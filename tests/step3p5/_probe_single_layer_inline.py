# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Frontend probe: does pl.inline(attention_swa._func) accept a SINGLE-LAYER
weight tensor whose leading dim (HIDDEN) is smaller than the inline param
annotation (LAYER_HIDDEN_ROWS_DYN = 12*HIDDEN)?

This decides whether the N=1 real-per-layer swa-MoE attention can host-slice a
single-layer slab (clean) or must fall back to 12-layer sub-stacks + kernel
index (uglier). We call the inline with wq/wk/wv/wo/w_g declared single-layer
and attn_layer_idx=0 so the kernel's base=attn_idx*HIDDEN reads [0:HIDDEN].

    python -m tests.step3p5._probe_single_layer_inline -p a2a3

Compile-only (skip_ptoas): reaches the frontend trace + IR shape checks, which
is where an inline arg-shape mismatch would surface.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-p", "--platform", default="a2a3",
                   choices=["a2a3", "a2a3sim"])
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415
    set_backend_type(BackendType.Ascend910B)

    import pypto.language as pl  # noqa: PLC0415
    import pypto.language.distributed as pld  # noqa: PLC0415
    import models.step3p5.config as cfg  # noqa: PLC0415
    from models.step3p5 import attention_swa as swa_mod  # noqa: PLC0415

    # pl.inline resolves the inlined body's free vars against the CALLER
    # module globals (mirror decode_layer.py, which imports all config consts).
    globals().update({k: v for k, v in vars(cfg).items() if k.isupper()})

    tp = cfg.TP_WORLD_SIZE
    BATCH = cfg.BATCH
    HIDDEN = cfg.HIDDEN
    HEAD_DIM = cfg.HEAD_DIM
    HQ_SWA = cfg.HIDDEN_Q_SWA_LOCAL
    KVH = cfg.KV_HIDDEN_LOCAL
    NHS_PAD = cfg.NUM_HEADS_SWA_LOCAL_PAD
    ROT_SWA = cfg.ROTARY_HALF_SWA * 2
    LAYER_DYN = cfg.LAYER_DYN
    KVC = cfg.KV_CACHE_ROWS_DYN
    UBD = cfg.USER_BATCH_DYN
    BTF = cfg.BLOCK_TABLE_FLAT_DYN
    RSD = cfg.ROPE_SEQ_DYN

    attention_swa_inline = pl.inline(swa_mod.attention_swa._func)

    @pl.program
    class ProbeSingleLayer:
        @pl.function(type=pl.FunctionType.InCore)
        def tp_all_reduce(
            self,
            local: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
            signal_window: pld.DistributedTensor[[tp, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            group_size = tp
            ar_chunk = HIDDEN // 8
            for k0 in pl.range(0, HIDDEN, ar_chunk):
                stage_tile = pl.load(local, [0, k0], [BATCH, ar_chunk])
                pl.store(stage_tile, [0, k0], tmp_window)
            for peer in pl.range(group_size):
                if peer != my_rank:
                    pld.system.notify(
                        target=signal_window, peer=peer,
                        offsets=[my_rank, 0], value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for src in pl.range(group_size):
                if src != my_rank:
                    pld.system.wait(
                        signal=signal_window, offsets=[src, 0],
                        expected=1, cmp=pld.WaitCmp.Ge,
                    )
            for k0 in pl.range(0, HIDDEN, ar_chunk):
                own_tile = pl.load(tmp_window, [0, k0], [BATCH, ar_chunk])
                acc = pl.cast(own_tile, target_type=pl.FP32)
                for peer in pl.range(group_size):
                    if peer != my_rank:
                        recv = pld.tile.remote_load(
                            tmp_window, peer=peer,
                            offsets=[0, k0], shape=[BATCH, ar_chunk],
                        )
                        acc = pl.add(acc, pl.cast(recv, target_type=pl.FP32))
                pl.store(pl.cast(acc, target_type=pl.BF16), [0, k0], local)
            return local

        @pl.function(type=pl.FunctionType.Orchestration)
        def swa_attn_only(  # noqa: PLR0913
            self,
            current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            input_rms: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            wq: pl.Tensor[[HIDDEN, HQ_SWA], pl.BF16],           # single-layer!
            wk: pl.Tensor[[HIDDEN, KVH], pl.BF16],              # single-layer!
            wv: pl.Tensor[[HIDDEN, KVH], pl.BF16],              # single-layer!
            q_norm: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            seq_lens: pl.Tensor[[UBD], pl.INT32],
            block_table: pl.Tensor[[BTF], pl.INT32],
            slot_mapping: pl.Tensor[[UBD], pl.INT32],
            rope_cos: pl.Tensor[[RSD, ROT_SWA], pl.FP32],
            rope_sin: pl.Tensor[[RSD, ROT_SWA], pl.FP32],
            k_cache: pl.Tensor[[KVC, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[KVC, HEAD_DIM], pl.BF16],
            wo: pl.Tensor[[HQ_SWA, HIDDEN], pl.BF16],           # single-layer!
            w_g: pl.Tensor[[HIDDEN, NHS_PAD], pl.BF16],         # single-layer!
            gate_r: pl.Tensor[[NHS_PAD, HQ_SWA], pl.BF16],
            resid3_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            attn_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
            attn_signal_window: pld.DistributedTensor[[tp, 1], pl.INT32],
            norm_layer_idx: pl.Scalar[pl.INT32],
            attn_layer_idx: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            resid3_out = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            resid3_out = attention_swa_inline(
                current_hidden, input_rms,
                wq, wk, wv, q_norm, k_norm,
                seq_lens, block_table, slot_mapping,
                rope_cos, rope_sin, k_cache, v_cache,
                wo, w_g, gate_r, resid3_out,
                norm_layer_idx, attn_layer_idx,
                attn_tmp_window, attn_signal_window, my_rank,
            )
            return resid3_out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(  # noqa: PLR0913
            self,
            current_hidden: pl.Tensor[[tp, BATCH, HIDDEN], pl.BF16],
            input_rms: pl.Tensor[[tp, LAYER_DYN, HIDDEN], pl.FP32],
            wq: pl.Tensor[[tp, HIDDEN, HQ_SWA], pl.BF16],
            wk: pl.Tensor[[tp, HIDDEN, KVH], pl.BF16],
            wv: pl.Tensor[[tp, HIDDEN, KVH], pl.BF16],
            q_norm: pl.Tensor[[tp, LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm: pl.Tensor[[tp, LAYER_DYN, HEAD_DIM], pl.FP32],
            seq_lens: pl.Tensor[[tp, UBD], pl.INT32],
            block_table: pl.Tensor[[tp, BTF], pl.INT32],
            slot_mapping: pl.Tensor[[tp, UBD], pl.INT32],
            rope_cos: pl.Tensor[[tp, RSD, ROT_SWA], pl.FP32],
            rope_sin: pl.Tensor[[tp, RSD, ROT_SWA], pl.FP32],
            k_cache: pl.Tensor[[tp, KVC, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[tp, KVC, HEAD_DIM], pl.BF16],
            wo: pl.Tensor[[tp, HQ_SWA, HIDDEN], pl.BF16],
            w_g: pl.Tensor[[tp, HIDDEN, NHS_PAD], pl.BF16],
            gate_r: pl.Tensor[[tp, NHS_PAD, HQ_SWA], pl.BF16],
            resid3_out: pl.Out[pl.Tensor[[tp, BATCH, HIDDEN], pl.BF16]],
        ):
            tmp = pld.alloc_window_buffer(BATCH * HIDDEN * 2)
            sig = pld.alloc_window_buffer(tp * 4)
            for r in pl.range(pld.world_size()):
                self.swa_attn_only(
                    current_hidden[r], input_rms[r],
                    wq[r], wk[r], wv[r], q_norm[r], k_norm[r],
                    seq_lens[r], block_table[r], slot_mapping[r],
                    rope_cos[r], rope_sin[r], k_cache[r], v_cache[r],
                    wo[r], w_g[r], gate_r[r], resid3_out[r],
                    pld.window(tmp, [BATCH, HIDDEN], dtype=pl.BF16),
                    pld.window(sig, [tp, 1], dtype=pl.INT32),
                    0, 0, r, device=r,
                )

    from pypto import ir  # noqa: PLC0415
    from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415

    print("[probe] compiling ProbeSingleLayer (single-layer swa inline) ...",
          flush=True)
    ir.compile(
        ProbeSingleLayer, platform=args.platform,
        distributed_config=DistributedConfig(
            device_ids=list(range(tp)), num_sub_workers=0),
        skip_ptoas=True, dump_passes=False,
    )
    print("[probe] RESULT=SINGLE_LAYER_INLINE_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
