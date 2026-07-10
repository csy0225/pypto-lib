# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Device dispatch of WholeDecodeFaithful on canonical TP=8 (Wall-2 experiment).

Compiles the module-level ``whole_decode_faithful`` @pl.program (ONE program,
per-protocol-separated TP-attn methods + EP-MoE method) under canonical TP=8 +
``DistributedConfig(device_ids=...)`` and actually DISPATCHES it on 8 cards with
dummy (zero) weights.

Decisive question: does per-protocol separation avoid the Wall-2
``TaskMapSize=0`` / 507018 dispatch fault that a fused TP+EP dispatch hits?
Numerical correctness is NOT the goal here — only whether host_orch produces a
non-empty chip task map and the run completes without 507018.

    python -m tests.step3p5._stage_whole_faithful_device -p a2a3 -d 0,1,2,3,4,5,6,7

env: source /usr/local/Ascend/cann/set_env.sh && source $WS/activate.sh &&
     export PTO_ISA_ROOT=$WS/pto-isa
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-p", "--platform", default="a2a3",
                   choices=["a2a3", "a2a3sim"])
    p.add_argument("-d", "--device", default="0,1,2,3,4,5,6,7")
    p.add_argument("--layer-name", default="whole_decode_faithful")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    device_ids = [int(d) for d in str(args.device).split(",")]
    n_ranks = len(device_ids)

    from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415
    set_backend_type({
        "a2a3": BackendType.Ascend910B,
        "a2a3sim": BackendType.Ascend910B,
    }[args.platform])

    # Canonical TP=8 — NO patch.
    import models.step3p5.config as cfg  # noqa: PLC0415
    import models.step3p5.decode_layer as dl  # noqa: PLC0415

    assert n_ranks == cfg.TP_WORLD_SIZE, (
        f"need {cfg.TP_WORLD_SIZE} cards; got {n_ranks}"
    )

    def C(name: str) -> int:
        v = getattr(dl, name, None)
        if v is None:
            v = getattr(cfg, name, None)
        if v is None:
            raise KeyError(f"const {name} not found on dl/cfg")
        return int(v)

    tp = cfg.TP_WORLD_SIZE
    BATCH = C("BATCH")
    HIDDEN = C("HIDDEN")
    HEAD_DIM = C("HEAD_DIM")
    LAYER_DYN = C("LAYER_DYN")
    LHR = C("LAYER_HIDDEN_ROWS_DYN")
    LIR = C("LAYER_INTER_ROWS_DYN")
    HQ_FULL = C("HIDDEN_Q_FULL_LOCAL")
    HQ_SWA = C("HIDDEN_Q_SWA_LOCAL")
    KVH = C("KV_HIDDEN_LOCAL")
    NHF_PAD = C("NUM_HEADS_FULL_LOCAL_PAD")
    NHS_PAD = C("NUM_HEADS_SWA_LOCAL_PAD")
    LQ_FULL = C("LAYER_QHIDDEN_ROWS_DYN_FULL")
    LQ_SWA = C("LAYER_QHIDDEN_ROWS_DYN_SWA")
    INTER_LOCAL = C("INTER_LOCAL")
    N_EXPERTS = C("N_EXPERTS")
    N_LOCAL_EXPERTS = C("N_LOCAL_EXPERTS")
    INTER = C("MOE_INTERMEDIATE")
    SH_INTER_LOCAL = C("INTER_S_LOCAL")
    ROT_FULL = cfg.ROTARY_HALF_FULL * 2
    ROT_SWA = cfg.ROTARY_HALF_SWA * 2
    UBD = C("USER_BATCH_DYN")
    BTF = C("BLOCK_TABLE_FLAT_DYN")
    RSD = C("ROPE_SEQ_DYN")
    KVC = C("KV_CACHE_ROWS_DYN")
    VOCAB_LOCAL = C("VOCAB_LOCAL")

    bf16, f32, i32 = torch.bfloat16, torch.float32, torch.int32

    def z(*shape, dtype=bf16):
        return torch.zeros(shape, dtype=dtype)

    inputs: list[torch.Tensor] = []

    def add(t: torch.Tensor):
        inputs.append(t)

    # current_hidden
    add(z(tp, BATCH, HIDDEN))

    # ---- L0 full-dense (13) ----
    add(z(tp, LAYER_DYN, HIDDEN, dtype=f32))       # l0_input_rms
    add(z(tp, LHR, HQ_FULL))                        # l0_wq
    add(z(tp, LHR, KVH))                            # l0_wk
    add(z(tp, LHR, KVH))                            # l0_wv
    add(z(tp, LAYER_DYN, HEAD_DIM, dtype=f32))      # l0_q_norm
    add(z(tp, LAYER_DYN, HEAD_DIM, dtype=f32))      # l0_k_norm
    add(z(tp, LQ_FULL, HIDDEN))                     # l0_wo
    add(z(tp, LHR, NHF_PAD))                        # l0_w_g
    add(z(tp, NHF_PAD, HQ_FULL))                    # l0_gate_r
    add(z(tp, LAYER_DYN, HIDDEN, dtype=f32))        # l0_post_rms
    add(z(tp, LHR, INTER_LOCAL))                    # l0_w_gate
    add(z(tp, LHR, INTER_LOCAL))                    # l0_w_up
    add(z(tp, LIR, HIDDEN))                         # l0_w_down

    # ---- L1 + L2 swa-dense (13 each, identical shapes) ----
    for _ in range(2):
        add(z(tp, LAYER_DYN, HIDDEN, dtype=f32))    # input_rms
        add(z(tp, LHR, HQ_SWA))                     # wq
        add(z(tp, LHR, KVH))                        # wk
        add(z(tp, LHR, KVH))                        # wv
        add(z(tp, LAYER_DYN, HEAD_DIM, dtype=f32))  # q_norm
        add(z(tp, LAYER_DYN, HEAD_DIM, dtype=f32))  # k_norm
        add(z(tp, LQ_SWA, HIDDEN))                  # wo
        add(z(tp, LHR, NHS_PAD))                    # w_g
        add(z(tp, NHS_PAD, HQ_SWA))                 # gate_r
        add(z(tp, LAYER_DYN, HIDDEN, dtype=f32))    # post_rms
        add(z(tp, LHR, INTER_LOCAL))                # w_gate
        add(z(tp, LHR, INTER_LOCAL))                # w_up
        add(z(tp, LIR, HIDDEN))                     # w_down

    # ---- L3..44 MoE attn (swa) (9) ----
    add(z(tp, LAYER_DYN, HIDDEN, dtype=f32))        # ma_input_rms
    add(z(tp, LHR, HQ_SWA))                         # ma_wq
    add(z(tp, LHR, KVH))                            # ma_wk
    add(z(tp, LHR, KVH))                            # ma_wv
    add(z(tp, LAYER_DYN, HEAD_DIM, dtype=f32))      # ma_q_norm
    add(z(tp, LAYER_DYN, HEAD_DIM, dtype=f32))      # ma_k_norm
    add(z(tp, LQ_SWA, HIDDEN))                      # ma_wo
    add(z(tp, LHR, NHS_PAD))                        # ma_w_g
    add(z(tp, NHS_PAD, HQ_SWA))                     # ma_gate_r

    # ---- L3..44 MoE block (18) ----
    add(z(tp, LAYER_DYN, HIDDEN, dtype=f32))        # m_post_rms
    add(z(tp, LHR, HQ_SWA))                         # m_wq
    add(z(tp, LHR, KVH))                            # m_wk
    add(z(tp, LHR, KVH))                            # m_wv
    add(z(tp, LAYER_DYN, HEAD_DIM, dtype=f32))      # m_q_norm
    add(z(tp, LAYER_DYN, HEAD_DIM, dtype=f32))      # m_k_norm
    add(z(tp, LQ_SWA, HIDDEN))                      # m_wo
    add(z(tp, LHR, NHS_PAD))                        # m_w_g
    add(z(tp, NHS_PAD, HQ_SWA))                     # m_gate_r
    add(z(tp, HIDDEN, N_EXPERTS, dtype=f32))        # m_gate_w
    add(z(tp, N_EXPERTS, dtype=f32))                # m_router_bias
    add(z(tp, N_LOCAL_EXPERTS, HIDDEN, INTER))      # m_w_gate_r
    add(z(tp, N_LOCAL_EXPERTS, HIDDEN, INTER))      # m_w_up_r
    add(z(tp, N_LOCAL_EXPERTS, INTER, HIDDEN))      # m_w_down_r
    add(z(tp, HIDDEN, SH_INTER_LOCAL))              # m_w_gate_s
    add(z(tp, HIDDEN, SH_INTER_LOCAL))              # m_w_up_s
    add(z(tp, SH_INTER_LOCAL, HIDDEN))             # m_w_down_s

    # ---- shared runtime (9) ----
    add(torch.ones(tp, UBD, dtype=i32))             # seq_lens
    add(torch.zeros(tp, BTF, dtype=i32))            # block_table
    add(torch.arange(UBD, dtype=i32).unsqueeze(0).repeat(tp, 1))  # slot_mapping
    add(z(tp, RSD, ROT_FULL, dtype=f32))            # rope_cos_full
    add(z(tp, RSD, ROT_FULL, dtype=f32))            # rope_sin_full
    add(z(tp, RSD, ROT_SWA, dtype=f32))             # rope_cos_swa
    add(z(tp, RSD, ROT_SWA, dtype=f32))             # rope_sin_swa
    add(z(tp, KVC, HEAD_DIM))                       # k_cache
    add(z(tp, KVC, HEAD_DIM))                       # v_cache

    # ---- resident handoff (2 Out) ----
    h_mid_out = z(tp, BATCH, HIDDEN)
    next_hidden_out = z(tp, BATCH, HIDDEN)
    add(h_mid_out)
    add(next_hidden_out)

    # ---- tail (3) ----
    add(z(tp, 1, HIDDEN, dtype=f32))                # final_norm_weight
    add(z(tp, VOCAB_LOCAL, HIDDEN))                 # lm_head_weight
    logits_shard_out = z(tp, UBD, VOCAB_LOCAL, dtype=f32)
    add(logits_shard_out)                           # logits_shard_out (Out)

    print(f"[faithful-dev] built {len(inputs)} inputs; "
          f"tp={tp} HIDDEN={HIDDEN} N_LOCAL_EXPERTS={N_LOCAL_EXPERTS} "
          f"VOCAB_LOCAL={VOCAB_LOCAL}", flush=True)

    from pypto import ir  # noqa: PLC0415
    from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415

    build_dir = "/data/chensiyu/hw_project/pypto/workspace/build_output"
    os.environ["PYPTO_PROG_BUILD_DIR"] = build_dir

    program = getattr(dl, args.layer_name)
    print(f"[faithful-dev] compiling {args.layer_name} platform={args.platform} "
          f"device_ids={device_ids} ...", flush=True)
    compiled = ir.compile(
        program, platform=args.platform,
        distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
        skip_ptoas=False, dump_passes=False,
    )
    print(f"[faithful-dev] compile OK => {compiled.output_dir}", flush=True)

    t0 = time.time()
    compiled(*inputs)
    dt = time.time() - t0
    print(f"[faithful-dev] DISPATCH+RUN done in {dt:.2f}s "
          f"max|next_hidden|={next_hidden_out.float().abs().max():.4f} "
          f"max|logits|={logits_shard_out.abs().max():.4f}", flush=True)
    print("[faithful-dev] RESULT=DISPATCH_CLEAN", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
