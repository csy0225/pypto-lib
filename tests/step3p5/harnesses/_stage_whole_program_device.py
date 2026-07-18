# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Device dispatch of WholeDecodeProgram (1 dense + 1 MoE + tail) on TP=8.

Scheduler-timeout bisect endpoint. ``whole_decode_faithful`` (88 pass-blocks,
42 MoE layers REUSING one set of comm/scratch windows) dispatches clean but
STALLS mid-execution (sched_error_code=100 / 507018). ``whole_decode_program``
is the minimal MoE-in-whole-program case: 4 passes (L0 full-dense ->
L3 MoE-attn -> L3 MoE-block -> tail) with DISTINCT per-layer windows (each
window used exactly once, no cross-layer reuse).

Decisive discriminator:
  - RUNS clean  => a single MoE layer forward-progresses in whole-program
    context; the faithful stall is depth or cross-layer window reuse.
  - STALLS      => the stall is in a single MoE-block scope itself, in the
    whole-program context (not reuse / depth).

    python -m tests.step3p5.harnesses._stage_whole_program_device -p a2a3 -d 0,1,2,3,4,5,6,7

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
    p.add_argument("--layer-name", default="whole_decode_program")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))

    device_ids = [int(d) for d in str(args.device).split(",")]
    n_ranks = len(device_ids)

    from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415
    set_backend_type({
        "a2a3": BackendType.Ascend910B,
        "a2a3sim": BackendType.Ascend910B,
    }[args.platform])

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
    KVH = C("KV_HIDDEN_LOCAL")
    NHF_PAD = C("NUM_HEADS_FULL_LOCAL_PAD")
    LQ_FULL = C("LAYER_QHIDDEN_ROWS_DYN_FULL")
    INTER_LOCAL = C("INTER_LOCAL")
    N_EXPERTS = C("N_EXPERTS")
    N_LOCAL_EXPERTS = C("N_LOCAL_EXPERTS")
    INTER = C("MOE_INTERMEDIATE")
    SH_INTER_LOCAL = C("INTER_S_LOCAL")
    ROT_FULL = cfg.ROTARY_HALF_FULL * 2
    UBD = C("USER_BATCH_DYN")
    BTF = C("BLOCK_TABLE_FLAT_DYN")
    RSD = C("ROPE_SEQ_DYN")
    KVC = C("KV_CACHE_ROWS_DYN")
    VOCAB_LOCAL = C("VOCAB_LOCAL")

    bf16, f32, i32 = torch.bfloat16, torch.float32, torch.int32

    def z(*shape, dtype=bf16):
        return torch.zeros(shape, dtype=dtype)

    inputs: list[torch.Tensor] = []
    add = inputs.append

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

    # ---- L3 MoE attention (full) (9) ----
    add(z(tp, LAYER_DYN, HIDDEN, dtype=f32))        # l3_input_rms
    add(z(tp, LHR, HQ_FULL))                        # l3_wq
    add(z(tp, LHR, KVH))                            # l3_wk
    add(z(tp, LHR, KVH))                            # l3_wv
    add(z(tp, LAYER_DYN, HEAD_DIM, dtype=f32))      # l3_q_norm
    add(z(tp, LAYER_DYN, HEAD_DIM, dtype=f32))      # l3_k_norm
    add(z(tp, LQ_FULL, HIDDEN))                     # l3_wo
    add(z(tp, LHR, NHF_PAD))                        # l3_w_g
    add(z(tp, NHF_PAD, HQ_FULL))                    # l3_gate_r

    # ---- L3 MoE block (18) ----
    add(z(tp, LAYER_DYN, HIDDEN, dtype=f32))        # l3_post_rms
    add(z(tp, HIDDEN, N_EXPERTS, dtype=f32))        # l3_gate_w
    add(z(tp, N_EXPERTS, dtype=f32))                # l3_router_bias
    add(z(tp, N_LOCAL_EXPERTS, HIDDEN, INTER))      # l3_w_gate_r
    add(z(tp, N_LOCAL_EXPERTS, HIDDEN, INTER))      # l3_w_up_r
    add(z(tp, N_LOCAL_EXPERTS, INTER, HIDDEN))      # l3_w_down_r
    add(z(tp, HIDDEN, SH_INTER_LOCAL))              # l3_w_gate_s
    add(z(tp, HIDDEN, SH_INTER_LOCAL))              # l3_w_up_s
    add(z(tp, SH_INTER_LOCAL, HIDDEN))             # l3_w_down_s

    # ---- shared runtime (7) ----
    add(torch.ones(tp, UBD, dtype=i32))             # seq_lens
    add(torch.zeros(tp, BTF, dtype=i32))            # block_table
    add(torch.arange(UBD, dtype=i32).unsqueeze(0).repeat(tp, 1))  # slot_mapping
    add(z(tp, RSD, ROT_FULL, dtype=f32))            # rope_cos
    add(z(tp, RSD, ROT_FULL, dtype=f32))            # rope_sin
    add(z(tp, KVC, HEAD_DIM))                       # k_cache
    add(z(tp, KVC, HEAD_DIM))                       # v_cache

    # ---- resident handoff (3 Out) ----
    h0_out = z(tp, BATCH, HIDDEN)
    resid1_out = z(tp, BATCH, HIDDEN)
    h3_out = z(tp, BATCH, HIDDEN)
    add(h0_out)
    add(resid1_out)
    add(h3_out)

    # ---- tail (3) ----
    add(z(tp, 1, HIDDEN, dtype=f32))                # final_norm_weight
    add(z(tp, VOCAB_LOCAL, HIDDEN))                 # lm_head_weight
    logits_shard_out = z(tp, UBD, VOCAB_LOCAL, dtype=f32)
    add(logits_shard_out)                           # logits_shard_out (Out)

    print(f"[wdp-dev] built {len(inputs)} inputs; tp={tp} HIDDEN={HIDDEN} "
          f"N_LOCAL_EXPERTS={N_LOCAL_EXPERTS} VOCAB_LOCAL={VOCAB_LOCAL}",
          flush=True)

    from pypto import ir  # noqa: PLC0415
    from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415

    build_dir = "/data/chensiyu/hw_project/pypto/workspace/build_output"
    os.environ["PYPTO_PROG_BUILD_DIR"] = build_dir

    program = getattr(dl, args.layer_name)
    print(f"[wdp-dev] compiling {args.layer_name} platform={args.platform} "
          f"device_ids={device_ids} ...", flush=True)
    compiled = ir.compile(
        program, platform=args.platform,
        distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
        skip_ptoas=False, dump_passes=False,
    )
    print(f"[wdp-dev] compile OK => {compiled.output_dir}", flush=True)

    t0 = time.time()
    compiled(*inputs)
    dt = time.time() - t0
    print(f"[wdp-dev] DISPATCH+RUN done in {dt:.2f}s "
          f"max|h3|={h3_out.float().abs().max():.4f} "
          f"max|logits|={logits_shard_out.abs().max():.4f}", flush=True)
    print("[wdp-dev] RESULT=RUN_CLEAN", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
