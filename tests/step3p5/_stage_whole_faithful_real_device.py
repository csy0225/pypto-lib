# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Device dispatch / compile of WholeDecodeFaithfulReal (real per-layer weights).

Unified stacked host_orch signature: norm[45] + full-attn[12] + swa-attn[33] +
dense-mlp[3] + MoE-experts[42], per-layer full/swa routing. Dummy-zero inputs by
default (forward-progress / compile check); real weights come via the weight_loader
integration (task #2 validation) later.

    # compile only (fast, no device)
    python -m tests.step3p5._stage_whole_faithful_real_device -p a2a3 --compile-only
    # full 8-card dispatch
    python -m tests.step3p5._stage_whole_faithful_real_device -p a2a3 -d 0,1,2,3,4,5,6,7

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
    p.add_argument("--layer-name", default="whole_decode_faithful_real")
    p.add_argument("--compile-only", action="store_true")
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
    HQ_FULL = C("HIDDEN_Q_FULL_LOCAL")
    HQ_SWA = C("HIDDEN_Q_SWA_LOCAL")
    KVH = C("KV_HIDDEN_LOCAL")
    NHF_PAD = C("NUM_HEADS_FULL_LOCAL_PAD")
    NHS_PAD = C("NUM_HEADS_SWA_LOCAL_PAD")
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
    N_FULL = sum(1 for li in range(cfg.NUM_HIDDEN_LAYERS)
                 if cfg.is_full_attention(li))          # 12
    N_SWA = cfg.NUM_HIDDEN_LAYERS - N_FULL              # 33
    N_MOE = len(cfg.MOE_LAYER_INDICES)                  # 42
    N_DENSE = cfg.NUM_HIDDEN_LAYERS - N_MOE             # 3

    bf16, f32, i32 = torch.bfloat16, torch.float32, torch.int32

    def z(*shape, dtype=bf16):
        return torch.zeros(shape, dtype=dtype)

    inputs: list[torch.Tensor] = []
    add = inputs.append

    # current_hidden
    add(z(tp, BATCH, HIDDEN))
    # norm [45]
    add(z(tp, LAYER_DYN, HIDDEN, dtype=f32))            # input_rms
    add(z(tp, LAYER_DYN, HIDDEN, dtype=f32))            # post_rms
    add(z(tp, LAYER_DYN, HEAD_DIM, dtype=f32))          # q_norm
    add(z(tp, LAYER_DYN, HEAD_DIM, dtype=f32))          # k_norm
    # full attn [12] single-layer
    add(z(tp, N_FULL, HIDDEN, HQ_FULL))                 # full_wq
    add(z(tp, N_FULL, HIDDEN, KVH))                     # full_wk
    add(z(tp, N_FULL, HIDDEN, KVH))                     # full_wv
    add(z(tp, N_FULL, HQ_FULL, HIDDEN))                 # full_wo
    add(z(tp, N_FULL, HIDDEN, NHF_PAD))                 # full_w_g
    add(z(tp, N_FULL, NHF_PAD, HQ_FULL))               # full_gate_r
    # swa attn [33] single-layer
    add(z(tp, N_SWA, HIDDEN, HQ_SWA))                   # swa_wq
    add(z(tp, N_SWA, HIDDEN, KVH))                      # swa_wk
    add(z(tp, N_SWA, HIDDEN, KVH))                      # swa_wv
    add(z(tp, N_SWA, HQ_SWA, HIDDEN))                   # swa_wo
    add(z(tp, N_SWA, HIDDEN, NHS_PAD))                  # swa_w_g
    add(z(tp, N_SWA, NHS_PAD, HQ_SWA))                 # swa_gate_r
    # dense mlp [3] single-layer
    add(z(tp, N_DENSE, HIDDEN, INTER_LOCAL))           # dense_w_gate
    add(z(tp, N_DENSE, HIDDEN, INTER_LOCAL))           # dense_w_up
    add(z(tp, N_DENSE, INTER_LOCAL, HIDDEN))           # dense_w_down
    # MoE experts [42]
    add(z(tp, N_MOE, HIDDEN, N_EXPERTS, dtype=f32))    # moe_gate_w
    add(z(tp, N_MOE, N_EXPERTS, dtype=f32))            # moe_router_bias
    add(z(tp, N_MOE, N_LOCAL_EXPERTS, HIDDEN, INTER))  # moe_w_gate_r
    add(z(tp, N_MOE, N_LOCAL_EXPERTS, HIDDEN, INTER))  # moe_w_up_r
    add(z(tp, N_MOE, N_LOCAL_EXPERTS, INTER, HIDDEN))  # moe_w_down_r
    add(z(tp, N_MOE, HIDDEN, SH_INTER_LOCAL))          # moe_w_gate_s
    add(z(tp, N_MOE, HIDDEN, SH_INTER_LOCAL))          # moe_w_up_s
    add(z(tp, N_MOE, SH_INTER_LOCAL, HIDDEN))          # moe_w_down_s
    # runtime [9]
    add(torch.ones(tp, UBD, dtype=i32))                # seq_lens
    add(torch.zeros(tp, BTF, dtype=i32))               # block_table
    add(torch.arange(UBD, dtype=i32).unsqueeze(0).repeat(tp, 1))  # slot_mapping
    add(z(tp, RSD, ROT_FULL, dtype=f32))               # rope_cos_full
    add(z(tp, RSD, ROT_FULL, dtype=f32))               # rope_sin_full
    add(z(tp, RSD, ROT_SWA, dtype=f32))                # rope_cos_swa
    add(z(tp, RSD, ROT_SWA, dtype=f32))                # rope_sin_swa
    add(z(tp, KVC, HEAD_DIM))                           # k_cache
    add(z(tp, KVC, HEAD_DIM))                           # v_cache
    # handoff [2 Out]
    h_mid_out = z(tp, BATCH, HIDDEN)
    next_hidden_out = z(tp, BATCH, HIDDEN)
    add(h_mid_out)
    add(next_hidden_out)
    # tail [3]
    add(z(tp, 1, HIDDEN, dtype=f32))                    # final_norm_weight
    add(z(tp, VOCAB_LOCAL, HIDDEN))                     # lm_head_weight
    logits_shard_out = z(tp, UBD, VOCAB_LOCAL, dtype=f32)
    add(logits_shard_out)

    print(f"[real-dev] built {len(inputs)} inputs; tp={tp} "
          f"N_FULL={N_FULL} N_SWA={N_SWA} N_MOE={N_MOE}", flush=True)

    from pypto import ir  # noqa: PLC0415
    from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415

    build_dir = "/data/chensiyu/hw_project/pypto/workspace/build_output"
    os.environ["PYPTO_PROG_BUILD_DIR"] = build_dir

    program = getattr(dl, args.layer_name)
    print(f"[real-dev] compiling {args.layer_name} device_ids={device_ids} ...",
          flush=True)
    compiled = ir.compile(
        program, platform=args.platform,
        distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
        skip_ptoas=args.compile_only, dump_passes=False,
    )
    print(f"[real-dev] compile OK => {compiled.output_dir}", flush=True)
    if args.compile_only:
        print("[real-dev] RESULT=COMPILE_OK", flush=True)
        return 0

    t0 = time.time()
    compiled(*inputs)
    dt = time.time() - t0
    print(f"[real-dev] DISPATCH+RUN done in {dt:.2f}s "
          f"max|next_hidden|={next_hidden_out.float().abs().max():.4f} "
          f"max|logits|={logits_shard_out.abs().max():.4f}", flush=True)
    print("[real-dev] RESULT=DISPATCH_CLEAN", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
