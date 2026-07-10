# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""DEPRECATED — DO NOT USE. Kept only as a negative example.

Wrong architecture on two counts:
  1. It loads the checkpoint itself, sequentially, for all 8 ranks in ONE
     driver process (8x serial cold jfs reads → 40-min timeout, rc=124).
     Production loads per-rank INSIDE each forked chip process, in parallel,
     each reading only its own TP/EP slice (see step3p5_decode.py:275-293).
  2. More fundamentally, the whole-decode program is NOT supposed to load
     weights from a checkpoint at all. The intended design is the vLLM-IPC
     handoff: vLLM loads the weights into HBM per rank, and pypto zero-copy
     IPC-imports them (the "47GiB single-key weight IPC" of Task#3, same
     pattern as the zero-copy KV handoff). Real-weight execution + per-layer
     numeric alignment vs vLLM therefore only happen in the LIVE path
     (8001 pypto whole-net vs 8000 vanilla), which is gated on the HCCL
     same-card co-tenancy blocker (HcclCommInitRootInfo failed:7 — the
     forked whole-decode worker's HCCL world conflicts with vLLM's).

Use the dummy-weight device harness `_stage_whole_faithful_real_device.py`
for structure/dispatch validation; pursue real weights only via the live
vLLM-IPC integration.

--- original (flawed) docstring below ---
Real-weight 8-card device run of WholeDecodeFaithfulReal.

Loads the real W8A8 step3p5 checkpoint per rank via weight_loader, stacks each
bundle key over the rank axis (1:1 with the host_orch unified-stack signature —
see weight_loader.expected_shapes), and runs the whole-decode program on 8 cards.

Scope: proves the real per-layer weight pipeline runs end-to-end on device and
produces non-trivial logits. Per-layer numeric alignment vs the vLLM eager dump
is NOT done here — the dumps are 18-token PREFILL while these decode kernels are
BATCH=16 (single-token decode); a decode-step golden (or the live A/B path) is
required for token-exact comparison.

    python -m tests.step3p5._stage_whole_faithful_real_weights -p a2a3 \
        -d 0,1,2,3,4,5,6,7 \
        --ckpt /mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp

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
    p.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    p.add_argument("-d", "--device", default="0,1,2,3,4,5,6,7")
    p.add_argument("--layer-name", default="whole_decode_faithful_real")
    p.add_argument(
        "--ckpt",
        default="/mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    if not Path(args.ckpt).exists():
        print(f"[real-w] ckpt not found: {args.ckpt}", flush=True)
        return 2

    device_ids = [int(d) for d in str(args.device).split(",")]
    tp = len(device_ids)

    from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415
    set_backend_type({
        "a2a3": BackendType.Ascend910B,
        "a2a3sim": BackendType.Ascend910B,
    }[args.platform])

    import models.step3p5.config as cfg  # noqa: PLC0415
    import models.step3p5.decode_layer as dl  # noqa: PLC0415
    from models.step3p5 import weight_loader as wl  # noqa: PLC0415

    assert tp == cfg.TP_WORLD_SIZE, f"need {cfg.TP_WORLD_SIZE} cards; got {tp}"

    HIDDEN = cfg.HIDDEN
    HEAD_DIM = cfg.HEAD_DIM
    BATCH = cfg.BATCH
    UBD = int(getattr(dl, "USER_BATCH_DYN", getattr(cfg, "USER_BATCH_DYN")))
    BTF = int(getattr(dl, "BLOCK_TABLE_FLAT_DYN", getattr(cfg, "BLOCK_TABLE_FLAT_DYN")))
    RSD = int(getattr(dl, "ROPE_SEQ_DYN", getattr(cfg, "ROPE_SEQ_DYN")))
    KVC = int(getattr(dl, "KV_CACHE_ROWS_DYN", getattr(cfg, "KV_CACHE_ROWS_DYN")))
    ROT_FULL = cfg.ROTARY_HALF_FULL * 2
    ROT_SWA = cfg.ROTARY_HALF_SWA * 2
    bf16, f32, i32 = torch.bfloat16, torch.float32, torch.int32

    print(f"[real-w] loading {tp} rank bundles from {args.ckpt} ...", flush=True)
    bundles = [
        wl.load_step3p5_weights_for_rank(args.ckpt, rank=r, tp_world_size=tp)
        for r in range(tp)
    ]

    def stack(key: str, *, dtype=None) -> torch.Tensor:
        t = torch.stack([bundles[r][key] for r in range(tp)], dim=0)
        return t.to(dtype) if dtype is not None else t

    K = wl
    inputs: list[torch.Tensor] = []
    add = inputs.append

    # current_hidden — synthetic deterministic decode-token hidden (real numeric
    # alignment vs vLLM needs a decode-step golden; see module docstring).
    torch.manual_seed(0)
    add((torch.randn(tp, BATCH, HIDDEN) * 0.02).to(bf16))
    # norm [45]
    add(stack(K.KEY_INPUT_RMS, dtype=f32))
    add(stack(K.KEY_POST_ATTN_RMS, dtype=f32))
    add(stack(K.KEY_Q_NORM, dtype=f32))
    add(stack(K.KEY_K_NORM, dtype=f32))
    # full attn [12]
    add(stack(K.KEY_WQ_FULL, dtype=bf16))
    add(stack(K.KEY_WK_FULL, dtype=bf16))
    add(stack(K.KEY_WV_FULL, dtype=bf16))
    add(stack(K.KEY_WO_FULL, dtype=bf16))
    add(stack(K.KEY_WG_FULL, dtype=bf16))
    # full_gate_r: head-gate is derived at runtime (not a checkpoint weight);
    # zero here (numeric alignment is gated on the live path anyway).
    add(torch.zeros(tp, 12, cfg.NUM_HEADS_FULL_LOCAL_PAD,
                    cfg.HIDDEN_Q_FULL_LOCAL, dtype=bf16))
    # swa attn [33]
    add(stack(K.KEY_WQ_SWA, dtype=bf16))
    add(stack(K.KEY_WK_SWA, dtype=bf16))
    add(stack(K.KEY_WV_SWA, dtype=bf16))
    add(stack(K.KEY_WO_SWA, dtype=bf16))
    add(stack(K.KEY_WG_SWA, dtype=bf16))
    add(torch.zeros(tp, 33, cfg.NUM_HEADS_SWA_LOCAL_PAD,
                    cfg.HIDDEN_Q_SWA_LOCAL, dtype=bf16))  # swa_gate_r
    # dense mlp [3]
    add(stack(K.KEY_DENSE_GATE, dtype=bf16))
    add(stack(K.KEY_DENSE_UP, dtype=bf16))
    add(stack(K.KEY_DENSE_DOWN, dtype=bf16))
    # MoE experts [42]
    add(stack(K.KEY_MOE_GATE_W, dtype=f32))
    add(stack(K.KEY_MOE_ROUTER_BIAS, dtype=f32))
    add(stack(K.KEY_MOE_W_GATE_R, dtype=bf16))
    add(stack(K.KEY_MOE_W_UP_R, dtype=bf16))
    add(stack(K.KEY_MOE_W_DOWN_R, dtype=bf16))
    add(stack(K.KEY_MOE_W_GATE_S, dtype=bf16))
    add(stack(K.KEY_MOE_W_UP_S, dtype=bf16))
    add(stack(K.KEY_MOE_W_DOWN_S, dtype=bf16))
    # runtime [9] — synthetic (decode step, empty KV → biases-dominated output).
    add(torch.ones(tp, UBD, dtype=i32))
    add(torch.zeros(tp, BTF, dtype=i32))
    add(torch.arange(UBD, dtype=i32).unsqueeze(0).repeat(tp, 1))
    add(torch.zeros(tp, RSD, ROT_FULL, dtype=f32))
    add(torch.zeros(tp, RSD, ROT_FULL, dtype=f32))
    add(torch.zeros(tp, RSD, ROT_SWA, dtype=f32))
    add(torch.zeros(tp, RSD, ROT_SWA, dtype=f32))
    add(torch.zeros(tp, KVC, HEAD_DIM, dtype=bf16))
    add(torch.zeros(tp, KVC, HEAD_DIM, dtype=bf16))
    # handoff [2 Out]
    h_mid_out = torch.zeros(tp, BATCH, HIDDEN, dtype=bf16)
    next_hidden_out = torch.zeros(tp, BATCH, HIDDEN, dtype=bf16)
    add(h_mid_out)
    add(next_hidden_out)
    # tail [3]
    add(stack(K.KEY_FINAL_NORM, dtype=f32).reshape(tp, 1, HIDDEN))
    add(stack(K.KEY_LM_HEAD, dtype=bf16))
    VOCAB_LOCAL = int(bundles[0][K.KEY_LM_HEAD].shape[0])
    logits_shard_out = torch.zeros(tp, UBD, VOCAB_LOCAL, dtype=f32)
    add(logits_shard_out)

    print(f"[real-w] built {len(inputs)} inputs; VOCAB_LOCAL={VOCAB_LOCAL}", flush=True)

    from pypto import ir  # noqa: PLC0415
    from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415

    os.environ["PYPTO_PROG_BUILD_DIR"] = "/data/chensiyu/hw_project/pypto/workspace/build_output"
    program = getattr(dl, args.layer_name)
    print(f"[real-w] compiling {args.layer_name} device_ids={device_ids} ...", flush=True)
    compiled = ir.compile(
        program, platform=args.platform,
        distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
        skip_ptoas=False, dump_passes=False,
    )
    print(f"[real-w] compile OK => {compiled.output_dir}", flush=True)

    t0 = time.time()
    compiled(*inputs)
    dt = time.time() - t0
    full_logits = torch.cat([logits_shard_out[r, 0] for r in range(tp)], dim=0)
    print(f"[real-w] RUN done {dt:.2f}s max|logits|={logits_shard_out.abs().max():.4f} "
          f"argmax(tok0)={int(full_logits.argmax())}", flush=True)
    print("[real-w] RESULT=REAL_WEIGHT_RUN_CLEAN", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
