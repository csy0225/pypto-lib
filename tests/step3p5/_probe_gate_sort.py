# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Sort-only device probe for the MoE gate_topk cascade (device-vs-torch).

Isolates gate_topk sort+gather+renorm (gate.py Stage-2/3) from the gate_matmul
(which overflows Mat/Vec on a single card with full unsliced N_EXPERTS=288 -- a
pre-existing single-card-shape issue, unrelated to the sort). Feeds pre-computed
biased/score buffers directly so the mrgsort cascade fix (format1 block_len=256)
is validated on device vs a torch topk reference.

Usage::
    cd /data/chensiyu/hw_project/pypto/workspace/pypto-lib
    python -m tests.step3p5._probe_gate_sort -p a2a3 -d 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Module-level so the @pl.jit annotation/scope resolution finds them.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pypto.language as pl  # noqa: E402
from models.step3p5.gate import (  # noqa: E402
    T, N_EXPERTS, TOPK, TOPK_PAD, SORT_PAD, SCORE_PAD, ROUTE_SCALE,
)


@pl.jit
def gate_sort_probe(
    biased_buf: pl.Tensor[[T, SCORE_PAD], pl.FP32],
    score_buf: pl.Tensor[[T, SCORE_PAD], pl.FP32],
    expert_indices: pl.Out[pl.Tensor[[T, TOPK], pl.INT32]],
    expert_weights: pl.Out[pl.Tensor[[T, TOPK], pl.FP32]],
):
    # Verbatim gate.py Stage-2/3 (the fixed sort cascade).
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_topk"):
        topk_idx_tile = pl.create_tensor([T, TOPK_PAD], dtype=pl.INT32)
        for tt in pl.range(T):
            row = biased_buf[tt : tt + 1, :]
            idx_init = pl.arange(0, [1, SCORE_PAD], dtype=pl.UINT32)
            srt = pl.sort32(row, idx_init)
            srt = pl.mrgsort(srt, block_len=64)          # 16 runs of 64 -> 4 runs of 256
            srt = pl.mrgsort(srt, block_len=256)         # 4 runs of 256 -> 1 sorted run
            pairs = srt[:, 0:SORT_PAD]
            top_idx = pl.gather(
                pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32,
            )
            topk_idx_tile[tt : tt + 1, :] = top_idx
        gather_all = pl.gather(score_buf, dim=-1, index=topk_idx_tile)
        gather_valid = pl.set_validshape(gather_all, T, TOPK)
        topk_vals_pad = pl.fillpad(gather_valid, pad_value=pl.PadValue.zero)
        denom = pl.reshape(pl.row_sum(topk_vals_pad), [T, 1])
        weights_pad = pl.mul(pl.row_expand_div(topk_vals_pad, denom), ROUTE_SCALE)
        for tt in pl.range(T):
            for k in pl.range(TOPK):
                pl.write(expert_indices, [tt, k], pl.read(topk_idx_tile, [tt, k]))
                pl.write(expert_weights, [tt, k], pl.read(weights_pad, [tt, k]))
    return expert_indices, expert_weights


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-p", "--platform", default="a2a3",
                   choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    p.add_argument("-d", "--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    import torch  # noqa: PLC0415
    torch.manual_seed(args.seed)

    from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415
    set_backend_type({"a2a3": BackendType.Ascend910B, "a2a3sim": BackendType.Ascend910B,
                      "a5": BackendType.Ascend950, "a5sim": BackendType.Ascend950}[args.platform])

    from golden import TensorSpec, ratio_allclose, run_jit, topk_pair_compare  # noqa: PLC0415

    def init_biased():
        # pad columns < all real scores so sort ranks them last.
        b = torch.full((T, SCORE_PAD), -1e30, dtype=torch.float32)
        b[:, :N_EXPERTS] = torch.rand(T, N_EXPERTS) * 1.2
        return b

    def init_score():
        s = torch.zeros(T, SCORE_PAD, dtype=torch.float32)
        s[:, :N_EXPERTS] = torch.rand(T, N_EXPERTS)
        return s

    _biased = init_biased()
    _score = init_score()

    def golden(tensors):
        biased = tensors["biased_buf"].float()
        score = tensors["score_buf"].float()
        idx = torch.argsort(-biased, dim=-1, stable=True)[:, :TOPK]
        vals = torch.gather(score, dim=-1, index=idx.long())
        wts = (vals / vals.sum(dim=-1, keepdim=True)) * ROUTE_SCALE
        tensors["expert_indices"][:] = idx.to(torch.int32)
        tensors["expert_weights"][:] = wts.to(torch.float32)

    specs = [
        TensorSpec("biased_buf", [T, SCORE_PAD], torch.float32, init_value=lambda: _biased.clone()),
        TensorSpec("score_buf", [T, SCORE_PAD], torch.float32, init_value=lambda: _score.clone()),
        TensorSpec("expert_indices", [T, TOPK], torch.int32, is_output=True),
        TensorSpec("expert_weights", [T, TOPK], torch.float32, is_output=True),
    ]

    print(f"[gate_sort_probe] platform={args.platform} device={args.device} T={T}", flush=True)
    if args.smoke or args.platform.endswith("sim"):
        r = run_jit(fn=gate_sort_probe, specs=specs,
                    runtime_cfg=dict(platform=args.platform, device_id=args.device),
                    compile_only=True)
        print(f"[gate_sort_probe] SMOKE: {r}", flush=True)
        return 0 if r.passed else 1

    r = run_jit(
        fn=gate_sort_probe, specs=specs, golden_fn=golden,
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=6e-3, atol=6e-3,
        compare_fn={
            "expert_indices": topk_pair_compare("expert_weights"),
            "expert_weights": ratio_allclose(atol=6e-3, rtol=6e-3, max_error_ratio=0.06),
        },
    )
    print(f"[gate_sort_probe] DEVICE RESULT: {r}", flush=True)
    return 0 if r.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
