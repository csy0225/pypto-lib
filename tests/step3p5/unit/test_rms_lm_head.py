# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Single-card precision UT for ``rms_lm_head`` at canonical TP=8 LOCAL dims.

step3p5 kernels are written for per-rank LOCAL widths (1/8 of the model
dim, e.g. ``VOCAB_LOCAL = VOCAB / TP_WORLD_SIZE = 16112``). Each rank in
the canonical TP=8 deployment receives those LOCAL shards as inputs and
produces a LOCAL output shard. This UT runs the per-rank chip_orch body
exactly as it would run on one of the 8 production ranks - with the
LOCAL widths the kernel was tiled for.

If a kernel cannot fit its L0/UB budget at the LOCAL widths, multi-card
e2e cannot help (each rank would hit the same budget); so this single-
card UT is also the canonical buffer-budget validator.

Usage::

    cd /data/chensiyu/hw_project/pypto/workspace/pypto-lib
    python -m tests.step3p5.unit.test_rms_lm_head --smoke
    python -m tests.step3p5.unit.test_rms_lm_head -p a2a3 -d 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pypto.language as pl
import torch

from models.step3p5.config import (
    BATCH,
    HIDDEN,
    USER_BATCH_DYN,
    VOCAB_LOCAL,
)
from models.step3p5.rms_lm_head import golden_rms_lm_head, rms_lm_head


# Thin standalone @pl.jit wrapper for the @pl.jit.inline rms_lm_head body.
# Matches the gate_test pattern: a single-rank kernel callable from
# golden.run_jit, with the canonical LOCAL widths intact.
@pl.jit
def rms_lm_head_test(
    hidden_states: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    lm_head_weight: pl.Tensor[[VOCAB_LOCAL, HIDDEN], pl.BF16],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    logits_shard_out: pl.Out[
        pl.Tensor[[USER_BATCH_DYN, VOCAB_LOCAL], pl.FP32]
    ],
):
    logits_shard_out = rms_lm_head(
        hidden_states, final_norm_weight, lm_head_weight,
        seq_lens, logits_shard_out,
    )
    return logits_shard_out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--smoke", action="store_true",
                        help="compile-only (no device run)")
    return parser.parse_args()


def _build_inputs(seed: int) -> dict[str, torch.Tensor]:
    """Per-rank LOCAL-width inputs (what one of the 8 ranks would see)."""
    g = torch.Generator().manual_seed(seed)

    hidden_states = (
        torch.empty(BATCH, HIDDEN, dtype=torch.float32)
        .normal_(mean=0.0, std=0.02, generator=g)
        .to(torch.bfloat16)
    )
    final_norm_weight = (
        torch.empty(1, HIDDEN, dtype=torch.float32)
        .normal_(mean=0.0, std=0.05, generator=g)
    )
    lm_head_weight = (
        torch.empty(VOCAB_LOCAL, HIDDEN, dtype=torch.float32)
        .normal_(mean=0.0, std=0.02, generator=g)
        .to(torch.bfloat16)
    )
    seq_lens = torch.full((USER_BATCH_DYN,), USER_BATCH_DYN, dtype=torch.int32)

    return {
        "hidden_states": hidden_states,
        "final_norm_weight": final_norm_weight,
        "lm_head_weight": lm_head_weight,
        "seq_lens": seq_lens,
    }


def _build_specs(inputs: dict[str, torch.Tensor]) -> list:
    from golden import TensorSpec  # noqa: PLC0415

    return [
        TensorSpec("hidden_states", list(inputs["hidden_states"].shape),
                   torch.bfloat16, init_value=inputs["hidden_states"]),
        TensorSpec("final_norm_weight",
                   list(inputs["final_norm_weight"].shape),
                   torch.float32, init_value=inputs["final_norm_weight"]),
        TensorSpec("lm_head_weight", list(inputs["lm_head_weight"].shape),
                   torch.bfloat16, init_value=inputs["lm_head_weight"]),
        TensorSpec("seq_lens", list(inputs["seq_lens"].shape),
                   torch.int32, init_value=inputs["seq_lens"]),
        TensorSpec("logits_shard_out",
                   [USER_BATCH_DYN, VOCAB_LOCAL],
                   torch.float32, is_output=True),
    ]


def _golden_fn(values: dict[str, torch.Tensor]) -> None:
    """Per-rank torch reference, mirrors rms_lm_head chip_orch on one rank."""
    out = golden_rms_lm_head(
        values["hidden_states"],
        values["final_norm_weight"],
        values["lm_head_weight"],
    )
    values["logits_shard_out"][:] = out.to(torch.float32)


def main() -> int:
    args = _parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))

    from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415

    set_backend_type({
        "a2a3": BackendType.Ascend910B,
        "a2a3sim": BackendType.Ascend910B,
        "a5": BackendType.Ascend950,
        "a5sim": BackendType.Ascend950,
    }[args.platform])

    print(
        f"[test_rms_lm_head] LOCAL dims: BATCH={BATCH} HIDDEN={HIDDEN} "
        f"VOCAB_LOCAL={VOCAB_LOCAL} (= VOCAB/TP_WORLD_SIZE)",
        flush=True,
    )

    inputs = _build_inputs(args.seed)
    specs = _build_specs(inputs)

    from golden import ratio_allclose, run_jit  # noqa: PLC0415

    if args.smoke or args.platform.endswith("sim"):
        result = run_jit(
            fn=rms_lm_head_test,
            specs=specs,
            runtime_cfg=dict(platform=args.platform, device_id=args.device),
            compile_only=True,
        )
        print(f"[test_rms_lm_head] SMOKE RESULT: {result}", flush=True)
        return 0 if result.passed else 1

    result = run_jit(
        fn=rms_lm_head_test,
        specs=specs,
        golden_fn=_golden_fn,
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=3e-3,
        atol=3e-3,
        # bf16 LM-head matmul: a 4096-wide K-reduction across a 16112-row
        # vocab slab. The per-rank slab is 1/8 of the unsliced VOCAB, so
        # ULP noise scales the same per-output cell, just with 1/8 the
        # cell count. 6% outlier cap matches the qwen3-style bf16 budget.
        compare_fn={
            "logits_shard_out": ratio_allclose(
                atol=3e-3, rtol=3e-3, max_error_ratio=0.06,
            ),
        },
    )
    print(f"[test_rms_lm_head] DEVICE RESULT: {result}", flush=True)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
