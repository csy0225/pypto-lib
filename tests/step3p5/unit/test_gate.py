# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Single-card precision UT for the MoE gate.

The step3p5 router is REPLICATED across all ranks (gate_w + router_bias
broadcast unmodified to every TP rank), so no TP=1 monkey-patch is
needed - we run the kernel on a single device with the full unsliced
weights and compare against ``golden_gate``.

Tolerance:
- ``expert_indices`` (int32): exact match for non-tied scores; for
  near-ties the topk_pair_compare lets the kernel reorder as long as
  the actual score column stays monotonically descending.
- ``expert_weights`` (bfloat16): bf16 noise on the ``score / sum * 3.0``
  renormalisation; 6e-3 atol/rtol covers 1-2 ULP at the 0..3 weight
  range with up to 6% outliers (matches the rms_lm_head tolerance).

Usage::

    cd /data/chensiyu/hw_project/pypto/workspace/pypto-lib
    python -m tests.step3p5.unit.test_gate --smoke
    python -m tests.step3p5.unit.test_gate -p a2a3 -d 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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


def main() -> int:
    args = _parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))

    import torch  # noqa: PLC0415

    torch.manual_seed(args.seed)

    from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415

    backend_map = {
        "a2a3": BackendType.Ascend910B,
        "a2a3sim": BackendType.Ascend910B,
        "a5": BackendType.Ascend950,
        "a5sim": BackendType.Ascend950,
    }
    set_backend_type(backend_map[args.platform])

    from models.step3p5.gate import (  # noqa: PLC0415
        build_tensor_specs,
        gate_test,
        golden_gate,
    )

    specs = build_tensor_specs()
    print(
        f"[test_gate] platform={args.platform} device={args.device} "
        f"smoke={args.smoke}",
        flush=True,
    )

    from golden import ratio_allclose, run_jit, topk_pair_compare  # noqa: PLC0415

    if args.smoke or args.platform.endswith("sim"):
        result = run_jit(
            fn=gate_test,
            specs=specs,
            runtime_cfg=dict(platform=args.platform, device_id=args.device),
            compile_only=True,
        )
        print(f"[test_gate] SMOKE RESULT: {result}", flush=True)
        return 0 if result.passed else 1

    result = run_jit(
        fn=gate_test,
        specs=specs,
        golden_fn=golden_gate,
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=6e-3,
        atol=6e-3,
        compare_fn={
            # idx column: tolerate score-tie reordering as long as the
            # paired weights remain monotonically descending.
            "expert_indices": topk_pair_compare("expert_weights"),
            # weights: top-k weights are renormalised to sum=1 then *3.0,
            # so the dynamic range is small (<3.0) - 1-2 ULP bf16 noise
            # falls within 6e-3 for >94% of cells.
            "expert_weights": ratio_allclose(
                atol=6e-3, rtol=6e-3, max_error_ratio=0.06,
            ),
        },
    )
    print(f"[test_gate] DEVICE RESULT: {result}", flush=True)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
