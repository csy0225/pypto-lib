# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Single-card precision UT for the TP-sliced shared-expert MLP.

Validates ``expert_shared_silu`` (and the swiglu16 variant) at canonical
``SHARE_EXPERT_DIM_LOCAL = 160`` - the per-rank shard one card sees in
production TP=8.

Why ``@pl.program`` (not ``@pl.jit``)
-------------------------------------
``expert_shared.py`` uses a factory ``_build_expert_shared`` whose body
references a Python *closure* variable ``use_swiglu_step`` (baked from
``swiglu_limit``). The pypto ``@pl.jit`` static AST parser cannot
resolve closure cells (only module-level globals), so the gate_test
pattern fails with ``Undefined variable 'use_swiglu_step'``. The
``@pl.program`` -> ``@pl.function`` route triggers Python evaluation of
the body at trace time, which preserves the closure and yields linear
IR; that is exactly how the production ``moe.py`` / ``decode_layer.py``
inlines these factory products. Mirror that by wrapping the inline body
in a single-rank ``@pl.program`` test scaffold.

Usage::

    cd /data/chensiyu/hw_project/pypto/workspace/pypto-lib
    python -m tests.step3p5.test_expert_shared --smoke
    python -m tests.step3p5.test_expert_shared -p a2a3 -d 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from models.step3p5.config import BATCH, HIDDEN, SHARE_EXPERT_DIM_LOCAL
from models.step3p5.expert_shared import (
    DOWN_K_CHUNK,
    DOWN_N_CHUNK,
    GATE_K_CHUNK,
    GATE_N_CHUNK,
    _torch_shared_local,
    expert_shared_silu,
    expert_shared_swiglu16,
)


T = BATCH
INTER = SHARE_EXPERT_DIM_LOCAL

# Re-export expert_shared.py's per-module constants into test-module scope
# so pl.inline can re-parse the body in this scope without losing
# GATE_K_CHUNK / GATE_N_CHUNK / DOWN_K_CHUNK / DOWN_N_CHUNK references.
_ = (GATE_K_CHUNK, GATE_N_CHUNK, DOWN_K_CHUNK, DOWN_N_CHUNK)


_expert_shared_silu_inline = pl.inline(expert_shared_silu._func)
_expert_shared_swiglu16_inline = pl.inline(expert_shared_swiglu16._func)


def _build_silu_program():
    @pl.program
    class ExpertSharedSiluTest:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            x: pl.Tensor[[T, HIDDEN], pl.BF16],
            w_gate: pl.Tensor[[HIDDEN, INTER], pl.BF16],
            w_up: pl.Tensor[[HIDDEN, INTER], pl.BF16],
            w_down: pl.Tensor[[INTER, HIDDEN], pl.BF16],
            sh_y_shard: pl.Out[pl.Tensor[[T, HIDDEN], pl.BF16]],
        ) -> pl.Tensor[[T, HIDDEN], pl.BF16]:
            sh_y_shard = _expert_shared_silu_inline(
                x, w_gate, w_up, w_down, sh_y_shard,
            )
            return sh_y_shard

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            x: pl.Tensor[[1, T, HIDDEN], pl.BF16],
            w_gate: pl.Tensor[[1, HIDDEN, INTER], pl.BF16],
            w_up: pl.Tensor[[1, HIDDEN, INTER], pl.BF16],
            w_down: pl.Tensor[[1, INTER, HIDDEN], pl.BF16],
            sh_y_shard: pl.Out[pl.Tensor[[1, T, HIDDEN], pl.BF16]],
        ):
            for r in pl.range(pld.world_size()):
                self.chip_orch(
                    x[r], w_gate[r], w_up[r], w_down[r], sh_y_shard[r],
                    device=r,
                )

    return ExpertSharedSiluTest


def _build_swiglu16_program():
    @pl.program
    class ExpertSharedSwiglu16Test:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            x: pl.Tensor[[T, HIDDEN], pl.BF16],
            w_gate: pl.Tensor[[HIDDEN, INTER], pl.BF16],
            w_up: pl.Tensor[[HIDDEN, INTER], pl.BF16],
            w_down: pl.Tensor[[INTER, HIDDEN], pl.BF16],
            sh_y_shard: pl.Out[pl.Tensor[[T, HIDDEN], pl.BF16]],
        ) -> pl.Tensor[[T, HIDDEN], pl.BF16]:
            sh_y_shard = _expert_shared_swiglu16_inline(
                x, w_gate, w_up, w_down, sh_y_shard,
            )
            return sh_y_shard

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            x: pl.Tensor[[1, T, HIDDEN], pl.BF16],
            w_gate: pl.Tensor[[1, HIDDEN, INTER], pl.BF16],
            w_up: pl.Tensor[[1, HIDDEN, INTER], pl.BF16],
            w_down: pl.Tensor[[1, INTER, HIDDEN], pl.BF16],
            sh_y_shard: pl.Out[pl.Tensor[[1, T, HIDDEN], pl.BF16]],
        ):
            for r in pl.range(pld.world_size()):
                self.chip_orch(
                    x[r], w_gate[r], w_up[r], w_down[r], sh_y_shard[r],
                    device=r,
                )

    return ExpertSharedSwiglu16Test


def _build_expert_shared_test_program(variant: str):
    if variant == "silu":
        return _build_silu_program()
    if variant == "swiglu16":
        return _build_swiglu16_program()
    raise ValueError(f"unknown variant {variant!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--variant", default="silu",
                        choices=["silu", "swiglu16"])
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _build_inputs(seed: int) -> dict[str, torch.Tensor]:
    """Per-rank LOCAL inputs wrapped in a [1, ...] leading dim for host_orch."""
    g = torch.Generator().manual_seed(seed)
    x = (
        torch.empty(1, T, HIDDEN, dtype=torch.float32)
        .normal_(mean=0.0, std=0.3, generator=g)
        .to(torch.bfloat16)
    )
    w_gate = (
        torch.empty(1, HIDDEN, INTER, dtype=torch.float32)
        .normal_(mean=0.0, std=HIDDEN ** -0.5, generator=g)
        .to(torch.bfloat16)
    )
    w_up = (
        torch.empty(1, HIDDEN, INTER, dtype=torch.float32)
        .normal_(mean=0.0, std=HIDDEN ** -0.5, generator=g)
        .to(torch.bfloat16)
    )
    w_down = (
        torch.empty(1, INTER, HIDDEN, dtype=torch.float32)
        .normal_(mean=0.0, std=INTER ** -0.5, generator=g)
        .to(torch.bfloat16)
    )
    return {"x": x, "w_gate": w_gate, "w_up": w_up, "w_down": w_down}


def _build_specs(inputs: dict[str, torch.Tensor]) -> list:
    from golden import TensorSpec  # noqa: PLC0415

    return [
        TensorSpec("x", list(inputs["x"].shape), torch.bfloat16,
                   init_value=inputs["x"]),
        TensorSpec("w_gate", list(inputs["w_gate"].shape), torch.bfloat16,
                   init_value=inputs["w_gate"]),
        TensorSpec("w_up", list(inputs["w_up"].shape), torch.bfloat16,
                   init_value=inputs["w_up"]),
        TensorSpec("w_down", list(inputs["w_down"].shape), torch.bfloat16,
                   init_value=inputs["w_down"]),
        TensorSpec("sh_y_shard", [1, T, HIDDEN], torch.bfloat16,
                   is_output=True),
    ]


def _golden_fn_for(swiglu_limit: float):
    def golden_fn(values: dict[str, torch.Tensor]) -> None:
        out = _torch_shared_local(
            swiglu_limit, values["x"][0], values["w_gate"][0],
            values["w_up"][0], values["w_down"][0],
        )
        values["sh_y_shard"][:] = out.unsqueeze(0).contiguous()

    return golden_fn


def main() -> int:
    args = _parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415

    set_backend_type({
        "a2a3": BackendType.Ascend910B,
        "a2a3sim": BackendType.Ascend910B,
        "a5": BackendType.Ascend950,
        "a5sim": BackendType.Ascend950,
    }[args.platform])

    print(
        f"[test_expert_shared] LOCAL: T={T} HIDDEN={HIDDEN} "
        f"INTER={INTER} (= SHARE_EXPERT_DIM_LOCAL) variant={args.variant}",
        flush=True,
    )

    program_cls = _build_expert_shared_test_program(args.variant)
    inputs = _build_inputs(args.seed)
    specs = _build_specs(inputs)
    swiglu_limit = 16.0 if args.variant == "swiglu16" else 0.0

    from golden import ratio_allclose, run  # noqa: PLC0415

    if args.smoke or args.platform.endswith("sim"):
        result = run(
            program=program_cls,
            specs=specs,
            runtime_cfg=dict(platform=args.platform, device_id=args.device),
            compile_only=True,
        )
        print(f"[test_expert_shared/{args.variant}] SMOKE: {result}", flush=True)
        return 0 if result.passed else 1

    result = run(
        program=program_cls,
        specs=specs,
        golden_fn=_golden_fn_for(swiglu_limit),
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=6e-3,
        atol=6e-3,
        # bf16 SwiGLU + 160-lane down matmul. Output ~0.3 scale; 6e-3
        # tolerance with 6% outlier cap matches the rest of the suite.
        compare_fn={
            "sh_y_shard": ratio_allclose(
                atol=6e-3, rtol=6e-3, max_error_ratio=0.06,
            ),
        },
    )
    print(f"[test_expert_shared/{args.variant}] DEVICE: {result}", flush=True)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
