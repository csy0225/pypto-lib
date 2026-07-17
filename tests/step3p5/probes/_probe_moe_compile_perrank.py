# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Probe: compile DecodeLayerMoE under PER-RANK patch (TP=8 widths preserved).

Single-card per-rank compile test. Uses ``apply_perrank_patch()`` which
flips TP_WORLD_SIZE/EP_WORLD_SIZE to 1 but KEEPS all ``*_LOCAL`` at
canonical TP=8 slice widths (160 / 1408 / 36 / 1 / 8). Validates the
"single-card ST/UT shape iron rule" from CLAUDE.md.

Expected outcome (post-iron-rule):
- sh_mlp / gate_matmul L1/UB overflow that previously surfaced under
  ``apply_tp1_patch`` (unsliced full widths) → should DISAPPEAR if the
  overflows were shape-choice artifacts.
- gate_topk codegen error (``pto.tci ui32 {descending=false}``) →
  expected to PERSIST (upstream PTOAS parser bug, shape-independent).
- Any overflow that PERSISTS at per-rank widths is a real issue
  (e.g. gate_matmul replicates N_EXPERTS=288 regardless of TP).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "-p", "--platform", default="a2a3sim",
        choices=["a2a3", "a2a3sim"],
    )
    p.add_argument(
        "--layer-name", default="decode_layer_full_moe_silu_silu",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))

    from tests.step3p5.common._perrank_setup import apply_perrank_patch  # noqa: PLC0415

    summary = apply_perrank_patch(reload_modules=[
        "models.step3p5.attention_full",
        "models.step3p5.attention_swa",
        "models.step3p5.decode_layer",
    ])
    print(f"[probe_moe_perrank] per-rank patch: {summary}", flush=True)

    from pypto import ir  # noqa: PLC0415
    from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415
    from pypto.ir.distributed_compiled_program import (  # noqa: PLC0415
        DistributedConfig,
    )

    set_backend_type({
        "a2a3": BackendType.Ascend910B,
        "a2a3sim": BackendType.Ascend910B,
    }[args.platform])

    import models.step3p5.decode_layer as decode_layer  # noqa: PLC0415

    program = getattr(decode_layer, args.layer_name)
    prog_name = getattr(program, "name", None) or type(program).__name__
    print(f"[probe_moe_perrank] resolved program={prog_name}", flush=True)

    dist_cfg = DistributedConfig(device_ids=[0], num_sub_workers=0)
    compiled = ir.compile(
        program,
        platform=args.platform,
        distributed_config=dist_cfg,
        skip_ptoas=False,
        dump_passes=False,
    )
    print(f"[probe_moe_perrank] COMPILE OK output_dir={compiled.output_dir}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
