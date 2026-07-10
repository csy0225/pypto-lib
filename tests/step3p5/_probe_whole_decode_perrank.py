# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Probe: compile the N=1 whole-decode dense-prefix program under per-rank patch.

Single-card per-rank compile test for ``whole_decode_dense_prefix`` — the ONE
``@pl.program`` running the dense decode prefix (L0 full + L1 swa + L2 swa) +
tail (final RMSNorm + LM head) threaded via resident ``pl.Out`` GM tensors.

Mirrors ``_probe_moe_compile_perrank.py``: builds under ``apply_perrank_patch()``
(TP=8 slice widths preserved, TP_WORLD_SIZE=EP_WORLD_SIZE=1) and runs
``ir.compile(..., platform="a2a3sim", skip_ptoas=False)``.

Expected outcome: ``COMPILE OK rc=0``. Failure modes to iterate on:
- const-fold "must be ConstInt" (host_orch window/loop bounds),
- UB/L1 overflow (per-layer scratch windows, dense_mlp chunks),
- name collisions / SSA scope errors,
- weight-index tensor shape mismatch vs inline signatures.
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
        "--layer-name", default="whole_decode_dense_prefix",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    from tests.step3p5._perrank_setup import apply_perrank_patch  # noqa: PLC0415

    summary = apply_perrank_patch(reload_modules=[
        "models.step3p5.attention_full",
        "models.step3p5.attention_swa",
        "models.step3p5.decode_layer",
    ])
    print(f"[probe_whole_decode_perrank] per-rank patch: {summary}", flush=True)

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
    print(f"[probe_whole_decode_perrank] resolved program={prog_name}",
          flush=True)

    dist_cfg = DistributedConfig(device_ids=[0], num_sub_workers=0)
    compiled = ir.compile(
        program,
        platform=args.platform,
        distributed_config=dist_cfg,
        skip_ptoas=False,
        dump_passes=False,
    )
    print(
        f"[probe_whole_decode_perrank] COMPILE OK output_dir={compiled.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
