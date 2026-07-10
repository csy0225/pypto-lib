# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Probe: compile the whole-decode DENSE-PREFIX chain (N=1 fusion milestone).

Builds a whole-decode fused ``@pl.program`` (default ``DenseChain3LmHead`` =
3 dense layers + tail in ONE program) under the per-rank patch (TP=8 slice
widths, single-card ST/UT iron rule) and runs the full compile
(frontend -> IR passes -> ptoas -> distributed codegen). This is the first
N=1 whole-network-fusion milestone: prove multi-layer chaining in ONE program
compiles on the upgraded stack (pypto 5e619dc7 + ptoas v0.45, with the
OUT_PROJ_N_CHUNK=64 tmov fix).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-p", "--platform", default="a2a3sim", choices=["a2a3", "a2a3sim"])
    p.add_argument("--builder", default="_build_dense_chain3_lmhead_program")
    p.add_argument("--full", action="store_true", default=True)
    p.add_argument("--swa", action="store_true", help="force full=False (swa attention)")
    p.add_argument("--routed-lim", type=float, default=0.0)
    p.add_argument("--shared-lim", type=float, default=0.0)
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
    print(f"[probe_chain3] per-rank patch: {summary}", flush=True)

    from pypto import ir  # noqa: PLC0415
    from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415
    from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415

    set_backend_type({
        "a2a3": BackendType.Ascend910B,
        "a2a3sim": BackendType.Ascend910B,
    }[args.platform])

    import models.step3p5.decode_layer as decode_layer  # noqa: PLC0415

    builder = getattr(decode_layer, args.builder)
    import inspect  # noqa: PLC0415
    params = inspect.signature(builder).parameters
    if "full" in params:
        program = builder(full=(not args.swa), routed_lim=args.routed_lim, shared_lim=args.shared_lim)
    else:
        program = builder()
    prog_name = getattr(program, "name", None) or type(program).__name__
    print(f"[probe_chain3] resolved builder={args.builder} program={prog_name}", flush=True)

    dist_cfg = DistributedConfig(device_ids=[0], num_sub_workers=0)
    compiled = ir.compile(
        program,
        platform=args.platform,
        distributed_config=dist_cfg,
        skip_ptoas=False,
        dump_passes=False,
    )
    print(f"[probe_chain3] COMPILE OK output_dir={compiled.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
