# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Probe: compile WholeDecodeFaithful under CANONICAL TP=8 + DistributedConfig.

Unlike ``_probe_whole_decode_perrank.py`` (which uses ``apply_perrank_patch``
to force TP=1 single-card compile), this probe compiles the module-level
``whole_decode_faithful`` program with NO patch — canonical TP=8/EP=8 — and
a real 8-card ``DistributedConfig(device_ids=[0..7])``.

Purpose (Wall-2 decisive experiment, compile stage): exercise pass-37
``materialize_comm_domain_scopes_pass`` at real TP=8/EP=8 to confirm the
per-protocol-separated program (TP attn method + EP MoE method, separate
dispatch passes) compiles at multi-rank. If this fails with a comm-domain /
TaskMapSize error at compile, Wall-2 is a compile-stage limit; if it compiles
clean, the next stage is an actual 8-card device dispatch.

    python -m tests.step3p5.probes._probe_whole_faithful_canonical \
        -p a2a3 --layer-name whole_decode_faithful -d 0,1,2,3,4,5,6,7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-p", "--platform", default="a2a3",
                   choices=["a2a3", "a2a3sim"])
    p.add_argument("--layer-name", default="whole_decode_faithful")
    p.add_argument("-d", "--device", default="0,1,2,3,4,5,6,7",
                   help="comma device_ids for DistributedConfig")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))

    device_ids = [int(d) for d in str(args.device).split(",")]

    from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415
    set_backend_type({
        "a2a3": BackendType.Ascend910B,
        "a2a3sim": BackendType.Ascend910B,
    }[args.platform])

    # Canonical TP=8 — NO apply_perrank_patch / apply_tp1_patch.
    import models.step3p5.config as cfg  # noqa: PLC0415
    print(f"[probe_canonical] TP_WORLD_SIZE={cfg.TP_WORLD_SIZE} "
          f"EP_WORLD_SIZE={cfg.EP_WORLD_SIZE} device_ids={device_ids}",
          flush=True)

    import models.step3p5.decode_layer as decode_layer  # noqa: PLC0415

    program = getattr(decode_layer, args.layer_name)
    prog_name = getattr(program, "name", None) or type(program).__name__
    print(f"[probe_canonical] resolved program={prog_name}", flush=True)

    from pypto import ir  # noqa: PLC0415
    from pypto.ir.distributed_compiled_program import (  # noqa: PLC0415
        DistributedConfig,
    )

    dist_cfg = DistributedConfig(device_ids=device_ids, num_sub_workers=0)
    compiled = ir.compile(
        program,
        platform=args.platform,
        distributed_config=dist_cfg,
        skip_ptoas=False,
        dump_passes=False,
    )
    print(
        f"[probe_canonical] COMPILE OK output_dir={compiled.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
