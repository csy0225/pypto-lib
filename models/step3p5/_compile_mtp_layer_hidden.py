"""Compile-only smoke probe for the 3 baseline MTP hidden programs.

Mirrors ``_compile_prefill_layer_moe.py``: builds each
``MtpLayerHidden`` ``@pl.program`` (layer 0/1/2) and runs it through
``pypto.ir.compile`` against ``a2a3sim``. No NPU execution.

Goal: confirm the baseline MTP factory compiles clean on the current
perf2 base (independent of the canonical Main program). Decision (A): MTP
stays an independent program + host wiring, not merged into the Main program.

Usage (from pypto-lib/):
    PYPTO_PROG_BUILD_DIR=/tmp/mtp_smoke \\
      python -m models.step3p5._compile_mtp_layer_hidden -p a2a3sim
"""
from __future__ import annotations

import argparse
import sys

from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

from .config import TP_WORLD_SIZE
from .mtp_hidden_fwd import MTP_LAYER_HIDDEN_PROGRAMS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", default="a2a3sim",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    args = parser.parse_args()

    dist_cfg = DistributedConfig(
        device_ids=list(range(TP_WORLD_SIZE)),
        num_sub_workers=0,
    )

    for layer_idx, program in enumerate(MTP_LAYER_HIDDEN_PROGRAMS):
        prog_name = getattr(program, "name", None) or type(program).__name__
        print(f"[mtp-smoke] compiling layer {layer_idx} program={prog_name}",
              flush=True)
        compiled = ir.compile(
            program,
            platform=args.platform,
            distributed_config=dist_cfg,
            skip_ptoas=False,
            dump_passes=False,
        )
        print(f"[mtp-smoke] OK layer={layer_idx} output_dir={compiled.output_dir}",
              flush=True)

    print("=== mtp-smoke rc=0 ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
