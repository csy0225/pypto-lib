"""[中文摘要] Phase 14.D 编译验证驱动:把 `DecodeLayerMoE`(layer 3..44 用)单层
@pl.program 喂给 pypto.ir.compile,跑 EP+TP 8 卡 codegen。**不上 NPU**。
[关键装饰器] 无(纯 host Python 入口)。
[SPMD 角色] host 编译驱动。
[详见] 中文架构指南 §11

────── 以下为英文原 docstring ──────

Compile a step3p5 MoE decode layer (TP+EP=8) for codegen verification.

Compile-only driver — no NPU execution. Builds a multi-card
``DecodeLayerMoE`` ``@pl.program`` and runs it through ``pypto.ir.compile``
against the ``a2a3sim`` simulator backend.

Layer 3 is the first MoE layer (layers 0..2 are dense, see ``decode_layer``
LAYER_TYPES table).

Usage (from pypto-lib/):
    cd /data/chensiyu/hw_project/pypto/workspace/pypto-lib
    PYPTO_PROG_BUILD_DIR=/path/to/p14 \\
      python -m models.step3p5._compile_decode_layer_moe
"""
from __future__ import annotations

import argparse
import sys

from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

from .config import TP_WORLD_SIZE
from .decode_layer import select_decode_layer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", default="a2a3sim",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument(
        "--layer-idx", type=int, default=3,
        help="Layer index; first MoE layer is 3. Default: 3",
    )
    args = parser.parse_args()

    program, kind = select_decode_layer(args.layer_idx)
    prog_name = getattr(program, "name", None) or type(program).__name__
    print(f"[14.D] compiling layer {args.layer_idx} kind={kind} program={prog_name}",
          flush=True)

    dist_cfg = DistributedConfig(
        device_ids=list(range(TP_WORLD_SIZE)),
        num_sub_workers=0,
    )

    compiled = ir.compile(
        program,
        platform=args.platform,
        distributed_config=dist_cfg,
        skip_ptoas=False,
        dump_passes=True,
    )
    print(f"[14.D] OK output_dir={compiled.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
