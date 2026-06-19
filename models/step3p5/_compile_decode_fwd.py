"""[中文摘要] Phase 14.F 编译验证驱动:把 `Step3p5DecodeFwd` 顶层 @pl.program
喂给 pypto.ir.compile,跑完 frontend → IR → ptoas → distributed codegen
全链路,产物落 `$PYPTO_PROG_BUILD_DIR`。**不上 NPU**,只验证 codegen 路径。
[关键装饰器] 无(纯 host Python 入口)。
[SPMD 角色] host 编译驱动;构造 DistributedConfig 让 codegen 知道有 8 张卡。
[详见] 中文架构指南 §11

────── 以下为英文原 docstring ──────

Compile the step3p5 ``Step3p5DecodeFwd`` program (final RMSNorm + LM head).

Compile-only driver — no NPU execution. Builds the multi-card
``Step3p5DecodeFwd`` ``@pl.program`` from ``decode_fwd.py`` and runs it
through ``pypto.ir.compile`` against the ``a2a3sim`` simulator backend.

Phase 14.F target: the top-level decode forward pass program (45 layers
staged from host_orch via per-layer programs; this program compiles the
final RMSNorm + vocab-sliced LM head chip_orch plus the host orchestration
shell that routes per-rank calls).

Usage (from pypto-lib/):
    cd /data/chensiyu/hw_project/pypto/workspace/pypto-lib
    PYPTO_PROG_BUILD_DIR=/tmp/p14f \\
      python -m models.step3p5._compile_decode_fwd
"""
from __future__ import annotations

import argparse
import sys

from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

from .config import TP_WORLD_SIZE
from .decode_fwd import Step3p5DecodeFwd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", default="a2a3sim",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    args = parser.parse_args()

    prog_name = getattr(Step3p5DecodeFwd, "name", None) or type(Step3p5DecodeFwd).__name__
    print(f"[14.F] compiling Step3p5DecodeFwd program={prog_name}", flush=True)

    dist_cfg = DistributedConfig(
        device_ids=list(range(TP_WORLD_SIZE)),
        num_sub_workers=0,
    )

    compiled = ir.compile(
        Step3p5DecodeFwd,
        platform=args.platform,
        distributed_config=dist_cfg,
        skip_ptoas=False,
        dump_passes=True,
    )
    print(f"[14.F] OK output_dir={compiled.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
