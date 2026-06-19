"""[中文摘要] 最小 smoke probe:只构造 `Step3p5DecodeFwd` / `Step3p5PrefillFwd` /
各 per-layer @pl.program 类,**不进编译、不上 NPU**,只确认 frontend 能
解析当前代码;rc=0 是迁移期最重要的一根绿灯。
[关键装饰器] 无(纯 host Python;但 import 链会触发被 import 模块里的
@pl.jit / @pl.program 装饰器执行)。
[SPMD 角色] host 入口,只走 frontend 解析。
[详见] 中文架构指南 §11

────── 以下为英文原 docstring ──────

Smoke probe: instantiate Step3p5DecodeFwd / Step3p5PrefillFwd @pl.program

builders to confirm the pypto frontend can chew the Phase 9 codebase.
This does NOT touch the NPU — it only constructs the program objects so
we know the IR-build phase is sound before attempting compile/run.
"""
from __future__ import annotations

import sys
import traceback


def probe(name: str, factory):
    print(f"=== {name} ===", flush=True)
    try:
        prog = factory()
        print(f"  built: {type(prog).__name__}", flush=True)
        return True
    except Exception:  # noqa: BLE001
        print("  FAILED:", flush=True)
        traceback.print_exc()
        return False


def main() -> int:
    rc = 0
    # 1) Decode forward
    def _build_decode():
        from .decode_fwd import Step3p5DecodeFwd
        return Step3p5DecodeFwd

    if not probe("Step3p5DecodeFwd class", _build_decode):
        rc = 1

    # 2) Prefill forward
    def _build_prefill():
        from .prefill_fwd import Step3p5PrefillFwd
        return Step3p5PrefillFwd

    if not probe("Step3p5PrefillFwd class", _build_prefill):
        rc = 1

    # 3) per-layer programs via select_decode_layer
    def _build_layer(idx):
        from .decode_layer import select_decode_layer
        prog, kind = select_decode_layer(idx)
        print(f"    layer {idx}: kind={kind}, prog={prog}", flush=True)
        return (prog, kind)

    print("=== select_decode_layer probes ===", flush=True)
    for idx in (0, 1, 3, 4, 43, 44, 45, 47):
        try:
            _build_layer(idx)
        except Exception:  # noqa: BLE001
            print(f"  layer {idx} FAILED:", flush=True)
            traceback.print_exc()
            rc = 1

    # 4) MoE specialisation factory
    def _build_moe():
        from .moe import select_moe_block
        for idx in (3, 43, 44):
            cls = select_moe_block(idx)
            print(f"    moe layer {idx}: {cls}", flush=True)
        return True

    if not probe("select_moe_block", _build_moe):
        rc = 1

    print(f"=== probe rc={rc} ===", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
