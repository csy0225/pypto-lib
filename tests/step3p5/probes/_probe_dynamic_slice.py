"""B2 dynamic-offset pl.slice 设备探针.

目的
----
PERF-B2 计划把 `decode_layer_single_chip.py::whole_chip_orch`（chip_orch,
`@pl.function(type=Orchestration)`）里 42 层逐层 unroll 的静态 offset
`pl.slice(moe_w_gate_r, [n_local_experts*HIDDEN, inter], [K * (n_local_experts*HIDDEN), 0])`
（K=0,1,...,41）合并进一个 `for layer_idx in pl.range(42):` 循环, offset 改成
动态 scalar `layer_idx * (n_local_experts*HIDDEN)`.

在花大力气重写 B2 循环前, 本探针先证伪「`pl.slice` 接 dynamic scalar offset
在 step3p5 codegen + 设备上安全」. step3p5 的 `pl.dynamic`（leading dim）有丢
父 stride / 幻 int32 的前科（dev-constraints §C #119）; 虽然这里是静态 leading
dim + 动态 scalar offset（不动 stride）, 但仍需设备验证 codegen 正确下放
runtime GM base address.

本探针刻意最小独立: 一个 `@pl.program`, chip_orch body `for layer_idx in pl.range(42):`
+ `pl.cast(layer_idx * STRIDE, INT32)`（参考 DeepSeek v4 `decode_fwd.py:386-394`
的 `csa_layer = pl.cast(loop_i * 2 + 2, INT32)` 写法）+ `pl.slice` + 一个 trivial
InCore row_sum consume, 使 codegen 必须真正下放 dynamic-offset slice（read 端）
+ dynamic-offset Out slice（write 端, 对齐 B2 每层写 `routed_y_buf` 的形态）.

注意
----
- 本文件**不碰**共享生成器 `_gen_single_chip_real.py`（C1 本轮在改）.
- shape 用 scaled proxy（NUM_LAYERS / N_LOCAL_EXPERTS 取真值, HIDDEN/INTER 缩小）:
  生产 per-rank MoE routed gate 权重 = `[42*36*4096, 1408]` INT8 ≈ 8.1 GB, 设备探针
  跑不动. dynamic-offset `pl.slice` 的 codegen 路径与 shape 无关, proxy 等价.
- `pl.range` ≠ `pl.unroll`: `UnrollLoops` pass 只展开 `ForKind::Unroll`
  （`unroll_loops_pass.cpp:165`）, `pl.range(42)` 保留为 runtime loop, `layer_idx`
  是 runtime scalar → offset 真动态. 这是本探针要验证的路径.
- 设备验证前**不**下「安全」结论（falsify-before-assert）.

运行
----
- 本机前端 smoke（无 NPU, 只 build program）::

      python -m tests.step3p5.probes._probe_dynamic_slice

- 0162 设备 compile + RUN_CLEAN（team-lead 跑）::

      python tests/step3p5/probes/_probe_dynamic_slice.py -p a2a3 -d 8
"""
from __future__ import annotations

import argparse
import sys
import traceback

import pypto.language as pl

# ── per-rank shape 常量 ──────────────────────────────────────────────
# 生产值: NUM_LAYERS=42, N_LOCAL_EXPERTS=36, HIDDEN=4096, INTER=1408 (8.1 GB INT8).
# 探针 proxy: 保留 NUM_LAYERS / N_LOCAL_EXPERTS 真值, 缩小 HIDDEN / INTER 使设备可跑.
NUM_LAYERS = 42
N_LOCAL_EXPERTS = 36
HIDDEN = 16  # proxy (生产 4096)
INTER = 32  # proxy (生产 1408)
ROWS_PER_LAYER = N_LOCAL_EXPERTS * HIDDEN  # 576
TOTAL_ROWS = NUM_LAYERS * ROWS_PER_LAYER  # 24192


@pl.program
class DynamicSliceProbe:
    """单 @pl.program: chip_orch 用 pl.range(42) + dynamic-offset pl.slice."""

    @pl.function(type=pl.FunctionType.InCore)
    def layer_consume(
        self,
        w_slice: pl.Tensor[[ROWS_PER_LAYER, INTER], pl.INT8],
    ) -> pl.Tensor[[ROWS_PER_LAYER], pl.FP32]:
        # trivial consume: cast INT8->FP32, row_sum (reduce 内层 INTER).
        # 迫使 codegen 真正读取 dynamic-offset slice 并下放 runtime GM 地址.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="dyn_slice_row_sum"):
            w_fp32 = pl.cast(w_slice, pl.FP32)
            s = pl.row_sum(w_fp32)  # [ROWS_PER_LAYER, INTER] -> [ROWS_PER_LAYER]
        return s

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch(
        self,
        stacked_w: pl.Tensor[[TOTAL_ROWS, INTER], pl.INT8],
        out: pl.Out[pl.Tensor[[TOTAL_ROWS], pl.FP32]],
    ) -> pl.Tensor[[TOTAL_ROWS], pl.FP32]:
        # B2 核心形态: 单 pl.range(42) 循环 + dynamic scalar offset slice.
        # 参考 DeepSeek v4 decode_fwd.py:386-394 的
        #   csa_layer = pl.cast(loop_i * 2 + 2, INT32)
        #   pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [csa_layer * MIX_HC, 0])
        for layer_idx in pl.range(NUM_LAYERS):
            layer_offset: pl.Scalar[pl.INT32] = pl.cast(
                layer_idx * ROWS_PER_LAYER, pl.INT32
            )
            # read 端: dynamic-offset slice 取本层权重.
            w_slice: pl.Tensor[[ROWS_PER_LAYER, INTER], pl.INT8] = pl.slice(
                stacked_w, [ROWS_PER_LAYER, INTER], [layer_offset, 0]
            )
            # consume + 拿回本层 row_sum.
            s = self.layer_consume(w_slice)
            # write 端: dynamic-offset slice-assign 写本层输出槽
            # (对齐 B2 每层写 routed_y_buf; 参考 deepseek v4 combine.py:148
            #   ffn_out[t:t+1, :] = ...).
            out[layer_offset : layer_offset + ROWS_PER_LAYER] = s
        return out


def build_specs():
    """stacked_w INT8 + out FP32 (golden harness 输入规格)."""
    import torch
    from golden import TensorSpec

    return [
        TensorSpec("stacked_w", [TOTAL_ROWS, INTER], torch.int8),
        TensorSpec("out", [TOTAL_ROWS], torch.float32, is_output=True),
    ]


def golden_fn(values):
    """out[layer_offset : layer_offset+ROWS_PER_LAYER] = row_sum(cast(w_slice, FP32)).

    row_sum reduce 内层 INTER, 与 layer_consume kernel 一致.
    """
    import torch

    w = values["stacked_w"].to(torch.float32)  # [TOTAL_ROWS, INTER]
    row_sums = w.sum(dim=1)  # [TOTAL_ROWS]
    values["out"][:] = row_sums


def _frontend_smoke() -> int:
    """本机无 NPU: 只 build @pl.program 类, 确认 frontend 解析通过 (rc=0)."""
    rc = 0
    print("=== DynamicSliceProbe frontend smoke ===", flush=True)
    try:
        prog = DynamicSliceProbe  # noqa: F841  (装饰器在 import 时已 build IR)
        print(f"  built: {type(prog).__name__}", flush=True)
        print(
            f"  NUM_LAYERS={NUM_LAYERS} ROWS_PER_LAYER={ROWS_PER_LAYER} "
            f"TOTAL_ROWS={TOTAL_ROWS} INTER={INTER}",
            flush=True,
        )
    except Exception:  # noqa: BLE001
        print("  FAILED:", flush=True)
        traceback.print_exc()
        rc = 1
    print(f"=== probe rc={rc} ===", flush=True)
    return rc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--platform",
        type=str,
        default=None,
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=8)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    if args.platform is None:
        return _frontend_smoke()

    # 设备路径: golden.run 编译 + (可选)运行 + 校验.
    from golden import run

    result = run(
        program=DynamicSliceProbe,
        specs=build_specs(),
        golden_fn=golden_fn,
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        compile_only=args.compile_only,
        rtol=1e-5,
        atol=1e-5,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
