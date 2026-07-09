# gap-5: in-kernel `tile.cast` → INT8 cube A-operand miscompiled (~98% wrong, no fault)

**Status (2026-07-09):** device-proven; survives pypto origin/main HEAD + ptoas-bin
v0.49 + pto-isa latest + simpler latest. **No upstream fix exists** (searched
`git log --all` for int8/quant/tcvt/cast/cube/w8a8/matmul_mx — none address this).
INT8-native routed matmul is gated OFF (`select_moe_block(..., w8a8_native=False)`);
the BF16-dequant path is the working production path (0.9995 vs vLLM).

## Summary
`pl.cast(<bf16/fp32 tile>, pl.INT8)` whose result feeds the A-operand of a
`tile.matmul_mx*` (`pto.tmatmul.mx*`) cube matmul is silently miscompiled:
~98.4% of outputs wrong, max diff ~254, **no device fault**. An INT8 operand
COPIED from global memory (pre-quantized in a separate program) works; any
in-kernel cast-to-INT8 into a cube A-operand fails.

## Minimal repro (single device, Ascend 910B), in pypto-lib:
- PASS control: `tests/step3p5/_probe_p2_downmirror.py` — INT8 A-operand from GM, down-tile T=32/K=1280/N=128.
- FAIL: `tests/step3p5/_probe_fixb_onthefly.py` — same shape, A-operand from in-kernel `pl.cast(bf16,int8)`.

## Root-cause localization (upstream-scout, 2026-07-09)
- `src/ir/transforms/infer_tile_memory_space_pass.cpp:55-56` — **primary**: `tile.matmul_mx*`
  is in `kUnregisteredCubeOps`, so the INT8 cube A-operand fractal/layout is never derived.
  INT8 cube fractal = 32 rows (BF16 = 16); a `pto.tcvt` output keeps a plain Vec layout →
  cube reads garbage rows. GM-copied int8 is pre-fractalized at authoring time → works.
- `src/backend/common/pto_ops_common.cpp:3382-3390` (`tile.cast`→`pto.tcvt` via MakeTileCvtCodegenPTO),
  `:2307-2312` (`tile.matmul_mx*`); `src/ir/op/tile_ops/unary.cpp:112-152` (DeduceTileCastType — no
  cube/fractal-layout constraint on INT8 cast output).

## Suggested fix (for upstream)
Add `tile.matmul_mx*` to memory-space inference, OR force a re-fractalize/pad on `tile.cast`
outputs whose target dtype is INT8 and whose sole consumer is a cube op.

## Why upstream CI misses it
DeepSeek-v4 W8A8 quantizes in a separate program and reads int8 from GM — never in-kernel
cast→cube. This path is unexercised in production/CI.
