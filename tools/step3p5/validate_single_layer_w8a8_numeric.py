#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Validate step3p5 prefill single-layer W8A8 MoE numeric correctness on NPU.

Approach C (synthetic W8A8 round-trip). The production checkpoint at
``DEFAULT_CKPT_DIR`` (models/step3p5/weight_loader.py) is BF16-only -- no
per-expert INT8 routed weights, no ``_scale``/``_offset`` tensors, no
``quant_model_weights.safetensors.index.json`` -- so a real-ckpt on-device
comparison (Approach B) is impossible and load_step3p5_weights_for_rank(
int8_routed=True) raises ValueError.

This script ports the W8A8 *math* of ``PrefillLayerMoE._expert_routed``
(models/step3p5/prefill_fwd.py:1748-2082, commit e885c46) into a standalone
single-card ``@pl.jit.inline`` program and checks the NPU output against a
faithful torch W8A8 reference via ``golden.run_jit``.

Orchestration note (IMPORTANT): prefill_fwd._expert_routed uses a 2-D
no-``b_trans`` INT8 matmul with ``[in,out]`` weight layout and FUSED
matmul+dequant+activation spmd scopes. As a STANDALONE ``@pl.jit.inline``
kernel that form faults the AICPU orchestration with retCode=0x2a
(ACL_ERROR_RT_AICPU_EXCEPTION) at execute time (compile-clean: no codegen
errors, only PH001 perf hints, 13% Mat memory -- not OOM). The proven
single-card W8A8 pattern in models/deepseek/v4/expert_routed.py uses 3-D
``b_trans=True`` INT8 matmul with ``[out,in]`` weight layout and SEPARATED
spmd scopes (cube matmul distinct from VEC activation, GM INT32 buffers
between them). This script adopts the deepseek-proven orchestration, which
computes the IDENTICAL W8A8 linear projection (b_trans with [out,in] storage
is the transpose of no-b_trans with [in,out] storage -- same matmul result).
The W8A8 MATH validated here is faithful to prefill_fwd:

  * INT8 x INT8 -> INT32 matmul (exact integer accumulation, associative
    across K-chunks; golden reproduces exact INT32 via float64).
  * FP32 dequant: (INT32 acc) * per-token x_scale * per-channel w_scale.
  * silu = gate * 1/(1+exp(-gate)); SwigluStep: silu computed FIRST, then
    ``silu_c = min(silu, L)`` (upper-clamp only on the SILU output, NOT on
    the gate input -- this is the step3p5 order, prefill_fwd.py:1911-1917,
    and differs from deepseek v4 which clamps the gate before silu).
    ``up_c = clamp(up, -L, L)``, ``gated = silu_c * up_c``.
    Layer 4 = plain SiLU (L=0); layer 44 = SwigluStep@7.
  * Intermediate ``gated`` cast to BF16 (``h_bf16``) BEFORE the per-token
    INT8 requant (golden round-trips BF16 identically; prefill_fwd.py:1926).
  * Requant: amax (init 1e-4) over h_bf16 rows, scale=127/amax,
    FP32->INT32 (rint) -> FP16 (round) -> INT8 (trunc) (prefill_fwd.py:1941).
  * NO route-weight folding (prefill combine_step applies expert_weights via
    inverse_map; prefill_fwd.py:2053-2056 comment confirms NO fold).

Constants match models/step3p5/config.py (HIDDEN=4096, MOE_INTERMEDIATE=1280,
RECV_TILE=32). N_LOCAL_EXPERTS/RECV_MAX are reduced from 36/1024 to 4/128 for
compile tractability; per-expert math is identical regardless of expert count.

Usage:
  python tools/step3p5/validate_single_layer_w8a8_numeric.py -p a2a3 -d 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pypto.language as pl  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (match models/step3p5/config.py; matmul tile sizes are perf/tiling
# params, not math params -- INT32 accumulation is exact regardless of K-tile).
# ---------------------------------------------------------------------------
HIDDEN = 4096                 # config.HIDDEN
INTER = 1280                  # config.MOE_INTERMEDIATE_SIZE
N_LOCAL_EXPERTS = 4           # reduced from config.MOE_NUM_EXPERTS_LOCAL=36
RECV_MAX = 128                # per-expert recv slab (reduced from 1024 shared)
RECV_TILE = 32                # prefill_fwd RECV_TILE

# Cube matmul tile sizes (divide HIDDEN/INTER cleanly; exact INT32 math
# independent of these). b_trans=True: weights stored [out, in] (deepseek v4).
GATE_K_TILE = 512             # HIDDEN // 512 = 8 K-chunks
GATE_N_TILE = 64              # INTER // 64 = 20 N-blocks
DOWN_K_TILE = 128             # INTER // 128 = 10 K-chunks (MUST be even:
                              # pl.pipeline stage=2 with an odd iteration count
                              # drops the last partial accumulation -- confirmed
                              # by minimal reproducer; 256 -> 5 chunks FAILS)
DOWN_N_TILE = 128             # HIDDEN // 128 = 32 N-blocks
ACT_N_TILE = GATE_N_TILE      # activation/dequant N-block
QUANT_N_TILE = GATE_N_TILE    # requant amax/quant N-block

INT8_AMAX_EPS = 1e-4          # prefill_fwd eh_amax init (line 1942)
INT8_SCALE_MAX = 127.0

SWIGLU_LIMIT_LAYER4 = 0.0     # config.SWIGLU_LIMITS: layer 4 = (0.0, 0.0)
SWIGLU_LIMIT_LAYER44 = 7.0    # layer 44 = (7.0, 16.0) -> SwigluStep@7 routed

_EXPERT_COUNTS = [128, 96, 64, 48]   # expert 3: 48 = 1 full + 1 partial(16) tile


# ===========================================================================
# Standalone single-card W8A8 routed-expert kernels.
# Structure: models/deepseek/v4/expert_routed.py (proven on a2a3 device 0).
# W8A8 math: faithful to prefill_fwd.PrefillLayerMoE._expert_routed.
# ===========================================================================

@pl.jit.inline
def expert_routed_silu(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, HIDDEN], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    w_gate: pl.Tensor[[N_LOCAL_EXPERTS, INTER, HIDDEN], pl.INT8],
    w_gate_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
    w_up: pl.Tensor[[N_LOCAL_EXPERTS, INTER, HIDDEN], pl.INT8],
    w_up_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
    w_down: pl.Tensor[[N_LOCAL_EXPERTS, HIDDEN, INTER], pl.INT8],
    w_down_scale: pl.Tensor[[N_LOCAL_EXPERTS, HIDDEN], pl.FP32],
    recv_y: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, HIDDEN], pl.BF16],
):
    recv_y_flat = pl.reshape(recv_y, [N_LOCAL_EXPERTS * RECV_MAX, HIDDEN])

    for local_i in pl.parallel(N_LOCAL_EXPERTS):
        n_rows = pl.read(recv_expert_count, [local_i, 0])
        n_tiles = (n_rows + RECV_TILE - 1) // RECV_TILE
        flat_base = local_i * RECV_MAX

        for t in pl.parallel(n_tiles):
            t0 = t * RECV_TILE
            flat_t0 = flat_base + t0
            valid_rows = pl.min(RECV_TILE, n_rows - t0)

            h_bf16 = pl.create_tensor([RECV_TILE, INTER], dtype=pl.BF16)
            gate_i32 = pl.create_tensor([RECV_TILE, INTER], dtype=pl.INT32)
            up_i32 = pl.create_tensor([RECV_TILE, INTER], dtype=pl.INT32)

            # gate matmul: INT8 x INT8 -> INT32, b_trans (w [out,in]).
            with pl.spmd(INTER // GATE_N_TILE, name_hint="exp_gate_mm"):
                nb_idx = pl.tile.get_block_idx()
                n0 = nb_idx * GATE_N_TILE
                gate_acc = pl.create_tensor([1, RECV_TILE, GATE_N_TILE], dtype=pl.INT32)
                for k0 in pl.pipeline(0, HIDDEN, GATE_K_TILE, stage=2):
                    x_k = recv_x[local_i : local_i + 1, t0 : t0 + RECV_TILE, k0 : k0 + GATE_K_TILE]
                    w1_k = w_gate[local_i : local_i + 1, n0 : n0 + GATE_N_TILE, k0 : k0 + GATE_K_TILE]
                    if k0 == 0:
                        gate_acc = pl.matmul(x_k, w1_k, b_trans=True, out_dtype=pl.INT32)
                    else:
                        gate_acc = pl.matmul_acc(gate_acc, x_k, w1_k, b_trans=True)
                gate_i32[:, n0 : n0 + GATE_N_TILE] = pl.reshape(gate_acc, [RECV_TILE, GATE_N_TILE])

            # up matmul.
            with pl.spmd(INTER // GATE_N_TILE, name_hint="exp_up_mm"):
                nb_idx = pl.tile.get_block_idx()
                n0 = nb_idx * GATE_N_TILE
                up_acc = pl.create_tensor([1, RECV_TILE, GATE_N_TILE], dtype=pl.INT32)
                for k0 in pl.pipeline(0, HIDDEN, GATE_K_TILE, stage=2):
                    x_k = recv_x[local_i : local_i + 1, t0 : t0 + RECV_TILE, k0 : k0 + GATE_K_TILE]
                    w3_k = w_up[local_i : local_i + 1, n0 : n0 + GATE_N_TILE, k0 : k0 + GATE_K_TILE]
                    if k0 == 0:
                        up_acc = pl.matmul(x_k, w3_k, b_trans=True, out_dtype=pl.INT32)
                    else:
                        up_acc = pl.matmul_acc(up_acc, x_k, w3_k, b_trans=True)
                up_i32[:, n0 : n0 + GATE_N_TILE] = pl.reshape(up_acc, [RECV_TILE, GATE_N_TILE])

            # activation: dequant INT32->FP32, silu, -> h_bf16 (BF16 round-trip).
            with pl.spmd(INTER // ACT_N_TILE, name_hint="exp_gate_up_act"):
                nb_idx = pl.tile.get_block_idx()
                n0 = nb_idx * ACT_N_TILE
                gate_2d_i32 = gate_i32[:, n0 : n0 + ACT_N_TILE]
                up_2d_i32 = up_i32[:, n0 : n0 + ACT_N_TILE]
                x_scale_dq = pl.reshape(
                    recv_scale_dq[local_i : local_i + 1, t0 : t0 + RECV_TILE], [RECV_TILE, 1],
                )
                w1_scale_chunk = w_gate_scale[local_i : local_i + 1, n0 : n0 + ACT_N_TILE]
                w3_scale_chunk = w_up_scale[local_i : local_i + 1, n0 : n0 + ACT_N_TILE]
                gate_2d = pl.cast(gate_2d_i32, target_type=pl.FP32, mode="none")
                up_2d = pl.cast(up_2d_i32, target_type=pl.FP32, mode="none")
                gate_2d = pl.col_expand_mul(pl.row_expand_mul(gate_2d, x_scale_dq), w1_scale_chunk)
                up_2d = pl.col_expand_mul(pl.row_expand_mul(up_2d, x_scale_dq), w3_scale_chunk)
                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_2d)), 1.0))
                silu = pl.mul(gate_2d, sigmoid)
                gated = pl.mul(silu, up_2d)
                gated_valid = pl.set_validshape(gated, valid_rows, ACT_N_TILE)
                gated_masked = pl.fillpad(gated_valid, pad_value=pl.PadValue.zero)
                h_bf16[:, n0 : n0 + ACT_N_TILE] = pl.cast(gated_masked, target_type=pl.BF16)

            # Per-token INT8 requant of the SwiGLU intermediate (CORE_GROUP).
            h_i8 = pl.create_tensor([RECV_TILE, INTER], dtype=pl.INT8)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_h_q"):
                eh_amax = pl.full([1, RECV_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                for k0 in pl.range(INTER // QUANT_N_TILE):
                    hqa0 = k0 * QUANT_N_TILE
                    eh_a = pl.cast(h_bf16[:, hqa0 : hqa0 + QUANT_N_TILE], target_type=pl.FP32)
                    eh_amax = pl.maximum(
                        eh_amax,
                        pl.reshape(pl.row_max(pl.maximum(eh_a, pl.neg(eh_a))), [1, RECV_TILE]),
                    )
                eh_sq_row = pl.mul(
                    pl.recip(eh_amax),
                    pl.full([1, RECV_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX),
                )
                h_scale_dq = pl.reshape(pl.recip(eh_sq_row), [RECV_TILE, 1])
                eh_sq_col = pl.reshape(eh_sq_row, [RECV_TILE, 1])
                for k0 in pl.range(INTER // QUANT_N_TILE):
                    hqn0 = k0 * QUANT_N_TILE
                    eh_q = pl.cast(h_bf16[:, hqn0 : hqn0 + QUANT_N_TILE], target_type=pl.FP32)
                    eh_scaled = pl.row_expand_mul(eh_q, eh_sq_col)
                    eh_i32 = pl.cast(eh_scaled, target_type=pl.INT32, mode="rint")
                    eh_half = pl.cast(eh_i32, target_type=pl.FP16, mode="round")
                    h_i8[:, hqn0 : hqn0 + QUANT_N_TILE] = pl.cast(eh_half, target_type=pl.INT8, mode="trunc")

            # down matmul: INT8 h_i8 x INT8 w_down -> INT32, b_trans.
            y_i32 = pl.create_tensor([RECV_TILE, HIDDEN], dtype=pl.INT32)
            with pl.spmd(HIDDEN // DOWN_N_TILE, name_hint="exp_w2_mm"):
                nb_idx = pl.tile.get_block_idx()
                d0 = nb_idx * DOWN_N_TILE
                y_acc = pl.create_tensor([1, RECV_TILE, DOWN_N_TILE], dtype=pl.INT32)
                for k0 in pl.pipeline(0, INTER, DOWN_K_TILE, stage=2):
                    h_k = h_i8[:, k0 : k0 + DOWN_K_TILE]
                    w2_k = w_down[local_i : local_i + 1, d0 : d0 + DOWN_N_TILE, k0 : k0 + DOWN_K_TILE]
                    if k0 == 0:
                        y_acc = pl.matmul(h_k, w2_k, b_trans=True, out_dtype=pl.INT32)
                    else:
                        y_acc = pl.matmul_acc(y_acc, h_k, w2_k, b_trans=True)
                y_i32[:, d0 : d0 + DOWN_N_TILE] = pl.reshape(y_acc, [RECV_TILE, DOWN_N_TILE])

            # down dequant: INT32 -> FP32 (h_scale_dq * w_down_scale) -> BF16, NO route weight.
            recv_y_tile = pl.create_tensor([RECV_TILE, HIDDEN], dtype=pl.BF16)
            with pl.spmd(HIDDEN // DOWN_N_TILE, name_hint="exp_w2_act"):
                nb_idx = pl.tile.get_block_idx()
                d0 = nb_idx * DOWN_N_TILE
                y_2d_i32 = y_i32[:, d0 : d0 + DOWN_N_TILE]
                w2_scale_chunk = w_down_scale[local_i : local_i + 1, d0 : d0 + DOWN_N_TILE]
                y_2d = pl.cast(y_2d_i32, target_type=pl.FP32, mode="none")
                y_2d = pl.col_expand_mul(pl.row_expand_mul(y_2d, h_scale_dq), w2_scale_chunk)
                y_v = pl.set_validshape(y_2d, valid_rows, DOWN_N_TILE)
                y_m = pl.fillpad(y_v, pad_value=pl.PadValue.zero)
                recv_y_tile[:, d0 : d0 + DOWN_N_TILE] = pl.cast(y_m, target_type=pl.BF16)
            recv_y_flat = pl.assemble(recv_y_flat, recv_y_tile, [flat_t0, 0])

    return recv_y


@pl.jit.inline
def expert_routed_swiglu7(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, HIDDEN], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    w_gate: pl.Tensor[[N_LOCAL_EXPERTS, INTER, HIDDEN], pl.INT8],
    w_gate_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
    w_up: pl.Tensor[[N_LOCAL_EXPERTS, INTER, HIDDEN], pl.INT8],
    w_up_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
    w_down: pl.Tensor[[N_LOCAL_EXPERTS, HIDDEN, INTER], pl.INT8],
    w_down_scale: pl.Tensor[[N_LOCAL_EXPERTS, HIDDEN], pl.FP32],
    recv_y: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, HIDDEN], pl.BF16],
):
    """Layer 44 variant: SwigluStep@7 routed (silu first, then clamp silu)."""
    recv_y_flat = pl.reshape(recv_y, [N_LOCAL_EXPERTS * RECV_MAX, HIDDEN])

    for local_i in pl.parallel(N_LOCAL_EXPERTS):
        n_rows = pl.read(recv_expert_count, [local_i, 0])
        n_tiles = (n_rows + RECV_TILE - 1) // RECV_TILE
        flat_base = local_i * RECV_MAX

        for t in pl.parallel(n_tiles):
            t0 = t * RECV_TILE
            flat_t0 = flat_base + t0
            valid_rows = pl.min(RECV_TILE, n_rows - t0)

            h_bf16 = pl.create_tensor([RECV_TILE, INTER], dtype=pl.BF16)
            gate_i32 = pl.create_tensor([RECV_TILE, INTER], dtype=pl.INT32)
            up_i32 = pl.create_tensor([RECV_TILE, INTER], dtype=pl.INT32)

            with pl.spmd(INTER // GATE_N_TILE, name_hint="exp_gate_mm"):
                nb_idx = pl.tile.get_block_idx()
                n0 = nb_idx * GATE_N_TILE
                gate_acc = pl.create_tensor([1, RECV_TILE, GATE_N_TILE], dtype=pl.INT32)
                for k0 in pl.pipeline(0, HIDDEN, GATE_K_TILE, stage=2):
                    x_k = recv_x[local_i : local_i + 1, t0 : t0 + RECV_TILE, k0 : k0 + GATE_K_TILE]
                    w1_k = w_gate[local_i : local_i + 1, n0 : n0 + GATE_N_TILE, k0 : k0 + GATE_K_TILE]
                    if k0 == 0:
                        gate_acc = pl.matmul(x_k, w1_k, b_trans=True, out_dtype=pl.INT32)
                    else:
                        gate_acc = pl.matmul_acc(gate_acc, x_k, w1_k, b_trans=True)
                gate_i32[:, n0 : n0 + GATE_N_TILE] = pl.reshape(gate_acc, [RECV_TILE, GATE_N_TILE])

            with pl.spmd(INTER // GATE_N_TILE, name_hint="exp_up_mm"):
                nb_idx = pl.tile.get_block_idx()
                n0 = nb_idx * GATE_N_TILE
                up_acc = pl.create_tensor([1, RECV_TILE, GATE_N_TILE], dtype=pl.INT32)
                for k0 in pl.pipeline(0, HIDDEN, GATE_K_TILE, stage=2):
                    x_k = recv_x[local_i : local_i + 1, t0 : t0 + RECV_TILE, k0 : k0 + GATE_K_TILE]
                    w3_k = w_up[local_i : local_i + 1, n0 : n0 + GATE_N_TILE, k0 : k0 + GATE_K_TILE]
                    if k0 == 0:
                        up_acc = pl.matmul(x_k, w3_k, b_trans=True, out_dtype=pl.INT32)
                    else:
                        up_acc = pl.matmul_acc(up_acc, x_k, w3_k, b_trans=True)
                up_i32[:, n0 : n0 + GATE_N_TILE] = pl.reshape(up_acc, [RECV_TILE, GATE_N_TILE])

            with pl.spmd(INTER // ACT_N_TILE, name_hint="exp_gate_up_act"):
                nb_idx = pl.tile.get_block_idx()
                n0 = nb_idx * ACT_N_TILE
                gate_2d_i32 = gate_i32[:, n0 : n0 + ACT_N_TILE]
                up_2d_i32 = up_i32[:, n0 : n0 + ACT_N_TILE]
                x_scale_dq = pl.reshape(
                    recv_scale_dq[local_i : local_i + 1, t0 : t0 + RECV_TILE], [RECV_TILE, 1],
                )
                w1_scale_chunk = w_gate_scale[local_i : local_i + 1, n0 : n0 + ACT_N_TILE]
                w3_scale_chunk = w_up_scale[local_i : local_i + 1, n0 : n0 + ACT_N_TILE]
                gate_2d = pl.cast(gate_2d_i32, target_type=pl.FP32, mode="none")
                up_2d = pl.cast(up_2d_i32, target_type=pl.FP32, mode="none")
                gate_2d = pl.col_expand_mul(pl.row_expand_mul(gate_2d, x_scale_dq), w1_scale_chunk)
                up_2d = pl.col_expand_mul(pl.row_expand_mul(up_2d, x_scale_dq), w3_scale_chunk)
                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_2d)), 1.0))
                silu = pl.mul(gate_2d, sigmoid)
                # step3p5 SwigluStep: clamp silu OUTPUT (upper-only), NOT gate.
                silu_c = pl.minimum(silu, SWIGLU_LIMIT_LAYER44)
                up_c = pl.maximum(pl.minimum(up_2d, SWIGLU_LIMIT_LAYER44), -SWIGLU_LIMIT_LAYER44)
                gated = pl.mul(silu_c, up_c)
                gated_valid = pl.set_validshape(gated, valid_rows, ACT_N_TILE)
                gated_masked = pl.fillpad(gated_valid, pad_value=pl.PadValue.zero)
                h_bf16[:, n0 : n0 + ACT_N_TILE] = pl.cast(gated_masked, target_type=pl.BF16)

            h_i8 = pl.create_tensor([RECV_TILE, INTER], dtype=pl.INT8)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_h_q"):
                eh_amax = pl.full([1, RECV_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                for k0 in pl.range(INTER // QUANT_N_TILE):
                    hqa0 = k0 * QUANT_N_TILE
                    eh_a = pl.cast(h_bf16[:, hqa0 : hqa0 + QUANT_N_TILE], target_type=pl.FP32)
                    eh_amax = pl.maximum(
                        eh_amax,
                        pl.reshape(pl.row_max(pl.maximum(eh_a, pl.neg(eh_a))), [1, RECV_TILE]),
                    )
                eh_sq_row = pl.mul(
                    pl.recip(eh_amax),
                    pl.full([1, RECV_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX),
                )
                h_scale_dq = pl.reshape(pl.recip(eh_sq_row), [RECV_TILE, 1])
                eh_sq_col = pl.reshape(eh_sq_row, [RECV_TILE, 1])
                for k0 in pl.range(INTER // QUANT_N_TILE):
                    hqn0 = k0 * QUANT_N_TILE
                    eh_q = pl.cast(h_bf16[:, hqn0 : hqn0 + QUANT_N_TILE], target_type=pl.FP32)
                    eh_scaled = pl.row_expand_mul(eh_q, eh_sq_col)
                    eh_i32 = pl.cast(eh_scaled, target_type=pl.INT32, mode="rint")
                    eh_half = pl.cast(eh_i32, target_type=pl.FP16, mode="round")
                    h_i8[:, hqn0 : hqn0 + QUANT_N_TILE] = pl.cast(eh_half, target_type=pl.INT8, mode="trunc")

            y_i32 = pl.create_tensor([RECV_TILE, HIDDEN], dtype=pl.INT32)
            with pl.spmd(HIDDEN // DOWN_N_TILE, name_hint="exp_w2_mm"):
                nb_idx = pl.tile.get_block_idx()
                d0 = nb_idx * DOWN_N_TILE
                y_acc = pl.create_tensor([1, RECV_TILE, DOWN_N_TILE], dtype=pl.INT32)
                for k0 in pl.pipeline(0, INTER, DOWN_K_TILE, stage=2):
                    h_k = h_i8[:, k0 : k0 + DOWN_K_TILE]
                    w2_k = w_down[local_i : local_i + 1, d0 : d0 + DOWN_N_TILE, k0 : k0 + DOWN_K_TILE]
                    if k0 == 0:
                        y_acc = pl.matmul(h_k, w2_k, b_trans=True, out_dtype=pl.INT32)
                    else:
                        y_acc = pl.matmul_acc(y_acc, h_k, w2_k, b_trans=True)
                y_i32[:, d0 : d0 + DOWN_N_TILE] = pl.reshape(y_acc, [RECV_TILE, DOWN_N_TILE])

            recv_y_tile = pl.create_tensor([RECV_TILE, HIDDEN], dtype=pl.BF16)
            with pl.spmd(HIDDEN // DOWN_N_TILE, name_hint="exp_w2_act"):
                nb_idx = pl.tile.get_block_idx()
                d0 = nb_idx * DOWN_N_TILE
                y_2d_i32 = y_i32[:, d0 : d0 + DOWN_N_TILE]
                w2_scale_chunk = w_down_scale[local_i : local_i + 1, d0 : d0 + DOWN_N_TILE]
                y_2d = pl.cast(y_2d_i32, target_type=pl.FP32, mode="none")
                y_2d = pl.col_expand_mul(pl.row_expand_mul(y_2d, h_scale_dq), w2_scale_chunk)
                y_v = pl.set_validshape(y_2d, valid_rows, DOWN_N_TILE)
                y_m = pl.fillpad(y_v, pad_value=pl.PadValue.zero)
                recv_y_tile[:, d0 : d0 + DOWN_N_TILE] = pl.cast(y_m, target_type=pl.BF16)
            recv_y_flat = pl.assemble(recv_y_flat, recv_y_tile, [flat_t0, 0])

    return recv_y


@pl.jit
def expert_routed_silu_test(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, HIDDEN], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    w_gate: pl.Tensor[[N_LOCAL_EXPERTS, INTER, HIDDEN], pl.INT8],
    w_gate_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
    w_up: pl.Tensor[[N_LOCAL_EXPERTS, INTER, HIDDEN], pl.INT8],
    w_up_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
    w_down: pl.Tensor[[N_LOCAL_EXPERTS, HIDDEN, INTER], pl.INT8],
    w_down_scale: pl.Tensor[[N_LOCAL_EXPERTS, HIDDEN], pl.FP32],
    recv_y: pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, HIDDEN], pl.BF16]],
):
    expert_routed_silu(
        recv_x, recv_scale_dq, recv_expert_count,
        w_gate, w_gate_scale, w_up, w_up_scale,
        w_down, w_down_scale, recv_y,
    )
    return recv_y


@pl.jit
def expert_routed_swiglu7_test(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, HIDDEN], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    w_gate: pl.Tensor[[N_LOCAL_EXPERTS, INTER, HIDDEN], pl.INT8],
    w_gate_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
    w_up: pl.Tensor[[N_LOCAL_EXPERTS, INTER, HIDDEN], pl.INT8],
    w_up_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
    w_down: pl.Tensor[[N_LOCAL_EXPERTS, HIDDEN, INTER], pl.INT8],
    w_down_scale: pl.Tensor[[N_LOCAL_EXPERTS, HIDDEN], pl.FP32],
    recv_y: pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, HIDDEN], pl.BF16]],
):
    expert_routed_swiglu7(
        recv_x, recv_scale_dq, recv_expert_count,
        w_gate, w_gate_scale, w_up, w_up_scale,
        w_down, w_down_scale, recv_y,
    )
    return recv_y


# ===========================================================================
# Torch W8A8 reference (golden) + synthetic data generation.
# ===========================================================================

def _int8_quant_per_row(x):
    """Per-row (per-token) INT8 symmetric quant matching prefill dispatch.

    Returns (i8, scale_dequant) where scale_dequant = amax/127 so that
    i8.float() * scale_dequant reconstructs the FP value.
    """
    import torch
    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)


def gen_w8a8_weight(shape, dequant_std):
    """Synthesize a per-output-channel symmetric INT8 weight + FP32 scale.

    ``shape`` = [..., out, in]; scale shape = [..., out]. A plain randn weight
    is sufficient for numeric validation (we are not reproducing the real MXFP4
    grid; we are checking the W8A8 round-trip math). ``dequant_std`` sets the
    absolute magnitude of the dequantized weight.
    """
    import torch
    *lead, out, inn = shape
    w_fp = torch.randn(*shape, dtype=torch.float32)
    amax = w_fp.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    chan_scale = amax / INT8_SCALE_MAX
    w_i8 = torch.round(w_fp / chan_scale).clamp(-INT8_SCALE_MAX, INT8_SCALE_MAX).to(torch.int8)
    # Rescale so the dequantized weight has the target std.
    cur_std = (w_i8.float() * chan_scale).std()
    chan_scale = chan_scale * (dequant_std / cur_std.clamp_min(1e-12))
    return w_i8, chan_scale.squeeze(-1).reshape(*lead, out)


def build_tensor_specs():
    """Build TensorSpecs with correlated INT8 weights + FP32 scales.

    Per-expert recv slabs: rows [0, count_e) are valid INT8 activations with
    per-token dequant scale; rows [count_e, RECV_MAX) are zero-padded.
    """
    import torch
    from golden import TensorSpec

    ROUTED_DEQUANT_STD = {"w_gate": 2.47e-2, "w_up": 2.46e-2, "w_down": 2.44e-2}

    counts = torch.tensor(_EXPERT_COUNTS, dtype=torch.int32).reshape(N_LOCAL_EXPERTS, 1)

    # Per-token INT8 activations + dequant scale (dispatch quantizes; we simulate).
    x_bf16 = torch.randn(N_LOCAL_EXPERTS, RECV_MAX, HIDDEN, dtype=torch.bfloat16)
    valid_mask_3d = (
        torch.arange(RECV_MAX).reshape(1, RECV_MAX, 1) < counts.reshape(N_LOCAL_EXPERTS, 1, 1)
    )
    recv_x_i8, recv_scale_dq = _int8_quant_per_row(x_bf16)
    recv_x_i8 = torch.where(valid_mask_3d, recv_x_i8, torch.zeros_like(recv_x_i8))
    valid_mask_2d = valid_mask_3d.squeeze(-1)
    recv_scale_dq = torch.where(
        valid_mask_2d, recv_scale_dq.squeeze(-1), torch.zeros_like(recv_scale_dq.squeeze(-1)),
    ).float()

    w_gate_i8, w_gate_s = gen_w8a8_weight((N_LOCAL_EXPERTS, INTER, HIDDEN), ROUTED_DEQUANT_STD["w_gate"])
    w_up_i8, w_up_s = gen_w8a8_weight((N_LOCAL_EXPERTS, INTER, HIDDEN), ROUTED_DEQUANT_STD["w_up"])
    w_down_i8, w_down_s = gen_w8a8_weight((N_LOCAL_EXPERTS, HIDDEN, INTER), ROUTED_DEQUANT_STD["w_down"])

    return [
        TensorSpec("recv_x", [N_LOCAL_EXPERTS, RECV_MAX, HIDDEN], torch.int8, init_value=lambda: recv_x_i8),
        TensorSpec("recv_scale_dq", [N_LOCAL_EXPERTS, RECV_MAX], torch.float32, init_value=lambda: recv_scale_dq),
        TensorSpec("recv_expert_count", [N_LOCAL_EXPERTS, 1], torch.int32, init_value=lambda: counts),
        TensorSpec("w_gate", [N_LOCAL_EXPERTS, INTER, HIDDEN], torch.int8, init_value=lambda: w_gate_i8),
        TensorSpec("w_gate_scale", [N_LOCAL_EXPERTS, INTER], torch.float32, init_value=lambda: w_gate_s),
        TensorSpec("w_up", [N_LOCAL_EXPERTS, INTER, HIDDEN], torch.int8, init_value=lambda: w_up_i8),
        TensorSpec("w_up_scale", [N_LOCAL_EXPERTS, INTER], torch.float32, init_value=lambda: w_up_s),
        TensorSpec("w_down", [N_LOCAL_EXPERTS, HIDDEN, INTER], torch.int8, init_value=lambda: w_down_i8),
        TensorSpec("w_down_scale", [N_LOCAL_EXPERTS, HIDDEN], torch.float32, init_value=lambda: w_down_s),
        TensorSpec("recv_y", [N_LOCAL_EXPERTS, RECV_MAX, HIDDEN], torch.bfloat16, is_output=True),
    ]


def golden_expert_routed(swiglu_limit):
    """Return a golden_fn for the given SwigluStep limit (0.0 = plain SiLU).

    Faithful to prefill_fwd._expert_routed:
      * dequant: w_i8.float() * w_scale; x_i8.float() * x_scale
      * gate = x_q @ w_gate[e].T  (w stored [out,in]; .T -> [in,out])
      * silu = gate * sigmoid(gate); SwigluStep: clamp silu AFTER (upper-only)
      * h_bf16 = gated.cast(BF16)  (BF16 round-trip before requant)
      * requant: amax(init 1e-4), scale=127/amax, round->int8
      * y = h_i8.float() * h_sd @ w_down[e].T  (NO route weight)
      * recv_y = y.cast(BF16)
    Uses float64 for exact INT32 accumulator reproduction.
    """
    import torch

    def _fn(tensors):
        recv_x_i8 = tensors["recv_x"]
        recv_scale_dq = tensors["recv_scale_dq"].double()
        recv_expert_count = tensors["recv_expert_count"]
        wg_i8 = tensors["w_gate"]
        wg_scale = tensors["w_gate_scale"].double()
        wu_i8 = tensors["w_up"]
        wu_scale = tensors["w_up_scale"].double()
        wd_i8 = tensors["w_down"]
        wd_scale = tensors["w_down_scale"].double()

        wg_fp = wg_i8.to(torch.float64) * wg_scale.unsqueeze(-1)   # [E, INTER, HIDDEN]
        wu_fp = wu_i8.to(torch.float64) * wu_scale.unsqueeze(-1)   # [E, INTER, HIDDEN]
        wd_fp = wd_i8.to(torch.float64) * wd_scale.unsqueeze(-1)   # [E, HIDDEN, INTER]

        recv_y = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, HIDDEN, dtype=torch.bfloat16)
        for e in range(N_LOCAL_EXPERTS):
            n_rows = int(recv_expert_count[e, 0].item())
            if n_rows == 0:
                continue
            x_i8 = recv_x_i8[e, :n_rows, :].to(torch.float64)
            x_sd = recv_scale_dq[e, :n_rows].reshape(-1, 1)
            x_q = x_i8 * x_sd                                   # [n, HIDDEN]

            gate = x_q @ wg_fp[e].T                             # [n, INTER]
            up = x_q @ wu_fp[e].T
            sigmoid = torch.reciprocal(torch.add(torch.exp(torch.neg(gate)), 1.0))
            silu = gate * sigmoid
            if swiglu_limit > 0.0:
                # step3p5 SwigluStep: clamp silu OUTPUT (upper-only), NOT gate.
                silu_c = torch.clamp(silu, max=swiglu_limit)
                up_c = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
                gated = silu_c * up_c
            else:
                gated = silu * up

            # BF16 round-trip before requant (prefill_fwd h_bf16).
            h_bf16 = gated.to(torch.bfloat16).to(torch.float64)
            amax = h_bf16.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
            scale_q = INT8_SCALE_MAX / amax
            h_i8 = torch.round(h_bf16 * scale_q).clamp(-INT8_SCALE_MAX, INT8_SCALE_MAX).to(torch.int8)
            h_sd = (amax / INT8_SCALE_MAX)                      # = 1/scale_q

            h_fp = h_i8.to(torch.float64) * h_sd                # [n, INTER]
            y = h_fp @ wd_fp[e].T                               # [n, HIDDEN]
            recv_y[e, :n_rows, :] = y.to(torch.bfloat16)

        tensors["recv_y"][:] = recv_y
    _fn.__name__ = f"golden_expert_routed(swiglu_limit={swiglu_limit})"
    return _fn


# ===========================================================================
# Pass-rate comparator: isclose().mean() over valid rows only.
# ===========================================================================

def make_pass_rate_compare(threshold=0.997):
    """Comparator: pass_rate = isclose(actual, expected, rtol, atol).mean()
    over valid rows (rows [0, count_e) per expert). Pass iff pass_rate >= threshold.
    Reports pass_rate, max_abs_diff, max_rel_diff, and valid/total counts.
    """
    import torch

    def cmp(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        actual_f = actual.cpu().to(torch.float32)
        expected_f = expected.cpu().to(torch.float32)
        counts = inputs["recv_expert_count"].cpu()

        # Build a valid-row mask [E, RECV_MAX, HIDDEN] (broadcast over HIDDEN).
        valid_mask = (
            torch.arange(RECV_MAX).reshape(1, RECV_MAX, 1) < counts.reshape(N_LOCAL_EXPERTS, 1, 1)
        )
        valid_flat = valid_mask.expand(-1, -1, HIDDEN).reshape(-1)

        a_flat = actual_f.reshape(-1)
        e_flat = expected_f.reshape(-1)
        a_v = a_flat[valid_flat]
        e_v = e_flat[valid_flat]

        nan_count = int(torch.isnan(a_v).sum().item())
        inf_count = int(torch.isinf(a_v).sum().item())
        if nan_count or inf_count:
            return False, f"    illegal values in actual: NaN={nan_count} Inf={inf_count}"

        close = torch.isclose(a_v, e_v, rtol=rtol, atol=atol)
        pass_rate = float(close.float().mean().item())
        n_valid = int(a_v.numel())
        n_close = int(close.sum().item())

        diff_abs = (a_v - e_v).abs()
        max_abs = float(diff_abs.max().item())
        rel_denom = e_v.abs().clamp_min(1e-9)
        max_rel = float((diff_abs / rel_denom).max().item())

        passed = pass_rate >= threshold
        label = f"pass_rate_compare(threshold={threshold})"
        if passed:
            msg = (
                f"    pass_rate={pass_rate:.6f} ({n_close}/{n_valid} close) "
                f">= {threshold}  max_abs={max_abs:.6g} max_rel={max_rel:.6g} "
                f"rtol={rtol} atol={atol}"
            )
            print(f"[CMP] {msg}")
            return True, ""
        # Failure: show worst mismatches.
        bad = ~close
        bad_idx = torch.where(bad)[0]
        n_show = min(10, bad_idx.numel())
        lines = [
            f"    pass_rate={pass_rate:.6f} ({n_close}/{n_valid} close) < {threshold}\n"
            f"    max_abs={max_abs:.6g} max_rel={max_rel:.6g} rtol={rtol} atol={atol}\n"
            f"    {bad_idx.numel()} mismatched points; first {n_show}:"
        ]
        for i in range(n_show):
            idx = int(bad_idx[i].item())
            lines.append(
                f"      [{idx}] actual={a_v[idx].item():.8g} "
                f"expected={e_v[idx].item():.8g} "
                f"abs_diff={diff_abs[idx].item():.4g}"
            )
        return False, "\n".join(lines)

    cmp.__name__ = f"pass_rate_compare(threshold={threshold})"
    return cmp


# ===========================================================================
# Run harness.
# ===========================================================================

def run_variant(name, test_fn, swiglu_limit, platform, device_id, threshold=0.997,
                rtol=5e-3, atol=5e-3, dump_passes=False):
    """Compile + run one variant on device and return (passed, pass_rate, detail)."""
    from golden import run_jit

    print(f"\n{'='*72}", flush=True)
    print(f"[RUN] variant={name} swiglu_limit={swiglu_limit} "
          f"threshold={threshold} rtol={rtol} atol={atol}", flush=True)
    print(f"{'='*72}", flush=True)

    result = run_jit(
        fn=test_fn,
        specs=build_tensor_specs(),
        golden_fn=golden_expert_routed(swiglu_limit),
        compile_cfg=dict(dump_passes=dump_passes),
        runtime_cfg=dict(platform=platform, device_id=device_id, enable_l2_swimlane=0),
        rtol=rtol,
        atol=atol,
        compare_fn={"recv_y": make_pass_rate_compare(threshold=threshold)},
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.997,
                        help="pass_rate threshold (default 0.997)")
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--variant", choices=["both", "silu", "swiglu7"], default="both",
                        help="Which variant to run (default both)")
    args = parser.parse_args()

    results = {}
    if args.variant in ("both", "silu"):
        r = run_variant("layer4_silu", expert_routed_silu_test, SWIGLU_LIMIT_LAYER4,
                        args.platform, args.device, args.threshold, args.rtol, args.atol,
                        args.dump_passes)
        results["layer4_silu"] = r
    if args.variant in ("both", "swiglu7"):
        r = run_variant("layer44_swiglu7", expert_routed_swiglu7_test, SWIGLU_LIMIT_LAYER44,
                        args.platform, args.device, args.threshold, args.rtol, args.atol,
                        args.dump_passes)
        results["layer44_swiglu7"] = r

    print(f"\n{'='*72}", flush=True)
    print("[SUMMARY] W8A8 single-layer MoE numeric validation", flush=True)
    print(f"{'='*72}", flush=True)
    all_passed = True
    for name, r in results.items():
        passed = r.passed
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        err = r.error or ""
        print(f"  {name}: {status}  (exec={r.execution_time:.1f}s)", flush=True)
        if err:
            print(f"    error: {err}", flush=True)
    verdict = "CONFIRMED" if all_passed else "NOT CONFIRMED"
    print(f"\n  W8A8 numeric correctness: {verdict} (threshold={args.threshold})", flush=True)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
