# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""[中文摘要] 8 种每层特化的 @pl.program(`{full,swa} × {dense, moe×3 激活组合}`)+
`select_decode_layer(layer_idx)` 分发器。`DecodeLayerMoE` 把 EpTpMoE 整套
方法体逐字复制为 `@pl.function(Inline)` —— 因为 pypto frontend 不允许在一个
@pl.program body 里实例化另一个 @pl.program(指南 §10)。
[关键装饰器] @pl.program +
   @pl.function(level=HOST, role=Orchestrator)  ← host_orch
   @pl.function(type=Orchestration)             ← chip_orch
   @pl.function(type=InCore)                    ← attention / dense MLP / MoE 各 body
   @pl.function(type=Inline)                    ← EpTpMoE 拍扁后的方法
   @pl.jit.inline 共享 helper(_ops 复制 + tp_all_reduce 复制)
[SPMD 角色] 跨卡(TP/EP all-reduce/all-to-all)+ 片上多核 SPMD;chip_orch 用
Python 风格的 if/branches 在 layer_idx 上选 attention/MLP 类型。
[详见] 中文架构指南 §3, §4.2, §4.5, §10

────── 以下为英文原 docstring ──────

Step3p5 per-layer decode dispatcher — TP/EP wired (Phase 9 Wave 3).

Each per-layer program is a ``@pl.program`` class composing the Wave-2
TP-refactored attention path with either the Wave-2 EP-refactored MoE
program or a TP-sliced dense-MLP body. The per-layer dispatcher selects
the right class from ``layer_idx`` via ``config.is_full_attention``
(attention flavour) and ``config.is_moe_layer`` (MLP flavour):

  layer_idx  | attention   | MLP
  -----------|-------------|------------
  0, 1, 2    | full / swa  | TP dense MLP
  3..44      | full / swa  | EP+TP MoE
  45..47     | swa         | TP dense MLP (driven by ``mtp.py``)

Eight specialisations are exposed, matching the brief from Wave-3:

  - ``decode_layer_full_dense``           — full attention + TP dense MLP
  - ``decode_layer_swa_dense``            — SWA attention + TP dense MLP
  - ``decode_layer_full_moe_silu_silu``   — full + (silu, silu)
  - ``decode_layer_full_moe_swiglu7_silu``— full + (swiglu7, silu)
  - ``decode_layer_full_moe_swiglu7_swiglu16`` — full + (swiglu7, swiglu16)
  - ``decode_layer_swa_moe_silu_silu``    — SWA  + (silu, silu)
  - ``decode_layer_swa_moe_swiglu7_silu`` — SWA  + (swiglu7, silu)
  - ``decode_layer_swa_moe_swiglu7_swiglu16`` — SWA  + (swiglu7, swiglu16)

Dense-MLP TP slicing (documented in ``_dense_mlp_body_tp``):
  The Wave-2 attention path emits a fully-reduced residual stream on
  every rank (``resid1`` is replicated). The dense MLP that follows
  shards the intermediate axis ``INTERMEDIATE = 11264`` across the TP
  group:
    * ``w_gate / w_up``  per rank ``[HIDDEN, INTERMEDIATE_LOCAL=1408]``
    * ``w_down``         per rank ``[INTERMEDIATE_LOCAL, HIDDEN]``
  After the local ``w_down`` matmul each rank holds a *partial* hidden
  ``[BATCH, HIDDEN]`` BF16 (one term of the TP-group sum); the body
  invokes :func:`tp_all_reduce` to homogenise the partial sums across
  ranks, then adds the post-attention residual ``resid1`` back on top
  (residual is replicated, so the add commutes with the all-reduce).

Window contract (caller is responsible — see ``decode_fwd.py``):
  * Each ``decode_layer_*`` program's ``chip_orch`` takes a fresh
    per-layer ``tmp_window`` (BF16, ``[BATCH, HIDDEN // TP_WORLD_SIZE]``)
    and ``signal_window`` (INT32, ``[TP_WORLD_SIZE, 1]``) for the dense
    MLP's tp_all_reduce. AtomicAdd cells accumulate across the ring
    steps, so the caller MUST allocate a fresh signal window per
    call site (one per dense-MLP layer).
  * For MoE layers, the EP+TP windows for the embedded ``EpTpMoE``
    program are passed through verbatim (see ``moe.EpTpMoE`` for the
    full list).

Per-layer ``layer_idx`` is a runtime ``pl.Scalar[pl.INT32]`` — the slabs
inside ``input_rms_weight`` / ``post_rms_weight`` / ``q_norm_weight`` /
``k_norm_weight`` / ``wq`` / ``wo`` / ``w_gate`` etc. are sliced by it.
"""

# pyright: reportUndefinedVariable=false

from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld

from ._ops import zero_centered_rmsnorm_apply
from .attention_full import (
    LAYER_QHIDDEN_ROWS_DYN as LAYER_QHIDDEN_ROWS_DYN_FULL,
    attention_full,
)
from .attention_swa import (
    LAYER_QHIDDEN_ROWS_DYN as LAYER_QHIDDEN_ROWS_DYN_SWA,
    attention_swa,
)
from .config import (
    ATTN_SCALE,
    BATCH,
    BATCH_TILE,
    BLOCK_SIZE,
    BLOCK_TABLE_FLAT_DYN,
    EPS,
    HEAD_DIM,
    HEAD_DIM_INV,
    HIDDEN,
    HIDDEN_INV,
    HIDDEN_Q_FULL_LOCAL,
    HIDDEN_Q_SWA_LOCAL,
    INPUT_PROJ_K_CHUNK,
    INTERMEDIATE_LOCAL,
    K_CHUNK,
    KV_PROJ_K_CHUNK_LOCAL,
    KV_CACHE_ROWS_DYN,
    KV_HEADS_LOCAL,
    KV_HIDDEN_LOCAL,
    LAYER_DYN,
    LAYER_HIDDEN_ROWS_DYN,
    LAYER_INTER_ROWS_DYN,
    LAYER_ROPE_THETA,
    MAX_BLOCKS_PER_SEQ,
    MAX_SEQ_DEFAULT,
    MLP_OUT_CHUNK,
    MOE_INTERMEDIATE,
    MOE_NUM_EXPERTS,
    MOE_NUM_EXPERTS_LOCAL,
    NUM_HEADS_FULL_LOCAL,
    NUM_HEADS_FULL_LOCAL_PAD,
    NUM_HEADS_SWA_LOCAL,
    NUM_HEADS_SWA_LOCAL_PAD,
    OUT_PROJ_K_CHUNK,
    OUT_PROJ_N_CHUNK,
    Q_HEAD_BATCH_FULL,
    Q_HEAD_BATCH_SWA,
    Q_HEAD_PAD_FULL,
    Q_HEAD_PAD_SWA,
    Q_OUT_CHUNK,
    Q_PER_KV_FULL,
    Q_PER_KV_SWA,
    ROPE_SCALING,
    ROPE_SEQ_DYN,
    ROTARY_HALF_FULL,
    ROTARY_HALF_SWA,
    SHARE_EXPERT_DIM_LOCAL,
    SLIDING_WINDOW,
    SWIGLU_LIMITS,
    SWIGLU_LIMITS_SHARED,
    TP_WORLD_SIZE,
    USER_BATCH_DYN,
    is_full_attention,
    is_moe_layer,
)
from .dispatch import LOCAL_RECV_MAX, N_RANKS_PAD, PER_RANK_BUCKETS
from .moe import select_moe_block
from .config import VOCAB_LOCAL, VOCAB_CHUNK, FINAL_RMS_K_CHUNK, LM_HEAD_K_CHUNK
from .rms_lm_head import rms_lm_head


# -----------------------------------------------------------------------------
# Local re-exports for shorter signatures.
# -----------------------------------------------------------------------------
N_EXPERTS = MOE_NUM_EXPERTS
N_LOCAL_EXPERTS = MOE_NUM_EXPERTS_LOCAL
INTER_R = MOE_INTERMEDIATE
INTER_S_LOCAL = SHARE_EXPERT_DIM_LOCAL
INTER_LOCAL = INTERMEDIATE_LOCAL
TP_CHUNK = HIDDEN // TP_WORLD_SIZE
TOPK = 8
N_ROUTES_PER_RANK = BATCH * TOPK


assert HIDDEN % TP_WORLD_SIZE == 0
assert INTER_LOCAL % MLP_OUT_CHUNK == 0


# -----------------------------------------------------------------------------
# Phase X.8 — kernel-internal constants for the inlined MoE method bodies.
#
# pypto frontend rejects ``self._embedded_moe_cls().chip_orch(...)``
# (instantiating a ``@pl.program`` inside another ``@pl.program`` body is not a
# supported feature), so the entire body of ``EpTpMoE`` (from ``moe.py``) —
# every ``@pl.function`` method plus the ``chip_orch`` body — is inlined
# directly into ``DecodeLayerMoE``. Their tiling / sort widths / activation
# thresholds live here as module-level constants so the lifted method bodies
# can reference them via closure capture without depending on ``moe.py`` at
# parse time. The originals in ``moe.py`` remain intact (its ``__main__``
# harness still drives them as a standalone @pl.program).
# -----------------------------------------------------------------------------

# Router (gate) kernel constants — mirrors gate.py / moe.ROUTER_*.
ROUTER_SCORE_PAD = 512
ROUTER_TOPK_PAD = 16
ROUTER_SORT_PAD = ROUTER_TOPK_PAD * 2
ROUTER_GATE_K_CHUNK = 256   # 512→256: keeps x0/xk [16,256] FP32 = 16384 B in Vec budget
ROUTER_GATE_N_CHUNK = 32   # N-chunk for gate matmul; [K=256,N=32] FP32 = 32768 B (L0B limit)
ROUTER_FP32_NEG_INF = -3.4028235e38
ROUTER_SCALE = 3.0  # MOE_ROUTER_SCALING_FACTOR
assert TOPK <= ROUTER_TOPK_PAD
assert HIDDEN % ROUTER_GATE_K_CHUNK == 0
assert N_EXPERTS % ROUTER_GATE_N_CHUNK == 0

# Routed-expert kernel constants — mirrors expert_routed.py / moe.ROUTED_*.
ROUTED_GATE_K_CHUNK = 64    # A-tile K dim; [32,64] BF16 = 4096 B (L0A)
ROUTED_GATE_N_CHUNK = 64    # B-tile N dim; [64,64] BF16 = 8192 B (L0B)
ROUTED_DOWN_K_CHUNK = 64    # A-tile K dim; [32,64] BF16 = 4096 B (L0A)
ROUTED_DOWN_N_CHUNK = 128   # B-tile N dim; [64,128] BF16 = 16384 B (L0B)
ROUTED_MAX_TILE = LOCAL_RECV_MAX

# Per-tile row count for the routed-expert compute body. PTOAS Vec UB on
# A2/A3 caps tiles at 192 KB; allocating ``[ROUTED_MAX_TILE=1024, MOE_INTERMEDIATE=1280]``
# FP32 = 5.2 MB blows past that by ~28×. We adopt deepseek/v4's row-tiling:
# ``RECV_TILE`` rows per inner pass, with an outer ``for tile_idx in pl.range(n_tiles)``
# loop. ``32 * 1280 * 4 = 160 KB`` keeps headroom for intermediates.
RECV_TILE = 32
assert ROUTED_MAX_TILE % RECV_TILE == 0, (
    f"ROUTED_MAX_TILE ({ROUTED_MAX_TILE}) must be divisible by RECV_TILE ({RECV_TILE})"
)
N_RECV_TILES = ROUTED_MAX_TILE // RECV_TILE  # 32 outer iterations
# Diagnostic bisect knob (build-time constant): when >0, MoE chip_orch skips
# dispatch/routed/combine and returns resid + shared-expert output only.
_MOE_SHARED_ONLY = int(__import__("os").environ.get("P_MOE_SHARED_ONLY", "0"))
# Diagnostic bisect knob: when >0, MoE chip_orch returns resid1 (the h_mid input
# as read by chip_orch) directly — isolates the attention->MoE handoff.
_MOE_PASSTHROUGH = int(__import__("os").environ.get("P_MOE_PASSTHROUGH", "0"))
# Diagnostic bisect knob: when >0, MoE chip_orch returns post_norm (rmsnorm of
# resid1) directly — isolates the post-attention RMSNorm output magnitude.
_MOE_NORM_ONLY = int(__import__("os").environ.get("P_MOE_NORM_ONLY", "0"))
# Diagnostic bisect knob: when >0, the FUSED MoE orch returns resid1 (the
# post-attention hidden) directly — isolates whether the attention output (fed
# into the MoE) is the magnitude source, vs the MoE compute itself.
_FUSE_ATTN_ONLY = int(__import__("os").environ.get("P_FUSE_ATTN_ONLY", "0"))
# Diagnostic op-level dump (E2): when >0, the FUSED MoE orch writes a chosen
# intermediate stage into the dedicated `dbg_out` program-Out (a SEPARATE buffer
# written right after the stage, so it reliably captures that stage's value —
# unlike in-orch early-return knobs). 1=post_norm, 2=sh_y (shared expert),
# 4=moe_out (after combine, before residual). Isolates the first stage that goes
# ~1e11 to pin aliasing vs collective vs quant.
_DBG_STAGE = int(__import__("os").environ.get("P_DBG_STAGE", "0"))
assert HIDDEN % ROUTED_GATE_K_CHUNK == 0
assert HIDDEN % ROUTED_DOWN_N_CHUNK == 0
assert MOE_INTERMEDIATE % ROUTED_GATE_N_CHUNK == 0
assert MOE_INTERMEDIATE % ROUTED_DOWN_K_CHUNK == 0

# Shared-expert kernel constants — mirrors expert_shared.py / moe.SHARED_*.
SHARED_GATE_K_CHUNK = 256
SHARED_GATE_N_CHUNK = INTER_S_LOCAL  # 160 — one N tile covers the slice
SHARED_DOWN_K_CHUNK = INTER_S_LOCAL  # 160 — one K tile covers the slice
SHARED_DOWN_N_CHUNK = 256
assert HIDDEN % SHARED_GATE_K_CHUNK == 0
assert HIDDEN % SHARED_DOWN_N_CHUNK == 0


# =============================================================================
# Dense-MLP body — TP-sliced gate/up/down + tp_all_reduce.
#
# The intermediate axis ``INTERMEDIATE = 11264`` is sharded across the TP
# group: each rank holds ``INTERMEDIATE_LOCAL = 1408`` lanes of
# ``w_gate``/``w_up`` (column slice) and ``w_down`` (row slice). After
# the local ``w_down`` matmul each rank produces a partial
# ``[BATCH, HIDDEN]`` BF16 contribution to the full hidden output; the
# body's tp_all_reduce sums those partials across the group so every
# rank holds the same fully-reduced next hidden. The post-attention
# residual is replicated (the attention's tp_all_reduce already
# homogenised it) and gets added on top after the reduction.
# =============================================================================
@pl.jit.inline
def _dense_mlp_body_tp(
    resid1: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    w_gate: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTER_LOCAL], pl.BF16],
    w_up: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTER_LOCAL], pl.BF16],
    w_down: pl.Tensor[[LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16],
    next_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    norm_layer_idx: pl.Scalar[pl.INT32],
    mlp_layer_idx: pl.Scalar[pl.INT32],
    tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
    signal_window: pld.DistributedTensor[[TP_WORLD_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
    """Post-attention zero-centred RMSNorm + TP-sliced SwiGLU MLP + residual.

    The per-rank weight slabs are:
      * ``w_gate / w_up``  ``[HIDDEN, INTERMEDIATE_LOCAL=1408]`` per layer
      * ``w_down``         ``[INTERMEDIATE_LOCAL, HIDDEN]`` per layer

    After the local ``w_down`` matmul each rank holds a partial
    ``[BATCH, HIDDEN]`` BF16 (one of TP_WORLD_SIZE terms in the sum). The
    body then runs ``tp_all_reduce`` on the partial-hidden tile so every
    rank receives the fully-dm_reduced output before the residual add.
    Step3p5's dense layers (0..2 + the MTP layers' dense MLP) all use
    plain SiLU activation (``SWIGLU_LIMITS == 0`` everywhere on the dense
    path), matching the single-card draft's activation choice.
    """
    hidden_blocks = HIDDEN // K_CHUNK
    mlp_out_blocks = INTER_LOCAL // MLP_OUT_CHUNK
    layer_hidden_base = mlp_layer_idx * HIDDEN
    layer_inter_base = mlp_layer_idx * INTER_LOCAL

    # ── Step 1: post-attention zero-centred RMSNorm of resid1. ──────────
    dm_post_norm = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    dm_resid1_fp32 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dense_post_rmsnorm_zc"):
        for kb in pl.range(hidden_blocks):
            k0 = kb * K_CHUNK
            dm_rchunk = pl.cast(
                pl.slice(resid1, [BATCH, K_CHUNK], [0, k0]),
                target_type=pl.FP32,
            )
            dm_resid1_fp32 = pl.assemble(dm_resid1_fp32, dm_rchunk, [0, k0])

        dm_sq_sum = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
        for kb2 in pl.range(hidden_blocks):
            k0 = kb2 * K_CHUNK
            dm_ck = pl.slice(dm_resid1_fp32, [BATCH, K_CHUNK], [0, k0])
            dm_sq_sum = pl.add(
                dm_sq_sum,
                pl.reshape(
                    pl.row_sum(pl.mul(dm_ck, dm_ck)),
                    [1, BATCH],
                ),
            )
        inv_rms_dense = pl.recip(
            pl.sqrt(pl.add(pl.mul(dm_sq_sum, HIDDEN_INV), EPS)),
        )
        dm_inv_rms_col = pl.reshape(inv_rms_dense, [BATCH, 1])
        for kb3 in pl.range(hidden_blocks):
            k0 = kb3 * K_CHUNK
            dm_norm_chunk = pl.slice(
                dm_resid1_fp32, [BATCH, K_CHUNK], [0, k0],
            )
            dm_gamma = pl.slice(
                post_rms_weight, [1, K_CHUNK], [norm_layer_idx, k0],
            )
            dm_scaled = pl.row_expand_mul(dm_norm_chunk, dm_inv_rms_col)
            dm_normed = pl.col_expand_mul(dm_scaled, pl.add(dm_gamma, 1.0))
            dm_post_norm = pl.assemble(
                dm_post_norm,
                pl.cast(dm_normed, target_type=pl.BF16),
                [0, k0],
            )

    # ── Step 2: TP-sliced gate_up + plain SiLU into mlp_tile. ──────────
    # Phase A (2026-06-11): split the mixed AIC+AIV body so PTOAS does not
    # lower this scope to MixedKernels (the mixed-mode root is the 507018
    # VEC UB alignment crash site; the first such kernel in dispatch order
    # crashed deterministically — see phase-15 doc Phase A section).
    # Stage 2.a (cube): both gate and up matmuls into FP32 GM scratch.
    gate_acc_gm = pl.create_tensor([BATCH, INTER_LOCAL], dtype=pl.FP32)
    up_acc_gm = pl.create_tensor([BATCH, INTER_LOCAL], dtype=pl.FP32)
    for ob in pl.spmd(
        mlp_out_blocks, name_hint="dense_gate_up_matmul_tp",
        optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
    ):
        mlp_o0 = ob * MLP_OUT_CHUNK
        post_chunk_0 = pl.slice(dm_post_norm, [BATCH, K_CHUNK], [0, 0])
        wg_0 = pl.slice(
            w_gate, [K_CHUNK, MLP_OUT_CHUNK], [layer_hidden_base, mlp_o0],
        )
        wu_0 = pl.slice(
            w_up, [K_CHUNK, MLP_OUT_CHUNK], [layer_hidden_base, mlp_o0],
        )
        gate_acc = pl.matmul(post_chunk_0, wg_0, out_dtype=pl.FP32)
        up_acc = pl.matmul(post_chunk_0, wu_0, out_dtype=pl.FP32)
        for kb in pl.range(1, hidden_blocks):
            k0 = kb * K_CHUNK
            post_chunk = pl.slice(dm_post_norm, [BATCH, K_CHUNK], [0, k0])
            wg = pl.slice(
                w_gate, [K_CHUNK, MLP_OUT_CHUNK],
                [layer_hidden_base + k0, mlp_o0],
            )
            wu = pl.slice(
                w_up, [K_CHUNK, MLP_OUT_CHUNK],
                [layer_hidden_base + k0, mlp_o0],
            )
            gate_acc = pl.matmul_acc(gate_acc, post_chunk, wg)
            up_acc = pl.matmul_acc(up_acc, post_chunk, wu)
        gate_acc_gm = pl.assemble(gate_acc_gm, gate_acc, [0, mlp_o0])
        up_acc_gm = pl.assemble(up_acc_gm, up_acc, [0, mlp_o0])

    # Stage 2.b (vec): SiLU(gate) * up, cast to BF16.
    mlp_tile = pl.create_tensor([BATCH, INTER_LOCAL], dtype=pl.BF16)
    for ob in pl.spmd(mlp_out_blocks, name_hint="dense_silu_cast_tp"):
        mlp_o0 = ob * MLP_OUT_CHUNK
        gate_chunk = pl.slice(
            gate_acc_gm, [BATCH, MLP_OUT_CHUNK], [0, mlp_o0],
        )
        up_chunk = pl.slice(
            up_acc_gm, [BATCH, MLP_OUT_CHUNK], [0, mlp_o0],
        )
        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_chunk)), 1.0))
        mlp_chunk = pl.mul(pl.mul(gate_chunk, sigmoid), up_chunk)
        mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)
        mlp_tile = pl.assemble(mlp_tile, mlp_chunk_bf16, [0, mlp_o0])

    # ── Step 3: TP-sliced w_down -> partial [BATCH, HIDDEN] BF16. ──────
    # Phase A: split cube matmul + vec cast (mirror of full_out_proj fix).
    partial_hidden_fp32 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    for dob in pl.spmd(
        hidden_blocks, name_hint="dense_down_matmul_tp",
        optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
    ):
        d0 = dob * K_CHUNK
        mlp_chunk_0 = pl.slice(mlp_tile, [BATCH, MLP_OUT_CHUNK], [0, 0])
        w_down_chunk_0 = pl.slice(
            w_down, [MLP_OUT_CHUNK, K_CHUNK], [layer_inter_base, d0],
        )
        down_acc = pl.matmul(mlp_chunk_0, w_down_chunk_0, out_dtype=pl.FP32)
        for ob in pl.range(1, mlp_out_blocks):
            down_o0 = ob * MLP_OUT_CHUNK
            down_mlp_chunk_bf16 = pl.slice(
                mlp_tile, [BATCH, MLP_OUT_CHUNK], [0, down_o0],
            )
            w_down_chunk = pl.slice(
                w_down, [MLP_OUT_CHUNK, K_CHUNK],
                [layer_inter_base + down_o0, d0],
            )
            down_acc = pl.matmul_acc(down_acc, down_mlp_chunk_bf16, w_down_chunk)
        partial_hidden_fp32 = pl.assemble(
            partial_hidden_fp32, down_acc, [0, d0],
        )

    partial_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for dob in pl.spmd(hidden_blocks, name_hint="dense_down_cast_tp"):
        d0 = dob * K_CHUNK
        dense_fp32_chunk = pl.slice(
            partial_hidden_fp32, [BATCH, K_CHUNK], [0, d0],
        )
        partial_hidden = pl.assemble(
            partial_hidden,
            pl.cast(dense_fp32_chunk, target_type=pl.BF16),
            [0, d0],
        )

    # ── Step 4: TP all-reduce across the group's partial hiddens. ──────
    # Phase X.2: ``self.tp_all_reduce`` resolves to a method on the
    # enclosing @pl.program class (DecodeLayerDense / DecodeLayerMoE).
    # Phase 15.1 single-rank gate: at TP=1 skip (mirror of 15.B).
    if TP_WORLD_SIZE > 1:
        self.tp_all_reduce(
            partial_hidden, tmp_window, signal_window, my_rank,
        )

    # ── Step 5: replicated residual add — next_hidden = resid1 + dm_reduced.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dense_residual_add_tp"):
        for kb4 in pl.range(hidden_blocks):
            k0 = kb4 * K_CHUNK
            dm_reduced = pl.cast(
                pl.slice(partial_hidden, [BATCH, K_CHUNK], [0, k0]),
                target_type=pl.FP32,
            )
            dm_r = pl.slice(dm_resid1_fp32, [BATCH, K_CHUNK], [0, k0])
            next_hidden = pl.assemble(
                next_hidden,
                pl.cast(pl.add(dm_r, dm_reduced), target_type=pl.BF16),
                [0, k0],
            )

    return next_hidden



@pl.jit.inline
def _dense_mlp_body_tp_fused(
    resid1: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    w_gate: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTER_LOCAL], pl.BF16],
    w_up: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTER_LOCAL], pl.BF16],
    w_down: pl.Tensor[[LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16],
    next_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    norm_layer_idx: pl.Scalar[pl.INT32],
    mlp_layer_idx: pl.Scalar[pl.INT32],
    tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
    signal_window: pld.DistributedTensor[[TP_WORLD_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
    """Single-scope-safe dense MLP: no cross-scope post_norm/resid1_fp32 scratch."""
    hidden_blocks = HIDDEN // K_CHUNK
    mlp_out_blocks = INTER_LOCAL // MLP_OUT_CHUNK
    layer_hidden_base = mlp_layer_idx * HIDDEN
    layer_inter_base = mlp_layer_idx * INTER_LOCAL

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dfz_inv_rms"):
        dfz_sq = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
        for kb in pl.range(hidden_blocks):
            k0 = kb * K_CHUNK
            rc = pl.cast(pl.slice(resid1, [BATCH, K_CHUNK], [0, k0]), target_type=pl.FP32)
            dfz_sq = pl.add(dfz_sq, pl.reshape(pl.row_sum(pl.mul(rc, rc)), [1, BATCH]))
        dfz_inv = pl.reshape(
            pl.recip(pl.sqrt(pl.add(pl.mul(dfz_sq, HIDDEN_INV), EPS))), [BATCH, 1]
        )

    dfz_gate_gm = pl.create_tensor([BATCH, INTER_LOCAL], dtype=pl.FP32)
    dfz_up_gm = pl.create_tensor([BATCH, INTER_LOCAL], dtype=pl.FP32)
    for ob in pl.spmd(
        mlp_out_blocks, name_hint="dfz_gate_up_matmul",
    ):
        mo0 = ob * MLP_OUT_CHUNK
        rc0 = pl.cast(pl.slice(resid1, [BATCH, K_CHUNK], [0, 0]), target_type=pl.FP32)
        gm0 = pl.slice(post_rms_weight, [1, K_CHUNK], [norm_layer_idx, 0])
        pn0 = pl.cast(
            pl.col_expand_mul(pl.row_expand_mul(rc0, dfz_inv), pl.add(gm0, 1.0)),
            target_type=pl.BF16,
        )
        wg0 = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [layer_hidden_base, mo0])
        wu0 = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [layer_hidden_base, mo0])
        dfz_ga = pl.matmul(pn0, wg0, out_dtype=pl.FP32)
        dfz_ua = pl.matmul(pn0, wu0, out_dtype=pl.FP32)
        for kb in pl.range(1, hidden_blocks):
            k0 = kb * K_CHUNK
            rc = pl.cast(pl.slice(resid1, [BATCH, K_CHUNK], [0, k0]), target_type=pl.FP32)
            gm = pl.slice(post_rms_weight, [1, K_CHUNK], [norm_layer_idx, k0])
            pn = pl.cast(
                pl.col_expand_mul(pl.row_expand_mul(rc, dfz_inv), pl.add(gm, 1.0)),
                target_type=pl.BF16,
            )
            wg = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [layer_hidden_base + k0, mo0])
            wu = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [layer_hidden_base + k0, mo0])
            dfz_ga = pl.matmul_acc(dfz_ga, pn, wg)
            dfz_ua = pl.matmul_acc(dfz_ua, pn, wu)
        dfz_gate_gm = pl.assemble(dfz_gate_gm, dfz_ga, [0, mo0])
        dfz_up_gm = pl.assemble(dfz_up_gm, dfz_ua, [0, mo0])

    dfz_mlp = pl.create_tensor([BATCH, INTER_LOCAL], dtype=pl.BF16)
    for ob in pl.spmd(mlp_out_blocks, name_hint="dfz_silu"):
        mo0 = ob * MLP_OUT_CHUNK
        gc = pl.slice(dfz_gate_gm, [BATCH, MLP_OUT_CHUNK], [0, mo0])
        uc = pl.slice(dfz_up_gm, [BATCH, MLP_OUT_CHUNK], [0, mo0])
        sg = pl.recip(pl.add(pl.exp(pl.neg(gc)), 1.0))
        mc = pl.mul(pl.mul(gc, sg), uc)
        dfz_mlp = pl.assemble(dfz_mlp, pl.cast(mc, target_type=pl.BF16), [0, mo0])

    dfz_pf = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    for dob in pl.spmd(
        hidden_blocks, name_hint="dfz_down_matmul",
    ):
        d0 = dob * K_CHUNK
        mc0 = pl.slice(dfz_mlp, [BATCH, MLP_OUT_CHUNK], [0, 0])
        wd0 = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [layer_inter_base, d0])
        dfz_da = pl.matmul(mc0, wd0, out_dtype=pl.FP32)
        for ob in pl.range(1, mlp_out_blocks):
            do0 = ob * MLP_OUT_CHUNK
            mcb = pl.slice(dfz_mlp, [BATCH, MLP_OUT_CHUNK], [0, do0])
            wdb = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [layer_inter_base + do0, d0])
            dfz_da = pl.matmul_acc(dfz_da, mcb, wdb)
        dfz_pf = pl.assemble(dfz_pf, dfz_da, [0, d0])

    dfz_partial = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for dob in pl.spmd(hidden_blocks, name_hint="dfz_down_cast"):
        d0 = dob * K_CHUNK
        fc = pl.slice(dfz_pf, [BATCH, K_CHUNK], [0, d0])
        dfz_partial = pl.assemble(dfz_partial, pl.cast(fc, target_type=pl.BF16), [0, d0])

    if TP_WORLD_SIZE > 1:
        self.tp_all_reduce(dfz_partial, tmp_window, signal_window, my_rank)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dfz_residual_add"):
        for kb4 in pl.range(hidden_blocks):
            k0 = kb4 * K_CHUNK
            red = pl.cast(pl.slice(dfz_partial, [BATCH, K_CHUNK], [0, k0]), target_type=pl.FP32)
            rr = pl.cast(pl.slice(resid1, [BATCH, K_CHUNK], [0, k0]), target_type=pl.FP32)
            next_hidden = pl.assemble(
                next_hidden, pl.cast(pl.add(rr, red), target_type=pl.BF16), [0, k0]
            )
    return next_hidden


# =============================================================================
# Per-layer @pl.program builders.
#
# Each builder returns a freshly-constructed @pl.program class with a
# chip_orch (per-rank orchestration) and a host_orch (per-call-site
# window allocation + per-rank dispatch). The builders are constructed
# inside Python factory functions so the module imports even on hosts
# that have not finished bringing up the pypto runtime (deferred-build
# pattern, mirrors the in-tree TP+EP MoE reference).
# =============================================================================


# N1 whole-net single-submit entry. This generated module intentionally keeps
# only the final single-submit implementation and shared helpers; legacy
# 46-submit whole-net builders remain in decode_layer.py for generator input /
# historical baseline only and are not re-exported here.
def _build_whole_decode_faithful_real_single_chip_program(
    *,
    routed_lim: float = 0.0,
    shared_lim: float = 0.0,
    tp_size: int = TP_WORLD_SIZE,
):
    """Return a ``@pl.program`` = faithful 45-layer whole-decode network in
    ONE ``@pl.program``: L0 full-dense, L1/L2 swa-dense, L3..44 = 42 real
    per-protocol MoE layers (swa_attn_only_orch -> chip_orch each) + tail.

    Spliced from the compile-verified ``_build_whole_decode_mixed_min_program``
    (WholeDecodeMixedMin, rc=0): the entire method-set (tp_all_reduce + EP
    all_to_all + stage helpers + chip_orch + attn_dense_orch + lm_head_orch +
    full_chip_orch + swa_chip_orch + swa_attn_only_orch) is reused verbatim,
    class renamed to WholeDecodeFaithful. Only the host_orch is replaced with
    a SOURCE-UNROLLED 45-layer chain (explicit per-layer pass blocks, no
    Python ``for`` in the @pl.function body — modeled on WholeDecodeNetwork).

    Per-protocol separation holds: dense attn methods + MoE method are
    distinct Orchestration methods with distinct host_orch passes, so
    pass-37 does not collapse TP+EP comm-domains into one (Wall-2 avoidance).

    2-buffer resident handoff (mirrors WholeDecodeNetwork):
      A = h_mid_out (transient resid), B = next_hidden_out (layer output).
      L0 full_chip_orch: current_hidden -> B
      L1 swa_chip_orch:  B -> A
      L2 swa_chip_orch:  A -> B
      L3..44 (x42): swa_attn_only_orch B -> A ; chip_orch A -> B
      tail lm_head_orch: reads B

    Compile-milestone simplifications (runtime host-side stack slicing later):
      - swa-attn for ALL 42 MoE layers (full-attn MoE variant is a follow-up).
      - reuse-one-slab-per-method-type: layer_idx=0 for every layer (per-layer
        distinct weights = runtime, host-side stack slicing deferred).
      - windows allocated once, reused across all 45 layers.
    """
    if HIDDEN % tp_size != 0:
        raise ValueError(
            f"HIDDEN={HIDDEN} must divide tp_size={tp_size}"
        )
    attention_full_inline = pl.inline(attention_full._func)
    attention_swa_inline = pl.inline(attention_swa._func)
    # attn_dense_orch (reused verbatim from MixedMoeTail) references
    # attention_inline; L3 is a SWA-MoE layer so bind it to swa.
    attention_inline = attention_swa_inline
    dense_mlp_inline = pl.inline(_dense_mlp_body_tp._func)
    rms_lm_head_inline = pl.inline(rms_lm_head._func)

    # Activation choice — compile-time Python constants captured in closure.
    if routed_lim == 0.0:
        _routed_swiglu_step = False
    elif routed_lim == 7.0:
        _routed_swiglu_step = True
    else:
        raise ValueError(
            f"routed_lim must be 0.0 or 7.0, got {routed_lim}",
        )

    if shared_lim == 0.0:
        _shared_swiglu_step = False
    elif shared_lim == 16.0:
        _shared_swiglu_step = True
    else:
        raise ValueError(
            f"shared_lim must be 0.0 or 16.0, got {shared_lim}",
        )

    _routed_swiglu_limit = routed_lim
    _shared_swiglu_limit = shared_lim
    tp_chunk = HIDDEN // tp_size
    _FAITHFUL_MOE_LAYERS = int(__import__('os').environ.get('P_FAITHFUL_MOE_LAYERS', '42'))  # bisect: N MoE layers emitted (default 42 = full)

    # L3 is a SWA-MoE layer: chip_orch / attn_dense_orch vestigial
    # attention-shape params resolve to SWA values.
    rotary_dim = 128
    hidden_q_local = HIDDEN_Q_SWA_LOCAL
    num_heads_local = NUM_HEADS_SWA_LOCAL
    num_heads_local_pad = NUM_HEADS_SWA_LOCAL_PAD
    layer_qhidden_dyn = LAYER_QHIDDEN_ROWS_DYN_SWA

    # Dense-prefix layer consts (L0 full + L1/L2 swa distinct methods).
    rotary_dim_full = ROTARY_HALF_FULL * 2      # 64
    rotary_dim_swa = ROTARY_HALF_SWA * 2        # 128
    hidden_q_full = HIDDEN_Q_FULL_LOCAL
    hidden_q_swa = HIDDEN_Q_SWA_LOCAL
    nh_full_pad = NUM_HEADS_FULL_LOCAL_PAD
    nh_swa_pad = NUM_HEADS_SWA_LOCAL_PAD
    layer_qhidden_full = LAYER_QHIDDEN_ROWS_DYN_FULL
    layer_qhidden_swa = LAYER_QHIDDEN_ROWS_DYN_SWA

    n_ranks = tp_size
    n_ranks_pad = N_RANKS_PAD
    n_local_experts = N_LOCAL_EXPERTS
    # pub_counts cols padded to a multiple of 8 so a [n_ranks, n_local_experts_pad]
    # INT32 tile has 32B-aligned rows (40*4=160B), required by the burst-free
    # count exchange's remote_load (36*4=144B fails ptoas alloc_tile alignment).
    n_local_experts_pad = ((n_local_experts + 7) // 8) * 8
    idx_pad = 8
    stage_rows = 8
    inter = MOE_INTERMEDIATE
    sh_inter_local = INTER_S_LOCAL
    sh_tp_chunk = HIDDEN // tp_size
    local_recv_max = LOCAL_RECV_MAX  # matches dispatch.LOCAL_RECV_MAX (1024)
    n_routes_per_rank = BATCH * TOPK
    per_rank_buckets = PER_RANK_BUCKETS  # n_ranks * n_local_experts
    # A2/A3 comm-domain buffers are carved sequentially without per-slot
    # alignment. Reserve one full L2 cache line for every cross-rank control
    # signal so AtomicAdd/TWAIT traffic cannot share a line with adjacent
    # control or data windows. The logical tensor view remains [8, 1] INT32.
    COMM_CONTROL_SIGNAL_BYTES = 512
    COMM_SIGNAL_STRIDE_I32 = COMM_CONTROL_SIGNAL_BYTES // 4
    WHOLE_CHIP_DENSE_LAYERS = 3
    WHOLE_CHIP_MOE_LAYERS = 42

    @pl.program
    class WholeDecodeFaithfulRealSingleChip:
        # ===============================================================
        # MoE method-set + attn_dense_orch + lm_head_orch spliced VERBATIM
        # from _build_mixed_moe_tail_program (MixedMoeTail, compile rc=0).
        # Bodies kept byte-identical; only the enclosing class is renamed.
        # ===============================================================

        @pl.function(type=pl.FunctionType.InCore)
        def tp_all_reduce(
            self,
            local: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
            signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            group_size = tp_size

            # Phase 1: stage-in — copy local into my tmp_window slot (full HIDDEN).
            # All-reduce HIDDEN tiling width: fixed, INDEPENDENT of tp_size.
            # tp_chunk = HIDDEN // tp_size collapses to HIDDEN (4096) at
            # tp_size=1 (apply_tp1_patch single-card e2e), so [BATCH, 4096]
            # FP32 acc tiles (256KB) overflow the 188KB UB limit. A fixed
            # tile keeps the per-iteration working set bounded for every
            # tp_size (512 = canonical TP=8 chunk; HIDDEN is divisible by it).
            ar_chunk = HIDDEN // 8
            for k0 in pl.range(0, HIDDEN, ar_chunk):
                stage_tile = pl.load(local, [0, k0], [BATCH, ar_chunk])
                pl.store(stage_tile, [0, k0], tmp_window)

            # Phase 2: barrier — notify all peers (one round), then wait on all
            # peers (one round). Separate loops, matching the pypto own-test
            # pattern. expected=1 fixed (cells start zero, accumulate to N-1
            # after all notifies land; we only require >=1 from each peer slot).
            for peer in pl.range(group_size):
                if peer != my_rank:
                    pld.system.notify(
                        target=signal_window, peer=peer,
                        offsets=[my_rank, 0], value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for src in pl.range(group_size):
                if src != my_rank:
                    pld.system.wait(
                        signal=signal_window, offsets=[src, 0],
                        expected=1, cmp=pld.WaitCmp.Ge,
                    )

            # Phase 3: load own tmp slot, then for each peer remote_load + tadd
            # (FP32 — PTOAS bf16 tadd unsupported, cast through f32). Result lands
            # back in `local` (in-place reduction target).
            for k0 in pl.range(0, HIDDEN, ar_chunk):
                own_tile = pl.load(tmp_window, [0, k0], [BATCH, ar_chunk])
                acc = pl.cast(own_tile, target_type=pl.FP32)
                for peer in pl.range(group_size):
                    if peer != my_rank:
                        recv = pld.tile.remote_load(
                            tmp_window, peer=peer,
                            offsets=[0, k0], shape=[BATCH, ar_chunk],
                        )
                        acc = pl.add(acc, pl.cast(recv, target_type=pl.FP32))
                pl.store(
                    pl.cast(acc, target_type=pl.BF16),
                    [0, k0], local,
                )
            # Phase 4: completion barrier (framework two-wave protocol). Ensures every
            # rank finished Phase 3 reads before returning, so the next layer's
            # collective cannot race this one's reads. Single-wave (Phase 2 only) hung
            # at >=41 pipelined layers (507018 / S1:running-stalled). Same signal_window,
            # threshold escalated 1 -> 2.
            for peer in pl.range(group_size):
                if peer != my_rank:
                    pld.system.notify(
                        target=signal_window, peer=peer,
                        offsets=[my_rank, 0], value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for src in pl.range(group_size):
                if src != my_rank:
                    pld.system.wait(
                        signal=signal_window, offsets=[src, 0],
                        expected=2, cmp=pld.WaitCmp.Ge,
                    )
            return local

        # ===================================================================
        # Phase X.8 — inlined EpTpMoE @pl.function methods.
        # Bodies copied verbatim from ``moe.EpTpMoE`` (Phase X.3+X.4+X.5).
        # The activation choice (plain SiLU vs. SwigluStep@7/16) is baked at
        # factory build time via the ``_routed_swiglu_step`` /
        # ``_shared_swiglu_step`` Python closure constants — only one branch
        # is emitted per specialisation. Module-level constants from moe.py
        # (T, N_RANKS, INTER, etc.) are renamed to the local closure names
        # (BATCH, n_ranks, inter, ...) so the bodies type-check against this
        # factory's signatures.
        # ===================================================================

        # ---------- Collective: EP all_to_all ----------
        @pl.function(type=pl.FunctionType.Inline)
        def ep_all_to_all(
            self,
            send: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.BF16],
            recv: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.BF16],
            send_counts: pl.Tensor[[n_ranks], pl.INT32],
            recv_counts: pl.Tensor[[n_ranks], pl.INT32],
            send_offsets: pl.Tensor[[n_ranks], pl.INT32],
            recv_offsets: pl.Tensor[[n_ranks], pl.INT32],
            signal_window: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pld.DistributedTensor[[local_recv_max, HIDDEN], pl.BF16]:
            """Pull-side variable-length token-level all-to-all over EP."""
            group_size = n_ranks
            d_cols = HIDDEN

            # 1) Local self-bucket copy.
            n_self = pl.cast(pl.read(send_counts, [my_rank]), pl.INDEX)
            s_off_self = pl.cast(pl.read(send_offsets, [my_rank]), pl.INDEX)
            r_off_self = pl.cast(pl.read(recv_offsets, [my_rank]), pl.INDEX)
            for r in pl.range(n_self):
                self_tile = pl.load(
                    send, [s_off_self + r, 0], [1, d_cols],
                )
                pl.store(self_tile, [r_off_self + r, 0], recv)

            # 2) Set(1) notify every peer.
            for peer in pl.range(group_size):
                if peer != my_rank:
                    pld.system.notify(
                        target=signal_window,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )

            # 3) Ge(1) wait for every peer.
            for src in pl.range(group_size):
                if src != my_rank:
                    pld.system.wait(
                        signal=signal_window,
                        offsets=[src, 0],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )

            # 4) Pull every peer's bucket-for-me.
            for peer in pl.range(group_size):
                if peer != my_rank:
                    n_recv = pl.cast(
                        pl.read(recv_counts, [peer]), pl.INDEX,
                    )
                    r_off = pl.cast(
                        pl.read(recv_offsets, [peer]), pl.INDEX,
                    )
                    for r in pl.range(n_recv):
                        peer_tile = pld.tile.remote_load(
                            send,
                            peer=peer,
                            offsets=[r_off + r, 0],
                            shape=[1, d_cols],
                        )
                        pl.store(peer_tile, [r_off + r, 0], recv)

            return recv

        # ---------- Stage 1: gate (local, replicated) ----------
        @pl.function(type=pl.FunctionType.Inline)
        def _gate(
            self,
            x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            gate_w: pl.Tensor[[HIDDEN, N_EXPERTS], pl.FP32],
            router_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
            expert_indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
            expert_weights: pl.Tensor[[BATCH, TOPK], pl.FP32],
        ):
            score_buf = pl.create_tensor(
                [BATCH, ROUTER_SCORE_PAD], dtype=pl.FP32,
            )
            biased_buf = pl.create_tensor(
                [BATCH, ROUTER_SCORE_PAD], dtype=pl.FP32,
            )

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_matmul"):
                # Initialise output pads once — columns beyond N_EXPERTS
                # keep 0 / NEG_INF so topk is not tricked by uninitialised values.
                score_buf[:, :] = pl.full(
                    [BATCH, ROUTER_SCORE_PAD], dtype=pl.FP32, value=0.0,
                )
                biased_buf[:, :] = pl.full(
                    [BATCH, ROUTER_SCORE_PAD],
                    dtype=pl.FP32, value=ROUTER_FP32_NEG_INF,
                )
                for nb in pl.range(N_EXPERTS // ROUTER_GATE_N_CHUNK):
                    n0 = nb * ROUTER_GATE_N_CHUNK
                    # Cast x per K-chunk → [BATCH,K] FP32 = 16384 B
                    # (Site 5 fix: avoids full [BATCH,HIDDEN] FP32 = 262144 B).
                    x0 = pl.cast(
                        pl.slice(x, [BATCH, ROUTER_GATE_K_CHUNK], [0, 0]),
                        target_type=pl.FP32,
                    )
                    w0 = pl.slice(
                        gate_w,
                        [ROUTER_GATE_K_CHUNK, ROUTER_GATE_N_CHUNK],
                        [0, n0],
                    )
                    logits_n = pl.matmul(x0, w0, out_dtype=pl.FP32)
                    for kb in pl.range(1, HIDDEN // ROUTER_GATE_K_CHUNK):
                        k0 = kb * ROUTER_GATE_K_CHUNK
                        xk = pl.cast(
                            pl.slice(
                                x, [BATCH, ROUTER_GATE_K_CHUNK], [0, k0],
                            ),
                            target_type=pl.FP32,
                        )
                        wk = pl.slice(
                            gate_w,
                            [ROUTER_GATE_K_CHUNK, ROUTER_GATE_N_CHUNK],
                            [k0, n0],
                        )
                        logits_n = pl.matmul_acc(logits_n, xk, wk)
                    # Apply sigmoid per N-chunk — vec ops convert cube→vec
                    # block layout, avoiding blayout mismatch when storing
                    # into pre-created score_buf / biased_buf (vec layout).
                    score_n_chunk = pl.recip(
                        pl.add(pl.exp(pl.neg(logits_n)), 1.0),
                    )
                    bias_chunk = pl.slice(
                        router_bias, [ROUTER_GATE_N_CHUNK], [n0],
                    )
                    bias_row_chunk = pl.reshape(
                        bias_chunk, [1, ROUTER_GATE_N_CHUNK],
                    )
                    # ROUTER-BIAS-BF16 (align moe.py:485-490): vLLM runs
                    # router_bias in BF16; the FP32 loader value's ~0.015 rounding
                    # decides the top-8 tail. Without this the whole-net gate picks
                    # a different top-8 vs vLLM -> wrong routed experts -> argmax
                    # mismatch on the flat next-token distribution.
                    bias_row_chunk = pl.cast(
                        pl.cast(bias_row_chunk, target_type=pl.BF16),
                        target_type=pl.FP32,
                    )
                    biased_n_chunk = pl.add(
                        score_n_chunk,
                        pl.col_expand_mul(
                            pl.full(
                                [BATCH, ROUTER_GATE_N_CHUNK],
                                dtype=pl.FP32, value=1.0,
                            ),
                            bias_row_chunk,
                        ),
                    )
                    score_buf[:, n0 : n0 + ROUTER_GATE_N_CHUNK] = score_n_chunk
                    biased_buf[:, n0 : n0 + ROUTER_GATE_N_CHUNK] = biased_n_chunk

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_topk"):
                topk_idx_tile = pl.create_tensor(
                    [BATCH, ROUTER_TOPK_PAD], dtype=pl.INT32,
                )
                for tt in pl.range(BATCH):
                    row = biased_buf[tt : tt + 1, :]
                    idx_init = pl.arange(
                        0, [1, ROUTER_SCORE_PAD], dtype=pl.UINT32,
                    )
                    srt = pl.sort32(row, idx_init)
                    srt = pl.mrgsort(srt, block_len=64)
                    srt = pl.mrgsort(srt, block_len=256)
                    pairs = srt[:, 0:ROUTER_SORT_PAD]
                    top_idx = pl.gather(
                        pairs, mask_pattern=pl.tile.MaskPattern.P1010,
                        output_dtype=pl.INT32,
                    )
                    topk_idx_tile[tt : tt + 1, :] = top_idx

                gather_all = pl.gather(
                    score_buf, dim=-1, index=topk_idx_tile,
                )
                gather_valid = pl.set_validshape(gather_all, BATCH, TOPK)
                topk_vals_pad = pl.fillpad(
                    gather_valid, pad_value=pl.PadValue.zero,
                )

                denom = pl.reshape(pl.row_sum(topk_vals_pad), [BATCH, 1])
                weights_pad = pl.mul(
                    pl.row_expand_div(topk_vals_pad, denom),
                    ROUTER_SCALE,
                )

                for tt in pl.range(BATCH):
                    for k in pl.range(TOPK):
                        pl.write(
                            expert_indices, [tt, k],
                            pl.read(topk_idx_tile, [tt, k]),
                        )
                        pl.write(
                            expert_weights, [tt, k],
                            pl.read(weights_pad, [tt, k]),
                        )

            return expert_weights

        @pl.function(type=pl.FunctionType.Inline)
        def gate_step(
            self,
            x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            gate_w: pl.Tensor[[HIDDEN, N_EXPERTS], pl.FP32],
            router_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
            expert_indices: pl.Out[pl.Tensor[[BATCH, TOPK], pl.INT32]],
            expert_weights: pl.Out[pl.Tensor[[BATCH, TOPK], pl.FP32]],
        ) -> tuple[
            pl.Tensor[[BATCH, TOPK], pl.INT32],
            pl.Tensor[[BATCH, TOPK], pl.FP32]
        ]:
            self._gate(
                x, gate_w, router_bias, expert_indices, expert_weights,
            )
            return expert_indices, expert_weights

        # ---------- Stage 2: dispatch (EP all-to-all) ----------
        @pl.function(type=pl.FunctionType.Inline)
        def _histogram_and_prefix_sum(
            self,
            indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
            send_counts_per_bucket: pl.Tensor[[per_rank_buckets], pl.INT32],
            send_counts_per_rank: pl.Tensor[[n_ranks], pl.INT32],
            send_offsets_per_rank: pl.Tensor[[n_ranks], pl.INT32],
        ):
            """Local histogram + per-rank prefix-sum prelude."""
            for bkt in pl.range(per_rank_buckets):
                pl.write(
                    send_counts_per_bucket, [bkt], pl.cast(0, pl.INT32),
                )
            for r in pl.range(n_ranks):
                pl.write(
                    send_counts_per_rank, [r], pl.cast(0, pl.INT32),
                )

            for t in pl.range(BATCH):
                for k in pl.range(TOPK):
                    eid = pl.read(indices, [t, k])
                    dst = eid // n_local_experts
                    loc_e = eid - dst * n_local_experts
                    bkt = dst * n_local_experts + loc_e
                    cur = pl.read(send_counts_per_bucket, [bkt])
                    pl.write(
                        send_counts_per_bucket, [bkt],
                        pl.cast(cur + 1, pl.INT32),
                    )
                    r_cur = pl.read(send_counts_per_rank, [dst])
                    pl.write(
                        send_counts_per_rank, [dst],
                        pl.cast(r_cur + 1, pl.INT32),
                    )

            pl.write(send_offsets_per_rank, [0], pl.cast(0, pl.INT32))
            for r in pl.range(1, n_ranks):
                prev_off = pl.read(send_offsets_per_rank, [r - 1])
                prev_cnt = pl.read(send_counts_per_rank, [r - 1])
                pl.write(
                    send_offsets_per_rank, [r],
                    pl.cast(prev_off + prev_cnt, pl.INT32),
                )

        @pl.function(type=pl.FunctionType.Inline)
        def _pack_send_payload(
            self,
            x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
            send_counts_per_bucket: pl.Tensor[[per_rank_buckets], pl.INT32],
            send_offsets_per_rank: pl.Tensor[[n_ranks], pl.INT32],
            send_buf: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.BF16],
            cursor_per_bucket: pl.Tensor[[per_rank_buckets], pl.INT32],
            bucket_offset: pl.Tensor[[per_rank_buckets], pl.INT32],
        ):
            """Pack outgoing tokens into ``send_buf`` ordered by (dst, loc_e)."""
            for r in pl.range(n_ranks):
                rank_off = pl.read(send_offsets_per_rank, [r])
                pl.write(
                    bucket_offset, [r * n_local_experts],
                    pl.cast(rank_off, pl.INT32),
                )
                pl.write(
                    cursor_per_bucket, [r * n_local_experts],
                    pl.cast(rank_off, pl.INT32),
                )
                for e in pl.range(1, n_local_experts):
                    prev_off = pl.read(
                        bucket_offset, [r * n_local_experts + e - 1],
                    )
                    prev_cnt = pl.read(
                        send_counts_per_bucket,
                        [r * n_local_experts + e - 1],
                    )
                    new_off = pl.cast(prev_off + prev_cnt, pl.INT32)
                    pl.write(
                        bucket_offset, [r * n_local_experts + e], new_off,
                    )
                    pl.write(
                        cursor_per_bucket, [r * n_local_experts + e],
                        new_off,
                    )

            for t in pl.range(BATCH):
                for k in pl.range(TOPK):
                    eid = pl.read(indices, [t, k])
                    dst = eid // n_local_experts
                    loc_e = eid - dst * n_local_experts
                    bkt = dst * n_local_experts + loc_e
                    slot_i32 = pl.read(cursor_per_bucket, [bkt])
                    slot = pl.cast(slot_i32, pl.INDEX)
                    x_tile = pl.load(x, [t, 0], [1, HIDDEN])
                    pl.store(x_tile, [slot, 0], send_buf)
                    pl.write(
                        cursor_per_bucket, [bkt],
                        pl.cast(slot_i32 + 1, pl.INT32),
                    )

        @pl.function(type=pl.FunctionType.Inline)
        def _build_local_expert_csr(
            self,
            pub_counts: pld.DistributedTensor[
                [n_ranks * n_ranks, n_local_experts_pad], pl.INT32
            ],
            local_expert_offset: pl.Tensor[[n_local_experts], pl.INT32],
            local_expert_count: pl.Tensor[[n_local_experts], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ):
            """Receiver-side CSR: scan pub_counts for my dst slot."""
            for e in pl.range(n_local_experts):
                acc = pl.cast(0, pl.INT32)
                for s in pl.range(n_ranks):
                    acc = acc + pl.read(
                        pub_counts, [s * n_ranks + my_rank, e],
                    )
                pl.write(local_expert_count, [e], pl.cast(acc, pl.INT32))

            pl.write(local_expert_offset, [0], pl.cast(0, pl.INT32))
            for e in pl.range(1, n_local_experts):
                prev_off = pl.read(local_expert_offset, [e - 1])
                prev_cnt = pl.read(local_expert_count, [e - 1])
                pl.write(
                    local_expert_offset, [e],
                    pl.cast(prev_off + prev_cnt, pl.INT32),
                )

        @pl.function(type=pl.FunctionType.Inline)
        def _build_inverse_map(
            self,
            indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
            pub_counts: pld.DistributedTensor[
                [n_ranks * n_ranks, n_local_experts_pad], pl.INT32
            ],
            inverse_map: pl.Tensor[[BATCH, TOPK], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ):
            """Encode (dst_rank, dst_row_in_recv_buf) into one INT32 per (t,k)."""
            cursor = pl.create_tensor(
                [per_rank_buckets], dtype=pl.INT32,
            )
            for bkt in pl.range(per_rank_buckets):
                pl.write(cursor, [bkt], pl.cast(0, pl.INT32))

            for t in pl.range(BATCH):
                for k in pl.range(TOPK):
                    eid = pl.read(indices, [t, k])
                    dst = eid // n_local_experts
                    loc_e = eid - dst * n_local_experts
                    bkt = dst * n_local_experts + loc_e

                    src_off = pl.cast(0, pl.INT32)
                    for s in pl.range(n_ranks):
                        if s < my_rank:
                            src_off = src_off + pl.read(
                                pub_counts, [s * n_ranks + dst, loc_e],
                            )

                    loc_e_off = pl.cast(0, pl.INT32)
                    for prev_e in pl.range(n_local_experts):
                        if prev_e < loc_e:
                            for s in pl.range(n_ranks):
                                loc_e_off = loc_e_off + pl.read(
                                    pub_counts,
                                    [s * n_ranks + dst, prev_e],
                                )

                    my_cursor_val = pl.read(cursor, [bkt])
                    dst_row = loc_e_off + src_off + my_cursor_val
                    packed = (
                        dst * pl.cast(local_recv_max, pl.INT32) + dst_row
                    )
                    pl.write(inverse_map, [t, k], pl.cast(packed, pl.INT32))
                    pl.write(
                        cursor, [bkt],
                        pl.cast(my_cursor_val + 1, pl.INT32),
                    )

        @pl.function(type=pl.FunctionType.InCore)
        def _quant_moe_input(
            self,
            x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            x_i8_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.INT8]],
            x_scale_out: pl.Out[pl.Tensor[[BATCH, 8], pl.FP32]],
        ):
            q_amax = pl.full([1, BATCH], dtype=pl.FP32, value=1e-4)
            for qab in pl.range(HIDDEN // ROUTED_GATE_K_CHUNK):
                qa0 = qab * ROUTED_GATE_K_CHUNK
                qac = pl.cast(
                    pl.slice(x, [BATCH, ROUTED_GATE_K_CHUNK], [0, qa0]),
                    target_type=pl.FP32,
                )
                q_amax = pl.maximum(
                    q_amax,
                    pl.reshape(
                        pl.row_max(pl.maximum(qac, pl.neg(qac))), [1, BATCH],
                    ),
                )
            q_inv_row = pl.div(
                pl.full([1, BATCH], dtype=pl.FP32, value=127.0), q_amax,
            )
            q_inv = pl.reshape(q_inv_row, [BATCH, 1])
            q_scale = pl.reshape(pl.recip(q_inv_row), [BATCH, 1])
            x_scale_out[0:BATCH, 0:1] = q_scale
            for qnb in pl.range(HIDDEN // ROUTED_GATE_K_CHUNK):
                qn0 = qnb * ROUTED_GATE_K_CHUNK
                qch = pl.cast(
                    pl.slice(x, [BATCH, ROUTED_GATE_K_CHUNK], [0, qn0]),
                    target_type=pl.FP32,
                )
                qq = pl.cast(
                    pl.row_expand_mul(qch, q_inv),
                    target_type=pl.INT32, mode="rint",
                )
                qf = pl.cast(qq, target_type=pl.FP16, mode="round")
                qi8 = pl.cast(qf, target_type=pl.INT8, mode="trunc")
                x_i8_out[0:BATCH, qn0 : qn0 + ROUTED_GATE_K_CHUNK] = qi8
            return x_i8_out, x_scale_out


        # ---------- Stage 2: dispatch (EP all-to-all, PULL) ----------
        @pl.function(type=pl.FunctionType.InCore)
        def _dispatch_pack_publish(  # noqa: PLR0913
            self,
            x: pl.Tensor[[BATCH, HIDDEN], pl.INT8],
            x_scale: pl.Tensor[[BATCH, 8], pl.FP32],
            expert_indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
            send_x: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
            send_scale: pld.DistributedTensor[[local_recv_max, 8], pl.FP32],
            send_route: pld.DistributedTensor[
                [local_recv_max, idx_pad], pl.INT32
            ],
            pub_counts: pld.DistributedTensor[
                [n_ranks * n_ranks, n_local_experts_pad], pl.INT32
            ],
            my_rank: pl.Scalar[pl.INT32],
        ):
            # Dispatch task 1 (PULL): histogram -> pack own INT8 tokens + per-token
            # scale + route(t*TOPK+k) into own peer-readable send_* windows in
            # (dst,loc_e) bucket order -> publish my local pub_counts rows.
            # send_* and pub_counts writes are LOCAL; peers pull them later.
            # boundary drains both before the pull's pack_done rendezvous.
            send_counts_bkt = pl.create_tensor(
                [per_rank_buckets], dtype=pl.INT32,
            )
            send_counts_rank = pl.create_tensor([n_ranks], dtype=pl.INT32)
            send_offsets_rank = pl.create_tensor([n_ranks], dtype=pl.INT32)
            self._histogram_and_prefix_sum(
                expert_indices,
                send_counts_bkt, send_counts_rank, send_offsets_rank,
            )
            bucket_off = pl.create_tensor([per_rank_buckets], dtype=pl.INT32)
            cursor_bkt = pl.create_tensor([per_rank_buckets], dtype=pl.INT32)
            for r in pl.range(n_ranks):
                rank_off = pl.cast(r * n_routes_per_rank, pl.INT32)  # moe.py fixed dst-block base
                pl.write(
                    bucket_off, [r * n_local_experts],
                    pl.cast(rank_off, pl.INT32),
                )
                pl.write(
                    cursor_bkt, [r * n_local_experts],
                    pl.cast(rank_off, pl.INT32),
                )
                for e in pl.range(1, n_local_experts):
                    prev_off = pl.read(
                        bucket_off, [r * n_local_experts + e - 1],
                    )
                    prev_cnt = pl.read(
                        send_counts_bkt, [r * n_local_experts + e - 1],
                    )
                    new_off = pl.cast(prev_off + prev_cnt, pl.INT32)
                    pl.write(bucket_off, [r * n_local_experts + e], new_off)
                    pl.write(cursor_bkt, [r * n_local_experts + e], new_off)
            idx_tile = pl.tile.full([1, idx_pad], dtype=pl.INT32, value=0)
            for t in pl.range(BATCH):
                for k in pl.range(TOPK):
                    eid = pl.read(expert_indices, [t, k])
                    dst = eid // n_local_experts
                    loc_e = eid - dst * n_local_experts
                    bkt = dst * n_local_experts + loc_e
                    slot_i32 = pl.read(cursor_bkt, [bkt])
                    slot = pl.cast(slot_i32, pl.INDEX)
                    x_tile = pl.load(x, [t, 0], [1, HIDDEN])
                    pl.store(x_tile, [slot, 0], send_x)
                    sc_tile = pl.load(x_scale, [t, 0], [1, 8])
                    pl.store(sc_tile, [slot, 0], send_scale)
                    pl.tile.write(
                        idx_tile, [0, 0], pl.cast(t * TOPK + k, pl.INT32),
                    )
                    pl.store(idx_tile, [slot, 0], send_route)
                    pl.write(
                        cursor_bkt, [bkt], pl.cast(slot_i32 + 1, pl.INT32),
                    )
            for d in pl.range(n_ranks):
                for e in pl.range(n_local_experts):
                    v = pl.read(send_counts_bkt, [d * n_local_experts + e])
                    pl.write(pub_counts, [my_rank * n_ranks + d, e], v)

        @pl.function(type=pl.FunctionType.InCore)
        def _dispatch_pull(  # noqa: PLR0913
            self,
            send_x: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
            send_scale: pld.DistributedTensor[[local_recv_max, 8], pl.FP32],
            send_route: pld.DistributedTensor[
                [local_recv_max, idx_pad], pl.INT32
            ],
            expert_indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
            pub_counts: pld.DistributedTensor[
                [n_ranks * n_ranks, n_local_experts_pad], pl.INT32
            ],
            pack_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
            recv_x: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
            recv_scale: pld.DistributedTensor[[local_recv_max, 8], pl.FP32],
            recv_r_route: pld.DistributedTensor[
                [local_recv_max, idx_pad], pl.INT32
            ],
            recv_counts: pl.Out[
                pl.Tensor[[n_ranks, n_local_experts_pad], pl.INT32]
            ],
            inverse_map_out: pl.Out[pl.Tensor[[BATCH, TOPK], pl.INT32]],
            local_expert_offset: pl.Out[
                pl.Tensor[[n_local_experts], pl.INT32]
            ],
            local_expert_count: pl.Out[
                pl.Tensor[[n_local_experts], pl.INT32]
            ],
            my_rank: pl.Scalar[pl.INT32],
        ) -> tuple[
            pl.Tensor[[n_ranks, n_local_experts_pad], pl.INT32],
            pl.Tensor[[n_local_experts], pl.INT32],
            pl.Tensor[[n_local_experts], pl.INT32],
            pl.Tensor[[BATCH, TOPK], pl.INT32],
        ]:
            # Dispatch task 2 (PULL): AtomicAdd/Ge rendezvous after all peers pack send_* +
            # published pub_counts) -> per-expert CSR -> gather each incoming token
            # from its source's send_* via remote_load (TGET, local-observable). The
            # gather order (loc_e outer, s ascending, row inner) reproduces the push
            # dst_row exactly. Barrier BEFORE the read (ep_all_to_all pattern); no
            # trailing barrier (dst owns recv_x; send_* are per-layer distinct).
            for peer in pl.range(n_ranks):
                if peer != my_rank:
                    pld.system.notify(
                        target=pack_done_sig, peer=peer,
                        offsets=[my_rank, 0], value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for src in pl.range(n_ranks):
                if src != my_rank:
                    pld.system.wait(
                        signal=pack_done_sig, offsets=[src, 0],
                        expected=1, cmp=pld.WaitCmp.Ge,
                    )
            counts_all = pl.create_tensor(
                [n_ranks * n_ranks, n_local_experts_pad], dtype=pl.INT32,
            )
            for src in pl.range(n_ranks):
                if src == my_rank:
                    for d in pl.range(n_ranks):
                        for e in pl.range(n_local_experts):
                            c = pl.read(
                                pub_counts, [my_rank * n_ranks + d, e],
                            )
                            pl.write(
                                counts_all, [my_rank * n_ranks + d, e],
                                pl.cast(c, pl.INT32),
                            )
                            if d == my_rank:
                                pl.write(
                                    recv_counts, [src, e], pl.cast(c, pl.INT32),
                                )
                else:
                    for d in pl.range(n_ranks):
                        cnt_row = pld.tile.remote_load(
                            pub_counts, peer=src,
                            offsets=[src * n_ranks + d, 0],
                            shape=[1, n_local_experts_pad],
                        )
                        for e in pl.range(n_local_experts):
                            c = pl.read(cnt_row, [0, e])
                            pl.write(
                                counts_all, [src * n_ranks + d, e],
                                pl.cast(c, pl.INT32),
                            )
                            if d == my_rank:
                                pl.write(
                                    recv_counts, [src, e], pl.cast(c, pl.INT32),
                                )
            for e in pl.range(n_local_experts):
                acc = pl.cast(0, pl.INT32)
                for src in pl.range(n_ranks):
                    acc = acc + pl.read(recv_counts, [src, e])
                pl.write(local_expert_count, [e], pl.cast(acc, pl.INT32))
            pl.write(local_expert_offset, [0], pl.cast(0, pl.INT32))
            for e in pl.range(1, n_local_experts):
                prev_off = pl.read(local_expert_offset, [e - 1])
                prev_cnt = pl.read(local_expert_count, [e - 1])
                pl.write(
                    local_expert_offset, [e],
                    pl.cast(prev_off + prev_cnt, pl.INT32),
                )

            cursor_inv = pl.create_tensor([per_rank_buckets], dtype=pl.INT32)
            for bkt_inv in pl.range(per_rank_buckets):
                pl.write(cursor_inv, [bkt_inv], pl.cast(0, pl.INT32))
            for t_inv in pl.range(BATCH):
                for k_inv in pl.range(TOPK):
                    eid_inv = pl.read(expert_indices, [t_inv, k_inv])
                    dst_inv = eid_inv // n_local_experts
                    loc_e_inv = eid_inv - dst_inv * n_local_experts
                    bkt_inv = dst_inv * n_local_experts + loc_e_inv
                    src_off_inv = pl.cast(0, pl.INT32)
                    for s_inv in pl.range(n_ranks):
                        if s_inv < my_rank:
                            src_off_inv = src_off_inv + pl.read(
                                counts_all,
                                [s_inv * n_ranks + dst_inv, loc_e_inv],
                            )
                    loc_e_off_inv = pl.cast(0, pl.INT32)
                    for prev_e_inv in pl.range(n_local_experts):
                        if prev_e_inv < loc_e_inv:
                            for s2_inv in pl.range(n_ranks):
                                loc_e_off_inv = loc_e_off_inv + pl.read(
                                    counts_all,
                                    [s2_inv * n_ranks + dst_inv, prev_e_inv],
                                )
                    cur_inv = pl.read(cursor_inv, [bkt_inv])
                    dst_row_inv = loc_e_off_inv + src_off_inv + cur_inv
                    packed_inv = (
                        dst_inv * pl.cast(local_recv_max, pl.INT32)
                        + dst_row_inv
                    )
                    pl.write(
                        inverse_map_out, [t_inv, k_inv],
                        pl.cast(packed_inv, pl.INT32),
                    )
                    pl.write(
                        cursor_inv, [bkt_inv],
                        pl.cast(cur_inv + 1, pl.INT32),
                    )
            # moe.py ep_all_to_all static fixed-slot pull -> recv_x PEER-MAJOR
            # (peer block at peer*n_routes_per_rank). Self block copied locally;
            # peer blocks pulled via remote_load at compound-scalar my_rank*MAX.
            # NO cross-rank offset, NO runtime pub_counts bound in the pull loop.
            # _dispatch_stage re-packs peer-major -> expert-major.
            _self_base = pl.cast(my_rank * n_routes_per_rank, pl.INDEX)
            for r in pl.range(n_routes_per_rank):
                sxt = pl.load(send_x, [_self_base + r, 0], [1, HIDDEN])
                pl.store(sxt, [_self_base + r, 0], recv_x)
                sst = pl.load(send_scale, [_self_base + r, 0], [1, 8])
                pl.store(sst, [_self_base + r, 0], recv_scale)
                srt = pl.load(send_route, [_self_base + r, 0], [1, idx_pad])
                pl.store(srt, [_self_base + r, 0], recv_r_route)
            for peer in pl.range(n_ranks):
                if peer != my_rank:
                    _peer_base = pl.cast(peer * n_routes_per_rank, pl.INDEX)
                    for r in pl.range(n_routes_per_rank):
                        xt = pld.tile.remote_load(
                            send_x, peer=peer,
                            offsets=[_self_base + r, 0], shape=[1, HIDDEN],
                        )
                        pl.store(xt, [_peer_base + r, 0], recv_x)
                        st = pld.tile.remote_load(
                            send_scale, peer=peer,
                            offsets=[_self_base + r, 0], shape=[1, 8],
                        )
                        pl.store(st, [_peer_base + r, 0], recv_scale)
                        rt = pld.tile.remote_load(
                            send_route, peer=peer,
                            offsets=[_self_base + r, 0], shape=[1, idx_pad],
                        )
                        pl.store(rt, [_peer_base + r, 0], recv_r_route)
            return recv_counts, local_expert_offset, local_expert_count, inverse_map_out

        @pl.function(type=pl.FunctionType.InCore)
        def _dispatch_stage(  # noqa: PLR0913
            self,
            recv_x: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
            recv_scale: pld.DistributedTensor[[local_recv_max, 8], pl.FP32],
            recv_r_route: pld.DistributedTensor[
                [local_recv_max, idx_pad], pl.INT32
            ],
            local_expert_offset: pl.Tensor[[n_local_experts], pl.INT32],
            local_expert_count: pl.Tensor[[n_local_experts], pl.INT32],
            local_routed_x_out: pl.Out[
                pl.Tensor[[local_recv_max, HIDDEN], pl.INT8]
            ],
            local_routed_x_scale_out: pl.Out[pl.Tensor[[1, local_recv_max], pl.FP32]],
            recv_r_route_out: pl.Out[pl.Tensor[[local_recv_max], pl.INT32]],
            recv_counts: pl.Tensor[[n_ranks, n_local_experts_pad], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> tuple[
            pl.Tensor[[local_recv_max, HIDDEN], pl.INT8],
            pl.Tensor[[1, local_recv_max], pl.FP32],
            pl.Tensor[[local_recv_max], pl.INT32],
        ]:
            # Dispatch task 3: compact the peer-major fixed-slot pull windows
            # into expert-major tensors using the receiver-local count snapshot.
            # The InCore task boundary keeps pull and compaction separate.
            running = pl.cast(0, pl.INT32)
            for e in pl.range(n_local_experts):
                for src in pl.range(n_ranks):
                    rn = pl.cast(pl.read(recv_counts, [src, e]), pl.INDEX)
                    src_base = pl.cast(src * n_routes_per_rank, pl.INDEX)
                    src_e_off = pl.cast(0, pl.INT32)
                    for prev_e in pl.range(n_local_experts):
                        if prev_e < e:
                            src_e_off = src_e_off + pl.read(recv_counts, [src, prev_e])
                    for row in pl.range(rn):
                        src_row = src_base + pl.cast(src_e_off, pl.INDEX) + row
                        dst_row = pl.cast(running, pl.INDEX) + row
                        tile = pl.load(recv_x, [src_row, 0], [1, HIDDEN])
                        pl.store(tile, [dst_row, 0], local_routed_x_out)
                        pl.write(local_routed_x_scale_out, [0, dst_row], pl.read(recv_scale, [src_row, 0]))
                        pl.write(recv_r_route_out, [dst_row], pl.read(recv_r_route, [src_row, 0]))
                    running = running + pl.cast(rn, pl.INT32)
            return local_routed_x_out, local_routed_x_scale_out, recv_r_route_out

        @pl.function(type=pl.FunctionType.Inline)
        def dispatch_step(  # noqa: PLR0913
            self,
            x: pl.Tensor[[BATCH, HIDDEN], pl.INT8],
            x_scale: pl.Tensor[[BATCH, 8], pl.FP32],
            expert_indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
            local_routed_x_out: pl.Out[
                pl.Tensor[[local_recv_max, HIDDEN], pl.INT8]
            ],
            local_routed_x_scale_out: pl.Out[pl.Tensor[[1, local_recv_max], pl.FP32]],
            local_expert_offset: pl.Out[
                pl.Tensor[[n_local_experts], pl.INT32]
            ],
            local_expert_count: pl.Out[
                pl.Tensor[[n_local_experts], pl.INT32]
            ],
            recv_r_route_out: pl.Out[pl.Tensor[[local_recv_max], pl.INT32]],
            pub_counts: pld.DistributedTensor[
                [n_ranks * n_ranks, n_local_experts_pad], pl.INT32
            ],
            count_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
            recv_x: pld.DistributedTensor[
                [local_recv_max, HIDDEN], pl.INT8
            ],
            recv_scale: pld.DistributedTensor[[local_recv_max, 8], pl.FP32],
            recv_r_route: pld.DistributedTensor[
                [local_recv_max, idx_pad], pl.INT32
            ],
            data_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
            send_x: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
            send_scale: pld.DistributedTensor[[local_recv_max, 8], pl.FP32],
            send_route: pld.DistributedTensor[[local_recv_max, idx_pad], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> tuple[
            pl.Tensor[[local_recv_max, HIDDEN], pl.INT8],
            pl.Tensor[[1, local_recv_max], pl.FP32],
            pl.Tensor[[n_local_experts], pl.INT32],
            pl.Tensor[[n_local_experts], pl.INT32],
            pl.Tensor[[local_recv_max], pl.INT32],
            pl.Tensor[[BATCH, TOPK], pl.INT32]
        ]:
            # Pull EP dispatch, split into 3 InCore tasks so local pack,
            # pack_done rendezvous + remote_load pull, and stage compact have
            # explicit task boundaries.
            self._dispatch_pack_publish(
                x, x_scale, expert_indices,
                send_x, send_scale, send_route, pub_counts, my_rank,
            )
            recv_counts = pl.create_tensor(
                [n_ranks, n_local_experts_pad], dtype=pl.INT32,
            )
            inverse_map = pl.create_tensor([BATCH, TOPK], dtype=pl.INT32)
            recv_counts, local_expert_offset, local_expert_count, inverse_map = self._dispatch_pull(
                send_x, send_scale, send_route, expert_indices, pub_counts, count_done_sig,
                recv_x, recv_scale, recv_r_route, recv_counts, inverse_map,
                local_expert_offset, local_expert_count, my_rank,
            )
            local_routed_x_out, local_routed_x_scale_out, recv_r_route_out = self._dispatch_stage(
                recv_x, recv_scale, recv_r_route,
                local_expert_offset, local_expert_count,
                local_routed_x_out, local_routed_x_scale_out, recv_r_route_out,
                recv_counts, my_rank,
            )
            return (
                local_routed_x_out,
                local_routed_x_scale_out,
                local_expert_offset,
                local_expert_count,
                recv_r_route_out,
                inverse_map,
            )

        # ---------- Stage 3a: expert_routed (local 36 experts) ----------
        @pl.function(type=pl.FunctionType.Inline)
        def _expert_routed(  # noqa: PLR0913, PLR0915
            self,
            local_routed_x: pl.Tensor[[local_recv_max, HIDDEN], pl.INT8],
            local_routed_x_scale: pl.Tensor[[1, local_recv_max], pl.FP32],
            local_expert_offset: pl.Tensor[[n_local_experts], pl.INT32],
            local_expert_count: pl.Tensor[[n_local_experts], pl.INT32],
            w_gate: pl.Tensor[
                [n_local_experts, HIDDEN, inter], pl.INT8
            ],
            w_gate_scale: pl.Tensor[[n_local_experts, inter], pl.FP32],
            w_up: pl.Tensor[
                [n_local_experts, HIDDEN, inter], pl.INT8
            ],
            w_up_scale: pl.Tensor[[n_local_experts, inter], pl.FP32],
            w_down: pl.Tensor[
                [n_local_experts, inter, HIDDEN], pl.INT8
            ],
            w_down_scale: pl.Tensor[[n_local_experts, HIDDEN], pl.FP32],
            local_routed_y: pl.Tensor[
                [local_recv_max, HIDDEN], pl.BF16
            ],
        ):
            for e in pl.parallel(n_local_experts):
                n_rows = pl.read(local_expert_count, [e])
                offset_i32 = pl.read(local_expert_offset, [e])
                offset = pl.cast(offset_i32, pl.INDEX)
                for tile_idx in pl.range(N_RECV_TILES):
                    tile_row0_i32 = pl.cast(tile_idx * RECV_TILE, pl.INT32)
                    tile_rem = n_rows - tile_row0_i32
                    if tile_rem > 0:
                        tile_row0 = pl.cast(tile_row0_i32, pl.INDEX)
                        tile_offset = offset + tile_row0
                        tile_valid = pl.cast(
                            pl.min(pl.cast(RECV_TILE, pl.INT32), tile_rem),
                            pl.INDEX,
                        )

                        h_bf16 = pl.create_tensor(
                            [RECV_TILE, inter], dtype=pl.BF16,
                        )

                        for nb in pl.spmd(
                            inter // ROUTED_GATE_N_CHUNK,
                            name_hint="expert_gate_up",
                        ):
                            n0 = nb * ROUTED_GATE_N_CHUNK
                            x0 = pl.slice(
                                local_routed_x,
                                [RECV_TILE, ROUTED_GATE_K_CHUNK],
                                [tile_offset, 0],
                            )
                            wg0_2d = pl.reshape(
                                pl.slice(
                                    w_gate,
                                    [1, ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                                    [e, 0, n0],
                                ),
                                [ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                            )
                            wu0_2d = pl.reshape(
                                pl.slice(
                                    w_up,
                                    [1, ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                                    [e, 0, n0],
                                ),
                                [ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                            )
                            gate_acc = pl.matmul(x0, wg0_2d, out_dtype=pl.INT32)
                            up_acc = pl.matmul(x0, wu0_2d, out_dtype=pl.INT32)
                            for kb in pl.range(1, HIDDEN // ROUTED_GATE_K_CHUNK):
                                k0 = kb * ROUTED_GATE_K_CHUNK
                                xk = pl.slice(
                                    local_routed_x,
                                    [RECV_TILE, ROUTED_GATE_K_CHUNK],
                                    [tile_offset, k0],
                                )
                                wgk = pl.reshape(
                                    pl.slice(
                                        w_gate,
                                        [
                                            1,
                                            ROUTED_GATE_K_CHUNK,
                                            ROUTED_GATE_N_CHUNK,
                                        ],
                                        [e, k0, n0],
                                    ),
                                    [ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                                )
                                wuk = pl.reshape(
                                    pl.slice(
                                        w_up,
                                        [
                                            1,
                                            ROUTED_GATE_K_CHUNK,
                                            ROUTED_GATE_N_CHUNK,
                                        ],
                                        [e, k0, n0],
                                    ),
                                    [ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                                )
                                gate_acc = pl.matmul_acc(gate_acc, xk, wgk)
                                up_acc = pl.matmul_acc(up_acc, xk, wuk)

                            # Per-token act scale: contiguous [1,RECV_TILE] row-
                            # slice of the UNPADDED local_routed_x_scale + reshape [RECV_TILE,1]
                            # (ccec ND2ND-safe; a [RECV_TILE,1] col-slice of a
                            # padded tensor is a strided ColMajor TLOAD ccec rejects).
                            x_scale_col = pl.reshape(
                                pl.slice(
                                    local_routed_x_scale, [1, RECV_TILE], [0, tile_offset],
                                ),
                                [RECV_TILE, 1],
                            )
                            wg_scale_row = pl.slice(
                                w_gate_scale, [1, ROUTED_GATE_N_CHUNK], [e, n0],
                            )
                            wu_scale_row = pl.slice(
                                w_up_scale, [1, ROUTED_GATE_N_CHUNK], [e, n0],
                            )
                            gate_2d = pl.col_expand_mul(
                                pl.row_expand_mul(
                                    pl.cast(
                                        gate_acc, target_type=pl.FP32, mode="none",
                                    ),
                                    x_scale_col,
                                ),
                                wg_scale_row,
                            )
                            up_2d = pl.col_expand_mul(
                                pl.row_expand_mul(
                                    pl.cast(
                                        up_acc, target_type=pl.FP32, mode="none",
                                    ),
                                    x_scale_col,
                                ),
                                wu_scale_row,
                            )
                            sigmoid = pl.recip(
                                pl.add(pl.exp(pl.neg(gate_2d)), 1.0),
                            )
                            silu = pl.mul(gate_2d, sigmoid)
                            if _routed_swiglu_step:
                                silu_c = pl.minimum(silu, _routed_swiglu_limit)
                                up_c = pl.maximum(
                                    pl.minimum(up_2d, _routed_swiglu_limit),
                                    -_routed_swiglu_limit,
                                )
                                gated = pl.mul(silu_c, up_c)
                            else:
                                gated = pl.mul(silu, up_2d)

                            gated_v = pl.set_validshape(
                                gated, tile_valid, ROUTED_GATE_N_CHUNK,
                            )
                            gated_m = pl.fillpad(
                                gated_v, pad_value=pl.PadValue.zero,
                            )
                            h_bf16[
                                :, n0 : n0 + ROUTED_GATE_N_CHUNK
                            ] = pl.cast(gated_m, target_type=pl.BF16)

                        # Per-token INT8 requant of the swiglu intermediate for
                        # the INT8 down-proj (moe.py h_i8: BARE slice amax, no
                        # set_validshape/fillpad — h_bf16 padding rows are already
                        # zero from the gated fillpad above).
                        h_i8 = pl.create_tensor(
                            [RECV_TILE, inter], dtype=pl.INT8,
                        )
                        with pl.at(
                            level=pl.Level.CORE_GROUP, name_hint="routed_h_quant",
                        ):
                            eh_amax = pl.full(
                                [1, RECV_TILE], dtype=pl.FP32, value=1e-4,
                            )
                            for hqa in pl.range(inter // ROUTED_GATE_N_CHUNK):
                                hqa0 = hqa * ROUTED_GATE_N_CHUNK
                                eh_a = pl.cast(
                                    pl.slice(
                                        h_bf16,
                                        [RECV_TILE, ROUTED_GATE_N_CHUNK],
                                        [0, hqa0],
                                    ),
                                    target_type=pl.FP32,
                                )
                                eh_amax = pl.maximum(
                                    eh_amax,
                                    pl.reshape(
                                        pl.row_max(
                                            pl.maximum(eh_a, pl.neg(eh_a)),
                                        ),
                                        [1, RECV_TILE],
                                    ),
                                )
                            # Keep the native W8A8 row scale mathematically equal
                            # to 127 / amax. pl.recip also lowers through TDIVS on
                            # A2/A3, so this spelling is not a TDIV-avoidance fix;
                            # leave the math unchanged during the signal-layout A/B.
                            eh_sq_row = pl.mul(
                                pl.recip(eh_amax),
                                pl.full(
                                    [1, RECV_TILE],
                                    dtype=pl.FP32,
                                    value=127.0,
                                ),
                            )
                            h_scale_dq = pl.reshape(
                                pl.recip(eh_sq_row), [RECV_TILE, 1],
                            )
                            eh_sq_col = pl.reshape(eh_sq_row, [RECV_TILE, 1])
                            for hqn in pl.range(inter // ROUTED_GATE_N_CHUNK):
                                hqn0 = hqn * ROUTED_GATE_N_CHUNK
                                eh_q = pl.cast(
                                    pl.slice(
                                        h_bf16,
                                        [RECV_TILE, ROUTED_GATE_N_CHUNK],
                                        [0, hqn0],
                                    ),
                                    target_type=pl.FP32,
                                )
                                eh_scaled = pl.row_expand_mul(eh_q, eh_sq_col)
                                eh_i32 = pl.cast(
                                    eh_scaled, target_type=pl.INT32, mode="rint",
                                )
                                eh_half = pl.cast(
                                    eh_i32, target_type=pl.FP16, mode="round",
                                )
                                h_i8[
                                    :, hqn0 : hqn0 + ROUTED_GATE_N_CHUNK
                                ] = pl.cast(
                                    eh_half, target_type=pl.INT8, mode="trunc",
                                )

                        for db in pl.spmd(
                            HIDDEN // ROUTED_DOWN_N_CHUNK,
                            name_hint="expert_down",
                        ):
                            d0 = db * ROUTED_DOWN_N_CHUNK
                            h0 = pl.slice(
                                h_i8,
                                [RECV_TILE, ROUTED_DOWN_K_CHUNK],
                                [0, 0],
                            )
                            wd0 = pl.reshape(
                                pl.slice(
                                    w_down,
                                    [
                                        1,
                                        ROUTED_DOWN_K_CHUNK,
                                        ROUTED_DOWN_N_CHUNK,
                                    ],
                                    [e, 0, d0],
                                ),
                                [ROUTED_DOWN_K_CHUNK, ROUTED_DOWN_N_CHUNK],
                            )
                            y_acc = pl.matmul(h0, wd0, out_dtype=pl.INT32)
                            for kb2 in pl.range(1, inter // ROUTED_DOWN_K_CHUNK):
                                k0 = kb2 * ROUTED_DOWN_K_CHUNK
                                hk = pl.slice(
                                    h_i8,
                                    [RECV_TILE, ROUTED_DOWN_K_CHUNK],
                                    [0, k0],
                                )
                                wdk = pl.reshape(
                                    pl.slice(
                                        w_down,
                                        [
                                            1,
                                            ROUTED_DOWN_K_CHUNK,
                                            ROUTED_DOWN_N_CHUNK,
                                        ],
                                        [e, k0, d0],
                                    ),
                                    [
                                        ROUTED_DOWN_K_CHUNK,
                                        ROUTED_DOWN_N_CHUNK,
                                    ],
                                )
                                y_acc = pl.matmul_acc(y_acc, hk, wdk)

                            wd_scale_row = pl.slice(
                                w_down_scale, [1, ROUTED_DOWN_N_CHUNK], [e, d0],
                            )
                            y_2d = pl.col_expand_mul(
                                pl.row_expand_mul(
                                    pl.cast(
                                        y_acc, target_type=pl.FP32, mode="none",
                                    ),
                                    h_scale_dq,
                                ),
                                wd_scale_row,
                            )
                            y_v = pl.set_validshape(
                                y_2d, tile_valid, ROUTED_DOWN_N_CHUNK,
                            )
                            y_m = pl.fillpad(
                                y_v, pad_value=pl.PadValue.zero,
                            )
                            local_routed_y = pl.assemble(
                                local_routed_y,
                                pl.cast(y_m, target_type=pl.BF16),
                                [tile_offset, d0],
                            )

            return local_routed_y

        @pl.function(type=pl.FunctionType.Inline)
        def expert_routed_step(
            self,
            local_routed_x: pl.Tensor[[local_recv_max, HIDDEN], pl.INT8],
            local_routed_x_scale: pl.Tensor[[1, local_recv_max], pl.FP32],
            local_expert_offset: pl.Tensor[[n_local_experts], pl.INT32],
            local_expert_count: pl.Tensor[[n_local_experts], pl.INT32],
            w_gate_r: pl.Tensor[
                [n_local_experts, HIDDEN, inter], pl.INT8
            ],
            w_gate_r_scale: pl.Tensor[[n_local_experts, inter], pl.FP32],
            w_up_r: pl.Tensor[
                [n_local_experts, HIDDEN, inter], pl.INT8
            ],
            w_up_r_scale: pl.Tensor[[n_local_experts, inter], pl.FP32],
            w_down_r: pl.Tensor[
                [n_local_experts, inter, HIDDEN], pl.INT8
            ],
            w_down_r_scale: pl.Tensor[[n_local_experts, HIDDEN], pl.FP32],
            local_routed_y: pl.Out[
                pl.Tensor[[local_recv_max, HIDDEN], pl.BF16]
            ],
        ) -> pl.Tensor[[local_recv_max, HIDDEN], pl.BF16]:
            local_routed_y = self._expert_routed(
                local_routed_x,
                local_routed_x_scale,
                local_expert_offset, local_expert_count,
                w_gate_r, w_gate_r_scale,
                w_up_r, w_up_r_scale,
                w_down_r, w_down_r_scale,
                local_routed_y,
            )
            return local_routed_y

        # ---------- Stage 3b: expert_shared (TP-sliced + tp_all_reduce) ----
        @pl.function(type=pl.FunctionType.Inline)
        def _expert_shared_local(
            self,
            x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            w_gate: pl.Tensor[[HIDDEN, sh_inter_local], pl.BF16],
            w_up: pl.Tensor[[HIDDEN, sh_inter_local], pl.BF16],
            w_down: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
            sh_y_shard: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        ):
            # Gate+up and down projections merged into one InCore kernel so that
            # h_tile (Vec SRAM) is live across both projections in the same
            # kernel dispatch.  Two separate pl.at(CORE_GROUP) scopes would
            # become two separate InCore kernel dispatches when expert_shared_step
            # is Inline, and h_tile cannot cross that dispatch boundary.
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_mlp"):
                h_tile = pl.create_tensor(
                    [BATCH, sh_inter_local], dtype=pl.BF16,
                )

                x0 = pl.slice(x, [BATCH, SHARED_GATE_K_CHUNK], [0, 0])
                wg0 = pl.slice(
                    w_gate,
                    [SHARED_GATE_K_CHUNK, SHARED_GATE_N_CHUNK],
                    [0, 0],
                )
                wu0 = pl.slice(
                    w_up,
                    [SHARED_GATE_K_CHUNK, SHARED_GATE_N_CHUNK],
                    [0, 0],
                )
                gate_acc = pl.matmul(x0, wg0, out_dtype=pl.FP32)
                up_acc = pl.matmul(x0, wu0, out_dtype=pl.FP32)
                for kb in pl.range(1, HIDDEN // SHARED_GATE_K_CHUNK):
                    k0 = kb * SHARED_GATE_K_CHUNK
                    xk = pl.slice(x, [BATCH, SHARED_GATE_K_CHUNK], [0, k0])
                    wgk = pl.slice(
                        w_gate,
                        [SHARED_GATE_K_CHUNK, SHARED_GATE_N_CHUNK],
                        [k0, 0],
                    )
                    wuk = pl.slice(
                        w_up,
                        [SHARED_GATE_K_CHUNK, SHARED_GATE_N_CHUNK],
                        [k0, 0],
                    )
                    gate_acc = pl.matmul_acc(gate_acc, xk, wgk)
                    up_acc = pl.matmul_acc(up_acc, xk, wuk)

                sigmoid = pl.recip(
                    pl.add(pl.exp(pl.neg(gate_acc)), 1.0),
                )
                silu = pl.mul(gate_acc, sigmoid)
                if _shared_swiglu_step:
                    silu_c = pl.minimum(silu, _shared_swiglu_limit)
                    up_c = pl.maximum(
                        pl.minimum(up_acc, _shared_swiglu_limit),
                        -_shared_swiglu_limit,
                    )
                    gated = pl.mul(silu_c, up_c)
                else:
                    gated = pl.mul(silu, up_acc)

                h_tile[:, 0:SHARED_GATE_N_CHUNK] = pl.cast(
                    gated, target_type=pl.BF16,
                )

                # Explicit K-chunking for the down projection.
                # K=160 (sh_inter_local) would trigger backend auto-K-split
                # which emits an invalid ``tmov acc→acc``.  Expose the K-loop
                # at the user level using K_INNER=32 (5 passes of 32) so the
                # compiler sees a structured pl.matmul + pl.matmul_acc chain
                # (same pattern as gate/up above) and never needs the copy.
                _SH_DOWN_K_INNER = 32  # hardware K-tile; 160 // 32 == 5 passes
                for db in pl.range(HIDDEN // SHARED_DOWN_N_CHUNK):
                    d0 = db * SHARED_DOWN_N_CHUNK
                    h0_k0 = pl.slice(
                        h_tile, [BATCH, _SH_DOWN_K_INNER], [0, 0],
                    )
                    wd0_k0 = pl.slice(
                        w_down,
                        [_SH_DOWN_K_INNER, SHARED_DOWN_N_CHUNK],
                        [0, d0],
                    )
                    y_acc = pl.matmul(h0_k0, wd0_k0, out_dtype=pl.FP32)
                    for kk in pl.range(1, sh_inter_local // _SH_DOWN_K_INNER):
                        kk0 = kk * _SH_DOWN_K_INNER
                        h0_kk = pl.slice(
                            h_tile, [BATCH, _SH_DOWN_K_INNER], [0, kk0],
                        )
                        wd0_kk = pl.slice(
                            w_down,
                            [_SH_DOWN_K_INNER, SHARED_DOWN_N_CHUNK],
                            [kk0, d0],
                        )
                        y_acc = pl.matmul_acc(y_acc, h0_kk, wd0_kk)
                    sh_y_shard = pl.assemble(
                        sh_y_shard,
                        pl.cast(y_acc, target_type=pl.BF16),
                        [0, d0],
                    )

            return sh_y_shard

        @pl.function(type=pl.FunctionType.Inline)
        def expert_shared_step(  # noqa: PLR0913
            self,
            x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            w_gate_s: pl.Tensor[[HIDDEN, sh_inter_local], pl.BF16],
            w_up_s: pl.Tensor[[HIDDEN, sh_inter_local], pl.BF16],
            w_down_s: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
            sh_y: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            sh_tmp_window: pld.DistributedTensor[
                [BATCH, HIDDEN], pl.BF16
            ],
            sh_signal_window: pld.DistributedTensor[
                [n_ranks, 1], pl.INT32
            ],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            sh_y = self._expert_shared_local(
                x, w_gate_s, w_up_s, w_down_s, sh_y,
            )
            # Phase 15.1 single-rank gate: skip TP=1 (mirror of 15.B).
            if TP_WORLD_SIZE > 1:
                self.tp_all_reduce(
                    sh_y, sh_tmp_window, sh_signal_window, my_rank,
                )
            return sh_y

        # ---------- Stage 4: combine (EP a2a back + weighted gather) ------
        @pl.function(type=pl.FunctionType.InCore)
        def _publish_src_route_table(
            self,
            indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
            src_route_table: pld.DistributedTensor[
                [n_ranks, n_local_experts, n_routes_per_rank], pl.INT32
            ],
            my_rank: pl.Scalar[pl.INT32],
        ):
            cursor = pl.create_tensor(
                [per_rank_buckets], dtype=pl.INT32,
            )
            for i in pl.range(per_rank_buckets):
                pl.write(cursor, [i], pl.cast(0, pl.INT32))

            for t in pl.range(BATCH):
                for k in pl.range(TOPK):
                    eid = pl.read(indices, [t, k])
                    dst = eid // n_local_experts
                    loc_e = eid - dst * n_local_experts
                    bkt = dst * n_local_experts + loc_e
                    idx = pl.read(cursor, [bkt])
                    r_route = pl.cast(t * TOPK + k, pl.INT32)

                    if dst == my_rank:
                        # Self-rank publish: write the scalar r_route
                        # directly into the local view of the
                        # DistributedTensor. Mirrors the self-branch
                        # used for ``pub_counts`` above; avoids the
                        # ``tmp = create_tensor; load; store`` dance
                        # that the InCore tile verifier rejects with
                        # "tile.load requires TensorType ... got TileType".
                        pl.write(
                            src_route_table,
                            [my_rank, loc_e, pl.cast(idx, pl.INDEX)],
                            r_route,
                        )
                    else:
                        pld.system.notify(
                            target=src_route_table,
                            peer=dst,
                            offsets=[
                                my_rank, loc_e, pl.cast(idx, pl.INDEX),
                            ],
                            value=r_route,
                            op=pld.NotifyOp.Set,
                        )
                    pl.write(cursor, [bkt], pl.cast(idx + 1, pl.INT32))

        @pl.function(type=pl.FunctionType.InCore)
        def _push_routed_y_to_sources(  # noqa: PLR0913
            self,
            local_routed_y: pl.Tensor[
                [local_recv_max, HIDDEN], pl.BF16
            ],
            pub_counts: pld.DistributedTensor[
                [n_ranks * n_ranks, n_local_experts_pad], pl.INT32
            ],
            routed_y_buf: pld.DistributedTensor[
                [n_routes_per_rank, HIDDEN], pl.BF16
            ],
            combine_done: pld.DistributedTensor[
                [n_ranks, 1], pl.INT32
            ],
            recv_r_route_out: pl.Tensor[[local_recv_max], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ):
            # zero-done barrier (wave 1): every rank must finish _zero_routed_y_buf
            # (called immediately before this push in the combine caller) before any
            # peer pld.tensor.put lands, else a fast peer's push into a not-yet-zeroed
            # routed_y_buf is clobbered by the local zero -> racy moe_out. Mirrors
            # moe.py combine_step's pub_route_barrier (dropped with src_route_table).
            # Same combine_done window, two-wave: wave 1 =1 here, post-push barrier =2.
            for peer in pl.range(n_ranks):
                if peer != my_rank:
                    pld.system.notify(
                        target=combine_done, peer=peer,
                        offsets=[my_rank, 0], value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for src in pl.range(n_ranks):
                if src != my_rank:
                    pld.system.wait(
                        signal=combine_done, offsets=[src, 0],
                        expected=1, cmp=pld.WaitCmp.Ge,
                    )
            e_cursor = pl.cast(0, pl.INT32)
            for e in pl.range(n_local_experts):
                src_off = pl.cast(0, pl.INT32)
                for src in pl.range(n_ranks):
                    n = pl.cast(
                        pl.read(
                            pub_counts, [src * n_ranks + my_rank, e],
                        ),
                        pl.INDEX,
                    )
                    for row in pl.range(n):
                        local_row = (
                            pl.cast(e_cursor, pl.INDEX)
                            + pl.cast(src_off, pl.INDEX) + row
                        )
                        # r_route rode with the token at dispatch (recv_r_route);
                        # local_row is the same expert-major CSR row order.
                        r_route = pl.read(recv_r_route_out, [local_row])
                        if src == my_rank:
                            tile = pl.load(
                                local_routed_y,
                                [local_row, 0], [1, HIDDEN],
                            )
                            pl.store(tile, [r_route, 0], routed_y_buf)
                        else:
                            # moe.py-aligned combine push: pld.tensor.put
                            # establishes a RAW dep into the gather so the
                            # consumer waits for the push DMA to land. Fire-and-
                            # forget remote_store let the gather read partially-
                            # landed routed_y_buf -> racy moe_out (device: moe_out
                            # row0 5.6/5.0/26 across identical runs). Mirrors the
                            # dispatch recv_x push, which already uses tensor.put.
                            pld.tensor.put(
                                dst=routed_y_buf,
                                peer=src,
                                src=local_routed_y,
                                dst_offsets=[r_route, 0],
                                src_offsets=[local_row, 0],
                                shape=[1, HIDDEN],
                            )
                    src_off = src_off + pl.cast(n, pl.INT32)
                total_e = pl.cast(0, pl.INT32)
                for src2 in pl.range(n_ranks):
                    total_e = total_e + pl.read(
                        pub_counts, [src2 * n_ranks + my_rank, e],
                    )
                e_cursor = e_cursor + total_e

            pld.system.notify(
                target=combine_done, peer=my_rank,
                offsets=[my_rank, 0], value=0, op=pld.NotifyOp.AtomicAdd,
            )
            for peer in pl.range(n_ranks):
                if peer != my_rank:
                    pld.system.notify(
                        target=combine_done,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for src in pl.range(n_ranks):
                if src != my_rank:
                    pld.system.wait(
                        signal=combine_done,
                        offsets=[src, 0],
                        expected=2,
                        cmp=pld.WaitCmp.Ge,
                    )

        @pl.function(type=pl.FunctionType.Inline)
        def _weighted_gather_and_add(
            self,
            routed_y_buf: pld.DistributedTensor[
                [n_routes_per_rank, HIDDEN], pl.BF16
            ],
            expert_weights: pl.Tensor[[BATCH, TOPK], pl.FP32],
            sh_y: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            moe_out: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        ):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_combine"):
                for b in pl.range(BATCH):
                    acc = pl.cast(
                        pl.load(sh_y, [b, 0], [1, HIDDEN]),
                        target_type=pl.FP32,
                    )
                    for k in pl.range(TOPK):
                        w_fp = pl.read(expert_weights, [b, k])

                        r_route = b * TOPK + k
                        row_fp32 = pl.cast(
                            pl.load(
                                routed_y_buf, [r_route, 0], [1, HIDDEN],
                            ),
                            target_type=pl.FP32,
                        )
                        weighted = pl.mul(row_fp32, w_fp)
                        acc = pl.add(acc, weighted)

                    pl.store(
                        pl.cast(acc, target_type=pl.BF16),
                        [b, 0],
                        moe_out,
                    )

            return moe_out

        @pl.function(type=pl.FunctionType.InCore)
        def _zero_routed_y_buf(
            self,
            routed_y_buf: pld.DistributedTensor[
                [n_routes_per_rank, HIDDEN], pl.BF16
            ],
        ):
            # Zero-init: data windows are NOT auto-zeroed (only signal windows).
            # _weighted_gather_and_add reads all n_routes_per_rank rows; any slot
            # the combine push misses would else read uninitialised garbage
            # (nondeterministic ~0 / ~1e11 moe_out). Mirrors the validated
            # moe.EpTpMoE._zero_routed_y_buf.
            for r in pl.range(n_routes_per_rank):
                routed_y_buf[r : r + 1, :] = pl.full(
                    [1, HIDDEN], dtype=pl.BF16, value=0.0,
                )

        @pl.function(type=pl.FunctionType.InCore)
        def _stage_routed_src(
            self,
            local_routed_y: pl.Tensor[[local_recv_max, HIDDEN], pl.BF16],
            routed_src_buf: pld.DistributedTensor[
                [local_recv_max, HIDDEN], pl.BF16
            ],
        ):
            # Combine task 1 (PULL): copy the expert holder's routed output into
            # its OWN peer-readable window (local writes). The InCore task boundary
            # drains these before the pull rendezvous, so peers read landed data.
            for row in pl.range(0, local_recv_max, stage_rows):
                tile = pl.load(local_routed_y, [row, 0], [stage_rows, HIDDEN])
                pl.store(tile, [row, 0], routed_src_buf)

        @pl.function(type=pl.FunctionType.InCore)
        def _pull_routed_y(  # noqa: PLR0913
            self,
            routed_src_buf: pld.DistributedTensor[
                [local_recv_max, HIDDEN], pl.BF16
            ],
            expert_indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
            inverse_map: pl.Tensor[[BATCH, TOPK], pl.INT32],
            routed_y_buf: pld.DistributedTensor[
                [n_routes_per_rank, HIDDEN], pl.BF16
            ],
            combine_done: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ):
            # Combine task 2 (PULL): AtomicAdd/Ge rendezvous after all holders
            # stage routed_src_buf. Consume the source-local inverse_map produced
            # by dispatch, then load each routed row from its holder. Self rows use
            # local pl.load; peer rows use remote_load.
            for peer in pl.range(n_ranks):
                if peer != my_rank:
                    pld.system.notify(
                        target=combine_done, peer=peer,
                        offsets=[my_rank, 0], value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for src in pl.range(n_ranks):
                if src != my_rank:
                    pld.system.wait(
                        signal=combine_done, offsets=[src, 0],
                        expected=1, cmp=pld.WaitCmp.Ge,
                    )
            # Use source-local inverse_map produced by dispatch.
            for t in pl.range(BATCH):
                for k in pl.range(TOPK):
                    packed = pl.read(inverse_map, [t, k])
                    dst = packed // pl.cast(local_recv_max, pl.INT32)
                    dst_row = pl.cast(
                        packed - dst * pl.cast(local_recv_max, pl.INT32),
                        pl.INDEX,
                    )
                    r_route = pl.cast(t * TOPK + k, pl.INDEX)
                    if dst == my_rank:
                        tile_local = pl.load(
                            routed_src_buf, [dst_row, 0], [1, HIDDEN],
                        )
                        pl.store(tile_local, [r_route, 0], routed_y_buf)
                    else:
                        tile_remote = pld.tile.remote_load(
                            routed_src_buf, peer=dst,
                            offsets=[dst_row, 0], shape=[1, HIDDEN],
                        )
                        pl.store(tile_remote, [r_route, 0], routed_y_buf)

        @pl.function(type=pl.FunctionType.Inline)
        def combine_step(  # noqa: PLR0913
            self,
            local_routed_y: pl.Tensor[
                [local_recv_max, HIDDEN], pl.BF16
            ],
            recv_r_route_out: pl.Tensor[[local_recv_max], pl.INT32],
            expert_weights: pl.Tensor[[BATCH, TOPK], pl.FP32],
            sh_y: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            moe_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            pub_counts: pld.DistributedTensor[
                [n_ranks * n_ranks, n_local_experts_pad], pl.INT32
            ],
            routed_y_buf: pld.DistributedTensor[
                [n_routes_per_rank, HIDDEN], pl.BF16
            ],
            combine_done_sig: pld.DistributedTensor[
                [n_ranks, 1], pl.INT32
            ],
            expert_indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
            inverse_map: pl.Tensor[[BATCH, TOPK], pl.INT32],
            routed_src_buf: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.BF16],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            # Pull design: expert holders stage routed output locally; source
            # ranks use dispatch-produced inverse_map entries to pull rows back.
            self._zero_routed_y_buf(routed_y_buf)
            self._stage_routed_src(local_routed_y, routed_src_buf)
            self._pull_routed_y(
                routed_src_buf, expert_indices, inverse_map,
                routed_y_buf, combine_done_sig, my_rank,
            )
            moe_out = self._weighted_gather_and_add(
                routed_y_buf, expert_weights, sh_y, moe_out,
            )
            return moe_out

        @pl.function(type=pl.FunctionType.Orchestration)
        def attn_dense_orch(  # noqa: PLR0913, PLR0915
            self,
            current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, hidden_q_local], pl.BF16],
            wk: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16],
            wv: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16],
            q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            rope_cos: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim], pl.FP32],
            rope_sin: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim], pl.FP32],
            k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            wo: pl.Tensor[[layer_qhidden_dyn, HIDDEN], pl.BF16],
            w_g: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, num_heads_local_pad], pl.BF16],
            gate_r: pl.Tensor[[num_heads_local_pad, hidden_q_local], pl.BF16],
            post_rms_d: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            w_gate_d: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTER_LOCAL], pl.BF16],
            w_up_d: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTER_LOCAL], pl.BF16],
            w_down_d: pl.Tensor[[LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16],
            h_mid_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            attn_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
            attn_signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
            mlp_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
            mlp_signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
            layer_idx: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            resid1 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            resid1 = attention_inline(
                current_hidden, input_rms_weight,
                wq, wk, wv, q_norm_weight, k_norm_weight,
                seq_lens, block_table, slot_mapping,
                rope_cos, rope_sin, k_cache, v_cache,
                wo, w_g, gate_r, resid1, layer_idx, layer_idx,
                attn_tmp_window, attn_signal_window, my_rank,
            )
            h_mid_out = dense_mlp_inline(
                resid1, post_rms_d, w_gate_d, w_up_d, w_down_d,
                h_mid_out, layer_idx, layer_idx, mlp_tmp_window,
                mlp_signal_window, my_rank,
            )
            return h_mid_out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(  # noqa: PLR0913, PLR0915
            self,
            current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, hidden_q_local], pl.BF16],
            wk: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16],
            wv: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16],
            q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            rope_cos: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim], pl.FP32],
            rope_sin: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim], pl.FP32],
            k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            wo: pl.Tensor[[layer_qhidden_dyn, HIDDEN], pl.BF16],
            w_g: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, num_heads_local_pad], pl.BF16],
            gate_r: pl.Tensor[[num_heads_local_pad, hidden_q_local], pl.BF16],
            post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            gate_w: pl.Tensor[[HIDDEN, N_EXPERTS], pl.FP32],
            router_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
            w_gate_r: pl.Tensor[[n_local_experts, HIDDEN, inter], pl.INT8],
            w_gate_r_scale: pl.Tensor[[n_local_experts, inter], pl.FP32],
            w_up_r: pl.Tensor[[n_local_experts, HIDDEN, inter], pl.INT8],
            w_up_r_scale: pl.Tensor[[n_local_experts, inter], pl.FP32],
            w_down_r: pl.Tensor[[n_local_experts, inter, HIDDEN], pl.INT8],
            w_down_r_scale: pl.Tensor[[n_local_experts, HIDDEN], pl.FP32],
            w_gate_s: pl.Tensor[[HIDDEN, sh_inter_local], pl.BF16],
            w_up_s: pl.Tensor[[HIDDEN, sh_inter_local], pl.BF16],
            w_down_s: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
            next_hidden_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            attn_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
            attn_signal_window: pld.DistributedTensor[
                [tp_size, 1], pl.INT32
            ],
            pub_counts: pld.DistributedTensor[
                [n_ranks * n_ranks, n_local_experts_pad], pl.INT32
            ],
            count_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
            recv_x: pld.DistributedTensor[
                [local_recv_max, HIDDEN], pl.INT8
            ],
            recv_scale: pld.DistributedTensor[[local_recv_max, 8], pl.FP32],
            send_x: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
            send_scale: pld.DistributedTensor[[local_recv_max, 8], pl.FP32],
            send_route: pld.DistributedTensor[[local_recv_max, idx_pad], pl.INT32],
            data_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
            recv_r_route: pld.DistributedTensor[
                [local_recv_max, idx_pad], pl.INT32
            ],
            sh_tmp_window: pld.DistributedTensor[
                [BATCH, HIDDEN], pl.BF16
            ],
            sh_signal_window: pld.DistributedTensor[
                [n_ranks, 1], pl.INT32
            ],
            routed_y_buf: pld.DistributedTensor[
                [n_routes_per_rank, HIDDEN], pl.BF16
            ],
            combine_done_sig: pld.DistributedTensor[
                [n_ranks, 1], pl.INT32
            ],
            routed_src_buf: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.BF16],
            norm_layer_idx: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            # ── A': input IS h_mid (attn+dense_mlp already done in attn_dense_orch). ───
            resid1 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            for _ac in pl.range(HIDDEN // K_CHUNK):
                _c0 = _ac * K_CHUNK
                resid1 = pl.assemble(
                    resid1,
                    pl.slice(current_hidden, [BATCH, K_CHUNK], [0, _c0]),
                    [0, _c0],
                )

            # ── B: post-attention zero-centred RMSNorm of resid1. ──────
            hidden_blocks = HIDDEN // K_CHUNK
            post_norm = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            resid1_fp32 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
            with pl.at(
                level=pl.Level.CORE_GROUP, name_hint="moe_post_rmsnorm_zc",
            ):
                for kb in pl.range(hidden_blocks):
                    k0 = kb * K_CHUNK
                    rchunk = pl.cast(
                        pl.slice(resid1, [BATCH, K_CHUNK], [0, k0]),
                        target_type=pl.FP32,
                    )
                    resid1_fp32 = pl.assemble(resid1_fp32, rchunk, [0, k0])

                sq_sum = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
                for kb2 in pl.range(hidden_blocks):
                    k0 = kb2 * K_CHUNK
                    ck = pl.slice(resid1_fp32, [BATCH, K_CHUNK], [0, k0])
                    sq_sum = pl.add(
                        sq_sum,
                        pl.reshape(
                            pl.row_sum(pl.mul(ck, ck)),
                            [1, BATCH],
                        ),
                    )
                inv_rms_moe = pl.recip(
                    pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)),
                )
                inv_rms_col = pl.reshape(inv_rms_moe, [BATCH, 1])
                for kb3 in pl.range(hidden_blocks):
                    k0 = kb3 * K_CHUNK
                    norm_chunk = pl.slice(
                        resid1_fp32, [BATCH, K_CHUNK], [0, k0],
                    )
                    gamma = pl.slice(
                        post_rms_weight, [1, K_CHUNK], [norm_layer_idx, k0],
                    )
                    scaled = pl.row_expand_mul(norm_chunk, inv_rms_col)
                    normed = pl.col_expand_mul(scaled, pl.add(gamma, 1.0))
                    post_norm = pl.assemble(
                        post_norm,
                        pl.cast(normed, target_type=pl.BF16),
                        [0, k0],
                    )

            # ── C: EP+TP MoE chip_orch -> moe_out (Phase X.8 inlined). ─
            # Body copied verbatim from ``moe.EpTpMoE.chip_orch``; calls the
            # inlined per-stage methods (``self.gate_step`` /
            # ``self.dispatch_step`` / ``self.expert_routed_step`` /
            # ``self.expert_shared_step`` / ``self.combine_step``) directly
            # rather than instantiating a separate ``@pl.program``.
            moe_out = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)

            # 1) Gate (local, replicated).
            expert_indices = pl.create_tensor([BATCH, TOPK], dtype=pl.INT32)
            expert_weights = pl.create_tensor([BATCH, TOPK], dtype=pl.FP32)
            expert_indices, expert_weights = self.gate_step(
                post_norm, gate_w, router_bias,
                expert_indices, expert_weights,
            )

            # 2) Shared-expert lane (TP-sliced + tp_all_reduce).
            sh_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            sh_y = self.expert_shared_step(
                post_norm, w_gate_s, w_up_s, w_down_s, sh_y,
                sh_tmp_window, sh_signal_window, my_rank,
            )


            # 1A: per-token INT8 dynamic-quant of the MoE input BEFORE
            # dispatch (dispatch-side; shrinks recv_x 8→4MB/layer).
            x_disp_i8 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.INT8)
            x_disp_scale = pl.create_tensor([BATCH, 8], dtype=pl.FP32)
            (x_disp_i8, x_disp_scale) = self._quant_moe_input(
                post_norm, x_disp_i8, x_disp_scale,
            )
            # 3) Dispatch (EP fixed-slot pull).
            local_routed_x = pl.create_tensor(
                [local_recv_max, HIDDEN], dtype=pl.INT8,
            )
            local_routed_x_scale = pl.create_tensor(
                [1, local_recv_max], dtype=pl.FP32,
            )
            local_expert_offset = pl.create_tensor(
                [n_local_experts], dtype=pl.INT32,
            )
            local_expert_count = pl.create_tensor(
                [n_local_experts], dtype=pl.INT32,
            )
            # Per-recv-row r_route (= source t*TOPK+k), staged from recv_r_route;
            # combine uses it to scatter the routed output back to the source.
            recv_r_route_out = pl.create_tensor(
                [local_recv_max], dtype=pl.INT32,
            )
            (
                local_routed_x,
                local_routed_x_scale,
                local_expert_offset,
                local_expert_count,
                recv_r_route_out,
                inverse_map,
            ) = self.dispatch_step(
                x_disp_i8, x_disp_scale, expert_indices,
                local_routed_x, local_routed_x_scale,
                local_expert_offset, local_expert_count, recv_r_route_out,
                pub_counts, count_done_sig, recv_x, recv_scale, recv_r_route, data_done_sig,
                send_x, send_scale, send_route,
                my_rank,
            )

            # 4) Routed experts (local 36).
            local_routed_y = pl.create_tensor(
                [local_recv_max, HIDDEN], dtype=pl.BF16,
            )
            local_routed_y = self.expert_routed_step(
                local_routed_x,
                local_routed_x_scale,
                local_expert_offset, local_expert_count,
                w_gate_r, w_gate_r_scale, w_up_r, w_up_r_scale,
                w_down_r, w_down_r_scale,
                local_routed_y,
            )

            # 5) Combine (EP pull back + weighted gather + sh_y add).
            moe_out = self.combine_step(
                local_routed_y,
                recv_r_route_out, expert_weights, sh_y,
                moe_out,
                pub_counts,
                routed_y_buf, combine_done_sig,
                expert_indices, inverse_map, routed_src_buf,
                my_rank,
            )

            # ── D: residual add: next_hidden_out = resid1 + moe_out. ───
            with pl.at(
                level=pl.Level.CORE_GROUP, name_hint="moe_residual_add",
            ):
                for kb4 in pl.range(hidden_blocks):
                    k0 = kb4 * K_CHUNK
                    m = pl.cast(
                        pl.slice(moe_out, [BATCH, K_CHUNK], [0, k0]),
                        target_type=pl.FP32,
                    )
                    r = pl.slice(resid1_fp32, [BATCH, K_CHUNK], [0, k0])
                    next_hidden_out = pl.assemble(
                        next_hidden_out,
                        pl.cast(pl.add(r, m), target_type=pl.BF16),
                        [0, k0],
                    )
            return next_hidden_out

        # ---- Tail: final RMSNorm + LM head (separate SSA scope) ----------
        @pl.function(
            type=pl.FunctionType.Orchestration,
            attrs={"inline_orchestration": True},
        )
        def lm_head_orch(
            self,
            next_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
            lm_head_weight: pl.Tensor[[VOCAB_LOCAL, HIDDEN], pl.BF16],
            seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            logits_shard_out: pl.Out[
                pl.Tensor[[USER_BATCH_DYN, VOCAB_LOCAL], pl.FP32]
            ],
        ) -> pl.Tensor[[USER_BATCH_DYN, VOCAB_LOCAL], pl.FP32]:
            logits_shard_out = rms_lm_head_inline(
                next_hidden, final_norm_weight, lm_head_weight,
                seq_lens, logits_shard_out,
            )
            return logits_shard_out


        # ---- Dense-prefix full attention + dense MLP (single-layer host-slice). ----
        @pl.function(
            type=pl.FunctionType.Orchestration,
            attrs={"inline_orchestration": True},
        )
        def full_chip_orch(  # noqa: PLR0913
            self,
            current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            wq: pl.Tensor[[HIDDEN, hidden_q_full], pl.BF16],
            wk: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL], pl.BF16],
            wv: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL], pl.BF16],
            q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            rope_cos: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],
            rope_sin: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],
            k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            wo: pl.Tensor[[hidden_q_full, HIDDEN], pl.BF16],
            w_g: pl.Tensor[[HIDDEN, nh_full_pad], pl.BF16],
            gate_r: pl.Tensor[[nh_full_pad, hidden_q_full], pl.BF16],
            post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            w_gate: pl.Tensor[[HIDDEN, INTER_LOCAL], pl.BF16],
            w_up: pl.Tensor[[HIDDEN, INTER_LOCAL], pl.BF16],
            w_down: pl.Tensor[[INTER_LOCAL, HIDDEN], pl.BF16],
            h0_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            attn_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
            attn_signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
            mlp_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
            mlp_signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
            norm_layer_idx: pl.Scalar[pl.INT32],
            attn_layer_idx: pl.Scalar[pl.INT32],
            mlp_layer_idx: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            resid1 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            resid1 = attention_full_inline(
                current_hidden, input_rms_weight, wq, wk, wv,
                q_norm_weight, k_norm_weight,
                seq_lens, block_table, slot_mapping,
                rope_cos, rope_sin, k_cache, v_cache,
                wo, w_g, gate_r, resid1,
                norm_layer_idx, attn_layer_idx,
                attn_tmp_window, attn_signal_window, my_rank,
            )
            h0_out = dense_mlp_inline(
                resid1, post_rms_weight, w_gate, w_up, w_down,
                h0_out, norm_layer_idx, mlp_layer_idx,
                mlp_tmp_window, mlp_signal_window, my_rank,
            )
            return h0_out

        @pl.function(
            type=pl.FunctionType.Orchestration,
            attrs={"inline_orchestration": True},
        )
        def swa_chip_orch(  # noqa: PLR0913
            self,
            current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            wq: pl.Tensor[[HIDDEN, hidden_q_swa], pl.BF16],
            wk: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL], pl.BF16],
            wv: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL], pl.BF16],
            q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            rope_cos: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],
            rope_sin: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],
            k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            wo: pl.Tensor[[hidden_q_swa, HIDDEN], pl.BF16],
            w_g: pl.Tensor[[HIDDEN, nh_swa_pad], pl.BF16],
            gate_r: pl.Tensor[[nh_swa_pad, hidden_q_swa], pl.BF16],
            post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            w_gate: pl.Tensor[[HIDDEN, INTER_LOCAL], pl.BF16],
            w_up: pl.Tensor[[HIDDEN, INTER_LOCAL], pl.BF16],
            w_down: pl.Tensor[[INTER_LOCAL, HIDDEN], pl.BF16],
            hidden_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            attn_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
            attn_signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
            mlp_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
            mlp_signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
            norm_layer_idx: pl.Scalar[pl.INT32],
            attn_layer_idx: pl.Scalar[pl.INT32],
            mlp_layer_idx: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            resid1 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            resid1 = attention_swa_inline(
                current_hidden, input_rms_weight, wq, wk, wv,
                q_norm_weight, k_norm_weight,
                seq_lens, block_table, slot_mapping,
                rope_cos, rope_sin, k_cache, v_cache,
                wo, w_g, gate_r, resid1,
                norm_layer_idx, attn_layer_idx,
                attn_tmp_window, attn_signal_window, my_rank,
            )
            hidden_out = dense_mlp_inline(
                resid1, post_rms_weight, w_gate, w_up, w_down,
                hidden_out, norm_layer_idx, mlp_layer_idx,
                mlp_tmp_window, mlp_signal_window, my_rank,
            )
            return hidden_out

        # ---- MoE-layer full-attn FUSED with MoE-block: attention -> resid1
        # (local, intra-orch) -> post_norm -> EP/TP MoE -> residual. Mirrors the
        # dense full_chip_orch attn->MLP handoff so the attn->MoE handoff is
        # an intra-Submission local tensor, not a cross-orch shared h_mid_out. ----
        @pl.function(
            type=pl.FunctionType.Orchestration,
            attrs={"inline_orchestration": True},
        )
        def full_moe_chip_orch(  # noqa: PLR0913, PLR0915
            self,
            current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            wq: pl.Tensor[[HIDDEN, hidden_q_full], pl.BF16],
            wk: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL], pl.BF16],
            wv: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL], pl.BF16],
            q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            rope_cos: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],
            rope_sin: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],
            k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            wo: pl.Tensor[[hidden_q_full, HIDDEN], pl.BF16],
            w_g: pl.Tensor[[HIDDEN, nh_full_pad], pl.BF16],
            gate_r: pl.Tensor[[nh_full_pad, hidden_q_full], pl.BF16],
            post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            gate_w: pl.Tensor[[HIDDEN, N_EXPERTS], pl.FP32],
            router_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
            w_gate_r: pl.Tensor[[n_local_experts, HIDDEN, inter], pl.INT8],
            w_gate_r_scale: pl.Tensor[[n_local_experts, inter], pl.FP32],
            w_up_r: pl.Tensor[[n_local_experts, HIDDEN, inter], pl.INT8],
            w_up_r_scale: pl.Tensor[[n_local_experts, inter], pl.FP32],
            w_down_r: pl.Tensor[[n_local_experts, inter, HIDDEN], pl.INT8],
            w_down_r_scale: pl.Tensor[[n_local_experts, HIDDEN], pl.FP32],
            w_gate_s: pl.Tensor[[HIDDEN, sh_inter_local], pl.BF16],
            w_up_s: pl.Tensor[[HIDDEN, sh_inter_local], pl.BF16],
            w_down_s: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
            next_hidden_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            dbg_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            resid_hold: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            attn_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
            attn_signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
            pub_counts: pld.DistributedTensor[
                [n_ranks * n_ranks, n_local_experts_pad], pl.INT32
            ],
            count_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
            recv_x: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
            recv_scale: pld.DistributedTensor[[local_recv_max, 8], pl.FP32],
            data_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
            recv_r_route: pld.DistributedTensor[[local_recv_max, idx_pad], pl.INT32],
            send_x: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
            send_scale: pld.DistributedTensor[[local_recv_max, 8], pl.FP32],
            send_route: pld.DistributedTensor[[local_recv_max, idx_pad], pl.INT32],
            sh_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
            sh_signal_window: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
            routed_y_buf: pld.DistributedTensor[[n_routes_per_rank, HIDDEN], pl.BF16],
            combine_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
            routed_src_buf: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.BF16],
            norm_layer_idx: pl.Scalar[pl.INT32],
            attn_layer_idx: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            resid1 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            resid1 = attention_full_inline(
                current_hidden, input_rms_weight, wq, wk, wv,
                q_norm_weight, k_norm_weight,
                seq_lens, block_table, slot_mapping,
                rope_cos, rope_sin, k_cache, v_cache,
                wo, w_g, gate_r, resid1,
                norm_layer_idx, attn_layer_idx,
                attn_tmp_window, attn_signal_window, my_rank,
            )
            # Save the attention residual into a DEDICATED per-layer external
            # buffer resid_hold (write-once here, read-once by section D). The
            # local resid1 create_tensor is reused by gate/shared/routed InCore
            # scratch inside this big fused orch, so section D cannot read resid1
            # directly. Earlier code stashed into next_hidden_out, but that made
            # next_hidden_out have TWO writers (stash + residual_add) => a WAW that
            # RAW-only-v1 (single-value producer_index) cannot serialise across
            # submissions => nondeterministic output. resid_hold has ONE writer
            # (here) and ONE reader (section D); next_hidden_out has ONE writer
            # (section D). Both are clean single-producer RAW.
            with pl.at(
                level=pl.Level.CORE_GROUP, name_hint="stash_resid_hold",
            ):
                for _rs in pl.range(HIDDEN // K_CHUNK):
                    _r0 = _rs * K_CHUNK
                    resid_hold = pl.assemble(
                        resid_hold,
                        pl.slice(resid1, [BATCH, K_CHUNK], [0, _r0]),
                        [0, _r0],
                    )
            if _DBG_STAGE == 5:
                # Diagnostic: dump resid_hold (the attention residual = MoE-block
                # attention output) to isolate whether the valid-token race is in
                # attention (this) vs the MoE body (stages 1-4).
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="dbg_dump_5"):
                    for _d5 in pl.range(HIDDEN // K_CHUNK):
                        _d50 = _d5 * K_CHUNK
                        dbg_out = pl.assemble(
                            dbg_out,
                            pl.slice(resid_hold, [BATCH, K_CHUNK], [0, _d50]),
                            [0, _d50],
                        )
            # ── B: post-attention zero-centred RMSNorm of resid1. ──────
            hidden_blocks = HIDDEN // K_CHUNK
            post_norm = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            resid1_fp32 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
            with pl.at(
                level=pl.Level.CORE_GROUP, name_hint="moe_post_rmsnorm_zc",
            ):
                for kb in pl.range(hidden_blocks):
                    k0 = kb * K_CHUNK
                    rchunk = pl.cast(
                        pl.slice(resid1, [BATCH, K_CHUNK], [0, k0]),
                        target_type=pl.FP32,
                    )
                    resid1_fp32 = pl.assemble(resid1_fp32, rchunk, [0, k0])

                sq_sum = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
                for kb2 in pl.range(hidden_blocks):
                    k0 = kb2 * K_CHUNK
                    ck = pl.slice(resid1_fp32, [BATCH, K_CHUNK], [0, k0])
                    sq_sum = pl.add(
                        sq_sum,
                        pl.reshape(
                            pl.row_sum(pl.mul(ck, ck)),
                            [1, BATCH],
                        ),
                    )
                inv_rms_moe = pl.recip(
                    pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)),
                )
                inv_rms_col = pl.reshape(inv_rms_moe, [BATCH, 1])
                for kb3 in pl.range(hidden_blocks):
                    k0 = kb3 * K_CHUNK
                    norm_chunk = pl.slice(
                        resid1_fp32, [BATCH, K_CHUNK], [0, k0],
                    )
                    gamma = pl.slice(
                        post_rms_weight, [1, K_CHUNK], [norm_layer_idx, k0],
                    )
                    scaled = pl.row_expand_mul(norm_chunk, inv_rms_col)
                    normed = pl.col_expand_mul(scaled, pl.add(gamma, 1.0))
                    post_norm = pl.assemble(
                        post_norm,
                        pl.cast(normed, target_type=pl.BF16),
                        [0, k0],
                    )

            if _DBG_STAGE == 1:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="dbg_dump_1"):
                    for _dg in pl.range(HIDDEN // K_CHUNK):
                        _dg0 = _dg * K_CHUNK
                        dbg_out = pl.assemble(dbg_out, pl.slice(post_norm, [BATCH, K_CHUNK], [0, _dg0]), [0, _dg0])
            # ── C: EP+TP MoE chip_orch -> moe_out (Phase X.8 inlined). ─
            # Body copied verbatim from ``moe.EpTpMoE.chip_orch``; calls the
            # inlined per-stage methods (``self.gate_step`` /
            # ``self.dispatch_step`` / ``self.expert_routed_step`` /
            # ``self.expert_shared_step`` / ``self.combine_step``) directly
            # rather than instantiating a separate ``@pl.program``.
            moe_out = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)

            # 1) Gate (local, replicated).
            expert_indices = pl.create_tensor([BATCH, TOPK], dtype=pl.INT32)
            expert_weights = pl.create_tensor([BATCH, TOPK], dtype=pl.FP32)
            expert_indices, expert_weights = self.gate_step(
                post_norm, gate_w, router_bias,
                expert_indices, expert_weights,
            )

            # 2) Shared-expert lane (TP-sliced + tp_all_reduce).
            sh_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            sh_y = self.expert_shared_step(
                post_norm, w_gate_s, w_up_s, w_down_s, sh_y,
                sh_tmp_window, sh_signal_window, my_rank,
            )


            # 1A: per-token INT8 dynamic-quant of the MoE input BEFORE
            # dispatch (dispatch-side; shrinks recv_x 8→4MB/layer).
            x_disp_i8 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.INT8)
            x_disp_scale = pl.create_tensor([BATCH, 8], dtype=pl.FP32)
            (x_disp_i8, x_disp_scale) = self._quant_moe_input(
                post_norm, x_disp_i8, x_disp_scale,
            )
            if _DBG_STAGE == 2:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="dbg_dump_2"):
                    for _dg in pl.range(HIDDEN // K_CHUNK):
                        _dg0 = _dg * K_CHUNK
                        dbg_out = pl.assemble(dbg_out, pl.slice(sh_y, [BATCH, K_CHUNK], [0, _dg0]), [0, _dg0])
            # 3) Dispatch (EP fixed-slot pull).
            local_routed_x = pl.create_tensor(
                [local_recv_max, HIDDEN], dtype=pl.INT8,
            )
            local_routed_x_scale = pl.create_tensor(
                [1, local_recv_max], dtype=pl.FP32,
            )
            local_expert_offset = pl.create_tensor(
                [n_local_experts], dtype=pl.INT32,
            )
            local_expert_count = pl.create_tensor(
                [n_local_experts], dtype=pl.INT32,
            )
            # Per-recv-row r_route (= source t*TOPK+k), staged from recv_r_route;
            # combine uses it to scatter the routed output back to the source.
            recv_r_route_out = pl.create_tensor(
                [local_recv_max], dtype=pl.INT32,
            )
            (
                local_routed_x,
                local_routed_x_scale,
                local_expert_offset,
                local_expert_count,
                recv_r_route_out,
                inverse_map,
            ) = self.dispatch_step(
                x_disp_i8, x_disp_scale, expert_indices,
                local_routed_x, local_routed_x_scale,
                local_expert_offset, local_expert_count, recv_r_route_out,
                pub_counts, count_done_sig, recv_x, recv_scale, recv_r_route, data_done_sig,
                send_x, send_scale, send_route,
                my_rank,
            )

            if _DBG_STAGE == 32:
                # Diagnostic-only: dump local_routed_x_scale[0, :1024] into
                # dbg_out row 0. Other rows/cols remain host-initialized zero.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="dbg_dump_32_local_routed_x_scale"):
                    _scale_dbg = pl.cast(
                        pl.slice(local_routed_x_scale, [1, local_recv_max], [0, 0]),
                        target_type=pl.BF16,
                    )
                    dbg_out = pl.assemble(dbg_out, _scale_dbg, [0, 0])
            if _DBG_STAGE == 33:
                # Diagnostic-only: dump recv_r_route_out[:1024] into dbg_out row 0
                # as BF16 values. It is route metadata, not model data.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="dbg_dump_33_recv_route"):
                    _route_dbg = pl.reshape(
                        pl.cast(
                            pl.cast(
                                pl.slice(recv_r_route_out, [local_recv_max], [0]),
                                target_type=pl.FP32,
                            ),
                            target_type=pl.BF16,
                        ),
                        [1, local_recv_max],
                    )
                    dbg_out = pl.assemble(dbg_out, _route_dbg, [0, 0])

            # 4) Routed experts (local 36).
            local_routed_y = pl.create_tensor(
                [local_recv_max, HIDDEN], dtype=pl.BF16,
            )
            local_routed_y = self.expert_routed_step(
                local_routed_x,
                local_routed_x_scale,
                local_expert_offset, local_expert_count,
                w_gate_r, w_gate_r_scale, w_up_r, w_up_r_scale,
                w_down_r, w_down_r_scale,
                local_routed_y,
            )

            if _DBG_STAGE == 3:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="dbg_dump_3"):
                    for _dg in pl.range(HIDDEN // K_CHUNK):
                        _dg0 = _dg * K_CHUNK
                        dbg_out = pl.assemble(dbg_out, pl.slice(local_routed_y, [BATCH, K_CHUNK], [0, _dg0]), [0, _dg0])
            # 5) Combine (EP pull back + weighted gather + sh_y add).
            moe_out = self.combine_step(
                local_routed_y,
                recv_r_route_out, expert_weights, sh_y,
                moe_out,
                pub_counts,
                routed_y_buf, combine_done_sig,
                expert_indices, inverse_map, routed_src_buf,
                my_rank,
            )

            if _DBG_STAGE == 4:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="dbg_dump_4"):
                    for _dg in pl.range(HIDDEN // K_CHUNK):
                        _dg0 = _dg * K_CHUNK
                        dbg_out = pl.assemble(dbg_out, pl.slice(moe_out, [BATCH, K_CHUNK], [0, _dg0]), [0, _dg0])
            # ── D: residual add: next_hidden_out = resid1 + moe_out. ───
            with pl.at(
                level=pl.Level.CORE_GROUP, name_hint="moe_residual_add",
            ):
                for kb4 in pl.range(hidden_blocks):
                    k0 = kb4 * K_CHUNK
                    m = pl.cast(
                        pl.slice(moe_out, [BATCH, K_CHUNK], [0, k0]),
                        target_type=pl.FP32,
                    )
                    r = pl.cast(pl.slice(resid_hold, [BATCH, K_CHUNK], [0, k0]), target_type=pl.FP32)
                    next_hidden_out = pl.assemble(
                        next_hidden_out,
                        pl.cast(pl.add(r, m), target_type=pl.BF16),
                        [0, k0],
                    )
            return next_hidden_out

        # ---- MoE-layer swa-attn FUSED with MoE-block: attention -> resid1
        # (local, intra-orch) -> post_norm -> EP/TP MoE -> residual. Mirrors the
        # dense swa_chip_orch attn->MLP handoff so the attn->MoE handoff is
        # an intra-Submission local tensor, not a cross-orch shared h_mid_out. ----
        @pl.function(
            type=pl.FunctionType.Orchestration,
            attrs={"inline_orchestration": True},
        )
        def swa_moe_chip_orch(  # noqa: PLR0913, PLR0915
            self,
            current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            wq: pl.Tensor[[HIDDEN, hidden_q_swa], pl.BF16],
            wk: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL], pl.BF16],
            wv: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL], pl.BF16],
            q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            rope_cos: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],
            rope_sin: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],
            k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            wo: pl.Tensor[[hidden_q_swa, HIDDEN], pl.BF16],
            w_g: pl.Tensor[[HIDDEN, nh_swa_pad], pl.BF16],
            gate_r: pl.Tensor[[nh_swa_pad, hidden_q_swa], pl.BF16],
            post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            gate_w: pl.Tensor[[HIDDEN, N_EXPERTS], pl.FP32],
            router_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
            w_gate_r: pl.Tensor[[n_local_experts, HIDDEN, inter], pl.INT8],
            w_gate_r_scale: pl.Tensor[[n_local_experts, inter], pl.FP32],
            w_up_r: pl.Tensor[[n_local_experts, HIDDEN, inter], pl.INT8],
            w_up_r_scale: pl.Tensor[[n_local_experts, inter], pl.FP32],
            w_down_r: pl.Tensor[[n_local_experts, inter, HIDDEN], pl.INT8],
            w_down_r_scale: pl.Tensor[[n_local_experts, HIDDEN], pl.FP32],
            w_gate_s: pl.Tensor[[HIDDEN, sh_inter_local], pl.BF16],
            w_up_s: pl.Tensor[[HIDDEN, sh_inter_local], pl.BF16],
            w_down_s: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
            next_hidden_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            dbg_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            resid_hold: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            attn_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
            attn_signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
            pub_counts: pld.DistributedTensor[
                [n_ranks * n_ranks, n_local_experts_pad], pl.INT32
            ],
            count_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
            recv_x: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
            recv_scale: pld.DistributedTensor[[local_recv_max, 8], pl.FP32],
            data_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
            recv_r_route: pld.DistributedTensor[[local_recv_max, idx_pad], pl.INT32],
            send_x: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
            send_scale: pld.DistributedTensor[[local_recv_max, 8], pl.FP32],
            send_route: pld.DistributedTensor[[local_recv_max, idx_pad], pl.INT32],
            sh_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
            sh_signal_window: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
            routed_y_buf: pld.DistributedTensor[[n_routes_per_rank, HIDDEN], pl.BF16],
            combine_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
            routed_src_buf: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.BF16],
            norm_layer_idx: pl.Scalar[pl.INT32],
            attn_layer_idx: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            resid1 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            resid1 = attention_swa_inline(
                current_hidden, input_rms_weight, wq, wk, wv,
                q_norm_weight, k_norm_weight,
                seq_lens, block_table, slot_mapping,
                rope_cos, rope_sin, k_cache, v_cache,
                wo, w_g, gate_r, resid1,
                norm_layer_idx, attn_layer_idx,
                attn_tmp_window, attn_signal_window, my_rank,
            )
            # Save the attention residual into a DEDICATED per-layer external
            # buffer resid_hold (write-once here, read-once by section D). The
            # local resid1 create_tensor is reused by gate/shared/routed InCore
            # scratch inside this big fused orch, so section D cannot read resid1
            # directly. Earlier code stashed into next_hidden_out, but that made
            # next_hidden_out have TWO writers (stash + residual_add) => a WAW that
            # RAW-only-v1 (single-value producer_index) cannot serialise across
            # submissions => nondeterministic output. resid_hold has ONE writer
            # (here) and ONE reader (section D); next_hidden_out has ONE writer
            # (section D). Both are clean single-producer RAW.
            with pl.at(
                level=pl.Level.CORE_GROUP, name_hint="stash_resid_hold",
            ):
                for _rs in pl.range(HIDDEN // K_CHUNK):
                    _r0 = _rs * K_CHUNK
                    resid_hold = pl.assemble(
                        resid_hold,
                        pl.slice(resid1, [BATCH, K_CHUNK], [0, _r0]),
                        [0, _r0],
                    )
            if _DBG_STAGE == 5:
                # Diagnostic: dump resid_hold (the attention residual = MoE-block
                # attention output) to isolate whether the valid-token race is in
                # attention (this) vs the MoE body (stages 1-4).
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="dbg_dump_5"):
                    for _d5 in pl.range(HIDDEN // K_CHUNK):
                        _d50 = _d5 * K_CHUNK
                        dbg_out = pl.assemble(
                            dbg_out,
                            pl.slice(resid_hold, [BATCH, K_CHUNK], [0, _d50]),
                            [0, _d50],
                        )
            # ── B: post-attention zero-centred RMSNorm of resid1. ──────
            hidden_blocks = HIDDEN // K_CHUNK
            post_norm = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            resid1_fp32 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
            with pl.at(
                level=pl.Level.CORE_GROUP, name_hint="moe_post_rmsnorm_zc",
            ):
                for kb in pl.range(hidden_blocks):
                    k0 = kb * K_CHUNK
                    rchunk = pl.cast(
                        pl.slice(resid1, [BATCH, K_CHUNK], [0, k0]),
                        target_type=pl.FP32,
                    )
                    resid1_fp32 = pl.assemble(resid1_fp32, rchunk, [0, k0])

                sq_sum = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
                for kb2 in pl.range(hidden_blocks):
                    k0 = kb2 * K_CHUNK
                    ck = pl.slice(resid1_fp32, [BATCH, K_CHUNK], [0, k0])
                    sq_sum = pl.add(
                        sq_sum,
                        pl.reshape(
                            pl.row_sum(pl.mul(ck, ck)),
                            [1, BATCH],
                        ),
                    )
                inv_rms_moe = pl.recip(
                    pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)),
                )
                inv_rms_col = pl.reshape(inv_rms_moe, [BATCH, 1])
                for kb3 in pl.range(hidden_blocks):
                    k0 = kb3 * K_CHUNK
                    norm_chunk = pl.slice(
                        resid1_fp32, [BATCH, K_CHUNK], [0, k0],
                    )
                    gamma = pl.slice(
                        post_rms_weight, [1, K_CHUNK], [norm_layer_idx, k0],
                    )
                    scaled = pl.row_expand_mul(norm_chunk, inv_rms_col)
                    normed = pl.col_expand_mul(scaled, pl.add(gamma, 1.0))
                    post_norm = pl.assemble(
                        post_norm,
                        pl.cast(normed, target_type=pl.BF16),
                        [0, k0],
                    )

            if _DBG_STAGE == 1:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="dbg_dump_1"):
                    for _dg in pl.range(HIDDEN // K_CHUNK):
                        _dg0 = _dg * K_CHUNK
                        dbg_out = pl.assemble(dbg_out, pl.slice(post_norm, [BATCH, K_CHUNK], [0, _dg0]), [0, _dg0])
            # ── C: EP+TP MoE chip_orch -> moe_out (Phase X.8 inlined). ─
            # Body copied verbatim from ``moe.EpTpMoE.chip_orch``; calls the
            # inlined per-stage methods (``self.gate_step`` /
            # ``self.dispatch_step`` / ``self.expert_routed_step`` /
            # ``self.expert_shared_step`` / ``self.combine_step``) directly
            # rather than instantiating a separate ``@pl.program``.
            moe_out = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)

            # 1) Gate (local, replicated).
            expert_indices = pl.create_tensor([BATCH, TOPK], dtype=pl.INT32)
            expert_weights = pl.create_tensor([BATCH, TOPK], dtype=pl.FP32)
            expert_indices, expert_weights = self.gate_step(
                post_norm, gate_w, router_bias,
                expert_indices, expert_weights,
            )

            # 2) Shared-expert lane (TP-sliced + tp_all_reduce).
            sh_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            sh_y = self.expert_shared_step(
                post_norm, w_gate_s, w_up_s, w_down_s, sh_y,
                sh_tmp_window, sh_signal_window, my_rank,
            )


            # 1A: per-token INT8 dynamic-quant of the MoE input BEFORE
            # dispatch (dispatch-side; shrinks recv_x 8→4MB/layer).
            x_disp_i8 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.INT8)
            x_disp_scale = pl.create_tensor([BATCH, 8], dtype=pl.FP32)
            (x_disp_i8, x_disp_scale) = self._quant_moe_input(
                post_norm, x_disp_i8, x_disp_scale,
            )
            if _DBG_STAGE == 2:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="dbg_dump_2"):
                    for _dg in pl.range(HIDDEN // K_CHUNK):
                        _dg0 = _dg * K_CHUNK
                        dbg_out = pl.assemble(dbg_out, pl.slice(sh_y, [BATCH, K_CHUNK], [0, _dg0]), [0, _dg0])
            # 3) Dispatch (EP fixed-slot pull).
            local_routed_x = pl.create_tensor(
                [local_recv_max, HIDDEN], dtype=pl.INT8,
            )
            local_routed_x_scale = pl.create_tensor(
                [1, local_recv_max], dtype=pl.FP32,
            )
            local_expert_offset = pl.create_tensor(
                [n_local_experts], dtype=pl.INT32,
            )
            local_expert_count = pl.create_tensor(
                [n_local_experts], dtype=pl.INT32,
            )
            # Per-recv-row r_route (= source t*TOPK+k), staged from recv_r_route;
            # combine uses it to scatter the routed output back to the source.
            recv_r_route_out = pl.create_tensor(
                [local_recv_max], dtype=pl.INT32,
            )
            (
                local_routed_x,
                local_routed_x_scale,
                local_expert_offset,
                local_expert_count,
                recv_r_route_out,
                inverse_map,
            ) = self.dispatch_step(
                x_disp_i8, x_disp_scale, expert_indices,
                local_routed_x, local_routed_x_scale,
                local_expert_offset, local_expert_count, recv_r_route_out,
                pub_counts, count_done_sig, recv_x, recv_scale, recv_r_route, data_done_sig,
                send_x, send_scale, send_route,
                my_rank,
            )

            if _DBG_STAGE == 32:
                # Diagnostic-only: dump local_routed_x_scale[0, :1024] into
                # dbg_out row 0. Other rows/cols remain host-initialized zero.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="dbg_dump_32_local_routed_x_scale"):
                    _scale_dbg = pl.cast(
                        pl.slice(local_routed_x_scale, [1, local_recv_max], [0, 0]),
                        target_type=pl.BF16,
                    )
                    dbg_out = pl.assemble(dbg_out, _scale_dbg, [0, 0])
            if _DBG_STAGE == 33:
                # Diagnostic-only: dump recv_r_route_out[:1024] into dbg_out row 0
                # as BF16 values. It is route metadata, not model data.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="dbg_dump_33_recv_route"):
                    _route_dbg = pl.reshape(
                        pl.cast(
                            pl.cast(
                                pl.slice(recv_r_route_out, [local_recv_max], [0]),
                                target_type=pl.FP32,
                            ),
                            target_type=pl.BF16,
                        ),
                        [1, local_recv_max],
                    )
                    dbg_out = pl.assemble(dbg_out, _route_dbg, [0, 0])

            # 4) Routed experts (local 36).
            local_routed_y = pl.create_tensor(
                [local_recv_max, HIDDEN], dtype=pl.BF16,
            )
            local_routed_y = self.expert_routed_step(
                local_routed_x,
                local_routed_x_scale,
                local_expert_offset, local_expert_count,
                w_gate_r, w_gate_r_scale, w_up_r, w_up_r_scale,
                w_down_r, w_down_r_scale,
                local_routed_y,
            )

            if _DBG_STAGE == 3:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="dbg_dump_3"):
                    for _dg in pl.range(HIDDEN // K_CHUNK):
                        _dg0 = _dg * K_CHUNK
                        dbg_out = pl.assemble(dbg_out, pl.slice(local_routed_y, [BATCH, K_CHUNK], [0, _dg0]), [0, _dg0])
            # 5) Combine (EP pull back + weighted gather + sh_y add).
            moe_out = self.combine_step(
                local_routed_y,
                recv_r_route_out, expert_weights, sh_y,
                moe_out,
                pub_counts,
                routed_y_buf, combine_done_sig,
                expert_indices, inverse_map, routed_src_buf,
                my_rank,
            )

            if _DBG_STAGE == 4:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="dbg_dump_4"):
                    for _dg in pl.range(HIDDEN // K_CHUNK):
                        _dg0 = _dg * K_CHUNK
                        dbg_out = pl.assemble(dbg_out, pl.slice(moe_out, [BATCH, K_CHUNK], [0, _dg0]), [0, _dg0])
            # ── D: residual add: next_hidden_out = resid1 + moe_out. ───
            with pl.at(
                level=pl.Level.CORE_GROUP, name_hint="moe_residual_add",
            ):
                for kb4 in pl.range(hidden_blocks):
                    k0 = kb4 * K_CHUNK
                    m = pl.cast(
                        pl.slice(moe_out, [BATCH, K_CHUNK], [0, k0]),
                        target_type=pl.FP32,
                    )
                    r = pl.cast(pl.slice(resid_hold, [BATCH, K_CHUNK], [0, k0]), target_type=pl.FP32)
                    next_hidden_out = pl.assemble(
                        next_hidden_out,
                        pl.cast(pl.add(r, m), target_type=pl.BF16),
                        [0, k0],
                    )
            return next_hidden_out
        # ---- host_orch: real per-layer weights, full+swa routing. ----
        @pl.function(type=pl.FunctionType.Orchestration)
        def whole_chip_orch(  # noqa: PLR0913, PLR0915
            self,
            current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            input_rms: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            post_rms: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            q_norm: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            full_wq: pl.Tensor[[12 * HIDDEN, hidden_q_full], pl.BF16],
            full_wk: pl.Tensor[[12 * HIDDEN, KV_HIDDEN_LOCAL], pl.BF16],
            full_wv: pl.Tensor[[12 * HIDDEN, KV_HIDDEN_LOCAL], pl.BF16],
            full_wo: pl.Tensor[[12 * hidden_q_full, HIDDEN], pl.BF16],
            full_w_g: pl.Tensor[[12 * HIDDEN, nh_full_pad], pl.BF16],
            full_gate_r: pl.Tensor[[12 * nh_full_pad, hidden_q_full], pl.BF16],
            swa_wq: pl.Tensor[[33 * HIDDEN, hidden_q_swa], pl.BF16],
            swa_wk: pl.Tensor[[33 * HIDDEN, KV_HIDDEN_LOCAL], pl.BF16],
            swa_wv: pl.Tensor[[33 * HIDDEN, KV_HIDDEN_LOCAL], pl.BF16],
            swa_wo: pl.Tensor[[33 * hidden_q_swa, HIDDEN], pl.BF16],
            swa_w_g: pl.Tensor[[33 * HIDDEN, nh_swa_pad], pl.BF16],
            swa_gate_r: pl.Tensor[[33 * nh_swa_pad, hidden_q_swa], pl.BF16],
            dense_w_gate: pl.Tensor[[3 * HIDDEN, INTER_LOCAL], pl.BF16],
            dense_w_up: pl.Tensor[[3 * HIDDEN, INTER_LOCAL], pl.BF16],
            dense_w_down: pl.Tensor[[3 * INTER_LOCAL, HIDDEN], pl.BF16],
            moe_gate_w: pl.Tensor[[42 * HIDDEN, N_EXPERTS], pl.FP32],
            moe_router_bias: pl.Tensor[[42 * N_EXPERTS], pl.FP32],
            moe_w_gate_r: pl.Tensor[[42 * n_local_experts * HIDDEN, inter], pl.INT8],
            moe_w_gate_r_scale: pl.Tensor[[42 * n_local_experts, inter], pl.FP32],
            moe_w_up_r: pl.Tensor[[42 * n_local_experts * HIDDEN, inter], pl.INT8],
            moe_w_up_r_scale: pl.Tensor[[42 * n_local_experts, inter], pl.FP32],
            moe_w_down_r: pl.Tensor[[42 * n_local_experts * inter, HIDDEN], pl.INT8],
            moe_w_down_r_scale: pl.Tensor[[42 * n_local_experts, HIDDEN], pl.FP32],
            moe_w_gate_s: pl.Tensor[[42 * HIDDEN, sh_inter_local], pl.BF16],
            moe_w_up_s: pl.Tensor[[42 * HIDDEN, sh_inter_local], pl.BF16],
            moe_w_down_s: pl.Tensor[[42 * sh_inter_local, HIDDEN], pl.BF16],
            seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            rope_cos_full: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],
            rope_sin_full: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],
            rope_cos_swa: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],
            rope_sin_swa: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],
            k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            h_mid_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            next_hidden_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            dbg_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
            lm_head_weight: pl.Tensor[[VOCAB_LOCAL, HIDDEN], pl.BF16],
            logits_shard_out: pl.Out[pl.Tensor[[USER_BATCH_DYN, VOCAB_LOCAL], pl.FP32]],
            dense_attn_tmp_stack: pld.DistributedTensor[
                [WHOLE_CHIP_DENSE_LAYERS * BATCH, HIDDEN], pl.BF16
            ],
            dense_attn_signal_stack: pld.DistributedTensor[
                [WHOLE_CHIP_DENSE_LAYERS * COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
            ],
            dense_mlp_tmp_stack: pld.DistributedTensor[
                [WHOLE_CHIP_DENSE_LAYERS * BATCH, HIDDEN], pl.BF16
            ],
            dense_mlp_signal_stack: pld.DistributedTensor[
                [WHOLE_CHIP_DENSE_LAYERS * COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
            ],
            moe_attn_tmp_stack: pld.DistributedTensor[
                [WHOLE_CHIP_MOE_LAYERS * BATCH, HIDDEN], pl.BF16
            ],
            moe_attn_signal_stack: pld.DistributedTensor[
                [WHOLE_CHIP_MOE_LAYERS * COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
            ],
            moe_pub_counts_stack: pld.DistributedTensor[
                [WHOLE_CHIP_MOE_LAYERS * n_ranks * n_ranks, n_local_experts_pad], pl.INT32
            ],
            moe_count_done_stack: pld.DistributedTensor[
                [WHOLE_CHIP_MOE_LAYERS * COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
            ],
            moe_recv_x_stack: pld.DistributedTensor[
                [WHOLE_CHIP_MOE_LAYERS * local_recv_max, HIDDEN], pl.INT8
            ],
            moe_recv_scale_stack: pld.DistributedTensor[
                [WHOLE_CHIP_MOE_LAYERS * local_recv_max, 8], pl.FP32
            ],
            moe_recv_route_stack: pld.DistributedTensor[
                [WHOLE_CHIP_MOE_LAYERS * local_recv_max, idx_pad], pl.INT32
            ],
            moe_data_done_stack: pld.DistributedTensor[
                [WHOLE_CHIP_MOE_LAYERS * COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
            ],
            moe_send_x_stack: pld.DistributedTensor[
                [WHOLE_CHIP_MOE_LAYERS * local_recv_max, HIDDEN], pl.INT8
            ],
            moe_send_scale_stack: pld.DistributedTensor[
                [WHOLE_CHIP_MOE_LAYERS * local_recv_max, 8], pl.FP32
            ],
            moe_send_route_stack: pld.DistributedTensor[
                [WHOLE_CHIP_MOE_LAYERS * local_recv_max, idx_pad], pl.INT32
            ],
            moe_shared_tmp_stack: pld.DistributedTensor[
                [WHOLE_CHIP_MOE_LAYERS * BATCH, HIDDEN], pl.BF16
            ],
            moe_shared_signal_stack: pld.DistributedTensor[
                [WHOLE_CHIP_MOE_LAYERS * COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
            ],
            moe_routed_y_stack: pld.DistributedTensor[
                [WHOLE_CHIP_MOE_LAYERS * n_routes_per_rank, HIDDEN], pl.BF16
            ],
            moe_combine_done_stack: pld.DistributedTensor[
                [WHOLE_CHIP_MOE_LAYERS * COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
            ],
            moe_routed_src_stack: pld.DistributedTensor[
                [WHOLE_CHIP_MOE_LAYERS * local_recv_max, HIDDEN], pl.BF16
            ],
            my_rank: pl.Scalar[pl.INT32],
        ):
            # One rank-local task graph. Every communication argument below
            # is a distinct window allocation; signal allocations are 512 B
            # so AtomicAdd/TWAIT traffic cannot alias another layer.
            h_layer_0 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_0 = self.full_chip_orch(
                current_hidden,
                input_rms,
                pl.slice(full_wq, [HIDDEN, hidden_q_full], [0 * (HIDDEN), 0]),
                pl.slice(full_wk, [HIDDEN, KV_HIDDEN_LOCAL], [0 * (HIDDEN), 0]),
                pl.slice(full_wv, [HIDDEN, KV_HIDDEN_LOCAL], [0 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_full,
                rope_sin_full,
                k_cache,
                v_cache,
                pl.slice(full_wo, [hidden_q_full, HIDDEN], [0 * (hidden_q_full), 0]),
                pl.slice(full_w_g, [HIDDEN, nh_full_pad], [0 * (HIDDEN), 0]),
                pl.slice(full_gate_r, [nh_full_pad, hidden_q_full], [0 * (nh_full_pad), 0]),
                post_rms,
                pl.slice(dense_w_gate, [HIDDEN, INTER_LOCAL], [0 * (HIDDEN), 0]),
                pl.slice(dense_w_up, [HIDDEN, INTER_LOCAL], [0 * (HIDDEN), 0]),
                pl.slice(dense_w_down, [INTER_LOCAL, HIDDEN], [0 * (INTER_LOCAL), 0]),
                h_layer_0,
                pl.slice(dense_attn_tmp_stack, [BATCH, HIDDEN], [0 * BATCH, 0]),
                pl.slice(dense_attn_signal_stack, [tp_size, 1], [0 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(dense_mlp_tmp_stack, [BATCH, HIDDEN], [0 * BATCH, 0]),
                pl.slice(dense_mlp_signal_stack, [tp_size, 1], [0 * COMM_SIGNAL_STRIDE_I32, 0]),
                0,
                0,
                0,
                my_rank,
            )
            h_mid_out = self.swa_chip_orch(
                h_layer_0,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [0 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [0 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [0 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [0 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [0 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [0 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(dense_w_gate, [HIDDEN, INTER_LOCAL], [1 * (HIDDEN), 0]),
                pl.slice(dense_w_up, [HIDDEN, INTER_LOCAL], [1 * (HIDDEN), 0]),
                pl.slice(dense_w_down, [INTER_LOCAL, HIDDEN], [1 * (INTER_LOCAL), 0]),
                h_mid_out,
                pl.slice(dense_attn_tmp_stack, [BATCH, HIDDEN], [1 * BATCH, 0]),
                pl.slice(dense_attn_signal_stack, [tp_size, 1], [1 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(dense_mlp_tmp_stack, [BATCH, HIDDEN], [1 * BATCH, 0]),
                pl.slice(dense_mlp_signal_stack, [tp_size, 1], [1 * COMM_SIGNAL_STRIDE_I32, 0]),
                1,
                0,
                0,
                my_rank,
            )
            h_layer_2 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_2 = self.swa_chip_orch(
                h_mid_out,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [1 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [1 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [1 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [1 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [1 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [1 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(dense_w_gate, [HIDDEN, INTER_LOCAL], [2 * (HIDDEN), 0]),
                pl.slice(dense_w_up, [HIDDEN, INTER_LOCAL], [2 * (HIDDEN), 0]),
                pl.slice(dense_w_down, [INTER_LOCAL, HIDDEN], [2 * (INTER_LOCAL), 0]),
                h_layer_2,
                pl.slice(dense_attn_tmp_stack, [BATCH, HIDDEN], [2 * BATCH, 0]),
                pl.slice(dense_attn_signal_stack, [tp_size, 1], [2 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(dense_mlp_tmp_stack, [BATCH, HIDDEN], [2 * BATCH, 0]),
                pl.slice(dense_mlp_signal_stack, [tp_size, 1], [2 * COMM_SIGNAL_STRIDE_I32, 0]),
                2,
                0,
                0,
                my_rank,
            )
            h_layer_3 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_3 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_3 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_3 = self.swa_moe_chip_orch(
                h_layer_2,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [2 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [2 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [2 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [2 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [2 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [2 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [0 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [0 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [0 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [0 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [0 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [0 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [0 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [0 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [0 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [0 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [0 * (sh_inter_local), 0]),
                h_layer_3,
                dbg_layer_3,
                resid_hold_layer_3,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [0 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [0 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [0 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [0 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [0 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [0 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [0 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [0 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [0 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [0 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [0 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [0 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [0 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [0 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [0 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [0 * local_recv_max, 0]),
                3,
                0,
                my_rank,
            )
            h_layer_4 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_4 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_4 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_4 = self.full_moe_chip_orch(
                h_layer_3,
                input_rms,
                pl.slice(full_wq, [HIDDEN, hidden_q_full], [1 * (HIDDEN), 0]),
                pl.slice(full_wk, [HIDDEN, KV_HIDDEN_LOCAL], [1 * (HIDDEN), 0]),
                pl.slice(full_wv, [HIDDEN, KV_HIDDEN_LOCAL], [1 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_full,
                rope_sin_full,
                k_cache,
                v_cache,
                pl.slice(full_wo, [hidden_q_full, HIDDEN], [1 * (hidden_q_full), 0]),
                pl.slice(full_w_g, [HIDDEN, nh_full_pad], [1 * (HIDDEN), 0]),
                pl.slice(full_gate_r, [nh_full_pad, hidden_q_full], [1 * (nh_full_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [1 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [1 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [1 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [1 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [1 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [1 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [1 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [1 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [1 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [1 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [1 * (sh_inter_local), 0]),
                h_layer_4,
                dbg_layer_4,
                resid_hold_layer_4,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [1 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [1 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [1 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [1 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [1 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [1 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [1 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [1 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [1 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [1 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [1 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [1 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [1 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [1 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [1 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [1 * local_recv_max, 0]),
                4,
                0,
                my_rank,
            )
            h_layer_5 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_5 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_5 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_5 = self.swa_moe_chip_orch(
                h_layer_4,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [3 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [3 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [3 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [3 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [3 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [3 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [2 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [2 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [2 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [2 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [2 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [2 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [2 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [2 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [2 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [2 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [2 * (sh_inter_local), 0]),
                h_layer_5,
                dbg_layer_5,
                resid_hold_layer_5,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [2 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [2 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [2 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [2 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [2 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [2 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [2 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [2 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [2 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [2 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [2 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [2 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [2 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [2 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [2 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [2 * local_recv_max, 0]),
                5,
                0,
                my_rank,
            )
            h_layer_6 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_6 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_6 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_6 = self.swa_moe_chip_orch(
                h_layer_5,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [4 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [4 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [4 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [4 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [4 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [4 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [3 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [3 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [3 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [3 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [3 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [3 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [3 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [3 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [3 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [3 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [3 * (sh_inter_local), 0]),
                h_layer_6,
                dbg_layer_6,
                resid_hold_layer_6,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [3 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [3 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [3 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [3 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [3 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [3 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [3 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [3 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [3 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [3 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [3 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [3 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [3 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [3 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [3 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [3 * local_recv_max, 0]),
                6,
                0,
                my_rank,
            )
            h_layer_7 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_7 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_7 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_7 = self.swa_moe_chip_orch(
                h_layer_6,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [5 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [5 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [5 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [5 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [5 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [5 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [4 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [4 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [4 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [4 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [4 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [4 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [4 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [4 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [4 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [4 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [4 * (sh_inter_local), 0]),
                h_layer_7,
                dbg_layer_7,
                resid_hold_layer_7,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [4 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [4 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [4 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [4 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [4 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [4 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [4 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [4 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [4 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [4 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [4 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [4 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [4 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [4 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [4 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [4 * local_recv_max, 0]),
                7,
                0,
                my_rank,
            )
            h_layer_8 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_8 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_8 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_8 = self.full_moe_chip_orch(
                h_layer_7,
                input_rms,
                pl.slice(full_wq, [HIDDEN, hidden_q_full], [2 * (HIDDEN), 0]),
                pl.slice(full_wk, [HIDDEN, KV_HIDDEN_LOCAL], [2 * (HIDDEN), 0]),
                pl.slice(full_wv, [HIDDEN, KV_HIDDEN_LOCAL], [2 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_full,
                rope_sin_full,
                k_cache,
                v_cache,
                pl.slice(full_wo, [hidden_q_full, HIDDEN], [2 * (hidden_q_full), 0]),
                pl.slice(full_w_g, [HIDDEN, nh_full_pad], [2 * (HIDDEN), 0]),
                pl.slice(full_gate_r, [nh_full_pad, hidden_q_full], [2 * (nh_full_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [5 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [5 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [5 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [5 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [5 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [5 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [5 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [5 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [5 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [5 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [5 * (sh_inter_local), 0]),
                h_layer_8,
                dbg_layer_8,
                resid_hold_layer_8,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [5 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [5 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [5 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [5 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [5 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [5 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [5 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [5 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [5 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [5 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [5 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [5 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [5 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [5 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [5 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [5 * local_recv_max, 0]),
                8,
                0,
                my_rank,
            )
            h_layer_9 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_9 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_9 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_9 = self.swa_moe_chip_orch(
                h_layer_8,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [6 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [6 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [6 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [6 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [6 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [6 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [6 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [6 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [6 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [6 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [6 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [6 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [6 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [6 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [6 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [6 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [6 * (sh_inter_local), 0]),
                h_layer_9,
                dbg_layer_9,
                resid_hold_layer_9,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [6 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [6 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [6 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [6 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [6 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [6 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [6 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [6 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [6 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [6 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [6 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [6 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [6 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [6 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [6 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [6 * local_recv_max, 0]),
                9,
                0,
                my_rank,
            )
            h_layer_10 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_10 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_10 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_10 = self.swa_moe_chip_orch(
                h_layer_9,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [7 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [7 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [7 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [7 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [7 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [7 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [7 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [7 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [7 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [7 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [7 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [7 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [7 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [7 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [7 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [7 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [7 * (sh_inter_local), 0]),
                h_layer_10,
                dbg_layer_10,
                resid_hold_layer_10,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [7 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [7 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [7 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [7 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [7 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [7 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [7 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [7 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [7 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [7 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [7 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [7 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [7 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [7 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [7 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [7 * local_recv_max, 0]),
                10,
                0,
                my_rank,
            )
            h_layer_11 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_11 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_11 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_11 = self.swa_moe_chip_orch(
                h_layer_10,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [8 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [8 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [8 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [8 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [8 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [8 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [8 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [8 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [8 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [8 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [8 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [8 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [8 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [8 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [8 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [8 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [8 * (sh_inter_local), 0]),
                h_layer_11,
                dbg_layer_11,
                resid_hold_layer_11,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [8 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [8 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [8 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [8 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [8 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [8 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [8 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [8 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [8 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [8 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [8 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [8 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [8 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [8 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [8 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [8 * local_recv_max, 0]),
                11,
                0,
                my_rank,
            )
            h_layer_12 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_12 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_12 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_12 = self.full_moe_chip_orch(
                h_layer_11,
                input_rms,
                pl.slice(full_wq, [HIDDEN, hidden_q_full], [3 * (HIDDEN), 0]),
                pl.slice(full_wk, [HIDDEN, KV_HIDDEN_LOCAL], [3 * (HIDDEN), 0]),
                pl.slice(full_wv, [HIDDEN, KV_HIDDEN_LOCAL], [3 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_full,
                rope_sin_full,
                k_cache,
                v_cache,
                pl.slice(full_wo, [hidden_q_full, HIDDEN], [3 * (hidden_q_full), 0]),
                pl.slice(full_w_g, [HIDDEN, nh_full_pad], [3 * (HIDDEN), 0]),
                pl.slice(full_gate_r, [nh_full_pad, hidden_q_full], [3 * (nh_full_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [9 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [9 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [9 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [9 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [9 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [9 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [9 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [9 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [9 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [9 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [9 * (sh_inter_local), 0]),
                h_layer_12,
                dbg_layer_12,
                resid_hold_layer_12,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [9 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [9 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [9 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [9 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [9 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [9 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [9 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [9 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [9 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [9 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [9 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [9 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [9 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [9 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [9 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [9 * local_recv_max, 0]),
                12,
                0,
                my_rank,
            )
            h_layer_13 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_13 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_13 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_13 = self.swa_moe_chip_orch(
                h_layer_12,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [9 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [9 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [9 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [9 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [9 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [9 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [10 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [10 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [10 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [10 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [10 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [10 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [10 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [10 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [10 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [10 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [10 * (sh_inter_local), 0]),
                h_layer_13,
                dbg_layer_13,
                resid_hold_layer_13,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [10 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [10 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [10 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [10 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [10 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [10 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [10 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [10 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [10 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [10 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [10 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [10 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [10 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [10 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [10 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [10 * local_recv_max, 0]),
                13,
                0,
                my_rank,
            )
            h_layer_14 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_14 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_14 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_14 = self.swa_moe_chip_orch(
                h_layer_13,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [10 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [10 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [10 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [10 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [10 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [10 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [11 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [11 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [11 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [11 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [11 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [11 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [11 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [11 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [11 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [11 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [11 * (sh_inter_local), 0]),
                h_layer_14,
                dbg_layer_14,
                resid_hold_layer_14,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [11 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [11 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [11 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [11 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [11 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [11 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [11 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [11 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [11 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [11 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [11 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [11 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [11 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [11 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [11 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [11 * local_recv_max, 0]),
                14,
                0,
                my_rank,
            )
            h_layer_15 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_15 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_15 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_15 = self.swa_moe_chip_orch(
                h_layer_14,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [11 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [11 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [11 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [11 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [11 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [11 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [12 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [12 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [12 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [12 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [12 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [12 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [12 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [12 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [12 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [12 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [12 * (sh_inter_local), 0]),
                h_layer_15,
                dbg_layer_15,
                resid_hold_layer_15,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [12 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [12 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [12 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [12 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [12 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [12 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [12 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [12 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [12 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [12 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [12 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [12 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [12 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [12 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [12 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [12 * local_recv_max, 0]),
                15,
                0,
                my_rank,
            )
            h_layer_16 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_16 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_16 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_16 = self.full_moe_chip_orch(
                h_layer_15,
                input_rms,
                pl.slice(full_wq, [HIDDEN, hidden_q_full], [4 * (HIDDEN), 0]),
                pl.slice(full_wk, [HIDDEN, KV_HIDDEN_LOCAL], [4 * (HIDDEN), 0]),
                pl.slice(full_wv, [HIDDEN, KV_HIDDEN_LOCAL], [4 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_full,
                rope_sin_full,
                k_cache,
                v_cache,
                pl.slice(full_wo, [hidden_q_full, HIDDEN], [4 * (hidden_q_full), 0]),
                pl.slice(full_w_g, [HIDDEN, nh_full_pad], [4 * (HIDDEN), 0]),
                pl.slice(full_gate_r, [nh_full_pad, hidden_q_full], [4 * (nh_full_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [13 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [13 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [13 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [13 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [13 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [13 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [13 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [13 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [13 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [13 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [13 * (sh_inter_local), 0]),
                h_layer_16,
                dbg_layer_16,
                resid_hold_layer_16,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [13 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [13 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [13 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [13 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [13 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [13 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [13 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [13 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [13 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [13 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [13 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [13 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [13 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [13 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [13 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [13 * local_recv_max, 0]),
                16,
                0,
                my_rank,
            )
            h_layer_17 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_17 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_17 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_17 = self.swa_moe_chip_orch(
                h_layer_16,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [12 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [12 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [12 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [12 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [12 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [12 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [14 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [14 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [14 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [14 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [14 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [14 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [14 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [14 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [14 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [14 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [14 * (sh_inter_local), 0]),
                h_layer_17,
                dbg_layer_17,
                resid_hold_layer_17,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [14 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [14 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [14 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [14 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [14 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [14 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [14 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [14 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [14 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [14 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [14 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [14 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [14 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [14 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [14 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [14 * local_recv_max, 0]),
                17,
                0,
                my_rank,
            )
            h_layer_18 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_18 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_18 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_18 = self.swa_moe_chip_orch(
                h_layer_17,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [13 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [13 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [13 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [13 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [13 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [13 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [15 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [15 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [15 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [15 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [15 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [15 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [15 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [15 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [15 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [15 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [15 * (sh_inter_local), 0]),
                h_layer_18,
                dbg_layer_18,
                resid_hold_layer_18,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [15 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [15 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [15 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [15 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [15 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [15 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [15 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [15 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [15 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [15 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [15 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [15 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [15 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [15 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [15 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [15 * local_recv_max, 0]),
                18,
                0,
                my_rank,
            )
            h_layer_19 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_19 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_19 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_19 = self.swa_moe_chip_orch(
                h_layer_18,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [14 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [14 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [14 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [14 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [14 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [14 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [16 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [16 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [16 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [16 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [16 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [16 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [16 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [16 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [16 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [16 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [16 * (sh_inter_local), 0]),
                h_layer_19,
                dbg_layer_19,
                resid_hold_layer_19,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [16 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [16 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [16 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [16 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [16 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [16 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [16 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [16 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [16 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [16 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [16 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [16 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [16 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [16 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [16 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [16 * local_recv_max, 0]),
                19,
                0,
                my_rank,
            )
            h_layer_20 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_20 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_20 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_20 = self.full_moe_chip_orch(
                h_layer_19,
                input_rms,
                pl.slice(full_wq, [HIDDEN, hidden_q_full], [5 * (HIDDEN), 0]),
                pl.slice(full_wk, [HIDDEN, KV_HIDDEN_LOCAL], [5 * (HIDDEN), 0]),
                pl.slice(full_wv, [HIDDEN, KV_HIDDEN_LOCAL], [5 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_full,
                rope_sin_full,
                k_cache,
                v_cache,
                pl.slice(full_wo, [hidden_q_full, HIDDEN], [5 * (hidden_q_full), 0]),
                pl.slice(full_w_g, [HIDDEN, nh_full_pad], [5 * (HIDDEN), 0]),
                pl.slice(full_gate_r, [nh_full_pad, hidden_q_full], [5 * (nh_full_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [17 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [17 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [17 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [17 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [17 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [17 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [17 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [17 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [17 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [17 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [17 * (sh_inter_local), 0]),
                h_layer_20,
                dbg_layer_20,
                resid_hold_layer_20,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [17 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [17 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [17 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [17 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [17 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [17 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [17 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [17 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [17 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [17 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [17 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [17 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [17 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [17 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [17 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [17 * local_recv_max, 0]),
                20,
                0,
                my_rank,
            )
            h_layer_21 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_21 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_21 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_21 = self.swa_moe_chip_orch(
                h_layer_20,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [15 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [15 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [15 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [15 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [15 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [15 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [18 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [18 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [18 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [18 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [18 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [18 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [18 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [18 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [18 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [18 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [18 * (sh_inter_local), 0]),
                h_layer_21,
                dbg_layer_21,
                resid_hold_layer_21,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [18 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [18 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [18 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [18 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [18 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [18 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [18 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [18 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [18 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [18 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [18 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [18 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [18 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [18 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [18 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [18 * local_recv_max, 0]),
                21,
                0,
                my_rank,
            )
            h_layer_22 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_22 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_22 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_22 = self.swa_moe_chip_orch(
                h_layer_21,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [16 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [16 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [16 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [16 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [16 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [16 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [19 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [19 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [19 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [19 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [19 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [19 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [19 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [19 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [19 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [19 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [19 * (sh_inter_local), 0]),
                h_layer_22,
                dbg_layer_22,
                resid_hold_layer_22,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [19 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [19 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [19 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [19 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [19 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [19 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [19 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [19 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [19 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [19 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [19 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [19 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [19 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [19 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [19 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [19 * local_recv_max, 0]),
                22,
                0,
                my_rank,
            )
            h_layer_23 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_23 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_23 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_23 = self.swa_moe_chip_orch(
                h_layer_22,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [17 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [17 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [17 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [17 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [17 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [17 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [20 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [20 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [20 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [20 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [20 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [20 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [20 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [20 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [20 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [20 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [20 * (sh_inter_local), 0]),
                h_layer_23,
                dbg_layer_23,
                resid_hold_layer_23,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [20 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [20 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [20 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [20 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [20 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [20 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [20 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [20 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [20 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [20 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [20 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [20 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [20 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [20 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [20 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [20 * local_recv_max, 0]),
                23,
                0,
                my_rank,
            )
            h_layer_24 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_24 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_24 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_24 = self.full_moe_chip_orch(
                h_layer_23,
                input_rms,
                pl.slice(full_wq, [HIDDEN, hidden_q_full], [6 * (HIDDEN), 0]),
                pl.slice(full_wk, [HIDDEN, KV_HIDDEN_LOCAL], [6 * (HIDDEN), 0]),
                pl.slice(full_wv, [HIDDEN, KV_HIDDEN_LOCAL], [6 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_full,
                rope_sin_full,
                k_cache,
                v_cache,
                pl.slice(full_wo, [hidden_q_full, HIDDEN], [6 * (hidden_q_full), 0]),
                pl.slice(full_w_g, [HIDDEN, nh_full_pad], [6 * (HIDDEN), 0]),
                pl.slice(full_gate_r, [nh_full_pad, hidden_q_full], [6 * (nh_full_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [21 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [21 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [21 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [21 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [21 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [21 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [21 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [21 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [21 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [21 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [21 * (sh_inter_local), 0]),
                h_layer_24,
                dbg_layer_24,
                resid_hold_layer_24,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [21 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [21 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [21 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [21 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [21 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [21 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [21 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [21 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [21 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [21 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [21 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [21 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [21 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [21 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [21 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [21 * local_recv_max, 0]),
                24,
                0,
                my_rank,
            )
            h_layer_25 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_25 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_25 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_25 = self.swa_moe_chip_orch(
                h_layer_24,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [18 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [18 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [18 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [18 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [18 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [18 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [22 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [22 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [22 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [22 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [22 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [22 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [22 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [22 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [22 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [22 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [22 * (sh_inter_local), 0]),
                h_layer_25,
                dbg_layer_25,
                resid_hold_layer_25,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [22 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [22 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [22 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [22 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [22 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [22 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [22 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [22 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [22 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [22 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [22 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [22 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [22 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [22 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [22 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [22 * local_recv_max, 0]),
                25,
                0,
                my_rank,
            )
            h_layer_26 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_26 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_26 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_26 = self.swa_moe_chip_orch(
                h_layer_25,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [19 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [19 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [19 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [19 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [19 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [19 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [23 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [23 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [23 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [23 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [23 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [23 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [23 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [23 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [23 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [23 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [23 * (sh_inter_local), 0]),
                h_layer_26,
                dbg_layer_26,
                resid_hold_layer_26,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [23 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [23 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [23 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [23 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [23 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [23 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [23 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [23 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [23 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [23 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [23 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [23 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [23 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [23 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [23 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [23 * local_recv_max, 0]),
                26,
                0,
                my_rank,
            )
            h_layer_27 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_27 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_27 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_27 = self.swa_moe_chip_orch(
                h_layer_26,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [20 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [20 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [20 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [20 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [20 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [20 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [24 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [24 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [24 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [24 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [24 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [24 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [24 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [24 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [24 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [24 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [24 * (sh_inter_local), 0]),
                h_layer_27,
                dbg_layer_27,
                resid_hold_layer_27,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [24 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [24 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [24 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [24 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [24 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [24 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [24 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [24 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [24 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [24 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [24 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [24 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [24 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [24 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [24 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [24 * local_recv_max, 0]),
                27,
                0,
                my_rank,
            )
            h_layer_28 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_28 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_28 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_28 = self.full_moe_chip_orch(
                h_layer_27,
                input_rms,
                pl.slice(full_wq, [HIDDEN, hidden_q_full], [7 * (HIDDEN), 0]),
                pl.slice(full_wk, [HIDDEN, KV_HIDDEN_LOCAL], [7 * (HIDDEN), 0]),
                pl.slice(full_wv, [HIDDEN, KV_HIDDEN_LOCAL], [7 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_full,
                rope_sin_full,
                k_cache,
                v_cache,
                pl.slice(full_wo, [hidden_q_full, HIDDEN], [7 * (hidden_q_full), 0]),
                pl.slice(full_w_g, [HIDDEN, nh_full_pad], [7 * (HIDDEN), 0]),
                pl.slice(full_gate_r, [nh_full_pad, hidden_q_full], [7 * (nh_full_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [25 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [25 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [25 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [25 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [25 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [25 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [25 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [25 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [25 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [25 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [25 * (sh_inter_local), 0]),
                h_layer_28,
                dbg_layer_28,
                resid_hold_layer_28,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [25 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [25 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [25 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [25 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [25 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [25 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [25 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [25 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [25 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [25 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [25 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [25 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [25 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [25 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [25 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [25 * local_recv_max, 0]),
                28,
                0,
                my_rank,
            )
            h_layer_29 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_29 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_29 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_29 = self.swa_moe_chip_orch(
                h_layer_28,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [21 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [21 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [21 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [21 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [21 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [21 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [26 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [26 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [26 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [26 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [26 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [26 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [26 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [26 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [26 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [26 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [26 * (sh_inter_local), 0]),
                h_layer_29,
                dbg_layer_29,
                resid_hold_layer_29,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [26 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [26 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [26 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [26 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [26 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [26 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [26 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [26 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [26 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [26 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [26 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [26 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [26 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [26 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [26 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [26 * local_recv_max, 0]),
                29,
                0,
                my_rank,
            )
            h_layer_30 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_30 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_30 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_30 = self.swa_moe_chip_orch(
                h_layer_29,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [22 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [22 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [22 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [22 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [22 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [22 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [27 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [27 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [27 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [27 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [27 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [27 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [27 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [27 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [27 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [27 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [27 * (sh_inter_local), 0]),
                h_layer_30,
                dbg_layer_30,
                resid_hold_layer_30,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [27 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [27 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [27 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [27 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [27 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [27 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [27 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [27 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [27 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [27 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [27 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [27 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [27 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [27 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [27 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [27 * local_recv_max, 0]),
                30,
                0,
                my_rank,
            )
            h_layer_31 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_31 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_31 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_31 = self.swa_moe_chip_orch(
                h_layer_30,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [23 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [23 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [23 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [23 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [23 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [23 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [28 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [28 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [28 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [28 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [28 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [28 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [28 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [28 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [28 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [28 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [28 * (sh_inter_local), 0]),
                h_layer_31,
                dbg_layer_31,
                resid_hold_layer_31,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [28 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [28 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [28 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [28 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [28 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [28 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [28 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [28 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [28 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [28 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [28 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [28 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [28 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [28 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [28 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [28 * local_recv_max, 0]),
                31,
                0,
                my_rank,
            )
            h_layer_32 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_32 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_32 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_32 = self.full_moe_chip_orch(
                h_layer_31,
                input_rms,
                pl.slice(full_wq, [HIDDEN, hidden_q_full], [8 * (HIDDEN), 0]),
                pl.slice(full_wk, [HIDDEN, KV_HIDDEN_LOCAL], [8 * (HIDDEN), 0]),
                pl.slice(full_wv, [HIDDEN, KV_HIDDEN_LOCAL], [8 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_full,
                rope_sin_full,
                k_cache,
                v_cache,
                pl.slice(full_wo, [hidden_q_full, HIDDEN], [8 * (hidden_q_full), 0]),
                pl.slice(full_w_g, [HIDDEN, nh_full_pad], [8 * (HIDDEN), 0]),
                pl.slice(full_gate_r, [nh_full_pad, hidden_q_full], [8 * (nh_full_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [29 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [29 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [29 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [29 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [29 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [29 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [29 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [29 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [29 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [29 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [29 * (sh_inter_local), 0]),
                h_layer_32,
                dbg_layer_32,
                resid_hold_layer_32,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [29 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [29 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [29 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [29 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [29 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [29 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [29 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [29 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [29 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [29 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [29 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [29 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [29 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [29 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [29 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [29 * local_recv_max, 0]),
                32,
                0,
                my_rank,
            )
            h_layer_33 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_33 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_33 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_33 = self.swa_moe_chip_orch(
                h_layer_32,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [24 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [24 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [24 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [24 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [24 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [24 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [30 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [30 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [30 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [30 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [30 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [30 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [30 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [30 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [30 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [30 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [30 * (sh_inter_local), 0]),
                h_layer_33,
                dbg_layer_33,
                resid_hold_layer_33,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [30 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [30 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [30 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [30 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [30 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [30 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [30 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [30 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [30 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [30 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [30 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [30 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [30 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [30 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [30 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [30 * local_recv_max, 0]),
                33,
                0,
                my_rank,
            )
            h_layer_34 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_34 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_34 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_34 = self.swa_moe_chip_orch(
                h_layer_33,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [25 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [25 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [25 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [25 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [25 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [25 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [31 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [31 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [31 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [31 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [31 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [31 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [31 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [31 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [31 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [31 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [31 * (sh_inter_local), 0]),
                h_layer_34,
                dbg_layer_34,
                resid_hold_layer_34,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [31 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [31 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [31 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [31 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [31 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [31 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [31 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [31 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [31 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [31 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [31 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [31 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [31 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [31 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [31 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [31 * local_recv_max, 0]),
                34,
                0,
                my_rank,
            )
            h_layer_35 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_35 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_35 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_35 = self.swa_moe_chip_orch(
                h_layer_34,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [26 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [26 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [26 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [26 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [26 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [26 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [32 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [32 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [32 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [32 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [32 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [32 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [32 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [32 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [32 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [32 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [32 * (sh_inter_local), 0]),
                h_layer_35,
                dbg_layer_35,
                resid_hold_layer_35,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [32 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [32 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [32 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [32 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [32 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [32 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [32 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [32 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [32 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [32 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [32 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [32 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [32 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [32 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [32 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [32 * local_recv_max, 0]),
                35,
                0,
                my_rank,
            )
            h_layer_36 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_36 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_36 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_36 = self.full_moe_chip_orch(
                h_layer_35,
                input_rms,
                pl.slice(full_wq, [HIDDEN, hidden_q_full], [9 * (HIDDEN), 0]),
                pl.slice(full_wk, [HIDDEN, KV_HIDDEN_LOCAL], [9 * (HIDDEN), 0]),
                pl.slice(full_wv, [HIDDEN, KV_HIDDEN_LOCAL], [9 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_full,
                rope_sin_full,
                k_cache,
                v_cache,
                pl.slice(full_wo, [hidden_q_full, HIDDEN], [9 * (hidden_q_full), 0]),
                pl.slice(full_w_g, [HIDDEN, nh_full_pad], [9 * (HIDDEN), 0]),
                pl.slice(full_gate_r, [nh_full_pad, hidden_q_full], [9 * (nh_full_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [33 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [33 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [33 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [33 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [33 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [33 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [33 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [33 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [33 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [33 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [33 * (sh_inter_local), 0]),
                h_layer_36,
                dbg_layer_36,
                resid_hold_layer_36,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [33 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [33 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [33 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [33 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [33 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [33 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [33 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [33 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [33 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [33 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [33 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [33 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [33 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [33 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [33 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [33 * local_recv_max, 0]),
                36,
                0,
                my_rank,
            )
            h_layer_37 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_37 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_37 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_37 = self.swa_moe_chip_orch(
                h_layer_36,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [27 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [27 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [27 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [27 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [27 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [27 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [34 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [34 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [34 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [34 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [34 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [34 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [34 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [34 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [34 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [34 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [34 * (sh_inter_local), 0]),
                h_layer_37,
                dbg_layer_37,
                resid_hold_layer_37,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [34 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [34 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [34 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [34 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [34 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [34 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [34 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [34 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [34 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [34 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [34 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [34 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [34 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [34 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [34 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [34 * local_recv_max, 0]),
                37,
                0,
                my_rank,
            )
            h_layer_38 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_38 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_38 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_38 = self.swa_moe_chip_orch(
                h_layer_37,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [28 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [28 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [28 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [28 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [28 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [28 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [35 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [35 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [35 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [35 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [35 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [35 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [35 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [35 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [35 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [35 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [35 * (sh_inter_local), 0]),
                h_layer_38,
                dbg_layer_38,
                resid_hold_layer_38,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [35 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [35 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [35 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [35 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [35 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [35 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [35 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [35 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [35 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [35 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [35 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [35 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [35 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [35 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [35 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [35 * local_recv_max, 0]),
                38,
                0,
                my_rank,
            )
            h_layer_39 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_39 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_39 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_39 = self.swa_moe_chip_orch(
                h_layer_38,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [29 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [29 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [29 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [29 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [29 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [29 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [36 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [36 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [36 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [36 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [36 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [36 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [36 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [36 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [36 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [36 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [36 * (sh_inter_local), 0]),
                h_layer_39,
                dbg_layer_39,
                resid_hold_layer_39,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [36 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [36 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [36 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [36 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [36 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [36 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [36 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [36 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [36 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [36 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [36 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [36 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [36 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [36 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [36 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [36 * local_recv_max, 0]),
                39,
                0,
                my_rank,
            )
            h_layer_40 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_40 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_40 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_40 = self.full_moe_chip_orch(
                h_layer_39,
                input_rms,
                pl.slice(full_wq, [HIDDEN, hidden_q_full], [10 * (HIDDEN), 0]),
                pl.slice(full_wk, [HIDDEN, KV_HIDDEN_LOCAL], [10 * (HIDDEN), 0]),
                pl.slice(full_wv, [HIDDEN, KV_HIDDEN_LOCAL], [10 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_full,
                rope_sin_full,
                k_cache,
                v_cache,
                pl.slice(full_wo, [hidden_q_full, HIDDEN], [10 * (hidden_q_full), 0]),
                pl.slice(full_w_g, [HIDDEN, nh_full_pad], [10 * (HIDDEN), 0]),
                pl.slice(full_gate_r, [nh_full_pad, hidden_q_full], [10 * (nh_full_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [37 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [37 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [37 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [37 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [37 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [37 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [37 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [37 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [37 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [37 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [37 * (sh_inter_local), 0]),
                h_layer_40,
                dbg_layer_40,
                resid_hold_layer_40,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [37 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [37 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [37 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [37 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [37 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [37 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [37 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [37 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [37 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [37 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [37 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [37 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [37 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [37 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [37 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [37 * local_recv_max, 0]),
                40,
                0,
                my_rank,
            )
            h_layer_41 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_41 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_41 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_41 = self.swa_moe_chip_orch(
                h_layer_40,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [30 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [30 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [30 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [30 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [30 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [30 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [38 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [38 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [38 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [38 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [38 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [38 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [38 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [38 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [38 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [38 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [38 * (sh_inter_local), 0]),
                h_layer_41,
                dbg_layer_41,
                resid_hold_layer_41,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [38 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [38 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [38 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [38 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [38 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [38 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [38 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [38 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [38 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [38 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [38 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [38 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [38 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [38 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [38 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [38 * local_recv_max, 0]),
                41,
                0,
                my_rank,
            )
            h_layer_42 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_42 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_42 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_42 = self.swa_moe_chip_orch(
                h_layer_41,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [31 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [31 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [31 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [31 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [31 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [31 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [39 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [39 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [39 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [39 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [39 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [39 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [39 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [39 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [39 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [39 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [39 * (sh_inter_local), 0]),
                h_layer_42,
                dbg_layer_42,
                resid_hold_layer_42,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [39 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [39 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [39 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [39 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [39 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [39 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [39 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [39 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [39 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [39 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [39 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [39 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [39 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [39 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [39 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [39 * local_recv_max, 0]),
                42,
                0,
                my_rank,
            )
            h_layer_43 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            dbg_layer_43 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            resid_hold_layer_43 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            h_layer_43 = self.swa_moe_chip_orch(
                h_layer_42,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [32 * (HIDDEN), 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [32 * (HIDDEN), 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [32 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [32 * (hidden_q_swa), 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [32 * (HIDDEN), 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [32 * (nh_swa_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [40 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [40 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [40 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [40 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [40 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [40 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [40 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [40 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [40 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [40 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [40 * (sh_inter_local), 0]),
                h_layer_43,
                dbg_layer_43,
                resid_hold_layer_43,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [40 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [40 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [40 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [40 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [40 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [40 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [40 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [40 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [40 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [40 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [40 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [40 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [40 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [40 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [40 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [40 * local_recv_max, 0]),
                43,
                0,
                my_rank,
            )
            resid_hold_layer_44 = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            next_hidden_out = self.full_moe_chip_orch(
                h_layer_43,
                input_rms,
                pl.slice(full_wq, [HIDDEN, hidden_q_full], [11 * (HIDDEN), 0]),
                pl.slice(full_wk, [HIDDEN, KV_HIDDEN_LOCAL], [11 * (HIDDEN), 0]),
                pl.slice(full_wv, [HIDDEN, KV_HIDDEN_LOCAL], [11 * (HIDDEN), 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_full,
                rope_sin_full,
                k_cache,
                v_cache,
                pl.slice(full_wo, [hidden_q_full, HIDDEN], [11 * (hidden_q_full), 0]),
                pl.slice(full_w_g, [HIDDEN, nh_full_pad], [11 * (HIDDEN), 0]),
                pl.slice(full_gate_r, [nh_full_pad, hidden_q_full], [11 * (nh_full_pad), 0]),
                post_rms,
                pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [41 * (HIDDEN), 0]),
                pl.slice(moe_router_bias, [N_EXPERTS], [41 * (N_EXPERTS)]),
                pl.reshape(pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [41 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [41 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [41 * (n_local_experts * HIDDEN), 0]), [n_local_experts, HIDDEN, inter]),
                pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [41 * (n_local_experts), 0]),
                pl.reshape(pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [41 * (n_local_experts * inter), 0]), [n_local_experts, inter, HIDDEN]),
                pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [41 * (n_local_experts), 0]),
                pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [41 * (HIDDEN), 0]),
                pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [41 * (HIDDEN), 0]),
                pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [41 * (sh_inter_local), 0]),
                next_hidden_out,
                dbg_out,
                resid_hold_layer_44,
                pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [41 * BATCH, 0]),
                pl.slice(moe_attn_signal_stack, [tp_size, 1], [41 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [41 * n_ranks * n_ranks, 0]),
                pl.slice(moe_count_done_stack, [n_ranks, 1], [41 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [41 * local_recv_max, 0]),
                pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [41 * local_recv_max, 0]),
                pl.slice(moe_data_done_stack, [n_ranks, 1], [41 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_recv_route_stack, [local_recv_max, idx_pad], [41 * local_recv_max, 0]),
                pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [41 * local_recv_max, 0]),
                pl.slice(moe_send_scale_stack, [local_recv_max, 8], [41 * local_recv_max, 0]),
                pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [41 * local_recv_max, 0]),
                pl.slice(moe_shared_tmp_stack, [BATCH, HIDDEN], [41 * BATCH, 0]),
                pl.slice(moe_shared_signal_stack, [n_ranks, 1], [41 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [41 * n_routes_per_rank, 0]),
                pl.slice(moe_combine_done_stack, [n_ranks, 1], [41 * COMM_SIGNAL_STRIDE_I32, 0]),
                pl.slice(moe_routed_src_stack, [local_recv_max, HIDDEN], [41 * local_recv_max, 0]),
                44,
                0,
                my_rank,
            )
            logits_shard_out = self.lm_head_orch(
                next_hidden_out, final_norm_weight,
                lm_head_weight, seq_lens, logits_shard_out,
            )
            return logits_shard_out
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(  # noqa: PLR0913, PLR0915
            self,
            current_hidden: pl.Tensor[[tp_size, BATCH, HIDDEN], pl.BF16],
            input_rms: pl.Tensor[[tp_size, LAYER_DYN, HIDDEN], pl.FP32],
            post_rms: pl.Tensor[[tp_size, LAYER_DYN, HIDDEN], pl.FP32],
            q_norm: pl.Tensor[[tp_size, LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm: pl.Tensor[[tp_size, LAYER_DYN, HEAD_DIM], pl.FP32],
            full_wq: pl.Tensor[[tp_size, 12, HIDDEN, hidden_q_full], pl.BF16],
            full_wk: pl.Tensor[[tp_size, 12, HIDDEN, KV_HIDDEN_LOCAL], pl.BF16],
            full_wv: pl.Tensor[[tp_size, 12, HIDDEN, KV_HIDDEN_LOCAL], pl.BF16],
            full_wo: pl.Tensor[[tp_size, 12, hidden_q_full, HIDDEN], pl.BF16],
            full_w_g: pl.Tensor[[tp_size, 12, HIDDEN, nh_full_pad], pl.BF16],
            full_gate_r: pl.Tensor[[tp_size, 12, nh_full_pad, hidden_q_full], pl.BF16],
            swa_wq: pl.Tensor[[tp_size, 33, HIDDEN, hidden_q_swa], pl.BF16],
            swa_wk: pl.Tensor[[tp_size, 33, HIDDEN, KV_HIDDEN_LOCAL], pl.BF16],
            swa_wv: pl.Tensor[[tp_size, 33, HIDDEN, KV_HIDDEN_LOCAL], pl.BF16],
            swa_wo: pl.Tensor[[tp_size, 33, hidden_q_swa, HIDDEN], pl.BF16],
            swa_w_g: pl.Tensor[[tp_size, 33, HIDDEN, nh_swa_pad], pl.BF16],
            swa_gate_r: pl.Tensor[[tp_size, 33, nh_swa_pad, hidden_q_swa], pl.BF16],
            dense_w_gate: pl.Tensor[[tp_size, 3, HIDDEN, INTER_LOCAL], pl.BF16],
            dense_w_up: pl.Tensor[[tp_size, 3, HIDDEN, INTER_LOCAL], pl.BF16],
            dense_w_down: pl.Tensor[[tp_size, 3, INTER_LOCAL, HIDDEN], pl.BF16],
            moe_gate_w: pl.Tensor[[tp_size, 42, HIDDEN, N_EXPERTS], pl.FP32],
            moe_router_bias: pl.Tensor[[tp_size, 42, N_EXPERTS], pl.FP32],
            moe_w_gate_r: pl.Tensor[[tp_size, 42, n_local_experts, HIDDEN, inter], pl.INT8],
            moe_w_gate_r_scale: pl.Tensor[[tp_size, 42, n_local_experts, inter], pl.FP32],
            moe_w_up_r: pl.Tensor[[tp_size, 42, n_local_experts, HIDDEN, inter], pl.INT8],
            moe_w_up_r_scale: pl.Tensor[[tp_size, 42, n_local_experts, inter], pl.FP32],
            moe_w_down_r: pl.Tensor[[tp_size, 42, n_local_experts, inter, HIDDEN], pl.INT8],
            moe_w_down_r_scale: pl.Tensor[[tp_size, 42, n_local_experts, HIDDEN], pl.FP32],
            moe_w_gate_s: pl.Tensor[[tp_size, 42, HIDDEN, sh_inter_local], pl.BF16],
            moe_w_up_s: pl.Tensor[[tp_size, 42, HIDDEN, sh_inter_local], pl.BF16],
            moe_w_down_s: pl.Tensor[[tp_size, 42, sh_inter_local, HIDDEN], pl.BF16],
            seq_lens: pl.Tensor[[tp_size, USER_BATCH_DYN], pl.INT32],
            block_table: pl.Tensor[[tp_size, BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[tp_size, USER_BATCH_DYN], pl.INT32],
            rope_cos_full: pl.Tensor[[tp_size, ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],
            rope_sin_full: pl.Tensor[[tp_size, ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],
            rope_cos_swa: pl.Tensor[[tp_size, ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],
            rope_sin_swa: pl.Tensor[[tp_size, ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],
            k_cache: pl.Tensor[[tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            h_mid_out: pl.Out[pl.Tensor[[tp_size, BATCH, HIDDEN], pl.BF16]],
            next_hidden_out: pl.Out[pl.Tensor[[tp_size, BATCH, HIDDEN], pl.BF16]],
            dbg_out: pl.Out[pl.Tensor[[tp_size, BATCH, HIDDEN], pl.BF16]],
            final_norm_weight: pl.Tensor[[tp_size, 1, HIDDEN], pl.FP32],
            lm_head_weight: pl.Tensor[[tp_size, VOCAB_LOCAL, HIDDEN], pl.BF16],
            logits_shard_out: pl.Out[pl.Tensor[[tp_size, USER_BATCH_DYN, VOCAB_LOCAL], pl.FP32]],
        ):
            # Buffers are stacked by semantic category, not reused by
            # layer. Every per-layer slice therefore has a distinct
            # address. All category sizes are multiples of 512 bytes,
            # preserving both category-base and signal-slot alignment.
            dense_attn_tmp_stack_buf = pld.alloc_window_buffer(WHOLE_CHIP_DENSE_LAYERS * BATCH * HIDDEN * 2)
            dense_attn_signal_stack_buf = pld.alloc_window_buffer(WHOLE_CHIP_DENSE_LAYERS * COMM_CONTROL_SIGNAL_BYTES)
            dense_mlp_tmp_stack_buf = pld.alloc_window_buffer(WHOLE_CHIP_DENSE_LAYERS * BATCH * HIDDEN * 2)
            dense_mlp_signal_stack_buf = pld.alloc_window_buffer(WHOLE_CHIP_DENSE_LAYERS * COMM_CONTROL_SIGNAL_BYTES)
            moe_attn_tmp_stack_buf = pld.alloc_window_buffer(WHOLE_CHIP_MOE_LAYERS * BATCH * HIDDEN * 2)
            moe_attn_signal_stack_buf = pld.alloc_window_buffer(WHOLE_CHIP_MOE_LAYERS * COMM_CONTROL_SIGNAL_BYTES)
            moe_pub_counts_stack_buf = pld.alloc_window_buffer(WHOLE_CHIP_MOE_LAYERS * n_ranks * n_ranks * n_local_experts_pad * 4)
            moe_count_done_stack_buf = pld.alloc_window_buffer(WHOLE_CHIP_MOE_LAYERS * COMM_CONTROL_SIGNAL_BYTES)
            moe_recv_x_stack_buf = pld.alloc_window_buffer(WHOLE_CHIP_MOE_LAYERS * local_recv_max * HIDDEN)
            moe_recv_scale_stack_buf = pld.alloc_window_buffer(WHOLE_CHIP_MOE_LAYERS * local_recv_max * 8 * 4)
            moe_recv_route_stack_buf = pld.alloc_window_buffer(WHOLE_CHIP_MOE_LAYERS * local_recv_max * idx_pad * 4)
            moe_data_done_stack_buf = pld.alloc_window_buffer(WHOLE_CHIP_MOE_LAYERS * COMM_CONTROL_SIGNAL_BYTES)
            moe_send_x_stack_buf = pld.alloc_window_buffer(WHOLE_CHIP_MOE_LAYERS * local_recv_max * HIDDEN)
            moe_send_scale_stack_buf = pld.alloc_window_buffer(WHOLE_CHIP_MOE_LAYERS * local_recv_max * 8 * 4)
            moe_send_route_stack_buf = pld.alloc_window_buffer(WHOLE_CHIP_MOE_LAYERS * local_recv_max * idx_pad * 4)
            moe_shared_tmp_stack_buf = pld.alloc_window_buffer(WHOLE_CHIP_MOE_LAYERS * BATCH * HIDDEN * 2)
            moe_shared_signal_stack_buf = pld.alloc_window_buffer(WHOLE_CHIP_MOE_LAYERS * COMM_CONTROL_SIGNAL_BYTES)
            moe_routed_y_stack_buf = pld.alloc_window_buffer(WHOLE_CHIP_MOE_LAYERS * n_routes_per_rank * HIDDEN * 2)
            moe_combine_done_stack_buf = pld.alloc_window_buffer(WHOLE_CHIP_MOE_LAYERS * COMM_CONTROL_SIGNAL_BYTES)
            moe_routed_src_stack_buf = pld.alloc_window_buffer(WHOLE_CHIP_MOE_LAYERS * local_recv_max * HIDDEN * 2)
            for r in pl.range(pld.world_size()):
                self.whole_chip_orch(
                    current_hidden[r],
                    input_rms[r],
                    post_rms[r],
                    q_norm[r],
                    k_norm[r],
                    pl.reshape(full_wq[r], [12 * HIDDEN, hidden_q_full]),
                    pl.reshape(full_wk[r], [12 * HIDDEN, KV_HIDDEN_LOCAL]),
                    pl.reshape(full_wv[r], [12 * HIDDEN, KV_HIDDEN_LOCAL]),
                    pl.reshape(full_wo[r], [12 * hidden_q_full, HIDDEN]),
                    pl.reshape(full_w_g[r], [12 * HIDDEN, nh_full_pad]),
                    pl.reshape(full_gate_r[r], [12 * nh_full_pad, hidden_q_full]),
                    pl.reshape(swa_wq[r], [33 * HIDDEN, hidden_q_swa]),
                    pl.reshape(swa_wk[r], [33 * HIDDEN, KV_HIDDEN_LOCAL]),
                    pl.reshape(swa_wv[r], [33 * HIDDEN, KV_HIDDEN_LOCAL]),
                    pl.reshape(swa_wo[r], [33 * hidden_q_swa, HIDDEN]),
                    pl.reshape(swa_w_g[r], [33 * HIDDEN, nh_swa_pad]),
                    pl.reshape(swa_gate_r[r], [33 * nh_swa_pad, hidden_q_swa]),
                    pl.reshape(dense_w_gate[r], [3 * HIDDEN, INTER_LOCAL]),
                    pl.reshape(dense_w_up[r], [3 * HIDDEN, INTER_LOCAL]),
                    pl.reshape(dense_w_down[r], [3 * INTER_LOCAL, HIDDEN]),
                    pl.reshape(moe_gate_w[r], [42 * HIDDEN, N_EXPERTS]),
                    pl.reshape(moe_router_bias[r], [42 * N_EXPERTS]),
                    pl.reshape(moe_w_gate_r[r], [42 * n_local_experts * HIDDEN, inter]),
                    pl.reshape(moe_w_gate_r_scale[r], [42 * n_local_experts, inter]),
                    pl.reshape(moe_w_up_r[r], [42 * n_local_experts * HIDDEN, inter]),
                    pl.reshape(moe_w_up_r_scale[r], [42 * n_local_experts, inter]),
                    pl.reshape(moe_w_down_r[r], [42 * n_local_experts * inter, HIDDEN]),
                    pl.reshape(moe_w_down_r_scale[r], [42 * n_local_experts, HIDDEN]),
                    pl.reshape(moe_w_gate_s[r], [42 * HIDDEN, sh_inter_local]),
                    pl.reshape(moe_w_up_s[r], [42 * HIDDEN, sh_inter_local]),
                    pl.reshape(moe_w_down_s[r], [42 * sh_inter_local, HIDDEN]),
                    seq_lens[r],
                    block_table[r],
                    slot_mapping[r],
                    rope_cos_full[r],
                    rope_sin_full[r],
                    rope_cos_swa[r],
                    rope_sin_swa[r],
                    k_cache[r],
                    v_cache[r],
                    h_mid_out[r],
                    next_hidden_out[r],
                    dbg_out[r],
                    final_norm_weight[r],
                    lm_head_weight[r],
                    logits_shard_out[r],
                    pld.window(dense_attn_tmp_stack_buf, [WHOLE_CHIP_DENSE_LAYERS * BATCH, HIDDEN],
                               dtype=pl.BF16),
                    pld.window(dense_attn_signal_stack_buf, [WHOLE_CHIP_DENSE_LAYERS * COMM_SIGNAL_STRIDE_I32, 1],
                               dtype=pl.INT32),
                    pld.window(dense_mlp_tmp_stack_buf, [WHOLE_CHIP_DENSE_LAYERS * BATCH, HIDDEN],
                               dtype=pl.BF16),
                    pld.window(dense_mlp_signal_stack_buf, [WHOLE_CHIP_DENSE_LAYERS * COMM_SIGNAL_STRIDE_I32, 1],
                               dtype=pl.INT32),
                    pld.window(moe_attn_tmp_stack_buf, [WHOLE_CHIP_MOE_LAYERS * BATCH, HIDDEN],
                               dtype=pl.BF16),
                    pld.window(moe_attn_signal_stack_buf, [WHOLE_CHIP_MOE_LAYERS * COMM_SIGNAL_STRIDE_I32, 1],
                               dtype=pl.INT32),
                    pld.window(moe_pub_counts_stack_buf, [WHOLE_CHIP_MOE_LAYERS * n_ranks * n_ranks, n_local_experts_pad],
                               dtype=pl.INT32),
                    pld.window(moe_count_done_stack_buf, [WHOLE_CHIP_MOE_LAYERS * COMM_SIGNAL_STRIDE_I32, 1],
                               dtype=pl.INT32),
                    pld.window(moe_recv_x_stack_buf, [WHOLE_CHIP_MOE_LAYERS * local_recv_max, HIDDEN],
                               dtype=pl.INT8),
                    pld.window(moe_recv_scale_stack_buf, [WHOLE_CHIP_MOE_LAYERS * local_recv_max, 8],
                               dtype=pl.FP32),
                    pld.window(moe_recv_route_stack_buf, [WHOLE_CHIP_MOE_LAYERS * local_recv_max, idx_pad],
                               dtype=pl.INT32),
                    pld.window(moe_data_done_stack_buf, [WHOLE_CHIP_MOE_LAYERS * COMM_SIGNAL_STRIDE_I32, 1],
                               dtype=pl.INT32),
                    pld.window(moe_send_x_stack_buf, [WHOLE_CHIP_MOE_LAYERS * local_recv_max, HIDDEN],
                               dtype=pl.INT8),
                    pld.window(moe_send_scale_stack_buf, [WHOLE_CHIP_MOE_LAYERS * local_recv_max, 8],
                               dtype=pl.FP32),
                    pld.window(moe_send_route_stack_buf, [WHOLE_CHIP_MOE_LAYERS * local_recv_max, idx_pad],
                               dtype=pl.INT32),
                    pld.window(moe_shared_tmp_stack_buf, [WHOLE_CHIP_MOE_LAYERS * BATCH, HIDDEN],
                               dtype=pl.BF16),
                    pld.window(moe_shared_signal_stack_buf, [WHOLE_CHIP_MOE_LAYERS * COMM_SIGNAL_STRIDE_I32, 1],
                               dtype=pl.INT32),
                    pld.window(moe_routed_y_stack_buf, [WHOLE_CHIP_MOE_LAYERS * n_routes_per_rank, HIDDEN],
                               dtype=pl.BF16),
                    pld.window(moe_combine_done_stack_buf, [WHOLE_CHIP_MOE_LAYERS * COMM_SIGNAL_STRIDE_I32, 1],
                               dtype=pl.INT32),
                    pld.window(moe_routed_src_stack_buf, [WHOLE_CHIP_MOE_LAYERS * local_recv_max, HIDDEN],
                               dtype=pl.BF16),
                    r,
                    device=r,
                )

    return WholeDecodeFaithfulRealSingleChip

whole_decode_faithful_real_single_chip = _build_whole_decode_faithful_real_single_chip_program()
