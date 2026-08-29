"""Canonical whole-net Step3.5 decode program.

The 45-layer graph uses runtime ``pl.range`` loops over resident leading-
dimension weight/KV views.  Physical tensor shapes are compile-time capacity;
``num_tokens_per_owner`` supplies the runtime active-row bound.

MoE consumes the replicated post-attention hidden on every TP rank.  Each rank
runs the same gate, packs only routes owned by its local expert shard, combines
its local routed and TP-sharded shared partials, and performs one TP all-reduce
before the residual add.  Legacy EP dispatch/combine windows are removed from
the decode graph and whole-model ABI.
"""
# ruff: noqa: F401

from __future__ import annotations

from pathlib import Path

import pypto.language as pl
import pypto.language.distributed as pld

from models.step3p5.attention_full import attention_full
from models.step3p5.attention_swa import attention_swa
from models.step3p5.dense_mlp import dense_mlp_body_tp

# 注意：attention_full / attention_swa / dense_mlp_body_tp 的 kernel body 经
# ``pl.inline`` 拷进本模块后，body 内引用的 config 常量按 **本模块** 的 module
# global 解析（非 source 模块）。所以 dense 路径三个 kernel 用到的 config 常量必须
# 在这里一次性全导入，否则 codegen 报 UndefinedVariableError（已在
# ``INPUT_PROJ_K_CHUNK`` 上撞过）。LAYER_QHIDDEN_ROWS_DYN 在 full / swa 两个 source
# 模块里值不同（full=12288，swa=33*1536=50688），按 _FULL/_SWA 别名区分。
from models.step3p5.config import (
    ATTN_SCALE,
    BATCH,
    BATCH_TILE,
    BLOCK_SIZE,
    BLOCK_TABLE_FLAT_DYN,
    EPS,
    FULL_ATTN_OUT_PROJ_FUSE_CAST,
    FULL_ATTN_OUT_PROJ_MATMUL_N_CHUNK,
    FULL_ATTN_OUT_PROJ_MATMUL_TILES_PER_TASK,
    FULL_ATTN_OUT_PROJ_VEC_N_CHUNK,
    FULL_ATTN_QK_BLOCKS_PER_TASK,
    FULL_ATTN_QK_UNIFORM_O1,
    FULL_ATTN_SOFTMAX_BLOCKS_PER_TASK,
    FULL_ATTN_SOFTMAX_UNIFORM_O1,
    FULL_ATTN_ONLINE_SOFTMAX_BLOCKS_PER_TASK,
    FULL_ATTN_ONLINE_SOFTMAX_PARTIALS_PER_REDUCE_TASK,
    FULL_ATTN_ONLINE_SOFTMAX_UNIFORM_O1,
    FULL_ATTN_ONLINE_SOFTMAX_REDUCE_UNIFORM_O1,
    HEAD_DIM,
    HEAD_DIM_INV,
    HIDDEN,
    HIDDEN_INV,
    HIDDEN_Q_FULL_LOCAL,
    HIDDEN_Q_SWA_LOCAL,
    INPUT_PROJ_K_CHUNK,
    INTERMEDIATE_LOCAL,
    K_CHUNK,
    KV_CACHE_ROWS_DYN,
    KV_HEADS_LOCAL,
    KV_HIDDEN_LOCAL,
    KV_PROJ_K_CHUNK_LOCAL,
    LAYER_DYN,
    LAYER_HIDDEN_ROWS_DYN,
    LAYER_INTER_ROWS_DYN,
    MAX_SEQ_DEFAULT,
    MLP_OUT_CHUNK,
    MOE_INTERMEDIATE,
    MOE_NUM_EXPERTS,
    MOE_NUM_EXPERTS_LOCAL,
    NUM_HEADS_FULL_LOCAL_PAD,
    NUM_HEADS_SWA_LOCAL_PAD,
    OUT_PROJ_K_CHUNK,
    Q_HEAD_BATCH_FULL,
    Q_HEAD_BATCH_SWA,
    Q_HEAD_PAD_FULL,
    Q_OUT_CHUNK,
    Q_PER_KV_FULL,
    Q_PER_KV_SWA,
    ROPE_SEQ_DYN,
    ROTARY_HALF_FULL,
    ROTARY_HALF_SWA,
    SHARE_EXPERT_DIM_LOCAL,
    SLIDING_WINDOW,
    SWA_OUT_PROJ_FUSE_CAST,
    SWA_OUT_PROJ_MATMUL_N_CHUNK,
    SWA_OUT_PROJ_MATMUL_TILES_PER_TASK,
    SWA_OUT_PROJ_VEC_N_CHUNK,
    SWA_RMSNORM_ROWS_PER_TASK,
    TP_ALL_REDUCE_CHUNK,
    TP_WORLD_SIZE,
    USER_BATCH_DYN,
)
from models.step3p5.attention_full import (
    LAYER_QHIDDEN_ROWS_DYN as LAYER_QHIDDEN_ROWS_DYN_FULL,
)
from models.step3p5.attention_swa import (
    LAYER_QHIDDEN_ROWS_DYN as LAYER_QHIDDEN_ROWS_DYN_SWA,
)

# Per-rank slice widths (TP=8). Single-card ST/UT iron rule: keep per-rank
# widths, never unslice to full.
INTER_LOCAL = INTERMEDIATE_LOCAL                       # 1408
hidden_q_full = HIDDEN_Q_FULL_LOCAL                   # 1024
hidden_q_swa = HIDDEN_Q_SWA_LOCAL                     # 1536
nh_full_pad = NUM_HEADS_FULL_LOCAL_PAD                # 16
nh_swa_pad = NUM_HEADS_SWA_LOCAL_PAD                  # 16
KV_HIDDEN_LOCAL_R = KV_HIDDEN_LOCAL                   # 128
rotary_dim_full = ROTARY_HALF_FULL * 2                # 64
rotary_dim_swa = ROTARY_HALF_SWA * 2                  # 128
tp_size = TP_WORLD_SIZE                               # 8
TP_ALL_REDUCE_OWNED_CHUNK = HIDDEN // TP_WORLD_SIZE  # 512

# Dense-layer counts. L0 = full-attn dense, L1/L2 = swa-attn dense.
NUM_DENSE_LAYERS = 3
NUM_SWA_DENSE_LAYERS = 2                              # L1, L2 inside pl.range(2)
# Stacked leading-dim multipliers (B1): full-attn layers (12) own full_wq;
# swa-attn layers (33) own swa_wq; dense MLP layers (3) own dense_w_gate.
N_FULL_ATTN_LAYERS = 12
N_SWA_ATTN_LAYERS = 33

# Cross-rank control signal footprint: one 512 B cache line per layer slot so
# AtomicAdd / TWAIT traffic never shares a line with a neighbour.
COMM_CONTROL_SIGNAL_BYTES = 512
COMM_SIGNAL_STRIDE_I32 = COMM_CONTROL_SIGNAL_BYTES // 4
# Inline bodies from attention/dense_mlp refer to this symbolic formal.
# Canonical Main binds it to the physical stacked/reused 512B slot; standalone
# and MTP modules bind the same name to TP_WORLD_SIZE for compact signals.
SIGNAL_WINDOW_ROWS = COMM_SIGNAL_STRIDE_I32

# ── MoE-layer module-level constants (silu_silu 40-layer loop). ─────────
# Tiling / sort widths / activation thresholds live at module level so the
# inlined method bodies (gate / dispatch / expert_routed / expert_shared /
# combine) can reference them through parse-time closure capture.  The
# canonical decode ABI uses replicated input and local-owner expert execution.
N_EXPERTS = MOE_NUM_EXPERTS
N_LOCAL_EXPERTS = MOE_NUM_EXPERTS_LOCAL
INTER_R = MOE_INTERMEDIATE
INTER_S_LOCAL = SHARE_EXPERT_DIM_LOCAL
TOPK = 8

# Router (gate) kernel constants — mirrors gate.py / moe.ROUTER_*.
ROUTER_SCORE_PAD = 512
ROUTER_TOPK_PAD = 16
ROUTER_SORT_PAD = ROUTER_TOPK_PAD * 2
ROUTER_GATE_M_TILE = 16
ROUTER_GATE_K_CHUNK = 512
ROUTER_GATE_N_CHUNK = 16
assert BATCH % ROUTER_GATE_M_TILE == 0
ROUTER_FP32_NEG_INF = -3.4028235e38
ROUTER_SCALE = 3.0  # MOE_ROUTER_SCALING_FACTOR
# Eight rows are the smallest backend-safe FP32 scalar store tile. Split the
# configured decode storage into disjoint token workers without duplicating
# any norm/quant work.
MOE_NORM_TOKEN_TILE = 8
MOE_NORM_SCALAR_PAD = 8
assert BATCH % MOE_NORM_TOKEN_TILE == 0
MOE_NORM_BLOCKS = BATCH // MOE_NORM_TOKEN_TILE

# Routed-expert kernel constants — mirrors expert_routed.py / moe.ROUTED_*.
# Keep independent matmul and activation tiles so the cube task grain can be
# tuned without coupling it to the vector epilogue or changing W8A8 rounding.
# The routed weights are [K, N] with N contiguous in GM. K256 x N256 keeps the
# 64 KiB Right/L0B tile ceiling while widening each strided weight load from
# 64 to 256 contiguous bytes. Four 64 KiB V2C slots retain a 256 KiB pipe.
ROUTED_GATE_MM_K_CHUNK = 256
ROUTED_GATE_MM_N_CHUNK = 256
ROUTED_GATE_ACT_N_CHUNK = 64
assert HIDDEN % ROUTED_GATE_MM_K_CHUNK == 0
assert MOE_INTERMEDIATE % ROUTED_GATE_MM_N_CHUNK == 0
# Widen both rowwise quant passes from 20 serial slices to five while preserving
# the full-row amax domain and W8A8 requantization sequence.
ROUTED_H_QUANT_N_CHUNK = 256
assert MOE_INTERMEDIATE % ROUTED_H_QUANT_N_CHUNK == 0
# L43/L44 keep the existing specialization until they are tuned separately.
ROUTED_GATE_K_CHUNK = 64
ROUTED_GATE_N_CHUNK = 64
# INT8 accumulation is exact across K chunks. A 128-wide tile halves the
# routed down matmul_acc chain without changing its rounding contract.
ROUTED_DOWN_K_CHUNK = 128
ROUTED_DOWN_N_CHUNK = 256
ROUTED_NZ_M0 = 16
ROUTED_NZ_N0 = 32
ROUTED_W13_N1 = (2 * MOE_INTERMEDIATE) // ROUTED_NZ_N0
ROUTED_W13_M1 = HIDDEN // ROUTED_NZ_M0
ROUTED_W2_N1 = HIDDEN // ROUTED_NZ_N0
ROUTED_W2_M1 = MOE_INTERMEDIATE // ROUTED_NZ_M0
ROUTED_DOWN_PIPE_FLOATS_PER_BLOCK = 32768
RECV_TILE = 16
ROUTED_SPECIAL_DOWN_N_CHUNK = 128
RECV_SPECIAL_TILE = 32

# Shared-expert kernel constants — mirrors expert_shared.py / moe.SHARED_*.
SHARED_GATE_M_TILE = 16
SHARED_GATE_K_CHUNK = 1024
SHARED_GATE_N_CHUNK = INTER_S_LOCAL  # 160 — one N tile covers the slice
SHARED_DOWN_M_TILE = 16
SHARED_DOWN_K_CHUNK = INTER_S_LOCAL  # 160 — one K tile covers the slice
SHARED_DOWN_N_CHUNK = 256
SHARED_SWIGLU_N_CHUNK = 32
# Gate/up/activation launches one task per active M tile and 32-wide chunk.
# Shared down activates one worker for single-token decode and both workers
# for larger batches. Routed uses the spare-core budget left by shared down:
# 23 workers for one token and 22 workers for larger batches. This is soft
# scheduling, not physical affinity.
SHARED_GATE_UP_ACT_CHUNKS = INTER_S_LOCAL // SHARED_SWIGLU_N_CHUNK
assert BATCH % SHARED_GATE_M_TILE == 0
assert BATCH % SHARED_DOWN_M_TILE == 0
SHARED_DOWN_WORKERS = 2
# Regular routed stages use one grid per stage. Logical expert/tile/chunk work
# is grid-strided inside the kernel so AICPU no longer submits per expert.
ROUTED_GRID_WORKERS = 23
ROUTED_MULTIBATCH_GRID_WORKERS = 22
# Keep all mixed AIC/AIV GMM1 groups resident beside two shared-down groups.
# Logical work remains grid-strided across the 22 resident groups.
ROUTED_FUSED_GRID_WORKERS = 22
ROUTED_FUSED_AIV_WORKERS = ROUTED_FUSED_GRID_WORKERS

# MoE-local helper constants are module-level for parse-time closure capture.
n_ranks = tp_size
n_local_experts = N_LOCAL_EXPERTS
n_local_experts_pad = ((n_local_experts + 7) // 8) * 8
# A single expert can own every active ``(token, topk-slot)`` route.  Keep the
# existing 128-row expert slab because the generated routed-NZ kernels embed
# this stride and the resulting 36 * 128 = 4608 packed buffer extent.
expert_recv_max = BATCH * TOPK
assert expert_recv_max == 128
DISPATCH_SCALE_COLS = 1
inter = MOE_INTERMEDIATE
sh_inter_local = INTER_S_LOCAL
# The local expert ABI uses a fixed slab per owner-local expert. Counts describe
# valid rows only; they must not change the physical base of an expert or the
# shape of the routed-expert loop.
local_recv_max = n_local_experts * expert_recv_max
stage_rows = 8
n_routes_per_rank = BATCH * TOPK
# Counts occupy [0, 36), while the route plan occupies total/active plus at
# most 36 expert IDs in [0, 38). Two Soft producer counters start at index 64
# with a 16-int32 stride; their cache lines are disjoint and stay within
# [49, 96) for every 4-byte-aligned base. The complete plan and workspace are
# zero-published before GMM starts.
local_route_plan_valid_size = n_local_experts + 2
local_route_soft_sync_offset = 64
local_route_soft_sync_cache_line_slots = 16
local_route_soft_sync_counter_count = 2
local_route_soft_sync_slots = (
    local_route_soft_sync_cache_line_slots
    * local_route_soft_sync_counter_count
)
local_route_plan_size = (
    local_route_soft_sync_offset + local_route_soft_sync_slots
)
assert (
    local_route_soft_sync_offset
    - (local_route_soft_sync_cache_line_slots - 1)
    >= local_route_plan_valid_size
)
assert local_route_plan_size >= (
    local_route_soft_sync_offset + local_route_soft_sync_slots
)
sh_tp_chunk = HIDDEN // tp_size
# G1 runtime ABI: one active-token count per owner rank.  This is an ordinary
# host tensor, not a notify/wait signal window.  Its 128-element storage comes
# from the general 512B tensor-shape invariant and deliberately does not reuse
# COMM_SIGNAL_STRIDE_I32 as an ABI concept.
NUM_TOKENS_STORAGE_I32 = 128
# Whole-net host ABI uses the same ordinary padded INT32 storage as the
# per-layer active-token tensor.  Keep a distinct alias for the outer program
# signature so the storage contract is explicit rather than accidentally
# depending on an undefined symbolic shape.
NUM_TOKENS_RUNTIME = NUM_TOKENS_STORAGE_I32
# Canonical product graph is fixed; diagnostics/truncation live in probes.

# MoE-layer counts for the loop form. L3..L42 = 40 MoE silu_silu layers split
# by attention type: 10 full-attn (L3..L12) + 30 swa-attn (L13..L42).
NUM_FULL_MOE_LAYERS = 10
NUM_SWA_MOE_LAYERS = 30
NUM_MOE_LAYERS = NUM_FULL_MOE_LAYERS + NUM_SWA_MOE_LAYERS  # 40 (loop body L3..L42)
# Phase 4: L43 (swa_moe_swiglu7_silu) + L44 (full_moe_swiglu7_swiglu16) are
# post-loop explicit specialization layers that slice MoE weight/window stacks
# at offsets 40/41 → stacks sized to 42 (loop keeps iterating 40).
NUM_MOE_LAYERS_TOTAL = NUM_MOE_LAYERS + 2  # 42

# silu_silu activation closure constants（两个 limit 均为 0.0，因此
# swiglu-clip branch 为 dead code）。这些常量必须位于 module level，因为
# step3p5 是 flat @pl.program class，不是 builder closure。
# Phase 4 will add swiglu7_silu / swiglu7_swiglu16 as separate explicit layers.
_ROUTED_SWIGLU_STEP = False
_ROUTED_SWIGLU_LIMIT = 0.0
_SHARED_SWIGLU_STEP = False
_SHARED_SWIGLU_LIMIT = 0.0

# swiglu7 / swiglu16 activation closure constants (Phase 4 L43/L44 specializations).
# L43 = swa_moe_swiglu7_silu (routed_lim=7.0, shared_lim=0.0); L44 =
# full_moe_swiglu7_swiglu16 (routed_lim=7.0, shared_lim=16.0)，二者均为
# dedicated compile-time specialization。
_ROUTED_SWIGLU7_STEP = True
_ROUTED_SWIGLU7_LIMIT = 7.0
_SHARED_SWIGLU16_STEP = True
_SHARED_SWIGLU16_LIMIT = 16.0

# Inlined kernel functions (DeepSeek form: pl.inline of the _func).
attention_full_inline = pl.inline(attention_full._func)
attention_swa_inline = pl.inline(attention_swa._func)
dense_mlp_inline = pl.inline(dense_mlp_body_tp._func)


_ROUTED_NZ_KERNEL_DIR = (
    Path(__file__).resolve().parents[2] / "kernels" / "step3p5" / "routed_nz"
)


@pl.program
class WholeDecodeStep3p5:
    @pl.function(
        type=pl.FunctionType.AIC,
        external_source=(
            _ROUTED_NZ_KERNEL_DIR / "routed_gmm1_swiglu_quant_aic.cpp"
        ),
    )
    def routed_nz_gmm1_swiglu_quant_aic(
        self,
        local_route_count: pl.InOut[
            pl.Tensor[[local_route_plan_size], pl.INT32]
        ],
        gate_up_i32: pl.InOut[
            pl.Tensor[[local_recv_max, 2 * inter], pl.INT32]
        ],
        local_expert_count: pl.Tensor[[n_local_experts_pad], pl.INT32],
        local_routed_x: pl.Tensor[[local_recv_max, HIDDEN], pl.INT8],
        w13_nz: pl.Tensor[
            [
                n_local_experts,
                ROUTED_W13_N1,
                ROUTED_W13_M1,
                ROUTED_NZ_M0,
                ROUTED_NZ_N0,
            ],
            pl.INT8,
        ],
        h_bf16: pl.InOut[pl.Tensor[[local_recv_max, inter], pl.BF16]],
        local_routed_x_scale: pl.Tensor[[1, local_recv_max], pl.FP32],
        w13_scale: pl.Tensor[[n_local_experts, 2 * inter], pl.FP32],
        h_i8: pl.Out[pl.Tensor[[local_recv_max, inter], pl.INT8]],
        h_scale_dq: pl.Out[pl.Tensor[[1, local_recv_max], pl.FP32]],
    ) -> tuple[
        pl.Tensor[[local_route_plan_size], pl.INT32],
        pl.Tensor[[local_recv_max, 2 * inter], pl.INT32],
        pl.Tensor[[local_recv_max, inter], pl.BF16],
        pl.Tensor[[local_recv_max, inter], pl.INT8],
        pl.Tensor[[1, local_recv_max], pl.FP32],
    ]: ...

    @pl.function(
        type=pl.FunctionType.AIV,
        external_source=(
            _ROUTED_NZ_KERNEL_DIR / "routed_gmm1_swiglu_quant_aiv.cpp"
        ),
        attrs={"dual_aiv_dispatch": True},
    )
    def routed_nz_gmm1_swiglu_quant_aiv(
        self,
        local_route_count: pl.InOut[
            pl.Tensor[[local_route_plan_size], pl.INT32]
        ],
        gate_up_i32: pl.InOut[
            pl.Tensor[[local_recv_max, 2 * inter], pl.INT32]
        ],
        local_expert_count: pl.Tensor[[n_local_experts_pad], pl.INT32],
        local_routed_x: pl.Tensor[[local_recv_max, HIDDEN], pl.INT8],
        w13_nz: pl.Tensor[
            [
                n_local_experts,
                ROUTED_W13_N1,
                ROUTED_W13_M1,
                ROUTED_NZ_M0,
                ROUTED_NZ_N0,
            ],
            pl.INT8,
        ],
        h_bf16: pl.InOut[pl.Tensor[[local_recv_max, inter], pl.BF16]],
        local_routed_x_scale: pl.Tensor[[1, local_recv_max], pl.FP32],
        w13_scale: pl.Tensor[[n_local_experts, 2 * inter], pl.FP32],
        h_i8: pl.Out[pl.Tensor[[local_recv_max, inter], pl.INT8]],
        h_scale_dq: pl.Out[pl.Tensor[[1, local_recv_max], pl.FP32]],
    ) -> tuple[
        pl.Tensor[[local_route_plan_size], pl.INT32],
        pl.Tensor[[local_recv_max, 2 * inter], pl.INT32],
        pl.Tensor[[local_recv_max, inter], pl.BF16],
        pl.Tensor[[local_recv_max, inter], pl.INT8],
        pl.Tensor[[1, local_recv_max], pl.FP32],
    ]: ...

    @pl.function(type=pl.FunctionType.Group)
    def routed_nz_gmm1_swiglu_quant(
        self,
        local_route_count: pl.InOut[
            pl.Tensor[[local_route_plan_size], pl.INT32]
        ],
        gate_up_i32: pl.InOut[
            pl.Tensor[[local_recv_max, 2 * inter], pl.INT32]
        ],
        local_expert_count: pl.Tensor[[n_local_experts_pad], pl.INT32],
        local_routed_x: pl.Tensor[[local_recv_max, HIDDEN], pl.INT8],
        w13_nz: pl.Tensor[
            [
                n_local_experts,
                ROUTED_W13_N1,
                ROUTED_W13_M1,
                ROUTED_NZ_M0,
                ROUTED_NZ_N0,
            ],
            pl.INT8,
        ],
        h_bf16: pl.InOut[pl.Tensor[[local_recv_max, inter], pl.BF16]],
        local_routed_x_scale: pl.Tensor[[1, local_recv_max], pl.FP32],
        w13_scale: pl.Tensor[[n_local_experts, 2 * inter], pl.FP32],
        h_i8: pl.Out[pl.Tensor[[local_recv_max, inter], pl.INT8]],
        h_scale_dq: pl.Out[pl.Tensor[[1, local_recv_max], pl.FP32]],
    ) -> tuple[
        pl.Tensor[[local_route_plan_size], pl.INT32],
        pl.Tensor[[local_recv_max, 2 * inter], pl.INT32],
        pl.Tensor[[local_recv_max, inter], pl.BF16],
        pl.Tensor[[local_recv_max, inter], pl.INT8],
        pl.Tensor[[1, local_recv_max], pl.FP32],
    ]:
        aic_outputs = self.routed_nz_gmm1_swiglu_quant_aic(
            local_route_count,
            gate_up_i32,
            local_expert_count,
            local_routed_x,
            w13_nz,
            h_bf16,
            local_routed_x_scale,
            w13_scale,
            h_i8,
            h_scale_dq,
        )
        self.routed_nz_gmm1_swiglu_quant_aiv(
            local_route_count,
            gate_up_i32,
            local_expert_count,
            local_routed_x,
            w13_nz,
            h_bf16,
            local_routed_x_scale,
            w13_scale,
            h_i8,
            h_scale_dq,
        )
        return aic_outputs

    @pl.function(
        type=pl.FunctionType.AIV,
        external_source=(
            _ROUTED_NZ_KERNEL_DIR / "routed_gmm1_swiglu7_quant_aiv.cpp"
        ),
        attrs={"dual_aiv_dispatch": True},
    )
    def routed_nz_gmm1_swiglu7_quant_aiv(
        self,
        local_route_count: pl.InOut[
            pl.Tensor[[local_route_plan_size], pl.INT32]
        ],
        gate_up_i32: pl.InOut[
            pl.Tensor[[local_recv_max, 2 * inter], pl.INT32]
        ],
        local_expert_count: pl.Tensor[[n_local_experts_pad], pl.INT32],
        local_routed_x: pl.Tensor[[local_recv_max, HIDDEN], pl.INT8],
        w13_nz: pl.Tensor[
            [
                n_local_experts,
                ROUTED_W13_N1,
                ROUTED_W13_M1,
                ROUTED_NZ_M0,
                ROUTED_NZ_N0,
            ],
            pl.INT8,
        ],
        h_bf16: pl.InOut[pl.Tensor[[local_recv_max, inter], pl.BF16]],
        local_routed_x_scale: pl.Tensor[[1, local_recv_max], pl.FP32],
        w13_scale: pl.Tensor[[n_local_experts, 2 * inter], pl.FP32],
        h_i8: pl.Out[pl.Tensor[[local_recv_max, inter], pl.INT8]],
        h_scale_dq: pl.Out[pl.Tensor[[1, local_recv_max], pl.FP32]],
    ) -> tuple[
        pl.Tensor[[local_route_plan_size], pl.INT32],
        pl.Tensor[[local_recv_max, 2 * inter], pl.INT32],
        pl.Tensor[[local_recv_max, inter], pl.BF16],
        pl.Tensor[[local_recv_max, inter], pl.INT8],
        pl.Tensor[[1, local_recv_max], pl.FP32],
    ]: ...

    @pl.function(type=pl.FunctionType.Group)
    def routed_nz_gmm1_swiglu7_quant(
        self,
        local_route_count: pl.InOut[
            pl.Tensor[[local_route_plan_size], pl.INT32]
        ],
        gate_up_i32: pl.InOut[
            pl.Tensor[[local_recv_max, 2 * inter], pl.INT32]
        ],
        local_expert_count: pl.Tensor[[n_local_experts_pad], pl.INT32],
        local_routed_x: pl.Tensor[[local_recv_max, HIDDEN], pl.INT8],
        w13_nz: pl.Tensor[
            [
                n_local_experts,
                ROUTED_W13_N1,
                ROUTED_W13_M1,
                ROUTED_NZ_M0,
                ROUTED_NZ_N0,
            ],
            pl.INT8,
        ],
        h_bf16: pl.InOut[pl.Tensor[[local_recv_max, inter], pl.BF16]],
        local_routed_x_scale: pl.Tensor[[1, local_recv_max], pl.FP32],
        w13_scale: pl.Tensor[[n_local_experts, 2 * inter], pl.FP32],
        h_i8: pl.Out[pl.Tensor[[local_recv_max, inter], pl.INT8]],
        h_scale_dq: pl.Out[pl.Tensor[[1, local_recv_max], pl.FP32]],
    ) -> tuple[
        pl.Tensor[[local_route_plan_size], pl.INT32],
        pl.Tensor[[local_recv_max, 2 * inter], pl.INT32],
        pl.Tensor[[local_recv_max, inter], pl.BF16],
        pl.Tensor[[local_recv_max, inter], pl.INT8],
        pl.Tensor[[1, local_recv_max], pl.FP32],
    ]:
        aic_outputs = self.routed_nz_gmm1_swiglu_quant_aic(
            local_route_count,
            gate_up_i32,
            local_expert_count,
            local_routed_x,
            w13_nz,
            h_bf16,
            local_routed_x_scale,
            w13_scale,
            h_i8,
            h_scale_dq,
        )
        self.routed_nz_gmm1_swiglu7_quant_aiv(
            local_route_count,
            gate_up_i32,
            local_expert_count,
            local_routed_x,
            w13_nz,
            h_bf16,
            local_routed_x_scale,
            w13_scale,
            h_i8,
            h_scale_dq,
        )
        return aic_outputs

    @pl.function(
        type=pl.FunctionType.AIC,
        external_source=_ROUTED_NZ_KERNEL_DIR / "expert_down_aic.cpp",
    )
    def routed_nz_down_aic(
        self,
        local_route_count: pl.Tensor[[local_route_plan_size], pl.INT32],
        local_routed_y: pl.Out[
            pl.Tensor[[local_recv_max, HIDDEN], pl.BF16]
        ],
        local_expert_count: pl.Tensor[[n_local_experts_pad], pl.INT32],
        h_i8: pl.Tensor[[local_recv_max, inter], pl.INT8],
        w_down_nz: pl.Tensor[
            [
                n_local_experts,
                ROUTED_W2_N1,
                ROUTED_W2_M1,
                ROUTED_NZ_M0,
                ROUTED_NZ_N0,
            ],
            pl.INT8,
        ],
        w_down_scale: pl.Tensor[[n_local_experts, HIDDEN], pl.FP32],
        h_scale_dq: pl.Tensor[[1, local_recv_max], pl.FP32],
        local_routed_weight: pl.Tensor[[local_recv_max], pl.FP32],
        pipe_buffer: pl.Out[
            pl.Tensor[
                [ROUTED_DOWN_PIPE_FLOATS_PER_BLOCK * ROUTED_GRID_WORKERS],
                pl.FP32,
            ]
        ],
        routed_workers: pl.Scalar[pl.INDEX],
    ) -> tuple[
        pl.Tensor[[local_recv_max, HIDDEN], pl.BF16],
        pl.Tensor[
            [ROUTED_DOWN_PIPE_FLOATS_PER_BLOCK * ROUTED_GRID_WORKERS],
            pl.FP32,
        ],
    ]: ...

    @pl.function(
        type=pl.FunctionType.AIV,
        external_source=_ROUTED_NZ_KERNEL_DIR / "expert_down_aiv.cpp",
        attrs={"dual_aiv_dispatch": True},
    )
    def routed_nz_down_aiv(
        self,
        local_route_count: pl.Tensor[[local_route_plan_size], pl.INT32],
        local_routed_y: pl.Out[
            pl.Tensor[[local_recv_max, HIDDEN], pl.BF16]
        ],
        local_expert_count: pl.Tensor[[n_local_experts_pad], pl.INT32],
        h_i8: pl.Tensor[[local_recv_max, inter], pl.INT8],
        w_down_nz: pl.Tensor[
            [
                n_local_experts,
                ROUTED_W2_N1,
                ROUTED_W2_M1,
                ROUTED_NZ_M0,
                ROUTED_NZ_N0,
            ],
            pl.INT8,
        ],
        w_down_scale: pl.Tensor[[n_local_experts, HIDDEN], pl.FP32],
        h_scale_dq: pl.Tensor[[1, local_recv_max], pl.FP32],
        local_routed_weight: pl.Tensor[[local_recv_max], pl.FP32],
        pipe_buffer: pl.Out[
            pl.Tensor[
                [ROUTED_DOWN_PIPE_FLOATS_PER_BLOCK * ROUTED_GRID_WORKERS],
                pl.FP32,
            ]
        ],
        routed_workers: pl.Scalar[pl.INDEX],
    ) -> tuple[
        pl.Tensor[[local_recv_max, HIDDEN], pl.BF16],
        pl.Tensor[
            [ROUTED_DOWN_PIPE_FLOATS_PER_BLOCK * ROUTED_GRID_WORKERS],
            pl.FP32,
        ],
    ]: ...

    @pl.function(type=pl.FunctionType.Group)
    def routed_nz_down(
        self,
        local_route_count: pl.Tensor[[local_route_plan_size], pl.INT32],
        local_routed_y: pl.Out[
            pl.Tensor[[local_recv_max, HIDDEN], pl.BF16]
        ],
        local_expert_count: pl.Tensor[[n_local_experts_pad], pl.INT32],
        h_i8: pl.Tensor[[local_recv_max, inter], pl.INT8],
        w_down_nz: pl.Tensor[
            [
                n_local_experts,
                ROUTED_W2_N1,
                ROUTED_W2_M1,
                ROUTED_NZ_M0,
                ROUTED_NZ_N0,
            ],
            pl.INT8,
        ],
        w_down_scale: pl.Tensor[[n_local_experts, HIDDEN], pl.FP32],
        h_scale_dq: pl.Tensor[[1, local_recv_max], pl.FP32],
        local_routed_weight: pl.Tensor[[local_recv_max], pl.FP32],
        pipe_buffer: pl.Out[
            pl.Tensor[
                [ROUTED_DOWN_PIPE_FLOATS_PER_BLOCK * ROUTED_GRID_WORKERS],
                pl.FP32,
            ]
        ],
        routed_workers: pl.Scalar[pl.INDEX],
    ) -> tuple[
        pl.Tensor[[local_recv_max, HIDDEN], pl.BF16],
        pl.Tensor[
            [ROUTED_DOWN_PIPE_FLOATS_PER_BLOCK * ROUTED_GRID_WORKERS],
            pl.FP32,
        ],
    ]:
        aic_outputs = self.routed_nz_down_aic(
            local_route_count,
            local_routed_y,
            local_expert_count,
            h_i8,
            w_down_nz,
            w_down_scale,
            h_scale_dq,
            local_routed_weight,
            pipe_buffer,
            routed_workers,
        )
        self.routed_nz_down_aiv(
            local_route_count,
            local_routed_y,
            local_expert_count,
            h_i8,
            w_down_nz,
            w_down_scale,
            h_scale_dq,
            local_routed_weight,
            pipe_buffer,
            routed_workers,
        )
        return aic_outputs

    # ── TP all-reduce collective ────────────────────────────────────────
    # The inlined attention and dense-MLP bodies call this method to gather
    # o_proj and down_proj partial sums.  Keep the method on this program so
    # pl.inline resolves those calls locally. Single-row payloads use a
    # two-wave one-shot mesh; larger payloads use a static-bucket three-wave
    # path so partially occupied decode batches do not move all 16 rows.
    @pl.function(type=pl.FunctionType.InCore)
    def tp_all_reduce(
        self,
        local: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
        signal_window: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        active_rows_i32: pl.Scalar[pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
        group_size = tp_size

        # Keep the existing full-size communication window ABI.  The transfer
        # grain is configurable, while reduce-scatter ownership is defined by
        # the TP rank count rather than a fixed core count.
        ar_chunk = TP_ALL_REDUCE_CHUNK
        active_rows = pl.cast(active_rows_i32, pl.INDEX)
        if active_rows > BATCH:
            active_rows = pl.cast(BATCH, pl.INDEX)
        if active_rows < 1:
            # Preserve the legacy full-capacity fallback for defensive
            # out-of-contract zero/negative row counts.
            active_rows = pl.cast(BATCH, pl.INDEX)

        # Match HCCL's small-message selector: a single active BF16 row is
        # only 8 KiB, so a one-shot full-width mesh has fewer remote
        # transactions than reduce-scatter plus push all-gather. All transfer
        # extents in this branch are static, avoiding dynamic-TPUT and
        # dynamic-remote-load limitations in the pinned toolchain.
        if active_rows == 1:
            # Self-target TPUT drains before the publication wave (PTOAS#872).
            pld.tensor.put(
                dst=tmp_window,
                peer=my_rank,
                src=local,
                dst_offsets=[0, 0],
                src_offsets=[0, 0],
                shape=[1, HIDDEN],
                chunk_rows=1,
                chunk_cols=TP_ALL_REDUCE_CHUNK,
            )

            # Publication wave: every peer source row is now readable.
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

            # The full 1x4096 row fits in UB. Keep canonical peer order, one
            # FP32 accumulator, and one final BF16 cast for byte-exact output.
            own_row = pl.load(tmp_window, [0, 0], [1, HIDDEN])
            row_acc = pl.mul(
                pl.cast(own_row, target_type=pl.FP32), 0.0,
            )
            for peer in pl.range(group_size):
                if peer == my_rank:
                    row_acc = pl.add(
                        row_acc,
                        pl.cast(own_row, target_type=pl.FP32),
                    )
                else:
                    remote_row = pld.tile.remote_load(
                        tmp_window,
                        peer=peer,
                        offsets=[0, 0],
                        shape=[1, HIDDEN],
                    )
                    row_acc = pl.add(
                        row_acc,
                        pl.cast(remote_row, target_type=pl.FP32),
                    )
            pl.store(
                pl.cast(row_acc, target_type=pl.BF16),
                [0, 0],
                local,
            )

            # Completion wave protects peer reads before the per-layer window
            # can be reset or reused on a later invocation.
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
        else:
            # Runtime extents are not legal tensor shapes in the pinned
            # compiler. Round active rows up to a rank-uniform static bucket
            # instead. Rows are reduced independently, so a rounded inactive
            # tail cannot affect the active prefix consumed downstream.
            if active_rows <= 2:
                pld.tensor.put(
                    dst=tmp_window,
                    peer=my_rank,
                    src=local,
                    dst_offsets=[0, 0],
                    src_offsets=[0, 0],
                    shape=[2, HIDDEN],
                    chunk_rows=2,
                    chunk_cols=TP_ALL_REDUCE_CHUNK,
                )
            elif active_rows <= 4:
                pld.tensor.put(
                    dst=tmp_window,
                    peer=my_rank,
                    src=local,
                    dst_offsets=[0, 0],
                    src_offsets=[0, 0],
                    shape=[4, HIDDEN],
                    chunk_rows=4,
                    chunk_cols=TP_ALL_REDUCE_CHUNK,
                )
            elif active_rows <= 8:
                pld.tensor.put(
                    dst=tmp_window,
                    peer=my_rank,
                    src=local,
                    dst_offsets=[0, 0],
                    src_offsets=[0, 0],
                    shape=[8, HIDDEN],
                    chunk_rows=8,
                    chunk_cols=TP_ALL_REDUCE_CHUNK,
                )
            elif active_rows <= 16:
                pld.tensor.put(
                    dst=tmp_window,
                    peer=my_rank,
                    src=local,
                    dst_offsets=[0, 0],
                    src_offsets=[0, 0],
                    shape=[16, HIDDEN],
                    chunk_rows=16,
                    chunk_cols=TP_ALL_REDUCE_CHUNK,
                )
            else:
                # Configurations with capacity above 16 retain the original
                # complete-capacity fallback.
                pld.tensor.put(
                    dst=tmp_window,
                    peer=my_rank,
                    src=local,
                    chunk_rows=BATCH_TILE,
                    chunk_cols=TP_ALL_REDUCE_CHUNK,
                )

            # Wave 1 publishes all source partials.
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

            # Reduce-scatter: rank r uniquely owns chunk r.  Preserve the
            # canonical peer order 0..N-1, one FP32 accumulator, and one final
            # BF16 cast to retain the numerical contract of the pull mesh.
            owned_base = my_rank * TP_ALL_REDUCE_OWNED_CHUNK
            if active_rows <= 2:
                own_tile_2 = pl.load(
                    tmp_window,
                    [0, owned_base],
                    [2, TP_ALL_REDUCE_OWNED_CHUNK],
                )
                acc_2 = pl.mul(
                    pl.cast(own_tile_2, target_type=pl.FP32),
                    0.0,
                )
                for peer in pl.range(group_size):
                    if peer == my_rank:
                        acc_2 = pl.add(
                            acc_2,
                            pl.cast(own_tile_2, target_type=pl.FP32),
                        )
                    else:
                        remote_tile_2 = pld.tile.remote_load(
                            tmp_window,
                            peer=peer,
                            offsets=[0, owned_base],
                            shape=[2, TP_ALL_REDUCE_OWNED_CHUNK],
                        )
                        acc_2 = pl.add(
                            acc_2,
                            pl.cast(remote_tile_2, target_type=pl.FP32),
                        )
                reduced_tile_2 = pl.cast(acc_2, target_type=pl.BF16)

                # Publish the write-disjoint reduced shard with the existing push
                # path. Batch tiling changes only the transfer tile, not ownership.
                pl.store(reduced_tile_2, [0, owned_base], tmp_window)
                for dst in pl.range(group_size):
                    if dst != my_rank:
                        pld.tile.remote_store(
                            reduced_tile_2,
                            target=tmp_window,
                            peer=dst,
                            offsets=[0, owned_base],
                        )
            elif active_rows <= 4:
                own_tile_4 = pl.load(
                    tmp_window,
                    [0, owned_base],
                    [4, TP_ALL_REDUCE_OWNED_CHUNK],
                )
                acc_4 = pl.mul(
                    pl.cast(own_tile_4, target_type=pl.FP32),
                    0.0,
                )
                for peer in pl.range(group_size):
                    if peer == my_rank:
                        acc_4 = pl.add(
                            acc_4,
                            pl.cast(own_tile_4, target_type=pl.FP32),
                        )
                    else:
                        remote_tile_4 = pld.tile.remote_load(
                            tmp_window,
                            peer=peer,
                            offsets=[0, owned_base],
                            shape=[4, TP_ALL_REDUCE_OWNED_CHUNK],
                        )
                        acc_4 = pl.add(
                            acc_4,
                            pl.cast(remote_tile_4, target_type=pl.FP32),
                        )
                reduced_tile_4 = pl.cast(acc_4, target_type=pl.BF16)
                pl.store(reduced_tile_4, [0, owned_base], tmp_window)
                for dst in pl.range(group_size):
                    if dst != my_rank:
                        pld.tile.remote_store(
                            reduced_tile_4,
                            target=tmp_window,
                            peer=dst,
                            offsets=[0, owned_base],
                        )
            elif active_rows <= 8:
                own_tile_8 = pl.load(
                    tmp_window,
                    [0, owned_base],
                    [8, TP_ALL_REDUCE_OWNED_CHUNK],
                )
                acc_8 = pl.mul(
                    pl.cast(own_tile_8, target_type=pl.FP32),
                    0.0,
                )
                for peer in pl.range(group_size):
                    if peer == my_rank:
                        acc_8 = pl.add(
                            acc_8,
                            pl.cast(own_tile_8, target_type=pl.FP32),
                        )
                    else:
                        remote_tile_8 = pld.tile.remote_load(
                            tmp_window,
                            peer=peer,
                            offsets=[0, owned_base],
                            shape=[8, TP_ALL_REDUCE_OWNED_CHUNK],
                        )
                        acc_8 = pl.add(
                            acc_8,
                            pl.cast(remote_tile_8, target_type=pl.FP32),
                        )
                reduced_tile_8 = pl.cast(acc_8, target_type=pl.BF16)
                pl.store(reduced_tile_8, [0, owned_base], tmp_window)
                for dst in pl.range(group_size):
                    if dst != my_rank:
                        pld.tile.remote_store(
                            reduced_tile_8,
                            target=tmp_window,
                            peer=dst,
                            offsets=[0, owned_base],
                        )
            elif active_rows <= 16:
                own_tile_16 = pl.load(
                    tmp_window,
                    [0, owned_base],
                    [16, TP_ALL_REDUCE_OWNED_CHUNK],
                )
                acc_16 = pl.mul(
                    pl.cast(own_tile_16, target_type=pl.FP32),
                    0.0,
                )
                for peer in pl.range(group_size):
                    if peer == my_rank:
                        acc_16 = pl.add(
                            acc_16,
                            pl.cast(own_tile_16, target_type=pl.FP32),
                        )
                    else:
                        remote_tile_16 = pld.tile.remote_load(
                            tmp_window,
                            peer=peer,
                            offsets=[0, owned_base],
                            shape=[16, TP_ALL_REDUCE_OWNED_CHUNK],
                        )
                        acc_16 = pl.add(
                            acc_16,
                            pl.cast(remote_tile_16, target_type=pl.FP32),
                        )
                reduced_tile_16 = pl.cast(
                    acc_16,
                    target_type=pl.BF16,
                )
                pl.store(reduced_tile_16, [0, owned_base], tmp_window)
                for dst in pl.range(group_size):
                    if dst != my_rank:
                        pld.tile.remote_store(
                            reduced_tile_16,
                            target=tmp_window,
                            peer=dst,
                            offsets=[0, owned_base],
                        )
            else:
                for ar_b0 in pl.range(0, BATCH, BATCH_TILE):
                    own_tile = pl.load(
                        tmp_window,
                        [ar_b0, owned_base],
                        [BATCH_TILE, TP_ALL_REDUCE_OWNED_CHUNK],
                    )
                    acc = pl.mul(
                        pl.cast(own_tile, target_type=pl.FP32),
                        0.0,
                    )
                    for peer in pl.range(group_size):
                        if peer == my_rank:
                            acc = pl.add(
                                acc,
                                pl.cast(own_tile, target_type=pl.FP32),
                            )
                        else:
                            remote_tile = pld.tile.remote_load(
                                tmp_window,
                                peer=peer,
                                offsets=[ar_b0, owned_base],
                                shape=[
                                    BATCH_TILE,
                                    TP_ALL_REDUCE_OWNED_CHUNK,
                                ],
                            )
                            acc = pl.add(
                                acc,
                                pl.cast(remote_tile, target_type=pl.FP32),
                            )
                    reduced_tile = pl.cast(acc, target_type=pl.BF16)
                    pl.store(
                        reduced_tile,
                        [ar_b0, owned_base],
                        tmp_window,
                    )
                    for dst in pl.range(group_size):
                        if dst != my_rank:
                            pld.tile.remote_store(
                                reduced_tile,
                                target=tmp_window,
                                peer=dst,
                                offsets=[ar_b0, owned_base],
                            )

            # Wave 2 publishes all pushed result chunks.
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

            # Copy only the selected static bucket back to the local result.
            if active_rows <= 2:
                for k0 in pl.parallel(0, HIDDEN, ar_chunk):
                    result_tile_2 = pl.load(
                        tmp_window,
                        [0, k0],
                        [2, ar_chunk],
                    )
                    pl.store(result_tile_2, [0, k0], local)
            elif active_rows <= 4:
                for k0 in pl.parallel(0, HIDDEN, ar_chunk):
                    result_tile_4 = pl.load(
                        tmp_window,
                        [0, k0],
                        [4, ar_chunk],
                    )
                    pl.store(result_tile_4, [0, k0], local)
            elif active_rows <= 8:
                for k0 in pl.parallel(0, HIDDEN, ar_chunk):
                    result_tile_8 = pl.load(
                        tmp_window,
                        [0, k0],
                        [8, ar_chunk],
                    )
                    pl.store(result_tile_8, [0, k0], local)
            elif active_rows <= 16:
                for k0 in pl.parallel(0, HIDDEN, ar_chunk):
                    result_tile_16 = pl.load(
                        tmp_window,
                        [0, k0],
                        [16, ar_chunk],
                    )
                    pl.store(result_tile_16, [0, k0], local)
            else:
                ar_copy_tiles = (
                    (BATCH // BATCH_TILE) * (HIDDEN // ar_chunk)
                )
                for ar_copy in pl.parallel(ar_copy_tiles):
                    ar_b_idx = ar_copy // (HIDDEN // ar_chunk)
                    ar_k_idx = ar_copy % (HIDDEN // ar_chunk)
                    ar_b0 = ar_b_idx * BATCH_TILE
                    k0 = ar_k_idx * ar_chunk
                    result_tile = pl.load(
                        tmp_window,
                        [ar_b0, k0],
                        [BATCH_TILE, ar_chunk],
                    )
                    pl.store(result_tile, [ar_b0, k0], local)

            # Wave 3 closes the communication-window read lifetime.  Every rank
            # finishes its final local reads before the window can be reused.
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
                        expected=3, cmp=pld.WaitCmp.Ge,
                    )
        return local

    @pl.function(type=pl.FunctionType.InCore)
    def tp_all_reduce_residual_bs1(
        self,
        local: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        residual_out: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
        signal_window: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
        group_size = tp_size

        # Self-target TPUT drains before the publication wave (PTOAS#872).
        pld.tensor.put(
            dst=tmp_window,
            peer=my_rank,
            src=local,
            dst_offsets=[0, 0],
            src_offsets=[0, 0],
            shape=[1, HIDDEN],
            chunk_rows=1,
            chunk_cols=TP_ALL_REDUCE_CHUNK,
        )

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

        own_row = pl.load(tmp_window, [0, 0], [1, HIDDEN])
        row_acc = pl.mul(
            pl.cast(own_row, target_type=pl.FP32),
            0.0,
        )
        for peer in pl.range(group_size):
            if peer == my_rank:
                row_acc = pl.add(
                    row_acc,
                    pl.cast(own_row, target_type=pl.FP32),
                )
            else:
                remote_row = pld.tile.remote_load(
                    tmp_window,
                    peer=peer,
                    offsets=[0, 0],
                    shape=[1, HIDDEN],
                )
                row_acc = pl.add(
                    row_acc,
                    pl.cast(remote_row, target_type=pl.FP32),
                )
        reduced_bf16 = pl.cast(row_acc, target_type=pl.BF16)
        pl.store(reduced_bf16, [0, 0], local)

        # Close the peer-read lifetime before the local residual epilogue.
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

        for k0 in pl.range(0, HIDDEN, TP_ALL_REDUCE_CHUNK):
            reduced_chunk = pl.load(
                local,
                [0, k0],
                [1, TP_ALL_REDUCE_CHUNK],
            )
            residual_chunk = pl.load(
                residual_out,
                [0, k0],
                [1, TP_ALL_REDUCE_CHUNK],
            )
            residual_sum = pl.add(
                pl.cast(reduced_chunk, target_type=pl.FP32),
                pl.cast(residual_chunk, target_type=pl.FP32),
            )
            pl.store(
                pl.cast(residual_sum, target_type=pl.BF16),
                [0, k0],
                residual_out,
            )
        return residual_out

    @pl.function(
        type=pl.FunctionType.Orchestration,
        attrs={"inline_orchestration": True},
    )
    def full_chip_orch(  # noqa: PLR0913, PLR0915
        self,
        current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
        wq: pl.Tensor[[HIDDEN, hidden_q_full], pl.BF16],
        wk: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        wv: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
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
        attn_signal_window: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        mlp_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
        mlp_signal_window: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        norm_layer_idx: pl.Scalar[pl.INT32],
        attn_layer_idx: pl.Scalar[pl.INT32],
        mlp_layer_idx: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
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
            num_tokens,
            attn_tmp_window, attn_signal_window, my_rank,
        )
        h0_out = dense_mlp_inline(
            resid1, post_rms_weight, w_gate, w_up, w_down,
            h0_out, norm_layer_idx, mlp_layer_idx, num_tokens,
            mlp_tmp_window, mlp_signal_window, my_rank,
        )
        return h0_out

    @pl.function(
        type=pl.FunctionType.Orchestration,
        attrs={"inline_orchestration": True},
    )
    def swa_chip_orch(  # noqa: PLR0913, PLR0915
        self,
        current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
        wq: pl.Tensor[[HIDDEN, hidden_q_swa], pl.BF16],
        wk: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        wv: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
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
        attn_signal_window: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        mlp_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
        mlp_signal_window: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        norm_layer_idx: pl.Scalar[pl.INT32],
        attn_layer_idx: pl.Scalar[pl.INT32],
        mlp_layer_idx: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
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
            num_tokens,
            attn_tmp_window, attn_signal_window, my_rank,
        )
        hidden_out = dense_mlp_inline(
            resid1, post_rms_weight, w_gate, w_up, w_down,
            hidden_out, norm_layer_idx, mlp_layer_idx, num_tokens,
            mlp_tmp_window, mlp_signal_window, my_rank,
        )
        return hidden_out

    @pl.function(type=pl.FunctionType.Inline)
    def _gate(
        self,
        resid: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
        norm_layer_idx: pl.Scalar[pl.INT32],
        inv_rms: pl.Tensor[[BATCH, 1], pl.FP32],
        gate_w: pl.Tensor[[N_EXPERTS, HIDDEN], pl.FP32],
        router_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
        expert_indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
        expert_weights: pl.Tensor[[BATCH, TOPK], pl.FP32],
        num_tokens: pl.Scalar[pl.INT32],
    ) -> tuple[
        pl.Tensor[[BATCH, TOPK], pl.INT32],
        pl.Tensor[[BATCH, TOPK], pl.FP32],
    ]:
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > BATCH:
            active_tokens = pl.cast(BATCH, pl.INDEX)
        score_buf = pl.create_tensor(
            [BATCH, ROUTER_SCORE_PAD], dtype=pl.FP32,
        )
        biased_buf = pl.create_tensor(
            [BATCH, ROUTER_SCORE_PAD], dtype=pl.FP32,
        )
        # PERF/GATE-DECOUPLE: raw FP32 gate logits, pre inv_rms/sigmoid/bias.
        # Only columns below N_EXPERTS are ever read, so the pad needs no
        # initialization.
        logit_buf = pl.create_tensor(
            [BATCH, ROUTER_SCORE_PAD], dtype=pl.FP32,
        )
        # Materialize x * (gamma + 1) once per token row. The gate fanout
        # then reuses the FP32 matrix instead of rebuilding the same 16xK
        # tile independently for every expert-column worker.
        gate_xg = pl.create_tensor(
            [BATCH, HIDDEN], dtype=pl.FP32, manual_dep=True,
        )

        # G1a: materialize only active rows. The task count follows the
        # runtime batch, while the downstream cube keeps the hardware M=16
        # minimum tile.
        with pl.spmd(
            active_tokens,
            name_hint="gate_xg_precompute",
            allow_early_resolve=True,
        ) as gate_xg_tid:
            gate_row = pl.tile.get_block_idx()
            for kb in pl.range(HIDDEN // ROUTER_GATE_K_CHUNK):
                k0 = kb * ROUTER_GATE_K_CHUNK
                raw = pl.cast(
                    pl.slice(
                        resid, [1, ROUTER_GATE_K_CHUNK], [gate_row, k0],
                    ),
                    target_type=pl.FP32,
                )
                gamma = pl.slice(
                    post_rms_weight, [1, ROUTER_GATE_K_CHUNK],
                    [norm_layer_idx, k0],
                )
                gate_xg[
                    gate_row : gate_row + 1, k0 : k0 + ROUTER_GATE_K_CHUNK
                ] = pl.col_expand_mul(raw, pl.add(gamma, 1.0))

        # G1b: fan out over the active M tiles and expert-column tiles.
        # The checkpoint-native [N, K] layout gives every worker contiguous K
        # loads, matching vLLM-Ascend MatMulV2(false, true). This avoids the
        # 64-byte strided rows of the previous [K, N] storage.
        active_gate_tiles = (
            active_tokens + ROUTER_GATE_M_TILE - 1
        ) // ROUTER_GATE_M_TILE
        gate_n_blocks = N_EXPERTS // ROUTER_GATE_N_CHUNK
        with pl.spmd(
            active_gate_tiles * gate_n_blocks,
            name_hint="gate_expert_fanout",
            deps=[gate_xg_tid],
            allow_early_resolve=True,
        ) as _gate_fanout_tid:
            task = pl.tile.get_block_idx()
            mb = task // gate_n_blocks
            nb = task % gate_n_blocks
            m0 = mb * ROUTER_GATE_M_TILE
            n0 = nb * ROUTER_GATE_N_CHUNK
            x0 = pl.slice(
                gate_xg,
                [ROUTER_GATE_M_TILE, ROUTER_GATE_K_CHUNK],
                [m0, 0],
            )
            w0 = pl.slice(
                gate_w,
                [ROUTER_GATE_N_CHUNK, ROUTER_GATE_K_CHUNK],
                [n0, 0],
            )
            logits_n = pl.matmul(
                x0, w0, out_dtype=pl.FP32, b_trans=True,
            )
            for kb in pl.range(1, HIDDEN // ROUTER_GATE_K_CHUNK):
                k0 = kb * ROUTER_GATE_K_CHUNK
                xk = pl.slice(
                    gate_xg,
                    [ROUTER_GATE_M_TILE, ROUTER_GATE_K_CHUNK],
                    [m0, k0],
                )
                wk = pl.slice(
                    gate_w,
                    [ROUTER_GATE_N_CHUNK, ROUTER_GATE_K_CHUNK],
                    [n0, k0],
                )
                logits_n = pl.matmul_acc(
                    logits_n, xk, wk, b_trans=True,
                )
            # Store the FP32 accumulator directly through FIXPIPE. This keeps
            # the fanout cube-only and avoids a C2V handoff for an identity op.
            logit_buf = pl.assemble(logit_buf, logits_n, [m0, n0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_topk"):
            # TOPK=8 gives a legal 32-byte row for both INT32 and FP32.  One
            # producer fills complete 512-byte tiles and publishes each output
            # through a single TSTORE; inactive token rows stay zero.
            expert_indices_tile = pl.tile.full(
                [BATCH, TOPK], dtype=pl.INT32, value=0,
            )
            expert_weights_tile = pl.tile.full(
                [BATCH, TOPK], dtype=pl.FP32, value=0.0,
            )
            # Process only logical rows. The cube still executes its required
            # 16-row tile, but sigmoid, bias, sort and normalization scale with
            # the runtime batch like vLLM-Ascend MoeGatingTopK.
            for tt in pl.range(active_tokens):
                score_buf[tt : tt + 1, :] = pl.full(
                    [1, ROUTER_SCORE_PAD], dtype=pl.FP32, value=0.0,
                )
                biased_buf[tt : tt + 1, :] = pl.full(
                    [1, ROUTER_SCORE_PAD],
                    dtype=pl.FP32,
                    value=ROUTER_FP32_NEG_INF,
                )
                inv_rms_scalar = pl.read(inv_rms, [tt, 0])
                for nb2 in pl.range(
                    N_EXPERTS // ROUTER_GATE_N_CHUNK,
                ):
                    n0b = nb2 * ROUTER_GATE_N_CHUNK
                    logits_b = pl.mul(
                        pl.slice(
                            logit_buf,
                            [1, ROUTER_GATE_N_CHUNK],
                            [tt, n0b],
                        ),
                        inv_rms_scalar,
                    )
                    score_n_chunk = pl.recip(
                        pl.add(pl.exp(pl.neg(logits_b)), 1.0),
                    )
                    bias_chunk = pl.slice(
                        router_bias, [ROUTER_GATE_N_CHUNK], [n0b],
                    )
                    bias_row_chunk = pl.reshape(
                        bias_chunk, [1, ROUTER_GATE_N_CHUNK],
                    )
                    # vLLM applies the router bias in BF16. Preserve that
                    # rounding before widening it for the FP32 selection key.
                    bias_row_chunk = pl.cast(
                        pl.cast(bias_row_chunk, target_type=pl.BF16),
                        target_type=pl.FP32,
                    )
                    biased_n_chunk = pl.add(
                        score_n_chunk,
                        pl.col_expand_mul(
                            pl.full(
                                [1, ROUTER_GATE_N_CHUNK],
                                dtype=pl.FP32,
                                value=1.0,
                            ),
                            bias_row_chunk,
                        ),
                    )
                    score_buf[
                        tt : tt + 1,
                        n0b : n0b + ROUTER_GATE_N_CHUNK,
                    ] = score_n_chunk
                    biased_buf[
                        tt : tt + 1,
                        n0b : n0b + ROUTER_GATE_N_CHUNK,
                    ] = biased_n_chunk

                row = biased_buf[tt : tt + 1, :]
                idx_init = pl.arange(
                    0, [1, ROUTER_SCORE_PAD], dtype=pl.UINT32,
                )
                srt = pl.sort32(row, idx_init)
                srt = pl.mrgsort(srt, block_len=64)
                srt = pl.mrgsort(srt, block_len=256)
                pairs = srt[:, 0:ROUTER_SORT_PAD]
                top_idx = pl.gather(
                    pairs,
                    mask_pattern=pl.tile.MaskPattern.P1010,
                    output_dtype=pl.INT32,
                )
                topk_idx_tile = pl.create_tensor(
                    [1, ROUTER_TOPK_PAD], dtype=pl.INT32,
                )
                topk_idx_tile[:, :] = top_idx
                gather_all = pl.gather(
                    score_buf[tt : tt + 1, :],
                    dim=-1,
                    index=topk_idx_tile,
                )
                gather_valid = pl.set_validshape(gather_all, 1, TOPK)
                # A 1x1 FP32 reduction result has a 4-byte col-major
                # footprint, which ptoas rejects. Keep an aligned 8-row
                # workspace and scatter only its first row.
                topk_vals_work = pl.create_tensor(
                    [8, ROUTER_TOPK_PAD], dtype=pl.FP32,
                )
                for denom_row in pl.range(8):
                    topk_vals_work = pl.assemble(
                        topk_vals_work, gather_valid, [denom_row, 0],
                    )
                topk_vals_work_valid = pl.set_validshape(
                    topk_vals_work, 8, TOPK,
                )
                denom = pl.row_sum(topk_vals_work_valid)
                weights_work = pl.mul(
                    pl.row_expand_div(topk_vals_work_valid, denom),
                    ROUTER_SCALE,
                )
                for k in pl.range(TOPK):
                    pl.tile.write(
                        expert_indices_tile, [tt, k],
                        pl.read(topk_idx_tile, [0, k]),
                    )
                    pl.tile.write(
                        expert_weights_tile, [tt, k],
                        pl.read(weights_work, [0, k]),
                    )
            expert_indices = pl.store(
                expert_indices_tile, [0, 0], expert_indices,
            )
            expert_weights = pl.store(
                expert_weights_tile, [0, 0], expert_weights,
            )

        return expert_indices, expert_weights

    @pl.function(type=pl.FunctionType.Inline)
    def gate_step(
        self,
        resid: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
        norm_layer_idx: pl.Scalar[pl.INT32],
        inv_rms: pl.Tensor[[BATCH, 1], pl.FP32],
        gate_w: pl.Tensor[[N_EXPERTS, HIDDEN], pl.FP32],
        router_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
        expert_indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
        expert_weights: pl.Tensor[[BATCH, TOPK], pl.FP32],
        num_tokens: pl.Scalar[pl.INT32],
    ) -> tuple[
        pl.Tensor[[BATCH, TOPK], pl.INT32],
        pl.Tensor[[BATCH, TOPK], pl.FP32]
    ]:
        expert_indices, expert_weights = self._gate(
            resid, post_rms_weight, norm_layer_idx, inv_rms,
            gate_w, router_bias,
            expert_indices, expert_weights, num_tokens,
        )
        return expert_indices, expert_weights

    # ---------- Stage 2: MoE norm/quant input producer ----------
    @pl.function(type=pl.FunctionType.Inline)
    def _norm_quant_moe_input(
        self,
        resid: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
        norm_layer_idx: pl.Scalar[pl.INT32],
        post_norm_out: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        inv_rms_out: pl.Tensor[[BATCH, 1], pl.FP32],
        x_i8_out: pl.Tensor[[BATCH, HIDDEN], pl.INT8],
        x_scale_out: pl.Tensor[[BATCH, DISPATCH_SCALE_COLS], pl.FP32],
        num_tokens: pl.Scalar[pl.INT32],
    ) -> tuple[
        pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        pl.Tensor[[BATCH, 1], pl.FP32],
        pl.Tensor[[BATCH, HIDDEN], pl.INT8],
        pl.Tensor[[BATCH, DISPATCH_SCALE_COLS], pl.FP32],
    ]:
        """Deferred RMSNorm and INT8 producer.

        The first pass forms ``xg = resid * (gamma + 1)`` while reducing both
        ``sum(resid**2)`` and ``amax(xg)``.  The current backend recomputes xg
        chunk-wise in the second pass, which emits the BF16
        normalized value needed by the step3p5 BF16 shared expert and the
        mathematically equivalent INT8 payload ``quant(xg)``.  Its dequant
        scale carries the deferred positive RMS factor:
        ``inv_rms * amax(xg) / 127``.
        """
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > BATCH:
            active_tokens = pl.cast(BATCH, pl.INDEX)
        # Match the physical gate-tile contract: norm/quant/gate work on
        # complete 16-row cube tiles, while routing, packing and observable
        # outputs remain bounded by the logical active_tokens prefix.
        active_gate_tiles = (active_tokens + 15) // 16
        active_gate_tokens = active_gate_tiles * 16
        if active_gate_tokens > BATCH:
            active_gate_tokens = pl.cast(BATCH, pl.INDEX)

        for token_block in pl.spmd(
            MOE_NORM_BLOCKS,
            name_hint="norm_quant_moe_input",
        ):
            token0 = token_block * MOE_NORM_TOKEN_TILE
            sq_sum = pl.row_sum(
                pl.full(
                    [MOE_NORM_TOKEN_TILE, MOE_NORM_SCALAR_PAD],
                    dtype=pl.FP32,
                    value=0.0,
                ),
            )
            xg_amax = pl.row_max(
                pl.full(
                    [MOE_NORM_TOKEN_TILE, MOE_NORM_SCALAR_PAD],
                    dtype=pl.FP32,
                    value=1e-4,
                ),
            )
            for kb in pl.range(HIDDEN // K_CHUNK):
                k0 = kb * K_CHUNK
                raw = pl.cast(
                    pl.slice(
                        resid,
                        [MOE_NORM_TOKEN_TILE, K_CHUNK],
                        [token0, k0],
                    ),
                    target_type=pl.FP32,
                )
                gamma = pl.slice(
                    post_rms_weight,
                    [1, K_CHUNK],
                    [norm_layer_idx, k0],
                )
                xg = pl.col_expand_mul(
                    raw, pl.add(gamma, 1.0),
                )
                sq_sum = pl.add(
                    sq_sum,
                    pl.row_sum(pl.mul(raw, raw)),
                )
                xg_amax = pl.maximum(
                    xg_amax,
                    pl.row_max(pl.maximum(xg, pl.neg(xg))),
                )

            inv_rms = pl.recip(
                pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)),
            )
            inv_rms_out[
                token0 : token0 + MOE_NORM_TOKEN_TILE, 0:1
            ] = inv_rms
            quant_numerator = pl.row_max(
                pl.full(
                    [MOE_NORM_TOKEN_TILE, MOE_NORM_SCALAR_PAD],
                    dtype=pl.FP32,
                    value=127.0,
                ),
            )
            quant_mul = pl.div(quant_numerator, xg_amax)
            dequant_scale = pl.mul(
                inv_rms, pl.mul(xg_amax, 1.0 / 127.0),
            )
            x_scale_out[
                token0 : token0 + MOE_NORM_TOKEN_TILE, 0:1
            ] = dequant_scale

            for kb2 in pl.range(HIDDEN // K_CHUNK):
                k0 = kb2 * K_CHUNK
                raw = pl.cast(
                    pl.slice(
                        resid,
                        [MOE_NORM_TOKEN_TILE, K_CHUNK],
                        [token0, k0],
                    ),
                    target_type=pl.FP32,
                )
                gamma = pl.slice(
                    post_rms_weight,
                    [1, K_CHUNK],
                    [norm_layer_idx, k0],
                )
                # Current backend UB cannot retain the FP32 xg tile across
                # both passes, so recompute it in the emission pass.
                xg = pl.col_expand_mul(
                    raw, pl.add(gamma, 1.0),
                )
                normed = pl.row_expand_mul(xg, inv_rms)
                post_norm_out[
                    token0 : token0 + MOE_NORM_TOKEN_TILE,
                    k0 : k0 + K_CHUNK,
                ] = pl.cast(normed, target_type=pl.BF16)
                qi32 = pl.cast(
                    pl.row_expand_mul(xg, quant_mul),
                    target_type=pl.INT32,
                    mode="rint",
                )
                qf16 = pl.cast(
                    qi32, target_type=pl.FP16, mode="round",
                )
                x_i8_out[
                    token0 : token0 + MOE_NORM_TOKEN_TILE,
                    k0 : k0 + K_CHUNK,
                ] = pl.cast(
                    qf16, target_type=pl.INT8, mode="trunc",
                )
        return post_norm_out, inv_rms_out, x_i8_out, x_scale_out

    # ---------- Stage 2: replicated-input local expert packing ----------
    @pl.function(type=pl.FunctionType.Inline)
    def dispatch_step(  # noqa: PLR0913, PLR0915
        self,
        x: pl.Tensor[[BATCH, HIDDEN], pl.INT8],
        x_scale: pl.Tensor[[BATCH, DISPATCH_SCALE_COLS], pl.FP32],
        expert_indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
        expert_weights: pl.Tensor[[BATCH, TOPK], pl.FP32],
        local_routed_x_out: pl.Tensor[[local_recv_max, HIDDEN], pl.INT8],
        local_routed_x_scale_out: pl.Tensor[
            [1, local_recv_max], pl.FP32
        ],
        local_routed_weight_out: pl.Tensor[[local_recv_max], pl.FP32],
        local_route_row_out: pl.Tensor[
            [1, n_routes_per_rank], pl.INT32
        ],
        local_expert_count: pl.Tensor[[n_local_experts_pad], pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
    ) -> tuple[
        pl.Tensor[[local_recv_max, HIDDEN], pl.INT8],
        pl.Tensor[[1, local_recv_max], pl.FP32],
        pl.Tensor[[local_recv_max], pl.FP32],
        pl.Tensor[[1, n_routes_per_rank], pl.INT32],
        pl.Tensor[[n_local_experts_pad], pl.INT32],
        pl.Tensor[[local_route_plan_size], pl.INT32],
        pl.Scalar[pl.TASK_ID],
    ]:
        """Pack routes owned by this rank into fixed local-expert slabs."""
        # PTO tile.store requires a 2-D Tile. Keep the public rank-1 tensor ABI
        # and publish through aligned row views of the same backing storage.
        local_expert_count_view = pl.reshape(
            local_expert_count, [1, n_local_experts_pad],
        )
        local_routed_weight_out_view = pl.reshape(
            local_routed_weight_out, [1, local_recv_max],
        )
        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="local_route_map_init",
            allow_early_resolve=True,
        ) as route_map_init_tid:
            route_map = pl.tile.full(
                [1, n_routes_per_rank], dtype=pl.INT32, value=-1,
            )
            cursor = pl.array.create(n_local_experts, pl.INT32)
            for e in pl.range(n_local_experts):
                cursor[e] = 0

            active_tokens = pl.cast(num_tokens, pl.INDEX)
            if active_tokens < 0:
                active_tokens = pl.cast(0, pl.INDEX)
            if active_tokens > BATCH:
                active_tokens = pl.cast(BATCH, pl.INDEX)
            for t in pl.range(active_tokens):
                for k in pl.range(TOPK):
                    eid = pl.read(expert_indices, [t, k])
                    route_owner = eid // n_local_experts
                    route_local_e = eid - my_rank * n_local_experts
                    if route_owner == my_rank:
                        packed_count_i32 = cursor[route_local_e]
                        route_out_row_i32 = (
                            route_local_e * expert_recv_max
                            + packed_count_i32
                        )
                        route = t * TOPK + k
                        pl.tile.write(
                            route_map,
                            [0, route],
                            pl.cast(route_out_row_i32, pl.INT32),
                        )
                        cursor[route_local_e] = (
                            packed_count_i32 + pl.cast(1, pl.INT32)
                        )

            expert_count_tile = pl.tile.full(
                [1, n_local_experts_pad], dtype=pl.INT32, value=0,
            )
            for e in pl.range(n_local_experts):
                pl.tile.write(expert_count_tile, [0, e], cursor[e])
            local_expert_count_view = pl.store(
                expert_count_tile, [0, 0], local_expert_count_view,
            )
            local_route_row_out = pl.store(
                route_map, [0, 0], local_route_row_out,
            )

        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > BATCH:
            active_tokens = pl.cast(BATCH, pl.INDEX)
        with pl.spmd(
            n_local_experts,
            name_hint="local_route_pack",
            deps=[route_map_init_tid],
            allow_early_resolve=True,
        ) as local_pack_tid:
            local_e_idx = pl.tile.get_block_idx()
            pack_local_e_i32 = pl.cast(local_e_idx, pl.INT32)
            global_e = my_rank * n_local_experts + pack_local_e_i32
            pack_slab_begin = local_e_idx * expert_recv_max
            pack_slab_begin_i32 = pl.cast(pack_slab_begin, pl.INT32)
            pack_slab_end_i32 = pack_slab_begin_i32 + expert_recv_max
            scale_slab = pl.tile.full(
                [1, expert_recv_max], dtype=pl.FP32, value=0.0,
            )
            weight_slab = pl.tile.full(
                [1, expert_recv_max], dtype=pl.FP32, value=0.0,
            )
            for t in pl.range(active_tokens):
                for k in pl.range(TOPK):
                    eid = pl.read(expert_indices, [t, k])
                    if eid == global_e:
                        route = t * TOPK + k
                        pack_out_row_i32 = pl.read(
                            local_route_row_out, [0, route],
                        )
                        if pack_out_row_i32 >= pack_slab_begin_i32:
                            if pack_out_row_i32 < pack_slab_end_i32:
                                out_row = pl.cast(
                                    pack_out_row_i32, pl.INDEX,
                                )
                                pack_slab_row = pl.cast(
                                    pack_out_row_i32
                                    - pack_slab_begin_i32,
                                    pl.INDEX,
                                )
                                local_routed_x_out[
                                    out_row : out_row + 1, :
                                ] = x[t : t + 1, :]
                                pl.tile.write(
                                    scale_slab,
                                    [0, pack_slab_row],
                                    pl.read(x_scale, [t, 0]),
                                )
                                pl.tile.write(
                                    weight_slab,
                                    [0, pack_slab_row],
                                    pl.read(expert_weights, [t, k]),
                                )
            local_routed_x_scale_out = pl.store(
                scale_slab,
                [0, pack_slab_begin],
                local_routed_x_scale_out,
            )
            local_routed_weight_out_view = pl.store(
                weight_slab,
                [0, pack_slab_begin],
                local_routed_weight_out_view,
            )

        local_routed_weight_out = pl.reshape(
            local_routed_weight_out_view, [local_recv_max],
        )
        local_route_count = pl.create_tensor(
            [local_route_plan_size], dtype=pl.INT32,
        )
        local_route_count_view = pl.reshape(
            local_route_count, [1, local_route_plan_size],
        )
        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="local_route_plan",
            deps=[local_pack_tid],
            allow_early_resolve=True,
        ) as local_route_count_tid:
            total_count = pl.cast(0, pl.INT32)
            active_expert_count_i32 = pl.cast(0, pl.INT32)
            route_plan_tile = pl.tile.full(
                [1, local_route_plan_size], dtype=pl.INT32, value=0,
            )
            for e in pl.range(n_local_experts):
                expert_count_i32 = pl.read(local_expert_count_view, [0, e])
                if expert_count_i32 > 0:
                    pl.tile.write(
                        route_plan_tile,
                        [
                            0,
                            pl.cast(active_expert_count_i32, pl.INDEX)
                            + pl.cast(2, pl.INDEX)
                        ],
                        pl.cast(e, pl.INT32),
                    )
                    active_expert_count_i32 = (
                        active_expert_count_i32 + pl.cast(1, pl.INT32)
                    )
                total_count = total_count + expert_count_i32
            pl.tile.write(route_plan_tile, [0, 0], total_count)
            pl.tile.write(
                route_plan_tile, [0, 1], active_expert_count_i32,
            )
            local_route_count_view = pl.store(
                route_plan_tile, [0, 0], local_route_count_view,
            )

        local_route_count = pl.reshape(
            local_route_count_view, [local_route_plan_size],
        )
        # The rank-2 count view and this persistent rank-1 formal share storage.
        return (
            local_routed_x_out,
            local_routed_x_scale_out,
            local_routed_weight_out,
            local_route_row_out,
            local_expert_count,
            local_route_count,
            local_route_count_tid,
        )

    # ---------- Stage 3a: expert_routed (local 36 experts) ----------
    @pl.function(type=pl.FunctionType.Inline)
    def _expert_routed(  # noqa: PLR0913
        self,
        local_routed_x: pl.Tensor[[local_recv_max, HIDDEN], pl.INT8],
        local_routed_x_scale: pl.Tensor[[1, local_recv_max], pl.FP32],
        local_routed_weight: pl.Tensor[[local_recv_max], pl.FP32],
        local_expert_count: pl.Tensor[[n_local_experts_pad], pl.INT32],
        local_route_count: pl.Tensor[[local_route_plan_size], pl.INT32],
        local_route_count_tid: pl.Scalar[pl.TASK_ID],
        num_tokens: pl.Scalar[pl.INT32],
        w13: pl.Tensor[
            [
                n_local_experts,
                ROUTED_W13_N1,
                ROUTED_W13_M1,
                ROUTED_NZ_M0,
                ROUTED_NZ_N0,
            ],
            pl.INT8,
        ],
        w13_scale: pl.Tensor[[n_local_experts, 2 * inter], pl.FP32],
        w_down: pl.Tensor[
            [
                n_local_experts,
                ROUTED_W2_N1,
                ROUTED_W2_M1,
                ROUTED_NZ_M0,
                ROUTED_NZ_N0,
            ],
            pl.INT8,
        ],
        w_down_scale: pl.Tensor[[n_local_experts, HIDDEN], pl.FP32],
        local_routed_y: pl.Tensor[[local_recv_max, HIDDEN], pl.BF16],
    ):
        gate_up_i32 = pl.create_tensor(
            [local_recv_max, 2 * inter], dtype=pl.INT32,
        )
        h_bf16 = pl.create_tensor(
            [local_recv_max, inter], dtype=pl.BF16,
        )
        h_i8 = pl.create_tensor(
            [local_recv_max, inter], dtype=pl.INT8,
        )
        h_scale_dq_all = pl.create_tensor(
            [1, local_recv_max], dtype=pl.FP32,
        )
        (
            (
                local_route_count,
                gate_up_i32,
                h_bf16,
                h_i8,
                h_scale_dq_all,
            ),
            routed_fused_tid,
        ) = pl.spmd_submit(
            self.routed_nz_gmm1_swiglu_quant,
            local_route_count,
            gate_up_i32,
            local_expert_count,
            local_routed_x,
            w13,
            h_bf16,
            local_routed_x_scale,
            w13_scale,
            h_i8,
            h_scale_dq_all,
            core_num=ROUTED_FUSED_GRID_WORKERS,
            deps=[local_route_count_tid],
            predicate=(local_route_count[0] > 0),
            allow_early_resolve=True,
        )

        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > BATCH:
            active_tokens = pl.cast(BATCH, pl.INDEX)
        routed_workers = pl.cast(ROUTED_GRID_WORKERS, pl.INDEX)
        if active_tokens > 1:
            routed_workers = pl.cast(
                ROUTED_MULTIBATCH_GRID_WORKERS, pl.INDEX,
            )
        down_pipe_buffer = pl.create_tensor(
            [ROUTED_DOWN_PIPE_FLOATS_PER_BLOCK * ROUTED_GRID_WORKERS],
            dtype=pl.FP32,
        )
        (
            (local_routed_y, down_pipe_buffer),
            routed_down_tid,
        ) = pl.spmd_submit(
            self.routed_nz_down,
            local_route_count,
            local_routed_y,
            local_expert_count,
            h_i8,
            w_down,
            w_down_scale,
            h_scale_dq_all,
            local_routed_weight,
            down_pipe_buffer,
            routed_workers,
            core_num=routed_workers,
            deps=[local_route_count_tid, routed_fused_tid],
            predicate=(local_route_count[0] > 0),
            allow_early_resolve=True,
        )

        return local_routed_y, routed_down_tid


    @pl.function(type=pl.FunctionType.Inline)
    def expert_routed_step(
        self,
        local_routed_x: pl.Tensor[[local_recv_max, HIDDEN], pl.INT8],
        local_routed_x_scale: pl.Tensor[[1, local_recv_max], pl.FP32],
        local_routed_weight: pl.Tensor[[local_recv_max], pl.FP32],
        local_expert_count: pl.Tensor[[n_local_experts_pad], pl.INT32],
        local_route_count: pl.Tensor[[local_route_plan_size], pl.INT32],
        local_route_count_tid: pl.Scalar[pl.TASK_ID],
        num_tokens: pl.Scalar[pl.INT32],
        w13_r: pl.Tensor[
            [
                n_local_experts,
                ROUTED_W13_N1,
                ROUTED_W13_M1,
                ROUTED_NZ_M0,
                ROUTED_NZ_N0,
            ],
            pl.INT8,
        ],
        w13_r_scale: pl.Tensor[[n_local_experts, 2 * inter], pl.FP32],
        w_down_r: pl.Tensor[
            [
                n_local_experts,
                ROUTED_W2_N1,
                ROUTED_W2_M1,
                ROUTED_NZ_M0,
                ROUTED_NZ_N0,
            ],
            pl.INT8,
        ],
        w_down_r_scale: pl.Tensor[[n_local_experts, HIDDEN], pl.FP32],
        local_routed_y: pl.Tensor[[local_recv_max, HIDDEN], pl.BF16],
    ) -> tuple[
        pl.Tensor[[local_recv_max, HIDDEN], pl.BF16],
        pl.Scalar[pl.TASK_ID],
    ]:
        local_routed_y, routed_down_tid = self._expert_routed(
            local_routed_x,
            local_routed_x_scale,
            local_routed_weight,
            local_expert_count,
            local_route_count,
            local_route_count_tid,
            num_tokens,
            w13_r,
            w13_r_scale,
            w_down_r,
            w_down_r_scale,
            local_routed_y,
        )
        return local_routed_y, routed_down_tid

    # ---------- Stage 3b: expert_shared (5x32 narrow activation tiles) -
    @pl.function(type=pl.FunctionType.Inline)
    def _expert_shared_local(
        self,
        x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        w_gate: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
        w_up: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
        w_down: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
        sh_y_shard: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
        swiglu_limit: pl.Scalar[pl.FP32],
    ):
        # Gate and up are independent checkpoint-native [N, K] products.
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > BATCH:
            active_tokens = pl.cast(BATCH, pl.INDEX)
        active_shared_tiles = (
            active_tokens + SHARED_GATE_M_TILE - 1
        ) // SHARED_GATE_M_TILE
        shared_n_blocks = SHARED_GATE_UP_ACT_CHUNKS
        shared_mm_tasks = active_shared_tiles * shared_n_blocks
        sh_hidden = pl.create_tensor(
            [BATCH, sh_inter_local], dtype=pl.BF16, manual_dep=True,
        )
        sh_gate_acc = pl.create_tensor(
            [BATCH, sh_inter_local], dtype=pl.FP32, manual_dep=True,
        )
        sh_up_acc = pl.create_tensor(
            [BATCH, sh_inter_local], dtype=pl.FP32, manual_dep=True,
        )

        with pl.spmd(
            shared_mm_tasks,
            name_hint="sh_gate_mm",
            allow_early_resolve=True,
        ) as sh_gate_tid:
            task = pl.tile.get_block_idx()
            mb = task // shared_n_blocks
            chunk = task % shared_n_blocks
            m0 = mb * SHARED_GATE_M_TILE
            n0 = chunk * SHARED_SWIGLU_N_CHUNK
            x0 = pl.slice(
                x, [SHARED_GATE_M_TILE, SHARED_GATE_K_CHUNK], [m0, 0],
            )
            wg0 = pl.slice(
                w_gate,
                [SHARED_SWIGLU_N_CHUNK, SHARED_GATE_K_CHUNK],
                [n0, 0],
            )
            gate_acc = pl.matmul(
                x0, wg0, out_dtype=pl.FP32, b_trans=True,
            )
            for kb in pl.range(1, HIDDEN // SHARED_GATE_K_CHUNK):
                k0 = kb * SHARED_GATE_K_CHUNK
                xk = pl.slice(
                    x,
                    [SHARED_GATE_M_TILE, SHARED_GATE_K_CHUNK],
                    [m0, k0],
                )
                wgk = pl.slice(
                    w_gate,
                    [SHARED_SWIGLU_N_CHUNK, SHARED_GATE_K_CHUNK],
                    [n0, k0],
                )
                gate_acc = pl.matmul_acc(
                    gate_acc, xk, wgk, b_trans=True,
                )
            sh_gate_acc = pl.assemble(sh_gate_acc, gate_acc, [m0, n0])

        with pl.spmd(
            shared_mm_tasks,
            name_hint="sh_up_mm",
            allow_early_resolve=True,
        ) as sh_up_tid:
            task = pl.tile.get_block_idx()
            mb = task // shared_n_blocks
            chunk = task % shared_n_blocks
            m0 = mb * SHARED_GATE_M_TILE
            n0 = chunk * SHARED_SWIGLU_N_CHUNK
            x0 = pl.slice(
                x, [SHARED_GATE_M_TILE, SHARED_GATE_K_CHUNK], [m0, 0],
            )
            wu0 = pl.slice(
                w_up,
                [SHARED_SWIGLU_N_CHUNK, SHARED_GATE_K_CHUNK],
                [n0, 0],
            )
            up_acc = pl.matmul(
                x0, wu0, out_dtype=pl.FP32, b_trans=True,
            )
            for kb in pl.range(1, HIDDEN // SHARED_GATE_K_CHUNK):
                k0 = kb * SHARED_GATE_K_CHUNK
                xk = pl.slice(
                    x,
                    [SHARED_GATE_M_TILE, SHARED_GATE_K_CHUNK],
                    [m0, k0],
                )
                wuk = pl.slice(
                    w_up,
                    [SHARED_SWIGLU_N_CHUNK, SHARED_GATE_K_CHUNK],
                    [n0, k0],
                )
                up_acc = pl.matmul_acc(
                    up_acc, xk, wuk, b_trans=True,
                )
            sh_up_acc = pl.assemble(sh_up_acc, up_acc, [m0, n0])

        with pl.spmd(
            shared_mm_tasks,
            name_hint="sh_gate_up_act",
            deps=[sh_gate_tid, sh_up_tid],
            allow_early_resolve=True,
        ) as sh_gate_up_tid:
            task = pl.tile.get_block_idx()
            mb = task // shared_n_blocks
            chunk = task % shared_n_blocks
            m0 = mb * SHARED_GATE_M_TILE
            n0 = chunk * SHARED_SWIGLU_N_CHUNK
            gate_acc = pl.slice(
                sh_gate_acc,
                [SHARED_GATE_M_TILE, SHARED_SWIGLU_N_CHUNK],
                [m0, n0],
            )
            up_acc = pl.slice(
                sh_up_acc,
                [SHARED_GATE_M_TILE, SHARED_SWIGLU_N_CHUNK],
                [m0, n0],
            )
            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
            silu = pl.mul(gate_acc, sigmoid)
            if swiglu_limit > 0.0:
                silu_c = pl.minimum(silu, swiglu_limit)
                up_c = pl.maximum(
                    pl.minimum(up_acc, swiglu_limit),
                    -swiglu_limit,
                )
                gated = pl.mul(silu_c, up_c)
            else:
                gated = pl.mul(silu, up_acc)
            sh_hidden[
                m0 : m0 + SHARED_GATE_M_TILE,
                n0 : n0 + SHARED_SWIGLU_N_CHUNK,
            ] = pl.cast(gated, target_type=pl.BF16)

        active_down_tiles = (
            active_tokens + SHARED_DOWN_M_TILE - 1
        ) // SHARED_DOWN_M_TILE
        shared_down_tasks = active_down_tiles * SHARED_DOWN_WORKERS
        with pl.spmd(
            shared_down_tasks,
            name_hint="sh_down",
            deps=[sh_gate_up_tid],
            allow_early_resolve=True,
        ) as _sh_down_tid:
            task = pl.tile.get_block_idx()
            mb = task // SHARED_DOWN_WORKERS
            worker = task % SHARED_DOWN_WORKERS
            m0 = mb * SHARED_DOWN_M_TILE
            if active_tokens <= 1:
                if mb == 0 and worker == 0:
                    for db in pl.range(
                        HIDDEN // SHARED_DOWN_N_CHUNK,
                    ):
                        d0 = db * SHARED_DOWN_N_CHUNK
                        h0 = pl.slice(
                            sh_hidden,
                            [SHARED_DOWN_M_TILE, SHARED_SWIGLU_N_CHUNK],
                            [m0, 0],
                        )
                        wd0 = pl.slice(
                            w_down,
                            [
                                SHARED_SWIGLU_N_CHUNK,
                                SHARED_DOWN_N_CHUNK,
                            ],
                            [0, d0],
                        )
                        y_acc = pl.matmul(h0, wd0, out_dtype=pl.FP32)
                        for chunk in pl.range(
                            1, sh_inter_local // SHARED_SWIGLU_N_CHUNK,
                        ):
                            n0 = chunk * SHARED_SWIGLU_N_CHUNK
                            hk = pl.slice(
                                sh_hidden,
                                [
                                    SHARED_DOWN_M_TILE,
                                    SHARED_SWIGLU_N_CHUNK,
                                ],
                                [m0, n0],
                            )
                            wdk = pl.slice(
                                w_down,
                                [
                                    SHARED_SWIGLU_N_CHUNK,
                                    SHARED_DOWN_N_CHUNK,
                                ],
                                [n0, d0],
                            )
                            y_acc = pl.matmul_acc(y_acc, hk, wdk)
                        sh_y_shard = pl.assemble(
                            sh_y_shard,
                            pl.cast(y_acc, target_type=pl.BF16),
                            [m0, d0],
                        )
            else:
                for db in pl.range(
                    worker,
                    HIDDEN // SHARED_DOWN_N_CHUNK,
                    SHARED_DOWN_WORKERS,
                ):
                    d0 = db * SHARED_DOWN_N_CHUNK
                    h0 = pl.slice(
                        sh_hidden,
                        [SHARED_DOWN_M_TILE, SHARED_SWIGLU_N_CHUNK],
                        [m0, 0],
                    )
                    wd0 = pl.slice(
                        w_down,
                        [SHARED_SWIGLU_N_CHUNK, SHARED_DOWN_N_CHUNK],
                        [0, d0],
                    )
                    y_acc = pl.matmul(h0, wd0, out_dtype=pl.FP32)
                    for chunk in pl.range(
                        1, sh_inter_local // SHARED_SWIGLU_N_CHUNK,
                    ):
                        n0 = chunk * SHARED_SWIGLU_N_CHUNK
                        hk = pl.slice(
                            sh_hidden,
                            [SHARED_DOWN_M_TILE, SHARED_SWIGLU_N_CHUNK],
                            [m0, n0],
                        )
                        wdk = pl.slice(
                            w_down,
                            [SHARED_SWIGLU_N_CHUNK, SHARED_DOWN_N_CHUNK],
                            [n0, d0],
                        )
                        y_acc = pl.matmul_acc(y_acc, hk, wdk)
                    sh_y_shard = pl.assemble(
                        sh_y_shard,
                        pl.cast(y_acc, target_type=pl.BF16),
                        [m0, d0],
                    )

        return sh_y_shard

    @pl.function(type=pl.FunctionType.Inline)
    def expert_shared_step(  # noqa: PLR0913
        self,
        x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        w_gate_s: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
        w_up_s: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
        w_down_s: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
        sh_y: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
        return self._expert_shared_local(
            x,
            w_gate_s,
            w_up_s,
            w_down_s,
            sh_y,
            num_tokens,
            _SHARED_SWIGLU_LIMIT,
        )

    # ---------- Stage 4: local routed and shared partial combine ----------
    @pl.function(type=pl.FunctionType.Inline)
    def combine_step(
        self,
        local_routed_y: pl.Tensor[[local_recv_max, HIDDEN], pl.BF16],
        sh_y: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        moe_out: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        local_route_row: pl.Tensor[[1, n_routes_per_rank], pl.INT32],
        local_route_count_tid: pl.Scalar[pl.TASK_ID],
        routed_down_tid: pl.Scalar[pl.TASK_ID],
        num_tokens: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > BATCH:
            active_tokens = pl.cast(BATCH, pl.INDEX)
        with pl.spmd(
            BATCH,
            name_hint="local_combine_reduce",
            deps=[local_route_count_tid, routed_down_tid],
            allow_early_resolve=True,
        ) as _local_combine_tid:
            t = pl.tile.get_block_idx()
            acc = pl.cast(
                sh_y[t : t + 1, :], target_type=pl.FP32,
            )
            if t < active_tokens:
                for k in pl.range(TOPK):
                    route = t * TOPK + k
                    local_row_i32 = pl.read(local_route_row, [0, route])
                    if local_row_i32 >= 0:
                        if local_row_i32 < local_recv_max:
                            local_row = pl.cast(local_row_i32, pl.INDEX)
                            routed = pl.cast(
                                local_routed_y[local_row : local_row + 1, :],
                                target_type=pl.FP32,
                            )
                            acc = pl.add(acc, routed)
            moe_out[t : t + 1, :] = pl.cast(
                acc, target_type=pl.BF16, mode="rint",
            )
        return moe_out

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
        gate_w: pl.Tensor[[N_EXPERTS, HIDDEN], pl.FP32],
        router_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
        w13_r: pl.Tensor[
            [
                n_local_experts,
                ROUTED_W13_N1,
                ROUTED_W13_M1,
                ROUTED_NZ_M0,
                ROUTED_NZ_N0,
            ],
            pl.INT8,
        ],
        w13_r_scale: pl.Tensor[[n_local_experts, 2 * inter], pl.FP32],
        w_down_r: pl.Tensor[
            [
                n_local_experts,
                ROUTED_W2_N1,
                ROUTED_W2_M1,
                ROUTED_NZ_M0,
                ROUTED_NZ_N0,
            ],
            pl.INT8,
        ],
        w_down_r_scale: pl.Tensor[[n_local_experts, HIDDEN], pl.FP32],
        w_gate_s: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
        w_up_s: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
        w_down_s: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
        next_hidden_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
        resid_hold: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
        local_expert_count: pl.Out[
            pl.Tensor[[n_local_experts_pad], pl.INT32]
        ],
        attn_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
        attn_signal_window: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        sh_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
        sh_signal_window: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        norm_layer_idx: pl.Scalar[pl.INT32],
        attn_layer_idx: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
    ) -> tuple[
        pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        pl.Tensor[[n_local_experts_pad], pl.INT32],
    ]:
        # Write attention directly into the dedicated residual Out.  Avoid a
        # create_tensor -> reassign -> assemble handoff here: inside the
        # runtime MoE pl.range that pattern could read the pre-call zero SSA
        # version when stashing the residual, dropping the attention branch.
        resid_hold = attention_full_inline(
            current_hidden, input_rms_weight, wq, wk, wv,
            q_norm_weight, k_norm_weight,
            seq_lens, block_table, slot_mapping,
            rope_cos, rope_sin, k_cache, v_cache,
            wo, w_g, gate_r, resid_hold,
            norm_layer_idx, attn_layer_idx,
            num_tokens,
            attn_tmp_window, attn_signal_window, my_rank,
        )
        # ── B: V4-style deferred RMSNorm + INT8/scale producer. ──
        hidden_blocks = HIDDEN // K_CHUNK
        post_norm = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        moe_inv_rms = pl.create_tensor([BATCH, 1], dtype=pl.FP32)
        x_disp_i8 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.INT8)
        x_disp_scale = pl.create_tensor(
            [BATCH, DISPATCH_SCALE_COLS], dtype=pl.FP32,
        )
        post_norm, moe_inv_rms, x_disp_i8, x_disp_scale = (
            self._norm_quant_moe_input(
                resid_hold, post_rms_weight, norm_layer_idx,
                post_norm, moe_inv_rms, x_disp_i8, x_disp_scale, num_tokens,
            )
        )

        # ── C: replicated-input local-expert MoE. ──
        moe_out = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)

        expert_indices = pl.create_tensor([BATCH, TOPK], dtype=pl.INT32)
        expert_weights = pl.create_tensor([BATCH, TOPK], dtype=pl.FP32)
        expert_indices, expert_weights = self.gate_step(
            resid_hold, post_rms_weight, norm_layer_idx, moe_inv_rms,
            gate_w, router_bias, expert_indices, expert_weights, num_tokens,
        )

        sh_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        sh_y = self.expert_shared_step(
            post_norm, w_gate_s, w_up_s, w_down_s, sh_y, num_tokens,
        )

        local_routed_x = pl.create_tensor(
            [local_recv_max, HIDDEN], dtype=pl.INT8,
        )
        local_routed_x_scale = pl.create_tensor(
            [1, local_recv_max], dtype=pl.FP32,
        )
        local_routed_weight = pl.create_tensor(
            [local_recv_max], dtype=pl.FP32,
        )
        local_route_row = pl.create_tensor(
            [1, n_routes_per_rank], dtype=pl.INT32,
        )
        (
            local_routed_x,
            local_routed_x_scale,
            local_routed_weight,
            local_route_row,
            local_expert_count,
            local_route_count,
            local_route_count_tid,
        ) = self.dispatch_step(
            x_disp_i8, x_disp_scale, expert_indices, expert_weights,
            local_routed_x, local_routed_x_scale,
            local_routed_weight, local_route_row,
            local_expert_count,
            num_tokens, my_rank,
        )

        local_routed_y = pl.create_tensor(
            [local_recv_max, HIDDEN], dtype=pl.BF16,
        )
        local_routed_y, routed_down_tid = self.expert_routed_step(
            local_routed_x,
            local_routed_x_scale,
            local_routed_weight,
            local_expert_count,
            local_route_count, local_route_count_tid,
            num_tokens,
            w13_r, w13_r_scale,
            w_down_r, w_down_r_scale,
            local_routed_y,
        )

        moe_out = self.combine_step(
            local_routed_y, sh_y, moe_out, local_route_row,
            local_route_count_tid, routed_down_tid,
            num_tokens,
        )
        if TP_WORLD_SIZE > 1:
            moe_out = self.tp_all_reduce(
                moe_out, sh_tmp_window, sh_signal_window, num_tokens, my_rank,
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
                r = pl.cast(pl.slice(resid_hold, [BATCH, K_CHUNK], [0, k0]), target_type=pl.FP32)
                next_hidden_out = pl.assemble(
                    next_hidden_out,
                    pl.cast(pl.add(r, m), target_type=pl.BF16),
                    [0, k0],
                )
        return next_hidden_out, local_expert_count

    # ---- MoE-layer swa-attn FUSED with MoE-block: attention -> resid1
    # (local, intra-orch) -> post_norm -> local-owner MoE -> residual. Mirrors the
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
        gate_w: pl.Tensor[[N_EXPERTS, HIDDEN], pl.FP32],
        router_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
        w13_r: pl.Tensor[
            [
                n_local_experts,
                ROUTED_W13_N1,
                ROUTED_W13_M1,
                ROUTED_NZ_M0,
                ROUTED_NZ_N0,
            ],
            pl.INT8,
        ],
        w13_r_scale: pl.Tensor[[n_local_experts, 2 * inter], pl.FP32],
        w_down_r: pl.Tensor[
            [
                n_local_experts,
                ROUTED_W2_N1,
                ROUTED_W2_M1,
                ROUTED_NZ_M0,
                ROUTED_NZ_N0,
            ],
            pl.INT8,
        ],
        w_down_r_scale: pl.Tensor[[n_local_experts, HIDDEN], pl.FP32],
        w_gate_s: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
        w_up_s: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
        w_down_s: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
        next_hidden_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
        resid_hold: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
        local_expert_count: pl.Out[
            pl.Tensor[[n_local_experts_pad], pl.INT32]
        ],
        attn_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
        attn_signal_window: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        sh_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
        sh_signal_window: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        norm_layer_idx: pl.Scalar[pl.INT32],
        attn_layer_idx: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
    ) -> tuple[
        pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        pl.Tensor[[n_local_experts_pad], pl.INT32],
    ]:
        # See full_moe_chip_orch: write the post-attention hidden directly
        # into the dedicated residual Out so the loop body reads the post-call
        # SSA value rather than a zero-initialized local tensor version.
        resid_hold = attention_swa_inline(
            current_hidden, input_rms_weight, wq, wk, wv,
            q_norm_weight, k_norm_weight,
            seq_lens, block_table, slot_mapping,
            rope_cos, rope_sin, k_cache, v_cache,
            wo, w_g, gate_r, resid_hold,
            norm_layer_idx, attn_layer_idx,
            num_tokens,
            attn_tmp_window, attn_signal_window, my_rank,
        )
        # ── B: V4-style deferred RMSNorm + INT8/scale producer. ──
        hidden_blocks = HIDDEN // K_CHUNK
        post_norm = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        moe_inv_rms = pl.create_tensor([BATCH, 1], dtype=pl.FP32)
        x_disp_i8 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.INT8)
        x_disp_scale = pl.create_tensor(
            [BATCH, DISPATCH_SCALE_COLS], dtype=pl.FP32,
        )
        post_norm, moe_inv_rms, x_disp_i8, x_disp_scale = (
            self._norm_quant_moe_input(
                resid_hold, post_rms_weight, norm_layer_idx,
                post_norm, moe_inv_rms, x_disp_i8, x_disp_scale, num_tokens,
            )
        )

        # ── C: replicated-input local-expert MoE. ──
        moe_out = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)

        expert_indices = pl.create_tensor([BATCH, TOPK], dtype=pl.INT32)
        expert_weights = pl.create_tensor([BATCH, TOPK], dtype=pl.FP32)
        expert_indices, expert_weights = self.gate_step(
            resid_hold, post_rms_weight, norm_layer_idx, moe_inv_rms,
            gate_w, router_bias, expert_indices, expert_weights, num_tokens,
        )

        sh_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        sh_y = self.expert_shared_step(
            post_norm, w_gate_s, w_up_s, w_down_s, sh_y, num_tokens,
        )

        local_routed_x = pl.create_tensor(
            [local_recv_max, HIDDEN], dtype=pl.INT8,
        )
        local_routed_x_scale = pl.create_tensor(
            [1, local_recv_max], dtype=pl.FP32,
        )
        local_routed_weight = pl.create_tensor(
            [local_recv_max], dtype=pl.FP32,
        )
        local_route_row = pl.create_tensor(
            [1, n_routes_per_rank], dtype=pl.INT32,
        )
        (
            local_routed_x,
            local_routed_x_scale,
            local_routed_weight,
            local_route_row,
            local_expert_count,
            local_route_count,
            local_route_count_tid,
        ) = self.dispatch_step(
            x_disp_i8, x_disp_scale, expert_indices, expert_weights,
            local_routed_x, local_routed_x_scale,
            local_routed_weight, local_route_row,
            local_expert_count,
            num_tokens, my_rank,
        )

        local_routed_y = pl.create_tensor(
            [local_recv_max, HIDDEN], dtype=pl.BF16,
        )
        local_routed_y, routed_down_tid = self.expert_routed_step(
            local_routed_x,
            local_routed_x_scale,
            local_routed_weight,
            local_expert_count,
            local_route_count, local_route_count_tid,
            num_tokens,
            w13_r, w13_r_scale,
            w_down_r, w_down_r_scale,
            local_routed_y,
        )

        moe_out = self.combine_step(
            local_routed_y, sh_y, moe_out, local_route_row,
            local_route_count_tid, routed_down_tid,
            num_tokens,
        )
        if TP_WORLD_SIZE > 1:
            moe_out = self.tp_all_reduce(
                moe_out, sh_tmp_window, sh_signal_window, num_tokens, my_rank,
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
                r = pl.cast(pl.slice(resid_hold, [BATCH, K_CHUNK], [0, k0]), target_type=pl.FP32)
                next_hidden_out = pl.assemble(
                    next_hidden_out,
                    pl.cast(pl.add(r, m), target_type=pl.BF16),
                    [0, k0],
                )
        return next_hidden_out, local_expert_count
    @pl.function(type=pl.FunctionType.Inline)
    def _expert_routed_swiglu7(  # noqa: PLR0913
        self,
        local_routed_x: pl.Tensor[[local_recv_max, HIDDEN], pl.INT8],
        local_routed_x_scale: pl.Tensor[[1, local_recv_max], pl.FP32],
        local_routed_weight: pl.Tensor[[local_recv_max], pl.FP32],
        local_expert_count: pl.Tensor[[n_local_experts_pad], pl.INT32],
        local_route_count: pl.Tensor[[local_route_plan_size], pl.INT32],
        local_route_count_tid: pl.Scalar[pl.TASK_ID],
        num_tokens: pl.Scalar[pl.INT32],
        w13: pl.Tensor[
            [
                n_local_experts,
                ROUTED_W13_N1,
                ROUTED_W13_M1,
                ROUTED_NZ_M0,
                ROUTED_NZ_N0,
            ],
            pl.INT8,
        ],
        w13_scale: pl.Tensor[[n_local_experts, 2 * inter], pl.FP32],
        w_down: pl.Tensor[
            [
                n_local_experts,
                ROUTED_W2_N1,
                ROUTED_W2_M1,
                ROUTED_NZ_M0,
                ROUTED_NZ_N0,
            ],
            pl.INT8,
        ],
        w_down_scale: pl.Tensor[[n_local_experts, HIDDEN], pl.FP32],
        local_routed_y: pl.Tensor[[local_recv_max, HIDDEN], pl.BF16],
    ):
        gate_up_i32 = pl.create_tensor(
            [local_recv_max, 2 * inter], dtype=pl.INT32,
        )
        h_bf16 = pl.create_tensor(
            [local_recv_max, inter], dtype=pl.BF16,
        )
        h_i8 = pl.create_tensor(
            [local_recv_max, inter], dtype=pl.INT8,
        )
        h_scale_dq_all = pl.create_tensor(
            [1, local_recv_max], dtype=pl.FP32,
        )
        (
            (
                local_route_count,
                gate_up_i32,
                h_bf16,
                h_i8,
                h_scale_dq_all,
            ),
            routed_fused_tid,
        ) = pl.spmd_submit(
            self.routed_nz_gmm1_swiglu7_quant,
            local_route_count,
            gate_up_i32,
            local_expert_count,
            local_routed_x,
            w13,
            h_bf16,
            local_routed_x_scale,
            w13_scale,
            h_i8,
            h_scale_dq_all,
            core_num=ROUTED_FUSED_GRID_WORKERS,
            deps=[local_route_count_tid],
            predicate=(local_route_count[0] > 0),
            allow_early_resolve=True,
        )

        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > BATCH:
            active_tokens = pl.cast(BATCH, pl.INDEX)
        routed_workers = pl.cast(ROUTED_GRID_WORKERS, pl.INDEX)
        if active_tokens > 1:
            routed_workers = pl.cast(
                ROUTED_MULTIBATCH_GRID_WORKERS, pl.INDEX,
            )
        down_pipe_buffer = pl.create_tensor(
            [ROUTED_DOWN_PIPE_FLOATS_PER_BLOCK * ROUTED_GRID_WORKERS],
            dtype=pl.FP32,
        )
        (
            (local_routed_y, down_pipe_buffer),
            routed_down_tid,
        ) = pl.spmd_submit(
            self.routed_nz_down,
            local_route_count,
            local_routed_y,
            local_expert_count,
            h_i8,
            w_down,
            w_down_scale,
            h_scale_dq_all,
            local_routed_weight,
            down_pipe_buffer,
            routed_workers,
            core_num=routed_workers,
            deps=[local_route_count_tid, routed_fused_tid],
            predicate=(local_route_count[0] > 0),
            allow_early_resolve=True,
        )

        return local_routed_y, routed_down_tid

    @pl.function(type=pl.FunctionType.Inline)
    def expert_routed_step_swiglu7(
        self,
        local_routed_x: pl.Tensor[[local_recv_max, HIDDEN], pl.INT8],
        local_routed_x_scale: pl.Tensor[[1, local_recv_max], pl.FP32],
        local_routed_weight: pl.Tensor[[local_recv_max], pl.FP32],
        local_expert_count: pl.Tensor[[n_local_experts_pad], pl.INT32],
        local_route_count: pl.Tensor[[local_route_plan_size], pl.INT32],
        local_route_count_tid: pl.Scalar[pl.TASK_ID],
        num_tokens: pl.Scalar[pl.INT32],
        w13_r: pl.Tensor[
            [
                n_local_experts,
                ROUTED_W13_N1,
                ROUTED_W13_M1,
                ROUTED_NZ_M0,
                ROUTED_NZ_N0,
            ],
            pl.INT8,
        ],
        w13_r_scale: pl.Tensor[[n_local_experts, 2 * inter], pl.FP32],
        w_down_r: pl.Tensor[
            [
                n_local_experts,
                ROUTED_W2_N1,
                ROUTED_W2_M1,
                ROUTED_NZ_M0,
                ROUTED_NZ_N0,
            ],
            pl.INT8,
        ],
        w_down_r_scale: pl.Tensor[[n_local_experts, HIDDEN], pl.FP32],
        local_routed_y: pl.Tensor[[local_recv_max, HIDDEN], pl.BF16],
    ) -> tuple[
        pl.Tensor[[local_recv_max, HIDDEN], pl.BF16],
        pl.Scalar[pl.TASK_ID],
    ]:
        local_routed_y, routed_down_tid = self._expert_routed_swiglu7(
            local_routed_x,
            local_routed_x_scale,
            local_routed_weight,
            local_expert_count,
            local_route_count, local_route_count_tid,
            num_tokens,
            w13_r, w13_r_scale,
            w_down_r, w_down_r_scale,
            local_routed_y,
        )
        return local_routed_y, routed_down_tid

    @pl.function(type=pl.FunctionType.Inline)
    def _expert_shared_local_swiglu16(
        self,
        x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        w_gate: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
        w_up: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
        w_down: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
        sh_y_shard: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        return self._expert_shared_local(
            x,
            w_gate,
            w_up,
            w_down,
            sh_y_shard,
            num_tokens,
            _SHARED_SWIGLU16_LIMIT,
        )

    @pl.function(type=pl.FunctionType.Inline)
    def expert_shared_step_swiglu16(  # noqa: PLR0913
        self,
        x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        w_gate_s: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
        w_up_s: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
        w_down_s: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
        sh_y: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
        return self._expert_shared_local_swiglu16(
            x, w_gate_s, w_up_s, w_down_s, sh_y, num_tokens,
        )

    @pl.function(
        type=pl.FunctionType.Orchestration,
        attrs={"inline_orchestration": True},
    )
    def full_moe_chip_orch_swiglu7_swiglu16(  # noqa: PLR0913, PLR0915
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
        gate_w: pl.Tensor[[N_EXPERTS, HIDDEN], pl.FP32],
        router_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
        w13_r: pl.Tensor[
            [
                n_local_experts,
                ROUTED_W13_N1,
                ROUTED_W13_M1,
                ROUTED_NZ_M0,
                ROUTED_NZ_N0,
            ],
            pl.INT8,
        ],
        w13_r_scale: pl.Tensor[[n_local_experts, 2 * inter], pl.FP32],
        w_down_r: pl.Tensor[
            [
                n_local_experts,
                ROUTED_W2_N1,
                ROUTED_W2_M1,
                ROUTED_NZ_M0,
                ROUTED_NZ_N0,
            ],
            pl.INT8,
        ],
        w_down_r_scale: pl.Tensor[[n_local_experts, HIDDEN], pl.FP32],
        w_gate_s: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
        w_up_s: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
        w_down_s: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
        next_hidden_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
        resid_hold: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
        attn_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
        attn_signal_window: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        sh_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
        sh_signal_window: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        norm_layer_idx: pl.Scalar[pl.INT32],
        attn_layer_idx: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
        # Keep the specialized L44 path on the same direct-Out residual
        # discipline as the loop-form MoE layers.
        resid_hold = attention_full_inline(
            current_hidden, input_rms_weight, wq, wk, wv,
            q_norm_weight, k_norm_weight,
            seq_lens, block_table, slot_mapping,
            rope_cos, rope_sin, k_cache, v_cache,
            wo, w_g, gate_r, resid_hold,
            norm_layer_idx, attn_layer_idx,
            num_tokens,
            attn_tmp_window, attn_signal_window, my_rank,
        )
        # ── B: V4-style deferred RMSNorm + INT8/scale producer. ──
        hidden_blocks = HIDDEN // K_CHUNK
        post_norm = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        moe_inv_rms = pl.create_tensor([BATCH, 1], dtype=pl.FP32)
        x_disp_i8 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.INT8)
        x_disp_scale = pl.create_tensor(
            [BATCH, DISPATCH_SCALE_COLS], dtype=pl.FP32,
        )
        post_norm, moe_inv_rms, x_disp_i8, x_disp_scale = (
            self._norm_quant_moe_input(
                resid_hold, post_rms_weight, norm_layer_idx,
                post_norm, moe_inv_rms, x_disp_i8, x_disp_scale, num_tokens,
            )
        )

        # ── C: replicated-input local-expert MoE. ──
        moe_out = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)

        expert_indices = pl.create_tensor([BATCH, TOPK], dtype=pl.INT32)
        expert_weights = pl.create_tensor([BATCH, TOPK], dtype=pl.FP32)
        expert_indices, expert_weights = self.gate_step(
            resid_hold, post_rms_weight, norm_layer_idx, moe_inv_rms,
            gate_w, router_bias, expert_indices, expert_weights, num_tokens,
        )

        sh_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        sh_y = self.expert_shared_step_swiglu16(
            post_norm, w_gate_s, w_up_s, w_down_s, sh_y, num_tokens,
        )

        local_routed_x = pl.create_tensor(
            [local_recv_max, HIDDEN], dtype=pl.INT8,
        )
        local_routed_x_scale = pl.create_tensor(
            [1, local_recv_max], dtype=pl.FP32,
        )
        local_routed_weight = pl.create_tensor(
            [local_recv_max], dtype=pl.FP32,
        )
        local_route_row = pl.create_tensor(
            [1, n_routes_per_rank], dtype=pl.INT32,
        )
        local_expert_count = pl.create_tensor(
            [n_local_experts_pad], dtype=pl.INT32,
        )
        (
            local_routed_x,
            local_routed_x_scale,
            local_routed_weight,
            local_route_row,
            local_expert_count,
            local_route_count,
            local_route_count_tid,
        ) = self.dispatch_step(
            x_disp_i8, x_disp_scale, expert_indices, expert_weights,
            local_routed_x, local_routed_x_scale,
            local_routed_weight, local_route_row,
            local_expert_count,
            num_tokens, my_rank,
        )

        local_routed_y = pl.create_tensor(
            [local_recv_max, HIDDEN], dtype=pl.BF16,
        )
        local_routed_y, routed_down_tid = self.expert_routed_step_swiglu7(
            local_routed_x,
            local_routed_x_scale,
            local_routed_weight,
            local_expert_count,
            local_route_count, local_route_count_tid,
            num_tokens,
            w13_r, w13_r_scale,
            w_down_r, w_down_r_scale,
            local_routed_y,
        )

        moe_out = self.combine_step(
            local_routed_y, sh_y, moe_out, local_route_row,
            local_route_count_tid, routed_down_tid,
            num_tokens,
        )
        if TP_WORLD_SIZE > 1:
            moe_out = self.tp_all_reduce(
                moe_out, sh_tmp_window, sh_signal_window, num_tokens, my_rank,
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
                r = pl.cast(pl.slice(resid_hold, [BATCH, K_CHUNK], [0, k0]), target_type=pl.FP32)
                next_hidden_out = pl.assemble(
                    next_hidden_out,
                    pl.cast(pl.add(r, m), target_type=pl.BF16),
                    [0, k0],
                )
        return next_hidden_out

    @pl.function(
        type=pl.FunctionType.Orchestration,
        attrs={"inline_orchestration": True},
    )
    def swa_moe_chip_orch_swiglu7_silu(  # noqa: PLR0913, PLR0915
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
        gate_w: pl.Tensor[[N_EXPERTS, HIDDEN], pl.FP32],
        router_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
        w13_r: pl.Tensor[
            [
                n_local_experts,
                ROUTED_W13_N1,
                ROUTED_W13_M1,
                ROUTED_NZ_M0,
                ROUTED_NZ_N0,
            ],
            pl.INT8,
        ],
        w13_r_scale: pl.Tensor[[n_local_experts, 2 * inter], pl.FP32],
        w_down_r: pl.Tensor[
            [
                n_local_experts,
                ROUTED_W2_N1,
                ROUTED_W2_M1,
                ROUTED_NZ_M0,
                ROUTED_NZ_N0,
            ],
            pl.INT8,
        ],
        w_down_r_scale: pl.Tensor[[n_local_experts, HIDDEN], pl.FP32],
        w_gate_s: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
        w_up_s: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
        w_down_s: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
        next_hidden_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
        resid_hold: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
        attn_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
        attn_signal_window: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        sh_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
        sh_signal_window: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        norm_layer_idx: pl.Scalar[pl.INT32],
        attn_layer_idx: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
        # Keep the specialized L43 path on the same direct-Out residual
        # discipline as the loop-form MoE layers.
        resid_hold = attention_swa_inline(
            current_hidden, input_rms_weight, wq, wk, wv,
            q_norm_weight, k_norm_weight,
            seq_lens, block_table, slot_mapping,
            rope_cos, rope_sin, k_cache, v_cache,
            wo, w_g, gate_r, resid_hold,
            norm_layer_idx, attn_layer_idx,
            num_tokens,
            attn_tmp_window, attn_signal_window, my_rank,
        )
        # ── B: V4-style deferred RMSNorm + INT8/scale producer. ──
        hidden_blocks = HIDDEN // K_CHUNK
        post_norm = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        moe_inv_rms = pl.create_tensor([BATCH, 1], dtype=pl.FP32)
        x_disp_i8 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.INT8)
        x_disp_scale = pl.create_tensor(
            [BATCH, DISPATCH_SCALE_COLS], dtype=pl.FP32,
        )
        post_norm, moe_inv_rms, x_disp_i8, x_disp_scale = (
            self._norm_quant_moe_input(
                resid_hold, post_rms_weight, norm_layer_idx,
                post_norm, moe_inv_rms, x_disp_i8, x_disp_scale, num_tokens,
            )
        )

        # ── C: replicated-input local-expert MoE. ──
        moe_out = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)

        expert_indices = pl.create_tensor([BATCH, TOPK], dtype=pl.INT32)
        expert_weights = pl.create_tensor([BATCH, TOPK], dtype=pl.FP32)
        expert_indices, expert_weights = self.gate_step(
            resid_hold, post_rms_weight, norm_layer_idx, moe_inv_rms,
            gate_w, router_bias, expert_indices, expert_weights, num_tokens,
        )

        sh_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        sh_y = self.expert_shared_step(
            post_norm, w_gate_s, w_up_s, w_down_s, sh_y, num_tokens,
        )

        local_routed_x = pl.create_tensor(
            [local_recv_max, HIDDEN], dtype=pl.INT8,
        )
        local_routed_x_scale = pl.create_tensor(
            [1, local_recv_max], dtype=pl.FP32,
        )
        local_routed_weight = pl.create_tensor(
            [local_recv_max], dtype=pl.FP32,
        )
        local_route_row = pl.create_tensor(
            [1, n_routes_per_rank], dtype=pl.INT32,
        )
        local_expert_count = pl.create_tensor(
            [n_local_experts_pad], dtype=pl.INT32,
        )
        (
            local_routed_x,
            local_routed_x_scale,
            local_routed_weight,
            local_route_row,
            local_expert_count,
            local_route_count,
            local_route_count_tid,
        ) = self.dispatch_step(
            x_disp_i8, x_disp_scale, expert_indices, expert_weights,
            local_routed_x, local_routed_x_scale,
            local_routed_weight, local_route_row,
            local_expert_count,
            num_tokens, my_rank,
        )

        local_routed_y = pl.create_tensor(
            [local_recv_max, HIDDEN], dtype=pl.BF16,
        )
        local_routed_y, routed_down_tid = self.expert_routed_step_swiglu7(
            local_routed_x,
            local_routed_x_scale,
            local_routed_weight,
            local_expert_count,
            local_route_count, local_route_count_tid,
            num_tokens,
            w13_r, w13_r_scale,
            w_down_r, w_down_r_scale,
            local_routed_y,
        )

        moe_out = self.combine_step(
            local_routed_y, sh_y, moe_out, local_route_row,
            local_route_count_tid, routed_down_tid,
            num_tokens,
        )
        if TP_WORLD_SIZE > 1:
            moe_out = self.tp_all_reduce(
                moe_out, sh_tmp_window, sh_signal_window, num_tokens, my_rank,
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
                r = pl.cast(pl.slice(resid_hold, [BATCH, K_CHUNK], [0, k0]), target_type=pl.FP32)
                next_hidden_out = pl.assemble(
                    next_hidden_out,
                    pl.cast(pl.add(r, m), target_type=pl.BF16),
                    [0, k0],
                )
        return next_hidden_out

    @pl.function(type=pl.FunctionType.Orchestration)
    def whole_chip_orch(  # noqa: PLR0913, PLR0915
        self,
        current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        input_rms: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
        post_rms: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
        q_norm: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
        k_norm: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
        full_wq: pl.Tensor[[N_FULL_ATTN_LAYERS * HIDDEN, hidden_q_full], pl.BF16],
        full_wk: pl.Tensor[[N_FULL_ATTN_LAYERS * HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        full_wv: pl.Tensor[[N_FULL_ATTN_LAYERS * HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        full_wo: pl.Tensor[[N_FULL_ATTN_LAYERS * hidden_q_full, HIDDEN], pl.BF16],
        full_w_g: pl.Tensor[[N_FULL_ATTN_LAYERS * HIDDEN, nh_full_pad], pl.BF16],
        full_gate_r: pl.Tensor[[N_FULL_ATTN_LAYERS * nh_full_pad, hidden_q_full], pl.BF16],
        swa_wq: pl.Tensor[[N_SWA_ATTN_LAYERS * HIDDEN, hidden_q_swa], pl.BF16],
        swa_wk: pl.Tensor[[N_SWA_ATTN_LAYERS * HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        swa_wv: pl.Tensor[[N_SWA_ATTN_LAYERS * HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        swa_wo: pl.Tensor[[N_SWA_ATTN_LAYERS * hidden_q_swa, HIDDEN], pl.BF16],
        swa_w_g: pl.Tensor[[N_SWA_ATTN_LAYERS * HIDDEN, nh_swa_pad], pl.BF16],
        swa_gate_r: pl.Tensor[[N_SWA_ATTN_LAYERS * nh_swa_pad, hidden_q_swa], pl.BF16],
        dense_w_gate: pl.Tensor[[NUM_DENSE_LAYERS * HIDDEN, INTER_LOCAL], pl.BF16],
        dense_w_up: pl.Tensor[[NUM_DENSE_LAYERS * HIDDEN, INTER_LOCAL], pl.BF16],
        dense_w_down: pl.Tensor[[NUM_DENSE_LAYERS * INTER_LOCAL, HIDDEN], pl.BF16],
        # ── MoE silu_silu L3-L42 (40 layers) stacked weights ──
        # full-attn MoE layers (10): physical {4,8,12,16,20,24,28,32,36,40}
        moe_full_wq: pl.Tensor[[NUM_FULL_MOE_LAYERS * HIDDEN, hidden_q_full], pl.BF16],
        moe_full_wk: pl.Tensor[[NUM_FULL_MOE_LAYERS * HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        moe_full_wv: pl.Tensor[[NUM_FULL_MOE_LAYERS * HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        moe_full_wo: pl.Tensor[[NUM_FULL_MOE_LAYERS * hidden_q_full, HIDDEN], pl.BF16],
        moe_full_w_g: pl.Tensor[[NUM_FULL_MOE_LAYERS * HIDDEN, nh_full_pad], pl.BF16],
        moe_full_gate_r: pl.Tensor[[NUM_FULL_MOE_LAYERS * nh_full_pad, hidden_q_full], pl.BF16],
        # swa-attn MoE layers (30): physical {3,5,6,7,9,..,42}
        moe_swa_wq: pl.Tensor[[NUM_SWA_MOE_LAYERS * HIDDEN, hidden_q_swa], pl.BF16],
        moe_swa_wk: pl.Tensor[[NUM_SWA_MOE_LAYERS * HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        moe_swa_wv: pl.Tensor[[NUM_SWA_MOE_LAYERS * HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        moe_swa_wo: pl.Tensor[[NUM_SWA_MOE_LAYERS * hidden_q_swa, HIDDEN], pl.BF16],
        moe_swa_w_g: pl.Tensor[[NUM_SWA_MOE_LAYERS * HIDDEN, nh_swa_pad], pl.BF16],
        moe_swa_gate_r: pl.Tensor[[NUM_SWA_MOE_LAYERS * nh_swa_pad, hidden_q_swa], pl.BF16],
        # MoE router weights are replicated, routed weights are owner-local,
        # and shared-expert weights are TP-sliced. All three are stacked for
        # the 42 MoE layers.
        moe_gate_w: pl.Tensor[[NUM_MOE_LAYERS_TOTAL * N_EXPERTS, HIDDEN], pl.FP32],
        moe_router_bias: pl.Tensor[[NUM_MOE_LAYERS_TOTAL * N_EXPERTS], pl.FP32],
        moe_w13_r: pl.Tensor[[NUM_MOE_LAYERS_TOTAL * n_local_experts * ROUTED_W13_N1, ROUTED_W13_M1, ROUTED_NZ_M0, ROUTED_NZ_N0], pl.INT8],
        moe_w13_r_scale: pl.Tensor[[NUM_MOE_LAYERS_TOTAL * n_local_experts, 2 * inter], pl.FP32],
        moe_w_down_r: pl.Tensor[[NUM_MOE_LAYERS_TOTAL * n_local_experts * ROUTED_W2_N1, ROUTED_W2_M1, ROUTED_NZ_M0, ROUTED_NZ_N0], pl.INT8],
        moe_w_down_r_scale: pl.Tensor[[NUM_MOE_LAYERS_TOTAL * n_local_experts, HIDDEN], pl.FP32],
        moe_w_gate_s: pl.Tensor[[NUM_MOE_LAYERS_TOTAL * sh_inter_local, HIDDEN], pl.BF16],
        moe_w_up_s: pl.Tensor[[NUM_MOE_LAYERS_TOTAL * sh_inter_local, HIDDEN], pl.BF16],
        moe_w_down_s: pl.Tensor[[NUM_MOE_LAYERS_TOTAL * sh_inter_local, HIDDEN], pl.BF16],
        seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
        block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
        slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
        rope_cos_full: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],
        rope_sin_full: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],
        rope_cos_swa: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],
        rope_sin_swa: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],
        k_cache: pl.InOut[pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16]],
        v_cache: pl.InOut[pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16]],
        next_hidden_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
        dense_attn_tmp_stack: pld.DistributedTensor[
            [NUM_DENSE_LAYERS * BATCH, HIDDEN], pl.BF16
        ],
        dense_attn_signal_stack: pld.DistributedTensor[
            [NUM_DENSE_LAYERS * COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        dense_mlp_tmp_stack: pld.DistributedTensor[
            [NUM_DENSE_LAYERS * BATCH, HIDDEN], pl.BF16
        ],
        dense_mlp_signal_stack: pld.DistributedTensor[
            [NUM_DENSE_LAYERS * COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        # ── MoE TP all-reduce windows ──
        moe_attn_tmp_stack: pld.DistributedTensor[
            [NUM_MOE_LAYERS_TOTAL * BATCH, HIDDEN], pl.BF16
        ],
        moe_attn_signal_stack: pld.DistributedTensor[
            [NUM_MOE_LAYERS_TOTAL * COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        moe_sh_tmp_stack: pld.DistributedTensor[
            [NUM_MOE_LAYERS_TOTAL * BATCH, HIDDEN], pl.BF16
        ],
        moe_sh_signal_stack: pld.DistributedTensor[
            [NUM_MOE_LAYERS_TOTAL * COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        num_tokens_per_owner: pl.Tensor[[NUM_TOKENS_STORAGE_I32], pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
    ):
        # G1 packed-global batch contract: the holder writes the same active
        # row count to every TP owner and replicates valid rows/metadata.  The
        # owner vector is retained for the runtime ABI and diagnostics; taking
        # its max does not make owner-local heterogeneous rows valid.  All
        # active rows [0:num_tokens) must therefore be initialized on every
        # rank before attention/KV/MoE execution.
        num_tokens = pl.cast(0, pl.INT32)
        for owner_rank in pl.range(n_ranks):
            num_tokens = pl.max(
                num_tokens,
                pl.read(num_tokens_per_owner, [owner_rank]),
            )
        if num_tokens < 0:
            num_tokens = pl.cast(0, pl.INT32)
        if num_tokens > BATCH:
            num_tokens = pl.cast(BATCH, pl.INT32)

        # ── L0: full-attn dense layer (distinct shape, emitted pre-loop). ──
        h_layer_0 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        h_layer_0 = self.full_chip_orch(
            current_hidden,
            input_rms,
            pl.slice(full_wq, [HIDDEN, hidden_q_full], [0, 0]),
            pl.slice(full_wk, [HIDDEN, KV_HIDDEN_LOCAL_R], [0, 0]),
            pl.slice(full_wv, [HIDDEN, KV_HIDDEN_LOCAL_R], [0, 0]),
            q_norm,
            k_norm,
            seq_lens,
            block_table,
            slot_mapping,
            rope_cos_full,
            rope_sin_full,
            k_cache,
            v_cache,
            pl.slice(full_wo, [hidden_q_full, HIDDEN], [0, 0]),
            pl.slice(full_w_g, [HIDDEN, nh_full_pad], [0, 0]),
            pl.slice(full_gate_r, [nh_full_pad, hidden_q_full], [0, 0]),
            post_rms,
            pl.slice(dense_w_gate, [HIDDEN, INTER_LOCAL], [0, 0]),
            pl.slice(dense_w_up, [HIDDEN, INTER_LOCAL], [0, 0]),
            pl.slice(dense_w_down, [INTER_LOCAL, HIDDEN], [0, 0]),
            h_layer_0,
            pl.slice(dense_attn_tmp_stack, [BATCH, HIDDEN], [0, 0]),
            pl.slice(dense_attn_signal_stack, [COMM_SIGNAL_STRIDE_I32, 1], [0, 0]),
            pl.slice(dense_mlp_tmp_stack, [BATCH, HIDDEN], [0, 0]),
            pl.slice(dense_mlp_signal_stack, [COMM_SIGNAL_STRIDE_I32, 1], [0, 0]),
            0,
            0,
            0,
            num_tokens,
            my_rank,
        )

        # ── L1/L2: swa-attn dense layers inside a runtime pl.range loop. ──
        # layer_idx ∈ {0, 1} maps to physical layers {1, 2}: swa weight offset
        # = layer_idx, dense MLP offset = layer_idx + 1, norm/mlp idx = +1.
        prev_hidden = h_layer_0
        for layer_idx in pl.range(NUM_SWA_DENSE_LAYERS):
            swa_w_off = layer_idx * HIDDEN
            swa_wo_off = layer_idx * hidden_q_swa
            swa_gate_r_off = layer_idx * nh_swa_pad
            dense_w_off = (layer_idx + 1) * HIDDEN
            dense_down_off = (layer_idx + 1) * INTER_LOCAL
            win_off = (layer_idx + 1) * BATCH
            sig_off = (layer_idx + 1) * COMM_SIGNAL_STRIDE_I32
            norm_idx = layer_idx + 1
            dump_phys = layer_idx + 1
            h_next = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            h_next = self.swa_chip_orch(
                prev_hidden,
                input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [swa_w_off, 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL_R], [swa_w_off, 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL_R], [swa_w_off, 0]),
                q_norm,
                k_norm,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos_swa,
                rope_sin_swa,
                k_cache,
                v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [swa_wo_off, 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [swa_w_off, 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [swa_gate_r_off, 0]),
                post_rms,
                pl.slice(dense_w_gate, [HIDDEN, INTER_LOCAL], [dense_w_off, 0]),
                pl.slice(dense_w_up, [HIDDEN, INTER_LOCAL], [dense_w_off, 0]),
                pl.slice(dense_w_down, [INTER_LOCAL, HIDDEN], [dense_down_off, 0]),
                h_next,
                pl.slice(dense_attn_tmp_stack, [BATCH, HIDDEN], [win_off, 0]),
                pl.slice(dense_attn_signal_stack, [COMM_SIGNAL_STRIDE_I32, 1], [sig_off, 0]),
                pl.slice(dense_mlp_tmp_stack, [BATCH, HIDDEN], [win_off, 0]),
                pl.slice(dense_mlp_signal_stack, [COMM_SIGNAL_STRIDE_I32, 1], [sig_off, 0]),
                norm_idx,
                0,
                0,
                num_tokens,
                my_rank,
            )
            prev_hidden = h_next

        # ── L3-L42: MoE silu_silu 40 layers inside a runtime pl.range loop. ──
        # layer_idx ∈ 0..39 maps to physical layer 3 + layer_idx. The real
        # step3p5 layer table interleaves full_moe / swa_moe in a 3:1 pattern:
        # physical {4,8,12,16,20,24,28,32,36,40} = full_moe (10 layers), the
        # remaining 30 = swa_moe. In layer_idx space full_moe sits at
        # layer_idx % 4 == 1. DeepSeek decode_layer.py:201-226 proves pypto
        # supports runtime `if`/`elif` on a loop scalar dispatching to different
        # chip_orch methods — we mirror that here.
        for layer_idx in pl.range(NUM_MOE_LAYERS):
            phys_layer = layer_idx + 3
            norm_layer_idx = pl.cast(phys_layer, pl.INT32)
            # MoE weight/window offset = layer_idx * slot (0-indexed into the
            # 40-layer stacks).
            moe_gate_off = layer_idx * N_EXPERTS
            moe_bias_off = layer_idx * N_EXPERTS
            moe_sh_gate_off = layer_idx * sh_inter_local
            moe_r_off = layer_idx * (n_local_experts * ROUTED_W13_N1)
            moe_r_down_off = layer_idx * (n_local_experts * ROUTED_W2_N1)
            moe_r_scale_off = layer_idx * n_local_experts
            moe_sh_down_off = layer_idx * sh_inter_local
            moe_win_off = layer_idx * BATCH
            moe_sig_off = layer_idx * COMM_SIGNAL_STRIDE_I32
            h_moe = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            resid_hold_moe = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            local_expert_count_moe = pl.create_tensor(
                [n_local_experts_pad], dtype=pl.INT32,
            )
            # full_moe at layer_idx % 4 == 1 (physical 4,8,12,...). full_idx =
            # (layer_idx - 1) // 4 maps {1,5,9,...} -> {0,1,2,...,9}.
            # swa_idx = layer_idx - full_count_so_far; computed inside branch.
            if layer_idx % 4 == 1:
                full_idx = (layer_idx - 1) // 4
                fa_w_off = full_idx * HIDDEN
                fa_wo_off = full_idx * hidden_q_full
                fa_gate_r_off = full_idx * nh_full_pad
                h_moe, local_expert_count_moe = self.full_moe_chip_orch(
                    prev_hidden,
                    input_rms,
                    pl.slice(moe_full_wq, [HIDDEN, hidden_q_full], [fa_w_off, 0]),
                    pl.slice(moe_full_wk, [HIDDEN, KV_HIDDEN_LOCAL_R], [fa_w_off, 0]),
                    pl.slice(moe_full_wv, [HIDDEN, KV_HIDDEN_LOCAL_R], [fa_w_off, 0]),
                    q_norm,
                    k_norm,
                    seq_lens,
                    block_table,
                    slot_mapping,
                    rope_cos_full,
                    rope_sin_full,
                    k_cache,
                    v_cache,
                    pl.slice(moe_full_wo, [hidden_q_full, HIDDEN], [fa_wo_off, 0]),
                    pl.slice(moe_full_w_g, [HIDDEN, nh_full_pad], [fa_w_off, 0]),
                    pl.slice(moe_full_gate_r, [nh_full_pad, hidden_q_full], [fa_gate_r_off, 0]),
                    post_rms,
                    pl.slice(moe_gate_w, [N_EXPERTS, HIDDEN], [moe_gate_off, 0]),
                    pl.slice(moe_router_bias, [N_EXPERTS], [moe_bias_off]),
                    pl.reshape(
                        pl.slice(moe_w13_r, [n_local_experts * ROUTED_W13_N1, ROUTED_W13_M1, ROUTED_NZ_M0, ROUTED_NZ_N0], [moe_r_off, 0, 0, 0]),
                        [n_local_experts, ROUTED_W13_N1, ROUTED_W13_M1, ROUTED_NZ_M0, ROUTED_NZ_N0],
                    ),
                    pl.slice(moe_w13_r_scale, [n_local_experts, 2 * inter], [moe_r_scale_off, 0]),
                    pl.reshape(
                        pl.slice(moe_w_down_r, [n_local_experts * ROUTED_W2_N1, ROUTED_W2_M1, ROUTED_NZ_M0, ROUTED_NZ_N0], [moe_r_down_off, 0, 0, 0]),
                        [n_local_experts, ROUTED_W2_N1, ROUTED_W2_M1, ROUTED_NZ_M0, ROUTED_NZ_N0],
                    ),
                    pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [moe_r_scale_off, 0]),
                    pl.slice(moe_w_gate_s, [sh_inter_local, HIDDEN], [moe_sh_gate_off, 0]),
                    pl.slice(moe_w_up_s, [sh_inter_local, HIDDEN], [moe_sh_gate_off, 0]),
                    pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [moe_sh_down_off, 0]),
                    h_moe,
                    resid_hold_moe,
                    local_expert_count_moe,
                    pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [moe_win_off, 0]),
                    pl.slice(moe_attn_signal_stack, [COMM_SIGNAL_STRIDE_I32, 1], [moe_sig_off, 0]),
                    pl.slice(moe_sh_tmp_stack, [BATCH, HIDDEN], [moe_win_off, 0]),
                    pl.slice(moe_sh_signal_stack, [COMM_SIGNAL_STRIDE_I32, 1], [moe_sig_off, 0]),
                    norm_layer_idx,
                    0,
                    num_tokens,
                    my_rank,
                )
            else:
                # swa_idx = number of swa layers before this layer_idx =
                # layer_idx - (number of full_moe layers before it).
                # full_moe sits at layer_idx % 4 == 1 ({1,5,9,...}); count of
                # full layers strictly before layer_idx = (layer_idx + 2) // 4.
                full_before = (layer_idx + 2) // 4
                swa_idx = layer_idx - full_before
                swa_w_off = swa_idx * HIDDEN
                swa_wo_off = swa_idx * hidden_q_swa
                swa_gate_r_off = swa_idx * nh_swa_pad
                h_moe, local_expert_count_moe = self.swa_moe_chip_orch(
                    prev_hidden,
                    input_rms,
                    pl.slice(moe_swa_wq, [HIDDEN, hidden_q_swa], [swa_w_off, 0]),
                    pl.slice(moe_swa_wk, [HIDDEN, KV_HIDDEN_LOCAL_R], [swa_w_off, 0]),
                    pl.slice(moe_swa_wv, [HIDDEN, KV_HIDDEN_LOCAL_R], [swa_w_off, 0]),
                    q_norm,
                    k_norm,
                    seq_lens,
                    block_table,
                    slot_mapping,
                    rope_cos_swa,
                    rope_sin_swa,
                    k_cache,
                    v_cache,
                    pl.slice(moe_swa_wo, [hidden_q_swa, HIDDEN], [swa_wo_off, 0]),
                    pl.slice(moe_swa_w_g, [HIDDEN, nh_swa_pad], [swa_w_off, 0]),
                    pl.slice(moe_swa_gate_r, [nh_swa_pad, hidden_q_swa], [swa_gate_r_off, 0]),
                    post_rms,
                    pl.slice(moe_gate_w, [N_EXPERTS, HIDDEN], [moe_gate_off, 0]),
                    pl.slice(moe_router_bias, [N_EXPERTS], [moe_bias_off]),
                    pl.reshape(
                        pl.slice(moe_w13_r, [n_local_experts * ROUTED_W13_N1, ROUTED_W13_M1, ROUTED_NZ_M0, ROUTED_NZ_N0], [moe_r_off, 0, 0, 0]),
                        [n_local_experts, ROUTED_W13_N1, ROUTED_W13_M1, ROUTED_NZ_M0, ROUTED_NZ_N0],
                    ),
                    pl.slice(moe_w13_r_scale, [n_local_experts, 2 * inter], [moe_r_scale_off, 0]),
                    pl.reshape(
                        pl.slice(moe_w_down_r, [n_local_experts * ROUTED_W2_N1, ROUTED_W2_M1, ROUTED_NZ_M0, ROUTED_NZ_N0], [moe_r_down_off, 0, 0, 0]),
                        [n_local_experts, ROUTED_W2_N1, ROUTED_W2_M1, ROUTED_NZ_M0, ROUTED_NZ_N0],
                    ),
                    pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [moe_r_scale_off, 0]),
                    pl.slice(moe_w_gate_s, [sh_inter_local, HIDDEN], [moe_sh_gate_off, 0]),
                    pl.slice(moe_w_up_s, [sh_inter_local, HIDDEN], [moe_sh_gate_off, 0]),
                    pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [moe_sh_down_off, 0]),
                    h_moe,
                    resid_hold_moe,
                    local_expert_count_moe,
                    pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [moe_win_off, 0]),
                    pl.slice(moe_attn_signal_stack, [COMM_SIGNAL_STRIDE_I32, 1], [moe_sig_off, 0]),
                    pl.slice(moe_sh_tmp_stack, [BATCH, HIDDEN], [moe_win_off, 0]),
                    pl.slice(moe_sh_signal_stack, [COMM_SIGNAL_STRIDE_I32, 1], [moe_sig_off, 0]),
                    norm_layer_idx,
                    0,
                    num_tokens,
                    my_rank,
                )
            prev_hidden = h_moe

        # ── Phase 4: L43/L44 post-loop explicit specialization layers ──
        # L43 = swa_moe_swiglu7_silu (routed_lim=7.0, shared_lim=0.0): swa attn
        # weight offset 32 (the 33rd swa layer), MoE weight offset 40, norm 43.
        # L44 = full_moe_swiglu7_swiglu16 (routed_lim=7.0, shared_lim=16.0):
        # full attn weight offset 11 (the 12th full layer), MoE offset 41, norm 44.
        # swiglu7 / swiglu16 constants resolved at module level (always-True).
        h_layer_43 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        resid_hold_layer_43 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        swa_w_off_43 = 32 * HIDDEN
        swa_wo_off_43 = 32 * hidden_q_swa
        swa_gate_r_off_43 = 32 * nh_swa_pad
        moe_gate_off_43 = 40 * N_EXPERTS
        moe_sh_gate_off_43 = 40 * sh_inter_local
        moe_bias_off_43 = 40 * N_EXPERTS
        moe_r_off_43 = 40 * (n_local_experts * ROUTED_W13_N1)
        moe_r_scale_off_43 = 40 * n_local_experts
        moe_r_down_off_43 = 40 * (n_local_experts * ROUTED_W2_N1)
        moe_sh_down_off_43 = 40 * sh_inter_local
        moe_win_off_43 = 40 * BATCH
        moe_sig_off_43 = 40 * COMM_SIGNAL_STRIDE_I32
        norm_layer_idx_43 = pl.cast(43, pl.INT32)
        h_layer_43 = self.swa_moe_chip_orch_swiglu7_silu(
            prev_hidden,
            input_rms,
            pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [swa_w_off_43, 0]),
            pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL_R], [swa_w_off_43, 0]),
            pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL_R], [swa_w_off_43, 0]),
            q_norm,
            k_norm,
            seq_lens,
            block_table,
            slot_mapping,
            rope_cos_swa,
            rope_sin_swa,
            k_cache,
            v_cache,
            pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [swa_wo_off_43, 0]),
            pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [swa_w_off_43, 0]),
            pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [swa_gate_r_off_43, 0]),
            post_rms,
            pl.slice(moe_gate_w, [N_EXPERTS, HIDDEN], [moe_gate_off_43, 0]),
            pl.slice(moe_router_bias, [N_EXPERTS], [moe_bias_off_43]),
            pl.reshape(
                pl.slice(moe_w13_r, [n_local_experts * ROUTED_W13_N1, ROUTED_W13_M1, ROUTED_NZ_M0, ROUTED_NZ_N0], [moe_r_off_43, 0, 0, 0]),
                [n_local_experts, ROUTED_W13_N1, ROUTED_W13_M1, ROUTED_NZ_M0, ROUTED_NZ_N0],
            ),
            pl.slice(moe_w13_r_scale, [n_local_experts, 2 * inter], [moe_r_scale_off_43, 0]),
            pl.reshape(
                pl.slice(moe_w_down_r, [n_local_experts * ROUTED_W2_N1, ROUTED_W2_M1, ROUTED_NZ_M0, ROUTED_NZ_N0], [moe_r_down_off_43, 0, 0, 0]),
                [n_local_experts, ROUTED_W2_N1, ROUTED_W2_M1, ROUTED_NZ_M0, ROUTED_NZ_N0],
            ),
            pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [moe_r_scale_off_43, 0]),
            pl.slice(moe_w_gate_s, [sh_inter_local, HIDDEN], [moe_sh_gate_off_43, 0]),
            pl.slice(moe_w_up_s, [sh_inter_local, HIDDEN], [moe_sh_gate_off_43, 0]),
            pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [moe_sh_down_off_43, 0]),
            h_layer_43,
            resid_hold_layer_43,
            pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [moe_win_off_43, 0]),
            pl.slice(moe_attn_signal_stack, [COMM_SIGNAL_STRIDE_I32, 1], [moe_sig_off_43, 0]),
            pl.slice(moe_sh_tmp_stack, [BATCH, HIDDEN], [moe_win_off_43, 0]),
            pl.slice(moe_sh_signal_stack, [COMM_SIGNAL_STRIDE_I32, 1], [moe_sig_off_43, 0]),
            norm_layer_idx_43,
            0,
            num_tokens,
            my_rank,
        )
        prev_hidden = h_layer_43

        resid_hold_layer_44 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        full_w_off_44 = 11 * HIDDEN
        full_wo_off_44 = 11 * hidden_q_full
        full_gate_r_off_44 = 11 * nh_full_pad
        moe_gate_off_44 = 41 * N_EXPERTS
        moe_sh_gate_off_44 = 41 * sh_inter_local
        moe_bias_off_44 = 41 * N_EXPERTS
        moe_r_off_44 = 41 * (n_local_experts * ROUTED_W13_N1)
        moe_r_scale_off_44 = 41 * n_local_experts
        moe_r_down_off_44 = 41 * (n_local_experts * ROUTED_W2_N1)
        moe_sh_down_off_44 = 41 * sh_inter_local
        moe_win_off_44 = 41 * BATCH
        moe_sig_off_44 = 41 * COMM_SIGNAL_STRIDE_I32
        norm_layer_idx_44 = pl.cast(44, pl.INT32)
        next_hidden_out = self.full_moe_chip_orch_swiglu7_swiglu16(
            prev_hidden,
            input_rms,
            pl.slice(full_wq, [HIDDEN, hidden_q_full], [full_w_off_44, 0]),
            pl.slice(full_wk, [HIDDEN, KV_HIDDEN_LOCAL_R], [full_w_off_44, 0]),
            pl.slice(full_wv, [HIDDEN, KV_HIDDEN_LOCAL_R], [full_w_off_44, 0]),
            q_norm,
            k_norm,
            seq_lens,
            block_table,
            slot_mapping,
            rope_cos_full,
            rope_sin_full,
            k_cache,
            v_cache,
            pl.slice(full_wo, [hidden_q_full, HIDDEN], [full_wo_off_44, 0]),
            pl.slice(full_w_g, [HIDDEN, nh_full_pad], [full_w_off_44, 0]),
            pl.slice(full_gate_r, [nh_full_pad, hidden_q_full], [full_gate_r_off_44, 0]),
            post_rms,
            pl.slice(moe_gate_w, [N_EXPERTS, HIDDEN], [moe_gate_off_44, 0]),
            pl.slice(moe_router_bias, [N_EXPERTS], [moe_bias_off_44]),
            pl.reshape(
                pl.slice(moe_w13_r, [n_local_experts * ROUTED_W13_N1, ROUTED_W13_M1, ROUTED_NZ_M0, ROUTED_NZ_N0], [moe_r_off_44, 0, 0, 0]),
                [n_local_experts, ROUTED_W13_N1, ROUTED_W13_M1, ROUTED_NZ_M0, ROUTED_NZ_N0],
            ),
            pl.slice(moe_w13_r_scale, [n_local_experts, 2 * inter], [moe_r_scale_off_44, 0]),
            pl.reshape(
                pl.slice(moe_w_down_r, [n_local_experts * ROUTED_W2_N1, ROUTED_W2_M1, ROUTED_NZ_M0, ROUTED_NZ_N0], [moe_r_down_off_44, 0, 0, 0]),
                [n_local_experts, ROUTED_W2_N1, ROUTED_W2_M1, ROUTED_NZ_M0, ROUTED_NZ_N0],
            ),
            pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [moe_r_scale_off_44, 0]),
            pl.slice(moe_w_gate_s, [sh_inter_local, HIDDEN], [moe_sh_gate_off_44, 0]),
            pl.slice(moe_w_up_s, [sh_inter_local, HIDDEN], [moe_sh_gate_off_44, 0]),
            pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [moe_sh_down_off_44, 0]),
            next_hidden_out,
            resid_hold_layer_44,
            pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [moe_win_off_44, 0]),
            pl.slice(moe_attn_signal_stack, [COMM_SIGNAL_STRIDE_I32, 1], [moe_sig_off_44, 0]),
            pl.slice(moe_sh_tmp_stack, [BATCH, HIDDEN], [moe_win_off_44, 0]),
            pl.slice(moe_sh_signal_stack, [COMM_SIGNAL_STRIDE_I32, 1], [moe_sig_off_44, 0]),
            norm_layer_idx_44,
            0,
            num_tokens,
            my_rank,
        )
        return next_hidden_out

    @pl.function(
        level=pl.Level.HOST,
        role=pl.Role.Orchestrator,
    )
    def host_orch(  # noqa: PLR0913, PLR0915
        self,
        current_hidden: pl.Tensor[[tp_size, BATCH, HIDDEN], pl.BF16],
        input_rms: pl.Tensor[[tp_size, LAYER_DYN, HIDDEN], pl.FP32],
        post_rms: pl.Tensor[[tp_size, LAYER_DYN, HIDDEN], pl.FP32],
        q_norm: pl.Tensor[[tp_size, LAYER_DYN, HEAD_DIM], pl.FP32],
        k_norm: pl.Tensor[[tp_size, LAYER_DYN, HEAD_DIM], pl.FP32],
        full_wq: pl.Tensor[[tp_size, N_FULL_ATTN_LAYERS, HIDDEN, hidden_q_full], pl.BF16],
        full_wk: pl.Tensor[[tp_size, N_FULL_ATTN_LAYERS, HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        full_wv: pl.Tensor[[tp_size, N_FULL_ATTN_LAYERS, HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        full_wo: pl.Tensor[[tp_size, N_FULL_ATTN_LAYERS, hidden_q_full, HIDDEN], pl.BF16],
        full_w_g: pl.Tensor[[tp_size, N_FULL_ATTN_LAYERS, HIDDEN, nh_full_pad], pl.BF16],
        full_gate_r: pl.Tensor[[tp_size, N_FULL_ATTN_LAYERS, nh_full_pad, hidden_q_full], pl.BF16],
        swa_wq: pl.Tensor[[tp_size, N_SWA_ATTN_LAYERS, HIDDEN, hidden_q_swa], pl.BF16],
        swa_wk: pl.Tensor[[tp_size, N_SWA_ATTN_LAYERS, HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        swa_wv: pl.Tensor[[tp_size, N_SWA_ATTN_LAYERS, HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        swa_wo: pl.Tensor[[tp_size, N_SWA_ATTN_LAYERS, hidden_q_swa, HIDDEN], pl.BF16],
        swa_w_g: pl.Tensor[[tp_size, N_SWA_ATTN_LAYERS, HIDDEN, nh_swa_pad], pl.BF16],
        swa_gate_r: pl.Tensor[[tp_size, N_SWA_ATTN_LAYERS, nh_swa_pad, hidden_q_swa], pl.BF16],
        dense_w_gate: pl.Tensor[[tp_size, NUM_DENSE_LAYERS, HIDDEN, INTER_LOCAL], pl.BF16],
        dense_w_up: pl.Tensor[[tp_size, NUM_DENSE_LAYERS, HIDDEN, INTER_LOCAL], pl.BF16],
        dense_w_down: pl.Tensor[[tp_size, NUM_DENSE_LAYERS, INTER_LOCAL, HIDDEN], pl.BF16],
        # MoE silu_silu L3-L42 (40 layers) weights (per-rank tp_size leading dim).
        moe_full_wq: pl.Tensor[[tp_size, NUM_FULL_MOE_LAYERS, HIDDEN, hidden_q_full], pl.BF16],
        moe_full_wk: pl.Tensor[[tp_size, NUM_FULL_MOE_LAYERS, HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        moe_full_wv: pl.Tensor[[tp_size, NUM_FULL_MOE_LAYERS, HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        moe_full_wo: pl.Tensor[[tp_size, NUM_FULL_MOE_LAYERS, hidden_q_full, HIDDEN], pl.BF16],
        moe_full_w_g: pl.Tensor[[tp_size, NUM_FULL_MOE_LAYERS, HIDDEN, nh_full_pad], pl.BF16],
        moe_full_gate_r: pl.Tensor[[tp_size, NUM_FULL_MOE_LAYERS, nh_full_pad, hidden_q_full], pl.BF16],
        moe_swa_wq: pl.Tensor[[tp_size, NUM_SWA_MOE_LAYERS, HIDDEN, hidden_q_swa], pl.BF16],
        moe_swa_wk: pl.Tensor[[tp_size, NUM_SWA_MOE_LAYERS, HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        moe_swa_wv: pl.Tensor[[tp_size, NUM_SWA_MOE_LAYERS, HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        moe_swa_wo: pl.Tensor[[tp_size, NUM_SWA_MOE_LAYERS, hidden_q_swa, HIDDEN], pl.BF16],
        moe_swa_w_g: pl.Tensor[[tp_size, NUM_SWA_MOE_LAYERS, HIDDEN, nh_swa_pad], pl.BF16],
        moe_swa_gate_r: pl.Tensor[[tp_size, NUM_SWA_MOE_LAYERS, nh_swa_pad, hidden_q_swa], pl.BF16],
        moe_gate_w: pl.Tensor[[tp_size, NUM_MOE_LAYERS_TOTAL, N_EXPERTS, HIDDEN], pl.FP32],
        moe_router_bias: pl.Tensor[[tp_size, NUM_MOE_LAYERS_TOTAL, N_EXPERTS], pl.FP32],
        moe_w13_r: pl.Tensor[[tp_size, NUM_MOE_LAYERS_TOTAL, n_local_experts, ROUTED_W13_N1, ROUTED_W13_M1, ROUTED_NZ_M0, ROUTED_NZ_N0], pl.INT8],
        moe_w13_r_scale: pl.Tensor[[tp_size, NUM_MOE_LAYERS_TOTAL, n_local_experts, 2 * inter], pl.FP32],
        moe_w_down_r: pl.Tensor[[tp_size, NUM_MOE_LAYERS_TOTAL, n_local_experts, ROUTED_W2_N1, ROUTED_W2_M1, ROUTED_NZ_M0, ROUTED_NZ_N0], pl.INT8],
        moe_w_down_r_scale: pl.Tensor[[tp_size, NUM_MOE_LAYERS_TOTAL, n_local_experts, HIDDEN], pl.FP32],
        moe_w_gate_s: pl.Tensor[[tp_size, NUM_MOE_LAYERS_TOTAL, sh_inter_local, HIDDEN], pl.BF16],
        moe_w_up_s: pl.Tensor[[tp_size, NUM_MOE_LAYERS_TOTAL, sh_inter_local, HIDDEN], pl.BF16],
        moe_w_down_s: pl.Tensor[[tp_size, NUM_MOE_LAYERS_TOTAL, sh_inter_local, HIDDEN], pl.BF16],
        seq_lens: pl.Tensor[[tp_size, USER_BATCH_DYN], pl.INT32],
        block_table: pl.Tensor[[tp_size, BLOCK_TABLE_FLAT_DYN], pl.INT32],
        slot_mapping: pl.Tensor[[tp_size, USER_BATCH_DYN], pl.INT32],
        rope_cos_full: pl.Tensor[[tp_size, ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],
        rope_sin_full: pl.Tensor[[tp_size, ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],
        rope_cos_swa: pl.Tensor[[tp_size, ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],
        rope_sin_swa: pl.Tensor[[tp_size, ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],
        k_cache: pl.InOut[pl.Tensor[[tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16]],
        v_cache: pl.InOut[pl.Tensor[[tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16]],
        next_hidden_out: pl.Out[pl.Tensor[[tp_size, BATCH, HIDDEN], pl.BF16]],
        num_tokens_per_owner: pl.Tensor[[NUM_TOKENS_RUNTIME], pl.INT32],
    ):
        # K8: control (signal / arrived) buffers are declared first so that
        # they form ONE contiguous window prefix.  The reset path can then
        # restore them with a single blocking memset_all instead of one per
        # buffer -- broadcast count, not bytes, dominates that path.
        dense_attn_signal_stack_buf = pld.alloc_window_buffer(NUM_DENSE_LAYERS * COMM_CONTROL_SIGNAL_BYTES)
        dense_mlp_signal_stack_buf = pld.alloc_window_buffer(NUM_DENSE_LAYERS * COMM_CONTROL_SIGNAL_BYTES)
        moe_attn_signal_stack_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * COMM_CONTROL_SIGNAL_BYTES)
        moe_sh_signal_stack_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * COMM_CONTROL_SIGNAL_BYTES)
        dense_attn_tmp_stack_buf = pld.alloc_window_buffer(NUM_DENSE_LAYERS * BATCH * HIDDEN * 2)
        dense_mlp_tmp_stack_buf = pld.alloc_window_buffer(NUM_DENSE_LAYERS * BATCH * HIDDEN * 2)
        moe_attn_tmp_stack_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * BATCH * HIDDEN * 2)
        moe_sh_tmp_stack_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * BATCH * HIDDEN * 2)
        for r in pl.range(pld.world_size()):
            self.whole_chip_orch(
                current_hidden[r],
                input_rms[r],
                post_rms[r],
                q_norm[r],
                k_norm[r],
                pl.reshape(full_wq[r], [N_FULL_ATTN_LAYERS * HIDDEN, hidden_q_full]),
                pl.reshape(full_wk[r], [N_FULL_ATTN_LAYERS * HIDDEN, KV_HIDDEN_LOCAL_R]),
                pl.reshape(full_wv[r], [N_FULL_ATTN_LAYERS * HIDDEN, KV_HIDDEN_LOCAL_R]),
                pl.reshape(full_wo[r], [N_FULL_ATTN_LAYERS * hidden_q_full, HIDDEN]),
                pl.reshape(full_w_g[r], [N_FULL_ATTN_LAYERS * HIDDEN, nh_full_pad]),
                pl.reshape(full_gate_r[r], [N_FULL_ATTN_LAYERS * nh_full_pad, hidden_q_full]),
                pl.reshape(swa_wq[r], [N_SWA_ATTN_LAYERS * HIDDEN, hidden_q_swa]),
                pl.reshape(swa_wk[r], [N_SWA_ATTN_LAYERS * HIDDEN, KV_HIDDEN_LOCAL_R]),
                pl.reshape(swa_wv[r], [N_SWA_ATTN_LAYERS * HIDDEN, KV_HIDDEN_LOCAL_R]),
                pl.reshape(swa_wo[r], [N_SWA_ATTN_LAYERS * hidden_q_swa, HIDDEN]),
                pl.reshape(swa_w_g[r], [N_SWA_ATTN_LAYERS * HIDDEN, nh_swa_pad]),
                pl.reshape(swa_gate_r[r], [N_SWA_ATTN_LAYERS * nh_swa_pad, hidden_q_swa]),
                pl.reshape(dense_w_gate[r], [NUM_DENSE_LAYERS * HIDDEN, INTER_LOCAL]),
                pl.reshape(dense_w_up[r], [NUM_DENSE_LAYERS * HIDDEN, INTER_LOCAL]),
                pl.reshape(dense_w_down[r], [NUM_DENSE_LAYERS * INTER_LOCAL, HIDDEN]),
                pl.reshape(moe_full_wq[r], [NUM_FULL_MOE_LAYERS * HIDDEN, hidden_q_full]),
                pl.reshape(moe_full_wk[r], [NUM_FULL_MOE_LAYERS * HIDDEN, KV_HIDDEN_LOCAL_R]),
                pl.reshape(moe_full_wv[r], [NUM_FULL_MOE_LAYERS * HIDDEN, KV_HIDDEN_LOCAL_R]),
                pl.reshape(moe_full_wo[r], [NUM_FULL_MOE_LAYERS * hidden_q_full, HIDDEN]),
                pl.reshape(moe_full_w_g[r], [NUM_FULL_MOE_LAYERS * HIDDEN, nh_full_pad]),
                pl.reshape(moe_full_gate_r[r], [NUM_FULL_MOE_LAYERS * nh_full_pad, hidden_q_full]),
                pl.reshape(moe_swa_wq[r], [NUM_SWA_MOE_LAYERS * HIDDEN, hidden_q_swa]),
                pl.reshape(moe_swa_wk[r], [NUM_SWA_MOE_LAYERS * HIDDEN, KV_HIDDEN_LOCAL_R]),
                pl.reshape(moe_swa_wv[r], [NUM_SWA_MOE_LAYERS * HIDDEN, KV_HIDDEN_LOCAL_R]),
                pl.reshape(moe_swa_wo[r], [NUM_SWA_MOE_LAYERS * hidden_q_swa, HIDDEN]),
                pl.reshape(moe_swa_w_g[r], [NUM_SWA_MOE_LAYERS * HIDDEN, nh_swa_pad]),
                pl.reshape(moe_swa_gate_r[r], [NUM_SWA_MOE_LAYERS * nh_swa_pad, hidden_q_swa]),
                pl.reshape(moe_gate_w[r], [NUM_MOE_LAYERS_TOTAL * N_EXPERTS, HIDDEN]),
                pl.reshape(moe_router_bias[r], [NUM_MOE_LAYERS_TOTAL * N_EXPERTS]),
                pl.reshape(moe_w13_r[r], [NUM_MOE_LAYERS_TOTAL * n_local_experts * ROUTED_W13_N1, ROUTED_W13_M1, ROUTED_NZ_M0, ROUTED_NZ_N0]),
                pl.reshape(moe_w13_r_scale[r], [NUM_MOE_LAYERS_TOTAL * n_local_experts, 2 * inter]),
                pl.reshape(moe_w_down_r[r], [NUM_MOE_LAYERS_TOTAL * n_local_experts * ROUTED_W2_N1, ROUTED_W2_M1, ROUTED_NZ_M0, ROUTED_NZ_N0]),
                pl.reshape(moe_w_down_r_scale[r], [NUM_MOE_LAYERS_TOTAL * n_local_experts, HIDDEN]),
                pl.reshape(moe_w_gate_s[r], [NUM_MOE_LAYERS_TOTAL * sh_inter_local, HIDDEN]),
                pl.reshape(moe_w_up_s[r], [NUM_MOE_LAYERS_TOTAL * sh_inter_local, HIDDEN]),
                pl.reshape(moe_w_down_s[r], [NUM_MOE_LAYERS_TOTAL * sh_inter_local, HIDDEN]),
                seq_lens[r],
                block_table[r],
                slot_mapping[r],
                rope_cos_full[r],
                rope_sin_full[r],
                rope_cos_swa[r],
                rope_sin_swa[r],
                k_cache[r],
                v_cache[r],
                next_hidden_out[r],
                pld.window(dense_attn_tmp_stack_buf, [NUM_DENSE_LAYERS * BATCH, HIDDEN],
                           dtype=pl.BF16),
                pld.window(dense_attn_signal_stack_buf, [NUM_DENSE_LAYERS * COMM_SIGNAL_STRIDE_I32, 1],
                           dtype=pl.INT32),
                pld.window(dense_mlp_tmp_stack_buf, [NUM_DENSE_LAYERS * BATCH, HIDDEN],
                           dtype=pl.BF16),
                pld.window(dense_mlp_signal_stack_buf, [NUM_DENSE_LAYERS * COMM_SIGNAL_STRIDE_I32, 1],
                           dtype=pl.INT32),
                pld.window(moe_attn_tmp_stack_buf, [NUM_MOE_LAYERS_TOTAL * BATCH, HIDDEN],
                           dtype=pl.BF16),
                pld.window(moe_attn_signal_stack_buf, [NUM_MOE_LAYERS_TOTAL * COMM_SIGNAL_STRIDE_I32, 1],
                           dtype=pl.INT32),
                pld.window(moe_sh_tmp_stack_buf, [NUM_MOE_LAYERS_TOTAL * BATCH, HIDDEN],
                           dtype=pl.BF16),
                pld.window(moe_sh_signal_stack_buf, [NUM_MOE_LAYERS_TOTAL * COMM_SIGNAL_STRIDE_I32, 1],
                           dtype=pl.INT32),
                num_tokens_per_owner,
                r,
                device=r,
            )


whole_decode_step3p5 = WholeDecodeStep3p5
