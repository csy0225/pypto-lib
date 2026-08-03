"""Canonical whole-net Step3.5 decode program.

The 45-layer graph uses runtime ``pl.range`` loops over resident leading-
dimension weight/KV views.  Physical tensor shapes are compile-time capacity;
``num_tokens_per_owner`` supplies the runtime active-row bound.

MoE communication is being unified with the DeepSeek V4-Flash baseline:
local-expert-lane dispatch push/gather, independent metadata/payload arrivals,
combine scatter/arrival, and token-level FP32 reduction.  EP data windows are
one shared set across all 42 MoE calls and use a monotonic 1-based
``moe_epoch`` across independent metadata, payload, and combine arrival
lineages.  Metadata waits for ``epoch``; payload and combine waits count one
completion per local-expert lane and use ``epoch * N_LOCAL``.

Attention/shared TP all-reduce scratch remains independent from EP windows.
Its peer order, single FP32 accumulator, and final one-time BF16 store are not
part of the MoE communication migration.
"""
# ruff: noqa: F401

from __future__ import annotations

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
    FULL_ATTN_SOFTMAX_BLOCKS_PER_TASK,
    FULL_ATTN_ONLINE_SOFTMAX_BLOCKS_PER_TASK,
    FULL_ATTN_ONLINE_SOFTMAX_PARTIALS_PER_REDUCE_TASK,
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
from models.step3p5.dispatch import N_RANKS_PAD, PER_RANK_BUCKETS

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
# canonical ABI follows V4-Flash expert-lane push/gather.
N_EXPERTS = MOE_NUM_EXPERTS
N_LOCAL_EXPERTS = MOE_NUM_EXPERTS_LOCAL
INTER_R = MOE_INTERMEDIATE
INTER_S_LOCAL = SHARE_EXPERT_DIM_LOCAL
TOPK = 8
N_ROUTES_PER_RANK = BATCH * TOPK

# Router (gate) kernel constants — mirrors gate.py / moe.ROUTER_*.
ROUTER_SCORE_PAD = 512
ROUTER_TOPK_PAD = 16
ROUTER_SORT_PAD = ROUTER_TOPK_PAD * 2
ROUTER_GATE_K_CHUNK = 256
ROUTER_GATE_N_CHUNK = 32
ROUTER_FP32_NEG_INF = -3.4028235e38
ROUTER_SCALE = 3.0  # MOE_ROUTER_SCALING_FACTOR

# Routed-expert kernel constants — mirrors expert_routed.py / moe.ROUTED_*.
ROUTED_GATE_K_CHUNK = 64
ROUTED_GATE_N_CHUNK = 64
ROUTED_DOWN_K_CHUNK = 64
ROUTED_DOWN_N_CHUNK = 128
RECV_TILE = 32

# Shared-expert kernel constants — mirrors expert_shared.py / moe.SHARED_*.
SHARED_GATE_K_CHUNK = 256
SHARED_GATE_N_CHUNK = INTER_S_LOCAL  # 160 — one N tile covers the slice
SHARED_DOWN_K_CHUNK = INTER_S_LOCAL  # 160 — one K tile covers the slice
SHARED_DOWN_N_CHUNK = 256
SHARED_SWIGLU_N_CHUNK = 32

# MoE-local helper constants are module-level for parse-time closure capture.
n_ranks = tp_size
n_ranks_pad = N_RANKS_PAD
n_local_experts = N_LOCAL_EXPERTS
n_local_experts_pad = ((n_local_experts + 7) // 8) * 8
idx_pad = 8
# V4-Flash dispatch physical lanes: [local_expert, source_rank, token_slot].
dispatch_max_per_src = BATCH
dispatch_recv_per_expert = n_ranks * dispatch_max_per_src
dispatch_lane_rows = n_local_experts * dispatch_recv_per_expert
expert_recv_max = dispatch_recv_per_expert
N_RECV_TILES = expert_recv_max // RECV_TILE
DISPATCH_SCALE_COLS = 1  # V4-Flash: one per-token activation scale
dispatch_weight_col = 1  # aux[0]=scale, aux[1]=route weight
dispatch_aux_pad = 8  # physical FP32 row tile; logical cols 0..1
inter = MOE_INTERMEDIATE
sh_inter_local = INTER_S_LOCAL
# The local expert ABI is a fixed [expert, source, token_slot] lane slab.
# Counts describe valid rows only; they must not change the physical base of an
# expert or the shape of the routed-expert loop.
local_recv_max = n_local_experts * expert_recv_max
stage_rows = 8
n_routes_per_rank = BATCH * TOPK
per_rank_buckets = PER_RANK_BUCKETS
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


@pl.program
class WholeDecodeStep3p5:
    # ── TP all-reduce collective ────────────────────────────────────────
    # The inlined attention and dense-MLP bodies call this method to gather
    # o_proj and down_proj partial sums.  Keep the method on this program so
    # pl.inline resolves those calls locally.  The protocol uses three
    # completion waves with Ge thresholds 1, 2, and 3.
    @pl.function(type=pl.FunctionType.InCore)
    def tp_all_reduce(
        self,
        local: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
        signal_window: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
        group_size = tp_size

        # Keep the existing full-size communication window ABI.  The transfer
        # grain is configurable, while reduce-scatter ownership is defined by
        # the TP rank count rather than a fixed core count.
        ar_chunk = TP_ALL_REDUCE_CHUNK

        # Self-target TPUT drains before the following notify (PTOAS#872).
        pld.tensor.put(
            dst=tmp_window,
            peer=my_rank,
            src=local,
            chunk_rows=BATCH,
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
        owned_chunk = HIDDEN // group_size
        owned_base = my_rank * owned_chunk
        own_tile = pl.load(
            tmp_window, [0, owned_base], [BATCH, owned_chunk],
        )
        acc = pl.mul(pl.cast(own_tile, target_type=pl.FP32), 0.0)
        for peer in pl.range(group_size):
            if peer == my_rank:
                acc = pl.add(
                    acc,
                    pl.cast(own_tile, target_type=pl.FP32),
                )
            else:
                remote_tile = pld.tile.remote_load(
                    tmp_window, peer=peer,
                    offsets=[0, owned_base],
                    shape=[BATCH, owned_chunk],
                )
                acc = pl.add(
                    acc,
                    pl.cast(remote_tile, target_type=pl.FP32),
                )
        reduced_tile = pl.cast(acc, target_type=pl.BF16)

        # Publish the write-disjoint reduced shard with the existing push path.
        pl.store(reduced_tile, [0, owned_base], tmp_window)
        for dst in pl.range(group_size):
            if dst != my_rank:
                pld.tile.remote_store(
                    reduced_tile,
                    target=tmp_window,
                    peer=dst,
                    offsets=[0, owned_base],
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

        # The completed temporary window contains the complete reduced vector.
        for k0 in pl.parallel(0, HIDDEN, ar_chunk):
            result_tile = pl.load(
                tmp_window, [0, k0], [BATCH, ar_chunk],
            )
            pl.store(result_tile, [0, k0], local)

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
            h0_out, norm_layer_idx, mlp_layer_idx,
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
            hidden_out, norm_layer_idx, mlp_layer_idx,
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
        gate_w: pl.Tensor[[HIDDEN, N_EXPERTS], pl.FP32],
        router_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
        expert_indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
        expert_weights: pl.Tensor[[BATCH, TOPK], pl.FP32],
        num_tokens: pl.Scalar[pl.INT32],
    ):
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

        # Keep the initialization in a CORE_GROUP scope, but create the
        # expert-column SPMD fan-out at function scope.  PyPTO does not accept
        # an SPMD region nested inside an AT scope on all compiler versions.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_init"):
            # Initialise output pads once — columns beyond N_EXPERTS
            # keep 0 / NEG_INF so topk is not tricked by uninitialised values.
            score_buf[:, :] = pl.full(
                [BATCH, ROUTER_SCORE_PAD], dtype=pl.FP32, value=0.0,
            )
            biased_buf[:, :] = pl.full(
                [BATCH, ROUTER_SCORE_PAD],
                dtype=pl.FP32, value=ROUTER_FP32_NEG_INF,
            )

        # G1: fan the gate matmul over expert chunks.  The token dimension is
        # the inner sequential bound, so inactive storage rows never enter
        # the routing/top-k path.  The A2/A3 cube M tile stays the validated
        # static 16 rows; active_tokens gates the later row-wise stages.
        for nb in pl.spmd(
            N_EXPERTS // ROUTER_GATE_N_CHUNK,
            name_hint="gate_expert_fanout",
        ):
            n0 = nb * ROUTER_GATE_N_CHUNK
            raw0 = pl.cast(
                pl.slice(resid, [BATCH, ROUTER_GATE_K_CHUNK], [0, 0]),
                target_type=pl.FP32,
            )
            gamma0 = pl.slice(
                post_rms_weight, [1, ROUTER_GATE_K_CHUNK],
                [norm_layer_idx, 0],
            )
            # V4 deferred RMSNorm: gate consumes FP32 xg = resid*(gamma+1).
            # The full xg tensor is recomputed chunk-wise because the 0726
            # backend UB cannot retain [BATCH,HIDDEN] FP32 as one tile.
            x0 = pl.col_expand_mul(raw0, pl.add(gamma0, 1.0))
            w0 = pl.slice(
                gate_w,
                [ROUTER_GATE_K_CHUNK, ROUTER_GATE_N_CHUNK],
                [0, n0],
            )
            logits_n = pl.matmul(x0, w0, out_dtype=pl.FP32)
            for kb in pl.range(1, HIDDEN // ROUTER_GATE_K_CHUNK):
                k0 = kb * ROUTER_GATE_K_CHUNK
                rawk = pl.cast(
                    pl.slice(
                        resid, [BATCH, ROUTER_GATE_K_CHUNK], [0, k0],
                    ),
                    target_type=pl.FP32,
                )
                gammak = pl.slice(
                    post_rms_weight, [1, ROUTER_GATE_K_CHUNK],
                    [norm_layer_idx, k0],
                )
                xk = pl.col_expand_mul(rawk, pl.add(gammak, 1.0))
                wk = pl.slice(
                    gate_w,
                    [ROUTER_GATE_K_CHUNK, ROUTER_GATE_N_CHUNK],
                    [k0, n0],
                )
                logits_n = pl.matmul_acc(logits_n, xk, wk)
            # xg omits the positive per-token inv_rms factor. Apply it after
            # the FP32 matmul, exactly as V4-Flash, before step3p5 sigmoid.
            logits_n = pl.row_expand_mul(logits_n, inv_rms)
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
            for tt in pl.range(active_tokens):
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
                # footprint, which ptoas rejects.  Keep the one-token
                # semantics but reduce an aligned 8-row workspace: row 0
                # contains the real top-k values and rows 1..7 are zero.  Do
                # the assemble before fillpad: a padded source tile cannot
                # be assembled into an unpadded destination in current PTOAS.
                topk_vals_work = pl.create_tensor(
                    [8, ROUTER_TOPK_PAD], dtype=pl.FP32,
                )
                # Replicate the one real row into the aligned workspace so
                # every row has a non-zero denominator (avoid 0/0 in the
                # inactive helper rows; only row 0 is scattered afterward).
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
                    pl.write(
                        expert_indices, [tt, k],
                        pl.read(topk_idx_tile, [0, k]),
                    )
                    pl.write(
                        expert_weights, [tt, k],
                        pl.read(weights_work, [0, k]),
                    )

        return expert_weights

    @pl.function(type=pl.FunctionType.Inline)
    def gate_step(
        self,
        resid: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
        norm_layer_idx: pl.Scalar[pl.INT32],
        inv_rms: pl.Tensor[[BATCH, 1], pl.FP32],
        gate_w: pl.Tensor[[HIDDEN, N_EXPERTS], pl.FP32],
        router_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
        expert_indices: pl.Out[pl.Tensor[[BATCH, TOPK], pl.INT32]],
        expert_weights: pl.Out[pl.Tensor[[BATCH, TOPK], pl.FP32]],
        num_tokens: pl.Scalar[pl.INT32],
    ) -> tuple[
        pl.Tensor[[BATCH, TOPK], pl.INT32],
        pl.Tensor[[BATCH, TOPK], pl.FP32]
    ]:
        self._gate(
            resid, post_rms_weight, norm_layer_idx, inv_rms,
            gate_w, router_bias, expert_indices, expert_weights, num_tokens,
        )
        return expert_indices, expert_weights

    # ---------- Stage 2: V4-Flash norm/quant + expert-lane dispatch ----------
    @pl.function(type=pl.FunctionType.InCore)
    def _norm_quant_moe_input(
        self,
        resid: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
        norm_layer_idx: pl.Scalar[pl.INT32],
        post_norm_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
        inv_rms_out: pl.Out[pl.Tensor[[BATCH, 1], pl.FP32]],
        x_i8_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.INT8]],
        x_scale_out: pl.Out[
            pl.Tensor[[BATCH, DISPATCH_SCALE_COLS], pl.FP32]
        ],
        num_tokens: pl.Scalar[pl.INT32],
    ) -> tuple[
        pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        pl.Tensor[[BATCH, 1], pl.FP32],
        pl.Tensor[[BATCH, HIDDEN], pl.INT8],
        pl.Tensor[[BATCH, DISPATCH_SCALE_COLS], pl.FP32],
    ]:
        """V4-style deferred RMSNorm and INT8 producer.

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
        # Match V4-Flash's physical gate-tile contract: norm/quant/gate work on
        # complete 16-row cube tiles, while routing, dispatch and observable
        # outputs remain bounded by the logical active_tokens prefix.
        active_gate_tiles = (active_tokens + 15) // 16
        active_gate_tokens = active_gate_tiles * 16
        if active_gate_tokens > BATCH:
            active_gate_tokens = pl.cast(BATCH, pl.INDEX)

        sq_sum = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
        xg_amax = pl.full([1, BATCH], dtype=pl.FP32, value=1e-4)
        for kb in pl.range(HIDDEN // K_CHUNK):
            k0 = kb * K_CHUNK
            raw = pl.cast(
                pl.slice(
                    resid, [BATCH, K_CHUNK], [0, k0],
                    valid_shape=[active_gate_tokens, K_CHUNK],
                ),
                target_type=pl.FP32,
            )
            gamma = pl.slice(
                post_rms_weight, [1, K_CHUNK], [norm_layer_idx, k0],
            )
            xg = pl.col_expand_mul(raw, pl.add(gamma, 1.0))
            sq_sum = pl.add(
                sq_sum,
                pl.reshape(pl.row_sum(pl.mul(raw, raw)), [1, BATCH]),
            )
            xg_amax = pl.maximum(
                xg_amax,
                pl.reshape(
                    pl.row_max(pl.maximum(xg, pl.neg(xg))), [1, BATCH],
                ),
            )

        inv_rms_row = pl.recip(
            pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)),
        )
        inv_rms = pl.reshape(inv_rms_row, [BATCH, 1])
        inv_rms_out[0:BATCH, 0:1] = inv_rms
        quant_mul_row = pl.div(
            pl.full([1, BATCH], dtype=pl.FP32, value=127.0), xg_amax,
        )
        quant_mul = pl.reshape(quant_mul_row, [BATCH, 1])
        dequant_scale = pl.reshape(
            pl.mul(inv_rms_row, pl.mul(xg_amax, 1.0 / 127.0)),
            [BATCH, 1],
        )
        x_scale_out[0:BATCH, 0:1] = dequant_scale

        for kb2 in pl.range(HIDDEN // K_CHUNK):
            k0 = kb2 * K_CHUNK
            raw = pl.cast(
                pl.slice(
                    resid, [BATCH, K_CHUNK], [0, k0],
                    valid_shape=[active_gate_tokens, K_CHUNK],
                ),
                target_type=pl.FP32,
            )
            gamma = pl.slice(
                post_rms_weight, [1, K_CHUNK], [norm_layer_idx, k0],
            )
            # Current 0726 backend UB cannot retain a full [BATCH,HIDDEN]
            # FP32 xg buffer, so recompute xg in the emission pass.  This is a
            # backend/profile trade-off only; norm and quant still share one
            # producer and the same inv_rms/amax values.
            xg = pl.col_expand_mul(raw, pl.add(gamma, 1.0))
            normed = pl.row_expand_mul(xg, inv_rms)
            post_norm_out[0:BATCH, k0 : k0 + K_CHUNK] = pl.cast(
                normed, target_type=pl.BF16,
            )
            qi32 = pl.cast(
                pl.row_expand_mul(xg, quant_mul),
                target_type=pl.INT32, mode="rint",
            )
            qf16 = pl.cast(qi32, target_type=pl.FP16, mode="round")
            x_i8_out[0:BATCH, k0 : k0 + K_CHUNK] = pl.cast(
                qf16, target_type=pl.INT8, mode="trunc",
            )
        return post_norm_out, inv_rms_out, x_i8_out, x_scale_out

    # ---------- Stage 2: dispatch (V4-Flash expert-lane PUSH + gather) ----------
    @pl.function(type=pl.FunctionType.Inline)
    def dispatch_step(  # noqa: PLR0913, PLR0915
        self,
        x: pl.Tensor[[BATCH, HIDDEN], pl.INT8],
        x_scale: pl.Tensor[[BATCH, DISPATCH_SCALE_COLS], pl.FP32],
        expert_indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
        expert_weights: pl.Tensor[[BATCH, TOPK], pl.FP32],
        local_routed_x_out: pl.Out[
            pl.Tensor[[local_recv_max, HIDDEN], pl.INT8]
        ],
        local_routed_x_scale_out: pl.Out[pl.Tensor[[1, local_recv_max], pl.FP32]],
        local_routed_weight_out: pl.Out[pl.Tensor[[local_recv_max], pl.FP32]],
        local_route_out: pl.Out[pl.Tensor[[local_recv_max], pl.INT32]],
        local_expert_offset: pl.Out[pl.Tensor[[n_local_experts], pl.INT32]],
        local_expert_count: pl.Out[pl.Tensor[[n_local_experts], pl.INT32]],
        recv_meta: pld.DistributedTensor[
            [n_ranks, n_local_experts_pad], pl.INT32
        ],
        meta_arrived: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        recv_x: pld.DistributedTensor[
            [dispatch_lane_rows, HIDDEN], pl.INT8
        ],
        recv_aux: pld.DistributedTensor[
            [dispatch_lane_rows, dispatch_aux_pad], pl.FP32
        ],
        recv_route: pld.DistributedTensor[
            [dispatch_lane_rows, idx_pad], pl.INT32
        ],
        data_arrived: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
        moe_epoch: pl.Scalar[pl.INT32],
    ) -> tuple[
        pl.Tensor[[local_recv_max, HIDDEN], pl.INT8],
        pl.Tensor[[1, local_recv_max], pl.FP32],
        pl.Tensor[[local_recv_max], pl.FP32],
        pl.Tensor[[local_recv_max], pl.INT32],
        pl.Tensor[[n_local_experts], pl.INT32],
        pl.Tensor[[n_local_experts], pl.INT32],
        pl.Tensor[[n_ranks, n_local_experts_pad], pl.INT32],
    ]:
        """V4-Flash dispatch ABI; combine conversion is intentionally separate.

        ``recv_aux`` follows the V4-Flash ABI: logical column 0 carries the
        per-token activation scale and column 1 carries the route weight.
        Columns 2..7 are physical FP32 padding for the current PTOAS tile
        alignment ABI; they carry no model semantics. ``recv_route`` carries ``t * TOPK + k``.
        Physical lane capacity remains static while every token loop is bounded
        by the clamped runtime ``num_tokens`` value.
        """
        recv_meta_local = pl.create_tensor(
            [n_ranks, n_local_experts_pad], dtype=pl.INT32, manual_dep=True,
        )
        # Metadata has an independent arrival lineage so expert counts become
        # available without waiting for the bulk x/aux/route payload.
        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="dispatch_meta",
            allow_early_resolve=True,
        ) as meta_tid:
            active_tokens = pl.cast(num_tokens, pl.INDEX)
            if active_tokens < 0:
                active_tokens = pl.cast(0, pl.INDEX)
            if active_tokens > BATCH:
                active_tokens = pl.cast(BATCH, pl.INDEX)

            cursor = pl.array.create(n_ranks * n_local_experts, pl.INT32)
            for d in pl.range(n_ranks):
                for e in pl.range(n_local_experts):
                    cursor[d * n_local_experts + e] = 0
            for t in pl.range(active_tokens):
                for k in pl.range(TOPK):
                    eid = pl.read(expert_indices, [t, k])
                    dst = eid // n_local_experts
                    loc_e = eid - dst * n_local_experts
                    cursor[dst * n_local_experts + loc_e] = (
                        cursor[dst * n_local_experts + loc_e] + 1
                    )

            meta_tile = pl.tile.full(
                [1, n_local_experts_pad], dtype=pl.INT32, value=0,
            )
            for dst in pl.range(n_ranks):
                for e in pl.range(n_local_experts):
                    pl.tile.write(
                        meta_tile, [0, e], cursor[dst * n_local_experts + e],
                    )
                pld.tile.remote_store(
                    meta_tile, target=recv_meta, peer=dst,
                    offsets=[my_rank, 0],
                )
                if dst != my_rank:
                    pld.system.notify(
                        target=meta_arrived, peer=dst,
                        offsets=[my_rank, 0], value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )

            for src in pl.range(n_ranks):
                if src != my_rank:
                    pld.system.wait(
                        signal=meta_arrived, offsets=[src, 0],
                        expected=moe_epoch, cmp=pld.WaitCmp.Ge,
                    )

            for e in pl.range(n_local_experts):
                pl.write(
                    local_expert_offset, [e],
                    pl.cast(e * expert_recv_max, pl.INT32),
                )
                count = pl.cast(0, pl.INT32)
                for src in pl.range(n_ranks):
                    meta_count = pl.read(recv_meta, [src, e])
                    pl.write(recv_meta_local, [src, e], meta_count)
                    count = count + meta_count
                pl.write(local_expert_count, [e], count)

        # One block owns one local-expert lane on every destination. This is
        # the V4-Flash push layout [expert, source, slot].
        with pl.spmd(
            n_local_experts,
            name_hint="dispatch_push",
            allow_early_resolve=True,
        ) as dispatch_push_tid:
            loc_e = pl.tile.get_block_idx()
            active_tokens = pl.cast(num_tokens, pl.INDEX)
            if active_tokens < 0:
                active_tokens = pl.cast(0, pl.INDEX)
            if active_tokens > BATCH:
                active_tokens = pl.cast(BATCH, pl.INDEX)

            slot_ctr = pl.array.create(n_ranks, pl.INT32)
            for d in pl.range(n_ranks):
                slot_ctr[d] = 0
            lane_base = (
                loc_e * dispatch_recv_per_expert
                + my_rank * dispatch_max_per_src
            )
            aux_tile = pl.tile.full(
                [1, dispatch_aux_pad], dtype=pl.FP32, value=0.0,
            )
            route_tile = pl.tile.full([1, idx_pad], dtype=pl.INT32, value=0)
            for t in pl.range(active_tokens):
                for k in pl.range(TOPK):
                    eid = pl.read(expert_indices, [t, k])
                    dst = eid // n_local_experts
                    le = eid - dst * n_local_experts
                    if le == loc_e:
                        slot = slot_ctr[dst]
                        slot_ctr[dst] = slot + 1
                        row = lane_base + slot
                        pld.tensor.put(
                            dst=recv_x, peer=dst, src=x,
                            dst_offsets=[row, 0], src_offsets=[t, 0],
                            shape=[1, HIDDEN],
                        )
                        for sc in pl.range(DISPATCH_SCALE_COLS):
                            pl.tile.write(
                                aux_tile, [0, sc], pl.read(x_scale, [t, sc]),
                            )
                        pl.tile.write(
                            aux_tile, [0, dispatch_weight_col],
                            pl.read(expert_weights, [t, k]),
                        )
                        pld.tile.remote_store(
                            aux_tile, target=recv_aux, peer=dst,
                            offsets=[row, 0],
                        )
                        pl.tile.write(
                            route_tile, [0, 0],
                            pl.cast(t * TOPK + k, pl.INT32),
                        )
                        pld.tile.remote_store(
                            route_tile, target=recv_route, peer=dst,
                            offsets=[row, 0],
                        )

            for dst in pl.range(n_ranks):
                if dst != my_rank:
                    pld.system.notify(
                        target=data_arrived, peer=dst,
                        offsets=[my_rank, 0], value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )

        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="dispatch_wait",
            deps=[dispatch_push_tid],
            allow_early_resolve=True,
        ) as wait_tid:
            _route_anchor = pl.read(expert_indices, [0, 0])
            for src in pl.range(n_ranks):
                if src != my_rank:
                    pld.system.wait(
                        signal=data_arrived, offsets=[src, 0],
                        expected=pl.cast(
                            moe_epoch * n_local_experts, pl.INT32,
                        ),
                        cmp=pld.WaitCmp.Ge,
                    )

        # Gather each expert into its fixed V4 lane. Route weight and route id
        # stay beside the activation; counts only bound the valid prefix.
        with pl.spmd(
            n_local_experts,
            name_hint="dispatch_gather",
            deps=[wait_tid, meta_tid],
            allow_early_resolve=True,
        ) as dispatch_gather_tid:
            e = pl.tile.get_block_idx()
            out_base = pl.cast(e * expert_recv_max, pl.INDEX)
            expert_prefix = pl.cast(0, pl.INDEX)
            lane_e_base = e * dispatch_recv_per_expert
            for src in pl.range(n_ranks):
                route_count = pl.cast(
                    pl.read(recv_meta_local, [src, e]), pl.INDEX,
                )
                src_base = lane_e_base + src * dispatch_max_per_src
                for slot in pl.range(route_count):
                    in_row = src_base + slot
                    out_row = out_base + expert_prefix + slot
                    local_routed_x_out[out_row : out_row + 1, :] = (
                        recv_x[in_row : in_row + 1, :]
                    )
                    pl.write(
                        local_routed_x_scale_out, [0, out_row],
                        pl.read(recv_aux, [in_row, 0]),
                    )
                    pl.write(
                        local_routed_weight_out, [out_row],
                        pl.read(recv_aux, [in_row, dispatch_weight_col]),
                    )
                    route = pl.read(recv_route, [in_row, 0])
                    pl.write(local_route_out, [out_row], route)
                expert_prefix = expert_prefix + route_count

        return (
            local_routed_x_out,
            local_routed_x_scale_out,
            local_routed_weight_out,
            local_route_out,
            local_expert_offset,
            local_expert_count,
            recv_meta_local,
        )

    # ---------- Stage 3a: expert_routed (local 36 experts) ----------
    @pl.function(type=pl.FunctionType.Inline)
    def _expert_routed(  # noqa: PLR0913, PLR0915
        self,
        local_routed_x: pl.Tensor[[local_recv_max, HIDDEN], pl.INT8],
        local_routed_x_scale: pl.Tensor[[1, local_recv_max], pl.FP32],
        local_routed_weight: pl.Tensor[[local_recv_max], pl.FP32],
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
        for e in pl.range(n_local_experts):
            n_rows = pl.read(local_expert_count, [e])
            offset = pl.cast(e * expert_recv_max, pl.INDEX)
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
                        if _ROUTED_SWIGLU_STEP:
                            silu_c = pl.minimum(silu, _ROUTED_SWIGLU_LIMIT)
                            up_c = pl.maximum(
                                pl.minimum(up_2d, _ROUTED_SWIGLU_LIMIT),
                                -_ROUTED_SWIGLU_LIMIT,
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
                        route_weight = pl.reshape(
                            pl.slice(
                                local_routed_weight,
                                [RECV_TILE],
                                [tile_offset],
                                valid_shape=[tile_valid],
                            ),
                            [RECV_TILE, 1],
                        )
                        y_2d = pl.col_expand_mul(
                            pl.row_expand_mul(
                                pl.cast(
                                    y_acc, target_type=pl.FP32, mode="none",
                                ),
                                pl.mul(h_scale_dq, route_weight),
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
        local_routed_weight: pl.Tensor[[local_recv_max], pl.FP32],
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
            local_routed_weight,
            local_expert_offset, local_expert_count,
            w_gate_r, w_gate_r_scale,
            w_up_r, w_up_r_scale,
            w_down_r, w_down_r_scale,
            local_routed_y,
        )
        return local_routed_y

    # ---------- Stage 3b: expert_shared (5x32 narrow activation tiles) -
    @pl.function(type=pl.FunctionType.Inline)
    def _expert_shared_local(
        self,
        x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        w_gate: pl.Tensor[[HIDDEN, sh_inter_local], pl.BF16],
        w_up: pl.Tensor[[HIDDEN, sh_inter_local], pl.BF16],
        w_down: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
        sh_y_shard: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    ):
        # Keep gate/up activation and down projection in one InCore kernel,
        # but never assemble a full [BATCH,160] Vec tile.  The wide tile is
        # miscompiled on A2/A3; five [BATCH,32] tiles match moe.py's
        # device-validated implementation.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_mlp"):
            x0_0 = pl.slice(
                x, [BATCH, SHARED_GATE_K_CHUNK], [0, 0],
            )
            wg0_0 = pl.slice(
                w_gate,
                [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                [0, 0],
            )
            wu0_0 = pl.slice(
                w_up,
                [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                [0, 0],
            )
            gate_acc_0 = pl.matmul(x0_0, wg0_0, out_dtype=pl.FP32)
            up_acc_0 = pl.matmul(x0_0, wu0_0, out_dtype=pl.FP32)
            for kb in pl.range(1, HIDDEN // SHARED_GATE_K_CHUNK):
                k0 = kb * SHARED_GATE_K_CHUNK
                xk_0 = pl.slice(
                    x, [BATCH, SHARED_GATE_K_CHUNK], [0, k0],
                )
                wgk_0 = pl.slice(
                    w_gate,
                    [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                    [k0, 0],
                )
                wuk_0 = pl.slice(
                    w_up,
                    [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                    [k0, 0],
                )
                gate_acc_0 = pl.matmul_acc(gate_acc_0, xk_0, wgk_0)
                up_acc_0 = pl.matmul_acc(up_acc_0, xk_0, wuk_0)
            sigmoid_0 = pl.recip(
                pl.add(pl.exp(pl.neg(gate_acc_0)), 1.0),
            )
            silu_0 = pl.mul(gate_acc_0, sigmoid_0)
            if _SHARED_SWIGLU_STEP:
                silu_c_0 = pl.minimum(
                    silu_0, _SHARED_SWIGLU_LIMIT,
                )
                up_c_0 = pl.maximum(
                    pl.minimum(up_acc_0, _SHARED_SWIGLU_LIMIT),
                    -_SHARED_SWIGLU_LIMIT,
                )
                gated_0 = pl.mul(silu_c_0, up_c_0)
            else:
                gated_0 = pl.mul(silu_0, up_acc_0)
            h_c0 = pl.cast(gated_0, target_type=pl.BF16)

            x0_1 = pl.slice(
                x, [BATCH, SHARED_GATE_K_CHUNK], [0, 0],
            )
            wg0_1 = pl.slice(
                w_gate,
                [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                [0, 32],
            )
            wu0_1 = pl.slice(
                w_up,
                [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                [0, 32],
            )
            gate_acc_1 = pl.matmul(x0_1, wg0_1, out_dtype=pl.FP32)
            up_acc_1 = pl.matmul(x0_1, wu0_1, out_dtype=pl.FP32)
            for kb in pl.range(1, HIDDEN // SHARED_GATE_K_CHUNK):
                k0 = kb * SHARED_GATE_K_CHUNK
                xk_1 = pl.slice(
                    x, [BATCH, SHARED_GATE_K_CHUNK], [0, k0],
                )
                wgk_1 = pl.slice(
                    w_gate,
                    [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                    [k0, 32],
                )
                wuk_1 = pl.slice(
                    w_up,
                    [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                    [k0, 32],
                )
                gate_acc_1 = pl.matmul_acc(gate_acc_1, xk_1, wgk_1)
                up_acc_1 = pl.matmul_acc(up_acc_1, xk_1, wuk_1)
            sigmoid_1 = pl.recip(
                pl.add(pl.exp(pl.neg(gate_acc_1)), 1.0),
            )
            silu_1 = pl.mul(gate_acc_1, sigmoid_1)
            if _SHARED_SWIGLU_STEP:
                silu_c_1 = pl.minimum(
                    silu_1, _SHARED_SWIGLU_LIMIT,
                )
                up_c_1 = pl.maximum(
                    pl.minimum(up_acc_1, _SHARED_SWIGLU_LIMIT),
                    -_SHARED_SWIGLU_LIMIT,
                )
                gated_1 = pl.mul(silu_c_1, up_c_1)
            else:
                gated_1 = pl.mul(silu_1, up_acc_1)
            h_c1 = pl.cast(gated_1, target_type=pl.BF16)

            x0_2 = pl.slice(
                x, [BATCH, SHARED_GATE_K_CHUNK], [0, 0],
            )
            wg0_2 = pl.slice(
                w_gate,
                [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                [0, 64],
            )
            wu0_2 = pl.slice(
                w_up,
                [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                [0, 64],
            )
            gate_acc_2 = pl.matmul(x0_2, wg0_2, out_dtype=pl.FP32)
            up_acc_2 = pl.matmul(x0_2, wu0_2, out_dtype=pl.FP32)
            for kb in pl.range(1, HIDDEN // SHARED_GATE_K_CHUNK):
                k0 = kb * SHARED_GATE_K_CHUNK
                xk_2 = pl.slice(
                    x, [BATCH, SHARED_GATE_K_CHUNK], [0, k0],
                )
                wgk_2 = pl.slice(
                    w_gate,
                    [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                    [k0, 64],
                )
                wuk_2 = pl.slice(
                    w_up,
                    [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                    [k0, 64],
                )
                gate_acc_2 = pl.matmul_acc(gate_acc_2, xk_2, wgk_2)
                up_acc_2 = pl.matmul_acc(up_acc_2, xk_2, wuk_2)
            sigmoid_2 = pl.recip(
                pl.add(pl.exp(pl.neg(gate_acc_2)), 1.0),
            )
            silu_2 = pl.mul(gate_acc_2, sigmoid_2)
            if _SHARED_SWIGLU_STEP:
                silu_c_2 = pl.minimum(
                    silu_2, _SHARED_SWIGLU_LIMIT,
                )
                up_c_2 = pl.maximum(
                    pl.minimum(up_acc_2, _SHARED_SWIGLU_LIMIT),
                    -_SHARED_SWIGLU_LIMIT,
                )
                gated_2 = pl.mul(silu_c_2, up_c_2)
            else:
                gated_2 = pl.mul(silu_2, up_acc_2)
            h_c2 = pl.cast(gated_2, target_type=pl.BF16)

            x0_3 = pl.slice(
                x, [BATCH, SHARED_GATE_K_CHUNK], [0, 0],
            )
            wg0_3 = pl.slice(
                w_gate,
                [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                [0, 96],
            )
            wu0_3 = pl.slice(
                w_up,
                [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                [0, 96],
            )
            gate_acc_3 = pl.matmul(x0_3, wg0_3, out_dtype=pl.FP32)
            up_acc_3 = pl.matmul(x0_3, wu0_3, out_dtype=pl.FP32)
            for kb in pl.range(1, HIDDEN // SHARED_GATE_K_CHUNK):
                k0 = kb * SHARED_GATE_K_CHUNK
                xk_3 = pl.slice(
                    x, [BATCH, SHARED_GATE_K_CHUNK], [0, k0],
                )
                wgk_3 = pl.slice(
                    w_gate,
                    [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                    [k0, 96],
                )
                wuk_3 = pl.slice(
                    w_up,
                    [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                    [k0, 96],
                )
                gate_acc_3 = pl.matmul_acc(gate_acc_3, xk_3, wgk_3)
                up_acc_3 = pl.matmul_acc(up_acc_3, xk_3, wuk_3)
            sigmoid_3 = pl.recip(
                pl.add(pl.exp(pl.neg(gate_acc_3)), 1.0),
            )
            silu_3 = pl.mul(gate_acc_3, sigmoid_3)
            if _SHARED_SWIGLU_STEP:
                silu_c_3 = pl.minimum(
                    silu_3, _SHARED_SWIGLU_LIMIT,
                )
                up_c_3 = pl.maximum(
                    pl.minimum(up_acc_3, _SHARED_SWIGLU_LIMIT),
                    -_SHARED_SWIGLU_LIMIT,
                )
                gated_3 = pl.mul(silu_c_3, up_c_3)
            else:
                gated_3 = pl.mul(silu_3, up_acc_3)
            h_c3 = pl.cast(gated_3, target_type=pl.BF16)

            x0_4 = pl.slice(
                x, [BATCH, SHARED_GATE_K_CHUNK], [0, 0],
            )
            wg0_4 = pl.slice(
                w_gate,
                [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                [0, 128],
            )
            wu0_4 = pl.slice(
                w_up,
                [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                [0, 128],
            )
            gate_acc_4 = pl.matmul(x0_4, wg0_4, out_dtype=pl.FP32)
            up_acc_4 = pl.matmul(x0_4, wu0_4, out_dtype=pl.FP32)
            for kb in pl.range(1, HIDDEN // SHARED_GATE_K_CHUNK):
                k0 = kb * SHARED_GATE_K_CHUNK
                xk_4 = pl.slice(
                    x, [BATCH, SHARED_GATE_K_CHUNK], [0, k0],
                )
                wgk_4 = pl.slice(
                    w_gate,
                    [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                    [k0, 128],
                )
                wuk_4 = pl.slice(
                    w_up,
                    [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                    [k0, 128],
                )
                gate_acc_4 = pl.matmul_acc(gate_acc_4, xk_4, wgk_4)
                up_acc_4 = pl.matmul_acc(up_acc_4, xk_4, wuk_4)
            sigmoid_4 = pl.recip(
                pl.add(pl.exp(pl.neg(gate_acc_4)), 1.0),
            )
            silu_4 = pl.mul(gate_acc_4, sigmoid_4)
            if _SHARED_SWIGLU_STEP:
                silu_c_4 = pl.minimum(
                    silu_4, _SHARED_SWIGLU_LIMIT,
                )
                up_c_4 = pl.maximum(
                    pl.minimum(up_acc_4, _SHARED_SWIGLU_LIMIT),
                    -_SHARED_SWIGLU_LIMIT,
                )
                gated_4 = pl.mul(silu_c_4, up_c_4)
            else:
                gated_4 = pl.mul(silu_4, up_acc_4)
            h_c4 = pl.cast(gated_4, target_type=pl.BF16)

            # Down projection consumes the five narrow activation tiles
            # directly; recombining them would restore the compiler bug.
            for db in pl.range(HIDDEN // SHARED_DOWN_N_CHUNK):
                d0 = db * SHARED_DOWN_N_CHUNK
                wd_c0 = pl.slice(
                    w_down,
                    [SHARED_SWIGLU_N_CHUNK, SHARED_DOWN_N_CHUNK],
                    [0, d0],
                )
                y_acc = pl.matmul(h_c0, wd_c0, out_dtype=pl.FP32)
                wd_c1 = pl.slice(
                    w_down,
                    [SHARED_SWIGLU_N_CHUNK, SHARED_DOWN_N_CHUNK],
                    [32, d0],
                )
                y_acc = pl.matmul_acc(y_acc, h_c1, wd_c1)
                wd_c2 = pl.slice(
                    w_down,
                    [SHARED_SWIGLU_N_CHUNK, SHARED_DOWN_N_CHUNK],
                    [64, d0],
                )
                y_acc = pl.matmul_acc(y_acc, h_c2, wd_c2)
                wd_c3 = pl.slice(
                    w_down,
                    [SHARED_SWIGLU_N_CHUNK, SHARED_DOWN_N_CHUNK],
                    [96, d0],
                )
                y_acc = pl.matmul_acc(y_acc, h_c3, wd_c3)
                wd_c4 = pl.slice(
                    w_down,
                    [SHARED_SWIGLU_N_CHUNK, SHARED_DOWN_N_CHUNK],
                    [128, d0],
                )
                y_acc = pl.matmul_acc(y_acc, h_c4, wd_c4)
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
            [COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
        sh_y = self._expert_shared_local(
            x, w_gate_s, w_up_s, w_down_s, sh_y,
        )
        # Phase 15.1 single-rank gate: skip TP=1 (mirror of 15.B).
        if TP_WORLD_SIZE > 1:
            sh_y = self.tp_all_reduce(
                sh_y, sh_tmp_window, sh_signal_window, my_rank,
            )
        return sh_y

    # ---------- Stage 4: combine (EP a2a back + weighted gather) ------
    # ---------- Stage 4: V4-Flash combine scatter/wait/reduce ----------
    @pl.function(type=pl.FunctionType.Inline)
    def combine_step(  # noqa: PLR0913, PLR0915
        self,
        local_routed_y: pl.Tensor[[local_recv_max, HIDDEN], pl.BF16],
        sh_y: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        moe_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
        combine_arrived: pld.DistributedTensor[
            [COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        local_route: pl.Tensor[[local_recv_max], pl.INT32],
        routed_y_buf: pld.DistributedTensor[
            [n_routes_per_rank, HIDDEN], pl.BF16
        ],
        local_expert_count: pl.Tensor[[n_local_experts], pl.INT32],
        local_expert_offset: pl.Tensor[[n_local_experts], pl.INT32],
        recv_meta_local: pl.Tensor[[n_ranks, n_local_experts_pad], pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
        moe_epoch: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
        with pl.spmd(
            n_local_experts,
            name_hint="combine_scatter",
            allow_early_resolve=True,
        ) as combine_scatter_tid:
            e = pl.tile.get_block_idx()
            expert_base = pl.cast(e * expert_recv_max, pl.INDEX)
            # ``local_route`` is the V4 route id (token * TOPK + k), not a
            # packed source-rank route.  Preserve source provenance from the
            # dispatch lane: rows are [expert, source, slot], and
            # recv_meta_local supplies the exact source prefix for each lane.
            source_prefix = pl.cast(0, pl.INDEX)
            for src in pl.range(n_ranks):
                src_count = pl.cast(
                    pl.read(recv_meta_local, [src, e]), pl.INDEX,
                )
                for slot in pl.range(src_count):
                    expert_row = expert_base + source_prefix + slot
                    route = pl.cast(
                        pl.read(local_route, [expert_row]), pl.INDEX,
                    )
                    pld.tensor.put(
                        dst=routed_y_buf,
                        peer=src,
                        src=local_routed_y,
                        dst_offsets=[route, 0],
                        src_offsets=[expert_row, 0],
                        shape=[1, HIDDEN],
                    )
                source_prefix = source_prefix + src_count
            for peer in pl.range(n_ranks):
                if peer != my_rank:
                    pld.system.notify(
                        target=combine_arrived,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )

        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="combine_wait",
            deps=[combine_scatter_tid],
            allow_early_resolve=True,
        ) as combine_wait_tid:
            _routed_anchor = pl.read(local_routed_y, [0, 0])
            for src in pl.range(n_ranks):
                if src != my_rank:
                    pld.system.wait(
                        signal=combine_arrived,
                        offsets=[src, 0],
                        expected=pl.cast(
                            moe_epoch * n_local_experts, pl.INT32,
                        ),
                        cmp=pld.WaitCmp.Ge,
                    )

        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > BATCH:
            active_tokens = pl.cast(BATCH, pl.INDEX)
        with pl.spmd(
            BATCH,
            name_hint="combine_reduce",
            deps=[combine_wait_tid],
            allow_early_resolve=True,
        ) as combine_reduce_tid:
            t = pl.tile.get_block_idx()
            if t < active_tokens:
                acc = pl.cast(
                    sh_y[t : t + 1, :], target_type=pl.FP32,
                )
                for k in pl.range(TOPK):
                    route = t * TOPK + k
                    acc = pl.add(
                        acc,
                        pl.cast(
                            routed_y_buf[route : route + 1, :],
                            target_type=pl.FP32,
                        ),
                    )
                moe_out[t : t + 1, :] = pl.cast(
                    acc, target_type=pl.BF16, mode="rint",
                )
            else:
                moe_out[t : t + 1, :] = sh_y[t : t + 1, :]
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
        resid_hold: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
        attn_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
        attn_signal_window: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        recv_meta: pld.DistributedTensor[
            [n_ranks, n_local_experts_pad], pl.INT32
        ],
        meta_arrived: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        recv_x: pld.DistributedTensor[[dispatch_lane_rows, HIDDEN], pl.INT8],
        recv_aux: pld.DistributedTensor[
            [dispatch_lane_rows, dispatch_aux_pad], pl.FP32
        ],
        recv_route: pld.DistributedTensor[
            [dispatch_lane_rows, idx_pad], pl.INT32
        ],
        data_arrived: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        sh_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
        sh_signal_window: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        combine_arrived: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        routed_y_buf: pld.DistributedTensor[[n_routes_per_rank, HIDDEN], pl.BF16],
        norm_layer_idx: pl.Scalar[pl.INT32],
        attn_layer_idx: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
        moe_epoch: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
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
            resid_hold, post_rms_weight, norm_layer_idx, moe_inv_rms,
            gate_w, router_bias, expert_indices, expert_weights, num_tokens,
        )

        # 2) Shared-expert lane (TP-sliced + tp_all_reduce).
        sh_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        sh_y = self.expert_shared_step(
            post_norm, w_gate_s, w_up_s, w_down_s, sh_y,
            sh_tmp_window, sh_signal_window, my_rank,
        )


        # 3) Dispatch (V4-Flash expert-lane push/gather).
        local_routed_x = pl.create_tensor(
            [local_recv_max, HIDDEN], dtype=pl.INT8,
        )
        local_routed_x_scale = pl.create_tensor(
            [1, local_recv_max], dtype=pl.FP32,
        )
        local_routed_weight = pl.create_tensor(
            [local_recv_max], dtype=pl.FP32,
        )
        local_route = pl.create_tensor(
            [local_recv_max], dtype=pl.INT32,
        )
        local_expert_offset = pl.create_tensor(
            [n_local_experts], dtype=pl.INT32,
        )
        local_expert_count = pl.create_tensor(
            [n_local_experts], dtype=pl.INT32,
        )
        (
            local_routed_x,
            local_routed_x_scale,
            local_routed_weight,
            local_route,
            local_expert_offset,
            local_expert_count,
            recv_meta_local,
        ) = self.dispatch_step(
            x_disp_i8, x_disp_scale, expert_indices, expert_weights,
            local_routed_x, local_routed_x_scale,
            local_routed_weight, local_route,
            local_expert_offset, local_expert_count,
            recv_meta, meta_arrived, recv_x, recv_aux,
            recv_route, data_arrived,
            num_tokens, my_rank, moe_epoch,
        )

        # 4) Routed experts (local 36).
        local_routed_y = pl.create_tensor(
            [local_recv_max, HIDDEN], dtype=pl.BF16,
        )
        local_routed_y = self.expert_routed_step(
            local_routed_x,
            local_routed_x_scale,
            local_routed_weight,
            local_expert_offset, local_expert_count,
            w_gate_r, w_gate_r_scale, w_up_r, w_up_r_scale,
            w_down_r, w_down_r_scale,
            local_routed_y,
        )

        # 5) Combine (expert-lane scatter + arrival + FP32 token reduce).
        moe_out = self.combine_step(
            local_routed_y,
            sh_y,
            moe_out,
            combine_arrived,
            local_route, routed_y_buf,
            local_expert_count, local_expert_offset, recv_meta_local,
            num_tokens, my_rank, moe_epoch,
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
        resid_hold: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
        attn_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
        attn_signal_window: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        recv_meta: pld.DistributedTensor[
            [n_ranks, n_local_experts_pad], pl.INT32
        ],
        meta_arrived: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        recv_x: pld.DistributedTensor[[dispatch_lane_rows, HIDDEN], pl.INT8],
        recv_aux: pld.DistributedTensor[
            [dispatch_lane_rows, dispatch_aux_pad], pl.FP32
        ],
        recv_route: pld.DistributedTensor[
            [dispatch_lane_rows, idx_pad], pl.INT32
        ],
        data_arrived: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        sh_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
        sh_signal_window: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        combine_arrived: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        routed_y_buf: pld.DistributedTensor[[n_routes_per_rank, HIDDEN], pl.BF16],
        norm_layer_idx: pl.Scalar[pl.INT32],
        attn_layer_idx: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
        moe_epoch: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
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
            resid_hold, post_rms_weight, norm_layer_idx, moe_inv_rms,
            gate_w, router_bias, expert_indices, expert_weights, num_tokens,
        )

        # 2) Shared-expert lane (TP-sliced + tp_all_reduce).
        sh_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        sh_y = self.expert_shared_step(
            post_norm, w_gate_s, w_up_s, w_down_s, sh_y,
            sh_tmp_window, sh_signal_window, my_rank,
        )


        # 3) Dispatch (V4-Flash expert-lane push/gather).
        local_routed_x = pl.create_tensor(
            [local_recv_max, HIDDEN], dtype=pl.INT8,
        )
        local_routed_x_scale = pl.create_tensor(
            [1, local_recv_max], dtype=pl.FP32,
        )
        local_routed_weight = pl.create_tensor(
            [local_recv_max], dtype=pl.FP32,
        )
        local_route = pl.create_tensor(
            [local_recv_max], dtype=pl.INT32,
        )
        local_expert_offset = pl.create_tensor(
            [n_local_experts], dtype=pl.INT32,
        )
        local_expert_count = pl.create_tensor(
            [n_local_experts], dtype=pl.INT32,
        )
        (
            local_routed_x,
            local_routed_x_scale,
            local_routed_weight,
            local_route,
            local_expert_offset,
            local_expert_count,
            recv_meta_local,
        ) = self.dispatch_step(
            x_disp_i8, x_disp_scale, expert_indices, expert_weights,
            local_routed_x, local_routed_x_scale,
            local_routed_weight, local_route,
            local_expert_offset, local_expert_count,
            recv_meta, meta_arrived, recv_x, recv_aux,
            recv_route, data_arrived,
            num_tokens, my_rank, moe_epoch,
        )

        # 4) Routed experts (local 36).
        local_routed_y = pl.create_tensor(
            [local_recv_max, HIDDEN], dtype=pl.BF16,
        )
        local_routed_y = self.expert_routed_step(
            local_routed_x,
            local_routed_x_scale,
            local_routed_weight,
            local_expert_offset, local_expert_count,
            w_gate_r, w_gate_r_scale, w_up_r, w_up_r_scale,
            w_down_r, w_down_r_scale,
            local_routed_y,
        )

        # 5) Combine (expert-lane scatter + arrival + FP32 token reduce).
        moe_out = self.combine_step(
            local_routed_y,
            sh_y,
            moe_out,
            combine_arrived,
            local_route, routed_y_buf,
            local_expert_count, local_expert_offset, recv_meta_local,
            num_tokens, my_rank, moe_epoch,
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
    @pl.function(type=pl.FunctionType.Inline)
    def _expert_routed_swiglu7(  # noqa: PLR0913, PLR0915
        self,
        local_routed_x: pl.Tensor[[local_recv_max, HIDDEN], pl.INT8],
        local_routed_x_scale: pl.Tensor[[1, local_recv_max], pl.FP32],
        local_routed_weight: pl.Tensor[[local_recv_max], pl.FP32],
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
        for e in pl.range(n_local_experts):
            n_rows = pl.read(local_expert_count, [e])
            offset = pl.cast(e * expert_recv_max, pl.INDEX)
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
                        if _ROUTED_SWIGLU7_STEP:
                            silu_c = pl.minimum(silu, _ROUTED_SWIGLU7_LIMIT)
                            up_c = pl.maximum(
                                pl.minimum(up_2d, _ROUTED_SWIGLU7_LIMIT),
                                -_ROUTED_SWIGLU7_LIMIT,
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
                        route_weight = pl.reshape(
                            pl.slice(
                                local_routed_weight,
                                [RECV_TILE],
                                [tile_offset],
                                valid_shape=[tile_valid],
                            ),
                            [RECV_TILE, 1],
                        )
                        y_2d = pl.col_expand_mul(
                            pl.row_expand_mul(
                                pl.cast(
                                    y_acc, target_type=pl.FP32, mode="none",
                                ),
                                pl.mul(h_scale_dq, route_weight),
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
    def expert_routed_step_swiglu7(
        self,
        local_routed_x: pl.Tensor[[local_recv_max, HIDDEN], pl.INT8],
        local_routed_x_scale: pl.Tensor[[1, local_recv_max], pl.FP32],
        local_routed_weight: pl.Tensor[[local_recv_max], pl.FP32],
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
        local_routed_y = self._expert_routed_swiglu7(
            local_routed_x,
            local_routed_x_scale,
            local_routed_weight,
            local_expert_offset, local_expert_count,
            w_gate_r, w_gate_r_scale,
            w_up_r, w_up_r_scale,
            w_down_r, w_down_r_scale,
            local_routed_y,
        )
        return local_routed_y

    @pl.function(type=pl.FunctionType.Inline)
    def _expert_shared_local_swiglu16(
        self,
        x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        w_gate: pl.Tensor[[HIDDEN, sh_inter_local], pl.BF16],
        w_up: pl.Tensor[[HIDDEN, sh_inter_local], pl.BF16],
        w_down: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
        sh_y_shard: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    ):
        # Keep gate/up activation and down projection in one InCore kernel,
        # but never assemble a full [BATCH,160] Vec tile.  The wide tile is
        # miscompiled on A2/A3; five [BATCH,32] tiles match moe.py's
        # device-validated implementation.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_mlp"):
            x0_0 = pl.slice(
                x, [BATCH, SHARED_GATE_K_CHUNK], [0, 0],
            )
            wg0_0 = pl.slice(
                w_gate,
                [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                [0, 0],
            )
            wu0_0 = pl.slice(
                w_up,
                [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                [0, 0],
            )
            gate_acc_0 = pl.matmul(x0_0, wg0_0, out_dtype=pl.FP32)
            up_acc_0 = pl.matmul(x0_0, wu0_0, out_dtype=pl.FP32)
            for kb in pl.range(1, HIDDEN // SHARED_GATE_K_CHUNK):
                k0 = kb * SHARED_GATE_K_CHUNK
                xk_0 = pl.slice(
                    x, [BATCH, SHARED_GATE_K_CHUNK], [0, k0],
                )
                wgk_0 = pl.slice(
                    w_gate,
                    [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                    [k0, 0],
                )
                wuk_0 = pl.slice(
                    w_up,
                    [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                    [k0, 0],
                )
                gate_acc_0 = pl.matmul_acc(gate_acc_0, xk_0, wgk_0)
                up_acc_0 = pl.matmul_acc(up_acc_0, xk_0, wuk_0)
            sigmoid_0 = pl.recip(
                pl.add(pl.exp(pl.neg(gate_acc_0)), 1.0),
            )
            silu_0 = pl.mul(gate_acc_0, sigmoid_0)
            if _SHARED_SWIGLU16_STEP:
                silu_c_0 = pl.minimum(
                    silu_0, _SHARED_SWIGLU16_LIMIT,
                )
                up_c_0 = pl.maximum(
                    pl.minimum(up_acc_0, _SHARED_SWIGLU16_LIMIT),
                    -_SHARED_SWIGLU16_LIMIT,
                )
                gated_0 = pl.mul(silu_c_0, up_c_0)
            else:
                gated_0 = pl.mul(silu_0, up_acc_0)
            h_c0 = pl.cast(gated_0, target_type=pl.BF16)

            x0_1 = pl.slice(
                x, [BATCH, SHARED_GATE_K_CHUNK], [0, 0],
            )
            wg0_1 = pl.slice(
                w_gate,
                [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                [0, 32],
            )
            wu0_1 = pl.slice(
                w_up,
                [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                [0, 32],
            )
            gate_acc_1 = pl.matmul(x0_1, wg0_1, out_dtype=pl.FP32)
            up_acc_1 = pl.matmul(x0_1, wu0_1, out_dtype=pl.FP32)
            for kb in pl.range(1, HIDDEN // SHARED_GATE_K_CHUNK):
                k0 = kb * SHARED_GATE_K_CHUNK
                xk_1 = pl.slice(
                    x, [BATCH, SHARED_GATE_K_CHUNK], [0, k0],
                )
                wgk_1 = pl.slice(
                    w_gate,
                    [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                    [k0, 32],
                )
                wuk_1 = pl.slice(
                    w_up,
                    [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                    [k0, 32],
                )
                gate_acc_1 = pl.matmul_acc(gate_acc_1, xk_1, wgk_1)
                up_acc_1 = pl.matmul_acc(up_acc_1, xk_1, wuk_1)
            sigmoid_1 = pl.recip(
                pl.add(pl.exp(pl.neg(gate_acc_1)), 1.0),
            )
            silu_1 = pl.mul(gate_acc_1, sigmoid_1)
            if _SHARED_SWIGLU16_STEP:
                silu_c_1 = pl.minimum(
                    silu_1, _SHARED_SWIGLU16_LIMIT,
                )
                up_c_1 = pl.maximum(
                    pl.minimum(up_acc_1, _SHARED_SWIGLU16_LIMIT),
                    -_SHARED_SWIGLU16_LIMIT,
                )
                gated_1 = pl.mul(silu_c_1, up_c_1)
            else:
                gated_1 = pl.mul(silu_1, up_acc_1)
            h_c1 = pl.cast(gated_1, target_type=pl.BF16)

            x0_2 = pl.slice(
                x, [BATCH, SHARED_GATE_K_CHUNK], [0, 0],
            )
            wg0_2 = pl.slice(
                w_gate,
                [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                [0, 64],
            )
            wu0_2 = pl.slice(
                w_up,
                [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                [0, 64],
            )
            gate_acc_2 = pl.matmul(x0_2, wg0_2, out_dtype=pl.FP32)
            up_acc_2 = pl.matmul(x0_2, wu0_2, out_dtype=pl.FP32)
            for kb in pl.range(1, HIDDEN // SHARED_GATE_K_CHUNK):
                k0 = kb * SHARED_GATE_K_CHUNK
                xk_2 = pl.slice(
                    x, [BATCH, SHARED_GATE_K_CHUNK], [0, k0],
                )
                wgk_2 = pl.slice(
                    w_gate,
                    [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                    [k0, 64],
                )
                wuk_2 = pl.slice(
                    w_up,
                    [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                    [k0, 64],
                )
                gate_acc_2 = pl.matmul_acc(gate_acc_2, xk_2, wgk_2)
                up_acc_2 = pl.matmul_acc(up_acc_2, xk_2, wuk_2)
            sigmoid_2 = pl.recip(
                pl.add(pl.exp(pl.neg(gate_acc_2)), 1.0),
            )
            silu_2 = pl.mul(gate_acc_2, sigmoid_2)
            if _SHARED_SWIGLU16_STEP:
                silu_c_2 = pl.minimum(
                    silu_2, _SHARED_SWIGLU16_LIMIT,
                )
                up_c_2 = pl.maximum(
                    pl.minimum(up_acc_2, _SHARED_SWIGLU16_LIMIT),
                    -_SHARED_SWIGLU16_LIMIT,
                )
                gated_2 = pl.mul(silu_c_2, up_c_2)
            else:
                gated_2 = pl.mul(silu_2, up_acc_2)
            h_c2 = pl.cast(gated_2, target_type=pl.BF16)

            x0_3 = pl.slice(
                x, [BATCH, SHARED_GATE_K_CHUNK], [0, 0],
            )
            wg0_3 = pl.slice(
                w_gate,
                [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                [0, 96],
            )
            wu0_3 = pl.slice(
                w_up,
                [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                [0, 96],
            )
            gate_acc_3 = pl.matmul(x0_3, wg0_3, out_dtype=pl.FP32)
            up_acc_3 = pl.matmul(x0_3, wu0_3, out_dtype=pl.FP32)
            for kb in pl.range(1, HIDDEN // SHARED_GATE_K_CHUNK):
                k0 = kb * SHARED_GATE_K_CHUNK
                xk_3 = pl.slice(
                    x, [BATCH, SHARED_GATE_K_CHUNK], [0, k0],
                )
                wgk_3 = pl.slice(
                    w_gate,
                    [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                    [k0, 96],
                )
                wuk_3 = pl.slice(
                    w_up,
                    [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                    [k0, 96],
                )
                gate_acc_3 = pl.matmul_acc(gate_acc_3, xk_3, wgk_3)
                up_acc_3 = pl.matmul_acc(up_acc_3, xk_3, wuk_3)
            sigmoid_3 = pl.recip(
                pl.add(pl.exp(pl.neg(gate_acc_3)), 1.0),
            )
            silu_3 = pl.mul(gate_acc_3, sigmoid_3)
            if _SHARED_SWIGLU16_STEP:
                silu_c_3 = pl.minimum(
                    silu_3, _SHARED_SWIGLU16_LIMIT,
                )
                up_c_3 = pl.maximum(
                    pl.minimum(up_acc_3, _SHARED_SWIGLU16_LIMIT),
                    -_SHARED_SWIGLU16_LIMIT,
                )
                gated_3 = pl.mul(silu_c_3, up_c_3)
            else:
                gated_3 = pl.mul(silu_3, up_acc_3)
            h_c3 = pl.cast(gated_3, target_type=pl.BF16)

            x0_4 = pl.slice(
                x, [BATCH, SHARED_GATE_K_CHUNK], [0, 0],
            )
            wg0_4 = pl.slice(
                w_gate,
                [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                [0, 128],
            )
            wu0_4 = pl.slice(
                w_up,
                [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                [0, 128],
            )
            gate_acc_4 = pl.matmul(x0_4, wg0_4, out_dtype=pl.FP32)
            up_acc_4 = pl.matmul(x0_4, wu0_4, out_dtype=pl.FP32)
            for kb in pl.range(1, HIDDEN // SHARED_GATE_K_CHUNK):
                k0 = kb * SHARED_GATE_K_CHUNK
                xk_4 = pl.slice(
                    x, [BATCH, SHARED_GATE_K_CHUNK], [0, k0],
                )
                wgk_4 = pl.slice(
                    w_gate,
                    [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                    [k0, 128],
                )
                wuk_4 = pl.slice(
                    w_up,
                    [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK],
                    [k0, 128],
                )
                gate_acc_4 = pl.matmul_acc(gate_acc_4, xk_4, wgk_4)
                up_acc_4 = pl.matmul_acc(up_acc_4, xk_4, wuk_4)
            sigmoid_4 = pl.recip(
                pl.add(pl.exp(pl.neg(gate_acc_4)), 1.0),
            )
            silu_4 = pl.mul(gate_acc_4, sigmoid_4)
            if _SHARED_SWIGLU16_STEP:
                silu_c_4 = pl.minimum(
                    silu_4, _SHARED_SWIGLU16_LIMIT,
                )
                up_c_4 = pl.maximum(
                    pl.minimum(up_acc_4, _SHARED_SWIGLU16_LIMIT),
                    -_SHARED_SWIGLU16_LIMIT,
                )
                gated_4 = pl.mul(silu_c_4, up_c_4)
            else:
                gated_4 = pl.mul(silu_4, up_acc_4)
            h_c4 = pl.cast(gated_4, target_type=pl.BF16)

            # Down projection consumes the five narrow activation tiles
            # directly; recombining them would restore the compiler bug.
            for db in pl.range(HIDDEN // SHARED_DOWN_N_CHUNK):
                d0 = db * SHARED_DOWN_N_CHUNK
                wd_c0 = pl.slice(
                    w_down,
                    [SHARED_SWIGLU_N_CHUNK, SHARED_DOWN_N_CHUNK],
                    [0, d0],
                )
                y_acc = pl.matmul(h_c0, wd_c0, out_dtype=pl.FP32)
                wd_c1 = pl.slice(
                    w_down,
                    [SHARED_SWIGLU_N_CHUNK, SHARED_DOWN_N_CHUNK],
                    [32, d0],
                )
                y_acc = pl.matmul_acc(y_acc, h_c1, wd_c1)
                wd_c2 = pl.slice(
                    w_down,
                    [SHARED_SWIGLU_N_CHUNK, SHARED_DOWN_N_CHUNK],
                    [64, d0],
                )
                y_acc = pl.matmul_acc(y_acc, h_c2, wd_c2)
                wd_c3 = pl.slice(
                    w_down,
                    [SHARED_SWIGLU_N_CHUNK, SHARED_DOWN_N_CHUNK],
                    [96, d0],
                )
                y_acc = pl.matmul_acc(y_acc, h_c3, wd_c3)
                wd_c4 = pl.slice(
                    w_down,
                    [SHARED_SWIGLU_N_CHUNK, SHARED_DOWN_N_CHUNK],
                    [128, d0],
                )
                y_acc = pl.matmul_acc(y_acc, h_c4, wd_c4)
                sh_y_shard = pl.assemble(
                    sh_y_shard,
                    pl.cast(y_acc, target_type=pl.BF16),
                    [0, d0],
                )

        return sh_y_shard

    @pl.function(type=pl.FunctionType.Inline)
    def expert_shared_step_swiglu16(  # noqa: PLR0913
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
            [COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
        sh_y = self._expert_shared_local_swiglu16(
            x, w_gate_s, w_up_s, w_down_s, sh_y,
        )
        # Phase 15.1 single-rank gate: skip TP=1 (mirror of 15.B).
        if TP_WORLD_SIZE > 1:
            sh_y = self.tp_all_reduce(
                sh_y, sh_tmp_window, sh_signal_window, my_rank,
            )
        return sh_y

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
        resid_hold: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
        attn_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
        attn_signal_window: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        recv_meta: pld.DistributedTensor[
            [n_ranks, n_local_experts_pad], pl.INT32
        ],
        meta_arrived: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        recv_x: pld.DistributedTensor[[dispatch_lane_rows, HIDDEN], pl.INT8],
        recv_aux: pld.DistributedTensor[
            [dispatch_lane_rows, dispatch_aux_pad], pl.FP32
        ],
        recv_route: pld.DistributedTensor[
            [dispatch_lane_rows, idx_pad], pl.INT32
        ],
        data_arrived: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        sh_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
        sh_signal_window: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        combine_arrived: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        routed_y_buf: pld.DistributedTensor[[n_routes_per_rank, HIDDEN], pl.BF16],
        norm_layer_idx: pl.Scalar[pl.INT32],
        attn_layer_idx: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
        moe_epoch: pl.Scalar[pl.INT32],
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
            resid_hold, post_rms_weight, norm_layer_idx, moe_inv_rms,
            gate_w, router_bias, expert_indices, expert_weights, num_tokens,
        )

        # 2) Shared-expert lane (TP-sliced + tp_all_reduce).
        sh_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        sh_y = self.expert_shared_step_swiglu16(
            post_norm, w_gate_s, w_up_s, w_down_s, sh_y,
            sh_tmp_window, sh_signal_window, my_rank,
        )


        # 3) Dispatch (V4-Flash expert-lane push/gather).
        local_routed_x = pl.create_tensor(
            [local_recv_max, HIDDEN], dtype=pl.INT8,
        )
        local_routed_x_scale = pl.create_tensor(
            [1, local_recv_max], dtype=pl.FP32,
        )
        local_routed_weight = pl.create_tensor(
            [local_recv_max], dtype=pl.FP32,
        )
        local_route = pl.create_tensor(
            [local_recv_max], dtype=pl.INT32,
        )
        local_expert_offset = pl.create_tensor(
            [n_local_experts], dtype=pl.INT32,
        )
        local_expert_count = pl.create_tensor(
            [n_local_experts], dtype=pl.INT32,
        )
        (
            local_routed_x,
            local_routed_x_scale,
            local_routed_weight,
            local_route,
            local_expert_offset,
            local_expert_count,
            recv_meta_local,
        ) = self.dispatch_step(
            x_disp_i8, x_disp_scale, expert_indices, expert_weights,
            local_routed_x, local_routed_x_scale,
            local_routed_weight, local_route,
            local_expert_offset, local_expert_count,
            recv_meta, meta_arrived, recv_x, recv_aux,
            recv_route, data_arrived,
            num_tokens, my_rank, moe_epoch,
        )

        # 4) Routed experts (local 36).
        local_routed_y = pl.create_tensor(
            [local_recv_max, HIDDEN], dtype=pl.BF16,
        )
        local_routed_y = self.expert_routed_step_swiglu7(
            local_routed_x,
            local_routed_x_scale,
            local_routed_weight,
            local_expert_offset, local_expert_count,
            w_gate_r, w_gate_r_scale, w_up_r, w_up_r_scale,
            w_down_r, w_down_r_scale,
            local_routed_y,
        )

        # 5) Combine (expert-lane scatter + arrival + FP32 token reduce).
        moe_out = self.combine_step(
            local_routed_y,
            sh_y,
            moe_out,
            combine_arrived,
            local_route, routed_y_buf,
            local_expert_count, local_expert_offset, recv_meta_local,
            num_tokens, my_rank, moe_epoch,
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
        resid_hold: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
        attn_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
        attn_signal_window: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        recv_meta: pld.DistributedTensor[
            [n_ranks, n_local_experts_pad], pl.INT32
        ],
        meta_arrived: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        recv_x: pld.DistributedTensor[[dispatch_lane_rows, HIDDEN], pl.INT8],
        recv_aux: pld.DistributedTensor[
            [dispatch_lane_rows, dispatch_aux_pad], pl.FP32
        ],
        recv_route: pld.DistributedTensor[
            [dispatch_lane_rows, idx_pad], pl.INT32
        ],
        data_arrived: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        sh_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
        sh_signal_window: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        combine_arrived: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        routed_y_buf: pld.DistributedTensor[[n_routes_per_rank, HIDDEN], pl.BF16],
        norm_layer_idx: pl.Scalar[pl.INT32],
        attn_layer_idx: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
        moe_epoch: pl.Scalar[pl.INT32],
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
            resid_hold, post_rms_weight, norm_layer_idx, moe_inv_rms,
            gate_w, router_bias, expert_indices, expert_weights, num_tokens,
        )

        # 2) Shared-expert lane (TP-sliced + tp_all_reduce).
        sh_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        sh_y = self.expert_shared_step(
            post_norm, w_gate_s, w_up_s, w_down_s, sh_y,
            sh_tmp_window, sh_signal_window, my_rank,
        )


        # 3) Dispatch (V4-Flash expert-lane push/gather).
        local_routed_x = pl.create_tensor(
            [local_recv_max, HIDDEN], dtype=pl.INT8,
        )
        local_routed_x_scale = pl.create_tensor(
            [1, local_recv_max], dtype=pl.FP32,
        )
        local_routed_weight = pl.create_tensor(
            [local_recv_max], dtype=pl.FP32,
        )
        local_route = pl.create_tensor(
            [local_recv_max], dtype=pl.INT32,
        )
        local_expert_offset = pl.create_tensor(
            [n_local_experts], dtype=pl.INT32,
        )
        local_expert_count = pl.create_tensor(
            [n_local_experts], dtype=pl.INT32,
        )
        (
            local_routed_x,
            local_routed_x_scale,
            local_routed_weight,
            local_route,
            local_expert_offset,
            local_expert_count,
            recv_meta_local,
        ) = self.dispatch_step(
            x_disp_i8, x_disp_scale, expert_indices, expert_weights,
            local_routed_x, local_routed_x_scale,
            local_routed_weight, local_route,
            local_expert_offset, local_expert_count,
            recv_meta, meta_arrived, recv_x, recv_aux,
            recv_route, data_arrived,
            num_tokens, my_rank, moe_epoch,
        )

        # 4) Routed experts (local 36).
        local_routed_y = pl.create_tensor(
            [local_recv_max, HIDDEN], dtype=pl.BF16,
        )
        local_routed_y = self.expert_routed_step_swiglu7(
            local_routed_x,
            local_routed_x_scale,
            local_routed_weight,
            local_expert_offset, local_expert_count,
            w_gate_r, w_gate_r_scale, w_up_r, w_up_r_scale,
            w_down_r, w_down_r_scale,
            local_routed_y,
        )

        # 5) Combine (expert-lane scatter + arrival + FP32 token reduce).
        moe_out = self.combine_step(
            local_routed_y,
            sh_y,
            moe_out,
            combine_arrived,
            local_route, routed_y_buf,
            local_expert_count, local_expert_offset, recv_meta_local,
            num_tokens, my_rank, moe_epoch,
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
        # shared MoE weights (all 40 layers share the same shared-expert weights;
        # step3p5 shared expert is replicated, not per-layer — pass once, no stack)
        moe_gate_w: pl.Tensor[[NUM_MOE_LAYERS_TOTAL * HIDDEN, N_EXPERTS], pl.FP32],
        moe_router_bias: pl.Tensor[[NUM_MOE_LAYERS_TOTAL * N_EXPERTS], pl.FP32],
        moe_w_gate_r: pl.Tensor[[NUM_MOE_LAYERS_TOTAL * n_local_experts * HIDDEN, inter], pl.INT8],
        moe_w_gate_r_scale: pl.Tensor[[NUM_MOE_LAYERS_TOTAL * n_local_experts, inter], pl.FP32],
        moe_w_up_r: pl.Tensor[[NUM_MOE_LAYERS_TOTAL * n_local_experts * HIDDEN, inter], pl.INT8],
        moe_w_up_r_scale: pl.Tensor[[NUM_MOE_LAYERS_TOTAL * n_local_experts, inter], pl.FP32],
        moe_w_down_r: pl.Tensor[[NUM_MOE_LAYERS_TOTAL * n_local_experts * inter, HIDDEN], pl.INT8],
        moe_w_down_r_scale: pl.Tensor[[NUM_MOE_LAYERS_TOTAL * n_local_experts, HIDDEN], pl.FP32],
        moe_w_gate_s: pl.Tensor[[NUM_MOE_LAYERS_TOTAL * HIDDEN, sh_inter_local], pl.BF16],
        moe_w_up_s: pl.Tensor[[NUM_MOE_LAYERS_TOTAL * HIDDEN, sh_inter_local], pl.BF16],
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
        # ── MoE communication windows ──
        # Attention/shared TP all-reduce scratch stays per-layer because its
        # fixed expected=1/2 protocol is not epoch-aware. EP dispatch/combine
        # data windows are one shared set, protected by independent monotonic
        # metadata, payload, and combine arrival lineages.
        moe_attn_tmp_stack: pld.DistributedTensor[
            [NUM_MOE_LAYERS_TOTAL * BATCH, HIDDEN], pl.BF16
        ],
        moe_attn_signal_stack: pld.DistributedTensor[
            [NUM_MOE_LAYERS_TOTAL * COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        moe_recv_meta_stack: pld.DistributedTensor[
            [n_ranks, n_local_experts_pad], pl.INT32
        ],
        moe_meta_arrived_stack: pld.DistributedTensor[
            [COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        moe_recv_x_stack: pld.DistributedTensor[
            [dispatch_lane_rows, HIDDEN], pl.INT8
        ],
        moe_recv_aux_stack: pld.DistributedTensor[
            [dispatch_lane_rows, dispatch_aux_pad], pl.FP32
        ],
        moe_recv_route_stack: pld.DistributedTensor[
            [dispatch_lane_rows, idx_pad], pl.INT32
        ],
        moe_data_arrived_stack: pld.DistributedTensor[
            [COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        moe_sh_tmp_stack: pld.DistributedTensor[
            [NUM_MOE_LAYERS_TOTAL * BATCH, HIDDEN], pl.BF16
        ],
        moe_sh_signal_stack: pld.DistributedTensor[
            [NUM_MOE_LAYERS_TOTAL * COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        moe_combine_arrived_stack: pld.DistributedTensor[
            [COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        moe_routed_y_buf_stack: pld.DistributedTensor[
            [n_routes_per_rank, HIDDEN], pl.BF16
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
        # C1 protocol: EP dispatch/combine reuse one V4-Flash-style window
        # set across MoE epochs. ``moe_epoch`` tags metadata, payload, and
        # combine arrival lineages; a slot is reusable only after its final
        # semantic consumer from the previous epoch has completed. This is
        # distinct from the per-layer attention/shared TP all-reduce scratch,
        # whose expected count remains fixed at 1/2. The 512B stride is only a
        # step3p5 backend/profile isolation for stacked/reused control slots,
        # not a general DeepSeek signal/window ABI.
        for layer_idx in pl.range(NUM_MOE_LAYERS):
            phys_layer = layer_idx + 3
            norm_layer_idx = pl.cast(phys_layer, pl.INT32)
            # MoE weight/window offset = layer_idx * slot (0-indexed into the
            # 40-layer stacks).
            moe_w_off = layer_idx * HIDDEN
            moe_bias_off = layer_idx * N_EXPERTS
            moe_r_off = layer_idx * (n_local_experts * HIDDEN)
            moe_r_down_off = layer_idx * (n_local_experts * inter)
            moe_r_scale_off = layer_idx * n_local_experts
            moe_sh_down_off = layer_idx * sh_inter_local
            moe_win_off = layer_idx * BATCH
            moe_sig_off = layer_idx * COMM_SIGNAL_STRIDE_I32
            # Pass the invocation and layer-local epochs as independent
            # scalar variables.  The InCore pull helpers form the absolute
            # epoch there; orchestration codegen cannot lower a compound
            # arithmetic expression directly as a call argument.
            moe_epoch = pl.cast(layer_idx + 1, pl.INT32)
            h_moe = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            resid_hold_moe = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            # full_moe at layer_idx % 4 == 1 (physical 4,8,12,...). full_idx =
            # (layer_idx - 1) // 4 maps {1,5,9,...} -> {0,1,2,...,9}.
            # swa_idx = layer_idx - full_count_so_far; computed inside branch.
            if layer_idx % 4 == 1:
                full_idx = (layer_idx - 1) // 4
                fa_w_off = full_idx * HIDDEN
                fa_wo_off = full_idx * hidden_q_full
                fa_gate_r_off = full_idx * nh_full_pad
                h_moe = self.full_moe_chip_orch(
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
                    pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [moe_w_off, 0]),
                    pl.slice(moe_router_bias, [N_EXPERTS], [moe_bias_off]),
                    pl.reshape(
                        pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [moe_r_off, 0]),
                        [n_local_experts, HIDDEN, inter],
                    ),
                    pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [moe_r_scale_off, 0]),
                    pl.reshape(
                        pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [moe_r_off, 0]),
                        [n_local_experts, HIDDEN, inter],
                    ),
                    pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [moe_r_scale_off, 0]),
                    pl.reshape(
                        pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [moe_r_down_off, 0]),
                        [n_local_experts, inter, HIDDEN],
                    ),
                    pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [moe_r_scale_off, 0]),
                    pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [moe_w_off, 0]),
                    pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [moe_w_off, 0]),
                    pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [moe_sh_down_off, 0]),
                    h_moe,
                    resid_hold_moe,
                    pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [moe_win_off, 0]),
                    pl.slice(moe_attn_signal_stack, [COMM_SIGNAL_STRIDE_I32, 1], [moe_sig_off, 0]),
                    pl.slice(moe_recv_meta_stack, [n_ranks, n_local_experts_pad], [0, 0]),
                    pl.slice(moe_meta_arrived_stack, [COMM_SIGNAL_STRIDE_I32, 1], [0, 0]),
                    pl.slice(moe_recv_x_stack, [dispatch_lane_rows, HIDDEN], [0, 0]),
                    pl.slice(moe_recv_aux_stack, [dispatch_lane_rows, dispatch_aux_pad], [0, 0]),
                    pl.slice(moe_recv_route_stack, [dispatch_lane_rows, idx_pad], [0, 0]),
                    pl.slice(moe_data_arrived_stack, [COMM_SIGNAL_STRIDE_I32, 1], [0, 0]),
                            pl.slice(moe_sh_tmp_stack, [BATCH, HIDDEN], [moe_win_off, 0]),
                    pl.slice(moe_sh_signal_stack, [COMM_SIGNAL_STRIDE_I32, 1], [moe_sig_off, 0]),
                            pl.slice(moe_combine_arrived_stack, [COMM_SIGNAL_STRIDE_I32, 1], [0, 0]),
                    pl.slice(moe_routed_y_buf_stack, [n_routes_per_rank, HIDDEN], [0, 0]),
                    norm_layer_idx,
                    0,
                    num_tokens,
                    my_rank,
                    moe_epoch,
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
                h_moe = self.swa_moe_chip_orch(
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
                    pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [moe_w_off, 0]),
                    pl.slice(moe_router_bias, [N_EXPERTS], [moe_bias_off]),
                    pl.reshape(
                        pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [moe_r_off, 0]),
                        [n_local_experts, HIDDEN, inter],
                    ),
                    pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [moe_r_scale_off, 0]),
                    pl.reshape(
                        pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [moe_r_off, 0]),
                        [n_local_experts, HIDDEN, inter],
                    ),
                    pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [moe_r_scale_off, 0]),
                    pl.reshape(
                        pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [moe_r_down_off, 0]),
                        [n_local_experts, inter, HIDDEN],
                    ),
                    pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [moe_r_scale_off, 0]),
                    pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [moe_w_off, 0]),
                    pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [moe_w_off, 0]),
                    pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [moe_sh_down_off, 0]),
                    h_moe,
                    resid_hold_moe,
                    pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [moe_win_off, 0]),
                    pl.slice(moe_attn_signal_stack, [COMM_SIGNAL_STRIDE_I32, 1], [moe_sig_off, 0]),
                    pl.slice(moe_recv_meta_stack, [n_ranks, n_local_experts_pad], [0, 0]),
                    pl.slice(moe_meta_arrived_stack, [COMM_SIGNAL_STRIDE_I32, 1], [0, 0]),
                    pl.slice(moe_recv_x_stack, [dispatch_lane_rows, HIDDEN], [0, 0]),
                    pl.slice(moe_recv_aux_stack, [dispatch_lane_rows, dispatch_aux_pad], [0, 0]),
                    pl.slice(moe_recv_route_stack, [dispatch_lane_rows, idx_pad], [0, 0]),
                    pl.slice(moe_data_arrived_stack, [COMM_SIGNAL_STRIDE_I32, 1], [0, 0]),
                            pl.slice(moe_sh_tmp_stack, [BATCH, HIDDEN], [moe_win_off, 0]),
                    pl.slice(moe_sh_signal_stack, [COMM_SIGNAL_STRIDE_I32, 1], [moe_sig_off, 0]),
                            pl.slice(moe_combine_arrived_stack, [COMM_SIGNAL_STRIDE_I32, 1], [0, 0]),
                    pl.slice(moe_routed_y_buf_stack, [n_routes_per_rank, HIDDEN], [0, 0]),
                    norm_layer_idx,
                    0,
                    num_tokens,
                    my_rank,
                    moe_epoch,
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
        moe_w_off_43 = 40 * HIDDEN
        moe_bias_off_43 = 40 * N_EXPERTS
        moe_r_off_43 = 40 * (n_local_experts * HIDDEN)
        moe_r_scale_off_43 = 40 * n_local_experts
        moe_r_down_off_43 = 40 * (n_local_experts * inter)
        moe_sh_down_off_43 = 40 * sh_inter_local
        moe_win_off_43 = 40 * BATCH
        moe_sig_off_43 = 40 * COMM_SIGNAL_STRIDE_I32
        moe_epoch_43 = pl.cast(41, pl.INT32)
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
            pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [moe_w_off_43, 0]),
            pl.slice(moe_router_bias, [N_EXPERTS], [moe_bias_off_43]),
            pl.reshape(
                pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [moe_r_off_43, 0]),
                [n_local_experts, HIDDEN, inter],
            ),
            pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [moe_r_scale_off_43, 0]),
            pl.reshape(
                pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [moe_r_off_43, 0]),
                [n_local_experts, HIDDEN, inter],
            ),
            pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [moe_r_scale_off_43, 0]),
            pl.reshape(
                pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [moe_r_down_off_43, 0]),
                [n_local_experts, inter, HIDDEN],
            ),
            pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [moe_r_scale_off_43, 0]),
            pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [moe_w_off_43, 0]),
            pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [moe_w_off_43, 0]),
            pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [moe_sh_down_off_43, 0]),
            h_layer_43,
            resid_hold_layer_43,
            pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [moe_win_off_43, 0]),
            pl.slice(moe_attn_signal_stack, [COMM_SIGNAL_STRIDE_I32, 1], [moe_sig_off_43, 0]),
            pl.slice(moe_recv_meta_stack, [n_ranks, n_local_experts_pad], [0, 0]),
            pl.slice(moe_meta_arrived_stack, [COMM_SIGNAL_STRIDE_I32, 1], [0, 0]),
            pl.slice(moe_recv_x_stack, [dispatch_lane_rows, HIDDEN], [0, 0]),
            pl.slice(moe_recv_aux_stack, [dispatch_lane_rows, dispatch_aux_pad], [0, 0]),
            pl.slice(moe_recv_route_stack, [dispatch_lane_rows, idx_pad], [0, 0]),
            pl.slice(moe_data_arrived_stack, [COMM_SIGNAL_STRIDE_I32, 1], [0, 0]),
            pl.slice(moe_sh_tmp_stack, [BATCH, HIDDEN], [moe_win_off_43, 0]),
            pl.slice(moe_sh_signal_stack, [COMM_SIGNAL_STRIDE_I32, 1], [moe_sig_off_43, 0]),
            pl.slice(moe_combine_arrived_stack, [COMM_SIGNAL_STRIDE_I32, 1], [0, 0]),
            pl.slice(moe_routed_y_buf_stack, [n_routes_per_rank, HIDDEN], [0, 0]),
            norm_layer_idx_43,
            0,
            num_tokens,
            my_rank,
            moe_epoch_43,
        )
        prev_hidden = h_layer_43

        resid_hold_layer_44 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        full_w_off_44 = 11 * HIDDEN
        full_wo_off_44 = 11 * hidden_q_full
        full_gate_r_off_44 = 11 * nh_full_pad
        moe_w_off_44 = 41 * HIDDEN
        moe_bias_off_44 = 41 * N_EXPERTS
        moe_r_off_44 = 41 * (n_local_experts * HIDDEN)
        moe_r_scale_off_44 = 41 * n_local_experts
        moe_r_down_off_44 = 41 * (n_local_experts * inter)
        moe_sh_down_off_44 = 41 * sh_inter_local
        moe_win_off_44 = 41 * BATCH
        moe_sig_off_44 = 41 * COMM_SIGNAL_STRIDE_I32
        moe_epoch_44 = pl.cast(42, pl.INT32)
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
            pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [moe_w_off_44, 0]),
            pl.slice(moe_router_bias, [N_EXPERTS], [moe_bias_off_44]),
            pl.reshape(
                pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [moe_r_off_44, 0]),
                [n_local_experts, HIDDEN, inter],
            ),
            pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [moe_r_scale_off_44, 0]),
            pl.reshape(
                pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [moe_r_off_44, 0]),
                [n_local_experts, HIDDEN, inter],
            ),
            pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [moe_r_scale_off_44, 0]),
            pl.reshape(
                pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [moe_r_down_off_44, 0]),
                [n_local_experts, inter, HIDDEN],
            ),
            pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [moe_r_scale_off_44, 0]),
            pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [moe_w_off_44, 0]),
            pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [moe_w_off_44, 0]),
            pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [moe_sh_down_off_44, 0]),
            next_hidden_out,
            resid_hold_layer_44,
            pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [moe_win_off_44, 0]),
            pl.slice(moe_attn_signal_stack, [COMM_SIGNAL_STRIDE_I32, 1], [moe_sig_off_44, 0]),
            pl.slice(moe_recv_meta_stack, [n_ranks, n_local_experts_pad], [0, 0]),
            pl.slice(moe_meta_arrived_stack, [COMM_SIGNAL_STRIDE_I32, 1], [0, 0]),
            pl.slice(moe_recv_x_stack, [dispatch_lane_rows, HIDDEN], [0, 0]),
            pl.slice(moe_recv_aux_stack, [dispatch_lane_rows, dispatch_aux_pad], [0, 0]),
            pl.slice(moe_recv_route_stack, [dispatch_lane_rows, idx_pad], [0, 0]),
            pl.slice(moe_data_arrived_stack, [COMM_SIGNAL_STRIDE_I32, 1], [0, 0]),
            pl.slice(moe_sh_tmp_stack, [BATCH, HIDDEN], [moe_win_off_44, 0]),
            pl.slice(moe_sh_signal_stack, [COMM_SIGNAL_STRIDE_I32, 1], [moe_sig_off_44, 0]),
            pl.slice(moe_combine_arrived_stack, [COMM_SIGNAL_STRIDE_I32, 1], [0, 0]),
            pl.slice(moe_routed_y_buf_stack, [n_routes_per_rank, HIDDEN], [0, 0]),
            norm_layer_idx_44,
            0,
            num_tokens,
            my_rank,
            moe_epoch_44,
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
        moe_gate_w: pl.Tensor[[tp_size, NUM_MOE_LAYERS_TOTAL, HIDDEN, N_EXPERTS], pl.FP32],
        moe_router_bias: pl.Tensor[[tp_size, NUM_MOE_LAYERS_TOTAL, N_EXPERTS], pl.FP32],
        moe_w_gate_r: pl.Tensor[[tp_size, NUM_MOE_LAYERS_TOTAL, n_local_experts, HIDDEN, inter], pl.INT8],
        moe_w_gate_r_scale: pl.Tensor[[tp_size, NUM_MOE_LAYERS_TOTAL, n_local_experts, inter], pl.FP32],
        moe_w_up_r: pl.Tensor[[tp_size, NUM_MOE_LAYERS_TOTAL, n_local_experts, HIDDEN, inter], pl.INT8],
        moe_w_up_r_scale: pl.Tensor[[tp_size, NUM_MOE_LAYERS_TOTAL, n_local_experts, inter], pl.FP32],
        moe_w_down_r: pl.Tensor[[tp_size, NUM_MOE_LAYERS_TOTAL, n_local_experts, inter, HIDDEN], pl.INT8],
        moe_w_down_r_scale: pl.Tensor[[tp_size, NUM_MOE_LAYERS_TOTAL, n_local_experts, HIDDEN], pl.FP32],
        moe_w_gate_s: pl.Tensor[[tp_size, NUM_MOE_LAYERS_TOTAL, HIDDEN, sh_inter_local], pl.BF16],
        moe_w_up_s: pl.Tensor[[tp_size, NUM_MOE_LAYERS_TOTAL, HIDDEN, sh_inter_local], pl.BF16],
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
        dense_attn_tmp_stack_buf = pld.alloc_window_buffer(NUM_DENSE_LAYERS * BATCH * HIDDEN * 2)
        dense_attn_signal_stack_buf = pld.alloc_window_buffer(NUM_DENSE_LAYERS * COMM_CONTROL_SIGNAL_BYTES)
        dense_mlp_tmp_stack_buf = pld.alloc_window_buffer(NUM_DENSE_LAYERS * BATCH * HIDDEN * 2)
        dense_mlp_signal_stack_buf = pld.alloc_window_buffer(NUM_DENSE_LAYERS * COMM_CONTROL_SIGNAL_BYTES)
        # MoE attention/shared TP scratch remains per-layer. EP
        # dispatch/combine windows are a single epoch-protected set.
        moe_attn_tmp_stack_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * BATCH * HIDDEN * 2)
        moe_attn_signal_stack_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * COMM_CONTROL_SIGNAL_BYTES)
        moe_recv_meta_stack_buf = pld.alloc_window_buffer(
            n_ranks * n_local_experts_pad * 4
        )
        moe_meta_arrived_stack_buf = pld.alloc_window_buffer(COMM_CONTROL_SIGNAL_BYTES)
        moe_recv_x_stack_buf = pld.alloc_window_buffer(dispatch_lane_rows * HIDDEN)
        moe_recv_aux_stack_buf = pld.alloc_window_buffer(
            dispatch_lane_rows * dispatch_aux_pad * 4
        )
        moe_recv_route_stack_buf = pld.alloc_window_buffer(
            dispatch_lane_rows * idx_pad * 4
        )
        moe_data_arrived_stack_buf = pld.alloc_window_buffer(COMM_CONTROL_SIGNAL_BYTES)
        moe_sh_tmp_stack_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * BATCH * HIDDEN * 2)
        moe_sh_signal_stack_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * COMM_CONTROL_SIGNAL_BYTES)
        moe_combine_arrived_stack_buf = pld.alloc_window_buffer(COMM_CONTROL_SIGNAL_BYTES)
        moe_routed_y_buf_stack_buf = pld.alloc_window_buffer(n_routes_per_rank * HIDDEN * 2)
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
                pl.reshape(moe_gate_w[r], [NUM_MOE_LAYERS_TOTAL * HIDDEN, N_EXPERTS]),
                pl.reshape(moe_router_bias[r], [NUM_MOE_LAYERS_TOTAL * N_EXPERTS]),
                pl.reshape(moe_w_gate_r[r], [NUM_MOE_LAYERS_TOTAL * n_local_experts * HIDDEN, inter]),
                pl.reshape(moe_w_gate_r_scale[r], [NUM_MOE_LAYERS_TOTAL * n_local_experts, inter]),
                pl.reshape(moe_w_up_r[r], [NUM_MOE_LAYERS_TOTAL * n_local_experts * HIDDEN, inter]),
                pl.reshape(moe_w_up_r_scale[r], [NUM_MOE_LAYERS_TOTAL * n_local_experts, inter]),
                pl.reshape(moe_w_down_r[r], [NUM_MOE_LAYERS_TOTAL * n_local_experts * inter, HIDDEN]),
                pl.reshape(moe_w_down_r_scale[r], [NUM_MOE_LAYERS_TOTAL * n_local_experts, HIDDEN]),
                pl.reshape(moe_w_gate_s[r], [NUM_MOE_LAYERS_TOTAL * HIDDEN, sh_inter_local]),
                pl.reshape(moe_w_up_s[r], [NUM_MOE_LAYERS_TOTAL * HIDDEN, sh_inter_local]),
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
                pld.window(moe_recv_meta_stack_buf,
                           [n_ranks, n_local_experts_pad], dtype=pl.INT32),
                pld.window(moe_meta_arrived_stack_buf, [COMM_SIGNAL_STRIDE_I32, 1],
                           dtype=pl.INT32),
                pld.window(moe_recv_x_stack_buf, [dispatch_lane_rows, HIDDEN],
                           dtype=pl.INT8),
                pld.window(moe_recv_aux_stack_buf,
                           [dispatch_lane_rows, dispatch_aux_pad], dtype=pl.FP32),
                pld.window(moe_recv_route_stack_buf,
                           [dispatch_lane_rows, idx_pad], dtype=pl.INT32),
                pld.window(moe_data_arrived_stack_buf, [COMM_SIGNAL_STRIDE_I32, 1],
                           dtype=pl.INT32),
                pld.window(moe_sh_tmp_stack_buf, [NUM_MOE_LAYERS_TOTAL * BATCH, HIDDEN],
                           dtype=pl.BF16),
                pld.window(moe_sh_signal_stack_buf, [NUM_MOE_LAYERS_TOTAL * COMM_SIGNAL_STRIDE_I32, 1],
                           dtype=pl.INT32),
                pld.window(moe_combine_arrived_stack_buf, [COMM_SIGNAL_STRIDE_I32, 1],
                           dtype=pl.INT32),
                pld.window(moe_routed_y_buf_stack_buf, [n_routes_per_rank, HIDDEN],
                           dtype=pl.BF16),
                num_tokens_per_owner,
                r,
                device=r,
            )


whole_decode_step3p5 = WholeDecodeStep3p5
