"""Canonical whole-net Step3.5 decode program.

The 45-layer graph uses runtime ``pl.range`` loops over resident leading-
dimension weight/KV views.  Physical tensor shapes are compile-time capacity;
``num_tokens_per_owner`` supplies the runtime active-row bound.

MoE communication is being unified with the DeepSeek V4-Flash baseline:
local-expert-lane dispatch push/gather, independent metadata/payload arrivals,
combine scatter/arrival, and token-level FP32 reduction.  EP data windows are
one shared set across all 42 MoE calls and use a monotonic 1-based
``moe_epoch``.  The fixed-slot pull helpers still present below are migration
source only, not the target ABI; their ready/read-complete double wave must not
be retained after the V4-Flash arrival lineages land.

Attention/shared TP all-reduce scratch remains independent from EP windows.
Its peer order, single FP32 accumulator, and final one-time BF16 store are not
part of the MoE communication migration.
"""

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
    OUT_PROJ_N_CHUNK,
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
    TP_WORLD_SIZE,
    USER_BATCH_DYN,
)
from models.step3p5.attention_full import (
    LAYER_QHIDDEN_ROWS_DYN as LAYER_QHIDDEN_ROWS_DYN_FULL,
)
from models.step3p5.attention_swa import (
    LAYER_QHIDDEN_ROWS_DYN as LAYER_QHIDDEN_ROWS_DYN_SWA,
)
from models.step3p5.dispatch import LOCAL_RECV_MAX, N_RANKS_PAD, PER_RANK_BUCKETS

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
# canonical ABI is pull-only.
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
ROUTED_MAX_TILE = LOCAL_RECV_MAX
RECV_TILE = 32
N_RECV_TILES = ROUTED_MAX_TILE // RECV_TILE  # 32 outer iterations

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
inter = MOE_INTERMEDIATE
sh_inter_local = INTER_S_LOCAL
local_recv_max = LOCAL_RECV_MAX  # 1024
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
    # attention_full / attention_swa / dense_mlp_body_tp 的 inlined body 里
    # 都调 ``self.tp_all_reduce(...)`` 汇集 o_proj / down_proj 的 partial sum。
    # pl.inline 把这些 body 拷进本 program 后，``self.tp_all_reduce`` 解析到本
    # program 的 method，所以必须在这里定义。协议使用 two-wave completion
    # barrier（expected=1/2 Ge）。
    @pl.function(type=pl.FunctionType.InCore)
    def tp_all_reduce(
        self,
        local: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
        signal_window: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
        group_size = tp_size

        # Phase 1: stage-in — copy local into my tmp_window slot (full HIDDEN).
        # All-reduce HIDDEN tiling width: fixed, INDEPENDENT of tp_size.
        ar_chunk = HIDDEN // 8
        for k0 in pl.range(0, HIDDEN, ar_chunk):
            stage_tile = pl.load(local, [0, k0], [BATCH, ar_chunk])
            pl.store(stage_tile, [0, k0], tmp_window)

        # Phase 2: barrier — notify all peers (one round), then wait on all
        # peers (one round). expected=1 fixed (cells start zero, accumulate to
        # N-1 after all notifies land; we only require >=1 from each peer slot).
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
                        tmp_window, peer=peer,
                        offsets=[0, k0], shape=[BATCH, ar_chunk],
                    )
                    acc = pl.add(
                        acc,
                        pl.cast(remote_tile, target_type=pl.FP32),
                    )
                # Every TP rank accumulates in the same canonical peer order
                # 0..N-1.  Starting from the local shard made BF16 rounding
                # rank-dependent even though all ranks consumed the same set.
            pl.store(
                pl.cast(acc, target_type=pl.BF16),
                [0, k0], local,
            )
        # Phase 4: completion barrier (framework two-wave protocol). Ensures
        # every rank finished Phase 3 reads before returning, so the next
        # layer's collective cannot race this one's reads. threshold 1 -> 2.
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
        x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
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
        x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
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
            x, gate_w, router_bias, expert_indices, expert_weights,
            num_tokens,
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
        num_tokens: pl.Scalar[pl.INT32],
    ):
        """Local histogram + per-rank prefix-sum prelude."""
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > BATCH:
            active_tokens = pl.cast(BATCH, pl.INDEX)
        for bkt in pl.range(per_rank_buckets):
            pl.write(
                send_counts_per_bucket, [bkt], pl.cast(0, pl.INT32),
            )
        for r in pl.range(n_ranks):
            pl.write(
                send_counts_per_rank, [r], pl.cast(0, pl.INT32),
            )

        for t in pl.range(active_tokens):
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
        num_tokens: pl.Scalar[pl.INT32],
    ):
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > BATCH:
            active_tokens = pl.cast(BATCH, pl.INDEX)
        # Keep the proven 16-row vector tile but attach a runtime valid shape.
        # This avoids the 1-row UB/alignment path while suppressing inactive
        # rows in the vector quant chain.
        q_amax = pl.full([1, BATCH], dtype=pl.FP32, value=1e-4)
        for qab in pl.range(HIDDEN // ROUTED_GATE_K_CHUNK):
            qa0 = qab * ROUTED_GATE_K_CHUNK
            qac = pl.cast(
                pl.slice(
                    x,
                    [BATCH, ROUTED_GATE_K_CHUNK],
                    [0, qa0],
                    valid_shape=[active_tokens, ROUTED_GATE_K_CHUNK],
                ),
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
                pl.slice(
                    x,
                    [BATCH, ROUTED_GATE_K_CHUNK],
                    [0, qn0],
                    valid_shape=[active_tokens, ROUTED_GATE_K_CHUNK],
                ),
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
    def _wait_previous_dispatch(
        self,
        count_done_sig: pld.DistributedTensor[
            [COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        active_token_out: pl.Out[
            pl.Tensor[[COMM_SIGNAL_STRIDE_I32], pl.INT32]
        ],
        num_tokens: pl.Scalar[pl.INT32],
        moe_epoch: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[COMM_SIGNAL_STRIDE_I32], pl.INT32]:
        """在不持有 dispatch data window 的 control task 中等待上一波完成。"""
        previous_complete = pl.cast(moe_epoch * 2 - 2, pl.INT32)
        if moe_epoch > 1:
            for src in pl.range(n_ranks):
                pld.system.wait(
                    signal=count_done_sig,
                    offsets=[src, 0],
                    expected=previous_complete,
                    cmp=pld.WaitCmp.Ge,
                )
        active_tokens = pl.cast(num_tokens, pl.INT32)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INT32)
        if active_tokens > BATCH:
            active_tokens = pl.cast(BATCH, pl.INT32)
        # producer 直接使用该值作为真实 loop bound，避免 ``token * 0``
        # 被 constant-fold/DCE 后丢失 wait -> producer 的 RAW。
        pl.write(active_token_out, [0], active_tokens)
        return active_token_out

    @pl.function(type=pl.FunctionType.InCore)
    def _dispatch_pack_publish(  # noqa: PLR0913
        self,
        x: pl.Tensor[[BATCH, HIDDEN], pl.INT8],
        x_scale: pl.Tensor[[BATCH, 8], pl.FP32],
        expert_indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
        send_x: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
        send_scale: pld.DistributedTensor[[local_recv_max, 8], pl.FP32],
        pub_counts: pld.DistributedTensor[
            [n_ranks * n_ranks, n_local_experts_pad], pl.INT32
        ],
        active_token: pl.Tensor[[COMM_SIGNAL_STRIDE_I32], pl.INT32],
        count_done_sig: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
    ):
        # Dispatch task 1 (PULL): histogram -> pack own INT8 tokens + per-token
        # scale + route(t*TOPK+k) into own peer-readable send_* windows in
        # (dst,loc_e) bucket order -> publish my local pub_counts rows.
        # send_* and pub_counts writes are LOCAL; peers pull them later.
        # 先消费 control-only wait task 的 token，再写 producer-owned data
        # window；这样 producer 不会在持有 send_* / pub_counts 时等待。
        active_count = pl.read(active_token, [0])
        # The InCore boundary drains both payload and counts before the pull's
        # pack_done rendezvous.
        send_counts_bkt = pl.create_tensor(
            [per_rank_buckets], dtype=pl.INT32,
        )
        send_counts_rank = pl.create_tensor([n_ranks], dtype=pl.INT32)
        send_offsets_rank = pl.create_tensor([n_ranks], dtype=pl.INT32)
        self._histogram_and_prefix_sum(
            expert_indices,
            send_counts_bkt, send_counts_rank, send_offsets_rank,
            active_count,
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
        active_tokens = pl.cast(active_count, pl.INDEX)
        for t in pl.range(active_tokens):
            for k in pl.range(TOPK):
                eid = pl.read(expert_indices, [t, k])
                dst = eid // n_local_experts
                loc_e = eid - dst * n_local_experts
                bkt = dst * n_local_experts + loc_e
                slot_i32 = pl.cast(pl.read(cursor_bkt, [bkt]), pl.INT32)
                slot = pl.cast(slot_i32, pl.INDEX)
                x_tile = pl.load(x, [t, 0], [1, HIDDEN])
                pl.store(x_tile, [slot, 0], send_x)
                sc_tile = pl.load(x_scale, [t, 0], [1, 8])
                pl.store(sc_tile, [slot, 0], send_scale)
                pl.write(
                    cursor_bkt, [bkt], pl.cast(slot_i32 + 1, pl.INT32),
                )
        # C3: each destination owns one disjoint pub_counts row.
        for d in pl.range(n_ranks):
            for e in pl.range(n_local_experts):
                v = pl.read(send_counts_bkt, [d * n_local_experts + e])
                pl.write(pub_counts, [my_rank * n_ranks + d, e], v)
        # Producer-ready wave belongs after the producer writes.  The pull
        # task waits on this wave before touching send_*; placing the notify
        # in the consumer was a false dependency and allowed a peer to read
        # partially published payload.
        for peer in pl.range(n_ranks):
            pld.system.notify(
                target=count_done_sig,
                peer=peer,
                offsets=[my_rank, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )

    @pl.function(type=pl.FunctionType.InCore)
    def _wait_dispatch_ready(
        self,
        pack_done_sig: pld.DistributedTensor[
            [COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        active_token_out: pl.Out[
            pl.Tensor[[COMM_SIGNAL_STRIDE_I32], pl.INT32]
        ],
        num_tokens: pl.Scalar[pl.INT32],
        moe_epoch: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[COMM_SIGNAL_STRIDE_I32], pl.INT32]:
        """只持有 signal/token，等待本 epoch producer-ready 波。"""
        ready_epoch = pl.cast(moe_epoch * 2 - 1, pl.INT32)
        for src in pl.range(n_ranks):
            pld.system.wait(
                signal=pack_done_sig,
                offsets=[src, 0],
                expected=ready_epoch,
                cmp=pld.WaitCmp.Ge,
            )
        active_tokens = pl.cast(num_tokens, pl.INT32)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INT32)
        if active_tokens > BATCH:
            active_tokens = pl.cast(BATCH, pl.INT32)
        pl.write(active_token_out, [0], active_tokens)
        return active_token_out

    @pl.function(type=pl.FunctionType.InCore)
    def _dispatch_pull(  # noqa: PLR0913
        self,
        send_x: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
        send_scale: pld.DistributedTensor[[local_recv_max, 8], pl.FP32],
        expert_indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
        pub_counts: pld.DistributedTensor[
            [n_ranks * n_ranks, n_local_experts_pad], pl.INT32
        ],
        active_token: pl.Tensor[[COMM_SIGNAL_STRIDE_I32], pl.INT32],
        recv_x: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
        recv_scale: pld.DistributedTensor[[local_recv_max, 8], pl.FP32],
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
        # Dispatch task 2 (PULL): control-only ready task 已确认所有 peers 完成
        # send_* / pub_counts 发布；本 task 只做 per-expert CSR 与 payload pull，
        # 不在持有 data window 时等待 signal。
        # gather each incoming token
        # from its source's send_* via remote_load (TGET, local-observable). The
        # gather order (loc_e outer, s ascending, row inner) reproduces the push
        # dst_row exactly. The ready wave is before the read (ep_all_to_all
        # pattern); the read-complete wave is emitted by _dispatch_stage after
        # the downstream consumer has consumed recv_*.
        active_tokens = pl.cast(pl.read(active_token, [0]), pl.INDEX)
        active_routes = active_tokens * TOPK
        counts_all = pl.create_tensor(
            [n_ranks * n_ranks, n_local_experts_pad], dtype=pl.INT32,
        )
        # C3: source snapshots write disjoint counts_all/recv_counts rows.
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
        for t_inv in pl.range(active_tokens):
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
        for r in pl.range(active_routes):
            sxt = pl.load(send_x, [_self_base + r, 0], [1, HIDDEN])
            pl.store(sxt, [_self_base + r, 0], recv_x)
            sst = pl.load(send_scale, [_self_base + r, 0], [1, 8])
            pl.store(sst, [_self_base + r, 0], recv_scale)
        # 固定槽 payload pull 保持确定的 peer 顺序；C3 的并行 fan-out 需要在
        # orchestration 层拆成 write-disjoint per-peer task，不能在此 InCore
        # 循环中直接替换为 ``pl.parallel``。
        for peer in pl.range(n_ranks):
            if peer != my_rank:
                _peer_base = pl.cast(peer * n_routes_per_rank, pl.INDEX)
                for r in pl.range(active_routes):
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
        return recv_counts, local_expert_offset, local_expert_count, inverse_map_out

    @pl.function(type=pl.FunctionType.InCore)
    def _dispatch_stage(  # noqa: PLR0913
        self,
        recv_x: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
        recv_scale: pld.DistributedTensor[[local_recv_max, 8], pl.FP32],
        local_expert_offset: pl.Tensor[[n_local_experts], pl.INT32],
        local_expert_count: pl.Tensor[[n_local_experts], pl.INT32],
        local_routed_x_out: pl.Out[
            pl.Tensor[[local_recv_max, HIDDEN], pl.INT8]
        ],
        local_routed_x_scale_out: pl.Out[pl.Tensor[[1, local_recv_max], pl.FP32]],
        recv_counts: pl.Tensor[[n_ranks, n_local_experts_pad], pl.INT32],
        count_done_sig: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
        moe_epoch: pl.Scalar[pl.INT32],
    ) -> tuple[
        pl.Tensor[[local_recv_max, HIDDEN], pl.INT8],
        pl.Tensor[[1, local_recv_max], pl.FP32],
    ]:
        # Dispatch task 3: compact the peer-major fixed-slot pull windows
        # into expert-major tensors using the receiver-local count snapshot.
        # The InCore task boundary keeps pull and compaction separate. The
        # completion wave below is deliberately part of this consumer task:
        # the next layer may not overwrite producer-owned send_* / pub_counts
        # until every rank has finished consuming the pulled recv_* rows.
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
                running = running + pl.cast(rn, pl.INT32)
        for peer in pl.range(n_ranks):
            pld.system.notify(
                target=count_done_sig,
                peer=peer,
                offsets=[my_rank, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )
        return local_routed_x_out, local_routed_x_scale_out

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
        pub_counts: pld.DistributedTensor[
            [n_ranks * n_ranks, n_local_experts_pad], pl.INT32
        ],
        count_done_sig: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        recv_x: pld.DistributedTensor[
            [local_recv_max, HIDDEN], pl.INT8
        ],
        recv_scale: pld.DistributedTensor[[local_recv_max, 8], pl.FP32],
        send_x: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
        send_scale: pld.DistributedTensor[[local_recv_max, 8], pl.FP32],
        num_tokens: pl.Scalar[pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
        moe_epoch: pl.Scalar[pl.INT32],
    ) -> tuple[
        pl.Tensor[[local_recv_max, HIDDEN], pl.INT8],
        pl.Tensor[[1, local_recv_max], pl.FP32],
        pl.Tensor[[n_local_experts], pl.INT32],
        pl.Tensor[[n_local_experts], pl.INT32],
        pl.Tensor[[BATCH, TOPK], pl.INT32]
    ]:
        # Pull EP dispatch, split into a control-only wait plus 3 InCore data
        # tasks so local pack,
        # pack_done rendezvous + remote_load pull, and stage compact have
        # explicit task boundaries.  _dispatch_stage owns the completion
        # wave because it is the task that consumes the pulled recv_* data.
        dispatch_active_token = pl.create_tensor(
            [COMM_SIGNAL_STRIDE_I32], dtype=pl.INT32,
        )
        dispatch_active_token = self._wait_previous_dispatch(
            count_done_sig,
            dispatch_active_token,
            num_tokens,
            moe_epoch,
        )
        self._dispatch_pack_publish(
            x, x_scale, expert_indices,
            send_x, send_scale, pub_counts,
            dispatch_active_token,
            count_done_sig, my_rank,
        )
        dispatch_ready_token = pl.create_tensor(
            [COMM_SIGNAL_STRIDE_I32], dtype=pl.INT32,
        )
        dispatch_ready_token = self._wait_dispatch_ready(
            count_done_sig,
            dispatch_ready_token,
            num_tokens,
            moe_epoch,
        )
        recv_counts = pl.create_tensor(
            [n_ranks, n_local_experts_pad], dtype=pl.INT32,
        )
        inverse_map = pl.create_tensor([BATCH, TOPK], dtype=pl.INT32)
        recv_counts, local_expert_offset, local_expert_count, inverse_map = self._dispatch_pull(
            send_x, send_scale, expert_indices, pub_counts, dispatch_ready_token,
            recv_x, recv_scale, recv_counts, inverse_map,
            local_expert_offset, local_expert_count,
            my_rank,
        )
        local_routed_x_out, local_routed_x_scale_out = self._dispatch_stage(
            recv_x, recv_scale,
            local_expert_offset, local_expert_count,
            local_routed_x_out, local_routed_x_scale_out,
            recv_counts, count_done_sig, my_rank, moe_epoch,
        )
        return (
            local_routed_x_out,
            local_routed_x_scale_out,
            local_expert_offset,
            local_expert_count,
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
        for e in pl.range(n_local_experts):
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
            self.tp_all_reduce(
                sh_y, sh_tmp_window, sh_signal_window, my_rank,
            )
        return sh_y

    # ---------- Stage 4: combine (EP a2a back + weighted gather) ------
    @pl.function(type=pl.FunctionType.InCore)
    def _wait_previous_combine(
        self,
        combine_done: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        notify_token_out: pl.Out[
            pl.Tensor[[COMM_SIGNAL_STRIDE_I32], pl.INT32]
        ],
        moe_epoch: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[COMM_SIGNAL_STRIDE_I32], pl.INT32]:
        # Keep the epoch wait in a control-only task.  Waiting while already
        # holding the routed_y window as an output can deadlock the scheduler:
        # the next-layer zero task reserves the window while the prior
        # weighted-gather task still needs to read it before publishing
        # completion.
        previous_complete = pl.cast(moe_epoch * 2 - 2, pl.INT32)
        if moe_epoch > 1:
            for src in pl.range(n_ranks):
                pld.system.wait(
                    signal=combine_done,
                    offsets=[src, 0],
                    expected=previous_complete,
                    cmp=pld.WaitCmp.Ge,
                )
        # stage task 将该 token 直接作为 AtomicAdd 增量，形成不可省略的 RAW。
        pl.write(
            notify_token_out,
            [0],
            pl.cast(pl.min(moe_epoch, 1), target_type=pl.INT32),
        )
        return notify_token_out

    @pl.function(type=pl.FunctionType.InCore)
    def _stage_routed_src(
        self,
        local_routed_y: pl.Tensor[[local_recv_max, HIDDEN], pl.BF16],
        routed_src_buf: pld.DistributedTensor[
            [local_recv_max, HIDDEN], pl.BF16
        ],
        local_expert_count: pl.Tensor[[n_local_experts], pl.INT32],
        notify_token: pl.Tensor[[COMM_SIGNAL_STRIDE_I32], pl.INT32],
        combine_done: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
    ):
        # Combine task 1 (PULL): copy the expert holder's routed output into
        # its OWN peer-readable window (local writes). The InCore task boundary
        # drains these before the pull rendezvous, so peers read landed data.
        # 先消费 control-only wait task 的 token，再覆盖 routed_src_buf。
        ready_increment = pl.read(notify_token, [0])
        active_rows = pl.cast(0, pl.INT32)
        for e in pl.range(n_local_experts):
            active_rows = active_rows + pl.read(local_expert_count, [e])
        for row in pl.range(0, active_rows, stage_rows):
            tile = pl.load(local_routed_y, [row, 0], [stage_rows, HIDDEN])
            pl.store(tile, [row, 0], routed_src_buf)
        # Producer-ready wave belongs after routed_src_buf is fully staged.
        # Including self makes the local pull observe the same ordering as
        # peer pulls rather than relying on an incidental inline schedule.
        for peer in pl.range(n_ranks):
            pld.system.notify(
                target=combine_done,
                peer=peer,
                offsets=[my_rank, 0],
                value=ready_increment,
                op=pld.NotifyOp.AtomicAdd,
            )

    @pl.function(type=pl.FunctionType.InCore)
    def _wait_combine_ready(
        self,
        combine_done: pld.DistributedTensor[
            [COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        active_token_out: pl.Out[
            pl.Tensor[[COMM_SIGNAL_STRIDE_I32], pl.INT32]
        ],
        num_tokens: pl.Scalar[pl.INT32],
        moe_epoch: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[COMM_SIGNAL_STRIDE_I32], pl.INT32]:
        """只持有 signal/token，等待 routed_src producer-ready 波。"""
        ready_epoch = pl.cast(moe_epoch * 2 - 1, pl.INT32)
        for src in pl.range(n_ranks):
            pld.system.wait(
                signal=combine_done,
                offsets=[src, 0],
                expected=ready_epoch,
                cmp=pld.WaitCmp.Ge,
            )
        active_tokens = pl.cast(num_tokens, pl.INT32)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INT32)
        if active_tokens > BATCH:
            active_tokens = pl.cast(BATCH, pl.INT32)
        pl.write(active_token_out, [0], active_tokens)
        return active_token_out

    @pl.function(type=pl.FunctionType.InCore)
    def _pull_routed_y(  # noqa: PLR0913
        self,
        routed_src_buf: pld.DistributedTensor[
            [local_recv_max, HIDDEN], pl.BF16
        ],
        inverse_map: pl.Tensor[[BATCH, TOPK], pl.INT32],
        combine_done: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        active_token: pl.Tensor[[COMM_SIGNAL_STRIDE_I32], pl.INT32],
        expert_weights: pl.Tensor[[BATCH, TOPK], pl.FP32],
        sh_y: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        route_stage: pl.Out[pl.Tensor[[1, HIDDEN], pl.BF16]],
        moe_out: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
        # Combine task 2 (PULL + weighted gather): after every holder has
        # staged routed_src_buf, pull each route with the synchronous
        # tensor-level TGET primitive.  ``pld.tensor.get`` owns the complete
        # peer-GM -> local-GM transfer, unlike the tile-level remote_load path
        # whose direct accumulator use was invocation-dependent on A2/A3.
        # Keep the TGET and local weighted consumer in this same InCore task.
        active_tokens = pl.cast(pl.read(active_token, [0]), pl.INDEX)
        # 使用独立的 GM staging tensor 承接 local copy / remote TGET。这样
        # TGET 的写目标与最终 moe_out 分离，后续 load 对 route_stage 建立
        # 明确 RAW；每条 route 消费后才复用同一行。
        for t in pl.range(active_tokens):
            acc = pl.cast(
                pl.load(sh_y, [t, 0], [1, HIDDEN]),
                target_type=pl.FP32,
            )
            for k in pl.range(TOPK):
                packed = pl.read(inverse_map, [t, k])
                dst = packed // pl.cast(local_recv_max, pl.INT32)
                dst_row = pl.cast(
                    packed - dst * pl.cast(local_recv_max, pl.INT32),
                    pl.INDEX,
                )
                route_weight = pl.read(expert_weights, [t, k])
                if dst == my_rank:
                    local_row = pl.load(
                        routed_src_buf, [dst_row, 0], [1, HIDDEN],
                    )
                    pl.store(local_row, [0, 0], route_stage)
                else:
                    pld.tensor.get(
                        route_stage,
                        peer=dst,
                        src=routed_src_buf,
                        dst_offsets=[0, 0],
                        src_offsets=[dst_row, 0],
                        shape=[1, HIDDEN],
                    )
                route_row = pl.load(
                    route_stage, [0, 0], [1, HIDDEN],
                )
                acc = pl.add(
                    acc,
                    pl.mul(
                        pl.cast(route_row, target_type=pl.FP32),
                        route_weight,
                    ),
                )
            pl.store(
                pl.cast(acc, target_type=pl.BF16),
                [t, 0],
                moe_out,
            )
        for t_inactive in pl.range(active_tokens, BATCH):
            inactive = pl.mul(
                pl.cast(
                    pl.load(sh_y, [t_inactive, 0], [1, HIDDEN]),
                    target_type=pl.FP32,
                ),
                0.0,
            )
            pl.store(
                pl.cast(inactive, target_type=pl.BF16),
                [t_inactive, 0],
                moe_out,
            )

        # This point is after the last direct remote read and its accumulation,
        # so the completion wave is the exact routed_src consumer boundary.
        for peer in pl.range(n_ranks):
            pld.system.notify(
                target=combine_done,
                peer=peer,
                offsets=[my_rank, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )
        return moe_out
    @pl.function(type=pl.FunctionType.Inline)
    def combine_step(  # noqa: PLR0913
        self,
        local_routed_y: pl.Tensor[
            [local_recv_max, HIDDEN], pl.BF16
        ],
        expert_weights: pl.Tensor[[BATCH, TOPK], pl.FP32],
        sh_y: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        moe_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
        combine_done_sig: pld.DistributedTensor[
            [COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        inverse_map: pl.Tensor[[BATCH, TOPK], pl.INT32],
        routed_src_buf: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.BF16],
        local_expert_count: pl.Tensor[[n_local_experts], pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
        moe_epoch: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
        # Receiver-staged pull is the canonical return leg.  Dispatch already
        # produced ``inverse_map`` from the receiver-local count/CSR ordering.
        # The return leg therefore consumes receiver-local route provenance,
        # avoiding rank-indexed reads from another producer's private metadata.
        combine_reuse_token = pl.create_tensor(
            [COMM_SIGNAL_STRIDE_I32], dtype=pl.INT32,
        )
        combine_reuse_token = self._wait_previous_combine(
            combine_done_sig,
            combine_reuse_token,
            moe_epoch,
        )
        route_stage = pl.create_tensor([1, HIDDEN], dtype=pl.BF16)
        self._stage_routed_src(
            local_routed_y,
            routed_src_buf,
            local_expert_count,
            combine_reuse_token,
            combine_done_sig,
            my_rank,
        )
        combine_ready_token = pl.create_tensor(
            [COMM_SIGNAL_STRIDE_I32], dtype=pl.INT32,
        )
        combine_ready_token = self._wait_combine_ready(
            combine_done_sig,
            combine_ready_token,
            num_tokens,
            moe_epoch,
        )
        moe_out = self._pull_routed_y(
            routed_src_buf,
            inverse_map,
            combine_done_sig,
            combine_ready_token,
            expert_weights,
            sh_y,
            route_stage,
            moe_out,
            my_rank,
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
        pub_counts: pld.DistributedTensor[
            [n_ranks * n_ranks, n_local_experts_pad], pl.INT32
        ],
        count_done_sig: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        recv_x: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
        recv_scale: pld.DistributedTensor[[local_recv_max, 8], pl.FP32],
        send_x: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
        send_scale: pld.DistributedTensor[[local_recv_max, 8], pl.FP32],
        sh_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
        sh_signal_window: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        combine_done_sig: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        routed_src_buf: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.BF16],
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
        # ── B: post-attention zero-centred RMSNorm of resid_hold. ──
        hidden_blocks = HIDDEN // K_CHUNK
        post_norm = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        resid1_fp32 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
        with pl.at(
            level=pl.Level.CORE_GROUP, name_hint="moe_post_rmsnorm_zc",
        ):
            for kb in pl.range(hidden_blocks):
                k0 = kb * K_CHUNK
                rchunk = pl.cast(
                    pl.slice(resid_hold, [BATCH, K_CHUNK], [0, k0]),
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
            expert_indices, expert_weights, num_tokens,
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
            post_norm, x_disp_i8, x_disp_scale, num_tokens,
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
        (
            local_routed_x,
            local_routed_x_scale,
            local_expert_offset,
            local_expert_count,
            inverse_map,
        ) = self.dispatch_step(
            x_disp_i8, x_disp_scale, expert_indices,
            local_routed_x, local_routed_x_scale,
            local_expert_offset, local_expert_count,
            pub_counts, count_done_sig, recv_x, recv_scale,
            send_x, send_scale,
            num_tokens, my_rank, moe_epoch,
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
            expert_weights, sh_y,
            moe_out,
            combine_done_sig,
            inverse_map, routed_src_buf,
            local_expert_count, num_tokens, my_rank, moe_epoch,
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
        pub_counts: pld.DistributedTensor[
            [n_ranks * n_ranks, n_local_experts_pad], pl.INT32
        ],
        count_done_sig: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        recv_x: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
        recv_scale: pld.DistributedTensor[[local_recv_max, 8], pl.FP32],
        send_x: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
        send_scale: pld.DistributedTensor[[local_recv_max, 8], pl.FP32],
        sh_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
        sh_signal_window: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        combine_done_sig: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        routed_src_buf: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.BF16],
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
        # ── B: post-attention zero-centred RMSNorm of resid_hold. ──
        hidden_blocks = HIDDEN // K_CHUNK
        post_norm = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        resid1_fp32 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
        with pl.at(
            level=pl.Level.CORE_GROUP, name_hint="moe_post_rmsnorm_zc",
        ):
            for kb in pl.range(hidden_blocks):
                k0 = kb * K_CHUNK
                rchunk = pl.cast(
                    pl.slice(resid_hold, [BATCH, K_CHUNK], [0, k0]),
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
            expert_indices, expert_weights, num_tokens,
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
            post_norm, x_disp_i8, x_disp_scale, num_tokens,
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
        (
            local_routed_x,
            local_routed_x_scale,
            local_expert_offset,
            local_expert_count,
            inverse_map,
        ) = self.dispatch_step(
            x_disp_i8, x_disp_scale, expert_indices,
            local_routed_x, local_routed_x_scale,
            local_expert_offset, local_expert_count,
            pub_counts, count_done_sig, recv_x, recv_scale,
            send_x, send_scale,
            num_tokens, my_rank, moe_epoch,
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
            expert_weights, sh_y,
            moe_out,
            combine_done_sig,
            inverse_map, routed_src_buf,
            local_expert_count, num_tokens, my_rank, moe_epoch,
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
    def expert_routed_step_swiglu7(
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
        local_routed_y = self._expert_routed_swiglu7(
            local_routed_x,
            local_routed_x_scale,
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
            self.tp_all_reduce(
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
        pub_counts: pld.DistributedTensor[
            [n_ranks * n_ranks, n_local_experts_pad], pl.INT32
        ],
        count_done_sig: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        recv_x: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
        recv_scale: pld.DistributedTensor[[local_recv_max, 8], pl.FP32],
        send_x: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
        send_scale: pld.DistributedTensor[[local_recv_max, 8], pl.FP32],
        sh_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
        sh_signal_window: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        combine_done_sig: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        routed_src_buf: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.BF16],
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
        # ── B: post-attention zero-centred RMSNorm of resid_hold. ──
        hidden_blocks = HIDDEN // K_CHUNK
        post_norm = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        resid1_fp32 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
        with pl.at(
            level=pl.Level.CORE_GROUP, name_hint="moe_post_rmsnorm_zc",
        ):
            for kb in pl.range(hidden_blocks):
                k0 = kb * K_CHUNK
                rchunk = pl.cast(
                    pl.slice(resid_hold, [BATCH, K_CHUNK], [0, k0]),
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
            expert_indices, expert_weights, num_tokens,
        )

        # 2) Shared-expert lane (TP-sliced + tp_all_reduce).
        sh_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        sh_y = self.expert_shared_step_swiglu16(
            post_norm, w_gate_s, w_up_s, w_down_s, sh_y,
            sh_tmp_window, sh_signal_window, my_rank,
        )


        # 1A: per-token INT8 dynamic-quant of the MoE input BEFORE
        # dispatch (dispatch-side; shrinks recv_x 8→4MB/layer).
        x_disp_i8 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.INT8)
        x_disp_scale = pl.create_tensor([BATCH, 8], dtype=pl.FP32)
        (x_disp_i8, x_disp_scale) = self._quant_moe_input(
            post_norm, x_disp_i8, x_disp_scale, num_tokens,
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
        (
            local_routed_x,
            local_routed_x_scale,
            local_expert_offset,
            local_expert_count,
            inverse_map,
        ) = self.dispatch_step(
            x_disp_i8, x_disp_scale, expert_indices,
            local_routed_x, local_routed_x_scale,
            local_expert_offset, local_expert_count,
            pub_counts, count_done_sig, recv_x, recv_scale,
            send_x, send_scale,
            num_tokens, my_rank, moe_epoch,
        )

        # 4) Routed experts (local 36).
        local_routed_y = pl.create_tensor(
            [local_recv_max, HIDDEN], dtype=pl.BF16,
        )
        local_routed_y = self.expert_routed_step_swiglu7(
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
            expert_weights, sh_y,
            moe_out,
            combine_done_sig,
            inverse_map, routed_src_buf,
            local_expert_count, num_tokens, my_rank, moe_epoch,
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
        pub_counts: pld.DistributedTensor[
            [n_ranks * n_ranks, n_local_experts_pad], pl.INT32
        ],
        count_done_sig: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        recv_x: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
        recv_scale: pld.DistributedTensor[[local_recv_max, 8], pl.FP32],
        send_x: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
        send_scale: pld.DistributedTensor[[local_recv_max, 8], pl.FP32],
        sh_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
        sh_signal_window: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        combine_done_sig: pld.DistributedTensor[[COMM_SIGNAL_STRIDE_I32, 1], pl.INT32],
        routed_src_buf: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.BF16],
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
        # ── B: post-attention zero-centred RMSNorm of resid_hold. ──
        hidden_blocks = HIDDEN // K_CHUNK
        post_norm = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        resid1_fp32 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
        with pl.at(
            level=pl.Level.CORE_GROUP, name_hint="moe_post_rmsnorm_zc",
        ):
            for kb in pl.range(hidden_blocks):
                k0 = kb * K_CHUNK
                rchunk = pl.cast(
                    pl.slice(resid_hold, [BATCH, K_CHUNK], [0, k0]),
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
            expert_indices, expert_weights, num_tokens,
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
            post_norm, x_disp_i8, x_disp_scale, num_tokens,
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
        (
            local_routed_x,
            local_routed_x_scale,
            local_expert_offset,
            local_expert_count,
            inverse_map,
        ) = self.dispatch_step(
            x_disp_i8, x_disp_scale, expert_indices,
            local_routed_x, local_routed_x_scale,
            local_expert_offset, local_expert_count,
            pub_counts, count_done_sig, recv_x, recv_scale,
            send_x, send_scale,
            num_tokens, my_rank, moe_epoch,
        )

        # 4) Routed experts (local 36).
        local_routed_y = pl.create_tensor(
            [local_recv_max, HIDDEN], dtype=pl.BF16,
        )
        local_routed_y = self.expert_routed_step_swiglu7(
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
            expert_weights, sh_y,
            moe_out,
            combine_done_sig,
            inverse_map, routed_src_buf,
            local_expert_count, num_tokens, my_rank, moe_epoch,
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
        # data windows are one shared set, protected by the pull-safe monotonic
        # double-wave epoch in _dispatch_pull/_pull_routed_y.
        moe_attn_tmp_stack: pld.DistributedTensor[
            [NUM_MOE_LAYERS_TOTAL * BATCH, HIDDEN], pl.BF16
        ],
        moe_attn_signal_stack: pld.DistributedTensor[
            [NUM_MOE_LAYERS_TOTAL * COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        moe_pub_counts_stack: pld.DistributedTensor[
            [n_ranks * n_ranks, n_local_experts_pad], pl.INT32
        ],
        moe_count_done_sig_stack: pld.DistributedTensor[
            [COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        moe_recv_x_stack: pld.DistributedTensor[
            [local_recv_max, HIDDEN], pl.INT8
        ],
        moe_recv_scale_stack: pld.DistributedTensor[
            [local_recv_max, 8], pl.FP32
        ],
        moe_send_x_stack: pld.DistributedTensor[
            [local_recv_max, HIDDEN], pl.INT8
        ],
        moe_send_scale_stack: pld.DistributedTensor[
            [local_recv_max, 8], pl.FP32
        ],
        moe_sh_tmp_stack: pld.DistributedTensor[
            [NUM_MOE_LAYERS_TOTAL * BATCH, HIDDEN], pl.BF16
        ],
        moe_sh_signal_stack: pld.DistributedTensor[
            [NUM_MOE_LAYERS_TOTAL * COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        moe_combine_done_sig_stack: pld.DistributedTensor[
            [COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        moe_routed_src_buf_stack: pld.DistributedTensor[
            [local_recv_max, HIDDEN], pl.BF16
        ],
        num_tokens_per_owner: pl.Tensor[[NUM_TOKENS_STORAGE_I32], pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
    ):
        # G1 runtime active-token ABI.  Every owner publishes its number of
        # real decode rows; all ranks use the same max so dynamic MoE loop
        # bounds and communication counts remain rank-symmetric.
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
        # C1 protocol: EP dispatch/combine reuse one window set. A 1-based
        # moe_epoch drives ready/read-complete waits at 2e-1/2e, preventing a
        # fast rank from overwriting producer-owned send_*/routed_src while a
        # slow peer still reads the previous layer. Attention/shared TP
        # all-reduce scratch remains per-layer with fixed expected=1/2.
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
                    pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [0, 0]),
                    pl.slice(moe_count_done_sig_stack, [COMM_SIGNAL_STRIDE_I32, 1], [0, 0]),
                    pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [0, 0]),
                    pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [0, 0]),
                                    pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [0, 0]),
                    pl.slice(moe_send_scale_stack, [local_recv_max, 8], [0, 0]),
                            pl.slice(moe_sh_tmp_stack, [BATCH, HIDDEN], [moe_win_off, 0]),
                    pl.slice(moe_sh_signal_stack, [COMM_SIGNAL_STRIDE_I32, 1], [moe_sig_off, 0]),
                            pl.slice(moe_combine_done_sig_stack, [COMM_SIGNAL_STRIDE_I32, 1], [0, 0]),
                    pl.slice(moe_routed_src_buf_stack, [local_recv_max, HIDDEN], [0, 0]),
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
                    pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [0, 0]),
                    pl.slice(moe_count_done_sig_stack, [COMM_SIGNAL_STRIDE_I32, 1], [0, 0]),
                    pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [0, 0]),
                    pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [0, 0]),
                                    pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [0, 0]),
                    pl.slice(moe_send_scale_stack, [local_recv_max, 8], [0, 0]),
                            pl.slice(moe_sh_tmp_stack, [BATCH, HIDDEN], [moe_win_off, 0]),
                    pl.slice(moe_sh_signal_stack, [COMM_SIGNAL_STRIDE_I32, 1], [moe_sig_off, 0]),
                            pl.slice(moe_combine_done_sig_stack, [COMM_SIGNAL_STRIDE_I32, 1], [0, 0]),
                    pl.slice(moe_routed_src_buf_stack, [local_recv_max, HIDDEN], [0, 0]),
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
            pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [0, 0]),
            pl.slice(moe_count_done_sig_stack, [COMM_SIGNAL_STRIDE_I32, 1], [0, 0]),
            pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [0, 0]),
            pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [0, 0]),
            pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [0, 0]),
            pl.slice(moe_send_scale_stack, [local_recv_max, 8], [0, 0]),
            pl.slice(moe_sh_tmp_stack, [BATCH, HIDDEN], [moe_win_off_43, 0]),
            pl.slice(moe_sh_signal_stack, [COMM_SIGNAL_STRIDE_I32, 1], [moe_sig_off_43, 0]),
            pl.slice(moe_combine_done_sig_stack, [COMM_SIGNAL_STRIDE_I32, 1], [0, 0]),
            pl.slice(moe_routed_src_buf_stack, [local_recv_max, HIDDEN], [0, 0]),
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
            pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [0, 0]),
            pl.slice(moe_count_done_sig_stack, [COMM_SIGNAL_STRIDE_I32, 1], [0, 0]),
            pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [0, 0]),
            pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [0, 0]),
            pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [0, 0]),
            pl.slice(moe_send_scale_stack, [local_recv_max, 8], [0, 0]),
            pl.slice(moe_sh_tmp_stack, [BATCH, HIDDEN], [moe_win_off_44, 0]),
            pl.slice(moe_sh_signal_stack, [COMM_SIGNAL_STRIDE_I32, 1], [moe_sig_off_44, 0]),
            pl.slice(moe_combine_done_sig_stack, [COMM_SIGNAL_STRIDE_I32, 1], [0, 0]),
            pl.slice(moe_routed_src_buf_stack, [local_recv_max, HIDDEN], [0, 0]),
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
        moe_pub_counts_stack_buf = pld.alloc_window_buffer(
            n_ranks * n_ranks * n_local_experts_pad * 4
        )
        moe_count_done_sig_stack_buf = pld.alloc_window_buffer(COMM_CONTROL_SIGNAL_BYTES)
        moe_recv_x_stack_buf = pld.alloc_window_buffer(local_recv_max * HIDDEN)
        moe_recv_scale_stack_buf = pld.alloc_window_buffer(local_recv_max * 8 * 4)
        moe_send_x_stack_buf = pld.alloc_window_buffer(local_recv_max * HIDDEN)
        moe_send_scale_stack_buf = pld.alloc_window_buffer(local_recv_max * 8 * 4)
        moe_sh_tmp_stack_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * BATCH * HIDDEN * 2)
        moe_sh_signal_stack_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * COMM_CONTROL_SIGNAL_BYTES)
        moe_combine_done_sig_stack_buf = pld.alloc_window_buffer(COMM_CONTROL_SIGNAL_BYTES)
        moe_routed_src_buf_stack_buf = pld.alloc_window_buffer(local_recv_max * HIDDEN * 2)
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
                pld.window(moe_pub_counts_stack_buf,
                           [n_ranks * n_ranks, n_local_experts_pad],
                           dtype=pl.INT32),
                pld.window(moe_count_done_sig_stack_buf, [COMM_SIGNAL_STRIDE_I32, 1],
                           dtype=pl.INT32),
                pld.window(moe_recv_x_stack_buf, [local_recv_max, HIDDEN],
                           dtype=pl.INT8),
                pld.window(moe_recv_scale_stack_buf, [local_recv_max, 8],
                           dtype=pl.FP32),
                pld.window(moe_send_x_stack_buf, [local_recv_max, HIDDEN],
                           dtype=pl.INT8),
                pld.window(moe_send_scale_stack_buf, [local_recv_max, 8],
                           dtype=pl.FP32),
                pld.window(moe_sh_tmp_stack_buf, [NUM_MOE_LAYERS_TOTAL * BATCH, HIDDEN],
                           dtype=pl.BF16),
                pld.window(moe_sh_signal_stack_buf, [NUM_MOE_LAYERS_TOTAL * COMM_SIGNAL_STRIDE_I32, 1],
                           dtype=pl.INT32),
                pld.window(moe_combine_done_sig_stack_buf, [COMM_SIGNAL_STRIDE_I32, 1],
                           dtype=pl.INT32),
                pld.window(moe_routed_src_buf_stack_buf, [local_recv_max, HIDDEN],
                           dtype=pl.BF16),
                num_tokens_per_owner,
                r,
                device=r,
            )


whole_decode_step3p5 = WholeDecodeStep3p5
