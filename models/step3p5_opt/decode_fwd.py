"""Step3p5 decode whole-net, DeepSeek-aligned loop form (阶段 1 dense-only).

Architectural pivot (task #27): the baseline
``models/step3p5/decode_layer_single_chip_hidden.py`` unrolls all 45 layers
inline as a single ``@pl.program`` body, with per-layer ``*_chip_orch``
helpers marked ``inline_orchestration=True``. That 45x unroll is the
brittleness root: return-0-Var-like walls, view-escape-scope walls, 45x SSA
+ a single 766 MB comm domain with no inter-layer drain.

This module keeps the baseline ``@pl.program class`` form (proven to codegen
clean at pristine bc5eecb1) and changes exactly one variable per
criterion A: unroll → ``pl.range`` runtime loop + ``pl.Out`` return-params
on the per-layer ``chip_orch`` helpers (Var, never a view — removes the
return-0 / view-escape walls). Module-level ``@pl.jit.inline`` helpers were
a structural mismatch (parser rejects bare-call of module-level inline from
inside a ``@pl.function`` method); helpers stay as class methods with
``self.method(...)`` invocation (baseline convention).

* a single ``@pl.program class`` with a ``whole_chip_orch`` method whose
  body is a ``for layer_idx in pl.range(N)`` runtime loop (``ForStmt``,
  not ``ForKind::Unroll``);
* per-layer ``chip_orch`` helpers as ``@pl.function(type=Orchestration,
  attrs={'inline_orchestration': True})`` class methods (decorator + 签名
  逐字对齐 baseline) with ``pl.Out`` return-params (Var, never a view);
* per-layer weight slices via dynamic ``layer_idx * STRIDE`` scalar offsets
  into stacked leading-dim weight tensors (B1 resident="stacked");
* per-layer intermediates created fresh each iteration (carry pattern).

阶段 1 scope: dense layers only (L0 full-attn dense + L1/L2 swa-attn dense,
3 layers total). L0 is a distinct full-attn shape, so it is emitted as an
explicit pre-loop ``self.full_chip_orch`` call; L1/L2 share the swa-attn
shape and run inside ``pl.range(2)``. 目标 = 验证判据 A: the loop form
(``pl.range`` + ``pl.Out`` return-param) clears codegen with no return-0 /
view-escape walls, isolated from the class-vs-function question.

Components (attention, dense MLP) are imported verbatim from
``models.step3p5``; only the orchestration scaffolding is rewritten here.
"""

from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld

from models.step3p5.attention_full import attention_full
from models.step3p5.attention_swa import attention_swa
from models.step3p5.decode_layer_single_chip_hidden import _dense_mlp_body_tp

# 注意：attention_full / attention_swa / _dense_mlp_body_tp 的 kernel body 经
# ``pl.inline`` 拷进本模块后，body 内引用的 config 常量按 **本模块** 的 module
# global 解析（非 source 模块）。所以 dense 路径三个 kernel 用到的 config 常量必须
# 在这里一次性全导入，否则 codegen 报 UndefinedVariableError（已在
# ``INPUT_PROJ_K_CHUNK`` 上撞过）。LAYER_QHIDDEN_ROWS_DYN 在 full / swa 两个 source
# 模块里值不同（full=12288，swa=33*1536=50688），按 baseline 别名 _FULL/_SWA 区分。
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

# ── Phase 3: MoE-layer module-level constants (silu_silu 40-layer loop). ──
# Copied verbatim from baseline ``decode_layer_single_chip_hidden.py:98..192``.
# These tiling / sort widths / activation thresholds live at module level so the
# inlined MoE method bodies (gate / dispatch / expert_routed / expert_shared /
# combine) can reference them via closure capture at parse time. PULL dispatch /
# combine path only (ep_all_to_all / _push_routed_y_to_sources / _publish_src_
# route_table are PUSH-path dead code, NOT ported).
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

# moe-local helpers (baseline builder-function locals, lifted to module level).
n_ranks = tp_size
n_ranks_pad = N_RANKS_PAD
n_local_experts = N_LOCAL_EXPERTS
n_local_experts_pad = ((n_local_experts + 7) // 8) * 8
idx_pad = 8
inter = MOE_INTERMEDIATE
sh_inter_local = INTER_S_LOCAL
local_recv_max = LOCAL_RECV_MAX  # 1024
stage_rows = 8  # baseline builder-local closure constant (L644), hoisted
n_routes_per_rank = BATCH * TOPK
per_rank_buckets = PER_RANK_BUCKETS
sh_tp_chunk = HIDDEN // tp_size
# Diagnostic bisect knob (build-time constant): MoE chip_orch op-level dump.
_DBG_STAGE = int(__import__("os").environ.get("P_DBG_STAGE", "0"))

# MoE-layer counts for the loop form. L3..L42 = 40 MoE silu_silu layers split
# by attention type: 10 full-attn (L3..L12) + 30 swa-attn (L13..L42).
NUM_FULL_MOE_LAYERS = 10
NUM_SWA_MOE_LAYERS = 30
NUM_MOE_LAYERS = NUM_FULL_MOE_LAYERS + NUM_SWA_MOE_LAYERS  # 40 (loop body L3..L42)
# Phase 4: L43 (swa_moe_swiglu7_silu) + L44 (full_moe_swiglu7_swiglu16) are
# post-loop explicit specialization layers that slice MoE weight/window stacks
# at offsets 40/41 → stacks sized to 42 (loop keeps iterating 40).
NUM_MOE_LAYERS_TOTAL = NUM_MOE_LAYERS + 2  # 42

# silu_silu activation closure constants (Phase 3 = silu_silu only, both limits
# 0.0 → swiglu-clip branches dead). Mirrors baseline builder L588-607; lifted to
# module level because step3p5_opt is a flat @pl.program class, not a builder.
# Phase 4 will add swiglu7_silu / swiglu7_swiglu16 as separate explicit layers.
_ROUTED_SWIGLU_STEP = False
_ROUTED_SWIGLU_LIMIT = 0.0
_SHARED_SWIGLU_STEP = False
_SHARED_SWIGLU_LIMIT = 0.0

# swiglu7 / swiglu16 activation closure constants (Phase 4 L43/L44 specializations).
# L43 = swa_moe_swiglu7_silu (routed_lim=7.0, shared_lim=0.0); L44 =
# full_moe_swiglu7_swiglu16 (routed_lim=7.0, shared_lim=16.0). Mirrors baseline
# builder L611-614 (always-True dedicated specializations).
_ROUTED_SWIGLU7_STEP = True
_ROUTED_SWIGLU7_LIMIT = 7.0
_SHARED_SWIGLU16_STEP = True
_SHARED_SWIGLU16_LIMIT = 16.0

# Inlined kernel functions (DeepSeek form: pl.inline of the _func).
attention_full_inline = pl.inline(attention_full._func)
attention_swa_inline = pl.inline(attention_swa._func)
dense_mlp_inline = pl.inline(_dense_mlp_body_tp._func)


@pl.program
class WholeDecodeOpt:
    # ── TP all-reduce collective ────────────────────────────────────────
    # attention_full / attention_swa / _dense_mlp_body_tp 的 inlined body 里
    # 都调 ``self.tp_all_reduce(...)`` 汇集 o_proj / down_proj 的 partial sum。
    # pl.inline 把这些 body 拷进本 program 后，``self.tp_all_reduce`` 解析到本
    # program 的 method，所以必须在这里定义。定义按 baseline
    # decode_layer_single_chip_hidden.py 的 InCore 形态原样搬入（two-wave
    # completion barrier，expected=1/2 Ge），不改语义。
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
            [n_ranks, 1], pl.INT32
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
        sh_y = self.expert_shared_step_swiglu16(
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
        local_routed_y = self.expert_routed_step_swiglu7(
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
        local_routed_y = self.expert_routed_step_swiglu7(
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
        # ── MoE L3-L42 (40 layers) per-layer comm window stacks ──
        # (per-layer offset slice into one big stack; signal windows auto-zero,
        #  data windows do NOT — _zero_routed_y_buf handles routed_y_buf reset.)
        moe_attn_tmp_stack: pld.DistributedTensor[
            [NUM_MOE_LAYERS_TOTAL * BATCH, HIDDEN], pl.BF16
        ],
        moe_attn_signal_stack: pld.DistributedTensor[
            [NUM_MOE_LAYERS_TOTAL * COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        moe_pub_counts_stack: pld.DistributedTensor[
            [NUM_MOE_LAYERS_TOTAL * n_ranks * n_ranks, n_local_experts_pad], pl.INT32
        ],
        moe_count_done_sig_stack: pld.DistributedTensor[
            [NUM_MOE_LAYERS_TOTAL * n_ranks, 1], pl.INT32
        ],
        moe_recv_x_stack: pld.DistributedTensor[
            [NUM_MOE_LAYERS_TOTAL * local_recv_max, HIDDEN], pl.INT8
        ],
        moe_recv_scale_stack: pld.DistributedTensor[
            [NUM_MOE_LAYERS_TOTAL * local_recv_max, 8], pl.FP32
        ],
        moe_data_done_sig_stack: pld.DistributedTensor[
            [NUM_MOE_LAYERS_TOTAL * n_ranks, 1], pl.INT32
        ],
        moe_recv_r_route_stack: pld.DistributedTensor[
            [NUM_MOE_LAYERS_TOTAL * local_recv_max, idx_pad], pl.INT32
        ],
        moe_send_x_stack: pld.DistributedTensor[
            [NUM_MOE_LAYERS_TOTAL * local_recv_max, HIDDEN], pl.INT8
        ],
        moe_send_scale_stack: pld.DistributedTensor[
            [NUM_MOE_LAYERS_TOTAL * local_recv_max, 8], pl.FP32
        ],
        moe_send_route_stack: pld.DistributedTensor[
            [NUM_MOE_LAYERS_TOTAL * local_recv_max, idx_pad], pl.INT32
        ],
        moe_sh_tmp_stack: pld.DistributedTensor[
            [NUM_MOE_LAYERS_TOTAL * BATCH, HIDDEN], pl.BF16
        ],
        moe_sh_signal_stack: pld.DistributedTensor[
            [NUM_MOE_LAYERS_TOTAL * n_ranks, 1], pl.INT32
        ],
        moe_routed_y_buf_stack: pld.DistributedTensor[
            [NUM_MOE_LAYERS_TOTAL * n_routes_per_rank, HIDDEN], pl.BF16
        ],
        moe_combine_done_sig_stack: pld.DistributedTensor[
            [NUM_MOE_LAYERS_TOTAL * n_ranks, 1], pl.INT32
        ],
        moe_routed_src_buf_stack: pld.DistributedTensor[
            [NUM_MOE_LAYERS_TOTAL * local_recv_max, HIDDEN], pl.BF16
        ],
        my_rank: pl.Scalar[pl.INT32],
    ):
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
            pl.slice(dense_attn_signal_stack, [tp_size, 1], [0, 0]),
            pl.slice(dense_mlp_tmp_stack, [BATCH, HIDDEN], [0, 0]),
            pl.slice(dense_mlp_signal_stack, [tp_size, 1], [0, 0]),
            0,
            0,
            0,
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
                pl.slice(dense_attn_signal_stack, [tp_size, 1], [sig_off, 0]),
                pl.slice(dense_mlp_tmp_stack, [BATCH, HIDDEN], [win_off, 0]),
                pl.slice(dense_mlp_signal_stack, [tp_size, 1], [sig_off, 0]),
                norm_idx,
                0,
                0,
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
        # C1 protocol: per-layer offset slice into ONE big window stack per
        # kind (NOT single-window reuse — that would race the Ge(1) rendezvous
        # across layers). Pristine fixed-threshold expected=1 kept verbatim;
        # no moe_epoch. See memory moe-protocol-c1-handoff-loop-form.
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
            moe_pub_off = layer_idx * (n_ranks * n_ranks)
            moe_recv_off = layer_idx * local_recv_max
            moe_route_off = layer_idx * n_routes_per_rank
            h_moe = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            dbg_moe = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
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
                    dbg_moe,
                    resid_hold_moe,
                    pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [moe_win_off, 0]),
                    pl.slice(moe_attn_signal_stack, [tp_size, 1], [moe_sig_off, 0]),
                    pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [moe_pub_off, 0]),
                    pl.slice(moe_count_done_sig_stack, [n_ranks, 1], [moe_win_off, 0]),
                    pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [moe_recv_off, 0]),
                    pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [moe_recv_off, 0]),
                    pl.slice(moe_data_done_sig_stack, [n_ranks, 1], [moe_win_off, 0]),
                    pl.slice(moe_recv_r_route_stack, [local_recv_max, idx_pad], [moe_recv_off, 0]),
                    pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [moe_recv_off, 0]),
                    pl.slice(moe_send_scale_stack, [local_recv_max, 8], [moe_recv_off, 0]),
                    pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [moe_recv_off, 0]),
                    pl.slice(moe_sh_tmp_stack, [BATCH, HIDDEN], [moe_win_off, 0]),
                    pl.slice(moe_sh_signal_stack, [n_ranks, 1], [moe_win_off, 0]),
                    pl.slice(moe_routed_y_buf_stack, [n_routes_per_rank, HIDDEN], [moe_route_off, 0]),
                    pl.slice(moe_combine_done_sig_stack, [n_ranks, 1], [moe_win_off, 0]),
                    pl.slice(moe_routed_src_buf_stack, [local_recv_max, HIDDEN], [moe_recv_off, 0]),
                    norm_layer_idx,
                    0,
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
                    dbg_moe,
                    resid_hold_moe,
                    pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [moe_win_off, 0]),
                    pl.slice(moe_attn_signal_stack, [tp_size, 1], [moe_sig_off, 0]),
                    pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [moe_pub_off, 0]),
                    pl.slice(moe_count_done_sig_stack, [n_ranks, 1], [moe_win_off, 0]),
                    pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [moe_recv_off, 0]),
                    pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [moe_recv_off, 0]),
                    pl.slice(moe_data_done_sig_stack, [n_ranks, 1], [moe_win_off, 0]),
                    pl.slice(moe_recv_r_route_stack, [local_recv_max, idx_pad], [moe_recv_off, 0]),
                    pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [moe_recv_off, 0]),
                    pl.slice(moe_send_scale_stack, [local_recv_max, 8], [moe_recv_off, 0]),
                    pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [moe_recv_off, 0]),
                    pl.slice(moe_sh_tmp_stack, [BATCH, HIDDEN], [moe_win_off, 0]),
                    pl.slice(moe_sh_signal_stack, [n_ranks, 1], [moe_win_off, 0]),
                    pl.slice(moe_routed_y_buf_stack, [n_routes_per_rank, HIDDEN], [moe_route_off, 0]),
                    pl.slice(moe_combine_done_sig_stack, [n_ranks, 1], [moe_win_off, 0]),
                    pl.slice(moe_routed_src_buf_stack, [local_recv_max, HIDDEN], [moe_recv_off, 0]),
                    norm_layer_idx,
                    0,
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
        dbg_layer_43 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
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
        moe_pub_off_43 = 40 * (n_ranks * n_ranks)
        moe_recv_off_43 = 40 * local_recv_max
        moe_route_off_43 = 40 * n_routes_per_rank
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
            dbg_layer_43,
            resid_hold_layer_43,
            pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [moe_win_off_43, 0]),
            pl.slice(moe_attn_signal_stack, [tp_size, 1], [moe_sig_off_43, 0]),
            pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [moe_pub_off_43, 0]),
            pl.slice(moe_count_done_sig_stack, [n_ranks, 1], [moe_win_off_43, 0]),
            pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [moe_recv_off_43, 0]),
            pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [moe_recv_off_43, 0]),
            pl.slice(moe_data_done_sig_stack, [n_ranks, 1], [moe_win_off_43, 0]),
            pl.slice(moe_recv_r_route_stack, [local_recv_max, idx_pad], [moe_recv_off_43, 0]),
            pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [moe_recv_off_43, 0]),
            pl.slice(moe_send_scale_stack, [local_recv_max, 8], [moe_recv_off_43, 0]),
            pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [moe_recv_off_43, 0]),
            pl.slice(moe_sh_tmp_stack, [BATCH, HIDDEN], [moe_win_off_43, 0]),
            pl.slice(moe_sh_signal_stack, [n_ranks, 1], [moe_win_off_43, 0]),
            pl.slice(moe_routed_y_buf_stack, [n_routes_per_rank, HIDDEN], [moe_route_off_43, 0]),
            pl.slice(moe_combine_done_sig_stack, [n_ranks, 1], [moe_win_off_43, 0]),
            pl.slice(moe_routed_src_buf_stack, [local_recv_max, HIDDEN], [moe_recv_off_43, 0]),
            norm_layer_idx_43,
            0,
            my_rank,
        )
        prev_hidden = h_layer_43

        resid_hold_layer_44 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        dbg_layer_44 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
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
        moe_pub_off_44 = 41 * (n_ranks * n_ranks)
        moe_recv_off_44 = 41 * local_recv_max
        moe_route_off_44 = 41 * n_routes_per_rank
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
            dbg_layer_44,
            resid_hold_layer_44,
            pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [moe_win_off_44, 0]),
            pl.slice(moe_attn_signal_stack, [tp_size, 1], [moe_sig_off_44, 0]),
            pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts_pad], [moe_pub_off_44, 0]),
            pl.slice(moe_count_done_sig_stack, [n_ranks, 1], [moe_win_off_44, 0]),
            pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [moe_recv_off_44, 0]),
            pl.slice(moe_recv_scale_stack, [local_recv_max, 8], [moe_recv_off_44, 0]),
            pl.slice(moe_data_done_sig_stack, [n_ranks, 1], [moe_win_off_44, 0]),
            pl.slice(moe_recv_r_route_stack, [local_recv_max, idx_pad], [moe_recv_off_44, 0]),
            pl.slice(moe_send_x_stack, [local_recv_max, HIDDEN], [moe_recv_off_44, 0]),
            pl.slice(moe_send_scale_stack, [local_recv_max, 8], [moe_recv_off_44, 0]),
            pl.slice(moe_send_route_stack, [local_recv_max, idx_pad], [moe_recv_off_44, 0]),
            pl.slice(moe_sh_tmp_stack, [BATCH, HIDDEN], [moe_win_off_44, 0]),
            pl.slice(moe_sh_signal_stack, [n_ranks, 1], [moe_win_off_44, 0]),
            pl.slice(moe_routed_y_buf_stack, [n_routes_per_rank, HIDDEN], [moe_route_off_44, 0]),
            pl.slice(moe_combine_done_sig_stack, [n_ranks, 1], [moe_win_off_44, 0]),
            pl.slice(moe_routed_src_buf_stack, [local_recv_max, HIDDEN], [moe_recv_off_44, 0]),
            norm_layer_idx_44,
            0,
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
    ):
        dense_attn_tmp_stack_buf = pld.alloc_window_buffer(NUM_DENSE_LAYERS * BATCH * HIDDEN * 2)
        dense_attn_signal_stack_buf = pld.alloc_window_buffer(NUM_DENSE_LAYERS * COMM_CONTROL_SIGNAL_BYTES)
        dense_mlp_tmp_stack_buf = pld.alloc_window_buffer(NUM_DENSE_LAYERS * BATCH * HIDDEN * 2)
        dense_mlp_signal_stack_buf = pld.alloc_window_buffer(NUM_DENSE_LAYERS * COMM_CONTROL_SIGNAL_BYTES)
        # MoE L3-L42 (40 layers) per-layer comm window stacks.
        moe_attn_tmp_stack_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * BATCH * HIDDEN * 2)
        moe_attn_signal_stack_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * COMM_CONTROL_SIGNAL_BYTES)
        moe_pub_counts_stack_buf = pld.alloc_window_buffer(
            NUM_MOE_LAYERS_TOTAL * n_ranks * n_ranks * n_local_experts_pad * 4
        )
        moe_count_done_sig_stack_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * n_ranks * COMM_CONTROL_SIGNAL_BYTES)
        moe_recv_x_stack_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * local_recv_max * HIDDEN)
        moe_recv_scale_stack_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * local_recv_max * 8 * 4)
        moe_data_done_sig_stack_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * n_ranks * COMM_CONTROL_SIGNAL_BYTES)
        moe_recv_r_route_stack_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * local_recv_max * idx_pad * 4)
        moe_send_x_stack_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * local_recv_max * HIDDEN)
        moe_send_scale_stack_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * local_recv_max * 8 * 4)
        moe_send_route_stack_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * local_recv_max * idx_pad * 4)
        moe_sh_tmp_stack_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * BATCH * HIDDEN * 2)
        moe_sh_signal_stack_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * n_ranks * COMM_CONTROL_SIGNAL_BYTES)
        moe_routed_y_buf_stack_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * n_routes_per_rank * HIDDEN * 2)
        moe_combine_done_sig_stack_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * n_ranks * COMM_CONTROL_SIGNAL_BYTES)
        moe_routed_src_buf_stack_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * local_recv_max * HIDDEN * 2)
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
                           [NUM_MOE_LAYERS_TOTAL * n_ranks * n_ranks, n_local_experts_pad],
                           dtype=pl.INT32),
                pld.window(moe_count_done_sig_stack_buf, [NUM_MOE_LAYERS_TOTAL * n_ranks, 1],
                           dtype=pl.INT32),
                pld.window(moe_recv_x_stack_buf, [NUM_MOE_LAYERS_TOTAL * local_recv_max, HIDDEN],
                           dtype=pl.INT8),
                pld.window(moe_recv_scale_stack_buf, [NUM_MOE_LAYERS_TOTAL * local_recv_max, 8],
                           dtype=pl.FP32),
                pld.window(moe_data_done_sig_stack_buf, [NUM_MOE_LAYERS_TOTAL * n_ranks, 1],
                           dtype=pl.INT32),
                pld.window(moe_recv_r_route_stack_buf, [NUM_MOE_LAYERS_TOTAL * local_recv_max, idx_pad],
                           dtype=pl.INT32),
                pld.window(moe_send_x_stack_buf, [NUM_MOE_LAYERS_TOTAL * local_recv_max, HIDDEN],
                           dtype=pl.INT8),
                pld.window(moe_send_scale_stack_buf, [NUM_MOE_LAYERS_TOTAL * local_recv_max, 8],
                           dtype=pl.FP32),
                pld.window(moe_send_route_stack_buf, [NUM_MOE_LAYERS_TOTAL * local_recv_max, idx_pad],
                           dtype=pl.INT32),
                pld.window(moe_sh_tmp_stack_buf, [NUM_MOE_LAYERS_TOTAL * BATCH, HIDDEN],
                           dtype=pl.BF16),
                pld.window(moe_sh_signal_stack_buf, [NUM_MOE_LAYERS_TOTAL * n_ranks, 1],
                           dtype=pl.INT32),
                pld.window(moe_routed_y_buf_stack_buf, [NUM_MOE_LAYERS_TOTAL * n_routes_per_rank, HIDDEN],
                           dtype=pl.BF16),
                pld.window(moe_combine_done_sig_stack_buf, [NUM_MOE_LAYERS_TOTAL * n_ranks, 1],
                           dtype=pl.INT32),
                pld.window(moe_routed_src_buf_stack_buf, [NUM_MOE_LAYERS_TOTAL * local_recv_max, HIDDEN],
                           dtype=pl.BF16),
                r,
                device=r,
            )


whole_decode_opt = WholeDecodeOpt
