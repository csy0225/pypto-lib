# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Step3p5 prefill whole-network hidden-only single-chip program.

Canonical symbol: ``whole_prefill_step3p5``.  P2 whole-net body (PROGRESS.md
Session 20续3 方案 B): per-layer-type orchestration methods mirror decode
``WholeDecodeStep3p5`` (decode_fwd.py:364/436/1851/2049/3111/2919); MoE stage
methods migrated verbatim from ``PrefillLayerMoE`` (prefill_fwd.py:878-2250)
preserving the Session 20 pl.at / InCore / Inline paradigms.  Prefill uses BF16
MoE (W8A8 later), the prefill token-tiled TP ring (prefill_fwd.py:878),
``PREFILL_T=128`` token dim with per-tile (BATCH=16) MoE, ``position_ids``
resident, KV-write semantics, and stacked per-layer EP windows (no moe_epoch).

Deviation from decode (PROGRESS Session 9-10): decode's per-layer-type methods
carry ``attrs={"inline_orchestration": True}`` so they inline into
``whole_chip_orch``.  Prefill's MoE methods contain a per-tile
``pl.unroll(PREFILL_TILE_COUNT)`` loop whose DT params + pl.at scopes trigger a
pypto CommCtx materialization failure (orchestration_codegen.cpp:1197) under
inline_orchestration — decode avoids this because its MoE body is a single
BATCH=16 pass with no per-tile loop.  The prefill methods are therefore plain
``Orchestration`` (subroutine calls, mirror the single-layer
``PrefillLayerMoE.chip_orch`` that compiles 45/45); ``whole_chip_orch``
(Orchestration) calls them as nested subroutines.  See RECOVERY_PROGRESS.md
§"P2 整网 body 填入方案".
"""
from __future__ import annotations

# ruff: noqa: F401
# Config/chunk constants below are referenced by the pl.inline attention / MLP /
# MoE bodies, not by this module's own statements; ruff cannot trace that use.

import pypto.language as pl
import pypto.language.distributed as pld
# NOTE (mirror decode_fwd.py:30-35): attention_full_prefill / attention_swa_prefill
# / _prefill_dense_mlp_body_tp kernel bodies are captured via ``pl.inline`` and
# their referenced config constants resolve against THIS module's globals, so
# every constant those bodies use must be imported here once.
from models.step3p5.config import (
    ATTN_SCALE,
    BATCH,
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
# prefill attention bodies reference their own module aliases; bind them here so
# the inlined body's globals resolve (mirror decode_fwd.py:98-103).
LAYER_QHIDDEN_ROWS_DYN = LAYER_QHIDDEN_ROWS_DYN_FULL
from models.step3p5.dispatch import N_RANKS_PAD, PER_RANK_BUCKETS
from models.step3p5.prefill_attention_full import attention_full_prefill
from models.step3p5.prefill_attention_swa import attention_swa_prefill
from models.step3p5.prefill_fwd import _prefill_dense_mlp_body_tp
from models.step3p5.prefill_qkv_proj_rope import PREFILL_T, TOK_TILE

# ── per-rank slice widths / aliases (mirror decode_fwd.py:108-116) ──────────
INTER_LOCAL = INTERMEDIATE_LOCAL
hidden_q_full = HIDDEN_Q_FULL_LOCAL
hidden_q_swa = HIDDEN_Q_SWA_LOCAL
nh_full_pad = NUM_HEADS_FULL_LOCAL_PAD
nh_swa_pad = NUM_HEADS_SWA_LOCAL_PAD
KV_HIDDEN_LOCAL_R = KV_HIDDEN_LOCAL
rotary_dim_full = ROTARY_HALF_FULL * 2
rotary_dim_swa = ROTARY_HALF_SWA * 2
tp_size = TP_WORLD_SIZE

# ── layer schedule (mirror decode_fwd.py:118-124) ───────────────────────────
NUM_DENSE_LAYERS = 3
NUM_SWA_DENSE_LAYERS = 2
N_FULL_ATTN_LAYERS = 12
N_SWA_ATTN_LAYERS = 33
NUM_FULL_MOE_LAYERS = 10
NUM_SWA_MOE_LAYERS = 30
NUM_MOE_LAYERS = NUM_FULL_MOE_LAYERS + NUM_SWA_MOE_LAYERS          # 40 (L3..L42)
NUM_MOE_LAYERS_TOTAL = NUM_MOE_LAYERS + 2                          # 42 (incl L43/L44)

# ── MoE expert layout + lower-case aliases (mirror decode_fwd.py:140-194) ───
N_EXPERTS = MOE_NUM_EXPERTS
n_ranks = tp_size
n_ranks_pad = N_RANKS_PAD
n_local_experts = MOE_NUM_EXPERTS_LOCAL
n_local_experts_pad = ((n_local_experts + 7) // 8) * 8
inter = MOE_INTERMEDIATE
sh_inter_local = SHARE_EXPERT_DIM_LOCAL
TOPK = 8
topk = TOPK
per_rank_buckets = PER_RANK_BUCKETS
tp_chunk = HIDDEN // tp_size
sh_tp_chunk = HIDDEN // tp_size
# Prefill per-tile (BATCH) routed-row upper bound = n_ranks * BATCH * TOPK = 1024.
local_recv_max = n_ranks * BATCH * TOPK
n_routes_per_rank = BATCH * TOPK
PREFILL_TILE_COUNT = PREFILL_T // BATCH

# ── cross-rank control signal footprint (mirror decode_fwd.py:128-129) ──────
COMM_CONTROL_SIGNAL_BYTES = 512
COMM_SIGNAL_STRIDE_I32 = COMM_CONTROL_SIGNAL_BYTES // 4

# ── resident runtime token-count ABI (mirror decode_fwd.py:199-204) ─────────
NUM_TOKENS_STORAGE_I32 = 128
NUM_TOKENS_RUNTIME = NUM_TOKENS_STORAGE_I32

# ── MoE kernel chunk constants (mirror prefill_fwd.py:232-276) ──────────────
ROUTER_SCORE_PAD = 512
ROUTER_TOPK_PAD = 16
ROUTER_SORT_PAD = ROUTER_TOPK_PAD * 2
ROUTER_GATE_K_CHUNK = 512
ROUTER_GATE_N_CHUNK = 32
ROUTER_FP32_NEG_INF = -3.4028235e38
ROUTER_SCALE = 3.0
ROUTED_GATE_K_CHUNK = 64
ROUTED_GATE_N_CHUNK = 64
ROUTED_DOWN_K_CHUNK = 64
ROUTED_DOWN_N_CHUNK = 128
ROUTED_MAX_TILE = local_recv_max
RECV_TILE = 32
N_RECV_TILES = local_recv_max // RECV_TILE
# W8A8 routed-expert quant constants (mirror decode_fwd.py:182-184, design
# §4.2 step 2).  DISPATCH_SCALE_COLS=1 = one per-token activation scale; the
# per-row aux FP32 slab is padded to dispatch_aux_pad=8 for PTOAS tile
# alignment (decode recv_aux ABI, decode_fwd.py:184).  Logical col 0 carries
# the per-token dequant scale; cols 1..7 are physical padding.
DISPATCH_SCALE_COLS = 1
dispatch_aux_pad = 8
# SwiGLU7 routed activation limit for L43/L44 (mirror decode_fwd.py:231).
_ROUTED_SWIGLU7_LIMIT = 7.0
SHARED_GATE_K_CHUNK = 256
SHARED_GATE_N_CHUNK = sh_inter_local
SHARED_DOWN_K_CHUNK = sh_inter_local
SHARED_DOWN_N_CHUNK = 256

# ── module-level pl.inline atom kernels (PROGRESS Session 20续3 方案 B) ─────
# Captures the dual-index (norm_layer_idx + attn/mlp_layer_idx) + gate_r prefill
# bodies.  self.tp_all_reduce resolves to WholePrefillStep3p5.tp_all_reduce.
attention_full_inline = pl.inline(attention_full_prefill._func)
attention_swa_inline = pl.inline(attention_swa_prefill._func)
dense_mlp_inline = pl.inline(_prefill_dense_mlp_body_tp._func)

# P2 whole-net body landed (PROGRESS Session 20续3).  Holder reads this flag to
# decide whether to compile the real 45-layer dispatch.
IS_SCAFFOLD = False


@pl.program
class WholePrefillStep3p5:
    """Prefill dual of WholeDecodeStep3p5 (decode_fwd.py:241).  Single
    @pl.program running the 45-layer prefill forward, returning pre-final-norm
    BF16 hidden (hidden-only boundary, design §2.2 invariant 1)."""

    @pl.function(type=pl.FunctionType.InCore)
    def tp_all_reduce(
        self,
        local: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
        tmp_window: pld.DistributedTensor[[PREFILL_T, tp_chunk], pl.BF16],
        signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]:
        group_size = tp_size
        # Inline shape constants — pypto's tile shape inference cannot follow
        # Python-level aliases like ``t_rows = PREFILL_T`` past load/remote_load
        # boundaries (it preserves the alias name in the tile type, which
        # then mismatches the concrete shape from sibling pl.load calls).
        for step in pl.range(group_size - 1):
            send_idx = (my_rank - step + group_size) % group_size
            recv_idx = (my_rank - step - 1 + group_size) % group_size
            next_rank = (my_rank + 1) % group_size
            prev_rank = (my_rank - 1 + group_size) % group_size
            # Token-tiled Vec I/O (BATCH rows per sub-tile) so the per-step
            # working set fits UB; the ring comm stays whole-window with one
            # notify/wait per step (signal counts unchanged).
            for tt in pl.range(PREFILL_TILE_COUNT):
                ttr = tt * BATCH
                send_tile = pl.load(
                    local, [ttr, send_idx * tp_chunk], [BATCH, tp_chunk],
                )
                pl.store(send_tile, [ttr, 0], tmp_window)
            pld.system.notify(
                target=signal_window, peer=next_rank,
                offsets=[my_rank, 0], value=1,
                op=pld.NotifyOp.AtomicAdd,
            )
            pld.system.wait(
                signal=signal_window, offsets=[prev_rank, 0],
                expected=pl.cast(step + 1, pl.INT32), cmp=pld.WaitCmp.Ge,
            )
            for tt in pl.range(PREFILL_TILE_COUNT):
                ttr = tt * BATCH
                recv_tile = pld.tile.remote_load(
                    tmp_window, peer=prev_rank,
                    offsets=[ttr, 0], shape=[BATCH, tp_chunk],
                )
                old_tile = pl.load(
                    local, [ttr, recv_idx * tp_chunk], [BATCH, tp_chunk],
                )
                # PTOAS A2/A3 ``tadd`` doesn't support bf16; upcast to f32,
                # add, then downcast for the store.
                summed_fp32 = pl.add(
                    pl.cast(old_tile, target_type=pl.FP32),
                    pl.cast(recv_tile, target_type=pl.FP32),
                )
                pl.store(
                    pl.cast(summed_fp32, target_type=pl.BF16),
                    [ttr, recv_idx * tp_chunk], local,
                )

        for step in pl.range(group_size - 1):
            send_idx = (my_rank - step + 1 + group_size) % group_size
            recv_idx = (my_rank - step + group_size) % group_size
            next_rank = (my_rank + 1) % group_size
            prev_rank = (my_rank - 1 + group_size) % group_size
            for tt in pl.range(PREFILL_TILE_COUNT):
                ttr = tt * BATCH
                send_tile = pl.load(
                    local, [ttr, send_idx * tp_chunk], [BATCH, tp_chunk],
                )
                pl.store(send_tile, [ttr, 0], tmp_window)
            pld.system.notify(
                target=signal_window, peer=next_rank,
                offsets=[my_rank, 0], value=1,
                op=pld.NotifyOp.AtomicAdd,
            )
            pld.system.wait(
                signal=signal_window, offsets=[prev_rank, 0],
                expected=pl.cast(group_size - 1 + step + 1, pl.INT32),
                cmp=pld.WaitCmp.Ge,
            )
            for tt in pl.range(PREFILL_TILE_COUNT):
                ttr = tt * BATCH
                recv_tile = pld.tile.remote_load(
                    tmp_window, peer=prev_rank,
                    offsets=[ttr, 0], shape=[BATCH, tp_chunk],
                )
                pl.store(recv_tile, [ttr, recv_idx * tp_chunk], local)
        return local

    # ===================================================================
    # Phase X.10 — inlined EpTpMoE @pl.function methods.
    # Bodies copied verbatim from ``prefill_moe.PrefillMoE`` (which in
    # turn mirrors ``moe.EpTpMoE``). The activation choice (plain SiLU
    # vs. SwigluStep@7/16) is baked at factory build time via the
    # ``False`` / ``False`` Python closure
    # constants — only one branch is emitted per specialisation. The
    # per-tile token count is BATCH (= decode-T); ``chip_orch`` below
    # drives PREFILL_TILE_COUNT independent tile invocations to cover
    # the full PREFILL_T axis.
    # ===================================================================

    # ---------- Per-tile TP all_reduce (BATCH rows, used by shared lane) ----
    # NOTE: distinct from the outer PREFILL_T-sized ``tp_all_reduce`` above —
    # the shared-expert lane operates on per-tile [BATCH, HIDDEN] tiles, so
    # we use a dedicated tile-sized ring body. Same algorithm, baked
    # t_rows=BATCH.
    @pl.function(type=pl.FunctionType.InCore)
    def _tp_all_reduce_moe(
        self,
        local: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        tmp_window: pld.DistributedTensor[[BATCH, sh_tp_chunk], pl.BF16],
        signal_window: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
        group_size = n_ranks
        t_rows = BATCH
        d_cols = HIDDEN
        chunk = d_cols // group_size

        for step in pl.range(group_size - 1):
            send_idx = (my_rank - step + group_size) % group_size
            recv_idx = (my_rank - step - 1 + group_size) % group_size
            next_rank = (my_rank + 1) % group_size
            prev_rank = (my_rank - 1 + group_size) % group_size
            send_tile = pl.load(
                local, [0, send_idx * chunk], [t_rows, chunk],
            )
            pl.store(send_tile, [0, 0], tmp_window)
            pld.system.notify(
                target=signal_window, peer=next_rank,
                offsets=[my_rank, 0], value=1,
                op=pld.NotifyOp.AtomicAdd,
            )
            pld.system.wait(
                signal=signal_window, offsets=[prev_rank, 0],
                expected=step + 1, cmp=pld.WaitCmp.Ge,
            )
            recv_tile = pld.tile.remote_load(
                tmp_window, peer=prev_rank,
                offsets=[0, 0], shape=[t_rows, chunk],
            )
            old_tile = pl.load(
                local, [0, recv_idx * chunk], [t_rows, chunk],
            )
            pl.store(
                pl.add(old_tile, recv_tile),
                [0, recv_idx * chunk], local,
            )

        for step in pl.range(group_size - 1):
            send_idx = (my_rank - step + 1 + group_size) % group_size
            recv_idx = (my_rank - step + group_size) % group_size
            next_rank = (my_rank + 1) % group_size
            prev_rank = (my_rank - 1 + group_size) % group_size
            send_tile = pl.load(
                local, [0, send_idx * chunk], [t_rows, chunk],
            )
            pl.store(send_tile, [0, 0], tmp_window)
            pld.system.notify(
                target=signal_window, peer=next_rank,
                offsets=[my_rank, 0], value=1,
                op=pld.NotifyOp.AtomicAdd,
            )
            pld.system.wait(
                signal=signal_window, offsets=[prev_rank, 0],
                expected=pl.cast(group_size - 1 + step + 1, pl.INT32),
                cmp=pld.WaitCmp.Ge,
            )
            recv_tile = pld.tile.remote_load(
                tmp_window, peer=prev_rank,
                offsets=[0, 0], shape=[t_rows, chunk],
            )
            pl.store(recv_tile, [0, recv_idx * chunk], local)
        return local

    # ---------- Collective: EP all_to_all ----------
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

        # Gate N-chunk SPMD fan-out (mirrors decode_fwd.py:516-608,
        # PROGRESS Session 4 step 3 + Session 6 step 7).  The full
        # ``x_fp32`` cast is dropped in favour of per-K-chunk casts so the
        # ``[BATCH, HIDDEN] FP32`` tile never lives alongside the matmul
        # intermediates; the expert dimension is fanned out via SPMD so
        # each task's weight tile is ``[K_CHUNK, N_CHUNK] = 64 KiB``.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_init"):
            score_buf[:, :] = pl.full(
                [BATCH, ROUTER_SCORE_PAD], dtype=pl.FP32, value=0.0,
            )
            biased_buf[:, :] = pl.full(
                [BATCH, ROUTER_SCORE_PAD],
                dtype=pl.FP32, value=ROUTER_FP32_NEG_INF,
            )

        # Sigmoid is applied per N-chunk on the matmul output (the vec op
        # needs a TileType), NOT on the assembled full logits — assembling
        # first would force a ``[BATCH, N_EXPERTS] FP32`` tile back into
        # Vec UB.  PyPTO does not accept an SPMD region nested inside an
        # AT scope, so the fan-out lives at function scope (decode comment
        # decode_fwd.py:513-515).
        for nb in pl.spmd(
            N_EXPERTS // ROUTER_GATE_N_CHUNK,
            name_hint="gate_expert_fanout",
        ):
            n0 = nb * ROUTER_GATE_N_CHUNK
            raw0 = pl.cast(
                pl.slice(x, [BATCH, ROUTER_GATE_K_CHUNK], [0, 0]),
                target_type=pl.FP32,
            )
            w0 = pl.slice(
                gate_w,
                [ROUTER_GATE_K_CHUNK, ROUTER_GATE_N_CHUNK],
                [0, n0],
            )
            logits_n = pl.matmul(raw0, w0, out_dtype=pl.FP32)
            for kb in pl.range(1, HIDDEN // ROUTER_GATE_K_CHUNK):
                k0 = kb * ROUTER_GATE_K_CHUNK
                rawk = pl.cast(
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
                logits_n = pl.matmul_acc(logits_n, rawk, wk)
            score_n_chunk = pl.recip(
                pl.add(pl.exp(pl.neg(logits_n)), 1.0),
            )
            bias_chunk = pl.slice(
                router_bias, [ROUTER_GATE_N_CHUNK], [n0],
            )
            bias_row_chunk = pl.reshape(
                bias_chunk, [1, ROUTER_GATE_N_CHUNK],
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
                srt = pl.mrgsort(srt[:, 0:512], srt[:, 512:1024])
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
        # W8A8: x is the INT8 dispatch payload + per-token dequant scale
        # (decode_fwd.py:811-812, design §4.2 step 2).
        x: pl.Tensor[[BATCH, HIDDEN], pl.INT8],
        x_scale: pl.Tensor[[BATCH, DISPATCH_SCALE_COLS], pl.FP32],
        indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
        send_counts_per_bucket: pl.Tensor[[per_rank_buckets], pl.INT32],
        send_offsets_per_rank: pl.Tensor[[n_ranks], pl.INT32],
        # GM send buffer (DistributedTensor): written in place so the
        # original parameter SSA value reaches ep_all_to_all with its
        # materialized CommCtx intact.  pl.assemble re-assignment produced
        # a derived SSA value whose CommCtx get_comm_ctx could not resolve
        # (PROGRESS.md Session 20续).  pl.store(pl.load(...)) writes the GM
        # tile in place without re-assigning send_buf.
        send_buf: pld.DistributedTensor[
            [local_recv_max, HIDDEN], pl.INT8
        ],
        # slot_map records the send_buf row assigned to each (t, k) route so a
        # separate scope can write the per-row aux scale DT in isolation
        # (writing two DTs in one Inline function broke CommCtx materialization
        # for send_buf; slot_map is a regular tensor with no CommCtx).
        slot_map: pl.Out[pl.Tensor[[BATCH, TOPK], pl.INT32]],
        cursor_per_bucket: pl.Tensor[[per_rank_buckets], pl.INT32],
        bucket_offset: pl.Tensor[[per_rank_buckets], pl.INT32],
    ):
        # Wrap the pack body in an InCore scope (PROGRESS.md Session 20续
        # item 9, mirrors decode dispatch_push in decode_fwd.py:927).  The
        # pl.load(x, ..., target_memory=Vec) tile op would otherwise inline
        # into chip_orch (Orchestration) and InitMemRef would materialize it
        # as a misplaced tile.alloc at the Orchestration body top.
        with pl.at(
            level=pl.Level.CORE_GROUP, name_hint="dispatch_pack",
        ):
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
                    # In-place GM write: load the source row as a tile and
                    # store it into send_buf at [slot, 0].  No re-assignment
                    # of send_buf, so ep_all_to_all receives the original
                    # CommCtx.
                    pl.store(
                        pl.load(x, [t, 0], [1, HIDDEN]),
                        [slot, 0],
                        send_buf,
                    )
                    # Record the assigned send_buf row for (t, k) so the
                    # separate aux-pack scope can write the per-token scale.
                    pl.write(slot_map, [t, k], slot_i32)
                    pl.write(
                        cursor_per_bucket, [bkt],
                        pl.cast(slot_i32 + 1, pl.INT32),
                    )
        return slot_map

    # InCore (PROGRESS.md Session 20 item 7): pure scalar compute that reads
    # pub_counts (DistributedTensor) with no pl.at.  As an Inline function
    # the bare pl.read on a DistributedTensor inlines into chip_orch
    # (Orchestration) and materializes as a misplaced op; InCore keeps the
    # read inside an InCore body (mirrors decode _norm_quant_moe_input).
    @pl.function(type=pl.FunctionType.InCore)
    def _build_local_expert_csr(
        self,
        pub_counts: pld.DistributedTensor[
            [n_ranks * n_ranks, n_local_experts], pl.INT32
        ],
        local_expert_offset: pl.Tensor[[n_local_experts], pl.INT32],
        local_expert_count: pl.Tensor[[n_local_experts], pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
    ):
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

    # InCore (PROGRESS.md Session 20 item 8): pure scalar compute that reads
    # pub_counts (DistributedTensor); cursor is a small local DDR scratch.
    # Inline would inline the bare pl.read into chip_orch (Orchestration).
    @pl.function(type=pl.FunctionType.InCore)
    def _build_inverse_map(
        self,
        indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
        pub_counts: pld.DistributedTensor[
            [n_ranks * n_ranks, n_local_experts], pl.INT32
        ],
        inverse_map: pl.Tensor[[BATCH, TOPK], pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
    ):
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

    @pl.function(type=pl.FunctionType.Inline)
    def dispatch_step(  # noqa: PLR0913
        self,
        # W8A8: x is INT8 + per-token dequant scale (decode_fwd.py:811-812,
        # design §4.2 step 2).
        x: pl.Tensor[[BATCH, HIDDEN], pl.INT8],
        x_scale: pl.Tensor[[BATCH, DISPATCH_SCALE_COLS], pl.FP32],
        expert_indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
        local_routed_x_out: pl.Out[
            pl.Tensor[[local_recv_max, HIDDEN], pl.INT8]
        ],
        local_routed_x_scale_out: pl.Out[
            pl.Tensor[[1, local_recv_max], pl.FP32]
        ],
        local_expert_offset: pl.Out[
            pl.Tensor[[n_local_experts], pl.INT32]
        ],
        local_expert_count: pl.Out[
            pl.Tensor[[n_local_experts], pl.INT32]
        ],
        inverse_map: pl.Out[pl.Tensor[[BATCH, TOPK], pl.INT32]],
        pub_counts: pld.DistributedTensor[
            [n_ranks * n_ranks, n_local_experts], pl.INT32
        ],
        count_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        recv_x: pld.DistributedTensor[
            [local_recv_max, HIDDEN], pl.INT8
        ],
        data_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        # GM send buffer (mirrors decode moe.py): the InCore create_tensor
        # of [local_recv_max, HIDDEN] = 8 MiB overflows the 188 KiB Vec UB,
        # so the buffer is allocated on GM by host_orch and passed in as a
        # DistributedTensor.  See PROGRESS.md Session 4.
        send_buf: pld.DistributedTensor[
            [local_recv_max, HIDDEN], pl.INT8
        ],
        # Per-routed-row aux slab (decode recv_aux ABI, decode_fwd.py:830-832):
        # col 0 = per-token dequant scale, cols 1..7 physical FP32 padding
        # (dispatch_aux_pad) for PTOAS tile alignment.  Prefill does not
        # transport the route weight through the a2a (it is applied locally
        # in combine_step via expert_weights + inverse_map, numerically
        # equivalent to decode folding it at the expert down-proj).
        send_aux: pld.DistributedTensor[
            [local_recv_max, dispatch_aux_pad], pl.FP32
        ],
        recv_aux: pld.DistributedTensor[
            [local_recv_max, dispatch_aux_pad], pl.FP32
        ],
        my_rank: pl.Scalar[pl.INT32],
    ) -> tuple[
        pl.Tensor[[local_recv_max, HIDDEN], pl.INT8],
        pl.Tensor[[1, local_recv_max], pl.FP32],
        pl.Tensor[[n_local_experts], pl.INT32],
        pl.Tensor[[n_local_experts], pl.INT32],
        pl.Tensor[[BATCH, TOPK], pl.INT32]
    ]:
        send_counts_bkt = pl.create_tensor(
            [per_rank_buckets], dtype=pl.INT32,
        )
        send_counts_rank = pl.create_tensor([n_ranks_pad], dtype=pl.INT32)
        send_offsets_rank = pl.create_tensor([n_ranks_pad], dtype=pl.INT32)
        self._histogram_and_prefix_sum(
            expert_indices,
            send_counts_bkt, send_counts_rank, send_offsets_rank,
        )

        # Publish per-bucket counts to peers and signal count_done in one
        # InCore scope (PROGRESS.md Session 20 item 4, mirrors decode
        # dispatch_meta in decode_fwd.py:863).  Bare pl.write on a
        # DistributedTensor and bare pld.system.notify inline into chip_orch
        # (Orchestration) and materialize as misplaced ops; the count_done
        # notify is kept inside the same scope so it fires after publishes.
        with pl.at(
            level=pl.Level.CORE_GROUP, name_hint="dispatch_pub_counts",
        ):
            for peer in pl.range(n_ranks):
                for e in pl.range(n_local_experts):
                    v = pl.read(
                        send_counts_bkt, [peer * n_local_experts + e],
                    )
                    if peer == my_rank:
                        pl.write(
                            pub_counts,
                            [my_rank * n_ranks + my_rank, e],
                            v,
                        )
                    else:
                        pld.system.notify(
                            target=pub_counts,
                            peer=peer,
                            offsets=[my_rank * n_ranks + peer, e],
                            value=v,
                            op=pld.NotifyOp.Set,
                        )

            for peer in pl.range(n_ranks):
                if peer != my_rank:
                    pld.system.notify(
                        target=count_done_sig,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.Set,
                    )
        # Wait for all peers' count_done signals in an InCore scope
        # (PROGRESS.md Session 20 item 5).  Bare pld.system.wait inlines
        # into chip_orch (Orchestration) and materializes as a misplaced op.
        with pl.at(
            level=pl.Level.CORE_GROUP, name_hint="dispatch_wait_counts",
        ):
            for src in pl.range(n_ranks):
                if src != my_rank:
                    pld.system.wait(
                        signal=count_done_sig,
                        offsets=[src, 0],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )

        # send_buf is now a GM DistributedTensor parameter (allocated by
        # host_orch); the InCore create_tensor of 8 MiB overflowed the Vec
        # UB.  cursor_bkt / bucket_offset stay on-chip (small INT32).
        cursor_bkt = pl.create_tensor(
            [per_rank_buckets], dtype=pl.INT32,
        )
        bucket_offset = pl.create_tensor(
            [per_rank_buckets], dtype=pl.INT32,
        )
        slot_map = pl.create_tensor([BATCH, TOPK], dtype=pl.INT32)
        slot_map = self._pack_send_payload(
            x, x_scale, expert_indices,
            send_counts_bkt, send_offsets_rank,
            send_buf, slot_map, cursor_bkt, bucket_offset,
        )
        # Write the per-row aux scale DT in its own scope via pld.tensor.put
        # (decode-proven DT write, decode_fwd.py:959 — pl.store to send_aux
        # alongside send_buf in the Inline dispatch_step broke CommCtx
        # materialization for send_buf).  slot_map[t,k] gives the send_buf row
        # assigned to route (t,k); self-put (peer=my_rank) writes col 0.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_aux_pack"):
            for t in pl.range(BATCH):
                for k in pl.range(TOPK):
                    slot = pl.cast(pl.read(slot_map, [t, k]), pl.INDEX)
                    pl.store(
                        pl.load(x_scale, [t, 0], [1, DISPATCH_SCALE_COLS]),
                        [slot, 0],
                        send_aux,
                    )

        # Reduce peer-published counts into per-src recv_counts and prefix
        # into recv_offsets (PROGRESS.md Session 20 item 6).  pl.read on the
        # DistributedTensor pub_counts would inline into chip_orch
        # (Orchestration) as a misplaced op; create_tensor stays outside
        # (DDR scratch) and the read/write go inside the scope.
        recv_counts = pl.create_tensor([n_ranks_pad], dtype=pl.INT32)
        recv_offsets = pl.create_tensor([n_ranks_pad], dtype=pl.INT32)
        with pl.at(
            level=pl.Level.CORE_GROUP, name_hint="dispatch_recv_counts",
        ):
            for src in pl.range(n_ranks):
                acc = pl.cast(0, pl.INT32)
                for e in pl.range(n_local_experts):
                    acc = acc + pl.read(
                        pub_counts, [src * n_ranks + my_rank, e],
                    )
                pl.write(recv_counts, [src], pl.cast(acc, pl.INT32))
            pl.write(recv_offsets, [0], pl.cast(0, pl.INT32))
            for r in pl.range(1, n_ranks):
                prev_off = pl.read(recv_offsets, [r - 1])
                prev_cnt = pl.read(recv_counts, [r - 1])
                pl.write(
                    recv_offsets, [r],
                    pl.cast(prev_off + prev_cnt, pl.INT32),
                )

        """Pull-side variable-length token-level all-to-all over EP."""
        group_size = n_ranks
        d_cols = HIDDEN

        # All body ops are cross-core (load/store on DistributedTensor,
        # notify/wait, remote_load).  Wrap in CORE_GROUP scopes
        # (PROGRESS.md Session 20: bare cross-core ops inlined into the
        # caller become Misplaced / unregistered).  Self-copy and the
        # notify/wait/remote-load phases are separate scopes so the
        # OutlineIncoreScopes pass can outline each phase cleanly.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="ep_a2a_self_copy"):
            n_self = pl.cast(pl.read(send_counts_rank, [my_rank]), pl.INDEX)
            s_off_self = pl.cast(pl.read(send_offsets_rank, [my_rank]), pl.INDEX)
            r_off_self = pl.cast(pl.read(recv_offsets, [my_rank]), pl.INDEX)
            for r in pl.range(n_self):
                self_tile = pl.load(
                    send_buf, [s_off_self + r, 0], [1, d_cols],
                )
                pl.store(self_tile, [r_off_self + r, 0], recv_x)
        # Parallel aux (per-token scale) self-copy in its own scope (W8A8).
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="ep_a2a_aux_self_copy"):
            n_self_aux = pl.cast(pl.read(send_counts_rank, [my_rank]), pl.INDEX)
            s_off_self_aux = pl.cast(pl.read(send_offsets_rank, [my_rank]), pl.INDEX)
            r_off_self_aux = pl.cast(pl.read(recv_offsets, [my_rank]), pl.INDEX)
            for r in pl.range(n_self_aux):
                self_aux_tile = pl.load(
                    send_aux, [s_off_self_aux + r, 0], [1, dispatch_aux_pad],
                )
                pl.store(self_aux_tile, [r_off_self_aux + r, 0], recv_aux)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="ep_a2a_notify"):
            for peer in pl.range(group_size):
                if peer != my_rank:
                    pld.system.notify(
                        target=data_done_sig,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.Set,
                    )

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="ep_a2a_wait"):
            for src in pl.range(group_size):
                if src != my_rank:
                    pld.system.wait(
                        signal=data_done_sig,
                        offsets=[src, 0],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="ep_a2a_remote_gather"):
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
                            send_buf,
                            peer=peer,
                            offsets=[r_off + r, 0],
                            shape=[1, d_cols],
                        )
                        pl.store(peer_tile, [r_off + r, 0], recv_x)
        # Parallel aux (per-token scale) remote gather in its own scope (W8A8).
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="ep_a2a_aux_remote_gather"):
            for peer in pl.range(group_size):
                if peer != my_rank:
                    n_recv = pl.cast(
                        pl.read(recv_counts, [peer]), pl.INDEX,
                    )
                    r_off = pl.cast(
                        pl.read(recv_offsets, [peer]), pl.INDEX,
                    )
                    for r in pl.range(n_recv):
                        peer_aux_tile = pld.tile.remote_load(
                            send_aux,
                            peer=peer,
                            offsets=[r_off + r, 0],
                            shape=[1, dispatch_aux_pad],
                        )
                        pl.store(peer_aux_tile, [r_off + r, 0], recv_aux)


    # ---------- Stage 1: gate (local, replicated) ----------

        self._build_local_expert_csr(
            pub_counts,
            local_expert_offset, local_expert_count,
            my_rank,
        )
        # Gather routed rows from recv_x into the expert-contiguous
        # local_routed_x_out (PROGRESS.md Session 20 item 1, mirrors decode
        # dispatch_gather in decode_fwd.py:1012).  pl.load(recv_x, ...,
        # target_memory=Vec) is the tile op InitMemRef materializes as a Vec
        # tile.alloc; wrapping it in an InCore scope lets
        # OutlineIncoreScopes outline it so the alloc lands in the InCore
        # body instead of chip_orch (Orchestration).
        with pl.at(
            level=pl.Level.CORE_GROUP, name_hint="dispatch_gather",
        ):
            running = pl.cast(0, pl.INT32)
            for e in pl.range(n_local_experts):
                for src in pl.range(n_ranks):
                    n = pl.cast(
                        pl.read(pub_counts, [src * n_ranks + my_rank, e]),
                        pl.INDEX,
                    )
                    src_base = pl.cast(
                        pl.read(recv_offsets, [src]), pl.INDEX,
                    )
                    src_e_off = pl.cast(0, pl.INT32)
                    for prev_e in pl.range(n_local_experts):
                        if prev_e < e:
                            src_e_off = src_e_off + pl.read(
                                pub_counts,
                                [src * n_ranks + my_rank, prev_e],
                            )
                    for row in pl.range(n):
                        src_row = (
                            src_base
                            + pl.cast(src_e_off, pl.INDEX) + row
                        )
                        dst_row = pl.cast(running, pl.INDEX) + row
                        tile = pl.load(recv_x, [src_row, 0], [1, HIDDEN])
                        pl.store(tile, [dst_row, 0], local_routed_x_out)
                        # Gather the per-token dequant scale alongside the
                        # activation (decode_fwd.py:1033-1036).
                        scale_tile = pl.load(
                            recv_aux, [src_row, 0], [1, DISPATCH_SCALE_COLS],
                        )
                        pl.store(
                            scale_tile, [0, dst_row], local_routed_x_scale_out,
                        )
                    running = running + pl.cast(n, pl.INT32)

        self._build_inverse_map(
            expert_indices, pub_counts, inverse_map, my_rank,
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
        # W8A8: INT8 activation + per-token scale + INT8 weights + per-channel
        # scales (decode_fwd.py:1056-1078, design §4.2 step 3).
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

            # Row tiling (mirrors decode RECV_TILE / N_RECV_TILES,
            # decode_fwd.py:1083): the 1024 routed rows are processed in
            # 32-row tiles so the per-tile h_tile [32, inter] FP32 = 160 KiB
            # stays under the 188 KiB Vec UB (the monolithic [1024, inter]
            # FP32 = 5 MiB overflows).  gate_up and down are both inside
            # the per-tile loop; h_bf16 is a per-tile bridge tensor.
            for tile_idx in pl.range(N_RECV_TILES):
                tile_row0_i32 = pl.cast(tile_idx * RECV_TILE, pl.INT32)
                tile_rem = n_rows - tile_row0_i32
                if tile_rem > 0:
                    tile_offset = offset + pl.cast(tile_row0_i32, pl.INDEX)
                    tile_valid = pl.cast(
                        pl.min(pl.cast(RECV_TILE, pl.INT32), tile_rem),
                        pl.INDEX,
                    )

                    # Bridge-tensor h_bf16 at loop level (PROGRESS Session 5
                    # step 4, mirrors decode_fwd.py:1094): shared between
                    # the gate_up / down SPMD fans.
                    h_bf16 = pl.create_tensor(
                        [RECV_TILE, inter], dtype=pl.BF16,
                    )
                    # Per-token act scale for this tile (decode_fwd.py:1164-
                    # 1169): contiguous [1,RECV_TILE] row-slice of the
                    # UNPADDED local_routed_x_scale + reshape [RECV_TILE,1].
                    x_scale_col = pl.reshape(
                        pl.slice(
                            local_routed_x_scale,
                            [1, RECV_TILE],
                            [0, tile_offset],
                        ),
                        [RECV_TILE, 1],
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
                            valid_shape=[tile_valid, ROUTED_GATE_K_CHUNK],
                        )
                        wg0 = pl.slice(
                            w_gate,
                            [1, ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                            [e, 0, n0],
                        )
                        wu0 = pl.slice(
                            w_up,
                            [1, ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                            [e, 0, n0],
                        )
                        wg0_2d = pl.reshape(
                            wg0,
                            [ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                        )
                        wu0_2d = pl.reshape(
                            wu0,
                            [ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                        )
                        # INT8 x INT8 -> INT32 matmul (decode_fwd.py:1124).
                        gate_acc = pl.matmul(x0, wg0_2d, out_dtype=pl.INT32)
                        up_acc = pl.matmul(x0, wu0_2d, out_dtype=pl.INT32)
                        for kb in pl.range(1, HIDDEN // ROUTED_GATE_K_CHUNK):
                            k0 = kb * ROUTED_GATE_K_CHUNK
                            xk = pl.slice(
                                local_routed_x,
                                [RECV_TILE, ROUTED_GATE_K_CHUNK],
                                [tile_offset, k0],
                                valid_shape=[
                                    tile_valid, ROUTED_GATE_K_CHUNK,
                                ],
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
                        # Dequant: cast INT32->FP32, row-expand per-token act
                        # scale, col-expand per-channel weight scale
                        # (decode_fwd.py:1176-1193, design §4.2 step 3).
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

                    # Per-token INT8 requant of the SwiGLU intermediate for
                    # the INT8 down-proj (decode_fwd.py:1218-1287, design
                    # §4.2 step 3).  h_bf16 padding rows are already zero from
                    # the gated fillpad, so the bare amax slice is safe.
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
                            valid_shape=[
                                tile_valid, ROUTED_DOWN_K_CHUNK,
                            ],
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
                        # INT8 h_i8 x INT8 w_down -> INT32 (decode_fwd.py:1311).
                        y_acc = pl.matmul(h0, wd0, out_dtype=pl.INT32)
                        for kb2 in pl.range(1, inter // ROUTED_DOWN_K_CHUNK):
                            k0 = kb2 * ROUTED_DOWN_K_CHUNK
                            hk = pl.slice(
                                h_i8,
                                [RECV_TILE, ROUTED_DOWN_K_CHUNK],
                                [0, k0],
                                valid_shape=[
                                    tile_valid, ROUTED_DOWN_K_CHUNK,
                                ],
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

                        # Dequant down-proj: cast INT32->FP32, row-expand
                        # h_scale_dq (NO route_weight — prefill combine applies
                        # expert_weights locally via inverse_map, numerically
                        # equivalent to decode folding it at the expert
                        # down-proj), col-expand wd_scale_row
                        # (decode_fwd.py:1348-1356, design §4.2 step 3).
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
            local_routed_x, local_routed_x_scale,
            local_expert_offset, local_expert_count,
            w_gate_r, w_gate_r_scale, w_up_r, w_up_r_scale,
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
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_gate_up"):
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
            if False:
                silu_c = pl.minimum(silu, 0.0)
                up_c = pl.maximum(
                    pl.minimum(up_acc, 0.0),
                    -0.0,
                )
                gated = pl.mul(silu_c, up_c)
            else:
                gated = pl.mul(silu, up_acc)

            h_tile[:, 0:SHARED_GATE_N_CHUNK] = pl.cast(
                gated, target_type=pl.BF16,
            )

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_down"):
            for db in pl.range(HIDDEN // SHARED_DOWN_N_CHUNK):
                d0 = db * SHARED_DOWN_N_CHUNK
                h0 = pl.slice(
                    h_tile, [BATCH, SHARED_DOWN_K_CHUNK], [0, 0],
                )
                wd0 = pl.slice(
                    w_down,
                    [SHARED_DOWN_K_CHUNK, SHARED_DOWN_N_CHUNK],
                    [0, d0],
                )
                y_acc = pl.matmul(h0, wd0, out_dtype=pl.FP32)
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
            [BATCH, sh_tp_chunk], pl.BF16
        ],
        sh_signal_window: pld.DistributedTensor[
            [n_ranks, 1], pl.INT32
        ],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
        sh_y = self._expert_shared_local(
            x, w_gate_s, w_up_s, w_down_s, sh_y,
        )
        self._tp_all_reduce_moe(
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
                    # Direct write to the GM DistributedTensor (mirrors
                    # decode combine_step): the tmp create_tensor + write +
                    # load + store round-trip produced a TileType that
                    # tile.store rejected, and the tmp scratch is unneeded.
                    # See PROGRESS.md Session 4 (combine pl.write fix).
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
            [n_ranks * n_ranks, n_local_experts], pl.INT32
        ],
        routed_y_buf: pld.DistributedTensor[
            [n_routes_per_rank, HIDDEN], pl.BF16
        ],
        combine_done: pld.DistributedTensor[
            [n_ranks, 1], pl.INT32
        ],
        src_route_table: pld.DistributedTensor[
            [n_ranks, n_local_experts, n_routes_per_rank], pl.INT32
        ],
        my_rank: pl.Scalar[pl.INT32],
    ):
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
                    r_route = pl.read(
                        src_route_table,
                        [src, e, pl.cast(row, pl.INDEX)],
                    )
                    local_row = (
                        pl.cast(e_cursor, pl.INDEX)
                        + pl.cast(src_off, pl.INDEX) + row
                    )
                    # Per-row put (mirrors decode combine_step:1778-1785,
                    # PROGRESS.md Session 5 step 6): put only one
                    # [1, HIDDEN] = 8 KiB row from local_routed_y into the
                    # peer's routed_y_buf at [r_route, 0].  The previous
                    # X.4 minimal rewrite put the entire
                    # [n_routes_per_rank, HIDDEN] BF16 = 1 MiB buf, which
                    # overflowed the 188 KiB Vec UB.  A single put serves
                    # both local (peer==my_rank) and remote peers.
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

        for peer in pl.range(n_ranks):
            if peer != my_rank:
                pld.system.notify(
                    target=combine_done,
                    peer=peer,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.Set,
                )
        for src in pl.range(n_ranks):
            if src != my_rank:
                pld.system.wait(
                    signal=combine_done,
                    offsets=[src, 0],
                    expected=1,
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

    @pl.function(type=pl.FunctionType.Inline)
    def combine_step(  # noqa: PLR0913
        self,
        local_routed_y: pl.Tensor[
            [local_recv_max, HIDDEN], pl.BF16
        ],
        expert_indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
        expert_weights: pl.Tensor[[BATCH, TOPK], pl.FP32],
        sh_y: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        moe_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
        pub_counts: pld.DistributedTensor[
            [n_ranks * n_ranks, n_local_experts], pl.INT32
        ],
        src_route_table: pld.DistributedTensor[
            [n_ranks, n_local_experts, n_routes_per_rank], pl.INT32
        ],
        route_pub_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        routed_y_buf: pld.DistributedTensor[
            [n_routes_per_rank, HIDDEN], pl.BF16
        ],
        combine_done_sig: pld.DistributedTensor[
            [n_ranks, 1], pl.INT32
        ],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
        self._publish_src_route_table(
            expert_indices, src_route_table, my_rank,
        )
        # Wrap cross-core notify/wait in CORE_GROUP scopes (PROGRESS.md
        # Session 20: bare pld.system.notify/wait inlined into chip_orch
        # (Orchestration) become Misplaced builtin op).  Each scope holds
        # one control-plane phase so the OutlineIncoreScopes pass can
        # outline them correctly.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="combine_route_notify"):
            for peer in pl.range(n_ranks):
                if peer != my_rank:
                    pld.system.notify(
                        target=route_pub_sig,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.Set,
                    )
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="combine_route_wait"):
            for src in pl.range(n_ranks):
                if src != my_rank:
                    pld.system.wait(
                        signal=route_pub_sig,
                        offsets=[src, 0],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )

        self._push_routed_y_to_sources(
            local_routed_y,
            pub_counts,
            routed_y_buf,
            combine_done_sig,
            src_route_table,
            my_rank,
        )

        moe_out = self._weighted_gather_and_add(
            routed_y_buf, expert_weights, sh_y, moe_out,
        )
        return moe_out

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

            # Row tiling (mirrors decode RECV_TILE / N_RECV_TILES,
            # decode_fwd.py:1083): the 1024 routed rows are processed in
            # 32-row tiles so the per-tile h_tile [32, inter] FP32 = 160 KiB
            # stays under the 188 KiB Vec UB (the monolithic [1024, inter]
            # FP32 = 5 MiB overflows).  gate_up and down are both inside
            # the per-tile loop; h_bf16 is a per-tile bridge tensor.
            for tile_idx in pl.range(N_RECV_TILES):
                tile_row0_i32 = pl.cast(tile_idx * RECV_TILE, pl.INT32)
                tile_rem = n_rows - tile_row0_i32
                if tile_rem > 0:
                    tile_offset = offset + pl.cast(tile_row0_i32, pl.INDEX)
                    tile_valid = pl.cast(
                        pl.min(pl.cast(RECV_TILE, pl.INT32), tile_rem),
                        pl.INDEX,
                    )

                    # Bridge-tensor h_bf16 at loop level (PROGRESS Session 5
                    # step 4, mirrors decode_fwd.py:1094): shared between
                    # the gate_up / down SPMD fans.
                    h_bf16 = pl.create_tensor(
                        [RECV_TILE, inter], dtype=pl.BF16,
                    )
                    # Per-token act scale for this tile (decode_fwd.py:1164-
                    # 1169): contiguous [1,RECV_TILE] row-slice of the
                    # UNPADDED local_routed_x_scale + reshape [RECV_TILE,1].
                    x_scale_col = pl.reshape(
                        pl.slice(
                            local_routed_x_scale,
                            [1, RECV_TILE],
                            [0, tile_offset],
                        ),
                        [RECV_TILE, 1],
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
                            valid_shape=[tile_valid, ROUTED_GATE_K_CHUNK],
                        )
                        wg0 = pl.slice(
                            w_gate,
                            [1, ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                            [e, 0, n0],
                        )
                        wu0 = pl.slice(
                            w_up,
                            [1, ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                            [e, 0, n0],
                        )
                        wg0_2d = pl.reshape(
                            wg0,
                            [ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                        )
                        wu0_2d = pl.reshape(
                            wu0,
                            [ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                        )
                        # INT8 x INT8 -> INT32 matmul (decode_fwd.py:1124).
                        gate_acc = pl.matmul(x0, wg0_2d, out_dtype=pl.INT32)
                        up_acc = pl.matmul(x0, wu0_2d, out_dtype=pl.INT32)
                        for kb in pl.range(1, HIDDEN // ROUTED_GATE_K_CHUNK):
                            k0 = kb * ROUTED_GATE_K_CHUNK
                            xk = pl.slice(
                                local_routed_x,
                                [RECV_TILE, ROUTED_GATE_K_CHUNK],
                                [tile_offset, k0],
                                valid_shape=[
                                    tile_valid, ROUTED_GATE_K_CHUNK,
                                ],
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
                        # Dequant: cast INT32->FP32, row-expand per-token act
                        # scale, col-expand per-channel weight scale
                        # (decode_fwd.py:1176-1193, design §4.2 step 3).
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
                        # SwiGLU7 routed activation (L43/L44, routed_lim=7.0).
                        silu_c = pl.minimum(silu, _ROUTED_SWIGLU7_LIMIT)
                        up_c = pl.maximum(
                            pl.minimum(up_2d, _ROUTED_SWIGLU7_LIMIT),
                            -_ROUTED_SWIGLU7_LIMIT,
                        )
                        gated = pl.mul(silu_c, up_c)
                        gated_v = pl.set_validshape(
                            gated, tile_valid, ROUTED_GATE_N_CHUNK,
                        )
                        gated_m = pl.fillpad(
                            gated_v, pad_value=pl.PadValue.zero,
                        )
                        h_bf16[
                            :, n0 : n0 + ROUTED_GATE_N_CHUNK
                        ] = pl.cast(gated_m, target_type=pl.BF16)

                    # Per-token INT8 requant of the SwiGLU intermediate for
                    # the INT8 down-proj (decode_fwd.py:1218-1287, design
                    # §4.2 step 3).  h_bf16 padding rows are already zero from
                    # the gated fillpad, so the bare amax slice is safe.
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
                            valid_shape=[
                                tile_valid, ROUTED_DOWN_K_CHUNK,
                            ],
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
                        # INT8 h_i8 x INT8 w_down -> INT32 (decode_fwd.py:1311).
                        y_acc = pl.matmul(h0, wd0, out_dtype=pl.INT32)
                        for kb2 in pl.range(1, inter // ROUTED_DOWN_K_CHUNK):
                            k0 = kb2 * ROUTED_DOWN_K_CHUNK
                            hk = pl.slice(
                                h_i8,
                                [RECV_TILE, ROUTED_DOWN_K_CHUNK],
                                [0, k0],
                                valid_shape=[
                                    tile_valid, ROUTED_DOWN_K_CHUNK,
                                ],
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

                        # Dequant down-proj: cast INT32->FP32, row-expand
                        # h_scale_dq (NO route_weight — prefill combine applies
                        # expert_weights locally via inverse_map, numerically
                        # equivalent to decode folding it at the expert
                        # down-proj), col-expand wd_scale_row
                        # (decode_fwd.py:1348-1356, design §4.2 step 3).
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
            local_routed_x, local_routed_x_scale,
            local_expert_offset, local_expert_count,
            w_gate_r, w_gate_r_scale, w_up_r, w_up_r_scale,
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
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_gate_up"):
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
            if True:
                silu_c = pl.minimum(silu, 16.0)
                up_c = pl.maximum(
                    pl.minimum(up_acc, 16.0),
                    -16.0,
                )
                gated = pl.mul(silu_c, up_c)
            else:
                gated = pl.mul(silu, up_acc)

            h_tile[:, 0:SHARED_GATE_N_CHUNK] = pl.cast(
                gated, target_type=pl.BF16,
            )

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_down"):
            for db in pl.range(HIDDEN // SHARED_DOWN_N_CHUNK):
                d0 = db * SHARED_DOWN_N_CHUNK
                h0 = pl.slice(
                    h_tile, [BATCH, SHARED_DOWN_K_CHUNK], [0, 0],
                )
                wd0 = pl.slice(
                    w_down,
                    [SHARED_DOWN_K_CHUNK, SHARED_DOWN_N_CHUNK],
                    [0, d0],
                )
                y_acc = pl.matmul(h0, wd0, out_dtype=pl.FP32)
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
            [BATCH, sh_tp_chunk], pl.BF16
        ],
        sh_signal_window: pld.DistributedTensor[
            [n_ranks, 1], pl.INT32
        ],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
        sh_y = self._expert_shared_local_swiglu16(
            x, w_gate_s, w_up_s, w_down_s, sh_y,
        )
        self._tp_all_reduce_moe(
            sh_y, sh_tmp_window, sh_signal_window, my_rank,
        )
        return sh_y

    @pl.function(
        type=pl.FunctionType.Orchestration,
    )
    def full_chip_orch(  # noqa: PLR0913, PLR0915
        self,
        current_hidden: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
        input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
        wq: pl.Tensor[[HIDDEN, hidden_q_full], pl.BF16],
        wk: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        wv: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
        k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
        block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
        slot_mapping: pl.Tensor[[PREFILL_T], pl.INT32],
        rope_cos: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],
        rope_sin: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],
        k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
        v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
        wo: pl.Tensor[[hidden_q_full, HIDDEN], pl.BF16],
        w_g: pl.Tensor[[HIDDEN, nh_full_pad], pl.BF16],
        gate_r: pl.Tensor[[nh_full_pad, hidden_q_full], pl.BF16],
        positions: pl.Tensor[[PREFILL_T], pl.INT32],
        post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
        w_gate: pl.Tensor[[HIDDEN, INTER_LOCAL], pl.BF16],
        w_up: pl.Tensor[[HIDDEN, INTER_LOCAL], pl.BF16],
        w_down: pl.Tensor[[INTER_LOCAL, HIDDEN], pl.BF16],
        h0_out: pl.Out[pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]],
        attn_tmp_window: pld.DistributedTensor[[PREFILL_T, tp_chunk], pl.BF16],
        attn_signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
        mlp_tmp_window: pld.DistributedTensor[[PREFILL_T, tp_chunk], pl.BF16],
        mlp_signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
        norm_layer_idx: pl.Scalar[pl.INT32],
        mlp_layer_idx: pl.Scalar[pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]:
        resid1 = pl.create_tensor([PREFILL_T, HIDDEN], dtype=pl.BF16)
        resid1 = attention_full_inline(
            current_hidden, input_rms_weight, wq, wk, wv,
            q_norm_weight, k_norm_weight,
            block_table, slot_mapping,
            rope_cos, rope_sin, k_cache, v_cache,
            wo, w_g, gate_r, positions, resid1,
            norm_layer_idx, norm_layer_idx,
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
    )
    def swa_chip_orch(  # noqa: PLR0913, PLR0915
        self,
        current_hidden: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
        input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
        wq: pl.Tensor[[HIDDEN, hidden_q_swa], pl.BF16],
        wk: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        wv: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
        k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
        block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
        slot_mapping: pl.Tensor[[PREFILL_T], pl.INT32],
        rope_cos: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],
        rope_sin: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],
        k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
        v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
        wo: pl.Tensor[[hidden_q_swa, HIDDEN], pl.BF16],
        w_g: pl.Tensor[[HIDDEN, nh_swa_pad], pl.BF16],
        gate_r: pl.Tensor[[nh_swa_pad, hidden_q_swa], pl.BF16],
        positions: pl.Tensor[[PREFILL_T], pl.INT32],
        post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
        w_gate: pl.Tensor[[HIDDEN, INTER_LOCAL], pl.BF16],
        w_up: pl.Tensor[[HIDDEN, INTER_LOCAL], pl.BF16],
        w_down: pl.Tensor[[INTER_LOCAL, HIDDEN], pl.BF16],
        hidden_out: pl.Out[pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]],
        attn_tmp_window: pld.DistributedTensor[[PREFILL_T, tp_chunk], pl.BF16],
        attn_signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
        mlp_tmp_window: pld.DistributedTensor[[PREFILL_T, tp_chunk], pl.BF16],
        mlp_signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
        norm_layer_idx: pl.Scalar[pl.INT32],
        mlp_layer_idx: pl.Scalar[pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]:
        resid1 = pl.create_tensor([PREFILL_T, HIDDEN], dtype=pl.BF16)
        resid1 = attention_swa_inline(
            current_hidden, input_rms_weight, wq, wk, wv,
            q_norm_weight, k_norm_weight,
            block_table, slot_mapping,
            rope_cos, rope_sin, k_cache, v_cache,
            wo, w_g, gate_r, positions, resid1,
            norm_layer_idx, norm_layer_idx,
            attn_tmp_window, attn_signal_window, my_rank,
        )
        hidden_out = dense_mlp_inline(
            resid1, post_rms_weight, w_gate, w_up, w_down,
            hidden_out, norm_layer_idx, mlp_layer_idx,
            mlp_tmp_window, mlp_signal_window, my_rank,
        )
        return hidden_out

    @pl.function(
        type=pl.FunctionType.Orchestration,
    )
    def full_moe_chip_orch(  # noqa: PLR0913, PLR0915
        self,
        current_hidden: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
        input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
        wq: pl.Tensor[[HIDDEN, hidden_q_full], pl.BF16],
        wk: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        wv: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
        k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
        block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
        slot_mapping: pl.Tensor[[PREFILL_T], pl.INT32],
        rope_cos: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],
        rope_sin: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],
        k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
        v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
        wo: pl.Tensor[[hidden_q_full, HIDDEN], pl.BF16],
        w_g: pl.Tensor[[HIDDEN, nh_full_pad], pl.BF16],
        gate_r: pl.Tensor[[nh_full_pad, hidden_q_full], pl.BF16],
        positions: pl.Tensor[[PREFILL_T], pl.INT32],
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
        next_hidden_out: pl.Out[pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]],
        resid_hold: pl.Out[pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]],
        attn_tmp_window: pld.DistributedTensor[[PREFILL_T, tp_chunk], pl.BF16],
        attn_signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
        pub_counts: pld.DistributedTensor[[n_ranks * n_ranks, n_local_experts], pl.INT32],
        count_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        recv_x: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
        data_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        send_buf: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
        # Per-routed-row aux slab (W8A8, decode recv_aux ABI,
        # decode_fwd.py:830-832): col 0 = per-token dequant scale, cols 1..7
        # physical FP32 padding (dispatch_aux_pad) for PTOAS tile alignment.
        send_aux: pld.DistributedTensor[
            [local_recv_max, dispatch_aux_pad], pl.FP32
        ],
        recv_aux: pld.DistributedTensor[
            [local_recv_max, dispatch_aux_pad], pl.FP32
        ],
        sh_tmp_window: pld.DistributedTensor[[BATCH, sh_tp_chunk], pl.BF16],
        sh_signal_window: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        src_route_table: pld.DistributedTensor[[n_ranks, n_local_experts, n_routes_per_rank], pl.INT32],
        route_pub_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        routed_y_buf: pld.DistributedTensor[[n_routes_per_rank, HIDDEN], pl.BF16],
        combine_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        norm_layer_idx: pl.Scalar[pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]:
        # ── A: prefill attention + tp_all_reduce -> resid_hold. ────────
        resid_hold = attention_full_inline(
            current_hidden,
            input_rms_weight,
            wq, wk, wv,
            q_norm_weight, k_norm_weight,
            block_table, slot_mapping,
            rope_cos, rope_sin,
            k_cache, v_cache,
            wo, w_g,
            gate_r,
            positions,
            resid_hold,
            norm_layer_idx,
            norm_layer_idx,
            attn_tmp_window,
            attn_signal_window,
            my_rank,
        )

        # ── B: post-attention RMSNorm + residual add fused per-tile
        # (PROGRESS Session 6 step 8+9, mirrors decode BATCH=16).  The full
        # [PREFILL_T, HIDDEN] post_norm / resid_hold_fp32 / moe_out buffers and
        # their CORE_GROUP scopes are removed; each MoE tile runs a
        # two-pass chunked RMSNorm over resid_hold[t_lo:t_lo+BATCH] producing
        # tile_x (stage C) and a per-tile residual add producing
        # next_hidden_out[t_lo] (stage D).  inv_rms is per-token (over
        # HIDDEN features) so per-tile is bit-equivalent to the full pass.
        hidden_blocks = HIDDEN // K_CHUNK

        # ── C: prefill MoE adapter — Phase X.10 inlined per-tile loop.
        # Each tile runs a two-pass chunked RMSNorm on resid_hold[t_lo] ->
        # tile_x, then the inlined gate / dispatch / expert_routed /
        # expert_shared / combine pipeline -> tile_y, then a per-tile
        # residual add -> next_hidden_out[t_lo].  Routing is per-token, so
        # this is bit-equivalent to a single T=PREFILL_T pass.  The window
        # pool shared across tiles is safe because each tile flushes
        # (signal-wait pairs) before the next reads.
        for tile_idx in pl.unroll(PREFILL_TILE_COUNT):
            t_lo = tile_idx * BATCH
            tile_x = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            x_i8_tile = pl.create_tensor([BATCH, HIDDEN], dtype=pl.INT8)
            x_scale_tile = pl.create_tensor(
                [BATCH, DISPATCH_SCALE_COLS], dtype=pl.FP32,
            )
            # V4-style deferred RMSNorm + INT8 producer (decode_fwd.py:694-805
            # _norm_quant_moe_input, design §4.2 step 1).  Two-pass chunked over
            # resid_hold[t_lo]: pass 1 forms xg = resid*(gamma+1) while reducing
            # sq_sum (RMSNorm) and xg_amax (INT8 quant scale); pass 2 emits the
            # BF16 post-norm hidden (tile_x = shared-expert + gate input) and
            # the INT8 dispatch payload x_i8_tile whose dequant scale
            # x_scale_tile = inv_rms*amax(xg)/127 carries the deferred positive
            # RMS factor.  Prefill inlines the producer (per-tile slicing of
            # resid_hold[t_lo] is already inline here); prefill tiles are full
            # BATCH (no active_tokens clamping) and the gate consumes tile_x
            # directly, so inv_rms is folded into x_scale and not emitted.
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="prefill_moe_post_rmsnorm_tile",
            ):
                sq_sum = pl.full(
                    [1, BATCH], dtype=pl.FP32, value=0.0,
                )
                xg_amax = pl.full(
                    [1, BATCH], dtype=pl.FP32, value=1e-4,
                )
                for kb in pl.range(hidden_blocks):
                    k0 = kb * K_CHUNK
                    raw = pl.cast(
                        pl.slice(
                            resid_hold, [BATCH, K_CHUNK], [t_lo, k0],
                        ),
                        target_type=pl.FP32,
                    )
                    gamma = pl.slice(
                        post_rms_weight, [1, K_CHUNK], [norm_layer_idx, k0],
                    )
                    xg = pl.col_expand_mul(raw, pl.add(gamma, 1.0))
                    sq_sum = pl.add(
                        sq_sum,
                        pl.reshape(
                            pl.row_sum(pl.mul(raw, raw)),
                            [1, BATCH],
                        ),
                    )
                    xg_amax = pl.maximum(
                        xg_amax,
                        pl.reshape(
                            pl.row_max(pl.maximum(xg, pl.neg(xg))),
                            [1, BATCH],
                        ),
                    )
                inv_rms_row = pl.recip(
                    pl.sqrt(
                        pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS),
                    ),
                )
                inv_rms_col = pl.reshape(inv_rms_row, [BATCH, 1])
                quant_mul = pl.reshape(
                    pl.div(
                        pl.full([1, BATCH], dtype=pl.FP32, value=127.0),
                        xg_amax,
                    ),
                    [BATCH, 1],
                )
                x_scale_tile = pl.assemble(
                    x_scale_tile,
                    pl.reshape(
                        pl.mul(
                            inv_rms_row,
                            pl.mul(xg_amax, 1.0 / 127.0),
                        ),
                        [BATCH, 1],
                    ),
                    [0, 0],
                )
                for kb2 in pl.range(hidden_blocks):
                    k0 = kb2 * K_CHUNK
                    raw = pl.cast(
                        pl.slice(
                            resid_hold, [BATCH, K_CHUNK], [t_lo, k0],
                        ),
                        target_type=pl.FP32,
                    )
                    gamma = pl.slice(
                        post_rms_weight, [1, K_CHUNK], [norm_layer_idx, k0],
                    )
                    xg = pl.col_expand_mul(raw, pl.add(gamma, 1.0))
                    normed = pl.row_expand_mul(xg, inv_rms_col)
                    tile_x = pl.assemble(
                        tile_x,
                        pl.cast(normed, target_type=pl.BF16),
                        [0, k0],
                    )
                    qi32 = pl.cast(
                        pl.row_expand_mul(xg, quant_mul),
                        target_type=pl.INT32, mode="rint",
                    )
                    qf16 = pl.cast(qi32, target_type=pl.FP16, mode="round")
                    x_i8_tile = pl.assemble(
                        x_i8_tile,
                        pl.cast(qf16, target_type=pl.INT8, mode="trunc"),
                        [0, k0],
                    )

            # 1) Gate (local, replicated).
            expert_indices = pl.create_tensor(
                [BATCH, TOPK], dtype=pl.INT32,
            )
            expert_weights = pl.create_tensor([BATCH, TOPK], dtype=pl.FP32)
            expert_indices, expert_weights = self.gate_step(
                tile_x, gate_w, router_bias,
                expert_indices, expert_weights,
            )

            # 2) Shared-expert lane (TP-sliced + tp_all_reduce).
            sh_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            sh_y = self.expert_shared_step(
                tile_x, w_gate_s, w_up_s, w_down_s, sh_y,
                sh_tmp_window, sh_signal_window, my_rank,
            )

            # 3) Dispatch (EP all-to-all) — W8A8 INT8 payload + per-token
            # scale (decode_fwd.py:809-1047, design §4.2 step 2).
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
            inverse_map = pl.create_tensor(
                [BATCH, TOPK], dtype=pl.INT32,
            )
            (
                local_routed_x,
                local_routed_x_scale,
                local_expert_offset,
                local_expert_count,
                inverse_map,
            ) = self.dispatch_step(
                x_i8_tile, x_scale_tile, expert_indices,
                local_routed_x, local_routed_x_scale,
                local_expert_offset, local_expert_count, inverse_map,
                pub_counts, count_done_sig, recv_x, data_done_sig,
                send_buf, send_aux, recv_aux,
                my_rank,
            )

            # 4) Routed experts (local 36).
            local_routed_y = pl.create_tensor(
                [local_recv_max, HIDDEN], dtype=pl.BF16,
            )
            local_routed_y = self.expert_routed_step(
                local_routed_x, local_routed_x_scale,
                local_expert_offset, local_expert_count,
                w_gate_r, w_gate_r_scale, w_up_r, w_up_r_scale,
                w_down_r, w_down_r_scale,
                local_routed_y,
            )

            # 5) Combine (EP a2a back + weighted gather + sh_y add).
            tile_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            tile_y = self.combine_step(
                local_routed_y,
                expert_indices, expert_weights, sh_y,
                tile_y,
                pub_counts, src_route_table, route_pub_sig,
                routed_y_buf, combine_done_sig,
                my_rank,
            )

            # Per-tile residual add (PROGRESS Session 6 step 9, mirrors
            # decode): next_hidden_out[t_lo] = resid_hold[t_lo] + tile_y,
            # removing the full [PREFILL_T, HIDDEN] moe_out / resid_hold_fp32
            # buffers.  Chunked over K_CHUNK to keep Vec tiles at
            # [BATCH, K_CHUNK] FP32 = 16 KiB.
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="prefill_moe_residual_add_tile",
            ):
                for kb4 in pl.range(hidden_blocks):
                    k0 = kb4 * K_CHUNK
                    res_chunk = pl.cast(
                        pl.slice(
                            resid_hold, [BATCH, K_CHUNK], [t_lo, k0],
                        ),
                        target_type=pl.FP32,
                    )
                    y_chunk = pl.cast(
                        pl.slice(
                            tile_y, [BATCH, K_CHUNK], [0, k0],
                        ),
                        target_type=pl.FP32,
                    )
                    next_hidden_out = pl.assemble(
                        next_hidden_out,
                        pl.cast(
                            pl.add(res_chunk, y_chunk),
                            target_type=pl.BF16,
                        ),
                        [t_lo, k0],
                    )
        return next_hidden_out

    @pl.function(
        type=pl.FunctionType.Orchestration,
    )
    def swa_moe_chip_orch(  # noqa: PLR0913, PLR0915
        self,
        current_hidden: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
        input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
        wq: pl.Tensor[[HIDDEN, hidden_q_swa], pl.BF16],
        wk: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        wv: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
        k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
        block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
        slot_mapping: pl.Tensor[[PREFILL_T], pl.INT32],
        rope_cos: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],
        rope_sin: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],
        k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
        v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
        wo: pl.Tensor[[hidden_q_swa, HIDDEN], pl.BF16],
        w_g: pl.Tensor[[HIDDEN, nh_swa_pad], pl.BF16],
        gate_r: pl.Tensor[[nh_swa_pad, hidden_q_swa], pl.BF16],
        positions: pl.Tensor[[PREFILL_T], pl.INT32],
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
        next_hidden_out: pl.Out[pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]],
        resid_hold: pl.Out[pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]],
        attn_tmp_window: pld.DistributedTensor[[PREFILL_T, tp_chunk], pl.BF16],
        attn_signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
        pub_counts: pld.DistributedTensor[[n_ranks * n_ranks, n_local_experts], pl.INT32],
        count_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        recv_x: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
        data_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        send_buf: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
        # Per-routed-row aux slab (W8A8, decode recv_aux ABI,
        # decode_fwd.py:830-832): col 0 = per-token dequant scale, cols 1..7
        # physical FP32 padding (dispatch_aux_pad) for PTOAS tile alignment.
        send_aux: pld.DistributedTensor[
            [local_recv_max, dispatch_aux_pad], pl.FP32
        ],
        recv_aux: pld.DistributedTensor[
            [local_recv_max, dispatch_aux_pad], pl.FP32
        ],
        sh_tmp_window: pld.DistributedTensor[[BATCH, sh_tp_chunk], pl.BF16],
        sh_signal_window: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        src_route_table: pld.DistributedTensor[[n_ranks, n_local_experts, n_routes_per_rank], pl.INT32],
        route_pub_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        routed_y_buf: pld.DistributedTensor[[n_routes_per_rank, HIDDEN], pl.BF16],
        combine_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        norm_layer_idx: pl.Scalar[pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]:
        # ── A: prefill attention + tp_all_reduce -> resid_hold. ────────
        resid_hold = attention_swa_inline(
            current_hidden,
            input_rms_weight,
            wq, wk, wv,
            q_norm_weight, k_norm_weight,
            block_table, slot_mapping,
            rope_cos, rope_sin,
            k_cache, v_cache,
            wo, w_g,
            gate_r,
            positions,
            resid_hold,
            norm_layer_idx,
            norm_layer_idx,
            attn_tmp_window,
            attn_signal_window,
            my_rank,
        )

        # ── B: post-attention RMSNorm + residual add fused per-tile
        # (PROGRESS Session 6 step 8+9, mirrors decode BATCH=16).  The full
        # [PREFILL_T, HIDDEN] post_norm / resid_hold_fp32 / moe_out buffers and
        # their CORE_GROUP scopes are removed; each MoE tile runs a
        # two-pass chunked RMSNorm over resid_hold[t_lo:t_lo+BATCH] producing
        # tile_x (stage C) and a per-tile residual add producing
        # next_hidden_out[t_lo] (stage D).  inv_rms is per-token (over
        # HIDDEN features) so per-tile is bit-equivalent to the full pass.
        hidden_blocks = HIDDEN // K_CHUNK

        # ── C: prefill MoE adapter — Phase X.10 inlined per-tile loop.
        # Each tile runs a two-pass chunked RMSNorm on resid_hold[t_lo] ->
        # tile_x, then the inlined gate / dispatch / expert_routed /
        # expert_shared / combine pipeline -> tile_y, then a per-tile
        # residual add -> next_hidden_out[t_lo].  Routing is per-token, so
        # this is bit-equivalent to a single T=PREFILL_T pass.  The window
        # pool shared across tiles is safe because each tile flushes
        # (signal-wait pairs) before the next reads.
        for tile_idx in pl.unroll(PREFILL_TILE_COUNT):
            t_lo = tile_idx * BATCH
            tile_x = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            x_i8_tile = pl.create_tensor([BATCH, HIDDEN], dtype=pl.INT8)
            x_scale_tile = pl.create_tensor(
                [BATCH, DISPATCH_SCALE_COLS], dtype=pl.FP32,
            )
            # V4-style deferred RMSNorm + INT8 producer (decode_fwd.py:694-805
            # _norm_quant_moe_input, design §4.2 step 1).  Two-pass chunked over
            # resid_hold[t_lo]: pass 1 forms xg = resid*(gamma+1) while reducing
            # sq_sum (RMSNorm) and xg_amax (INT8 quant scale); pass 2 emits the
            # BF16 post-norm hidden (tile_x = shared-expert + gate input) and
            # the INT8 dispatch payload x_i8_tile whose dequant scale
            # x_scale_tile = inv_rms*amax(xg)/127 carries the deferred positive
            # RMS factor.  Prefill inlines the producer (per-tile slicing of
            # resid_hold[t_lo] is already inline here); prefill tiles are full
            # BATCH (no active_tokens clamping) and the gate consumes tile_x
            # directly, so inv_rms is folded into x_scale and not emitted.
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="prefill_moe_post_rmsnorm_tile",
            ):
                sq_sum = pl.full(
                    [1, BATCH], dtype=pl.FP32, value=0.0,
                )
                xg_amax = pl.full(
                    [1, BATCH], dtype=pl.FP32, value=1e-4,
                )
                for kb in pl.range(hidden_blocks):
                    k0 = kb * K_CHUNK
                    raw = pl.cast(
                        pl.slice(
                            resid_hold, [BATCH, K_CHUNK], [t_lo, k0],
                        ),
                        target_type=pl.FP32,
                    )
                    gamma = pl.slice(
                        post_rms_weight, [1, K_CHUNK], [norm_layer_idx, k0],
                    )
                    xg = pl.col_expand_mul(raw, pl.add(gamma, 1.0))
                    sq_sum = pl.add(
                        sq_sum,
                        pl.reshape(
                            pl.row_sum(pl.mul(raw, raw)),
                            [1, BATCH],
                        ),
                    )
                    xg_amax = pl.maximum(
                        xg_amax,
                        pl.reshape(
                            pl.row_max(pl.maximum(xg, pl.neg(xg))),
                            [1, BATCH],
                        ),
                    )
                inv_rms_row = pl.recip(
                    pl.sqrt(
                        pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS),
                    ),
                )
                inv_rms_col = pl.reshape(inv_rms_row, [BATCH, 1])
                quant_mul = pl.reshape(
                    pl.div(
                        pl.full([1, BATCH], dtype=pl.FP32, value=127.0),
                        xg_amax,
                    ),
                    [BATCH, 1],
                )
                x_scale_tile = pl.assemble(
                    x_scale_tile,
                    pl.reshape(
                        pl.mul(
                            inv_rms_row,
                            pl.mul(xg_amax, 1.0 / 127.0),
                        ),
                        [BATCH, 1],
                    ),
                    [0, 0],
                )
                for kb2 in pl.range(hidden_blocks):
                    k0 = kb2 * K_CHUNK
                    raw = pl.cast(
                        pl.slice(
                            resid_hold, [BATCH, K_CHUNK], [t_lo, k0],
                        ),
                        target_type=pl.FP32,
                    )
                    gamma = pl.slice(
                        post_rms_weight, [1, K_CHUNK], [norm_layer_idx, k0],
                    )
                    xg = pl.col_expand_mul(raw, pl.add(gamma, 1.0))
                    normed = pl.row_expand_mul(xg, inv_rms_col)
                    tile_x = pl.assemble(
                        tile_x,
                        pl.cast(normed, target_type=pl.BF16),
                        [0, k0],
                    )
                    qi32 = pl.cast(
                        pl.row_expand_mul(xg, quant_mul),
                        target_type=pl.INT32, mode="rint",
                    )
                    qf16 = pl.cast(qi32, target_type=pl.FP16, mode="round")
                    x_i8_tile = pl.assemble(
                        x_i8_tile,
                        pl.cast(qf16, target_type=pl.INT8, mode="trunc"),
                        [0, k0],
                    )

            # 1) Gate (local, replicated).
            expert_indices = pl.create_tensor(
                [BATCH, TOPK], dtype=pl.INT32,
            )
            expert_weights = pl.create_tensor([BATCH, TOPK], dtype=pl.FP32)
            expert_indices, expert_weights = self.gate_step(
                tile_x, gate_w, router_bias,
                expert_indices, expert_weights,
            )

            # 2) Shared-expert lane (TP-sliced + tp_all_reduce).
            sh_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            sh_y = self.expert_shared_step(
                tile_x, w_gate_s, w_up_s, w_down_s, sh_y,
                sh_tmp_window, sh_signal_window, my_rank,
            )

            # 3) Dispatch (EP all-to-all) — W8A8 INT8 payload + per-token
            # scale (decode_fwd.py:809-1047, design §4.2 step 2).
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
            inverse_map = pl.create_tensor(
                [BATCH, TOPK], dtype=pl.INT32,
            )
            (
                local_routed_x,
                local_routed_x_scale,
                local_expert_offset,
                local_expert_count,
                inverse_map,
            ) = self.dispatch_step(
                x_i8_tile, x_scale_tile, expert_indices,
                local_routed_x, local_routed_x_scale,
                local_expert_offset, local_expert_count, inverse_map,
                pub_counts, count_done_sig, recv_x, data_done_sig,
                send_buf, send_aux, recv_aux,
                my_rank,
            )

            # 4) Routed experts (local 36).
            local_routed_y = pl.create_tensor(
                [local_recv_max, HIDDEN], dtype=pl.BF16,
            )
            local_routed_y = self.expert_routed_step(
                local_routed_x, local_routed_x_scale,
                local_expert_offset, local_expert_count,
                w_gate_r, w_gate_r_scale, w_up_r, w_up_r_scale,
                w_down_r, w_down_r_scale,
                local_routed_y,
            )

            # 5) Combine (EP a2a back + weighted gather + sh_y add).
            tile_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            tile_y = self.combine_step(
                local_routed_y,
                expert_indices, expert_weights, sh_y,
                tile_y,
                pub_counts, src_route_table, route_pub_sig,
                routed_y_buf, combine_done_sig,
                my_rank,
            )

            # Per-tile residual add (PROGRESS Session 6 step 9, mirrors
            # decode): next_hidden_out[t_lo] = resid_hold[t_lo] + tile_y,
            # removing the full [PREFILL_T, HIDDEN] moe_out / resid_hold_fp32
            # buffers.  Chunked over K_CHUNK to keep Vec tiles at
            # [BATCH, K_CHUNK] FP32 = 16 KiB.
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="prefill_moe_residual_add_tile",
            ):
                for kb4 in pl.range(hidden_blocks):
                    k0 = kb4 * K_CHUNK
                    res_chunk = pl.cast(
                        pl.slice(
                            resid_hold, [BATCH, K_CHUNK], [t_lo, k0],
                        ),
                        target_type=pl.FP32,
                    )
                    y_chunk = pl.cast(
                        pl.slice(
                            tile_y, [BATCH, K_CHUNK], [0, k0],
                        ),
                        target_type=pl.FP32,
                    )
                    next_hidden_out = pl.assemble(
                        next_hidden_out,
                        pl.cast(
                            pl.add(res_chunk, y_chunk),
                            target_type=pl.BF16,
                        ),
                        [t_lo, k0],
                    )
        return next_hidden_out

    @pl.function(
        type=pl.FunctionType.Orchestration,
    )
    def swa_moe_chip_orch_swiglu7_silu(  # noqa: PLR0913, PLR0915
        self,
        current_hidden: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
        input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
        wq: pl.Tensor[[HIDDEN, hidden_q_swa], pl.BF16],
        wk: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        wv: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
        k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
        block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
        slot_mapping: pl.Tensor[[PREFILL_T], pl.INT32],
        rope_cos: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],
        rope_sin: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],
        k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
        v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
        wo: pl.Tensor[[hidden_q_swa, HIDDEN], pl.BF16],
        w_g: pl.Tensor[[HIDDEN, nh_swa_pad], pl.BF16],
        gate_r: pl.Tensor[[nh_swa_pad, hidden_q_swa], pl.BF16],
        positions: pl.Tensor[[PREFILL_T], pl.INT32],
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
        next_hidden_out: pl.Out[pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]],
        resid_hold: pl.Out[pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]],
        attn_tmp_window: pld.DistributedTensor[[PREFILL_T, tp_chunk], pl.BF16],
        attn_signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
        pub_counts: pld.DistributedTensor[[n_ranks * n_ranks, n_local_experts], pl.INT32],
        count_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        recv_x: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
        data_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        send_buf: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
        # Per-routed-row aux slab (W8A8, decode recv_aux ABI,
        # decode_fwd.py:830-832): col 0 = per-token dequant scale, cols 1..7
        # physical FP32 padding (dispatch_aux_pad) for PTOAS tile alignment.
        send_aux: pld.DistributedTensor[
            [local_recv_max, dispatch_aux_pad], pl.FP32
        ],
        recv_aux: pld.DistributedTensor[
            [local_recv_max, dispatch_aux_pad], pl.FP32
        ],
        sh_tmp_window: pld.DistributedTensor[[BATCH, sh_tp_chunk], pl.BF16],
        sh_signal_window: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        src_route_table: pld.DistributedTensor[[n_ranks, n_local_experts, n_routes_per_rank], pl.INT32],
        route_pub_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        routed_y_buf: pld.DistributedTensor[[n_routes_per_rank, HIDDEN], pl.BF16],
        combine_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        norm_layer_idx: pl.Scalar[pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]:
        # ── A: prefill attention + tp_all_reduce -> resid_hold. ────────
        resid_hold = attention_swa_inline(
            current_hidden,
            input_rms_weight,
            wq, wk, wv,
            q_norm_weight, k_norm_weight,
            block_table, slot_mapping,
            rope_cos, rope_sin,
            k_cache, v_cache,
            wo, w_g,
            gate_r,
            positions,
            resid_hold,
            norm_layer_idx,
            norm_layer_idx,
            attn_tmp_window,
            attn_signal_window,
            my_rank,
        )

        # ── B: post-attention RMSNorm + residual add fused per-tile
        # (PROGRESS Session 6 step 8+9, mirrors decode BATCH=16).  The full
        # [PREFILL_T, HIDDEN] post_norm / resid_hold_fp32 / moe_out buffers and
        # their CORE_GROUP scopes are removed; each MoE tile runs a
        # two-pass chunked RMSNorm over resid_hold[t_lo:t_lo+BATCH] producing
        # tile_x (stage C) and a per-tile residual add producing
        # next_hidden_out[t_lo] (stage D).  inv_rms is per-token (over
        # HIDDEN features) so per-tile is bit-equivalent to the full pass.
        hidden_blocks = HIDDEN // K_CHUNK

        # ── C: prefill MoE adapter — Phase X.10 inlined per-tile loop.
        # Each tile runs a two-pass chunked RMSNorm on resid_hold[t_lo] ->
        # tile_x, then the inlined gate / dispatch / expert_routed /
        # expert_shared / combine pipeline -> tile_y, then a per-tile
        # residual add -> next_hidden_out[t_lo].  Routing is per-token, so
        # this is bit-equivalent to a single T=PREFILL_T pass.  The window
        # pool shared across tiles is safe because each tile flushes
        # (signal-wait pairs) before the next reads.
        for tile_idx in pl.unroll(PREFILL_TILE_COUNT):
            t_lo = tile_idx * BATCH
            tile_x = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            x_i8_tile = pl.create_tensor([BATCH, HIDDEN], dtype=pl.INT8)
            x_scale_tile = pl.create_tensor(
                [BATCH, DISPATCH_SCALE_COLS], dtype=pl.FP32,
            )
            # V4-style deferred RMSNorm + INT8 producer (decode_fwd.py:694-805
            # _norm_quant_moe_input, design §4.2 step 1).  Two-pass chunked over
            # resid_hold[t_lo]: pass 1 forms xg = resid*(gamma+1) while reducing
            # sq_sum (RMSNorm) and xg_amax (INT8 quant scale); pass 2 emits the
            # BF16 post-norm hidden (tile_x = shared-expert + gate input) and
            # the INT8 dispatch payload x_i8_tile whose dequant scale
            # x_scale_tile = inv_rms*amax(xg)/127 carries the deferred positive
            # RMS factor.  Prefill inlines the producer (per-tile slicing of
            # resid_hold[t_lo] is already inline here); prefill tiles are full
            # BATCH (no active_tokens clamping) and the gate consumes tile_x
            # directly, so inv_rms is folded into x_scale and not emitted.
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="prefill_moe_post_rmsnorm_tile",
            ):
                sq_sum = pl.full(
                    [1, BATCH], dtype=pl.FP32, value=0.0,
                )
                xg_amax = pl.full(
                    [1, BATCH], dtype=pl.FP32, value=1e-4,
                )
                for kb in pl.range(hidden_blocks):
                    k0 = kb * K_CHUNK
                    raw = pl.cast(
                        pl.slice(
                            resid_hold, [BATCH, K_CHUNK], [t_lo, k0],
                        ),
                        target_type=pl.FP32,
                    )
                    gamma = pl.slice(
                        post_rms_weight, [1, K_CHUNK], [norm_layer_idx, k0],
                    )
                    xg = pl.col_expand_mul(raw, pl.add(gamma, 1.0))
                    sq_sum = pl.add(
                        sq_sum,
                        pl.reshape(
                            pl.row_sum(pl.mul(raw, raw)),
                            [1, BATCH],
                        ),
                    )
                    xg_amax = pl.maximum(
                        xg_amax,
                        pl.reshape(
                            pl.row_max(pl.maximum(xg, pl.neg(xg))),
                            [1, BATCH],
                        ),
                    )
                inv_rms_row = pl.recip(
                    pl.sqrt(
                        pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS),
                    ),
                )
                inv_rms_col = pl.reshape(inv_rms_row, [BATCH, 1])
                quant_mul = pl.reshape(
                    pl.div(
                        pl.full([1, BATCH], dtype=pl.FP32, value=127.0),
                        xg_amax,
                    ),
                    [BATCH, 1],
                )
                x_scale_tile = pl.assemble(
                    x_scale_tile,
                    pl.reshape(
                        pl.mul(
                            inv_rms_row,
                            pl.mul(xg_amax, 1.0 / 127.0),
                        ),
                        [BATCH, 1],
                    ),
                    [0, 0],
                )
                for kb2 in pl.range(hidden_blocks):
                    k0 = kb2 * K_CHUNK
                    raw = pl.cast(
                        pl.slice(
                            resid_hold, [BATCH, K_CHUNK], [t_lo, k0],
                        ),
                        target_type=pl.FP32,
                    )
                    gamma = pl.slice(
                        post_rms_weight, [1, K_CHUNK], [norm_layer_idx, k0],
                    )
                    xg = pl.col_expand_mul(raw, pl.add(gamma, 1.0))
                    normed = pl.row_expand_mul(xg, inv_rms_col)
                    tile_x = pl.assemble(
                        tile_x,
                        pl.cast(normed, target_type=pl.BF16),
                        [0, k0],
                    )
                    qi32 = pl.cast(
                        pl.row_expand_mul(xg, quant_mul),
                        target_type=pl.INT32, mode="rint",
                    )
                    qf16 = pl.cast(qi32, target_type=pl.FP16, mode="round")
                    x_i8_tile = pl.assemble(
                        x_i8_tile,
                        pl.cast(qf16, target_type=pl.INT8, mode="trunc"),
                        [0, k0],
                    )

            # 1) Gate (local, replicated).
            expert_indices = pl.create_tensor(
                [BATCH, TOPK], dtype=pl.INT32,
            )
            expert_weights = pl.create_tensor([BATCH, TOPK], dtype=pl.FP32)
            expert_indices, expert_weights = self.gate_step(
                tile_x, gate_w, router_bias,
                expert_indices, expert_weights,
            )

            # 2) Shared-expert lane (TP-sliced + tp_all_reduce).
            sh_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            sh_y = self.expert_shared_step(
                tile_x, w_gate_s, w_up_s, w_down_s, sh_y,
                sh_tmp_window, sh_signal_window, my_rank,
            )

            # 3) Dispatch (EP all-to-all) — W8A8 INT8 payload + per-token
            # scale (decode_fwd.py:809-1047, design §4.2 step 2).
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
            inverse_map = pl.create_tensor(
                [BATCH, TOPK], dtype=pl.INT32,
            )
            (
                local_routed_x,
                local_routed_x_scale,
                local_expert_offset,
                local_expert_count,
                inverse_map,
            ) = self.dispatch_step(
                x_i8_tile, x_scale_tile, expert_indices,
                local_routed_x, local_routed_x_scale,
                local_expert_offset, local_expert_count, inverse_map,
                pub_counts, count_done_sig, recv_x, data_done_sig,
                send_buf, send_aux, recv_aux,
                my_rank,
            )

            # 4) Routed experts (local 36).
            local_routed_y = pl.create_tensor(
                [local_recv_max, HIDDEN], dtype=pl.BF16,
            )
            local_routed_y = self.expert_routed_step_swiglu7(
                local_routed_x, local_routed_x_scale,
                local_expert_offset, local_expert_count,
                w_gate_r, w_gate_r_scale, w_up_r, w_up_r_scale,
                w_down_r, w_down_r_scale,
                local_routed_y,
            )

            # 5) Combine (EP a2a back + weighted gather + sh_y add).
            tile_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            tile_y = self.combine_step(
                local_routed_y,
                expert_indices, expert_weights, sh_y,
                tile_y,
                pub_counts, src_route_table, route_pub_sig,
                routed_y_buf, combine_done_sig,
                my_rank,
            )

            # Per-tile residual add (PROGRESS Session 6 step 9, mirrors
            # decode): next_hidden_out[t_lo] = resid_hold[t_lo] + tile_y,
            # removing the full [PREFILL_T, HIDDEN] moe_out / resid_hold_fp32
            # buffers.  Chunked over K_CHUNK to keep Vec tiles at
            # [BATCH, K_CHUNK] FP32 = 16 KiB.
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="prefill_moe_residual_add_tile",
            ):
                for kb4 in pl.range(hidden_blocks):
                    k0 = kb4 * K_CHUNK
                    res_chunk = pl.cast(
                        pl.slice(
                            resid_hold, [BATCH, K_CHUNK], [t_lo, k0],
                        ),
                        target_type=pl.FP32,
                    )
                    y_chunk = pl.cast(
                        pl.slice(
                            tile_y, [BATCH, K_CHUNK], [0, k0],
                        ),
                        target_type=pl.FP32,
                    )
                    next_hidden_out = pl.assemble(
                        next_hidden_out,
                        pl.cast(
                            pl.add(res_chunk, y_chunk),
                            target_type=pl.BF16,
                        ),
                        [t_lo, k0],
                    )
        return next_hidden_out

    @pl.function(
        type=pl.FunctionType.Orchestration,
    )
    def full_moe_chip_orch_swiglu7_swiglu16(  # noqa: PLR0913, PLR0915
        self,
        current_hidden: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
        input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
        wq: pl.Tensor[[HIDDEN, hidden_q_full], pl.BF16],
        wk: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        wv: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
        k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
        block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
        slot_mapping: pl.Tensor[[PREFILL_T], pl.INT32],
        rope_cos: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],
        rope_sin: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],
        k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
        v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
        wo: pl.Tensor[[hidden_q_full, HIDDEN], pl.BF16],
        w_g: pl.Tensor[[HIDDEN, nh_full_pad], pl.BF16],
        gate_r: pl.Tensor[[nh_full_pad, hidden_q_full], pl.BF16],
        positions: pl.Tensor[[PREFILL_T], pl.INT32],
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
        next_hidden_out: pl.Out[pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]],
        resid_hold: pl.Out[pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]],
        attn_tmp_window: pld.DistributedTensor[[PREFILL_T, tp_chunk], pl.BF16],
        attn_signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
        pub_counts: pld.DistributedTensor[[n_ranks * n_ranks, n_local_experts], pl.INT32],
        count_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        recv_x: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
        data_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        send_buf: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],
        # Per-routed-row aux slab (W8A8, decode recv_aux ABI,
        # decode_fwd.py:830-832): col 0 = per-token dequant scale, cols 1..7
        # physical FP32 padding (dispatch_aux_pad) for PTOAS tile alignment.
        send_aux: pld.DistributedTensor[
            [local_recv_max, dispatch_aux_pad], pl.FP32
        ],
        recv_aux: pld.DistributedTensor[
            [local_recv_max, dispatch_aux_pad], pl.FP32
        ],
        sh_tmp_window: pld.DistributedTensor[[BATCH, sh_tp_chunk], pl.BF16],
        sh_signal_window: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        src_route_table: pld.DistributedTensor[[n_ranks, n_local_experts, n_routes_per_rank], pl.INT32],
        route_pub_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        routed_y_buf: pld.DistributedTensor[[n_routes_per_rank, HIDDEN], pl.BF16],
        combine_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
        norm_layer_idx: pl.Scalar[pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]:
        # ── A: prefill attention + tp_all_reduce -> resid_hold. ────────
        resid_hold = attention_full_inline(
            current_hidden,
            input_rms_weight,
            wq, wk, wv,
            q_norm_weight, k_norm_weight,
            block_table, slot_mapping,
            rope_cos, rope_sin,
            k_cache, v_cache,
            wo, w_g,
            gate_r,
            positions,
            resid_hold,
            norm_layer_idx,
            norm_layer_idx,
            attn_tmp_window,
            attn_signal_window,
            my_rank,
        )

        # ── B: post-attention RMSNorm + residual add fused per-tile
        # (PROGRESS Session 6 step 8+9, mirrors decode BATCH=16).  The full
        # [PREFILL_T, HIDDEN] post_norm / resid_hold_fp32 / moe_out buffers and
        # their CORE_GROUP scopes are removed; each MoE tile runs a
        # two-pass chunked RMSNorm over resid_hold[t_lo:t_lo+BATCH] producing
        # tile_x (stage C) and a per-tile residual add producing
        # next_hidden_out[t_lo] (stage D).  inv_rms is per-token (over
        # HIDDEN features) so per-tile is bit-equivalent to the full pass.
        hidden_blocks = HIDDEN // K_CHUNK

        # ── C: prefill MoE adapter — Phase X.10 inlined per-tile loop.
        # Each tile runs a two-pass chunked RMSNorm on resid_hold[t_lo] ->
        # tile_x, then the inlined gate / dispatch / expert_routed /
        # expert_shared / combine pipeline -> tile_y, then a per-tile
        # residual add -> next_hidden_out[t_lo].  Routing is per-token, so
        # this is bit-equivalent to a single T=PREFILL_T pass.  The window
        # pool shared across tiles is safe because each tile flushes
        # (signal-wait pairs) before the next reads.
        for tile_idx in pl.unroll(PREFILL_TILE_COUNT):
            t_lo = tile_idx * BATCH
            tile_x = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            x_i8_tile = pl.create_tensor([BATCH, HIDDEN], dtype=pl.INT8)
            x_scale_tile = pl.create_tensor(
                [BATCH, DISPATCH_SCALE_COLS], dtype=pl.FP32,
            )
            # V4-style deferred RMSNorm + INT8 producer (decode_fwd.py:694-805
            # _norm_quant_moe_input, design §4.2 step 1).  Two-pass chunked over
            # resid_hold[t_lo]: pass 1 forms xg = resid*(gamma+1) while reducing
            # sq_sum (RMSNorm) and xg_amax (INT8 quant scale); pass 2 emits the
            # BF16 post-norm hidden (tile_x = shared-expert + gate input) and
            # the INT8 dispatch payload x_i8_tile whose dequant scale
            # x_scale_tile = inv_rms*amax(xg)/127 carries the deferred positive
            # RMS factor.  Prefill inlines the producer (per-tile slicing of
            # resid_hold[t_lo] is already inline here); prefill tiles are full
            # BATCH (no active_tokens clamping) and the gate consumes tile_x
            # directly, so inv_rms is folded into x_scale and not emitted.
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="prefill_moe_post_rmsnorm_tile",
            ):
                sq_sum = pl.full(
                    [1, BATCH], dtype=pl.FP32, value=0.0,
                )
                xg_amax = pl.full(
                    [1, BATCH], dtype=pl.FP32, value=1e-4,
                )
                for kb in pl.range(hidden_blocks):
                    k0 = kb * K_CHUNK
                    raw = pl.cast(
                        pl.slice(
                            resid_hold, [BATCH, K_CHUNK], [t_lo, k0],
                        ),
                        target_type=pl.FP32,
                    )
                    gamma = pl.slice(
                        post_rms_weight, [1, K_CHUNK], [norm_layer_idx, k0],
                    )
                    xg = pl.col_expand_mul(raw, pl.add(gamma, 1.0))
                    sq_sum = pl.add(
                        sq_sum,
                        pl.reshape(
                            pl.row_sum(pl.mul(raw, raw)),
                            [1, BATCH],
                        ),
                    )
                    xg_amax = pl.maximum(
                        xg_amax,
                        pl.reshape(
                            pl.row_max(pl.maximum(xg, pl.neg(xg))),
                            [1, BATCH],
                        ),
                    )
                inv_rms_row = pl.recip(
                    pl.sqrt(
                        pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS),
                    ),
                )
                inv_rms_col = pl.reshape(inv_rms_row, [BATCH, 1])
                quant_mul = pl.reshape(
                    pl.div(
                        pl.full([1, BATCH], dtype=pl.FP32, value=127.0),
                        xg_amax,
                    ),
                    [BATCH, 1],
                )
                x_scale_tile = pl.assemble(
                    x_scale_tile,
                    pl.reshape(
                        pl.mul(
                            inv_rms_row,
                            pl.mul(xg_amax, 1.0 / 127.0),
                        ),
                        [BATCH, 1],
                    ),
                    [0, 0],
                )
                for kb2 in pl.range(hidden_blocks):
                    k0 = kb2 * K_CHUNK
                    raw = pl.cast(
                        pl.slice(
                            resid_hold, [BATCH, K_CHUNK], [t_lo, k0],
                        ),
                        target_type=pl.FP32,
                    )
                    gamma = pl.slice(
                        post_rms_weight, [1, K_CHUNK], [norm_layer_idx, k0],
                    )
                    xg = pl.col_expand_mul(raw, pl.add(gamma, 1.0))
                    normed = pl.row_expand_mul(xg, inv_rms_col)
                    tile_x = pl.assemble(
                        tile_x,
                        pl.cast(normed, target_type=pl.BF16),
                        [0, k0],
                    )
                    qi32 = pl.cast(
                        pl.row_expand_mul(xg, quant_mul),
                        target_type=pl.INT32, mode="rint",
                    )
                    qf16 = pl.cast(qi32, target_type=pl.FP16, mode="round")
                    x_i8_tile = pl.assemble(
                        x_i8_tile,
                        pl.cast(qf16, target_type=pl.INT8, mode="trunc"),
                        [0, k0],
                    )

            # 1) Gate (local, replicated).
            expert_indices = pl.create_tensor(
                [BATCH, TOPK], dtype=pl.INT32,
            )
            expert_weights = pl.create_tensor([BATCH, TOPK], dtype=pl.FP32)
            expert_indices, expert_weights = self.gate_step(
                tile_x, gate_w, router_bias,
                expert_indices, expert_weights,
            )

            # 2) Shared-expert lane (TP-sliced + tp_all_reduce).
            sh_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            sh_y = self.expert_shared_step_swiglu16(
                tile_x, w_gate_s, w_up_s, w_down_s, sh_y,
                sh_tmp_window, sh_signal_window, my_rank,
            )

            # 3) Dispatch (EP all-to-all) — W8A8 INT8 payload + per-token
            # scale (decode_fwd.py:809-1047, design §4.2 step 2).
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
            inverse_map = pl.create_tensor(
                [BATCH, TOPK], dtype=pl.INT32,
            )
            (
                local_routed_x,
                local_routed_x_scale,
                local_expert_offset,
                local_expert_count,
                inverse_map,
            ) = self.dispatch_step(
                x_i8_tile, x_scale_tile, expert_indices,
                local_routed_x, local_routed_x_scale,
                local_expert_offset, local_expert_count, inverse_map,
                pub_counts, count_done_sig, recv_x, data_done_sig,
                send_buf, send_aux, recv_aux,
                my_rank,
            )

            # 4) Routed experts (local 36).
            local_routed_y = pl.create_tensor(
                [local_recv_max, HIDDEN], dtype=pl.BF16,
            )
            local_routed_y = self.expert_routed_step_swiglu7(
                local_routed_x, local_routed_x_scale,
                local_expert_offset, local_expert_count,
                w_gate_r, w_gate_r_scale, w_up_r, w_up_r_scale,
                w_down_r, w_down_r_scale,
                local_routed_y,
            )

            # 5) Combine (EP a2a back + weighted gather + sh_y add).
            tile_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            tile_y = self.combine_step(
                local_routed_y,
                expert_indices, expert_weights, sh_y,
                tile_y,
                pub_counts, src_route_table, route_pub_sig,
                routed_y_buf, combine_done_sig,
                my_rank,
            )

            # Per-tile residual add (PROGRESS Session 6 step 9, mirrors
            # decode): next_hidden_out[t_lo] = resid_hold[t_lo] + tile_y,
            # removing the full [PREFILL_T, HIDDEN] moe_out / resid_hold_fp32
            # buffers.  Chunked over K_CHUNK to keep Vec tiles at
            # [BATCH, K_CHUNK] FP32 = 16 KiB.
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="prefill_moe_residual_add_tile",
            ):
                for kb4 in pl.range(hidden_blocks):
                    k0 = kb4 * K_CHUNK
                    res_chunk = pl.cast(
                        pl.slice(
                            resid_hold, [BATCH, K_CHUNK], [t_lo, k0],
                        ),
                        target_type=pl.FP32,
                    )
                    y_chunk = pl.cast(
                        pl.slice(
                            tile_y, [BATCH, K_CHUNK], [0, k0],
                        ),
                        target_type=pl.FP32,
                    )
                    next_hidden_out = pl.assemble(
                        next_hidden_out,
                        pl.cast(
                            pl.add(res_chunk, y_chunk),
                            target_type=pl.BF16,
                        ),
                        [t_lo, k0],
                    )
        return next_hidden_out

    @pl.function(type=pl.FunctionType.Orchestration)
    def whole_chip_orch(  # noqa: PLR0913, PLR0915
        self,
        current_hidden: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
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
        moe_full_wq: pl.Tensor[[NUM_FULL_MOE_LAYERS * HIDDEN, hidden_q_full], pl.BF16],
        moe_full_wk: pl.Tensor[[NUM_FULL_MOE_LAYERS * HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        moe_full_wv: pl.Tensor[[NUM_FULL_MOE_LAYERS * HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        moe_full_wo: pl.Tensor[[NUM_FULL_MOE_LAYERS * hidden_q_full, HIDDEN], pl.BF16],
        moe_full_w_g: pl.Tensor[[NUM_FULL_MOE_LAYERS * HIDDEN, nh_full_pad], pl.BF16],
        moe_full_gate_r: pl.Tensor[[NUM_FULL_MOE_LAYERS * nh_full_pad, hidden_q_full], pl.BF16],
        moe_swa_wq: pl.Tensor[[NUM_SWA_MOE_LAYERS * HIDDEN, hidden_q_swa], pl.BF16],
        moe_swa_wk: pl.Tensor[[NUM_SWA_MOE_LAYERS * HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        moe_swa_wv: pl.Tensor[[NUM_SWA_MOE_LAYERS * HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        moe_swa_wo: pl.Tensor[[NUM_SWA_MOE_LAYERS * hidden_q_swa, HIDDEN], pl.BF16],
        moe_swa_w_g: pl.Tensor[[NUM_SWA_MOE_LAYERS * HIDDEN, nh_swa_pad], pl.BF16],
        moe_swa_gate_r: pl.Tensor[[NUM_SWA_MOE_LAYERS * nh_swa_pad, hidden_q_swa], pl.BF16],
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
        seq_lens: pl.Tensor[[1], pl.INT32],
        block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
        slot_mapping: pl.Tensor[[PREFILL_T], pl.INT32],
        position_ids: pl.Tensor[[PREFILL_T], pl.INT32],
        rope_cos_full: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],
        rope_sin_full: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],
        rope_cos_swa: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],
        rope_sin_swa: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],
        k_cache: pl.InOut[pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16]],
        v_cache: pl.InOut[pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16]],
        next_hidden_out: pl.Out[pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]],
        # ── per-chip communication windows (allocated by host_orch, stacked) ──
        dense_attn_tmp_stack: pld.DistributedTensor[[NUM_DENSE_LAYERS * PREFILL_T, tp_chunk], pl.BF16],
        dense_attn_signal_stack: pld.DistributedTensor[[NUM_DENSE_LAYERS * tp_size, 1], pl.INT32],
        dense_mlp_tmp_stack: pld.DistributedTensor[[NUM_DENSE_LAYERS * PREFILL_T, tp_chunk], pl.BF16],
        dense_mlp_signal_stack: pld.DistributedTensor[[NUM_DENSE_LAYERS * tp_size, 1], pl.INT32],
        moe_attn_tmp_stack: pld.DistributedTensor[[NUM_MOE_LAYERS_TOTAL * PREFILL_T, tp_chunk], pl.BF16],
        moe_attn_signal_stack: pld.DistributedTensor[[NUM_MOE_LAYERS_TOTAL * tp_size, 1], pl.INT32],
        moe_pub_counts_stack: pld.DistributedTensor[[NUM_MOE_LAYERS_TOTAL * n_ranks * n_ranks, n_local_experts], pl.INT32],
        moe_count_done_stack: pld.DistributedTensor[[NUM_MOE_LAYERS_TOTAL * n_ranks, 1], pl.INT32],
        moe_recv_x_stack: pld.DistributedTensor[[NUM_MOE_LAYERS_TOTAL * local_recv_max, HIDDEN], pl.INT8],
        moe_send_buf_stack: pld.DistributedTensor[[NUM_MOE_LAYERS_TOTAL * local_recv_max, HIDDEN], pl.INT8],
        # W8A8 per-routed-row aux slabs (decode recv_aux ABI): col 0 = scale.
        moe_send_aux_stack: pld.DistributedTensor[[NUM_MOE_LAYERS_TOTAL * local_recv_max, dispatch_aux_pad], pl.FP32],
        moe_recv_aux_stack: pld.DistributedTensor[[NUM_MOE_LAYERS_TOTAL * local_recv_max, dispatch_aux_pad], pl.FP32],
        moe_data_done_stack: pld.DistributedTensor[[NUM_MOE_LAYERS_TOTAL * n_ranks, 1], pl.INT32],
        moe_sh_tmp_stack: pld.DistributedTensor[[NUM_MOE_LAYERS_TOTAL * BATCH, sh_tp_chunk], pl.BF16],
        moe_sh_signal_stack: pld.DistributedTensor[[NUM_MOE_LAYERS_TOTAL * n_ranks, 1], pl.INT32],
        moe_src_route_stack: pld.DistributedTensor[[NUM_MOE_LAYERS_TOTAL * n_ranks, n_local_experts, n_routes_per_rank], pl.INT32],
        moe_route_pub_stack: pld.DistributedTensor[[NUM_MOE_LAYERS_TOTAL * n_ranks, 1], pl.INT32],
        moe_routed_y_stack: pld.DistributedTensor[[NUM_MOE_LAYERS_TOTAL * n_routes_per_rank, HIDDEN], pl.BF16],
        moe_combine_done_stack: pld.DistributedTensor[[NUM_MOE_LAYERS_TOTAL * n_ranks, 1], pl.INT32],
        num_tokens_per_owner: pl.Tensor[[NUM_TOKENS_STORAGE_I32], pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
    ):
        # ── L0: full-attn dense layer (mirror decode_fwd.py:3433-3467). ──
        h_layer_0 = pl.create_tensor([PREFILL_T, HIDDEN], dtype=pl.BF16)
        h_layer_0 = self.full_chip_orch(
            current_hidden, input_rms,
            pl.slice(full_wq, [HIDDEN, hidden_q_full], [0, 0]),
            pl.slice(full_wk, [HIDDEN, KV_HIDDEN_LOCAL_R], [0, 0]),
            pl.slice(full_wv, [HIDDEN, KV_HIDDEN_LOCAL_R], [0, 0]),
            q_norm, k_norm, block_table, slot_mapping,
            rope_cos_full, rope_sin_full, k_cache, v_cache,
            pl.slice(full_wo, [hidden_q_full, HIDDEN], [0, 0]),
            pl.slice(full_w_g, [HIDDEN, nh_full_pad], [0, 0]),
            pl.slice(full_gate_r, [nh_full_pad, hidden_q_full], [0, 0]),
            position_ids, post_rms,
            pl.slice(dense_w_gate, [HIDDEN, INTER_LOCAL], [0, 0]),
            pl.slice(dense_w_up, [HIDDEN, INTER_LOCAL], [0, 0]),
            pl.slice(dense_w_down, [INTER_LOCAL, HIDDEN], [0, 0]),
            h_layer_0,
            pl.slice(dense_attn_tmp_stack, [PREFILL_T, tp_chunk], [0, 0]),
            pl.slice(dense_attn_signal_stack, [tp_size, 1], [0, 0]),
            pl.slice(dense_mlp_tmp_stack, [PREFILL_T, tp_chunk], [0, 0]),
            pl.slice(dense_mlp_signal_stack, [tp_size, 1], [0, 0]),
            0, 0, my_rank,
        )
        prev_hidden = h_layer_0

        # ── L1/L2: swa-attn dense layers (mirror decode_fwd.py:3469-3517). ──
        for layer_idx in pl.range(NUM_SWA_DENSE_LAYERS):
            swa_w_off = layer_idx * HIDDEN
            swa_wo_off = layer_idx * hidden_q_swa
            swa_gate_r_off = layer_idx * nh_swa_pad
            dense_w_off = (layer_idx + 1) * HIDDEN
            dense_down_off = (layer_idx + 1) * INTER_LOCAL
            win_off_t = (layer_idx + 1) * PREFILL_T
            sig_off = (layer_idx + 1) * tp_size
            norm_idx = layer_idx + 1
            h_next = pl.create_tensor([PREFILL_T, HIDDEN], dtype=pl.BF16)
            h_next = self.swa_chip_orch(
                prev_hidden, input_rms,
                pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [swa_w_off, 0]),
                pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL_R], [swa_w_off, 0]),
                pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL_R], [swa_w_off, 0]),
                q_norm, k_norm, block_table, slot_mapping,
                rope_cos_swa, rope_sin_swa, k_cache, v_cache,
                pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [swa_wo_off, 0]),
                pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [swa_w_off, 0]),
                pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [swa_gate_r_off, 0]),
                position_ids, post_rms,
                pl.slice(dense_w_gate, [HIDDEN, INTER_LOCAL], [dense_w_off, 0]),
                pl.slice(dense_w_up, [HIDDEN, INTER_LOCAL], [dense_w_off, 0]),
                pl.slice(dense_w_down, [INTER_LOCAL, HIDDEN], [dense_down_off, 0]),
                h_next,
                pl.slice(dense_attn_tmp_stack, [PREFILL_T, tp_chunk], [win_off_t, 0]),
                pl.slice(dense_attn_signal_stack, [tp_size, 1], [sig_off, 0]),
                pl.slice(dense_mlp_tmp_stack, [PREFILL_T, tp_chunk], [win_off_t, 0]),
                pl.slice(dense_mlp_signal_stack, [tp_size, 1], [sig_off, 0]),
                norm_idx, 0, my_rank,
            )
            prev_hidden = h_next

        # ── L3-L42: MoE 40 layers (mirror decode_fwd.py:3519-3691). ──
        for layer_idx in pl.range(NUM_MOE_LAYERS):
            phys_layer = layer_idx + 3
            norm_layer_idx = pl.cast(phys_layer, pl.INT32)
            moe_w_off = layer_idx * HIDDEN
            moe_bias_off = layer_idx * N_EXPERTS
            moe_r_off = layer_idx * (n_local_experts * HIDDEN)
            moe_r_down_off = layer_idx * (n_local_experts * inter)
            # W8A8 per-channel weight scale offset (one [n_local_experts, *]
            # slab per layer; gate/up share the [n_local_experts, inter] slab
            # offset, down uses the same [n_local_experts, HIDDEN] row offset).
            moe_r_scale_off = layer_idx * n_local_experts
            moe_sh_down_off = layer_idx * sh_inter_local
            h_moe = pl.create_tensor([PREFILL_T, HIDDEN], dtype=pl.BF16)
            resid_hold_moe = pl.create_tensor([PREFILL_T, HIDDEN], dtype=pl.BF16)
            if layer_idx % 4 == 1:
                full_idx = (layer_idx - 1) // 4
                fa_w_off = full_idx * HIDDEN
                fa_wo_off = full_idx * hidden_q_full
                fa_gate_r_off = full_idx * nh_full_pad
                h_moe = self.full_moe_chip_orch(
                    prev_hidden, input_rms,
                    pl.slice(moe_full_wq, [HIDDEN, hidden_q_full], [fa_w_off, 0]),
                    pl.slice(moe_full_wk, [HIDDEN, KV_HIDDEN_LOCAL_R], [fa_w_off, 0]),
                    pl.slice(moe_full_wv, [HIDDEN, KV_HIDDEN_LOCAL_R], [fa_w_off, 0]),
                    q_norm, k_norm, block_table, slot_mapping,
                    rope_cos_full, rope_sin_full, k_cache, v_cache,
                    pl.slice(moe_full_wo, [hidden_q_full, HIDDEN], [fa_wo_off, 0]),
                    pl.slice(moe_full_w_g, [HIDDEN, nh_full_pad], [fa_w_off, 0]),
                    pl.slice(moe_full_gate_r, [nh_full_pad, hidden_q_full], [fa_gate_r_off, 0]),
                    position_ids, post_rms,
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
                    h_moe, resid_hold_moe,
            pl.slice(moe_attn_tmp_stack, [PREFILL_T, tp_chunk], [layer_idx * PREFILL_T, 0]),
            pl.slice(moe_attn_signal_stack, [tp_size, 1], [layer_idx * tp_size, 0]),
            pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts], [layer_idx * (n_ranks * n_ranks), 0]),
            pl.slice(moe_count_done_stack, [n_ranks, 1], [layer_idx * n_ranks, 0]),
            pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [layer_idx * local_recv_max, 0]),
            pl.slice(moe_data_done_stack, [n_ranks, 1], [layer_idx * n_ranks, 0]),
            pl.slice(moe_send_buf_stack, [local_recv_max, HIDDEN], [layer_idx * local_recv_max, 0]),
            pl.slice(moe_send_aux_stack, [local_recv_max, dispatch_aux_pad], [layer_idx * local_recv_max, 0]),
            pl.slice(moe_recv_aux_stack, [local_recv_max, dispatch_aux_pad], [layer_idx * local_recv_max, 0]),
            pl.slice(moe_sh_tmp_stack, [BATCH, sh_tp_chunk], [layer_idx * BATCH, 0]),
            pl.slice(moe_sh_signal_stack, [n_ranks, 1], [layer_idx * n_ranks, 0]),
            pl.slice(moe_src_route_stack, [n_ranks, n_local_experts, n_routes_per_rank], [layer_idx * n_ranks, 0, 0]),
            pl.slice(moe_route_pub_stack, [n_ranks, 1], [layer_idx * n_ranks, 0]),
            pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [layer_idx * n_routes_per_rank, 0]),
            pl.slice(moe_combine_done_stack, [n_ranks, 1], [layer_idx * n_ranks, 0]),
                    norm_layer_idx, my_rank,
                )
            else:
                full_before = (layer_idx + 2) // 4
                swa_idx = layer_idx - full_before
                swa_w_off = swa_idx * HIDDEN
                swa_wo_off = swa_idx * hidden_q_swa
                swa_gate_r_off = swa_idx * nh_swa_pad
                h_moe = self.swa_moe_chip_orch(
                    prev_hidden, input_rms,
                    pl.slice(moe_swa_wq, [HIDDEN, hidden_q_swa], [swa_w_off, 0]),
                    pl.slice(moe_swa_wk, [HIDDEN, KV_HIDDEN_LOCAL_R], [swa_w_off, 0]),
                    pl.slice(moe_swa_wv, [HIDDEN, KV_HIDDEN_LOCAL_R], [swa_w_off, 0]),
                    q_norm, k_norm, block_table, slot_mapping,
                    rope_cos_swa, rope_sin_swa, k_cache, v_cache,
                    pl.slice(moe_swa_wo, [hidden_q_swa, HIDDEN], [swa_wo_off, 0]),
                    pl.slice(moe_swa_w_g, [HIDDEN, nh_swa_pad], [swa_w_off, 0]),
                    pl.slice(moe_swa_gate_r, [nh_swa_pad, hidden_q_swa], [swa_gate_r_off, 0]),
                    position_ids, post_rms,
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
                    h_moe, resid_hold_moe,
            pl.slice(moe_attn_tmp_stack, [PREFILL_T, tp_chunk], [layer_idx * PREFILL_T, 0]),
            pl.slice(moe_attn_signal_stack, [tp_size, 1], [layer_idx * tp_size, 0]),
            pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts], [layer_idx * (n_ranks * n_ranks), 0]),
            pl.slice(moe_count_done_stack, [n_ranks, 1], [layer_idx * n_ranks, 0]),
            pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [layer_idx * local_recv_max, 0]),
            pl.slice(moe_data_done_stack, [n_ranks, 1], [layer_idx * n_ranks, 0]),
            pl.slice(moe_send_buf_stack, [local_recv_max, HIDDEN], [layer_idx * local_recv_max, 0]),
            pl.slice(moe_send_aux_stack, [local_recv_max, dispatch_aux_pad], [layer_idx * local_recv_max, 0]),
            pl.slice(moe_recv_aux_stack, [local_recv_max, dispatch_aux_pad], [layer_idx * local_recv_max, 0]),
            pl.slice(moe_sh_tmp_stack, [BATCH, sh_tp_chunk], [layer_idx * BATCH, 0]),
            pl.slice(moe_sh_signal_stack, [n_ranks, 1], [layer_idx * n_ranks, 0]),
            pl.slice(moe_src_route_stack, [n_ranks, n_local_experts, n_routes_per_rank], [layer_idx * n_ranks, 0, 0]),
            pl.slice(moe_route_pub_stack, [n_ranks, 1], [layer_idx * n_ranks, 0]),
            pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [layer_idx * n_routes_per_rank, 0]),
            pl.slice(moe_combine_done_stack, [n_ranks, 1], [layer_idx * n_ranks, 0]),
                    norm_layer_idx, my_rank,
                )
            prev_hidden = h_moe

        # ── L43: swa_moe_swiglu7_silu post-loop (mirror decode_fwd.py:3693). ──
        h_layer_43 = pl.create_tensor([PREFILL_T, HIDDEN], dtype=pl.BF16)
        resid_hold_43 = pl.create_tensor([PREFILL_T, HIDDEN], dtype=pl.BF16)
        norm_layer_idx_43 = pl.cast(43, pl.INT32)
        h_layer_43 = self.swa_moe_chip_orch_swiglu7_silu(
            prev_hidden, input_rms,
            pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [32 * HIDDEN, 0]),
            pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL_R], [32 * HIDDEN, 0]),
            pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL_R], [32 * HIDDEN, 0]),
            q_norm, k_norm, block_table, slot_mapping,
            rope_cos_swa, rope_sin_swa, k_cache, v_cache,
            pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [32 * hidden_q_swa, 0]),
            pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [32 * HIDDEN, 0]),
            pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [32 * nh_swa_pad, 0]),
            position_ids, post_rms,
            pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [40 * HIDDEN, 0]),
            pl.slice(moe_router_bias, [N_EXPERTS], [40 * N_EXPERTS]),
            pl.reshape(
                pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [40 * (n_local_experts * HIDDEN), 0]),
                [n_local_experts, HIDDEN, inter],
            ),
            pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [40 * n_local_experts, 0]),
            pl.reshape(
                pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [40 * (n_local_experts * HIDDEN), 0]),
                [n_local_experts, HIDDEN, inter],
            ),
            pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [40 * n_local_experts, 0]),
            pl.reshape(
                pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [40 * (n_local_experts * inter), 0]),
                [n_local_experts, inter, HIDDEN],
            ),
            pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [40 * n_local_experts, 0]),
            pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [40 * HIDDEN, 0]),
            pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [40 * HIDDEN, 0]),
            pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [40 * sh_inter_local, 0]),
            h_layer_43, resid_hold_43,
            pl.slice(moe_attn_tmp_stack, [PREFILL_T, tp_chunk], [40 * PREFILL_T, 0]),
            pl.slice(moe_attn_signal_stack, [tp_size, 1], [40 * tp_size, 0]),
            pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts], [40 * (n_ranks * n_ranks), 0]),
            pl.slice(moe_count_done_stack, [n_ranks, 1], [40 * n_ranks, 0]),
            pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [40 * local_recv_max, 0]),
            pl.slice(moe_data_done_stack, [n_ranks, 1], [40 * n_ranks, 0]),
            pl.slice(moe_send_buf_stack, [local_recv_max, HIDDEN], [40 * local_recv_max, 0]),
            pl.slice(moe_send_aux_stack, [local_recv_max, dispatch_aux_pad], [40 * local_recv_max, 0]),
            pl.slice(moe_recv_aux_stack, [local_recv_max, dispatch_aux_pad], [40 * local_recv_max, 0]),
            pl.slice(moe_sh_tmp_stack, [BATCH, sh_tp_chunk], [40 * BATCH, 0]),
            pl.slice(moe_sh_signal_stack, [n_ranks, 1], [40 * n_ranks, 0]),
            pl.slice(moe_src_route_stack, [n_ranks, n_local_experts, n_routes_per_rank], [40 * n_ranks, 0, 0]),
            pl.slice(moe_route_pub_stack, [n_ranks, 1], [40 * n_ranks, 0]),
            pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [40 * n_routes_per_rank, 0]),
            pl.slice(moe_combine_done_stack, [n_ranks, 1], [40 * n_ranks, 0]),
            norm_layer_idx_43, my_rank,
        )
        prev_hidden = h_layer_43

        # ── L44: full_moe_swiglu7_swiglu16 post-loop (mirror decode_fwd.py:3775). ──
        resid_hold_44 = pl.create_tensor([PREFILL_T, HIDDEN], dtype=pl.BF16)
        norm_layer_idx_44 = pl.cast(44, pl.INT32)
        next_hidden_out = self.full_moe_chip_orch_swiglu7_swiglu16(
            prev_hidden, input_rms,
            pl.slice(full_wq, [HIDDEN, hidden_q_full], [11 * HIDDEN, 0]),
            pl.slice(full_wk, [HIDDEN, KV_HIDDEN_LOCAL_R], [11 * HIDDEN, 0]),
            pl.slice(full_wv, [HIDDEN, KV_HIDDEN_LOCAL_R], [11 * HIDDEN, 0]),
            q_norm, k_norm, block_table, slot_mapping,
            rope_cos_full, rope_sin_full, k_cache, v_cache,
            pl.slice(full_wo, [hidden_q_full, HIDDEN], [11 * hidden_q_full, 0]),
            pl.slice(full_w_g, [HIDDEN, nh_full_pad], [11 * HIDDEN, 0]),
            pl.slice(full_gate_r, [nh_full_pad, hidden_q_full], [11 * nh_full_pad, 0]),
            position_ids, post_rms,
            pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [41 * HIDDEN, 0]),
            pl.slice(moe_router_bias, [N_EXPERTS], [41 * N_EXPERTS]),
            pl.reshape(
                pl.slice(moe_w_gate_r, [n_local_experts * HIDDEN, inter], [41 * (n_local_experts * HIDDEN), 0]),
                [n_local_experts, HIDDEN, inter],
            ),
            pl.slice(moe_w_gate_r_scale, [n_local_experts, inter], [41 * n_local_experts, 0]),
            pl.reshape(
                pl.slice(moe_w_up_r, [n_local_experts * HIDDEN, inter], [41 * (n_local_experts * HIDDEN), 0]),
                [n_local_experts, HIDDEN, inter],
            ),
            pl.slice(moe_w_up_r_scale, [n_local_experts, inter], [41 * n_local_experts, 0]),
            pl.reshape(
                pl.slice(moe_w_down_r, [n_local_experts * inter, HIDDEN], [41 * (n_local_experts * inter), 0]),
                [n_local_experts, inter, HIDDEN],
            ),
            pl.slice(moe_w_down_r_scale, [n_local_experts, HIDDEN], [41 * n_local_experts, 0]),
            pl.slice(moe_w_gate_s, [HIDDEN, sh_inter_local], [41 * HIDDEN, 0]),
            pl.slice(moe_w_up_s, [HIDDEN, sh_inter_local], [41 * HIDDEN, 0]),
            pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [41 * sh_inter_local, 0]),
            next_hidden_out, resid_hold_44,
            pl.slice(moe_attn_tmp_stack, [PREFILL_T, tp_chunk], [41 * PREFILL_T, 0]),
            pl.slice(moe_attn_signal_stack, [tp_size, 1], [41 * tp_size, 0]),
            pl.slice(moe_pub_counts_stack, [n_ranks * n_ranks, n_local_experts], [41 * (n_ranks * n_ranks), 0]),
            pl.slice(moe_count_done_stack, [n_ranks, 1], [41 * n_ranks, 0]),
            pl.slice(moe_recv_x_stack, [local_recv_max, HIDDEN], [41 * local_recv_max, 0]),
            pl.slice(moe_data_done_stack, [n_ranks, 1], [41 * n_ranks, 0]),
            pl.slice(moe_send_buf_stack, [local_recv_max, HIDDEN], [41 * local_recv_max, 0]),
            pl.slice(moe_send_aux_stack, [local_recv_max, dispatch_aux_pad], [41 * local_recv_max, 0]),
            pl.slice(moe_recv_aux_stack, [local_recv_max, dispatch_aux_pad], [41 * local_recv_max, 0]),
            pl.slice(moe_sh_tmp_stack, [BATCH, sh_tp_chunk], [41 * BATCH, 0]),
            pl.slice(moe_sh_signal_stack, [n_ranks, 1], [41 * n_ranks, 0]),
            pl.slice(moe_src_route_stack, [n_ranks, n_local_experts, n_routes_per_rank], [41 * n_ranks, 0, 0]),
            pl.slice(moe_route_pub_stack, [n_ranks, 1], [41 * n_ranks, 0]),
            pl.slice(moe_routed_y_stack, [n_routes_per_rank, HIDDEN], [41 * n_routes_per_rank, 0]),
            pl.slice(moe_combine_done_stack, [n_ranks, 1], [41 * n_ranks, 0]),
            norm_layer_idx_44, my_rank,
        )
        return next_hidden_out
    @pl.function(
        level=pl.Level.HOST,
        role=pl.Role.Orchestrator,
    )
    def host_orch(  # noqa: PLR0913, PLR0915
        self,
        current_hidden: pl.Tensor[[tp_size, PREFILL_T, HIDDEN], pl.BF16],
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
        seq_lens: pl.Tensor[[tp_size, 1], pl.INT32],
        block_table: pl.Tensor[[tp_size, BLOCK_TABLE_FLAT_DYN], pl.INT32],
        slot_mapping: pl.Tensor[[tp_size, PREFILL_T], pl.INT32],
        position_ids: pl.Tensor[[tp_size, PREFILL_T], pl.INT32],
        rope_cos_full: pl.Tensor[[tp_size, ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],
        rope_sin_full: pl.Tensor[[tp_size, ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],
        rope_cos_swa: pl.Tensor[[tp_size, ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],
        rope_sin_swa: pl.Tensor[[tp_size, ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],
        k_cache: pl.InOut[pl.Tensor[[tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16]],
        v_cache: pl.InOut[pl.Tensor[[tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16]],
        next_hidden_out: pl.Out[pl.Tensor[[tp_size, PREFILL_T, HIDDEN], pl.BF16]],
        num_tokens_per_owner: pl.Tensor[[NUM_TOKENS_RUNTIME], pl.INT32],
    ):
        # ── allocate per-chip communication windows (mirror decode host_orch
        # decode_fwd.py:3912-3935, stacked per-layer for prefill, no moe_epoch). ──
        dense_attn_tmp_buf = pld.alloc_window_buffer(NUM_DENSE_LAYERS * PREFILL_T * tp_chunk * 2)
        dense_attn_signal_buf = pld.alloc_window_buffer(NUM_DENSE_LAYERS * tp_size * 4)
        dense_mlp_tmp_buf = pld.alloc_window_buffer(NUM_DENSE_LAYERS * PREFILL_T * tp_chunk * 2)
        dense_mlp_signal_buf = pld.alloc_window_buffer(NUM_DENSE_LAYERS * tp_size * 4)
        moe_attn_tmp_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * PREFILL_T * tp_chunk * 2)
        moe_attn_signal_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * tp_size * 4)
        moe_pub_counts_buf = pld.alloc_window_buffer(
            NUM_MOE_LAYERS_TOTAL * n_ranks * n_ranks * n_local_experts * 4,
        )
        moe_count_done_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * n_ranks * 4)
        moe_recv_x_buf = pld.alloc_window_buffer(
            NUM_MOE_LAYERS_TOTAL * local_recv_max * HIDDEN,
        )
        moe_send_buf_buf = pld.alloc_window_buffer(
            NUM_MOE_LAYERS_TOTAL * local_recv_max * HIDDEN,
        )
        # W8A8 per-routed-row aux slabs (FP32, decode recv_aux ABI).
        moe_send_aux_buf = pld.alloc_window_buffer(
            NUM_MOE_LAYERS_TOTAL * local_recv_max * dispatch_aux_pad * 4,
        )
        moe_recv_aux_buf = pld.alloc_window_buffer(
            NUM_MOE_LAYERS_TOTAL * local_recv_max * dispatch_aux_pad * 4,
        )
        moe_data_done_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * n_ranks * 4)
        moe_sh_tmp_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * BATCH * sh_tp_chunk * 2)
        moe_sh_signal_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * n_ranks * 4)
        moe_src_route_buf = pld.alloc_window_buffer(
            NUM_MOE_LAYERS_TOTAL * n_ranks * n_local_experts * n_routes_per_rank * 4,
        )
        moe_route_pub_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * n_ranks * 4)
        moe_routed_y_buf = pld.alloc_window_buffer(
            NUM_MOE_LAYERS_TOTAL * n_routes_per_rank * HIDDEN * 2,
        )
        moe_combine_done_buf = pld.alloc_window_buffer(NUM_MOE_LAYERS_TOTAL * n_ranks * 4)

        # ── fan out to every chip (mirror decode_fwd.py:3936-4026, device=r). ──
        for r in pl.range(pld.world_size()):
            self.whole_chip_orch(
                current_hidden[r],
                input_rms[r], post_rms[r], q_norm[r], k_norm[r],
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
                seq_lens[r], block_table[r], slot_mapping[r], position_ids[r],
                rope_cos_full[r], rope_sin_full[r], rope_cos_swa[r], rope_sin_swa[r],
                k_cache[r], v_cache[r], next_hidden_out[r],
                pld.window(dense_attn_tmp_buf, [NUM_DENSE_LAYERS * PREFILL_T, tp_chunk], dtype=pl.BF16),
                pld.window(dense_attn_signal_buf, [NUM_DENSE_LAYERS * tp_size, 1], dtype=pl.INT32),
                pld.window(dense_mlp_tmp_buf, [NUM_DENSE_LAYERS * PREFILL_T, tp_chunk], dtype=pl.BF16),
                pld.window(dense_mlp_signal_buf, [NUM_DENSE_LAYERS * tp_size, 1], dtype=pl.INT32),
                pld.window(moe_attn_tmp_buf, [NUM_MOE_LAYERS_TOTAL * PREFILL_T, tp_chunk], dtype=pl.BF16),
                pld.window(moe_attn_signal_buf, [NUM_MOE_LAYERS_TOTAL * tp_size, 1], dtype=pl.INT32),
                pld.window(moe_pub_counts_buf, [NUM_MOE_LAYERS_TOTAL * n_ranks * n_ranks, n_local_experts], dtype=pl.INT32),
                pld.window(moe_count_done_buf, [NUM_MOE_LAYERS_TOTAL * n_ranks, 1], dtype=pl.INT32),
                pld.window(moe_recv_x_buf, [NUM_MOE_LAYERS_TOTAL * local_recv_max, HIDDEN], dtype=pl.INT8),
                pld.window(moe_send_buf_buf, [NUM_MOE_LAYERS_TOTAL * local_recv_max, HIDDEN], dtype=pl.INT8),
                pld.window(moe_send_aux_buf, [NUM_MOE_LAYERS_TOTAL * local_recv_max, dispatch_aux_pad], dtype=pl.FP32),
                pld.window(moe_recv_aux_buf, [NUM_MOE_LAYERS_TOTAL * local_recv_max, dispatch_aux_pad], dtype=pl.FP32),
                pld.window(moe_data_done_buf, [NUM_MOE_LAYERS_TOTAL * n_ranks, 1], dtype=pl.INT32),
                pld.window(moe_sh_tmp_buf, [NUM_MOE_LAYERS_TOTAL * BATCH, sh_tp_chunk], dtype=pl.BF16),
                pld.window(moe_sh_signal_buf, [NUM_MOE_LAYERS_TOTAL * n_ranks, 1], dtype=pl.INT32),
                pld.window(moe_src_route_buf, [NUM_MOE_LAYERS_TOTAL * n_ranks, n_local_experts, n_routes_per_rank], dtype=pl.INT32),
                pld.window(moe_route_pub_buf, [NUM_MOE_LAYERS_TOTAL * n_ranks, 1], dtype=pl.INT32),
                pld.window(moe_routed_y_buf, [NUM_MOE_LAYERS_TOTAL * n_routes_per_rank, HIDDEN], dtype=pl.BF16),
                pld.window(moe_combine_done_buf, [NUM_MOE_LAYERS_TOTAL * n_ranks, 1], dtype=pl.INT32),
                num_tokens_per_owner,
                r,
                device=r,
            )


whole_prefill_step3p5 = WholePrefillStep3p5
