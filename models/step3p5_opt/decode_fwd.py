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

        next_hidden_out = prev_hidden
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
                r,
                device=r,
            )


whole_decode_opt = WholeDecodeOpt
