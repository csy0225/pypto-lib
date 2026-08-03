# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Two-layer decoder program: L0 full-attention dense and L1 SWA dense.

This is a fast iteration fixture for attention critical-path optimization. Each
change to ``attention_full`` or ``attention_swa`` otherwise requires compiling
and running the canonical 45-layer ``whole_decode_step3p5`` program on eight
devices, which takes tens of minutes. A whole-network DFX capture also contains
more than 1,500 tasks, where MoE noise obscures the smaller attention dependency
chain. This program retains only the first two canonical layers:

- compile and runtime iteration remains on the order of minutes;
- swimlanes contain only attention, dense MLP, and TP all-reduce;
- the second layer preserves inter-layer overlap between the previous MLP
  all-reduce and the following attention prologue.

**No-drift contract**: ``pl.inline`` copies the attention and dense-MLP bodies
directly from ``models.step3p5.attention_full``, ``attention_swa``, and
``dense_mlp``. This module only mirrors the thin ``*_chip_orch`` wrappers from
``decode_fwd``. It also inherits the complete set of configuration globals from
``decode_fwd`` because free variables in an inlined body resolve against the
caller module. Manually copying selected constants can silently omit one, as
previously happened with ``INPUT_PROJ_K_CHUNK``.

Layer mapping, aligned with canonical ``whole_chip_orch``:

=============  ==============  ==========  ==========  ==========
logical layer  variant         norm_idx    attn_idx    mlp_idx
=============  ==============  ==========  ==========  ==========
L0             full_dense      0           0           0
L1             swa_dense       1           0           0
=============  ==============  ==========  ==========  ==========

``attn_idx`` and ``mlp_idx`` remain zero because weights are passed one layer at
a time rather than as a 45-layer stack, so each kernel's base offset is zero.
``norm_idx`` still selects rows from the canonical ``LAYER_DYN`` norm table.
"""
from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld

import models.step3p5.decode_fwd as _canonical
from models.step3p5.attention_full import attention_full
from models.step3p5.attention_swa import attention_swa
from models.step3p5.dense_mlp import dense_mlp_body_tp

# Inherit every canonical module global (config constants, per-rank slice
# widths, comm-signal footprint, tiling chunks). The inlined kernel bodies
# resolve their free variables against THIS module's globals, so the set must
# match canonical exactly; enumerating names by hand silently drops one.
globals().update(
    {
        key: value
        for key, value in vars(_canonical).items()
        if not key.startswith("_") and key != "WholeDecodeStep3p5"
    }
)

# Re-declared so static readers (and linters) see the names this module's
# signatures depend on; values come from the canonical inherit above.
BATCH = _canonical.BATCH
HIDDEN = _canonical.HIDDEN
HEAD_DIM = _canonical.HEAD_DIM
LAYER_DYN = _canonical.LAYER_DYN
USER_BATCH_DYN = _canonical.USER_BATCH_DYN
BLOCK_TABLE_FLAT_DYN = _canonical.BLOCK_TABLE_FLAT_DYN
ROPE_SEQ_DYN = _canonical.ROPE_SEQ_DYN
KV_CACHE_ROWS_DYN = _canonical.KV_CACHE_ROWS_DYN
COMM_SIGNAL_STRIDE_I32 = _canonical.COMM_SIGNAL_STRIDE_I32
COMM_CONTROL_SIGNAL_BYTES = _canonical.COMM_CONTROL_SIGNAL_BYTES
TP_ALL_REDUCE_CHUNK = _canonical.TP_ALL_REDUCE_CHUNK
NUM_TOKENS_RUNTIME = _canonical.NUM_TOKENS_RUNTIME
INTER_LOCAL = _canonical.INTER_LOCAL
KV_HIDDEN_LOCAL_R = _canonical.KV_HIDDEN_LOCAL_R
hidden_q_full = _canonical.hidden_q_full
hidden_q_swa = _canonical.hidden_q_swa
nh_full_pad = _canonical.nh_full_pad
nh_swa_pad = _canonical.nh_swa_pad
rotary_dim_full = _canonical.rotary_dim_full
rotary_dim_swa = _canonical.rotary_dim_swa
tp_size = _canonical.tp_size

# Two dense layers -> two window slots per collective kind.
N_LAYERS = 2

attention_full_inline = pl.inline(attention_full._func)
attention_swa_inline = pl.inline(attention_swa._func)
dense_mlp_inline = pl.inline(dense_mlp_body_tp._func)


@pl.program
class TwoLayerAttnPerf:
    # Mirrors WholeDecodeStep3p5.tp_all_reduce verbatim. The inlined attn /
    # dense-MLP bodies call ``self.tp_all_reduce`` to sum the o_proj /
    # down_proj partials, so it must exist as a method of THIS program.
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

    # Mirrors WholeDecodeStep3p5.full_chip_orch.
    @pl.function(
        type=pl.FunctionType.Orchestration,
        attrs={"inline_orchestration": True},
    )
    def full_layer(  # noqa: PLR0913
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

    # Mirrors WholeDecodeStep3p5.swa_chip_orch.
    @pl.function(
        type=pl.FunctionType.Orchestration,
        attrs={"inline_orchestration": True},
    )
    def swa_layer(  # noqa: PLR0913
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

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch(  # noqa: PLR0913
        self,
        current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        input_rms: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
        post_rms: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
        q_norm: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
        k_norm: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
        full_wq: pl.Tensor[[HIDDEN, hidden_q_full], pl.BF16],
        full_wk: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        full_wv: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        full_wo: pl.Tensor[[hidden_q_full, HIDDEN], pl.BF16],
        full_w_g: pl.Tensor[[HIDDEN, nh_full_pad], pl.BF16],
        full_gate_r: pl.Tensor[[nh_full_pad, hidden_q_full], pl.BF16],
        swa_wq: pl.Tensor[[HIDDEN, hidden_q_swa], pl.BF16],
        swa_wk: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        swa_wv: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        swa_wo: pl.Tensor[[hidden_q_swa, HIDDEN], pl.BF16],
        swa_w_g: pl.Tensor[[HIDDEN, nh_swa_pad], pl.BF16],
        swa_gate_r: pl.Tensor[[nh_swa_pad, hidden_q_swa], pl.BF16],
        dense_w_gate: pl.Tensor[[N_LAYERS * HIDDEN, INTER_LOCAL], pl.BF16],
        dense_w_up: pl.Tensor[[N_LAYERS * HIDDEN, INTER_LOCAL], pl.BF16],
        dense_w_down: pl.Tensor[[N_LAYERS * INTER_LOCAL, HIDDEN], pl.BF16],
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
        attn_tmp_stack: pld.DistributedTensor[[N_LAYERS * BATCH, HIDDEN], pl.BF16],
        attn_signal_stack: pld.DistributedTensor[
            [N_LAYERS * COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        mlp_tmp_stack: pld.DistributedTensor[[N_LAYERS * BATCH, HIDDEN], pl.BF16],
        mlp_signal_stack: pld.DistributedTensor[
            [N_LAYERS * COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        num_tokens_per_owner: pl.Tensor[[NUM_TOKENS_RUNTIME], pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
    ):
        # Packed-global active-row contract, mirroring canonical whole_chip_orch:
        # every owner is written the same count, so max() yields identical
        # dynamic bounds on every rank.
        num_tokens = pl.cast(0, pl.INT32)
        for owner_rank in pl.range(tp_size):
            num_tokens = pl.max(
                num_tokens,
                pl.read(num_tokens_per_owner, [owner_rank]),
            )
        if num_tokens < 0:
            num_tokens = pl.cast(0, pl.INT32)
        if num_tokens > BATCH:
            num_tokens = pl.cast(BATCH, pl.INT32)

        # ── L0: full-attn dense. ──
        h_layer_0 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        h_layer_0 = self.full_layer(
            current_hidden,
            input_rms,
            full_wq,
            full_wk,
            full_wv,
            q_norm,
            k_norm,
            seq_lens,
            block_table,
            slot_mapping,
            rope_cos_full,
            rope_sin_full,
            k_cache,
            v_cache,
            full_wo,
            full_w_g,
            full_gate_r,
            post_rms,
            pl.slice(dense_w_gate, [HIDDEN, INTER_LOCAL], [0, 0]),
            pl.slice(dense_w_up, [HIDDEN, INTER_LOCAL], [0, 0]),
            pl.slice(dense_w_down, [INTER_LOCAL, HIDDEN], [0, 0]),
            h_layer_0,
            pl.slice(attn_tmp_stack, [BATCH, HIDDEN], [0, 0]),
            pl.slice(attn_signal_stack, [COMM_SIGNAL_STRIDE_I32, 1], [0, 0]),
            pl.slice(mlp_tmp_stack, [BATCH, HIDDEN], [0, 0]),
            pl.slice(mlp_signal_stack, [COMM_SIGNAL_STRIDE_I32, 1], [0, 0]),
            0,
            0,
            0,
            num_tokens,
            my_rank,
        )

        # ── L1: swa-attn dense. ──
        next_hidden_out = self.swa_layer(
            h_layer_0,
            input_rms,
            swa_wq,
            swa_wk,
            swa_wv,
            q_norm,
            k_norm,
            seq_lens,
            block_table,
            slot_mapping,
            rope_cos_swa,
            rope_sin_swa,
            k_cache,
            v_cache,
            swa_wo,
            swa_w_g,
            swa_gate_r,
            post_rms,
            pl.slice(dense_w_gate, [HIDDEN, INTER_LOCAL], [HIDDEN, 0]),
            pl.slice(dense_w_up, [HIDDEN, INTER_LOCAL], [HIDDEN, 0]),
            pl.slice(dense_w_down, [INTER_LOCAL, HIDDEN], [INTER_LOCAL, 0]),
            next_hidden_out,
            pl.slice(attn_tmp_stack, [BATCH, HIDDEN], [BATCH, 0]),
            pl.slice(
                attn_signal_stack, [COMM_SIGNAL_STRIDE_I32, 1],
                [COMM_SIGNAL_STRIDE_I32, 0],
            ),
            pl.slice(mlp_tmp_stack, [BATCH, HIDDEN], [BATCH, 0]),
            pl.slice(
                mlp_signal_stack, [COMM_SIGNAL_STRIDE_I32, 1],
                [COMM_SIGNAL_STRIDE_I32, 0],
            ),
            1,
            0,
            0,
            num_tokens,
            my_rank,
        )
        return next_hidden_out

    @pl.function(
        level=pl.Level.HOST,
        role=pl.Role.Orchestrator,
    )
    def host_orch(  # noqa: PLR0913
        self,
        current_hidden: pl.Tensor[[tp_size, BATCH, HIDDEN], pl.BF16],
        input_rms: pl.Tensor[[tp_size, LAYER_DYN, HIDDEN], pl.FP32],
        post_rms: pl.Tensor[[tp_size, LAYER_DYN, HIDDEN], pl.FP32],
        q_norm: pl.Tensor[[tp_size, LAYER_DYN, HEAD_DIM], pl.FP32],
        k_norm: pl.Tensor[[tp_size, LAYER_DYN, HEAD_DIM], pl.FP32],
        full_wq: pl.Tensor[[tp_size, HIDDEN, hidden_q_full], pl.BF16],
        full_wk: pl.Tensor[[tp_size, HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        full_wv: pl.Tensor[[tp_size, HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        full_wo: pl.Tensor[[tp_size, hidden_q_full, HIDDEN], pl.BF16],
        full_w_g: pl.Tensor[[tp_size, HIDDEN, nh_full_pad], pl.BF16],
        full_gate_r: pl.Tensor[[tp_size, nh_full_pad, hidden_q_full], pl.BF16],
        swa_wq: pl.Tensor[[tp_size, HIDDEN, hidden_q_swa], pl.BF16],
        swa_wk: pl.Tensor[[tp_size, HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        swa_wv: pl.Tensor[[tp_size, HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16],
        swa_wo: pl.Tensor[[tp_size, hidden_q_swa, HIDDEN], pl.BF16],
        swa_w_g: pl.Tensor[[tp_size, HIDDEN, nh_swa_pad], pl.BF16],
        swa_gate_r: pl.Tensor[[tp_size, nh_swa_pad, hidden_q_swa], pl.BF16],
        dense_w_gate: pl.Tensor[[tp_size, N_LAYERS, HIDDEN, INTER_LOCAL], pl.BF16],
        dense_w_up: pl.Tensor[[tp_size, N_LAYERS, HIDDEN, INTER_LOCAL], pl.BF16],
        dense_w_down: pl.Tensor[[tp_size, N_LAYERS, INTER_LOCAL, HIDDEN], pl.BF16],
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
        attn_tmp_stack_buf = pld.alloc_window_buffer(N_LAYERS * BATCH * HIDDEN * 2)
        attn_signal_stack_buf = pld.alloc_window_buffer(
            N_LAYERS * COMM_CONTROL_SIGNAL_BYTES
        )
        mlp_tmp_stack_buf = pld.alloc_window_buffer(N_LAYERS * BATCH * HIDDEN * 2)
        mlp_signal_stack_buf = pld.alloc_window_buffer(
            N_LAYERS * COMM_CONTROL_SIGNAL_BYTES
        )
        for r in pl.range(pld.world_size()):
            self.chip_orch(
                current_hidden[r],
                input_rms[r],
                post_rms[r],
                q_norm[r],
                k_norm[r],
                full_wq[r],
                full_wk[r],
                full_wv[r],
                full_wo[r],
                full_w_g[r],
                full_gate_r[r],
                swa_wq[r],
                swa_wk[r],
                swa_wv[r],
                swa_wo[r],
                swa_w_g[r],
                swa_gate_r[r],
                pl.reshape(dense_w_gate[r], [N_LAYERS * HIDDEN, INTER_LOCAL]),
                pl.reshape(dense_w_up[r], [N_LAYERS * HIDDEN, INTER_LOCAL]),
                pl.reshape(dense_w_down[r], [N_LAYERS * INTER_LOCAL, HIDDEN]),
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
                pld.window(attn_tmp_stack_buf, [N_LAYERS * BATCH, HIDDEN],
                           dtype=pl.BF16),
                pld.window(attn_signal_stack_buf,
                           [N_LAYERS * COMM_SIGNAL_STRIDE_I32, 1],
                           dtype=pl.INT32),
                pld.window(mlp_tmp_stack_buf, [N_LAYERS * BATCH, HIDDEN],
                           dtype=pl.BF16),
                pld.window(mlp_signal_stack_buf,
                           [N_LAYERS * COMM_SIGNAL_STRIDE_I32, 1],
                           dtype=pl.INT32),
                num_tokens_per_owner,
                r,
                device=r,
            )


two_layer_attn_perf = TwoLayerAttnPerf
