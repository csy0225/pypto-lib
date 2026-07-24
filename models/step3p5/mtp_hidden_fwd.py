# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Selected Step3p5 MTP body with a hidden-only production boundary.

The vLLM proposer invokes one compile-time layer variant at a time.  PyPTO
owns token embedding plus the selected MTP decoder body and writes only the
raw hidden state.  vLLM owns every operation after that boundary.
"""
from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld

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
    HIDDEN_Q_SWA_LOCAL,
    INPUT_PROJ_K_CHUNK,
    INTERMEDIATE_LOCAL,
    K_CHUNK,
    LAYER_DYN,
    LAYER_HIDDEN_ROWS_DYN,
    LAYER_INTER_ROWS_DYN,
    KV_HEADS_LOCAL,
    KV_HIDDEN_LOCAL,
    KV_PROJ_K_CHUNK_LOCAL,
    MLP_OUT_CHUNK,
    MTP_KV_CACHE_ROWS_DYN,
    NUM_HEADS_SWA_LOCAL_PAD,
    NUM_NEXTN_PREDICT_LAYERS,
    OUT_PROJ_K_CHUNK,
    OUT_PROJ_N_CHUNK,
    Q_HEAD_BATCH_SWA,
    Q_OUT_CHUNK,
    Q_PER_KV_SWA,
    ROPE_SEQ_DYN,
    ROTARY_HALF_SWA,
    SLIDING_WINDOW,
    TP_WORLD_SIZE,
    USER_BATCH_DYN,
    VOCAB,
)
from .attention_swa import (
    LAYER_QHIDDEN_ROWS_DYN as LAYER_QHIDDEN_ROWS_DYN_SWA,
    attention_swa,
)
from .decode_layer_single_chip_hidden import _dense_mlp_body_tp


NUM_MTP = NUM_NEXTN_PREDICT_LAYERS
HIDDEN_LOCAL = HIDDEN // TP_WORLD_SIZE
INTER_LOCAL = INTERMEDIATE_LOCAL
MTP_EH_ROWS = NUM_MTP * HIDDEN_LOCAL
MTP_HIDDEN_ROWS = NUM_MTP * HIDDEN
MTP_QHIDDEN_ROWS = NUM_MTP * HIDDEN_Q_SWA_LOCAL
MTP_INTER_ROWS = NUM_MTP * INTERMEDIATE_LOCAL
MTP_CACHE_ROWS = NUM_MTP * MTP_KV_CACHE_ROWS_DYN
COMM_CONTROL_SIGNAL_BYTES = 512
MTP_EH_ROWS_DYN = pl.dynamic("MTP_EH_ROWS_DYN")
EH_OUT_CHUNK = (
    OUT_PROJ_N_CHUNK
    if OUT_PROJ_N_CHUNK <= HIDDEN_LOCAL
    else HIDDEN_LOCAL
)

assert HIDDEN_LOCAL % EH_OUT_CHUNK == 0


@pl.jit.inline
def _mtp_input_proj_body(
    prev_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    embed_next: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    enorm_weight: pl.Tensor[[NUM_NEXTN_PREDICT_LAYERS, HIDDEN], pl.FP32],
    hnorm_weight: pl.Tensor[[NUM_NEXTN_PREDICT_LAYERS, HIDDEN], pl.FP32],
    eh_proj_weight: pl.Tensor[[MTP_EH_ROWS_DYN, 2 * HIDDEN], pl.BF16],
    mtp_in_out: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    mtp_layer_idx: pl.Scalar[pl.INT32],
    eh_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
    eh_signal_window: pld.DistributedTensor[[TP_WORLD_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
    """Build the selected MTP input without importing the legacy MTP tail.

    ``enorm(embed_next)`` and ``hnorm(prev_hidden)`` are replicated. Each
    rank computes its row-sliced ``eh_proj`` output, writes it into its own
    hidden-width slot, and the program's TP all-reduce assembles the complete
    hidden state. This helper exposes no logits, token selection, or
    acceptance state.
    """
    hidden_blocks = HIDDEN // K_CHUNK
    eh_blocks = (2 * HIDDEN) // K_CHUNK
    eh_out_blocks = HIDDEN_LOCAL // EH_OUT_CHUNK
    layer_out_base = mtp_layer_idx * HIDDEN_LOCAL

    concat_tile = pl.create_tensor([BATCH, 2 * HIDDEN], dtype=pl.BF16)

    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_enorm"):
            e_sq_sum = pl.full(
                [1, BATCH_TILE], dtype=pl.FP32, value=0.0
            )
            for kb in pl.range(hidden_blocks):
                e_sq_k0 = kb * K_CHUNK
                e_sq_chunk = pl.cast(
                    pl.slice(
                        embed_next,
                        [BATCH_TILE, K_CHUNK],
                        [b0, e_sq_k0],
                    ),
                    target_type=pl.FP32,
                )
                e_sq_sum = pl.add(
                    e_sq_sum,
                    pl.reshape(
                        pl.row_sum(pl.mul(e_sq_chunk, e_sq_chunk)),
                        [1, BATCH_TILE],
                    ),
                )
            e_inv_rms = pl.reshape(
                pl.recip(
                    pl.sqrt(
                        pl.add(pl.mul(e_sq_sum, HIDDEN_INV), EPS)
                    )
                ),
                [BATCH_TILE, 1],
            )
            for kb in pl.range(hidden_blocks):
                e_norm_k0 = kb * K_CHUNK
                e_chunk = pl.cast(
                    pl.slice(
                        embed_next,
                        [BATCH_TILE, K_CHUNK],
                        [b0, e_norm_k0],
                    ),
                    target_type=pl.FP32,
                )
                e_gamma = pl.slice(
                    enorm_weight,
                    [1, K_CHUNK],
                    [mtp_layer_idx, e_norm_k0],
                )
                e_scaled = pl.row_expand_mul(e_chunk, e_inv_rms)
                e_normed = pl.col_expand_mul(
                    e_scaled, pl.add(e_gamma, 1.0)
                )
                concat_tile = pl.assemble(
                    concat_tile,
                    pl.cast(e_normed, target_type=pl.BF16),
                    [b0, e_norm_k0],
                )

    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_hnorm"):
            h_sq_sum = pl.full(
                [1, BATCH_TILE], dtype=pl.FP32, value=0.0
            )
            for kb in pl.range(hidden_blocks):
                h_sq_k0 = kb * K_CHUNK
                h_sq_chunk = pl.cast(
                    pl.slice(
                        prev_hidden,
                        [BATCH_TILE, K_CHUNK],
                        [b0, h_sq_k0],
                    ),
                    target_type=pl.FP32,
                )
                h_sq_sum = pl.add(
                    h_sq_sum,
                    pl.reshape(
                        pl.row_sum(pl.mul(h_sq_chunk, h_sq_chunk)),
                        [1, BATCH_TILE],
                    ),
                )
            h_inv_rms = pl.reshape(
                pl.recip(
                    pl.sqrt(
                        pl.add(pl.mul(h_sq_sum, HIDDEN_INV), EPS)
                    )
                ),
                [BATCH_TILE, 1],
            )
            for kb in pl.range(hidden_blocks):
                h_norm_k0 = kb * K_CHUNK
                h_chunk = pl.cast(
                    pl.slice(
                        prev_hidden,
                        [BATCH_TILE, K_CHUNK],
                        [b0, h_norm_k0],
                    ),
                    target_type=pl.FP32,
                )
                h_gamma = pl.slice(
                    hnorm_weight,
                    [1, K_CHUNK],
                    [mtp_layer_idx, h_norm_k0],
                )
                h_scaled = pl.row_expand_mul(h_chunk, h_inv_rms)
                h_normed = pl.col_expand_mul(
                    h_scaled, pl.add(h_gamma, 1.0)
                )
                concat_tile = pl.assemble(
                    concat_tile,
                    pl.cast(h_normed, target_type=pl.BF16),
                    [b0, HIDDEN + h_norm_k0],
                )

    partial = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    rank_col_base = my_rank * HIDDEN_LOCAL
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="mtp_eh_zero_partial",
    ):
        for kb0 in pl.range(hidden_blocks):
            zero_k0 = kb0 * K_CHUNK
            zero_chunk = pl.full(
                [BATCH, K_CHUNK], dtype=pl.BF16, value=0.0
            )
            partial = pl.assemble(
                partial, zero_chunk, [0, zero_k0]
            )

    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        for ob in pl.spmd(
            eh_out_blocks,
            name_hint="mtp_eh_proj_tp",
            optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
        ):
            o0 = ob * EH_OUT_CHUNK
            mtp_eh_a_chunk_0 = pl.slice(
                concat_tile, [BATCH_TILE, K_CHUNK], [b0, 0]
            )
            mtp_eh_w_chunk_0 = pl.slice(
                eh_proj_weight,
                [EH_OUT_CHUNK, K_CHUNK],
                [layer_out_base + o0, 0],
            )
            eh_acc = pl.matmul(
                mtp_eh_a_chunk_0,
                mtp_eh_w_chunk_0,
                out_dtype=pl.FP32,
                b_trans=True,
            )
            for kb in pl.range(1, eh_blocks):
                k0 = kb * K_CHUNK
                mtp_eh_a_chunk = pl.slice(
                    concat_tile,
                    [BATCH_TILE, K_CHUNK],
                    [b0, k0],
                )
                mtp_eh_w_chunk = pl.slice(
                    eh_proj_weight,
                    [EH_OUT_CHUNK, K_CHUNK],
                    [layer_out_base + o0, k0],
                )
                eh_acc = pl.matmul_acc(
                    eh_acc,
                    mtp_eh_a_chunk,
                    mtp_eh_w_chunk,
                    b_trans=True,
                )
            partial = pl.assemble(
                partial,
                pl.cast(eh_acc, target_type=pl.BF16),
                [b0, rank_col_base + o0],
            )

    if TP_WORLD_SIZE > 1:
        self.tp_all_reduce(
            partial,
            eh_tmp_window,
            eh_signal_window,
            my_rank,
        )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_eh_copy_out"):
        for kb in pl.range(hidden_blocks):
            k0 = kb * K_CHUNK
            chunk = pl.slice(
                partial, [BATCH, K_CHUNK], [0, k0]
            )
            mtp_in_out = pl.assemble(
                mtp_in_out, chunk, [0, k0]
            )
    return mtp_in_out


def _build_mtp_layer_hidden_program(
    layer_idx: int,
    *,
    tp_size: int = TP_WORLD_SIZE,
):
    """Build one compile-time selected MTP layer program."""
    if layer_idx not in range(NUM_MTP):
        raise ValueError(f"layer_idx must be 0..{NUM_MTP - 1}, got {layer_idx}")
    if tp_size != TP_WORLD_SIZE:
        raise ValueError(
            f"MTP hidden program requires TP={TP_WORLD_SIZE}, got {tp_size}"
        )

    # Do not call the composite ``mtp_layer_hidden`` helper here.  PyPTO's
    # frontend only accepts an external ``@pl.jit.inline`` body when it is
    # explicitly lifted into the enclosing @pl.program method.  The composite
    # helper itself calls other inline bodies and therefore is not a valid
    # cross-function call target in this context.
    input_proj_inline = pl.inline(_mtp_input_proj_body._func)
    attention_inline = pl.inline(attention_swa._func)
    dense_mlp_inline = pl.inline(_dense_mlp_body_tp._func)

    @pl.program
    class MtpLayerHidden:
        @pl.function(type=pl.FunctionType.InCore)
        def tp_all_reduce(
            self,
            local: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
            signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            """Released two-wave barrier all-reduce, with fresh call windows."""
            ar_chunk = HIDDEN // 8
            for k0 in pl.range(0, HIDDEN, ar_chunk):
                stage_tile = pl.load(local, [0, k0], [BATCH, ar_chunk])
                pl.store(stage_tile, [0, k0], tmp_window)

            for peer in pl.range(tp_size):
                if peer != my_rank:
                    pld.system.notify(
                        target=signal_window,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for src in pl.range(tp_size):
                if src != my_rank:
                    pld.system.wait(
                        signal=signal_window,
                        offsets=[src, 0],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )

            for k0 in pl.range(0, HIDDEN, ar_chunk):
                own_tile = pl.load(tmp_window, [0, k0], [BATCH, ar_chunk])
                acc = pl.cast(own_tile, target_type=pl.FP32)
                for peer in pl.range(tp_size):
                    if peer != my_rank:
                        recv = pld.tile.remote_load(
                            tmp_window,
                            peer=peer,
                            offsets=[0, k0],
                            shape=[BATCH, ar_chunk],
                        )
                        acc = pl.add(acc, pl.cast(recv, target_type=pl.FP32))
                pl.store(
                    pl.cast(acc, target_type=pl.BF16),
                    [0, k0],
                    local,
                )

            for peer in pl.range(tp_size):
                if peer != my_rank:
                    pld.system.notify(
                        target=signal_window,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for src in pl.range(tp_size):
                if src != my_rank:
                    pld.system.wait(
                        signal=signal_window,
                        offsets=[src, 0],
                        expected=2,
                        cmp=pld.WaitCmp.Ge,
                    )
            return local

        @pl.function(type=pl.FunctionType.InCore)
        def mask_rows(
            self,
            source: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            active_mask: pl.Tensor[[BATCH], pl.INT32],
            output: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            for b in pl.parallel(BATCH):
                active = pl.read(active_mask, [b])
                for k0 in pl.range(0, HIDDEN, 256):
                    if active != 0:
                        row = pl.slice(source, [1, 256], [b, k0])
                    else:
                        row = pl.full([1, 256], dtype=pl.BF16, value=0.0)
                    output = pl.assemble(output, row, [b, k0])
            return output

        @pl.function(type=pl.FunctionType.InCore)
        def embedding_lookup(
            self,
            token_ids: pl.Tensor[[BATCH], pl.INT32],
            active_mask: pl.Tensor[[BATCH], pl.INT32],
            seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            embed_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
            embed_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            """Lookup the vLLM-sampled token and keep the position-0 contract."""
            for b in pl.parallel(BATCH):
                active = pl.read(active_mask, [b])
                seq_len = pl.read(seq_lens, [b])
                if active != 0:
                    if seq_len != 1:
                        token_id = pl.cast(
                            pl.read(token_ids, [b]), pl.INDEX
                        )
                        for k0 in pl.range(0, HIDDEN, 256):
                            row = pl.slice(
                                embed_weight,
                                [1, 256],
                                [token_id, k0],
                            )
                            embed_out = pl.assemble(
                                embed_out, row, [b, k0]
                            )
                    else:
                        for k0 in pl.range(0, HIDDEN, 256):
                            zero_row = pl.full(
                                [1, 256], dtype=pl.BF16, value=0.0
                            )
                            embed_out = pl.assemble(
                                embed_out, zero_row, [b, k0]
                            )
                else:
                    for k0 in pl.range(0, HIDDEN, 256):
                        zero_row = pl.full(
                            [1, 256], dtype=pl.BF16, value=0.0
                        )
                        embed_out = pl.assemble(
                            embed_out, zero_row, [b, k0]
                        )
            return embed_out

        @pl.function(type=pl.FunctionType.Orchestration)
        def layer_orch(  # noqa: PLR0913
            self,
            previous_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            input_token_ids: pl.Tensor[[BATCH], pl.INT32],
            active_mask: pl.Tensor[[BATCH], pl.INT32],
            embed_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
            enorm_weight: pl.Tensor[[NUM_MTP, HIDDEN], pl.FP32],
            hnorm_weight: pl.Tensor[[NUM_MTP, HIDDEN], pl.FP32],
            eh_proj_weight: pl.Tensor[[MTP_EH_ROWS, 2 * HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[NUM_MTP, HIDDEN], pl.FP32],
            wq: pl.Tensor[[MTP_HIDDEN_ROWS, HIDDEN_Q_SWA_LOCAL], pl.BF16],
            wk: pl.Tensor[[MTP_HIDDEN_ROWS, KV_HIDDEN_LOCAL], pl.BF16],
            wv: pl.Tensor[[MTP_HIDDEN_ROWS, KV_HIDDEN_LOCAL], pl.BF16],
            q_norm_weight: pl.Tensor[[NUM_MTP, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[NUM_MTP, HEAD_DIM], pl.FP32],
            seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            rope_cos: pl.Tensor[[ROPE_SEQ_DYN, HEAD_DIM], pl.FP32],
            rope_sin: pl.Tensor[[ROPE_SEQ_DYN, HEAD_DIM], pl.FP32],
            k_cache: pl.InOut[
                pl.Tensor[[MTP_CACHE_ROWS, HEAD_DIM], pl.BF16]
            ],
            v_cache: pl.InOut[
                pl.Tensor[[MTP_CACHE_ROWS, HEAD_DIM], pl.BF16]
            ],
            wo: pl.Tensor[[MTP_QHIDDEN_ROWS, HIDDEN], pl.BF16],
            w_g: pl.Tensor[
                [MTP_HIDDEN_ROWS, NUM_HEADS_SWA_LOCAL_PAD], pl.BF16
            ],
            gate_r: pl.Tensor[
                [NUM_HEADS_SWA_LOCAL_PAD, HIDDEN_Q_SWA_LOCAL], pl.BF16
            ],
            post_rms_weight: pl.Tensor[[NUM_MTP, HIDDEN], pl.FP32],
            w_gate: pl.Tensor[
                [MTP_HIDDEN_ROWS, INTERMEDIATE_LOCAL], pl.BF16
            ],
            w_up: pl.Tensor[
                [MTP_HIDDEN_ROWS, INTERMEDIATE_LOCAL], pl.BF16
            ],
            w_down: pl.Tensor[[MTP_INTER_ROWS, HIDDEN], pl.BF16],
            hidden_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            eh_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
            eh_signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
            attn_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
            attn_signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
            mlp_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
            mlp_signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            embed_next = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            embed_next = self.embedding_lookup(
                input_token_ids,
                active_mask,
                seq_lens,
                embed_weight,
                embed_next,
            )
            masked_previous = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            masked_previous = self.mask_rows(
                previous_hidden,
                active_mask,
                masked_previous,
            )
            mtp_in = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            mtp_in = input_proj_inline(
                masked_previous,
                embed_next,
                enorm_weight,
                hnorm_weight,
                eh_proj_weight,
                mtp_in,
                layer_idx,
                eh_tmp_window,
                eh_signal_window,
                my_rank,
            )
            resid1 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            resid1 = attention_inline(
                mtp_in,
                input_rms_weight,
                wq,
                wk,
                wv,
                q_norm_weight,
                k_norm_weight,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos,
                rope_sin,
                k_cache,
                v_cache,
                wo,
                w_g,
                gate_r,
                resid1,
                layer_idx,
                layer_idx,
                attn_tmp_window,
                attn_signal_window,
                my_rank,
            )
            raw_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            raw_hidden = dense_mlp_inline(
                resid1,
                post_rms_weight,
                w_gate,
                w_up,
                w_down,
                raw_hidden,
                layer_idx,
                layer_idx,
                mlp_tmp_window,
                mlp_signal_window,
                my_rank,
            )
            return self.mask_rows(raw_hidden, active_mask, hidden_out)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(  # noqa: PLR0913
            self,
            previous_hidden: pl.Tensor[
                [tp_size, BATCH, HIDDEN], pl.BF16
            ],
            input_token_ids: pl.Tensor[[tp_size, BATCH], pl.INT32],
            active_mask: pl.Tensor[[tp_size, BATCH], pl.INT32],
            embed_weight: pl.Tensor[[tp_size, VOCAB, HIDDEN], pl.BF16],
            enorm_weight: pl.Tensor[[tp_size, NUM_MTP, HIDDEN], pl.FP32],
            hnorm_weight: pl.Tensor[[tp_size, NUM_MTP, HIDDEN], pl.FP32],
            eh_proj_weight: pl.Tensor[
                [tp_size, MTP_EH_ROWS, 2 * HIDDEN], pl.BF16
            ],
            input_rms_weight: pl.Tensor[
                [tp_size, NUM_MTP, HIDDEN], pl.FP32
            ],
            wq: pl.Tensor[
                [tp_size, MTP_HIDDEN_ROWS, HIDDEN_Q_SWA_LOCAL], pl.BF16
            ],
            wk: pl.Tensor[
                [tp_size, MTP_HIDDEN_ROWS, KV_HIDDEN_LOCAL], pl.BF16
            ],
            wv: pl.Tensor[
                [tp_size, MTP_HIDDEN_ROWS, KV_HIDDEN_LOCAL], pl.BF16
            ],
            q_norm_weight: pl.Tensor[
                [tp_size, NUM_MTP, HEAD_DIM], pl.FP32
            ],
            k_norm_weight: pl.Tensor[
                [tp_size, NUM_MTP, HEAD_DIM], pl.FP32
            ],
            wo: pl.Tensor[
                [tp_size, MTP_QHIDDEN_ROWS, HIDDEN], pl.BF16
            ],
            w_g: pl.Tensor[
                [tp_size, MTP_HIDDEN_ROWS, NUM_HEADS_SWA_LOCAL_PAD],
                pl.BF16,
            ],
            gate_r: pl.Tensor[
                [tp_size, NUM_HEADS_SWA_LOCAL_PAD, HIDDEN_Q_SWA_LOCAL],
                pl.BF16,
            ],
            post_rms_weight: pl.Tensor[
                [tp_size, NUM_MTP, HIDDEN], pl.FP32
            ],
            w_gate: pl.Tensor[
                [tp_size, MTP_HIDDEN_ROWS, INTERMEDIATE_LOCAL], pl.BF16
            ],
            w_up: pl.Tensor[
                [tp_size, MTP_HIDDEN_ROWS, INTERMEDIATE_LOCAL], pl.BF16
            ],
            w_down: pl.Tensor[
                [tp_size, MTP_INTER_ROWS, HIDDEN], pl.BF16
            ],
            seq_lens: pl.Tensor[[tp_size, USER_BATCH_DYN], pl.INT32],
            block_table: pl.Tensor[
                [tp_size, BLOCK_TABLE_FLAT_DYN], pl.INT32
            ],
            slot_mapping: pl.Tensor[
                [tp_size, USER_BATCH_DYN], pl.INT32
            ],
            rope_cos: pl.Tensor[
                [tp_size, ROPE_SEQ_DYN, HEAD_DIM], pl.FP32
            ],
            rope_sin: pl.Tensor[
                [tp_size, ROPE_SEQ_DYN, HEAD_DIM], pl.FP32
            ],
            k_cache: pl.InOut[
                pl.Tensor[
                    [tp_size, MTP_CACHE_ROWS, HEAD_DIM], pl.BF16
                ]
            ],
            v_cache: pl.InOut[
                pl.Tensor[
                    [tp_size, MTP_CACHE_ROWS, HEAD_DIM], pl.BF16
                ]
            ],
            hidden_out: pl.Out[
                pl.Tensor[[tp_size, BATCH, HIDDEN], pl.BF16]
            ],
        ) -> pl.Tensor[[tp_size, BATCH, HIDDEN], pl.BF16]:
            eh_tmp = pld.alloc_window_buffer(BATCH * HIDDEN * 2)
            eh_sig = pld.alloc_window_buffer(COMM_CONTROL_SIGNAL_BYTES)
            attn_tmp = pld.alloc_window_buffer(BATCH * HIDDEN * 2)
            attn_sig = pld.alloc_window_buffer(COMM_CONTROL_SIGNAL_BYTES)
            mlp_tmp = pld.alloc_window_buffer(BATCH * HIDDEN * 2)
            mlp_sig = pld.alloc_window_buffer(COMM_CONTROL_SIGNAL_BYTES)
            for rank in pl.range(pld.world_size()):
                self.layer_orch(
                    previous_hidden[rank],
                    input_token_ids[rank],
                    active_mask[rank],
                    embed_weight[rank],
                    enorm_weight[rank],
                    hnorm_weight[rank],
                    eh_proj_weight[rank],
                    input_rms_weight[rank],
                    wq[rank],
                    wk[rank],
                    wv[rank],
                    q_norm_weight[rank],
                    k_norm_weight[rank],
                    seq_lens[rank],
                    block_table[rank],
                    slot_mapping[rank],
                    rope_cos[rank],
                    rope_sin[rank],
                    k_cache[rank],
                    v_cache[rank],
                    wo[rank],
                    w_g[rank],
                    gate_r[rank],
                    post_rms_weight[rank],
                    w_gate[rank],
                    w_up[rank],
                    w_down[rank],
                    hidden_out[rank],
                    pld.window(eh_tmp, [BATCH, HIDDEN], dtype=pl.BF16),
                    pld.window(eh_sig, [tp_size, 1], dtype=pl.INT32),
                    pld.window(attn_tmp, [BATCH, HIDDEN], dtype=pl.BF16),
                    pld.window(attn_sig, [tp_size, 1], dtype=pl.INT32),
                    pld.window(mlp_tmp, [BATCH, HIDDEN], dtype=pl.BF16),
                    pld.window(mlp_sig, [tp_size, 1], dtype=pl.INT32),
                    rank,
                    device=rank,
                )
            return hidden_out

    return MtpLayerHidden


mtp_layer_hidden_0 = _build_mtp_layer_hidden_program(0)
mtp_layer_hidden_1 = _build_mtp_layer_hidden_program(1)
mtp_layer_hidden_2 = _build_mtp_layer_hidden_program(2)
MTP_LAYER_HIDDEN_PROGRAMS = (
    mtp_layer_hidden_0,
    mtp_layer_hidden_1,
    mtp_layer_hidden_2,
)


__all__ = [
    "MTP_CACHE_ROWS",
    "MTP_LAYER_HIDDEN_PROGRAMS",
    "_build_mtp_layer_hidden_program",
    "mtp_layer_hidden_0",
    "mtp_layer_hidden_1",
    "mtp_layer_hidden_2",
]
