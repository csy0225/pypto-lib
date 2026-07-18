# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""N=1 three-layer Step3p5 MTP forward program.

The 45-layer target network and the MTP drafter are separated by the target
sampler.  This module owns the device-resident side of that boundary: given
the target model's final hidden and the already-sampled first token, run MTP
layers 45/46/47 in one ``@pl.program`` and emit three draft-token logits and
global greedy token ids. The three layer-boundary hidden states are exposed
through one ``[TP, 3, BATCH, HIDDEN]`` output and are reused directly as the
layer45→46→47 handoff buffers.

All MTP checkpoint tensors are native BF16 in the Ascend W8A8 checkpoint.
This file does not dequantize any W8A8 routed-MoE weight.
"""

# pyright: reportUndefinedVariable=false

from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld

from .attention_swa import attention_swa
from .config import (
    ATTN_SCALE,
    BATCH,
    BATCH_TILE,
    BLOCK_SIZE,
    BLOCK_TABLE_FLAT_DYN,
    EPS,
    FINAL_RMS_K_CHUNK,
    HEAD_DIM,
    HEAD_DIM_INV,
    HIDDEN,
    HIDDEN_INV,
    HIDDEN_Q_SWA_LOCAL,
    INPUT_PROJ_K_CHUNK,
    INTERMEDIATE_LOCAL,
    K_CHUNK,
    KV_CACHE_ROWS_DYN,
    KV_HEADS_LOCAL,
    KV_HIDDEN_LOCAL,
    KV_PROJ_K_CHUNK_LOCAL,
    LM_HEAD_K_CHUNK,
    MLP_OUT_CHUNK,
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
    VOCAB_CHUNK,
    VOCAB_LOCAL,
)
from .decode_layer import _dense_mlp_body_tp
from .mtp import _mtp_input_proj_body, _mtp_shared_head_body


NUM_MTP = NUM_NEXTN_PREDICT_LAYERS
HIDDEN_LOCAL = HIDDEN // TP_WORLD_SIZE
INTER_LOCAL = INTERMEDIATE_LOCAL
MTP_EH_ROWS = NUM_MTP * HIDDEN_LOCAL
MTP_HIDDEN_ROWS = NUM_MTP * HIDDEN
MTP_QHIDDEN_ROWS = NUM_MTP * HIDDEN_Q_SWA_LOCAL
MTP_INTER_ROWS = NUM_MTP * INTERMEDIATE_LOCAL
MTP_VOCAB_ROWS = NUM_MTP * VOCAB_LOCAL
MTP_CACHE_ROWS = NUM_MTP * KV_CACHE_ROWS_DYN
LOCAL_VOCAB_CHUNKS = VOCAB_LOCAL // VOCAB_CHUNK
LOCAL_VOCAB_CHUNKS_PAD = (
    (LOCAL_VOCAB_CHUNKS + 1) // 2 * 2
)
LOCAL_CANDIDATE_GROUP = 16
LOCAL_CANDIDATE_GROUPS = (
    LOCAL_VOCAB_CHUNKS_PAD // LOCAL_CANDIDATE_GROUP
)
LOCAL_CANDIDATE_GROUPS_PAD = (
    (LOCAL_CANDIDATE_GROUPS + 1) // 2 * 2
)

COMM_CONTROL_SIGNAL_BYTES = 512
COMM_SCALAR_DATA_BYTES = 512


def _build_whole_mtp3_program(tp_size: int = TP_WORLD_SIZE):
    """Build the TP=8 N=1 MTP3 program."""
    if tp_size != TP_WORLD_SIZE:
        raise ValueError(
            f"whole_mtp3 requires canonical TP={TP_WORLD_SIZE}, got {tp_size}"
        )
    if VOCAB % tp_size != 0:
        raise ValueError(f"VOCAB={VOCAB} must divide tp_size={tp_size}")

    mtp_input_proj_inline = pl.inline(_mtp_input_proj_body._func)
    attention_swa_inline = pl.inline(attention_swa._func)
    dense_mlp_inline = pl.inline(_dense_mlp_body_tp._func)
    mtp_shared_head_inline = pl.inline(_mtp_shared_head_body._func)

    @pl.program
    class WholeMtp3:
        @pl.function(type=pl.FunctionType.InCore)
        def tp_all_reduce(
            self,
            local: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
            signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            """Two-wave barrier all-reduce used by the released P42 program."""
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

            # Completion wave: no following collective may overwrite a scratch
            # slot before every peer has completed its remote reads.
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
        def embedding_lookup(
            self,
            token_ids: pl.Tensor[[BATCH], pl.INT32],
            active_mask: pl.Tensor[[BATCH], pl.INT32],
            seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            embed_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
            embed_out: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            """Replicated lookup with vLLM's position-0 MTP mask.

            The attention ABI derives the current position as
            ``seq_lens[b] - 1``.  Keep that same metadata as the single source
            of truth here: an active row at ``seq_len == 1`` is position 0 and
            must receive a zero embedding, matching
            ``patch_deepseek_mtp.py``.  Inactive rows are also explicitly
            zeroed.
            """
            for b in pl.parallel(BATCH):
                active = pl.read(active_mask, [b])
                seq_len = pl.read(seq_lens, [b])
                if active != 0 and seq_len != 1:
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
            return embed_out

        @pl.function(type=pl.FunctionType.InCore)
        def mask_previous_hidden(
            self,
            previous_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            active_mask: pl.Tensor[[BATCH], pl.INT32],
            masked_hidden_out: pl.Out[
                pl.Tensor[[BATCH, HIDDEN], pl.BF16]
            ],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            """Make the layer handoff boundary explicit for inactive rows."""
            for b in pl.parallel(BATCH):
                active = pl.read(active_mask, [b])
                for k0 in pl.range(0, HIDDEN, 256):
                    if active != 0:
                        hidden_row = pl.slice(
                            previous_hidden,
                            [1, 256],
                            [b, k0],
                        )
                    else:
                        hidden_row = pl.full(
                            [1, 256], dtype=pl.BF16, value=0.0
                        )
                    masked_hidden_out = pl.assemble(
                        masked_hidden_out,
                        hidden_row,
                        [b, k0],
                    )
            return masked_hidden_out

        @pl.function(type=pl.FunctionType.InCore)
        def local_vocab_chunk_candidates(
            self,
            logits: pl.Tensor[[USER_BATCH_DYN, VOCAB_LOCAL], pl.FP32],
            chunk_values_out: pl.Out[
                pl.Tensor[[BATCH, LOCAL_VOCAB_CHUNKS_PAD], pl.FP32]
            ],
            chunk_ids_out: pl.Out[
                pl.Tensor[[BATCH, LOCAL_VOCAB_CHUNKS_PAD], pl.INT32]
            ],
            my_rank: pl.Scalar[pl.INT32],
        ):
            """Reduce each 16-token vocab chunk with explicit tile ops."""
            # Initialize the full aligned table first.  The final padded
            # column remains -inf/0 and therefore cannot win a reduction.
            for chunk_block in pl.range(
                LOCAL_VOCAB_CHUNKS_PAD // VOCAB_CHUNK
            ):
                chunk_block_base = chunk_block * VOCAB_CHUNK
                pl.store(
                    pl.tile.full(
                        [BATCH, VOCAB_CHUNK],
                        dtype=pl.FP32,
                        value=-3.4028235e38,
                    ),
                    [0, chunk_block_base],
                    chunk_values_out,
                )
                pl.store(
                    pl.tile.full(
                        [BATCH, VOCAB_CHUNK],
                        dtype=pl.INT32,
                        value=0,
                    ),
                    [0, chunk_block_base],
                    chunk_ids_out,
                )

            for chunk_idx in pl.parallel(LOCAL_VOCAB_CHUNKS):
                vocab_base = chunk_idx * VOCAB_CHUNK
                logits_tile = pl.load(
                    logits,
                    [0, vocab_base],
                    [BATCH, VOCAB_CHUNK],
                )
                argmax_tmp = pl.tile.create(
                    [BATCH, VOCAB_CHUNK],
                    dtype=pl.FP32,
                    target_memory=pl.Mem.Vec,
                )
                chunk_local_idx = pl.tile.row_argmax(
                    logits_tile, argmax_tmp
                )
                chunk_value = pl.tile.row_max(
                    logits_tile, argmax_tmp
                )
                # row_argmax/row_max return [BATCH,1] ColMajor tiles.  A
                # direct tile.store into one column of the row-major candidate
                # table corrupts rows > 0 on the current lowering.  Flatten to
                # one aligned [1,BATCH] row and write scalar-by-scalar.
                chunk_value_row = pl.reshape(
                    chunk_value,
                    [1, BATCH],
                )
                chunk_local_idx_row = pl.reshape(
                    chunk_local_idx,
                    [1, BATCH],
                )
                for b in pl.range(BATCH):
                    pl.write(
                        chunk_values_out,
                        [b, chunk_idx],
                        pl.read(chunk_value_row, [0, b]),
                    )
                    pl.write(
                        chunk_ids_out,
                        [b, chunk_idx],
                        pl.read(chunk_local_idx_row, [0, b]),
                    )
            return chunk_values_out, chunk_ids_out

        @pl.function(type=pl.FunctionType.InCore)
        def local_vocab_group_candidates(
            self,
            chunk_values: pl.Tensor[
                [BATCH, LOCAL_VOCAB_CHUNKS_PAD], pl.FP32
            ],
            chunk_ids: pl.Tensor[
                [BATCH, LOCAL_VOCAB_CHUNKS_PAD], pl.INT32
            ],
            group_values_out: pl.Out[
                pl.Tensor[[BATCH, LOCAL_CANDIDATE_GROUPS_PAD], pl.FP32]
            ],
            group_chunks_out: pl.Out[
                pl.Tensor[[BATCH, LOCAL_CANDIDATE_GROUPS_PAD], pl.INT32]
            ],
        ):
            """Reduce 16 chunk candidates into one candidate per group."""
            pl.store(
                pl.tile.full(
                    [BATCH, LOCAL_CANDIDATE_GROUPS_PAD],
                    dtype=pl.FP32,
                    value=-3.4028235e38,
                ),
                [0, 0],
                group_values_out,
            )
            pl.store(
                pl.tile.full(
                    [BATCH, LOCAL_CANDIDATE_GROUPS_PAD],
                    dtype=pl.INT32,
                    value=0,
                ),
                [0, 0],
                group_chunks_out,
            )

            for group_idx in pl.parallel(LOCAL_CANDIDATE_GROUPS):
                chunk_base = group_idx * LOCAL_CANDIDATE_GROUP
                values_tile = pl.load(
                    chunk_values,
                    [0, chunk_base],
                    [BATCH, LOCAL_CANDIDATE_GROUP],
                )
                argmax_tmp = pl.tile.create(
                    [BATCH, LOCAL_CANDIDATE_GROUP],
                    dtype=pl.FP32,
                    target_memory=pl.Mem.Vec,
                )
                best_in_group = pl.tile.row_argmax(
                    values_tile, argmax_tmp
                )
                group_value = pl.tile.row_max(
                    values_tile, argmax_tmp
                )
                group_value_row = pl.reshape(
                    group_value,
                    [1, BATCH],
                )
                best_in_group_row = pl.reshape(
                    best_in_group,
                    [1, BATCH],
                )
                for b in pl.range(BATCH):
                    pl.write(
                        group_values_out,
                        [b, group_idx],
                        pl.read(group_value_row, [0, b]),
                    )
                    pl.write(
                        group_chunks_out,
                        [b, group_idx],
                        pl.read(best_in_group_row, [0, b]),
                    )
            return group_values_out, group_chunks_out

        @pl.function(type=pl.FunctionType.InCore)
        def local_vocab_best(
            self,
            group_values: pl.Tensor[
                [BATCH, LOCAL_CANDIDATE_GROUPS_PAD], pl.FP32
            ],
            group_chunks: pl.Tensor[
                [BATCH, LOCAL_CANDIDATE_GROUPS_PAD], pl.INT32
            ],
            chunk_local_ids: pl.Tensor[
                [BATCH, LOCAL_VOCAB_CHUNKS_PAD], pl.INT32
            ],
            candidate_value_out: pl.Out[
                pl.Tensor[[BATCH, tp_size], pl.FP32]
            ],
            candidate_id_out: pl.Out[
                pl.Tensor[[BATCH, tp_size], pl.INT32]
            ],
            my_rank: pl.Scalar[pl.INT32],
        ):
            """Reduce to one candidate in this rank's aligned matrix column.

            A row-major ``[BATCH, 1]`` FP32/INT32 tile has only four bytes per
            row and cannot be loaded by PTOAS' ND TLOAD path. Use the complete
            ``[BATCH, TP]`` matrix instead (16 * 8 * 4 == 512 bytes): this rank
            writes only column ``my_rank`` and leaves every other value/id
            column at ``-inf``/zero.
            """
            pl.store(
                pl.tile.full(
                    [BATCH, tp_size],
                    dtype=pl.FP32,
                    value=-3.4028235e38,
                ),
                [0, 0],
                candidate_value_out,
            )
            pl.store(
                pl.tile.full(
                    [BATCH, tp_size],
                    dtype=pl.INT32,
                    value=0,
                ),
                [0, 0],
                candidate_id_out,
            )
            values_tile = pl.load(
                group_values,
                [0, 0],
                [BATCH, LOCAL_CANDIDATE_GROUPS_PAD],
            )
            argmax_tmp = pl.tile.create(
                [BATCH, LOCAL_CANDIDATE_GROUPS_PAD],
                dtype=pl.FP32,
                target_memory=pl.Mem.Vec,
            )
            best_group = pl.tile.row_argmax(
                values_tile, argmax_tmp
            )
            best_value = pl.tile.row_max(
                values_tile, argmax_tmp
            )
            # Do not scalar-read the [BATCH,1] row_argmax tile directly.
            # Flatten it to one aligned row first; the direct [b,0] read
            # mis-addresses rows > 0 on the current ColMajor lowering, while
            # [1,BATCH] has a 64-byte row and is a valid row-major tile.
            best_group_row = pl.reshape(best_group, [1, BATCH])
            best_value_row = pl.reshape(best_value, [1, BATCH])
            for b in pl.range(BATCH):
                group_idx = pl.cast(
                    pl.read(best_group_row, [0, b]),
                    pl.INDEX,
                )
                candidate_value = pl.read(
                    best_value_row,
                    [0, b],
                )
                pl.write(
                    candidate_value_out,
                    [b, pl.cast(my_rank, pl.INDEX)],
                    candidate_value,
                )
                chunk_in_group = pl.read(
                    group_chunks,
                    [b, group_idx],
                )
                local_chunk_idx = (
                    group_idx * LOCAL_CANDIDATE_GROUP
                    + chunk_in_group
                )
                token_in_chunk = pl.read(
                    chunk_local_ids,
                    [b, pl.cast(local_chunk_idx, pl.INDEX)],
                )
                global_token_id = (
                    pl.cast(my_rank * VOCAB_LOCAL, pl.INT32)
                    + pl.cast(
                        local_chunk_idx * VOCAB_CHUNK,
                        pl.INT32,
                    )
                    + pl.cast(token_in_chunk, pl.INT32)
                )
                pl.write(
                    candidate_id_out,
                    [b, pl.cast(my_rank, pl.INDEX)],
                    global_token_id,
                )
            return candidate_value_out, candidate_id_out

        @pl.function(type=pl.FunctionType.InCore)
        def gather_vocab_candidates(
            self,
            local_candidate_value: pl.Tensor[
                [BATCH, tp_size], pl.FP32
            ],
            local_candidate_id: pl.Tensor[
                [BATCH, tp_size], pl.INT32
            ],
            candidate_values_out: pl.Out[
                pl.Tensor[[BATCH, tp_size], pl.FP32]
            ],
            candidate_ids_out: pl.Out[
                pl.Tensor[[BATCH, tp_size], pl.INT32]
            ],
            candidate_value_window: pld.DistributedTensor[
                [BATCH, tp_size], pl.FP32
            ],
            candidate_id_window: pld.DistributedTensor[
                [BATCH, tp_size], pl.INT32
            ],
            ready_signal_window: pld.DistributedTensor[
                [tp_size, 1], pl.INT32
            ],
            my_rank: pl.Scalar[pl.INT32],
        ):
            """Gather aligned sparse rank-column candidate matrices.

            Every source matrix has a value/id only in its own rank column.
            Elementwise max merges the value matrices; integer add merges the
            disjoint id columns. All local/remote transfers are full
            ``[16,8]`` 512-byte ND tiles, avoiding unsupported ``[16,1]``
            column-major loads.
            """
            local_value_tile = pl.load(
                local_candidate_value, [0, 0], [BATCH, tp_size]
            )
            global_idx_tile = pl.load(
                local_candidate_id, [0, 0], [BATCH, tp_size]
            )
            pl.store(
                local_value_tile, [0, 0], candidate_value_window
            )
            pl.store(global_idx_tile, [0, 0], candidate_id_window)

            for peer in pl.range(tp_size):
                if peer != my_rank:
                    pld.system.notify(
                        target=ready_signal_window,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for src in pl.range(tp_size):
                if src != my_rank:
                    pld.system.wait(
                        signal=ready_signal_window,
                        offsets=[src, 0],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )

            own_value_matrix = pl.load(
                candidate_value_window, [0, 0], [BATCH, tp_size]
            )
            own_id_matrix = pl.load(
                candidate_id_window, [0, 0], [BATCH, tp_size]
            )
            gathered_values = own_value_matrix
            gathered_ids = own_id_matrix
            for peer in pl.range(tp_size):
                if peer != my_rank:
                    remote_value_matrix = pld.tile.remote_load(
                        candidate_value_window,
                        peer=peer,
                        offsets=[0, 0],
                        shape=[BATCH, tp_size],
                    )
                    remote_id_matrix = pld.tile.remote_load(
                        candidate_id_window,
                        peer=peer,
                        offsets=[0, 0],
                        shape=[BATCH, tp_size],
                    )
                    gathered_values = pl.maximum(
                        gathered_values,
                        remote_value_matrix,
                    )
                    gathered_ids = pl.add(
                        gathered_ids,
                        remote_id_matrix,
                    )
            pl.store(gathered_values, [0, 0], candidate_values_out)
            pl.store(gathered_ids, [0, 0], candidate_ids_out)
            return candidate_values_out, candidate_ids_out

        @pl.function(type=pl.FunctionType.InCore)
        def select_global_candidate(
            self,
            candidate_values: pl.Tensor[[BATCH, tp_size], pl.FP32],
            candidate_ids: pl.Tensor[[BATCH, tp_size], pl.INT32],
            token_ids_out: pl.Out[pl.Tensor[[BATCH], pl.INT32]],
        ) -> pl.Tensor[[BATCH], pl.INT32]:
            """Select the full-vocabulary winner after candidate communication."""
            values_tile = pl.load(
                candidate_values, [0, 0], [BATCH, tp_size]
            )
            argmax_tmp = pl.tile.create(
                [BATCH, tp_size],
                dtype=pl.FP32,
                target_memory=pl.Mem.Vec,
            )
            winning_rank = pl.tile.row_argmax(
                values_tile, argmax_tmp
            )
            winning_rank_row = pl.reshape(
                winning_rank,
                [1, BATCH],
            )
            for b in pl.range(BATCH):
                winner = pl.read(winning_rank_row, [0, b])
                winning_id = pl.read(
                    candidate_ids,
                    [b, pl.cast(winner, pl.INDEX)],
                )
                pl.write(token_ids_out, [b], winning_id)
            return token_ids_out

        @pl.function(type=pl.FunctionType.Orchestration)
        def mtp_layer_orch(  # noqa: PLR0913, PLR0915
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
            k_cache: pl.Tensor[[MTP_CACHE_ROWS, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[MTP_CACHE_ROWS, HEAD_DIM], pl.BF16],
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
            shared_head_norm_weight: pl.Tensor[
                [NUM_MTP, HIDDEN], pl.FP32
            ],
            shared_head_output_weight: pl.Tensor[
                [MTP_VOCAB_ROWS, HIDDEN], pl.BF16
            ],
            hidden_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            logits_out: pl.Out[
                pl.Tensor[[USER_BATCH_DYN, VOCAB_LOCAL], pl.FP32]
            ],
            token_ids_out: pl.Out[pl.Tensor[[BATCH], pl.INT32]],
            eh_tmp_window: pld.DistributedTensor[
                [BATCH, HIDDEN], pl.BF16
            ],
            eh_signal_window: pld.DistributedTensor[
                [tp_size, 1], pl.INT32
            ],
            attn_tmp_window: pld.DistributedTensor[
                [BATCH, HIDDEN], pl.BF16
            ],
            attn_signal_window: pld.DistributedTensor[
                [tp_size, 1], pl.INT32
            ],
            mlp_tmp_window: pld.DistributedTensor[
                [BATCH, HIDDEN], pl.BF16
            ],
            mlp_signal_window: pld.DistributedTensor[
                [tp_size, 1], pl.INT32
            ],
            candidate_value_window: pld.DistributedTensor[
                [BATCH, tp_size], pl.FP32
            ],
            candidate_id_window: pld.DistributedTensor[
                [BATCH, tp_size], pl.INT32
            ],
            ready_signal_window: pld.DistributedTensor[
                [tp_size, 1], pl.INT32
            ],
            layer_idx: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ):
            embed_next = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            embed_next = self.embedding_lookup(
                input_token_ids,
                active_mask,
                seq_lens,
                embed_weight,
                embed_next,
            )
            masked_previous_hidden = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
            masked_previous_hidden = self.mask_previous_hidden(
                previous_hidden,
                active_mask,
                masked_previous_hidden,
            )

            mtp_in = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            mtp_in = mtp_input_proj_inline(
                masked_previous_hidden,
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
            resid1 = attention_swa_inline(
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

            raw_hidden = pl.create_tensor(
                [BATCH, HIDDEN], dtype=pl.BF16
            )
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
            # Attention still executes over the fixed BATCH=16 ABI.  Padding
            # metadata must use conflict-free cache slots, and this boundary
            # mask guarantees that no value computed for an inactive row is
            # handed to the shared head or the next MTP layer.
            hidden_out = self.mask_previous_hidden(
                raw_hidden,
                active_mask,
                hidden_out,
            )

            logits_out = mtp_shared_head_inline(
                hidden_out,
                shared_head_norm_weight,
                shared_head_output_weight,
                seq_lens,
                logits_out,
                layer_idx,
            )

            local_chunk_values = pl.create_tensor(
                [BATCH, LOCAL_VOCAB_CHUNKS_PAD], dtype=pl.FP32
            )
            local_chunk_ids = pl.create_tensor(
                [BATCH, LOCAL_VOCAB_CHUNKS_PAD], dtype=pl.INT32
            )
            local_chunk_values, local_chunk_ids = (
                self.local_vocab_chunk_candidates(
                    logits_out,
                    local_chunk_values,
                    local_chunk_ids,
                    my_rank,
                )
            )

            local_group_values = pl.create_tensor(
                [BATCH, LOCAL_CANDIDATE_GROUPS_PAD],
                dtype=pl.FP32,
            )
            local_group_ids = pl.create_tensor(
                [BATCH, LOCAL_CANDIDATE_GROUPS_PAD],
                dtype=pl.INT32,
            )
            local_group_values, local_group_ids = (
                self.local_vocab_group_candidates(
                    local_chunk_values,
                    local_chunk_ids,
                    local_group_values,
                    local_group_ids,
                )
            )

            local_candidate_value = pl.create_tensor(
                [BATCH, tp_size], dtype=pl.FP32
            )
            local_candidate_id = pl.create_tensor(
                [BATCH, tp_size], dtype=pl.INT32
            )
            local_candidate_value, local_candidate_id = (
                self.local_vocab_best(
                    local_group_values,
                    local_group_ids,
                    local_chunk_ids,
                    local_candidate_value,
                    local_candidate_id,
                    my_rank,
                )
            )
            candidate_values = pl.create_tensor(
                [BATCH, tp_size], dtype=pl.FP32
            )
            candidate_ids = pl.create_tensor(
                [BATCH, tp_size], dtype=pl.INT32
            )
            candidate_values, candidate_ids = (
                self.gather_vocab_candidates(
                    local_candidate_value,
                    local_candidate_id,
                    candidate_values,
                    candidate_ids,
                    candidate_value_window,
                    candidate_id_window,
                    ready_signal_window,
                    my_rank,
                )
            )
            token_ids_out = self.select_global_candidate(
                candidate_values,
                candidate_ids,
                token_ids_out,
            )
            return hidden_out, logits_out, token_ids_out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(  # noqa: PLR0913, PLR0915
            self,
            previous_hidden: pl.Tensor[
                [tp_size, BATCH, HIDDEN], pl.BF16
            ],
            first_token_ids: pl.Tensor[[tp_size, BATCH], pl.INT32],
            active_mask: pl.Tensor[[tp_size, BATCH], pl.INT32],
            embed_weight: pl.Tensor[
                [tp_size, VOCAB, HIDDEN], pl.BF16
            ],
            enorm_weight: pl.Tensor[
                [tp_size, NUM_MTP, HIDDEN], pl.FP32
            ],
            hnorm_weight: pl.Tensor[
                [tp_size, NUM_MTP, HIDDEN], pl.FP32
            ],
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
                [
                    tp_size,
                    NUM_HEADS_SWA_LOCAL_PAD,
                    HIDDEN_Q_SWA_LOCAL,
                ],
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
            shared_head_norm_weight: pl.Tensor[
                [tp_size, NUM_MTP, HIDDEN], pl.FP32
            ],
            shared_head_output_weight: pl.Tensor[
                [tp_size, MTP_VOCAB_ROWS, HIDDEN], pl.BF16
            ],
            seq_lens: pl.Tensor[
                [tp_size, USER_BATCH_DYN], pl.INT32
            ],
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
            k_cache: pl.Tensor[
                [tp_size, MTP_CACHE_ROWS, HEAD_DIM], pl.BF16
            ],
            v_cache: pl.Tensor[
                [tp_size, MTP_CACHE_ROWS, HEAD_DIM], pl.BF16
            ],
            hidden_out: pl.Out[
                pl.Tensor[
                    [tp_size, NUM_MTP, BATCH, HIDDEN], pl.BF16
                ]
            ],
            logits_out: pl.Out[
                pl.Tensor[
                    [tp_size, NUM_MTP, USER_BATCH_DYN, VOCAB_LOCAL],
                    pl.FP32,
                ]
            ],
            draft_token_ids_out: pl.Out[
                pl.Tensor[[tp_size, NUM_MTP, BATCH], pl.INT32]
            ],
        ):
            # Every collective call-site owns a distinct scratch/signal pair.
            l0_eh_tmp = pld.alloc_window_buffer(BATCH * HIDDEN * 2)
            l0_eh_sig = pld.alloc_window_buffer(COMM_CONTROL_SIGNAL_BYTES)
            l0_attn_tmp = pld.alloc_window_buffer(BATCH * HIDDEN * 2)
            l0_attn_sig = pld.alloc_window_buffer(
                COMM_CONTROL_SIGNAL_BYTES
            )
            l0_mlp_tmp = pld.alloc_window_buffer(BATCH * HIDDEN * 2)
            l0_mlp_sig = pld.alloc_window_buffer(COMM_CONTROL_SIGNAL_BYTES)
            l0_candidate_value = pld.alloc_window_buffer(
                COMM_SCALAR_DATA_BYTES
            )
            l0_candidate_id = pld.alloc_window_buffer(
                COMM_SCALAR_DATA_BYTES
            )
            l0_candidate_sig = pld.alloc_window_buffer(
                COMM_CONTROL_SIGNAL_BYTES
            )

            l1_eh_tmp = pld.alloc_window_buffer(BATCH * HIDDEN * 2)
            l1_eh_sig = pld.alloc_window_buffer(COMM_CONTROL_SIGNAL_BYTES)
            l1_attn_tmp = pld.alloc_window_buffer(BATCH * HIDDEN * 2)
            l1_attn_sig = pld.alloc_window_buffer(
                COMM_CONTROL_SIGNAL_BYTES
            )
            l1_mlp_tmp = pld.alloc_window_buffer(BATCH * HIDDEN * 2)
            l1_mlp_sig = pld.alloc_window_buffer(COMM_CONTROL_SIGNAL_BYTES)
            l1_candidate_value = pld.alloc_window_buffer(
                COMM_SCALAR_DATA_BYTES
            )
            l1_candidate_id = pld.alloc_window_buffer(
                COMM_SCALAR_DATA_BYTES
            )
            l1_candidate_sig = pld.alloc_window_buffer(
                COMM_CONTROL_SIGNAL_BYTES
            )

            l2_eh_tmp = pld.alloc_window_buffer(BATCH * HIDDEN * 2)
            l2_eh_sig = pld.alloc_window_buffer(COMM_CONTROL_SIGNAL_BYTES)
            l2_attn_tmp = pld.alloc_window_buffer(BATCH * HIDDEN * 2)
            l2_attn_sig = pld.alloc_window_buffer(
                COMM_CONTROL_SIGNAL_BYTES
            )
            l2_mlp_tmp = pld.alloc_window_buffer(BATCH * HIDDEN * 2)
            l2_mlp_sig = pld.alloc_window_buffer(COMM_CONTROL_SIGNAL_BYTES)
            l2_candidate_value = pld.alloc_window_buffer(
                COMM_SCALAR_DATA_BYTES
            )
            l2_candidate_id = pld.alloc_window_buffer(
                COMM_SCALAR_DATA_BYTES
            )
            l2_candidate_sig = pld.alloc_window_buffer(
                COMM_CONTROL_SIGNAL_BYTES
            )

            for r0 in pl.range(pld.world_size()):
                self.mtp_layer_orch(
                    previous_hidden[r0],
                    first_token_ids[r0],
                    active_mask[r0],
                    embed_weight[r0],
                    enorm_weight[r0],
                    hnorm_weight[r0],
                    eh_proj_weight[r0],
                    input_rms_weight[r0],
                    wq[r0],
                    wk[r0],
                    wv[r0],
                    q_norm_weight[r0],
                    k_norm_weight[r0],
                    seq_lens[r0],
                    block_table[r0],
                    slot_mapping[r0],
                    rope_cos[r0],
                    rope_sin[r0],
                    k_cache[r0],
                    v_cache[r0],
                    wo[r0],
                    w_g[r0],
                    gate_r[r0],
                    post_rms_weight[r0],
                    w_gate[r0],
                    w_up[r0],
                    w_down[r0],
                    shared_head_norm_weight[r0],
                    shared_head_output_weight[r0],
                    hidden_out[r0, 0],
                    logits_out[r0, 0],
                    draft_token_ids_out[r0, 0],
                    pld.window(
                        l0_eh_tmp, [BATCH, HIDDEN], dtype=pl.BF16
                    ),
                    pld.window(
                        l0_eh_sig, [tp_size, 1], dtype=pl.INT32
                    ),
                    pld.window(
                        l0_attn_tmp, [BATCH, HIDDEN], dtype=pl.BF16
                    ),
                    pld.window(
                        l0_attn_sig, [tp_size, 1], dtype=pl.INT32
                    ),
                    pld.window(
                        l0_mlp_tmp, [BATCH, HIDDEN], dtype=pl.BF16
                    ),
                    pld.window(
                        l0_mlp_sig, [tp_size, 1], dtype=pl.INT32
                    ),
                    pld.window(
                        l0_candidate_value,
                        [BATCH, tp_size],
                        dtype=pl.FP32,
                    ),
                    pld.window(
                        l0_candidate_id,
                        [BATCH, tp_size],
                        dtype=pl.INT32,
                    ),
                    pld.window(
                        l0_candidate_sig, [tp_size, 1], dtype=pl.INT32
                    ),
                    0,
                    r0,
                    device=r0,
                )

            for r1 in pl.range(pld.world_size()):
                self.mtp_layer_orch(
                    hidden_out[r1, 0],
                    draft_token_ids_out[r1, 0],
                    active_mask[r1],
                    embed_weight[r1],
                    enorm_weight[r1],
                    hnorm_weight[r1],
                    eh_proj_weight[r1],
                    input_rms_weight[r1],
                    wq[r1],
                    wk[r1],
                    wv[r1],
                    q_norm_weight[r1],
                    k_norm_weight[r1],
                    seq_lens[r1],
                    block_table[r1],
                    slot_mapping[r1],
                    rope_cos[r1],
                    rope_sin[r1],
                    k_cache[r1],
                    v_cache[r1],
                    wo[r1],
                    w_g[r1],
                    gate_r[r1],
                    post_rms_weight[r1],
                    w_gate[r1],
                    w_up[r1],
                    w_down[r1],
                    shared_head_norm_weight[r1],
                    shared_head_output_weight[r1],
                    hidden_out[r1, 1],
                    logits_out[r1, 1],
                    draft_token_ids_out[r1, 1],
                    pld.window(
                        l1_eh_tmp, [BATCH, HIDDEN], dtype=pl.BF16
                    ),
                    pld.window(
                        l1_eh_sig, [tp_size, 1], dtype=pl.INT32
                    ),
                    pld.window(
                        l1_attn_tmp, [BATCH, HIDDEN], dtype=pl.BF16
                    ),
                    pld.window(
                        l1_attn_sig, [tp_size, 1], dtype=pl.INT32
                    ),
                    pld.window(
                        l1_mlp_tmp, [BATCH, HIDDEN], dtype=pl.BF16
                    ),
                    pld.window(
                        l1_mlp_sig, [tp_size, 1], dtype=pl.INT32
                    ),
                    pld.window(
                        l1_candidate_value,
                        [BATCH, tp_size],
                        dtype=pl.FP32,
                    ),
                    pld.window(
                        l1_candidate_id,
                        [BATCH, tp_size],
                        dtype=pl.INT32,
                    ),
                    pld.window(
                        l1_candidate_sig, [tp_size, 1], dtype=pl.INT32
                    ),
                    1,
                    r1,
                    device=r1,
                )

            for r2 in pl.range(pld.world_size()):
                self.mtp_layer_orch(
                    hidden_out[r2, 1],
                    draft_token_ids_out[r2, 1],
                    active_mask[r2],
                    embed_weight[r2],
                    enorm_weight[r2],
                    hnorm_weight[r2],
                    eh_proj_weight[r2],
                    input_rms_weight[r2],
                    wq[r2],
                    wk[r2],
                    wv[r2],
                    q_norm_weight[r2],
                    k_norm_weight[r2],
                    seq_lens[r2],
                    block_table[r2],
                    slot_mapping[r2],
                    rope_cos[r2],
                    rope_sin[r2],
                    k_cache[r2],
                    v_cache[r2],
                    wo[r2],
                    w_g[r2],
                    gate_r[r2],
                    post_rms_weight[r2],
                    w_gate[r2],
                    w_up[r2],
                    w_down[r2],
                    shared_head_norm_weight[r2],
                    shared_head_output_weight[r2],
                    hidden_out[r2, 2],
                    logits_out[r2, 2],
                    draft_token_ids_out[r2, 2],
                    pld.window(
                        l2_eh_tmp, [BATCH, HIDDEN], dtype=pl.BF16
                    ),
                    pld.window(
                        l2_eh_sig, [tp_size, 1], dtype=pl.INT32
                    ),
                    pld.window(
                        l2_attn_tmp, [BATCH, HIDDEN], dtype=pl.BF16
                    ),
                    pld.window(
                        l2_attn_sig, [tp_size, 1], dtype=pl.INT32
                    ),
                    pld.window(
                        l2_mlp_tmp, [BATCH, HIDDEN], dtype=pl.BF16
                    ),
                    pld.window(
                        l2_mlp_sig, [tp_size, 1], dtype=pl.INT32
                    ),
                    pld.window(
                        l2_candidate_value,
                        [BATCH, tp_size],
                        dtype=pl.FP32,
                    ),
                    pld.window(
                        l2_candidate_id,
                        [BATCH, tp_size],
                        dtype=pl.INT32,
                    ),
                    pld.window(
                        l2_candidate_sig, [tp_size, 1], dtype=pl.INT32
                    ),
                    2,
                    r2,
                    device=r2,
                )

    return WholeMtp3


whole_mtp3 = _build_whole_mtp3_program()


__all__ = [
    "NUM_MTP",
    "MTP_CACHE_ROWS",
    "whole_mtp3",
    "_build_whole_mtp3_program",
]
