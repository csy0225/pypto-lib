# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Instrumented L0-L4 program for exact Step3p5 MoE route metadata.

The focused graph keeps exactly the first five canonical decode layers:

* L0: full attention + dense MLP
* L1: sliding-window attention + dense MLP
* L2: sliding-window attention + dense MLP
* L3: sliding-window attention + MoE
* L4: full attention + MoE

This diagnostic-only graph snapshots the reused distributed ``moe_recv_meta``
window after L3 and again after L4.  Each snapshot also copies the matching
hidden state.  L4 consumes the copied L3 hidden state, creating a real
producer-consumer fence that prevents L4 from overwriting metadata before the
L3 snapshot completes.

The compute bodies are not copied.  This module composes the canonical IR
functions from ``models.step3p5.decode_fwd`` with diagnostic orchestration and
one local snapshot kernel.  The formal normal/DFX program remains unchanged.
"""
from __future__ import annotations

from pypto import ir
import pypto.language as pl
import pypto.language.distributed as pld

import models.step3p5.decode_fwd as _canonical


# Re-export the canonical shape/config globals used by the focused signatures.
BATCH = _canonical.BATCH
HIDDEN = _canonical.HIDDEN
HEAD_DIM = _canonical.HEAD_DIM
LAYER_DYN = _canonical.LAYER_DYN
USER_BATCH_DYN = _canonical.USER_BATCH_DYN
BLOCK_TABLE_FLAT_DYN = _canonical.BLOCK_TABLE_FLAT_DYN
ROPE_SEQ_DYN = _canonical.ROPE_SEQ_DYN
KV_CACHE_ROWS_DYN = _canonical.KV_CACHE_ROWS_DYN
NUM_TOKENS_RUNTIME = _canonical.NUM_TOKENS_RUNTIME
COMM_CONTROL_SIGNAL_BYTES = _canonical.COMM_CONTROL_SIGNAL_BYTES
COMM_SIGNAL_STRIDE_I32 = _canonical.COMM_SIGNAL_STRIDE_I32

INTER_LOCAL = _canonical.INTER_LOCAL
hidden_q_full = _canonical.hidden_q_full
hidden_q_swa = _canonical.hidden_q_swa
nh_full_pad = _canonical.nh_full_pad
nh_swa_pad = _canonical.nh_swa_pad
KV_HIDDEN_LOCAL_R = _canonical.KV_HIDDEN_LOCAL_R
rotary_dim_full = _canonical.rotary_dim_full
rotary_dim_swa = _canonical.rotary_dim_swa
tp_size = _canonical.tp_size

N_EXPERTS = _canonical.N_EXPERTS
n_ranks = _canonical.n_ranks
n_local_experts = _canonical.n_local_experts
n_local_experts_pad = _canonical.n_local_experts_pad
inter = _canonical.inter
sh_inter_local = _canonical.sh_inter_local
dispatch_lane_rows = _canonical.dispatch_lane_rows
dispatch_aux_pad = _canonical.dispatch_aux_pad
idx_pad = _canonical.idx_pad
n_routes_per_rank = _canonical.n_routes_per_rank

N_FULL_FIVE = 2
N_SWA_FIVE = 3
N_DENSE_FIVE = 3
N_MOE_FIVE = 2
SNAPSHOT_HIDDEN_CHUNK = 512

assert HIDDEN % SNAPSHOT_HIDDEN_CHUNK == 0


_CANONICAL_PROGRAM = _canonical.whole_decode_step3p5

# Bare-name calls to IR Functions are supported by the PyPTO program parser.
# The final Program below also adds the canonical transitive dependency set.
full_chip_orch = _CANONICAL_PROGRAM.get_function("full_chip_orch")
swa_chip_orch = _CANONICAL_PROGRAM.get_function("swa_chip_orch")
full_moe_chip_orch = _CANONICAL_PROGRAM.get_function("full_moe_chip_orch")
swa_moe_chip_orch = _CANONICAL_PROGRAM.get_function("swa_moe_chip_orch")


@pl.program
class FiveLayerMoeRouteInstrumented:
    @pl.function(type=pl.FunctionType.InCore)
    def snapshot_recv_meta_and_hidden(
        self,
        recv_meta: pld.DistributedTensor[
            [n_ranks, n_local_experts_pad], pl.INT32
        ],
        my_rank: pl.Scalar[pl.INT32],
        hidden_in: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        recv_meta_out: pl.Out[
            pl.Tensor[[n_ranks, n_local_experts_pad], pl.INT32]
        ],
        hidden_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
    ) -> tuple[
        pl.Tensor[[n_ranks, n_local_experts_pad], pl.INT32],
        pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    ]:
        meta_tile = pl.load(
            recv_meta,
            [0, 0],
            [n_ranks, n_local_experts_pad],
        )
        # Self-target stores have no peer notification/fence.  Keep that row
        # deterministic and reconstruct it from explicit dispatch counts.
        for expert in pl.range(n_local_experts):
            pl.tile.write(
                meta_tile,
                [my_rank, expert],
                pl.cast(0, pl.INT32),
            )
        for src in pl.range(n_ranks):
            for expert in pl.range(
                n_local_experts, n_local_experts_pad
            ):
                pl.tile.write(meta_tile, [src, expert], pl.cast(0, pl.INT32))
        pl.store(meta_tile, [0, 0], recv_meta_out)
        for k0 in pl.range(0, HIDDEN, SNAPSHOT_HIDDEN_CHUNK):
            hidden_tile = pl.load(
                hidden_in,
                [0, k0],
                [BATCH, SNAPSHOT_HIDDEN_CHUNK],
            )
            pl.store(hidden_tile, [0, k0], hidden_out)
        return recv_meta_out, hidden_out

    @pl.function(type=pl.FunctionType.Orchestration)
    def five_layer_route_chip_orch(  # noqa: PLR0913, PLR0915
        self,
        current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        input_rms: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
        post_rms: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
        q_norm: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
        k_norm: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
        full_wq: pl.Tensor[[N_FULL_FIVE * HIDDEN, hidden_q_full], pl.BF16],
        full_wk: pl.Tensor[
            [N_FULL_FIVE * HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16
        ],
        full_wv: pl.Tensor[
            [N_FULL_FIVE * HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16
        ],
        full_wo: pl.Tensor[
            [N_FULL_FIVE * hidden_q_full, HIDDEN], pl.BF16
        ],
        full_w_g: pl.Tensor[
            [N_FULL_FIVE * HIDDEN, nh_full_pad], pl.BF16
        ],
        full_gate_r: pl.Tensor[
            [N_FULL_FIVE * nh_full_pad, hidden_q_full], pl.BF16
        ],
        swa_wq: pl.Tensor[[N_SWA_FIVE * HIDDEN, hidden_q_swa], pl.BF16],
        swa_wk: pl.Tensor[
            [N_SWA_FIVE * HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16
        ],
        swa_wv: pl.Tensor[
            [N_SWA_FIVE * HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16
        ],
        swa_wo: pl.Tensor[[N_SWA_FIVE * hidden_q_swa, HIDDEN], pl.BF16],
        swa_w_g: pl.Tensor[[N_SWA_FIVE * HIDDEN, nh_swa_pad], pl.BF16],
        swa_gate_r: pl.Tensor[
            [N_SWA_FIVE * nh_swa_pad, hidden_q_swa], pl.BF16
        ],
        dense_w_gate: pl.Tensor[
            [N_DENSE_FIVE * HIDDEN, INTER_LOCAL], pl.BF16
        ],
        dense_w_up: pl.Tensor[
            [N_DENSE_FIVE * HIDDEN, INTER_LOCAL], pl.BF16
        ],
        dense_w_down: pl.Tensor[
            [N_DENSE_FIVE * INTER_LOCAL, HIDDEN], pl.BF16
        ],
        moe_gate_w: pl.Tensor[
            [N_MOE_FIVE * N_EXPERTS, HIDDEN], pl.FP32
        ],
        moe_router_bias: pl.Tensor[
            [N_MOE_FIVE * N_EXPERTS], pl.FP32
        ],
        moe_w_gate_r: pl.Tensor[
            [N_MOE_FIVE * n_local_experts * HIDDEN, inter], pl.INT8
        ],
        moe_w_gate_r_scale: pl.Tensor[
            [N_MOE_FIVE * n_local_experts, inter], pl.FP32
        ],
        moe_w_up_r: pl.Tensor[
            [N_MOE_FIVE * n_local_experts * HIDDEN, inter], pl.INT8
        ],
        moe_w_up_r_scale: pl.Tensor[
            [N_MOE_FIVE * n_local_experts, inter], pl.FP32
        ],
        moe_w_down_r: pl.Tensor[
            [N_MOE_FIVE * n_local_experts * inter, HIDDEN], pl.INT8
        ],
        moe_w_down_r_scale: pl.Tensor[
            [N_MOE_FIVE * n_local_experts, HIDDEN], pl.FP32
        ],
        moe_w_gate_s: pl.Tensor[
            [N_MOE_FIVE * sh_inter_local, HIDDEN], pl.BF16
        ],
        moe_w_up_s: pl.Tensor[
            [N_MOE_FIVE * sh_inter_local, HIDDEN], pl.BF16
        ],
        moe_w_down_s: pl.Tensor[
            [N_MOE_FIVE * sh_inter_local, HIDDEN], pl.BF16
        ],
        seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
        block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
        slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
        rope_cos_full: pl.Tensor[
            [ROPE_SEQ_DYN, rotary_dim_full], pl.FP32
        ],
        rope_sin_full: pl.Tensor[
            [ROPE_SEQ_DYN, rotary_dim_full], pl.FP32
        ],
        rope_cos_swa: pl.Tensor[
            [ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32
        ],
        rope_sin_swa: pl.Tensor[
            [ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32
        ],
        k_cache: pl.InOut[
            pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16]
        ],
        v_cache: pl.InOut[
            pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16]
        ],
        hidden_l3: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
        hidden_l4: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
        local_expert_count_l3: pl.Out[
            pl.Tensor[[n_local_experts], pl.INT32]
        ],
        local_expert_count_l4: pl.Out[
            pl.Tensor[[n_local_experts], pl.INT32]
        ],
        recv_meta_l3: pl.Out[
            pl.Tensor[[n_ranks, n_local_experts_pad], pl.INT32]
        ],
        recv_meta_l4: pl.Out[
            pl.Tensor[[n_ranks, n_local_experts_pad], pl.INT32]
        ],
        dense_attn_tmp_stack: pld.DistributedTensor[
            [N_DENSE_FIVE * BATCH, HIDDEN], pl.BF16
        ],
        dense_attn_signal_stack: pld.DistributedTensor[
            [N_DENSE_FIVE * COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        dense_mlp_tmp_stack: pld.DistributedTensor[
            [N_DENSE_FIVE * BATCH, HIDDEN], pl.BF16
        ],
        dense_mlp_signal_stack: pld.DistributedTensor[
            [N_DENSE_FIVE * COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        moe_attn_tmp_stack: pld.DistributedTensor[
            [N_MOE_FIVE * BATCH, HIDDEN], pl.BF16
        ],
        moe_attn_signal_stack: pld.DistributedTensor[
            [N_MOE_FIVE * COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        moe_recv_meta: pld.DistributedTensor[
            [n_ranks, n_local_experts_pad], pl.INT32
        ],
        moe_meta_arrived: pld.DistributedTensor[
            [COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        moe_recv_x: pld.DistributedTensor[
            [dispatch_lane_rows, HIDDEN], pl.INT8
        ],
        moe_recv_aux: pld.DistributedTensor[
            [dispatch_lane_rows, dispatch_aux_pad], pl.FP32
        ],
        moe_recv_route: pld.DistributedTensor[
            [dispatch_lane_rows, idx_pad], pl.INT32
        ],
        moe_data_arrived: pld.DistributedTensor[
            [COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        moe_sh_tmp_stack: pld.DistributedTensor[
            [N_MOE_FIVE * BATCH, HIDDEN], pl.BF16
        ],
        moe_sh_signal_stack: pld.DistributedTensor[
            [N_MOE_FIVE * COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        moe_combine_arrived: pld.DistributedTensor[
            [COMM_SIGNAL_STRIDE_I32, 1], pl.INT32
        ],
        moe_routed_y_buf: pld.DistributedTensor[
            [n_routes_per_rank, HIDDEN], pl.BF16
        ],
        num_tokens_per_owner: pl.Tensor[
            [NUM_TOKENS_RUNTIME], pl.INT32
        ],
        my_rank: pl.Scalar[pl.INT32],
    ):
        # Packed-global contract: every TP rank owns the same valid row prefix.
        # The holder rejects heterogeneous owner-local counts before rt.run().
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

        # L0: full attention + dense MLP.
        h0 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        h0 = full_chip_orch(
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
            h0,
            pl.slice(dense_attn_tmp_stack, [BATCH, HIDDEN], [0, 0]),
            pl.slice(
                dense_attn_signal_stack,
                [COMM_SIGNAL_STRIDE_I32, 1],
                [0, 0],
            ),
            pl.slice(dense_mlp_tmp_stack, [BATCH, HIDDEN], [0, 0]),
            pl.slice(
                dense_mlp_signal_stack,
                [COMM_SIGNAL_STRIDE_I32, 1],
                [0, 0],
            ),
            0,
            0,
            0,
            num_tokens,
            my_rank,
        )

        # L1: SWA + dense MLP.
        h1 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        h1 = swa_chip_orch(
            h0,
            input_rms,
            pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [0, 0]),
            pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL_R], [0, 0]),
            pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL_R], [0, 0]),
            q_norm,
            k_norm,
            seq_lens,
            block_table,
            slot_mapping,
            rope_cos_swa,
            rope_sin_swa,
            k_cache,
            v_cache,
            pl.slice(swa_wo, [hidden_q_swa, HIDDEN], [0, 0]),
            pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [0, 0]),
            pl.slice(swa_gate_r, [nh_swa_pad, hidden_q_swa], [0, 0]),
            post_rms,
            pl.slice(dense_w_gate, [HIDDEN, INTER_LOCAL], [HIDDEN, 0]),
            pl.slice(dense_w_up, [HIDDEN, INTER_LOCAL], [HIDDEN, 0]),
            pl.slice(
                dense_w_down,
                [INTER_LOCAL, HIDDEN],
                [INTER_LOCAL, 0],
            ),
            h1,
            pl.slice(dense_attn_tmp_stack, [BATCH, HIDDEN], [BATCH, 0]),
            pl.slice(
                dense_attn_signal_stack,
                [COMM_SIGNAL_STRIDE_I32, 1],
                [COMM_SIGNAL_STRIDE_I32, 0],
            ),
            pl.slice(dense_mlp_tmp_stack, [BATCH, HIDDEN], [BATCH, 0]),
            pl.slice(
                dense_mlp_signal_stack,
                [COMM_SIGNAL_STRIDE_I32, 1],
                [COMM_SIGNAL_STRIDE_I32, 0],
            ),
            1,
            0,
            0,
            num_tokens,
            my_rank,
        )

        # L2: SWA + dense MLP.
        h2 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        h2 = swa_chip_orch(
            h1,
            input_rms,
            pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [HIDDEN, 0]),
            pl.slice(
                swa_wk,
                [HIDDEN, KV_HIDDEN_LOCAL_R],
                [HIDDEN, 0],
            ),
            pl.slice(
                swa_wv,
                [HIDDEN, KV_HIDDEN_LOCAL_R],
                [HIDDEN, 0],
            ),
            q_norm,
            k_norm,
            seq_lens,
            block_table,
            slot_mapping,
            rope_cos_swa,
            rope_sin_swa,
            k_cache,
            v_cache,
            pl.slice(
                swa_wo,
                [hidden_q_swa, HIDDEN],
                [hidden_q_swa, 0],
            ),
            pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [HIDDEN, 0]),
            pl.slice(
                swa_gate_r,
                [nh_swa_pad, hidden_q_swa],
                [nh_swa_pad, 0],
            ),
            post_rms,
            pl.slice(
                dense_w_gate,
                [HIDDEN, INTER_LOCAL],
                [2 * HIDDEN, 0],
            ),
            pl.slice(
                dense_w_up,
                [HIDDEN, INTER_LOCAL],
                [2 * HIDDEN, 0],
            ),
            pl.slice(
                dense_w_down,
                [INTER_LOCAL, HIDDEN],
                [2 * INTER_LOCAL, 0],
            ),
            h2,
            pl.slice(
                dense_attn_tmp_stack,
                [BATCH, HIDDEN],
                [2 * BATCH, 0],
            ),
            pl.slice(
                dense_attn_signal_stack,
                [COMM_SIGNAL_STRIDE_I32, 1],
                [2 * COMM_SIGNAL_STRIDE_I32, 0],
            ),
            pl.slice(
                dense_mlp_tmp_stack,
                [BATCH, HIDDEN],
                [2 * BATCH, 0],
            ),
            pl.slice(
                dense_mlp_signal_stack,
                [COMM_SIGNAL_STRIDE_I32, 1],
                [2 * COMM_SIGNAL_STRIDE_I32, 0],
            ),
            2,
            0,
            0,
            num_tokens,
            my_rank,
        )

        # L3: SWA + MoE, epoch 1. Snapshot before L4 reuses moe_recv_meta.
        resid_l3 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        hidden_l3_raw = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        hidden_l3_raw = swa_moe_chip_orch(
            h2,
            input_rms,
            pl.slice(swa_wq, [HIDDEN, hidden_q_swa], [2 * HIDDEN, 0]),
            pl.slice(
                swa_wk,
                [HIDDEN, KV_HIDDEN_LOCAL_R],
                [2 * HIDDEN, 0],
            ),
            pl.slice(
                swa_wv,
                [HIDDEN, KV_HIDDEN_LOCAL_R],
                [2 * HIDDEN, 0],
            ),
            q_norm,
            k_norm,
            seq_lens,
            block_table,
            slot_mapping,
            rope_cos_swa,
            rope_sin_swa,
            k_cache,
            v_cache,
            pl.slice(
                swa_wo,
                [hidden_q_swa, HIDDEN],
                [2 * hidden_q_swa, 0],
            ),
            pl.slice(swa_w_g, [HIDDEN, nh_swa_pad], [2 * HIDDEN, 0]),
            pl.slice(
                swa_gate_r,
                [nh_swa_pad, hidden_q_swa],
                [2 * nh_swa_pad, 0],
            ),
            post_rms,
            pl.slice(moe_gate_w, [N_EXPERTS, HIDDEN], [0, 0]),
            pl.slice(moe_router_bias, [N_EXPERTS], [0]),
            pl.reshape(
                pl.slice(
                    moe_w_gate_r,
                    [n_local_experts * HIDDEN, inter],
                    [0, 0],
                ),
                [n_local_experts, HIDDEN, inter],
            ),
            pl.slice(
                moe_w_gate_r_scale,
                [n_local_experts, inter],
                [0, 0],
            ),
            pl.reshape(
                pl.slice(
                    moe_w_up_r,
                    [n_local_experts * HIDDEN, inter],
                    [0, 0],
                ),
                [n_local_experts, HIDDEN, inter],
            ),
            pl.slice(
                moe_w_up_r_scale,
                [n_local_experts, inter],
                [0, 0],
            ),
            pl.reshape(
                pl.slice(
                    moe_w_down_r,
                    [n_local_experts * inter, HIDDEN],
                    [0, 0],
                ),
                [n_local_experts, inter, HIDDEN],
            ),
            pl.slice(
                moe_w_down_r_scale,
                [n_local_experts, HIDDEN],
                [0, 0],
            ),
            pl.slice(moe_w_gate_s, [sh_inter_local, HIDDEN], [0, 0]),
            pl.slice(moe_w_up_s, [sh_inter_local, HIDDEN], [0, 0]),
            pl.slice(moe_w_down_s, [sh_inter_local, HIDDEN], [0, 0]),
            hidden_l3_raw,
            resid_l3,
            local_expert_count_l3,
            pl.slice(moe_attn_tmp_stack, [BATCH, HIDDEN], [0, 0]),
            pl.slice(
                moe_attn_signal_stack,
                [COMM_SIGNAL_STRIDE_I32, 1],
                [0, 0],
            ),
            moe_recv_meta,
            moe_meta_arrived,
            moe_recv_x,
            moe_recv_aux,
            moe_recv_route,
            moe_data_arrived,
            pl.slice(moe_sh_tmp_stack, [BATCH, HIDDEN], [0, 0]),
            pl.slice(
                moe_sh_signal_stack,
                [COMM_SIGNAL_STRIDE_I32, 1],
                [0, 0],
            ),
            moe_combine_arrived,
            moe_routed_y_buf,
            3,
            0,
            num_tokens,
            my_rank,
            1,
        )
        recv_meta_l3, hidden_l3 = self.snapshot_recv_meta_and_hidden(
            moe_recv_meta,
            my_rank,
            hidden_l3_raw,
            recv_meta_l3,
            hidden_l3,
        )

        # L4 consumes the fenced, bit-identical L3 snapshot.
        resid_l4 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        hidden_l4_raw = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        hidden_l4_raw = full_moe_chip_orch(
            hidden_l3,
            input_rms,
            pl.slice(full_wq, [HIDDEN, hidden_q_full], [HIDDEN, 0]),
            pl.slice(
                full_wk,
                [HIDDEN, KV_HIDDEN_LOCAL_R],
                [HIDDEN, 0],
            ),
            pl.slice(
                full_wv,
                [HIDDEN, KV_HIDDEN_LOCAL_R],
                [HIDDEN, 0],
            ),
            q_norm,
            k_norm,
            seq_lens,
            block_table,
            slot_mapping,
            rope_cos_full,
            rope_sin_full,
            k_cache,
            v_cache,
            pl.slice(
                full_wo,
                [hidden_q_full, HIDDEN],
                [hidden_q_full, 0],
            ),
            pl.slice(full_w_g, [HIDDEN, nh_full_pad], [HIDDEN, 0]),
            pl.slice(
                full_gate_r,
                [nh_full_pad, hidden_q_full],
                [nh_full_pad, 0],
            ),
            post_rms,
            pl.slice(moe_gate_w, [N_EXPERTS, HIDDEN], [N_EXPERTS, 0]),
            pl.slice(moe_router_bias, [N_EXPERTS], [N_EXPERTS]),
            pl.reshape(
                pl.slice(
                    moe_w_gate_r,
                    [n_local_experts * HIDDEN, inter],
                    [n_local_experts * HIDDEN, 0],
                ),
                [n_local_experts, HIDDEN, inter],
            ),
            pl.slice(
                moe_w_gate_r_scale,
                [n_local_experts, inter],
                [n_local_experts, 0],
            ),
            pl.reshape(
                pl.slice(
                    moe_w_up_r,
                    [n_local_experts * HIDDEN, inter],
                    [n_local_experts * HIDDEN, 0],
                ),
                [n_local_experts, HIDDEN, inter],
            ),
            pl.slice(
                moe_w_up_r_scale,
                [n_local_experts, inter],
                [n_local_experts, 0],
            ),
            pl.reshape(
                pl.slice(
                    moe_w_down_r,
                    [n_local_experts * inter, HIDDEN],
                    [n_local_experts * inter, 0],
                ),
                [n_local_experts, inter, HIDDEN],
            ),
            pl.slice(
                moe_w_down_r_scale,
                [n_local_experts, HIDDEN],
                [n_local_experts, 0],
            ),
            pl.slice(
                moe_w_gate_s,
                [sh_inter_local, HIDDEN],
                [sh_inter_local, 0],
            ),
            pl.slice(
                moe_w_up_s,
                [sh_inter_local, HIDDEN],
                [sh_inter_local, 0],
            ),
            pl.slice(
                moe_w_down_s,
                [sh_inter_local, HIDDEN],
                [sh_inter_local, 0],
            ),
            hidden_l4_raw,
            resid_l4,
            local_expert_count_l4,
            pl.slice(
                moe_attn_tmp_stack,
                [BATCH, HIDDEN],
                [BATCH, 0],
            ),
            pl.slice(
                moe_attn_signal_stack,
                [COMM_SIGNAL_STRIDE_I32, 1],
                [COMM_SIGNAL_STRIDE_I32, 0],
            ),
            moe_recv_meta,
            moe_meta_arrived,
            moe_recv_x,
            moe_recv_aux,
            moe_recv_route,
            moe_data_arrived,
            pl.slice(moe_sh_tmp_stack, [BATCH, HIDDEN], [BATCH, 0]),
            pl.slice(
                moe_sh_signal_stack,
                [COMM_SIGNAL_STRIDE_I32, 1],
                [COMM_SIGNAL_STRIDE_I32, 0],
            ),
            moe_combine_arrived,
            moe_routed_y_buf,
            4,
            0,
            num_tokens,
            my_rank,
            2,
        )
        recv_meta_l4, hidden_l4 = self.snapshot_recv_meta_and_hidden(
            moe_recv_meta,
            my_rank,
            hidden_l4_raw,
            recv_meta_l4,
            hidden_l4,
        )
        return hidden_l3, hidden_l4, recv_meta_l3, recv_meta_l4

    @pl.function(
        level=pl.Level.HOST,
        role=pl.Role.Orchestrator,
    )
    def five_layer_route_host_orch(  # noqa: PLR0913, PLR0915
        self,
        current_hidden: pl.Tensor[
            [tp_size, BATCH, HIDDEN], pl.BF16
        ],
        input_rms: pl.Tensor[
            [tp_size, LAYER_DYN, HIDDEN], pl.FP32
        ],
        post_rms: pl.Tensor[
            [tp_size, LAYER_DYN, HIDDEN], pl.FP32
        ],
        q_norm: pl.Tensor[
            [tp_size, LAYER_DYN, HEAD_DIM], pl.FP32
        ],
        k_norm: pl.Tensor[
            [tp_size, LAYER_DYN, HEAD_DIM], pl.FP32
        ],
        full_wq: pl.Tensor[
            [tp_size, N_FULL_FIVE, HIDDEN, hidden_q_full], pl.BF16
        ],
        full_wk: pl.Tensor[
            [tp_size, N_FULL_FIVE, HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16
        ],
        full_wv: pl.Tensor[
            [tp_size, N_FULL_FIVE, HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16
        ],
        full_wo: pl.Tensor[
            [tp_size, N_FULL_FIVE, hidden_q_full, HIDDEN], pl.BF16
        ],
        full_w_g: pl.Tensor[
            [tp_size, N_FULL_FIVE, HIDDEN, nh_full_pad], pl.BF16
        ],
        full_gate_r: pl.Tensor[
            [tp_size, N_FULL_FIVE, nh_full_pad, hidden_q_full],
            pl.BF16,
        ],
        swa_wq: pl.Tensor[
            [tp_size, N_SWA_FIVE, HIDDEN, hidden_q_swa], pl.BF16
        ],
        swa_wk: pl.Tensor[
            [tp_size, N_SWA_FIVE, HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16
        ],
        swa_wv: pl.Tensor[
            [tp_size, N_SWA_FIVE, HIDDEN, KV_HIDDEN_LOCAL_R], pl.BF16
        ],
        swa_wo: pl.Tensor[
            [tp_size, N_SWA_FIVE, hidden_q_swa, HIDDEN], pl.BF16
        ],
        swa_w_g: pl.Tensor[
            [tp_size, N_SWA_FIVE, HIDDEN, nh_swa_pad], pl.BF16
        ],
        swa_gate_r: pl.Tensor[
            [tp_size, N_SWA_FIVE, nh_swa_pad, hidden_q_swa], pl.BF16
        ],
        dense_w_gate: pl.Tensor[
            [tp_size, N_DENSE_FIVE, HIDDEN, INTER_LOCAL], pl.BF16
        ],
        dense_w_up: pl.Tensor[
            [tp_size, N_DENSE_FIVE, HIDDEN, INTER_LOCAL], pl.BF16
        ],
        dense_w_down: pl.Tensor[
            [tp_size, N_DENSE_FIVE, INTER_LOCAL, HIDDEN], pl.BF16
        ],
        moe_gate_w: pl.Tensor[
            [tp_size, N_MOE_FIVE, N_EXPERTS, HIDDEN], pl.FP32
        ],
        moe_router_bias: pl.Tensor[
            [tp_size, N_MOE_FIVE, N_EXPERTS], pl.FP32
        ],
        moe_w_gate_r: pl.Tensor[
            [
                tp_size,
                N_MOE_FIVE,
                n_local_experts,
                HIDDEN,
                inter,
            ],
            pl.INT8,
        ],
        moe_w_gate_r_scale: pl.Tensor[
            [tp_size, N_MOE_FIVE, n_local_experts, inter], pl.FP32
        ],
        moe_w_up_r: pl.Tensor[
            [
                tp_size,
                N_MOE_FIVE,
                n_local_experts,
                HIDDEN,
                inter,
            ],
            pl.INT8,
        ],
        moe_w_up_r_scale: pl.Tensor[
            [tp_size, N_MOE_FIVE, n_local_experts, inter], pl.FP32
        ],
        moe_w_down_r: pl.Tensor[
            [
                tp_size,
                N_MOE_FIVE,
                n_local_experts,
                inter,
                HIDDEN,
            ],
            pl.INT8,
        ],
        moe_w_down_r_scale: pl.Tensor[
            [tp_size, N_MOE_FIVE, n_local_experts, HIDDEN], pl.FP32
        ],
        moe_w_gate_s: pl.Tensor[
            [tp_size, N_MOE_FIVE, sh_inter_local, HIDDEN], pl.BF16
        ],
        moe_w_up_s: pl.Tensor[
            [tp_size, N_MOE_FIVE, sh_inter_local, HIDDEN], pl.BF16
        ],
        moe_w_down_s: pl.Tensor[
            [tp_size, N_MOE_FIVE, sh_inter_local, HIDDEN], pl.BF16
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
        rope_cos_full: pl.Tensor[
            [tp_size, ROPE_SEQ_DYN, rotary_dim_full], pl.FP32
        ],
        rope_sin_full: pl.Tensor[
            [tp_size, ROPE_SEQ_DYN, rotary_dim_full], pl.FP32
        ],
        rope_cos_swa: pl.Tensor[
            [tp_size, ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32
        ],
        rope_sin_swa: pl.Tensor[
            [tp_size, ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32
        ],
        k_cache: pl.InOut[
            pl.Tensor[
                [tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16
            ]
        ],
        v_cache: pl.InOut[
            pl.Tensor[
                [tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16
            ]
        ],
        hidden_l3: pl.Out[
            pl.Tensor[[tp_size, BATCH, HIDDEN], pl.BF16]
        ],
        hidden_l4: pl.Out[
            pl.Tensor[[tp_size, BATCH, HIDDEN], pl.BF16]
        ],
        local_expert_count_l3: pl.Out[
            pl.Tensor[[tp_size, n_local_experts], pl.INT32]
        ],
        local_expert_count_l4: pl.Out[
            pl.Tensor[[tp_size, n_local_experts], pl.INT32]
        ],
        recv_meta_l3: pl.Out[
            pl.Tensor[
                [tp_size, n_ranks, n_local_experts_pad], pl.INT32
            ]
        ],
        recv_meta_l4: pl.Out[
            pl.Tensor[
                [tp_size, n_ranks, n_local_experts_pad], pl.INT32
            ]
        ],
        num_tokens_per_owner: pl.Tensor[
            [NUM_TOKENS_RUNTIME], pl.INT32
        ],
    ):
        dense_attn_tmp_buf = pld.alloc_window_buffer(
            N_DENSE_FIVE * BATCH * HIDDEN * 2
        )
        dense_attn_signal_buf = pld.alloc_window_buffer(
            N_DENSE_FIVE * COMM_CONTROL_SIGNAL_BYTES
        )
        dense_mlp_tmp_buf = pld.alloc_window_buffer(
            N_DENSE_FIVE * BATCH * HIDDEN * 2
        )
        dense_mlp_signal_buf = pld.alloc_window_buffer(
            N_DENSE_FIVE * COMM_CONTROL_SIGNAL_BYTES
        )
        moe_attn_tmp_buf = pld.alloc_window_buffer(
            N_MOE_FIVE * BATCH * HIDDEN * 2
        )
        moe_attn_signal_buf = pld.alloc_window_buffer(
            N_MOE_FIVE * COMM_CONTROL_SIGNAL_BYTES
        )
        moe_recv_meta_buf = pld.alloc_window_buffer(
            n_ranks * n_local_experts_pad * 4
        )
        moe_meta_arrived_buf = pld.alloc_window_buffer(
            COMM_CONTROL_SIGNAL_BYTES
        )
        moe_recv_x_buf = pld.alloc_window_buffer(
            dispatch_lane_rows * HIDDEN
        )
        moe_recv_aux_buf = pld.alloc_window_buffer(
            dispatch_lane_rows * dispatch_aux_pad * 4
        )
        moe_recv_route_buf = pld.alloc_window_buffer(
            dispatch_lane_rows * idx_pad * 4
        )
        moe_data_arrived_buf = pld.alloc_window_buffer(
            COMM_CONTROL_SIGNAL_BYTES
        )
        moe_sh_tmp_buf = pld.alloc_window_buffer(
            N_MOE_FIVE * BATCH * HIDDEN * 2
        )
        moe_sh_signal_buf = pld.alloc_window_buffer(
            N_MOE_FIVE * COMM_CONTROL_SIGNAL_BYTES
        )
        moe_combine_arrived_buf = pld.alloc_window_buffer(
            COMM_CONTROL_SIGNAL_BYTES
        )
        moe_routed_y_buf = pld.alloc_window_buffer(
            n_routes_per_rank * HIDDEN * 2
        )

        for rank in pl.range(pld.world_size()):
            self.five_layer_route_chip_orch(
                current_hidden[rank],
                input_rms[rank],
                post_rms[rank],
                q_norm[rank],
                k_norm[rank],
                pl.reshape(
                    full_wq[rank],
                    [N_FULL_FIVE * HIDDEN, hidden_q_full],
                ),
                pl.reshape(
                    full_wk[rank],
                    [N_FULL_FIVE * HIDDEN, KV_HIDDEN_LOCAL_R],
                ),
                pl.reshape(
                    full_wv[rank],
                    [N_FULL_FIVE * HIDDEN, KV_HIDDEN_LOCAL_R],
                ),
                pl.reshape(
                    full_wo[rank],
                    [N_FULL_FIVE * hidden_q_full, HIDDEN],
                ),
                pl.reshape(
                    full_w_g[rank],
                    [N_FULL_FIVE * HIDDEN, nh_full_pad],
                ),
                pl.reshape(
                    full_gate_r[rank],
                    [N_FULL_FIVE * nh_full_pad, hidden_q_full],
                ),
                pl.reshape(
                    swa_wq[rank],
                    [N_SWA_FIVE * HIDDEN, hidden_q_swa],
                ),
                pl.reshape(
                    swa_wk[rank],
                    [N_SWA_FIVE * HIDDEN, KV_HIDDEN_LOCAL_R],
                ),
                pl.reshape(
                    swa_wv[rank],
                    [N_SWA_FIVE * HIDDEN, KV_HIDDEN_LOCAL_R],
                ),
                pl.reshape(
                    swa_wo[rank],
                    [N_SWA_FIVE * hidden_q_swa, HIDDEN],
                ),
                pl.reshape(
                    swa_w_g[rank],
                    [N_SWA_FIVE * HIDDEN, nh_swa_pad],
                ),
                pl.reshape(
                    swa_gate_r[rank],
                    [N_SWA_FIVE * nh_swa_pad, hidden_q_swa],
                ),
                pl.reshape(
                    dense_w_gate[rank],
                    [N_DENSE_FIVE * HIDDEN, INTER_LOCAL],
                ),
                pl.reshape(
                    dense_w_up[rank],
                    [N_DENSE_FIVE * HIDDEN, INTER_LOCAL],
                ),
                pl.reshape(
                    dense_w_down[rank],
                    [N_DENSE_FIVE * INTER_LOCAL, HIDDEN],
                ),
                pl.reshape(
                    moe_gate_w[rank],
                    [N_MOE_FIVE * N_EXPERTS, HIDDEN],
                ),
                pl.reshape(
                    moe_router_bias[rank],
                    [N_MOE_FIVE * N_EXPERTS],
                ),
                pl.reshape(
                    moe_w_gate_r[rank],
                    [N_MOE_FIVE * n_local_experts * HIDDEN, inter],
                ),
                pl.reshape(
                    moe_w_gate_r_scale[rank],
                    [N_MOE_FIVE * n_local_experts, inter],
                ),
                pl.reshape(
                    moe_w_up_r[rank],
                    [N_MOE_FIVE * n_local_experts * HIDDEN, inter],
                ),
                pl.reshape(
                    moe_w_up_r_scale[rank],
                    [N_MOE_FIVE * n_local_experts, inter],
                ),
                pl.reshape(
                    moe_w_down_r[rank],
                    [N_MOE_FIVE * n_local_experts * inter, HIDDEN],
                ),
                pl.reshape(
                    moe_w_down_r_scale[rank],
                    [N_MOE_FIVE * n_local_experts, HIDDEN],
                ),
                pl.reshape(
                    moe_w_gate_s[rank],
                    [N_MOE_FIVE * sh_inter_local, HIDDEN],
                ),
                pl.reshape(
                    moe_w_up_s[rank],
                    [N_MOE_FIVE * sh_inter_local, HIDDEN],
                ),
                pl.reshape(
                    moe_w_down_s[rank],
                    [N_MOE_FIVE * sh_inter_local, HIDDEN],
                ),
                seq_lens[rank],
                block_table[rank],
                slot_mapping[rank],
                rope_cos_full[rank],
                rope_sin_full[rank],
                rope_cos_swa[rank],
                rope_sin_swa[rank],
                k_cache[rank],
                v_cache[rank],
                hidden_l3[rank],
                hidden_l4[rank],
                local_expert_count_l3[rank],
                local_expert_count_l4[rank],
                recv_meta_l3[rank],
                recv_meta_l4[rank],
                pld.window(
                    dense_attn_tmp_buf,
                    [N_DENSE_FIVE * BATCH, HIDDEN],
                    dtype=pl.BF16,
                ),
                pld.window(
                    dense_attn_signal_buf,
                    [N_DENSE_FIVE * COMM_SIGNAL_STRIDE_I32, 1],
                    dtype=pl.INT32,
                ),
                pld.window(
                    dense_mlp_tmp_buf,
                    [N_DENSE_FIVE * BATCH, HIDDEN],
                    dtype=pl.BF16,
                ),
                pld.window(
                    dense_mlp_signal_buf,
                    [N_DENSE_FIVE * COMM_SIGNAL_STRIDE_I32, 1],
                    dtype=pl.INT32,
                ),
                pld.window(
                    moe_attn_tmp_buf,
                    [N_MOE_FIVE * BATCH, HIDDEN],
                    dtype=pl.BF16,
                ),
                pld.window(
                    moe_attn_signal_buf,
                    [N_MOE_FIVE * COMM_SIGNAL_STRIDE_I32, 1],
                    dtype=pl.INT32,
                ),
                pld.window(
                    moe_recv_meta_buf,
                    [n_ranks, n_local_experts_pad],
                    dtype=pl.INT32,
                ),
                pld.window(
                    moe_meta_arrived_buf,
                    [COMM_SIGNAL_STRIDE_I32, 1],
                    dtype=pl.INT32,
                ),
                pld.window(
                    moe_recv_x_buf,
                    [dispatch_lane_rows, HIDDEN],
                    dtype=pl.INT8,
                ),
                pld.window(
                    moe_recv_aux_buf,
                    [dispatch_lane_rows, dispatch_aux_pad],
                    dtype=pl.FP32,
                ),
                pld.window(
                    moe_recv_route_buf,
                    [dispatch_lane_rows, idx_pad],
                    dtype=pl.INT32,
                ),
                pld.window(
                    moe_data_arrived_buf,
                    [COMM_SIGNAL_STRIDE_I32, 1],
                    dtype=pl.INT32,
                ),
                pld.window(
                    moe_sh_tmp_buf,
                    [N_MOE_FIVE * BATCH, HIDDEN],
                    dtype=pl.BF16,
                ),
                pld.window(
                    moe_sh_signal_buf,
                    [N_MOE_FIVE * COMM_SIGNAL_STRIDE_I32, 1],
                    dtype=pl.INT32,
                ),
                pld.window(
                    moe_combine_arrived_buf,
                    [COMM_SIGNAL_STRIDE_I32, 1],
                    dtype=pl.INT32,
                ),
                pld.window(
                    moe_routed_y_buf,
                    [n_routes_per_rank, HIDDEN],
                    dtype=pl.BF16,
                ),
                num_tokens_per_owner,
                rank,
                device=rank,
            )


_ROUTE_PARSED = FiveLayerMoeRouteInstrumented
_REQUIRED_CANONICAL = (
    "tp_all_reduce",
    "full_chip_orch",
    "swa_chip_orch",
    "_gate",
    "gate_step",
    "_norm_quant_moe_input",
    "dispatch_step",
    "_expert_routed",
    "expert_routed_step",
    "_expert_shared_local",
    "expert_shared_step",
    "combine_step",
    "full_moe_chip_orch",
    "swa_moe_chip_orch",
)

_FUNCTIONS = {
    function.name: function
    for function in _ROUTE_PARSED.functions.values()
}
for _name in _REQUIRED_CANONICAL:
    _function = _CANONICAL_PROGRAM.get_function(_name)
    if _function is None:
        raise RuntimeError(f"canonical program is missing required function {_name}")
    _FUNCTIONS[_name] = _function

five_layer_moe_route = ir.Program(
    list(_FUNCTIONS.values()),
    "FiveLayerMoeRoute",
    _ROUTE_PARSED.span,
)


__all__ = [
    "N_DENSE_FIVE",
    "N_FULL_FIVE",
    "N_MOE_FIVE",
    "N_SWA_FIVE",
    "five_layer_moe_route",
]
