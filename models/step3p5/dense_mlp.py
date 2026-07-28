# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Step3p5 dense MLP inline kernel shared by canonical Main and MTP.

本模块只承载 TP-sliced dense MLP 计算，不定义任何 ``@pl.program``、
host ABI、debug 输出或历史 whole-net 入口。跨 rank control signal 的物理
byte-span 由 enclosing program 的真实 task formal 决定：

- canonical Main 从 stacked per-layer backing 取 512B-strided slot；
- standalone/MTP 使用各自独立的 compact signal backing。

inline formal 通过 ``SIGNAL_WINDOW_ROWS`` 由 enclosing module 解析：
canonical 为 128 行，MTP/standalone 为 ``TP_WORLD_SIZE`` 行；通信 loop
始终只访问前 ``TP_WORLD_SIZE`` 个 INT32。
"""
from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld

from .config import (
    BATCH,
    EPS,
    HIDDEN,
    HIDDEN_INV,
    INTERMEDIATE_LOCAL,
    K_CHUNK,
    LAYER_DYN,
    LAYER_HIDDEN_ROWS_DYN,
    LAYER_INTER_ROWS_DYN,
    MLP_OUT_CHUNK,
    TP_WORLD_SIZE,
)


INTER_LOCAL = INTERMEDIATE_LOCAL
# MTP/standalone callers use a compact independent signal.  Canonical Main
# supplies the same symbolic name as 128-row stacked/reused control storage.
SIGNAL_WINDOW_ROWS = TP_WORLD_SIZE


@pl.jit.inline
def dense_mlp_body_tp(
    resid1: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    w_gate: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTER_LOCAL], pl.BF16],
    w_up: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTER_LOCAL], pl.BF16],
    w_down: pl.Tensor[[LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16],
    next_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    norm_layer_idx: pl.Scalar[pl.INT32],
    mlp_layer_idx: pl.Scalar[pl.INT32],
    tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
    signal_window: pld.DistributedTensor[[SIGNAL_WINDOW_ROWS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
    """执行 post-attention RMSNorm、TP-sliced SwiGLU MLP 与 residual add。"""
    hidden_blocks = HIDDEN // K_CHUNK
    mlp_out_blocks = INTER_LOCAL // MLP_OUT_CHUNK
    layer_hidden_base = mlp_layer_idx * HIDDEN
    layer_inter_base = mlp_layer_idx * INTER_LOCAL

    dm_post_norm = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    dm_resid1_fp32 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dense_post_rmsnorm_zc"):
        for kb in pl.range(hidden_blocks):
            k0 = kb * K_CHUNK
            dm_rchunk = pl.cast(
                pl.slice(resid1, [BATCH, K_CHUNK], [0, k0]),
                target_type=pl.FP32,
            )
            dm_resid1_fp32 = pl.assemble(
                dm_resid1_fp32, dm_rchunk, [0, k0],
            )

        dm_sq_sum = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
        for kb2 in pl.range(hidden_blocks):
            k0 = kb2 * K_CHUNK
            dm_ck = pl.slice(
                dm_resid1_fp32, [BATCH, K_CHUNK], [0, k0],
            )
            dm_sq_sum = pl.add(
                dm_sq_sum,
                pl.reshape(
                    pl.row_sum(pl.mul(dm_ck, dm_ck)),
                    [1, BATCH],
                ),
            )
        inv_rms_dense = pl.recip(
            pl.sqrt(pl.add(pl.mul(dm_sq_sum, HIDDEN_INV), EPS)),
        )
        dm_inv_rms_col = pl.reshape(inv_rms_dense, [BATCH, 1])
        for kb3 in pl.range(hidden_blocks):
            k0 = kb3 * K_CHUNK
            dm_norm_chunk = pl.slice(
                dm_resid1_fp32, [BATCH, K_CHUNK], [0, k0],
            )
            dm_gamma = pl.slice(
                post_rms_weight, [1, K_CHUNK], [norm_layer_idx, k0],
            )
            dm_scaled = pl.row_expand_mul(dm_norm_chunk, dm_inv_rms_col)
            dm_normed = pl.col_expand_mul(
                dm_scaled, pl.add(dm_gamma, 1.0),
            )
            dm_post_norm = pl.assemble(
                dm_post_norm,
                pl.cast(dm_normed, target_type=pl.BF16),
                [0, k0],
            )

    gate_acc_gm = pl.create_tensor([BATCH, INTER_LOCAL], dtype=pl.FP32)
    up_acc_gm = pl.create_tensor([BATCH, INTER_LOCAL], dtype=pl.FP32)
    for ob in pl.spmd(
        mlp_out_blocks,
        name_hint="dense_gate_up_matmul_tp",
        optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
    ):
        mlp_o0 = ob * MLP_OUT_CHUNK
        post_chunk_0 = pl.slice(
            dm_post_norm, [BATCH, K_CHUNK], [0, 0],
        )
        wg_0 = pl.slice(
            w_gate,
            [K_CHUNK, MLP_OUT_CHUNK],
            [layer_hidden_base, mlp_o0],
        )
        wu_0 = pl.slice(
            w_up,
            [K_CHUNK, MLP_OUT_CHUNK],
            [layer_hidden_base, mlp_o0],
        )
        gate_acc = pl.matmul(
            post_chunk_0, wg_0, out_dtype=pl.FP32,
        )
        up_acc = pl.matmul(
            post_chunk_0, wu_0, out_dtype=pl.FP32,
        )
        for kb in pl.range(1, hidden_blocks):
            k0 = kb * K_CHUNK
            post_chunk = pl.slice(
                dm_post_norm, [BATCH, K_CHUNK], [0, k0],
            )
            wg = pl.slice(
                w_gate,
                [K_CHUNK, MLP_OUT_CHUNK],
                [layer_hidden_base + k0, mlp_o0],
            )
            wu = pl.slice(
                w_up,
                [K_CHUNK, MLP_OUT_CHUNK],
                [layer_hidden_base + k0, mlp_o0],
            )
            gate_acc = pl.matmul_acc(gate_acc, post_chunk, wg)
            up_acc = pl.matmul_acc(up_acc, post_chunk, wu)
        gate_acc_gm = pl.assemble(
            gate_acc_gm, gate_acc, [0, mlp_o0],
        )
        up_acc_gm = pl.assemble(
            up_acc_gm, up_acc, [0, mlp_o0],
        )

    mlp_tile = pl.create_tensor([BATCH, INTER_LOCAL], dtype=pl.BF16)
    for ob in pl.spmd(
        mlp_out_blocks, name_hint="dense_silu_cast_tp",
    ):
        mlp_o0 = ob * MLP_OUT_CHUNK
        gate_chunk = pl.slice(
            gate_acc_gm, [BATCH, MLP_OUT_CHUNK], [0, mlp_o0],
        )
        up_chunk = pl.slice(
            up_acc_gm, [BATCH, MLP_OUT_CHUNK], [0, mlp_o0],
        )
        sigmoid = pl.recip(
            pl.add(pl.exp(pl.neg(gate_chunk)), 1.0),
        )
        mlp_chunk = pl.mul(
            pl.mul(gate_chunk, sigmoid), up_chunk,
        )
        mlp_tile = pl.assemble(
            mlp_tile,
            pl.cast(mlp_chunk, target_type=pl.BF16),
            [0, mlp_o0],
        )

    partial_hidden_fp32 = pl.create_tensor(
        [BATCH, HIDDEN], dtype=pl.FP32,
    )
    for dob in pl.spmd(
        hidden_blocks,
        name_hint="dense_down_matmul_tp",
        optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
    ):
        d0 = dob * K_CHUNK
        mlp_chunk_0 = pl.slice(
            mlp_tile, [BATCH, MLP_OUT_CHUNK], [0, 0],
        )
        w_down_chunk_0 = pl.slice(
            w_down,
            [MLP_OUT_CHUNK, K_CHUNK],
            [layer_inter_base, d0],
        )
        down_acc = pl.matmul(
            mlp_chunk_0, w_down_chunk_0, out_dtype=pl.FP32,
        )
        for ob in pl.range(1, mlp_out_blocks):
            down_o0 = ob * MLP_OUT_CHUNK
            down_mlp_chunk_bf16 = pl.slice(
                mlp_tile,
                [BATCH, MLP_OUT_CHUNK],
                [0, down_o0],
            )
            w_down_chunk = pl.slice(
                w_down,
                [MLP_OUT_CHUNK, K_CHUNK],
                [layer_inter_base + down_o0, d0],
            )
            down_acc = pl.matmul_acc(
                down_acc, down_mlp_chunk_bf16, w_down_chunk,
            )
        partial_hidden_fp32 = pl.assemble(
            partial_hidden_fp32, down_acc, [0, d0],
        )

    partial_hidden = pl.create_tensor(
        [BATCH, HIDDEN], dtype=pl.BF16,
    )
    for dob in pl.spmd(
        hidden_blocks, name_hint="dense_down_cast_tp",
    ):
        d0 = dob * K_CHUNK
        dense_fp32_chunk = pl.slice(
            partial_hidden_fp32, [BATCH, K_CHUNK], [0, d0],
        )
        partial_hidden = pl.assemble(
            partial_hidden,
            pl.cast(dense_fp32_chunk, target_type=pl.BF16),
            [0, d0],
        )

    if TP_WORLD_SIZE > 1:
        partial_hidden = self.tp_all_reduce(
            partial_hidden, tmp_window, signal_window, my_rank,
        )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dense_residual_add_tp"):
        for kb4 in pl.range(hidden_blocks):
            k0 = kb4 * K_CHUNK
            dm_reduced = pl.cast(
                pl.slice(
                    partial_hidden, [BATCH, K_CHUNK], [0, k0],
                ),
                target_type=pl.FP32,
            )
            dm_r = pl.slice(
                dm_resid1_fp32, [BATCH, K_CHUNK], [0, k0],
            )
            next_hidden = pl.assemble(
                next_hidden,
                pl.cast(
                    pl.add(dm_r, dm_reduced),
                    target_type=pl.BF16,
                ),
                [0, k0],
            )

    return next_hidden


__all__ = ["dense_mlp_body_tp"]
