# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""One-shot code generator: derive ``_build_whole_decode_faithful_real_program``
(class ``WholeDecodeFaithfulReal``, module binding ``whole_decode_faithful_real``)
from the compile+device-verified reuse-one-slab ``_build_whole_decode_faithful_program``.

Real per-layer weights, N=1 whole-decode, full+swa complete routing:
  - norm (input_rms/post_rms/q_norm/k_norm): full [45] stack, kernel-indexed by
    absolute layer_idx L (KV-cache base = norm*rows needs absolute L).
  - attn (wq/wk/wv/wo/w_g/gate_r) + dense-MLP (w_gate/w_up/w_down): host-sliced
    single-layer slabs; kernel weight-index = 0. (Probe-verified: pl.inline
    accepts a smaller actual leading dim than the annotation.)
  - MoE experts (gate_w/router_bias/w_*_{r,s}): host-sliced [42] stack by pos.
  - per-layer full/swa routing: 11 full-attn MoE layers (L4,8,..,44) go through
    full_attn_only_orch; 31 swa-attn MoE layers go through swa_attn_only_orch.

The existing method-set (tp_all_reduce/ep_all_to_all/gate/dispatch/expert/combine/
attn_dense_orch/lm_head_orch) is reused verbatim. chip_orch is reused with
layer_idx renamed to norm_layer_idx. full_chip_orch/swa_chip_orch/
swa_attn_only_orch are rewritten single-layer; full_attn_only_orch is new;
host_orch is rewritten with the unified stacked signature + real per-layer args.

Run once (idempotent — refuses if the real builder already exists):
    python tools/step3p5/_gen_faithful_real.py
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "models" / "step3p5" / "decode_layer.py"

DEF_MARKER = "def _build_whole_decode_faithful_program("
BIND_MARKER = "\nwhole_decode_faithful = _build_whole_decode_faithful_program()"
CHIP_MARKER = (
    "        @pl.function(type=pl.FunctionType.Orchestration)\n"
    "        def chip_orch(  # noqa: PLR0913, PLR0915"
)
LMHEAD_MARKER = "        # ---- Tail: final RMSNorm + LM head (separate SSA scope) ----------"
DENSE_METHODS_MARKER = "        # ---- Dense-prefix attention methods (spliced verbatim from WDP). ----"


def _layer_classification():
    sys.path.insert(0, str(REPO))
    import models.step3p5.config as cfg  # noqa: PLC0415
    n_layers = cfg.NUM_HIDDEN_LAYERS
    full = [li for li in range(n_layers) if cfg.is_full_attention(li)]
    swa = [li for li in range(n_layers) if not cfg.is_full_attention(li)]
    moe = list(cfg.MOE_LAYER_INDICES)
    full_local = {li: i for i, li in enumerate(full)}
    swa_local = {li: i for i, li in enumerate(swa)}
    dense = [li for li in range(n_layers) if li not in moe]
    return {
        "N_FULL": len(full), "N_SWA": len(swa), "N_DENSE": len(dense),
        "N_MOE": len(moe), "full_local": full_local, "swa_local": swa_local,
        "is_full": {li: cfg.is_full_attention(li) for li in range(n_layers)},
        "moe": moe, "dense": dense,
    }


FRESH_FULL_CHIP_ORCH = '''
        # ---- Dense-prefix full attention + dense MLP (single-layer host-slice). ----
        @pl.function(type=pl.FunctionType.Orchestration)
        def full_chip_orch(  # noqa: PLR0913
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
'''

FRESH_SWA_CHIP_ORCH = '''
        @pl.function(type=pl.FunctionType.Orchestration)
        def swa_chip_orch(  # noqa: PLR0913
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
'''

# ── A1: routed-expert `_expert_routed` rewritten byte-faithful to the
# device-PASS moe.py `EpTpMoE._expert_routed` (ratio_allclose atol=0.04 vs
# torch W8A8).  The input INT8 quant is a single FLAT pre-pass over the full
# `local_recv_max` recv buffer (mirrors moe.py `_quant_moe_input`: per-token
# amax over FULL RECV_TILE tiles — NO partial-tile quant, avoiding the gap-5
# in-expert partial-tile miscompile that made the whole-net routed output
# ~1e11).  Everything downstream (INT8 gate/up/down cube, dequant, intermediate
# `h_i8` requant with BARE slice, gated `fillpad(zero)`) is identical to moe.py.
# `lrx_scale` is UNPADDED [1, local_recv_max]; the per-token dequant scale is
# read as a contiguous [1,RECV_TILE] row-slice + reshape (ccec ND2ND-safe).
# Numerically == dispatch-side quant because per-token quantization is invariant
# under the dispatch row permutation. The final path carries INT8 + scale across
# the dispatch boundary without a BF16 dequantization fallback.
FRESH_EXPERT_ROUTED = '''        def _expert_routed(  # noqa: PLR0913, PLR0915
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
                valid_rows = pl.cast(n_rows, pl.INDEX)

                for tile_idx in pl.range(N_RECV_TILES):
                    tile_row0 = tile_idx * RECV_TILE
                    tile_offset = offset + tile_row0
                    tile_valid = pl.min(RECV_TILE, valid_rows - tile_row0)
                    if tile_valid > 0:

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
                            if _routed_swiglu_step:
                                silu_c = pl.minimum(silu, _routed_swiglu_limit)
                                up_c = pl.maximum(
                                    pl.minimum(up_2d, _routed_swiglu_limit),
                                    -_routed_swiglu_limit,
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

            return local_routed_y'''


# ── 1A dispatch-side INT8 quant kernel (byte-faithful moe.py `_quant_moe_input`,
# moe.py:1947). Per-token INT8 dynamic-quant of the routed-expert INPUT done in
# chip_orch BEFORE dispatch (router `_gate` + unquantized shared already used the
# ORIGINAL BF16 x). Output INT8 x + per-token FP32 dequant scale (amax/127) in
# col 0 of an 8-wide (=32B) window and transported through dispatch as INT8
# recv_x + recv_scale. This preserves native W8A8 and reduces the communication
# footprint; it is retained as a prerequisite, not claimed as the final 512B
# signal-isolation A/B variable. BATCH tokens == one MOE_IN_QUANT_T_TILE (=16).
# InCore + pl.range (NOT pl.spmd → avoids nested InCore ScopeStmt rejected by
# SplitIncoreOrch #1828). INT8 chain: rint INT32 → round FP16 → trunc INT8.
FRESH_QUANT_MOE_INPUT = '''        @pl.function(type=pl.FunctionType.InCore)
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

'''


def _strip_moe_knobs(text: str) -> str:
    """Remove diagnostic ``if _MOE_{PASSTHROUGH,NORM_ONLY,SHARED_ONLY} > 0:``
    early-return blocks (each ends in ``return next_hidden_out``) so the spliced
    MoE body is production-clean."""
    out = []
    skip = False
    for ln in text.split("\n"):
        s = ln.strip()
        if s.startswith("if _MOE_") and s.endswith("> 0:"):
            skip = True
            continue
        if skip:
            if s == "return next_hidden_out":
                skip = False
            continue
        out.append(ln)
    return "\n".join(out)


def _fused_moe_head(variant: str) -> str:
    """Signature + attention front for the FUSED MoE-layer orchestration.

    Mirrors the dense ``{variant}_chip_orch``: attention runs first and writes
    a LOCAL ``resid1`` tensor (intra-Submission RAW, guaranteed non-aliasing),
    which the spliced MoE body (post_norm -> MoE -> residual) then consumes.
    Folding attention into the MoE orch removes the fragile cross-orch handoff
    on the shared ``h_mid_out`` program-Out that produced the M3b uninitialised
    read. Returns the def header through the ``attention_{variant}_inline`` call;
    the caller appends the shared MoE body.
    """
    hq = f"hidden_q_{variant}"
    nhp = f"nh_{variant}_pad"
    rdim = f"rotary_dim_{variant}"
    attn_inline = f"attention_{variant}_inline"
    return f'''
        # ---- MoE-layer {variant}-attn FUSED with MoE-block: attention -> resid1
        # (local, intra-orch) -> post_norm -> EP/TP MoE -> residual. Mirrors the
        # dense {variant}_chip_orch attn->MLP handoff so the attn->MoE handoff is
        # an intra-Submission local tensor, not a cross-orch shared h_mid_out. ----
        @pl.function(type=pl.FunctionType.Orchestration)
        def {variant}_moe_chip_orch(  # noqa: PLR0913, PLR0915
            self,
            current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            wq: pl.Tensor[[HIDDEN, {hq}], pl.BF16],
            wk: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL], pl.BF16],
            wv: pl.Tensor[[HIDDEN, KV_HIDDEN_LOCAL], pl.BF16],
            q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            rope_cos: pl.Tensor[[ROPE_SEQ_DYN, {rdim}], pl.FP32],
            rope_sin: pl.Tensor[[ROPE_SEQ_DYN, {rdim}], pl.FP32],
            k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            wo: pl.Tensor[[{hq}, HIDDEN], pl.BF16],
            w_g: pl.Tensor[[HIDDEN, {nhp}], pl.BF16],
            gate_r: pl.Tensor[[{nhp}, {hq}], pl.BF16],
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
            resid1 = {attn_inline}(
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
'''


# ==========================================================================
# Canonical fixed-slot PULL dispatch used by the device-verified real builder.
# The full stage-2 task boundary is kept as one template so regeneration cannot
# move recv-count or inverse-map work across InCore boundaries.
#
# WHY: the final protocol uses consumer-side gather-by-PULL so transfer
# completion is observed by the consuming rank. Historical kernel-position
# mappings motivated this direction, but they do not by themselves prove that
# TPUT was the root cause. The final minimal layout A/B that removed the
# intermittent canonical stall was 512B control-signal isolation.
#
# INVARIANT PRESERVED: recv_x / recv_scale / recv_r_route / local_expert_offset|
# count are reproduced in the SAME expert-major CSR order (loc_e outer, src rank
# ascending, source-cursor inner) that the push produced. Dispatch also produces
# the source-local inverse_map at the same InCore boundary; combine consumes that
# map directly instead of re-reading the distributed count matrix.
#
# OFFSET MATH (validated): source s packs its outgoing tokens into its OWN
# peer-readable send_x window in (dst,loc_e) bucket order (bkt=dst*nle+loc_e,
# base = prefix over send_counts_bkt). So dst=my_rank's tokens from source s for
# local expert loc_e begin in s.send_x at:
#   off_s = sum_{d'<my_rank} sum_{e'} pub_counts[s*nr+d', e']      # buckets dst<my_rank
#         + sum_{e'<loc_e}       pub_counts[s*nr+my_rank, e']      # buckets (my_rank,e'<loc_e)
# and there are n = pub_counts[s*nr+my_rank, loc_e] of them. All from the
# already-peer-published pub_counts (NO new cross-card publish needed).
FRESH_DISPATCH_PULL_INT8 = '''
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

'''

# PULL combine (push->pull, step 5). Expert holder stages its routed output into
# its OWN peer-readable routed_src_buf window; an AtomicAdd/Ge rendezvous; then
# SOURCE gathers its tokens' routed outputs with remote_load using inverse_map
# (dst_rank, dst_row) — the SAME (dst,dst_row) that dispatch placed them at, so
# routed_y_buf[r_route] is filled exactly as the push produced it and
# _weighted_gather_and_add is unchanged. This is the retained final pull+pull
# protocol; it avoids a peer-write completion dependency without claiming that
# the historical push helper alone was the proven root cause.
FRESH_COMBINE_PULL = '''
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

'''



def _host_orch(cls) -> str:
    N_FULL = cls["N_FULL"]
    N_SWA = cls["N_SWA"]
    N_DENSE = cls["N_DENSE"]
    N_MOE = cls["N_MOE"]
    moe = cls["moe"]
    is_full = cls["is_full"]
    full_local = cls["full_local"]
    swa_local = cls["swa_local"]

    L = []
    A = L.append
    A("        # ---- host_orch: real per-layer weights, full+swa routing. ----")
    A("        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)")
    A("        def host_orch(  # noqa: PLR0913, PLR0915")
    A("            self,")
    A("            current_hidden: pl.Tensor[[tp_size, BATCH, HIDDEN], pl.BF16],")
    A("            input_rms: pl.Tensor[[tp_size, LAYER_DYN, HIDDEN], pl.FP32],")
    A("            post_rms: pl.Tensor[[tp_size, LAYER_DYN, HIDDEN], pl.FP32],")
    A("            q_norm: pl.Tensor[[tp_size, LAYER_DYN, HEAD_DIM], pl.FP32],")
    A("            k_norm: pl.Tensor[[tp_size, LAYER_DYN, HEAD_DIM], pl.FP32],")
    A(f"            full_wq: pl.Tensor[[tp_size, {N_FULL}, HIDDEN, hidden_q_full], pl.BF16],")
    A(f"            full_wk: pl.Tensor[[tp_size, {N_FULL}, HIDDEN, KV_HIDDEN_LOCAL], pl.BF16],")
    A(f"            full_wv: pl.Tensor[[tp_size, {N_FULL}, HIDDEN, KV_HIDDEN_LOCAL], pl.BF16],")
    A(f"            full_wo: pl.Tensor[[tp_size, {N_FULL}, hidden_q_full, HIDDEN], pl.BF16],")
    A(f"            full_w_g: pl.Tensor[[tp_size, {N_FULL}, HIDDEN, nh_full_pad], pl.BF16],")
    A(f"            full_gate_r: pl.Tensor[[tp_size, {N_FULL}, nh_full_pad, hidden_q_full], pl.BF16],")
    A(f"            swa_wq: pl.Tensor[[tp_size, {N_SWA}, HIDDEN, hidden_q_swa], pl.BF16],")
    A(f"            swa_wk: pl.Tensor[[tp_size, {N_SWA}, HIDDEN, KV_HIDDEN_LOCAL], pl.BF16],")
    A(f"            swa_wv: pl.Tensor[[tp_size, {N_SWA}, HIDDEN, KV_HIDDEN_LOCAL], pl.BF16],")
    A(f"            swa_wo: pl.Tensor[[tp_size, {N_SWA}, hidden_q_swa, HIDDEN], pl.BF16],")
    A(f"            swa_w_g: pl.Tensor[[tp_size, {N_SWA}, HIDDEN, nh_swa_pad], pl.BF16],")
    A(f"            swa_gate_r: pl.Tensor[[tp_size, {N_SWA}, nh_swa_pad, hidden_q_swa], pl.BF16],")
    A(f"            dense_w_gate: pl.Tensor[[tp_size, {N_DENSE}, HIDDEN, INTER_LOCAL], pl.BF16],")
    A(f"            dense_w_up: pl.Tensor[[tp_size, {N_DENSE}, HIDDEN, INTER_LOCAL], pl.BF16],")
    A(f"            dense_w_down: pl.Tensor[[tp_size, {N_DENSE}, INTER_LOCAL, HIDDEN], pl.BF16],")
    A(f"            moe_gate_w: pl.Tensor[[tp_size, {N_MOE}, HIDDEN, N_EXPERTS], pl.FP32],")
    A(f"            moe_router_bias: pl.Tensor[[tp_size, {N_MOE}, N_EXPERTS], pl.FP32],")
    A(f"            moe_w_gate_r: pl.Tensor[[tp_size, {N_MOE}, n_local_experts, HIDDEN, inter], pl.INT8],")
    A(f"            moe_w_gate_r_scale: pl.Tensor[[tp_size, {N_MOE}, n_local_experts, inter], pl.FP32],")
    A(f"            moe_w_up_r: pl.Tensor[[tp_size, {N_MOE}, n_local_experts, HIDDEN, inter], pl.INT8],")
    A(f"            moe_w_up_r_scale: pl.Tensor[[tp_size, {N_MOE}, n_local_experts, inter], pl.FP32],")
    A(f"            moe_w_down_r: pl.Tensor[[tp_size, {N_MOE}, n_local_experts, inter, HIDDEN], pl.INT8],")
    A(f"            moe_w_down_r_scale: pl.Tensor[[tp_size, {N_MOE}, n_local_experts, HIDDEN], pl.FP32],")
    A(f"            moe_w_gate_s: pl.Tensor[[tp_size, {N_MOE}, HIDDEN, sh_inter_local], pl.BF16],")
    A(f"            moe_w_up_s: pl.Tensor[[tp_size, {N_MOE}, HIDDEN, sh_inter_local], pl.BF16],")
    A(f"            moe_w_down_s: pl.Tensor[[tp_size, {N_MOE}, sh_inter_local, HIDDEN], pl.BF16],")
    A("            seq_lens: pl.Tensor[[tp_size, USER_BATCH_DYN], pl.INT32],")
    A("            block_table: pl.Tensor[[tp_size, BLOCK_TABLE_FLAT_DYN], pl.INT32],")
    A("            slot_mapping: pl.Tensor[[tp_size, USER_BATCH_DYN], pl.INT32],")
    A("            rope_cos_full: pl.Tensor[[tp_size, ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],")
    A("            rope_sin_full: pl.Tensor[[tp_size, ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],")
    A("            rope_cos_swa: pl.Tensor[[tp_size, ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],")
    A("            rope_sin_swa: pl.Tensor[[tp_size, ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],")
    A("            k_cache: pl.Tensor[[tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],")
    A("            v_cache: pl.Tensor[[tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],")
    A("            h_mid_out: pl.Out[pl.Tensor[[tp_size, BATCH, HIDDEN], pl.BF16]],")
    A("            next_hidden_out: pl.Out[pl.Tensor[[tp_size, BATCH, HIDDEN], pl.BF16]],")
    A("            dbg_out: pl.Out[pl.Tensor[[tp_size, BATCH, HIDDEN], pl.BF16]],")
    A("            final_norm_weight: pl.Tensor[[tp_size, 1, HIDDEN], pl.FP32],")
    A("            lm_head_weight: pl.Tensor[[tp_size, VOCAB_LOCAL, HIDDEN], pl.BF16],")
    A("            logits_shard_out: pl.Out[pl.Tensor[[tp_size, USER_BATCH_DYN, VOCAB_LOCAL], pl.FP32]],")
    A("        ):")

    A("            l0_attn_tmp = pld.alloc_window_buffer(BATCH * HIDDEN * 2)")
    A("            l0_attn_sig = pld.alloc_window_buffer(COMM_CONTROL_SIGNAL_BYTES)")
    A("            l0_mlp_tmp = pld.alloc_window_buffer(BATCH * HIDDEN * 2)")
    A("            l0_mlp_sig = pld.alloc_window_buffer(COMM_CONTROL_SIGNAL_BYTES)")
    A("            l1_attn_tmp = pld.alloc_window_buffer(BATCH * HIDDEN * 2)")
    A("            l1_attn_sig = pld.alloc_window_buffer(COMM_CONTROL_SIGNAL_BYTES)")
    A("            l1_mlp_tmp = pld.alloc_window_buffer(BATCH * HIDDEN * 2)")
    A("            l1_mlp_sig = pld.alloc_window_buffer(COMM_CONTROL_SIGNAL_BYTES)")
    # L2 (2nd swa-dense) MUST have its OWN comm windows, not reuse l1's. Signal
    # windows use AtomicAdd + Ge(1) and are not reset between layers; reusing
    # l1_*_sig could let L2 observe the previous layer's completion value.
    # Retain distinct windows as a framework invariant. This is a potential
    # correctness/stall hazard, not the final 512B layout A/B variable.
    A("            l2_attn_tmp = pld.alloc_window_buffer(BATCH * HIDDEN * 2)")
    A("            l2_attn_sig = pld.alloc_window_buffer(COMM_CONTROL_SIGNAL_BYTES)")
    A("            l2_mlp_tmp = pld.alloc_window_buffer(BATCH * HIDDEN * 2)")
    A("            l2_mlp_sig = pld.alloc_window_buffer(COMM_CONTROL_SIGNAL_BYTES)")

    def dense_win(g):
        return (
            f"                    pld.window({g}_attn_tmp, [BATCH, HIDDEN], dtype=pl.BF16),\n"
            f"                    pld.window({g}_attn_sig, [tp_size, 1], dtype=pl.INT32),\n"
            f"                    pld.window({g}_mlp_tmp, [BATCH, HIDDEN], dtype=pl.BF16),\n"
            f"                    pld.window({g}_mlp_sig, [tp_size, 1], dtype=pl.INT32),"
        )

    # ── Per-layer distinct hidden buffers (fixes the RAW-only-v1 WAR/WAW race of
    # the old 2-buffer ping-pong across 45 sequential layers). Each hidden buffer
    # is written EXACTLY ONCE and read only by the next layer, so the RAW-only-v1
    # runtime (single-value producer_index; ADR-013 non-aliasing precondition)
    # serialises the chain correctly. Dense L0->h_d0, L1->h_mid_out (Out, written
    # once, read once by L2), L2->(next_hidden_out if no MoE layers else h_d2);
    # MoE layers chain through create_tensor h_moe_L*. The LAST executed layer
    # writes next_hidden_out (program Out) for the tail lm_head + harness readback.
    A("            h_d0 = pl.create_tensor([tp_size, BATCH, HIDDEN], dtype=pl.BF16)")
    A("            h_d2 = pl.create_tensor([tp_size, BATCH, HIDDEN], dtype=pl.BF16)")
    # All per-layer hidden buffers are defined UNCONDITIONALLY at function-body
    # level so they are defined on every path (pypto SSA requires a variable be
    # defined on all paths reaching its use; buffers defined inside an `if pos<N`
    # block are not visible to the next layer's block -> "used outside defining
    # scope"). Unused ones (gated-out layers / the last layer that writes
    # next_hidden_out instead) are dead locals and DCE'd.
    for _p in range(len(moe)):
        A(f"            h_moe_L{_p} = pl.create_tensor([tp_size, BATCH, HIDDEN], dtype=pl.BF16)")
    # Per-layer write-once residual-hold buffers (attention residual saved by the
    # fused orch's stash, read by its section-D residual add). Distinct per layer
    # so no cross-layer WAW; external (param) so the orch's internal MoE scratch
    # does not reuse/clobber them.
    for _p in range(len(moe)):
        A(f"            resid_hold_L{_p} = pl.create_tensor([tp_size, BATCH, HIDDEN], dtype=pl.BF16)")
    A("            # ---- L0 full-dense: current_hidden -> h_d0. ----")
    A("            for r in pl.range(pld.world_size()):")
    A("                self.full_chip_orch(")
    A("                    current_hidden[r], input_rms[r],")
    A("                    full_wq[r, 0], full_wk[r, 0], full_wv[r, 0],")
    A("                    q_norm[r], k_norm[r],")
    A("                    seq_lens[r], block_table[r], slot_mapping[r],")
    A("                    rope_cos_full[r], rope_sin_full[r], k_cache[r], v_cache[r],")
    A("                    full_wo[r, 0], full_w_g[r, 0], full_gate_r[r, 0],")
    A("                    post_rms[r], dense_w_gate[r, 0], dense_w_up[r, 0], dense_w_down[r, 0],")
    A("                    h_d0[r],")
    A(dense_win("l0"))
    A("                    0, 0, 0, r, device=r,")
    A("                )")
    A("            # ---- L1 swa-dense: h_d0 -> h_mid_out. ----")
    A("            for r in pl.range(pld.world_size()):")
    A("                self.swa_chip_orch(")
    A("                    h_d0[r], input_rms[r],")
    A("                    swa_wq[r, 0], swa_wk[r, 0], swa_wv[r, 0],")
    A("                    q_norm[r], k_norm[r],")
    A("                    seq_lens[r], block_table[r], slot_mapping[r],")
    A("                    rope_cos_swa[r], rope_sin_swa[r], k_cache[r], v_cache[r],")
    A("                    swa_wo[r, 0], swa_w_g[r, 0], swa_gate_r[r, 0],")
    A("                    post_rms[r], dense_w_gate[r, 1], dense_w_up[r, 1], dense_w_down[r, 1],")
    A("                    h_mid_out[r],")
    A(dense_win("l1"))
    A("                    1, 0, 0, r, device=r,")
    A("                )")
    A("            # ---- L2 swa-dense: h_mid_out -> next_hidden_out (0 MoE) or h_d2. ----")
    # dst passed DIRECTLY indexed (bare alias `_l2_dst = h_d2` breaks device
    # host_orch codegen — see MoE loop note). Compile-time branch picks the dst.
    def _emit_l2(dst_expr, pad):
        A(f"{pad}            for r in pl.range(pld.world_size()):")
        A(f"{pad}                self.swa_chip_orch(")
        A(f"{pad}                    h_mid_out[r], input_rms[r],")
        A(f"{pad}                    swa_wq[r, 1], swa_wk[r, 1], swa_wv[r, 1],")
        A(f"{pad}                    q_norm[r], k_norm[r],")
        A(f"{pad}                    seq_lens[r], block_table[r], slot_mapping[r],")
        A(f"{pad}                    rope_cos_swa[r], rope_sin_swa[r], k_cache[r], v_cache[r],")
        A(f"{pad}                    swa_wo[r, 1], swa_w_g[r, 1], swa_gate_r[r, 1],")
        A(f"{pad}                    post_rms[r], dense_w_gate[r, 2], dense_w_up[r, 2], dense_w_down[r, 2],")
        A(f"{pad}                    {dst_expr}[r],")
        A(dense_win("l2") if pad == "" else "\n".join("    " + ln for ln in dense_win("l2").split("\n")))
        # attn_layer_idx MUST be 0: wq/wk/wv/w_g are pre-sliced to this layer at the
        # call site (swa_wq[r, 1]); attention_swa uses layer_hidden_base =
        # attn_layer_idx*HIDDEN as a K-offset into those weights, so a non-zero index
        # slices PAST the single-layer weight into the next layer's contiguous data
        # (was 1 -> read layer-3 weights -> L2 cos 0.93). L1 + all 42 MoE layers pass 0.
        A(f"{pad}                    2, 0, 0, r, device=r,")
        A(f"{pad}                )")
    A("            if _FAITHFUL_MOE_LAYERS == 0:")
    _emit_l2("next_hidden_out", "    ")
    A("            else:")
    _emit_l2("h_d2", "    ")

    # MoE layers chain through per-layer DISTINCT create_tensor hidden buffers
    # (each written once) instead of the old 2-buffer ping-pong:
    #   pos 0 reads h_d2 (dense L2 output); pos k reads h_moe_L{k-1}.
    #   the LAST executed layer writes next_hidden_out (program Out); others
    #   write their own h_moe_L{pos}. This makes every inter-layer hidden buffer
    #   non-aliasing (RAW-only-v1 precondition), removing the WAR race that made
    #   the ping-pong output explode + go nondeterministic past ~3 layers.
    # Each MoE layer is ONE fused orch (attn->resid1 local -> MoE -> residual),
    # so the attn->MoE handoff never crosses an orch boundary (M3b fix).
    for pos, Labs in enumerate(moe):
        sfx = f"L{pos}"
        src = "h_d2" if pos == 0 else f"h_moe_L{pos - 1}"
        wpre = "full" if is_full[Labs] else "swa"
        li = full_local[Labs] if is_full[Labs] else swa_local[Labs]
        orch = f"{wpre}_moe_chip_orch"
        A(f"            if {pos} < _FAITHFUL_MOE_LAYERS:")
        for buf, sz in [
            ("attn_tmp_buf", "BATCH * HIDDEN * 2"),
            ("attn_sig_buf", "COMM_CONTROL_SIGNAL_BYTES"),
            ("pub_counts_buf", "n_ranks * n_ranks * n_local_experts_pad * 4"),
            ("count_done_buf", "COMM_CONTROL_SIGNAL_BYTES"),
            ("recv_x_buf", "local_recv_max * HIDDEN * 1"),
            ("recv_scale_buf", "local_recv_max * 8 * 4"),
            ("recv_r_route_buf", "local_recv_max * idx_pad * 4"),
            ("data_done_buf", "COMM_CONTROL_SIGNAL_BYTES"),
            ("send_x_buf", "local_recv_max * HIDDEN * 1"),
            ("send_scale_buf", "local_recv_max * 8 * 4"),
            ("send_route_buf", "local_recv_max * idx_pad * 4"),
            ("sh_tmp_buf", "BATCH * HIDDEN * 2"),
            ("sh_sig_buf", "COMM_CONTROL_SIGNAL_BYTES"),
            ("routed_y_window_buf", "n_routes_per_rank * HIDDEN * 2"),
            ("combine_done_buf", "COMM_CONTROL_SIGNAL_BYTES"),
            ("routed_src_window_buf", "local_recv_max * HIDDEN * 2"),
        ]:
            A(f"                {buf}_{sfx} = pld.alloc_window_buffer({sz})")
        # dst passed DIRECTLY indexed: create_tensor buffers survive the device
        # host_orch codegen only as tensors[...] entries accessed by index; a bare
        # alias `_x = h_moe_L{pos}` emits an undefined bare name -> NameError at
        # device runtime. So branch the whole call at compile time instead of
        # aliasing a dst variable. The LAST executed layer writes next_hidden_out
        # (program Out, read by tail+harness); every other layer writes its own
        # distinct h_moe_L{pos}.
        def _emit_moe_call(dst_expr, pad):
            A(f"{pad}                for r in pl.range(pld.world_size()):")
            A(f"{pad}                    attn_tmp_window = pld.window(attn_tmp_buf_{sfx}, [BATCH, HIDDEN], dtype=pl.BF16)")
            A(f"{pad}                    attn_signal_window = pld.window(attn_sig_buf_{sfx}, [tp_size, 1], dtype=pl.INT32)")
            A(f"{pad}                    pub_counts = pld.window(pub_counts_buf_{sfx}, [n_ranks * n_ranks, n_local_experts_pad], dtype=pl.INT32)")
            A(f"{pad}                    count_done_sig = pld.window(count_done_buf_{sfx}, [n_ranks, 1], dtype=pl.INT32)")
            A(f"{pad}                    recv_x = pld.window(recv_x_buf_{sfx}, [local_recv_max, HIDDEN], dtype=pl.INT8)")
            A(f"{pad}                    recv_scale = pld.window(recv_scale_buf_{sfx}, [local_recv_max, 8], dtype=pl.FP32)")
            A(f"{pad}                    data_done_sig = pld.window(data_done_buf_{sfx}, [n_ranks, 1], dtype=pl.INT32)")
            A(f"{pad}                    recv_r_route = pld.window(recv_r_route_buf_{sfx}, [local_recv_max, idx_pad], dtype=pl.INT32)")
            A(f"{pad}                    send_x = pld.window(send_x_buf_{sfx}, [local_recv_max, HIDDEN], dtype=pl.INT8)")
            A(f"{pad}                    send_scale = pld.window(send_scale_buf_{sfx}, [local_recv_max, 8], dtype=pl.FP32)")
            A(f"{pad}                    send_route = pld.window(send_route_buf_{sfx}, [local_recv_max, idx_pad], dtype=pl.INT32)")
            A(f"{pad}                    sh_tmp_window = pld.window(sh_tmp_buf_{sfx}, [BATCH, HIDDEN], dtype=pl.BF16)")
            A(f"{pad}                    sh_signal_window = pld.window(sh_sig_buf_{sfx}, [n_ranks, 1], dtype=pl.INT32)")
            A(f"{pad}                    routed_y_buf = pld.window(routed_y_window_buf_{sfx}, [n_routes_per_rank, HIDDEN], dtype=pl.BF16)")
            A(f"{pad}                    combine_done_sig = pld.window(combine_done_buf_{sfx}, [n_ranks, 1], dtype=pl.INT32)")
            A(f"{pad}                    routed_src_buf = pld.window(routed_src_window_buf_{sfx}, [local_recv_max, HIDDEN], dtype=pl.BF16)")
            A(f"{pad}                    self.{orch}(")
            A(f"{pad}                        {src}[r], input_rms[r],")
            A(f"{pad}                        {wpre}_wq[r, {li}], {wpre}_wk[r, {li}], {wpre}_wv[r, {li}], q_norm[r], k_norm[r],")
            A(f"{pad}                        seq_lens[r], block_table[r], slot_mapping[r],")
            A(f"{pad}                        rope_cos_{wpre}[r], rope_sin_{wpre}[r], k_cache[r], v_cache[r],")
            A(f"{pad}                        {wpre}_wo[r, {li}], {wpre}_w_g[r, {li}], {wpre}_gate_r[r, {li}],")
            A(f"{pad}                        post_rms[r],")
            A(f"{pad}                        moe_gate_w[r, {pos}], moe_router_bias[r, {pos}],")
            A(f"{pad}                        moe_w_gate_r[r, {pos}], moe_w_gate_r_scale[r, {pos}], moe_w_up_r[r, {pos}], moe_w_up_r_scale[r, {pos}], moe_w_down_r[r, {pos}], moe_w_down_r_scale[r, {pos}],")
            A(f"{pad}                        moe_w_gate_s[r, {pos}], moe_w_up_s[r, {pos}], moe_w_down_s[r, {pos}], {dst_expr}[r],")
            A(f"{pad}                        dbg_out[r],")
            A(f"{pad}                        resid_hold_{sfx}[r],")
            A(f"{pad}                        attn_tmp_window, attn_signal_window, pub_counts, count_done_sig,")
            A(f"{pad}                        recv_x, recv_scale, data_done_sig, recv_r_route,")
            A(f"{pad}                        send_x, send_scale, send_route,")
            A(f"{pad}                        sh_tmp_window, sh_signal_window, routed_y_buf, combine_done_sig,")
            A(f"{pad}                        routed_src_buf,")
            A(f"{pad}                        {Labs}, 0, r, device=r,")
            A(f"{pad}                    )")
        A(f"                # ---- layer {Labs}: {wpre}-attn + MoE FUSED (pos={pos}); {src} -> dst. ----")
        A(f"                if {pos} == _FAITHFUL_MOE_LAYERS - 1:")
        _emit_moe_call("next_hidden_out", "    ")
        A("                else:")
        _emit_moe_call(f"h_moe_{sfx}", "    ")

    A("            # ── Tail: final RMSNorm + LM head on every rank. With per-layer")
    A("            # distinct buffers the LAST executed layer always wrote")
    A("            # next_hidden_out, so the tail unconditionally reads it. ──")
    A("            for rt in pl.range(pld.world_size()):")
    A("                self.lm_head_orch(")
    A("                    next_hidden_out[rt], final_norm_weight[rt], lm_head_weight[rt],")
    A("                    seq_lens[rt], logits_shard_out[rt], device=rt,")
    A("                )")
    return "\n".join(L) + "\n"


def main() -> int:
    cls = _layer_classification()
    text = SRC.read_text()
    if "_build_whole_decode_faithful_real_program" in text:
        print("[gen] real builder already present — refusing (idempotent).")
        return 1

    b0 = text.index(DEF_MARKER)
    b1 = text.index(BIND_MARKER)
    builder = text[b0:b1]

    chip_at = builder.index(CHIP_MARKER)
    lm_at = builder.index(LMHEAD_MARKER)
    dense_methods_at = builder.index(DENSE_METHODS_MARKER)

    head_and_setA = builder[:chip_at]
    chip_orch_text = builder[chip_at:lm_at]
    lm_head_text = builder[lm_at:dense_methods_at]

    head_and_setA = head_and_setA.replace(
        "def _build_whole_decode_faithful_program(",
        "def _build_whole_decode_faithful_real_program(",
    ).replace(
        "class WholeDecodeFaithful:",
        "class WholeDecodeFaithfulReal:",
    )
    _signal_policy_anchor = (
        "    per_rank_buckets = PER_RANK_BUCKETS  # n_ranks * n_local_experts\n"
        "\n"
        "    @pl.program\n"
        "    class WholeDecodeFaithfulReal:\n"
    )
    _signal_policy_replacement = (
        "    per_rank_buckets = PER_RANK_BUCKETS  # n_ranks * n_local_experts\n"
        "    # A2/A3 comm-domain buffers are carved sequentially without per-slot\n"
        "    # alignment. Reserve one full L2 cache line for every cross-rank control\n"
        "    # signal so AtomicAdd/TWAIT traffic cannot share a line with adjacent\n"
        "    # control or data windows. The logical tensor view remains [8, 1] INT32.\n"
        "    COMM_CONTROL_SIGNAL_BYTES = 512\n"
        "\n"
        "    @pl.program\n"
        "    class WholeDecodeFaithfulReal:\n"
    )
    assert head_and_setA.count(_signal_policy_anchor) == 1
    head_and_setA = head_and_setA.replace(
        _signal_policy_anchor, _signal_policy_replacement, 1,
    )

    # ── A1: replace the base builder's in-expert-quant `_expert_routed`
    # with the byte-faithful moe.py device-PASS version (FRESH_EXPERT_ROUTED:
    # flat pre-quant of the whole recv buffer + INT8 gate/up/down + bare-slice
    # h-requant). The @pl.function decorator + comment before `def _expert_routed`
    # are kept; only the function body (def .. return local_routed_y) is swapped.
    _er_start = head_and_setA.index("        def _expert_routed(  # noqa: PLR0913, PLR0915")
    _ers_marker = (
        "        @pl.function(type=pl.FunctionType.Inline)\n"
        "        def expert_routed_step("
    )
    _er_end = head_and_setA.index(_ers_marker, _er_start)
    head_and_setA = (
        head_and_setA[:_er_start]
        + FRESH_EXPERT_ROUTED
        + "\n\n"
        + head_and_setA[_er_end:]
    )
    # ── Combine DMA fence: restore moe.py's self-notify AtomicAdd+0 drain
    # (mirrors combine.cpp:177 pipe_barrier between the last TPUT and TNOTIFY).
    # The whole-net base `_push_routed_y_to_sources` DROPPED it: `remote_store`
    # (TPUT) has no trailing dsb (pto-isa TPut.hpp), so `combine_done` can outrun
    # the push DMA -> the consumer `_weighted_gather_and_add` gathers a STALE
    # routed_y_buf -> racy `moe_out` (the M4 nondeterminism; device 1000 vs 1008).
    # A notify emits dsb(DSB_DDR)+pipe_barrier(PIPE_ALL) (TNotify.hpp) draining the
    # TPUTs. combine_done[my_rank,0] is never waited (waiters use src!=my_rank),
    # so AtomicAdd+0 is a pure no-op fence write. (SKILL D / P4.)
    _push_notify = (
        "                e_cursor = e_cursor + total_e\n"
        "\n"
        "            for peer in pl.range(n_ranks):\n"
    )
    _push_fence = (
        "                e_cursor = e_cursor + total_e\n"
        "\n"
        "            pld.system.notify(\n"
        "                target=combine_done, peer=my_rank,\n"
        "                offsets=[my_rank, 0], value=0, op=pld.NotifyOp.AtomicAdd,\n"
        "            )\n"
        "            for peer in pl.range(n_ranks):\n"
    )
    assert head_and_setA.count(_push_notify) == 1, head_and_setA.count(_push_notify)
    head_and_setA = head_and_setA.replace(_push_notify, _push_fence, 1)
    # ── 1A dispatch-side INT8: quant post_norm→INT8+scale in chip_orch BEFORE
    # dispatch and carry INT8 recv_x + per-token FP32 scale through the final
    # pull boundary. This preserves native W8A8 and reduces the communication
    # footprint. It remains a required design choice, but was not the final
    # 512B control-signal isolation variable.

    # PULL: replace base _dispatch_publish + _dispatch_push (PUSH scatter) with
    # the INT8 quant kernel + FRESH pull methods (_dispatch_pack_publish +
    # _dispatch_pull). _histogram_and_prefix_sum / _build_local_expert_csr (above)
    # are reused; the exact device-verified stage-2 boundary is emitted below.
    _pub_dec = (
        "        @pl.function(type=pl.FunctionType.InCore)\n"
        "        def _dispatch_publish(\n"
    )
    _stage3_marker = "        # ---------- Stage 3a: expert_routed (local 36 experts) ----------\n"
    assert head_and_setA.count(_pub_dec) == 1, head_and_setA.count(_pub_dec)
    assert head_and_setA.count(_stage3_marker) == 1, head_and_setA.count(_stage3_marker)
    _pub_at = head_and_setA.index(_pub_dec)
    _stage3_at = head_and_setA.index(_stage3_marker)
    assert _stage3_at > _pub_at, (_stage3_at, _pub_at)
    head_and_setA = (
        head_and_setA[:_pub_at]
        + FRESH_QUANT_MOE_INPUT + "\n"
        + FRESH_DISPATCH_PULL_INT8.strip("\n") + "\n"
        + head_and_setA[_stage3_at:]
    )

    def _edit_span(text, start_anchor, end_anchor, fn):
        s = text.index(start_anchor)
        e = text.index(end_anchor, s + len(start_anchor))
        return text[:s] + fn(text[s:e]) + text[e:]

    # Stage-2 signatures and wiring are canonical in FRESH_DISPATCH_PULL_INT8.

    # expert_routed_step wrapper: FRESH_EXPERT_ROUTED already consumes INT8
    # local_routed_x + scale, but its caller wrapper still had the BF16 sig +
    # scale-less _expert_routed call — thread INT8 + local_routed_x_scale.
    def _ers_edits(seg):
        _s = "            local_routed_x: pl.Tensor[[local_recv_max, HIDDEN], pl.BF16],\n"
        _s_new = ("            local_routed_x: pl.Tensor[[local_recv_max, HIDDEN], pl.INT8],\n"
                  "            local_routed_x_scale: pl.Tensor[[1, local_recv_max], pl.FP32],\n")
        assert seg.count(_s) == 1, ("ers sig", seg.count(_s))
        seg = seg.replace(_s, _s_new, 1)
        _c = ("            local_routed_y = self._expert_routed(\n"
              "                local_routed_x,\n")
        _c_new = ("            local_routed_y = self._expert_routed(\n"
                  "                local_routed_x,\n"
                  "                local_routed_x_scale,\n")
        assert seg.count(_c) == 1, ("ers call", seg.count(_c))
        return seg.replace(_c, _c_new, 1)
    head_and_setA = _edit_span(
        head_and_setA,
        "        def expert_routed_step(\n",
        "        # ---------- Stage 3b: expert_shared",
        _ers_edits,
    )

    # ── PULL combine (step 5): splice _stage_routed_src + _pull_routed_y before
    # combine_step; rewrite combine_step to stage->consume inverse_map->pull->gather
    # + add expert_indices/inverse_map/routed_src_buf params. The dispatch task
    # owns inverse-map construction; combine only consumes it.
    _cs_dec = (
        "        @pl.function(type=pl.FunctionType.Inline)\n"
        "        def combine_step(  # noqa: PLR0913\n"
    )
    assert head_and_setA.count(_cs_dec) == 1, head_and_setA.count(_cs_dec)
    head_and_setA = head_and_setA.replace(
        _cs_dec, FRESH_COMBINE_PULL.strip("\n") + "\n" + _cs_dec, 1,
    )
    # combine_step sig: add expert_indices + routed_src_buf (multiline
    # combine_done_sig + my_rank is unique to combine_step).
    _cs_sig = ("            combine_done_sig: pld.DistributedTensor[\n"
               "                [n_ranks, 1], pl.INT32\n"
               "            ],\n"
               "            my_rank: pl.Scalar[pl.INT32],\n")
    _cs_sig_new = ("            combine_done_sig: pld.DistributedTensor[\n"
                   "                [n_ranks, 1], pl.INT32\n"
                   "            ],\n"
                   "            expert_indices: pl.Tensor[[BATCH, TOPK], pl.INT32],\n"
                   "            inverse_map: pl.Tensor[[BATCH, TOPK], pl.INT32],\n"
                   "            routed_src_buf: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.BF16],\n"
                   "            my_rank: pl.Scalar[pl.INT32],\n")
    assert head_and_setA.count(_cs_sig) == 1, head_and_setA.count(_cs_sig)
    head_and_setA = head_and_setA.replace(_cs_sig, _cs_sig_new, 1)
    # combine_step body: push -> stage + inverse_map + pull.
    _cs_body = ("            self._zero_routed_y_buf(routed_y_buf)\n"
                "            self._push_routed_y_to_sources(\n"
                "                local_routed_y,\n"
                "                pub_counts,\n"
                "                routed_y_buf,\n"
                "                combine_done_sig,\n"
                "                recv_r_route_out,\n"
                "                my_rank,\n"
                "            )\n"
                "\n"
                "            moe_out = self._weighted_gather_and_add(\n"
                "                routed_y_buf, expert_weights, sh_y, moe_out,\n"
                "            )\n"
                "            return moe_out\n")
    _cs_body_new = ("            self._zero_routed_y_buf(routed_y_buf)\n"
                    "            self._stage_routed_src(local_routed_y, routed_src_buf)\n"
                    "            self._pull_routed_y(\n"
                    "                routed_src_buf, expert_indices, inverse_map,\n"
                    "                routed_y_buf, combine_done_sig, my_rank,\n"
                    "            )\n"
                    "            moe_out = self._weighted_gather_and_add(\n"
                    "                routed_y_buf, expert_weights, sh_y, moe_out,\n"
                    "            )\n"
                    "            return moe_out\n")
    assert head_and_setA.count(_cs_body) == 1, head_and_setA.count(_cs_body)
    head_and_setA = head_and_setA.replace(_cs_body, _cs_body_new, 1)
    _cs_comment = (
        "            # Push design: r_route rode with each token into recv_r_route at\n"
        "            # dispatch, so combine drops the src_route_table publish + its\n"
        "            # barrier and scatters the routed output straight back.\n"
        "            # Zero routed_y_buf first: unwritten slots (not targeted by any push)\n"
        "            # else feed uninitialised garbage into _weighted_gather_and_add.\n"
    )
    _cs_comment_new = (
        "            # Pull design: expert holders stage routed output locally; source\n"
        "            # ranks use dispatch-produced inverse_map entries to pull rows back.\n"
    )
    assert head_and_setA.count(_cs_comment) == 1, head_and_setA.count(_cs_comment)
    head_and_setA = head_and_setA.replace(_cs_comment, _cs_comment_new, 1)

    # Knob-strip is toggleable: GEN_STRIP_KNOBS=0 keeps the _MOE_{NORM_ONLY,
    # SHARED_ONLY} bisect knobs in the fused MoE body (diagnostic regen); the
    # default =1 strips them for a production-clean builder.
    _renamed = chip_orch_text.replace("layer_idx", "norm_layer_idx")
    if os.environ.get("GEN_STRIP_KNOBS", "1") == "1":
        chip_orch_text = _strip_moe_knobs(_renamed)
    else:
        chip_orch_text = _renamed

    # ── 1A chip_orch dispatch-side INT8 wiring. Quant post_norm→INT8+scale
    # BEFORE dispatch; carry INT8 local_routed_x + a [1,local_recv_max] scale
    # through dispatch_step → expert_routed_step (router `_gate` + shared already
    # consumed the ORIGINAL BF16 post_norm). Edits apply to chip_orch_text (which
    # is emitted directly AND sliced into the fused moe_body).
    _q_anchor = "            # 3) Dispatch (EP push: tokens remote_store'd into peer recv_x).\n"
    _q_new = (
        "            # 1A: per-token INT8 dynamic-quant of the MoE input BEFORE\n"
        "            # dispatch (dispatch-side; shrinks recv_x 8→4MB/layer).\n"
        "            x_disp_i8 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.INT8)\n"
        "            x_disp_scale = pl.create_tensor([BATCH, 8], dtype=pl.FP32)\n"
        "            (x_disp_i8, x_disp_scale) = self._quant_moe_input(\n"
        "                post_norm, x_disp_i8, x_disp_scale,\n"
        "            )\n"
        "            # 3) Dispatch (EP fixed-slot pull).\n"
    )
    assert chip_orch_text.count(_q_anchor) == 1, chip_orch_text.count(_q_anchor)
    chip_orch_text = chip_orch_text.replace(_q_anchor, _q_new, 1)

    _lrx_c = ("            local_routed_x = pl.create_tensor(\n"
              "                [local_recv_max, HIDDEN], dtype=pl.BF16,\n"
              "            )\n")
    _lrx_c_new = ("            local_routed_x = pl.create_tensor(\n"
                  "                [local_recv_max, HIDDEN], dtype=pl.INT8,\n"
                  "            )\n"
                  "            local_routed_x_scale = pl.create_tensor(\n"
                  "                [1, local_recv_max], dtype=pl.FP32,\n"
                  "            )\n")
    assert chip_orch_text.count(_lrx_c) == 1, chip_orch_text.count(_lrx_c)
    chip_orch_text = chip_orch_text.replace(_lrx_c, _lrx_c_new, 1)

    _dcall = ("            (\n"
              "                local_routed_x,\n"
              "                local_expert_offset,\n"
              "                local_expert_count,\n"
              "                recv_r_route_out,\n"
              "            ) = self.dispatch_step(\n"
              "                post_norm, expert_indices,\n"
              "                local_routed_x,\n"
              "                local_expert_offset, local_expert_count, recv_r_route_out,\n"
              "                pub_counts, count_done_sig, recv_x, recv_r_route, data_done_sig,\n"
              "                my_rank,\n"
              "            )\n")
    _dcall_new = ("            (\n"
                  "                local_routed_x,\n"
                  "                local_routed_x_scale,\n"
                  "                local_expert_offset,\n"
                  "                local_expert_count,\n"
                  "                recv_r_route_out,\n"
                  "                inverse_map,\n"
                  "            ) = self.dispatch_step(\n"
                  "                x_disp_i8, x_disp_scale, expert_indices,\n"
                  "                local_routed_x, local_routed_x_scale,\n"
                  "                local_expert_offset, local_expert_count, recv_r_route_out,\n"
                  "                pub_counts, count_done_sig, recv_x, recv_scale, recv_r_route, data_done_sig,\n"
                  "                send_x, send_scale, send_route,\n"
                  "                my_rank,\n"
                  "            )\n")
    assert chip_orch_text.count(_dcall) == 1, chip_orch_text.count(_dcall)
    chip_orch_text = chip_orch_text.replace(_dcall, _dcall_new, 1)

    _ecall = ("            local_routed_y = self.expert_routed_step(\n"
              "                local_routed_x,\n"
              "                local_expert_offset, local_expert_count,\n")
    _ecall_new = ("            local_routed_y = self.expert_routed_step(\n"
                  "                local_routed_x,\n"
                  "                local_routed_x_scale,\n"
                  "                local_expert_offset, local_expert_count,\n")
    assert chip_orch_text.count(_ecall) == 1, chip_orch_text.count(_ecall)
    chip_orch_text = chip_orch_text.replace(_ecall, _ecall_new, 1)

    # Keep the (dead-but-emitted) base chip_orch signature self-consistent with
    # its now-INT8 dispatch body (recv_x INT8 + recv_scale param), so it stays
    # compilable even if the frontend traces it. This is BEFORE _b_marker, so the
    # extracted moe_body (which lives in the recv_scale-carrying fused orchs) is
    # unaffected.
    _co_rx = ("            recv_x: pld.DistributedTensor[\n"
              "                [local_recv_max, HIDDEN], pl.BF16\n"
              "            ],\n")
    _co_rx_new = ("            recv_x: pld.DistributedTensor[\n"
                  "                [local_recv_max, HIDDEN], pl.INT8\n"
                  "            ],\n"
                  "            recv_scale: pld.DistributedTensor[[local_recv_max, 8], pl.FP32],\n"
                  "            send_x: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.INT8],\n"
                  "            send_scale: pld.DistributedTensor[[local_recv_max, 8], pl.FP32],\n"
                  "            send_route: pld.DistributedTensor[[local_recv_max, idx_pad], pl.INT32],\n")
    assert chip_orch_text.count(_co_rx) == 1, chip_orch_text.count(_co_rx)
    chip_orch_text = chip_orch_text.replace(_co_rx, _co_rx_new, 1)

    # ── PULL combine chip_orch wiring: add routed_src_buf to the base chip_orch
    # sig (dead-but-emitted self-consistency) + thread expert_indices +
    # routed_src_buf into the combine_step call (this call lives in moe_body, so
    # the edit propagates to the fused orchs when moe_body is sliced below).
    _co_cds = ("            combine_done_sig: pld.DistributedTensor[\n"
               "                [n_ranks, 1], pl.INT32\n"
               "            ],\n"
               "            norm_layer_idx: pl.Scalar[pl.INT32],\n")
    _co_cds_new = ("            combine_done_sig: pld.DistributedTensor[\n"
                   "                [n_ranks, 1], pl.INT32\n"
                   "            ],\n"
                   "            routed_src_buf: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.BF16],\n"
                   "            norm_layer_idx: pl.Scalar[pl.INT32],\n")
    assert chip_orch_text.count(_co_cds) == 1, chip_orch_text.count(_co_cds)
    chip_orch_text = chip_orch_text.replace(_co_cds, _co_cds_new, 1)

    _combine_call = ("            moe_out = self.combine_step(\n"
                     "                local_routed_y,\n"
                     "                recv_r_route_out, expert_weights, sh_y,\n"
                     "                moe_out,\n"
                     "                pub_counts,\n"
                     "                routed_y_buf, combine_done_sig,\n"
                     "                my_rank,\n"
                     "            )\n")
    _combine_call_new = ("            moe_out = self.combine_step(\n"
                         "                local_routed_y,\n"
                         "                recv_r_route_out, expert_weights, sh_y,\n"
                         "                moe_out,\n"
                         "                pub_counts,\n"
                         "                routed_y_buf, combine_done_sig,\n"
                         "                expert_indices, inverse_map, routed_src_buf,\n"
                         "                my_rank,\n"
                         "            )\n")
    assert chip_orch_text.count(_combine_call) == 1, chip_orch_text.count(_combine_call)
    chip_orch_text = chip_orch_text.replace(_combine_call, _combine_call_new, 1)
    _combine_label = (
        "            # 5) Combine (EP push back + weighted gather + sh_y add).\n"
    )
    _combine_label_new = (
        "            # 5) Combine (EP pull back + weighted gather + sh_y add).\n"
    )
    assert chip_orch_text.count(_combine_label) == 1, chip_orch_text.count(_combine_label)
    chip_orch_text = chip_orch_text.replace(
        _combine_label, _combine_label_new, 1,
    )

    # Extract the shared MoE body (post_norm -> gate/shared/dispatch/routed/
    # combine -> residual) from chip_orch and splice it after each fused
    # attention front. Anchor the end on section D's ``moe_residual_add`` scope
    # so the slice is correct even when the _MOE_* knobs are kept (their
    # 16-space ``return next_hidden_out`` would otherwise substring-match a bare
    # 12-space search and truncate the body before shared/dispatch/routed/D).
    _b_marker = "            # ── B: post-attention"
    _d_marker = 'name_hint="moe_residual_add"'
    _ret = "            return next_hidden_out\n"
    _bs = chip_orch_text.index(_b_marker)
    _dm = chip_orch_text.index(_d_marker, _bs)
    _be = chip_orch_text.index(_ret, _dm) + len(_ret)
    moe_body = chip_orch_text[_bs:_be]
    # Residual protection (fused orch only): section D reads the residual from
    # the dedicated write-once resid_hold buffer (the _fused_moe_head stashed
    # resid1 there) instead of the local resid1_fp32, which gate/shared InCore
    # scratch clobbers inside the enlarged fused orch. resid_hold (not
    # next_hidden_out) avoids the WAW that made the output nondeterministic.
    # Only the "r = pl.slice(resid1_fp32" in section D matches (B uses
    # ck/norm_chunk, SHARED_ONLY uses _sr).
    moe_body = moe_body.replace(
        "r = pl.slice(resid1_fp32, [BATCH, K_CHUNK], [0, k0])",
        "r = pl.cast(pl.slice(resid_hold, [BATCH, K_CHUNK], [0, k0]), "
        "target_type=pl.FP32)",
    )

    # E2 op-level dumps: inject a `dbg_out` write right AFTER each stage
    # (post_norm / sh_y / moe_out) so the harness can read that stage's value
    # via a SEPARATE Out (reliable — no early-return; the write RAW-depends on
    # the stage output). Compile-time `_DBG_STAGE` selects which one is emitted.
    def _dbg(stage: str, src: str) -> str:
        return (
            f"            if _DBG_STAGE == {stage}:\n"
            f'                with pl.at(level=pl.Level.CORE_GROUP, name_hint="dbg_dump_{stage}"):\n'
            f"                    for _dg in pl.range(HIDDEN // K_CHUNK):\n"
            f"                        _dg0 = _dg * K_CHUNK\n"
            f"                        dbg_out = pl.assemble(dbg_out, pl.slice("
            f"{src}, [BATCH, K_CHUNK], [0, _dg0]), [0, _dg0])\n"
        )
    moe_body = moe_body.replace(
        "            # ── C: EP+TP MoE",
        _dbg("1", "post_norm") + "            # ── C: EP+TP MoE", 1,
    )
    moe_body = moe_body.replace(
        "            # 3) Dispatch",
        _dbg("2", "sh_y") + "            # 3) Dispatch", 1,
    )
    moe_body = moe_body.replace(
        "            # ── D: residual",
        _dbg("4", "moe_out") + "            # ── D: residual", 1,
    )
    # (1A: stage-5 dump of local_routed_x dropped — it is INT8 post-dispatch now,
    # which cannot assemble into the FP32 dbg_out; stage 3 = local_routed_y is the
    # routed-output magnitude check used for P=1 verification.)
    moe_body = moe_body.replace(
        "            # 5) Combine",
        _dbg("3", "local_routed_y") + "            # 5) Combine", 1,
    )
    fused_full = _fused_moe_head("full") + moe_body
    fused_swa = _fused_moe_head("swa") + moe_body

    new_builder = (
        head_and_setA
        + chip_orch_text
        + lm_head_text
        + FRESH_FULL_CHIP_ORCH
        + FRESH_SWA_CHIP_ORCH
        + fused_full
        + fused_swa
        + _host_orch(cls)
        + "\n    return WholeDecodeFaithfulReal\n"
    )

    binding = "\n\nwhole_decode_faithful_real = _build_whole_decode_faithful_real_program()\n"

    nl = text.index("\n", text.index(BIND_MARKER) + 1)
    new_text = text[: nl + 1] + "\n\n" + new_builder + binding + text[nl + 1:]


    # Dispatch and inverse-map task boundaries are canonical in the templates above.

    # Restore signed tile_valid in the final generated real builder.  This avoids
    # INDEX/unsigned underflow for empty tail tiles.
    def _apply_signed_tile_valid(src: str) -> str:
        old = """                valid_rows = pl.cast(n_rows, pl.INDEX)\n\n                for tile_idx in pl.range(N_RECV_TILES):\n                    tile_row0 = tile_idx * RECV_TILE\n                    tile_offset = offset + tile_row0\n                    tile_valid = pl.min(RECV_TILE, valid_rows - tile_row0)\n                    if tile_valid > 0:\n"""
        new = """                for tile_idx in pl.range(N_RECV_TILES):\n                    tile_row0_i32 = pl.cast(tile_idx * RECV_TILE, pl.INT32)\n                    tile_rem = n_rows - tile_row0_i32\n                    if tile_rem > 0:\n                        tile_row0 = pl.cast(tile_row0_i32, pl.INDEX)\n                        tile_offset = offset + tile_row0\n                        tile_valid = pl.cast(\n                            pl.min(pl.cast(RECV_TILE, pl.INT32), tile_rem),\n                            pl.INDEX,\n                        )\n"""
        n = src.count(old)
        if n == 0:
            raise RuntimeError("signed tile patch pattern not found")
        return src.replace(old, new)

    new_text = _apply_signed_tile_valid(new_text)
    # The canonical templates include their own terminating newline. Normalize
    # the two splice joins so strip -> regenerate is byte-exact, not merely AST
    # equivalent.
    new_text = new_text.replace(
        "                inverse_map,\n"
        "            )\n"
        "        # ---------- Stage 3a: expert_routed",
        "                inverse_map,\n"
        "            )\n\n"
        "        # ---------- Stage 3a: expert_routed",
        1,
    )
    new_text = new_text.replace(
        "                        pl.store(tile_remote, [r_route, 0], routed_y_buf)\n"
        "        @pl.function(type=pl.FunctionType.Inline)\n"
        "        def combine_step",
        "                        pl.store(tile_remote, [r_route, 0], routed_y_buf)\n"
        "\n"
        "        @pl.function(type=pl.FunctionType.Inline)\n"
        "        def combine_step",
        1,
    )

    bak = SRC.with_suffix(".py.bak.pregen_real")
    if not bak.exists():
        bak.write_text(text)
    SRC.write_text(new_text.rstrip("\n") + "\n")
    print(f"[gen] wrote real builder: N_FULL={cls['N_FULL']} N_SWA={cls['N_SWA']} "
          f"N_DENSE={cls['N_DENSE']} N_MOE={cls['N_MOE']}; backup={bak.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
