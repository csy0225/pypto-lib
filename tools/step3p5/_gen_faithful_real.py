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

FRESH_FULL_ATTN_ONLY = '''
        # ---- MoE-layer full attention only (single-layer host-slice) -> resid. ----
        @pl.function(type=pl.FunctionType.Orchestration)
        def full_attn_only_orch(  # noqa: PLR0913
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
            resid3_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            attn_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
            attn_signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
            norm_layer_idx: pl.Scalar[pl.INT32],
            attn_layer_idx: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            resid3_out = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            resid3_out = attention_full_inline(
                current_hidden, input_rms_weight, wq, wk, wv,
                q_norm_weight, k_norm_weight,
                seq_lens, block_table, slot_mapping,
                rope_cos, rope_sin, k_cache, v_cache,
                wo, w_g, gate_r, resid3_out,
                norm_layer_idx, attn_layer_idx,
                attn_tmp_window, attn_signal_window, my_rank,
            )
            return resid3_out
'''

FRESH_SWA_ATTN_ONLY = '''
        # ---- MoE-layer swa attention only (single-layer host-slice) -> resid. ----
        @pl.function(type=pl.FunctionType.Orchestration)
        def swa_attn_only_orch(  # noqa: PLR0913
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
            resid3_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            attn_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
            attn_signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
            norm_layer_idx: pl.Scalar[pl.INT32],
            attn_layer_idx: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            resid3_out = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            resid3_out = attention_swa_inline(
                current_hidden, input_rms_weight, wq, wk, wv,
                q_norm_weight, k_norm_weight,
                seq_lens, block_table, slot_mapping,
                rope_cos, rope_sin, k_cache, v_cache,
                wo, w_g, gate_r, resid3_out,
                norm_layer_idx, attn_layer_idx,
                attn_tmp_window, attn_signal_window, my_rank,
            )
            return resid3_out
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
    A(f"            moe_w_gate_r: pl.Tensor[[tp_size, {N_MOE}, n_local_experts, HIDDEN, inter], pl.BF16],")
    A(f"            moe_w_up_r: pl.Tensor[[tp_size, {N_MOE}, n_local_experts, HIDDEN, inter], pl.BF16],")
    A(f"            moe_w_down_r: pl.Tensor[[tp_size, {N_MOE}, n_local_experts, inter, HIDDEN], pl.BF16],")
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
    A("            final_norm_weight: pl.Tensor[[tp_size, 1, HIDDEN], pl.FP32],")
    A("            lm_head_weight: pl.Tensor[[tp_size, VOCAB_LOCAL, HIDDEN], pl.BF16],")
    A("            logits_shard_out: pl.Out[pl.Tensor[[tp_size, USER_BATCH_DYN, VOCAB_LOCAL], pl.FP32]],")
    A("        ):")

    A("            l0_attn_tmp = pld.alloc_window_buffer(BATCH * HIDDEN * 2)")
    A("            l0_attn_sig = pld.alloc_window_buffer(tp_size * 4)")
    A("            l0_mlp_tmp = pld.alloc_window_buffer(BATCH * HIDDEN * 2)")
    A("            l0_mlp_sig = pld.alloc_window_buffer(tp_size * 4)")
    A("            l1_attn_tmp = pld.alloc_window_buffer(BATCH * HIDDEN * 2)")
    A("            l1_attn_sig = pld.alloc_window_buffer(tp_size * 4)")
    A("            l1_mlp_tmp = pld.alloc_window_buffer(BATCH * HIDDEN * 2)")
    A("            l1_mlp_sig = pld.alloc_window_buffer(tp_size * 4)")

    def dense_win(g):
        return (
            f"                    pld.window({g}_attn_tmp, [BATCH, HIDDEN], dtype=pl.BF16),\n"
            f"                    pld.window({g}_attn_sig, [tp_size, 1], dtype=pl.INT32),\n"
            f"                    pld.window({g}_mlp_tmp, [BATCH, HIDDEN], dtype=pl.BF16),\n"
            f"                    pld.window({g}_mlp_sig, [tp_size, 1], dtype=pl.INT32),"
        )

    A("            # ---- L0 full-dense: current_hidden -> next_hidden_out. ----")
    A("            for r in pl.range(pld.world_size()):")
    A("                self.full_chip_orch(")
    A("                    current_hidden[r], input_rms[r],")
    A("                    full_wq[r, 0], full_wk[r, 0], full_wv[r, 0],")
    A("                    q_norm[r], k_norm[r],")
    A("                    seq_lens[r], block_table[r], slot_mapping[r],")
    A("                    rope_cos_full[r], rope_sin_full[r], k_cache[r], v_cache[r],")
    A("                    full_wo[r, 0], full_w_g[r, 0], full_gate_r[r, 0],")
    A("                    post_rms[r], dense_w_gate[r, 0], dense_w_up[r, 0], dense_w_down[r, 0],")
    A("                    next_hidden_out[r],")
    A(dense_win("l0"))
    A("                    0, 0, 0, r, device=r,")
    A("                )")
    A("            # ---- L1 swa-dense: next_hidden_out -> h_mid_out. ----")
    A("            for r in pl.range(pld.world_size()):")
    A("                self.swa_chip_orch(")
    A("                    next_hidden_out[r], input_rms[r],")
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
    A("            # ---- L2 swa-dense: h_mid_out -> next_hidden_out. ----")
    A("            for r in pl.range(pld.world_size()):")
    A("                self.swa_chip_orch(")
    A("                    h_mid_out[r], input_rms[r],")
    A("                    swa_wq[r, 1], swa_wk[r, 1], swa_wv[r, 1],")
    A("                    q_norm[r], k_norm[r],")
    A("                    seq_lens[r], block_table[r], slot_mapping[r],")
    A("                    rope_cos_swa[r], rope_sin_swa[r], k_cache[r], v_cache[r],")
    A("                    swa_wo[r, 1], swa_w_g[r, 1], swa_gate_r[r, 1],")
    A("                    post_rms[r], dense_w_gate[r, 2], dense_w_up[r, 2], dense_w_down[r, 2],")
    A("                    next_hidden_out[r],")
    A(dense_win("l1"))
    A("                    2, 1, 0, r, device=r,")
    A("                )")

    for pos, Labs in enumerate(moe):
        sfx = f"L{pos}"
        A(f"            if {pos} < _FAITHFUL_MOE_LAYERS:")
        for buf, sz in [
            ("ad_attn_tmp_buf", "BATCH * HIDDEN * 2"),
            ("ad_attn_sig_buf", "tp_size * 4"),
            ("attn_tmp_buf", "BATCH * HIDDEN * 2"),
            ("attn_sig_buf", "tp_size * 4"),
            ("pub_counts_buf", "n_ranks * n_ranks * n_local_experts_pad * 4"),
            ("count_done_buf", "n_ranks * 4"),
            ("recv_x_buf", "local_recv_max * HIDDEN * 2"),
            ("recv_r_route_buf", "local_recv_max * idx_pad * 4"),
            ("data_done_buf", "n_ranks * 4"),
            ("sh_tmp_buf", "BATCH * HIDDEN * 2"),
            ("sh_sig_buf", "n_ranks * 4"),
            ("routed_y_window_buf", "n_routes_per_rank * HIDDEN * 2"),
            ("combine_done_buf", "n_ranks * 4"),
        ]:
            A(f"                {buf}_{sfx} = pld.alloc_window_buffer({sz})")
        if is_full[Labs]:
            fl = full_local[Labs]
            A(f"                # ---- layer {Labs}: MoE full-attn only -> h_mid_out. ----")
            A("                for rd in pl.range(pld.world_size()):")
            A(f"                    ad_attn_tmp_window = pld.window(ad_attn_tmp_buf_{sfx}, [BATCH, HIDDEN], dtype=pl.BF16)")
            A(f"                    ad_attn_signal_window = pld.window(ad_attn_sig_buf_{sfx}, [tp_size, 1], dtype=pl.INT32)")
            A("                    self.full_attn_only_orch(")
            A("                        next_hidden_out[rd], input_rms[rd],")
            A(f"                        full_wq[rd, {fl}], full_wk[rd, {fl}], full_wv[rd, {fl}], q_norm[rd], k_norm[rd],")
            A("                        seq_lens[rd], block_table[rd], slot_mapping[rd],")
            A("                        rope_cos_full[rd], rope_sin_full[rd], k_cache[rd], v_cache[rd],")
            A(f"                        full_wo[rd, {fl}], full_w_g[rd, {fl}], full_gate_r[rd, {fl}], h_mid_out[rd],")
            A(f"                        ad_attn_tmp_window, ad_attn_signal_window, {Labs}, 0, rd, device=rd,")
            A("                    )")
        else:
            sl = swa_local[Labs]
            A(f"                # ---- layer {Labs}: MoE swa-attn only -> h_mid_out. ----")
            A("                for rd in pl.range(pld.world_size()):")
            A(f"                    ad_attn_tmp_window = pld.window(ad_attn_tmp_buf_{sfx}, [BATCH, HIDDEN], dtype=pl.BF16)")
            A(f"                    ad_attn_signal_window = pld.window(ad_attn_sig_buf_{sfx}, [tp_size, 1], dtype=pl.INT32)")
            A("                    self.swa_attn_only_orch(")
            A("                        next_hidden_out[rd], input_rms[rd],")
            A(f"                        swa_wq[rd, {sl}], swa_wk[rd, {sl}], swa_wv[rd, {sl}], q_norm[rd], k_norm[rd],")
            A("                        seq_lens[rd], block_table[rd], slot_mapping[rd],")
            A("                        rope_cos_swa[rd], rope_sin_swa[rd], k_cache[rd], v_cache[rd],")
            A(f"                        swa_wo[rd, {sl}], swa_w_g[rd, {sl}], swa_gate_r[rd, {sl}], h_mid_out[rd],")
            A(f"                        ad_attn_tmp_window, ad_attn_signal_window, {Labs}, 0, rd, device=rd,")
            A("                    )")
        A(f"                # ---- layer {Labs}: MoE-block -> next_hidden_out (pos={pos}). ----")
        A("                for r in pl.range(pld.world_size()):")
        A(f"                    attn_tmp_window = pld.window(attn_tmp_buf_{sfx}, [BATCH, HIDDEN], dtype=pl.BF16)")
        A(f"                    attn_signal_window = pld.window(attn_sig_buf_{sfx}, [tp_size, 1], dtype=pl.INT32)")
        A(f"                    pub_counts = pld.window(pub_counts_buf_{sfx}, [n_ranks * n_ranks, n_local_experts_pad], dtype=pl.INT32)")
        A(f"                    count_done_sig = pld.window(count_done_buf_{sfx}, [n_ranks, 1], dtype=pl.INT32)")
        A(f"                    recv_x = pld.window(recv_x_buf_{sfx}, [local_recv_max, HIDDEN], dtype=pl.BF16)")
        A(f"                    data_done_sig = pld.window(data_done_buf_{sfx}, [n_ranks, 1], dtype=pl.INT32)")
        A(f"                    recv_r_route = pld.window(recv_r_route_buf_{sfx}, [local_recv_max, idx_pad], dtype=pl.INT32)")
        A(f"                    sh_tmp_window = pld.window(sh_tmp_buf_{sfx}, [BATCH, HIDDEN], dtype=pl.BF16)")
        A(f"                    sh_signal_window = pld.window(sh_sig_buf_{sfx}, [n_ranks, 1], dtype=pl.INT32)")
        A(f"                    routed_y_buf = pld.window(routed_y_window_buf_{sfx}, [n_routes_per_rank, HIDDEN], dtype=pl.BF16)")
        A(f"                    combine_done_sig = pld.window(combine_done_buf_{sfx}, [n_ranks, 1], dtype=pl.INT32)")
        A("                    self.chip_orch(")
        A("                        h_mid_out[r], input_rms[r],")
        A("                        swa_wq[r, 0], swa_wk[r, 0], swa_wv[r, 0], q_norm[r], k_norm[r],")
        A("                        seq_lens[r], block_table[r], slot_mapping[r],")
        A("                        rope_cos_swa[r], rope_sin_swa[r], k_cache[r], v_cache[r],")
        A("                        swa_wo[r, 0], swa_w_g[r, 0], swa_gate_r[r, 0], post_rms[r],")
        A(f"                        moe_gate_w[r, {pos}], moe_router_bias[r, {pos}],")
        A(f"                        moe_w_gate_r[r, {pos}], moe_w_up_r[r, {pos}], moe_w_down_r[r, {pos}],")
        A(f"                        moe_w_gate_s[r, {pos}], moe_w_up_s[r, {pos}], moe_w_down_s[r, {pos}], next_hidden_out[r],")
        A("                        attn_tmp_window, attn_signal_window, pub_counts, count_done_sig,")
        A("                        recv_x, data_done_sig, recv_r_route,")
        A("                        sh_tmp_window, sh_signal_window, routed_y_buf, combine_done_sig,")
        A(f"                        {Labs}, r, device=r,")
        A("                    )")

    A("            # ── Tail: final RMSNorm + LM head on every rank. ──")
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
    chip_orch_text = chip_orch_text.replace("layer_idx", "norm_layer_idx")

    new_builder = (
        head_and_setA
        + chip_orch_text
        + lm_head_text
        + FRESH_FULL_CHIP_ORCH
        + FRESH_SWA_CHIP_ORCH
        + FRESH_FULL_ATTN_ONLY
        + FRESH_SWA_ATTN_ONLY
        + _host_orch(cls)
        + "\n    return WholeDecodeFaithfulReal\n"
    )

    binding = "\n\nwhole_decode_faithful_real = _build_whole_decode_faithful_real_program()\n"

    nl = text.index("\n", text.index(BIND_MARKER) + 1)
    new_text = text[: nl + 1] + "\n\n" + new_builder + binding + text[nl + 1:]

    bak = SRC.with_suffix(".py.bak.pregen_real")
    if not bak.exists():
        bak.write_text(text)
    SRC.write_text(new_text)
    print(f"[gen] wrote real builder: N_FULL={cls['N_FULL']} N_SWA={cls['N_SWA']} "
          f"N_DENSE={cls['N_DENSE']} N_MOE={cls['N_MOE']}; backup={bak.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
