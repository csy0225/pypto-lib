# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""ST: parameterised decode_layer MoE precision validation on device 0.

Covers the 6 MoE variants in ``select_decode_layer``:

* full_moe_silu_silu       - layer 3 (full attn + silu routed + silu shared)
* full_moe_swiglu7_silu    - layer 7
* full_moe_swiglu7_swiglu16 - layer 11
* swa_moe_silu_silu        - layer 4 (swa attn + silu routed + silu shared)
* swa_moe_swiglu7_silu     - layer 8
* swa_moe_swiglu7_swiglu16 - layer 12

Topology: per-rank single-card. Uses ``apply_perrank_patch()`` from
``_perrank_setup`` (TP_WORLD=EP_WORLD=1, ``*_LOCAL`` widths preserved
at TP=8 canonical 8/12/1/1408/160/36 etc.). Per the iron rule in
CLAUDE.md, ``apply_tp1_patch()`` (unsliced widths) is INCORRECT for
ST/UT - it forces 8-card aggregate widths onto the kernel and overflows
chunks that follow the slice (sh_mlp etc.).

Kernel + golden semantics (zeroed MoE weights):
  Set ``gate_w / router_bias / w_gate_r / w_up_r / w_down_r / w_gate_s
  / w_up_s / w_down_s`` to all-zero so the routed + shared MoE block
  output is identically zero. ``next_hidden = resid1 + 0 = resid1``
  where ``resid1`` is the attention residual (RMSNorm + QKV + RoPE +
  flash-attn + head-gate-bypass + o_proj + residual). Matches the Phase 15
  e2e zero-weight smoke pattern.

  This validates:
    1. The MoE program's host_orch + chip_orch wiring compiles + runs
       without faulting on device.
    2. Attention math is correct end-to-end through the MoE program.
    3. The MoE block's allocation / dispatch / combine path executes
       (gate scores all uniform, topk arbitrary, expert MLPs produce 0,
       combine -> 0).
  Does NOT validate MoE expert routing / weighting precision; that
  requires non-zero expert weights and a full MoE golden_fn (TODO when
  device runs unblocked).

Current status (2026-06-17): smoke (compile-only) is the only mode
that surfaces meaningful results - device runs are gated on upstream
PTOAS gate_topk parser bug (Blocker 1, see
docs/step3p5/phases/19-moe-st-blockers.md). The ``--smoke`` path
validates that all blockers except #1 are resolved.

Usage::

    cd /data/chensiyu/hw_project/pypto/workspace/pypto-lib
    python -m tests.step3p5.system.test_decode_layer_moe_st --variant full_silu_silu --smoke
    python -m tests.step3p5.system.test_decode_layer_moe_st --variant full_silu_silu -p a2a3 -d 0    # gated on Blocker 1
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch


_VARIANTS: dict[str, tuple[bool, str, int]] = {
    # name -> (full_attn, registry-key suffix, default-layer-idx)
    "full_silu_silu":           (True,  "decode_layer_full_moe_silu_silu",       3),
    "full_swiglu7_silu":        (True,  "decode_layer_full_moe_swiglu7_silu",    7),
    "full_swiglu7_swiglu16":    (True,  "decode_layer_full_moe_swiglu7_swiglu16", 11),
    "swa_silu_silu":            (False, "decode_layer_swa_moe_silu_silu",        4),
    "swa_swiglu7_silu":         (False, "decode_layer_swa_moe_swiglu7_silu",     8),
    "swa_swiglu7_swiglu16":     (False, "decode_layer_swa_moe_swiglu7_swiglu16", 12),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--variant", required=True, choices=list(_VARIANTS.keys()),
        help="Which MoE variant to test. Must match _VARIANTS keys.",
    )
    p.add_argument(
        "-p", "--platform", default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    p.add_argument("-d", "--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--with-lmhead", action="store_true")
    p.add_argument("--with-dense-mlp", action="store_true")
    p.add_argument("--two-method", action="store_true")
    p.add_argument(
        "--world-size", type=int, default=1,
        choices=[1, 8],
        help="TP_WORLD_SIZE=EP_WORLD_SIZE. 1 (default) runs the per-rank "
             "single-card path via apply_perrank_patch (`*_LOCAL` widths "
             "preserved at TP=8 canonical). 8 runs the production multi-"
             "card path with DistributedConfig(device_ids=[0..7]) — only "
             "valid with -p a2a3sim since the dev env has 1 physical NPU.",
    )
    p.add_argument(
        "--layer-idx", type=int, default=None,
        help="Override layer_idx (default: variant-specific canonical).",
    )
    return p.parse_args()


def _torch_attn_no_gate(*, full, hidden_states, input_rms_weight, wq, wk, wv,
                        q_norm_weight, k_norm_weight, wo,
                        seq_lens, block_table, slot_mapping,
                        rope_cos, rope_sin, k_cache, v_cache,
                        num_heads_local, num_kv_heads_local, head_dim,
                        rotary_dim, rotary_half, rotary_pass,
                        q_per_kv, eps, block_size, max_blocks_per_seq,
                        slide_window):
    """Per-rank slim torch attention (head_gate bypassed)."""
    batch = hidden_states.shape[0]
    hidden_q = num_heads_local * head_dim
    scale = 1.0 / math.sqrt(head_dim)

    def zc(x, g):
        return x * (g + 1.0)

    x = hidden_states.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    normed_bf16 = zc(
        x * torch.rsqrt(var + eps), input_rms_weight.float(),
    ).bfloat16()

    q_proj = normed_bf16.float() @ wq.float()
    k_proj = normed_bf16.float() @ wk.float()
    v_proj = normed_bf16.float() @ wv.float()

    q_h = q_proj.view(batch, num_heads_local, head_dim)
    q_h = zc(q_h * torch.rsqrt(q_h.pow(2).mean(-1, keepdim=True) + eps),
             q_norm_weight.float())
    k_h = k_proj.view(batch, num_kv_heads_local, head_dim)
    k_h = zc(k_h * torch.rsqrt(k_h.pow(2).mean(-1, keepdim=True) + eps),
             k_norm_weight.float())

    k_cache_w = k_cache.clone()
    v_cache_w = v_cache.clone()
    attn_out = torch.zeros(batch, hidden_q, dtype=torch.bfloat16)

    for b in range(batch):
        ctx_len = int(seq_lens[b].item())
        ctx_blocks = (ctx_len + block_size - 1) // block_size
        pos = ctx_len - 1
        cr = rope_cos[pos:pos + 1, :]
        sr = rope_sin[pos:pos + 1, :]
        c_lo = cr[:, :rotary_half]
        c_hi = cr[:, rotary_half:rotary_dim]
        s_lo = sr[:, :rotary_half]
        s_hi = sr[:, rotary_half:rotary_dim]

        kh = k_h[b]
        if rotary_pass > 0:
            k_rot = torch.cat([
                kh[:, :rotary_half] * c_lo - kh[:, rotary_half:rotary_dim] * s_lo,
                kh[:, rotary_half:rotary_dim] * c_hi + kh[:, :rotary_half] * s_hi,
                kh[:, rotary_dim:rotary_dim + rotary_pass],
            ], dim=-1)
        else:
            k_rot = torch.cat([
                kh[:, :rotary_half] * c_lo - kh[:, rotary_half:] * s_lo,
                kh[:, rotary_half:] * c_hi + kh[:, :rotary_half] * s_hi,
            ], dim=-1)

        slot = int(slot_mapping[b].item())
        sb_blk = slot // block_size
        sb_off = slot % block_size
        for ki in range(num_kv_heads_local):
            row = (sb_blk * num_kv_heads_local + ki) * block_size + sb_off
            k_cache_w[row, :] = k_rot[ki].to(torch.bfloat16)
            v_cache_w[row, :] = v_proj[
                b, ki * head_dim:(ki + 1) * head_dim,
            ].to(torch.bfloat16)

        qh = q_h[b]
        if rotary_pass > 0:
            q_rot = torch.cat([
                qh[:, :rotary_half] * c_lo - qh[:, rotary_half:rotary_dim] * s_lo,
                qh[:, rotary_half:rotary_dim] * c_hi + qh[:, :rotary_half] * s_hi,
                qh[:, rotary_dim:rotary_dim + rotary_pass],
            ], dim=-1)
        else:
            q_rot = torch.cat([
                qh[:, :rotary_half] * c_lo - qh[:, rotary_half:] * s_lo,
                qh[:, rotary_half:] * c_hi + qh[:, :rotary_half] * s_hi,
            ], dim=-1)

        attn_row = torch.zeros(1, hidden_q, dtype=torch.bfloat16)
        for kvh in range(num_kv_heads_local):
            q_base = kvh * q_per_kv
            q_grp = q_rot[q_base:q_base + q_per_kv, :].to(torch.bfloat16)
            oi = torch.zeros(q_per_kv, head_dim)
            li = torch.zeros(q_per_kv, 1)
            mi = torch.zeros(q_per_kv, 1)
            for sb in range(ctx_blocks):
                valid_len = min(block_size, ctx_len - sb * block_size)
                pbid = int(block_table[b * max_blocks_per_seq + sb].item())
                cr0 = (pbid * num_kv_heads_local + kvh) * block_size
                kt = k_cache_w[cr0:cr0 + block_size, :]
                vt = v_cache_w[cr0:cr0 + block_size, :]
                rs = q_grp.float() @ kt.float().T
                if valid_len < block_size:
                    rs[:, valid_len:] = torch.finfo(torch.float32).min
                if not full and slide_window > 0:
                    cur_pos = ctx_len - 1
                    sb_start = sb * block_size
                    abs_min = max(0, cur_pos - slide_window + 1)
                    for col in range(block_size):
                        if (sb_start + col) < abs_min:
                            rs[:, col] = torch.finfo(torch.float32).min
                scores = rs * scale
                cm = scores.max(dim=-1, keepdim=True).values
                es = torch.exp(scores - cm)
                es_b = es.to(torch.bfloat16)
                cl = es_b.float().sum(dim=-1, keepdim=True)
                ot = es_b.float() @ vt.float()
                if sb == 0:
                    oi, li, mi = ot, cl, cm
                else:
                    mn = torch.maximum(mi, cm)
                    a = torch.exp(mi - mn)
                    bw = torch.exp(cm - mn)
                    li = a * li + bw * cl
                    oi = oi * a + ot * bw
                    mi = mn
            ctx = oi / li
            attn_row[:, q_base * head_dim:(q_base + q_per_kv) * head_dim] = (
                ctx.reshape(1, -1).to(torch.bfloat16)
            )
        attn_out[b:b + 1, :] = attn_row

    o = attn_out.float() @ wo.float()
    resid1 = (o + hidden_states.float()).bfloat16()
    return resid1


def main() -> int:  # noqa: PLR0915
    args = _parse_args()
    full_attn, prog_name_key, default_layer_idx = _VARIANTS[args.variant]
    layer_idx = args.layer_idx if args.layer_idx is not None else default_layer_idx

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))

    # Topology selection: per-rank single-card (world_size=1) or canonical
    # multi-card TP=8/EP=8 (world_size=8). The latter validates that our
    # FP32 expert_weights + PER_RANK_BUCKETS/N_RANKS_PAD pad fixes are
    # numerically/dimensionally correct in production config — the simulator
    # exercises full 8-rank EP all-to-all + TP all-reduce with real
    # collective state (not a stub). Only a2a3sim supports world_size=8
    # in our dev env (1 physical NPU).
    if args.world_size == 1:
        # Iron-rule per-rank patch: TP=1/EP=1 codegen + TP=8 widths preserved.
        from tests.step3p5.common._perrank_setup import apply_perrank_patch  # noqa: PLC0415

        summary = apply_perrank_patch(reload_modules=[
            "models.step3p5.attention_full",
            "models.step3p5.attention_swa",
            "models.step3p5.decode_layer",
        ])
        print(f"[ST-MoE {args.variant}] per-rank patch: {summary}", flush=True)
    else:
        # Canonical TP=8/EP=8 — no patch, config defaults.
        if args.platform == "a2a3" and False:  # relaxed: 0162 has 16 NPUs
            raise ValueError(
                "world_size=8 requires -p a2a3sim (dev env has 1 NPU; "
                "8-rank canonical needs simulator).",
            )
        print(
            f"[ST-MoE {args.variant}] canonical TP=8/EP=8 (no per-rank patch)",
            flush=True,
        )

    if args.world_size > 1 and not full_attn:
        # Standalone SWA builds need the dynamic WO row bound materialized
        # before decode_layer imports LAYER_QHIDDEN_ROWS_DYN_SWA.
        import models.step3p5.attention_swa as attention_swa_mod  # noqa: PLC0415
        import models.step3p5.config as cfg_for_swa_dyn  # noqa: PLC0415

        n_swa_attn_layers = 33
        attention_swa_mod.LAYER_QHIDDEN_ROWS_DYN = (
            n_swa_attn_layers * cfg_for_swa_dyn.HIDDEN_Q_SWA_LOCAL
        )

    from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415

    set_backend_type({
        "a2a3": BackendType.Ascend910B,
        "a2a3sim": BackendType.Ascend910B,
        "a5": BackendType.Ascend950,
        "a5sim": BackendType.Ascend950,
    }[args.platform])

    from models.step3p5.config import (  # noqa: PLC0415
        BATCH,
        BLOCK_SIZE,
        EPS,
        HEAD_DIM,
        HIDDEN,
        MAX_BLOCKS_PER_SEQ,
        MAX_SEQ_DEFAULT,
        MOE_INTERMEDIATE,
        MOE_NUM_EXPERTS,
        NUM_HIDDEN_LAYERS,
        Q_PER_KV_FULL,
        Q_PER_KV_SWA,
        ROTARY_HALF_FULL,
        ROTARY_HALF_SWA,
        SLIDING_WINDOW,
    )
    import models.step3p5.config as cfg_mod  # noqa: PLC0415
    import models.step3p5.decode_layer as decode_layer  # noqa: PLC0415
    if args.two_method:
        args.with_dense_mlp = True

    if args.two_method:
        program = decode_layer._build_mixed_2method_program(
            full=full_attn, routed_lim=0.0, shared_lim=0.0,
        )
    elif args.with_dense_mlp:
        program = decode_layer._build_fused_dense_moe_program(
            full=full_attn, routed_lim=0.0, shared_lim=0.0,
        )
    else:
        program = getattr(decode_layer, prog_name_key)
    prog_name = getattr(program, "name", None) or type(program).__name__

    if full_attn:
        H_Q = cfg_mod.NUM_HEADS_FULL_LOCAL * HEAD_DIM
        ROTARY_DIM = ROTARY_HALF_FULL * 2
        ROTARY_HALF = ROTARY_HALF_FULL
        Q_PER_KV = Q_PER_KV_FULL
        N_HEADS = cfg_mod.NUM_HEADS_FULL_LOCAL
        PAD = cfg_mod.NUM_HEADS_FULL_LOCAL_PAD
    else:
        H_Q = cfg_mod.NUM_HEADS_SWA_LOCAL * HEAD_DIM
        ROTARY_DIM = ROTARY_HALF_SWA * 2
        ROTARY_HALF = ROTARY_HALF_SWA
        Q_PER_KV = Q_PER_KV_SWA
        N_HEADS = cfg_mod.NUM_HEADS_SWA_LOCAL
        PAD = cfg_mod.NUM_HEADS_SWA_LOCAL_PAD
    KV_H = cfg_mod.KV_HEADS_LOCAL * HEAD_DIM
    INT_S = cfg_mod.SHARE_EXPERT_DIM_LOCAL
    N_LOC_E = cfg_mod.MOE_NUM_EXPERTS_LOCAL
    INT_R = MOE_INTERMEDIATE
    ROTARY_PASS = HEAD_DIM - ROTARY_DIM

    print(
        f"[ST-MoE {args.variant}] program={prog_name} layer_idx={layer_idx} "
        f"platform={args.platform} device={args.device} "
        f"  dims: H={HIDDEN} H_Q={H_Q} KV={KV_H} INT_S={INT_S} "
        f"N_LOC_E={N_LOC_E} INT_R={INT_R} BATCH={BATCH}",
        flush=True,
    )

    g = torch.Generator().manual_seed(args.seed)
    bf16 = torch.bfloat16

    # Match the COMPILED program's stacked bound (LAYER_HIDDEN_ROWS_DYN =
    # n_full_attn_layers * HIDDEN); NUM_HIDDEN_LAYERS (45) over-provisions
    # wo to 3GB and OOMs once dense/lm_head weights are added.
    n_attn = cfg_mod.LAYER_HIDDEN_ROWS_DYN // HIDDEN
    w_input_rms = torch.empty(NUM_HIDDEN_LAYERS, HIDDEN).normal_(
        0.0, 0.05, generator=g)
    w_post_rms = torch.empty(NUM_HIDDEN_LAYERS, HIDDEN).normal_(
        0.0, 0.05, generator=g)
    w_q_norm = torch.empty(NUM_HIDDEN_LAYERS, HEAD_DIM).normal_(
        0.0, 0.05, generator=g)
    w_k_norm = torch.empty(NUM_HIDDEN_LAYERS, HEAD_DIM).normal_(
        0.0, 0.05, generator=g)

    def _randn(shape, dtype=bf16, std=0.02):
        return torch.empty(shape, dtype=torch.float32).normal_(
            0.0, std, generator=g).to(dtype)

    # Attention weights (per-layer stacked along axis 0).
    wq = _randn([n_attn, HIDDEN, H_Q])
    wk = _randn([n_attn, HIDDEN, KV_H])
    wv = _randn([n_attn, HIDDEN, KV_H])
    wo = _randn([n_attn, H_Q, HIDDEN])
    w_g = _randn([n_attn, HIDDEN, PAD])

    # MoE weights - mostly ZERO except gate_w (random small) so that:
    #   1. gate produces UNIQUE per-expert scores (avoids sort32 hardware tie
    #      edge cases when all 288 experts have identical sigmoid(0)=0.5
    #      logits — sort32 picks arbitrary tied indices, may include id >= 36
    #      which routes to non-existent peer rank).
    #   2. router_bias forces top-K ⊂ [0, N_LOC_E) so dispatch's
    #      `dst_rank = expert_id // N_LOCAL_EXPERTS` always == 0 == my_rank,
    #      avoiding `pld.system.notify(peer=non_existent_rank)` AICore hang
    #      → 507018 timeout.
    #   3. routed/shared expert weights (w_*_r, w_*_s) stay zero so MoE
    #      block math output = 0, and the golden_fn = attention resid1.
    gate_w = _randn([HIDDEN, MOE_NUM_EXPERTS], dtype=torch.float32, std=0.02).float()
    if args.world_size > 1:
        # Multi-card: a REAL EP all-to-all needs routing spread across ALL
        # MOE_NUM_EXPERTS experts so dst_rank = eid // N_LOC_E covers every
        # rank 0..world_size-1, load-balanced. Zero bias makes every expert
        # eligible; the random gate_w gives unique per-(token,expert) scores so
        # top-K is a balanced random spread (no sort32 ties), and the worst-case
        # rows/expert stays far below local_recv_max. The old [0, N_LOC_E) mask
        # forced every token to rank 0 (a degenerate all-to-one, not all-to-all).
        router_bias = torch.zeros([MOE_NUM_EXPERTS], dtype=torch.float32)
    else:
        # Single physical card: restrict top-K to [0, N_LOC_E) so dispatch's
        # dst_rank == 0 == my_rank and nothing is pushed to a non-existent peer.
        router_bias = torch.cat([
            torch.full([N_LOC_E], 10.0, dtype=torch.float32),
            torch.full([MOE_NUM_EXPERTS - N_LOC_E], -10.0, dtype=torch.float32),
        ])
    w_gate_r = torch.zeros(N_LOC_E, HIDDEN, INT_R, dtype=bf16)
    w_up_r = torch.zeros(N_LOC_E, HIDDEN, INT_R, dtype=bf16)
    w_down_r = torch.zeros(N_LOC_E, INT_R, HIDDEN, dtype=bf16)
    w_gate_s = torch.zeros(HIDDEN, INT_S, dtype=bf16)
    w_up_s = torch.zeros(HIDDEN, INT_S, dtype=bf16)
    w_down_s = torch.zeros(INT_S, HIDDEN, dtype=bf16)

    # Runtime inputs.
    current_hidden = _randn([1, BATCH, HIDDEN], dtype=bf16, std=0.02)

    seq_lens = torch.ones(1, BATCH, dtype=torch.int32)
    block_table = torch.zeros(1, MAX_BLOCKS_PER_SEQ * BATCH, dtype=torch.int32)
    slot_mapping = torch.arange(BATCH, dtype=torch.int32).unsqueeze(0)
    rope_cos = torch.empty(1, MAX_SEQ_DEFAULT, ROTARY_DIM).normal_(
        0.0, 0.5, generator=g)
    rope_sin = torch.empty(1, MAX_SEQ_DEFAULT, ROTARY_DIM).normal_(
        0.0, 0.5, generator=g)
    k_cache = torch.zeros(1, MAX_SEQ_DEFAULT, HEAD_DIM, dtype=bf16)
    v_cache = torch.zeros(1, MAX_SEQ_DEFAULT, HEAD_DIM, dtype=bf16)

    next_hidden_out = torch.zeros(1, BATCH, HIDDEN, dtype=bf16)
    if args.two_method:
        h_mid_out = torch.zeros(1, BATCH, HIDDEN, dtype=bf16)
    if args.with_dense_mlp:
        _LHR = cfg_mod.LAYER_HIDDEN_ROWS_DYN
        _LIR = cfg_mod.LAYER_INTER_ROWS_DYN
        _ILC = cfg_mod.INTERMEDIATE_LOCAL
        post_rms_d = w_post_rms.float().unsqueeze(0)
        w_gate_d = torch.zeros(1, _LHR, _ILC, dtype=bf16)
        w_up_d = torch.zeros(1, _LHR, _ILC, dtype=bf16)
        w_down_d = torch.zeros(1, _LIR, HIDDEN, dtype=bf16)
    if args.with_lmhead:
        from models.step3p5.config import VOCAB, VOCAB_LOCAL  # noqa: PLC0415,F401
        _fnorm = torch.empty(1, HIDDEN).normal_(0.0, 0.05, generator=g)
        _lmh_full = torch.empty(
            VOCAB, HIDDEN, dtype=torch.float32,
        ).normal_(0.0, 0.02, generator=g)
        final_norm_weight = _fnorm.reshape(1, 1, HIDDEN)
        lm_head_weight = _lmh_full[:VOCAB_LOCAL].to(bf16).unsqueeze(0)
        logits_shard_out = torch.zeros(
            1, BATCH, VOCAB_LOCAL, dtype=torch.float32,
        )

    def flat3(t):
        L, M, N = t.shape
        return t.reshape(1, L * M, N)

    inputs = {
        "current_hidden": current_hidden,
        "input_rms_weight": w_input_rms.float().unsqueeze(0),
        "wq": flat3(wq), "wk": flat3(wk), "wv": flat3(wv),
        "q_norm_weight": w_q_norm.float().unsqueeze(0),
        "k_norm_weight": w_k_norm.float().unsqueeze(0),
        "seq_lens": seq_lens, "block_table": block_table,
        "slot_mapping": slot_mapping, "rope_cos": rope_cos,
        "rope_sin": rope_sin, "k_cache": k_cache, "v_cache": v_cache,
        "wo": flat3(wo), "w_g": flat3(w_g),
        "gate_r": torch.ones(1, PAD, H_Q, dtype=bf16),
        "post_rms_weight": w_post_rms.float().unsqueeze(0),
        "gate_w": gate_w.unsqueeze(0),
        "router_bias": router_bias.unsqueeze(0),
        "w_gate_r": w_gate_r.unsqueeze(0),
        "w_up_r": w_up_r.unsqueeze(0),
        "w_down_r": w_down_r.unsqueeze(0),
        "w_gate_s": w_gate_s.unsqueeze(0),
        "w_up_s": w_up_s.unsqueeze(0),
        "w_down_s": w_down_s.unsqueeze(0),
        "next_hidden_out": next_hidden_out,
    }
    if args.with_dense_mlp:
        inputs["post_rms_d"] = post_rms_d
        inputs["w_gate_d"] = w_gate_d
        inputs["w_up_d"] = w_up_d
        inputs["w_down_d"] = w_down_d
    if args.two_method:
        inputs["h_mid_out"] = h_mid_out
    if args.with_lmhead:
        inputs["final_norm_weight"] = final_norm_weight
        inputs["lm_head_weight"] = lm_head_weight
        inputs["logits_shard_out"] = logits_shard_out

    # Multi-rank: broadcast every host input to [N_RANKS, ...] so the
    # canonical 8-rank host_orch can index tensors[name][r_idx] for
    # r_idx in [0, N_RANKS). Expert weights are zero, so the golden is
    # attention TP all-reduce + residual; MoE dispatch/combine/shared lanes
    # must execute but contribute numerically zero.
    if args.world_size > 1:
        _NR = args.world_size
        inputs = {
            k: (v if v.dim() == 0
                else v.expand(_NR, *v.shape[1:]).contiguous())
            for k, v in inputs.items()
        }
        if args.with_lmhead:
            from models.step3p5.config import VOCAB_LOCAL  # noqa: PLC0415
            _lmh_d = torch.zeros(_NR, VOCAB_LOCAL, HIDDEN, dtype=bf16)
            for _r in range(_NR):
                _lmh_d[_r] = _lmh_full[
                    _r * VOCAB_LOCAL:(_r + 1) * VOCAB_LOCAL
                ].to(bf16)
            inputs["lm_head_weight"] = _lmh_d

    from golden import ScalarSpec, TensorSpec, ratio_allclose, run  # noqa: PLC0415

    def _spec(name, dtype, is_out=False):
        t = inputs[name]
        return TensorSpec(
            name, list(t.shape), dtype,
            init_value=None if is_out else t,
            is_output=is_out,
        )

    specs = [
        _spec("current_hidden", bf16),
        _spec("input_rms_weight", torch.float32),
        _spec("wq", bf16), _spec("wk", bf16), _spec("wv", bf16),
        _spec("q_norm_weight", torch.float32),
        _spec("k_norm_weight", torch.float32),
        _spec("seq_lens", torch.int32),
        _spec("block_table", torch.int32),
        _spec("slot_mapping", torch.int32),
        _spec("rope_cos", torch.float32),
        _spec("rope_sin", torch.float32),
        _spec("k_cache", bf16), _spec("v_cache", bf16),
        _spec("wo", bf16), _spec("w_g", bf16),
        _spec("gate_r", bf16),
        _spec("post_rms_weight", torch.float32),
        _spec("gate_w", torch.float32),
        _spec("router_bias", torch.float32),
        _spec("w_gate_r", bf16), _spec("w_up_r", bf16),
        _spec("w_down_r", bf16),
        _spec("w_gate_s", bf16), _spec("w_up_s", bf16),
        _spec("w_down_s", bf16),
        _spec("next_hidden_out", bf16, is_out=True),
        ScalarSpec("layer_idx", torch.int32,
                   value=torch.tensor(layer_idx, dtype=torch.int32)),
    ]
    if args.two_method:
        specs.insert(-1, _spec("h_mid_out", bf16, is_out=True))
    if args.with_dense_mlp:
        specs.insert(-1, _spec("post_rms_d", torch.float32))
        specs.insert(-1, _spec("w_gate_d", bf16))
        specs.insert(-1, _spec("w_up_d", bf16))
        specs.insert(-1, _spec("w_down_d", bf16))
    if args.with_lmhead:
        specs.insert(-1, _spec("final_norm_weight", torch.float32))
        specs.insert(-1, _spec("lm_head_weight", bf16))
        specs.insert(-1, _spec("logits_shard_out", torch.float32, is_out=True))

    def golden_fn(values):
        # Rank-wise slim torch ref. Each rank computes its local attention
        # o_proj partial; TP all-reduce sums partials, then the replicated
        # residual is added once. MoE expert weights are zero, so the MoE
        # block contributes 0 and next_hidden = attention resid1.
        nranks = values["current_hidden"].shape[0]
        TP1_KV_HEADS = 1
        TP1_NUM_HEADS = Q_PER_KV
        partial_sum = torch.zeros(BATCH, HIDDEN, dtype=torch.float32)
        hidden_ref = values["current_hidden"][0].float()

        for rank in range(nranks):
            hidden = values["current_hidden"][rank]
            irms = values["input_rms_weight"][rank][layer_idx]
            wq_l = values["wq"][rank][layer_idx * HIDDEN:(layer_idx + 1) * HIDDEN]
            wk_l = values["wk"][rank][layer_idx * HIDDEN:(layer_idx + 1) * HIDDEN]
            wv_l = values["wv"][rank][layer_idx * HIDDEN:(layer_idx + 1) * HIDDEN]
            wo_l = values["wo"][rank][layer_idx * H_Q:(layer_idx + 1) * H_Q]
            qn = values["q_norm_weight"][rank][layer_idx]
            kn = values["k_norm_weight"][rank][layer_idx]
            sl = values["seq_lens"][rank]
            bt = values["block_table"][rank]
            sm = values["slot_mapping"][rank]
            rc = values["rope_cos"][rank]
            rs = values["rope_sin"][rank]
            kc = values["k_cache"][rank]
            vc = values["v_cache"][rank]

            wq_slim = wq_l[:, :TP1_NUM_HEADS * HEAD_DIM]
            wk_slim = wk_l[:, :TP1_KV_HEADS * HEAD_DIM]
            wv_slim = wv_l[:, :TP1_KV_HEADS * HEAD_DIM]
            wo_slim = wo_l[:TP1_NUM_HEADS * HEAD_DIM, :]
            local_resid1 = _torch_attn_no_gate(
                full=full_attn, hidden_states=hidden,
                input_rms_weight=irms,
                wq=wq_slim, wk=wk_slim, wv=wv_slim,
                q_norm_weight=qn, k_norm_weight=kn, wo=wo_slim,
                seq_lens=sl, block_table=bt, slot_mapping=sm,
                rope_cos=rc, rope_sin=rs,
                k_cache=kc, v_cache=vc,
                num_heads_local=TP1_NUM_HEADS,
                num_kv_heads_local=TP1_KV_HEADS,
                head_dim=HEAD_DIM, rotary_dim=ROTARY_DIM,
                rotary_half=ROTARY_HALF, rotary_pass=ROTARY_PASS,
                q_per_kv=Q_PER_KV, eps=EPS, block_size=BLOCK_SIZE,
                max_blocks_per_seq=MAX_BLOCKS_PER_SEQ,
                slide_window=SLIDING_WINDOW,
            )
            partial_sum += local_resid1.float() - hidden.float()

        expected = (partial_sum + hidden_ref).bfloat16()
        for rank in range(nranks):
            values["next_hidden_out"][rank] = expected
        if args.two_method:
            for rank in range(nranks):
                values["h_mid_out"][rank] = expected
        if args.with_lmhead:
            _x = expected.float()
            _var = _x.pow(2).mean(-1, keepdim=True)
            _fn = values["final_norm_weight"][0].reshape(-1).float()
            _fnormed = _x * torch.rsqrt(_var + EPS) * (_fn + 1.0)
            for rank in range(nranks):
                _lmh_r = values["lm_head_weight"][rank].float()
                values["logits_shard_out"][rank] = _fnormed @ _lmh_r.t()

    from simpler.orchestrator import Orchestrator  # noqa: PLC0415
    from simpler.task_interface import (  # noqa: PLC0415
        ChipDomainContext,
        CommDomainHandle,
    )
    _orig_alloc_domain = Orchestrator.allocate_domain

    def _single_rank_alloc_domain(self, *, name, workers, window_size, buffers):
        workers = tuple(int(w) for w in workers)
        if len(workers) > 1:
            return _orig_alloc_domain(
                self, name=name, workers=workers,
                window_size=window_size, buffers=buffers,
            )
        chip_idx = workers[0]
        # 64-byte align each buffer offset — MoE has 13 buffers with mixed
        # sizes (1MB BF16 routed_y_buf, 4 B INT32 signal flags, 144 B counts).
        # Unaligned offsets can fault on cell-level pld.system.notify /
        # pld.tile.remote_load that the kernel emits at the chip level.
        # Pre-compute the aligned cumulative size so the underlying malloc
        # covers all aligned offsets even if larger than the caller's
        # window_size hint (the host_orch's window_size is a sum-of-nbytes
        # without padding).
        aligned_offsets: list[int] = []
        running = 0
        for spec in buffers:
            aligned_offsets.append(running)
            n = int(spec.nbytes)
            n_aligned = ((n + 63) // 64) * 64
            running += n_aligned
        actual_window_size = max(int(window_size), running)
        base = int(self.malloc(chip_idx, actual_window_size))
        ptrs: dict[str, int] = {
            spec.name: base + off
            for spec, off in zip(buffers, aligned_offsets, strict=True)
        }
        ctx = ChipDomainContext(
            name=str(name), domain_rank=0, domain_size=1, device_ctx=0,
            local_window_base=base, actual_window_size=actual_window_size,
            buffer_ptrs=ptrs,
        )

        def _release(_h):
            try:
                self.free(chip_idx, base)
            except Exception:
                pass

        return CommDomainHandle(
            name=str(name), workers=workers, contexts={chip_idx: ctx},
            allocation_id=-1, _release_fn=_release,
        )

    # Stub install: only at world_size=1 (real allocate_domain at single
    # rank fails on shmem_map_exbus). At world_size=8 the simulator
    # provides full allocate_domain support out-of-box.
    if args.world_size == 1:
        Orchestrator.allocate_domain = _single_rank_alloc_domain

    # Build compile_cfg for canonical TP=8 multi-rank case so that
    # ir.compile uses DistributedConfig(device_ids=[0..world_size-1]).
    compile_cfg = {}
    if args.world_size != 1:
        from pypto.ir.distributed_compiled_program import (  # noqa: PLC0415
            DistributedConfig,
        )
        # MOE_ST_DEV_OFFSET lets the 8-card run target cards [off..off+7] so it
        # can run on a free card group (e.g. 8-15) without disturbing an
        # oracle vLLM on cards 0-7. Default 0 = original behaviour.
        _dev_off = int(os.environ.get("MOE_ST_DEV_OFFSET", "0"))
        compile_cfg["distributed_config"] = DistributedConfig(
            device_ids=[_dev_off + d for d in range(args.world_size)],
            num_sub_workers=0,
        )

    try:
        runtime_cfg = dict(platform=args.platform, device_id=args.device)
        if args.smoke:
            result = run(
                program=program, specs=specs,
                runtime_cfg=runtime_cfg, compile_only=True,
                compile_cfg=compile_cfg,
            )
            print(f"[ST-MoE {args.variant} ws={args.world_size}] SMOKE: {result}",
                  flush=True)
            return 0 if result.passed else 1

        _cmp_fns = {
            "next_hidden_out": ratio_allclose(
                atol=4e-2, rtol=4e-2, max_error_ratio=0.10,
            ),
        }
        if args.with_lmhead:
            _cmp_fns["logits_shard_out"] = ratio_allclose(
                atol=6e-2, rtol=6e-2, max_error_ratio=0.20,
            )
        result = run(
            program=program, specs=specs,
            golden_fn=golden_fn,
            runtime_cfg=runtime_cfg, rtol=4e-2, atol=4e-2,
            compile_cfg=compile_cfg,
            compare_fn=_cmp_fns,
        )
        print(f"[ST-MoE {args.variant} ws={args.world_size}] {args.platform.upper()}: {result}",
              flush=True)
        return 0 if result.passed else 1
    finally:
        if args.world_size == 1:
            Orchestrator.allocate_domain = _orig_alloc_domain


if __name__ == "__main__":
    raise SystemExit(main())
