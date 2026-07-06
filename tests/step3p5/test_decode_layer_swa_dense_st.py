# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""ST-2: swa_dense decode_layer (layers 1, 2) precision on device 0.

Mirrors ST-1 (test_decode_layer_full_dense_st.py) but exercises the SWA
attention path instead of full-attention. Covers
`select_decode_layer(1)` -> `DecodeLayerDense` @pl.program built with
``full=False``. Layer 2 is the same kind and reuses the same harness.

Differences vs ST-1:
- attention_swa instead of attention_full (sliding window math + full
  rotary_dim=128, no rotary_pass)
- HIDDEN_Q_SWA_LOCAL = 12288 (TP=1) vs HIDDEN_Q_FULL_LOCAL = 8192
- Q_PER_KV_SWA = 12 vs Q_PER_KV_FULL = 8
- ROTARY_HALF_SWA = 64 -> rotary_dim = 128 = HEAD_DIM (full rotation)
- LAYER_QHIDDEN_ROWS_DYN_SWA = 33 * HIDDEN_Q_SWA_LOCAL (33 SWA layers)
- SAME slim attention assumption as ST-1: attention_swa kernel also
  hardcodes "KV_HEADS_LOCAL=1, Q_GROUPS=1" in its FA stages, so under
  TP=1 monkey-patch only KV head 0's attention is computed; torch ref
  models that by setting num_kv_heads=1, num_heads=Q_PER_KV_SWA=12

Usage::

    cd /data/chensiyu/hw_project/pypto/workspace/pypto-lib
    python -m tests.step3p5.test_decode_layer_swa_dense_st --smoke
    python -m tests.step3p5.test_decode_layer_swa_dense_st -p a2a3 -d 0
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _torch_swa_attn_no_gate(*, hidden_states, input_rms_weight,
                            wq_full, wk_full, wv_full,
                            q_norm_weight, k_norm_weight, wo_full,
                            seq_lens, block_table, slot_mapping,
                            rope_cos, rope_sin,
                            k_cache_full, v_cache_full,
                            num_heads_full, num_kv_heads_full, head_dim,
                            rotary_dim, rotary_half, q_per_kv,
                            eps, block_size, max_blocks_per_seq,
                            sliding_window, w_g_full=None):
    """Per-token sliding-window paged attention. head_gate applied iff
    ``w_g_full`` given (``sigmoid(input_RMSNorm(hidden) @ w_g_full)`` per head,
    applied to attn_out before o_proj — matches vLLM Step3p5Attention and the
    kernel's worker-precomputed gate_r). Default ``None`` keeps the no-gate
    behaviour so existing ST callers are unchanged.
    rotary_dim == head_dim for SWA (no pass-through)."""
    batch = hidden_states.shape[0]
    hidden_q = num_heads_full * head_dim
    scale = 1.0 / math.sqrt(head_dim)

    def zc(x, g):
        return x * (g + 1.0)

    x = hidden_states.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    normed_bf16 = zc(
        x * torch.rsqrt(var + eps), input_rms_weight.float(),
    ).bfloat16()

    q_proj = normed_bf16.float() @ wq_full.float()
    k_proj = normed_bf16.float() @ wk_full.float()
    v_proj = normed_bf16.float() @ wv_full.float()

    q_h = q_proj.view(batch, num_heads_full, head_dim)
    q_h = zc(q_h * torch.rsqrt(q_h.pow(2).mean(-1, keepdim=True) + eps),
             q_norm_weight.float())
    k_h = k_proj.view(batch, num_kv_heads_full, head_dim)
    k_h = zc(k_h * torch.rsqrt(k_h.pow(2).mean(-1, keepdim=True) + eps),
             k_norm_weight.float())

    k_cache = k_cache_full.clone()
    v_cache = v_cache_full.clone()
    attn_out = torch.zeros(batch, hidden_q, dtype=torch.bfloat16)
    for b in range(batch):
        ctx_len = int(seq_lens[b].item())
        eff_ctx_len = min(ctx_len, sliding_window)
        ctx_blocks = (eff_ctx_len + block_size - 1) // block_size
        pos = ctx_len - 1
        cr = rope_cos[pos:pos + 1, :]
        sr = rope_sin[pos:pos + 1, :]
        c_lo, c_hi = cr[:, :rotary_half], cr[:, rotary_half:rotary_dim]
        s_lo, s_hi = sr[:, :rotary_half], sr[:, rotary_half:rotary_dim]

        kh = k_h[b]
        k_rot = torch.cat([
            kh[:, :rotary_half] * c_lo - kh[:, rotary_half:rotary_dim] * s_lo,
            kh[:, rotary_half:rotary_dim] * c_hi + kh[:, :rotary_half] * s_hi,
        ], dim=-1)

        slot = int(slot_mapping[b].item())
        sb_blk = slot // block_size
        sb_off = slot % block_size
        for ki in range(num_kv_heads_full):
            row = (sb_blk * num_kv_heads_full + ki) * block_size + sb_off
            k_cache[row, :] = k_rot[ki].to(torch.bfloat16)
            v_cache[row, :] = v_proj[
                b, ki * head_dim:(ki + 1) * head_dim,
            ].to(torch.bfloat16)

        qh = q_h[b]
        q_rot = torch.cat([
            qh[:, :rotary_half] * c_lo - qh[:, rotary_half:rotary_dim] * s_lo,
            qh[:, rotary_half:rotary_dim] * c_hi + qh[:, :rotary_half] * s_hi,
        ], dim=-1)

        attn_row = torch.zeros(1, hidden_q, dtype=torch.bfloat16)
        for kvh in range(num_kv_heads_full):
            q_base = kvh * q_per_kv
            q_grp = q_rot[q_base:q_base + q_per_kv, :].to(torch.bfloat16)
            oi = torch.zeros(q_per_kv, head_dim)
            li = torch.zeros(q_per_kv, 1)
            mi = torch.zeros(q_per_kv, 1)
            for sb in range(ctx_blocks):
                valid_len = min(block_size, eff_ctx_len - sb * block_size)
                pbid = int(block_table[b * max_blocks_per_seq + sb].item())
                cr0 = (pbid * num_kv_heads_full + kvh) * block_size
                kt = k_cache[cr0:cr0 + block_size, :]
                vt = v_cache[cr0:cr0 + block_size, :]
                rs = q_grp.float() @ kt.float().T
                if valid_len < block_size:
                    rs[:, valid_len:] = torch.finfo(torch.float32).min
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

    # head_gate: identity when w_g_full is None (ST no-gate path), else the real
    # step3p5 per-head sigmoid gate applied to attn_out before o_proj. gate uses
    # normed_bf16 (input-RMSNorm'd hidden), matching vLLM sigmoid(g_proj(
    # input_layernorm(hidden))) and the kernel's worker-precomputed gate_r.
    if w_g_full is not None:
        gate = torch.sigmoid(normed_bf16.float() @ w_g_full.float())
        attn_g = (attn_out.float().view(batch, num_heads_full, head_dim)
                  * gate.unsqueeze(-1)).reshape(batch, hidden_q)
        o = attn_g @ wo_full.float()
    else:
        o = attn_out.float() @ wo_full.float()
    resid1 = (o + hidden_states.float()).bfloat16()
    return resid1


def _torch_dense_mlp(resid1, post_rms_weight_layer, w_gate, w_up, w_down, eps):
    x = resid1.float()
    var = x.pow(2).mean(-1, keepdim=True)
    normed = (
        x * torch.rsqrt(var + eps) * (post_rms_weight_layer.float() + 1.0)
    ).bfloat16()
    gate = normed.float() @ w_gate.float()
    up = normed.float() @ w_up.float()
    silu = gate * torch.sigmoid(gate)
    h = (silu * up).bfloat16()
    mlp_partial = (h.float() @ w_down.float()).bfloat16()
    next_hidden = (resid1.float() + mlp_partial.float()).bfloat16()
    return next_hidden


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    from tests.step3p5._tp1_setup import apply_tp1_patch  # noqa: PLC0415

    summary = apply_tp1_patch(reload_modules=[
        "models.step3p5.attention_full",
        "models.step3p5.attention_swa",
        "models.step3p5.decode_layer",
    ])
    print(f"[ST-2 swa_dense] TP=1 patch: {summary}", flush=True)

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
        DENSE_LAYER_INDICES,
        EPS,
        HEAD_DIM,
        HIDDEN,
        LAYER_TYPE_SWA,
        LAYER_TYPES,
        MAX_BLOCKS_PER_SEQ,
        MAX_SEQ_DEFAULT,
        NUM_HIDDEN_LAYERS,
        Q_PER_KV_SWA,
        ROTARY_HALF_SWA,
        SLIDING_WINDOW,
    )
    import models.step3p5.config as cfg_mod  # noqa: PLC0415
    from models.step3p5.decode_layer import select_decode_layer  # noqa: PLC0415

    program, kind = select_decode_layer(1)
    assert kind == "swa_dense", f"layer 1 should be swa_dense, got {kind}"
    prog_name = getattr(program, "name", None) or type(program).__name__

    H_Q_SWA = cfg_mod.NUM_HEADS_SWA_LOCAL * HEAD_DIM
    KV_H = cfg_mod.KV_HEADS_LOCAL * HEAD_DIM
    INT_LOC = cfg_mod.INTERMEDIATE_LOCAL
    PAD_SWA = cfg_mod.NUM_HEADS_SWA_LOCAL_PAD
    ROTARY_DIM_SWA = ROTARY_HALF_SWA * 2
    n_swa = sum(1 for t in LAYER_TYPES if t == LAYER_TYPE_SWA)
    n_dense = len(DENSE_LAYER_INDICES)

    print(
        f"[ST-2 swa_dense] program={prog_name} platform={args.platform} "
        f"device={args.device}  dims: H={HIDDEN} H_Q_SWA={H_Q_SWA} KV={KV_H} "
        f"INT={INT_LOC} BATCH={BATCH} MAX_SEQ={MAX_SEQ_DEFAULT} "
        f"slide={SLIDING_WINDOW} n_swa={n_swa}",
        flush=True,
    )

    g = torch.Generator().manual_seed(args.seed)
    bf16 = torch.bfloat16

    def _randn(shape, dtype=torch.bfloat16, std=0.02):
        return (
            torch.empty(shape, dtype=torch.float32)
            .normal_(0.0, std, generator=g)
            .to(dtype)
        )

    w_input_rms = torch.empty(NUM_HIDDEN_LAYERS, HIDDEN).normal_(
        0.0, 0.05, generator=g)
    w_post_rms = torch.empty(NUM_HIDDEN_LAYERS, HIDDEN).normal_(
        0.0, 0.05, generator=g)
    w_q_norm = torch.empty(NUM_HIDDEN_LAYERS, HEAD_DIM).normal_(
        0.0, 0.05, generator=g)
    w_k_norm = torch.empty(NUM_HIDDEN_LAYERS, HEAD_DIM).normal_(
        0.0, 0.05, generator=g)
    wq = _randn([n_swa, HIDDEN, H_Q_SWA])
    wk = _randn([n_swa, HIDDEN, KV_H])
    wv = _randn([n_swa, HIDDEN, KV_H])
    wo = _randn([n_swa, H_Q_SWA, HIDDEN])
    w_g = _randn([n_swa, HIDDEN, PAD_SWA])
    # gate_r = ones -> o_proj multiplies attn_out by 1 (identity), matching the
    # no-gate golden (_torch_swa_attn_no_gate). The kernel now applies the head
    # gate inline in o_proj via gate_r (worker-precomputed in production); here
    # we feed ones so this ST keeps validating the un-gated attention path.
    gate_r = torch.ones(PAD_SWA, H_Q_SWA, dtype=bf16)
    w_gate = _randn([n_dense, HIDDEN, INT_LOC])
    w_up = _randn([n_dense, HIDDEN, INT_LOC])
    w_down = _randn([n_dense, INT_LOC, HIDDEN])

    current_hidden = _randn([1, BATCH, HIDDEN], dtype=bf16, std=0.02)
    seq_lens = torch.ones(1, BATCH, dtype=torch.int32)
    block_table = torch.zeros(1, MAX_BLOCKS_PER_SEQ * BATCH, dtype=torch.int32)
    slot_mapping = torch.arange(BATCH, dtype=torch.int32).unsqueeze(0)

    rope_cos = torch.empty(1, MAX_SEQ_DEFAULT, ROTARY_DIM_SWA).normal_(
        0.0, 0.5, generator=g)
    rope_sin = torch.empty(1, MAX_SEQ_DEFAULT, ROTARY_DIM_SWA).normal_(
        0.0, 0.5, generator=g)
    k_cache = torch.zeros(1, MAX_SEQ_DEFAULT, HEAD_DIM, dtype=bf16)
    v_cache = torch.zeros(1, MAX_SEQ_DEFAULT, HEAD_DIM, dtype=bf16)

    def flat3(t):
        L, M, N = t.shape
        return t.reshape(1, L * M, N)

    next_hidden_out = torch.zeros(1, BATCH, HIDDEN, dtype=bf16)
    layer_idx = 0

    inputs = {
        "current_hidden": current_hidden,
        "input_rms_weight": w_input_rms.float().unsqueeze(0),
        "wq": flat3(wq), "wk": flat3(wk), "wv": flat3(wv),
        "q_norm_weight": w_q_norm.float().unsqueeze(0),
        "k_norm_weight": w_k_norm.float().unsqueeze(0),
        "seq_lens": seq_lens,
        "block_table": block_table,
        "slot_mapping": slot_mapping,
        "rope_cos": rope_cos,
        "rope_sin": rope_sin,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "wo": flat3(wo),
        "w_g": flat3(w_g),
        "gate_r": gate_r.unsqueeze(0),
        "post_rms_weight": w_post_rms.float().unsqueeze(0),
        "w_gate": flat3(w_gate),
        "w_up": flat3(w_up),
        "w_down": flat3(w_down),
        "next_hidden_out": next_hidden_out,
    }

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
        _spec("w_gate", bf16), _spec("w_up", bf16), _spec("w_down", bf16),
        _spec("next_hidden_out", bf16, is_out=True),
        ScalarSpec("layer_idx", torch.int32,
                   value=torch.tensor(layer_idx, dtype=torch.int32)),
    ]

    def golden_fn(values):
        hidden = values["current_hidden"][0]
        irms_layer = values["input_rms_weight"][0][layer_idx]
        wq_layer = values["wq"][0][:HIDDEN]
        wk_layer = values["wk"][0][:HIDDEN]
        wv_layer = values["wv"][0][:HIDDEN]
        wo_layer = values["wo"][0][:H_Q_SWA]
        qn_layer = values["q_norm_weight"][0][layer_idx]
        kn_layer = values["k_norm_weight"][0][layer_idx]
        sl = values["seq_lens"][0]
        bt = values["block_table"][0]
        sm = values["slot_mapping"][0]
        rc = values["rope_cos"][0]
        rs = values["rope_sin"][0]
        kc = values["k_cache"][0]
        vc = values["v_cache"][0]
        post_rms_layer = values["post_rms_weight"][0][layer_idx]
        w_gate_layer = values["w_gate"][0][:HIDDEN]
        w_up_layer = values["w_up"][0][:HIDDEN]
        w_down_layer = values["w_down"][0][:INT_LOC]

        # Mirror kernel "KV_HEADS_LOCAL=1, Q_GROUPS=1" hardcoded assumption.
        TP1_KV_HEADS = 1
        TP1_NUM_HEADS = Q_PER_KV_SWA  # 12 Q heads served by kvh=0
        wq_slim = wq_layer[:, :TP1_NUM_HEADS * HEAD_DIM]
        wk_slim = wk_layer[:, :TP1_KV_HEADS * HEAD_DIM]
        wv_slim = wv_layer[:, :TP1_KV_HEADS * HEAD_DIM]
        wo_slim = wo_layer[:TP1_NUM_HEADS * HEAD_DIM, :]

        resid1 = _torch_swa_attn_no_gate(
            hidden_states=hidden, input_rms_weight=irms_layer,
            wq_full=wq_slim, wk_full=wk_slim, wv_full=wv_slim,
            q_norm_weight=qn_layer, k_norm_weight=kn_layer,
            wo_full=wo_slim,
            seq_lens=sl, block_table=bt, slot_mapping=sm,
            rope_cos=rc, rope_sin=rs,
            k_cache_full=kc, v_cache_full=vc,
            num_heads_full=TP1_NUM_HEADS,
            num_kv_heads_full=TP1_KV_HEADS,
            head_dim=HEAD_DIM, rotary_dim=ROTARY_DIM_SWA,
            rotary_half=ROTARY_HALF_SWA, q_per_kv=Q_PER_KV_SWA,
            eps=EPS, block_size=BLOCK_SIZE,
            max_blocks_per_seq=MAX_BLOCKS_PER_SEQ,
            sliding_window=SLIDING_WINDOW,
        )
        next_h = _torch_dense_mlp(
            resid1, post_rms_layer, w_gate_layer, w_up_layer, w_down_layer,
            EPS,
        )
        values["next_hidden_out"][0] = next_h

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
        base = int(self.malloc(chip_idx, int(window_size)))
        offset = 0
        ptrs: dict[str, int] = {}
        for spec in buffers:
            ptrs[spec.name] = base + offset
            offset += int(spec.nbytes)
        ctx = ChipDomainContext(
            name=str(name), domain_rank=0, domain_size=1, device_ctx=0,
            local_window_base=base, actual_window_size=int(window_size),
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

    Orchestrator.allocate_domain = _single_rank_alloc_domain
    try:
        runtime_cfg = dict(platform=args.platform, device_id=args.device)
        if args.smoke or args.platform.endswith("sim"):
            result = run(
                program=program, specs=specs,
                runtime_cfg=runtime_cfg, compile_only=True,
            )
            print(f"[ST-2 swa_dense] SMOKE: {result}", flush=True)
            return 0 if result.passed else 1

        result = run(
            program=program, specs=specs,
            golden_fn=golden_fn,
            runtime_cfg=runtime_cfg,
            rtol=4e-2, atol=4e-2,
            compare_fn={
                "next_hidden_out": ratio_allclose(
                    atol=4e-2, rtol=4e-2, max_error_ratio=0.10,
                ),
            },
        )
        print(f"[ST-2 swa_dense] DEVICE: {result}", flush=True)
        return 0 if result.passed else 1
    finally:
        Orchestrator.allocate_domain = _orig_alloc_domain


if __name__ == "__main__":
    raise SystemExit(main())
