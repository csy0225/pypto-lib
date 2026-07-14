#!/usr/bin/env python3
"""Faithful ctx=1 torch golden for the first N DENSE layers of step3p5.

Dense layers (0,1,2) are pure BF16 (no INT8, no MoE) -> this torch forward is
EXACT vs vLLM-ascend for the ctx=1 single-token case. At ctx=1, attention output
== value_states (softmax over a single element = 1), so q/k/rope/scale are NOT
exercised; only v_proj / head-gate(g_proj) / o_proj / norms / dense MLP / residual.

Mirrors modeling_step3p5.py:
  RMSNorm: x_fp32 * rsqrt(mean(x^2)+eps) * (weight+1)      (EPS=1e-5)
  DecoderLayer: resid=h; h=input_ln(h); h=attn(h); h=resid+h;
                resid=h; h=post_attn_ln(h); h=mlp(h); h=resid+h
  Attention (ctx=1): attn_out[head] = repeat_kv(v_proj(h))[head];
                     gate=sigmoid(g_proj(h)); attn_out*=gate[:,None];
                     o = o_proj(attn_out.flatten)
  MLP (dense, silu): down(silu(gate_proj(h)) * up_proj(h))

Emits golden next_hidden row0 (after N dense layers) + per-layer attention-residual
row0 to compare against pypto device P=0 dumps (cos / ratio_allclose row0, NOT max).
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from safetensors import safe_open

CKPT_DEFAULT = "/mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp"
HIDDEN = 4096
HEAD_DIM = 128
NUM_KV_HEADS = 8
EPS = 1e-5
# per-layer attention-head count (from ckpt o_proj shapes): L0 full=64, L1/L2 swa=96
LAYER_HEADS = {0: 64, 1: 96, 2: 96}


def _rmsnorm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return xf * torch.rsqrt(var + EPS) * (w.float() + 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--token", type=int, default=6127)
    ap.add_argument("--layers", type=int, default=3, help="num dense layers (0..2)")
    ap.add_argument("--layer-seq", default="", help="comma layer ids overriding 0..layers-1 (e.g. 0,1,1)")
    ap.add_argument("--out", default="/tmp/n1_vec")
    ap.add_argument("--no-hg", action="store_true", help="skip head-gate (gate=1)")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    args = ap.parse_args()
    seq = [int(x) for x in args.layer_seq.split(",")] if args.layer_seq else list(range(args.layers))

    idx = json.load(open(os.path.join(args.ckpt, "quant_model_weights.safetensors.index.json")))["weight_map"]
    handles: dict[str, object] = {}

    def get(name: str) -> torch.Tensor:
        shard = idx[name]
        h = handles.get(shard)
        if h is None:
            h = safe_open(os.path.join(args.ckpt, shard), framework="pt")
            h.__enter__()
            handles[shard] = h
        return h.get_tensor(name).float()

    emb = get("model.embed_tokens.weight")[args.token, :].clone()  # [4096]
    h = emb.view(1, HIDDEN)  # [1, 4096], single token, ctx=1
    print(f"[golden] embed(token={args.token}) row0 |.|={h.abs().max():.4f} norm={h.norm():.4f}")

    os.makedirs(args.out, exist_ok=True)
    for step_i, L in enumerate(seq):
        n_heads = LAYER_HEADS[L]
        n_rep = n_heads // NUM_KV_HEADS
        pn = f"model.layers.{L}"
        p = f"{pn}.self_attn"
        # --- attention block (ctx=1: attn_out = repeat_kv(v)) ---
        resid = h
        normed = _rmsnorm(h, get(f"{pn}.input_layernorm.weight"))
        v = normed @ get(f"{p}.v_proj.weight").t()                     # [1, KV_HIDDEN=1024]
        v = v.view(1, NUM_KV_HEADS, HEAD_DIM)                          # [1,8,128]
        attn = v.repeat_interleave(n_rep, dim=1)                       # [1, n_heads, 128]
        gate = torch.sigmoid(normed @ get(f"{p}.g_proj.weight").t())   # [1, n_heads]
        if not args.no_hg:
            attn = attn * gate.view(1, n_heads, 1)
        attn = attn.reshape(1, n_heads * HEAD_DIM)                     # [1, HIDDEN_Q]
        o = attn @ get(f"{p}.o_proj.weight").t()                       # [1, 4096]
        h = resid + o
        torch.save(h.squeeze(0).cpu(), os.path.join(args.out, f"golden_S{step_i}_resid1{args.tag}.pt"))
        print(f"[golden] step{step_i} L{L} heads={n_heads} resid1 |.|={h.abs().max():.4f} norm={h.norm():.4f}")
        # --- dense MLP block (silu, no clamp for layers 0-2) ---
        resid = h
        normed2 = _rmsnorm(h, get(f"{pn}.post_attention_layernorm.weight"))
        gate_mlp = torch.nn.functional.silu(normed2 @ get(f"{pn}.mlp.gate_proj.weight").t())
        up_mlp = normed2 @ get(f"{pn}.mlp.up_proj.weight").t()
        mlp = (gate_mlp * up_mlp) @ get(f"{pn}.mlp.down_proj.weight").t()
        h = resid + mlp
        torch.save(h.squeeze(0).cpu(), os.path.join(args.out, f"golden_S{step_i}_out{args.tag}.pt"))
        print(f"[golden] step{step_i} L{L} layer_out |.|={h.abs().max():.4f} norm={h.norm():.4f}")

    torch.save(h.squeeze(0).cpu(), os.path.join(args.out, f"golden_P0_nh{args.tag}.pt"))
    print(f"[golden] SAVED golden_P0_nh{args.tag}.pt (after {args.layers} dense layers) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
