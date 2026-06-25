#!/usr/bin/env python3
"""Compare PyPTO all-layer torch math against vLLM detailed layer dumps."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_dump_name(path: Path) -> tuple[int, int, str]:
    stem = path.name.removesuffix(".pt")
    prefix, rank_part, tensor_name = stem.split("_", 2)
    return int(prefix), int(rank_part.removeprefix("rank")), tensor_name


def _select_files(dump_root: Path) -> dict[str, dict[int, Path]]:
    selected: dict[str, dict[int, Path]] = defaultdict(dict)
    for path in sorted(dump_root.glob("*.pt"), key=_parse_dump_name):
        _idx, rank, name = _parse_dump_name(path)
        selected[name][rank] = path
    return selected


def _load_obj(path: Path) -> dict[str, Any]:
    import torch

    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"expected dict dump: {path}")
    return obj


def _rank_tensor(files: dict[str, dict[int, Path]], name: str, rank: int, key: str):
    path = files.get(name, {}).get(rank)
    if path is None:
        raise FileNotFoundError(f"missing dump {name} rank={rank}")
    return _load_obj(path)[key]


def _zc_rmsnorm(x, gamma, eps: float):
    import torch

    x_f = x.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    return (x_f * torch.rsqrt(var + eps) * (gamma.float() + 1.0)).bfloat16()


def _dense_mlp_partial(post_norm, w_gate, w_up, w_down):
    import torch

    gate = post_norm.float() @ w_gate.float()
    up = post_norm.float() @ w_up.float()
    hidden = (gate * torch.sigmoid(gate) * up).bfloat16()
    return (hidden.float() @ w_down.float()).bfloat16()


def _moe_ref_dynamic(
    routed_swiglu_limit: float,
    shared_swiglu_limit: float,
    x,
    gate_w_full,
    router_bias_full,
    w_gate_r_full,
    w_up_r_full,
    w_down_r_full,
    w_gate_s_full,
    w_up_s_full,
    w_down_s_full,
    topk_ids=None,
    topk_weights=None,
):
    import torch
    import torch.nn.functional as F

    top_k = 8
    router_scaling = 3.0
    token_count = x.shape[0]
    hidden = x.shape[1]

    if topk_ids is None or topk_weights is None:
        logits = x.float() @ gate_w_full.float()
        score = torch.sigmoid(logits)
        biased = score + router_bias_full.float().view(1, -1)
        indices = torch.topk(biased, k=top_k, dim=-1, sorted=False)[1]
        topk_vals = torch.gather(score, dim=-1, index=indices.long())
        weights = (topk_vals / topk_vals.sum(dim=-1, keepdim=True)) * router_scaling
    else:
        indices = topk_ids.long()
        weights = topk_weights.float()
    weights_bf = weights.float()

    routed_acc = torch.zeros(token_count, hidden, dtype=torch.float32)
    for token_idx in range(token_count):
        for top_idx in range(top_k):
            expert_id = int(indices[token_idx, top_idx].item())
            x_row = x[token_idx:token_idx + 1, :].float()
            gate_a = x_row @ w_gate_r_full[expert_id].float()
            up_a = x_row @ w_up_r_full[expert_id].float()
            if routed_swiglu_limit > 0.0:
                silu_g = F.silu(gate_a).clamp(max=routed_swiglu_limit)
                up_c = up_a.clamp(
                    min=-routed_swiglu_limit,
                    max=routed_swiglu_limit,
                )
                moe_hidden = silu_g * up_c
            else:
                moe_hidden = F.silu(gate_a) * up_a
            y = moe_hidden.to(torch.bfloat16).float() @ w_down_r_full[expert_id].float()
            routed_acc[token_idx, :] += weights_bf[token_idx, top_idx] * y[0]

    sh_gate = x.float() @ w_gate_s_full.float()
    sh_up = x.float() @ w_up_s_full.float()
    if shared_swiglu_limit > 0.0:
        sh_silu = F.silu(sh_gate).clamp(max=shared_swiglu_limit)
        sh_up_c = sh_up.clamp(
            min=-shared_swiglu_limit,
            max=shared_swiglu_limit,
        )
        sh_hidden = sh_silu * sh_up_c
    else:
        sh_hidden = F.silu(sh_gate) * sh_up
    sh_out = sh_hidden.to(torch.bfloat16).float() @ w_down_s_full.float()
    return (sh_out + routed_acc).to(torch.bfloat16)


def _compare(name: str, candidate, expected, *, rtol: float, atol: float,
             pass_rate_threshold: float) -> dict[str, Any]:
    import torch

    if tuple(candidate.shape) != tuple(expected.shape):
        return {
            "name": name,
            "shape": tuple(candidate.shape),
            "expected_shape": tuple(expected.shape),
            "shape_match": False,
            "finite": bool(torch.isfinite(candidate.float()).all().item()),
            "pass_rate": 0.0,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "ok": False,
        }
    diff = (candidate.float() - expected.float()).abs()
    close = torch.isclose(candidate.float(), expected.float(), rtol=rtol, atol=atol)
    pass_rate = float(close.float().mean().item())
    return {
        "name": name,
        "shape": tuple(candidate.shape),
        "expected_shape": tuple(expected.shape),
        "shape_match": True,
        "finite": bool(torch.isfinite(candidate.float()).all().item()),
        "pass_rate": pass_rate,
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "ok": pass_rate >= pass_rate_threshold,
    }


def _dense_pos(layer_idx: int, dense_indices: tuple[int, ...]) -> int:
    return dense_indices.index(layer_idx)


def _moe_pos(layer_idx: int, moe_indices: tuple[int, ...]) -> int:
    return moe_indices.index(layer_idx)


def _attn_pos(layer_idx: int, *, full: bool, layer_types: tuple[str, ...],
              full_type: str) -> int:
    count = 0
    for idx in range(layer_idx + 1):
        is_full = layer_types[idx] == full_type
        if is_full == full:
            count += 1
    return count - 1


def compare(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from models.step3p5.config import (
        EPS,
        HEAD_DIM,
        LAYER_TYPE_FULL,
        LAYER_TYPES,
        MOE_LAYER_INDICES,
        NUM_HIDDEN_LAYERS,
        NUM_HEADS_FULL_LOCAL,
        NUM_HEADS_SWA_LOCAL,
        DENSE_LAYER_INDICES,
        SWIGLU_LIMITS,
        SWIGLU_LIMITS_SHARED,
        is_full_attention,
        is_moe_layer,
    )
    from models.step3p5.weight_loader import (
        KEY_DENSE_DOWN,
        KEY_DENSE_GATE,
        KEY_DENSE_UP,
        KEY_INPUT_RMS,
        KEY_K_NORM,
        KEY_MOE_GATE_W,
        KEY_MOE_ROUTER_BIAS,
        KEY_MOE_W_DOWN_R,
        KEY_MOE_W_DOWN_S,
        KEY_MOE_W_GATE_R,
        KEY_MOE_W_GATE_S,
        KEY_MOE_W_UP_R,
        KEY_MOE_W_UP_S,
        KEY_POST_ATTN_RMS,
        KEY_Q_NORM,
        KEY_WG_FULL,
        KEY_WG_SWA,
        KEY_WK_FULL,
        KEY_WK_SWA,
        KEY_WQ_FULL,
        KEY_WQ_SWA,
        KEY_WV_FULL,
        KEY_WV_SWA,
        load_step3p5_weights_for_rank,
    )

    dump_root = Path(args.dump_root)
    files = _select_files(dump_root)
    bundles = [
        load_step3p5_weights_for_rank(args.ckpt_dir, rank, args.tp_world_size)
        for rank in range(args.tp_world_size)
    ]
    reports: list[dict[str, Any]] = []
    layer_reports = []

    for layer_idx in range(NUM_HIDDEN_LAYERS):
        full = is_full_attention(layer_idx)
        attn_pos = _attn_pos(
            layer_idx, full=full, layer_types=LAYER_TYPES, full_type=LAYER_TYPE_FULL,
        )
        q_key = KEY_WQ_FULL if full else KEY_WQ_SWA
        k_key = KEY_WK_FULL if full else KEY_WK_SWA
        v_key = KEY_WV_FULL if full else KEY_WV_SWA
        gate_key = KEY_WG_FULL if full else KEY_WG_SWA
        num_heads_local = NUM_HEADS_FULL_LOCAL if full else NUM_HEADS_SWA_LOCAL

        layer_checks = []
        ffn_partials = []
        post_attn_resid_rank0 = _rank_tensor(
            files, f"layer_{layer_idx:02d}_post_attn_residual", 0, "hidden_states",
        )
        post_attn_norm_rank0 = _rank_tensor(
            files, f"layer_{layer_idx:02d}_post_attn_norm", 0, "hidden_states",
        )

        for rank, bundle in enumerate(bundles):
            layer_input = _rank_tensor(
                files, f"layer_{layer_idx:02d}_layer_input", rank, "hidden_states",
            )
            input_norm_expected = _rank_tensor(
                files, f"layer_{layer_idx:02d}_input_norm", rank, "hidden_states",
            )
            input_norm = _zc_rmsnorm(layer_input, bundle[KEY_INPUT_RMS][layer_idx], EPS)
            layer_checks.append(_compare(
                f"layer{layer_idx:02d}.rank{rank}.input_norm",
                input_norm,
                input_norm_expected,
                rtol=args.rtol,
                atol=args.atol,
                pass_rate_threshold=args.pass_rate,
            ))

            projection_input = input_norm_expected
            q_expected = _rank_tensor(files, f"layer_{layer_idx:02d}_qkv_proj", rank, "q")
            k_expected = _rank_tensor(files, f"layer_{layer_idx:02d}_qkv_proj", rank, "k")
            v_expected = _rank_tensor(files, f"layer_{layer_idx:02d}_qkv_proj", rank, "v")
            q = (projection_input.float() @ bundle[q_key][attn_pos].float()).bfloat16()
            k = (projection_input.float() @ bundle[k_key][attn_pos].float()).bfloat16()
            v = (projection_input.float() @ bundle[v_key][attn_pos].float()).bfloat16()
            layer_checks.extend([
                _compare(f"layer{layer_idx:02d}.rank{rank}.q_proj", q, q_expected,
                         rtol=args.rtol, atol=args.atol, pass_rate_threshold=args.pass_rate),
                _compare(f"layer{layer_idx:02d}.rank{rank}.k_proj", k, k_expected,
                         rtol=args.rtol, atol=args.atol, pass_rate_threshold=args.pass_rate),
                _compare(f"layer{layer_idx:02d}.rank{rank}.v_proj", v, v_expected,
                         rtol=args.rtol, atol=args.atol, pass_rate_threshold=args.pass_rate),
            ])

            q_norm_expected = _rank_tensor(files, f"layer_{layer_idx:02d}_qk_norm", rank, "q")
            k_norm_expected = _rank_tensor(files, f"layer_{layer_idx:02d}_qk_norm", rank, "k")
            q_norm = _zc_rmsnorm(
                q.view(q.shape[0], num_heads_local, HEAD_DIM),
                bundle[KEY_Q_NORM][layer_idx],
                EPS,
            ).view(q.shape)
            k_norm = _zc_rmsnorm(
                k.view(k.shape[0], 1, HEAD_DIM),
                bundle[KEY_K_NORM][layer_idx],
                EPS,
            ).view(k.shape)
            layer_checks.extend([
                _compare(f"layer{layer_idx:02d}.rank{rank}.q_norm", q_norm, q_norm_expected,
                         rtol=args.rtol, atol=args.atol, pass_rate_threshold=args.pass_rate),
                _compare(f"layer{layer_idx:02d}.rank{rank}.k_norm", k_norm, k_norm_expected,
                         rtol=args.rtol, atol=args.atol, pass_rate_threshold=args.pass_rate),
            ])

            gate_expected = _rank_tensor(
                files, f"layer_{layer_idx:02d}_attn_gate_logits", rank, "gate",
            )
            gate = (projection_input.float() @ bundle[gate_key][attn_pos].float()).bfloat16()
            gate = gate[:, :gate_expected.shape[-1]]
            layer_checks.append(_compare(
                f"layer{layer_idx:02d}.rank{rank}.attn_gate_logits",
                gate,
                gate_expected,
                rtol=args.rtol,
                atol=args.atol,
                pass_rate_threshold=args.pass_rate,
            ))

            post_attn_resid = _rank_tensor(
                files, f"layer_{layer_idx:02d}_post_attn_residual", rank, "hidden_states",
            )
            post_attn_norm_expected = _rank_tensor(
                files, f"layer_{layer_idx:02d}_post_attn_norm", rank, "hidden_states",
            )
            post_attn_norm = _zc_rmsnorm(
                post_attn_resid, bundle[KEY_POST_ATTN_RMS][layer_idx], EPS,
            )
            layer_checks.append(_compare(
                f"layer{layer_idx:02d}.rank{rank}.post_attn_norm",
                post_attn_norm,
                post_attn_norm_expected,
                rtol=args.rtol,
                atol=args.atol,
                pass_rate_threshold=args.pass_rate,
            ))

            attn_delta = _rank_tensor(
                files, f"layer_{layer_idx:02d}_post_attn_residual", rank, "attn_delta",
            )
            layer_checks.append(_compare(
                f"layer{layer_idx:02d}.rank{rank}.post_attn_residual_identity",
                (layer_input.float() + attn_delta.float()).bfloat16(),
                post_attn_resid,
                rtol=0.0,
                atol=0.0,
                pass_rate_threshold=1.0,
            ))

            if not is_moe_layer(layer_idx):
                dense_pos = _dense_pos(layer_idx, DENSE_LAYER_INDICES)
                ffn_partials.append(_dense_mlp_partial(
                    post_attn_norm_expected,
                    bundle[KEY_DENSE_GATE][dense_pos],
                    bundle[KEY_DENSE_UP][dense_pos],
                    bundle[KEY_DENSE_DOWN][dense_pos],
                ))

        if is_moe_layer(layer_idx):
            moe_pos = _moe_pos(layer_idx, MOE_LAYER_INDICES)
            base = bundles[0]
            gate_w_full = base[KEY_MOE_GATE_W][moe_pos]
            router_bias_full = base[KEY_MOE_ROUTER_BIAS][moe_pos]
            w_gate_r_full = torch.cat(
                [bundle[KEY_MOE_W_GATE_R][moe_pos] for bundle in bundles], dim=0,
            )
            w_up_r_full = torch.cat(
                [bundle[KEY_MOE_W_UP_R][moe_pos] for bundle in bundles], dim=0,
            )
            w_down_r_full = torch.cat(
                [bundle[KEY_MOE_W_DOWN_R][moe_pos] for bundle in bundles], dim=0,
            )
            w_gate_s_full = torch.cat(
                [bundle[KEY_MOE_W_GATE_S][moe_pos] for bundle in bundles], dim=-1,
            )
            w_up_s_full = torch.cat(
                [bundle[KEY_MOE_W_UP_S][moe_pos] for bundle in bundles], dim=-1,
            )
            w_down_s_full = torch.cat(
                [bundle[KEY_MOE_W_DOWN_S][moe_pos] for bundle in bundles], dim=0,
            )
            ffn_reduced = _moe_ref_dynamic(
                float(SWIGLU_LIMITS[layer_idx]),
                float(SWIGLU_LIMITS_SHARED[layer_idx]),
                post_attn_norm_rank0,
                gate_w_full,
                router_bias_full,
                w_gate_r_full,
                w_up_r_full,
                w_down_r_full,
                w_gate_s_full,
                w_up_s_full,
                w_down_s_full,
                topk_ids=(
                    _rank_tensor(files, f"layer_{layer_idx:02d}_moe_router", 0, "topk_ids")
                    if f"layer_{layer_idx:02d}_moe_router" in files else None
                ),
                topk_weights=(
                    _rank_tensor(files, f"layer_{layer_idx:02d}_moe_router", 0, "topk_weights")
                    if f"layer_{layer_idx:02d}_moe_router" in files else None
                ),
            )
        else:
            ffn_reduced = torch.stack(
                [partial.float() for partial in ffn_partials], dim=0,
            ).sum(dim=0).bfloat16()

        for rank in range(args.tp_world_size):
            ffn_obj = _load_obj(files[f"layer_{layer_idx:02d}_ffn_out"][rank])
            ffn_expected = ffn_obj["ffn_output"]
            layer_out_expected = ffn_obj["hidden_states"]
            layer_checks.extend([
                _compare(
                    f"layer{layer_idx:02d}.rank{rank}.ffn_output",
                    ffn_reduced,
                    ffn_expected,
                    rtol=args.mlp_rtol,
                    atol=args.mlp_atol,
                    pass_rate_threshold=args.pass_rate,
                ),
                _compare(
                    f"layer{layer_idx:02d}.rank{rank}.layer_out",
                    (ffn_reduced.float() + post_attn_resid_rank0.float()).bfloat16(),
                    layer_out_expected,
                    rtol=args.mlp_rtol,
                    atol=args.mlp_atol,
                    pass_rate_threshold=args.pass_rate,
                ),
            ])

        reports.extend(layer_checks)
        layer_reports.append({
            "layer": layer_idx,
            "kind": "moe" if is_moe_layer(layer_idx) else "dense",
            "num_checks": len(layer_checks),
            "ok": all(item["ok"] for item in layer_checks),
            "worst_pass_rate": min(item["pass_rate"] for item in layer_checks),
            "failed": [item for item in layer_checks if not item["ok"]],
        })

    return {
        "dump_root": str(dump_root),
        "ckpt_dir": args.ckpt_dir,
        "tp_world_size": args.tp_world_size,
        "num_checks": len(reports),
        "ok": all(item["ok"] for item in reports),
        "worst_pass_rate": min((item["pass_rate"] for item in reports), default=0.0),
        "layers": layer_reports,
        "note": (
            "This validates all non-attention-core per-layer math against vLLM "
            "detail tensors. Attention backend output is consumed from vLLM "
            "post_attn_residual because current dump does not expose KV-cache "
            "backend internals needed to recompute it on CPU."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-root", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--tp-world-size", type=int, default=8)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--mlp-rtol", type=float, default=8e-2)
    parser.add_argument("--mlp-atol", type=float, default=8e-2)
    parser.add_argument("--pass-rate", type=float, default=0.999)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(_repo_root()))
    report = compare(args)
    payload = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(payload)
    print(payload)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
