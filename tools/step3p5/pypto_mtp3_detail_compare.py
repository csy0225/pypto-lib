#!/usr/bin/env python3
"""Compare PyPTO MTP3 torch math against vLLM detailed MTP dumps."""
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


def _full_logits(normed, full_head, chunk_size: int):
    import torch

    chunks = []
    for start in range(0, full_head.shape[0], chunk_size):
        stop = min(start + chunk_size, full_head.shape[0])
        chunks.append(normed.float() @ full_head[start:stop].float().T)
    return torch.cat(chunks, dim=-1)


def compare(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from models.step3p5.config import EPS, HEAD_DIM, NUM_HEADS_SWA_LOCAL, NUM_HIDDEN_LAYERS
    from models.step3p5.weight_loader import (
        KEY_MTP_DENSE_DOWN,
        KEY_MTP_DENSE_GATE,
        KEY_MTP_DENSE_UP,
        KEY_MTP_EH_PROJ,
        KEY_MTP_ENORM,
        KEY_MTP_HNORM,
        KEY_MTP_INPUT_RMS,
        KEY_MTP_K_NORM,
        KEY_MTP_POST_ATTN_RMS,
        KEY_MTP_Q_NORM,
        KEY_MTP_SH_NORM,
        KEY_MTP_SH_OUT,
        KEY_MTP_WG,
        KEY_MTP_WK,
        KEY_MTP_WQ,
        KEY_MTP_WV,
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

    for global_layer in range(NUM_HIDDEN_LAYERS, NUM_HIDDEN_LAYERS + 3):
        mtp_idx = global_layer - NUM_HIDDEN_LAYERS
        layer_checks = []

        # MTP input projection is full on vLLM; PyPTO stores row shards.
        mtp_input = _load_obj(files[f"layer_{global_layer:02d}_mtp_input"][0])
        previous_hidden = mtp_input["previous_hidden_states"]
        embed_raw = _rank_tensor(files, f"layer_{global_layer:02d}_mtp_embed", 0, "inputs_embeds")
        norms = _load_obj(files[f"layer_{global_layer:02d}_mtp_norms"][0])
        enormed_expected = norms["enormed_inputs_embeds"]
        hnormed_expected = norms["hnormed_previous_hidden_states"]
        enormed = _zc_rmsnorm(embed_raw, bundles[0][KEY_MTP_ENORM][mtp_idx], EPS)
        hnormed = _zc_rmsnorm(previous_hidden, bundles[0][KEY_MTP_HNORM][mtp_idx], EPS)
        layer_checks.extend([
            _compare(
                f"layer{global_layer}.mtp_enorm",
                enormed,
                enormed_expected,
                rtol=args.rtol,
                atol=args.atol,
                pass_rate_threshold=args.pass_rate,
            ),
            _compare(
                f"layer{global_layer}.mtp_hnorm",
                hnormed,
                hnormed_expected,
                rtol=args.rtol,
                atol=args.atol,
                pass_rate_threshold=args.pass_rate,
            ),
        ])
        eh_input = torch.cat([enormed_expected, hnormed_expected], dim=-1)
        eh_full = torch.cat(
            [bundle[KEY_MTP_EH_PROJ][mtp_idx] for bundle in bundles], dim=0,
        )
        eh_out = (eh_input.float() @ eh_full.float().T).bfloat16()
        eh_expected = _rank_tensor(files, f"layer_{global_layer:02d}_mtp_eh_proj", 0, "hidden_states")
        layer_checks.append(_compare(
            f"layer{global_layer}.mtp_eh_proj",
            eh_out,
            eh_expected,
            rtol=args.mlp_rtol,
            atol=args.mlp_atol,
            pass_rate_threshold=args.pass_rate,
        ))

        ffn_partials = []
        post_resid_rank0 = _rank_tensor(
            files, f"layer_{global_layer:02d}_post_attn_residual", 0, "hidden_states",
        )
        for rank, bundle in enumerate(bundles):
            layer_input = _rank_tensor(
                files, f"layer_{global_layer:02d}_layer_input", rank, "hidden_states",
            )
            input_norm_expected = _rank_tensor(
                files, f"layer_{global_layer:02d}_input_norm", rank, "hidden_states",
            )
            input_norm = _zc_rmsnorm(layer_input, bundle[KEY_MTP_INPUT_RMS][mtp_idx], EPS)
            layer_checks.append(_compare(
                f"layer{global_layer}.rank{rank}.input_norm",
                input_norm,
                input_norm_expected,
                rtol=args.rtol,
                atol=args.atol,
                pass_rate_threshold=args.pass_rate,
            ))
            q_expected = _rank_tensor(files, f"layer_{global_layer:02d}_qkv_proj", rank, "q")
            k_expected = _rank_tensor(files, f"layer_{global_layer:02d}_qkv_proj", rank, "k")
            v_expected = _rank_tensor(files, f"layer_{global_layer:02d}_qkv_proj", rank, "v")
            q = (input_norm_expected.float() @ bundle[KEY_MTP_WQ][mtp_idx].float()).bfloat16()
            k = (input_norm_expected.float() @ bundle[KEY_MTP_WK][mtp_idx].float()).bfloat16()
            v = (input_norm_expected.float() @ bundle[KEY_MTP_WV][mtp_idx].float()).bfloat16()
            layer_checks.extend([
                _compare(f"layer{global_layer}.rank{rank}.q_proj", q, q_expected,
                         rtol=args.rtol, atol=args.atol, pass_rate_threshold=args.pass_rate),
                _compare(f"layer{global_layer}.rank{rank}.k_proj", k, k_expected,
                         rtol=args.rtol, atol=args.atol, pass_rate_threshold=args.pass_rate),
                _compare(f"layer{global_layer}.rank{rank}.v_proj", v, v_expected,
                         rtol=args.rtol, atol=args.atol, pass_rate_threshold=args.pass_rate),
            ])
            q_norm_expected = _rank_tensor(files, f"layer_{global_layer:02d}_qk_norm", rank, "q")
            k_norm_expected = _rank_tensor(files, f"layer_{global_layer:02d}_qk_norm", rank, "k")
            q_norm = _zc_rmsnorm(
                q.view(q.shape[0], NUM_HEADS_SWA_LOCAL, HEAD_DIM),
                bundle[KEY_MTP_Q_NORM][mtp_idx],
                EPS,
            ).view(q.shape)
            k_norm = _zc_rmsnorm(
                k.view(k.shape[0], 1, HEAD_DIM),
                bundle[KEY_MTP_K_NORM][mtp_idx],
                EPS,
            ).view(k.shape)
            layer_checks.extend([
                _compare(f"layer{global_layer}.rank{rank}.q_norm", q_norm, q_norm_expected,
                         rtol=args.rtol, atol=args.atol, pass_rate_threshold=args.pass_rate),
                _compare(f"layer{global_layer}.rank{rank}.k_norm", k_norm, k_norm_expected,
                         rtol=args.rtol, atol=args.atol, pass_rate_threshold=args.pass_rate),
            ])
            gate_expected = _rank_tensor(
                files, f"layer_{global_layer:02d}_attn_gate_logits", rank, "gate",
            )
            gate = (input_norm_expected.float() @ bundle[KEY_MTP_WG][mtp_idx].float()).bfloat16()
            gate = gate[:, :gate_expected.shape[-1]]
            layer_checks.append(_compare(
                f"layer{global_layer}.rank{rank}.attn_gate_logits",
                gate,
                gate_expected,
                rtol=args.rtol,
                atol=args.atol,
                pass_rate_threshold=args.pass_rate,
            ))
            post_norm_expected = _rank_tensor(
                files, f"layer_{global_layer:02d}_post_attn_norm", rank, "hidden_states",
            )
            post_resid = _rank_tensor(
                files, f"layer_{global_layer:02d}_post_attn_residual", rank, "hidden_states",
            )
            post_norm = _zc_rmsnorm(
                post_resid, bundle[KEY_MTP_POST_ATTN_RMS][mtp_idx], EPS,
            )
            layer_checks.append(_compare(
                f"layer{global_layer}.rank{rank}.post_attn_norm",
                post_norm,
                post_norm_expected,
                rtol=args.rtol,
                atol=args.atol,
                pass_rate_threshold=args.pass_rate,
            ))
            attn_delta = _rank_tensor(
                files, f"layer_{global_layer:02d}_post_attn_residual", rank, "attn_delta",
            )
            layer_checks.append(_compare(
                f"layer{global_layer}.rank{rank}.post_attn_residual_identity",
                (layer_input.float() + attn_delta.float()).bfloat16(),
                post_resid,
                rtol=0.0,
                atol=0.0,
                pass_rate_threshold=1.0,
            ))
            ffn_partials.append(_dense_mlp_partial(
                post_norm_expected,
                bundle[KEY_MTP_DENSE_GATE][mtp_idx],
                bundle[KEY_MTP_DENSE_UP][mtp_idx],
                bundle[KEY_MTP_DENSE_DOWN][mtp_idx],
            ))

        ffn_reduced = torch.stack([p.float() for p in ffn_partials], dim=0).sum(dim=0).bfloat16()
        for rank in range(args.tp_world_size):
            ffn_obj = _load_obj(files[f"layer_{global_layer:02d}_ffn_out"][rank])
            layer_checks.extend([
                _compare(
                    f"layer{global_layer}.rank{rank}.ffn_output",
                    ffn_reduced,
                    ffn_obj["ffn_output"],
                    rtol=args.mlp_rtol,
                    atol=args.mlp_atol,
                    pass_rate_threshold=args.pass_rate,
                ),
                _compare(
                    f"layer{global_layer}.rank{rank}.layer_out",
                    (ffn_reduced.float() + post_resid_rank0.float()).bfloat16(),
                    ffn_obj["hidden_states"],
                    rtol=args.mlp_rtol,
                    atol=args.mlp_atol,
                    pass_rate_threshold=args.pass_rate,
                ),
            ])

        logits_obj = _load_obj(files[f"layer_{global_layer:02d}_mtp_logits"][0])
        logits_hidden = logits_obj["hidden_states"]
        sh_norm = _zc_rmsnorm(logits_hidden, bundles[0][KEY_MTP_SH_NORM][mtp_idx], EPS)
        sh_full = torch.cat([bundle[KEY_MTP_SH_OUT][mtp_idx] for bundle in bundles], dim=0)
        logits = _full_logits(sh_norm, sh_full, args.logits_chunk_size).bfloat16()
        layer_checks.extend([
            _compare(
                f"layer{global_layer}.shared_head_norm",
                sh_norm,
                logits_obj["normed_hidden_states"],
                rtol=args.rtol,
                atol=args.atol,
                pass_rate_threshold=args.pass_rate,
            ),
            _compare(
                f"layer{global_layer}.shared_head_logits",
                logits,
                logits_obj["logits"],
                rtol=args.logits_rtol,
                atol=args.logits_atol,
                pass_rate_threshold=args.pass_rate,
            ),
        ])

        reports.extend(layer_checks)
        layer_reports.append({
            "layer": global_layer,
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
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-root", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--tp-world-size", type=int, default=8)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--mlp-rtol", type=float, default=8e-2)
    parser.add_argument("--mlp-atol", type=float, default=2e-1)
    parser.add_argument("--logits-rtol", type=float, default=8e-2)
    parser.add_argument("--logits-atol", type=float, default=8e-2)
    parser.add_argument("--logits-chunk-size", type=int, default=4096)
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
