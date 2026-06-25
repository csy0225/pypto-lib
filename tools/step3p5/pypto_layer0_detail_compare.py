#!/usr/bin/env python3
"""Compare PyPTO layer0 torch math against vLLM detailed layer0 dumps.

This consumes the real vLLM eager/all-to-all tensor scene produced by
``STEP3P5_DUMP_DETAIL=layer_debug STEP3P5_DUMP_LAYERS=0`` and recomputes the
same layer0 boundaries with PyPTO's checkpoint loader:

* input RMSNorm
* per-rank Q/K/V projection
* per-rank Q/K RMSNorm
* per-rank attention gate logits
* post-attention residual identity
* post-attention RMSNorm
* dense MLP output reduced across all TP ranks
* final layer output

The attention core itself is not recomputed here; vLLM's attention backend does
not dump KV-cache block tables or the post-core attention tensor in the current
scene.  The comparison closes the real tensor-input validation around all
non-attention-core math needed by PyPTO layer0.
"""
from __future__ import annotations

import argparse
import json
import math
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


def _load_obj(path: Path) -> dict[str, Any]:
    import torch

    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"expected dict dump: {path}")
    return obj


def _select_files(dump_root: Path, *, occurrence: int) -> dict[str, dict[int, Path]]:
    grouped: dict[tuple[str, int], list[Path]] = defaultdict(list)
    for path in dump_root.glob("*.pt"):
        _idx, rank, name = _parse_dump_name(path)
        grouped[(name, rank)].append(path)
    selected: dict[str, dict[int, Path]] = defaultdict(dict)
    for (name, rank), paths in grouped.items():
        paths = sorted(paths, key=lambda p: _parse_dump_name(p)[0])
        if occurrence < len(paths):
            selected[name][rank] = paths[occurrence]
    return selected


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


def _compare(name: str, candidate, expected, *, rtol: float, atol: float) -> dict[str, Any]:
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
        "ok": pass_rate >= 0.999,
    }


def _rank_tensor(files: dict[str, dict[int, Path]], name: str, rank: int, key: str):
    path = files.get(name, {}).get(rank)
    if path is None:
        raise FileNotFoundError(f"missing dump {name} rank={rank}")
    obj = _load_obj(path)
    return obj[key]


def compare(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from models.step3p5.config import EPS, HEAD_DIM, NUM_HEADS_FULL_LOCAL
    from models.step3p5.weight_loader import (
        KEY_DENSE_DOWN,
        KEY_DENSE_GATE,
        KEY_DENSE_UP,
        KEY_INPUT_RMS,
        KEY_K_NORM,
        KEY_POST_ATTN_RMS,
        KEY_Q_NORM,
        KEY_WG_FULL,
        KEY_WK_FULL,
        KEY_WQ_FULL,
        KEY_WV_FULL,
        load_step3p5_weights_for_rank,
    )

    dump_root = Path(args.dump_root)
    files = _select_files(dump_root, occurrence=args.occurrence)

    reports: list[dict[str, Any]] = []
    rank_bundles = [
        load_step3p5_weights_for_rank(args.ckpt_dir, rank, args.tp_world_size)
        for rank in range(args.tp_world_size)
    ]

    layer_input_rank0 = _rank_tensor(files, "layer_00_layer_input", 0, "hidden_states")
    post_attn_resid_rank0 = _rank_tensor(files, "layer_00_post_attn_residual", 0, "hidden_states")
    post_attn_norm_rank0 = _rank_tensor(files, "layer_00_post_attn_norm", 0, "hidden_states")

    mlp_partials = []
    for rank, bundle in enumerate(rank_bundles):
        layer_input = _rank_tensor(files, "layer_00_layer_input", rank, "hidden_states")
        input_norm_expected = _rank_tensor(files, "layer_00_input_norm", rank, "hidden_states")
        input_norm = _zc_rmsnorm(layer_input, bundle[KEY_INPUT_RMS][0], EPS)
        reports.append(_compare(
            f"rank{rank}.input_norm", input_norm, input_norm_expected,
            rtol=args.rtol, atol=args.atol,
        ))

        # Use vLLM's dumped input_norm as the tensor-input boundary for
        # projection checks.  The input RMSNorm check above is reported
        # separately; using its CPU recomputation here would conflate a small
        # RMSNorm/NPU numerical delta with QKV/gate weight mapping.
        projection_input = input_norm_expected

        q_expected = _rank_tensor(files, "layer_00_qkv_proj", rank, "q")
        k_expected = _rank_tensor(files, "layer_00_qkv_proj", rank, "k")
        v_expected = _rank_tensor(files, "layer_00_qkv_proj", rank, "v")
        q = (projection_input.float() @ bundle[KEY_WQ_FULL][0].float()).bfloat16()
        k = (projection_input.float() @ bundle[KEY_WK_FULL][0].float()).bfloat16()
        v = (projection_input.float() @ bundle[KEY_WV_FULL][0].float()).bfloat16()
        reports.extend([
            _compare(f"rank{rank}.q_proj", q, q_expected, rtol=args.rtol, atol=args.atol),
            _compare(f"rank{rank}.k_proj", k, k_expected, rtol=args.rtol, atol=args.atol),
            _compare(f"rank{rank}.v_proj", v, v_expected, rtol=args.rtol, atol=args.atol),
        ])

        q_norm_expected = _rank_tensor(files, "layer_00_qk_norm", rank, "q")
        k_norm_expected = _rank_tensor(files, "layer_00_qk_norm", rank, "k")
        q_norm = _zc_rmsnorm(
            q.view(q.shape[0], NUM_HEADS_FULL_LOCAL, HEAD_DIM),
            bundle[KEY_Q_NORM][0],
            EPS,
        ).view(q.shape)
        k_norm = _zc_rmsnorm(
            k.view(k.shape[0], 1, HEAD_DIM),
            bundle[KEY_K_NORM][0],
            EPS,
        ).view(k.shape)
        reports.extend([
            _compare(f"rank{rank}.q_norm", q_norm, q_norm_expected, rtol=args.rtol, atol=args.atol),
            _compare(f"rank{rank}.k_norm", k_norm, k_norm_expected, rtol=args.rtol, atol=args.atol),
        ])

        gate_expected = _rank_tensor(files, "layer_00_attn_gate_logits", rank, "gate")
        gate = (projection_input.float() @ bundle[KEY_WG_FULL][0].float()).bfloat16()
        gate = gate[:, :gate_expected.shape[-1]]
        reports.append(_compare(
            f"rank{rank}.attn_gate_logits", gate, gate_expected,
            rtol=args.rtol, atol=args.atol,
        ))

        post_attn_resid = _rank_tensor(files, "layer_00_post_attn_residual", rank, "hidden_states")
        post_attn_norm_expected = _rank_tensor(files, "layer_00_post_attn_norm", rank, "hidden_states")
        post_attn_norm = _zc_rmsnorm(
            post_attn_resid, bundle[KEY_POST_ATTN_RMS][0], EPS,
        )
        reports.append(_compare(
            f"rank{rank}.post_attn_norm", post_attn_norm, post_attn_norm_expected,
            rtol=args.rtol, atol=args.atol,
        ))

        # Dense MLP is TP-sliced.  vLLM returns a reduced full-hidden ffn_output,
        # so we compute rank-local partials and reduce after the loop.
        mlp_partials.append(_dense_mlp_partial(
            post_attn_norm,
            bundle[KEY_DENSE_GATE][0],
            bundle[KEY_DENSE_UP][0],
            bundle[KEY_DENSE_DOWN][0],
        ))

        # Residual identity checks are exact semantic checks using vLLM tensors.
        attn_delta = _rank_tensor(files, "layer_00_post_attn_residual", rank, "attn_delta")
        resid_hidden = _rank_tensor(files, "layer_00_post_attn_residual", rank, "hidden_states")
        reports.append(_compare(
            f"rank{rank}.post_attn_residual_identity",
            (layer_input + attn_delta).bfloat16(),
            resid_hidden,
            rtol=0.0,
            atol=0.0,
        ))

    ffn_reduced = torch.stack([p.float() for p in mlp_partials], dim=0).sum(dim=0).bfloat16()
    for rank in range(args.tp_world_size):
        ffn_obj = _load_obj(files["layer_00_ffn_out"][rank])
        ffn_expected = ffn_obj["ffn_output"]
        layer_out_expected = ffn_obj["hidden_states"]
        reports.extend([
            _compare(
                f"rank{rank}.ffn_output_reduced",
                ffn_reduced,
                ffn_expected,
                rtol=args.mlp_rtol,
                atol=args.mlp_atol,
            ),
            _compare(
                f"rank{rank}.layer_out",
                (ffn_reduced.float() + post_attn_resid_rank0.float()).bfloat16(),
                layer_out_expected,
                rtol=args.mlp_rtol,
                atol=args.mlp_atol,
            ),
        ])

    ok = all(item["ok"] for item in reports)
    worst = min((item["pass_rate"] for item in reports), default=0.0)
    return {
        "dump_root": str(dump_root),
        "ckpt_dir": args.ckpt_dir,
        "occurrence": args.occurrence,
        "tp_world_size": args.tp_world_size,
        "rtol": args.rtol,
        "atol": args.atol,
        "mlp_rtol": args.mlp_rtol,
        "mlp_atol": args.mlp_atol,
        "num_checks": len(reports),
        "worst_pass_rate": worst,
        "ok": ok,
        "checks": reports,
        "note": (
            "This is a real layer0 tensor-input validation against vLLM dumps. "
            "The attention core is not recomputed because current dumps do not "
            "include cache block-table/backend attention intermediate state."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-root", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--occurrence", type=int, default=0)
    parser.add_argument("--tp-world-size", type=int, default=8)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--mlp-rtol", type=float, default=8e-2)
    parser.add_argument("--mlp-atol", type=float, default=8e-2)
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
