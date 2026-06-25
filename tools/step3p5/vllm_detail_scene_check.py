#!/usr/bin/env python3
"""Check a vLLM Step3p5 detailed layer-dump validation scene."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REQUIRED_LAYER0_TENSORS = {
    "model_input",
    "layer_00_layer_input",
    "layer_00_input_norm",
    "layer_00_qkv_proj",
    "layer_00_qk_norm",
    "layer_00_rope",
    "layer_00_attn_gate_logits",
    "layer_00_post_attn_residual",
    "layer_00_post_attn_norm",
    "layer_00_ffn_out",
    "layer_00_out",
    "main_logits",
}


def _parse_name(path: Path) -> tuple[int, int, str]:
    stem = path.name.removesuffix(".pt")
    prefix, rank_part, tensor_name = stem.split("_", 2)
    return int(prefix), int(rank_part.removeprefix("rank")), tensor_name


def _tensor_meta(path: Path) -> dict[str, Any]:
    import torch

    obj = torch.load(path, map_location="cpu")
    meta = obj.get("__meta__", {}) if isinstance(obj, dict) else {}
    tensors = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if torch.is_tensor(value):
                tensors[key] = {
                    "shape": tuple(value.shape),
                    "dtype": str(value.dtype),
                    "finite": bool(torch.isfinite(value.float()).all().item()),
                    "max_abs": float(value.float().abs().max().item()) if value.numel() else 0.0,
                }
    return {"meta": meta, "tensors": tensors}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.dump_root)
    files = sorted(root.glob("*.pt"), key=lambda p: _parse_name(p))
    counts = Counter(_parse_name(path)[2] for path in files)
    by_name_rank: dict[str, dict[int, Path]] = defaultdict(dict)
    for path in files:
        _idx, rank, name = _parse_name(path)
        by_name_rank[name][rank] = path

    required = sorted(REQUIRED_LAYER0_TENSORS)
    missing = [
        name for name in required
        if name not in by_name_rank
    ]
    incomplete_ranks = {
        name: sorted(set(range(args.tp_world_size)) - set(by_name_rank.get(name, {})))
        for name in required
        if name in by_name_rank
        and set(by_name_rank[name]) != set(range(args.tp_world_size))
    }

    samples = {}
    for name in required:
        path = by_name_rank.get(name, {}).get(0)
        if path is not None:
            samples[name] = {"file": str(path), **_tensor_meta(path)}

    finite_ok = True
    for name in required:
        for rank in range(args.tp_world_size):
            path = by_name_rank.get(name, {}).get(rank)
            if path is None:
                finite_ok = False
                continue
            meta = _tensor_meta(path)
            finite_ok = finite_ok and all(t["finite"] for t in meta["tensors"].values())

    return {
        "dump_root": str(root),
        "num_files": len(files),
        "tp_world_size": args.tp_world_size,
        "required_tensors": required,
        "counts": dict(sorted(counts.items())),
        "missing": missing,
        "incomplete_ranks": incomplete_ranks,
        "rank0_samples": samples,
        "finite_ok": finite_ok,
        "ok": bool(files) and not missing and not incomplete_ranks and finite_ok,
        "scenario": (
            "Real vLLM eager all-to-all run with layer0 detailed dump. "
            "Use layer_00_layer_input/input_norm/qkv_proj/qk_norm/rope/"
            "attn_gate_logits/post_attn_residual/post_attn_norm/ffn_out "
            "as PyPTO one-layer tensor-input validation boundaries."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-root", required=True)
    parser.add_argument("--tp-world-size", type=int, default=8)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = build_report(args)
    payload = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(payload)
    print(payload)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
