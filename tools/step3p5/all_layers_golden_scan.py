#!/usr/bin/env python3
"""Scan vLLM Step3p5 layer dumps and validate all decoder-layer artifacts.

This is a lightweight all-layer gate for the current tensor-dump workflow.  It
does not recompute attention/MLP math; the dump does not include KV-cache or
intermediate tensors needed to rerun a decoder layer independently.  Instead it
checks that every dumped decode/prefill step contains all 45 main-layer outputs
on all TP ranks, and that the tensors have consistent shapes and finite values.
Rank-to-rank exactness is recorded as a diagnostic, not a pass criterion.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from bisect import bisect_right
from pathlib import Path
from typing import Any


def _parse_dump_index(path: str) -> tuple[int, int, str]:
    stem = Path(path).name.removesuffix(".pt")
    prefix, rank_part, tensor_name = stem.split("_", 2)
    return int(prefix), int(rank_part.removeprefix("rank")), tensor_name


def _layer_id(name: str) -> int | None:
    if not name.startswith("layer_") or not name.endswith("_out"):
        return None
    return int(name.removeprefix("layer_").removesuffix("_out"))


def _load_hidden(path: Path):
    import torch

    obj = torch.load(path, map_location="cpu")
    hidden = obj["hidden_states"].float()
    return hidden


def _compare_hidden(reference: Path, candidate: Path) -> dict[str, Any]:
    import torch

    ref = _load_hidden(reference)
    cand = _load_hidden(candidate)
    if tuple(ref.shape) != tuple(cand.shape):
        return {
            "shape_match": False,
            "finite": bool(torch.isfinite(cand).all().item()),
            "exact": False,
            "pass_rate": 0.0,
            "max_abs_diff": None,
        }
    diff = (ref - cand).abs()
    exact = ref == cand
    return {
        "shape_match": True,
        "finite": bool(torch.isfinite(cand).all().item()),
        "exact": bool(exact.all().item()),
        "pass_rate": float(exact.float().mean().item()),
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "shape": tuple(cand.shape),
    }


def scan_case(case: dict[str, Any], *, tp_world_size: int, num_layers: int) -> dict[str, Any]:
    step_starts = sorted(
        _parse_dump_index(item["file"])[0]
        for item in case.get("dump_files", [])
        if item.get("meta", {}).get("name") == "model_input"
        and item.get("meta", {}).get("rank") == 0
    )
    by_step: dict[int, dict[int, dict[int, Path]]] = defaultdict(lambda: defaultdict(dict))
    for item in case.get("dump_files", []):
        meta = item.get("meta", {})
        layer = _layer_id(meta.get("name", ""))
        if layer is None:
            continue
        dump_index, rank, _ = _parse_dump_index(item["file"])
        step_pos = bisect_right(step_starts, dump_index) - 1
        if step_pos < 0:
            continue
        by_step[step_pos][layer][rank] = Path(item["file"])

    step_reports = []
    for step in sorted(by_step):
        missing_layers = [
            layer for layer in range(num_layers)
            if layer not in by_step[step]
        ]
        layer_reports = []
        for layer in range(num_layers):
            rank_files = by_step[step].get(layer, {})
            missing_ranks = [
                rank for rank in range(tp_world_size)
                if rank not in rank_files
            ]
            rank0 = rank_files.get(0)
            comparisons = {}
            if rank0 is not None:
                for rank, path in sorted(rank_files.items()):
                    comparisons[str(rank)] = _compare_hidden(rank0, path)
            all_shape_match = bool(comparisons) and all(
                c["shape_match"] for c in comparisons.values()
            )
            all_finite = bool(comparisons) and all(
                c["finite"] for c in comparisons.values()
            )
            layer_reports.append({
                "layer": layer,
                "num_ranks": len(rank_files),
                "missing_ranks": missing_ranks,
                "all_shape_match": all_shape_match,
                "all_finite": all_finite,
                "rank_exact_match": (
                    not missing_ranks
                    and bool(comparisons)
                    and all(c["shape_match"] and c["finite"] and c["exact"]
                            for c in comparisons.values())
                ),
                "shape": next(
                    (c.get("shape") for c in comparisons.values() if c.get("shape")),
                    None,
                ),
                "rank_comparisons_to_rank0": comparisons,
            })

        step_ok = (
            not missing_layers
            and all(not item["missing_ranks"] and item["all_shape_match"] and item["all_finite"]
                    for item in layer_reports)
        )
        step_reports.append({
            "step": step,
            "start_dump_index": step_starts[step] if step < len(step_starts) else None,
            "missing_layers": missing_layers,
            "layers": layer_reports,
            "ok": step_ok,
        })

    return {
        "name": case["name"],
        "num_steps": len(step_reports),
        "steps": step_reports,
        "ok": bool(step_reports) and all(step["ok"] for step in step_reports),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    golden_root = Path(args.golden_root)
    manifest = json.loads((golden_root / "manifest.json").read_text())

    cases = [
        scan_case(case, tp_world_size=args.tp_world_size, num_layers=args.num_layers)
        for case in manifest.get("cases", [])
        if case.get("dump_files")
    ]
    return {
        "golden_root": str(golden_root),
        "tp_world_size": args.tp_world_size,
        "num_layers": args.num_layers,
        "cases": cases,
        "ok": bool(cases) and all(case["ok"] for case in cases),
        "note": (
            "This validates all dumped layer outputs across ranks. It does not "
            "rerun PyPTO layer math because current golden dumps do not include "
            "the KV-cache/intermediate tensors required to recompute attention. "
            "Layer hidden states are checked for completeness/shape/finite; rank "
            "exactness is reported diagnostically because TP workers may expose "
            "pre-gather residuals at this hook point."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-root", required=True)
    parser.add_argument("--tp-world-size", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=45)
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
