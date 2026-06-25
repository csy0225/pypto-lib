#!/usr/bin/env python3
"""Inspect vLLM Step3p5 golden dumps and report PyPTO comparison readiness.

This is the first bridge toward a tensor-input PyPTO decode precision harness:
it deliberately bypasses tokenizer/chat concerns by consuming vLLM tensor dumps
directly, and it implements the deterministic temperature=0 sampler as argmax
over dumped logits.

The current PyPTO Step3p5 implementation does not yet expose a full-network
tensor-input decode runner. Until that is wired, this tool reports the vLLM
golden oracle and the concrete blockers that prevent true end-to-end parity.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _parse_dump_index(path: str) -> tuple[int, int, str]:
    """Parse ``000140_rank0_main_logits.pt`` into index, rank, name."""
    name = Path(path).name
    stem = name.removesuffix(".pt")
    prefix, rank_part, tensor_name = stem.split("_", 2)
    return int(prefix), int(rank_part.removeprefix("rank")), tensor_name


def _load_logits(path: Path):
    import torch

    obj = torch.load(path, map_location="cpu")
    logits = obj["logits"].float()
    return logits, str(obj["logits"].dtype)


def _logits_stats(path: Path) -> dict[str, Any]:
    import torch

    logits, dtype = _load_logits(path)
    last = logits[-1]
    value, token = last.max(dim=0)
    return {
        "token_id": int(token.item()),
        "logit": float(value.item()),
        "shape": tuple(logits.shape),
        "dtype": dtype,
        "finite": bool(torch.isfinite(logits).all().item()),
        "max_abs": float(logits.abs().max().item()) if logits.numel() else 0.0,
        "row_index": int(logits.shape[0] - 1),
    }


def _compare_logits(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    import torch

    ref, _ = _load_logits(reference_path)
    cand, _ = _load_logits(candidate_path)
    if tuple(ref.shape) != tuple(cand.shape):
        return {
            "shape_match": False,
            "pass_rate": 0.0,
            "max_abs_diff": None,
            "mean_abs_diff": None,
        }
    diff = (ref - cand).abs()
    close = torch.isclose(ref, cand, rtol=0.0, atol=0.0)
    return {
        "shape_match": True,
        "pass_rate": float(close.float().mean().item()),
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
    }


def _summarize_case(case: dict[str, Any], *, max_steps: int | None) -> dict[str, Any]:
    by_step: dict[int, dict[int, Path]] = defaultdict(dict)
    tensor_counts: dict[str, int] = defaultdict(int)
    first_meta: dict[str, Any] = {}

    for item in case.get("dump_files", []):
        index, rank, tensor_name = _parse_dump_index(item["file"])
        tensor_counts[tensor_name] += 1
        first_meta.setdefault(tensor_name, item.get("meta", {}))
        if tensor_name == "main_logits":
            by_step[index][rank] = Path(item["file"])

    steps = sorted(by_step)
    if max_steps is not None:
        steps = steps[:max_steps]

    sampler = []
    logits_correctness = []
    for step in steps:
        rank_to_file = by_step[step]
        rank_results = {}
        for rank, file_path in sorted(rank_to_file.items()):
            stats = _logits_stats(file_path)
            stats["file"] = str(file_path)
            rank_results[str(rank)] = stats
        tokens = {entry["token_id"] for entry in rank_results.values()}
        shapes = {tuple(entry["shape"]) for entry in rank_results.values()}
        rank0_path = rank_to_file.get(0)
        rank_comparisons = {}
        if rank0_path is not None:
            for rank, file_path in sorted(rank_to_file.items()):
                rank_comparisons[str(rank)] = _compare_logits(rank0_path, file_path)
        all_rank_exact = bool(rank_comparisons) and all(
            item["shape_match"] and item["pass_rate"] == 1.0
            for item in rank_comparisons.values()
        )
        all_finite = all(entry["finite"] for entry in rank_results.values())
        sampler.append({
            "dump_index": step,
            "num_ranks": len(rank_results),
            "rank0_token_id": rank_results.get("0", {}).get("token_id"),
            "rank_argmax_consensus": len(tokens) == 1,
            "rank_results": rank_results,
        })
        logits_correctness.append({
            "dump_index": step,
            "num_ranks": len(rank_results),
            "shape_consensus": len(shapes) == 1,
            "all_finite": all_finite,
            "rank_exact_match": all_rank_exact,
            "rank_comparisons_to_rank0": rank_comparisons,
        })

    return {
        "name": case["name"],
        "num_dump_files": case.get("num_dump_files"),
        "tensor_counts": dict(sorted(tensor_counts.items())),
        "first_meta": first_meta,
        "sampler_temperature0": sampler,
        "final_logits_correctness": logits_correctness,
        "response_summary": case.get("response_summary"),
    }


def _run_pypto_acceptance(repo_root: Path, ckpt_dir: str) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(repo_root / "tools" / "step3p5" / "decode_acceptance.py"),
        "--ckpt-dir",
        ckpt_dir,
        "--tp-world-size",
        "8",
        "--rank",
        "0",
        "--json",
    ]
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError:
        parsed = None
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout_json": parsed,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    golden_root = Path(args.golden_root)
    manifest_path = golden_root / "manifest.json"
    summary_path = golden_root / "summary.json"
    manifest = _load_json(manifest_path)
    summary = _load_json(summary_path) if summary_path.exists() else {}

    case_reports = [
        _summarize_case(case, max_steps=args.max_steps)
        for case in manifest.get("cases", [])
    ]

    repo_root = Path(__file__).resolve().parents[2]
    pypto_acceptance = None
    if args.run_pypto_acceptance:
        pypto_acceptance = _run_pypto_acceptance(repo_root, args.ckpt_dir)

    return {
        "status": "GOLDEN_READY_PYPTO_E2E_BLOCKED",
        "golden_root": str(golden_root),
        "manifest": str(manifest_path),
        "summary": str(summary_path),
        "service": summary.get("service"),
        "cases": case_reports,
        "full_e2e": summary.get("full_e2e"),
        "pypto_acceptance": pypto_acceptance,
        "sampler_policy": {
            "temperature": 0,
            "implementation": "argmax over dumped main_logits",
            "note": "Tokenizer is bypassed for tensor-input comparison; token IDs are the comparable artifact.",
        },
        "pypto_tensor_input_decode_status": {
            "implemented": False,
            "next_step": (
                "Wire a PyPTO runner that accepts vLLM model_input hidden_states, "
                "positions, and per-rank checkpoint slices, then emits layer_XX_out "
                "and main_logits tensors with the same names/shapes as this report."
            ),
        },
        "blockers": [
            "PyPTO does not yet expose a full-network tensor-input Step3p5 decode runner.",
            "Existing PyPTO --no-smoke path compiles/runs only layer 0, not 45+3 layers.",
            "Existing real-NPU layer0 path still needs runtime environment and TP=8 weight packing cleanup.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden-root",
        required=True,
        help="Directory containing vLLM golden manifest.json and case dumps.",
    )
    parser.add_argument(
        "--ckpt-dir",
        default="/mnt/nvme1/chensiyu/step3p5_flash_release_hf_mtp3_bf16",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--run-pypto-acceptance", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = build_report(args)
    payload = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
