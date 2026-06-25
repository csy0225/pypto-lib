#!/usr/bin/env python3
"""Generate PyPTO final-logits artifacts from vLLM dumped final hidden states.

This is a fast tokenizer-free Step3p5 precision closure for the tail of the
end-to-end decode path:

    vLLM dumped final hidden_states -> PyPTO final RMSNorm + lm_head -> logits

It intentionally reuses PyPTO's checkpoint naming/slicing assumptions while the
full tensor-input 45-layer decode runner is still being wired.  By default it
emits full-vocabulary logits for every rank artifact because the current vLLM
``main_logits`` dump is replicated full-vocab on each TP rank.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_dump_index(path: str) -> tuple[int, int, str]:
    name = Path(path).name.removesuffix(".pt")
    prefix, rank_part, tensor_name = name.split("_", 2)
    return int(prefix), int(rank_part.removeprefix("rank")), tensor_name


def _zero_centered_rmsnorm(x, gamma, eps: float = 1e-5):
    import torch

    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    return x.float() * torch.rsqrt(var + eps) * (gamma.float() + 1.0)


def _load_head_weights(ckpt_dir: str):
    from models.step3p5.weight_loader import _ShardCache, _read_index, _to_bf16

    weight_map = _read_index(ckpt_dir)
    with _ShardCache(ckpt_dir, weight_map) as cache:
        final_norm = _to_bf16(cache.get("model.norm.weight").contiguous())
        lm_head = _to_bf16(cache.get("lm_head.weight").contiguous())
    return final_norm, lm_head


def _load_vllm_main_logits_items(golden_root: Path, case_name: str) -> list[dict[str, Any]]:
    manifest = json.loads((golden_root / "manifest.json").read_text())
    for case in manifest.get("cases", []):
        if case["name"] == case_name:
            items = [
                item for item in case.get("dump_files", [])
                if item.get("meta", {}).get("name") == "main_logits"
            ]
            return sorted(items, key=lambda x: _parse_dump_index(x["file"]))
    raise KeyError(f"case {case_name!r} not found in {golden_root / 'manifest.json'}")


def _rank0_per_step(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_step: dict[int, dict[int, dict[str, Any]]] = {}
    for item in items:
        step, rank, _ = _parse_dump_index(item["file"])
        by_step.setdefault(step, {})[rank] = item
    return [by_step[step][0] for step in sorted(by_step) if 0 in by_step[step]]


def _logits_from_full_head(normed, lm_head, chunk_size: int):
    import torch

    chunks = []
    for start in range(0, lm_head.shape[0], chunk_size):
        stop = min(start + chunk_size, lm_head.shape[0])
        chunk = normed.float() @ lm_head[start:stop].float().T
        chunks.append(chunk)
    return torch.cat(chunks, dim=-1)


def _tensor_report(candidate, expected, *, rtol: float, atol: float) -> dict[str, Any]:
    import torch

    if tuple(candidate.shape) != tuple(expected.shape):
        return {
            "shape": tuple(candidate.shape),
            "expected_shape": tuple(expected.shape),
            "shape_match": False,
            "finite": bool(torch.isfinite(candidate).all().item()),
            "pass_rate": 0.0,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "argmax_match": False,
            "argmax_token": None,
            "expected_argmax_token": None,
        }

    diff = (candidate.float() - expected.float()).abs()
    close = torch.isclose(candidate.float(), expected.float(), rtol=rtol, atol=atol)
    return {
        "shape": tuple(candidate.shape),
        "expected_shape": tuple(expected.shape),
        "shape_match": True,
        "finite": bool(torch.isfinite(candidate).all().item()),
        "pass_rate": float(close.float().mean().item()),
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "argmax_match": int(candidate[-1].argmax().item()) == int(expected[-1].argmax().item()),
        "argmax_token": int(candidate[-1].argmax().item()),
        "expected_argmax_token": int(expected[-1].argmax().item()),
    }


def _cases_from_args(golden_root: Path, selected: list[str] | None) -> list[str]:
    if selected:
        return selected
    manifest = json.loads((golden_root / "manifest.json").read_text())
    return [case["name"] for case in manifest.get("cases", []) if case.get("dump_files")]


def generate(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    golden_root = Path(args.golden_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    final_norm, lm_head = _load_head_weights(args.ckpt_dir)
    cases = []
    for case_name in _cases_from_args(golden_root, args.case):
        case_root = output_root / case_name
        case_root.mkdir(parents=True, exist_ok=True)

        items = _rank0_per_step(_load_vllm_main_logits_items(golden_root, case_name))
        if args.max_steps is not None:
            items = items[: args.max_steps]

        steps = []
        for local_step, item in enumerate(items):
            vllm_obj = torch.load(item["file"], map_location="cpu")
            hidden = vllm_obj["hidden_states"].bfloat16()
            expected_full = vllm_obj["logits"].float()

            normed = _zero_centered_rmsnorm(hidden, final_norm).bfloat16()
            pypto_full = _logits_from_full_head(normed, lm_head, args.chunk_size)
            full_report = _tensor_report(
                pypto_full, expected_full, rtol=args.rtol, atol=args.atol,
            )

            rank_reports = []
            vocab_local = pypto_full.shape[-1] // args.tp_world_size
            for rank in range(args.tp_world_size):
                full_path = case_root / f"main_logits_step{local_step:03d}_rank{rank}.pt"
                shard_path = case_root / f"main_logits_shard_step{local_step:03d}_rank{rank}.pt"
                lo = rank * vocab_local
                hi = lo + vocab_local
                shard = pypto_full[:, lo:hi].contiguous()
                torch.save({
                    "logits": pypto_full.bfloat16(),
                    "source": "pypto_final_norm_lm_head_from_vllm_hidden",
                    "vllm_file": item["file"],
                    "rank": rank,
                    "local_step": local_step,
                    "logits_layout": "replicated_full_vocab",
                }, full_path)
                torch.save({
                    "logits": shard.bfloat16(),
                    "source": "pypto_final_norm_lm_head_from_vllm_hidden",
                    "vllm_file": item["file"],
                    "rank": rank,
                    "local_step": local_step,
                    "vocab_range": [lo, hi],
                    "logits_layout": "tp_vocab_shard",
                }, shard_path)
                rank_reports.append({
                    "rank": rank,
                    "artifact": str(full_path),
                    "shard_artifact": str(shard_path),
                    "vocab_range": [lo, hi],
                    **full_report,
                })

            steps.append({
                "local_step": local_step,
                "vllm_file": item["file"],
                "hidden_shape": tuple(hidden.shape),
                "full_logits": full_report,
                "rank_artifacts": rank_reports,
                "pass": (
                    full_report["shape_match"]
                    and full_report["finite"]
                    and full_report["pass_rate"] >= args.pass_rate
                    and full_report["argmax_match"]
                ),
            })

        cases.append({
            "name": case_name,
            "steps": steps,
            "pass": bool(steps) and all(step["pass"] for step in steps),
        })

    report = {
        "golden_root": str(golden_root),
        "output_root": str(output_root),
        "ckpt_dir": args.ckpt_dir,
        "tp_world_size": args.tp_world_size,
        "chunk_size": args.chunk_size,
        "rtol": args.rtol,
        "atol": args.atol,
        "pass_rate_threshold": args.pass_rate,
        "artifact_layout": {
            "full": "<output>/<case>/main_logits_stepNNN_rankR.pt",
            "shard": "<output>/<case>/main_logits_shard_stepNNN_rankR.pt",
        },
        "cases": cases,
        "ok": bool(cases) and all(case["pass"] for case in cases),
    }
    (output_root / "final_logits_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--case", action="append", default=None,
                        help="Golden case to process. Repeatable. Default: all dumped cases.")
    parser.add_argument("--tp-world-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--pass-rate", type=float, default=0.999)
    args = parser.parse_args()

    sys.path.insert(0, str(_repo_root()))
    report = generate(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
