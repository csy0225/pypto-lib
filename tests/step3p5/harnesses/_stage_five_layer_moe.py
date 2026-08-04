#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint L0-L4 MoE correctness, timing, and DFX harness."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch


TP = 8
BATCH = 16
HIDDEN = 4096
BLOCK_SIZE = 128
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="8,9,10,11,12,13,14,15")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-blocks", type=int, default=32)
    parser.add_argument("--context-len", type=int, default=1)
    parser.add_argument("--active-batch", type=int, default=BATCH)
    parser.add_argument("--seed-token", type=int, default=6127)
    parser.add_argument(
        "--input-tokens",
        default="",
        help=(
            "comma-separated token ids; when set, provide exactly "
            "--active-batch ids instead of repeating --seed-token"
        ),
    )
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    parser.add_argument("--reuse-exporters", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--write-golden",
        default="",
        help="write hidden_l3.pt, hidden_l4.pt, and provenance to this directory",
    )
    parser.add_argument(
        "--golden-dir",
        default="",
        help="compare both outputs against an existing frozen golden directory",
    )
    parser.add_argument(
        "--allow-tolerance",
        action="store_true",
        help="report tolerance metrics instead of requiring bit-exact equality",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="absolute threshold used with --allow-tolerance",
    )
    parser.add_argument(
        "--dfx",
        action="store_true",
        help="capture separate warm dep_gen and l2_swimlane iterations",
    )
    parser.add_argument("--pmu", action="store_true")
    return parser.parse_args()


def _devices(text: str) -> list[int]:
    devices = [int(item) for item in text.split(",") if item.strip()]
    if len(devices) != TP or len(set(devices)) != TP:
        raise ValueError(f"expected {TP} distinct devices, got {devices}")
    return devices


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha256(root: Path, relative: str) -> str:
    path = root / relative
    return _sha256(path) if path.exists() else ""


def _input_token_ids(args: argparse.Namespace) -> list[int]:
    if not args.input_tokens:
        return [args.seed_token] * args.active_batch
    tokens = [
        int(item.strip())
        for item in args.input_tokens.split(",")
        if item.strip()
    ]
    if len(tokens) != args.active_batch:
        raise ValueError(
            "--input-tokens must contain exactly active_batch ids: "
            f"active_batch={args.active_batch}, tokens={tokens}"
        )
    if any(token < 0 for token in tokens):
        raise ValueError(f"token ids must be non-negative, got {tokens}")
    return tokens


def _tensor_health(tensor: torch.Tensor, active_batch: int) -> dict[str, object]:
    active = tensor[:, :active_batch].float()
    row_abs_max = active.abs().amax(dim=-1)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": bool(torch.isfinite(active).all()),
        "nonzero_rank_rows": int(torch.count_nonzero(row_abs_max > 0).item()),
        "expected_nonzero_rank_rows": TP * active_batch,
        "tp_spread_max": float(
            (active - active[0:1]).abs().amax().item()
        ),
        "abs_max": float(active.abs().amax().item()),
        "abs_mean": float(active.abs().mean().item()),
    }


def _comparison(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
) -> dict[str, object]:
    if tuple(actual.shape) != tuple(expected.shape):
        raise ValueError(
            f"shape mismatch: actual={tuple(actual.shape)}, "
            f"expected={tuple(expected.shape)}"
        )
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    denom = torch.linalg.vector_norm(actual_f.flatten()) * torch.linalg.vector_norm(
        expected_f.flatten()
    )
    cosine = (
        float(
            torch.dot(actual_f.flatten(), expected_f.flatten()).item()
            / denom.item()
        )
        if float(denom.item()) != 0.0
        else 0.0
    )
    return {
        "exact": bool(torch.equal(actual, expected)),
        "max_abs": float(diff.amax().item()),
        "mean_abs": float(diff.mean().item()),
        "bad_ratio": float((diff > atol).float().mean().item()),
        "cosine": cosine,
        "atol": float(atol),
    }


def _write_golden(
    golden_dir: Path,
    *,
    hidden_l3: torch.Tensor,
    hidden_l4: torch.Tensor,
    manifest: dict[str, object],
) -> None:
    if golden_dir.exists() and any(golden_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite golden directory {golden_dir}")
    golden_dir.mkdir(parents=True, exist_ok=True)
    l3_path = golden_dir / "hidden_l3.pt"
    l4_path = golden_dir / "hidden_l4.pt"
    torch.save(hidden_l3, l3_path)
    torch.save(hidden_l4, l4_path)
    hashes = {
        "hidden_l3.pt": _sha256(l3_path),
        "hidden_l4.pt": _sha256(l4_path),
    }
    manifest = dict(manifest)
    manifest["files"] = hashes
    (golden_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (golden_dir / "sha256.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items()),
        encoding="utf-8",
    )


def _configure(args: argparse.Namespace) -> None:
    if not 1 <= args.active_batch <= BATCH:
        raise ValueError(f"--active-batch must be in [1,{BATCH}]")
    if args.context_len <= 0:
        raise ValueError("--context-len must be positive")
    blocks_per_row = (args.context_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    if args.active_batch * blocks_per_row > args.num_blocks:
        raise ValueError(
            f"active_batch={args.active_batch}, context={args.context_len} "
            f"needs at least {args.active_batch * blocks_per_row} blocks, "
            f"got {args.num_blocks}"
        )
    if args.iters <= 0 or args.warmup < 0:
        raise ValueError("--iters must be positive and --warmup non-negative")
    _input_token_ids(args)

    physical_blocks = args.num_blocks + (BATCH - 1)
    os.environ["PYPTO_STEP3P5_MAX_SEQ"] = str(args.num_blocks * BLOCK_SIZE)
    os.environ["PYPTO_STEP3P5_BLOCK_TABLE_FLAT"] = str(
        BATCH * args.num_blocks
    )
    os.environ["PYPTO_STEP3P5_KV_CACHE_ROWS"] = str(
        45 * physical_blocks * BLOCK_SIZE
    )
    os.environ["PYPTO_STEP3P5_ROPE_SEQ"] = str(args.num_blocks * BLOCK_SIZE)
    # The release image defines a generic /tmp build root.  Override it for
    # this harness so raw dep/swimlane/PMU artifacts survive the container and
    # remain colocated with the immutable run directory.
    os.environ["PYPTO_PROG_BUILD_DIR"] = str(Path(args.out) / "build_output")


def _postprocess_dfx(build_dir: Path, out: Path) -> None:
    from tools.step3p5.analyze_five_layer_moe_dfx import analyze

    dfx_out = out / "dfx_analysis"
    analyze(build_dir, dfx_out)


def main() -> int:
    args = _parse_args()
    devices = _devices(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _configure(args)

    from tools.step3p5.five_layer_moe_holder import FiveLayerMoeHolder

    if args.compile_only:
        holder = FiveLayerMoeHolder(
            devices,
            str(out),
            args.ckpt,
            platform=args.platform,
            kv_ipc=False,
        ).build()
        (out / "compile_report.json").write_text(
            json.dumps(
                {
                    "ok": True,
                    "program": holder.program_name,
                    "output_dir": str(holder.compiled.output_dir),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0

    # Reuse the canonical exporters so the focused path sees the exact same
    # real-checkpoint IPC tensors and full 45-layer KV pool as product Main.
    from tests.step3p5.harnesses import _stage_main_hidden_only as main_stage

    args.kv_probe = False
    procs = [] if args.reuse_exporters else main_stage._start_exporters(args, devices)
    if args.reuse_exporters and not all(
        main_stage._ready(out, rank) for rank in range(TP)
    ):
        raise RuntimeError("reuse-exporters requested but IPC maps are incomplete")

    repo_root = Path(__file__).resolve().parents[3]
    holder = FiveLayerMoeHolder(
        devices,
        str(out),
        args.ckpt,
        platform=args.platform,
        kv_ipc=True,
    ).build()
    try:
        with holder:
            input_tokens = _input_token_ids(args)
            active_hidden = torch.stack(
                [
                    main_stage._load_embedding_row(args.ckpt, token)
                    for token in input_tokens
                ],
                dim=0,
            )
            seq, pos, table, slot = main_stage._step_metadata(
                step=args.context_len - 1,
                scheduler_num_blocks=args.num_blocks,
                valid_rows=args.active_batch,
            )
            active_hidden = active_hidden.contiguous()
            set_kwargs = {
                "seq_lens": seq,
                "positions": pos,
                "block_table": table,
                "slot_mapping": slot,
            }

            for _ in range(args.warmup):
                holder.set_live_step(active_hidden, **set_kwargs)
                holder.run()

            samples_ms: list[float] = []
            last_result = None
            for _ in range(args.iters):
                holder.set_live_step(active_hidden, **set_kwargs)
                started = time.time()
                last_result = holder.run()
                samples_ms.append((time.time() - started) * 1000.0)
            if last_result is None:
                raise RuntimeError("no measured focused iteration ran")

            hidden_l3 = (
                last_result["hidden_l3"][:, : args.active_batch]
                .to(torch.bfloat16)
                .clone()
                .cpu()
            )
            hidden_l4 = (
                last_result["hidden_l4"][:, : args.active_batch]
                .to(torch.bfloat16)
                .clone()
                .cpu()
            )
            torch.save(hidden_l3, out / "hidden_l3.pt")
            torch.save(hidden_l4, out / "hidden_l4.pt")

            health = {
                "hidden_l3": _tensor_health(hidden_l3, args.active_batch),
                "hidden_l4": _tensor_health(hidden_l4, args.active_batch),
            }
            for name, item in health.items():
                if not item["finite"]:
                    raise AssertionError(f"{name} contains non-finite values")
                if (
                    item["nonzero_rank_rows"]
                    != item["expected_nonzero_rank_rows"]
                ):
                    raise AssertionError(f"{name} has zero active rank/rows: {item}")
                if item["tp_spread_max"] != 0.0:
                    raise AssertionError(f"{name} TP spread is non-zero: {item}")

            comparisons: dict[str, object] = {}
            if args.golden_dir:
                golden_dir = Path(args.golden_dir)
                comparisons["hidden_l3"] = _comparison(
                    hidden_l3,
                    torch.load(
                        golden_dir / "hidden_l3.pt",
                        map_location="cpu",
                    ),
                    atol=args.atol,
                )
                comparisons["hidden_l4"] = _comparison(
                    hidden_l4,
                    torch.load(
                        golden_dir / "hidden_l4.pt",
                        map_location="cpu",
                    ),
                    atol=args.atol,
                )
                if not args.allow_tolerance and not all(
                    item["exact"] for item in comparisons.values()
                ):
                    raise AssertionError(
                        f"focused hidden outputs are not bit-exact: {comparisons}"
                    )
                if args.allow_tolerance and any(
                    item["bad_ratio"] != 0.0 for item in comparisons.values()
                ):
                    raise AssertionError(
                        f"focused hidden outputs exceed tolerance: {comparisons}"
                    )

            ordered = sorted(samples_ms)
            timing = {
                "iters": len(ordered),
                "warmup": args.warmup,
                "min_ms": ordered[0],
                "mean_ms": statistics.fmean(ordered),
                "p50_ms": ordered[len(ordered) // 2],
                "p99_ms": ordered[
                    min(len(ordered) - 1, int(len(ordered) * 0.99))
                ],
                "max_ms": ordered[-1],
            }
            manifest = {
                "schema": "step3p5.five-layer-moe.v1",
                "program": "FiveLayerMoe",
                "layers": [
                    "L0_full_dense",
                    "L1_swa_dense",
                    "L2_swa_dense",
                    "L3_swa_moe",
                    "L4_full_moe",
                ],
                "outputs": ["hidden_l3", "hidden_l4"],
                "devices": devices,
                "checkpoint": args.ckpt,
                "workload": {
                    "seed_token": args.seed_token,
                    "input_tokens": input_tokens,
                    "input_kind": (
                        "heterogeneous"
                        if len(set(input_tokens)) > 1
                        else "repeated"
                    ),
                    "active_batch": args.active_batch,
                    "context_len": args.context_len,
                    "num_blocks": args.num_blocks,
                },
                "source": {
                    "decode_fwd_sha256": _source_sha256(
                        repo_root,
                        "models/step3p5/decode_fwd.py",
                    ),
                    "program_sha256": _source_sha256(
                        repo_root,
                        "tests/step3p5/harnesses/_five_layer_moe_program.py",
                    ),
                    "holder_sha256": _source_sha256(
                        repo_root,
                        "tools/step3p5/five_layer_moe_holder.py",
                    ),
                    "harness_sha256": _source_sha256(
                        repo_root,
                        "tests/step3p5/harnesses/_stage_five_layer_moe.py",
                    ),
                },
                "build_output": str(holder.compiled.output_dir),
                "timing": timing,
                "health": health,
                "comparisons": comparisons,
            }
            (out / "five_layer_moe_report.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            if args.write_golden:
                _write_golden(
                    Path(args.write_golden),
                    hidden_l3=hidden_l3,
                    hidden_l4=hidden_l4,
                    manifest=manifest,
                )

            if args.dfx:
                # Keep dep generation and swimlane capture on separate warm
                # submissions, with an unprofiled separator in between.
                for _ in range(2):
                    holder.set_live_step(active_hidden, **set_kwargs)
                    holder.run()
                holder.set_live_step(active_hidden, **set_kwargs)
                holder.run(dfx="dep")
                holder.set_live_step(active_hidden, **set_kwargs)
                holder.run()
                holder.set_live_step(active_hidden, **set_kwargs)
                holder.run(dfx="swim")
            if args.pmu:
                holder.set_live_step(active_hidden, **set_kwargs)
                holder.run(dfx="pmu")
    finally:
        if not args.reuse_exporters:
            main_stage._stop_exporters(out, procs)

    if args.dfx:
        _postprocess_dfx(Path(holder.compiled.output_dir), out)
    print(
        json.dumps(
            {
                "ok": True,
                "report": str(out / "five_layer_moe_report.json"),
                "hidden_l3": str(out / "hidden_l3.pt"),
                "hidden_l4": str(out / "hidden_l4.pt"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    print("[worker] RUN done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
