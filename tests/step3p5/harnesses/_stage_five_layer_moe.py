#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint L0-L4 MoE correctness, timing, and DFX harness."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import time
from pathlib import Path

import torch

from tools.step3p5.five_layer_moe_golden_contract import (
    GOLDEN_SCHEMA,
    LEGACY_PROTOCOL_PROFILE,
    LOCAL_OWNER_PROTOCOL_PROFILE,
    canonical_hidden_only_moe_protocol_fields,
    golden_protocol_fields,
    source_protocol_binding_fields,
)

TP = 8
BATCH = 16
HIDDEN = 4096
BLOCK_SIZE = 128
IMAGE_DIGEST_PATTERN = re.compile(r".+@sha256:[0-9a-f]{64}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="8,9,10,11,12,13,14,15")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=32,
        help="per-sequence block-table capacity, not a batch-wide block budget",
    )
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
        "--recv-meta-sidecar",
        default=os.environ.get("PYPTO_RECV_META_SIDECAR", ""),
    )
    parser.add_argument(
        "--write-golden",
        default="",
        help="write hidden_l3.pt, hidden_l4.pt, and provenance to this directory",
    )
    parser.add_argument(
        "--image-digest",
        default=os.environ.get("PYPTO_IMAGE_DIGEST", ""),
        help="immutable producer image digest required by --write-golden",
    )
    parser.add_argument(
        "--source-run",
        default=os.environ.get("PYPTO_SOURCE_RUN", ""),
        help="immutable run identifier required by --write-golden",
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
        help="capture separate warm dep_gen and chip-swimlane iterations",
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


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return value


def _image_digest(value: object, *, field: str) -> str:
    if not (
        isinstance(value, str)
        and IMAGE_DIGEST_PATTERN.fullmatch(value)
    ):
        raise ValueError(f"{field} must be an immutable image digest")
    return value


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


def _workload_layout(args: argparse.Namespace) -> dict[str, int]:
    blocks_per_sequence = (
        int(args.context_len) + BLOCK_SIZE - 1
    ) // BLOCK_SIZE
    if blocks_per_sequence > int(args.num_blocks):
        raise ValueError(
            f"context={args.context_len} needs {blocks_per_sequence} blocks "
            f"per sequence, but --num-blocks={args.num_blocks}"
        )
    scheduler_num_blocks = int(args.active_batch) * blocks_per_sequence
    return {
        "blocks_per_sequence": blocks_per_sequence,
        "block_table_blocks_per_row_capacity": int(args.num_blocks),
        "scheduler_num_blocks": scheduler_num_blocks,
        "physical_num_blocks": scheduler_num_blocks + (BATCH - 1),
        "max_sequence_tokens": int(args.num_blocks) * BLOCK_SIZE,
    }


def _step_metadata(
    *,
    context_len: int,
    active_batch: int,
    blocks_per_row_capacity: int,
    scheduler_num_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build independent, compact paged contexts for every active sequence."""
    from tools.step3p5.kv_padding import (  # noqa: PLC0415
        make_padding_reserve,
        validate_fixed_batch_metadata,
    )

    blocks_per_sequence = (
        int(context_len) + BLOCK_SIZE - 1
    ) // BLOCK_SIZE
    expected_scheduler_blocks = int(active_batch) * blocks_per_sequence
    if int(scheduler_num_blocks) != expected_scheduler_blocks:
        raise ValueError(
            "scheduler block count must cover every sequence independently: "
            f"expected={expected_scheduler_blocks}, "
            f"got={scheduler_num_blocks}"
        )
    reserve = make_padding_reserve(
        scheduler_num_blocks,
        scheduler_num_blocks + (BATCH - 1),
        storage_capacity=BATCH,
    )
    seq = torch.ones(BATCH, dtype=torch.int32)
    pos = torch.zeros(BATCH, dtype=torch.int32)
    table = torch.zeros(
        BATCH,
        blocks_per_row_capacity,
        dtype=torch.int32,
    )
    slot = torch.zeros(BATCH, dtype=torch.int32)

    step = int(context_len) - 1
    for row in range(int(active_batch)):
        first_block = row * blocks_per_sequence
        block_ids = first_block + torch.arange(
            blocks_per_sequence,
            dtype=torch.int32,
        )
        table[row, :blocks_per_sequence] = block_ids
        seq[row] = int(context_len)
        pos[row] = step
        slot[row] = (
            int(block_ids[step // BLOCK_SIZE]) * BLOCK_SIZE
            + step % BLOCK_SIZE
        )
    for padding_idx, block_id in enumerate(
        reserve.padding_block_ids[: BATCH - int(active_batch)],
        start=int(active_batch),
    ):
        table[padding_idx, 0] = int(block_id)
        slot[padding_idx] = int(block_id) * BLOCK_SIZE

    validate_fixed_batch_metadata(
        seq_lens=seq,
        positions=pos,
        block_table=table,
        slot_mapping=slot,
        valid_rows=active_batch,
        reserve=reserve,
        where=(
            f"five-layer BS{active_batch} context={context_len} "
            "per-sequence"
        ),
    )
    return seq, pos, table, slot


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
    image_digest: str,
    source_run: str,
    protocol_profile: str = LEGACY_PROTOCOL_PROFILE,
) -> None:
    _image_digest(image_digest, field="golden.image_ref")
    if not source_run:
        raise ValueError("golden.source_run must be non-empty")
    workload = manifest.get("workload")
    if not isinstance(workload, dict):
        raise ValueError("golden producer workload is missing")
    active_batch = workload.get("active_batch")
    context_len = workload.get("context_len")
    if type(active_batch) is not int or not 1 <= active_batch <= BATCH:
        raise ValueError("golden active_batch is invalid")
    if context_len != 65536:
        raise ValueError("golden context_len must be 65536 per sequence")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("golden producer source manifest is missing")
    for field in (
        "decode_fwd_sha256",
        "program_sha256",
        "holder_sha256",
        "harness_sha256",
    ):
        _require_sha256(source.get(field), field=f"golden.source.{field}")
    protocol_fields = golden_protocol_fields(protocol_profile)
    source_contract = source.get("declared_moe_protocol_contract")
    if not isinstance(source_contract, dict):
        raise ValueError("golden producer source protocol is missing")
    if source_contract != protocol_fields:
        raise ValueError(
            "golden producer source protocol differs from requested protocol"
        )
    source_manifest_sha256 = _json_sha256(source)
    source_binding = source_protocol_binding_fields(
        source_manifest_sha256=source_manifest_sha256,
        decode_fwd_sha256=str(source["decode_fwd_sha256"]),
        moe_protocol_contract_sha256=str(
            source.get("moe_protocol_contract_sha256", "")
        ),
        protocol_contract=source_contract,
    )

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
    golden_manifest = {
        "schema": GOLDEN_SCHEMA,
        "source_run": source_run,
        "source_decode_fwd_sha256": source["decode_fwd_sha256"],
        "source_manifest_sha256": source_manifest_sha256,
        **source_binding,
        "active_batch": active_batch,
        "context_len_per_sequence": context_len,
        "image_ref": image_digest,
        "files": hashes,
        **protocol_fields,
    }
    (golden_dir / "manifest.json").write_text(
        json.dumps(golden_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (golden_dir / "sha256.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items()),
        encoding="utf-8",
    )


def _configure(args: argparse.Namespace) -> dict[str, int]:
    if not 1 <= args.active_batch <= BATCH:
        raise ValueError(f"--active-batch must be in [1,{BATCH}]")
    if args.context_len <= 0:
        raise ValueError("--context-len must be positive")
    if args.iters <= 0 or args.warmup < 0:
        raise ValueError("--iters must be positive and --warmup non-negative")
    if args.dfx and args.pmu:
        raise ValueError(
            "--dfx and --pmu must use separate processes/output directories"
        )
    if args.write_golden:
        _image_digest(args.image_digest, field="--image-digest")
        if not args.source_run:
            raise ValueError("--source-run or PYPTO_SOURCE_RUN is required")
        if args.context_len != 65536:
            raise ValueError("--write-golden requires --context-len=65536")
        if args.allow_tolerance:
            raise ValueError("--write-golden requires bit-exact outputs")
    _input_token_ids(args)
    layout = _workload_layout(args)

    os.environ["PYPTO_STEP3P5_MAX_SEQ"] = str(
        layout["max_sequence_tokens"]
    )
    os.environ["PYPTO_STEP3P5_BLOCK_TABLE_FLAT"] = str(
        BATCH * args.num_blocks
    )
    os.environ["PYPTO_STEP3P5_KV_NUM_LAYERS"] = "5"
    os.environ["PYPTO_STEP3P5_KV_CACHE_ROWS"] = str(
        5 * layout["physical_num_blocks"] * BLOCK_SIZE
    )
    os.environ["PYPTO_STEP3P5_ROPE_SEQ"] = str(
        layout["max_sequence_tokens"]
    )
    # The release image defines a generic /tmp build root.  Override it for
    # this harness so raw dep/swimlane/PMU artifacts survive the container and
    # remain colocated with the immutable run directory.
    os.environ["PYPTO_PROG_BUILD_DIR"] = str(Path(args.out) / "build_output")
    return layout


def _wait_for_artifacts(
    build_dir: Path,
    name: str,
    *,
    expected: int = TP,
    timeout_sec: float = 30.0,
) -> dict[str, str]:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        paths = sorted((build_dir / "dfx_outputs").rglob(name))
        if len(paths) == expected:
            return {
                str(path.relative_to(build_dir)): _sha256(path)
                for path in paths
            }
        time.sleep(0.2)
    raise RuntimeError(
        f"expected exactly {expected} {name} artifacts under {build_dir}, "
        f"found {len(paths)}"
    )


def _postprocess_dfx(
    build_dir: Path,
    out: Path,
    *,
    source_decode_sha256: str,
    recv_meta_sidecar: Path | None,
) -> None:
    from tools.step3p5.analyze_five_layer_moe_dfx import analyze

    dfx_out = out / "dfx_analysis"
    analyze(
        build_dir,
        dfx_out,
        recv_meta_sidecar=recv_meta_sidecar,
        source_decode_sha256=source_decode_sha256,
    )


def main() -> int:
    args = _parse_args()
    devices = _devices(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    layout = _configure(args)

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
    export_args = argparse.Namespace(**vars(args))
    export_args.num_blocks = layout["scheduler_num_blocks"]
    export_args.kv_num_layers = 5
    procs = (
        []
        if args.reuse_exporters
        else main_stage._start_exporters(export_args, devices)
    )
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
            seq, pos, table, slot = _step_metadata(
                context_len=args.context_len,
                active_batch=args.active_batch,
                blocks_per_row_capacity=args.num_blocks,
                scheduler_num_blocks=layout["scheduler_num_blocks"],
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
            source_protocol = canonical_hidden_only_moe_protocol_fields()
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
                    "kv_num_layers": 5,
                    "kv_rows_per_rank": (
                        5 * layout["physical_num_blocks"] * BLOCK_SIZE
                    ),
                    "context_semantics": "per_active_sequence",
                    **layout,
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
                    "moe_protocol_contract_sha256": _source_sha256(
                        repo_root,
                        "tools/step3p5/five_layer_moe_golden_contract.py",
                    ),
                    "declared_moe_protocol_profile": source_protocol[
                        "protocol_profile"
                    ],
                    "declared_moe_protocol_contract": source_protocol,
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
                    image_digest=args.image_digest,
                    source_run=args.source_run,
                    protocol_profile=LOCAL_OWNER_PROTOCOL_PROFILE,
                )

            if args.dfx:
                from pypto.runtime.runner import (  # noqa: PLC0415
                    _CHIP_SWIMLANE_RECORDS_NAME,
                )

                # Keep dep generation and swimlane capture on separate warm
                # submissions, with an unprofiled separator in between.
                for _ in range(2):
                    holder.set_live_step(active_hidden, **set_kwargs)
                    holder.run()
                holder.set_live_step(active_hidden, **set_kwargs)
                holder.run(dfx="dep")
                dep_hashes = _wait_for_artifacts(
                    Path(holder.compiled.output_dir),
                    "deps.json",
                )
                holder.set_live_step(active_hidden, **set_kwargs)
                holder.run()
                holder.set_live_step(active_hidden, **set_kwargs)
                holder.run(dfx="swim")
                dep_hashes_after_swim = _wait_for_artifacts(
                    Path(holder.compiled.output_dir),
                    "deps.json",
                )
                if dep_hashes_after_swim != dep_hashes:
                    raise RuntimeError(
                        "prepared swimlane capture changed the dep-gen "
                        "artifacts"
                    )
                swim_hashes = _wait_for_artifacts(
                    Path(holder.compiled.output_dir),
                    _CHIP_SWIMLANE_RECORDS_NAME,
                )
                (out / "dfx_protocol_report.json").write_text(
                    json.dumps(
                        {
                            "schema": (
                                "step3p5.five-layer-moe-dfx-protocol.v1"
                            ),
                            "dep_gen_artifacts": dep_hashes,
                            "dep_gen_preserved_after_swim": True,
                            "swimlane_artifacts": swim_hashes,
                            "rank_count": TP,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            if args.pmu:
                holder.set_live_step(active_hidden, **set_kwargs)
                holder.run(dfx="pmu")
    finally:
        if not args.reuse_exporters:
            main_stage._stop_exporters(out, procs)

    if args.dfx:
        _postprocess_dfx(
            Path(holder.compiled.output_dir),
            out,
            source_decode_sha256=_source_sha256(
                repo_root,
                "models/step3p5/decode_fwd.py",
            ),
            recv_meta_sidecar=(
                Path(args.recv_meta_sidecar)
                if args.recv_meta_sidecar
                else None
            ),
        )
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
