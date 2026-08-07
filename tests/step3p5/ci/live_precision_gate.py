#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed parsing for the live N-token precision release gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


_ORACLE_PREFIX = "ORACLE_IDS_JSON="
_CHECKPOINT_SCHEMA = "step3p5.checkpoint-identity.v1"


def validate_config(
    *,
    expected: int,
    threshold: float,
    seed: int,
    release: bool = False,
) -> None:
    """Validate release-gate scalars before launching either model."""
    if type(expected) is not int or expected <= 0:
        raise ValueError("expected token count must be a positive integer")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 100.0:
        raise ValueError("alignment threshold must be finite and within [0, 100]")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed token must be a non-negative integer")
    if release and expected != 128:
        raise ValueError("release precision gate requires exactly N=128")
    if release and threshold < 95.0:
        raise ValueError("release precision gate requires threshold >= 95")


def validate_oracle_ids(value: Any, *, expected: int) -> list[int]:
    """Return exact native token IDs or reject a malformed oracle."""
    if not isinstance(value, list) or len(value) != expected:
        count: object = len(value) if isinstance(value, list) else "invalid"
        raise ValueError(
            f"oracle token count={count}, expected={expected}"
        )
    if any(type(token) is not int or token < 0 for token in value):
        raise ValueError("oracle ids must be non-negative integers")
    return value


def extract_oracle_ids(log_path: Path, *, expected: int) -> list[int]:
    """Extract the unique oracle marker from a generator log."""
    markers = [
        line.strip()[len(_ORACLE_PREFIX):]
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(_ORACLE_PREFIX)
    ]
    if len(markers) != 1:
        raise ValueError(
            "oracle log must contain exactly one ORACLE_IDS_JSON marker"
        )
    try:
        value = json.loads(markers[0])
    except json.JSONDecodeError as exc:
        raise ValueError("ORACLE_IDS_JSON is not valid JSON") from exc
    return validate_oracle_ids(value, expected=expected)


def load_oracle_ids(path: Path, *, expected: int) -> list[int]:
    """Load and revalidate a frozen oracle JSON entity."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("oracle JSON entity is invalid") from exc
    return validate_oracle_ids(value, expected=expected)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_checkpoint_identity(path: Path) -> dict[str, Any]:
    """Load a complete checkpoint identity manifest."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("checkpoint manifest is invalid JSON") from exc
    if not isinstance(value, dict) or value.get("schema") != _CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint manifest schema mismatch")
    logical_id = value.get("logical_id")
    index_file = value.get("index_file")
    if not isinstance(logical_id, str) or not logical_id:
        raise ValueError("checkpoint logical identity is missing")
    if (
        not isinstance(index_file, str)
        or not index_file.endswith(".safetensors.index.json")
    ):
        raise ValueError("checkpoint index identity is invalid")
    files = value.get("files")
    if (
        not isinstance(files, dict)
        or "config.json" not in files
        or index_file not in files
    ):
        raise ValueError("checkpoint file identity is incomplete")
    for name, record in files.items():
        if (
            not isinstance(name, str)
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or not isinstance(record, dict)
            or type(record.get("size_bytes")) is not int
            or record["size_bytes"] <= 0
            or not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in record["sha256"])
        ):
            raise ValueError(f"checkpoint file identity is invalid: {name}")
    if value.get("identity_sha256") != _json_sha256(files):
        raise ValueError("checkpoint identity digest mismatch")
    shard_count = value.get("weight_shard_count")
    tensor_count = value.get("weight_tensor_count")
    if (
        type(shard_count) is not int
        or shard_count <= 0
        or len(files) != shard_count + 2
        or type(tensor_count) is not int
        or tensor_count <= 0
    ):
        raise ValueError("checkpoint shard/tensor count is invalid")
    return value


def verify_checkpoint(
    checkpoint: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Re-hash every checkpoint entity and return immutable evidence."""
    identity = load_checkpoint_identity(manifest_path)
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_dir():
        raise ValueError(f"checkpoint directory is missing: {checkpoint}")
    if checkpoint.name != identity["logical_id"]:
        raise ValueError("checkpoint logical identity mismatch")
    files = identity["files"]
    for name, record in files.items():
        entity = checkpoint / name
        if not entity.is_file():
            raise ValueError(f"checkpoint file is missing: {name}")
        if entity.stat().st_size != record["size_bytes"]:
            raise ValueError(f"checkpoint file size mismatch: {name}")
        if _sha256(entity) != record["sha256"]:
            raise ValueError(f"checkpoint file hash mismatch: {name}")
    return {
        "schema": _CHECKPOINT_SCHEMA,
        "logical_id": identity["logical_id"],
        "identity_sha256": identity["identity_sha256"],
        "manifest_sha256": _sha256(manifest_path),
        "weight_shard_count": identity["weight_shard_count"],
        "weight_tensor_count": identity["weight_tensor_count"],
    }


def compare_checkpoint_evidence(
    oracle_path: Path,
    pypto_path: Path,
) -> dict[str, Any]:
    """Require both precision stages to report one checkpoint identity."""
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    pypto = json.loads(pypto_path.read_text(encoding="utf-8"))
    if not isinstance(oracle, dict) or not isinstance(pypto, dict):
        raise ValueError("checkpoint evidence must be JSON objects")
    if oracle != pypto:
        raise ValueError("oracle/PyPTO checkpoint identities differ")
    return oracle


def load_step_rows(log_path: Path) -> list[dict[str, Any]]:
    """Load compact per-step JSON rows and ignore the final pretty report."""
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        log_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        text = line.strip()
        if not (
            text.startswith("{")
            and '"output_token"' in text
            and '"step"' in text
        ):
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid teacher-forced JSON row at line {line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(
                f"teacher-forced row at line {line_number} is not an object"
            )
        rows.append(value)
    return rows


def validate_result(
    log_path: Path,
    oracle_path: Path,
    *,
    expected: int,
    threshold: float,
    seed: int,
    release: bool = False,
) -> tuple[int, float]:
    """Validate row lineage and return the exact-match count and percentage."""
    validate_config(
        expected=expected,
        threshold=threshold,
        seed=seed,
        release=release,
    )
    oracle = load_oracle_ids(oracle_path, expected=expected)
    rows = load_step_rows(log_path)
    if len(rows) != expected:
        raise ValueError(
            f"teacher-forced rows={len(rows)}, expected={expected}"
        )

    scalar_fields = ("step", "input_token", "output_token", "expected_token")
    for index, row in enumerate(rows):
        for field in scalar_fields:
            if type(row.get(field)) is not int:
                raise ValueError(
                    f"step row {index} has invalid integer field {field}"
                )
            if row[field] < 0:
                raise ValueError(
                    f"step row {index} field {field} must be non-negative"
                )
        if type(row.get("token_exact")) is not bool:
            raise ValueError(f"step row {index} has invalid token_exact")

    steps = [row["step"] for row in rows]
    if steps != list(range(expected)):
        raise ValueError(
            f"teacher-forced steps={steps}, expected contiguous 0..{expected - 1}"
        )
    expected_tokens = [row["expected_token"] for row in rows]
    if expected_tokens != oracle:
        raise ValueError("teacher-forced expected tokens differ from frozen oracle")
    input_tokens = [row["input_token"] for row in rows]
    if input_tokens != [seed, *oracle[:-1]]:
        raise ValueError("teacher-forced input lineage differs from frozen oracle")

    matches = 0
    for index, row in enumerate(rows):
        exact = row["output_token"] == row["expected_token"]
        if row["token_exact"] is not exact:
            raise ValueError(f"step row {index} token_exact is inconsistent")
        matches += int(exact)
    aligned = 100.0 * matches / expected
    if aligned < threshold:
        raise ValueError(
            "live alignment %.3f%% is below %.3f%%"
            % (aligned, threshold)
        )
    return matches, aligned


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    config = subparsers.add_parser("validate-config")
    config.add_argument("--expected", type=int, required=True)
    config.add_argument("--threshold", type=float, required=True)
    config.add_argument("--seed", type=int, required=True)
    config.add_argument("--release", action="store_true")

    extract = subparsers.add_parser("extract-oracle")
    extract.add_argument("--log", type=Path, required=True)
    extract.add_argument("--expected", type=int, required=True)
    extract.add_argument("--out", type=Path, required=True)

    render = subparsers.add_parser("render-args")
    render.add_argument("--oracle-json", type=Path, required=True)
    render.add_argument("--expected", type=int, required=True)

    result = subparsers.add_parser("validate-result")
    result.add_argument("--log", type=Path, required=True)
    result.add_argument("--oracle-json", type=Path, required=True)
    result.add_argument("--expected", type=int, required=True)
    result.add_argument("--threshold", type=float, required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--release", action="store_true")

    checkpoint = subparsers.add_parser("verify-checkpoint")
    checkpoint.add_argument("--checkpoint", type=Path, required=True)
    checkpoint.add_argument("--manifest", type=Path, required=True)
    checkpoint.add_argument("--out", type=Path, required=True)

    compare = subparsers.add_parser("compare-checkpoint")
    compare.add_argument("--oracle", type=Path, required=True)
    compare.add_argument("--pypto", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.command == "validate-config":
            validate_config(
                expected=args.expected,
                threshold=args.threshold,
                seed=args.seed,
                release=args.release,
            )
        elif args.command == "extract-oracle":
            ids = extract_oracle_ids(args.log, expected=args.expected)
            args.out.write_text(
                json.dumps(ids, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
        elif args.command == "render-args":
            ids = load_oracle_ids(args.oracle_json, expected=args.expected)
            print(" ".join(f"--oracle-token {token}" for token in ids))
        elif args.command == "validate-result":
            matches, aligned = validate_result(
                args.log,
                args.oracle_json,
                expected=args.expected,
                threshold=args.threshold,
                seed=args.seed,
                release=args.release,
            )
            marker = (
                "LIVE_AB_ALIGNED"
                if args.release
                else "LIVE_AB_DIAGNOSTIC_ALIGNED"
            )
            print(
                "%s=%d/%d (%.1f%%), threshold=%.1f%%"
                % (marker, matches, args.expected, aligned, args.threshold)
            )
        elif args.command == "verify-checkpoint":
            evidence = verify_checkpoint(args.checkpoint, args.manifest)
            args.out.write_text(
                json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            evidence = compare_checkpoint_evidence(
                args.oracle,
                args.pypto,
            )
            print(
                "CHECKPOINT_IDENTITY=PASS "
                f"{evidence.get('identity_sha256')}"
            )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"FAIL: {exc}") from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
