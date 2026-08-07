# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Card-free contract checks for the live precision release gate."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.step3p5.ci import live_precision_gate


_SCRIPT = Path(__file__).with_name("run_live_precision_ab.sh")


def test_live_precision_script_fails_closed() -> None:
    source = _SCRIPT.read_text(encoding="utf-8")

    assert "set -euo pipefail" in source
    assert "MIN_ALIGNMENT_PCT=${MIN_ALIGNMENT_PCT:-95}" in source
    assert 'LIB=${LIB:-$(cd "$SCRIPT_DIR/../../.." && pwd)}' in source
    assert 'export PYPTO_PROG_BUILD_DIR="$OUT/build_output"' in source
    assert 'ORACLE_PYTHON=${ORACLE_PYTHON:?' in source
    assert 'CHECKPOINT_MANIFEST=${CHECKPOINT_MANIFEST:?' in source
    assert 'read -r -a ORACLE_CMD' not in source
    assert '"$GATE" validate-config' in source
    assert '"$GATE" extract-oracle' in source
    assert '"$GATE" validate-result' in source
    assert source.count('"$GATE" verify-checkpoint') == 2
    assert '"$GATE" compare-checkpoint' in source


def test_live_precision_script_uses_the_frozen_oracle_for_stage2() -> None:
    source = _SCRIPT.read_text(encoding="utf-8")

    assert '"$GATE" render-args' in source
    assert '--oracle-json "$OUT/oracle_ids.json"' in source
    assert source.index('"$GATE" extract-oracle') < source.index(
        '"$PYPTO_PY" -m tests.step3p5.harnesses._stage_main_hidden_only'
    )


def _write_rows(
    path: Path,
    *,
    oracle: list[int],
    seed: int,
    outputs: list[int] | None = None,
) -> None:
    actual = oracle if outputs is None else outputs
    inputs = [seed, *oracle[:-1]]
    rows = [
        {
            "step": step,
            "input_token": inputs[step],
            "output_token": actual[step],
            "expected_token": oracle[step],
            "token_exact": actual[step] == oracle[step],
        }
        for step in range(len(oracle))
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_live_precision_gate_validates_frozen_lineage(tmp_path: Path) -> None:
    oracle = [11, 12, 13, 14]
    oracle_path = tmp_path / "oracle.json"
    oracle_path.write_text(json.dumps(oracle), encoding="utf-8")
    log_path = tmp_path / "pypto.log"
    _write_rows(
        log_path,
        oracle=oracle,
        seed=7,
        outputs=[11, 99, 13, 14],
    )

    matches, aligned = live_precision_gate.validate_result(
        log_path,
        oracle_path,
        expected=4,
        threshold=75.0,
        seed=7,
    )

    assert matches == 3
    assert aligned == 75.0


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -1.0, 101.0])
def test_live_precision_gate_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        live_precision_gate.validate_config(
            expected=128,
            threshold=threshold,
            seed=6127,
        )


@pytest.mark.parametrize(
    ("expected", "threshold", "message"),
    [
        (127, 95.0, "N=128"),
        (129, 95.0, "N=128"),
        (128, 94.9, "threshold"),
    ],
)
def test_release_gate_rejects_relaxed_contract(
    expected: int,
    threshold: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        live_precision_gate.validate_config(
            expected=expected,
            threshold=threshold,
            seed=6127,
            release=True,
        )


def test_live_precision_gate_rejects_boolean_oracle_id(tmp_path: Path) -> None:
    log_path = tmp_path / "oracle.log"
    log_path.write_text(
        "ORACLE_IDS_JSON=[303, true]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-negative integers"):
        live_precision_gate.extract_oracle_ids(log_path, expected=2)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("step", "contiguous"),
        ("expected", "frozen oracle"),
        ("input", "input lineage"),
        ("exact", "inconsistent"),
    ],
)
def test_live_precision_gate_rejects_row_contract_drift(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    oracle = [11, 12]
    oracle_path = tmp_path / "oracle.json"
    oracle_path.write_text(json.dumps(oracle), encoding="utf-8")
    log_path = tmp_path / "pypto.log"
    _write_rows(log_path, oracle=oracle, seed=7)
    rows = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    if mutation == "step":
        rows[1]["step"] = 2
    elif mutation == "expected":
        rows[1]["expected_token"] = 99
        rows[1]["token_exact"] = False
    elif mutation == "input":
        rows[1]["input_token"] = 99
    else:
        rows[1]["token_exact"] = False
    log_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        live_precision_gate.validate_result(
            log_path,
            oracle_path,
            expected=2,
            threshold=95.0,
            seed=7,
        )


def test_live_precision_gate_rejects_negative_token_fields(
    tmp_path: Path,
) -> None:
    oracle = [11]
    oracle_path = tmp_path / "oracle.json"
    oracle_path.write_text(json.dumps(oracle), encoding="utf-8")
    log_path = tmp_path / "pypto.log"
    _write_rows(log_path, oracle=oracle, seed=7)
    row = json.loads(log_path.read_text(encoding="utf-8"))
    row["output_token"] = -1
    row["token_exact"] = False
    log_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-negative"):
        live_precision_gate.validate_result(
            log_path,
            oracle_path,
            expected=1,
            threshold=0.0,
            seed=7,
        )


def test_checkpoint_identity_rehashes_every_shard(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = checkpoint / "config.json"
    index = checkpoint / "model.safetensors.index.json"
    shard = checkpoint / "weights.safetensors"
    config.write_text("{}", encoding="utf-8")
    index.write_text(
        json.dumps({"weight_map": {"weight": shard.name}}),
        encoding="utf-8",
    )
    shard.write_bytes(b"AAAA")
    files = {
        path.name: {
            "size_bytes": path.stat().st_size,
            "sha256": live_precision_gate._sha256(path),
        }
        for path in (config, index, shard)
    }
    manifest = tmp_path / "checkpoint_identity.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "step3p5.checkpoint-identity.v1",
                "logical_id": checkpoint.name,
                "index_file": index.name,
                "weight_tensor_count": 1,
                "weight_shard_count": 1,
                "files": files,
                "identity_sha256": live_precision_gate._json_sha256(files),
            }
        ),
        encoding="utf-8",
    )

    evidence = live_precision_gate.verify_checkpoint(checkpoint, manifest)
    assert evidence["weight_shard_count"] == 1

    shard.write_bytes(b"BBBB")
    with pytest.raises(ValueError, match="file hash mismatch"):
        live_precision_gate.verify_checkpoint(checkpoint, manifest)


def test_checkpoint_evidence_must_match_between_stages(
    tmp_path: Path,
) -> None:
    oracle = tmp_path / "oracle.json"
    pypto = tmp_path / "pypto.json"
    oracle.write_text(
        json.dumps({"identity_sha256": "a" * 64}),
        encoding="utf-8",
    )
    pypto.write_text(
        json.dumps({"identity_sha256": "b" * 64}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="identities differ"):
        live_precision_gate.compare_checkpoint_evidence(oracle, pypto)
