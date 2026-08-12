# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SWA input-RMSNorm logical task-grain and aligned-reduction contracts."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _ROOT / "models" / "step3p5" / "config.py"
_SWA_PATH = _ROOT / "models" / "step3p5" / "attention_swa.py"
_CONFIG = _CONFIG_PATH.read_text(encoding="utf-8")
_SWA = _SWA_PATH.read_text(encoding="utf-8")
_RMS_SCOPE = _SWA.split(
    "# ----- Scope 1.a — zero-centred input RMSNorm. -----",
    maxsplit=1,
)[1].split(
    "# ----- Scope 1.f — on-device head-gate (RESTORED, path (a)). -----",
    maxsplit=1,
)[0]
_ENV_KEY = "PYPTO_STEP3P5_SWA_RMSNORM_ROWS_PER_TASK"


def _grain_config_nodes() -> tuple[ast.Assign, list[ast.If]]:
    tree = ast.parse(_CONFIG)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "SWA_RMSNORM_ROWS_PER_TASK"
            for target in node.targets
        )
    )
    validations = [
        node
        for node in tree.body[tree.body.index(assignment) + 1 :]
        if isinstance(node, ast.If)
        and "SWA_RMSNORM_ROWS_PER_TASK" in ast.unparse(node.test)
    ]
    assert len(validations) == 2
    return assignment, validations


def _evaluate_grain(value: str | None) -> int:
    assignment, validations = _grain_config_nodes()
    env = {} if value is None else {_ENV_KEY: value}
    namespace = {
        "BATCH": 16,
        "os": SimpleNamespace(environ=env),
    }
    module = ast.fix_missing_locations(
        ast.Module(body=[assignment, *validations], type_ignores=[]),
    )
    exec(compile(module, str(_CONFIG_PATH), "exec"), namespace)
    return int(namespace["SWA_RMSNORM_ROWS_PER_TASK"])


def test_swa_rmsnorm_grain_default_is_calibrated_to_eight_tasks() -> None:
    assert _evaluate_grain(None) == 2
    assert 16 // _evaluate_grain(None) == 8
    assert _ENV_KEY in _CONFIG
    assert '"SWA_RMSNORM_ROWS_PER_TASK"' in _CONFIG


def test_swa_rmsnorm_grain_accepts_the_calibrated_value() -> None:
    assert _evaluate_grain("2") == 2


@pytest.mark.parametrize("value", ["-1", "0", "3", "5", "17"])
def test_swa_rmsnorm_grain_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="divide BATCH exactly"):
        _evaluate_grain(value)


@pytest.mark.parametrize("value", ["1", "4", "8", "16"])
def test_swa_rmsnorm_grain_rejects_uncalibrated_divisors(value: str) -> None:
    with pytest.raises(ValueError, match="currently supports only"):
        _evaluate_grain(value)


def test_swa_rmsnorm_uses_workload_derived_logical_tasks() -> None:
    tree = ast.parse(_SWA)
    matches: list[ast.Call] = []
    for call in ast.walk(tree):
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "pl"
            and call.func.attr == "spmd"
        ):
            continue
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        name_hint = keywords.get("name_hint")
        if isinstance(name_hint, ast.Constant) and name_hint.value == "swa_rmsnorm_zc":
            matches.append(call)

    assert len(matches) == 1
    task_count = matches[0].args[0]
    assert isinstance(task_count, ast.BinOp)
    assert isinstance(task_count.op, ast.FloorDiv)
    assert isinstance(task_count.left, ast.Name)
    assert task_count.left.id == "BATCH"
    assert isinstance(task_count.right, ast.Name)
    assert task_count.right.id == "SWA_RMSNORM_ROWS_PER_TASK"
    assert all(keyword.arg != "optimizations" for keyword in matches[0].keywords)
    early_resolve = next(
        keyword.value
        for keyword in matches[0].keywords
        if keyword.arg == "allow_early_resolve"
    )
    assert isinstance(early_resolve, ast.Constant)
    assert early_resolve.value is True
    assert "rms_b0 = rms_spmd_idx * SWA_RMSNORM_ROWS_PER_TASK" in _RMS_SCOPE


def test_swa_rmsnorm_two_rows_are_packed_into_aligned_lanes() -> None:
    assert "[swa_rmsnorm_parts_per_row, 8]" in _RMS_SCOPE
    assert _RMS_SCOPE.count("pl.tensor.write(") == 32
    assert "partial_pairs, [0, 0], pl.tensor.read(row_partials, [0, 0])" in _RMS_SCOPE
    assert "partial_pairs, [0, 1], pl.tensor.read(row_partials, [1, 0])" in _RMS_SCOPE
    assert "pl.slice(partial_pairs, [1, 8]" in _RMS_SCOPE

def test_swa_rmsnorm_reuses_full_row_load_and_preserves_chunk_order() -> None:
    assert _RMS_SCOPE.count("norm_chunk = pl.cast(") == 1
    assert "[SWA_RMSNORM_ROWS_PER_TASK, swa_rmsnorm_norm_k_chunk]" in _RMS_SCOPE
    assert "valid_shape=[\n                    SWA_RMSNORM_ROWS_PER_TASK," in _RMS_SCOPE
    assert "[swa_rmsnorm_reduction_rows, swa_rmsnorm_reduce_chunk]" in _RMS_SCOPE
    assert "pl.row_sum(sq_rows)" in _RMS_SCOPE
    assert _RMS_SCOPE.count("partial_sq = pl.add(") == 15
    assert "pl.tensor.read(inv_rms, [0, 0])" in _RMS_SCOPE
    assert "pl.tensor.read(inv_rms, [0, 1])" in _RMS_SCOPE
    assert "[rms_b0 + 1, 0]" in _RMS_SCOPE
