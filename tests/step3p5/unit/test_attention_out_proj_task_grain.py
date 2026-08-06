# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Decode attention out-projection task-grain release contracts."""
from __future__ import annotations

import ast
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[3]
_CONFIG = _ROOT / "models" / "step3p5" / "config.py"
_FULL = _ROOT / "models" / "step3p5" / "attention_full.py"
_SWA = _ROOT / "models" / "step3p5" / "attention_swa.py"

_ENV_KEYS = (
    "PYPTO_STEP3P5_FULL_ATTN_OUT_PROJ_N_CHUNK",
    "PYPTO_STEP3P5_SWA_OUT_PROJ_N_CHUNK",
    "PYPTO_STEP3P5_FULL_ATTN_OUT_PROJ_MATMUL_N_CHUNK",
    "PYPTO_STEP3P5_FULL_ATTN_OUT_PROJ_VEC_N_CHUNK",
    "PYPTO_STEP3P5_FULL_ATTN_OUT_PROJ_MATMUL_TILES_PER_TASK",
    "PYPTO_STEP3P5_FULL_ATTN_OUT_PROJ_FUSE_CAST",
    "PYPTO_STEP3P5_SWA_OUT_PROJ_MATMUL_N_CHUNK",
    "PYPTO_STEP3P5_SWA_OUT_PROJ_VEC_N_CHUNK",
    "PYPTO_STEP3P5_SWA_OUT_PROJ_MATMUL_TILES_PER_TASK",
    "PYPTO_STEP3P5_SWA_OUT_PROJ_FUSE_CAST",
)


def test_out_proj_release_defaults_are_calibrated_independently() -> None:
    tree = ast.parse(_CONFIG.read_text(encoding="utf-8"))
    defaults: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not (
            isinstance(call.func, ast.Name)
            and call.func.id == "int"
            and call.args
            and isinstance(call.args[0], ast.Call)
        ):
            continue
        get_call = call.args[0]
        if len(get_call.args) < 2:
            continue
        env_key, default = get_call.args[:2]
        if (
            isinstance(env_key, ast.Constant)
            and isinstance(env_key.value, str)
            and isinstance(default, ast.Constant)
            and isinstance(default.value, str)
        ):
            defaults[target.id] = default.value

    expected = {
        "FULL_ATTN_OUT_PROJ_N_CHUNK": "64",
        "SWA_OUT_PROJ_N_CHUNK": "64",
        "FULL_ATTN_OUT_PROJ_VEC_N_CHUNK": "128",
        "SWA_OUT_PROJ_VEC_N_CHUNK": "128",
        "FULL_ATTN_OUT_PROJ_MATMUL_TILES_PER_TASK": "3",
        "SWA_OUT_PROJ_MATMUL_TILES_PER_TASK": "3",
    }
    assert {key: defaults[key] for key in expected} == expected


def test_full_and_swa_out_proj_use_logical_task_grouping() -> None:
    full = _FULL.read_text(encoding="utf-8")
    swa = _SWA.read_text(encoding="utf-8")

    assert "full_out_proj_tasks = (" in full
    assert "for out_task in pl.spmd(" in full
    assert "FULL_ATTN_OUT_PROJ_MATMUL_TILES_PER_TASK" in full
    assert "if out_tile < full_out_proj_n_tiles:" in full
    assert "HIDDEN // FULL_ATTN_OUT_PROJ_VEC_N_CHUNK" in full

    assert "swa_out_proj_tasks = (" in swa
    assert "for out_task in pl.spmd(" in swa
    assert "SWA_OUT_PROJ_MATMUL_TILES_PER_TASK" in swa
    assert "if out_tile < swa_out_proj_n_tiles:" in swa
    assert "HIDDEN // SWA_OUT_PROJ_VEC_N_CHUNK" in swa


def test_full_fused_out_proj_does_not_split_batch_rows() -> None:
    tree = ast.parse(_FULL.read_text(encoding="utf-8"))
    matches = []
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
        if (
            isinstance(name_hint, ast.Constant)
            and name_hint.value == "full_out_proj_matmul"
        ):
            matches.append(keywords)

    assert len(matches) == 1
    assert "optimizations" not in matches[0]


def test_out_proj_grain_knobs_remain_runtime_overridable() -> None:
    config = _CONFIG.read_text(encoding="utf-8")
    for key in _ENV_KEYS:
        assert key in config


def test_out_proj_cast_fusion_is_explicit_and_enabled_by_default() -> None:
    config = _CONFIG.read_text(encoding="utf-8")
    assert (
        '"PYPTO_STEP3P5_FULL_ATTN_OUT_PROJ_FUSE_CAST",\n'
        '        "1",'
    ) in config
    assert (
        '"PYPTO_STEP3P5_SWA_OUT_PROJ_FUSE_CAST",\n'
        '        "1",'
    ) in config

    full = _FULL.read_text(encoding="utf-8")
    swa = _SWA.read_text(encoding="utf-8")
    assert "if FULL_ATTN_OUT_PROJ_FUSE_CAST != 0:" in full
    assert "if FULL_ATTN_OUT_PROJ_FUSE_CAST == 0:" in full
    assert "if SWA_OUT_PROJ_FUSE_CAST != 0:" in swa
    assert "if SWA_OUT_PROJ_FUSE_CAST == 0:" in swa
    assert "pl.cast(o_acc, target_type=pl.BF16)" in full
    assert "pl.cast(o_acc, target_type=pl.BF16)" in swa
    assert (
        full.index(
            "partial_attn_proj = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)",
        )
        < full.index("if FULL_ATTN_OUT_PROJ_FUSE_CAST != 0:")
    )
    assert (
        full.index(
            "partial_attn_proj_fp32 = pl.create_tensor("
            "[BATCH, HIDDEN], dtype=pl.FP32)",
        )
        < full.index("if FULL_ATTN_OUT_PROJ_FUSE_CAST != 0:")
    )
    assert (
        swa.index(
            "partial_attn_proj = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)",
        )
        < swa.index("if SWA_OUT_PROJ_FUSE_CAST != 0:")
    )
    assert (
        swa.index(
            "partial_attn_proj_fp32 = pl.create_tensor("
            "[BATCH, HIDDEN], dtype=pl.FP32)",
        )
        < swa.index("if SWA_OUT_PROJ_FUSE_CAST != 0:")
    )
