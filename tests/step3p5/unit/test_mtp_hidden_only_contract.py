#!/usr/bin/env python3
"""Card-free source/AST gates for the selected MTP hidden-only program."""
from __future__ import annotations

import ast
from pathlib import Path


_SOURCE_PATH = (
    Path(__file__).resolve().parents[3]
    / "models"
    / "step3p5"
    / "mtp_hidden_fwd.py"
)


def _source() -> str:
    return _SOURCE_PATH.read_text(encoding="utf-8")


def test_production_program_exports_three_compile_time_variants():
    source = _source()
    assert "mtp_layer_hidden_0" in source
    assert "mtp_layer_hidden_1" in source
    assert "mtp_layer_hidden_2" in source
    assert "MTP_LAYER_HIDDEN_PROGRAMS" in source


def test_production_program_has_hidden_only_output():
    source = _source()
    tree = ast.parse(source)
    class_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }
    assert "MtpLayerHidden" in class_names
    host_orch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "host_orch"
    )
    argument_names = {arg.arg for arg in host_orch.args.args}
    assert "hidden_out" in argument_names
    forbidden = (
        "logits_out",
        "draft_token_ids",
        "argmax",
        "candidate_value",
        "candidate_id",
    )
    for name in forbidden:
        assert name not in argument_names
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and node.name == name
            for node in ast.walk(tree)
        )


def test_production_program_has_no_legacy_main_or_mtp_imports():
    tree = ast.parse(_source())
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert "decode_layer" not in imports
    assert "mtp" not in imports
    assert "dense_mlp" in imports


def test_each_call_owns_fresh_collective_windows():
    source = _source()
    assert source.count("pld.alloc_window_buffer(BATCH * HIDDEN * 2)") == 3
    assert source.count("pld.alloc_window_buffer(tp_size * 4)") == 3
    assert "COMM_CONTROL_SIGNAL_BYTES" not in source
    assert "COMM_SIGNAL_STRIDE_I32" not in source
    assert "run_gen" not in source


def test_selected_mtp_declares_persistent_kv_inout():
    tree = ast.parse(_source())
    layer_orch = None
    host_orch = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "layer_orch":
            layer_orch = node
        elif isinstance(node, ast.FunctionDef) and node.name == "host_orch":
            host_orch = node
    assert layer_orch is not None
    assert host_orch is not None
    for function in (layer_orch, host_orch):
        annotations = {
            arg.arg: ast.unparse(arg.annotation)
            for arg in function.args.args
            if arg.annotation is not None
        }
        for cache_name in ("k_cache", "v_cache"):
            assert annotations[cache_name].startswith("pl.InOut["), (
                f"{function.name}.{cache_name} must preserve selected-MTP "
                "paged KV across proposer rounds"
            )
