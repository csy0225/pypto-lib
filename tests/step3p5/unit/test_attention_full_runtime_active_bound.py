# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Source contracts for full-attention runtime active-row isolation.

These checks intentionally target only the full-attention implementation.  The
static ``BATCH`` dimension remains the storage/tiling capacity, while
``num_tokens`` controls which request rows may read request metadata, execute
RoPE, or update the resident KV cache.
"""
from __future__ import annotations

import ast
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_PATH = _ROOT / "models" / "step3p5" / "attention_full.py"
_SOURCE = _SOURCE_PATH.read_text(encoding="utf-8")
_TREE = ast.parse(_SOURCE)


def _function(name: str) -> ast.FunctionDef:
    for node in ast.walk(_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name!r}")


def _call_name(node: ast.AST) -> str:
    if not isinstance(node, ast.Call):
        return ""
    parts: list[str] = []
    value: ast.AST = node.func
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


def _is_active_guard(node: ast.If, row_name: str) -> bool:
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == row_name
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Lt)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Name)
        and test.comparators[0].id == "active_tokens"
    )


def test_full_attention_rope_and_kv_writes_are_runtime_guarded() -> None:
    fn = _function("attention_full")
    assert "num_tokens" in {arg.arg for arg in fn.args.args}
    assert "b_safe" not in ast.unparse(fn)

    row_loop = None
    for node in ast.walk(fn):
        if not isinstance(node, (ast.For, ast.AsyncFor)):
            continue
        if not isinstance(node.target, ast.Name) or node.target.id != "b":
            continue
        if _call_name(node.iter) != "pl.parallel":
            continue
        if ast.unparse(node.iter) == "pl.parallel(BATCH)":
            row_loop = node
            break
    assert row_loop is not None, "full-attention must retain static-capacity tiling"
    assert len(row_loop.body) == 1 and isinstance(row_loop.body[0], ast.If)
    guard = row_loop.body[0]
    assert _is_active_guard(guard, "b")

    guarded_source = ast.unparse(guard)
    assert "pl.tensor.read(seq_lens, [b])" in guarded_source
    assert "pl.tensor.read(slot_mapping, [b])" in guarded_source
    assert "pl.slice(rope_cos" in guarded_source
    assert "pl.slice(rope_sin" in guarded_source
    assert "k_cache = pl.assemble(k_cache" in guarded_source
    assert "v_cache = pl.assemble(v_cache" in guarded_source


def test_all_full_attention_request_spmd_stages_use_runtime_bound() -> None:
    fn = _function("attention_full")
    required = {
        "full_qk_matmul",
        "full_softmax",
        "full_sv_matmul",
        "full_online_softmax",
    }
    guarded: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.For) or not isinstance(node.target, ast.Name):
            continue
        if _call_name(node.iter) != "pl.spmd":
            continue
        hint = next(
            (
                kw.value.value
                for kw in node.iter.keywords
                if kw.arg == "name_hint" and isinstance(kw.value, ast.Constant)
            ),
            None,
        )
        if hint in required and node.body and isinstance(node.body[0], ast.If):
            if _is_active_guard(node.body[0], node.target.id):
                guarded.add(hint)
    assert guarded == required


def test_standalone_tp_wrapper_forwards_runtime_num_tokens() -> None:
    builder = _function("_build_tp_attention_full_program")
    class_node = next(
        node for node in builder.body if isinstance(node, ast.ClassDef) and node.name == "TpAttentionFull"
    )
    chip = next(
        node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "chip_orch"
    )
    host = next(
        node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "host_orch"
    )
    assert "num_tokens" in {arg.arg for arg in chip.args.args}
    assert "num_tokens" in {arg.arg for arg in host.args.args}

    inline_call = next(
        node for node in ast.walk(chip) if _call_name(node) == "attention_full_inline"
    )
    inline_args = [ast.unparse(arg) for arg in inline_call.args]
    assert "num_tokens" in inline_args
    assert inline_args[inline_args.index("attn_layer_idx") + 1] == "num_tokens"

    chip_call = next(node for node in ast.walk(host) if _call_name(node) == "self.chip_orch")
    assert "num_tokens" in [ast.unparse(arg) for arg in chip_call.args]
