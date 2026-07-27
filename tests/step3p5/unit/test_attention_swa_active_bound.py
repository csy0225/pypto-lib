"""SWA decode attention contracts for runtime active-row KV writes."""
from __future__ import annotations

import ast
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_PATH = _ROOT / "models" / "step3p5" / "attention_swa.py"


def _source() -> str:
    return _SOURCE_PATH.read_text(encoding="utf-8")


def _attention_function() -> ast.FunctionDef:
    tree = ast.parse(_source())
    matches = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "attention_swa"
    ]
    assert len(matches) == 1
    return matches[0]


def _contains_string(node: ast.AST, value: str) -> bool:
    return any(
        isinstance(child, ast.Constant) and child.value == value
        for child in ast.walk(node)
    )


def test_swa_formal_keeps_capacity_and_runtime_num_tokens() -> None:
    function = _attention_function()
    args = {arg.arg: ast.unparse(arg.annotation) for arg in function.args.args}
    assert "num_tokens" in args
    assert "BATCH" in args["current_hidden"]
    assert "USER_BATCH_DYN" in args["seq_lens"]
    assert "USER_BATCH_DYN" in args["slot_mapping"]
    assert ast.unparse(function.returns) == "pl.Tensor[[BATCH, HIDDEN], pl.BF16]"


def test_swa_rope_kv_producer_is_guarded_by_active_tokens() -> None:
    source = _source()
    function = _attention_function()
    loops = [
        node for node in ast.walk(function)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Attribute)
        and node.iter.func.attr == "parallel"
        and ast.unparse(node.iter.args[0]) == "BATCH"
    ]
    rope_loops = [
        node for node in loops
        if _contains_string(node, "swa_rope_kv_cache")
    ]
    assert len(rope_loops) == 1
    loop = rope_loops[0]
    guards = [
        node for node in ast.walk(loop)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "b < active_tokens"
    ]
    assert len(guards) == 1
    guarded_source = ast.get_source_segment(source, guards[0])
    assert guarded_source is not None
    assert "pl.tensor.read(seq_lens, [b])" in guarded_source
    assert "pl.tensor.read(slot_mapping, [b])" in guarded_source
    assert "b_safe" not in guarded_source


def test_swa_has_no_padding_slot_fallback_in_decode_rope_kv_path() -> None:
    source = _source()
    start = source.index("# Scope-2 runtime bound:")
    end = source.index("# ----- fa_fused (SWA)", start)
    scope2 = source[start:end]
    assert "b_safe" not in scope2
    assert "slot_mapping, [b]" in scope2
    assert "slot_mapping, [b_safe]" not in scope2
    assert "for b in pl.parallel(BATCH):" in scope2
    assert "if b < active_tokens:" in scope2
