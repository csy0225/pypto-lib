# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Card-free contracts for the focused Step3p5 L0-L4 MoE graph."""
from __future__ import annotations

import ast
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[3]
_PROGRAM = (
    _ROOT
    / "tests"
    / "step3p5"
    / "harnesses"
    / "_five_layer_moe_program.py"
)
_HOLDER = _ROOT / "tools" / "step3p5" / "five_layer_moe_holder.py"


def _parse(path: Path) -> tuple[str, ast.Module]:
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source)


def _method(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, f"expected one {name}, found {len(matches)}"
    return matches[0]


def _segment(source: str, node: ast.AST) -> str:
    result = ast.get_source_segment(source, node)
    assert result is not None
    return result


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _calls(function: ast.FunctionDef, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and _call_name(node) == name
    ]


def test_focused_graph_is_exactly_l0_through_l4() -> None:
    source, tree = _parse(_PROGRAM)
    function = _method(tree, "five_layer_chip_orch")
    body = _segment(source, function)

    assert body.count("full_chip_orch(") == 1
    assert body.count("swa_chip_orch(") == 2
    assert body.count("swa_moe_chip_orch(") == 1
    assert body.count("full_moe_chip_orch(") == 1
    assert body.index("# L0:") < body.index("# L1:")
    assert body.index("# L1:") < body.index("# L2:")
    assert body.index("# L2:") < body.index("# L3:")
    assert body.index("# L3:") < body.index("# L4:")

    returned = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Return)
    ]
    assert len(returned) == 1
    assert ast.unparse(returned[0].value) == "(hidden_l3, hidden_l4)"


def test_l4_consumes_the_actual_l3_output_and_both_are_host_outputs() -> None:
    _, tree = _parse(_PROGRAM)
    chip = _method(tree, "five_layer_chip_orch")
    full_moe_calls = _calls(chip, "full_moe_chip_orch")
    assert len(full_moe_calls) == 1
    assert ast.unparse(full_moe_calls[0].args[0]) == "hidden_l3"

    for function_name in ("five_layer_chip_orch", "five_layer_host_orch"):
        function = _method(tree, function_name)
        annotations = {
            arg.arg: ast.unparse(arg.annotation)
            for arg in function.args.args
            if arg.annotation is not None
        }
        assert annotations["hidden_l3"].startswith("pl.Out[")
        assert annotations["hidden_l4"].startswith("pl.Out[")


def test_layer_slots_indices_and_epochs_match_canonical_l0_l4() -> None:
    _, tree = _parse(_PROGRAM)
    function = _method(tree, "five_layer_chip_orch")

    calls = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        target = ast.unparse(node.targets[0])
        name = _call_name(node.value)
        if target in {"h0", "h1", "h2", "hidden_l3", "hidden_l4"}:
            calls.append((target, name, node.value))
    by_target = {target: (name, call) for target, name, call in calls}

    assert by_target["h0"][0] == "full_chip_orch"
    assert by_target["h1"][0] == "swa_chip_orch"
    assert by_target["h2"][0] == "swa_chip_orch"
    assert by_target["hidden_l3"][0] == "swa_moe_chip_orch"
    assert by_target["hidden_l4"][0] == "full_moe_chip_orch"

    assert [ast.unparse(arg) for arg in by_target["h0"][1].args[-5:]] == [
        "0", "0", "0", "num_tokens", "my_rank",
    ]
    assert [ast.unparse(arg) for arg in by_target["h1"][1].args[-5:]] == [
        "1", "0", "0", "num_tokens", "my_rank",
    ]
    assert [ast.unparse(arg) for arg in by_target["h2"][1].args[-5:]] == [
        "2", "0", "0", "num_tokens", "my_rank",
    ]
    assert [ast.unparse(arg) for arg in by_target["hidden_l3"][1].args[-5:]] == [
        "3", "0", "num_tokens", "my_rank", "1",
    ]
    assert [ast.unparse(arg) for arg in by_target["hidden_l4"][1].args[-5:]] == [
        "4", "0", "num_tokens", "my_rank", "2",
    ]

    holder = _HOLDER.read_text(encoding="utf-8")
    assert "_FULL_SLOTS = (0, 1)  # L0, L4" in holder
    assert "_SWA_SLOTS = (0, 1, 2)  # L1, L2, L3" in holder
    assert "_MOE_SLOTS = (0, 1)  # L3, L4" in holder


def test_focused_program_reuses_canonical_compute_functions() -> None:
    source, tree = _parse(_PROGRAM)
    assert "import models.step3p5.decode_fwd as _canonical" in source
    assert "_CANONICAL_PROGRAM = _canonical.whole_decode_step3p5" in source
    assert "ir.Program(" in source

    required = {
        "tp_all_reduce",
        "full_chip_orch",
        "swa_chip_orch",
        "_gate",
        "gate_step",
        "_norm_quant_moe_input",
        "dispatch_step",
        "_expert_routed",
        "expert_routed_step",
        "_expert_shared_local",
        "expert_shared_step",
        "combine_step",
        "full_moe_chip_orch",
        "swa_moe_chip_orch",
    }
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_REQUIRED_CANONICAL"
            for target in node.targets
        )
    )
    assert isinstance(assignment.value, ast.Tuple)
    assert {ast.literal_eval(item) for item in assignment.value.elts} == required

    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FiveLayerMoeFocused"
    )
    focused_methods = {
        node.name for node in class_node.body if isinstance(node, ast.FunctionDef)
    }
    assert focused_methods == {"five_layer_chip_orch", "five_layer_host_orch"}
    assert not required.intersection(focused_methods)
