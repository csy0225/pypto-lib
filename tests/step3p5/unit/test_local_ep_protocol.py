# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Static contracts for replicated-input, local-expert Step3p5 MoE."""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

from tools.step3p5.five_layer_moe_golden_contract import (
    CANONICAL_HIDDEN_ONLY_MOE_PROTOCOL_PROFILE,
    LOCAL_OWNER_PROTOCOL_PROFILE,
    canonical_hidden_only_moe_protocol_fields,
)


_ROOT = Path(__file__).resolve().parents[3]
_CANONICAL = _ROOT / "models" / "step3p5" / "decode_fwd.py"
_CONFIG = _ROOT / "models" / "step3p5" / "config.py"
_MOE_ORCHESTRATORS = (
    "full_moe_chip_orch",
    "swa_moe_chip_orch",
    "full_moe_chip_orch_swiglu7_swiglu16",
    "swa_moe_chip_orch_swiglu7_silu",
)
_REMOTE_ROUTE_PRIMITIVES = {
    "pld.system.notify",
    "pld.system.wait",
    "pld.tensor.get",
    "pld.tensor.put",
    "pld.tile.remote_store",
}


def test_canonical_hidden_only_moe_declares_local_owner_protocol() -> None:
    assert (
        CANONICAL_HIDDEN_ONLY_MOE_PROTOCOL_PROFILE
        == LOCAL_OWNER_PROTOCOL_PROFILE
    )
    contract = canonical_hidden_only_moe_protocol_fields()
    assert contract["protocol_profile"] == LOCAL_OWNER_PROTOCOL_PROFILE
    assert contract["bit_exact"] is True


def _parse() -> tuple[str, ast.Module]:
    source = _CANONICAL.read_text(encoding="utf-8")
    return source, ast.parse(source)


def _methods(tree: ast.AST) -> dict[str, ast.FunctionDef]:
    methods = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    assert all(name in methods for name in _MOE_ORCHESTRATORS)
    return methods


def _call_path(call: ast.Call) -> str:
    parts: list[str] = []
    node: ast.expr = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _self_call_name(call: ast.Call) -> str | None:
    if (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "self"
    ):
        return call.func.attr
    return None


def _self_calls(function: ast.FunctionDef) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and _self_call_name(node) is not None
    ]


def _is_inline_method(function: ast.FunctionDef) -> bool:
    return any(
        isinstance(decorator, ast.Call)
        and _call_path(decorator) == "pl.function"
        and any(
            keyword.arg == "type"
            and ast.unparse(keyword.value) == "pl.FunctionType.Inline"
            for keyword in decorator.keywords
        )
        for decorator in function.decorator_list
    )


def _reachable_methods(
    methods: dict[str, ast.FunctionDef],
    roots: tuple[str, ...],
) -> set[str]:
    pending = list(roots)
    reachable: set[str] = set()
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        reachable.add(name)
        for call in _self_calls(methods[name]):
            callee = _self_call_name(call)
            if callee in methods and callee not in reachable:
                pending.append(callee)
    return reachable


def _assignment_targets(node: ast.Assign | ast.AnnAssign) -> list[str]:
    target_nodes: list[ast.expr]
    if isinstance(node, ast.Assign):
        target_nodes = node.targets
    else:
        target_nodes = [node.target]
    result: list[str] = []
    for target in target_nodes:
        result.extend(
            child.id
            for child in ast.walk(target)
            if isinstance(child, ast.Name)
        )
    return result


def _value_tags(
    value: ast.AST,
    tags_by_name: dict[str, set[str]],
) -> set[str]:
    tags: set[str] = set()
    for node in ast.walk(value):
        if isinstance(node, ast.Name):
            tags.update(tags_by_name.get(node.id, ()))
        elif isinstance(node, ast.Call):
            callee = _self_call_name(node) or _call_path(node)
            if "shared" in callee:
                tags.add("shared")
            if "routed" in callee:
                tags.add("routed")
    return tags


def _tags_before_call(
    function: ast.FunctionDef,
    target_call: ast.Call,
) -> tuple[set[str], set[str]]:
    tags_by_name: dict[str, set[str]] = {}
    result_names: set[str] = set()
    statements = sorted(
        (
            node
            for node in ast.walk(function)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and node.lineno <= target_call.lineno
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    for statement in statements:
        value = statement.value
        if value is None:
            continue
        if (
            isinstance(value, ast.Call)
            and value is target_call
        ):
            result_names.update(_assignment_targets(statement))
            break
        value_tags = _value_tags(value, tags_by_name)
        for name in _assignment_targets(statement):
            tags_by_name[name] = set(value_tags)
    return _value_tags(target_call.args[0], tags_by_name), result_names


def _parent_map(root: ast.AST) -> dict[ast.AST, ast.AST]:
    return {
        child: parent
        for parent in ast.walk(root)
        for child in ast.iter_child_nodes(parent)
    }


def _enclosing_conditions(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    stop: ast.AST,
) -> list[ast.expr]:
    conditions: list[ast.expr] = []
    current = node
    while current is not stop:
        current = parents[current]
        if isinstance(current, ast.If):
            conditions.append(current.test)
    return conditions


def _owner_filter_present(
    methods: dict[str, ast.FunctionDef],
    reachable: set[str],
) -> bool:
    """Accept either a named owner/destination or an equivalent direct test."""
    for name in reachable:
        function = methods[name]
        source_assignments: dict[str, str] = {}
        for node in ast.walk(function):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if value is None:
                continue
            value_source = ast.unparse(value)
            for target in _assignment_targets(node):
                source_assignments[target] = value_source

        for node in ast.walk(function):
            if not isinstance(node, ast.Compare):
                continue
            if not any(
                isinstance(op, (ast.Eq, ast.NotEq))
                for op in node.ops
            ):
                continue
            operands = [node.left, *node.comparators]
            expanded_operands: list[str] = []
            for operand in operands:
                expanded = ast.unparse(operand)
                for referenced in (
                    child.id
                    for child in ast.walk(operand)
                    if isinstance(child, ast.Name)
                ):
                    producer = source_assignments.get(referenced)
                    if producer is not None:
                        expanded += f" ({producer})"
                expanded_operands.append(expanded)

            comparison_source = " ".join(expanded_operands)
            has_owner_expression = (
                "my_rank" in comparison_source
                and (
                    "n_local_experts" in comparison_source
                    or "N_LOCAL" in comparison_source
                )
            )
            has_gate_expert = any(
                "expert_indices" in operand
                or "eid" in operand
                or "expert_id" in operand
                for operand in expanded_operands
            )
            if has_owner_expression and has_gate_expert:
                return True
    return False


def _route_ownership(
    expert_indices: list[list[int]],
    *,
    tp_size: int,
    experts_per_rank: int,
) -> list[list[int]]:
    """Model the replicated gate result followed by rank-local owner filtering."""
    topk = len(expert_indices[0]) if expert_indices else 0
    routes_by_rank: list[list[int]] = [[] for _ in range(tp_size)]
    for rank in range(tp_size):
        for token, row in enumerate(expert_indices):
            assert len(row) == topk
            for slot, expert in enumerate(row):
                owner = expert // experts_per_rank
                assert 0 <= owner < tp_size
                if owner == rank:
                    routes_by_rank[rank].append(token * topk + slot)
    return routes_by_rank


def test_product_moe_route_path_is_local_only_and_owner_filtered() -> None:
    _, tree = _parse()
    methods = _methods(tree)
    reachable = _reachable_methods(methods, _MOE_ORCHESTRATORS)

    # The one permitted distributed helper is the final TP all-reduce.
    routed_path = reachable - {"tp_all_reduce", "tp_all_reduce_residual_bs1"}
    remote_calls = {
        (name, _call_path(call))
        for name in routed_path
        for call in ast.walk(methods[name])
        if isinstance(call, ast.Call)
        and _call_path(call) in _REMOTE_ROUTE_PRIMITIVES
    }
    assert not remote_calls
    assert _owner_filter_present(methods, routed_path)


def test_local_ep_inline_helpers_use_bare_tensor_parameters() -> None:
    _, tree = _parse()
    methods = _methods(tree)
    reachable = _reachable_methods(methods, _MOE_ORCHESTRATORS)

    inline_methods = {
        name
        for name in reachable
        if _is_inline_method(methods[name])
    }
    assert {
        "gate_step",
        "_norm_quant_moe_input",
        "dispatch_step",
        "expert_routed_step",
        "expert_shared_step",
        "combine_step",
        "expert_routed_step_swiglu7",
        "expert_shared_step_swiglu16",
    }.issubset(inline_methods)
    for name in inline_methods:
        for argument in methods[name].args.args:
            if argument.annotation is None:
                continue
            annotation = ast.unparse(argument.annotation)
            assert not annotation.startswith(("pl.Out[", "pl.InOut[")), (
                name,
                argument.arg,
                annotation,
            )


def test_shared_expert_wrappers_return_local_partials_without_collective() -> None:
    _, tree = _parse()
    methods = _methods(tree)
    reachable = _reachable_methods(methods, _MOE_ORCHESTRATORS)
    shared_methods = {
        name
        for name in reachable
        if "shared" in name
    }
    assert shared_methods
    for name in shared_methods:
        assert all(
            _self_call_name(call) != "tp_all_reduce"
            for call in _self_calls(methods[name])
        ), name


def test_each_moe_orchestrator_reduces_merged_local_partial_once() -> None:
    _, tree = _parse()
    methods = _methods(tree)

    for name in _MOE_ORCHESTRATORS:
        function = methods[name]
        collectives = [
            call
            for call in _self_calls(function)
            if _self_call_name(call) == "tp_all_reduce"
        ]
        assert len(collectives) == 1, name
        collective = collectives[0]

        local_tags, collective_result_names = _tags_before_call(
            function, collective,
        )
        assert local_tags == {"shared", "routed"}, name
        assert collective_result_names, name

        residual_scopes = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.With)
            and any(
                isinstance(item.context_expr, ast.Call)
                and any(
                    keyword.arg == "name_hint"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value == "moe_residual_add"
                    for keyword in item.context_expr.keywords
                )
                for item in node.items
            )
        ]
        assert len(residual_scopes) == 1, name
        residual_scope = residual_scopes[0]
        assert collective.lineno < residual_scope.lineno, name
        residual_names = {
            node.id
            for node in ast.walk(residual_scope)
            if isinstance(node, ast.Name)
        }
        assert collective_result_names & residual_names, name


def test_zero_route_rank_cannot_skip_the_final_collective() -> None:
    _, tree = _parse()
    methods = _methods(tree)

    for name in _MOE_ORCHESTRATORS:
        function = methods[name]
        collectives = [
            call
            for call in _self_calls(function)
            if _self_call_name(call) == "tp_all_reduce"
        ]
        assert len(collectives) == 1, name
        collective = collectives[0]
        parents = _parent_map(function)
        for condition in _enclosing_conditions(
            collective, parents, function,
        ):
            names = {
                node.id
                for node in ast.walk(condition)
                if isinstance(node, ast.Name)
            }
            assert not names.intersection(
                {
                    "active_expert_count",
                    "local_expert_count",
                    "local_route_count",
                    "my_rank",
                    "num_tokens",
                    "route_count",
                }
            ), (name, ast.unparse(condition))


def test_replicated_routes_are_computed_once_by_their_unique_owner() -> None:
    tp_size = 8
    experts_per_rank = 36
    topk = 8
    expert_indices = [
        [
            (token * 37 + slot * 53 + token // 3)
            % (tp_size * experts_per_rank)
            for slot in range(topk)
        ]
        for token in range(16)
    ]
    routes_by_rank = _route_ownership(
        expert_indices,
        tp_size=tp_size,
        experts_per_rank=experts_per_rank,
    )
    all_routes = [
        route
        for local_routes in routes_by_rank
        for route in local_routes
    ]
    expected_routes = list(range(len(expert_indices) * topk))

    assert sorted(all_routes) == expected_routes
    assert Counter(all_routes) == Counter(expected_routes)
    for rank, local_routes in enumerate(routes_by_rank):
        assert all(
            expert_indices[route // topk][route % topk]
            // experts_per_rank
            == rank
            for route in local_routes
        )

    # A legal skew can leave seven ranks empty. Those ranks still enter the
    # source-level unconditional collective checked above.
    all_rank_zero = [[slot % experts_per_rank for slot in range(topk)]]
    skewed = _route_ownership(
        all_rank_zero,
        tp_size=tp_size,
        experts_per_rank=experts_per_rank,
    )
    assert skewed[0] == list(range(topk))
    assert all(not routes for routes in skewed[1:])


def test_local_owner_protocol_requires_colocated_tp_ep_ranks() -> None:
    config = _CONFIG.read_text(encoding="utf-8")
    assert "if (TP_WORLD_SIZE, EP_WORLD_SIZE) != (8, 8):" in config
    assert "requires co-located TP=EP=8" in config
    canonical = _CANONICAL.read_text(encoding="utf-8")
    assert "n_ranks = tp_size" in canonical
    assert (
        "global_e = my_rank * n_local_experts + local_e_i32"
        in canonical
    )
    assert "if eid == global_e:" in canonical


def test_local_route_pack_does_not_shadow_expert_indices() -> None:
    source, tree = _parse()
    dispatch = _methods(tree)["dispatch_step"]
    body = ast.get_source_segment(source, dispatch)
    assert body is not None
    assert "local_e_idx = pl.tile.get_block_idx()" in body
    assert "local_e_i32 = pl.cast(local_e_idx, pl.INT32)" in body
    assert (
        "global_e = my_rank * n_local_experts + local_e_i32"
        in body
    )
    assert "for local_e in pl.range" not in body
    assert "local_e = pl.cast" not in body


def test_local_route_rows_are_bounds_checked_before_routed_load() -> None:
    _, tree = _parse()
    combine = _methods(tree)["combine_step"]
    parents = _parent_map(combine)
    routed_loads = [
        node
        for node in ast.walk(combine)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "local_routed_y"
    ]
    assert len(routed_loads) == 1
    conditions = _enclosing_conditions(
        routed_loads[0], parents, combine,
    )
    condition_source = " ".join(
        ast.unparse(condition)
        for condition in conditions
    )
    assert "local_row_i32 >= 0" in condition_source
    assert "local_row_i32 < local_recv_max" in condition_source
    guarded_casts = [
        node
        for node in ast.walk(combine)
        if isinstance(node, ast.Call)
        and _call_path(node) == "pl.cast"
        and node.args
        and ast.unparse(node.args[0]) == "local_row_i32"
    ]
    assert len(guarded_casts) == 1
    cast_condition_source = " ".join(
        ast.unparse(condition)
        for condition in _enclosing_conditions(
            guarded_casts[0], parents, combine,
        )
    )
    assert "local_row_i32 >= 0" in cast_condition_source
    assert "local_row_i32 < local_recv_max" in cast_condition_source
