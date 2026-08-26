# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Static contracts for replicated-input, local-expert Step3p5 MoE."""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import torch

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


def _independent_local_owner_fixture(
    expert_indices: torch.Tensor,
    *,
    tp_size: int,
    experts_per_rank: int,
    expert_capacity: int,
) -> dict[str, torch.Tensor]:
    """Build the owner-local packed route table without product helpers."""
    if expert_indices.ndim != 2:
        raise ValueError("expert_indices must have [token, topk_slot] shape")
    if expert_capacity <= 0:
        raise ValueError("expert_capacity must be positive")

    num_tokens, topk = expert_indices.shape
    next_row = torch.zeros(
        (tp_size, experts_per_rank),
        dtype=torch.int64,
    )
    records: dict[str, list[int]] = {
        "route_id": [],
        "token": [],
        "topk_slot": [],
        "expert_id": [],
        "owner_rank": [],
        "local_expert": [],
        "packed_row": [],
    }

    # Model the actual replicated-input schedule: every rank sees every route
    # and only the unique owner accepts it into its local expert slab.
    for rank in range(tp_size):
        for token in range(num_tokens):
            for topk_slot in range(topk):
                expert_id = int(expert_indices[token, topk_slot])
                owner_rank = expert_id // experts_per_rank
                if not 0 <= owner_rank < tp_size:
                    raise ValueError(f"expert_id={expert_id} has no owner")
                if owner_rank != rank:
                    continue
                local_expert = expert_id % experts_per_rank
                ordinal = int(next_row[rank, local_expert])
                if ordinal >= expert_capacity:
                    raise ValueError(
                        f"expert_id={expert_id} exceeds packed capacity"
                    )
                next_row[rank, local_expert] += 1

                records["route_id"].append(token * topk + topk_slot)
                records["token"].append(token)
                records["topk_slot"].append(topk_slot)
                records["expert_id"].append(expert_id)
                records["owner_rank"].append(owner_rank)
                records["local_expert"].append(local_expert)
                records["packed_row"].append(
                    local_expert * expert_capacity + ordinal
                )

    return {
        name: torch.tensor(values, dtype=torch.int64)
        for name, values in records.items()
    }


def _swiglu_mlp(
    x: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    """Small dense SwiGLU reference with checkpoint-native weight layout."""
    return (torch.nn.functional.silu(x @ gate) * (x @ up)) @ down


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


def test_independent_torch_oracle_matches_dense_reference_moe() -> None:
    """Prove the local-owner partial protocol independent of product routing."""
    tp_size = 4
    experts_per_rank = 3
    num_experts = tp_size * experts_per_rank
    hidden_size = 6
    routed_intermediate = 5
    shared_intermediate_per_rank = 2
    expert_indices = torch.tensor(
        [
            [0, 3, 8],
            [11, 4, 1],
            [6, 2, 9],
            [7, 10, 5],
            [3, 8, 0],
        ],
        dtype=torch.int64,
    )
    expert_weight_logits = torch.tensor(
        [
            [1.0, 2.0, 4.0],
            [3.0, 5.0, 2.0],
            [7.0, 1.0, 3.0],
            [2.0, 6.0, 5.0],
            [4.0, 3.0, 8.0],
        ],
        dtype=torch.float64,
    )
    expert_weights = expert_weight_logits / expert_weight_logits.sum(
        dim=1,
        keepdim=True,
    )
    num_tokens, topk = expert_indices.shape
    expert_capacity = num_tokens * topk
    fixture = _independent_local_owner_fixture(
        expert_indices,
        tp_size=tp_size,
        experts_per_rank=experts_per_rank,
        expert_capacity=expert_capacity,
    )

    expected_route_ids = torch.arange(num_tokens * topk, dtype=torch.int64)
    route_counts = torch.bincount(
        fixture["route_id"],
        minlength=num_tokens * topk,
    )
    assert torch.equal(route_counts, torch.ones_like(expected_route_ids))
    assert torch.equal(
        fixture["owner_rank"],
        fixture["expert_id"] // experts_per_rank,
    )
    assert torch.equal(
        fixture["local_expert"],
        fixture["expert_id"] % experts_per_rank,
    )
    owner_packed_row = (
        fixture["owner_rank"] * experts_per_rank * expert_capacity
        + fixture["packed_row"]
    )
    assert torch.unique(owner_packed_row).numel() == num_tokens * topk

    for rank in range(tp_size):
        for local_expert in range(experts_per_rank):
            selected = (
                (fixture["owner_rank"] == rank)
                & (fixture["local_expert"] == local_expert)
            )
            rows = fixture["packed_row"][selected]
            expected_rows = (
                local_expert * expert_capacity
                + torch.arange(rows.numel(), dtype=torch.int64)
            )
            assert torch.equal(rows, expected_rows)
            assert torch.unique(rows).numel() == rows.numel()

    generator = torch.Generator().manual_seed(20260825)
    hidden = torch.randn(
        (num_tokens, hidden_size),
        dtype=torch.float64,
        generator=generator,
    )
    replicated_hidden = hidden.unsqueeze(0).repeat(tp_size, 1, 1)
    assert all(
        torch.equal(replicated_hidden[0], replicated_hidden[rank])
        for rank in range(1, tp_size)
    )

    routed_gate = torch.randn(
        (num_experts, hidden_size, routed_intermediate),
        dtype=torch.float64,
        generator=generator,
    )
    routed_up = torch.randn(
        (num_experts, hidden_size, routed_intermediate),
        dtype=torch.float64,
        generator=generator,
    )
    routed_down = torch.randn(
        (num_experts, routed_intermediate, hidden_size),
        dtype=torch.float64,
        generator=generator,
    )
    shared_gate = torch.randn(
        (
            tp_size,
            hidden_size,
            shared_intermediate_per_rank,
        ),
        dtype=torch.float64,
        generator=generator,
    )
    shared_up = torch.randn(
        (
            tp_size,
            hidden_size,
            shared_intermediate_per_rank,
        ),
        dtype=torch.float64,
        generator=generator,
    )
    shared_down = torch.randn(
        (
            tp_size,
            shared_intermediate_per_rank,
            hidden_size,
        ),
        dtype=torch.float64,
        generator=generator,
    )

    rank_partials: list[torch.Tensor] = []
    computed_route_ids: list[int] = []
    for rank in range(tp_size):
        shared_partial = _swiglu_mlp(
            replicated_hidden[rank],
            shared_gate[rank],
            shared_up[rank],
            shared_down[rank],
        )
        routed_partial = torch.zeros_like(hidden)
        local_records = torch.nonzero(
            fixture["owner_rank"] == rank,
            as_tuple=False,
        ).flatten()
        for record in local_records.tolist():
            token = int(fixture["token"][record])
            topk_slot = int(fixture["topk_slot"][record])
            expert_id = int(fixture["expert_id"][record])
            routed_partial[token] += expert_weights[token, topk_slot] * (
                _swiglu_mlp(
                    replicated_hidden[rank, token : token + 1],
                    routed_gate[expert_id],
                    routed_up[expert_id],
                    routed_down[expert_id],
                )[0]
            )
            computed_route_ids.append(int(fixture["route_id"][record]))
        rank_partials.append(shared_partial + routed_partial)

    assert Counter(computed_route_ids) == Counter(expected_route_ids.tolist())

    dense_shared = _swiglu_mlp(
        hidden,
        torch.cat(shared_gate.unbind(), dim=1),
        torch.cat(shared_up.unbind(), dim=1),
        torch.cat(shared_down.unbind(), dim=0),
    )
    dense_routed = torch.zeros_like(hidden)
    for token in range(num_tokens):
        for topk_slot in range(topk):
            expert_id = int(expert_indices[token, topk_slot])
            dense_routed[token] += expert_weights[token, topk_slot] * (
                _swiglu_mlp(
                    hidden[token : token + 1],
                    routed_gate[expert_id],
                    routed_up[expert_id],
                    routed_down[expert_id],
                )[0]
            )

    tp_reduced = torch.stack(rank_partials).sum(dim=0)
    torch.testing.assert_close(
        tp_reduced,
        dense_shared + dense_routed,
        rtol=1e-10,
        atol=1e-10,
    )


def test_local_owner_protocol_requires_colocated_tp_ep_ranks() -> None:
    config = _CONFIG.read_text(encoding="utf-8")
    assert "if (TP_WORLD_SIZE, EP_WORLD_SIZE) != (8, 8):" in config
    assert "requires co-located TP=EP=8" in config
    canonical = _CANONICAL.read_text(encoding="utf-8")
    assert "n_ranks = tp_size" in canonical
    assert (
        "global_e = my_rank * n_local_experts + pack_local_e_i32"
        in canonical
    )
    assert "if eid == global_e:" in canonical


def test_local_route_pack_does_not_shadow_expert_indices() -> None:
    source, tree = _parse()
    dispatch = _methods(tree)["dispatch_step"]
    body = ast.get_source_segment(source, dispatch)
    assert body is not None
    assert "local_e_idx = pl.tile.get_block_idx()" in body
    assert "pack_local_e_i32 = pl.cast(local_e_idx, pl.INT32)" in body
    assert (
        "global_e = my_rank * n_local_experts + pack_local_e_i32"
        in body
    )
    assert "pack_out_row_i32 = pl.read(" in body
    assert "local_route_row_out, [0, route]," in body
    assert "if pack_out_row_i32 >= pack_slab_begin_i32:" in body
    assert "if pack_out_row_i32 < pack_slab_end_i32:" in body
    assert "for local_e in pl.range" not in body
    assert "local_e = pl.cast" not in body


def test_route_metadata_has_one_core_group_writer() -> None:
    source, tree = _parse()
    dispatch = _methods(tree)["dispatch_step"]
    parents = _parent_map(dispatch)
    metadata_names = {
        "local_route_row_out",
        "local_expert_count_view",
    }
    writes: dict[str, list[ast.Call]] = {
        name: []
        for name in metadata_names
    }
    for node in ast.walk(dispatch):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if _call_path(node) not in {"pl.write", "pl.store"}:
            continue
        target_arg = (
            node.args[0]
            if _call_path(node) == "pl.write"
            else node.args[2]
        )
        target = ast.unparse(target_arg)
        if target in writes:
            writes[target].append(node)

    for tensor_name, tensor_writes in writes.items():
        assert tensor_writes, tensor_name
        for write in tensor_writes:
            assert _call_path(write) == "pl.store"
            current: ast.AST = write
            scopes: list[ast.With] = []
            while current is not dispatch:
                current = parents[current]
                if isinstance(current, ast.With):
                    scopes.append(current)
            assert any(
                any(
                    isinstance(item.context_expr, ast.Call)
                    and _call_path(item.context_expr) == "pl.at"
                    and any(
                        keyword.arg == "name_hint"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value == "local_route_map_init"
                        for keyword in item.context_expr.keywords
                    )
                    for item in scope.items
                )
                for scope in scopes
            ), (tensor_name, ast.get_source_segment(source, write))


def test_cross_task_route_publications_are_full_tile_stores() -> None:
    _, tree = _parse()
    methods = _methods(tree)
    expected = {
        "expert_indices": ("expert_indices_tile", "[0, 0]"),
        "expert_weights": ("expert_weights_tile", "[0, 0]"),
        "local_expert_count_view": ("expert_count_tile", "[0, 0]"),
        "local_route_row_out": ("route_map", "[0, 0]"),
        "local_routed_x_scale_out": (
            "scale_slab",
            "[0, pack_slab_begin]",
        ),
        "local_routed_weight_out_view": (
            "weight_slab",
            "[0, pack_slab_begin]",
        ),
        "local_route_count_view": ("route_plan_tile", "[0, 0]"),
    }
    publication_calls: dict[str, list[tuple[ast.Call, ast.AST]]] = {
        name: []
        for name in expected
    }
    for function_name in ("_gate", "dispatch_step"):
        function = methods[function_name]
        parents = _parent_map(function)
        for call in (
            node for node in ast.walk(function) if isinstance(node, ast.Call)
        ):
            call_path = _call_path(call)
            if call_path == "pl.write" and call.args:
                assert ast.unparse(call.args[0]) not in expected
            if call_path != "pl.store" or len(call.args) != 3:
                continue
            target = ast.unparse(call.args[2])
            if target in publication_calls:
                publication_calls[target].append((call, parents[call]))

    for target, publications in publication_calls.items():
        assert len(publications) == 1, target
        call, parent = publications[0]
        assert (
            ast.unparse(call.args[0]),
            ast.unparse(call.args[1]),
        ) == expected[target]
        assert isinstance(parent, ast.Assign), target
        assert ast.unparse(parent.targets[0]) == target


def test_physical_metadata_padding_preserves_36_expert_semantics() -> None:
    source, tree = _parse()
    dispatch = _methods(tree)["dispatch_step"]
    body = ast.get_source_segment(source, dispatch)
    assert body is not None

    assert "n_local_experts = N_LOCAL_EXPERTS" in source
    assert "n_local_experts_pad = ((n_local_experts + 7) // 8) * 8" in source
    assert "local_route_plan_valid_size = n_local_experts + 2" in source
    assert "local_route_plan_size = n_local_experts_pad" in source
    assert "[1, n_local_experts_pad], dtype=pl.INT32, value=0" in body
    assert "[1, local_route_plan_size], dtype=pl.INT32, value=0" in body
    assert (
        "local_expert_count, [1, n_local_experts_pad]"
        in body
    )
    assert (
        "local_route_count, [1, local_route_plan_size]"
        in body
    )
    assert "pl.read(local_expert_count_view, [0, e])" in body
    assert (
        "local_routed_weight_out_view, [local_recv_max]"
        in body
    )
    assert (
        "local_route_count_view, [local_route_plan_size]"
        in body
    )
    assert body.count("for e in pl.range(n_local_experts):") == 3
    assert "with pl.spmd(\n            n_local_experts," in body
    assert "route_owner = eid // n_local_experts" in body
    assert "if route_owner == my_rank:" in body
    assert "route_local_e = eid - my_rank * n_local_experts" in body
    assert "if route_local_e >= 0:" not in body
    assert (
        "global_e = my_rank * n_local_experts + pack_local_e_i32"
        in body
    )
    assert "pl.set_validshape" not in body


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
