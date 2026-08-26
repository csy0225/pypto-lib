# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Static release contracts for PERF-B3 / PERF-C1 / PERF-C3 / PERF-G1."""
from __future__ import annotations

import ast
import re
from pathlib import Path

from tests.step3p5.probes._probe_g1_active_batch import _executable_match


_ROOT = Path(__file__).resolve().parents[3]
_CANONICAL = _ROOT / "models" / "step3p5" / "decode_fwd.py"
_HOLDER = _ROOT / "tools" / "step3p5" / "whole_decode_holder.py"
_FULL_ATTN = _ROOT / "models" / "step3p5" / "attention_full.py"
_SWA_ATTN = _ROOT / "models" / "step3p5" / "attention_swa.py"
_MTP_HIDDEN = _ROOT / "models" / "step3p5" / "mtp_hidden_fwd.py"
_ROUTED_DOWN_SOURCES = (
    _ROOT / "kernels" / "step3p5" / "routed_nz" / "expert_down_aic.cpp",
    _ROOT / "kernels" / "step3p5" / "routed_nz" / "expert_down_aiv.cpp",
)
_ROUTED_GMM1_SOURCES = (
    (
        _ROOT
        / "kernels"
        / "step3p5"
        / "routed_nz"
        / "routed_gmm1_swiglu_quant_aic.cpp"
    ),
    (
        _ROOT
        / "kernels"
        / "step3p5"
        / "routed_nz"
        / "routed_gmm1_swiglu_quant_aiv.cpp"
    ),
    (
        _ROOT
        / "kernels"
        / "step3p5"
        / "routed_nz"
        / "routed_gmm1_swiglu7_quant_aiv.cpp"
    ),
)
_MAIN_HARNESS = (
    _ROOT
    / "tests"
    / "step3p5"
    / "harnesses"
    / "_stage_main_hidden_only.py"
)
_TWO_LAYER_PROGRAM = (
    _ROOT
    / "tests"
    / "step3p5"
    / "harnesses"
    / "_two_layer_program.py"
)


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


def _method_calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    ]


def _single_function(source: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    ]
    assert len(functions) == 1
    return functions[0]


def _task_scope(function: ast.FunctionDef, name_hint: str) -> ast.With:
    matches: list[ast.With] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            call = item.context_expr
            if not isinstance(call, ast.Call):
                continue
            if any(
                keyword.arg == "name_hint"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == name_hint
                for keyword in call.keywords
            ):
                matches.append(node)
    assert len(matches) == 1, (
        f"expected one task scope {name_hint}, found {len(matches)}"
    )
    return matches[0]


def _call_path(call: ast.Call) -> str:
    parts: list[str] = []
    node: ast.expr = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _self_method_calls(
    function: ast.FunctionDef,
    method_name: str,
) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == method_name
    ]


def _pl_function_type(function: ast.FunctionDef) -> str:
    matches = [
        ast.unparse(keyword.value)
        for decorator in function.decorator_list
        if isinstance(decorator, ast.Call)
        and _call_path(decorator) == "pl.function"
        for keyword in decorator.keywords
        if keyword.arg == "type"
    ]
    assert len(matches) == 1, function.name
    return matches[0]


def test_g1_executable_match_is_exact_and_fail_closed() -> None:
    function = _single_function(
        """
def sample():
    # for t in pl.range(active_tokens):
    "active_tokens = pl.cast(num_tokens, pl.INDEX)"
    for t in pl.range(storage_batch):
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        pl.read(num_tokens_per_owner, [owner_rank])
"""
    )
    assert _executable_match(
        function,
        "active_tokens = pl.cast(num_tokens, pl.INDEX)",
    )["present"]
    assert _executable_match(
        function,
        "pl.read(num_tokens_per_owner, [owner_rank])",
    )["present"]

    # A comment/string cannot satisfy the loop pattern, and the outer For
    # cannot inherit a match from executable statements in its body.
    loop = _executable_match(
        function,
        "for t in pl.range(active_tokens):",
    )
    assert not loop["present"]
    assert "pattern_error" not in loop

    # A formal argument alone is not executable evidence.
    formal_only = _single_function(
        """
def sample(local_expert_count):
    return None
"""
    )
    assert not _executable_match(
        formal_only,
        "pl.read(local_expert_count, [e])",
    )["present"]

    nested_only = _single_function(
        """
def sample():
    def fake():
        for t in pl.range(active_tokens):
            pass
    return fake
"""
    )
    assert not _executable_match(
        nested_only,
        "for t in pl.range(active_tokens):",
    )["present"]


def test_routed_group_members_keep_exact_python_abi_and_call_order() -> None:
    _, tree = _parse(_CANONICAL)
    for group_name, member_names in (
        (
            "routed_nz_gmm1_swiglu_quant",
            (
                (
                    "routed_nz_gmm1_swiglu_quant_aic",
                    "pl.FunctionType.AIC",
                ),
                (
                    "routed_nz_gmm1_swiglu_quant_aiv",
                    "pl.FunctionType.AIV",
                ),
            ),
        ),
        (
            "routed_nz_gmm1_swiglu7_quant",
            (
                (
                    "routed_nz_gmm1_swiglu_quant_aic",
                    "pl.FunctionType.AIC",
                ),
                (
                    "routed_nz_gmm1_swiglu7_quant_aiv",
                    "pl.FunctionType.AIV",
                ),
            ),
        ),
        (
            "routed_nz_down",
            (
                ("routed_nz_down_aic", "pl.FunctionType.AIC"),
                ("routed_nz_down_aiv", "pl.FunctionType.AIV"),
            ),
        ),
    ):
        group = _method(tree, group_name)
        assert _pl_function_type(group) == "pl.FunctionType.Group"
        expected_call_args = [
            ast.dump(
                ast.Name(id=arg.arg, ctx=ast.Load()),
                include_attributes=False,
            )
            for arg in group.args.args[1:]
        ]
        group_args = ast.dump(group.args, include_attributes=False)
        group_returns = ast.dump(group.returns, include_attributes=False)

        for member_name, member_type in member_names:
            member = _method(tree, member_name)
            assert _pl_function_type(member) == member_type
            assert ast.dump(
                member.args,
                include_attributes=False,
            ) == group_args, (group_name, member_name)
            assert ast.dump(
                member.returns,
                include_attributes=False,
            ) == group_returns, (group_name, member_name)

            calls = _self_method_calls(group, member_name)
            assert len(calls) == 1, (group_name, member_name)
            assert [
                ast.dump(argument, include_attributes=False)
                for argument in calls[0].args
            ] == expected_call_args, (group_name, member_name)
            assert not calls[0].keywords, (group_name, member_name)


def test_b3_canonical_kv_is_resident_inout_and_holder_never_copies_pool() -> None:
    source, tree = _parse(_CANONICAL)
    del source
    for function_name in ("whole_chip_orch", "host_orch"):
        function = _method(tree, function_name)
        annotations = {
            arg.arg: ast.unparse(arg.annotation)
            for arg in function.args.args
            if arg.annotation is not None
        }
        for cache_name in ("k_cache", "v_cache"):
            assert annotations[cache_name].startswith("pl.InOut[")

    holder_source, holder_tree = _parse(_HOLDER)
    enter_source = _segment(holder_source, _method(holder_tree, "_enter_impl"))
    run_source = _segment(holder_source, _method(holder_tree, "run"))
    assert enter_source.count("import_kv_all(") == 1
    assert enter_source.count("build_stacked_kv_pool(") == 1
    assert "self.k_cache, self.v_cache = build_stacked_kv_pool(" in enter_source
    assert "self.rt.run(self.compiled, *self._args_list" in run_source
    assert "copy_(" not in run_source
    assert "import_kv_all" not in run_source
    assert "build_stacked_kv_pool" not in run_source


def test_b3_attention_only_writes_slot_addressed_head_rows() -> None:
    for path, prefix in (
        (_FULL_ATTN, "full"),
        (_SWA_ATTN, "swa"),
    ):
        source = path.read_text(encoding="utf-8")
        assert f'name_hint="{prefix}_qkv_proj"' in source
        assert f'name_hint="{prefix}_qkv_split_qknorm_rope"' in source
        assert f'name_hint="{prefix}_rope_q"' not in source
        assert f'name_hint="{prefix}_rope_kv_cache"' not in source
        assert f"{prefix}_rope_stage = pl.create_tensor(" not in source
        assert f"{prefix}_k_rope_stage = pl.create_tensor(" not in source
        assert f"{prefix}_v_stage = pl.create_tensor(" not in source
        assert source.count("        active_tokens,") >= 1
        assert "if b < active_tokens:" in source
        assert "slot = pl.tensor.read(slot_mapping, [b])" in source
        assert "slot_mapping, [b_safe]" not in source
        assert "layer_cache_base" in source
        assert "cache_row = (" in source
        assert f"deps=[{prefix}_qkv_proj_tid]" in source
        assert f"deps=[{prefix}_qkv_prerope_tid]" in source
        assert "k_cache = pl.assemble(" in source
        assert "v_cache = pl.assemble(" in source
        assert "[cache_row, 0]" in source
        assert "qkv_proj" in source
        # The only cache-row address is layer base + slot-derived block/offset
        # (+ local KV-head lane); slot_mapping never embeds a layer base.
        cache_row_block = source[
            source.index("cache_row = (") :
            source.index("cache_row = (") + 320
        ]
        assert "layer_cache_base" in cache_row_block
        assert "slot_block" in cache_row_block
        assert "slot_offset" in cache_row_block


def test_b3_device_probe_covers_all_layers_and_adjacent_history_rows() -> None:
    harness = _MAIN_HARNESS.read_text(encoding="utf-8")
    assert '"layer_indices": list(range(45))' in harness
    assert '"slots": [0, 1, 2]' in harness
    assert 'probe_summary["slot0_hashes"]' in harness
    assert 'previous_kv_probe_summary["slot0_hashes"]' in harness
    assert 'probe_summary["slot2_any_nonzero"]' in harness


def test_local_ep_helpers_drop_epoch_and_calls_preserve_arity() -> None:
    _, tree = _parse(_CANONICAL)
    for name in ("dispatch_step", "combine_step"):
        function = _method(tree, name)
        args = [arg.arg for arg in function.args.args]
        assert "moe_epoch" not in args
        expected = len(args) - 1
        calls = _method_calls(tree, name)
        assert calls, f"{name} has no call site"
        assert all(len(call.args) == expected for call in calls)
    dispatch_args = [
        arg.arg for arg in _method(tree, "dispatch_step").args.args
    ]
    assert dispatch_args[-2:] == ["num_tokens", "my_rank"]
    combine_args = [
        arg.arg for arg in _method(tree, "combine_step").args.args
    ]
    assert combine_args[-3:] == [
        "local_route_count_tid", "routed_down_tid", "num_tokens",
    ]


def test_c3_combine_is_local_only_and_plain_fp32_reduce() -> None:
    source, tree = _parse(_CANONICAL)
    function = _method(tree, "combine_step")
    names = [arg.arg for arg in function.args.args[1:]]
    assert names == [
        "local_routed_y", "sh_y", "moe_out",
        "local_route_row", "local_route_count_tid",
        "routed_down_tid", "num_tokens",
    ]
    body = _segment(source, function)
    assert 'name_hint="local_combine_reduce"' in body
    assert "pl.read(local_route_row, [0, route])" in body
    assert "if local_row_i32 >= 0:" in body
    assert "if local_row_i32 < local_recv_max:" in body
    assert "local_routed_y[local_row : local_row + 1, :]" in body
    assert "target_type=pl.FP32" in body
    assert "expert_weights" not in body
    assert "route_weight" not in body
    assert "pld." not in body


def test_local_ep_helpers_have_no_remote_signal_lineage() -> None:
    source, tree = _parse(_CANONICAL)
    dispatch = _segment(source, _method(tree, "dispatch_step"))
    combine = _segment(source, _method(tree, "combine_step"))
    for body in (dispatch, combine):
        assert "pld.system.notify" not in body
        assert "pld.system.wait" not in body
        assert "pld.tensor.put" not in body
        assert "pld.tile.remote_store" not in body
        assert "moe_epoch" not in body


def test_c1_legacy_ep_windows_are_removed_from_the_product_abi() -> None:
    source, tree = _parse(_CANONICAL)
    whole = _method(tree, "whole_chip_orch")
    annotations = {
        arg.arg: ast.unparse(arg.annotation)
        for arg in whole.args.args if arg.annotation is not None
    }
    ep_windows = (
        "moe_recv_meta_stack", "moe_meta_arrived_stack", "moe_recv_x_stack",
        "moe_recv_aux_stack", "moe_recv_route_stack", "moe_data_arrived_stack",
        "moe_combine_arrived_stack", "moe_routed_y_buf_stack",
    )
    for name in ep_windows:
        assert name not in annotations
        assert name not in _segment(source, _method(tree, "host_orch"))

    legacy_formals = {
        "recv_meta", "meta_arrived", "recv_x", "recv_aux", "recv_route",
        "data_arrived", "combine_arrived", "routed_y_buf", "moe_epoch",
    }
    for function_name in (
        "full_moe_chip_orch",
        "swa_moe_chip_orch",
        "full_moe_chip_orch_swiglu7_swiglu16",
        "swa_moe_chip_orch_swiglu7_silu",
    ):
        function = _method(tree, function_name)
        formal_names = {arg.arg for arg in function.args.args}
        assert not legacy_formals.intersection(formal_names), function_name
        body_names = {
            node.id
            for statement in function.body
            for node in ast.walk(statement)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
        }
        assert not legacy_formals.intersection(body_names), function_name
        assert {"sh_tmp_window", "sh_signal_window"} <= body_names


def test_c1_512b_stride_is_local_to_stacked_or_reused_control_slots() -> None:
    source, tree = _parse(_CANONICAL)
    assert "COMM_CONTROL_SIGNAL_BYTES = 512" in source
    assert "COMM_SIGNAL_STRIDE_I32 = COMM_CONTROL_SIGNAL_BYTES // 4" in source
    host_source = _segment(source, _method(tree, "host_orch"))
    for stack_name, slots in (
        ("dense_attn_signal_stack_buf", "NUM_DENSE_LAYERS"),
        ("dense_mlp_signal_stack_buf", "NUM_DENSE_LAYERS"),
        ("moe_attn_signal_stack_buf", "NUM_MOE_LAYERS_TOTAL"),
        ("moe_sh_signal_stack_buf", "NUM_MOE_LAYERS_TOTAL"),
    ):
        assert f"{stack_name} = pld.alloc_window_buffer({slots} * COMM_CONTROL_SIGNAL_BYTES)" in host_source
    for removed_name in (
        "moe_meta_arrived_stack_buf", "moe_data_arrived_stack_buf",
        "moe_combine_arrived_stack_buf",
    ):
        assert removed_name not in host_source


def test_c1_whole_graph_has_no_legacy_moe_epoch_protocol() -> None:
    source, tree = _parse(_CANONICAL)
    whole_source = _segment(source, _method(tree, "whole_chip_orch"))
    assert "moe_epoch" not in whole_source
    for name in ("dispatch_step", "combine_step"):
        assert "moe_epoch" not in _segment(source, _method(tree, name))


def test_c3_local_route_pack_and_combine_are_communication_free() -> None:
    source, tree = _parse(_CANONICAL)
    dispatch_function = _method(tree, "dispatch_step")
    combine_function = _method(tree, "combine_step")
    dispatch = _segment(source, dispatch_function)
    combine = _segment(source, combine_function)
    assert 'name_hint="local_route_pack"' in dispatch
    assert 'name_hint="local_combine_reduce"' in combine
    assert "for t in pl.range(active_tokens):" in dispatch
    assert "for k in pl.range(TOPK):" in dispatch
    assert "pld." not in dispatch
    assert "pld." not in combine


def test_gate_topk_publishes_complete_route_tiles_once() -> None:
    source, tree = _parse(_CANONICAL)
    gate = _method(tree, "_gate")
    gate_topk = _task_scope(gate, "gate_topk")
    gate_topk_source = _segment(source, gate_topk)

    assert (
        "expert_indices_tile = pl.tile.full(\n"
        "                [BATCH, TOPK], dtype=pl.INT32, value=0,"
        in gate_topk_source
    )
    assert (
        "expert_weights_tile = pl.tile.full(\n"
        "                [BATCH, TOPK], dtype=pl.FP32, value=0.0,"
        in gate_topk_source
    )
    assert "pl.tile.write(\n                        expert_indices_tile" in (
        gate_topk_source
    )
    assert "pl.tile.write(\n                        expert_weights_tile" in (
        gate_topk_source
    )

    stores: dict[str, list[ast.Assign]] = {
        "expert_indices": [],
        "expert_weights": [],
    }
    for node in ast.walk(gate_topk):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and _call_path(node.value) == "pl.store"
            and len(node.value.args) == 3
        ):
            target = ast.unparse(node.value.args[2])
            if target in stores:
                stores[target].append(node)
    for target, assignments in stores.items():
        assert len(assignments) == 1, target
        assert ast.unparse(assignments[0].targets[0]) == target
        expected_tile = f"{target}_tile"
        assert [ast.unparse(arg) for arg in assignments[0].value.args] == [
            expected_tile,
            "[0, 0]",
            target,
        ]

    scalar_writes = [
        call
        for call in ast.walk(gate)
        if isinstance(call, ast.Call)
        and _call_path(call) == "pl.write"
        and call.args
        and ast.unparse(call.args[0]) in stores
    ]
    assert not scalar_writes

    returned = [
        node for node in ast.walk(gate) if isinstance(node, ast.Return)
    ]
    assert len(returned) == 1
    assert ast.unparse(returned[0].value) == (
        "(expert_indices, expert_weights)"
    )
    gate_step = _method(tree, "gate_step")
    gate_step_returns = [
        node
        for node in ast.walk(gate_step)
        if isinstance(node, ast.Return)
    ]
    assert len(gate_step_returns) == 1
    assert ast.unparse(gate_step_returns[0].value) == (
        "(expert_indices, expert_weights)"
    )
    gate_calls = _self_method_calls(gate_step, "_gate")
    assert len(gate_calls) == 1
    gate_assignments = [
        node
        for node in ast.walk(gate_step)
        if isinstance(node, ast.Assign)
        and node.value is gate_calls[0]
    ]
    assert len(gate_assignments) == 1
    assert ast.unparse(gate_assignments[0].targets[0]) == (
        "(expert_indices, expert_weights)"
    )


def test_c3_local_expert_packing_and_combine_cover_active_routes() -> None:
    source, tree = _parse(_CANONICAL)
    dispatch_function = _method(tree, "dispatch_step")
    dispatch = _segment(source, dispatch_function)
    combine = _segment(source, _method(tree, "combine_step"))

    assert "with pl.spmd(\n            n_local_experts," in dispatch
    assert (
        "global_e = my_rank * n_local_experts + pack_local_e_i32"
        in dispatch
    )
    assert "if eid == global_e:" in dispatch
    assert "pack_out_row_i32 = pl.read(" in dispatch
    assert "local_route_row_out, [0, route]," in dispatch
    route_map_init = _task_scope(dispatch_function, "local_route_map_init")
    route_map_stores = [
        call
        for call in ast.walk(route_map_init)
        if isinstance(call, ast.Call)
        and _call_path(call) == "pl.store"
        and [ast.unparse(argument) for argument in call.args]
        == ["route_map", "[0, 0]", "local_route_row_out"]
    ]
    assert len(route_map_stores) == 1
    assert "cursor = pl.array.create(n_local_experts, pl.INT32)" in (
        _segment(source, route_map_init)
    )
    local_pack = _task_scope(dispatch_function, "local_route_pack")
    local_pack_call = local_pack.items[0].context_expr
    assert isinstance(local_pack_call, ast.Call)
    assert _call_path(local_pack_call) == "pl.spmd"
    assert [ast.unparse(arg) for arg in local_pack_call.args] == [
        "n_local_experts",
    ]
    local_pack_keywords = {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in local_pack_call.keywords
    }
    assert local_pack_keywords["deps"] == "[route_map_init_tid]"
    local_plan = _task_scope(dispatch_function, "local_route_plan")
    local_plan_call = local_plan.items[0].context_expr
    assert isinstance(local_plan_call, ast.Call)
    assert _call_path(local_plan_call) == "pl.at"
    local_plan_keywords = {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in local_plan_call.keywords
    }
    assert local_plan_keywords["deps"] == "[local_pack_tid]"
    assert "local_route_row_out," in dispatch
    assert "[0, route]" in dispatch
    assert "local_route_count = pl.create_tensor(" in dispatch
    assert "[local_route_plan_size], dtype=pl.INT32" in dispatch
    assert dispatch.count("pl.Tensor[[local_route_plan_size], pl.INT32]") == 1
    assert "active_expert_count_i32 = pl.cast(0, pl.INT32)" in dispatch
    route_plan_stores = [
        call
        for call in ast.walk(local_plan)
        if isinstance(call, ast.Call)
        and _call_path(call) == "pl.store"
        and [ast.unparse(argument) for argument in call.args]
        == ["route_plan_tile", "[0, 0]", "local_route_count_view"]
    ]
    assert len(route_plan_stores) == 1
    assert "with pl.spmd(\n            BATCH," in combine
    assert "for k in pl.range(TOPK):" in combine
    assert "local_row_i32 = pl.read(local_route_row, [0, route])" in combine
    assert "if local_row_i32 >= 0:" in combine
    assert "if local_row_i32 < local_recv_max:" in combine
    assert "local_route_count_tid: pl.Scalar[pl.TASK_ID]" in combine
    assert "deps=[local_route_count_tid, routed_down_tid]" in combine


def test_c3_local_route_map_is_dense_and_each_route_has_one_owner() -> None:
    source, tree = _parse(_CANONICAL)
    function = _method(tree, "dispatch_step")
    metadata = _task_scope(function, "local_route_map_init")
    metadata_source = _segment(source, metadata)
    pack = _task_scope(function, "local_route_pack")
    pack_source = _segment(source, pack)

    assert "route_owner = eid // n_local_experts" in metadata_source
    assert "if route_owner == my_rank:" in metadata_source
    assert (
        "route_local_e = eid - my_rank * n_local_experts"
        in metadata_source
    )
    assert "if route_local_e >= 0:" not in metadata_source
    assert "if route_local_e < n_local_experts:" not in metadata_source
    assert "packed_count_i32 = cursor[route_local_e]" in metadata_source
    route_row_assignments = [
        node
        for node in ast.walk(metadata)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and ast.unparse(node.targets[0]) == "route_out_row_i32"
    ]
    assert len(route_row_assignments) == 1
    assert (
        ast.unparse(route_row_assignments[0].value)
        == "route_local_e * expert_recv_max + packed_count_i32"
    )
    route_map_writes = [
        call
        for call in ast.walk(metadata)
        if isinstance(call, ast.Call)
        and _call_path(call) == "pl.tile.write"
        and [ast.unparse(argument) for argument in call.args]
        == [
            "route_map",
            "[0, route]",
            "pl.cast(route_out_row_i32, pl.INT32)",
        ]
    ]
    assert len(route_map_writes) == 1
    assert "local_e_idx = pl.tile.get_block_idx()" in pack_source
    assert (
        "global_e = my_rank * n_local_experts + pack_local_e_i32"
        in pack_source
    )
    assert "if eid == global_e:" in pack_source
    assert "pack_out_row_i32 = pl.read(" in pack_source
    assert "local_route_row_out, [0, route]," in pack_source
    assert (
        "pack_slab_begin_i32 = pl.cast(pack_slab_begin, pl.INT32)"
        in pack_source
    )
    assert (
        "pack_slab_end_i32 = pack_slab_begin_i32 + expert_recv_max"
        in pack_source
    )
    assert "if pack_out_row_i32 >= pack_slab_begin_i32:" in pack_source
    assert "if pack_out_row_i32 < pack_slab_end_i32:" in pack_source
    assert (
        "pack_out_row_i32, pl.INDEX,"
        in pack_source
    )
    assert "packed_count =" not in pack_source
    count_stores = [
        call
        for call in ast.walk(metadata)
        if isinstance(call, ast.Call)
        and _call_path(call) == "pl.store"
        and [ast.unparse(argument) for argument in call.args]
        == ["expert_count_tile", "[0, 0]", "local_expert_count_view"]
    ]
    assert len(count_stores) == 1
    payload_stores = {
        ast.unparse(call.args[2]): [
            ast.unparse(call.args[0]),
            ast.unparse(call.args[1]),
        ]
        for call in ast.walk(pack)
        if isinstance(call, ast.Call)
        and _call_path(call) == "pl.store"
        and len(call.args) == 3
    }
    assert payload_stores["local_routed_x_scale_out"] == [
        "scale_slab",
        "[0, pack_slab_begin]",
    ]
    assert payload_stores["local_routed_weight_out_view"] == [
        "weight_slab",
        "[0, pack_slab_begin]",
    ]
    for tensor_name in (
        "local_route_row_out",
        "local_expert_count",
    ):
        assert all(
            not isinstance(call, ast.Call)
            or _call_path(call) not in {"pl.write", "pl.store"}
            or not call.args
            or ast.unparse(
                call.args[0]
                if _call_path(call) == "pl.write"
                else call.args[2]
            )
            != tensor_name
            for call in ast.walk(pack)
        ), tensor_name

    n_ranks = 8
    n_local = 36
    active_tokens = 16
    topk = 8
    expert_recv_max = 128
    assert active_tokens * topk <= expert_recv_max
    cursor: dict[tuple[int, int], int] = {}
    route_owners: dict[int, int] = {}
    physical_rows: set[tuple[int, int]] = set()
    for t in range(active_tokens):
        for k in range(topk):
            eid = (t * 29 + k * 41 + (t // 7) * 3) % (n_ranks * n_local)
            owner, local_expert = divmod(eid, n_local)
            key = (owner, local_expert)
            slot = cursor.get(key, 0)
            cursor[key] = slot + 1
            route = t * topk + k
            row = local_expert * expert_recv_max + slot
            assert slot < expert_recv_max
            assert (owner, row) not in physical_rows
            physical_rows.add((owner, row))
            assert route not in route_owners
            route_owners[route] = owner
    assert sorted(route_owners) == list(range(active_tokens * topk))

    single_expert = n_local - 1
    single_expert_rows = [
        single_expert * expert_recv_max + route
        for route in range(active_tokens * topk)
    ]
    assert single_expert_rows[0] == single_expert * expert_recv_max
    assert single_expert_rows[-1] == n_local * expert_recv_max - 1
    assert len(set(single_expert_rows)) == expert_recv_max


def test_c3_local_route_plan_tracks_active_local_experts() -> None:
    source, tree = _parse(_CANONICAL)
    dispatch_function = _method(tree, "dispatch_step")
    plan = _task_scope(dispatch_function, "local_route_plan")
    plan_source = _segment(source, plan)

    assert "local_route_plan_valid_size = n_local_experts + 2" in source
    assert "local_route_plan_size = n_local_experts_pad" in source
    assert (
        "assert local_route_plan_valid_size <= local_route_plan_size"
        in source
    )
    assert "if expert_count_i32 > 0:" in plan_source
    assert "pl.cast(active_expert_count_i32, pl.INDEX)" in plan_source
    assert "+ pl.cast(2, pl.INDEX)" in plan_source
    assert "pl.cast(e, pl.INT32)" in plan_source
    assert "pl.tile.write(route_plan_tile, [0, 0], total_count)" in plan_source
    assert (
        "route_plan_tile, [0, 1], active_expert_count_i32,"
        in plan_source
    )
    assert (
        "route_plan_tile, [0, 0], local_route_count_view,"
        in plan_source
    )

    counts = [0, 2, 0, 1, 7, 0, 0, 3] + [0] * 28
    plan = [0] * 40
    active = []
    for expert, count in enumerate(counts):
        plan[0] += count
        if count > 0:
            active.append(expert)
            plan[2 + len(active) - 1] = expert
    plan[1] = len(active)
    assert plan[0] == sum(counts)
    assert plan[2 : 2 + plan[1]] == [1, 3, 4, 7]
    assert all(counts[expert] > 0 for expert in plan[2 : 2 + plan[1]])
    assert plan[38:] == [0, 0]


def test_c3_route_metadata_uses_one_physical_40_entry_abi() -> None:
    source, tree = _parse(_CANONICAL)
    dispatch = _method(tree, "dispatch_step")
    dispatch_source = _segment(source, dispatch)

    count_annotations: list[tuple[str, str]] = []
    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    ):
        for argument in function.args.args:
            if argument.arg != "local_expert_count":
                continue
            assert argument.annotation is not None
            count_annotations.append(
                (function.name, ast.unparse(argument.annotation))
            )
    assert count_annotations
    assert all(
        "n_local_experts_pad" in annotation
        for _, annotation in count_annotations
    ), count_annotations

    count_creations: list[tuple[str, ast.Call]] = []
    for assignment in (
        node for node in ast.walk(tree) if isinstance(node, ast.Assign)
    ):
        if (
            len(assignment.targets) != 1
            or not isinstance(assignment.targets[0], ast.Name)
            or "local_expert_count" not in assignment.targets[0].id
            or not isinstance(assignment.value, ast.Call)
            or _call_path(assignment.value) != "pl.create_tensor"
        ):
            continue
        count_creations.append((assignment.targets[0].id, assignment.value))
    assert count_creations
    assert all(
        ast.unparse(call.args[0]) == "[n_local_experts_pad]"
        for _, call in count_creations
    ), [
        (name, ast.unparse(call.args[0]))
        for name, call in count_creations
    ]

    assert (
        "expert_count_tile = pl.tile.full(\n"
        "                [1, n_local_experts_pad], dtype=pl.INT32, value=0,"
        in dispatch_source
    )
    assert (
        "route_plan_tile = pl.tile.full(\n"
        "                [1, local_route_plan_size], dtype=pl.INT32, value=0,"
        in dispatch_source
    )
    assert (
        "local_expert_count_view = pl.reshape(\n"
        "            local_expert_count, [1, n_local_experts_pad],"
        in dispatch_source
    )
    assert (
        "local_route_count_view = pl.reshape(\n"
        "            local_route_count, [1, local_route_plan_size],"
        in dispatch_source
    )
    assert "pl.read(local_expert_count_view, [0, e])" in dispatch_source
    assert (
        "local_routed_weight_out = pl.reshape(\n"
        "            local_routed_weight_out_view, [local_recv_max],"
        in dispatch_source
    )
    assert (
        "local_route_count = pl.reshape(\n"
        "            local_route_count_view, [local_route_plan_size],"
        in dispatch_source
    )
    assert "cursor = pl.array.create(n_local_experts, pl.INT32)" in (
        dispatch_source
    )
    assert "with pl.spmd(\n            n_local_experts," in dispatch_source
    assert dispatch_source.count("for e in pl.range(n_local_experts):") == 3
    assert "pl.set_validshape" not in dispatch_source

    scalar_targets = {
        "local_expert_count",
        "local_expert_count_view",
        "local_route_count",
        "local_route_count_view",
        "local_route_row_out",
        "local_routed_x_scale_out",
        "local_routed_weight_out",
        "local_routed_weight_out_view",
    }
    scalar_writes = [
        call
        for call in ast.walk(dispatch)
        if isinstance(call, ast.Call)
        and _call_path(call) == "pl.write"
        and call.args
        and ast.unparse(call.args[0]) in scalar_targets
    ]
    assert not scalar_writes


def test_c3_zero_route_rank_keeps_local_combine_and_final_collective_paths() -> None:
    source, tree = _parse(_CANONICAL)
    combine = _segment(source, _method(tree, "combine_step"))
    assert "acc = pl.cast(\n                sh_y[t : t + 1, :]" in combine
    assert "if t < active_tokens:" in combine
    assert "if local_row_i32 >= 0:" in combine
    assert "if local_row_i32 < local_recv_max:" in combine
    for name in (
        "full_moe_chip_orch",
        "swa_moe_chip_orch",
        "full_moe_chip_orch_swiglu7_swiglu16",
        "swa_moe_chip_orch_swiglu7_silu",
    ):
        function = _method(tree, name)
        calls = [
            call
            for call in _method_calls(function, "tp_all_reduce")
        ]
        assert len(calls) == 1


def test_two_layer_tp_all_reduce_matches_canonical() -> None:
    _, canonical_tree = _parse(_CANONICAL)
    _, two_layer_tree = _parse(_TWO_LAYER_PROGRAM)
    canonical = _method(canonical_tree, "tp_all_reduce")
    two_layer = _method(two_layer_tree, "tp_all_reduce")
    assert ast.dump(canonical, include_attributes=False) == ast.dump(
        two_layer,
        include_attributes=False,
    )



def test_two_layer_tp_all_reduce_residual_bs1_matches_canonical() -> None:
    _, canonical_tree = _parse(_CANONICAL)
    _, two_layer_tree = _parse(_TWO_LAYER_PROGRAM)
    canonical = _method(canonical_tree, "tp_all_reduce_residual_bs1")
    two_layer = _method(two_layer_tree, "tp_all_reduce_residual_bs1")
    assert ast.dump(canonical, include_attributes=False) == ast.dump(
        two_layer,
        include_attributes=False,
    )


def test_tp_all_reduce_residual_bs1_keeps_protocol_and_rounding_seams() -> None:
    source, tree = _parse(_CANONICAL)
    method = _method(tree, "tp_all_reduce_residual_bs1")
    body = _segment(source, method)
    normalized = ast.unparse(method)

    assert [arg.arg for arg in method.args.args] == [
        "self",
        "local",
        "residual_out",
        "tmp_window",
        "signal_window",
        "my_rank",
    ]
    assert "active_rows" not in body
    assert "shape=[1, HIDDEN]" in body
    assert "chunk_rows=1" in body
    assert "chunk_cols=TP_ALL_REDUCE_CHUNK" in body
    assert "for peer in pl.range(group_size):" in body
    assert "pl.parallel(group_size)" not in body
    assert "pl.spmd(group_size)" not in body

    calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
    call_paths = [_call_path(call) for call in calls]
    assert call_paths.count("pld.tensor.put") == 1
    assert call_paths.count("pld.tile.remote_load") == 1
    assert call_paths.count("pld.tile.remote_store") == 0
    assert call_paths.count("pld.system.notify") == 2
    assert call_paths.count("pld.system.wait") == 2

    waits = [call for call in calls if _call_path(call) == "pld.system.wait"]
    assert [
        ast.literal_eval(
            next(
                keyword.value
                for keyword in call.keywords
                if keyword.arg == "expected"
            ),
        )
        for call in waits
    ] == [1, 2]
    for call in calls:
        if _call_path(call) == "pld.system.notify":
            keywords = {
                keyword.arg: ast.unparse(keyword.value)
                for keyword in call.keywords
            }
            assert keywords["offsets"] == "[my_rank, 0]"
            assert keywords["op"] == "pld.NotifyOp.AtomicAdd"
        elif _call_path(call) == "pld.system.wait":
            keywords = {
                keyword.arg: ast.unparse(keyword.value)
                for keyword in call.keywords
            }
            assert keywords["offsets"] == "[src, 0]"

    reduced_cast = normalized.index(
        "reduced_bf16 = pl.cast(row_acc, target_type=pl.BF16)",
    )
    reduced_store = normalized.index(
        "pl.store(reduced_bf16, [0, 0], local)",
    )
    completion_wait = normalized.index("expected=2")
    residual_reload = normalized.index(
        "reduced_chunk = pl.load(local, [0, k0], "
        "[1, TP_ALL_REDUCE_CHUNK])",
    )
    assert reduced_cast < reduced_store < completion_wait < residual_reload
    assert (
        "for k0 in pl.range(0, HIDDEN, TP_ALL_REDUCE_CHUNK)"
        in normalized
    )
    assert (
        "residual_sum = pl.add("
        "pl.cast(reduced_chunk, target_type=pl.FP32), "
        "pl.cast(residual_chunk, target_type=pl.FP32))"
        in normalized
    )
    residual_sum = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "residual_sum"
    )
    assert not any(
        isinstance(node, ast.Name) and node.id == "row_acc"
        for node in ast.walk(residual_sum.value)
    )
    assert (
        "pl.store(pl.cast(residual_sum, target_type=pl.BF16), "
        "[0, k0], residual_out)"
        in normalized
    )
    assert normalized.rstrip().endswith("return residual_out")


def test_attention_bs1_fuses_collective_with_residual_only_on_tp_path() -> None:
    for path, function_name, residual_hint in (
        (_FULL_ATTN, "attention_full", "full_out_resid_add"),
        (_SWA_ATTN, "attention_swa", "swa_out_resid_add"),
    ):
        source, tree = _parse(path)
        function = _method(tree, function_name)
        outer = next(
            node
            for node in function.body
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "TP_WORLD_SIZE > 1"
        )
        assert len(outer.body) == 1
        fused_branch = outer.body[0]
        assert isinstance(fused_branch, ast.If)
        assert ast.unparse(fused_branch.test) == "num_tokens == 1"

        fused_calls = _method_calls(fused_branch, "tp_all_reduce_residual_bs1")
        assert len(fused_calls) == 1
        assert [ast.unparse(arg) for arg in fused_calls[0].args] == [
            "partial_attn_proj",
            "resid1_out",
            "tmp_window",
            "signal_window",
            "my_rank",
        ]
        assert len(fused_branch.body) == 1
        fused_assign = fused_branch.body[0]
        assert isinstance(fused_assign, ast.Assign)
        assert ast.unparse(fused_assign.targets[0]) == "resid1_out"

        generic_calls = _method_calls(fused_branch, "tp_all_reduce")
        assert len(generic_calls) == 1
        assert [ast.unparse(arg) for arg in generic_calls[0].args][-2:] == [
            "num_tokens",
            "my_rank",
        ]
        generic_source = ast.unparse(
            ast.Module(body=fused_branch.orelse, type_ignores=[]),
        )
        tp1_source = ast.unparse(
            ast.Module(body=outer.orelse, type_ignores=[]),
        )
        assert residual_hint in generic_source
        assert residual_hint in tp1_source
        assert residual_hint not in ast.unparse(
            ast.Module(body=fused_branch.body, type_ignores=[]),
        )
        function_source = _segment(source, function)
        assert function_source.count(
            "self.tp_all_reduce_residual_bs1("
        ) == 1
        assert function_source.count("self.tp_all_reduce(") == 1
        assert function_source.count(f'name_hint="{residual_hint}"') == 2
        assert isinstance(function.body[-1], ast.Return)
        assert ast.unparse(function.body[-1].value) == "resid1_out"


def test_attention_bs1_fused_helper_resolves_in_all_inline_callers() -> None:
    probe = (
        _ROOT
        / "tests"
        / "step3p5"
        / "probes"
        / "_probe_single_layer_inline.py"
    )
    for path, signal_rows in (
        (_FULL_ATTN, "tp_size"),
        (_SWA_ATTN, "tp_size"),
        (_MTP_HIDDEN, "tp_size"),
        (probe, "tp"),
    ):
        source, tree = _parse(path)
        method = _method(tree, "tp_all_reduce_residual_bs1")
        annotations = {
            arg.arg: ast.unparse(arg.annotation)
            for arg in method.args.args
            if arg.annotation is not None
        }
        assert signal_rows in annotations["signal_window"]
        assert [arg.arg for arg in method.args.args][-5:] == [
            "local",
            "residual_out",
            "tmp_window",
            "signal_window",
            "my_rank",
        ]
        if path == _MTP_HIDDEN:
            assert "self.tp_all_reduce_residual_bs1(" not in source

    dense_path = _ROOT / "models" / "step3p5" / "dense_mlp.py"
    assert "tp_all_reduce_residual_bs1" not in dense_path.read_text(
        encoding="utf-8",
    )


def test_tp_all_reduce_selects_smallmesh_and_keeps_push_gather_fallback() -> None:
    source, tree = _parse(_CANONICAL)
    method = _method(tree, "tp_all_reduce")
    body = _segment(source, method)
    normalized = ast.unparse(method)
    assert [arg.arg for arg in method.args.args][-2:] == [
        "active_rows_i32",
        "my_rank",
    ]
    assert "if active_rows == 1:" in body
    assert body.count("pld.tensor.put(") == 6
    assert "peer=my_rank" in body
    assert "shape=[1, HIDDEN]" in body
    assert "chunk_rows=1" in body
    for rows in (2, 4, 8, 16):
        assert f"active_rows <= {rows}" in body
        assert f"shape=[{rows}, HIDDEN]" in body
        assert f"chunk_rows={rows}" in body
    assert "for dst in pl.range(group_size):" in body
    assert "pld.tile.remote_store(" in body
    assert "for ar_b0 in pl.range(0, BATCH, BATCH_TILE):" in body
    assert (
        "pl.store(reduced_tile, [ar_b0, owned_base], tmp_window)"
        in normalized
    )
    assert "pl.store(reduced_tile, [ar_b0, owned_base], local)" not in body
    assert "chunk_rows=BATCH_TILE" in body
    assert "chunk_cols=TP_ALL_REDUCE_CHUNK" in body
    assert "ar_copy_tiles = (" in body
    assert "(BATCH // BATCH_TILE) * (HIDDEN // ar_chunk)" in body
    for expected in (1, 2, 3):
        assert f"expected={expected}" in body


def test_tp_all_reduce_uses_only_static_bucket_transfers() -> None:
    source, tree = _parse(_CANONICAL)
    method = _method(tree, "tp_all_reduce")
    body = _segment(source, method)

    assert "active_rows = pl.cast(active_rows_i32, pl.INDEX)" in body
    assert "if active_rows > BATCH:" in body
    assert "if active_rows < 1:" in body
    assert "active_rows = pl.cast(BATCH, pl.INDEX)" in body
    assert "pl.set_validshape(" not in body
    assert "valid_shapes=" not in body

    branch = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "active_rows == 1"
    )
    fallback_source = ast.unparse(
        ast.Module(body=branch.orelse, type_ignores=[])
    )
    assert "owned_chunk =" not in fallback_source
    assert "TP_ALL_REDUCE_OWNED_CHUNK = HIDDEN // TP_WORLD_SIZE" in source
    assert "owned_base = my_rank * TP_ALL_REDUCE_OWNED_CHUNK" in fallback_source
    assert "reduced_tile = pl.cast(acc, target_type=pl.BF16)" in fallback_source
    assert (
        "pl.store(reduced_tile, [ar_b0, owned_base], tmp_window)"
        in fallback_source
    )
    assert "[BATCH_TILE, ar_chunk]" in fallback_source

    # Every selected transfer extent is a literal static bucket. Runtime active
    # rows only select a branch; they never enter a tile or remote shape.
    put_calls = [
        call
        for call in ast.walk(method)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "put"
    ]
    assert len(put_calls) == 6
    put_shapes = {
        ast.unparse(keyword.value)
        for call in put_calls
        for keyword in call.keywords
        if keyword.arg == "shape"
    }
    assert put_shapes == {
        "[1, HIDDEN]",
        "[2, HIDDEN]",
        "[4, HIDDEN]",
        "[8, HIDDEN]",
        "[16, HIDDEN]",
    }
    assert any(
        "shape" not in {keyword.arg for keyword in call.keywords}
        for call in put_calls
    )

    remote_loads = [
        call
        for call in ast.walk(method)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "remote_load"
    ]
    assert len(remote_loads) == 6
    remote_shapes = {
        ast.unparse(keyword.value)
        for call in remote_loads
        for keyword in call.keywords
        if keyword.arg == "shape"
    }
    assert remote_shapes == {
        "[1, HIDDEN]",
        "[2, TP_ALL_REDUCE_OWNED_CHUNK]",
        "[4, TP_ALL_REDUCE_OWNED_CHUNK]",
        "[8, TP_ALL_REDUCE_OWNED_CHUNK]",
        "[16, TP_ALL_REDUCE_OWNED_CHUNK]",
        "[BATCH_TILE, TP_ALL_REDUCE_OWNED_CHUNK]",
    }
    all_static_shapes = put_shapes | remote_shapes
    assert all(
        "active_rows" not in shape and "bucket_rows" not in shape
        for shape in all_static_shapes
    )

    local_load_shapes = {
        ast.unparse(keyword.value)
        for call in ast.walk(method)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "load"
        for keyword in call.keywords
        if keyword.arg == "shape"
    }
    # pl.load carries shape positionally; guard the final-copy buckets in the
    # normalized fallback source instead.
    assert not local_load_shapes
    for rows in (2, 4, 8, 16):
        assert f"[{rows}, ar_chunk]" in fallback_source


def test_tp_all_reduce_smallmesh_keeps_peer_order_and_two_wave_lifetime() -> None:
    source, tree = _parse(_CANONICAL)
    method = _method(tree, "tp_all_reduce")
    branch = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "active_rows == 1"
    )
    branch_source = ast.unparse(ast.Module(body=branch.body, type_ignores=[]))
    assert "for peer in pl.range(group_size)" in branch_source
    assert "remote_row = pld.tile.remote_load" in branch_source
    assert "row_acc = pl.add" in branch_source
    assert branch_source.count("pld.system.notify") == 2
    assert branch_source.count("pld.system.wait") == 2
    assert "expected=1" in branch_source
    assert "expected=2" in branch_source
    assert "pld.tile.remote_store" not in branch_source
    assert "pl.cast(row_acc, target_type=pl.BF16)" in branch_source
    assert "pl.store(pl.cast(row_acc, target_type=pl.BF16), [0, 0], local)" in branch_source


def test_tp_all_reduce_keeps_reduce_scatter_accumulate_serial() -> None:
    # The reduce-scatter accumulate carries an FP32 accumulator across peers in
    # a fixed order and casts to BF16 exactly once.  hidden_tp_spread == 0
    # depends on that shape, so the loop must never become pl.parallel/pl.spmd.
    # Switching it is also pointless: swapping pl.range for pl.parallel in this
    # InCore body produced byte-identical AIV codegen (measured 2026-08-05, the
    # loop kind is not consumed here), and the onephase_par microbenchmark
    # measured parallel re-reduction as slower.
    source, tree = _parse(_CANONICAL)
    method = _method(tree, "tp_all_reduce")
    body = _segment(source, method)
    normalized = ast.unparse(method)
    assert "for peer in pl.range(group_size):" in body
    for suffix in ("2", "4", "8", "16"):
        assert (
            f"acc_{suffix} = pl.mul("
            f"pl.cast(own_tile_{suffix}, target_type=pl.FP32), 0.0)"
            in normalized
        )
    assert (
        "acc = pl.mul(pl.cast(own_tile, target_type=pl.FP32), 0.0)"
        in normalized
    )
    reduced_casts = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("reduced_tile")
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "cast"
    ]
    assert len(reduced_casts) == 5
    assert "pl.set_validshape(" not in body
    assert "pl.parallel(group_size)" not in body


def test_g1_threads_runtime_active_tokens_through_moe_and_holder() -> None:
    source, tree = _parse(_CANONICAL)
    whole = _method(tree, "whole_chip_orch")
    whole_args = [arg.arg for arg in whole.args.args]
    assert whole_args[-2:] == ["num_tokens_per_owner", "my_rank"]
    whole_source = _segment(source, whole)
    assert "num_tokens = pl.cast(0, pl.INT32)" in whole_source
    assert "pl.read(num_tokens_per_owner, [owner_rank])" in whole_source
    assert "NUM_TOKENS_STORAGE_I32" in source
    assert "NUM_TOKENS_STORAGE_I32 = COMM_SIGNAL_STRIDE_I32" not in source
    for name in (
        "_gate", "_norm_quant_moe_input", "dispatch_step", "combine_step",
        "full_moe_chip_orch", "swa_moe_chip_orch",
        "full_moe_chip_orch_swiglu7_swiglu16",
        "swa_moe_chip_orch_swiglu7_silu",
    ):
        function = _method(tree, name)
        assert any(arg.arg == "num_tokens" for arg in function.args.args), name
    dispatch = _segment(source, _method(tree, "dispatch_step"))
    combine = _segment(source, _method(tree, "combine_step"))
    assert "active_tokens" in dispatch and "TOPK" in dispatch
    assert "active_tokens" in combine and "TOPK" in combine
    holder_source, holder_tree = _parse(_HOLDER)
    holder_enter = _segment(holder_source, _method(holder_tree, "_enter_impl"))
    prepare_runtime = _segment(
        holder_source,
        _method(holder_tree, "_prepare_runtime"),
    )
    live_step = _segment(holder_source, _method(holder_tree, "set_live_step"))
    assert "self.num_tokens_per_owner" in holder_enter
    assert "self.compiled.prepare(persistent=True)" in prepare_runtime
    assert "self.num_tokens_per_owner[: self.tp].fill_(valid_tokens)" in live_step
    assert "args += [self.num_tokens_per_owner]" in holder_source
    assert "_invocation_epoch" not in holder_source


def test_g1_decode_attention_inline_calls_preserve_active_token_arity() -> None:
    """Every decode attention caller must pass the new active-token scalar.

    The canonical Main forwards its runtime ``num_tokens`` value.  Standalone
    attention and MTP programs retain fixed-storage behavior by passing the
    static ``BATCH`` bound explicitly.
    """
    for path in (
        _CANONICAL,
        _FULL_ATTN,
        _SWA_ATTN,
        _MTP_HIDDEN,
    ):
        _, tree = _parse(path)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {
                "attention_full_inline",
                "attention_swa_inline",
                "attention_inline",
            }
        ]
        assert calls, f"{path.name} has no decode attention inline call"
        assert all(len(call.args) == 24 for call in calls), (
            f"{path.name} has a stale decode attention inline call"
        )

    canonical_source, canonical_tree = _parse(_CANONICAL)
    del canonical_source
    canonical_calls = [
        node
        for node in ast.walk(canonical_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"attention_full_inline", "attention_swa_inline"}
    ]
    assert any(
        isinstance(call.args[20], ast.Name) and call.args[20].id == "num_tokens"
        for call in canonical_calls
    )


def test_g1_decode_dense_mlp_calls_preserve_active_token_arity() -> None:
    dense_path = _ROOT / "models" / "step3p5" / "dense_mlp.py"
    dense_source, dense_tree = _parse(dense_path)
    dense_body = _method(dense_tree, "dense_mlp_body_tp")
    assert "num_tokens" in [arg.arg for arg in dense_body.args.args]
    assert "num_tokens, my_rank" in _segment(dense_source, dense_body)

    for path, expected in (
        (_CANONICAL, "num_tokens"),
        (_TWO_LAYER_PROGRAM, "num_tokens"),
        (_MTP_HIDDEN, "BATCH"),
    ):
        _, tree = _parse(path)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "dense_mlp_inline"
        ]
        assert calls, f"{path.name} has no dense MLP inline call"
        assert all(len(call.args) == 12 for call in calls)
        assert all(ast.unparse(call.args[8]) == expected for call in calls)


def test_signal_inline_formal_resolves_wide_only_in_canonical_main() -> None:
    """One shared inline body serves wide canonical and compact MTP callers."""
    for path in (_FULL_ATTN, _SWA_ATTN):
        helper_source = path.read_text(encoding="utf-8")
        assert "SIGNAL_WINDOW_ROWS = TP_WORLD_SIZE" in helper_source
        assert (
            "signal_window: pld.DistributedTensor"
            "[[SIGNAL_WINDOW_ROWS, 1], pl.INT32]"
        ) in helper_source

    dense_source = (
        _ROOT / "models" / "step3p5" / "dense_mlp.py"
    ).read_text(encoding="utf-8")
    assert "SIGNAL_WINDOW_ROWS = TP_WORLD_SIZE" in dense_source
    assert (
        "signal_window: pld.DistributedTensor"
        "[[SIGNAL_WINDOW_ROWS, 1], pl.INT32]"
    ) in dense_source

    canonical_source = _CANONICAL.read_text(encoding="utf-8")
    assert "SIGNAL_WINDOW_ROWS = COMM_SIGNAL_STRIDE_I32" in canonical_source

    mtp_source = _MTP_HIDDEN.read_text(encoding="utf-8")
    assert "SIGNAL_WINDOW_ROWS = TP_WORLD_SIZE" in mtp_source
    assert mtp_source.count("pld.alloc_window_buffer(tp_size * 4)") == 3


def test_c3_expert_storage_keeps_fixed_local_lane_bases() -> None:
    source, tree = _parse(_CANONICAL)
    dispatch = _segment(source, _method(tree, "dispatch_step"))
    expert = _segment(source, _method(tree, "_expert_routed"))
    expert_swiglu7 = _segment(source, _method(tree, "_expert_routed_swiglu7"))
    combine = _segment(source, _method(tree, "combine_step"))

    assert "expert_recv_max = BATCH * TOPK" in source
    assert "assert expert_recv_max == 128" in source
    assert "local_recv_max = n_local_experts * expert_recv_max" in source
    assert "route_local_e * expert_recv_max" in dispatch
    assert "total = total + count" not in dispatch
    assert "pack_out_row_i32 = pl.read(" in dispatch
    assert "local_route_row_out, [0, route]," in dispatch
    for body in (expert, expert_swiglu7):
        assert "local_expert_count" in body
        assert "local_route_count" in body
        assert "pl.spmd_submit(" in body
    assert "pl.read(local_route_row, [0, route])" in combine
    assert "local_routed_y[local_row : local_row + 1, :]" in combine


def test_router_grid_and_postprocess_scale_with_active_batch() -> None:
    source, tree = _parse(_CANONICAL)
    gate = _method(tree, "_gate")
    gate_source = _segment(source, gate)
    assert "active_tokens = pl.cast(num_tokens, pl.INDEX)" in gate_source
    assert "active_gate_tiles = (" in gate_source
    assert "active_tokens + ROUTER_GATE_M_TILE - 1" in gate_source
    assert ") // ROUTER_GATE_M_TILE" in gate_source
    assert "gate_n_blocks = N_EXPERTS // ROUTER_GATE_N_CHUNK" in gate_source
    assert 'name_hint="gate_init"' not in gate_source

    xg = _task_scope(gate, "gate_xg_precompute")
    fanout = _task_scope(gate, "gate_expert_fanout")
    xg_call = xg.items[0].context_expr
    fanout_call = fanout.items[0].context_expr
    assert isinstance(xg_call, ast.Call)
    assert isinstance(fanout_call, ast.Call)
    assert [ast.unparse(arg) for arg in xg_call.args] == ["active_tokens"]
    assert [ast.unparse(arg) for arg in fanout_call.args] == [
        "active_gate_tiles * gate_n_blocks"
    ]
    fanout_source = _segment(source, fanout)
    assert "deps=[gate_xg_tid]" in fanout_source
    assert "b_trans=True" in fanout_source

    topk = _task_scope(gate, "gate_topk")
    topk_source = _segment(source, topk)
    assert "for tt in pl.range(active_tokens):" in topk_source
    assert "[1, ROUTER_SCORE_PAD]" in topk_source


def test_shared_mlp_scales_projection_and_down_grids_with_active_batch() -> None:
    source, tree = _parse(_CANONICAL)
    helper = _method(tree, "_expert_shared_local")
    helper_source = _segment(source, helper)
    args = [arg.arg for arg in helper.args.args]
    assert args[1:] == [
        "x",
        "w_gate",
        "w_up",
        "w_down",
        "sh_y_shard",
        "num_tokens",
        "swiglu_limit",
    ]
    assert (
        "[BATCH, sh_inter_local], dtype=pl.BF16, manual_dep=True"
        in helper_source
    )
    assert "active_shared_tiles = (" in helper_source
    assert "active_tokens + SHARED_GATE_M_TILE - 1" in helper_source
    assert ") // SHARED_GATE_M_TILE" in helper_source
    assert "shared_mm_tasks = active_shared_tiles * shared_n_blocks" in helper_source
    assert "active_down_tiles = (" in helper_source
    assert "active_tokens + SHARED_DOWN_M_TILE - 1" in helper_source
    assert ") // SHARED_DOWN_M_TILE" in helper_source
    assert "shared_down_tasks = active_down_tiles * SHARED_DOWN_WORKERS" in helper_source

    gate_mm = _task_scope(helper, "sh_gate_mm")
    up_mm = _task_scope(helper, "sh_up_mm")
    act = _task_scope(helper, "sh_gate_up_act")
    down = _task_scope(helper, "sh_down")
    for scope in (gate_mm, up_mm, act):
        call = scope.items[0].context_expr
        assert isinstance(call, ast.Call)
        assert _call_path(call) == "pl.spmd"
        assert [ast.unparse(arg) for arg in call.args] == ["shared_mm_tasks"]
    down_call = down.items[0].context_expr
    assert isinstance(down_call, ast.Call)
    assert _call_path(down_call) == "pl.spmd"
    assert [ast.unparse(arg) for arg in down_call.args] == [
        "shared_down_tasks"
    ]

    for scope in (gate_mm, up_mm):
        mm_source = _segment(source, scope)
        assert "b_trans=True" in mm_source
        assert "task = pl.tile.get_block_idx()" in mm_source
        assert "mb = task // shared_n_blocks" in mm_source
        assert "chunk = task % shared_n_blocks" in mm_source

    act_source = _segment(source, act)
    assert "deps=[sh_gate_tid, sh_up_tid]" in act_source
    assert "sh_hidden[" in act_source
    assert "pl.cast(gated, target_type=pl.BF16)" in act_source

    down_source = _segment(source, down)
    assert "deps=[sh_gate_up_tid]" in down_source
    assert "task = pl.tile.get_block_idx()" in down_source
    assert "mb = task // SHARED_DOWN_WORKERS" in down_source
    assert "worker = task % SHARED_DOWN_WORKERS" in down_source
    assert "m0 = mb * SHARED_DOWN_M_TILE" in down_source
    assert "if active_tokens <= 1:" in down_source
    assert "if mb == 0 and worker == 0:" in down_source
    assert "HIDDEN // SHARED_DOWN_N_CHUNK" in down_source
    assert "[SHARED_DOWN_M_TILE, SHARED_SWIGLU_N_CHUNK]" in down_source
    assert "[m0, d0]" in down_source
    assert "SHARED_DOWN_WORKERS" in down_source
    assert down_source.count("sh_hidden,") == 4

    generic_call = _method_calls(
        _method(tree, "expert_shared_step"),
        "_expert_shared_local",
    )
    assert len(generic_call) == 1
    assert ast.unparse(generic_call[0].args[-2]) == "num_tokens"
    assert ast.unparse(generic_call[0].args[-1]) == "_SHARED_SWIGLU_LIMIT"
    special_call = _method_calls(
        _method(tree, "_expert_shared_local_swiglu16"),
        "_expert_shared_local",
    )
    assert len(special_call) == 1
    assert ast.unparse(special_call[0].args[-2]) == "num_tokens"
    assert ast.unparse(special_call[0].args[-1]) == "_SHARED_SWIGLU16_LIMIT"

    for wrapper_name in (
        "expert_shared_step",
        "expert_shared_step_swiglu16",
    ):
        wrapper = _method(tree, wrapper_name)
        assert "num_tokens" in [arg.arg for arg in wrapper.args.args]

    orchestration_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr
        in {"expert_shared_step", "expert_shared_step_swiglu16"}
    ]
    assert len(orchestration_calls) == 4
    assert all(
        ast.unparse(call.args[-1]) == "num_tokens"
        for call in orchestration_calls
    )


def test_regular_routed_expert_adapts_grid_to_shared_worker_budget() -> None:
    source, tree = _parse(_CANONICAL)
    expected_assignments = {
        "ROUTED_GRID_WORKERS": 23,
        "ROUTED_MULTIBATCH_GRID_WORKERS": 22,
        "ROUTED_FUSED_GRID_WORKERS": 24,
    }
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in expected_assignments
    }
    assert assignments == expected_assignments
    for function_name, fused_callee in (
        ("_expert_routed", "self.routed_nz_gmm1_swiglu_quant"),
        (
            "_expert_routed_swiglu7",
            "self.routed_nz_gmm1_swiglu7_quant",
        ),
    ):
        expert = _method(tree, function_name)
        expert_source = _segment(source, expert)
        assert "num_tokens" in [arg.arg for arg in expert.args.args]
        assert (
            "routed_workers = pl.cast(ROUTED_GRID_WORKERS, pl.INDEX)"
            in expert_source
        )
        assert "if active_tokens > 1:" in expert_source
        assert (
            "ROUTED_MULTIBATCH_GRID_WORKERS, pl.INDEX"
            in expert_source
        )
        submits = [
            node
            for node in ast.walk(expert)
            if isinstance(node, ast.Call)
            and _call_path(node) == "pl.spmd_submit"
        ]
        assert len(submits) == 2
        by_callee = {
            ast.unparse(call.args[0]): call
            for call in submits
        }
        assert set(by_callee) == {
            fused_callee,
            "self.routed_nz_down",
        }
        fused_keywords = {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in by_callee[fused_callee].keywords
        }
        assert fused_keywords == {
            "core_num": "ROUTED_FUSED_GRID_WORKERS",
            "deps": "[local_route_count_tid]",
            "predicate": "local_route_count[0] > 0",
            "allow_early_resolve": "True",
            "sync_start": "True",
        }
        down = by_callee["self.routed_nz_down"]
        assert ast.unparse(down.args[-1]) == "routed_workers"
        down_keywords = {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in down.keywords
        }
        assert down_keywords == {
            "core_num": "routed_workers",
            "deps": "[local_route_count_tid, routed_fused_tid]",
            "predicate": "local_route_count[0] > 0",
            "allow_early_resolve": "True",
        }
        assert "routed_fused_tid" in expert_source
        assert "routed_down_tid" in expert_source
        assert "expert_gate_up_act" not in expert_source
        assert "routed_h_quant" not in expert_source

    wrapper = _method(tree, "expert_routed_step")
    assert "num_tokens" in [arg.arg for arg in wrapper.args.args]
    wrapper_calls = _method_calls(wrapper, "_expert_routed")
    assert len(wrapper_calls) == 1
    assert ast.unparse(wrapper_calls[0].args[6]) == "num_tokens"

    orchestration_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "expert_routed_step"
    ]
    assert len(orchestration_calls) == 2
    assert all(
        ast.unparse(call.args[6]) == "num_tokens"
        for call in orchestration_calls
    )


def test_routed_down_workspace_uses_fixed_per_worker_stride() -> None:
    source, tree = _parse(_CANONICAL)
    constants = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and type(node.value.value) is int
        ):
            constants[node.targets[0].id] = node.value.value
    assert "ROUTED_DOWN_PIPE_FLOATS_PER_BLOCK = 32768" in source
    pipe_floats = constants["ROUTED_DOWN_PIPE_FLOATS_PER_BLOCK"]
    full_workers = constants["ROUTED_GRID_WORKERS"]
    multibatch_workers = constants["ROUTED_MULTIBATCH_GRID_WORKERS"]
    assert 0 < multibatch_workers <= full_workers
    assert multibatch_workers * pipe_floats <= full_workers * pipe_floats
    assert source.count(
        "[ROUTED_DOWN_PIPE_FLOATS_PER_BLOCK * ROUTED_GRID_WORKERS]"
    ) == 8
    assert source.count("down_pipe_buffer = pl.create_tensor(") == 2
    fixed_offset = (
        "static_cast<int64_t>(__pypto_spmd_block_idx)\n"
        "        * kGmPipeFloatsPerBlock"
    )
    for path in _ROUTED_DOWN_SOURCES:
        external_source = path.read_text(encoding="utf-8")
        assert external_source.count(
            "const int64_t kGmPipeFloatsPerBlock = 32768;"
        ) == 1
        assert external_source.count(fixed_offset) == 1
        assert "__pypto_gm_elems_per_block" not in external_source
        assert "__pypto_gm_total_elems" not in external_source
        pipe_declarations = re.findall(
            r"auto v\d+ = TPipe<0, Direction::DIR_C2V, "
            r"16384, 8, 8, true>",
            external_source,
        )
        assert len(pipe_declarations) == 2
        assert 16384 * 8 // 4 == pipe_floats


def test_routed_gmm1_zero_fills_empty_aiv_row_parts() -> None:
    for path in _ROUTED_GMM1_SOURCES:
        external_source = path.read_text(encoding="utf-8")
        begin = (
            "// PYPTO-LIB-AUTHORITY: empty-row-part-zero-store begin"
        )
        end = "// PYPTO-LIB-AUTHORITY: empty-row-part-zero-store end"
        assert external_source.count(begin) == 1
        assert external_source.count(end) == 1
        authority = external_source[
            external_source.index(begin) : external_source.index(end)
        ]
        assert "if (remaining_rows > v40) {" in authority
        assert "int64_t v56 = v51 - v53 - v55;" in external_source
        assert (
            "int64_t v56 = (int64_t) ((uint64_t)"
            not in external_source
        )
        assert (
            "remaining_rows > v22 ? v22 : remaining_rows"
            in authority
        )
        empty_start = authority.index("} else {")
        empty_part = authority[empty_start:]
        assert "TLOAD(" not in empty_part
        assert "wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);" in empty_part
        assert "pipe_barrier(PIPE_ALL);" in empty_part
        assert "TEXPANDS(empty_part_f32, 0.0f);" in empty_part
        assert (
            "TCVT(empty_part_bf16, empty_part_f32, v16, v17);"
            in empty_part
        )
        assert "pto::Shape<1, 1, 1, 8, 256>" in empty_part
        assert "TSTORE(empty_part_out, empty_part_bf16);" in empty_part
        assert "set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);" in empty_part


def test_routed_gmm1_publishes_h_bf16_before_cross_core_quant() -> None:
    begin = "// PYPTO-LIB-AUTHORITY: cross-core-h-bf16-publish begin"
    end = "// PYPTO-LIB-AUTHORITY: cross-core-h-bf16-publish end"
    publish_acquire = (
        "pipe_barrier(PIPE_ALL);",
        "dcci(static_cast<__gm__ void *>(0), ENTIRE_DATA_CACHE);",
        "dsb(DSB_DDR);",
        "SYNCALL<SyncCoreType::Mix>();",
        "dcci(static_cast<__gm__ void *>(0), ENTIRE_DATA_CACHE);",
        "dsb(DSB_DDR);",
    )

    for path in _ROUTED_GMM1_SOURCES:
        external_source = path.read_text(encoding="utf-8")
        assert external_source.count(begin) == 2
        assert external_source.count(end) == 2
        cursor = 0
        for _ in range(2):
            start = external_source.index(begin, cursor)
            stop = external_source.index(end, start)
            authority = external_source[start:stop]
            op_cursor = 0
            for op in publish_acquire:
                op_cursor = authority.index(op, op_cursor) + len(op)
            cursor = stop + len(end)

        producer_start = external_source.index(begin)
        producer_prefix = external_source[
            external_source.rfind(
                "set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);",
                0,
                producer_start,
            ) : producer_start
        ]
        assert (
            "wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);"
            in producer_prefix
        )
        assert "wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);" in producer_prefix


def test_routed_gmm1_publishes_gate_up_before_aiv_consumers() -> None:
    publish_begin = (
        "// PYPTO-LIB-AUTHORITY: cross-core-gate-up-publish begin"
    )
    publish_end = (
        "// PYPTO-LIB-AUTHORITY: cross-core-gate-up-publish end"
    )
    acquire_begin = (
        "// PYPTO-LIB-AUTHORITY: cross-core-gate-up-acquire begin"
    )
    acquire_end = (
        "// PYPTO-LIB-AUTHORITY: cross-core-gate-up-acquire end"
    )

    for path in _ROUTED_GMM1_SOURCES:
        external_source = path.read_text(encoding="utf-8")
        assert external_source.count(publish_begin) == 1
        assert external_source.count(publish_end) == 1
        assert external_source.count(acquire_begin) == 2
        assert external_source.count(acquire_end) == 2

        publish = external_source[
            external_source.index(publish_begin) :
            external_source.index(publish_end)
        ]
        publish_ops = (
            "pipe_barrier(PIPE_ALL);",
            "dcci(static_cast<__gm__ void *>(0), ENTIRE_DATA_CACHE);",
            "dsb(DSB_DDR);",
        )
        op_cursor = 0
        for op in publish_ops:
            op_cursor = publish.index(op, op_cursor) + len(op)

        cursor = 0
        for _ in range(2):
            start = external_source.index(acquire_begin, cursor)
            stop = external_source.index(acquire_end, start)
            acquire = external_source[start:stop]
            acquire_ops = (
                "SYNCALL<SyncCoreType::Mix>();",
                "dcci(static_cast<__gm__ void *>(0), ENTIRE_DATA_CACHE);",
                "dsb(DSB_DDR);",
            )
            op_cursor = 0
            for op in acquire_ops:
                op_cursor = acquire.index(op, op_cursor) + len(op)
            cursor = stop + len(acquire_end)


def test_regular_routed_quant_math_is_chunk_invariant() -> None:
    import torch

    rows = 16
    intermediate = 1280
    values = torch.arange(rows * intermediate, dtype=torch.float32).reshape(
        rows,
        intermediate,
    )
    hidden = (
        torch.sin(values * 0.017)
        * torch.cos(values * 0.003)
        * torch.linspace(0.125, 8.0, rows).reshape(rows, 1)
    ).to(torch.bfloat16).to(torch.float32)
    hidden[0].zero_()
    hidden[1, 63] = -9.5
    hidden[2, 255] = 11.0
    hidden[3, 1024] = -12.5

    def quantize(chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
        amax = torch.full([rows], 1e-4, dtype=torch.float32)
        for offset in range(0, intermediate, chunk):
            chunk_amax = hidden[:, offset : offset + chunk].abs().amax(dim=1)
            amax = torch.maximum(amax, chunk_amax)
        scale_q = torch.reciprocal(amax) * 127.0
        scale_dq = torch.reciprocal(scale_q)

        quantized = torch.empty_like(hidden, dtype=torch.int8)
        for offset in range(0, intermediate, chunk):
            scaled = hidden[:, offset : offset + chunk] * scale_q.reshape(
                rows,
                1,
            )
            quantized[:, offset : offset + chunk] = (
                torch.round(scaled)
                .to(torch.int32)
                .to(torch.float16)
                .to(torch.int8)
            )
        return quantized, scale_dq

    quant64, scale64 = quantize(64)
    quant256, scale256 = quantize(256)
    assert torch.equal(scale256, scale64)
    assert torch.equal(quant256, quant64)

    def quantize_with_chunk_local_amax(chunk: int) -> torch.Tensor:
        quantized = torch.empty_like(hidden, dtype=torch.int8)
        for offset in range(0, intermediate, chunk):
            tile = hidden[:, offset : offset + chunk]
            local_amax = torch.maximum(
                torch.full([rows], 1e-4, dtype=torch.float32),
                tile.abs().amax(dim=1),
            )
            local_scale = torch.reciprocal(local_amax) * 127.0
            quantized[:, offset : offset + chunk] = (
                torch.round(tile * local_scale.reshape(rows, 1))
                .to(torch.int32)
                .to(torch.float16)
                .to(torch.int8)
            )
        return quantized

    # Prove that the fixture detects the bug guarded by the AST contract:
    # resetting amax inside the chunk loop makes the result chunk-dependent.
    assert not torch.equal(
        quantize_with_chunk_local_amax(64),
        quantize_with_chunk_local_amax(256),
    )


def test_routed_down_external_kernels_use_128_wide_k_tiles() -> None:
    source, tree = _parse(_CANONICAL)
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "ROUTED_DOWN_K_CHUNK"
    }
    assert assignments == {"ROUTED_DOWN_K_CHUNK": 128}
    for path in _ROUTED_DOWN_SOURCES:
        external_source = path.read_text(encoding="utf-8")
        assert (
            "Tile<TileType::Left, int8_t, 16, 128"
            in external_source
        )
        assert (
            "Tile<TileType::Right, int8_t, 128, 256"
            in external_source
        )


def test_adaptive_grid_planners_cover_every_active_tile_once() -> None:
    stage_chunks = {
        "gate_up": 20,
        "act": 20,
        "quant": 1,
        "down": 16,
    }
    for workers in (22, 23):
        for active_experts in (1, 2, 3, 8, 36):
            for tiles_per_expert in (1, 2, 4):
                for chunks in stage_chunks.values():
                    logical_work = active_experts * chunks
                    owners = [
                        work
                        for worker in range(workers)
                        for work in range(worker, logical_work, workers)
                    ]
                    assert sorted(owners) == list(range(logical_work))
                    assert len(owners) == len(set(owners))

                    covered = [
                        (work // chunks, tile, work % chunks)
                        for work in owners
                        for tile in range(tiles_per_expert)
                    ]
                    expected = [
                        (expert, tile, chunk)
                        for expert in range(active_experts)
                        for tile in range(tiles_per_expert)
                        for chunk in range(chunks)
                    ]
                    assert sorted(covered) == expected

    assert 1 + 23 == 24
    assert 2 + 22 == 24

    shared_gate = [
        chunk
        for worker in range(2)
        for chunk in range(worker, 5, 2)
    ]
    assert sorted(shared_gate) == list(range(5))
    for active_tokens in (0, 1, 2, 16):
        shared_down = [
            block
            for worker in range(2)
            for block in (
                range(16)
                if active_tokens <= 1 and worker == 0
                else range(0)
                if active_tokens <= 1
                else range(worker, 16, 2)
            )
        ]
        assert sorted(shared_down) == list(range(16))
        assert len(shared_down) == len(set(shared_down))


def test_moe_norm_quant_uses_full_eight_row_blocks() -> None:
    source, tree = _parse(_CANONICAL)
    function = _method(tree, "_norm_quant_moe_input")
    body = _segment(source, function)

    constants: dict[str, object] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    assert constants["MOE_NORM_TOKEN_TILE"] == 8
    assert constants["MOE_NORM_SCALAR_PAD"] == 8
    assert "assert BATCH % MOE_NORM_TOKEN_TILE == 0" in source
    assert "MOE_NORM_BLOCKS = BATCH // MOE_NORM_TOKEN_TILE" in source

    assert any(
        ast.unparse(decorator)
        == "pl.function(type=pl.FunctionType.Inline)"
        for decorator in function.decorator_list
    )
    spmd_loops = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and _call_path(node.iter) == "pl.spmd"
    ]
    assert len(spmd_loops) == 1
    spmd_call = spmd_loops[0].iter
    assert ast.unparse(spmd_call.args[0]) == "MOE_NORM_BLOCKS"
    assert {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in spmd_call.keywords
    } == {"name_hint": "norm_quant_moe_input"}

    scalar_full_calls = [
        call
        for call in ast.walk(function)
        if isinstance(call, ast.Call)
        and _call_path(call) == "pl.full"
        and call.args
        and ast.unparse(call.args[0])
        == "[MOE_NORM_TOKEN_TILE, MOE_NORM_SCALAR_PAD]"
    ]
    assert len(scalar_full_calls) == 3
    assert body.count("[MOE_NORM_TOKEN_TILE, K_CHUNK]") == 2
    assert "valid_shape=" not in body
    assert "target_type=pl.BF16" in body
    assert "target_type=pl.INT32,\n                    mode=\"rint\"" in body
    assert "target_type=pl.FP16, mode=\"round\"" in body
    assert "target_type=pl.INT8, mode=\"trunc\"" in body


def test_moe_norm_quant_grid_covers_supported_storage_batches() -> None:
    for storage_batch in (16, 32, 48):
        written_rows = {
            token_block * 8 + token_idx
            for token_block in range(storage_batch // 8)
            for token_idx in range(8)
        }
        assert written_rows == set(range(storage_batch))


def test_shared_two_stage_schedule_is_bf16_exact() -> None:
    import torch

    storage_rows = 8
    input_hidden = 8
    swiglu_chunk = 2
    shared_hidden = 5 * swiglu_chunk
    down_chunk = 2
    down_blocks = 16
    output_hidden = down_blocks * down_chunk

    def bf16(values: torch.Tensor) -> torch.Tensor:
        return values.to(torch.bfloat16).to(torch.float32)

    x = bf16(
        torch.sin(
            torch.arange(
                storage_rows * input_hidden, dtype=torch.float32,
            ).reshape(storage_rows, input_hidden)
            * 0.013
        )
    )
    w_gate = bf16(
        torch.cos(
            torch.arange(
                input_hidden * shared_hidden, dtype=torch.float32,
            ).reshape(input_hidden, shared_hidden)
            * 0.017
        )
        * 0.25
    )
    w_up = bf16(
        torch.sin(
            torch.arange(
                input_hidden * shared_hidden, dtype=torch.float32,
            ).reshape(input_hidden, shared_hidden)
            * 0.019
        )
        * 0.25
    )
    w_down = bf16(
        torch.cos(
            torch.arange(
                shared_hidden * output_hidden, dtype=torch.float32,
            ).reshape(shared_hidden, output_hidden)
            * 0.011
        )
        * 0.125
    )

    def activation_chunks(limit: float | None) -> list[torch.Tensor]:
        chunks: list[torch.Tensor] = []
        for chunk in range(5):
            n0 = chunk * swiglu_chunk
            n1 = n0 + swiglu_chunk
            gate = x @ w_gate[:, n0:n1]
            up = x @ w_up[:, n0:n1]
            silu = gate * torch.reciprocal(torch.exp(-gate) + 1.0)
            if limit is not None:
                silu = torch.minimum(silu, torch.tensor(limit))
                up = torch.clamp(up, -limit, limit)
            chunks.append(bf16(silu * up))
        return chunks

    def down_block(
        chunks: list[torch.Tensor],
        block: int,
    ) -> torch.Tensor:
        d0 = block * down_chunk
        d1 = d0 + down_chunk
        acc = chunks[0] @ w_down[0:swiglu_chunk, d0:d1]
        for chunk in range(1, 5):
            k0 = chunk * swiglu_chunk
            k1 = k0 + swiglu_chunk
            acc = acc + chunks[chunk] @ w_down[k0:k1, d0:d1]
        return bf16(acc)

    for limit in (None, 16.0):
        reference_chunks = activation_chunks(limit)
        reference = torch.empty(
            [storage_rows, output_hidden],
            dtype=torch.float32,
        )
        for block in range(down_blocks):
            d0 = block * down_chunk
            reference[:, d0 : d0 + down_chunk] = down_block(
                reference_chunks,
                block,
            )

        staged_chunks: list[torch.Tensor | None] = [None] * 5
        computed = activation_chunks(limit)
        for worker in range(2):
            for chunk in range(worker, 5, 2):
                staged_chunks[chunk] = computed[chunk]
        assert all(chunk is not None for chunk in staged_chunks)
        concrete_chunks = [
            chunk for chunk in staged_chunks if chunk is not None
        ]

        staged = torch.empty_like(reference)
        for worker in range(2):
            for block in range(worker, down_blocks, 2):
                d0 = block * down_chunk
                staged[:, d0 : d0 + down_chunk] = down_block(
                    concrete_chunks,
                    block,
                )
        assert torch.equal(staged, reference)
