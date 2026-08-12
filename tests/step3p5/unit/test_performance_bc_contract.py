# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Static release contracts for PERF-B3 / PERF-C1 / PERF-C3 / PERF-G1."""
from __future__ import annotations

import ast
from pathlib import Path

from tests.step3p5.probes._probe_g1_active_batch import _executable_match


_ROOT = Path(__file__).resolve().parents[3]
_CANONICAL = _ROOT / "models" / "step3p5" / "decode_fwd.py"
_HOLDER = _ROOT / "tools" / "step3p5" / "whole_decode_holder.py"
_FULL_ATTN = _ROOT / "models" / "step3p5" / "attention_full.py"
_SWA_ATTN = _ROOT / "models" / "step3p5" / "attention_swa.py"
_MTP_HIDDEN = _ROOT / "models" / "step3p5" / "mtp_hidden_fwd.py"
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
    enter_source = _segment(holder_source, _method(holder_tree, "__enter__"))
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


def test_c1_epoch_owners_keep_last_scalar_and_calls_preserve_arity() -> None:
    _, tree = _parse(_CANONICAL)
    names = (
        "dispatch_step",
        "combine_step",
        "full_moe_chip_orch",
        "swa_moe_chip_orch",
        "full_moe_chip_orch_swiglu7_swiglu16",
        "swa_moe_chip_orch_swiglu7_silu",
    )
    for name in names:
        function = _method(tree, name)
        args = [arg.arg for arg in function.args.args]
        assert args[-1] == "moe_epoch"
        assert args.count("moe_epoch") == 1
        expected = len(args) - 1
        calls = _method_calls(tree, name)
        assert calls, f"{name} has no call site"
        assert all(len(call.args) == expected for call in calls)
        assert all(
            isinstance(call.args[-1], ast.Name)
            and call.args[-1].id.startswith("moe_epoch")
            for call in calls
        )


def test_c3_combine_owns_scatter_wait_and_plain_fp32_reduce() -> None:
    source, tree = _parse(_CANONICAL)
    function = _method(tree, "combine_step")
    names = [arg.arg for arg in function.args.args[1:]]
    assert names == [
        "local_routed_y", "sh_y", "moe_out",
        "combine_arrived", "local_route", "routed_y_buf",
        "local_expert_count", "local_expert_offset", "recv_meta_local",
        "local_route_count", "local_route_count_tid",
        "num_tokens", "my_rank", "moe_epoch",
    ]
    body = _segment(source, function)
    assert 'name_hint="combine_scatter"' in body
    assert "pld.tensor.put(" in body
    assert 'name_hint="combine_wait"' in body
    assert "moe_epoch * n_local_experts, pl.INT32" in body
    assert 'name_hint="combine_reduce"' in body
    assert "target_type=pl.FP32" in body
    assert "expert_weights" not in body
    assert "route_weight" not in body
    assert "pld.tensor.get(" not in body


def test_c1_c2_have_three_independent_monotonic_signal_lineages() -> None:
    source, tree = _parse(_CANONICAL)
    dispatch = _segment(source, _method(tree, "dispatch_step"))
    combine = _segment(source, _method(tree, "combine_step"))
    assert "target=meta_arrived" in dispatch
    assert "signal=meta_arrived" in dispatch
    assert "expected=moe_epoch" in dispatch
    assert "target=data_arrived" in dispatch
    assert "signal=data_arrived" in dispatch
    assert "moe_epoch * n_local_experts, pl.INT32" in dispatch
    assert "target=combine_arrived" in combine
    assert "signal=combine_arrived" in combine
    assert "moe_epoch * n_local_experts, pl.INT32" in combine
    for body in (dispatch, combine):
        assert "pld.NotifyOp.AtomicAdd" in body
        assert "pld.WaitCmp.Ge" in body
        assert "moe_epoch * 2" not in body
        assert "2 * moe_epoch" not in body
        assert "ready_epoch" not in body
        assert "previous_complete" not in body


def test_c1_ep_windows_are_single_set_but_tp_scratch_stays_per_layer() -> None:
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
        assert "NUM_MOE_LAYERS_TOTAL" not in annotations[name]
    for name in (
        "moe_attn_tmp_stack", "moe_attn_signal_stack",
        "moe_sh_tmp_stack", "moe_sh_signal_stack",
    ):
        assert "NUM_MOE_LAYERS_TOTAL" in annotations[name]
    host_source = _segment(source, _method(tree, "host_orch"))
    for name in (
        "moe_recv_meta_stack_buf", "moe_meta_arrived_stack_buf",
        "moe_recv_x_stack_buf", "moe_recv_aux_stack_buf",
        "moe_recv_route_stack_buf", "moe_data_arrived_stack_buf",
        "moe_combine_arrived_stack_buf", "moe_routed_y_buf_stack_buf",
    ):
        line = next(line for line in host_source.splitlines()
                    if line.strip().startswith(f"{name} ="))
        assert "NUM_MOE_LAYERS_TOTAL" not in line
    assert "dispatch_lane_rows" in host_source
    assert "n_routes_per_rank" in host_source


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
    for reused_name in (
        "moe_meta_arrived_stack_buf", "moe_data_arrived_stack_buf",
        "moe_combine_arrived_stack_buf",
    ):
        assert f"{reused_name} = pld.alloc_window_buffer(COMM_CONTROL_SIGNAL_BYTES)" in host_source


def test_c1_whole_graph_uses_epochs_one_through_42() -> None:
    source, tree = _parse(_CANONICAL)
    whole_source = _segment(source, _method(tree, "whole_chip_orch"))
    assert "moe_epoch = pl.cast(layer_idx + 1, pl.INT32)" in whole_source
    assert "moe_epoch_43 = pl.cast(41, pl.INT32)" in whole_source
    assert "moe_epoch_44 = pl.cast(42, pl.INT32)" in whole_source
    for name in (
        "full_moe_chip_orch",
        "swa_moe_chip_orch",
        "swa_moe_chip_orch_swiglu7_silu",
        "full_moe_chip_orch_swiglu7_swiglu16",
    ):
        for call in _method_calls(tree, name):
            assert isinstance(call.args[-1], ast.Name)
            assert call.args[-1].id.startswith("moe_epoch")
    for stale in ("moe_pub_off", "moe_recv_off", "moe_route_off"):
        assert stale not in whole_source


def test_c3_expert_lane_fanout_uses_spmd_and_no_incore_parallel() -> None:
    source, tree = _parse(_CANONICAL)
    # This is a C3 expert-lane contract, not a blanket restriction on every
    # InCore helper in the canonical program.  In particular, tp_all_reduce
    # uses write-disjoint pl.parallel copies for the final local copy.
    for name in ("dispatch_step", "combine_step"):
        function = _method(tree, name)
        assert not any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "pl" and call.func.attr == "parallel"
            for call in ast.walk(function)
        )
    dispatch = _segment(source, _method(tree, "dispatch_step"))
    combine = _segment(source, _method(tree, "combine_step"))
    assert 'name_hint="dispatch_push"' in dispatch
    assert 'name_hint="dispatch_gather"' in dispatch
    assert 'name_hint="combine_scatter"' in combine
    assert "for t in pl.range(active_tokens):" in dispatch
    assert "for k in pl.range(TOPK):" in dispatch
    assert "pld.tensor.put(" in dispatch
    assert "pld.tensor.put(" in combine


def test_c3_moe_hybrid_grids_follow_tokens_routes_and_skip_empty_ranks() -> None:
    source, tree = _parse(_CANONICAL)
    dispatch = _segment(source, _method(tree, "dispatch_step"))
    combine = _segment(source, _method(tree, "combine_step"))

    assert "active_routes = active_tokens * TOPK" in dispatch
    assert "push_blocks = active_tokens" in dispatch
    assert "if push_blocks < 1:" in dispatch
    assert "if push_blocks > n_local_experts:" in dispatch
    assert "scan_blocks = active_routes" in dispatch
    assert "if scan_blocks < 1:" in dispatch
    assert "if scan_blocks > n_local_experts:" in dispatch
    assert "active_blocks" not in dispatch
    assert dispatch.count("with pl.spmd(\n            push_blocks,") == 1
    assert dispatch.count("with pl.spmd(\n            scan_blocks,") == 1
    assert 'name_hint="dispatch_count_publish"' in dispatch
    assert 'name_hint="dispatch_meta"' in dispatch
    assert "route_slot = pl.create_tensor([BATCH, TOPK], dtype=pl.INT32)" in dispatch
    assert "self_meta = pl.create_tensor([n_local_experts], dtype=pl.INT32)" in dispatch
    assert "deps=[meta_publish_tid]" in dispatch
    assert "pl.read(route_slot, [t, k])" in dispatch
    assert "moe_epoch * n_local_experts, pl.INT32" in dispatch
    assert "local_route_count = pl.create_tensor(" in dispatch
    assert "[local_route_plan_size], dtype=pl.INT32" in dispatch
    assert dispatch.count("pl.Tensor[[local_route_plan_size], pl.INT32]") == 1
    assert "active_expert_count_i32 = pl.cast(0, pl.INT32)" in dispatch
    assert (
        "pl.write(local_route_count, [1], active_expert_count_i32)"
        in dispatch
    )
    assert "predicate=(local_route_count[0] > 0)" in dispatch
    assert "deps=[wait_tid, meta_collect_tid]" in dispatch
    assert "meta_collect_tid," in dispatch
    assert "worker, active_tokens, push_blocks" in dispatch
    # Scatter-only experiment: retain the frozen hybrid gather grid.
    assert "worker, n_local_experts, scan_blocks" in dispatch

    assert (
        "local_route_count: pl.Tensor[[local_route_plan_size], pl.INT32]"
        in combine
    )
    assert "active_expert_count = pl.cast(" in combine
    assert "pl.read(local_route_count, [1]), pl.INDEX" in combine
    assert "scatter_blocks = active_expert_count" in combine
    assert "if scatter_blocks < 1:" in combine
    assert "if scatter_blocks > n_local_experts:" in combine
    assert "active_routes" not in combine
    assert "active_blocks" not in combine
    assert "with pl.spmd(\n            scatter_blocks," in combine
    assert 'name_hint="combine_route_gate"' not in combine
    assert "predicate=(local_route_count[0] > 0)" not in combine
    assert "if pl.read(local_route_count, [0]) > 0:" in combine
    assert "allow_early_resolve=True" in combine
    assert "local_route_count_tid: pl.Scalar[pl.TASK_ID]" in combine
    assert "deps=[local_route_count_tid]" in combine
    assert "pl.read(local_route_count, [worker + 2])" in combine
    assert "for e in pl.range(worker, n_local_experts, scatter_blocks)" not in combine
    assert "- pl.cast(scatter_blocks, pl.INT32)" in combine
    assert "Empty-rank scatter did not publish data or credits." in combine
    assert "moe_epoch * n_local_experts, pl.INT32" in combine
    assert "with pl.spmd(\n            BATCH," in combine


def test_c3_route_slot_is_dense_and_token_workers_cover_every_route() -> None:
    source, tree = _parse(_CANONICAL)
    function = _method(tree, "dispatch_step")
    publish = _task_scope(function, "dispatch_count_publish")
    push = _task_scope(function, "dispatch_push")
    publish_source = _segment(source, publish)
    push_source = _segment(source, push)

    # The metadata pass assigns the slot before incrementing its
    # (destination, local-expert) cursor. Token workers then reuse that stable
    # slot for every TOPK route without rebuilding any expert-lane scan.
    slot_write = "pl.write(route_slot, [t, k], slot)"
    cursor_update = "cursor[cursor_idx] ="
    assert publish_source.index(slot_write) < publish_source.index(cursor_update)
    assert "cursor_idx = dst * n_local_experts + loc_e" in publish_source
    assert "slot = cursor[cursor_idx]" in publish_source
    assert "deps=[meta_publish_tid]" in push_source
    assert "worker = pl.tile.get_block_idx()" in push_source
    assert (
        "for t in pl.range(worker, active_tokens, push_blocks):"
        in push_source
    )
    assert "for k in pl.range(TOPK):" in push_source
    assert "route_idx = t * TOPK + k" in push_source
    assert "pl.read(route_slot, [t, k])" in push_source
    assert "for route_idx in pl.range(" not in push_source
    assert "slot_ctr" not in push_source
    assert "if le == loc_e:" not in push_source

    # Pure-Python scheduling oracle: the token grid covers every route once,
    # and dense slots make every physical destination row write-disjoint.
    n_ranks = 8
    n_local = 36
    active_tokens = 16
    topk = 8
    push_blocks = min(max(active_tokens, 1), n_local)
    covered_routes = [
        t * topk + k
        for worker in range(push_blocks)
        for t in range(worker, active_tokens, push_blocks)
        for k in range(topk)
    ]
    assert sorted(covered_routes) == list(range(active_tokens * topk))
    assert len(covered_routes) == len(set(covered_routes))

    dispatch_max_per_src = 16
    source_rank = 3
    cursor: dict[tuple[int, int], int] = {}
    slots: dict[tuple[int, int], list[int]] = {}
    physical_rows: set[tuple[int, int]] = set()
    for t in range(active_tokens):
        for k in range(topk):
            eid = (t * 29 + k * 41 + (t // 7) * 3) % (n_ranks * n_local)
            dst, local_expert = divmod(eid, n_local)
            key = (dst, local_expert)
            slot = cursor.get(key, 0)
            cursor[key] = slot + 1
            slots.setdefault(key, []).append(slot)
            row = (
                local_expert * n_ranks * dispatch_max_per_src
                + source_rank * dispatch_max_per_src
                + slot
            )
            assert (dst, row) not in physical_rows
            physical_rows.add((dst, row))
            assert 0 <= slot < dispatch_max_per_src
    assert len(physical_rows) == active_tokens * topk
    for key, count in cursor.items():
        assert slots[key] == list(range(count))


def test_c3_combine_scatter_uses_compact_active_expert_plan() -> None:
    source, tree = _parse(_CANONICAL)
    dispatch_function = _method(tree, "dispatch_step")
    combine_function = _method(tree, "combine_step")
    meta_collect = _task_scope(dispatch_function, "dispatch_meta")
    gather = _task_scope(dispatch_function, "dispatch_gather")
    scatter = _task_scope(combine_function, "combine_scatter")
    meta_source = _segment(source, meta_collect)
    gather_source = _segment(source, gather)
    scatter_source = _segment(source, scatter)

    assert "local_route_plan_size = n_local_experts + 2" in source
    assert "if count > 0:" in meta_source
    assert "pl.cast(active_expert_count_i32, pl.INDEX)" in meta_source
    assert "+ pl.cast(2, pl.INDEX)" in meta_source
    assert "pl.cast(e, pl.INT32)" in meta_source
    assert "pl.write(local_route_count, [0], total_count)" in meta_source
    assert (
        "pl.write(local_route_count, [1], active_expert_count_i32)"
        in meta_source
    )
    assert "pl.read(local_route_count, [worker + 2])" in scatter_source

    # This candidate changes combine only: dispatch gather remains the frozen
    # hybrid full-expert scan rather than consuming the compact worklist.
    assert "for e in pl.range(worker, n_local_experts, scan_blocks):" in gather_source
    assert "pl.read(local_route_count, [work_idx + 2])" not in gather_source

    # Sparse local expert IDs are packed densely and preserve source order.
    counts = [0, 2, 0, 1, 7, 0, 0, 3] + [0] * 28
    plan = [0] * (len(counts) + 2)
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


def test_c3_fixed_lane_credits_cover_hybrid_work_and_cannot_arrive_early() -> None:
    source, tree = _parse(_CANONICAL)
    dispatch_function = _method(tree, "dispatch_step")
    combine_function = _method(tree, "combine_step")
    dispatch_push = _task_scope(dispatch_function, "dispatch_push")
    dispatch_wait = _task_scope(dispatch_function, "dispatch_wait")
    combine_scatter = _task_scope(combine_function, "combine_scatter")
    combine_wait = _task_scope(combine_function, "combine_wait")

    dispatch_push_source = _segment(source, dispatch_push)
    dispatch_wait_source = _segment(source, dispatch_wait)
    combine_scatter_source = _segment(source, combine_scatter)
    combine_wait_source = _segment(source, combine_wait)

    assert "completion_delta = pl.cast(1, pl.INT32)" in dispatch_push_source
    assert "if worker == 0:" in dispatch_push_source
    assert "- pl.cast(push_blocks, pl.INT32)" in dispatch_push_source
    assert "target=data_arrived" in dispatch_push_source
    assert "target=data_arrived" not in dispatch_wait_source
    assert "moe_epoch * n_local_experts, pl.INT32" in dispatch_wait_source

    assert "completion_delta = pl.cast(1, pl.INT32)" in combine_scatter_source
    assert "if worker == 0:" in combine_scatter_source
    assert "- pl.cast(scatter_blocks, pl.INT32)" in combine_scatter_source
    assert "target=combine_arrived" in combine_scatter_source
    assert "pl.read(local_route_count, [0]) == 0" in combine_wait_source
    assert "moe_epoch * n_local_experts, pl.INT32" in combine_wait_source
    empty_branches = [
        node
        for node in ast.walk(combine_wait)
        if isinstance(node, ast.If)
        and "pl.read(local_route_count, [0]) == 0" in ast.unparse(node.test)
    ]
    assert len(empty_branches) == 1
    wait_notifies = [
        call
        for call in ast.walk(combine_wait)
        if isinstance(call, ast.Call)
        and _call_path(call) == "pld.system.notify"
        and "combine_arrived" in ast.unparse(call)
    ]
    empty_notifies = [
        call
        for call in ast.walk(empty_branches[0])
        if isinstance(call, ast.Call)
        and _call_path(call) == "pld.system.notify"
        and "combine_arrived" in ast.unparse(call)
    ]
    assert wait_notifies
    assert {id(call) for call in wait_notifies} == {
        id(call) for call in empty_notifies
    }
    assert all(
        "value=pl.cast(n_local_experts, pl.INT32)" in ast.unparse(call)
        for call in empty_notifies
    )

    # Every non-empty producer notifies in the same task, after its payload
    # operations. Thus reaching the fixed threshold proves every physical
    # token/scatter worker completed.
    for scope, payload_paths, notify_target in (
        (
            dispatch_push,
            {"pld.tensor.put", "pld.tile.remote_store"},
            "data_arrived",
        ),
        (combine_scatter, {"pld.tensor.put"}, "combine_arrived"),
    ):
        payload_lines = [
            call.lineno
            for call in ast.walk(scope)
            if isinstance(call, ast.Call)
            and _call_path(call) in payload_paths
        ]
        notify_lines = [
            call.lineno
            for call in ast.walk(scope)
            if isinstance(call, ast.Call)
            and _call_path(call) == "pld.system.notify"
            and notify_target in ast.unparse(call)
        ]
        assert payload_lines and notify_lines
        assert max(payload_lines) < min(notify_lines)

    n_local = 36
    topk = 8
    for active_tokens in range(17):
        active_routes = active_tokens * topk
        push_blocks = min(max(active_tokens, 1), n_local)
        scan_blocks = min(max(active_routes, 1), n_local)

        covered_routes = [
            t * topk + k
            for worker in range(push_blocks)
            for t in range(worker, active_tokens, push_blocks)
            for k in range(topk)
        ]
        assert sorted(covered_routes) == list(range(active_routes))
        assert len(covered_routes) == len(set(covered_routes))

        scanned_experts = [
            expert
            for worker in range(scan_blocks)
            for expert in range(worker, n_local, scan_blocks)
        ]
        assert sorted(scanned_experts) == list(range(n_local))

        dispatch_credits = [
            1 + (n_local - push_blocks if worker == 0 else 0)
            for worker in range(push_blocks)
        ]
        assert sum(dispatch_credits) == n_local
        assert all(credit > 0 for credit in dispatch_credits)
        for credit in dispatch_credits:
            assert sum(dispatch_credits) - credit < n_local

        max_active_experts = min(active_routes, n_local)
        for active_experts in range(max_active_experts + 1):
            scatter_blocks = min(max(active_experts, 1), n_local)
            if active_experts == 0:
                # The non-predicated defensive block is an in-kernel no-op;
                # combine_wait publishes all 36 credits instead.
                assert scatter_blocks == 1
                continue
            assert scatter_blocks == active_experts
            combine_credits = [
                1 + (n_local - scatter_blocks if worker == 0 else 0)
                for worker in range(scatter_blocks)
            ]
            assert sum(combine_credits) == n_local
            assert all(credit > 0 for credit in combine_credits)
            for credit in combine_credits:
                assert sum(combine_credits) - credit < n_local

    # A rank receiving no routes launches one no-op scatter block, then its
    # symmetric control task advances peers by the same fixed lane budget.
    assert n_local == 36


def test_two_layer_tp_all_reduce_matches_canonical() -> None:
    _, canonical_tree = _parse(_CANONICAL)
    _, two_layer_tree = _parse(_TWO_LAYER_PROGRAM)
    canonical = _method(canonical_tree, "tp_all_reduce")
    two_layer = _method(two_layer_tree, "tp_all_reduce")
    assert ast.dump(canonical, include_attributes=False) == ast.dump(
        two_layer,
        include_attributes=False,
    )


def test_tp_all_reduce_uses_tput_source_and_existing_push_gather() -> None:
    source, tree = _parse(_CANONICAL)
    method = _method(tree, "tp_all_reduce")
    body = _segment(source, method)
    assert [arg.arg for arg in method.args.args][-2:] == [
        "active_rows_i32",
        "my_rank",
    ]
    assert body.count("pld.tensor.put(") == 1
    assert "peer=my_rank" in body
    assert "dst=tmp_window,\n            peer=my_rank,\n            src=local" in body
    assert "for dst in pl.range(group_size):" in body
    assert "pld.tile.remote_store(" in body
    assert "for ar_b0 in pl.range(0, BATCH, BATCH_TILE):" in body
    assert "pl.store(reduced_tile, [ar_b0, owned_base], tmp_window)" in body
    assert "pl.store(reduced_tile, [ar_b0, owned_base], local)" not in body
    assert "chunk_rows=BATCH_TILE" in body
    assert "chunk_cols=TP_ALL_REDUCE_CHUNK" in body
    assert "shape=[BATCH_TILE, owned_chunk]" in body
    assert "ar_copy_tiles = (BATCH // BATCH_TILE) * (HIDDEN // ar_chunk)" in body
    for expected in (1, 2, 3):
        assert f"expected={expected}" in body


def test_tp_all_reduce_limits_only_publish_and_final_copy_rows() -> None:
    source, tree = _parse(_CANONICAL)
    body = _segment(source, _method(tree, "tp_all_reduce"))

    assert "active_rows = pl.cast(active_rows_i32, pl.INDEX)" in body
    assert "if active_rows < 0:" not in body
    assert "if active_rows > BATCH:" in body
    assert "active_rows = pl.cast(BATCH, pl.INDEX)" in body
    assert body.count(
        "BATCH_TILE, pl.max(0, active_rows - ar_b0)"
    ) == 2
    assert "reduced_tile_raw = pl.cast(acc, target_type=pl.BF16)" in body
    assert (
        "reduced_tile_raw, publish_active_rows, owned_chunk" in body
    )
    assert "valid_shapes=[copy_active_rows, ar_chunk]" in body

    # Source publication and peer reads intentionally stay static: the pinned
    # PTOAS rejects dynamic TPUT destinations and remote_load has no separate
    # runtime-valid-shape API.
    put_call = next(
        call
        for call in ast.walk(_method(tree, "tp_all_reduce"))
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "put"
    )
    assert "shape" not in {keyword.arg for keyword in put_call.keywords}
    remote_loads = [
        call
        for call in ast.walk(_method(tree, "tp_all_reduce"))
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "remote_load"
    ]
    assert len(remote_loads) == 1
    assert any(
        keyword.arg == "shape"
        and ast.unparse(keyword.value) == "[BATCH_TILE, owned_chunk]"
        for keyword in remote_loads[0].keywords
    )


def test_tp_all_reduce_keeps_reduce_scatter_accumulate_serial() -> None:
    # The reduce-scatter accumulate carries an FP32 accumulator across peers in
    # a fixed order and casts to BF16 exactly once.  hidden_tp_spread == 0
    # depends on that shape, so the loop must never become pl.parallel/pl.spmd.
    # Switching it is also pointless: swapping pl.range for pl.parallel in this
    # InCore body produced byte-identical AIV codegen (measured 2026-08-05, the
    # loop kind is not consumed here), and the onephase_par microbenchmark
    # measured parallel re-reduction as slower.
    source, tree = _parse(_CANONICAL)
    body = _segment(source, _method(tree, "tp_all_reduce"))
    assert "for peer in pl.range(group_size):" in body
    assert "acc = pl.mul(pl.cast(own_tile, target_type=pl.FP32), 0.0)" in body
    assert body.count(
        "reduced_tile_raw = pl.cast(acc, target_type=pl.BF16)"
    ) == 1
    assert body.count("reduced_tile = pl.set_validshape(") == 1
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
    holder_enter = _segment(holder_source, _method(holder_tree, "__enter__"))
    live_step = _segment(holder_source, _method(holder_tree, "set_live_step"))
    assert "self.num_tokens_per_owner" in holder_enter
    assert "self.compiled.prepare(persistent=True)" in holder_enter
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


def test_c3_expert_storage_keeps_fixed_v4_lane_bases() -> None:
    source, tree = _parse(_CANONICAL)
    dispatch = _segment(source, _method(tree, "dispatch_step"))
    expert = _segment(source, _method(tree, "_expert_routed"))
    expert_swiglu7 = _segment(source, _method(tree, "_expert_routed_swiglu7"))
    combine = _segment(source, _method(tree, "combine_step"))

    assert "expert_recv_max = dispatch_recv_per_expert" in source
    assert "local_recv_max = n_local_experts * expert_recv_max" in source
    assert "pl.cast(e * expert_recv_max, pl.INT32)" in dispatch
    assert "total = total + count" not in dispatch
    assert "out_base = pl.cast(e * expert_recv_max, pl.INDEX)" in dispatch
    for body in (expert, expert_swiglu7):
        # The ordinary routed path names the fixed lane base ``expert_base``
        # while the untouched SwiGLU specialization retains ``offset``.
        # Both forms must be derived directly from the static expert lane;
        # neither may consume the dynamic prefix built by dispatch.
        assert (
            "expert_base = e * expert_recv_max" in body
            or "offset = pl.cast(e * expert_recv_max, pl.INDEX)" in body
        )
        assert "tile_offset = " in body
        assert " + tile_row0" in body
        assert "pl.read(local_expert_offset, [e])" not in body
    assert "expert_base = pl.cast(e * expert_recv_max, pl.INDEX)" in combine
    assert "pl.read(local_expert_offset, [e])" not in combine


def test_shared_mlp_adapts_down_ownership_without_dynamic_grid() -> None:
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

    gate = _task_scope(helper, "sh_gate_up_act")
    down = _task_scope(helper, "sh_down")
    for scope, blocks in (
        (gate, "SHARED_GATE_UP_ACT_BLOCKS"),
        (down, "SHARED_DOWN_WORKERS"),
    ):
        call = scope.items[0].context_expr
        assert isinstance(call, ast.Call)
        assert _call_path(call) == "pl.spmd"
        assert [ast.unparse(arg) for arg in call.args] == [blocks]

    gate_source = _segment(source, gate)
    assert (
        "for chunk in pl.range(\n"
        "                worker,\n"
        "                SHARED_GATE_UP_ACT_CHUNKS,\n"
        "                SHARED_GATE_UP_ACT_BLOCKS,"
    ) in gate_source
    assert "sh_hidden[" in gate_source
    assert "pl.cast(gated, target_type=pl.BF16)" in gate_source

    down_source = _segment(source, down)
    assert "deps=[sh_gate_up_tid]" in down_source
    assert "active_tokens = pl.cast(num_tokens, pl.INDEX)" in down_source
    assert "if active_tokens <= 1:" in down_source
    assert "if worker == 0:" in down_source
    assert "HIDDEN // SHARED_DOWN_N_CHUNK" in down_source
    assert (
        "worker,\n"
        "                    HIDDEN // SHARED_DOWN_N_CHUNK,\n"
        "                    SHARED_DOWN_WORKERS,"
        in down_source
    )
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
        ast.unparse(call.args[-2]) == "num_tokens"
        for call in orchestration_calls
    )


def test_regular_routed_expert_adapts_grid_to_shared_worker_budget() -> None:
    source, tree = _parse(_CANONICAL)
    expected_assignments = {
        "ROUTED_GATE_MM_K_CHUNK": 256,
        "ROUTED_GATE_MM_N_CHUNK": 256,
        "ROUTED_GATE_ACT_N_CHUNK": 64,
        "ROUTED_H_QUANT_N_CHUNK": 256,
        "ROUTED_GATE_K_CHUNK": 64,
        "ROUTED_GATE_N_CHUNK": 64,
        "ROUTED_GRID_WORKERS": 23,
        "ROUTED_MULTIBATCH_GRID_WORKERS": 22,
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
    for divisibility_contract in (
        "HIDDEN % ROUTED_GATE_MM_K_CHUNK == 0",
        "MOE_INTERMEDIATE % ROUTED_GATE_MM_N_CHUNK == 0",
        "MOE_INTERMEDIATE % ROUTED_H_QUANT_N_CHUNK == 0",
    ):
        assert sum(
            isinstance(node, ast.Assert)
            and ast.unparse(node.test) == divisibility_contract
            for node in tree.body
        ) == 1

    expert = _method(tree, "_expert_routed")
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
    stages = (
        ("expert_gate_up", "local_route_count_tid"),
        ("expert_gate_up_act", "routed_gate_up_tid"),
        ("routed_h_quant", "routed_act_tid"),
        ("expert_down", "routed_quant_tid"),
    )
    for name_hint, dependency in stages:
        scope = _task_scope(expert, name_hint)
        call = scope.items[0].context_expr
        assert isinstance(call, ast.Call)
        assert _call_path(call) == "pl.spmd"
        assert [ast.unparse(arg) for arg in call.args] == [
            "routed_workers",
        ]
        keywords = {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        }
        assert keywords["deps"] == f"[{dependency}]"
        assert keywords["predicate"] == "local_route_count[0] > 0"
        if name_hint == "expert_gate_up":
            assert keywords["optimizations"] == (
                "[pl.split(pl.SplitMode.NONE, slot_num=4)]"
            )
        else:
            assert "optimizations" not in keywords
        scope_source = _segment(source, scope)
        assert "pl.read(local_route_count, [1])" in scope_source
        assert "pl.read(local_route_count, [active_slot + 2])" in scope_source
        grid_stride_loops = [
            node
            for node in ast.walk(scope)
            if isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and _call_path(node.iter) == "pl.range"
            and len(node.iter.args) == 3
            and ast.unparse(node.iter.args[0]) == "worker"
            and ast.unparse(node.iter.args[2]) == "routed_workers"
        ]
        assert len(grid_stride_loops) == 1

    gate = _segment(source, _task_scope(expert, "expert_gate_up"))
    assert (
        "chunks_per_expert = inter // ROUTED_GATE_MM_N_CHUNK"
        in gate
    )
    assert "gate_up_work = active_expert_count * chunks_per_expert" in gate
    assert "gate_acc = pl.matmul(x0, wg0, out_dtype=pl.INT32)" in gate
    assert "up_acc = pl.matmul(x0, wu0, out_dtype=pl.INT32)" in gate
    assert "projection" not in gate
    assert "if projection" not in expert_source
    assert "for e in pl.parallel(n_local_experts)" not in expert_source

    for function_name, chunk, rows in (
        ("_expert_routed", "ROUTED_H_QUANT_N_CHUNK", "RECV_TILE"),
        ("_expert_routed_swiglu7", "ROUTED_GATE_N_CHUNK", "RECV_SPECIAL_TILE"),
    ):
        quant = _task_scope(_method(tree, function_name), "routed_h_quant")
        quant_source = _segment(source, quant)
        loops = [
            node
            for node in ast.walk(quant)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id in {"hqa", "hqn"}
        ]
        assert len(loops) == 2
        assert {loop.target.id for loop in loops} == {"hqa", "hqn"}
        for loop in loops:
            assert ast.unparse(loop.iter) == f"pl.range(inter // {chunk})"
            assert [
                ast.unparse(call.args[1])
                for call in ast.walk(loop)
                if isinstance(call, ast.Call)
                and _call_path(call) == "pl.slice"
                and ast.unparse(call.args[0]) == "h_bf16"
            ] == [f"[{rows}, {chunk}]"]

        hqa = next(loop for loop in loops if loop.target.id == "hqa")
        parents = [
            node
            for node in ast.walk(quant)
            if hasattr(node, "body") and hqa in node.body
        ]
        assert len(parents) == 1
        parent_body = parents[0].body
        amax_initializers = [
            node
            for node in ast.walk(quant)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and ast.unparse(node.targets[0]) == "eh_amax"
            and isinstance(node.value, ast.Call)
            and _call_path(node.value) == "pl.full"
        ]
        assert len(amax_initializers) == 1
        initializer = amax_initializers[0]
        assert initializer in parent_body
        assert parent_body.index(initializer) < parent_body.index(hqa)
        assert ast.unparse(initializer.value.args[0]) == f"[1, {rows}]"
        hqa_amax_assignments = [
            node
            for node in ast.walk(hqa)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and ast.unparse(node.targets[0]) == "eh_amax"
        ]
        assert len(hqa_amax_assignments) == 1
        hqa_amax_update = hqa_amax_assignments[0]
        assert isinstance(hqa_amax_update.value, ast.Call)
        assert _call_path(hqa_amax_update.value) == "pl.maximum"
        assert ast.unparse(hqa_amax_update.value.args[0]) == "eh_amax"
        assert {
            keyword.arg: ast.literal_eval(keyword.value)
            if isinstance(keyword.value, ast.Constant)
            else ast.unparse(keyword.value)
            for keyword in initializer.value.keywords
        } == {"dtype": "pl.FP32", "value": 1e-4}

        if function_name == "_expert_routed_swiglu7":
            assert "ROUTED_H_QUANT_N_CHUNK" not in quant_source

    wrapper = _method(tree, "expert_routed_step")
    assert "num_tokens" in [arg.arg for arg in wrapper.args.args]
    wrapper_calls = _method_calls(wrapper, "_expert_routed")
    assert len(wrapper_calls) == 1
    assert ast.unparse(wrapper_calls[0].args[7]) == "num_tokens"

    orchestration_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "expert_routed_step"
    ]
    assert len(orchestration_calls) == 2
    assert all(
        ast.unparse(call.args[7]) == "num_tokens"
        for call in orchestration_calls
    )


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


def test_routed_down_halves_the_int8_accumulation_chain() -> None:
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

    regular_scope = _task_scope(
        _method(tree, "_expert_routed"), "expert_down",
    )
    special_function = _method(tree, "_expert_routed_swiglu7")
    regular = _segment(source, regular_scope)
    special = _segment(source, special_function)
    for node in (regular_scope, special_function):
        loops = [
            item
            for item in ast.walk(node)
            if isinstance(item, ast.For)
            and isinstance(item.iter, ast.Call)
            and ast.unparse(item.iter)
            == "pl.range(1, inter // ROUTED_DOWN_K_CHUNK)"
        ]
        assert len(loops) == 1
    assert regular.count("[RECV_TILE, ROUTED_DOWN_K_CHUNK]") == 2
    assert (
        special.count("[RECV_SPECIAL_TILE, ROUTED_DOWN_K_CHUNK]")
        == 2
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
