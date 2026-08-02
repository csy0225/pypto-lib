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
    for path, name_hint in (
        (_FULL_ATTN, "full_rope_kv_cache"),
        (_SWA_ATTN, "swa_rope_kv_cache"),
    ):
        source = path.read_text(encoding="utf-8")
        assert "for b in pl.parallel(BATCH):" in source
        assert "if b < active_tokens:" in source
        assert "slot = pl.tensor.read(slot_mapping, [b])" in source
        assert "slot_mapping, [b_safe]" not in source
        assert "layer_cache_base" in source
        assert "cache_row = (" in source
        assert name_hint in source
        assert "[cache_row, 0]" in source
        assert "v_cache = pl.assemble(" in source
        assert "pl.slice(v_proj, [1, HEAD_DIM]" in source
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
    # legitimately uses write-disjoint pl.parallel copies for local staging.
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
        assert "offset = pl.cast(e * expert_recv_max, pl.INDEX)" in body
        assert "pl.read(local_expert_offset, [e])" not in body
    assert "expert_base = pl.cast(e * expert_recv_max, pl.INDEX)" in combine
    assert "pl.read(local_expert_offset, [e])" not in combine
