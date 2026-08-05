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


def _contains_name(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(nested, ast.Name) and nested.id == name
        for nested in ast.walk(node)
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
        "full_online_softmax_reduce",
        "full_online_softmax_finalize",
    }
    guarded: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, (ast.For, ast.With)):
            continue
        if isinstance(node, ast.For):
            if not isinstance(node.target, ast.Name):
                continue
            spmd_call = node.iter
            loop_target = node.target.id
            body = node.body
        else:
            if len(node.items) != 1:
                continue
            spmd_call = node.items[0].context_expr
            loop_target = None
            body = node.body
        if _call_name(spmd_call) != "pl.spmd":
            continue
        hint = next(
            (
                kw.value.value
                for kw in spmd_call.keywords
                if kw.arg == "name_hint" and isinstance(kw.value, ast.Constant)
            ),
            None,
        )
        if hint not in required:
            continue
        if body and isinstance(body[0], ast.If):
            if _is_active_guard(body[0], loop_target or ""):
                guarded.add(hint)
                continue
        # A context-stage may either use a static-capacity lane grid with a
        # nested active-row guard, or launch a runtime-sized logical grid whose
        # extent is derived from active_tokens.  Both forms isolate inactive
        # request rows; the latter is the work-quantized implementation.
        if any(
            isinstance(nested, ast.If)
            and _is_active_guard(nested, loop_target or "fa_b")
            for nested in ast.walk(node)
        ):
            guarded.add(hint)
            continue
        if spmd_call.args and (
            _contains_name(spmd_call.args[0], "active_tokens")
            or (
                isinstance(spmd_call.args[0], ast.Name)
                and spmd_call.args[0].id in {
                    "full_qk_active_tasks",
                    "full_softmax_active_tasks",
                    "full_sv_active_tasks",
                    "full_online_softmax_active_tasks",
                    "full_online_softmax_reduce_tasks",
                    "full_online_softmax_active_rows",
                }
            )
        ):
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


def test_full_attention_core_stages_capture_task_ids_and_chain_dependencies() -> None:
    """Dynamic launch extents and scratch consumers must use captured tasks."""
    fn_source = ast.unparse(_function("attention_full"))
    assert "with pl.spmd(full_qk_active_tasks" in fn_source
    assert "as full_qk_tid" in fn_source
    assert "with pl.spmd(full_softmax_active_tasks" in fn_source
    assert "deps=[full_qk_tid]" in fn_source
    assert "as full_softmax_tid" in fn_source
    assert "deps=[full_softmax_tid]" in fn_source


def test_full_online_softmax_writeback_casts_after_flatten() -> None:
    """Finalize separately so out-proj consumes attn_out, not partial scratch."""
    fn = _function("attention_full")
    fn_source = ast.unparse(fn)
    assert "name_hint='full_sv_matmul'" in fn_source
    assert "pl.system.syncall(core_type='mix')" not in fn_source
    assert "pl.split_aiv(2, mode=pl.SplitMode.NONE)" not in fn_source
    full_sv_spmd = next(
        node.items[0].context_expr
        for node in ast.walk(fn)
        if isinstance(node, ast.With)
        and len(node.items) == 1
        and _call_name(node.items[0].context_expr) == "pl.spmd"
        and any(
            keyword.arg == "name_hint"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "full_sv_matmul"
            for keyword in node.items[0].context_expr.keywords
        )
    )
    assert all(
        keyword.arg != "optimizations"
        for keyword in full_sv_spmd.keywords
    ), "recurrent M/L state must have one AIV owner, not two split subblocks"
    assert "name_hint='full_online_softmax_pass_a'" not in fn_source
    assert "name_hint='full_online_softmax_pass_b'" not in fn_source
    assert "name_hint='full_online_softmax_pass_c'" not in fn_source
    assert "name_hint='full_online_softmax_reduce'" in fn_source
    assert "as full_online_softmax_reduce_tid" in fn_source
    assert "name_hint='full_online_softmax_finalize'" in fn_source
    assert "deps=[full_sv_online_tid]" in fn_source
    assert "deps=[full_online_softmax_reduce_tid]" in fn_source
    assert (
        "online_partial_ml = pl.create_tensor([BATCH * MAX_CTX_BLOCKS, "
        "2 * Q_HEAD_BATCH_FULL], dtype=pl.FP32)"
        in fn_source
    )
    assert "online_partial_mi" not in fn_source
    assert "online_partial_li" not in fn_source
    assert (
        "fa_sm_exp_zero_heads = pl.full([Q_HEAD_PAD_FULL - "
        "Q_HEAD_BATCH_FULL, BLOCK_SIZE], dtype=pl.BF16, value=0.0)"
        in fn_source
    )
    assert (
        "all_exp_padded = pl.assemble(all_exp_padded, "
        "fa_sm_exp_zero_heads, [fa_sm_scratch_row + "
        "Q_HEAD_BATCH_FULL, 0])"
        in fn_source
    )
    assert "fa_sv_ml = pl.concat(fa_sv_mi_real, fa_sv_li_real)" in fn_source
    assert (
        "fa_acc_ml_new_row = pl.concat(fa_acc_mi_new_row, "
        "fa_acc_li_new_row)"
        in fn_source
    )
    assert (
        "ctx_flat = pl.reshape(ctx, [1, Q_HEAD_BATCH_FULL * HEAD_DIM])"
        in fn_source
    )
    assert (
        "ctx_flat_bf16 = pl.cast(ctx_flat, target_type=pl.BF16)"
        in fn_source
    )
    assert "attn_out = pl.assemble(attn_out, ctx_flat_bf16, [fa_b, 0])" in fn_source
    assert "ctx_bf16 = pl.cast(ctx, target_type=pl.BF16)" not in fn_source


def test_full_attention_task_profiles_are_explicit_and_portable_by_default() -> None:
    """A2A3 tuning must not silently become a cross-architecture default."""
    config_path = _ROOT / "models" / "step3p5" / "config.py"
    config_tree = ast.parse(config_path.read_text(encoding="utf-8"))
    profiles_node = next(
        node.value
        for node in config_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_ATTN_TASK_PROFILES"
            for target in node.targets
        )
    )
    profiles = ast.literal_eval(profiles_node)
    assert profiles["portable"]["online_blocks_per_task"] == 16
    assert profiles["portable"]["qk_uniform_o1"] == 0
    assert profiles["portable"]["softmax_uniform_o1"] == 0
    assert profiles["portable"]["online_uniform_o1"] == 0
    assert profiles["portable"]["online_reduce_uniform_o1"] == 0
    assert profiles["a2a3"] == {
        "qk_blocks_per_task": 22,
        "softmax_blocks_per_task": 12,
        "online_blocks_per_task": 22,
        "online_reduce_fan_in": 8,
        "qk_uniform_o1": 1,
        "softmax_uniform_o1": 1,
        "online_uniform_o1": 1,
        "online_reduce_uniform_o1": 1,
    }

    profile_default = next(
        node.value
        for node in config_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "ATTN_TASK_PROFILE"
            for target in node.targets
        )
    )
    assert isinstance(profile_default, ast.Call)
    assert ast.unparse(profile_default.func) == "os.environ.get"
    assert [
        ast.literal_eval(argument)
        for argument in profile_default.args
    ] == [
        "PYPTO_STEP3P5_ATTN_TASK_PROFILE",
        "portable",
    ]
    assert not profile_default.keywords

    config_source = config_path.read_text(encoding="utf-8")
    for env_name in (
        "PYPTO_STEP3P5_FULL_ATTN_QK_BLOCKS_PER_TASK",
        "PYPTO_STEP3P5_FULL_ATTN_SOFTMAX_BLOCKS_PER_TASK",
        "PYPTO_STEP3P5_FULL_ATTN_ONLINE_SOFTMAX_BLOCKS_PER_TASK",
        "PYPTO_STEP3P5_FULL_ATTN_ONLINE_SOFTMAX_PARTIALS_PER_REDUCE_TASK",
        "PYPTO_STEP3P5_FULL_ATTN_QK_UNIFORM_O1",
        "PYPTO_STEP3P5_FULL_ATTN_SOFTMAX_UNIFORM_O1",
        "PYPTO_STEP3P5_FULL_ATTN_ONLINE_SOFTMAX_UNIFORM_O1",
        "PYPTO_STEP3P5_FULL_ATTN_ONLINE_SOFTMAX_REDUCE_UNIFORM_O1",
    ):
        assert env_name in config_source


def test_full_qk_uniform_rows_use_task_major_constant_work_mapping() -> None:
    """Uniform rows avoid scans and queue full groups before row tails."""
    fn_source = ast.unparse(_function("attention_full"))
    assert (
        "full_qk_uniform_tasks_per_row = pl.cast(0, pl.INDEX)"
        in fn_source
    )
    assert (
        "full_qk_tasks_uniform = pl.cast(FULL_ATTN_QK_UNIFORM_O1, "
        "pl.INDEX)"
        in fn_source
    )
    assert "+ 1 - FULL_ATTN_QK_UNIFORM_O1" not in fn_source
    assert (
        "fa_count_qk_tasks != full_qk_uniform_tasks_per_row"
        in fn_source
    )
    assert "if active_tokens != 1:" in fn_source
    assert (
        "fa_qk_task_in_b = fa_task // active_tokens"
        in fn_source
    )
    assert (
        "fa_qk_b = fa_task - fa_qk_task_in_b * active_tokens"
        in fn_source
    )
    assert "fa_task // full_qk_uniform_tasks_per_row" not in fn_source
    assert "for fa_qk_scan_b in pl.range(active_tokens):" in fn_source


def test_every_uniform_o1_mapping_has_an_independent_compile_time_guard() -> None:
    fn_source = ast.unparse(_function("attention_full"))
    for guard in (
        "FULL_ATTN_QK_UNIFORM_O1",
        "FULL_ATTN_SOFTMAX_UNIFORM_O1",
        "FULL_ATTN_ONLINE_SOFTMAX_UNIFORM_O1",
        "FULL_ATTN_ONLINE_SOFTMAX_REDUCE_UNIFORM_O1",
    ):
        assert f"pl.cast({guard}, pl.INDEX)" in fn_source
        assert f"- {guard}" not in fn_source
    assert "full_softmax_uniform_tasks_per_row" in fn_source
    assert "full_online_uniform_tasks_per_row" in fn_source
    assert "full_online_reduce_uniform_tasks_per_row" in fn_source
    for task, local_task, row in (
        ("fa_task", "fa_qk_task_in_b", "fa_qk_b"),
        ("fa_task", "fa_sm_task_in_b", "fa_sm_b"),
        ("fa_task", "fa_sv_task_in_b", "fa_sv_b"),
        (
            "fa_reduce_task",
            "fa_reduce_task_in_b",
            "fa_reduce_b",
        ),
    ):
        assert f"{local_task} = {task} // active_tokens" in fn_source
        assert (
            f"{row} = {task} - {local_task} * active_tokens"
            in fn_source
        )
