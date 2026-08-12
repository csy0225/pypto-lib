# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Source contracts for full-attention runtime active-row isolation.

These checks intentionally target only the full-attention implementation. The
static ``BATCH`` dimension remains the storage capacity, while ``num_tokens``
controls the fused split/QKNorm/RoPE/KV-publication logical task grid.
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


def _spmd_scope(function: ast.FunctionDef, hint: str) -> ast.With:
    matches = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.With)
        and len(node.items) == 1
        and _call_name(node.items[0].context_expr) == "pl.spmd"
        and any(
            keyword.arg == "name_hint"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == hint
            for keyword in node.items[0].context_expr.keywords
        )
    ]
    assert len(matches) == 1, f"expected one {hint!r} SPMD scope"
    return matches[0]


def _guarded_body(scope: ast.With, row_name: str) -> ast.If:
    guards = [
        node
        for node in scope.body
        if isinstance(node, ast.If) and _is_active_guard(node, row_name)
    ]
    assert len(guards) == 1
    return guards[0]


def test_full_attention_packs_qkv_and_fuses_active_prerope_publication() -> None:
    fn = _function("attention_full")
    assert "num_tokens" in {arg.arg for arg in fn.args.args}
    assert "b_safe" not in ast.unparse(fn)

    rms_calls = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and _call_name(node) == "pl.at"
        and any(
            keyword.arg == "name_hint"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "full_rmsnorm_zc"
            for keyword in node.keywords
        )
    ]
    assert len(rms_calls) == 1
    assert any(
        keyword.arg == "allow_early_resolve"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in rms_calls[0].keywords
    )

    proj_scope = _spmd_scope(fn, "full_qkv_proj")
    proj_call = proj_scope.items[0].context_expr
    assert ast.unparse(proj_call.args[0]) == "BATCH // BATCH_TILE * full_qkv_tiles"
    assert any(
        keyword.arg == "allow_early_resolve"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in proj_call.keywords
    )
    assert (
        isinstance(proj_scope.items[0].optional_vars, ast.Name)
        and proj_scope.items[0].optional_vars.id == "full_qkv_proj_tid"
    )
    proj_source = ast.unparse(proj_scope)
    assert "qkv_proj = pl.assemble(qkv_proj" in proj_source
    assert "full_qkv_q_offset + q_o0" in proj_source
    assert "full_qkv_k_offset + kv_kind * KV_HIDDEN_LOCAL + kv_o0" in proj_source
    for weight in ("wq", "wk", "wv"):
        assert f"pl.slice({weight}" in proj_source

    head_logits_scope = _spmd_scope(fn, "full_head_gate_logits_mm")
    head_logits_call = head_logits_scope.items[0].context_expr
    assert any(
        keyword.arg == "deps"
        and ast.unparse(keyword.value) == "[full_attn_out_zero_tid]"
        for keyword in head_logits_call.keywords
    )
    assert "hg_part = pl.tile.get_block_idx()" in (
        ast.get_source_segment(_SOURCE, head_logits_scope) or ""
    )
    assert _SOURCE.index('name_hint="full_attn_out_zero"') < _SOURCE.index(
        'name_hint="full_head_gate_logits_mm"'
    )

    prerope_scope = _spmd_scope(fn, "full_qkv_split_qknorm_rope")
    prerope_call = prerope_scope.items[0].context_expr
    assert ast.unparse(prerope_call.args[0]) == "active_tokens"
    assert any(
        keyword.arg == "allow_early_resolve"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in prerope_call.keywords
    )
    assert any(
        keyword.arg == "deps"
        and ast.unparse(keyword.value) == "[full_qkv_proj_tid]"
        for keyword in prerope_call.keywords
    )
    assert (
        isinstance(prerope_scope.items[0].optional_vars, ast.Name)
        and prerope_scope.items[0].optional_vars.id == "full_qkv_prerope_tid"
    )
    prerope_source = ast.unparse(_guarded_body(prerope_scope, "b"))
    assert "pl.tensor.read(seq_lens, [b])" in prerope_source
    assert "pl.tensor.read(slot_mapping, [b])" in prerope_source
    assert "pl.slice(rope_cos" in prerope_source
    assert "pl.slice(rope_sin" in prerope_source
    assert "q_chunk_flat = pl.slice(qkv_proj" in prerope_source
    assert "q_chunk = pl.reshape(q_chunk_flat, [Q_HEAD_BATCH_FULL, HEAD_DIM])" in prerope_source
    assert "q_sq = pl.row_sum(pl.mul(q_chunk, q_chunk))" in prerope_source
    assert "k_chunk_8 = pl.reshape(pl.concat(k_chunk_4, k_chunk_4)" in prerope_source
    assert "k_sq = pl.row_sum(pl.mul(k_chunk_8, k_chunk_8))" in prerope_source
    assert "pl.add(q_gamma, 1.0)" in prerope_source
    assert "pl.add(k_gamma, 1.0)" in prerope_source
    assert "full_qkv_k_offset + kv_col" in prerope_source
    assert "full_qkv_v_offset + kv_col" in prerope_source
    assert "all_q_padded = pl.assemble(all_q_padded" in prerope_source
    assert "k_cache = pl.assemble(k_cache" in prerope_source
    assert "v_cache = pl.assemble(v_cache" in prerope_source
    # Full attention is partial-RoPE: the complete normed Q/K row is
    # published before the rotated 32-lane halves overwrite lanes 0:64.
    assert "pl.cast(q_head, target_type=pl.BF16)" in prerope_source
    assert "pl.cast(k_normed, target_type=pl.BF16)" in prerope_source

    fn_source = ast.unparse(fn)
    for retired in (
        "full_q_proj",
        "full_kv_proj",
        "full_qk_norm_zc",
        "full_rope_q",
        "full_rope_kv_cache",
        "q_proj_norm",
        "k_proj_norm",
    ):
        assert retired not in fn_source
    assert "full_rope_stage = pl.create_tensor" not in fn_source
    assert "full_k_rope_stage = pl.create_tensor" not in fn_source
    assert "full_v_stage = pl.create_tensor" not in fn_source

    mix_call = _spmd_scope(fn, "full_attn_mix").items[0].context_expr
    assert any(
        keyword.arg == "deps"
        and ast.unparse(keyword.value) == "[full_qkv_prerope_tid]"
        for keyword in mix_call.keywords
    )

def test_all_full_attention_request_spmd_stages_use_runtime_bound() -> None:
    fn = _function("attention_full")
    required = {
        "full_qkv_split_qknorm_rope",
        "full_attn_mix",
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
    """The mixed producer owns QK/softmax/SV and publishes reduce partials."""
    fn_source = ast.unparse(_function("attention_full"))
    assert "name_hint='full_qkv_proj'" in fn_source
    assert "as full_qkv_proj_tid" in fn_source
    assert (
        "with pl.spmd(active_tokens, name_hint='full_qkv_split_qknorm_rope', "
        "deps=[full_qkv_proj_tid], allow_early_resolve=True)"
    ) in fn_source
    assert "as full_qkv_prerope_tid" in fn_source
    assert "with pl.spmd(full_online_softmax_active_tasks" in fn_source
    assert "name_hint='full_attn_mix'" in fn_source
    assert "deps=[full_qkv_prerope_tid]" in fn_source
    assert "as full_attn_mix_tid" in fn_source
    assert "name_hint='full_online_softmax_reduce'" in fn_source
    assert "deps=[full_attn_mix_tid]" in fn_source
    assert "as full_online_softmax_reduce_tid" in fn_source
    assert "name_hint='full_online_softmax_finalize'" in fn_source
    assert "deps=[full_online_softmax_reduce_tid]" in fn_source

def test_full_online_softmax_writeback_casts_after_flatten() -> None:
    """Mixed attention keeps one partial owner and a separate final reduce."""
    fn = _function("attention_full")
    fn_source = ast.unparse(fn)
    assert "name_hint='full_attn_mix'" in fn_source
    assert "name_hint='full_qk_matmul'" not in fn_source
    assert "name_hint='full_softmax'" not in fn_source
    assert "name_hint='full_sv_matmul'" not in fn_source
    assert "pl.system.syncall(core_type='mix')" not in fn_source
    mix_spmd = _spmd_scope(fn, "full_attn_mix").items[0].context_expr
    assert all(
        keyword.arg != "optimizations"
        for keyword in mix_spmd.keywords
    ), "segment recurrence must have one AIV owner"
    assert "fa_mix_raw_scores = pl.matmul(" in fn_source
    assert "fa_mix_scores = pl.col_expand_add(" in fn_source
    assert "fa_mix_cur_mi = pl.row_max(fa_mix_scores)" in fn_source
    assert "fa_mix_exp_bf16 = pl.cast(" in fn_source
    assert "fa_mix_cur_li = pl.row_sum(fa_mix_exp_fp32)" in fn_source
    assert "fa_mix_oi = pl.matmul(" in fn_source
    assert "all_raw_scores" not in fn_source
    assert "all_exp_padded" not in fn_source
    assert (
        "MAX_SEGMENTS = (MAX_CTX_BLOCKS + "
        "FULL_ATTN_ONLINE_SOFTMAX_BLOCKS_PER_TASK - 1) // "
        "FULL_ATTN_ONLINE_SOFTMAX_BLOCKS_PER_TASK"
    ) in fn_source
    assert (
        "online_partial = pl.create_tensor([BATCH * MAX_SEGMENTS * "
        "Q_HEAD_PAD_FULL, HEAD_DIM], dtype=pl.FP32)"
    ) in fn_source
    assert (
        "fa_mix_segment_row = fa_mix_b * MAX_SEGMENTS + "
        "fa_mix_task_in_b"
    ) in fn_source
    assert (
        "online_partial_ml = pl.create_tensor([BATCH * MAX_SEGMENTS, "
        "2 * Q_HEAD_BATCH_FULL], dtype=pl.FP32)"
        in fn_source
    )
    assert "fa_mix_ml = pl.concat(fa_mix_mi_real, fa_mix_li_real)" in fn_source
    assert "name_hint='full_online_softmax_reduce'" in fn_source
    assert "deps=[full_attn_mix_tid]" in fn_source
    assert "name_hint='full_online_softmax_finalize'" in fn_source
    assert "deps=[full_online_softmax_reduce_tid]" in fn_source
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
        "softmax_blocks_per_task": 16,
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


def test_full_mixed_uniform_rows_use_task_major_constant_work_mapping() -> None:
    """Uniform rows map mixed segment tasks without orchestration scans."""
    fn_source = ast.unparse(_function("attention_full"))
    assert (
        "full_online_uniform_tasks_per_row = pl.cast(0, pl.INDEX)"
        in fn_source
    )
    assert (
        "full_online_tasks_uniform = pl.cast("
        "FULL_ATTN_ONLINE_SOFTMAX_UNIFORM_O1, pl.INDEX)"
        in fn_source
    )
    assert (
        "fa_count_online_tasks != full_online_uniform_tasks_per_row"
        in fn_source
    )
    assert "if active_tokens == 1:" in fn_source
    assert "fa_mix_task_in_b = fa_task // active_tokens" in fn_source
    assert (
        "fa_mix_b = fa_task - fa_mix_task_in_b * active_tokens"
        in fn_source
    )
    assert "for fa_mix_scan_b in pl.range(active_tokens):" in fn_source
    assert (
        "fa_mix_segment_row = fa_mix_b * MAX_SEGMENTS + "
        "fa_mix_task_in_b"
        in fn_source
    )

def test_mixed_and_reduce_uniform_mappings_have_compile_time_guards() -> None:
    fn_source = ast.unparse(_function("attention_full"))
    for guard in (
        "FULL_ATTN_ONLINE_SOFTMAX_UNIFORM_O1",
        "FULL_ATTN_ONLINE_SOFTMAX_REDUCE_UNIFORM_O1",
    ):
        assert f"pl.cast({guard}, pl.INDEX)" in fn_source
        assert f"- {guard}" not in fn_source
    assert "full_online_uniform_tasks_per_row" in fn_source
    assert "full_online_reduce_uniform_tasks_per_row" in fn_source
    for task, local_task, row in (
        ("fa_task", "fa_mix_task_in_b", "fa_mix_b"),
        ("fa_reduce_task", "fa_reduce_task_in_b", "fa_reduce_b"),
    ):
        assert f"{local_task} = {task} // active_tokens" in fn_source
        assert f"{row} = {task} - {local_task} * active_tokens" in fn_source
