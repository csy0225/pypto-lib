# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Correctness contracts for the focused two-layer attention harness."""
from __future__ import annotations

import itertools
import json

import pytest
import torch

from tests.step3p5.harnesses._stage_two_layer_attn import (
    _attention_codegen_contract_errors,
    _aggregate_rank_uniformity,
    _collective_low_wait_reference,
    _full_kv_slot_oracle,
    _kv_slot_audit_layout,
    _linear_percentile,
    _make_kv_fixture,
    _run_alternating_input_audit,
    _runtime_active_row_counts,
    _step_metadata,
    _tensor_probe,
    _tilewise_numerical_report,
    _torch_swa_local_partial_oracle,
    analyze_uniformity,
)


def _valid_attention_codegen_source() -> str:
    return """
// Spmd full_qkv_proj_spmd: full_qkv_proj
L0TaskArgs params_t5;
params_t5.add_inout(qkv_proj_v1);
params_t5.launch_spec.set_block_num(10);
params_t5.set_allow_early_resolve(true);
TaskOutputTensors task_5_outs = rt_submit_aic_task(6, params_t5);
PTO2TaskId full_qkv_proj_tid = task_5_outs.task_id();
// Task 6: full_attn_out_zero
L0TaskArgs params_t6;
params_t6.add_inout(attn_out_full);
TaskOutputTensors task_6_outs = rt_submit_aiv_task(7, params_t6);
// Spmd full_qkv_split_qknorm_rope_spmd: full_qkv_split_qknorm_rope
L0TaskArgs params_t7;
params_t7.add_input(qkv_proj_v1);
params_t7.add_output(all_q_padded_v1);
params_t7.add_inout(ext_k_cache);
params_t7.add_inout(ext_v_cache);
params_t7.add_scalar(active_tokens__rv_v2_inline1);
params_t7.launch_spec.set_block_num(active_tokens__rv_v2_inline1);
params_t7.set_allow_early_resolve(true);
PTO2TaskId params_t7_deps[1];
uint32_t params_t7_deps_count = 0;
params_t7_deps[params_t7_deps_count++] = full_qkv_proj_tid;
params_t7.set_dependencies(params_t7_deps, params_t7_deps_count);
TaskOutputTensors task_7_outs = rt_submit_aiv_task(8, params_t7);
PTO2TaskId full_qkv_prerope_tid = task_7_outs.task_id();
// Group full_attn_mix: MixedKernels (AIC + AIV lanes)
L0TaskArgs params_t10;
params_t10.add_input(all_q_padded_v1);
params_t10.add_input(ext_k_cache);
params_t10.add_input(ext_v_cache);
params_t10.add_scalar(full_online_softmax_active_tasks__rv_v2_inline149);
params_t10.launch_spec.set_block_num(
    full_online_softmax_active_tasks__rv_v2_inline149);
params_t10.set_allow_early_resolve(true);
PTO2TaskId params_t10_deps[1];
uint32_t params_t10_deps_count = 0;
params_t10_deps[params_t10_deps_count++] = full_qkv_prerope_tid;
params_t10.set_dependencies(params_t10_deps, params_t10_deps_count);
TaskOutputTensors task_10_outs = rt_submit_task(mixed_10, params_t10);
PTO2TaskId full_attn_mix_tid = task_10_outs.task_id();
// Spmd full_online_softmax_reduce_spmd: full_online_softmax_reduce
L0TaskArgs params_t13;
params_t13.add_scalar(
    full_online_softmax_reduce_tasks__rv_v2_inline117);
params_t13.launch_spec.set_block_num(
    full_online_softmax_reduce_tasks__rv_v2_inline117);
PTO2TaskId params_t13_deps[1];
uint32_t params_t13_deps_count = 0;
params_t13_deps[params_t13_deps_count++] = full_attn_mix_tid;
params_t13.set_dependencies(params_t13_deps, params_t13_deps_count);
TaskOutputTensors task_13_outs = rt_submit_aiv_task(15, params_t13);
PTO2TaskId full_online_softmax_reduce_tid = task_13_outs.task_id();
// Spmd full_online_softmax_finalize_spmd: full_online_softmax_finalize
L0TaskArgs params_t14;
params_t14.launch_spec.set_block_num(
    full_online_softmax_active_rows__rv_v2_inline19);
PTO2TaskId params_t14_deps[1];
uint32_t params_t14_deps_count = 0;
params_t14_deps[params_t14_deps_count++] =
    full_online_softmax_reduce_tid;
params_t14.set_dependencies(params_t14_deps, params_t14_deps_count);
TaskOutputTensors task_14_outs = rt_submit_aiv_task(16, params_t14);
// Spmd swa_qkv_proj_spmd: swa_qkv_proj
L0TaskArgs params_t31;
params_t31.add_inout(qkv_proj_swa_v1);
params_t31.launch_spec.set_block_num(14);
params_t31.set_allow_early_resolve(true);
TaskOutputTensors task_31_outs = rt_submit_aic_task(35, params_t31);
PTO2TaskId swa_qkv_proj_tid = task_31_outs.task_id();
// Task 32: swa_attn_out_zero
L0TaskArgs params_t31_zero;
params_t31_zero.add_inout(attn_out_swa);
TaskOutputTensors task_31_zero_outs = rt_submit_aiv_task(36, params_t31_zero);
// Spmd swa_qkv_split_qknorm_rope_spmd: swa_qkv_split_qknorm_rope
L0TaskArgs params_t32;
params_t32.add_input(qkv_proj_swa_v1);
params_t32.add_output(all_q_padded_swa_v1);
params_t32.add_inout(ext_k_cache_swa);
params_t32.add_inout(ext_v_cache_swa);
params_t32.add_scalar(active_tokens__rv_v2_inline1);
params_t32.launch_spec.set_block_num(active_tokens__rv_v2_inline1);
params_t32.set_allow_early_resolve(true);
PTO2TaskId params_t32_deps[1];
uint32_t params_t32_deps_count = 0;
params_t32_deps[params_t32_deps_count++] = swa_qkv_proj_tid;
params_t32.set_dependencies(params_t32_deps, params_t32_deps_count);
TaskOutputTensors task_32_outs = rt_submit_aiv_task(36, params_t32);
PTO2TaskId swa_qkv_prerope_tid = task_32_outs.task_id();
// Group swa_attn_mix: MixedKernels (AIC + AIV lanes)
L0TaskArgs params_t34;
params_t34.add_input(all_q_padded_swa_v1);
params_t34.add_input(ext_k_cache_swa);
params_t34.add_input(ext_v_cache_swa);
params_t34.launch_spec.set_block_num(swa_active_tasks__rv_v2_inline207);
params_t34.set_allow_early_resolve(true);
PTO2TaskId params_t34_deps[1];
uint32_t params_t34_deps_count = 0;
params_t34_deps[params_t34_deps_count++] = swa_qkv_prerope_tid;
params_t34.set_dependencies(params_t34_deps, params_t34_deps_count);
TaskOutputTensors task_34_outs = rt_submit_task(mixed_34, params_t34);
"""


def test_attention_codegen_contract_accepts_dynamic_bounds_and_task_chain() -> None:
    assert _attention_codegen_contract_errors(
        _valid_attention_codegen_source(),
    ) == []


def test_attention_codegen_contract_rejects_stale_bound_and_dependency() -> None:
    source = _valid_attention_codegen_source().replace(
        "params_t10.launch_spec.set_block_num(\n"
        "    full_online_softmax_active_tasks__rv_v2_inline149);",
        "params_t10.launch_spec.set_block_num(stale_full_task_bound);",
    ).replace(
        "params_t34_deps[params_t34_deps_count++] = swa_qkv_prerope_tid;",
        "params_t34_deps[params_t34_deps_count++] = stale_swa_task;",
    )
    errors = _attention_codegen_contract_errors(source)
    assert "full mixed attention dynamic launch" in errors
    assert (
        "SWA mixed attention dependency from swa_qkv_prerope_tid"
        in errors
    )


def test_attention_codegen_contract_rejects_incomplete_dependency_wiring() -> None:
    source = _valid_attention_codegen_source().replace(
        "params_t13.set_dependencies(params_t13_deps, "
        "params_t13_deps_count);",
        "",
    )
    assert (
        "full reduce dependency from full_attn_mix_tid"
        in _attention_codegen_contract_errors(source)
    )


@pytest.mark.parametrize(
    ("needle", "expected_error"),
    [
        (
            "params_t7_deps[params_t7_deps_count++] = full_qkv_proj_tid;",
            "full packed split/QKNorm/RoPE dependency from full_qkv_proj_tid",
        ),
        (
            "params_t10_deps[params_t10_deps_count++] = full_qkv_prerope_tid;",
            "full mixed attention dependency from full_qkv_prerope_tid",
        ),
        (
            "params_t32_deps[params_t32_deps_count++] = swa_qkv_proj_tid;",
            "SWA packed split/QKNorm/RoPE dependency from swa_qkv_proj_tid",
        ),
        (
            "params_t34_deps[params_t34_deps_count++] = swa_qkv_prerope_tid;",
            "SWA mixed attention dependency from swa_qkv_prerope_tid",
        ),
    ],
)
def test_attention_codegen_contract_requires_packed_prerope_chain(
    needle: str,
    expected_error: str,
) -> None:
    source = _valid_attention_codegen_source().replace(needle, "")
    assert expected_error in _attention_codegen_contract_errors(source)


@pytest.mark.parametrize(
    ("needle", "expected_error"),
    [
        (
            "params_t5.set_allow_early_resolve(true);",
            "full packed QKV projection early-resolve hint",
        ),
        (
            "params_t7.set_allow_early_resolve(true);",
            "full packed split/QKNorm/RoPE early-resolve hint",
        ),
        (
            "params_t10.set_allow_early_resolve(true);",
            "full mixed attention early-resolve hint",
        ),
        (
            "params_t31.set_allow_early_resolve(true);",
            "SWA packed QKV projection early-resolve hint",
        ),
        (
            "params_t32.set_allow_early_resolve(true);",
            "SWA packed split/QKNorm/RoPE early-resolve hint",
        ),
        (
            "params_t34.set_allow_early_resolve(true);",
            "SWA mixed attention early-resolve hint",
        ),
    ],
)
def test_attention_codegen_contract_requires_early_resolve(
    needle: str,
    expected_error: str,
) -> None:
    source = _valid_attention_codegen_source().replace(needle, "")
    assert expected_error in _attention_codegen_contract_errors(source)


def test_attention_codegen_contract_checks_packed_projection_bounds() -> None:
    source = _valid_attention_codegen_source().replace(
        "params_t5.launch_spec.set_block_num(10);",
        "params_t5.launch_spec.set_block_num(9);",
    ).replace(
        "params_t31.launch_spec.set_block_num(14);",
        "params_t31.launch_spec.set_block_num(13);",
    )
    errors = _attention_codegen_contract_errors(source)
    assert "full packed QKV projection dynamic launch" in errors
    assert "SWA packed QKV projection dynamic launch" in errors


def test_attention_codegen_contract_checks_swa_mixed_bound() -> None:
    source = _valid_attention_codegen_source().replace(
        "params_t34.launch_spec.set_block_num("
        "swa_active_tasks__rv_v2_inline207);",
        "params_t34.launch_spec.set_block_num(1);",
    )
    assert _attention_codegen_contract_errors(source) == [
        "SWA mixed attention dynamic launch",
    ]


def test_attention_codegen_contract_checks_launch_scalar_ssa_identity() -> None:
    source = _valid_attention_codegen_source().replace(
        "params_t10.add_scalar("
        "full_online_softmax_active_tasks__rv_v2_inline149);",
        "params_t10.add_scalar("
        "full_online_softmax_active_tasks__rv_v2_inline0);",
    )
    assert (
        "full mixed attention launch/scalar SSA agreement"
        in _attention_codegen_contract_errors(source)
    )


@pytest.mark.parametrize(
    ("prefix", "needle", "replacement"),
    [
        (
            "full",
            "params_t7.add_input(qkv_proj_v1);",
            "params_t7.add_input(qkv_proj_v2);",
        ),
        (
            "full",
            "params_t10.add_input(all_q_padded_v1);",
            "params_t10.add_input(all_q_padded_v2);",
        ),
        (
            "full",
            "params_t10.add_input(ext_k_cache);",
            "params_t10.add_input(ext_k_cache_v2);",
        ),
        (
            "full",
            "params_t10.add_input(ext_v_cache);",
            "params_t10.add_input(ext_v_cache_v2);",
        ),
        (
            "swa",
            "params_t32.add_input(qkv_proj_swa_v1);",
            "params_t32.add_input(qkv_proj_swa_v2);",
        ),
        (
            "swa",
            "params_t34.add_input(all_q_padded_swa_v1);",
            "params_t34.add_input(all_q_padded_swa_v2);",
        ),
        (
            "swa",
            "params_t34.add_input(ext_k_cache_swa);",
            "params_t34.add_input(ext_k_cache_swa_v2);",
        ),
        (
            "swa",
            "params_t34.add_input(ext_v_cache_swa);",
            "params_t34.add_input(ext_v_cache_swa_v2);",
        ),
    ],
)
def test_attention_codegen_contract_rejects_packed_prerope_lineage_mismatch(
    prefix: str,
    needle: str,
    replacement: str,
) -> None:
    source = _valid_attention_codegen_source().replace(needle, replacement)
    assert (
        f"{prefix} packed pre-attention tensor lineage"
        in _attention_codegen_contract_errors(source)
    )


def _scan_mapping(tasks_per_row: tuple[int, ...]) -> list[tuple[int, int]]:
    mapping = []
    for row, task_count in enumerate(tasks_per_row):
        mapping.extend((row, local) for local in range(task_count))
    return mapping


def _uniform_mapping(tasks_per_row: tuple[int, ...]) -> list[tuple[int, int]]:
    assert tasks_per_row
    assert len(set(tasks_per_row)) == 1
    per_row = tasks_per_row[0]
    if per_row == 0:
        return []
    return [
        (task % len(tasks_per_row), task // len(tasks_per_row))
        for task in range(len(tasks_per_row) * per_row)
    ]


def _fail_closed_mapping(
    tasks_per_row: tuple[int, ...],
    *,
    uniform_enabled: bool,
) -> list[tuple[int, int]]:
    if uniform_enabled and len(set(tasks_per_row)) == 1:
        return _uniform_mapping(tasks_per_row)
    return _scan_mapping(tasks_per_row)


def test_uniform_o1_mapping_matches_scan_exhaustively() -> None:
    for rows in range(1, 7):
        for tasks_per_row in itertools.product(range(5), repeat=rows):
            if len(set(tasks_per_row)) == 1:
                assert sorted(_uniform_mapping(tasks_per_row)) == (
                    _scan_mapping(tasks_per_row)
                )


def test_uniform_o1_mapping_queues_same_local_group_across_rows() -> None:
    assert _uniform_mapping((2, 2, 2, 2)) == [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
    ]


def test_uniform_o1_route_is_fail_closed_when_disabled() -> None:
    tasks_per_row = (2, 2, 2)
    assert _fail_closed_mapping(
        tasks_per_row,
        uniform_enabled=False,
    ) == _scan_mapping(tasks_per_row)
    assert _fail_closed_mapping(
        tasks_per_row,
        uniform_enabled=False,
    ) != _uniform_mapping(tasks_per_row)


@pytest.mark.parametrize("grain", [12, 16, 22, 24])
def test_mapping_boundaries_cover_every_local_task_once(grain: int) -> None:
    blocks = (
        grain - 1,
        grain,
        grain + 1,
        2 * grain - 1,
        2 * grain,
        2 * grain + 1,
    )
    tasks_per_row = tuple(
        (block_count + grain - 1) // grain
        for block_count in blocks
    )
    mapping = _scan_mapping(tasks_per_row)
    assert len(mapping) == sum(tasks_per_row)
    assert len(set(mapping)) == len(mapping)
    assert mapping == sorted(mapping)


@pytest.mark.parametrize("grain", [12, 16, 22, 24])
def test_grain_plus_one_heterogeneous_rows_always_fall_back(grain: int) -> None:
    tasks_per_row = tuple(
        (block_count + grain - 1) // grain
        for block_count in (grain, grain + 1)
    )
    assert tasks_per_row == (1, 2)
    expected = [(0, 0), (1, 0), (1, 1)]
    assert _fail_closed_mapping(
        tasks_per_row,
        uniform_enabled=False,
    ) == expected
    assert _fail_closed_mapping(
        tasks_per_row,
        uniform_enabled=True,
    ) == expected


@pytest.mark.parametrize("active_rows", [1, 2, 4, 7, 8, 16])
def test_compact_metadata_gives_every_request_its_own_64k_pages(
    active_rows: int,
) -> None:
    contexts = [65536] * active_rows
    physical_blocks = active_rows * 512 + 15
    _seq, _pos, table, slot = _step_metadata(
        context_lens=contexts,
        num_blocks=512,
        batch=16,
        active_rows=active_rows,
        physical_blocks=physical_blocks,
    )
    table = table.reshape(16, 512)
    for row in range(active_rows):
        block0 = row * 512
        assert table[row].tolist() == list(range(block0, block0 + 512))
        assert int(slot[row]) == (block0 + 512) * 128 - 1
    assert table[active_rows:].count_nonzero().item() == 0


def test_compact_metadata_can_reverse_chronological_page_order() -> None:
    _seq, _pos, table, slot = _step_metadata(
        context_lens=[384],
        num_blocks=4,
        batch=2,
        active_rows=1,
        physical_blocks=4,
        block_table_order="reverse",
    )
    table = table.reshape(2, 4)
    assert table[0].tolist() == [2, 1, 0, 0]
    assert int(slot[0]) == 127
    assert table[1].count_nonzero().item() == 0


def test_compact_metadata_rejects_unknown_page_order() -> None:
    with pytest.raises(ValueError, match="block_table_order"):
        _step_metadata(
            context_lens=[128],
            num_blocks=1,
            batch=1,
            active_rows=1,
            block_table_order="random",
        )


def test_runtime_batch_matrix_preserves_requested_non_power_of_two_order() -> None:
    assert _runtime_active_row_counts(
        "1,2,4,7,8,16",
        default=1,
        capacity=16,
    ) == [1, 2, 4, 7, 8, 16]
    assert _runtime_active_row_counts(
        "",
        default=7,
        capacity=16,
    ) == [7]
    with pytest.raises(ValueError, match="distinct"):
        _runtime_active_row_counts("1,1", default=1, capacity=16)
    with pytest.raises(ValueError, match=r"in \[0,16\]"):
        _runtime_active_row_counts("17", default=1, capacity=16)


def test_alternating_input_audit_is_cold_discriminating_and_restores_a() -> None:
    current = torch.zeros(2, 2, 3, dtype=torch.bfloat16)
    output = torch.empty_like(current)
    input_a = (
        torch.arange(6, dtype=torch.float32)
        .reshape(1, 2, 3)
        .expand(2, -1, -1)
        .clone()
        .bfloat16()
    )
    input_b = (input_a.float() * -2.0 + 0.5).bfloat16()
    calls = []

    def run_once() -> None:
        calls.append(current.clone())
        output.copy_((current.float() * 3.0 + 1.0).bfloat16())

    report = _run_alternating_input_audit(
        run_once=run_once,
        current_hidden=current,
        next_hidden_out=output,
        input_a=input_a,
        input_b=input_b,
        active_rows=2,
        iterations=6,
    )

    assert report["passed"]
    assert report["zero_warmup"]
    assert report["reference_sha256"]["A"] != report["reference_sha256"]["B"]
    assert [entry["variant"] for entry in report["iteration_hashes"]] == [
        "A", "B", "A", "B", "A", "B",
    ]
    assert torch.equal(calls[0], input_a)
    assert torch.equal(calls[1], input_b)
    assert torch.equal(current, input_a)


def test_alternating_input_audit_rejects_partially_unwritten_output() -> None:
    current = torch.zeros(2, 2, 3, dtype=torch.bfloat16)
    output = torch.empty_like(current)
    input_a = torch.ones_like(current)
    input_b = torch.full_like(current, 2.0)

    def run_once() -> None:
        output[:, :, 1:].copy_(current[:, :, 1:])

    with pytest.raises(RuntimeError, match="unwritten/poisoned"):
        _run_alternating_input_audit(
            run_once=run_once,
            current_hidden=current,
            next_hidden_out=output,
            input_a=input_a,
            input_b=input_b,
            active_rows=2,
            iterations=4,
        )
    assert torch.equal(current, input_a)


def test_alternating_input_audit_rejects_stale_variant_publication() -> None:
    current = torch.zeros(2, 2, 3, dtype=torch.bfloat16)
    output = torch.empty_like(current)
    input_a = torch.ones_like(current)
    input_b = torch.full_like(current, 2.0)
    invocation = 0

    def run_once() -> None:
        nonlocal invocation
        invocation += 1
        published = input_b if invocation == 3 else current
        output.copy_(published)

    with pytest.raises(RuntimeError, match="stale/intermittent publication"):
        _run_alternating_input_audit(
            run_once=run_once,
            current_hidden=current,
            next_hidden_out=output,
            input_a=input_a,
            input_b=input_b,
            active_rows=2,
            iterations=4,
        )
    assert torch.equal(current, input_a)


def test_inactive_row_audit_requires_identical_active_output() -> None:
    current = torch.zeros(2, 3, 4, dtype=torch.bfloat16)
    output = torch.empty_like(current)
    input_a = torch.ones_like(current)
    input_b = input_a.clone()
    input_b[:, 2].fill_(99.0)

    def run_once() -> None:
        output.copy_(current)

    report = _run_alternating_input_audit(
        run_once=run_once,
        current_hidden=current,
        next_hidden_out=output,
        input_a=input_a,
        input_b=input_b,
        active_rows=2,
        iterations=4,
        expected_output_relation="same",
        zero_warmup=False,
    )

    assert report["passed"]
    assert not report["zero_warmup"]
    assert report["expected_output_relation"] == "same"
    assert report["reference_sha256"]["A"] == report["reference_sha256"]["B"]


def test_alternating_input_audit_rejects_unchanged_output_tile() -> None:
    current = torch.zeros(2, 2, 128, dtype=torch.bfloat16)
    output = torch.empty_like(current)
    input_a = torch.ones_like(current)
    input_b = torch.full_like(current, 2.0)

    def run_once() -> None:
        output.copy_(current)
        output[:, :, 64:].fill_(7.0)

    with pytest.raises(RuntimeError, match="non-discriminating row/column"):
        _run_alternating_input_audit(
            run_once=run_once,
            current_hidden=current,
            next_hidden_out=output,
            input_a=input_a,
            input_b=input_b,
            active_rows=2,
            iterations=4,
        )


def test_alternating_audit_can_discriminate_external_variant_source() -> None:
    current = torch.ones(2, 1, 128, dtype=torch.bfloat16)
    output = torch.empty_like(current)
    invocation = 0

    def run_once() -> None:
        nonlocal invocation
        scale = 1.0 if invocation % 2 == 0 else 2.0
        invocation += 1
        output.copy_((current.float() * scale).bfloat16())

    report = _run_alternating_input_audit(
        run_once=run_once,
        current_hidden=current,
        next_hidden_out=output,
        input_a=current.clone(),
        input_b=current.clone(),
        active_rows=1,
        iterations=4,
        require_distinct_active_inputs=False,
    )

    assert report["passed"]
    assert report["active_inputs_equal"]
    assert not report["require_distinct_active_inputs"]
    assert report["reference_sha256"]["A"] != report["reference_sha256"]["B"]


def test_full_kv_slot_oracle_applies_bf16_norm_and_rope() -> None:
    hidden = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0]]],
        dtype=torch.bfloat16,
    )
    identity = (
        torch.eye(4, dtype=torch.float32)
        .reshape(1, 4, 4)
        .bfloat16()
    )
    k_row, v_row = _full_kv_slot_oracle(
        hidden=hidden,
        input_rms_weight=torch.zeros(1, 4),
        k_norm_weight=torch.zeros(1, 4),
        wk=identity,
        wv=identity,
        rope_cos=torch.tensor([[0.0, 0.0]]),
        rope_sin=torch.tensor([[1.0, 1.0]]),
        eps=1.0e-5,
    )

    normed = (
        hidden.float()
        * torch.rsqrt(
            hidden.float().pow(2).mean(dim=-1, keepdim=True) + 1.0e-5,
        )
    ).bfloat16()
    k_normed = (
        normed.float()
        * torch.rsqrt(
            normed.float().pow(2).mean(dim=-1, keepdim=True) + 1.0e-5,
        )
    )
    expected_k = torch.cat(
        (
            -k_normed[..., 1:2],
            k_normed[..., 0:1],
            k_normed[..., 2:],
        ),
        dim=-1,
    ).bfloat16()

    assert torch.equal(v_row, normed)
    assert torch.equal(k_row, expected_k)


def test_swa_local_partial_oracle_matches_reverse_table_reference() -> None:
    hidden = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0]],
        dtype=torch.bfloat16,
    )
    wq = torch.zeros(4, 4, dtype=torch.bfloat16)
    wk = torch.zeros(4, 2, dtype=torch.bfloat16)
    wv = torch.eye(4).bfloat16()[:, :2].contiguous()
    wo = torch.eye(4).bfloat16()
    w_g = torch.zeros(4, 2, dtype=torch.bfloat16)
    gate_r = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]],
        dtype=torch.bfloat16,
    )
    seq_lens = torch.tensor([3], dtype=torch.int32)
    block_table = torch.zeros(1, 2, dtype=torch.int32)
    block_table[0, 0] = 1
    block_table[0, 1] = 0
    slot_mapping = torch.tensor([0], dtype=torch.int32)
    rope_cos = torch.ones(3, 2)
    rope_sin = torch.zeros(3, 2)
    k_cache = torch.tensor(
        [[0.25, 0.5], [0.75, 1.0], [1.25, 1.5], [1.75, 2.0]],
        dtype=torch.bfloat16,
    )
    v_cache = torch.tensor(
        [[2.0, 1.75], [1.5, 1.25], [1.0, 0.75], [0.5, 0.25]],
        dtype=torch.bfloat16,
    )

    partial = _torch_swa_local_partial_oracle(
        hidden=hidden,
        input_rms_weight=torch.zeros(4),
        q_norm_weight=torch.zeros(2),
        k_norm_weight=torch.zeros(2),
        wq=wq,
        wk=wk,
        wv=wv,
        wo=wo,
        w_g=w_g,
        gate_r=gate_r,
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=slot_mapping,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        k_cache_layer=k_cache,
        v_cache_layer=v_cache,
        eps=1.0e-5,
        block_size=2,
        sliding_window=3,
        out_proj_k_chunk=2,
        out_proj_n_chunk=2,
    )
    normed = (
        hidden.float()
        * torch.rsqrt(
            hidden.float().pow(2).mean(dim=-1, keepdim=True) + 1.0e-5,
        )
    ).bfloat16()
    current_v = (normed.float() @ wv.float()).bfloat16()[0]
    context = (
        (
            v_cache[2].float()
            + v_cache[3].float()
            + current_v.float()
        )
        / 3.0
    ).bfloat16()
    expected = (
        torch.cat((context, context)).float() * 0.5
    ).bfloat16().reshape(1, 4)

    assert torch.equal(partial, expected)


def test_kv_slot_audit_layout_separates_layers_targets_and_canaries() -> None:
    layout = _kv_slot_audit_layout(
        page_base=512,
        active_rows=2,
        layer_cache_rows=1024 * 128,
    )
    assert len(layout) == 12
    assert [slot["attention_kind"] for slot in layout[:6]] == ["full"] * 6
    assert [slot["attention_kind"] for slot in layout[6:]] == ["swa"] * 6
    assert [slot["role"] for slot in layout[:3]] == [
        "target",
        "left_canary",
        "right_canary",
    ]
    target = layout[0]["cache_row"]
    assert layout[1]["cache_row"] == target - 1
    assert layout[2]["cache_row"] == target + 1
    assert layout[6]["cache_row"] - target == 1024 * 128


def test_tilewise_numerical_report_catches_one_bad_64_column_tile() -> None:
    expected = torch.zeros(8, 7, 128)
    actual = expected.clone()
    actual[0, 0, 0] = 1.0
    aggregate = _tilewise_numerical_report(
        actual=actual,
        expected=expected,
        tile_width=64,
        atol=0.05,
        rtol=0.05,
        max_bad_ratio=0.01,
    )
    assert aggregate["passed"]

    per_rank_row = _tilewise_numerical_report(
        actual=actual[0, 0],
        expected=expected[0, 0],
        tile_width=64,
        atol=0.05,
        rtol=0.05,
        max_bad_ratio=0.01,
    )
    assert not per_rank_row["passed"]
    assert per_rank_row["tiles"][0]["bad_ratio"] == pytest.approx(1.0 / 64.0)


def test_compact_metadata_rejects_small_physical_pool() -> None:
    with pytest.raises(ValueError, match="needs 512 physical blocks"):
        _step_metadata(
            context_lens=[4096] * 16,
            num_blocks=512,
            batch=16,
            active_rows=16,
            physical_blocks=511,
        )


def test_kv_fixture_is_full_shared_deterministic_upload() -> None:
    fixture = _make_kv_fixture(
        total_rows=12,
        layer_cache_rows=4,
        head_dim=8,
        initialized_layers=2,
        cache_kind=0,
        seed=7,
    )
    repeated = _make_kv_fixture(
        total_rows=12,
        layer_cache_rows=4,
        head_dim=8,
        initialized_layers=2,
        cache_kind=0,
        seed=7,
    )
    value_fixture = _make_kv_fixture(
        total_rows=12,
        layer_cache_rows=4,
        head_dim=8,
        initialized_layers=2,
        cache_kind=1,
        seed=7,
    )

    assert fixture.is_shared()
    assert fixture.shape == (12, 8)
    assert (fixture[:8] != 0).all()
    assert (fixture[8:] == 0).all()
    assert fixture.equal(repeated)
    assert not fixture[:8].equal(value_fixture[:8])


def test_tensor_probe_is_bounded_and_detects_both_edges() -> None:
    fixture = _make_kv_fixture(
        total_rows=12,
        layer_cache_rows=4,
        head_dim=8,
        initialized_layers=2,
        cache_kind=0,
        seed=7,
    )
    changed_first = fixture.clone()
    changed_first[0, 0] += 1
    changed_last = fixture.clone()
    changed_last[-1, -1] += 1

    probe = _tensor_probe(fixture, edge_elements=8)
    assert probe["sample_numel"] == 16
    assert probe["sample_sha256"] != _tensor_probe(
        changed_first,
        edge_elements=8,
    )["sample_sha256"]
    assert probe["sample_sha256"] != _tensor_probe(
        changed_last,
        edge_elements=8,
    )["sample_sha256"]


def test_linear_percentile_matches_wall_and_slice_definition() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert _linear_percentile(values, 0.5) == 2.5
    assert _linear_percentile(values, 0.99) == pytest.approx(3.97)


def test_family_span_is_summed_per_invocation_and_packing_is_bounded(
    tmp_path,
) -> None:
    rank_dir = tmp_path / "rank0" / "d0"
    rank_dir.mkdir(parents=True)
    (rank_dir / "deps.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {"task_id": 1, "block_num": 2, "kernel_ids": [7]},
                    {"task_id": 2, "block_num": 2, "kernel_ids": [7]},
                ],
            },
        ),
        encoding="utf-8",
    )
    (rank_dir / "name_map.json").write_text(
        json.dumps({"callable_id_to_name": {"7": "mix_aic"}}),
        encoding="utf-8",
    )
    (rank_dir / "l2_swimlane_records.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "clock_freq_hz": 1_000_000,
                    "num_cores": 4,
                    "core_types": ["aic", "aic", "aiv", "aiv"],
                },
                "aicore_tasks": [
                    [0, 1, 0, 0, 10, 0],
                    [1, 1, 1, 0, 10, 0],
                    [2, 1, 0, 0, 8, 0],
                    [0, 2, 0, 100, 110, 0],
                    [1, 2, 1, 100, 110, 0],
                    [2, 2, 0, 100, 108, 0],
                ],
            },
        ),
        encoding="utf-8",
    )

    report = analyze_uniformity(rank_dir)

    assert report is not None
    aic = report["families"]["mix_aic"]
    assert aic["invocation_count"] == 2
    assert aic["stage_span_us"] == 20.0
    assert aic["resource_slices"] == 4
    assert aic["packing_efficiency"] == 1.0
    assert report["families"]["mix_aiv"]["packing_efficiency"] <= 1.0

    aggregate = _aggregate_rank_uniformity(
        {"rank0/d0": report, "rank1/d0": report},
    )
    assert aggregate["rank_count"] == 2
    assert aggregate["families"]["mix_aic"]["stage_span_us"] == {
        "min": 20.0,
        "median": 20.0,
        "max": 20.0,
    }


def test_collective_low_wait_reference_uses_collective_span_not_makespan() -> None:
    reports = {
        "rank0/d0": {
            "makespan_us": 700.0,
            "families": {
                "tp_all_reduce": {"stage_span_us": 90.0},
                "full_qk_matmul": {"stage_span_us": 16.0},
            },
        },
        "rank1/d0": {
            "makespan_us": 650.0,
            "families": {
                "tp_all_reduce": {"stage_span_us": 120.0},
            },
        },
    }
    reference = _collective_low_wait_reference(reports)
    assert reference is not None
    assert reference["rank_tag"] == "rank0/d0"
    assert reference["tp_all_reduce_stage_span_us"] == 90.0
    assert "heuristic" in reference["interpretation"]
