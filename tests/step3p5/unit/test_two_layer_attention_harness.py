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

from tests.step3p5.harnesses._stage_two_layer_attn import (
    _attention_codegen_contract_errors,
    _aggregate_rank_uniformity,
    _collective_low_wait_reference,
    _linear_percentile,
    _make_kv_fixture,
    _step_metadata,
    _tensor_probe,
    analyze_uniformity,
)


def _valid_attention_codegen_source() -> str:
    return """
// Spmd full_qk_matmul_spmd: full_qk_matmul
L0TaskArgs params_t10;
params_t10.add_scalar(full_qk_active_tasks__rv_v2_inline153);
params_t10.launch_spec.set_block_num(full_qk_active_tasks__rv_v2_inline153);
TaskOutputTensors task_10_outs = rt_submit_aic_task(11, params_t10);
PTO2TaskId full_qk_tid = task_10_outs.task_id();
// Spmd full_softmax_spmd: full_softmax
L0TaskArgs params_t11;
params_t11.add_scalar(full_softmax_active_tasks__rv_v2_inline119);
params_t11.launch_spec.set_block_num(
    full_softmax_active_tasks__rv_v2_inline119);
PTO2TaskId params_t11_deps[1];
uint32_t params_t11_deps_count = 0;
params_t11_deps[params_t11_deps_count++] = full_qk_tid;
params_t11.set_dependencies(params_t11_deps, params_t11_deps_count);
TaskOutputTensors task_11_outs = rt_submit_aiv_task(12, params_t11);
PTO2TaskId full_softmax_tid = task_11_outs.task_id();
// Group full_sv_matmul: MixedKernels (AIC + AIV lanes)
L0TaskArgs params_t12;
params_t12.add_scalar(full_online_softmax_active_tasks__rv_v2_inline149);
params_t12.launch_spec.set_block_num(
    full_online_softmax_active_tasks__rv_v2_inline149);
PTO2TaskId params_t12_deps[1];
uint32_t params_t12_deps_count = 0;
params_t12_deps[params_t12_deps_count++] = full_softmax_tid;
params_t12.set_dependencies(params_t12_deps, params_t12_deps_count);
TaskOutputTensors task_12_outs = rt_submit_task(mixed_12, params_t12);
PTO2TaskId full_sv_online_tid = task_12_outs.task_id();
// Spmd full_online_softmax_reduce_spmd: full_online_softmax_reduce
L0TaskArgs params_t13;
params_t13.add_scalar(
    full_online_softmax_reduce_tasks__rv_v2_inline117);
params_t13.launch_spec.set_block_num(
    full_online_softmax_reduce_tasks__rv_v2_inline117);
PTO2TaskId params_t13_deps[1];
uint32_t params_t13_deps_count = 0;
params_t13_deps[params_t13_deps_count++] = full_sv_online_tid;
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
// Spmd swa_qk_matmul_spmd: swa_qk_matmul
L0TaskArgs params_t34;
params_t34.launch_spec.set_block_num(swa_active_tasks__rv_v2_inline207);
TaskOutputTensors task_34_outs = rt_submit_aic_task(38, params_t34);
PTO2TaskId swa_qk_tid = task_34_outs.task_id();
// Spmd swa_softmax_spmd: swa_softmax
L0TaskArgs params_t35;
params_t35.launch_spec.set_block_num(swa_active_tasks__rv_v2_inline207);
PTO2TaskId params_t35_deps[1];
uint32_t params_t35_deps_count = 0;
params_t35_deps[params_t35_deps_count++] = swa_qk_tid;
params_t35.set_dependencies(params_t35_deps, params_t35_deps_count);
TaskOutputTensors task_35_outs = rt_submit_aiv_task(39, params_t35);
PTO2TaskId swa_softmax_tid = task_35_outs.task_id();
// Spmd swa_sv_matmul_spmd: swa_sv_matmul
L0TaskArgs params_t36;
params_t36.launch_spec.set_block_num(swa_active_tasks__rv_v2_inline207);
PTO2TaskId params_t36_deps[1];
uint32_t params_t36_deps_count = 0;
params_t36_deps[params_t36_deps_count++] = swa_softmax_tid;
params_t36.set_dependencies(params_t36_deps, params_t36_deps_count);
TaskOutputTensors task_36_outs = rt_submit_aic_task(40, params_t36);
PTO2TaskId swa_sv_tid = task_36_outs.task_id();
// Spmd swa_online_softmax_spmd: swa_online_softmax
L0TaskArgs params_t37;
params_t37.launch_spec.set_block_num(swa_active_tasks__rv_v2_inline207);
PTO2TaskId params_t37_deps[1];
uint32_t params_t37_deps_count = 0;
params_t37_deps[params_t37_deps_count++] = swa_sv_tid;
params_t37.set_dependencies(params_t37_deps, params_t37_deps_count);
TaskOutputTensors task_37_outs = rt_submit_aiv_task(41, params_t37);
"""


def test_attention_codegen_contract_accepts_dynamic_bounds_and_task_chain() -> None:
    assert _attention_codegen_contract_errors(
        _valid_attention_codegen_source(),
    ) == []


def test_attention_codegen_contract_rejects_stale_bound_and_dependency() -> None:
    source = _valid_attention_codegen_source().replace(
        "set_block_num(full_qk_active_tasks__rv_v2_inline153)",
        "set_block_num(0)",
    ).replace(
        "= swa_sv_tid;",
        "= stale_swa_task;",
    )
    errors = _attention_codegen_contract_errors(source)
    assert "full QK dynamic launch" in errors
    assert "SWA online softmax dependency from swa_sv_tid" in errors


def test_attention_codegen_contract_rejects_incomplete_dependency_wiring() -> None:
    source = _valid_attention_codegen_source().replace(
        "params_t13.set_dependencies(params_t13_deps, "
        "params_t13_deps_count);",
        "",
    )
    assert (
        "full reduce dependency from full_sv_online_tid"
        in _attention_codegen_contract_errors(source)
    )


def test_attention_codegen_contract_checks_every_swa_stage_bound() -> None:
    source = _valid_attention_codegen_source().replace(
        "params_t36.launch_spec.set_block_num("
        "swa_active_tasks__rv_v2_inline207);",
        "params_t36.launch_spec.set_block_num(1);",
    )
    errors = _attention_codegen_contract_errors(source)
    assert "SWA SV dynamic launch" in errors
    assert "SWA QK dynamic launch" not in errors
    assert "SWA softmax dynamic launch" not in errors
    assert "SWA online softmax dynamic launch" not in errors


def test_attention_codegen_contract_checks_launch_scalar_ssa_identity() -> None:
    source = _valid_attention_codegen_source().replace(
        "params_t12.add_scalar("
        "full_online_softmax_active_tasks__rv_v2_inline149);",
        "params_t12.add_scalar("
        "full_online_softmax_active_tasks__rv_v2_inline0);",
    )
    assert (
        "full SV launch/scalar SSA agreement"
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


def test_compact_metadata_uses_fixed_total_pages_for_bs4() -> None:
    contexts = [16384] * 4
    _seq, _pos, table, slot = _step_metadata(
        context_lens=contexts,
        num_blocks=512,
        batch=16,
        active_rows=4,
        physical_blocks=527,
    )
    table = table.reshape(16, 512)
    assert table[0, :128].tolist() == list(range(0, 128))
    assert table[1, :128].tolist() == list(range(128, 256))
    assert table[2, :128].tolist() == list(range(256, 384))
    assert table[3, :128].tolist() == list(range(384, 512))
    assert slot[:4].tolist() == [
        16383,
        32767,
        49151,
        65535,
    ]


def test_compact_metadata_bs12_64k_uses_512_pages() -> None:
    contexts = [5504] * 8 + [5376] * 4
    _seq, _pos, table, _slot = _step_metadata(
        context_lens=contexts,
        num_blocks=512,
        batch=16,
        active_rows=12,
        physical_blocks=527,
    )
    table = table.reshape(16, 512)
    used = []
    for row, context_len in enumerate(contexts):
        blocks = (context_len + 127) // 128
        used.extend(table[row, :blocks].tolist())
    assert used == list(range(512))


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
