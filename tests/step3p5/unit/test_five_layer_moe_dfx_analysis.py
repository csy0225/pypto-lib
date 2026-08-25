# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Card-free tests for the focused L0-L4 MoE DFX analyzer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.step3p5.analyze_five_layer_moe_dfx import (
    RankTrace,
    Slice,
    Task,
    _CHIP_SWIMLANE_RECORDS_NAME,
    _admission_contract,
    _aggregate_findings,
    _arrival_analysis,
    _arrival_pairs,
    _clock_alignment,
    _combine_dependency_contract,
    _diagnostic_stage_resources,
    _duration_distribution,
    _execution_limit_classification,
    _expert_kernel_release_contract,
    _external_correctness_contract,
    _find_layer_task_ids,
    _local_ep_dependency_contract,
    _local_ep_route_execution_contract,
    _markdown_stage_order,
    _percentile,
    _predicated_skip_task_ids,
    _raw_swimlane_metadata,
    _rank_metrics,
    _route_histogram_contract,
    _routed_slice_profile_contract,
    _source_identity_contract,
    _source_policy,
    _stage_metrics,
    _task_id_contract,
    _timing_evidence_contract,
    _timing_profile_stages,
    _task_timing_evidence,
    _validate_structural_contracts,
)


def _trace(
    tag: str,
    *,
    end_tick: int,
    frequency_hz: int = 50_000_000,
) -> RankTrace:
    return RankTrace(
        tag=tag,
        rank_dir=Path(tag),
        frequency_hz=frequency_hz,
        core_types=["aic"],
        tasks=[],
        task_by_id={},
        slices_by_task={
            "terminal": [
                Slice(
                    core=0,
                    task_id="terminal",
                    start=end_tick - 100,
                    end=end_tick,
                    resource="aic",
                ),
            ],
        },
        edges=[],
        critical_path={},
    )


def _combine_trace() -> RankTrace:
    specs = (
        ("l3-scatter", "swa_moe_chip_orch_combine_scatter"),
        ("l3-wait", "swa_moe_chip_orch_combine_wait"),
        ("l3-reduce", "swa_moe_chip_orch_combine_reduce"),
        ("l4-scatter", "combine_scatter"),
        ("l4-wait", "combine_wait"),
        ("l4-reduce", "combine_reduce"),
    )
    tasks = [
        Task(
            task_id=task_id,
            order=order,
            name=name,
            block_num=1,
            kernel_ids=(-1, order + 1, -1),
            early_dispatch=True,
        )
        for order, (task_id, name) in enumerate(specs)
    ]
    slices_by_task = {
        task.task_id: [
            Slice(
                core=24,
                task_id=task.task_id,
                start=task.order * 20,
                end=task.order * 20 + 10,
                resource="aiv",
            )
        ]
        for task in tasks
    }
    edges = [
        {"pred": "l3-scatter", "succ": "l3-wait", "source": "explicit"},
        {"pred": "l3-wait", "succ": "l3-reduce", "source": "explicit"},
        {
            "pred": "l3-scatter",
            "succ": "l3-reduce",
            "source": "tensormap",
        },
        {"pred": "l4-scatter", "succ": "l4-wait", "source": "explicit"},
        {"pred": "l4-wait", "succ": "l4-reduce", "source": "explicit"},
        {
            "pred": "l4-scatter",
            "succ": "l4-reduce",
            "source": "tensormap",
        },
    ]
    return RankTrace(
        tag="rank0/d0",
        rank_dir=Path("rank0/d0"),
        frequency_hz=1_000_000,
        core_types=["aic"] * 24 + ["aiv"] * 48,
        tasks=tasks,
        task_by_id={task.task_id: task for task in tasks},
        slices_by_task=slices_by_task,
        edges=edges,
        critical_path={},
        swimlane_level=4,
    )


def _shared_trace(*, layer: str, split: bool) -> RankTrace:
    prefix = "swa_moe_chip_orch_" if layer == "L3" else ""
    shared_specs = (
        (
            ("shared-old", f"{prefix}sh_mlp", 1),
        )
        if not split
        else (
            ("shared-mm", f"{prefix}sh_gate_up_mm", 5),
            ("shared-act", f"{prefix}sh_gate_up_act", 5),
            ("shared-down", f"{prefix}sh_down", 16),
        )
    )
    specs = (
        *shared_specs,
        ("shared-ar", "tp_all_reduce", 1),
        ("dispatch", f"{prefix}dispatch_meta", 1),
    )
    tasks = [
        Task(
            task_id=task_id,
            order=order,
            name=name,
            block_num=blocks,
            kernel_ids=(order + 1, order + 101, -1),
            early_dispatch=True,
        )
        for order, (task_id, name, blocks) in enumerate(specs)
    ]
    return RankTrace(
        tag="rank0/d0",
        rank_dir=Path("rank0/d0"),
        frequency_hz=1_000_000,
        core_types=["aic"] * 24 + ["aiv"] * 48,
        tasks=tasks,
        task_by_id={task.task_id: task for task in tasks},
        slices_by_task={},
        edges=[],
        critical_path={},
        swimlane_level=4,
    )


def _packed_nz_trace(*, skip_l4: bool = False) -> RankTrace:
    specs = (
        ("l3-gather", "swa_moe_chip_orch_dispatch_gather", (-1, 10, -1)),
        (
            "l3-fused",
            "routed_nz_gmm1_swiglu_quant_aic",
            (11, 12, 12),
        ),
        ("l3-down", "routed_nz_down_aic", (13, 14, 14)),
        ("l3-scatter", "swa_moe_chip_orch_combine_scatter", (-1, 15, -1)),
        ("l4-gather", "dispatch_gather", (-1, 20, -1)),
        (
            "l4-fused",
            "routed_nz_gmm1_swiglu_quant_aic",
            (11, 12, 12),
        ),
        ("l4-down", "routed_nz_down_aic", (13, 14, 14)),
        ("l4-scatter", "combine_scatter", (-1, 21, -1)),
    )
    tasks = [
        Task(
            task_id=task_id,
            order=order,
            name=name,
            block_num=1,
            kernel_ids=kernel_ids,
            early_dispatch=True,
        )
        for order, (task_id, name, kernel_ids) in enumerate(specs)
    ]
    skipped = {"l4-fused", "l4-down"} if skip_l4 else set()
    slices_by_task = {
        task.task_id: [
            Slice(
                core=(
                    0
                    if task.kernel_ids[0] >= 0
                    else 24
                ),
                task_id=task.task_id,
                start=task.order * 100,
                end=task.order * 100 + 10,
                resource=(
                    "aic"
                    if task.kernel_ids[0] >= 0
                    else "aiv"
                ),
            )
        ]
        for task in tasks
        if task.task_id not in skipped
    }
    edges = [
        {"pred": "l3-gather", "succ": "l3-fused", "source": "tensormap"},
        {"pred": "l3-fused", "succ": "l3-down", "source": "explicit"},
        {"pred": "l3-down", "succ": "l3-scatter", "source": "explicit"},
        {"pred": "l4-gather", "succ": "l4-fused", "source": "tensormap"},
        {"pred": "l4-fused", "succ": "l4-down", "source": "explicit"},
        {"pred": "l4-down", "succ": "l4-scatter", "source": "explicit"},
    ]
    return RankTrace(
        tag="rank0/d0",
        rank_dir=Path("rank0/d0"),
        frequency_hz=1_000_000,
        core_types=["aic"] * 24 + ["aiv"] * 48,
        tasks=tasks,
        task_by_id={task.task_id: task for task in tasks},
        slices_by_task=slices_by_task,
        edges=edges,
        critical_path={},
        swimlane_level=4,
        predicated_skip_task_ids=tuple(sorted(skipped)),
    )


def _local_ep_trace(
    *,
    skip_layer: str | None = None,
) -> RankTrace:
    specs = (
        ("l3-shared", "swa_moe_chip_orch_sh_down", (1, 2, -1)),
        (
            "l3-pack",
            "swa_moe_chip_orch_local_route_pack",
            (-1, 3, -1),
        ),
        (
            "l3-fused",
            "routed_nz_gmm1_swiglu_quant_aic",
            (4, 5, 5),
        ),
        ("l3-down", "routed_nz_down_aic", (6, 7, 7)),
        (
            "l3-combine",
            "swa_moe_chip_orch_local_combine_reduce",
            (-1, 8, -1),
        ),
        ("l3-ar", "tp_all_reduce", (-1, 9, -1)),
        (
            "l3-residual",
            "swa_moe_chip_orch_moe_residual_add",
            (-1, 10, -1),
        ),
        ("l4-shared", "sh_down", (11, 12, -1)),
        ("l4-pack", "local_route_pack", (-1, 13, -1)),
        (
            "l4-fused",
            "routed_nz_gmm1_swiglu_quant_aic",
            (14, 15, 15),
        ),
        ("l4-down", "routed_nz_down_aic", (16, 17, 17)),
        ("l4-combine", "local_combine_reduce", (-1, 18, -1)),
        ("l4-ar", "tp_all_reduce", (-1, 19, -1)),
        ("l4-residual", "moe_residual_add", (-1, 20, -1)),
    )
    tasks = [
        Task(
            task_id=task_id,
            order=order,
            name=name,
            block_num=1,
            kernel_ids=kernel_ids,
            early_dispatch=True,
        )
        for order, (task_id, name, kernel_ids) in enumerate(specs)
    ]
    skipped = (
        {f"{skip_layer.lower()}-fused", f"{skip_layer.lower()}-down"}
        if skip_layer is not None
        else set()
    )
    slices_by_task = {
        task.task_id: [
            Slice(
                core=0 if task.kernel_ids[0] >= 0 else 24,
                task_id=task.task_id,
                start=task.order * 100,
                end=task.order * 100 + 10,
                resource=(
                    "aic" if task.kernel_ids[0] >= 0 else "aiv"
                ),
            )
        ]
        for task in tasks
        if task.task_id not in skipped
    }
    edges = []
    for layer in ("l3", "l4"):
        edges.extend(
            [
                {
                    "pred": f"{layer}-pack",
                    "succ": f"{layer}-fused",
                    "source": "explicit",
                },
                {
                    "pred": f"{layer}-fused",
                    "succ": f"{layer}-down",
                    "source": "explicit",
                },
                {
                    "pred": f"{layer}-pack",
                    "succ": f"{layer}-combine",
                    "source": "explicit",
                },
                {
                    "pred": f"{layer}-down",
                    "succ": f"{layer}-combine",
                    "source": "explicit",
                },
                {
                    "pred": f"{layer}-shared",
                    "succ": f"{layer}-combine",
                    "source": "tensormap",
                },
                {
                    "pred": f"{layer}-combine",
                    "succ": f"{layer}-ar",
                    "source": "tensormap",
                },
                {
                    "pred": f"{layer}-ar",
                    "succ": f"{layer}-residual",
                    "source": "tensormap",
                },
            ]
        )
    return RankTrace(
        tag="rank0/d0",
        rank_dir=Path("rank0/d0"),
        frequency_hz=1_000_000,
        core_types=["aic"] * 24 + ["aiv"] * 48,
        tasks=tasks,
        task_by_id={task.task_id: task for task in tasks},
        slices_by_task=slices_by_task,
        edges=edges,
        critical_path={},
        swimlane_level=4,
        predicated_skip_task_ids=tuple(sorted(skipped)),
    )


def _fake_resource(
    *,
    p50_us: float = 20.0,
    p90_us: float = 25.0,
    p99_us: float = 40.0,
    max_us: float = 50.0,
    observed_slices: int = 24,
    available_cores: int = 24,
    peak_concurrency: int = 24,
    expected_slices: int | None = None,
    distinct_cores: int | None = None,
) -> dict:
    return {
        "available": observed_slices > 0,
        "expected_slices": (
            observed_slices
            if expected_slices is None
            else expected_slices
        ),
        "observed_slices": observed_slices,
        "distinct_cores": (
            observed_slices
            if distinct_cores is None
            else distinct_cores
        ),
        "available_cores": available_cores,
        "peak_concurrency": peak_concurrency,
        "duration_distribution": {
            "available": observed_slices > 0,
            "count": observed_slices,
            "p50_us": p50_us,
            "p90_us": p90_us,
            "p99_us": p99_us,
            "max_us": max_us,
        },
    }


def _packed_nz_stage(aic: int, aiv: int) -> dict:
    return {
        "task_instances": 1,
        "blocks_per_task": [aic],
        "resources": {
            "aic": _fake_resource(
                observed_slices=aic,
                available_cores=24,
                peak_concurrency=aic,
            ),
            "aiv": _fake_resource(
                observed_slices=aiv,
                available_cores=48,
                peak_concurrency=aiv,
            ),
        },
        "task_instance_details": [],
    }


def _valid_packed_nz_rank() -> dict:
    return {
        "layers": {
            layer: {
                "expert_gate_up": _packed_nz_stage(24, 48),
                "expert_down": _packed_nz_stage(23, 46),
            }
            for layer in ("L3", "L4")
        }
    }


def _fake_stage(
    *,
    resource: str = "aic",
    queue_delay_us: float = 0.0,
    **resource_overrides,
) -> dict:
    empty = _fake_resource(observed_slices=0, peak_concurrency=0)
    profiled = _fake_resource(**resource_overrides)
    return {
        "resources": {
            "aic": profiled if resource == "aic" else empty,
            "aiv": profiled if resource == "aiv" else empty,
        },
        "task_instance_details": [
            {
                "task_id": "task-0",
                "timing_evidence": {
                    "queue_delay": {
                        "available": True,
                        "value_us": queue_delay_us,
                    }
                },
            }
        ],
    }


def _valid_expert_rank() -> dict:
    stages = {
        "expert_gate_up": _fake_stage(),
        "expert_gate": _fake_stage(),
        "expert_up": _fake_stage(),
        "expert_down": _fake_stage(),
        "expert_gate_up_act": _fake_stage(resource="aiv"),
        "routed_h_quant": _fake_stage(resource="aiv"),
    }
    return {
        "layers": {
            "L3": dict(stages),
            "L4": {
                "expert_gate_up": _fake_stage(),
                "expert_gate": _fake_stage(),
                "expert_up": _fake_stage(),
                "expert_down": _fake_stage(),
                "expert_gate_up_act": _fake_stage(resource="aiv"),
                "routed_h_quant": _fake_stage(resource="aiv"),
            },
        }
    }


def _local_owner_payload() -> dict:
    owner_route_counts = [
        [
            [[0 for _field in range(40)] for _route_owner in range(8)]
            for _owner in range(8)
        ]
        for _layer in range(2)
    ]
    owner_route_counts[0][0][0][0] = 5
    owner_route_counts[0][2][2][2] = 3
    owner_route_counts[1][2][2][6] = 4
    owner_route_counts[1][7][7][35] = 4
    local_expert_count = [
        [
            owner_route_counts[layer][owner][owner][:36]
            for owner in range(8)
        ]
        for layer in range(2)
    ]
    checkpoint_files = {
        "config.json": {"size_bytes": 1, "sha256": "1" * 64},
        "quant_model_weights.safetensors.index.json": {
            "size_bytes": 2,
            "sha256": "2" * 64,
        },
        "weights.safetensors": {
            "size_bytes": 3,
            "sha256": "3" * 64,
        },
    }
    source = {
        "source_tree_manifest_sha256": "4" * 64,
        "decode_fwd_sha256": "5" * 64,
        "formal_program_sha256": "6" * 64,
        "route_program_sha256": "7" * 64,
        "route_holder_sha256": "8" * 64,
        "route_stage_sha256": "9" * 64,
    }
    input_contract = {
        "workload": {
            "active_batch": 1,
            "context_len": 65536,
            "context_semantics": "per_active_sequence",
        },
        "input_tokens": [6127],
        "tensor_sha256": {
            "active_hidden": "a" * 64,
            "seq_lens": "b" * 64,
            "positions": "c" * 64,
            "block_table": "d" * 64,
            "slot_mapping": "e" * 64,
        },
    }

    def json_sha256(value) -> str:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    return {
        "schema": "step3p5.five-layer-moe-local-routes.v2",
        "layers": ["L3", "L4"],
        "axes": [
            "layer",
            "owner_rank",
            "route_owner_rank",
            "local_expert_pad",
        ],
        "owner_route_counts": owner_route_counts,
        "local_expert_count": local_expert_count,
        "snapshot_provenance": [
            {
                "layer": "L3",
                "snapshot_id": "local-owner-routes-l3",
                "shape": [8, 40],
                "dtype": "int32",
                "byte_size": 1280,
                "source_tensor": "local_expert_count",
                "source_protocol": "replicated_input_local_owner",
                "capture_point": "after_l3_before_l4",
            },
            {
                "layer": "L4",
                "snapshot_id": "local-owner-routes-l4",
                "shape": [8, 40],
                "dtype": "int32",
                "byte_size": 1280,
                "source_tensor": "local_expert_count",
                "source_protocol": "replicated_input_local_owner",
                "capture_point": "after_l4",
            },
        ],
        "provenance": {
            "image_digest": "image@sha256:" + "f" * 64,
            "checkpoint": {
                "schema": "step3p5.checkpoint-identity.v1",
                "logical_id": "checkpoint",
                "index_file": (
                    "quant_model_weights.safetensors.index.json"
                ),
                "weight_tensor_count": 10,
                "weight_shard_count": 1,
                "files": checkpoint_files,
                "identity_sha256": json_sha256(checkpoint_files),
            },
            "source": source,
            "source_manifest_sha256": json_sha256(source),
            "input_contract": input_contract,
            "input_contract_sha256": json_sha256(input_contract),
            "formal_golden": {
                "schema": "step3p5.five-layer-moe-golden.v3",
                "manifest_sha256": "0" * 64,
                "source_run": "baseline-r1-normal-bs1-64k",
                "source_kind": "baseline",
                "source_decode_fwd_sha256": "5" * 64,
                "source_manifest_sha256": "4" * 64,
                "active_batch": 1,
                "context_len_per_sequence": 65536,
                "image_ref": "image@sha256:" + "f" * 64,
                "files": {
                    "hidden_l3.pt": "1" * 64,
                    "hidden_l4.pt": "2" * 64,
                },
                "bit_exact": True,
            },
        },
    }


def test_clock_alignment_never_infers_a_common_clock_from_terminals() -> None:
    aligned = _clock_alignment(
        [
            _trace("rank0/d0", end_tick=1_000_000),
            _trace("rank1/d0", end_tick=1_000_250),
        ],
    )
    assert aligned["same_frequency"]
    assert aligned["terminal_end_skew_us"] is None
    assert not aligned["terminal_end_skew_computed"]
    assert not aligned["cross_rank_tick_math_enabled"]
    assert aligned["external_common_clock_anchor"] is None

    misaligned = _clock_alignment(
        [
            _trace("rank0/d0", end_tick=1_000_000),
            _trace("rank1/d0", end_tick=1_010_000),
        ],
    )
    assert misaligned["terminal_end_skew_us"] is None
    assert not misaligned["cross_rank_tick_math_enabled"]

    mixed_frequency = _clock_alignment(
        [
            _trace("rank0/d0", end_tick=1_000_000),
            _trace(
                "rank1/d0",
                end_tick=1_000_000,
                frequency_hz=1_000_000_000,
            ),
        ],
    )
    assert not mixed_frequency["same_frequency"]
    assert not mixed_frequency["cross_rank_tick_math_enabled"]


def test_raw_swimlane_metadata_uses_full_hardware_capacity(tmp_path) -> None:
    path = tmp_path / _CHIP_SWIMLANE_RECORDS_NAME
    path.write_text(
        json.dumps(
            {
                "chip_swimlane_level": 4,
                "metadata": {
                    "num_cores": 72,
                    "core_types": ["aic"] * 24 + ["aiv"] * 48,
                },
            }
        ),
        encoding="utf-8",
    )
    level, core_types = _raw_swimlane_metadata(path)
    assert level == 4
    assert core_types.count("aic") == 24
    assert core_types.count("aiv") == 48


def test_predicated_skip_task_ids_use_explicit_scheduler_evidence() -> None:
    assert _predicated_skip_task_ids(
        {
            "aicpu_scheduler_phases": [
                [
                    {"phase": "dispatch", "tasks_processed": 1},
                    {
                        "phase": "predicated_skip",
                        "task_id": 8589934713,
                    },
                ],
                [
                    {
                        "phase": "predicated_skip",
                        "task_id": "8589934714",
                    },
                ],
            ],
        }
    ) == ("8589934713", "8589934714")
    assert _predicated_skip_task_ids({}) == ()


def test_stage_metrics_separate_dag_task_span_from_core_slices() -> None:
    task = Task(
        task_id="7",
        order=3,
        name="expert_gate_mm",
        block_num=2,
        kernel_ids=(11, 12, 13),
        early_dispatch=True,
    )
    slices = [
        Slice(core=0, task_id="7", start=0, end=10, resource="aic"),
        Slice(core=1, task_id="7", start=3, end=15, resource="aic"),
        Slice(core=24, task_id="7", start=1, end=8, resource="aiv"),
        Slice(core=25, task_id="7", start=2, end=9, resource="aiv"),
        Slice(core=26, task_id="7", start=20, end=28, resource="aiv"),
        Slice(core=27, task_id="7", start=21, end=30, resource="aiv"),
    ]
    trace = RankTrace(
        tag="rank0/d0",
        rank_dir=Path("rank0/d0"),
        frequency_hz=1_000_000,
        core_types=["aic"] * 24 + ["aiv"] * 48,
        tasks=[task],
        task_by_id={"7": task},
        slices_by_task={"7": slices},
        edges=[],
        critical_path={},
        swimlane_level=4,
    )
    metrics = _stage_metrics(trace, ["7"])
    assert metrics is not None
    detail = metrics["task_instance_details"][0]
    assert detail["dag_task_span_us"] == 30.0
    assert detail["resources"]["aic"]["slice_duration_us"] == [10.0, 12.0]
    assert detail["resources"]["aic"]["expected_slices"] == 2
    assert detail["resources"]["aiv"]["expected_slices"] == 4
    assert metrics["resources"]["aic"]["available_cores"] == 24
    assert metrics["resources"]["aiv"]["available_cores"] == 48
    assert len(metrics["resources"]["aic"]["physical_slices"]) == 2
    assert metrics["resources"]["aic"]["duration_distribution"]["gates"] == {
        "lt_10_us": {"count": 0, "ratio": 0.0},
        "from_10_to_30_us": {"count": 2, "ratio": 1.0},
        "gt_30_us": {"count": 0, "ratio": 0.0},
    }
    profile_contract = _routed_slice_profile_contract(
        {
            "rank0/d0": {
                "layers": {
                    "L3": {"expert_gate": metrics},
                    "L4": {},
                }
            }
        }
    )
    assert profile_contract["pass"]
    aic_profile = (
        profile_contract["coverage"]["L3"]["rank0/d0"]["expert_gate"]["resources"]["aic"]
    )
    assert aic_profile["observed_slices"] == 2
    assert aic_profile["physical_record_count"] == 2
    assert aic_profile["duration_gate_count"] == 2


@pytest.mark.parametrize("layer", ["L3", "L4"])
@pytest.mark.parametrize("split", [False, True])
def test_find_layer_task_ids_supports_old_and_split_shared_layouts(
    layer: str,
    split: bool,
) -> None:
    stages = _find_layer_task_ids(
        _shared_trace(layer=layer, split=split),
        layer,
    )
    assert stages["shared_all_reduce"] == ["shared-ar"]
    if split:
        assert stages["shared_mlp"] == []
        assert stages["shared_gate_up"] == ["shared-mm"]
        assert stages["shared_gate_up_act"] == ["shared-act"]
        assert stages["shared_down"] == ["shared-down"]
        assert stages["shared_split"] == [
            "shared-mm",
            "shared-act",
            "shared-down",
        ]
    else:
        assert stages["shared_mlp"] == ["shared-old"]
        assert stages["shared_gate_up"] == []
        assert stages["shared_gate_up_act"] == []
        assert stages["shared_down"] == []
        assert stages["shared_split"] == []


def test_packed_nz_external_tasks_are_mapped_by_layer_window_and_task_id() -> None:
    trace = _packed_nz_trace()

    l3 = _find_layer_task_ids(trace, "L3")
    l4 = _find_layer_task_ids(trace, "L4")
    assert l3["expert_gate_up"] == ["l3-fused"]
    assert l3["expert_down"] == ["l3-down"]
    assert l4["expert_gate_up"] == ["l4-fused"]
    assert l4["expert_down"] == ["l4-down"]

    metrics = _rank_metrics(trace)["layers"]
    assert metrics["L3"]["expert_gate_up"]["task_ids"] == ["l3-fused"]
    assert metrics["L3"]["expert_gate_up"]["task_instances"] == 1
    assert metrics["L4"]["expert_gate_up"]["task_ids"] == ["l4-fused"]
    assert metrics["L4"]["expert_gate_up"]["task_instances"] == 1


def test_packed_nz_mapping_accepts_route_empty_predicated_skips() -> None:
    trace = _packed_nz_trace(skip_l4=True)

    l4 = _find_layer_task_ids(trace, "L4")
    assert l4["expert_gate_up"] == ["l4-fused"]
    assert l4["expert_down"] == ["l4-down"]
    assert "expert_gate_up" not in _rank_metrics(trace)["layers"]["L4"]

    task_ids = _task_id_contract(trace)
    assert task_ids["pass"]
    assert task_ids["predicated_skip_without_physical_slices"] == [
        "l4-down",
        "l4-fused",
    ]


def test_packed_nz_mapping_rejects_a_broken_dependency_chain() -> None:
    trace = _packed_nz_trace()
    trace.edges = [
        edge
        for edge in trace.edges
        if not (
            edge["pred"] == "l4-fused"
            and edge["succ"] == "l4-down"
        )
    ]

    with pytest.raises(RuntimeError, match="invalid packed-NZ dependency chain"):
        _find_layer_task_ids(trace, "L4")


def test_local_ep_external_tasks_are_mapped_by_pack_combine_window() -> None:
    trace = _local_ep_trace()

    l3 = _find_layer_task_ids(trace, "L3", "local-ep")
    l4 = _find_layer_task_ids(trace, "L4", "local-ep")

    assert l3["local_route_pack"] == ["l3-pack"]
    assert l3["expert_gate_up"] == ["l3-fused"]
    assert l3["expert_down"] == ["l3-down"]
    assert l3["local_combine_reduce"] == ["l3-combine"]
    assert l3["moe_all_reduce"] == ["l3-ar"]
    assert l3["moe_residual_add"] == ["l3-residual"]
    assert l4["local_route_pack"] == ["l4-pack"]
    assert l4["expert_gate_up"] == ["l4-fused"]
    assert l4["expert_down"] == ["l4-down"]
    assert l4["local_combine_reduce"] == ["l4-combine"]
    assert l4["moe_all_reduce"] == ["l4-ar"]
    assert l4["moe_residual_add"] == ["l4-residual"]

    metrics = _rank_metrics(trace, "local-ep")["layers"]
    assert metrics["L3"]["local_route_pack"]["task_ids"] == ["l3-pack"]
    assert metrics["L3"]["moe_all_reduce"]["task_ids"] == ["l3-ar"]
    assert metrics["L4"]["local_combine_reduce"]["task_ids"] == [
        "l4-combine"
    ]


def test_local_ep_dependency_contract_accepts_complete_two_layer_chain() -> None:
    trace = _local_ep_trace()

    contract = _local_ep_dependency_contract(trace)

    assert contract["pass"]
    assert _validate_structural_contracts(
        [trace],
        "local-ep",
    )["pass"]
    for layer in ("L3", "L4"):
        layer_contract = contract["layers"][layer]
        assert layer_contract["pass"]
        assert layer_contract["task_order"]["pass"]
        assert all(
            edge["pass"]
            for edge in layer_contract["required_edges"].values()
        )
        assert all(
            item["pass"]
            for item in layer_contract["execution"].values()
        )


def test_local_ep_dependency_contract_allows_wider_pack_grid() -> None:
    trace = _local_ep_trace()
    original = trace.task_by_id["l3-pack"]
    wide_pack = Task(
        task_id=original.task_id,
        order=original.order,
        name=original.name,
        block_num=36,
        kernel_ids=original.kernel_ids,
        early_dispatch=original.early_dispatch,
    )
    trace.tasks = [
        wide_pack if task.task_id == wide_pack.task_id else task
        for task in trace.tasks
    ]
    trace.task_by_id[wide_pack.task_id] = wide_pack

    contract = _local_ep_dependency_contract(trace)

    assert contract["pass"]
    l3_pack = contract["layers"]["L3"]["execution"][
        "local_route_pack"
    ]
    assert l3_pack["pass"]
    assert l3_pack["task_count"] == 1
    assert trace.task_by_id["l3-pack"].block_num == 36


def test_local_ep_dependency_contract_rejects_orphan_pack_producer() -> None:
    trace = _local_ep_trace()
    extra_pack = Task(
        task_id="l3-pack-wide",
        order=1,
        name="swa_moe_chip_orch_local_route_pack",
        block_num=36,
        kernel_ids=(-1, 21, -1),
        early_dispatch=True,
    )
    trace.tasks.append(extra_pack)
    trace.task_by_id[extra_pack.task_id] = extra_pack
    trace.slices_by_task[extra_pack.task_id] = [
        Slice(24, extra_pack.task_id, 100, 110, "aiv")
    ]

    contract = _local_ep_dependency_contract(trace)

    assert not contract["pass"]
    required_edges = contract["layers"]["L3"]["required_edges"]
    assert not required_edges["pack_to_expert_explicit"]["pass"]
    assert not required_edges["pack_to_combine_explicit"]["pass"]
    assert (
        required_edges["pack_to_expert_explicit"][
            "matches_by_pred_task"
        ][extra_pack.task_id]
        == []
    )


def test_local_ep_dependency_contract_accepts_connected_pack_producers() -> None:
    trace = _local_ep_trace()
    extra_pack = Task(
        task_id="l3-pack-wide",
        order=1,
        name="swa_moe_chip_orch_local_route_pack",
        block_num=36,
        kernel_ids=(-1, 21, -1),
        early_dispatch=True,
    )
    trace.tasks.append(extra_pack)
    trace.task_by_id[extra_pack.task_id] = extra_pack
    trace.slices_by_task[extra_pack.task_id] = [
        Slice(24, extra_pack.task_id, 100, 110, "aiv")
    ]
    trace.edges.extend(
        [
            {
                "pred": extra_pack.task_id,
                "succ": "l3-fused",
                "source": "explicit",
            },
            {
                "pred": extra_pack.task_id,
                "succ": "l3-combine",
                "source": "explicit",
            },
        ]
    )

    contract = _local_ep_dependency_contract(trace)

    assert contract["pass"]
    assert contract["layers"]["L3"]["stage_task_counts"][
        "local_route_pack"
    ] == 2


@pytest.mark.parametrize(
    ("pred", "succ", "edge_name"),
    [
        ("l4-pack", "l4-fused", "pack_to_expert_explicit"),
        ("l4-fused", "l4-down", "expert_to_down_explicit"),
        ("l4-pack", "l4-combine", "pack_to_combine_explicit"),
        ("l4-down", "l4-combine", "down_to_combine_explicit"),
        (
            "l4-shared",
            "l4-combine",
            "shared_down_to_combine_data",
        ),
        ("l4-combine", "l4-ar", "combine_to_all_reduce_data"),
        ("l4-ar", "l4-residual", "all_reduce_to_residual_data"),
    ],
)
def test_local_ep_dependency_contract_rejects_missing_required_edges(
    pred: str,
    succ: str,
    edge_name: str,
) -> None:
    trace = _local_ep_trace()
    trace.edges = [
        edge
        for edge in trace.edges
        if not (
            edge["pred"] == pred
            and edge["succ"] == succ
        )
    ]

    contract = _local_ep_dependency_contract(trace)

    assert not contract["pass"]
    assert not contract["layers"]["L4"]["required_edges"][edge_name]["pass"]
    with pytest.raises(RuntimeError, match="local_ep_dependency"):
        _validate_structural_contracts([trace], "local-ep")


def test_local_ep_zero_route_allows_only_expert_predicate_skips() -> None:
    trace = _local_ep_trace(skip_layer="L4")

    contract = _local_ep_dependency_contract(trace)

    assert contract["pass"]
    l4_execution = contract["layers"]["L4"]["execution"]
    assert l4_execution["expert_gate_up"]["predicated_skip"]
    assert l4_execution["expert_down"]["predicated_skip"]
    for stage in (
        "local_route_pack",
        "local_combine_reduce",
        "moe_all_reduce",
        "moe_residual_add",
    ):
        assert l4_execution[stage]["has_physical_slices"]
        assert not l4_execution[stage]["predicated_skip"]

    trace.slices_by_task["l4-combine"] = []
    trace.predicated_skip_task_ids += ("l4-combine",)
    rejected = _local_ep_dependency_contract(trace)
    assert not rejected["pass"]
    assert not rejected["layers"]["L4"]["execution"][
        "local_combine_reduce"
    ]["pass"]


def test_local_ep_route_execution_matches_zero_and_nonzero_routes() -> None:
    trace = _local_ep_trace(skip_layer="L4")
    structural = {
        "local_ep_dependency": {
            "rank0/d0": _local_ep_dependency_contract(trace),
        }
    }
    route_histogram = {
        "L3": {
            "available": True,
            "histogram": {
                "rank0/d0": [1] + [0] * 35,
            },
        },
        "L4": {
            "available": True,
            "histogram": {
                "rank0/d0": [0] * 36,
            },
        },
    }

    contract = _local_ep_route_execution_contract(
        structural,
        route_histogram,
        "local-ep",
    )

    assert contract["available"]
    assert contract["pass"]
    assert contract["layers"]["L3"]["ranks"]["rank0/d0"][
        "expected_expert_state"
    ] == "executed"
    assert contract["layers"]["L4"]["ranks"]["rank0/d0"][
        "expected_expert_state"
    ] == "predicated_skip"
    for layer in ("L3", "L4"):
        rank = contract["layers"][layer]["ranks"]["rank0/d0"]
        assert all(
            stage["pass"]
            for stage in rank["expert_stages"].values()
        )
        assert all(
            stage["pass"]
            for stage in rank["mandatory_stages"].values()
        )


@pytest.mark.parametrize(
    ("skip_layer", "l4_route_total", "expected_state"),
    [
        (None, 0, "predicated_skip"),
        ("L4", 1, "executed"),
    ],
)
def test_local_ep_route_execution_rejects_route_state_mismatch(
    skip_layer: str | None,
    l4_route_total: int,
    expected_state: str,
) -> None:
    trace = _local_ep_trace(skip_layer=skip_layer)
    structural = {
        "local_ep_dependency": {
            "rank0/d0": _local_ep_dependency_contract(trace),
        }
    }
    route_histogram = {
        "L3": {
            "available": True,
            "histogram": {
                "rank0/d0": [1] + [0] * 35,
            },
        },
        "L4": {
            "available": True,
            "histogram": {
                "rank0/d0": [l4_route_total] + [0] * 35,
            },
        },
    }

    contract = _local_ep_route_execution_contract(
        structural,
        route_histogram,
        "local-ep",
    )

    assert contract["available"]
    assert not contract["pass"]
    rank = contract["layers"]["L4"]["ranks"]["rank0/d0"]
    assert rank["expected_expert_state"] == expected_state
    assert {
        error["stage"]
        for error in rank["errors"]
        if error["code"] == "expert_execution_route_mismatch"
    } == {"expert_gate_up", "expert_down"}


def test_local_ep_route_execution_requires_mandatory_stages() -> None:
    trace = _local_ep_trace(skip_layer="L4")
    trace.slices_by_task["l4-ar"] = []
    trace.predicated_skip_task_ids += ("l4-ar",)
    structural = {
        "local_ep_dependency": {
            "rank0/d0": _local_ep_dependency_contract(trace),
        }
    }
    route_histogram = {
        layer: {
            "available": True,
            "histogram": {
                "rank0/d0": (
                    [1] + [0] * 35
                    if layer == "L3"
                    else [0] * 36
                ),
            },
        }
        for layer in ("L3", "L4")
    }

    contract = _local_ep_route_execution_contract(
        structural,
        route_histogram,
        "local-ep",
    )

    assert not contract["pass"]
    errors = contract["layers"]["L4"]["ranks"]["rank0/d0"]["errors"]
    assert any(
        error["code"] == "mandatory_stage_not_executed"
        and error["stage"] == "moe_all_reduce"
        for error in errors
    )


def test_local_ep_route_execution_waits_for_exact_route_evidence() -> None:
    contract = _local_ep_route_execution_contract(
        {"local_ep_dependency": {}},
        _route_histogram_contract(),
        "local-ep",
    )

    assert contract["applicable"]
    assert not contract["available"]
    assert contract["pass"] is None


def test_local_ep_swim_envelope_overlap_is_diagnostic() -> None:
    trace = _local_ep_trace()
    trace.slices_by_task["l4-fused"] = [
        Slice(0, "l4-fused", 920, 1110, "aic")
    ]

    contract = _local_ep_dependency_contract(trace)

    assert contract["pass"]
    edge = contract["layers"]["L4"]["local_swim_order"]["edges"][
        "expert_to_down_explicit"
    ]
    assert edge["pass"]
    assert edge["envelopes_overlap"]
    assert "allow_early_resolve" in edge["semantics"]


def test_duration_distribution_uses_closed_10_to_30_us_gate() -> None:
    distribution = _duration_distribution([9.999, 10.0, 30.0, 30.001])
    assert distribution["min_us"] == 9.999
    assert distribution["p50_us"] == 10.0
    assert distribution["p90_us"] == 30.001
    assert distribution["p99_us"] == 30.001
    assert distribution["max_us"] == 30.001
    assert distribution["gates"] == {
        "lt_10_us": {"count": 1, "ratio": 0.25},
        "from_10_to_30_us": {"count": 2, "ratio": 0.5},
        "gt_30_us": {"count": 1, "ratio": 0.25},
    }


def test_percentile_uses_conservative_nearest_rank_for_small_samples() -> None:
    values = [10.0, 100.0]
    assert _percentile([], 0.99) == 0.0
    assert _percentile([7.0], 0.99) == 7.0
    assert _percentile(values, 0.0) == 10.0
    assert _percentile(values, 0.50) == 10.0
    assert _percentile(values, 0.90) == 100.0
    assert _percentile(values, 0.99) == 100.0
    assert _percentile(values, 1.0) == 100.0
    with pytest.raises(ValueError, match="quantile"):
        _percentile(values, 1.01)
    with pytest.raises(ValueError, match="quantile"):
        _percentile([], -0.01)


def test_source_identity_contract_matches_only_the_selected_policy() -> None:
    matching = _source_identity_contract(
        "candidate",
        (
            "671a5df8a07e09303c398871fd1772f306b2998e"
            "a3e8168048588de6cc3fa323"
        ),
    )
    assert matching["available"]
    assert matching["pass"]

    mismatch = _source_identity_contract(
        "candidate",
        "671a5df8" + "0" * 56,
    )
    assert mismatch["available"]
    assert not mismatch["pass"]
    assert mismatch["expected_decode_sha256_prefix"] == "671a5df8"

    missing = _source_identity_contract("candidate", None)
    assert not missing["available"]
    assert missing["pass"] is None
    with pytest.raises(ValueError, match="lowercase SHA256"):
        _source_identity_contract("candidate", "not-a-digest")

    packed_sha256 = (
        "da36c09dc275838ee364f76342d74717338ef313"
        "d912ba2b372808530489dd14"
    )
    packed = _source_identity_contract("packed-nz", packed_sha256)
    assert packed["available"]
    assert packed["pass"]
    assert packed["policy_id"].startswith("release-packed-nz-")

    local_ep_sha256 = (
        "26c1b06d739c8d32c04c455c23854c4e"
        "45436049fc60e9496a64df895712a85e"
    )
    local_ep = _source_identity_contract("local-ep", local_ep_sha256)
    assert local_ep["available"]
    assert local_ep["pass"]
    assert local_ep["policy_id"].startswith("release-local-ep-")

    local_ep_mismatch = _source_identity_contract(
        "local-ep",
        "0" * 64,
    )
    assert local_ep_mismatch["available"]
    assert not local_ep_mismatch["pass"]
    assert local_ep_mismatch["expected_decode_sha256"] == local_ep_sha256


def test_route_histogram_awaits_local_owner_sidecar_without_task_counts() -> None:
    result = _route_histogram_contract()
    assert not result["L3"]["available"]
    assert not result["L3"]["blocking"]
    assert not result["L3"]["release_gate"]
    assert result["L3"]["expected_sidecar"] == "local_owner_route_counts"
    assert not result["L3"]["proxy_fallback_allowed"]
    assert result["L3"]["histogram"] is None
    assert result["L3"]["publication_evidence_required"]
    assert not result["L3"]["publication_evidence_ready"]
    assert "receive_tiles_by_rank" not in result["L3"]
    assert "task count" in " ".join(result["L3"]["rejected_proxies"])


def test_route_histogram_validates_exact_local_owner_sidecar(tmp_path) -> None:
    sidecar = tmp_path / "local_owner_routes.json"
    sidecar.write_text(json.dumps(_local_owner_payload()), encoding="utf-8")
    result = _route_histogram_contract(sidecar)
    assert result["L3"]["available"]
    assert result["L3"]["histogram"]["rank0/d0"][0] == 5
    assert result["L3"]["total_routed_tokens_by_rank"]["rank0/d0"] == 5
    assert result["L4"]["histogram"]["rank7/d0"][35] == 4
    assert result["L4"]["snapshot_independence_validated"]
    assert result["L3"]["publication_evidence_ready"]
    assert not result["L3"]["proxy_fallback_allowed"]
    assert result["L3"]["route_totals_validated"]
    assert result["L3"]["per_layer_per_owner"] == [
        5, 0, 3, 0, 0, 0, 0, 0
    ]
    assert result["L4"]["per_layer_per_owner"] == [
        0, 0, 4, 0, 0, 0, 0, 4
    ]
    assert result["L3"]["global_per_layer"] == [8, 8]
    assert result["L4"]["expected_global_per_layer"] == 8
    assert result["L3"]["owner_rows_diagonal"]
    assert (
        result["L3"]["provenance"]["protocol_profile"]
        == "legacy_distributed_ep"
    )
    assert result["L3"]["provenance"]["numeric_contract"] == {
        "name": "legacy_baseline_bit_exact_v1",
        "comparison": "bit_exact_to_protocol_golden",
        "bit_exact": True,
    }
    assert (
        result["L3"]["snapshot_provenance"]["source_tensor"]
        == "local_expert_count"
    )

    ranks = {
        "rank0/d0": {"layers": {"L3": {}, "L4": {}}},
        "rank1/d0": {"layers": {"L3": {}, "L4": {}}},
    }
    classification = _execution_limit_classification(ranks, result)
    assert (
        classification["coverage"]["L3"]["rank0/d0"]["route_empty"][
            "classification"
        ]
        == "not_observed"
    )
    assert (
        classification["coverage"]["L3"]["rank1/d0"]["route_empty"][
            "classification"
        ]
        == "observed"
    )


def test_route_histogram_sidecar_must_match_source_policy(tmp_path) -> None:
    payload = _local_owner_payload()
    sidecar = tmp_path / "local_owner_routes.json"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match source policy"):
        _route_histogram_contract(sidecar, profile="candidate")

    exact_decode_sha = (
        "671a5df8a07e09303c398871fd1772f306b2998e"
        "a3e8168048588de6cc3fa323"
    )
    payload["provenance"]["source"]["decode_fwd_sha256"] = exact_decode_sha
    payload["provenance"]["formal_golden"][
        "source_decode_fwd_sha256"
    ] = exact_decode_sha
    payload["provenance"]["source_manifest_sha256"] = hashlib.sha256(
        json.dumps(
            payload["provenance"]["source"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    result = _route_histogram_contract(
        sidecar,
        profile="candidate",
        source_decode_sha256=exact_decode_sha,
    )
    assert result["L3"]["provenance"]["source_policy_id"].startswith(
        "campaign-candidate-"
    )

    with pytest.raises(ValueError, match="does not match live source"):
        _route_histogram_contract(
            sidecar,
            profile="candidate",
            source_decode_sha256="0" * 64,
        )


def test_route_histogram_accepts_exact_packed_nz_source_policy(
    tmp_path,
) -> None:
    payload = _local_owner_payload()
    packed_sha256 = (
        "da36c09dc275838ee364f76342d74717338ef313"
        "d912ba2b372808530489dd14"
    )
    payload["provenance"]["source"]["decode_fwd_sha256"] = packed_sha256
    payload["provenance"]["formal_golden"][
        "source_decode_fwd_sha256"
    ] = packed_sha256
    payload["provenance"]["source_manifest_sha256"] = hashlib.sha256(
        json.dumps(
            payload["provenance"]["source"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    sidecar = tmp_path / "packed-nz-local-owner-routes.json"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    result = _route_histogram_contract(
        sidecar,
        profile="packed-nz",
        source_decode_sha256=packed_sha256,
    )

    assert result["L3"]["available"]
    assert result["L3"]["provenance"]["source_policy_id"].startswith(
        "release-packed-nz-"
    )


def test_route_histogram_accepts_exact_local_ep_source_policy(
    tmp_path,
) -> None:
    payload = _local_owner_payload()
    local_ep_sha256 = (
        "26c1b06d739c8d32c04c455c23854c4e"
        "45436049fc60e9496a64df895712a85e"
    )
    payload["provenance"]["source"]["decode_fwd_sha256"] = (
        local_ep_sha256
    )
    payload["provenance"]["formal_golden"][
        "source_decode_fwd_sha256"
    ] = local_ep_sha256
    payload["provenance"]["formal_golden"].update(
        {
            "source_run": "local-ep-r10-formal-bs1-64k",
            "source_kind": "local-ep",
            "protocol_profile": "replicated_input_local_owner",
            "numeric_contract": {
                "name": "local_owner_partial_tp_all_reduce_bf16_v1",
                "comparison": "bit_exact_to_protocol_golden",
                "bit_exact": True,
            },
        }
    )
    payload["provenance"]["source_manifest_sha256"] = hashlib.sha256(
        json.dumps(
            payload["provenance"]["source"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    sidecar = tmp_path / "local-ep-local-owner-routes.json"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    result = _route_histogram_contract(
        sidecar,
        profile="local-ep",
        source_decode_sha256=local_ep_sha256,
    )

    assert result["L3"]["available"]
    assert result["L3"]["provenance"]["source_policy_id"].startswith(
        "release-local-ep-"
    )
    assert result["L3"]["provenance"]["source_kind"] == "local-ep"
    assert (
        result["L3"]["provenance"]["protocol_profile"]
        == "replicated_input_local_owner"
    )
    assert result["L3"]["provenance"]["numeric_contract"] == {
        "name": "local_owner_partial_tp_all_reduce_bf16_v1",
        "comparison": "bit_exact_to_protocol_golden",
        "bit_exact": True,
    }


def test_local_ep_profile_rejects_legacy_or_malformed_protocol_golden(
    tmp_path,
) -> None:
    sidecar = tmp_path / "local-ep-golden-contract.json"
    payload = _local_owner_payload()
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match expected"):
        _route_histogram_contract(sidecar, profile="local-ep")

    formal_golden = payload["provenance"]["formal_golden"]
    formal_golden.update(
        {
            "source_kind": "local-ep",
            "protocol_profile": "replicated_input_local_owner",
        }
    )
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="declare protocol_profile and numeric_contract together",
    ):
        _route_histogram_contract(sidecar, profile="local-ep")

    formal_golden["numeric_contract"] = {
        "name": "local_owner_partial_tp_all_reduce_bf16_v1",
        "comparison": "bit_exact_to_legacy_baseline",
        "bit_exact": True,
    }
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="numeric_contract.comparison"):
        _route_histogram_contract(sidecar, profile="local-ep")


def test_route_histogram_rejects_invalid_local_owner_evidence(
    tmp_path,
) -> None:
    payload = _local_owner_payload()
    payload["snapshot_provenance"][1]["snapshot_id"] = (
        "local-owner-routes-l3"
    )
    sidecar = tmp_path / "overlap.json"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="distinct route snapshot IDs"):
        _route_histogram_contract(sidecar)

    payload = _local_owner_payload()
    payload["local_expert_count"][0][0][0] += 1
    sidecar = tmp_path / "bad-count.json"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="diagonal owner route row"):
        _route_histogram_contract(sidecar)

    payload = _local_owner_payload()
    payload["owner_route_counts"][0][0][0][0] += 1
    payload["local_expert_count"][0][0][0] += 1
    sidecar = tmp_path / "bad-route-total.json"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="route totals invalid"):
        _route_histogram_contract(sidecar)

    payload = _local_owner_payload()
    payload["owner_route_counts"][0][0][1][0] = 1
    sidecar = tmp_path / "off-owner-row.json"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="must be diagonal"):
        _route_histogram_contract(sidecar)

    payload = _local_owner_payload()
    payload["snapshot_provenance"][0]["source_tensor"] = "moe_recv_meta"
    sidecar = tmp_path / "forged-source-tensor.json"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source_tensor"):
        _route_histogram_contract(sidecar)

    payload = _local_owner_payload()
    payload["snapshot_provenance"][0]["source_protocol"] = "source_rank_ep"
    sidecar = tmp_path / "forged-source-protocol.json"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source_protocol"):
        _route_histogram_contract(sidecar)

    payload = _local_owner_payload()
    del payload["provenance"]["formal_golden"]
    sidecar = tmp_path / "missing-golden.json"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="formal_golden"):
        _route_histogram_contract(sidecar)


def test_task_id_contract_reports_both_mismatch_directions() -> None:
    tasks = [
        Task("dep-1", 0, "one", 1, (1,), False),
        Task("dep-2", 1, "two", 1, (2,), False),
    ]
    trace = RankTrace(
        tag="rank0/d0",
        rank_dir=Path("rank0/d0"),
        frequency_hz=1_000_000,
        core_types=["aic"],
        tasks=tasks,
        task_by_id={task.task_id: task for task in tasks},
        slices_by_task={
            "dep-1": [Slice(0, "dep-1", 0, 1, "aic")],
            "swim-only": [Slice(0, "swim-only", 2, 3, "aic")],
        },
        edges=[],
        critical_path={},
    )
    contract = _task_id_contract(trace)
    assert not contract["pass"]
    assert contract["missing_on_swim"] == ["dep-2"]
    assert contract["unknown_on_swim"] == ["swim-only"]


def test_task_id_contract_ignores_only_non_executable_dependency_tasks() -> None:
    executable = Task("kernel", 0, "kernel", 1, (1,), False)
    runtime_only = Task("runtime", 1, "runtime_or_creator", 0, (-1, -1), False)
    trace = RankTrace(
        tag="rank0/d0",
        rank_dir=Path("rank0/d0"),
        frequency_hz=1_000_000,
        core_types=["aic"],
        tasks=[executable, runtime_only],
        task_by_id={"kernel": executable, "runtime": runtime_only},
        slices_by_task={"kernel": [Slice(0, "kernel", 0, 1, "aic")]},
        edges=[],
        critical_path={},
    )
    contract = _task_id_contract(trace)
    assert contract["pass"]
    assert contract["dep_task_ids"] == ["kernel"]
    assert contract["swim_task_ids"] == ["kernel"]
    assert contract["ignored_non_executable_dep_task_ids"] == ["runtime"]


def test_task_id_contract_accepts_explicit_predicated_skip() -> None:
    executed = Task("executed", 0, "one", 1, (1,), False)
    skipped = Task("skipped", 1, "two", 1, (2,), True)
    trace = RankTrace(
        tag="rank0/d0",
        rank_dir=Path("rank0/d0"),
        frequency_hz=1_000_000,
        core_types=["aic"],
        tasks=[executed, skipped],
        task_by_id={"executed": executed, "skipped": skipped},
        slices_by_task={
            "executed": [Slice(0, "executed", 0, 1, "aic")],
        },
        edges=[],
        critical_path={},
        predicated_skip_task_ids=("skipped",),
    )

    contract = _task_id_contract(trace)

    assert contract["pass"]
    assert contract["missing_on_swim"] == []
    assert contract["predicated_skip_task_ids"] == ["skipped"]
    assert contract["predicated_skip_without_physical_slices"] == [
        "skipped"
    ]


def test_task_id_contract_rejects_invalid_predicated_skip_evidence() -> None:
    executed = Task("executed", 0, "one", 1, (1,), False)
    skipped = Task("skipped", 1, "two", 1, (2,), True)
    trace = RankTrace(
        tag="rank0/d0",
        rank_dir=Path("rank0/d0"),
        frequency_hz=1_000_000,
        core_types=["aic"],
        tasks=[executed, skipped],
        task_by_id={"executed": executed, "skipped": skipped},
        slices_by_task={
            "executed": [Slice(0, "executed", 0, 1, "aic")],
        },
        edges=[],
        critical_path={},
        predicated_skip_task_ids=(
            "executed",
            "skipped",
            "skipped",
            "unknown",
        ),
    )

    contract = _task_id_contract(trace)

    assert not contract["pass"]
    assert contract["missing_on_swim"] == []
    assert contract["duplicate_predicated_skip_task_ids"] == ["skipped"]
    assert contract["unexpected_predicated_skip_task_ids"] == ["unknown"]
    assert contract["predicated_skip_with_physical_slices"] == ["executed"]
    with pytest.raises(RuntimeError, match="task-level structural"):
        _validate_structural_contracts([trace])


def test_combine_dependency_contract_accepts_complete_two_layer_chain() -> None:
    trace = _combine_trace()
    contract = _combine_dependency_contract(trace)
    assert contract["pass"]
    assert _validate_structural_contracts([trace])["pass"]
    for layer in ("L3", "L4"):
        assert contract["layers"][layer]["chain_count"] == 1
        chain = contract["layers"][layer]["chains"][0]
        required = chain["required_edges"]
        assert required["scatter_to_wait_explicit"]["pass"]
        assert required["wait_to_reduce_explicit"]["pass"]
        assert required["scatter_to_reduce_data"]["pass"]
        assert chain["local_swim_order"]["pass"]


def test_combine_dependency_contract_rejects_missing_and_reverse_edges() -> None:
    trace = _combine_trace()
    trace.edges = [
        edge for edge in trace.edges if not (edge["pred"] == "l4-wait" and edge["succ"] == "l4-reduce")
    ]
    trace.edges.append({"pred": "l4-reduce", "succ": "l4-wait", "source": "explicit"})
    contract = _combine_dependency_contract(trace)
    assert not contract["pass"]
    error_codes = {error["code"] for error in contract["layers"]["L4"]["errors"]}
    assert "wait_to_reduce_out_degree" in error_codes
    assert "wait_to_reduce_in_degree" in error_codes
    assert "unexpected_or_cross_layer_explicit_edge" in error_codes
    with pytest.raises(RuntimeError, match="combine_dependency"):
        _validate_structural_contracts([trace])


def test_combine_dependency_contract_rejects_local_swim_reordering() -> None:
    trace = _combine_trace()
    trace.slices_by_task["l4-scatter"] = [
        Slice(24, "l4-scatter", 60, 95, "aiv")
    ]
    contract = _combine_dependency_contract(trace)
    assert not contract["pass"]
    local_order = contract["layers"]["L4"]["chains"][0]["local_swim_order"]
    assert local_order["available"]
    assert not local_order["pass"]
    assert not local_order["dependency_completion_ordered"]
    assert any(
        error["code"] == "local_swim_execution_order"
        for error in contract["layers"]["L4"]["errors"]
    )


def test_combine_dependency_contract_accepts_multiple_one_to_one_chains() -> None:
    trace = _combine_trace()
    extra_specs = (
        ("l3-scatter-2", "swa_moe_chip_orch_combine_scatter"),
        ("l3-wait-2", "swa_moe_chip_orch_combine_wait"),
        ("l3-reduce-2", "swa_moe_chip_orch_combine_reduce"),
    )
    for offset, (task_id, name) in enumerate(extra_specs, start=len(trace.tasks)):
        task = Task(
            task_id=task_id,
            order=offset,
            name=name,
            block_num=1,
            kernel_ids=(-1, offset + 1, -1),
            early_dispatch=True,
        )
        trace.tasks.append(task)
        trace.task_by_id[task_id] = task
        trace.slices_by_task[task_id] = [
            Slice(
                core=24,
                task_id=task_id,
                start=offset * 20,
                end=offset * 20 + 10,
                resource="aiv",
            )
        ]
    trace.edges.extend(
        [
            {
                "pred": "l3-scatter-2",
                "succ": "l3-wait-2",
                "source": "explicit",
            },
            {
                "pred": "l3-wait-2",
                "succ": "l3-reduce-2",
                "source": "explicit",
            },
            {
                "pred": "l3-scatter-2",
                "succ": "l3-reduce-2",
                "source": "tensormap",
            },
        ]
    )
    contract = _combine_dependency_contract(trace)
    assert contract["pass"]
    assert contract["layers"]["L3"]["stage_task_counts"] == {
        "combine_scatter": 2,
        "combine_wait": 2,
        "combine_reduce": 2,
    }
    assert contract["layers"]["L3"]["chain_count"] == 2
    assert all(chain["pass"] for chain in contract["layers"]["L3"]["chains"])


@pytest.mark.parametrize(
    ("field", "value", "failed_check"),
    [
        ("p50_us", 9.0, "p50_ge_limit"),
        ("p50_us", 31.0, "p50_le_limit"),
        ("p90_us", 31.0, "p90_le_limit"),
        ("p99_us", 61.0, "p99_le_limit"),
        ("max_us", 101.0, "max_le_limit"),
    ],
)
@pytest.mark.parametrize(
    "stage",
    ["expert_gate", "expert_up", "expert_down"],
)
def test_expert_release_enforces_every_aic_duration_limit(
    stage: str,
    field: str,
    value: float,
    failed_check: str,
) -> None:
    ranks = {"rank0/d0": _valid_expert_rank()}
    ranks["rank0/d0"]["layers"]["L3"][stage]["resources"]["aic"][
        "duration_distribution"
    ][field] = value
    contract = _expert_kernel_release_contract(ranks, profile="row16")
    assert not contract["duration_pass"]
    gate_error = next(
        error
        for error in contract["duration_errors"]
        if error["stage"] == stage
    )
    assert failed_check in gate_error["failed_checks"]


def test_expert_release_requires_staged_fused_aic_and_aiv_stages() -> None:
    ranks = {
        "rank0/d0": _valid_expert_rank(),
        "rank1/d0": _valid_expert_rank(),
    }
    del ranks["rank1/d0"]["layers"]["L3"]["expert_gate_up"]
    del ranks["rank1/d0"]["layers"]["L3"]["expert_gate_up_act"]
    del ranks["rank1/d0"]["layers"]["L3"]["routed_h_quant"]
    ranks["rank1/d0"]["layers"]["L4"] = {}
    contract = _expert_kernel_release_contract(ranks)
    assert not contract["pass"]
    assert any(
        error["rank"] == "rank1/d0"
        and error["stage"] == "expert_gate_up"
        and error["code"] == "missing_aic_stage"
        for error in contract["duration_errors"]
    )
    assert any(
        error["rank"] == "rank1/d0"
        and error["code"] == "activation_quant_must_be_aiv_only"
        for error in contract["activation_errors"]
    )
    empty_l4 = contract["coverage"]["L4"]["rank1/d0"]
    assert not empty_l4["execution_nonempty"]
    assert not empty_l4["route_empty_inferred"]


def test_packed_nz_release_enforces_mixed_resource_grids() -> None:
    ranks = {
        "rank0/d0": _valid_packed_nz_rank(),
        "rank1/d0": {"layers": {"L3": {}, "L4": {}}},
    }
    passing = _expert_kernel_release_contract(
        ranks,
        profile="packed-nz",
    )
    assert passing["pass"]
    assert passing["mixed_resource_grid_pass"]
    assert passing["activation_pass"]
    assert passing["duration_limits_us"] == {}
    assert passing["mixed_resource_targets"] == {
        "expert_gate_up": {"aic": 24, "aiv": 48},
        "expert_down": {"aic": 23, "aiv": 46},
    }
    assert not passing["coverage"]["L3"]["rank1/d0"][
        "execution_nonempty"
    ]

    ranks["rank0/d0"]["layers"]["L4"]["expert_down"]["resources"][
        "aiv"
    ]["observed_slices"] = 45
    blocked = _expert_kernel_release_contract(
        ranks,
        profile="packed-nz",
    )
    assert not blocked["pass"]
    assert not blocked["mixed_resource_grid_pass"]
    assert blocked["mixed_resource_errors"][0]["stage"] == "expert_down"
    assert "aiv_observed_slices" in blocked["mixed_resource_errors"][0][
        "failed_checks"
    ]


def test_expert_release_accepts_valid_nonempty_rank_and_rejects_aic_vector_stage() -> None:
    ranks = {"rank0/d0": _valid_expert_rank()}
    assert _expert_kernel_release_contract(ranks)["pass"]

    activation = ranks["rank0/d0"]["layers"]["L3"]["expert_gate_up_act"]
    activation["resources"]["aic"] = _fake_resource(observed_slices=1)
    contract = _expert_kernel_release_contract(ranks)
    assert not contract["activation_pass"]
    assert "aic_not_observed" in contract["activation_errors"][0]["failed_checks"]


def test_staged_fused_gate_up_uses_only_r5_proven_upper_bounds() -> None:
    ranks = {"rank0/d0": _valid_expert_rank()}
    gate_up = ranks["rank0/d0"]["layers"]["L3"]["expert_gate_up"]
    gate_up["resources"]["aic"]["duration_distribution"]["p50_us"] = 1.0
    passing = _expert_kernel_release_contract(ranks)
    assert passing["duration_pass"]
    assert passing["release_family"] == "staged_fused_gate_up"
    assert passing["duration_limits_us"] == {
        "p50_max": 200.0,
        "p90_max": 220.0,
        "p99_max": 320.0,
        "max": 500.0,
    }
    assert "R5 packed-fused" in passing["duration_limit_source"]

    gate_up["resources"]["aic"]["duration_distribution"]["p90_us"] = 220.1
    blocked = _expert_kernel_release_contract(ranks)
    assert not blocked["duration_pass"]
    assert blocked["duration_errors"][0]["failed_checks"] == [
        "p90_le_limit"
    ]


def test_expert_release_rejects_a_layer_with_no_observed_routed_compute() -> None:
    ranks = {"rank0/d0": _valid_expert_rank()}
    ranks["rank0/d0"]["layers"]["L4"] = {}
    contract = _expert_kernel_release_contract(ranks)
    assert not contract["coverage_pass"]
    assert contract["coverage_errors"] == [
        {
            "layer": "L4",
            "code": "no_routed_compute_observed",
            "reason": (
                "No rank exposes routed physical compute for this MoE layer; "
                "the capture cannot satisfy the expert release gate."
            ),
        }
    ]


def test_frozen_source_policies_match_the_actual_campaign_families() -> None:
    baseline = _source_policy("baseline")
    assert {
        key: baseline[key]
        for key in (
            "frozen_ref",
            "decode_sha256_prefix",
            "source_role",
            "storage_family",
            "schedule_family",
            "task_partition",
            "enforce_candidate_release_gate",
        )
    } == {
        "frozen_ref": "stepfun/develop@56b3d477",
        "decode_sha256_prefix": "3553664c",
        "source_role": "baseline",
        "storage_family": "row32_no_graph_wide_gate_up_scratch",
        "schedule_family": "fused_expert_gate_up",
        "task_partition": "tile_local_activation_quant_down",
        "enforce_candidate_release_gate": False,
    }

    candidate = _source_policy("candidate")
    assert {
        key: candidate[key]
        for key in (
            "decode_sha256_prefix",
            "source_role",
            "storage_family",
            "schedule_family",
            "task_partition",
            "enforce_candidate_release_gate",
        )
    } == {
        "decode_sha256_prefix": "671a5df8",
        "source_role": "candidate",
        "storage_family": "row16_staged_fused_gate_up_local_tiles",
        "schedule_family": "staged_fused_gate_up_then_aiv_act_quant_down",
        "task_partition": "aic_gate_up_aiv_activation_quant_aic_down",
        "enforce_candidate_release_gate": True,
    }
    compatibility = candidate["origin_main_compatibility_reference"]
    assert compatibility["frozen_ref"] == "origin/main@1f48761c"
    assert not compatibility["campaign_baseline"]
    assert "never inferred" in candidate["selection_semantics"]

    row16 = _source_policy("row16")
    assert row16["decode_sha256_prefix"] == "65b0b8bf"
    assert row16["source_role"] == "reference"
    assert row16["enforce_candidate_release_gate"]

    shared_split = _source_policy("shared-split")
    assert shared_split["decode_sha256_prefix"] == "572ea2a2"
    assert shared_split["source_role"] == "candidate"
    assert shared_split["task_partition"].endswith(
        "shared_5_gate_up_5_activation_16_down",
    )
    assert shared_split["enforce_candidate_release_gate"]


def test_shared_experiment_profiles_remain_distinct_in_admission() -> None:
    ranks = {"rank0/d0": _valid_expert_rank()}
    for profile in ("row16", "shared-split"):
        release = _expert_kernel_release_contract(
            ranks,
            profile=profile,
        )
        assert release["profile"] == profile
        assert release["release_enforced"]
        admission = _admission_contract(
            _route_histogram_contract(),
            {"fields": {}},
            {"pass": True, "errors": []},
            release,
        )
        assert admission["profile"] == profile
        assert admission["release_readiness"]["status"] == "NOT_EVALUABLE"


def test_baseline_profile_keeps_candidate_split_gate_diagnostic_only() -> None:
    fused_rank = {
        "layers": {
            layer: {
                "expert_gate_up": _fake_stage(),
                "expert_down": _fake_stage(),
            }
            for layer in ("L3", "L4")
        }
    }
    baseline = _expert_kernel_release_contract(
        {"rank0/d0": fused_rank},
        profile="baseline",
    )
    assert baseline["pass"] is None
    assert not baseline["diagnostic_pass"]
    assert not baseline["release_enforced"]
    assert baseline["release_gate_pass"] is None
    assert baseline["release_gate_status"] == "NOT_APPLICABLE"
    assert (
        baseline["source_policy"]["task_partition"]
        == "tile_local_activation_quant_down"
    )
    admission = _admission_contract(
        _route_histogram_contract(),
        {"fields": {}},
        {"pass": True, "errors": []},
        baseline,
    )
    assert admission["pass"]
    assert admission["profile"] == "baseline"
    assert admission["release_readiness"]["status"] == "DIAGNOSTIC_ONLY"
    assert not admission["release_readiness"]["blocked"]
    assert any(
        item["code"] == "baseline_expert_release_diagnostic_only"
        for item in admission["non_blocking_limitations"]
    )

    candidate = _expert_kernel_release_contract(
        {"rank0/d0": fused_rank},
        profile="candidate",
    )
    assert candidate["release_enforced"]
    assert candidate["source_policy"]["schedule_family"] == (
        "staged_fused_gate_up_then_aiv_act_quant_down"
    )
    candidate_admission = _admission_contract(
        _route_histogram_contract(),
        {"fields": {}},
        {"pass": True, "errors": []},
        candidate,
    )
    assert not candidate_admission["pass"]
    assert candidate_admission["release_readiness"]["status"] == "BLOCKED"
    assert {
        blocker["code"] for blocker in candidate_admission["blockers"]
    } >= {
        "expert_activation_aiv_release_failed",
    }


def test_admission_uses_frozen_policy_as_candidate_enforcement_authority() -> None:
    admission = _admission_contract(
        _route_histogram_contract(),
        {"fields": {}},
        {"pass": True, "errors": []},
        {
            "profile": "candidate",
            "release_enforced": False,
            "source_policy": {
                "source_role": "baseline",
                "storage_family": "task_name_inferred",
            },
            "coverage_pass": True,
            "duration_pass": False,
            "activation_pass": False,
            "coverage_errors": [],
            "duration_errors": [{"stage": "expert_gate"}],
            "activation_errors": [{"stage": "expert_gate_up_act"}],
        },
    )
    assert not admission["pass"]
    assert admission["expert_release_enforced"]
    assert (
        admission["source_policy"]["storage_family"]
        == "row16_staged_fused_gate_up_local_tiles"
    )
    blocker_codes = {blocker["code"] for blocker in admission["blockers"]}
    assert blocker_codes >= {
        "expert_release_policy_mismatch",
        "expert_aic_duration_release_failed",
        "expert_activation_aiv_release_failed",
    }


def test_execution_limit_classification_keeps_four_causes_distinct() -> None:
    ranks = {
        "rank0/d0": {
            "layers": {
                "L3": {
                    "expert_gate": _fake_stage(
                        observed_slices=2,
                        available_cores=24,
                        peak_concurrency=2,
                        queue_delay_us=5.0,
                    ),
                    "expert_up": _fake_stage(
                        observed_slices=30,
                        available_cores=24,
                        peak_concurrency=10,
                    ),
                },
                "L4": {},
            }
        }
    }
    result = _execution_limit_classification(
        ranks,
        _route_histogram_contract(),
    )
    rank = result["coverage"]["L3"]["rank0/d0"]
    assert rank["route_empty"]["classification"] == "unknown"
    assert "task count" in " ".join(rank["route_empty"]["rejected_proxies"])

    gate = rank["stages"]["expert_gate"]
    assert gate["queue_delay"]["classification"] == "observed"
    assert (
        gate["insufficient_ready_parallelism"]["classification"]
        == "observed_work_width_limit"
    )
    assert gate["scheduler_packing"]["classification"] == "not_observed"

    up = rank["stages"]["expert_up"]
    assert up["insufficient_ready_parallelism"]["classification"] == "undetermined"
    assert up["scheduler_packing"]["classification"] == "suspected"


def test_local_ep_profile_selects_local_topology_diagnostics() -> None:
    assert _arrival_pairs("local-ep") == (
        (
            "moe_collective",
            "local_combine_reduce",
            "moe_all_reduce",
        ),
    )
    assert set(_diagnostic_stage_resources("local-ep")) == {
        "local_route_pack",
        "expert_gate_up",
        "expert_down",
        "local_combine_reduce",
        "moe_all_reduce",
    }
    timing_stages = _timing_profile_stages("local-ep")
    assert "local_route_pack" in timing_stages
    assert "local_combine_reduce" in timing_stages
    assert "moe_all_reduce" in timing_stages
    assert "dispatch_wait" not in timing_stages
    markdown_stages = _markdown_stage_order("local-ep")
    assert "local_route_pack" in markdown_stages
    assert "local_combine_reduce" in markdown_stages
    assert "moe_all_reduce" in markdown_stages
    assert "dispatch_wait" not in markdown_stages
    assert "combine_wait" not in markdown_stages

    legacy_timing_stages = _timing_profile_stages("candidate")
    assert "combine_wait" in legacy_timing_stages
    assert "local_combine_reduce" not in legacy_timing_stages


def test_local_ep_execution_limit_uses_local_collective_stages() -> None:
    ranks = {
        "rank0/d0": {
            "layers": {
                "L3": {
                    "local_route_pack": _fake_stage(resource="aiv"),
                    "local_combine_reduce": _fake_stage(resource="aiv"),
                    "moe_all_reduce": _fake_stage(resource="aiv"),
                },
                "L4": {},
            }
        }
    }

    result = _execution_limit_classification(
        ranks,
        _route_histogram_contract(),
        "local-ep",
    )

    stages = result["coverage"]["L3"]["rank0/d0"]["stages"]
    assert set(stages) == set(_diagnostic_stage_resources("local-ep"))
    assert stages["local_route_pack"]["execution_observed"]
    assert stages["local_combine_reduce"]["execution_observed"]
    assert stages["moe_all_reduce"]["execution_observed"]
    assert "combine_wait" not in stages


def test_hidden_bit_exact_is_explicitly_owned_by_the_outer_gate() -> None:
    contract = _external_correctness_contract()["hidden_state_bit_exact"]
    assert not contract["enforced_here"]
    assert contract["required_artifacts"] == ["hidden_l3.pt", "hidden_l4.pt"]
    assert "protocol-specific" in contract["comparison"]
    assert contract["required_manifest_fields"] == [
        "source_kind",
        "bit_exact",
    ]
    assert contract["local_ep_required_manifest_fields"] == [
        "source_kind",
        "protocol_profile",
        "numeric_contract",
        "bit_exact",
    ]


def test_candidate_without_local_owner_routes_is_not_evaluable() -> None:
    route_histogram = _route_histogram_contract()
    timing_evidence = {
        "fields": {
            "critical_path_contribution": {
                "unavailable_task_count": 4,
                "unavailable": [
                    {
                        "reason": "no structured per-task contribution sidecar",
                    }
                ],
            }
        }
    }
    routed_slice_profiles = {"pass": True, "errors": []}
    expert_release = {
        "coverage_pass": True,
        "duration_pass": True,
        "activation_pass": True,
        "coverage_errors": [],
        "duration_errors": [],
        "activation_errors": [],
    }
    admission = _admission_contract(
        route_histogram,
        timing_evidence,
        routed_slice_profiles,
        expert_release,
    )
    assert admission["pass"]
    assert not admission["blockers"]
    limitation_codes = {
        limitation["code"] for limitation in admission["non_blocking_limitations"]
    }
    assert "route_histogram_awaiting_recv_meta" in limitation_codes
    assert "critical_path_contribution_incomplete" in limitation_codes
    readiness = admission["release_readiness"]
    assert readiness["status"] == "NOT_EVALUABLE"
    assert readiness["blocked"]
    assert readiness["analyzer_ready"]
    assert not readiness["recv_meta_publication_evidence_ready"]
    assert not readiness["publication_allowed"]


def test_candidate_with_local_owner_routes_is_pending_external_hidden_gate(
    tmp_path,
) -> None:
    sidecar = tmp_path / "local_owner_routes.json"
    sidecar.write_text(json.dumps(_local_owner_payload()), encoding="utf-8")
    admission = _admission_contract(
        _route_histogram_contract(sidecar),
        {"fields": {}},
        {"pass": True, "errors": []},
        {
            "profile": "candidate",
            "release_enforced": True,
            "coverage_pass": True,
            "duration_pass": True,
            "activation_pass": True,
            "coverage_errors": [],
            "duration_errors": [],
            "activation_errors": [],
        },
    )
    assert admission["pass"]
    readiness = admission["release_readiness"]
    assert readiness["status"] == "PENDING_EXTERNAL_GATE"
    assert readiness["blocked"]
    assert readiness["analyzer_ready"]
    assert readiness["recv_meta_publication_evidence_ready"]
    assert readiness["external_hidden_state_gate_required"]
    assert not readiness["publication_allowed"]


def test_task_timing_evidence_is_fail_closed() -> None:
    pred = Task("pred", 0, "producer", 1, (1,), False)
    task = Task("task", 1, "consumer", 1, (2,), False)
    trace = RankTrace(
        tag="rank0/d0",
        rank_dir=Path("rank0/d0"),
        frequency_hz=1_000_000,
        core_types=["aic"],
        tasks=[pred, task],
        task_by_id={"pred": pred, "task": task},
        slices_by_task={
            "pred": [Slice(0, "pred", 0, 10, "aic")],
            "task": [Slice(0, "task", 15, 35, "aic")],
        },
        edges=[{"pred": "pred", "succ": "task", "source": "tensormap"}],
        critical_path={},
    )
    evidence = _task_timing_evidence(trace, task)
    assert evidence["queue_delay"]["available"]
    assert evidence["queue_delay"]["value_us"] == 5.0
    assert evidence["service_span"]["distribution_us"]["p50_us"] == 20.0
    assert evidence["dag_span"]["value_us"] == 20.0
    assert not evidence["critical_path_contribution"]["available"]

    trace.slices_by_task["pred"] = [Slice(0, "pred", 0, 20, "aic")]
    overlap = _task_timing_evidence(trace, task)
    assert not overlap["queue_delay"]["available"]
    assert "overlaps task start" in overlap["queue_delay"]["reason"]

    trace.slices_by_task["pred"] = []
    missing = _task_timing_evidence(trace, task)
    assert not missing["queue_delay"]["available"]
    assert missing["queue_delay"]["missing_predecessor_timing"] == ["pred"]


def test_arrival_analysis_disables_cross_rank_subtraction_without_anchor() -> None:
    traces = [
        _trace("rank0/d0", end_tick=1_000_000),
        _trace("rank1/d0", end_tick=1_010_000),
    ]
    for index, trace in enumerate(traces):
        wait_task_id = f"wait-{index}"
        wait_task = Task(
            wait_task_id,
            0,
            "combine_wait",
            1,
            (-1, 1, -1),
            True,
        )
        trace.tasks.append(wait_task)
        trace.task_by_id[wait_task_id] = wait_task
        trace.slices_by_task[wait_task_id] = trace.slices_by_task["terminal"]
    ranks = {
        trace.tag: {
            "layers": {
                "L3": {
                    "combine_scatter": {
                        "start_tick": trace.all_slices[0].start,
                        "end_tick": trace.all_slices[0].end,
                        "stage_span_us": 2.0,
                    },
                    "combine_wait": {
                        "task_ids": ["wait-0" if trace.tag == "rank0/d0" else "wait-1"],
                        "start_tick": trace.all_slices[0].start,
                        "end_tick": trace.all_slices[0].end,
                        "stage_span_us": 2.0,
                    },
                },
            },
        }
        for trace in traces
    }
    alignment = _clock_alignment(traces)
    result = _arrival_analysis(traces, ranks, alignment)
    combine = result["L3"]["combine"]
    assert not combine["clock_domain_comparable"]
    assert combine["producer_end_skew_us"] is None
    assert combine["earliest_producer_rank"] is None
    assert combine["latest_producer_rank"] is None
    assert all("remote_arrival_after_wait_start_us" not in waiter for waiter in combine["wait_ranks"])


def test_local_ep_arrival_reports_only_moe_collective() -> None:
    traces = [_local_ep_trace(), _local_ep_trace()]
    traces[1].tag = "rank1/d0"
    traces[1].rank_dir = Path("rank1/d0")
    ranks = {
        trace.tag: _rank_metrics(trace, "local-ep")
        for trace in traces
    }

    result = _arrival_analysis(
        traces,
        ranks,
        _clock_alignment(traces),
        "local-ep",
    )

    for layer in ("L3", "L4"):
        assert set(result[layer]) == {"moe_collective"}
        collective = result[layer]["moe_collective"]
        assert collective["producer_stage"] == "local_combine_reduce"
        assert collective["consumer_stage"] == "moe_all_reduce"
        assert collective["consumer_ranks"] == collective["wait_ranks"]
        assert len(collective["consumer_ranks"]) == 2
        assert all(
            consumer["has_producer_dependency"]
            for consumer in collective["consumer_ranks"]
        )
        assert all(
            consumer["producer_dependency_sources"] == ["tensormap"]
            for consumer in collective["consumer_ranks"]
        )

    timing = _timing_evidence_contract(ranks, "local-ep")
    assert timing["profiled_stages"] == list(
        _timing_profile_stages("local-ep")
    )
    assert "local_route_pack" in timing["profiled_stages"]
    assert "moe_all_reduce" in timing["profiled_stages"]
    assert "combine_wait" not in timing["profiled_stages"]


def test_findings_accept_noncomparable_producer_skew() -> None:
    ranks = {
        "rank0/d0": {
            "layers": {
                "L3": {},
                "L4": {},
            },
        },
    }
    arrivals = {
        "L3": {"combine": {"producer_end_skew_us": None}},
        "L4": {"combine": {"producer_end_skew_us": None}},
    }
    findings = _aggregate_findings(
        ranks,
        arrivals,
        _route_histogram_contract(),
    )
    assert findings
