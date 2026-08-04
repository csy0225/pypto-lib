# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Analyze L0-L4 Step3p5 MoE dependency and swimlane traces.

The focused graph has two MoE layers:

* L3: sliding-window attention + MoE
* L4: full attention + MoE

This analyzer keeps those layers separate, keeps AIC and AIV accounting
separate for mixed kernels, and compares dispatch/combine arrival times across
all ranks.  It intentionally treats long ``tp_all_reduce`` spans as possible
in-kernel peer wait rather than arithmetic time.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_LAYER_PREFIX = {
    "L3": "swa_moe_chip_orch_",
    "L4": "",
}
_STAGE_SUFFIXES = {
    "gate_init": ("gate_init",),
    "gate_fanout": ("gate_expert_fanout",),
    "gate_topk": ("gate_topk",),
    "shared_mlp": ("sh_mlp",),
    "dispatch_meta": ("dispatch_meta",),
    "dispatch_push": ("dispatch_push",),
    "dispatch_wait": ("dispatch_wait",),
    "dispatch_gather": ("dispatch_gather",),
    "expert_gate_up": ("expert_gate_up",),
    "expert_gate": ("expert_gate_mm",),
    "expert_up": ("expert_up_mm",),
    "expert_gate_up_act": ("expert_gate_up_act",),
    "routed_h_quant": ("routed_h_quant",),
    "expert_down": ("expert_down",),
    "combine_scatter": ("combine_scatter",),
    "combine_wait": ("combine_wait",),
    "combine_reduce": ("combine_reduce",),
    "moe_residual_add": ("moe_residual_add",),
}
_ARRIVAL_PAIRS = (
    ("dispatch", "dispatch_push", "dispatch_wait"),
    ("combine", "combine_scatter", "combine_wait"),
)


@dataclass(frozen=True)
class Task:
    """One dependency-graph task."""

    task_id: str
    order: int
    name: str
    block_num: int
    kernel_ids: tuple[int, ...]
    early_dispatch: bool

    def resource_slices_per_block(self, resource: str) -> int:
        """Return how many physical resource slices one logical block owns."""
        if resource == "aic":
            return int(bool(self.kernel_ids and self.kernel_ids[0] >= 0))
        if resource == "aiv":
            return sum(kernel_id >= 0 for kernel_id in self.kernel_ids[1:])
        return 0


@dataclass(frozen=True)
class Slice:
    """One physical AIC/AIV execution slice."""

    core: int
    task_id: str
    start: int
    end: int
    resource: str


@dataclass
class RankTrace:
    """Parsed dependency and swimlane data for one rank."""

    tag: str
    rank_dir: Path
    frequency_hz: int
    core_types: list[str]
    tasks: list[Task]
    task_by_id: dict[str, Task]
    slices_by_task: dict[str, list[Slice]]
    edges: list[dict[str, Any]]
    critical_path: dict[str, Any]

    @property
    def all_slices(self) -> list[Slice]:
        return [
            item
            for task_slices in self.slices_by_task.values()
            for item in task_slices
        ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-dir",
        required=True,
        help="compiled program directory containing dfx_outputs",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--skip-critical-path",
        action="store_true",
        help="do not invoke simpler_setup.tools.critical_path",
    )
    parser.add_argument("--top", type=int, default=30)
    return parser.parse_args()


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * quantile))
    return ordered[index]


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _strip_resource_suffix(name: str) -> str:
    return re.sub(r"_(?:aic|aiv)$", "", name)


def _callable_name(task: dict[str, Any], name_map: dict[str, str]) -> str:
    callable_id = next(
        (
            kernel_id
            for kernel_id in task.get("kernel_ids", [])
            if kernel_id is not None and int(kernel_id) >= 0
        ),
        None,
    )
    if callable_id is None:
        return "runtime_or_creator"
    return name_map.get(str(callable_id), f"callable_{callable_id}")


def _parse_critical_path(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")

    def number(pattern: str) -> float | None:
        match = re.search(pattern, text)
        return float(match.group(1)) if match else None

    result: dict[str, Any] = {
        "report": str(path),
        "makespan_ms": number(r"\*\*makespan\*\*: ([0-9.]+) ms"),
        "static_cpm_ms": number(r"\*\*static CPM path\*\*: ([0-9.]+) ms"),
        "compute_ms": number(r"  - compute: ([0-9.]+) ms"),
        "stall_ms": number(r"  - stall \(runtime scheduling\): ([0-9.]+) ms"),
        "data_wait_ms": number(r"    - data-wait: ([0-9.]+) ms"),
        "core_wait_ms": number(r"    - core-wait: ([0-9.]+) ms"),
        "front_gap_ms": number(r"    - front-gap: ([0-9.]+) ms"),
    }
    return {key: value for key, value in result.items() if value is not None}


def _load_rank(rank_dir: Path, dfx_root: Path) -> RankTrace | None:
    deps_path = rank_dir / "deps.json"
    names_path = rank_dir / "name_map.json"
    swim_path = rank_dir / "l2_swimlane_records.json"
    if not (deps_path.exists() and names_path.exists() and swim_path.exists()):
        return None

    deps = json.loads(deps_path.read_text(encoding="utf-8"))
    names = json.loads(names_path.read_text(encoding="utf-8")).get(
        "callable_id_to_name",
        {},
    )
    from simpler_setup.tools.swimlane_converter import (  # noqa: PLC0415
        read_perf_data,
    )

    swim = read_perf_data(swim_path)
    perf_tasks = list(swim.get("tasks", []))
    max_core = max(
        (int(item["core_id"]) for item in perf_tasks),
        default=-1,
    )
    core_types = ["unknown"] * (max_core + 1)
    for item in perf_tasks:
        core_types[int(item["core_id"])] = str(
            item.get("core_type", "unknown"),
        )
    tasks = [
        Task(
            task_id=str(item["task_id"]),
            order=order,
            name=_callable_name(item, names),
            block_num=int(item.get("block_num", 0)),
            kernel_ids=tuple(
                int(value) if value is not None else -1
                for value in item.get("kernel_ids", [])
            ),
            early_dispatch=bool(item.get("early_dispatch", False)),
        )
        for order, item in enumerate(deps.get("tasks", []))
    ]
    task_by_id = {task.task_id: task for task in tasks}
    slices_by_task: dict[str, list[Slice]] = collections.defaultdict(list)
    for item in perf_tasks:
        core = int(item["core_id"])
        task_id = str(item["task_id"])
        # Store integer nanoseconds to preserve sub-microsecond precision.
        # read_perf_data owns all raw-schema joins and cycle conversion.
        start = round(float(item["start_time_us"]) * 1000.0)
        end = round(float(item["end_time_us"]) * 1000.0)
        slices_by_task[task_id].append(
            Slice(
                core=core,
                task_id=task_id,
                start=start,
                end=end,
                resource=str(item.get("core_type", "unknown")),
            ),
        )
    return RankTrace(
        tag=str(rank_dir.relative_to(dfx_root)),
        rank_dir=rank_dir,
        frequency_hz=1_000_000_000,
        core_types=core_types,
        tasks=tasks,
        task_by_id=task_by_id,
        slices_by_task=dict(slices_by_task),
        edges=list(deps.get("edges", [])),
        critical_path=_parse_critical_path(rank_dir / "critical_path_report.md"),
    )


def _task_matches_layer(task: Task, layer: str, suffix: str) -> bool:
    prefix = _LAYER_PREFIX[layer]
    base = _strip_resource_suffix(task.name)
    expected = f"{prefix}{suffix}"
    if layer == "L3":
        return base == expected
    return base == expected and not base.startswith(_LAYER_PREFIX["L3"])


def _find_layer_task_ids(
    trace: RankTrace,
    layer: str,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for stage, suffixes in _STAGE_SUFFIXES.items():
        result[stage] = [
            task.task_id
            for task in trace.tasks
            if any(
                _task_matches_layer(task, layer, suffix)
                for suffix in suffixes
            )
        ]

    gate_tasks = [
        trace.task_by_id[task_id]
        for task_id in result["gate_init"]
    ]
    if gate_tasks:
        gate_order = min(task.order for task in gate_tasks)
        prior_norm = [
            task
            for task in trace.tasks
            if task.name == "_norm_quant_moe_input" and task.order < gate_order
        ]
        if prior_norm:
            result["norm_quant"] = [max(prior_norm, key=lambda task: task.order).task_id]
        else:
            result["norm_quant"] = []
    else:
        result["norm_quant"] = []

    shared_tasks = [
        trace.task_by_id[task_id]
        for task_id in result["shared_mlp"]
    ]
    dispatch_tasks = [
        trace.task_by_id[task_id]
        for task_id in result["dispatch_meta"]
    ]
    if shared_tasks and dispatch_tasks:
        shared_order = max(task.order for task in shared_tasks)
        dispatch_order = min(task.order for task in dispatch_tasks)
        all_reduces = [
            task
            for task in trace.tasks
            if task.name == "tp_all_reduce"
            and shared_order < task.order < dispatch_order
        ]
        result["shared_all_reduce"] = [
            min(all_reduces, key=lambda task: task.order).task_id
        ] if all_reduces else []
    else:
        result["shared_all_reduce"] = []
    return result


def _concurrency(intervals: list[tuple[int, int]]) -> tuple[int, float]:
    if not intervals:
        return 0, 0.0
    events: list[tuple[int, int]] = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    active = 0
    peak = 0
    for _tick, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        peak = max(peak, active)
    span = max(end for _start, end in intervals) - min(
        start for start, _end in intervals
    )
    busy = sum(end - start for start, end in intervals)
    return peak, busy / span if span else 0.0


def _resource_metrics(
    trace: RankTrace,
    tasks: list[Task],
    resource: str,
) -> dict[str, Any] | None:
    slices = [
        item
        for task in tasks
        for item in trace.slices_by_task.get(task.task_id, [])
        if item.resource == resource
    ]
    if not slices:
        return None
    frequency = trace.frequency_hz
    intervals = [(item.start, item.end) for item in slices]
    durations = [
        (item.end - item.start) / frequency * 1e6
        for item in slices
    ]
    start = min(item.start for item in slices)
    end = max(item.end for item in slices)
    span_ticks = end - start
    busy_ticks = sum(item.end - item.start for item in slices)
    available_cores = sum(kind == resource for kind in trace.core_types)
    expected_slices = sum(
        task.block_num * task.resource_slices_per_block(resource)
        for task in tasks
    )
    peak, average = _concurrency(intervals)
    schedulable = min(available_cores, expected_slices)
    return {
        "available_cores": available_cores,
        "expected_slices": expected_slices,
        "observed_slices": len(slices),
        "distinct_cores": len({item.core for item in slices}),
        "waves_at_full_resource": (
            math.ceil(expected_slices / available_cores)
            if expected_slices and available_cores
            else 0
        ),
        "slice_duration_us_p50": _round(_percentile(durations, 0.50)),
        "slice_duration_us_p99": _round(_percentile(durations, 0.99)),
        "slice_duration_us_max": _round(max(durations)),
        "span_us": _round(span_ticks / frequency * 1e6),
        "busy_us": _round(busy_ticks / frequency * 1e6),
        "peak_concurrency": peak,
        "average_concurrency": _round(average),
        "unused_cores_at_peak": max(0, available_cores - peak),
        "full_resource_utilization": (
            _round(busy_ticks / (span_ticks * available_cores), 4)
            if span_ticks and available_cores
            else 0.0
        ),
        "packing_efficiency": (
            _round(average / schedulable, 4)
            if schedulable
            else 0.0
        ),
    }


def _stage_metrics(
    trace: RankTrace,
    task_ids: list[str],
) -> dict[str, Any] | None:
    tasks = [
        trace.task_by_id[task_id]
        for task_id in task_ids
        if task_id in trace.task_by_id
    ]
    all_slices = [
        item
        for task in tasks
        for item in trace.slices_by_task.get(task.task_id, [])
    ]
    if not tasks or not all_slices:
        return None
    frequency = trace.frequency_hz
    task_spans = []
    for task in tasks:
        slices = trace.slices_by_task.get(task.task_id, [])
        if slices:
            task_spans.append(
                (max(item.end for item in slices) - min(item.start for item in slices))
                / frequency
                * 1e6,
            )
    start = min(item.start for item in all_slices)
    end = max(item.end for item in all_slices)
    result: dict[str, Any] = {
        "task_ids": [task.task_id for task in tasks],
        "task_instances": len(tasks),
        "logical_blocks": sum(task.block_num for task in tasks),
        "blocks_per_task": [task.block_num for task in tasks],
        "start_tick": start,
        "end_tick": end,
        "stage_span_us": _round((end - start) / frequency * 1e6),
        "task_instance_span_us_p50": _round(_percentile(task_spans, 0.50)),
        "task_instance_span_us_p99": _round(_percentile(task_spans, 0.99)),
        "task_instance_span_us_max": _round(max(task_spans)),
        "early_dispatch": all(task.early_dispatch for task in tasks),
        "resources": {},
    }
    for resource in ("aic", "aiv"):
        metrics = _resource_metrics(trace, tasks, resource)
        if metrics is not None:
            result["resources"][resource] = metrics
    return result


def _rank_metrics(trace: RankTrace) -> dict[str, Any]:
    all_slices = trace.all_slices
    if not all_slices:
        return {}
    frequency = trace.frequency_hz
    start = min(item.start for item in all_slices)
    end = max(item.end for item in all_slices)
    layers: dict[str, Any] = {}
    for layer in _LAYER_PREFIX:
        stage_ids = _find_layer_task_ids(trace, layer)
        layers[layer] = {
            stage: metrics
            for stage, task_ids in stage_ids.items()
            if (metrics := _stage_metrics(trace, task_ids)) is not None
        }

    tp_spans = []
    for task in trace.tasks:
        if task.name != "tp_all_reduce":
            continue
        slices = trace.slices_by_task.get(task.task_id, [])
        if slices:
            tp_spans.append(
                {
                    "task_id": task.task_id,
                    "order": task.order,
                    "span_us": _round(
                        (
                            max(item.end for item in slices)
                            - min(item.start for item in slices)
                        )
                        / frequency
                        * 1e6,
                    ),
                },
            )
    return {
        "trace_start_tick": start,
        "trace_end_tick": end,
        "makespan_us": _round((end - start) / frequency * 1e6),
        "critical_path": trace.critical_path,
        "tp_all_reduce": {
            "task_count": len(tp_spans),
            "spans": tp_spans,
            "span_us_p50": _round(
                _percentile([item["span_us"] for item in tp_spans], 0.50),
            ),
            "span_us_max": _round(
                max((item["span_us"] for item in tp_spans), default=0.0),
            ),
            "interpretation": (
                "Long spans may include peer-arrival spin inside the kernel; "
                "do not interpret them as pure reduction arithmetic."
            ),
        },
        "layers": layers,
    }


def _clock_alignment(traces: list[RankTrace]) -> dict[str, Any]:
    """Report timestamp metadata without claiming a shared rank clock.

    ``read_perf_data`` normalizes every rank to its own local origin. Similar
    terminal makespans do not reconstruct the removed offsets, so cross-rank
    timestamp subtraction remains disabled unless a future caller supplies a
    documented external common-clock anchor.
    """
    frequencies = sorted({trace.frequency_hz for trace in traces})
    terminal_ticks = {
        trace.tag: max((item.end for item in trace.all_slices), default=0)
        for trace in traces
    }
    same_frequency = len(frequencies) == 1 and bool(frequencies)
    positive_terminals = [
        tick for tick in terminal_ticks.values() if tick > 0
    ]
    terminal_skew_us: float | None = None
    if same_frequency and positive_terminals:
        terminal_skew_us = (
            max(positive_terminals) - min(positive_terminals)
        ) / frequencies[0] * 1e6
    return {
        "frequencies_hz": frequencies,
        "same_frequency": same_frequency,
        "terminal_end_ticks": terminal_ticks,
        "terminal_end_skew_us": (
            _round(terminal_skew_us)
            if terminal_skew_us is not None
            else None
        ),
        "cross_rank_tick_math_enabled": False,
        "external_common_clock_anchor": None,
        "evidence_level": "per-rank normalized",
        "interpretation": (
            "The runtime reader normalizes every rank to a local origin. "
            "Terminal alignment is descriptive only and cannot authorize "
            "cross-rank timestamp subtraction."
        ),
    }


def _routed_tile_stage(stages: dict[str, Any]) -> dict[str, Any]:
    """Choose one non-duplicated stage that represents active receive tiles."""
    for stage in (
        "expert_gate_up_act",
        "expert_gate",
        "expert_gate_up",
    ):
        if stage in stages:
            return stages[stage]
    return {}


def _receive_tile_imbalance(
    ranks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Summarize route-dependent receive-tile counts for each MoE layer."""
    result: dict[str, Any] = {}
    for layer in _LAYER_PREFIX:
        counts = {
            rank: int(
                _routed_tile_stage(
                    rank_data.get("layers", {}).get(layer, {}),
                ).get("task_instances", 0)
            )
            for rank, rank_data in ranks.items()
        }
        values = list(counts.values())
        total = sum(values)
        mean = statistics.fmean(values) if values else 0.0
        variance = (
            statistics.fmean((value - mean) ** 2 for value in values)
            if values
            else 0.0
        )
        minimum = min(values, default=0)
        maximum = max(values, default=0)
        result[layer] = {
            "receive_tiles_by_rank": counts,
            "total_receive_tiles": total,
            "min_receive_tiles": minimum,
            "max_receive_tiles": maximum,
            "max_min_skew_tiles": maximum - minimum,
            "mean_receive_tiles": _round(mean),
            "coefficient_of_variation": (
                _round(math.sqrt(variance) / mean, 4) if mean else 0.0
            ),
            "zero_tile_ranks": [
                rank for rank, value in counts.items() if value == 0
            ],
            "interpretation": (
                "One selected routed task instance corresponds to one active "
                "receive-tile slice. Split kernels prefer activation, then "
                "gate; legacy kernels use fused gate/up."
            ),
        }
    return result


def _predecessors(trace: RankTrace, task_id: str) -> list[dict[str, Any]]:
    return [
        edge
        for edge in trace.edges
        if str(edge.get("succ")) == str(task_id)
    ]


def _task_end(trace: RankTrace, task_id: str) -> int | None:
    slices = trace.slices_by_task.get(str(task_id), [])
    return max((item.end for item in slices), default=None)


def _arrival_analysis(
    traces: list[RankTrace],
    ranks: dict[str, dict[str, Any]],
    clock_alignment: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not traces:
        return result
    comparable_ticks = bool(
        clock_alignment.get("cross_rank_tick_math_enabled")
    )
    frequency = traces[0].frequency_hz
    for layer in _LAYER_PREFIX:
        layer_result: dict[str, Any] = {}
        for label, producer_stage, wait_stage in _ARRIVAL_PAIRS:
            producers = []
            waiters = []
            for trace in traces:
                rank_data = ranks[trace.tag]
                stage_data = rank_data.get("layers", {}).get(layer, {})
                producer = stage_data.get(producer_stage)
                waiter = stage_data.get(wait_stage)
                if producer is not None:
                    producers.append(
                        {
                            "rank": trace.tag,
                            "start_tick": producer["start_tick"],
                            "end_tick": producer["end_tick"],
                            "span_us": producer["stage_span_us"],
                        },
                    )
                if waiter is not None:
                    wait_task_id = waiter["task_ids"][0]
                    predecessors = _predecessors(trace, wait_task_id)
                    explicit = [
                        edge
                        for edge in predecessors
                        if edge.get("source") == "explicit"
                    ]
                    non_explicit_ends = [
                        end
                        for edge in predecessors
                        if edge.get("source") != "explicit"
                        if (end := _task_end(trace, str(edge.get("pred"))))
                        is not None
                    ]
                    waiters.append(
                        {
                            "rank": trace.tag,
                            "start_tick": waiter["start_tick"],
                            "end_tick": waiter["end_tick"],
                            "span_us": waiter["stage_span_us"],
                            "has_explicit_producer_dependency": bool(explicit),
                            "explicit_predecessors": [
                                {
                                    "task_id": str(edge.get("pred")),
                                    "name": trace.task_by_id.get(
                                        str(edge.get("pred")),
                                        Task("", -1, "unknown", 0, (), False),
                                    ).name,
                                }
                                for edge in explicit
                            ],
                            "anchor_ready_tick_estimate": (
                                max(non_explicit_ends)
                                if non_explicit_ends
                                else None
                            ),
                        },
                    )
            if not producers or not waiters:
                continue
            latest = (
                max(producers, key=lambda item: item["end_tick"])
                if comparable_ticks
                else None
            )
            earliest = (
                min(producers, key=lambda item: item["end_tick"])
                if comparable_ticks
                else None
            )
            producer_by_rank = {
                item["rank"]: item
                for item in producers
            }
            waiter_details = []
            for waiter in waiters:
                rank = waiter["rank"]
                local_producer = producer_by_rank.get(rank)
                peer_producers = [
                    item for item in producers if item["rank"] != rank
                ]
                latest_peer = (
                    max(peer_producers, key=lambda item: item["end_tick"])
                    if peer_producers
                    else None
                )
                detail = dict(waiter)
                if comparable_ticks and latest_peer is not None:
                    detail["latest_peer_producer_rank"] = latest_peer["rank"]
                    detail["remote_arrival_after_wait_start_us"] = _round(
                        max(
                            0,
                            latest_peer["end_tick"] - waiter["start_tick"],
                        )
                        / frequency
                        * 1e6,
                    )
                if (
                    comparable_ticks
                    and local_producer is not None
                    and waiter["anchor_ready_tick_estimate"] is not None
                ):
                    detail["explicit_dependency_delay_us"] = _round(
                        max(
                            0,
                            local_producer["end_tick"]
                            - int(waiter["anchor_ready_tick_estimate"]),
                        )
                        / frequency
                        * 1e6,
                    )
                if comparable_ticks and local_producer is not None:
                    all_ready_tick = max(
                        item["end_tick"]
                        for item in producers
                    )
                    detail["wait_overlap_completion_upper_bound_us"] = _round(
                        max(0, waiter["end_tick"] - all_ready_tick)
                        / frequency
                        * 1e6,
                    )
                waiter_details.append(detail)
            layer_result[label] = {
                "clock_domain_comparable": comparable_ticks,
                "clock_evidence_level": clock_alignment.get("evidence_level"),
                "producer_end_skew_us": (
                    _round(
                        (latest["end_tick"] - earliest["end_tick"])
                        / frequency
                        * 1e6,
                    )
                    if latest is not None and earliest is not None
                    else None
                ),
                "earliest_producer_rank": (
                    earliest["rank"] if earliest is not None else None
                ),
                "latest_producer_rank": (
                    latest["rank"] if latest is not None else None
                ),
                "producer_ranks": producers,
                "wait_ranks": waiter_details,
            }
        result[layer] = layer_result
    return result


def _aggregate_findings(
    ranks: dict[str, dict[str, Any]],
    arrivals: dict[str, Any],
    receive_imbalance: dict[str, Any],
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for layer in _LAYER_PREFIX:
        fused_gate_rows = []
        split_gate_rows: dict[str, list[tuple[str, dict[str, Any]]]] = {
            "expert_gate": [],
            "expert_up": [],
            "expert_gate_up_act": [],
        }
        shared_rows = []
        scatter_rows = []
        wait_rows = []
        for rank, rank_data in ranks.items():
            stages = rank_data.get("layers", {}).get(layer, {})
            gate = stages.get("expert_gate_up")
            if gate and "aic" in gate["resources"]:
                fused_gate_rows.append((rank, gate))
            for stage in split_gate_rows:
                split = stages.get(stage)
                if split and "aic" in split["resources"]:
                    split_gate_rows[stage].append((rank, split))
            shared = stages.get("shared_mlp")
            if shared and "aic" in shared["resources"]:
                shared_rows.append((rank, shared))
            scatter = stages.get("combine_scatter")
            if scatter and "aiv" in scatter["resources"]:
                scatter_rows.append((rank, scatter))
            wait = stages.get("combine_wait")
            if wait and "aiv" in wait["resources"]:
                wait_rows.append((rank, wait))

        imbalance = receive_imbalance.get(layer, {})
        if imbalance.get("max_min_skew_tiles", 0) > 0:
            findings.append(
                {
                    "layer": layer,
                    "severity": "high",
                    "finding": (
                        "Routed receive-tile counts are imbalanced across "
                        f"ranks: {imbalance.get('receive_tiles_by_rank', {})}; "
                        "max-min skew is "
                        f"{imbalance.get('max_min_skew_tiles', 0)} tiles."
                    ),
                    "action": (
                        "Treat the latest routed expert as the combine critical "
                        "producer. Validate scheduling changes on a second "
                        "heterogeneous-token workload before generalizing."
                    ),
                },
            )

        if any(split_gate_rows.values()):
            for stage, rows in split_gate_rows.items():
                if not rows:
                    continue
                resource = (
                    "aiv" if stage == "expert_gate_up_act" else "aic"
                )
                resource_rows = [
                    (rank, row)
                    for rank, row in rows
                    if resource in row["resources"]
                ]
                if not resource_rows:
                    continue
                p50 = statistics.median(
                    row["resources"][resource]["slice_duration_us_p50"]
                    for _rank, row in resource_rows
                )
                blocks = sorted(
                    {
                        block
                        for _rank, row in resource_rows
                        for block in row["blocks_per_task"]
                    },
                )
                findings.append(
                    {
                        "layer": layer,
                        "severity": (
                            "high"
                            if p50 > 30.0
                            else "info"
                            if p50 >= 10.0
                            else "medium"
                        ),
                        "finding": (
                            f"Routed {stage} {resource.upper()} slice p50 is "
                            f"{p50:.1f} us with {blocks} logical blocks per "
                            "active receive tile."
                        ),
                        "action": (
                            "Keep routed compute tasks near the requested "
                            "10-30 us range while minimizing stage span and "
                            "waves; do not optimize slice time in isolation."
                        ),
                    },
                )
        elif fused_gate_rows:
            gate_p50 = statistics.median(
                row["resources"]["aic"]["slice_duration_us_p50"]
                for _rank, row in fused_gate_rows
            )
            blocks = sorted(
                {
                    block
                    for _rank, row in fused_gate_rows
                    for block in row["blocks_per_task"]
                },
            )
            findings.append(
                {
                    "layer": layer,
                    "severity": "high" if gate_p50 > 30.0 else "info",
                    "finding": (
                        f"Routed gate/up AIC slice p50 is {gate_p50:.1f} us "
                        f"with {blocks} logical blocks per active expert tile."
                    ),
                    "action": (
                        "Sweep a smaller routed gate N tile and compare task "
                        "duration, waves, stage span, and unprofiled wall p50."
                    ),
                },
            )
        else:
            findings.append(
                {
                    "layer": layer,
                    "severity": "high",
                    "finding": (
                        "Some ranks have no routed expert compute while other "
                        "ranks do, indicating route-dependent EP imbalance."
                    ),
                    "action": (
                        "Use all-rank arrival analysis; do not attribute the "
                        "entire combine_wait span to the wait kernel itself."
                    ),
                },
            )

        combine = arrivals.get(layer, {}).get("combine")
        if combine and (combine.get("producer_end_skew_us") or 0.0) > 100.0:
            findings.append(
                {
                    "layer": layer,
                    "severity": "high",
                    "finding": (
                        "Combine producer completion skew is "
                        f"{combine['producer_end_skew_us']:.1f} us; latest "
                        f"producer is {combine['latest_producer_rank']}."
                    ),
                    "action": (
                        "Optimize the late routed-expert/scatter producer. "
                        "Removing an explicit wait dependency alone cannot "
                        "eliminate cross-rank producer skew."
                    ),
                },
            )
        elif wait_rows:
            max_wait_rank, max_wait = max(
                wait_rows,
                key=lambda item: item[1]["stage_span_us"],
            )
            max_wait_us = max_wait["stage_span_us"]
            local_tiles = int(
                ranks[max_wait_rank]
                .get("layers", {})
                .get(layer, {})
                .get("expert_gate_up_act",
                     ranks[max_wait_rank]
                     .get("layers", {})
                     .get(layer, {})
                     .get("expert_gate",
                          ranks[max_wait_rank]
                          .get("layers", {})
                          .get(layer, {})
                          .get("expert_gate_up", {})))
                .get("task_instances", 0)
            )
            if max_wait_us > 100.0:
                findings.append(
                    {
                        "layer": layer,
                        "severity": "high",
                        "finding": (
                            f"Combine wait reaches {max_wait_us:.1f} us on "
                            f"{max_wait_rank}, which has {local_tiles} local "
                            "routed receive tiles. Cross-rank completion "
                            "subtraction is unavailable for this capture."
                        ),
                        "action": (
                            "Treat this as a remote-producer tail signal, not "
                            "wait-kernel arithmetic. Reduce routed compute and "
                            "scatter tails; separately check whether local "
                            "scatter overlaps the wait."
                        ),
                    },
                )

        if shared_rows:
            max_peak = max(
                row["resources"]["aic"]["peak_concurrency"]
                for _rank, row in shared_rows
            )
            max_span = max(row["stage_span_us"] for _rank, row in shared_rows)
            if max_peak <= 1:
                findings.append(
                    {
                        "layer": layer,
                        "severity": "medium",
                        "finding": (
                            "Shared expert is one mixed task using at most one "
                            f"AIC at a time; observed span reaches {max_span:.1f} us."
                        ),
                        "action": (
                            "Split its five 32-channel gate/up tiles and "
                            "write-disjoint down-projection tiles across cores "
                            "without creating a wide [BATCH,160] Vec tile."
                        ),
                    },
                )

        if scatter_rows:
            max_slice = max(
                row["resources"]["aiv"]["slice_duration_us_max"]
                for _rank, row in scatter_rows
            )
            max_span = max(row["stage_span_us"] for _rank, row in scatter_rows)
            if max_slice > 30.0 or max_span > 100.0:
                findings.append(
                    {
                        "layer": layer,
                        "severity": "medium",
                        "finding": (
                            "Combine scatter has route-skewed AIV blocks: "
                            f"max slice {max_slice:.1f} us, max span "
                            f"{max_span:.1f} us."
                        ),
                        "action": (
                            "Evaluate write-disjoint (expert, source-rank) "
                            "scatter blocks while preserving one notify per "
                            "expert lane and peer."
                        ),
                    },
                )
    return findings


def _markdown_table(rows: list[list[Any]], headers: list[str]) -> list[str]:
    output = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    output.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    )
    return output


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Step3p5 L0-L4 MoE DFX report",
        "",
        "This report keeps L3 and L4 separate and reports AIC/AIV resources "
        "independently. `tp_all_reduce` spans may include peer-arrival spin.",
        "",
        f"- LOW-WAIT reference: `{report['reference_rank']}`",
        f"- DFX root: `{report['dfx_root']}`",
        "- Cross-rank tick math: "
        f"`{report['clock_alignment']['cross_rank_tick_math_enabled']}` "
        "(per-rank normalized clocks; no external common-clock anchor)",
        "- Terminal end skew: "
        f"`{report['clock_alignment'].get('terminal_end_skew_us')} us`",
        "",
        "## Rank overview",
        "",
    ]
    rank_rows = []
    for rank, data in report["ranks"].items():
        critical = data.get("critical_path", {})
        rank_rows.append(
            [
                rank,
                f"{data['makespan_us'] / 1000:.3f}",
                f"{critical.get('static_cpm_ms', 0.0):.3f}",
                f"{critical.get('compute_ms', 0.0):.3f}",
                f"{critical.get('data_wait_ms', 0.0):.3f}",
                f"{data['tp_all_reduce']['span_us_max']:.1f}",
            ],
        )
    lines.extend(
        _markdown_table(
            rank_rows,
            [
                "rank",
                "makespan ms",
                "static CPM ms",
                "observed compute ms",
                "data-wait ms",
                "max TP AR span us",
            ],
        ),
    )

    for layer in _LAYER_PREFIX:
        lines.extend(["", f"## {layer} MoE stages", ""])
        imbalance = report["receive_tile_imbalance"].get(layer, {})
        lines.extend(
            [
                "Routed receive tiles by rank: "
                f"`{imbalance.get('receive_tiles_by_rank', {})}`; "
                "max-min skew "
                f"`{imbalance.get('max_min_skew_tiles', 0)}`.",
                "",
            ],
        )
        stage_rows = []
        for rank, data in report["ranks"].items():
            stages = data.get("layers", {}).get(layer, {})
            for stage in (
                "norm_quant",
                "gate_fanout",
                "gate_topk",
                "shared_mlp",
                "shared_all_reduce",
                "dispatch_push",
                "dispatch_wait",
                "dispatch_gather",
                "expert_gate_up",
                "expert_gate",
                "expert_up",
                "expert_gate_up_act",
                "routed_h_quant",
                "expert_down",
                "combine_scatter",
                "combine_wait",
                "combine_reduce",
            ):
                metrics = stages.get(stage)
                if metrics is None:
                    continue
                aic = metrics["resources"].get("aic", {})
                aiv = metrics["resources"].get("aiv", {})
                stage_rows.append(
                    [
                        rank,
                        stage,
                        metrics["task_instances"],
                        metrics["logical_blocks"],
                        f"{metrics['stage_span_us']:.1f}",
                        (
                            f"{aic.get('slice_duration_us_p50', 0.0):.1f}/"
                            f"{aic.get('slice_duration_us_p99', 0.0):.1f}/"
                            f"{aic.get('slice_duration_us_max', 0.0):.1f}"
                            if aic
                            else "-"
                        ),
                        (
                            f"{aiv.get('slice_duration_us_p50', 0.0):.1f}/"
                            f"{aiv.get('slice_duration_us_p99', 0.0):.1f}/"
                            f"{aiv.get('slice_duration_us_max', 0.0):.1f}"
                            if aiv
                            else "-"
                        ),
                        (
                            f"{aic.get('peak_concurrency', 0)}/"
                            f"{aiv.get('peak_concurrency', 0)}"
                        ),
                    ],
                )
        lines.extend(
            _markdown_table(
                stage_rows,
                [
                    "rank",
                    "stage",
                    "tasks",
                    "blocks",
                    "span us",
                    "AIC p50/p99/max us",
                    "AIV p50/p99/max us",
                    "peak AIC/AIV",
                ],
            ),
        )

        lines.extend(["", f"### {layer} all-rank arrivals", ""])
        for label in ("dispatch", "combine"):
            arrival = report["arrivals"].get(layer, {}).get(label)
            if not arrival:
                continue
            producer_skew = arrival.get("producer_end_skew_us")
            producer_skew_text = (
                f"{producer_skew:.1f} us"
                if producer_skew is not None
                else "not comparable"
            )
            latest_producer = (
                f"`{arrival['latest_producer_rank']}`"
                if arrival.get("latest_producer_rank") is not None
                else "`not comparable`"
            )
            lines.append(
                f"- **{label}** producer end skew: "
                f"`{producer_skew_text}`; "
                f"latest producer: {latest_producer}.",
            )
            arrival_rows = []
            producer_by_rank = {
                item["rank"]: item
                for item in arrival["producer_ranks"]
            }
            for waiter in arrival["wait_ranks"]:
                producer = producer_by_rank.get(waiter["rank"], {})
                arrival_rows.append(
                    [
                        waiter["rank"],
                        f"{producer.get('span_us', 0.0):.1f}",
                        f"{waiter['span_us']:.1f}",
                        (
                            f"{waiter['remote_arrival_after_wait_start_us']:.1f}"
                            if "remote_arrival_after_wait_start_us" in waiter
                            else "-"
                        ),
                        str(waiter["has_explicit_producer_dependency"]),
                        (
                            f"{waiter['explicit_dependency_delay_us']:.1f}"
                            if "explicit_dependency_delay_us" in waiter
                            else "-"
                        ),
                        (
                            f"{waiter['wait_overlap_completion_upper_bound_us']:.1f}"
                            if "wait_overlap_completion_upper_bound_us" in waiter
                            else "-"
                        ),
                    ],
                )
            lines.extend(
                _markdown_table(
                    arrival_rows,
                    [
                        "rank",
                        "producer span us",
                        "wait span us",
                        "remote arrival after wait start us",
                        "explicit dep",
                        "explicit delay us",
                        "completion saving upper bound us",
                    ],
                ),
            )
            lines.append("")

    lines.extend(["", "## Findings", ""])
    for finding in report["findings"]:
        lines.append(
            f"- **{finding['severity'].upper()} {finding['layer']}** — "
            f"{finding['finding']} {finding['action']}",
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(
    build_dir: Path,
    out: Path,
    *,
    run_critical_path: bool = True,
    top: int = 30,
) -> dict[str, Any]:
    """Analyze one compiled L0-L4 DFX directory and write JSON/Markdown."""
    dfx_root = build_dir / "dfx_outputs"
    if not dfx_root.exists():
        raise FileNotFoundError(f"no dfx_outputs under {build_dir}")
    out.mkdir(parents=True, exist_ok=True)

    if run_critical_path:
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "simpler_setup.tools.critical_path",
                str(dfx_root),
                "--top",
                str(top),
                "--stdout",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        (out / "critical_path_stdout.txt").write_text(
            process.stdout + process.stderr,
            encoding="utf-8",
        )
        if process.returncode != 0:
            print(
                "[five-layer-moe-dfx] critical_path returned "
                f"{process.returncode}",
                file=sys.stderr,
            )

    traces = [
        trace
        for path in sorted(dfx_root.rglob("l2_swimlane_records.json"))
        if (trace := _load_rank(path.parent, dfx_root)) is not None
    ]
    if not traces:
        raise RuntimeError(f"no complete rank traces under {dfx_root}")
    ranks = {trace.tag: _rank_metrics(trace) for trace in traces}
    reference_rank = min(ranks, key=lambda tag: ranks[tag]["makespan_us"])
    clock_alignment = _clock_alignment(traces)
    arrivals = _arrival_analysis(traces, ranks, clock_alignment)
    receive_imbalance = _receive_tile_imbalance(ranks)
    report = {
        "schema": "step3p5.five-layer-moe-dfx.v1",
        "build_dir": str(build_dir),
        "dfx_root": str(dfx_root),
        "reference_rank": reference_rank,
        "reference_note": (
            "Minimum makespan is a LOW-WAIT heuristic only. Compare every "
            "rank for dispatch/combine arrival and TP all-reduce spin."
        ),
        "clock_alignment": clock_alignment,
        "ranks": ranks,
        "arrivals": arrivals,
        "receive_tile_imbalance": receive_imbalance,
        "findings": _aggregate_findings(
            ranks,
            arrivals,
            receive_imbalance,
        ),
    }
    json_path = out / "moe_dfx_report.json"
    markdown_path = out / "moe_critical_path_report.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(report, markdown_path)
    print(
        f"[five-layer-moe-dfx] LOW-WAIT reference={reference_rank} "
        f"makespan={ranks[reference_rank]['makespan_us'] / 1000:.3f} ms",
        flush=True,
    )
    print(f"MOE_DFX_REPORT={json_path}", flush=True)
    print(f"MOE_CRITICAL_PATH_REPORT={markdown_path}", flush=True)
    return report


def main() -> int:
    args = _parse_args()
    analyze(
        Path(args.build_dir),
        Path(args.out),
        run_critical_path=not args.skip_critical_path,
        top=args.top,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
