# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Card-free tests for the focused L0-L4 MoE DFX analyzer."""
from __future__ import annotations

from pathlib import Path

from tools.step3p5.analyze_five_layer_moe_dfx import (
    RankTrace,
    Slice,
    _aggregate_findings,
    _arrival_analysis,
    _clock_alignment,
    _receive_tile_imbalance,
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


def test_clock_alignment_never_infers_a_common_clock_from_terminals() -> None:
    aligned = _clock_alignment(
        [
            _trace("rank0/d0", end_tick=1_000_000),
            _trace("rank1/d0", end_tick=1_000_250),
        ],
    )
    assert aligned["same_frequency"]
    assert aligned["terminal_end_skew_us"] == 5.0
    assert not aligned["cross_rank_tick_math_enabled"]
    assert aligned["external_common_clock_anchor"] is None

    misaligned = _clock_alignment(
        [
            _trace("rank0/d0", end_tick=1_000_000),
            _trace("rank1/d0", end_tick=1_010_000),
        ],
    )
    assert misaligned["terminal_end_skew_us"] == 200.0
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


def test_receive_tile_imbalance_uses_gate_up_task_instances() -> None:
    ranks = {
        "rank0/d0": {
            "layers": {
                "L3": {"expert_gate_up": {"task_instances": 0}},
                "L4": {"expert_gate_up": {"task_instances": 4}},
            },
        },
        "rank1/d0": {
            "layers": {
                "L3": {"expert_gate_up": {"task_instances": 8}},
                "L4": {"expert_gate_up": {"task_instances": 4}},
            },
        },
    }
    result = _receive_tile_imbalance(ranks)
    assert result["L3"]["receive_tiles_by_rank"] == {
        "rank0/d0": 0,
        "rank1/d0": 8,
    }
    assert result["L3"]["max_min_skew_tiles"] == 8
    assert result["L3"]["zero_tile_ranks"] == ["rank0/d0"]
    assert result["L4"]["max_min_skew_tiles"] == 0


def test_receive_tile_imbalance_prefers_split_activation_instances() -> None:
    ranks = {
        "rank0/d0": {
            "layers": {
                "L3": {
                    "expert_gate": {"task_instances": 4},
                    "expert_up": {"task_instances": 4},
                    "expert_gate_up_act": {"task_instances": 4},
                },
                "L4": {},
            },
        },
        "rank1/d0": {
            "layers": {
                "L3": {
                    "expert_gate": {"task_instances": 8},
                    "expert_up": {"task_instances": 8},
                    "expert_gate_up_act": {"task_instances": 8},
                },
                "L4": {},
            },
        },
    }
    result = _receive_tile_imbalance(ranks)
    assert result["L3"]["receive_tiles_by_rank"] == {
        "rank0/d0": 4,
        "rank1/d0": 8,
    }
    assert result["L3"]["total_receive_tiles"] == 12


def test_arrival_analysis_disables_cross_rank_subtraction_without_anchor() -> None:
    traces = [
        _trace("rank0/d0", end_tick=1_000_000),
        _trace("rank1/d0", end_tick=1_010_000),
    ]
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
                        "task_ids": ["missing"],
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
    assert all(
        "remote_arrival_after_wait_start_us" not in waiter
        for waiter in combine["wait_ranks"]
    )


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
        _receive_tile_imbalance(ranks),
    )
    assert findings
