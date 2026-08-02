# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Production-faithful two-layer attention harness for eight-way tensor parallelism.

Run one attention critical-path tuning iteration with:

    python -m tests.step3p5.harnesses._stage_two_layer_attn \
        --device 0,1,2,3,4,5,6,7 --out /tmp/attn_iter1 \
        --num-blocks 512 --context-lens 1024,65536

The output directory contains:

``itl_report.json``
    Two-layer wall-clock latency for each context length, including
    min/mean/p50/p99/max. Its ITL accounting matches the canonical whole-network
    harness and can be compared across commits.
``critical_path_report.md`` (per rank)
    Output from ``simpler_setup.tools.critical_path``: static CPM dependency
    lower bound, observed critical path, and compute/data-wait/core-wait splits.
``uniformity_report.json`` and terminal summary
    Swimlane uniformity: per-AIC/AIV-core busy distribution, total bubbles, and
    lane spans for each attention kernel family. This is the dual metric of
    shortest critical path: when the path no longer improves, check whether
    physical cores remain idle.

Design choices
--------------
**Why two layers instead of the whole network**: a canonical 45-layer run takes
tens of minutes and produces more than 1,500 swimlane tasks. Attention accounts
for only a fraction of them, so MoE noise obscures its dependency chain. This
harness contains only attention, dense MLP, and TP all-reduce.

**Why two layers instead of one**: two layers preserve the handoff between
layers, including overlap between the L0 MLP all-reduce and the L1 attention
prologue.

**Why random weights are sufficient**: attention work is independent of weight
and KV-cache values. This harness measures scheduling and pipeline shape;
precision regression remains the responsibility of
``tests.step3p5.ci.run_whole_network_ci``.

**Why DFX capture uses two separate warm iterations**: the historical
``--dfx`` path set ``N1_DFX`` before the entire ITL loop, so every iteration was
profiled and the captured iteration was effectively cold. The first
``tp_all_reduce`` then spun inside the kernel while waiting for the slowest
rank. ``critical_path.py`` classified that wait as compute rather than stall.
Observed outliers were 70.8 ms, 96.8 ms, and 763 ms for a task whose median was
about 50 microseconds, incorrectly assigning 63 percent of the critical path to
all-reduce. Removing that first task made ranks converge at 44.18 ms with only
1 microsecond of spread. This harness therefore gives dependency generation and
swimlane capture one separate warmed iteration each; combining them would also
mix dependency-submit overhead into the swimlane measurement.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

BLOCK_SIZE = 128
# config.STORAGE_BATCH_CAPACITY, duplicated because the block-table size env
# must be set BEFORE models.step3p5.config is first imported (it computes
# BLOCK_TABLE_FLAT_DYN at module level). main() asserts the two agree.
STORAGE_BATCH = 16
_BF16 = torch.bfloat16
_F32 = torch.float32
_I32 = torch.int32


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="0,1,2,3,4,5,6,7",
                   help="TP=8 device ids, comma separated")
    p.add_argument("--out", required=True)
    p.add_argument("--num-blocks", type=int, default=512,
                   help="scheduler blocks per layer; KV slab = num_blocks*128 rows")
    p.add_argument("--context-lens", default="65536",
                   help="comma list of decode context lengths to time "
                        "(benchmark point is bs=1 / 65536)")
    p.add_argument("--iters", type=int, default=20, help="measured iters per context")
    p.add_argument("--warmup", type=int, default=3, help="warmup iters per context")
    p.add_argument(
        "--active-rows",
        type=int,
        default=1,
        help="runtime active rows per step; active rows use disjoint KV pages",
    )
    p.add_argument(
        "--active-context-lens",
        default="",
        help="optional comma-separated context length for each active row; "
             "defaults to the benchmark context for every active row",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="torch RNG seed used for all host fixtures",
    )
    p.add_argument(
        "--full-attn-qk-blocks-per-task",
        type=int,
        default=0,
        help="paged-cache blocks per full-attention QK logical task "
             "(default: config/env value)",
    )
    p.add_argument(
        "--full-attn-softmax-blocks-per-task",
        type=int,
        default=0,
        help="paged-cache blocks per full-attention softmax logical task "
             "(default: config/env value)",
    )
    p.add_argument(
        "--full-attn-online-softmax-blocks-per-task",
        type=int,
        default=0,
        help="paged-cache blocks reduced by each full-attention online-softmax "
        "SV+segment-recurrence task (default: config/env value)",
    )
    p.add_argument(
        "--full-attn-online-softmax-partials-per-reduce-task",
        type=int,
        default=0,
        help="segment partials merged by each parallel full-attention "
        "online-softmax reduce task (default: config/env value)",
    )
    p.add_argument(
        "--full-attn-out-proj-n-chunk",
        type=int,
        default=0,
        help="compatibility override for both full-attention out_proj matmul "
        "and vector N tiles (default: config/env value)",
    )
    p.add_argument(
        "--swa-out-proj-n-chunk",
        type=int,
        default=0,
        help="compatibility override for both SWA out_proj matmul and vector "
        "N tiles (default: config/env value)",
    )
    p.add_argument(
        "--full-attn-out-proj-matmul-n-chunk",
        type=int,
        default=0,
        help="decode full-attention out_proj matmul N tile "
        "(default: compatibility/config value)",
    )
    p.add_argument(
        "--full-attn-out-proj-vec-n-chunk",
        type=int,
        default=0,
        help="decode full-attention out_proj cast/residual N tile "
        "(default: compatibility/config value)",
    )
    p.add_argument(
        "--full-attn-out-proj-matmul-tiles-per-task",
        type=int,
        default=0,
        help="legal matmul N tiles processed sequentially by each full-attention "
        "out_proj logical task (default: config/env value)",
    )
    p.add_argument(
        "--swa-out-proj-matmul-n-chunk",
        type=int,
        default=0,
        help="decode SWA out_proj matmul N tile "
        "(default: compatibility/config value)",
    )
    p.add_argument(
        "--swa-out-proj-vec-n-chunk",
        type=int,
        default=0,
        help="decode SWA out_proj cast/residual N tile "
        "(default: compatibility/config value)",
    )
    p.add_argument(
        "--swa-out-proj-matmul-tiles-per-task",
        type=int,
        default=0,
        help="legal matmul N tiles processed sequentially by each SWA out_proj "
        "logical task (default: config/env value)",
    )
    p.add_argument(
        "--full-attn-out-proj-fuse-cast",
        action="store_true",
        help="experimental: fuse full-attention out_proj FP32-to-BF16 cast "
        "into each matmul logical task",
    )
    p.add_argument(
        "--swa-out-proj-fuse-cast",
        action="store_true",
        help="experimental: fuse SWA out_proj FP32-to-BF16 cast into each "
        "matmul logical task",
    )
    p.add_argument(
        "--tp-all-reduce-chunk",
        type=int,
        default=0,
        help="TP all-reduce hidden-column tile width "
        "(default: config/env value)",
    )
    p.add_argument("--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    p.add_argument("--dfx", action="store_true",
                   help="after timing, spend two extra WARM iters on dep_gen then "
                        "l2_swimlane, then run critical_path + uniformity")
    p.add_argument("--dfx-context", type=int, default=0,
                   help="context length for the DFX iters (default: largest --context-lens)")
    p.add_argument("--compile-only", action="store_true",
                   help="compile and exit (card-free preflight)")
    return p.parse_args()


def _devices(text: str) -> list[int]:
    ids = [int(x) for x in str(text).split(",") if x.strip()]
    if len(ids) != 8 or len(set(ids)) != 8:
        raise ValueError(f"need 8 distinct devices for TP=8, got {ids}")
    return ids


# --------------------------------------------------------------------------- #
# metadata                                                                    #
# --------------------------------------------------------------------------- #
def _step_metadata(
    *,
    context_len: int | None = None,
    context_lens: list[int] | None = None,
    num_blocks: int,
    batch: int,
    active_rows: int = 1,
):
    """Decode metadata with active rows and disjoint paged sequences."""
    if not 0 <= int(active_rows) <= int(batch):
        raise ValueError(
            f"active_rows must be in [0,{batch}], got {active_rows}",
        )
    if context_lens is None:
        if context_len is None:
            raise ValueError("one of context_len or context_lens is required")
        row_context_lens = [int(context_len)] * int(active_rows)
    else:
        row_context_lens = [int(length) for length in context_lens]
        if len(row_context_lens) != int(active_rows):
            raise ValueError(
                f"context_lens must contain exactly active_rows={active_rows} "
                f"values, got {row_context_lens}",
            )
    for row, row_context_len in enumerate(row_context_lens):
        if row_context_len <= 0:
            raise ValueError(
                f"context length for active row {row} must be positive, "
                f"got {row_context_len}",
            )
        row_blocks = (row_context_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        if row_blocks > num_blocks:
            raise ValueError(
                f"context {row_context_len} for active row {row} needs "
                f"{row_blocks} blocks > --num-blocks {num_blocks}",
            )
    seq = torch.ones(batch, dtype=_I32)
    pos = torch.zeros(batch, dtype=_I32)
    table = torch.zeros(batch, num_blocks, dtype=_I32)
    slot = torch.zeros(batch, dtype=_I32)
    for row, row_context_len in enumerate(row_context_lens):
        step = row_context_len - 1
        row_blocks = (row_context_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        table[row, :row_blocks] = (
            row * num_blocks
            + torch.arange(row_blocks, dtype=_I32)
        )
        seq[row] = row_context_len
        pos[row] = step
        slot[row] = row * num_blocks * BLOCK_SIZE + step
    return seq, pos, table.reshape(-1), slot


def _logical_task_count(
    context_lens: list[int],
    blocks_per_task: int,
) -> int:
    """Count actual-work logical tasks for a list of active rows."""
    if blocks_per_task <= 0:
        raise ValueError(f"blocks_per_task must be positive, got {blocks_per_task}")
    return sum(
        (
            (int(context_len) + BLOCK_SIZE - 1) // BLOCK_SIZE
            + blocks_per_task - 1
        )
        // blocks_per_task
        for context_len in context_lens
    )


def _sv_online_logical_task_count(
    context_lens: list[int],
    *,
    online_blocks_per_task: int,
    **_ignored: int,
) -> int:
    """The fused SV/online kernel launches one block per online logical task."""
    return _logical_task_count(context_lens, online_blocks_per_task)


# --------------------------------------------------------------------------- #
# swimlane uniformity                                                         #
# --------------------------------------------------------------------------- #
def _family(name: str) -> str:
    return re.sub(r"(_\d+)?(_(?:aic|aiv))?$", "", name)


def _resource_family(name: str, resource: str) -> str:
    """Keep mixed-kernel AIC/AIV slices in separate resource families."""
    family = _family(name)
    if name.endswith(("_aic", "_aiv")):
        return f"{family}_{resource}"
    return family


def analyze_uniformity(
    rank_dir: Path,
    *,
    expected_logical_blocks: dict[str, int] | None = None,
) -> dict | None:
    """Per-core occupancy + bubble accounting from one rank's swimlane records.

    critical_path.py answers "how long is the chain"; this answers "while the
    chain ran, how evenly were the cores loaded". A short chain with idle cores
    means work can still be moved off the chain; a long chain with every core
    saturated means the kernels themselves must get cheaper.

    Reported twice: over all tasks, and excluding ``tp_all_reduce``. The
    exclusion matters because an all-reduce spin-waits for peer ranks *inside*
    the kernel, so its swimlane span is arrival skew rather than work.

    A task's swimlane span (min start -> max end over its blocks) is NOT
    per-core exec time: a 24-lane SPMD task's span includes lane skew. Per-core
    busy time is summed from the raw slices, so it is skew-free.
    """
    rec = rank_dir / "l2_swimlane_records.json"
    if not rec.exists():
        return None
    sw = json.loads(rec.read_text(encoding="utf-8"))
    freq = int(sw["metadata"]["clock_freq_hz"])
    rows = sw["aicore_tasks"]
    if not rows:
        return None
    deps_path = rank_dir / "deps.json"
    name_path = rank_dir / "name_map.json"
    names: dict[str, str] = {}
    task_block_num: dict[str, int] = {}
    if deps_path.exists() and name_path.exists():
        deps = json.loads(deps_path.read_text(encoding="utf-8"))
        nmap = json.loads(name_path.read_text(encoding="utf-8")).get(
            "callable_id_to_name", {}
        )
        for t in deps["tasks"]:
            task_id = str(t["task_id"])
            task_block_num[task_id] = int(t.get("block_num", 0))
            ks = t.get("kernel_ids") or []
            cid = next((k for k in ks if k is not None and k >= 0), None)
            if cid is not None:
                names[task_id] = nmap.get(str(cid), f"cid{cid}")

    us = lambda tk: tk / freq * 1e6  # noqa: E731
    busy_all: collections.Counter = collections.Counter()
    busy_excl: collections.Counter = collections.Counter()
    fam_cores: dict[str, set] = collections.defaultdict(set)
    fam_busy: collections.Counter = collections.Counter()
    fam_intervals: dict[str, list[tuple[int, int, int]]] = collections.defaultdict(list)
    fam_tasks: dict[str, set[str]] = collections.defaultdict(set)
    t0, t1 = None, None
    for core, tid, _seq, st, en, _recv in rows:
        fam = _family(names.get(str(tid), "unknown"))
        busy_all[core] += en - st
        if not fam.startswith("tp_all_reduce"):
            busy_excl[core] += en - st
        t0 = st if t0 is None else min(t0, st)
        t1 = en if t1 is None else max(t1, en)
        fam_cores[fam].add(core)
        fam_busy[fam] += en - st
        fam_intervals[fam].append((st, en, core))
        fam_tasks[fam].add(str(tid))
    makespan = t1 - t0
    core_types = sw["metadata"].get("core_types", [])
    all_core_ids = list(range(int(sw["metadata"].get("num_cores", len(core_types)))))
    # Mixed-kernel tasks can expose both AIC and AIV slices under one task id.
    # Re-key them by actual core resource so AIC task grain is not accidentally
    # reported against all 72 physical cores.
    fam_cores.clear()
    fam_busy.clear()
    fam_intervals.clear()
    fam_tasks.clear()
    for core, tid, _seq, st, en, _recv in rows:
        resource = (
            core_types[core]
            if 0 <= core < len(core_types)
            else "unknown"
        )
        fam = _resource_family(names.get(str(tid), "unknown"), resource)
        fam_cores[fam].add(core)
        fam_busy[fam] += en - st
        fam_intervals[fam].append((st, en, core))
        fam_tasks[fam].add(str(tid))

    def occupancy(busy: collections.Counter, core_ids: list[int]) -> dict:
        vals = sorted(busy.get(c, 0) / makespan for c in core_ids)
        used = sum(busy.values())
        cores = len(core_ids)
        return {
            "occupancy_min": round(vals[0], 4),
            "occupancy_p50": round(vals[len(vals) // 2], 4),
            "occupancy_mean": round(statistics.fmean(vals), 4),
            "occupancy_max": round(vals[-1], 4),
            "occupancy_stdev": round(statistics.pstdev(vals), 4),
            "bubble_ratio": round(1.0 - used / (makespan * cores), 4),
            "busy_us_total": round(us(used), 1),
        }

    ncores = len(all_core_ids)
    resource_core_counts = collections.Counter(core_types)
    resource_core_ids = {
        resource: [core for core, kind in enumerate(core_types) if kind == resource]
        for resource in sorted(resource_core_counts)
    }

    def percentile(vals: list[float], q: float) -> float:
        if not vals:
            return 0.0
        ordered = sorted(vals)
        idx = min(len(ordered) - 1, int((len(ordered) - 1) * q))
        return ordered[idx]

    def family_stats(fam: str) -> dict:
        intervals = fam_intervals[fam]
        starts = [st for st, _en, _core in intervals]
        ends = [en for _st, en, _core in intervals]
        durations_us = [us(en - st) for st, en, _core in intervals]
        events = []
        for st, en, _core in intervals:
            events.append((st, 1))
            events.append((en, -1))
        active = 0
        peak = 0
        for _tick, delta in sorted(events, key=lambda item: (item[0], item[1])):
            active += delta
            peak = max(peak, active)
        task_ids = fam_tasks[fam]
        logical_blocks = sum(task_block_num.get(tid, 0) for tid in task_ids)
        expected = (
            expected_logical_blocks.get(fam)
            if expected_logical_blocks is not None
            else None
        )
        if expected is not None and logical_blocks != expected:
            print(
                f"[two-layer] warn: {fam} deps block count {logical_blocks} "
                f"!= expected {expected}",
                file=sys.stderr,
            )
        resource_types = sorted(
            {
                core_types[core]
                for _st, _en, core in intervals
                if 0 <= core < len(core_types)
            }
        )
        available_cores = sum(resource_core_counts[t] for t in resource_types)
        span_ticks = max(ends) - min(starts)
        average_concurrency = fam_busy[fam] / span_ticks
        schedulable_concurrency = min(available_cores, logical_blocks)
        return {
            "busy_us": round(us(fam_busy[fam]), 1),
            "distinct_cores": len(fam_cores[fam]),
            "resource_types": resource_types,
            "available_cores": available_cores,
            "logical_blocks": logical_blocks,
            "expected_logical_blocks": expected,
            "waves_at_full_resource": (
                (logical_blocks + available_cores - 1) // available_cores
                if logical_blocks and available_cores
                else 0
            ),
            "observed_slices": len(intervals),
            "slice_duration_us_p50": round(percentile(durations_us, 0.50), 3),
            "slice_duration_us_p99": round(percentile(durations_us, 0.99), 3),
            "slice_duration_us_max": round(max(durations_us), 3),
            "stage_span_us": round(us(span_ticks), 3),
            "peak_concurrency": peak,
            "average_concurrency": round(average_concurrency, 3),
            "full_resource_utilization": (
                round(fam_busy[fam] / (span_ticks * available_cores), 4)
                if available_cores
                else 0.0
            ),
            "packing_efficiency": (
                round(average_concurrency / schedulable_concurrency, 4)
                if schedulable_concurrency
                else 0.0
            ),
        }

    return {
        "makespan_us": round(us(makespan), 1),
        "n_cores": ncores,
        "resource_core_counts": dict(resource_core_counts),
        "all": occupancy(busy_all, all_core_ids),
        "excl_tp_all_reduce": occupancy(busy_excl, all_core_ids),
        "resource_occupancy": {
            resource: {
                "all": occupancy(
                    collections.Counter(
                        {core: busy_all.get(core, 0) for core in core_ids}
                    ),
                    core_ids,
                ),
                "excl_tp_all_reduce": occupancy(
                    collections.Counter(
                        {core: busy_excl.get(core, 0) for core in core_ids}
                    ),
                    core_ids,
                ),
            }
            for resource, core_ids in resource_core_ids.items()
        },
        "per_core_occupancy_excl_ar": {
            str(c): round(busy_excl.get(c, 0) / makespan, 4) for c in all_core_ids
        },
        "families": {
            fam: family_stats(fam)
            for fam in sorted(fam_busy, key=lambda f: -fam_busy[f])
        },
    }


def print_uniformity(tag: str, u: dict, *, reference: bool = False) -> None:
    mark = "  <== LOW-WAIT REFERENCE (minimum rank makespan)" if reference else ""
    print(f"\n[uniformity {tag}] makespan={u['makespan_us'] / 1000:.3f}ms "
          f"cores={u['n_cores']}{mark}")
    for label, key in (("all", "all"), ("excl tp_all_reduce", "excl_tp_all_reduce")):
        o = u[key]
        print(f"  {label:<20} busy={o['busy_us_total']:.0f}µs bubble={o['bubble_ratio'] * 100:.1f}% "
              f"occ min={o['occupancy_min']:.3f} p50={o['occupancy_p50']:.3f} "
              f"mean={o['occupancy_mean']:.3f} max={o['occupancy_max']:.3f} "
              f"sd={o['occupancy_stdev']:.3f}")
    for resource, reports in u["resource_occupancy"].items():
        o = reports["excl_tp_all_reduce"]
        print(
            f"  {resource:<20} excl-ar bubble={o['bubble_ratio'] * 100:.1f}% "
            f"occ p50={o['occupancy_p50']:.3f} mean={o['occupancy_mean']:.3f} "
            f"max={o['occupancy_max']:.3f}"
        )
    print("  | kernel family | busy µs | cores | blocks | span µs | "
          "slice p50/p99/max µs | peak/avg | waves | pack |")
    print("  |---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for fam, info in list(u["families"].items())[:18]:
        print(
            f"  | {fam} | {info['busy_us']:.1f} | {info['distinct_cores']} | "
            f"{info['logical_blocks']} | {info['stage_span_us']:.1f} | "
            f"{info['slice_duration_us_p50']:.2f}/"
            f"{info['slice_duration_us_p99']:.2f}/"
            f"{info['slice_duration_us_max']:.2f} | "
            f"{info['peak_concurrency']}/{info['average_concurrency']:.1f} | "
            f"{info['waves_at_full_resource']} | "
            f"{info['packing_efficiency'] * 100:.1f}% |"
        )


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> int:
    args = _parse_args()
    devices = _devices(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if not 0 <= args.active_rows <= STORAGE_BATCH:
        raise ValueError(
            f"--active-rows must be in [0,{STORAGE_BATCH}], got {args.active_rows}",
        )
    active_context_lens = [
        int(value)
        for value in str(args.active_context_lens).split(",")
        if value.strip()
    ]
    if args.active_rows == 0 and active_context_lens:
        raise ValueError("--active-context-lens is invalid when --active-rows=0")
    if active_context_lens and len(active_context_lens) != args.active_rows:
        raise ValueError(
            "--active-context-lens must contain exactly one value per active "
            f"row ({args.active_rows}), got {active_context_lens}",
        )

    # Config env MUST be set before models.step3p5.config is first imported:
    # KV_CACHE_ROWS_DYN / BLOCK_TABLE_FLAT_DYN / ROPE_SEQ_DYN are module-level.
    # The kernel derives its per-layer KV slab as
    #   k_cache.rows // input_rms.rows  (attention_full.py:220-221)
    # and input_rms keeps canonical LAYER_DYN=45 rows, so the pool must be
    # 45 slabs wide even though only slabs 0 and 1 are touched. Keeping the
    # canonical stride matters for perf: the DDR distance between a layer's KV
    # rows is part of what we are measuring.
    max_seq = int(args.num_blocks) * BLOCK_SIZE
    os.environ["PYPTO_STEP3P5_MAX_SEQ"] = str(max_seq)
    physical_blocks = int(args.num_blocks) * max(1, int(args.active_rows)) + 15
    os.environ["PYPTO_STEP3P5_KV_CACHE_ROWS"] = str(45 * physical_blocks * BLOCK_SIZE)
    os.environ["PYPTO_STEP3P5_ROPE_SEQ"] = str(max_seq)
    os.environ["PYPTO_STEP3P5_BLOCK_TABLE_FLAT"] = str(
        STORAGE_BATCH * int(args.num_blocks)
    )
    for arg_name, env_name in (
        (
            "tp_all_reduce_chunk",
            "PYPTO_STEP3P5_TP_ALL_REDUCE_CHUNK",
        ),
        (
            "full_attn_qk_blocks_per_task",
            "PYPTO_STEP3P5_FULL_ATTN_QK_BLOCKS_PER_TASK",
        ),
        (
            "full_attn_softmax_blocks_per_task",
            "PYPTO_STEP3P5_FULL_ATTN_SOFTMAX_BLOCKS_PER_TASK",
        ),
        (
            "full_attn_online_softmax_blocks_per_task",
            "PYPTO_STEP3P5_FULL_ATTN_ONLINE_SOFTMAX_BLOCKS_PER_TASK",
        ),
        (
            "full_attn_online_softmax_partials_per_reduce_task",
            "PYPTO_STEP3P5_FULL_ATTN_ONLINE_SOFTMAX_PARTIALS_PER_REDUCE_TASK",
        ),
        (
            "full_attn_out_proj_n_chunk",
            "PYPTO_STEP3P5_FULL_ATTN_OUT_PROJ_N_CHUNK",
        ),
        (
            "swa_out_proj_n_chunk",
            "PYPTO_STEP3P5_SWA_OUT_PROJ_N_CHUNK",
        ),
        (
            "full_attn_out_proj_matmul_n_chunk",
            "PYPTO_STEP3P5_FULL_ATTN_OUT_PROJ_MATMUL_N_CHUNK",
        ),
        (
            "full_attn_out_proj_vec_n_chunk",
            "PYPTO_STEP3P5_FULL_ATTN_OUT_PROJ_VEC_N_CHUNK",
        ),
        (
            "full_attn_out_proj_matmul_tiles_per_task",
            "PYPTO_STEP3P5_FULL_ATTN_OUT_PROJ_MATMUL_TILES_PER_TASK",
        ),
        (
            "swa_out_proj_matmul_n_chunk",
            "PYPTO_STEP3P5_SWA_OUT_PROJ_MATMUL_N_CHUNK",
        ),
        (
            "swa_out_proj_vec_n_chunk",
            "PYPTO_STEP3P5_SWA_OUT_PROJ_VEC_N_CHUNK",
        ),
        (
            "swa_out_proj_matmul_tiles_per_task",
            "PYPTO_STEP3P5_SWA_OUT_PROJ_MATMUL_TILES_PER_TASK",
        ),
    ):
        value = int(getattr(args, arg_name))
        if value < 0:
            raise ValueError(f"--{arg_name.replace('_', '-')} must be non-negative")
        if value:
            os.environ[env_name] = str(value)
    if args.full_attn_out_proj_fuse_cast:
        os.environ["PYPTO_STEP3P5_FULL_ATTN_OUT_PROJ_FUSE_CAST"] = "1"
    if args.swa_out_proj_fuse_cast:
        os.environ["PYPTO_STEP3P5_SWA_OUT_PROJ_FUSE_CAST"] = "1"
    os.environ.setdefault("PYPTO_PROG_BUILD_DIR", str(out / "build_output"))

    from pypto.backend import BackendType, set_backend_type
    set_backend_type(BackendType.Ascend910B)

    import models.step3p5.config as cfg
    if cfg.BATCH != STORAGE_BATCH:
        raise ValueError(
            f"harness assumes STORAGE_BATCH={STORAGE_BATCH} to size the block "
            f"table before config import, but config.BATCH is {cfg.BATCH}"
        )
    if cfg.MAX_SEQ_DEFAULT != max_seq:
        raise ValueError(
            f"config.MAX_SEQ_DEFAULT={cfg.MAX_SEQ_DEFAULT} does not match "
            f"--num-blocks capacity {max_seq}"
        )
    task_grains = {
        "qk": getattr(cfg, "FULL_ATTN_QK_BLOCKS_PER_TASK", None),
        "softmax": getattr(cfg, "FULL_ATTN_SOFTMAX_BLOCKS_PER_TASK", None),
        "online_softmax": getattr(
            cfg,
            "FULL_ATTN_ONLINE_SOFTMAX_BLOCKS_PER_TASK",
            None,
        ),
    }
    online_softmax_reduce_fan_in = getattr(
        cfg,
        "FULL_ATTN_ONLINE_SOFTMAX_PARTIALS_PER_REDUCE_TASK",
        None,
    )
    out_proj_grains = {
        "full": {
            "matmul": getattr(
                cfg,
                "FULL_ATTN_OUT_PROJ_MATMUL_N_CHUNK",
                getattr(cfg, "FULL_ATTN_OUT_PROJ_N_CHUNK", 64),
            ),
            "vec": getattr(
                cfg,
                "FULL_ATTN_OUT_PROJ_VEC_N_CHUNK",
                getattr(cfg, "FULL_ATTN_OUT_PROJ_N_CHUNK", 64),
            ),
            "matmul_tiles_per_task": getattr(
                cfg,
                "FULL_ATTN_OUT_PROJ_MATMUL_TILES_PER_TASK",
                1,
            ),
        },
        "swa": {
            "matmul": getattr(
                cfg,
                "SWA_OUT_PROJ_MATMUL_N_CHUNK",
                getattr(cfg, "SWA_OUT_PROJ_N_CHUNK", 64),
            ),
            "vec": getattr(
                cfg,
                "SWA_OUT_PROJ_VEC_N_CHUNK",
                getattr(cfg, "SWA_OUT_PROJ_N_CHUNK", 64),
            ),
            "matmul_tiles_per_task": getattr(
                cfg,
                "SWA_OUT_PROJ_MATMUL_TILES_PER_TASK",
                1,
            ),
        },
    }
    task_limit = getattr(cfg, "PTO2_LOGICAL_BLOCK_LIMIT", 2**15 - 1)
    if any(value is None for value in task_grains.values()):
        requested = (
            args.full_attn_qk_blocks_per_task,
            args.full_attn_softmax_blocks_per_task,
            args.full_attn_online_softmax_blocks_per_task,
            args.full_attn_online_softmax_partials_per_reduce_task,
        )
        if any(requested):
            raise ValueError(
                "this source tree lacks workload-grain full-attention "
                "layout and does not support blocks-per-task overrides"
            )
        raise ValueError(
            "this source tree does not expose workload-grain full-attention "
            "task metadata; refusing to report a legacy static layout"
        )
    else:
        task_layout = "logical_tasks_by_work_grain"
        logical_tasks_at_capacity = {
            stage: args.active_rows
            * ((args.num_blocks + int(grain) - 1) // int(grain))
            for stage, grain in task_grains.items()
        }
        if any(value > task_limit for value in logical_tasks_at_capacity.values()):
            raise ValueError(
                "requested full-attention task layout exceeds the runtime "
                f"logical-block limit {task_limit}: {logical_tasks_at_capacity}",
            )

    from tests.step3p5.harnesses import _two_layer_program as prog_mod

    BATCH, HIDDEN, HEAD_DIM = cfg.BATCH, cfg.HIDDEN, cfg.HEAD_DIM
    LAYER_DYN = prog_mod.LAYER_DYN
    UBD = prog_mod.USER_BATCH_DYN
    BTF = prog_mod.BLOCK_TABLE_FLAT_DYN
    RSD = prog_mod.ROPE_SEQ_DYN
    KVC = prog_mod.KV_CACHE_ROWS_DYN
    HQF, HQS = prog_mod.hidden_q_full, prog_mod.hidden_q_swa
    NHF, NHS = prog_mod.nh_full_pad, prog_mod.nh_swa_pad
    KVH = prog_mod.KV_HIDDEN_LOCAL_R
    ROTF, ROTS = prog_mod.rotary_dim_full, prog_mod.rotary_dim_swa
    INTER = prog_mod.INTER_LOCAL
    NL = prog_mod.N_LAYERS
    tp = 8
    if int(args.num_blocks) * BLOCK_SIZE > RSD:
        raise ValueError(f"ROPE_SEQ_DYN {RSD} < num_blocks*128")

    from pypto import ir
    from pypto.ir.distributed_compiled_program import DistributedConfig

    t_compile = time.time()
    compiled = ir.compile(
        prog_mod.two_layer_attn_perf,
        platform=args.platform,
        distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
        skip_ptoas=False,
        dump_passes=False,
    )
    print(f"[two-layer] compile OK in {time.time() - t_compile:.1f}s "
          f"=> {compiled.output_dir}", flush=True)
    if args.compile_only:
        return 0

    # ---- host tensors: only what mutates per step. Must be share_memory_ and
    # allocated BEFORE prepare() so the forked chip children inherit them.
    def zsh(*shape, dtype=_BF16):
        return torch.zeros(shape, dtype=dtype).share_memory_()

    current_hidden = zsh(tp, BATCH, HIDDEN)
    current_hidden.normal_(0.0, 0.02)
    next_hidden_out = zsh(tp, BATCH, HIDDEN)
    seq_lens_h = torch.ones(tp, UBD, dtype=_I32).share_memory_()
    block_table_h = torch.zeros(tp, BTF, dtype=_I32).share_memory_()
    slot_mapping_h = torch.zeros(tp, UBD, dtype=_I32).share_memory_()
    num_tokens_h = torch.zeros(prog_mod.NUM_TOKENS_RUNTIME, dtype=_I32).share_memory_()
    num_tokens_h[:tp].fill_(args.active_rows)

    # ---- device-upload sources. alloc_tensor(init=...) runs the copy inside the
    # forked chip child, which can only read memory it inherited at fork, so
    # every init buffer must be shared-memory AND allocated before prepare().
    def h_rand(*shape, dtype=_BF16, scale=0.02):
        return ((torch.randn(shape, dtype=torch.float32) * scale)
                .to(dtype).share_memory_())

    def h_ones(*shape, dtype=_F32):
        return torch.ones(shape, dtype=dtype).share_memory_()

    # gate_r is the block-diagonal expand constant R, NOT random:
    # R[h, h*HEAD_DIM + d] = 1 for each real local head. The head-gate expand is
    # a matmul against it, so a zero R would zero the whole attention output.
    def h_gate_r(n_pad: int, hq: int):
        t = torch.zeros(tp, n_pad, hq, dtype=_BF16)
        for h in range(hq // HEAD_DIM):
            t[:, h, h * HEAD_DIM:(h + 1) * HEAD_DIM] = 1.0
        return t.share_memory_()

    init_h = {
        "input_rms": h_ones(tp, LAYER_DYN, HIDDEN),
        "post_rms": h_ones(tp, LAYER_DYN, HIDDEN),
        "q_norm": h_ones(tp, LAYER_DYN, HEAD_DIM),
        "k_norm": h_ones(tp, LAYER_DYN, HEAD_DIM),
        "full_wq": h_rand(tp, HIDDEN, HQF),
        "full_wk": h_rand(tp, HIDDEN, KVH),
        "full_wv": h_rand(tp, HIDDEN, KVH),
        "full_wo": h_rand(tp, HQF, HIDDEN),
        "full_w_g": h_rand(tp, HIDDEN, NHF),
        "full_gate_r": h_gate_r(NHF, HQF),
        "swa_wq": h_rand(tp, HIDDEN, HQS),
        "swa_wk": h_rand(tp, HIDDEN, KVH),
        "swa_wv": h_rand(tp, HIDDEN, KVH),
        "swa_wo": h_rand(tp, HQS, HIDDEN),
        "swa_w_g": h_rand(tp, HIDDEN, NHS),
        "swa_gate_r": h_gate_r(NHS, HQS),
        "dense_w_gate": h_rand(tp, NL, HIDDEN, INTER),
        "dense_w_up": h_rand(tp, NL, HIDDEN, INTER),
        "dense_w_down": h_rand(tp, NL, INTER, HIDDEN),
        # cos=1 / sin=0 is the no-rotation identity: RoPE is elementwise so the
        # table content does not move the timing, but finite values keep the
        # output interpretable enough to notice a kernel that never ran.
        "rope_cos_full": h_ones(tp, RSD, ROTF),
        "rope_sin_full": torch.zeros(tp, RSD, ROTF, dtype=_F32).share_memory_(),
        "rope_cos_swa": h_ones(tp, RSD, ROTS),
        "rope_sin_swa": torch.zeros(tp, RSD, ROTS, dtype=_F32).share_memory_(),
    }

    prepare_cm = compiled.prepare(persistent=True)
    rt = prepare_cm.__enter__()
    owned = []

    def dev(shape, dtype, *, key=None):
        """[tp, *tail] device-resident stacked tensor, one shard per rank.

        ``key`` names an ``init_h`` upload source; omit it to leave the buffer
        uninitialised (used for the KV pool, whose content cannot affect timing
        and which is far too large to stage through host memory).
        """
        from pypto.runtime.device_tensor import StackedDeviceTensor
        src = init_h[key] if key else None
        shards = []
        for r in range(tp):
            init = src[r] if src is not None else None
            shards.append(rt.alloc_tensor(tuple(shape[1:]), dtype, init=init, worker_id=r))
        st = StackedDeviceTensor(shards, tuple(shape), tuple(range(tp)))
        owned.append(st)
        return st

    try:
        input_rms = dev((tp, LAYER_DYN, HIDDEN), _F32, key="input_rms")
        post_rms = dev((tp, LAYER_DYN, HIDDEN), _F32, key="post_rms")
        q_norm = dev((tp, LAYER_DYN, HEAD_DIM), _F32, key="q_norm")
        k_norm = dev((tp, LAYER_DYN, HEAD_DIM), _F32, key="k_norm")
        full_wq = dev((tp, HIDDEN, HQF), _BF16, key="full_wq")
        full_wk = dev((tp, HIDDEN, KVH), _BF16, key="full_wk")
        full_wv = dev((tp, HIDDEN, KVH), _BF16, key="full_wv")
        full_wo = dev((tp, HQF, HIDDEN), _BF16, key="full_wo")
        full_w_g = dev((tp, HIDDEN, NHF), _BF16, key="full_w_g")
        full_gate_r = dev((tp, NHF, HQF), _BF16, key="full_gate_r")
        swa_wq = dev((tp, HIDDEN, HQS), _BF16, key="swa_wq")
        swa_wk = dev((tp, HIDDEN, KVH), _BF16, key="swa_wk")
        swa_wv = dev((tp, HIDDEN, KVH), _BF16, key="swa_wv")
        swa_wo = dev((tp, HQS, HIDDEN), _BF16, key="swa_wo")
        swa_w_g = dev((tp, HIDDEN, NHS), _BF16, key="swa_w_g")
        swa_gate_r = dev((tp, NHS, HQS), _BF16, key="swa_gate_r")
        dense_w_gate = dev((tp, NL, HIDDEN, INTER), _BF16, key="dense_w_gate")
        dense_w_up = dev((tp, NL, HIDDEN, INTER), _BF16, key="dense_w_up")
        dense_w_down = dev((tp, NL, INTER, HIDDEN), _BF16, key="dense_w_down")
        rope_cf = dev((tp, RSD, ROTF), _F32, key="rope_cos_full")
        rope_sf = dev((tp, RSD, ROTF), _F32, key="rope_sin_full")
        rope_cs = dev((tp, RSD, ROTS), _F32, key="rope_cos_swa")
        rope_ss = dev((tp, RSD, ROTS), _F32, key="rope_sin_swa")
        # KV pool: 45 slabs x num_blocks x 128 rows per rank (~755 MB each),
        # left uninitialised on purpose.
        k_cache = dev((tp, KVC, HEAD_DIM), _BF16)
        v_cache = dev((tp, KVC, HEAD_DIM), _BF16)

        # Arg order MUST match TwoLayerAttnPerf.host_orch exactly.
        arglist = [
            current_hidden, input_rms, post_rms, q_norm, k_norm,
            full_wq, full_wk, full_wv, full_wo, full_w_g, full_gate_r,
            swa_wq, swa_wk, swa_wv, swa_wo, swa_w_g, swa_gate_r,
            dense_w_gate, dense_w_up, dense_w_down,
            seq_lens_h, block_table_h, slot_mapping_h,
            rope_cf, rope_sf, rope_cs, rope_ss,
            k_cache, v_cache, next_hidden_out, num_tokens_h,
        ]

        def set_step(context_len: int) -> None:
            row_context_lens = (
                active_context_lens
                if active_context_lens
                else [context_len] * args.active_rows
            )
            seq, _pos, table, slot = _step_metadata(
                context_lens=row_context_lens,
                num_blocks=args.num_blocks,
                batch=BATCH,
                active_rows=args.active_rows,
            )
            seq_lens_h.copy_(seq.unsqueeze(0).expand(tp, -1))
            block_table_h.copy_(table.reshape(1, -1).expand(tp, -1))
            slot_mapping_h.copy_(slot.unsqueeze(0).expand(tp, -1))

        ctx_lens = [int(x) for x in str(args.context_lens).split(",") if x.strip()]
        if active_context_lens and len(ctx_lens) != 1:
            raise ValueError(
                "--active-context-lens requires exactly one benchmark "
                "--context-lens label",
            )
        if not ctx_lens:
            raise ValueError("--context-lens must contain at least one value")
        results = []
        for length in ctx_lens:
            set_step(length)
            for _ in range(max(0, args.warmup)):
                rt.run(compiled, *arglist)
            samples = []
            for _ in range(max(1, args.iters)):
                t = time.time()
                rt.run(compiled, *arglist)
                samples.append((time.time() - t) * 1000.0)
            ms = sorted(samples)
            n = len(ms)
            res = {
                "context_len": length,
                "iters": n,
                "two_layer_ms_min": round(ms[0], 4),
                "two_layer_ms_mean": round(statistics.fmean(ms), 4),
                "two_layer_ms_p50": round(ms[n // 2], 4),
                "two_layer_ms_p99": round(ms[min(n - 1, int(n * 0.99))], 4),
                "two_layer_ms_max": round(ms[-1], 4),
            }
            results.append(res)
            print(json.dumps(res, sort_keys=True), flush=True)

        report = {
            "kind": "two_layer_attn_perf",
            "layers": ["L0_full_dense", "L1_swa_dense"],
            "seed": args.seed,
            "num_blocks": args.num_blocks,
            "block_size": BLOCK_SIZE,
            "max_seq": cfg.MAX_SEQ_DEFAULT,
            "batch_capacity": BATCH,
            "active_rows": args.active_rows,
            "active_context_lens": active_context_lens or None,
            "warmup": args.warmup,
            "full_attn_task_layout": task_layout,
            "full_attn_blocks_per_task": task_grains,
            "full_attn_online_softmax_reduce_fan_in": (
                online_softmax_reduce_fan_in
            ),
            "full_attn_logical_tasks_at_capacity": logical_tasks_at_capacity,
            "decode_out_proj_n_chunks": out_proj_grains,
            "decode_out_proj_fuse_cast": {
                "full": bool(getattr(cfg, "FULL_ATTN_OUT_PROJ_FUSE_CAST", 0)),
                "swa": bool(getattr(cfg, "SWA_OUT_PROJ_FUSE_CAST", 0)),
            },
            "decode_out_proj_logical_tasks_at_capacity": {
                attention_kind: {
                    "matmul": (BATCH // cfg.BATCH_TILE)
                    * (
                        (
                            HIDDEN // int(stage_grains["matmul"])
                            + int(stage_grains["matmul_tiles_per_task"]) - 1
                        )
                        // int(stage_grains["matmul_tiles_per_task"])
                    ),
                    "vec": (BATCH // cfg.BATCH_TILE)
                    * (HIDDEN // int(stage_grains["vec"])),
                }
                for attention_kind, stage_grains in out_proj_grains.items()
            },
            "full_attn_logical_tasks_by_context": {
                str(length): {
                    stage: _logical_task_count(
                        (
                            active_context_lens
                            if active_context_lens
                            else [length] * args.active_rows
                        ),
                        int(grain),
                    )
                    for stage, grain in task_grains.items()
                }
                for length in ctx_lens
            },
            "full_attn_sv_online_logical_tasks_by_context": {
                str(length): _sv_online_logical_task_count(
                    (
                        active_context_lens
                        if active_context_lens
                        else [length] * args.active_rows
                    ),
                    online_blocks_per_task=int(task_grains["online_softmax"]),
                )
                for length in ctx_lens
            },
            "results": results,
        }
        (out / "itl_report.json").write_text(json.dumps(report, indent=2))
        print(f"ITL_REPORT={out / 'itl_report.json'}", flush=True)

        if args.dfx:
            from pypto.runtime.runner import RunConfig
            dfx_ctx = args.dfx_context or max(ctx_lens)
            set_step(dfx_ctx)
            # Both DFX iters run WARM (the timing loop above already ran), and
            # dep_gen / swimlane are kept in separate iters. See module docstring
            # for why co-running them (or profiling a cold iter) corrupts the
            # critical path with a multi-ms cold barrier.
            for _ in range(2):
                rt.run(compiled, *arglist)
            print(f"[two-layer] DFX dep_gen iter (ctx={dfx_ctx})", flush=True)
            rt.run(compiled, *arglist,
                   config=RunConfig(platform=args.platform, enable_dep_gen=True))
            rt.run(compiled, *arglist)
            print(f"[two-layer] DFX l2_swimlane iter (ctx={dfx_ctx})", flush=True)
            rt.run(compiled, *arglist,
                   config=RunConfig(platform=args.platform, enable_l2_swimlane=True))
    finally:
        for st in owned:
            try:
                rt.free_stacked_tensor(st)
            except Exception as exc:  # noqa: BLE001
                print(f"[two-layer] warn: free failed: {exc}", file=sys.stderr)
        prepare_cm.__exit__(None, None, None)

    if args.dfx:
        dfx_ctx = args.dfx_context or max(ctx_lens)
        dfx_context_lens = (
            active_context_lens
            if active_context_lens
            else [dfx_ctx] * args.active_rows
        )
        full_sv_online_logical_tasks = _sv_online_logical_task_count(
            dfx_context_lens,
            online_blocks_per_task=int(task_grains["online_softmax"]),
        )
        expected_logical_blocks = {
            "full_qk_matmul": _logical_task_count(
                dfx_context_lens,
                int(task_grains["qk"]),
            ),
            "full_softmax": _logical_task_count(
                dfx_context_lens,
                int(task_grains["softmax"]),
            ),
            "full_sv_matmul_aic": full_sv_online_logical_tasks,
            "full_sv_matmul_aiv": full_sv_online_logical_tasks,
            "full_online_softmax_reduce": sum(
                (
                    (
                        (int(context_len) + BLOCK_SIZE - 1) // BLOCK_SIZE
                        + int(task_grains["online_softmax"])
                        - 1
                    )
                    // int(task_grains["online_softmax"])
                    + int(online_softmax_reduce_fan_in)
                    - 1
                )
                // int(online_softmax_reduce_fan_in)
                for context_len in dfx_context_lens
            ),
            "full_online_softmax_finalize": args.active_rows,
            "full_out_proj_matmul_aic": (
                HIDDEN // int(out_proj_grains["full"]["matmul"])
                + int(out_proj_grains["full"]["matmul_tiles_per_task"]) - 1
            ) // int(out_proj_grains["full"]["matmul_tiles_per_task"]),
            "swa_out_proj_matmul_aic": (
                HIDDEN // int(out_proj_grains["swa"]["matmul"])
                + int(out_proj_grains["swa"]["matmul_tiles_per_task"]) - 1
            ) // int(out_proj_grains["swa"]["matmul_tiles_per_task"]),
            "full_out_resid_add": HIDDEN // int(out_proj_grains["full"]["vec"]),
            "swa_out_resid_add": HIDDEN // int(out_proj_grains["swa"]["vec"]),
        }
        if not getattr(cfg, "FULL_ATTN_OUT_PROJ_FUSE_CAST", 0):
            expected_logical_blocks["full_out_proj_cast"] = (
                HIDDEN // int(out_proj_grains["full"]["vec"])
            )
        if not getattr(cfg, "SWA_OUT_PROJ_FUSE_CAST", 0):
            expected_logical_blocks["swa_out_proj_cast"] = (
                HIDDEN // int(out_proj_grains["swa"]["vec"])
            )
        _postprocess_dfx(
            Path(compiled.output_dir),
            out,
            expected_logical_blocks=expected_logical_blocks,
        )
    return 0


def _postprocess_dfx(
    build_dir: Path,
    out: Path,
    *,
    expected_logical_blocks: dict[str, int] | None = None,
) -> None:
    """Run critical_path over the captured ranks and summarize uniformity."""
    dfx_root = build_dir / "dfx_outputs"
    if not dfx_root.exists():
        print(f"[two-layer] no dfx_outputs under {build_dir}", file=sys.stderr)
        return
    print(f"\n[two-layer] critical_path over {dfx_root}", flush=True)
    proc = subprocess.run(
        [sys.executable, "-m", "simpler_setup.tools.critical_path",
         str(dfx_root), "--top", "20", "--stdout"],
        capture_output=True, text=True, check=False,
    )
    (out / "critical_path_stdout.txt").write_text(proc.stdout + proc.stderr)
    print(proc.stdout[-4000:] if proc.stdout else proc.stderr[-2000:], flush=True)

    summary = {}
    for rank_dir in sorted(dfx_root.rglob("l2_swimlane_records.json")):
        d = rank_dir.parent
        u = analyze_uniformity(
            d,
            expected_logical_blocks=expected_logical_blocks,
        )
        if u is None:
            continue
        summary[str(d.relative_to(dfx_root))] = u
    if not summary:
        return
    # Minimum makespan is a useful low-wait heuristic, but this program has
    # multiple collectives and one rank need not be the last arrival at every
    # barrier. Keep the selected rank for convenient A/B reporting without
    # treating it as a uniquely wait-free measurement.
    ref = min(summary, key=lambda k: summary[k]["makespan_us"])
    for tag, u in summary.items():
        print_uniformity(tag, u, reference=(tag == ref))
    summary["_reference_rank"] = ref
    print(f"\n[two-layer] LOW-WAIT REFERENCE rank = {ref} "
          f"(device makespan {summary[ref]['makespan_us'] / 1000:.3f} ms). "
          f"Read {dfx_root / ref / 'critical_path_report.md'} for its chain; "
          f"compare per-collective minima when attributing all-reduce time.")
    (out / "uniformity_report.json").write_text(json.dumps(summary, indent=2))
    print(f"UNIFORMITY_REPORT={out / 'uniformity_report.json'}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
