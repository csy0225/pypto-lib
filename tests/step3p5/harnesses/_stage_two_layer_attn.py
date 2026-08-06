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
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

BLOCK_SIZE = 128
# The storage capacity must be selected before models.step3p5.config is first
# imported because BLOCK_TABLE_FLAT_DYN is computed at module load. Keep the
# environment as the single source of truth so the same harness can compile
# capacity-16 and capacity-32 profiles without editing source.
STORAGE_BATCH = int(os.environ.get("PYPTO_STEP3P5_STORAGE_BATCH_CAPACITY", "16"))
if STORAGE_BATCH <= 0 or STORAGE_BATCH % 16 != 0:
    raise ValueError(
        "PYPTO_STEP3P5_STORAGE_BATCH_CAPACITY must be a positive multiple "
        f"of 16, got {STORAGE_BATCH}",
    )
_BF16 = torch.bfloat16
_F32 = torch.float32
_I32 = torch.int32


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="0,1,2,3,4,5,6,7",
                   help="TP=8 device ids, comma separated")
    p.add_argument("--out", required=True)
    p.add_argument("--num-blocks", type=int, default=512,
                   help="maximum paged-cache blocks per request row")
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
        "--active-row-counts",
        default="",
        help="optional comma-separated runtime batch matrix executed by one "
             "capacity-shaped compiled artifact; for example 1,2,4,7,8,16",
    )
    p.add_argument(
        "--active-context-lens",
        default="",
        help="optional comma-separated context length for each active row; "
             "defaults to the benchmark context for every active row",
    )
    p.add_argument(
        "--block-table-order",
        choices=("linear", "reverse"),
        default="linear",
        help="chronological KV-block to physical-page mapping used by the "
             "fixture; reverse hardens block-table/addressing validation",
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
        "--attn-task-profile",
        default="",
        help="compile-time attention task profile "
        "(portable or a2a3; default: config portable profile)",
    )
    for stage in (
        "qk",
        "softmax",
        "online-softmax",
        "online-softmax-reduce",
    ):
        p.add_argument(
            f"--full-attn-{stage}-uniform-o1",
            type=int,
            choices=(0, 1),
            default=-1,
            help=f"override uniform-row O(1) mapping for {stage}; "
            "-1 keeps the profile default",
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
    p.add_argument(
        "--dfx-active-rows",
        type=int,
        default=0,
        help="runtime batch for DFX capture (default: largest active-row count)",
    )
    p.add_argument("--compile-only", action="store_true",
                   help="compile and exit (card-free preflight)")
    p.add_argument(
        "--reference-output-dir",
        default="",
        help="optional output directory from a baseline run; compare each "
        "captured active output against its matching tensor",
    )
    p.add_argument(
        "--replacement-atol",
        type=float,
        default=0.02,
        help="absolute tolerance for --reference-output-dir comparison",
    )
    p.add_argument(
        "--replacement-rtol",
        type=float,
        default=0.02,
        help="relative tolerance for --reference-output-dir comparison",
    )
    p.add_argument(
        "--replacement-max-bad-ratio",
        type=float,
        default=0.02,
        help="maximum fraction outside replacement atol/rtol",
    )
    p.add_argument(
        "--audit-iteration-outputs",
        action="store_true",
        help="hash the active output after every measured iteration and fail "
        "if one prepared program produces more than one result",
    )
    p.add_argument(
        "--alternating-input-audit-iters",
        type=int,
        default=0,
        help="before benchmarking, run this many zero-warmup A/B alternating-"
        "input iterations for every requested batch/context case; poison the "
        "host output before each run and require bitwise-stable publication",
    )
    p.add_argument(
        "--inactive-row-audit-iters",
        type=int,
        default=0,
        help="before benchmarking, alternate only inactive-row contents and "
        "require the active output to remain bitwise identical",
    )
    p.add_argument(
        "--q-publication-audit-iters",
        type=int,
        default=0,
        help="before benchmarking, alternate only the Full or SWA q_norm row "
        "while keeping hidden/K/V/residual inputs fixed; require stable and "
        "discriminating final publication for every requested workload case",
    )
    p.add_argument(
        "--kv-slot-audit-iters",
        type=int,
        default=0,
        help="before benchmarking, alternate independent inputs and disjoint "
        "reserve-page ranges at ctx=1, poison resident Full/SWA K/V rows, then "
        "compare device readback with independent torch oracles and canaries",
    )
    p.add_argument(
        "--swa-direct-oracle-audit",
        action="store_true",
        help="run a real-device bs1/ctx65535/reverse-table SWA invocation with "
        "Full and both dense residual branches reduced to identity, then compare "
        "every rank/row/64-column tile with an independent torch oracle",
    )
    return p.parse_args()


_ATTENTION_CODEGEN_STAGES = (
    {
        "description": "full Q RoPE",
        "marker": "full_rope_q",
        "bound_prefix": "active_tokens__",
        "bound_is_scalar": True,
        "producer": "full_rope_q_tid",
        "dependency": None,
        "allow_early_resolve": True,
    },
    {
        "description": "full KV RoPE/cache",
        "marker": "full_rope_kv_cache",
        "bound_prefix": "active_tokens__",
        "bound_is_scalar": True,
        "producer": "full_rope_kv_tid",
        "dependency": None,
        "allow_early_resolve": True,
    },
    {
        "description": "full QK",
        "marker": "full_qk_matmul",
        "bound_prefix": "full_qk_active_tasks__rv_",
        "bound_is_scalar": True,
        "producer": "full_qk_tid",
        "dependency": ("full_rope_q_tid", "full_rope_kv_tid"),
    },
    {
        "description": "full softmax",
        "marker": "full_softmax",
        "bound_prefix": "full_softmax_active_tasks__rv_",
        "bound_is_scalar": True,
        "producer": "full_softmax_tid",
        "dependency": "full_qk_tid",
    },
    {
        "description": "full SV",
        "marker": "full_sv_matmul",
        "bound_prefix": "full_online_softmax_active_tasks__rv_",
        "bound_is_scalar": True,
        "producer": "full_sv_online_tid",
        "dependency": "full_softmax_tid",
    },
    {
        "description": "full reduce",
        "marker": "full_online_softmax_reduce",
        "bound_prefix": "full_online_softmax_reduce_tasks__rv_",
        "bound_is_scalar": True,
        "producer": "full_online_softmax_reduce_tid",
        "dependency": "full_sv_online_tid",
    },
    {
        "description": "full finalize",
        "marker": "full_online_softmax_finalize",
        "bound_prefix": "full_online_softmax_active_rows__rv_",
        "bound_is_scalar": False,
        "producer": None,
        "dependency": "full_online_softmax_reduce_tid",
    },
    {
        "description": "SWA Q RoPE",
        "marker": "swa_rope_q",
        "bound_prefix": "active_tokens__",
        "bound_is_scalar": True,
        "producer": "swa_rope_q_tid",
        "dependency": None,
        "allow_early_resolve": True,
    },
    {
        "description": "SWA KV RoPE/cache",
        "marker": "swa_rope_kv_cache",
        "bound_prefix": "active_tokens__",
        "bound_is_scalar": True,
        "producer": "swa_rope_kv_tid",
        "dependency": None,
        "allow_early_resolve": True,
    },
    {
        "description": "SWA QK",
        "marker": "swa_qk_matmul",
        "bound_prefix": "swa_active_tasks__rv_",
        "bound_is_scalar": False,
        "producer": "swa_qk_tid",
        "dependency": ("swa_rope_q_tid", "swa_rope_kv_tid"),
    },
    {
        "description": "SWA softmax",
        "marker": "swa_softmax",
        "bound_prefix": "swa_active_tasks__rv_",
        "bound_is_scalar": False,
        "producer": "swa_softmax_tid",
        "dependency": "swa_qk_tid",
    },
    {
        "description": "SWA SV",
        "marker": "swa_sv_matmul",
        "bound_prefix": "swa_active_tasks__rv_",
        "bound_is_scalar": False,
        "producer": "swa_sv_tid",
        "dependency": "swa_softmax_tid",
    },
    {
        "description": "SWA online softmax",
        "marker": "swa_online_softmax",
        "bound_prefix": "swa_active_tasks__rv_",
        "bound_is_scalar": False,
        "producer": None,
        "dependency": "swa_sv_tid",
    },
)
_CODEGEN_STAGE_COMMENT = re.compile(
    r"^[ \t]*// (?:Spmd|Group) [^\n]+$",
    flags=re.MULTILINE,
)


def _attention_codegen_stage_blocks(source: str) -> tuple[dict[str, str], list[str]]:
    comments = list(_CODEGEN_STAGE_COMMENT.finditer(source))
    blocks = {}
    errors = []
    for stage in _ATTENTION_CODEGEN_STAGES:
        matches = [
            (index, comment)
            for index, comment in enumerate(comments)
            if stage["marker"] in comment.group(0)
        ]
        if len(matches) != 1:
            errors.append(
                f"{stage['description']} stage marker count={len(matches)}",
            )
            continue
        index, comment = matches[0]
        end = (
            comments[index + 1].start()
            if index + 1 < len(comments)
            else len(source)
        )
        blocks[stage["marker"]] = source[comment.start():end]
    return blocks, errors


def _attention_codegen_contract_errors(source: str) -> list[str]:
    blocks, errors = _attention_codegen_stage_blocks(source)
    for stage in _ATTENTION_CODEGEN_STAGES:
        block = blocks.get(stage["marker"])
        if block is None:
            continue
        description = stage["description"]
        params_matches = re.findall(
            r"\bL0TaskArgs\s+(params_t\d+)\s*;",
            block,
        )
        if len(params_matches) != 1:
            errors.append(
                f"{description} task-argument declaration count="
                f"{len(params_matches)}",
            )
            continue
        params = params_matches[0]
        launch = re.search(
            rf"\b{re.escape(params)}\.launch_spec\.set_block_num\(\s*"
            rf"({re.escape(stage['bound_prefix'])}\w*)\s*\);",
            block,
        )
        if launch is None:
            errors.append(f"{description} dynamic launch")
            bound = None
        else:
            bound = launch.group(1)
        if stage["bound_is_scalar"] and (
            bound is None
            or re.search(
                rf"\b{re.escape(params)}\.add_scalar\(\s*"
                rf"{re.escape(bound)}\s*\);",
                block,
            )
            is None
        ):
            errors.append(f"{description} launch/scalar SSA agreement")
        if stage.get("allow_early_resolve") and re.search(
            rf"\b{re.escape(params)}\.set_allow_early_resolve\(\s*true\s*\);",
            block,
        ) is None:
            errors.append(f"{description} early-resolve hint")

        dependency = stage["dependency"]
        dependencies = (
            ()
            if dependency is None
            else dependency
            if isinstance(dependency, tuple)
            else (dependency,)
        )
        if dependencies:
            deps = f"{params}_deps"
            deps_count = f"{params}_deps_count"
            dependency_patterns = [
                rf"\bPTO2TaskId\s+{re.escape(deps)}"
                rf"\[{len(dependencies)}\]\s*;",
                rf"\buint32_t\s+{re.escape(deps_count)}\s*=\s*0\s*;",
                rf"\b{re.escape(params)}\.set_dependencies\(\s*"
                rf"{re.escape(deps)}\s*,\s*{re.escape(deps_count)}\s*\);",
            ]
            dependency_patterns.extend(
                rf"\b{re.escape(deps)}\[\s*{re.escape(deps_count)}"
                rf"\+\+\s*\]\s*=\s*{re.escape(expected)}\s*;"
                for expected in dependencies
            )
            if any(
                re.search(pattern, block) is None
                for pattern in dependency_patterns
            ):
                errors.append(
                    f"{description} dependency from "
                    f"{', '.join(dependencies)}",
                )

        producer = stage["producer"]
        if producer is not None:
            submit = re.search(
                rf"\bTaskOutputTensors\s+(\w+)\s*=\s*"
                rf"rt_submit_\w+\([^;]*\b{re.escape(params)}\s*\)\s*;",
                block,
            )
            if (
                submit is None
                or re.search(
                    rf"\bPTO2TaskId\s+{re.escape(producer)}\s*=\s*"
                    rf"{re.escape(submit.group(1))}\.task_id\(\)\s*;",
                    block,
                )
                is None
            ):
                errors.append(f"{description} task publication")

    for prefix in ("full", "swa"):
        q_rope_marker = f"{prefix}_rope_q"
        kv_rope_marker = f"{prefix}_rope_kv_cache"
        qk_marker = f"{prefix}_qk_matmul"
        sv_marker = (
            "full_sv_matmul"
            if prefix == "full"
            else "swa_sv_matmul"
        )
        q_rope_block = blocks.get(q_rope_marker)
        kv_rope_block = blocks.get(kv_rope_marker)
        qk_block = blocks.get(qk_marker)
        sv_block = blocks.get(sv_marker)
        if (
            q_rope_block is None
            or kv_rope_block is None
            or qk_block is None
            or sv_block is None
        ):
            continue
        q_rope_writes = re.findall(
            r"\.add_(?:output|inout)\(\s*(\w+)\s*\);",
            q_rope_block,
        )
        kv_rope_writes = re.findall(
            r"\.add_(?:output|inout)\(\s*(\w+)\s*\);",
            kv_rope_block,
        )
        qk_inputs = set(re.findall(
            r"\.add_input\(\s*(\w+)\s*\);",
            qk_block,
        ))
        sv_inputs = set(re.findall(
            r"\.add_input\(\s*(\w+)\s*\);",
            sv_block,
        ))
        q_tensors = [
            tensor
            for tensor in q_rope_writes
            if re.fullmatch(r"all_q_padded\w*", tensor)
        ]
        k_tensors = [
            tensor
            for tensor in kv_rope_writes
            if re.fullmatch(r"(?:ext_)?k_cache\w*", tensor)
        ]
        v_tensors = [
            tensor
            for tensor in kv_rope_writes
            if re.fullmatch(r"(?:ext_)?v_cache\w*", tensor)
        ]
        if (
            len(q_tensors) != 1
            or len(k_tensors) != 1
            or len(v_tensors) != 1
            or any(
                re.fullmatch(r"(?:ext_)?[kv]_cache\w*", tensor)
                for tensor in q_rope_writes
            )
            or any(
                re.fullmatch(r"all_q_padded\w*", tensor)
                for tensor in kv_rope_writes
            )
            or q_tensors[0] not in qk_inputs
            or k_tensors[0] not in qk_inputs
            or v_tensors[0] not in sv_inputs
        ):
            errors.append(f"{prefix} split RoPE tensor lineage")
    return errors


def _verify_attention_codegen_contract(build_dir: Path) -> Path:
    candidates = [
        path
        for path in build_dir.rglob("orchestration/chip_orch.cpp")
        if "full_qk_matmul" in path.read_text(encoding="utf-8")
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "expected exactly one attention chip orchestration source below "
            f"{build_dir}, found {len(candidates)}",
        )
    source_path = candidates[0]
    errors = _attention_codegen_contract_errors(
        source_path.read_text(encoding="utf-8"),
    )
    if errors:
        raise RuntimeError(
            "attention lowering contract failed: " + ", ".join(errors),
        )
    return source_path


def _devices(text: str) -> list[int]:
    ids = [int(x) for x in str(text).split(",") if x.strip()]
    if len(ids) != 8 or len(set(ids)) != 8:
        raise ValueError(f"need 8 distinct devices for TP=8, got {ids}")
    return ids


def _runtime_active_row_counts(
    text: str,
    *,
    default: int,
    capacity: int,
) -> list[int]:
    """Parse a runtime-batch matrix while preserving the requested order."""
    values = [
        int(value)
        for value in str(text).split(",")
        if value.strip()
    ] or [int(default)]
    if len(set(values)) != len(values):
        raise ValueError(f"active row counts must be distinct, got {values}")
    for value in values:
        if not 0 <= value <= int(capacity):
            raise ValueError(
                f"active row count must be in [0,{capacity}], got {value}",
            )
    return values


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
    physical_blocks: int | None = None,
    block_table_order: str = "linear",
):
    """Decode metadata with active rows and compact disjoint physical pages."""
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
    if block_table_order not in ("linear", "reverse"):
        raise ValueError(
            "block_table_order must be 'linear' or 'reverse', got "
            f"{block_table_order!r}",
        )
    next_physical_block = 0
    for row, row_context_len in enumerate(row_context_lens):
        step = row_context_len - 1
        row_blocks = (row_context_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        page_offsets = torch.arange(row_blocks, dtype=_I32)
        if block_table_order == "reverse":
            page_offsets = torch.flip(page_offsets, dims=(0,))
        table[row, :row_blocks] = next_physical_block + page_offsets
        seq[row] = row_context_len
        pos[row] = step
        slot[row] = (
            table[row, step // BLOCK_SIZE] * BLOCK_SIZE
            + step % BLOCK_SIZE
        )
        next_physical_block += row_blocks
    if (
        physical_blocks is not None
        and next_physical_block > int(physical_blocks)
    ):
        raise ValueError(
            f"compact metadata needs {next_physical_block} physical blocks, "
            f"but the KV slab only has {physical_blocks}",
        )
    return seq, pos, table.reshape(-1), slot


def _linear_percentile(values: list[float], q: float) -> float:
    """Linearly interpolated percentile shared by wall and DFX reports."""
    if not values:
        return 0.0
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"percentile q must be in [0,1], got {q}")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash a tensor's dtype, shape and raw CPU bytes deterministically."""
    cpu = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("ascii"))
    digest.update(str(tuple(cpu.shape)).encode("ascii"))
    digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _kv_slot_oracle(
    *,
    hidden: torch.Tensor,
    input_rms_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent host oracle for one current-token K/V cache row."""
    x = hidden.float()
    normed = (
        x
        * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + float(eps))
        * (input_rms_weight.float().unsqueeze(1) + 1.0)
    ).bfloat16()
    k_proj = torch.bmm(normed.float(), wk.float())
    v_proj = torch.bmm(normed.float(), wv.float()).bfloat16()
    k_normed = (
        k_proj
        * torch.rsqrt(
            k_proj.pow(2).mean(dim=-1, keepdim=True) + float(eps),
        )
        * (k_norm_weight.float().unsqueeze(1) + 1.0)
    )
    rotary_dim = rope_cos.shape[-1]
    rotary_half = rotary_dim // 2
    cos = rope_cos.float().unsqueeze(1)
    sin = rope_sin.float().unsqueeze(1)
    k_lo = k_normed[..., :rotary_half]
    k_hi = k_normed[..., rotary_half:rotary_dim]
    k_rotated = torch.cat(
        (
            k_lo * cos[..., :rotary_half]
            - k_hi * sin[..., :rotary_half],
            k_hi * cos[..., rotary_half:]
            + k_lo * sin[..., rotary_half:],
            k_normed[..., rotary_dim:],
        ),
        dim=-1,
    ).bfloat16()
    return k_rotated, v_proj


def _full_kv_slot_oracle(**kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility name retained for downstream harness unit tests."""
    return _kv_slot_oracle(**kwargs)


def _torch_swa_local_partial_oracle(
    *,
    hidden: torch.Tensor,
    input_rms_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
    w_g: torch.Tensor,
    gate_r: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    k_cache_layer: torch.Tensor,
    v_cache_layer: torch.Tensor,
    eps: float,
    block_size: int,
    sliding_window: int,
    out_proj_k_chunk: int,
    out_proj_n_chunk: int,
) -> torch.Tensor:
    """Independent per-rank SWA partial matching production dtype boundaries."""
    batch, hidden_size = hidden.shape
    head_dim = wk.shape[-1]
    hidden_q = wq.shape[-1]
    num_heads = hidden_q // head_dim
    q_per_kv = num_heads
    rotary_dim = rope_cos.shape[-1]
    rotary_half = rotary_dim // 2
    scale = 1.0 / math.sqrt(head_dim)

    x = hidden.float()
    normed = (
        x
        * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + float(eps))
        * (input_rms_weight.float().reshape(1, -1) + 1.0)
    ).bfloat16()
    q_proj = normed.float() @ wq.float()
    k_proj = normed.float() @ wk.float()
    v_proj = (normed.float() @ wv.float()).bfloat16()

    q_heads = q_proj.reshape(batch, num_heads, head_dim)
    q_heads = (
        q_heads
        * torch.rsqrt(
            q_heads.pow(2).mean(dim=-1, keepdim=True) + float(eps),
        )
        * (q_norm_weight.float().reshape(1, 1, -1) + 1.0)
    )
    k_head = k_proj.reshape(batch, 1, head_dim)
    k_head = (
        k_head
        * torch.rsqrt(
            k_head.pow(2).mean(dim=-1, keepdim=True) + float(eps),
        )
        * (k_norm_weight.float().reshape(1, 1, -1) + 1.0)
    )

    gate_score = torch.sigmoid(normed.float() @ w_g.float()).bfloat16()
    gate_exp = (gate_score.float() @ gate_r.float()).bfloat16()
    attn_out = torch.empty(batch, hidden_q, dtype=_BF16)
    if block_table.ndim != 2 or block_table.shape[0] != batch:
        raise ValueError(
            "SWA oracle block_table must have shape [batch, blocks]",
        )
    if rotary_dim != head_dim:
        raise ValueError(
            "SWA oracle requires full-head RoPE "
            f"(rotary_dim={rotary_dim}, head_dim={head_dim})",
        )
    if hidden_q % out_proj_k_chunk or hidden_size % out_proj_n_chunk:
        raise ValueError("SWA oracle out-projection chunks must divide shapes")

    for batch_index in range(batch):
        context_len = int(seq_lens[batch_index].item())
        window_start = max(0, context_len - int(sliding_window))
        first_block = window_start // block_size
        end_block = (context_len + block_size - 1) // block_size
        position = context_len - 1
        cos = rope_cos[position].float()
        sin = rope_sin[position].float()
        q_lo = q_heads[batch_index, :, :rotary_half]
        q_hi = q_heads[
            batch_index,
            :,
            rotary_half:rotary_dim,
        ]
        q_rotated = torch.cat(
            (
                q_lo * cos[:rotary_half]
                - q_hi * sin[:rotary_half],
                q_hi * cos[rotary_half:]
                + q_lo * sin[rotary_half:],
            ),
            dim=-1,
        ).bfloat16()
        k_lo = k_head[batch_index, 0, :rotary_half]
        k_hi = k_head[batch_index, 0, rotary_half:rotary_dim]
        k_current = torch.cat(
            (
                k_lo * cos[:rotary_half]
                - k_hi * sin[:rotary_half],
                k_hi * cos[rotary_half:]
                + k_lo * sin[rotary_half:],
            ),
            dim=-1,
        ).bfloat16()
        current_slot = int(slot_mapping[batch_index].item())
        current_page = current_slot // block_size
        current_offset = current_slot % block_size

        running_o = torch.zeros(q_per_kv, head_dim)
        running_l = torch.zeros(q_per_kv, 1)
        running_m = torch.zeros(q_per_kv, 1)
        for logical_block in range(first_block, end_block):
            if logical_block >= block_table.shape[1]:
                raise ValueError(
                    "SWA oracle logical block exceeds block-table capacity: "
                    f"{logical_block} >= {block_table.shape[1]}",
                )
            physical_page = int(
                block_table[batch_index, logical_block].item(),
            )
            cache_row = physical_page * block_size
            key_block = k_cache_layer[
                cache_row:cache_row + block_size
            ].clone()
            value_block = v_cache_layer[
                cache_row:cache_row + block_size
            ].clone()
            if physical_page == current_page:
                key_block[current_offset].copy_(k_current)
                value_block[current_offset].copy_(v_proj[batch_index])

            block_token0 = logical_block * block_size
            valid_lo = max(window_start, block_token0) - block_token0
            valid_hi = min(
                context_len,
                block_token0 + block_size,
            ) - block_token0
            raw_scores = q_rotated.float() @ key_block.float().T
            scores = raw_scores * scale
            scores[:, :valid_lo] = -1.0e20
            scores[:, valid_hi:] = -1.0e20
            current_m = scores.max(dim=-1, keepdim=True).values
            exp_scores = torch.exp(scores - current_m).bfloat16()
            current_l = exp_scores.float().sum(dim=-1, keepdim=True)
            current_o = exp_scores.float() @ value_block.float()
            if logical_block == first_block:
                running_o = current_o
                running_l = current_l
                running_m = current_m
            else:
                merged_m = torch.maximum(running_m, current_m)
                old_weight = torch.exp(running_m - merged_m)
                new_weight = torch.exp(current_m - merged_m)
                running_l = (
                    old_weight * running_l
                    + new_weight * current_l
                )
                running_o = (
                    old_weight * running_o
                    + new_weight * current_o
                )
                running_m = merged_m
        attn_out[batch_index].copy_(
            (running_o / running_l).reshape(-1).bfloat16(),
        )

    gated = (attn_out.float() * gate_exp.float()).bfloat16()
    partial = torch.empty(batch, hidden_size, dtype=_BF16)
    for n0 in range(0, hidden_size, out_proj_n_chunk):
        acc = (
            gated[:, :out_proj_k_chunk].float()
            @ wo[:out_proj_k_chunk, n0:n0 + out_proj_n_chunk].float()
        )
        for k0 in range(out_proj_k_chunk, hidden_q, out_proj_k_chunk):
            acc = acc + (
                gated[:, k0:k0 + out_proj_k_chunk].float()
                @ wo[
                    k0:k0 + out_proj_k_chunk,
                    n0:n0 + out_proj_n_chunk,
                ].float()
            )
        partial[:, n0:n0 + out_proj_n_chunk].copy_(acc.bfloat16())
    return partial


def _kv_slot_audit_layout(
    *,
    page_base: int,
    active_rows: int,
    layer_cache_rows: int,
) -> list[dict]:
    """Map Full/SWA target rows plus adjacent canaries into audit slots."""
    if page_base <= 0:
        raise ValueError("KV-slot audit page_base must leave a left canary row")
    if not 0 < active_rows <= 7:
        raise ValueError(f"KV-slot audit active_rows must be in [1,7], got {active_rows}")
    slots = []
    for layer, attention_kind in ((0, "full"), (1, "swa")):
        layer_base = layer * int(layer_cache_rows)
        for active_row in range(active_rows):
            target = layer_base + (page_base + active_row) * BLOCK_SIZE
            for role, cache_row in (
                ("target", target),
                ("left_canary", target - 1),
                ("right_canary", target + 1),
            ):
                slots.append({
                    "attention_kind": attention_kind,
                    "layer": layer,
                    "active_row": active_row,
                    "role": role,
                    "cache_row": cache_row,
                })
    return slots


def _tilewise_numerical_report(
    *,
    actual: torch.Tensor,
    expected: torch.Tensor,
    tile_width: int,
    atol: float,
    rtol: float,
    max_bad_ratio: float,
) -> dict:
    """Require every final-dimension tile to pass its own numerical gate."""
    if actual.shape != expected.shape:
        raise ValueError(
            f"tilewise comparison shape mismatch: {actual.shape} vs {expected.shape}",
        )
    if actual.ndim < 1 or actual.shape[-1] % tile_width:
        raise ValueError(
            f"last dimension {actual.shape[-1] if actual.ndim else None} "
            f"must be divisible by tile_width={tile_width}",
        )
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    bad = diff > (float(atol) + float(rtol) * expected_f.abs())
    tile_reports = []
    for tile0 in range(0, actual.shape[-1], tile_width):
        tile_bad = bad[..., tile0:tile0 + tile_width]
        tile_diff = diff[..., tile0:tile0 + tile_width]
        bad_ratio = float(tile_bad.float().mean().item())
        tile_reports.append({
            "column_start": tile0,
            "bad_ratio": bad_ratio,
            "max_abs_diff": float(tile_diff.max().item()),
            "passed": bad_ratio <= max_bad_ratio,
        })
    return {
        "passed": all(tile["passed"] for tile in tile_reports),
        "atol": float(atol),
        "rtol": float(rtol),
        "max_bad_ratio_per_tile": float(max_bad_ratio),
        "max_abs_diff": float(diff.max().item()),
        "tiles": tile_reports,
    }


def _run_alternating_input_audit(
    *,
    run_once,
    current_hidden: torch.Tensor,
    next_hidden_out: torch.Tensor,
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    active_rows: int,
    iterations: int,
    discrimination_tile_width: int = 64,
    expected_output_relation: str = "different",
    require_distinct_active_inputs: bool = True,
    zero_warmup: bool = True,
) -> dict:
    """Detect stale or partially published output with a cold A/B sequence."""
    if iterations < 4 or iterations % 2:
        raise ValueError(
            "alternating-input audit iterations must be an even value >= 4",
        )
    if not 0 < active_rows <= current_hidden.shape[1]:
        raise ValueError(
            "alternating-input audit needs active_rows in "
            f"[1,{current_hidden.shape[1]}], got {active_rows}",
        )
    if (
        input_a.shape != current_hidden.shape
        or input_b.shape != current_hidden.shape
        or next_hidden_out.shape[:2] != current_hidden.shape[:2]
    ):
        raise ValueError("alternating-input audit tensor shapes do not agree")
    if expected_output_relation not in ("different", "same"):
        raise ValueError(
            "expected_output_relation must be 'different' or 'same'",
        )
    active_inputs_equal = torch.equal(
        input_a[:, :active_rows],
        input_b[:, :active_rows],
    )
    if (
        expected_output_relation == "different"
        and require_distinct_active_inputs
        and active_inputs_equal
    ):
        raise ValueError("alternating-input audit A and B prefixes are equal")
    if expected_output_relation == "same" and not active_inputs_equal:
        raise ValueError(
            "inactive-row audit A and B active prefixes must be equal",
        )
    if expected_output_relation == "same" and torch.equal(input_a, input_b):
        raise ValueError("inactive-row audit A and B full inputs are equal")
    if discrimination_tile_width <= 0:
        raise ValueError("discrimination_tile_width must be positive")

    variants = {"A": input_a, "B": input_b}
    references: dict[str, torch.Tensor] = {}
    reference_hashes: dict[str, str] = {}
    iteration_hashes = []
    try:
        for iteration in range(iterations):
            label = "A" if iteration % 2 == 0 else "B"
            current_hidden.copy_(variants[label])
            # This catches a missing host publication directly. A stale
            # device-side publication is caught by the per-variant reference.
            next_hidden_out[:, :active_rows].fill_(float("nan"))
            run_once()
            output = (
                next_hidden_out[:, :active_rows]
                .detach()
                .clone()
                .contiguous()
            )
            if not bool(torch.isfinite(output).all().item()):
                raise RuntimeError(
                    "alternating-input audit found an unwritten/poisoned "
                    f"output at iteration={iteration} variant={label}",
                )
            tp_spread = (
                output.float() - output[0:1].float()
            ).abs().max().item()
            if tp_spread != 0.0:
                raise RuntimeError(
                    "alternating-input audit found TP divergence at "
                    f"iteration={iteration} variant={label}: "
                    f"max_abs_spread={tp_spread}",
                )
            digest = _tensor_sha256(output)
            iteration_hashes.append({
                "iteration": iteration,
                "variant": label,
                "sha256": digest,
            })
            reference = references.get(label)
            if reference is None:
                references[label] = output
                reference_hashes[label] = digest
            elif not torch.equal(output, reference):
                diff = (output.float() - reference.float()).abs()
                raise RuntimeError(
                    "alternating-input audit found stale/intermittent "
                    f"publication at iteration={iteration} variant={label}: "
                    f"expected_sha256={reference_hashes[label]}, "
                    f"actual_sha256={digest}, "
                    f"max_abs_diff={float(diff.max().item())}",
                )
        variants_equal = reference_hashes["A"] == reference_hashes["B"]
        if expected_output_relation == "different" and variants_equal:
            raise RuntimeError(
                "alternating-input audit is non-discriminating: A and B "
                "produced the same output",
            )
        if expected_output_relation == "same" and not variants_equal:
            diff = (
                references["A"].float() - references["B"].float()
            ).abs()
            raise RuntimeError(
                "inactive-row audit found active-output contamination: "
                f"A_sha256={reference_hashes['A']}, "
                f"B_sha256={reference_hashes['B']}, "
                f"max_abs_diff={float(diff.max().item())}",
            )
        nondiscriminating_tiles = []
        if expected_output_relation == "different":
            output_a = references["A"][0]
            output_b = references["B"][0]
            for row in range(active_rows):
                for col0 in range(
                    0,
                    output_a.shape[-1],
                    discrimination_tile_width,
                ):
                    col1 = min(
                        output_a.shape[-1],
                        col0 + discrimination_tile_width,
                    )
                    if torch.equal(
                        output_a[row, col0:col1],
                        output_b[row, col0:col1],
                    ):
                        nondiscriminating_tiles.append({
                            "row": row,
                            "col0": col0,
                            "col1": col1,
                        })
        if (
            expected_output_relation == "different"
            and nondiscriminating_tiles
        ):
            raise RuntimeError(
                "alternating-input audit has non-discriminating row/column "
                f"tiles: {nondiscriminating_tiles}",
            )
        return {
            "iterations": iterations,
            "active_rows": active_rows,
            "zero_warmup": zero_warmup,
            "output_poison": "nan",
            "expected_output_relation": expected_output_relation,
            "active_inputs_equal": active_inputs_equal,
            "require_distinct_active_inputs": require_distinct_active_inputs,
            "reference_sha256": reference_hashes,
            "discrimination_tile_width": discrimination_tile_width,
            "nondiscriminating_tiles": nondiscriminating_tiles,
            "iteration_hashes": iteration_hashes,
            "passed": True,
        }
    finally:
        current_hidden.copy_(input_a)


def _tensor_probe(tensor: torch.Tensor, edge_elements: int = 4096) -> dict:
    """Fingerprint bounded tensor edges without hashing multi-GB fixtures."""
    flat = tensor.detach().reshape(-1)
    edge = min(int(edge_elements), flat.numel())
    if edge == 0:
        sample = flat
    elif flat.numel() <= 2 * edge:
        sample = flat
    else:
        sample = torch.cat((flat[:edge], flat[-edge:]))
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": tensor.numel(),
        "sample_numel": sample.numel(),
        "sample_sha256": _tensor_sha256(sample),
    }


def _context_workload(
    label: int,
    *,
    active_rows: int,
    active_context_lens: list[int],
) -> list[int]:
    """Resolve a benchmark label into the actual per-row context lengths."""
    return (
        list(active_context_lens)
        if active_context_lens
        else [int(label)] * int(active_rows)
    )


def _context_summary(label: int, context_lens: list[int]) -> dict:
    blocks = [
        (int(context_len) + BLOCK_SIZE - 1) // BLOCK_SIZE
        for context_len in context_lens
    ]
    return {
        "context_label": int(label),
        "per_row_context_lens": [int(value) for value in context_lens],
        "total_context_tokens": sum(int(value) for value in context_lens),
        "per_row_context_blocks": blocks,
        "total_context_blocks": sum(blocks),
    }


def _make_kv_fixture(
    *,
    total_rows: int,
    layer_cache_rows: int,
    head_dim: int,
    initialized_layers: int,
    cache_kind: int,
    seed: int,
) -> torch.Tensor:
    """Create a full provenance-safe KV upload with deterministic active slabs."""
    if total_rows <= 0 or layer_cache_rows <= 0 or head_dim <= 0:
        raise ValueError("KV fixture dimensions must be positive")
    if total_rows % layer_cache_rows != 0:
        raise ValueError(
            f"KV fixture rows {total_rows} are not divisible by slab rows "
            f"{layer_cache_rows}",
        )
    layer_count = total_rows // layer_cache_rows
    if not 0 <= initialized_layers <= layer_count:
        raise ValueError(
            f"initialized_layers must be in [0,{layer_count}], got "
            f"{initialized_layers}",
        )

    fixture = torch.empty(total_rows, head_dim, dtype=_BF16).share_memory_()
    fixture.zero_()
    chunk_rows = min(8192, layer_cache_rows)
    col_ids = torch.arange(head_dim, dtype=torch.int64).reshape(1, -1)
    for layer in range(initialized_layers):
        layer_row0 = layer * layer_cache_rows
        for row0 in range(0, layer_cache_rows, chunk_rows):
            rows = min(chunk_rows, layer_cache_rows - row0)
            row_ids = torch.arange(
                row0,
                row0 + rows,
                dtype=torch.int64,
            ).reshape(-1, 1)
            values = (
                row_ids * 131
                + col_ids * 17
                + layer * 43
                + cache_kind * 71
                + seed * 97
            ) % 255
            fixture[
                layer_row0 + row0:layer_row0 + row0 + rows
            ].copy_(((values.to(torch.float32) + 1.0) / 4096.0).to(_BF16))
    return fixture


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
    fam_intervals: dict[
        str,
        list[tuple[str, int, int, int]],
    ] = collections.defaultdict(list)
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
        fam_intervals[fam].append((str(tid), st, en, core))
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
        fam_intervals[fam].append((str(tid), st, en, core))
        fam_tasks[fam].add(str(tid))

    def occupancy(busy: collections.Counter, core_ids: list[int]) -> dict:
        vals = sorted(busy.get(c, 0) / makespan for c in core_ids)
        used = sum(busy.values())
        cores = len(core_ids)
        return {
            "occupancy_min": round(vals[0], 4),
            "occupancy_p50": round(_linear_percentile(vals, 0.50), 4),
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

    def family_stats(fam: str) -> dict:
        intervals = fam_intervals[fam]
        durations_us = [us(en - st) for _tid, st, en, _core in intervals]
        events = []
        for _tid, st, en, _core in intervals:
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
                for _tid, _st, _en, core in intervals
                if 0 <= core < len(core_types)
            }
        )
        available_cores = sum(resource_core_counts[t] for t in resource_types)
        intervals_by_task: dict[
            str,
            list[tuple[int, int, int]],
        ] = collections.defaultdict(list)
        for task_id, st, en, core in intervals:
            intervals_by_task[task_id].append((st, en, core))

        invocation_spans: list[int] = []
        invocation_waves: list[int] = []
        invocation_pack_denominator = 0
        invocation_full_resource_denominator = 0
        resource_slices = 0
        for task_intervals in intervals_by_task.values():
            task_start = min(st for st, _en, _core in task_intervals)
            task_end = max(en for _st, en, _core in task_intervals)
            task_span = task_end - task_start
            task_slices = len(task_intervals)
            invocation_spans.append(task_span)
            resource_slices += task_slices
            if available_cores:
                invocation_waves.append(
                    (task_slices + available_cores - 1) // available_cores,
                )
                invocation_full_resource_denominator += (
                    task_span * available_cores
                )
                invocation_pack_denominator += (
                    task_span * min(available_cores, task_slices)
                )

        # Sum per-invocation spans instead of taking the first start and last end
        # across a family. The latter incorrectly counts gaps between distinct
        # collectives (and can make a short all-reduce family look millisecond
        # long).
        span_ticks = sum(invocation_spans)
        average_concurrency = (
            fam_busy[fam] / span_ticks if span_ticks else 0.0
        )
        invocation_spans_us = [us(value) for value in invocation_spans]
        return {
            "busy_us": round(us(fam_busy[fam]), 1),
            "distinct_cores": len(fam_cores[fam]),
            "resource_types": resource_types,
            "available_cores": available_cores,
            "logical_blocks": logical_blocks,
            "expected_logical_blocks": expected,
            "resource_slices": resource_slices,
            "waves_at_full_resource": sum(invocation_waves),
            "observed_slices": resource_slices,
            "invocation_count": len(invocation_spans),
            "invocation_span_us_p50": round(
                _linear_percentile(invocation_spans_us, 0.50),
                3,
            ),
            "invocation_span_us_p99": round(
                _linear_percentile(invocation_spans_us, 0.99),
                3,
            ),
            "invocation_span_us_max": round(max(invocation_spans_us), 3),
            "slice_duration_us_p50": round(
                _linear_percentile(durations_us, 0.50),
                3,
            ),
            "slice_duration_us_p99": round(
                _linear_percentile(durations_us, 0.99),
                3,
            ),
            "slice_duration_us_max": round(max(durations_us), 3),
            "stage_span_us": round(us(span_ticks), 3),
            "peak_concurrency": peak,
            "average_concurrency": round(average_concurrency, 3),
            "full_resource_utilization": (
                round(
                    fam_busy[fam] / invocation_full_resource_denominator,
                    4,
                )
                if invocation_full_resource_denominator
                else 0.0
            ),
            "packing_efficiency": (
                round(fam_busy[fam] / invocation_pack_denominator, 4)
                if invocation_pack_denominator
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


def print_uniformity(tag: str, u: dict) -> None:
    print(f"\n[uniformity {tag}] makespan={u['makespan_us'] / 1000:.3f}ms "
          f"cores={u['n_cores']}")
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
    print("  | kernel family | busy µs | cores | invokes | slices | span-sum µs | "
          "slice p50/p99/max µs | peak/avg | waves | pack |")
    print("  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for fam, info in list(u["families"].items())[:18]:
        print(
            f"  | {fam} | {info['busy_us']:.1f} | {info['distinct_cores']} | "
            f"{info['invocation_count']} | {info['resource_slices']} | "
            f"{info['stage_span_us']:.1f} | "
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
    active_row_counts = _runtime_active_row_counts(
        args.active_row_counts,
        default=args.active_rows,
        capacity=STORAGE_BATCH,
    )
    matrix_mode = bool(str(args.active_row_counts).strip())
    active_context_lens = [
        int(value)
        for value in str(args.active_context_lens).split(",")
        if value.strip()
    ]
    if len(active_row_counts) != 1 and active_context_lens:
        raise ValueError(
            "--active-context-lens cannot be combined with "
            "--active-row-counts containing multiple batches",
        )
    benchmark_active_rows = active_row_counts[0]
    if benchmark_active_rows == 0 and active_context_lens:
        raise ValueError("--active-context-lens is invalid when --active-rows=0")
    if (
        active_context_lens
        and len(active_context_lens) != benchmark_active_rows
    ):
        raise ValueError(
            "--active-context-lens must contain exactly one value per active "
            f"row ({benchmark_active_rows}), got {active_context_lens}",
        )
    ctx_lens = [
        int(value)
        for value in str(args.context_lens).split(",")
        if value.strip()
    ]
    if active_context_lens and len(ctx_lens) != 1:
        raise ValueError(
            "--active-context-lens requires exactly one benchmark "
            "--context-lens label",
        )
    if not ctx_lens:
        raise ValueError("--context-lens must contain at least one value")
    if args.replacement_atol < 0.0 or args.replacement_rtol < 0.0:
        raise ValueError("replacement atol/rtol must be non-negative")
    if not 0.0 <= args.replacement_max_bad_ratio <= 1.0:
        raise ValueError("--replacement-max-bad-ratio must be in [0,1]")
    for option, iterations in (
        (
            "--alternating-input-audit-iters",
            args.alternating_input_audit_iters,
        ),
        ("--inactive-row-audit-iters", args.inactive_row_audit_iters),
        ("--q-publication-audit-iters", args.q_publication_audit_iters),
        ("--kv-slot-audit-iters", args.kv_slot_audit_iters),
    ):
        if iterations and (iterations < 4 or iterations % 2):
            raise ValueError(f"{option} must be an even value >= 4")
    dfx_active_rows = (
        int(args.dfx_active_rows)
        if args.dfx_active_rows
        else max(active_row_counts)
    )
    if not 0 <= dfx_active_rows <= STORAGE_BATCH:
        raise ValueError(
            f"--dfx-active-rows must be in [0,{STORAGE_BATCH}], "
            f"got {dfx_active_rows}",
        )
    dfx_ctx = int(args.dfx_context or max(ctx_lens))

    benchmark_cases = []
    for active_rows in active_row_counts:
        for label in ctx_lens:
            row_context_lens = _context_workload(
                int(label),
                active_rows=active_rows,
                active_context_lens=active_context_lens,
            )
            case_id = f"bs{active_rows}_ctx{int(label)}"
            summary = _context_summary(int(label), row_context_lens)
            summary["case_id"] = case_id
            summary["active_rows"] = int(active_rows)
            benchmark_cases.append({
                "case_id": case_id,
                "context_label": int(label),
                "active_rows": int(active_rows),
                "context_lens": row_context_lens,
                "summary": summary,
            })

    allocation_cases = list(benchmark_cases)
    if args.dfx and not any(
        case["context_label"] == dfx_ctx
        and case["active_rows"] == dfx_active_rows
        for case in allocation_cases
    ):
        row_context_lens = _context_workload(
            dfx_ctx,
            active_rows=dfx_active_rows,
            active_context_lens=[],
        )
        case_id = f"bs{dfx_active_rows}_ctx{dfx_ctx}"
        summary = _context_summary(dfx_ctx, row_context_lens)
        summary["case_id"] = case_id
        summary["active_rows"] = dfx_active_rows
        allocation_cases.append({
            "case_id": case_id,
            "context_label": dfx_ctx,
            "active_rows": dfx_active_rows,
            "context_lens": row_context_lens,
            "summary": summary,
        })
    workload_summaries = {
        case["case_id"]: case["summary"]
        for case in allocation_cases
    }

    # Config env MUST be set before models.step3p5.config is first imported:
    # KV_CACHE_ROWS_DYN / BLOCK_TABLE_FLAT_DYN / ROPE_SEQ_DYN are module-level.
    # The kernel derives its per-layer KV slab as
    #   k_cache.rows // input_rms.rows  (attention_full.py:220-221)
    # and input_rms keeps canonical LAYER_DYN=45 rows, so the pool must be
    # 45 slabs wide even though only slabs 0 and 1 are touched. Keeping the
    # canonical stride matters for perf: the DDR distance between a layer's KV
    # rows is part of what we are measuring.
    max_seq = int(args.num_blocks) * BLOCK_SIZE
    if args.swa_direct_oracle_audit and max_seq < 65535:
        raise ValueError(
            "--swa-direct-oracle-audit requires --num-blocks capacity "
            "for context_len=65535",
        )
    for summary in workload_summaries.values():
        for row, context_len in enumerate(summary["per_row_context_lens"]):
            if not 0 < context_len <= max_seq:
                raise ValueError(
                    f"context length {context_len} for active row {row} must "
                    f"be in [1,{max_seq}]",
                )
    os.environ["PYPTO_STEP3P5_MAX_SEQ"] = str(max_seq)
    max_workload_blocks = max(
        summary["total_context_blocks"]
        for summary in workload_summaries.values()
    )
    if args.swa_direct_oracle_audit:
        max_workload_blocks = max(
            max_workload_blocks,
            (65535 + BLOCK_SIZE - 1) // BLOCK_SIZE,
        )
    # Pages are compact across active rows, so the KV slab scales with the sum
    # of the per-row context blocks rather than the static row capacity.
    # Keep the canonical capacity-1 reserve for tail/current-token writes.
    physical_blocks = max(1, int(max_workload_blocks)) + STORAGE_BATCH - 1
    os.environ["PYPTO_STEP3P5_KV_CACHE_ROWS"] = str(45 * physical_blocks * BLOCK_SIZE)
    os.environ["PYPTO_STEP3P5_ROPE_SEQ"] = str(max_seq)
    os.environ["PYPTO_STEP3P5_BLOCK_TABLE_FLAT"] = str(
        STORAGE_BATCH * int(args.num_blocks)
    )
    if args.attn_task_profile:
        os.environ["PYPTO_STEP3P5_ATTN_TASK_PROFILE"] = (
            args.attn_task_profile
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
    for arg_name, env_name in (
        (
            "full_attn_qk_uniform_o1",
            "PYPTO_STEP3P5_FULL_ATTN_QK_UNIFORM_O1",
        ),
        (
            "full_attn_softmax_uniform_o1",
            "PYPTO_STEP3P5_FULL_ATTN_SOFTMAX_UNIFORM_O1",
        ),
        (
            "full_attn_online_softmax_uniform_o1",
            "PYPTO_STEP3P5_FULL_ATTN_ONLINE_SOFTMAX_UNIFORM_O1",
        ),
        (
            "full_attn_online_softmax_reduce_uniform_o1",
            "PYPTO_STEP3P5_FULL_ATTN_ONLINE_SOFTMAX_REDUCE_UNIFORM_O1",
        ),
    ):
        value = int(getattr(args, arg_name))
        if value >= 0:
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
            f"harness storage capacity {STORAGE_BATCH} does not match "
            f"config.BATCH {cfg.BATCH}",
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
    uniform_o1_mapping = {
        "qk": int(getattr(cfg, "FULL_ATTN_QK_UNIFORM_O1", 0)),
        "softmax": int(
            getattr(cfg, "FULL_ATTN_SOFTMAX_UNIFORM_O1", 0),
        ),
        "online_softmax": int(
            getattr(cfg, "FULL_ATTN_ONLINE_SOFTMAX_UNIFORM_O1", 0),
        ),
        "online_softmax_reduce": int(
            getattr(
                cfg,
                "FULL_ATTN_ONLINE_SOFTMAX_REDUCE_UNIFORM_O1",
                0,
            ),
        ),
    }
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
            stage: int(cfg.BATCH)
            * ((args.num_blocks + int(grain) - 1) // int(grain))
            for stage, grain in task_grains.items()
        }
        if any(value > task_limit for value in logical_tasks_at_capacity.values()):
            raise ValueError(
                "requested full-attention task layout exceeds the runtime "
                f"logical-block limit {task_limit}: {logical_tasks_at_capacity}",
            )

    from tests.step3p5.harnesses import _two_layer_program as prog_mod
    kv_io_mod = None
    if args.kv_slot_audit_iters or args.swa_direct_oracle_audit:
        from tests.step3p5.harnesses import (
            _kv_slot_io_program as kv_io_mod,
        )

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
    layer_cache_rows = KVC // LAYER_DYN
    expected_layer_cache_rows = physical_blocks * BLOCK_SIZE
    if layer_cache_rows != expected_layer_cache_rows:
        raise RuntimeError(
            f"KV slab rows {layer_cache_rows} != compact layout rows "
            f"{expected_layer_cache_rows}",
        )

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
    kv_io_compiled = None
    if kv_io_mod is not None:
        kv_io_compiled = ir.compile(
            kv_io_mod.kv_slot_io,
            platform=args.platform,
            distributed_config=DistributedConfig(
                device_ids=devices,
                num_sub_workers=0,
            ),
            skip_ptoas=False,
            dump_passes=False,
        )
    print(f"[two-layer] compile OK in {time.time() - t_compile:.1f}s "
          f"=> {compiled.output_dir}", flush=True)
    codegen_contract_path = _verify_attention_codegen_contract(
        Path(compiled.output_dir),
    )
    print(
        "[two-layer] attention codegen contract OK: "
        f"{codegen_contract_path}",
        flush=True,
    )
    if args.compile_only:
        return 0

    # ---- host tensors: only what mutates per step. Must be share_memory_ and
    # allocated BEFORE prepare() so the forked chip children inherit them.
    fixture_generator = torch.Generator(device="cpu")
    fixture_generator.manual_seed(args.seed)

    def zsh(*shape, dtype=_BF16):
        return torch.zeros(shape, dtype=dtype).share_memory_()

    current_hidden = zsh(tp, BATCH, HIDDEN)
    current_hidden[0].normal_(
        0.0,
        0.02,
        generator=fixture_generator,
    )
    for rank in range(1, tp):
        current_hidden[rank].copy_(current_hidden[0])
    next_hidden_out = zsh(tp, BATCH, HIDDEN)
    kv_io_rows_h = None
    kv_io_write_mask_h = None
    kv_io_k_payload_h = None
    kv_io_v_payload_h = None
    kv_io_mode_h = None
    kv_io_slot_count_h = None
    kv_io_k_readback_h = None
    kv_io_v_readback_h = None
    full_wo_zero_h = None
    dense_w_down_zero_h = None
    if kv_io_mod is not None:
        kv_slot_capacity = int(kv_io_mod.KV_SLOT_IO_CAPACITY)
        kv_io_rows_h = zsh(tp, kv_slot_capacity, dtype=_I32)
        kv_io_write_mask_h = zsh(tp, kv_slot_capacity, dtype=_I32)
        kv_io_k_payload_h = zsh(tp, kv_slot_capacity, HEAD_DIM)
        kv_io_v_payload_h = zsh(tp, kv_slot_capacity, HEAD_DIM)
        kv_io_mode_h = zsh(tp, 1, dtype=_I32)
        kv_io_slot_count_h = zsh(tp, 1, dtype=_I32)
        kv_io_k_readback_h = zsh(tp, kv_slot_capacity, HEAD_DIM)
        kv_io_v_readback_h = zsh(tp, kv_slot_capacity, HEAD_DIM)
        full_wo_zero_h = zsh(HQF, HIDDEN)
        dense_w_down_zero_h = zsh(NL, INTER, HIDDEN)
    seq_lens_h = torch.ones(tp, UBD, dtype=_I32).share_memory_()
    block_table_h = torch.zeros(tp, BTF, dtype=_I32).share_memory_()
    slot_mapping_h = torch.zeros(tp, UBD, dtype=_I32).share_memory_()
    num_tokens_h = torch.zeros(prog_mod.NUM_TOKENS_RUNTIME, dtype=_I32).share_memory_()
    num_tokens_h[:tp].fill_(max(active_row_counts))

    # ---- device-upload sources. alloc_tensor(init=...) runs the copy inside the
    # forked chip child, which can only read memory it inherited at fork, so
    # every init buffer must be shared-memory AND allocated before prepare().
    def h_rand(*shape, dtype=_BF16, scale=0.02):
        return (
            (
                torch.randn(
                    shape,
                    dtype=torch.float32,
                    generator=fixture_generator,
                )
                * scale
            )
            .to(dtype)
            .share_memory_()
        )

    def h_ones(*shape, dtype=_F32):
        return torch.ones(shape, dtype=dtype).share_memory_()

    rope_phase = (
        torch.arange(RSD, dtype=_F32).reshape(1, RSD, 1) * 0.001
        + torch.arange(tp, dtype=_F32).reshape(tp, 1, 1) * 0.03125
    )
    rope_cos_base = torch.cos(rope_phase)
    rope_sin_base = torch.sin(rope_phase)

    def h_rope(base: torch.Tensor, width: int) -> torch.Tensor:
        return (
            base.expand(tp, RSD, width)
            .contiguous()
            .share_memory_()
        )

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
        # Vary RoPE by position and rank so a stale position or wrong-rank
        # lineage cannot accidentally pass an output comparison.
        "rope_cos_full": h_rope(rope_cos_base, ROTF),
        "rope_sin_full": h_rope(rope_sin_base, ROTF),
        "rope_cos_swa": h_rope(rope_cos_base, ROTS),
        "rope_sin_swa": h_rope(rope_sin_base, ROTS),
    }
    q_norm_publication_variants = {}
    if args.q_publication_audit_iters or args.swa_direct_oracle_audit:
        for attention_kind, layer_index in (("full", 0), ("swa", 1)):
            variant = init_h["q_norm"].clone()
            # q_norm is zero-centred: gamma=-1 makes the selected layer's Q
            # exactly zero while K/V, residual, gate, and every other layer
            # input remain unchanged.
            variant[:, layer_index].fill_(-1.0)
            q_norm_publication_variants[attention_kind] = (
                variant.share_memory_()
            )
    k_cache_fixture = _make_kv_fixture(
        total_rows=KVC,
        layer_cache_rows=layer_cache_rows,
        head_dim=HEAD_DIM,
        initialized_layers=2,
        cache_kind=0,
        seed=args.seed,
    )
    v_cache_fixture = _make_kv_fixture(
        total_rows=KVC,
        layer_cache_rows=layer_cache_rows,
        head_dim=HEAD_DIM,
        initialized_layers=2,
        cache_kind=1,
        seed=args.seed,
    )
    fixture_probe = {
        "seed": args.seed,
        "current_hidden": _tensor_probe(current_hidden),
        "init_h": {
            key: _tensor_probe(value)
            for key, value in sorted(init_h.items())
        },
        "k_cache": _tensor_probe(k_cache_fixture),
        "v_cache": _tensor_probe(v_cache_fixture),
    }
    (out / "fixture_probe.json").write_text(
        json.dumps(fixture_probe, indent=2),
    )

    extra_compiled = (
        [kv_io_compiled]
        if kv_io_compiled is not None
        else []
    )
    prepare_cm = compiled.prepare(
        extra_compiled=extra_compiled,
        persistent=True,
    )
    rt = prepare_cm.__enter__()
    owned = []

    def dev(shape, dtype, *, key=None, shared_init=None):
        """[tp, *tail] device-resident stacked tensor, one shard per rank.

        ``key`` names a per-rank ``init_h`` source. ``shared_init`` reuses one
        full-shape source for every rank, as required by the large KV fixtures.
        """
        from pypto.runtime.device_tensor import StackedDeviceTensor
        if key is not None and shared_init is not None:
            raise ValueError("dev accepts either key or shared_init, not both")
        src = init_h[key] if key else None
        shards = []
        for r in range(tp):
            init = shared_init if shared_init is not None else (
                src[r] if src is not None else None
            )
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
        # KV pool keeps the canonical 45-slab stride, but pages inside each slab
        # are compact across active rows.
        k_cache = dev(
            (tp, KVC, HEAD_DIM),
            _BF16,
            shared_init=k_cache_fixture,
        )
        v_cache = dev(
            (tp, KVC, HEAD_DIM),
            _BF16,
            shared_init=v_cache_fixture,
        )

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

        def set_step(case: dict) -> None:
            active_rows = int(case["active_rows"])
            row_context_lens = list(case["context_lens"])
            seq, _pos, table, slot = _step_metadata(
                context_lens=row_context_lens,
                num_blocks=args.num_blocks,
                batch=BATCH,
                active_rows=active_rows,
                physical_blocks=physical_blocks,
                block_table_order=args.block_table_order,
            )
            num_tokens_h[:tp].fill_(active_rows)
            seq_lens_h.copy_(seq.unsqueeze(0).expand(tp, -1))
            block_table_h.copy_(table.reshape(1, -1).expand(tp, -1))
            slot_mapping_h.copy_(slot.unsqueeze(0).expand(tp, -1))

        outputs_dir = out / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        reference_root = (
            Path(args.reference_output_dir)
            if args.reference_output_dir
            else None
        )
        if reference_root is not None and (reference_root / "outputs").is_dir():
            reference_root = reference_root / "outputs"
        alternating_input_audit = None
        inactive_row_audit = None
        q_publication_audit = None
        kv_slot_audit = None
        swa_direct_oracle_audit = None
        kv_io_args = None
        if kv_io_compiled is not None:
            assert kv_io_rows_h is not None
            assert kv_io_write_mask_h is not None
            assert kv_io_k_payload_h is not None
            assert kv_io_v_payload_h is not None
            assert kv_io_mode_h is not None
            assert kv_io_slot_count_h is not None
            assert kv_io_k_readback_h is not None
            assert kv_io_v_readback_h is not None
            kv_io_args = [
                k_cache,
                v_cache,
                kv_io_rows_h,
                kv_io_write_mask_h,
                kv_io_k_payload_h,
                kv_io_v_payload_h,
                kv_io_mode_h,
                kv_io_slot_count_h,
                kv_io_k_readback_h,
                kv_io_v_readback_h,
            ]

        q_norm_nbytes = init_h["q_norm"][0].numel() * (
            init_h["q_norm"][0].element_size()
        )

        def install_q_norm(source: torch.Tensor) -> None:
            for rank in range(tp):
                rt.copy_to(
                    q_norm.shards[rank].data_ptr,
                    source[rank].data_ptr(),
                    q_norm_nbytes,
                    worker_id=rank,
                )

        def copy_audit_weights(*, restore: bool) -> None:
            assert full_wo_zero_h is not None
            assert dense_w_down_zero_h is not None
            full_wo_nbytes = full_wo_zero_h.numel() * (
                full_wo_zero_h.element_size()
            )
            dense_down_nbytes = dense_w_down_zero_h.numel() * (
                dense_w_down_zero_h.element_size()
            )
            for rank in range(tp):
                full_src = (
                    init_h["full_wo"][rank]
                    if restore
                    else full_wo_zero_h
                )
                down_src = (
                    init_h["dense_w_down"][rank]
                    if restore
                    else dense_w_down_zero_h
                )
                rt.copy_to(
                    full_wo.shards[rank].data_ptr,
                    full_src.data_ptr(),
                    full_wo_nbytes,
                    worker_id=rank,
                )
                rt.copy_to(
                    dense_w_down.shards[rank].data_ptr,
                    down_src.data_ptr(),
                    dense_down_nbytes,
                    worker_id=rank,
                )

        if args.swa_direct_oracle_audit:
            assert kv_io_compiled is not None
            assert kv_io_args is not None
            assert kv_io_rows_h is not None
            assert kv_io_write_mask_h is not None
            assert kv_io_k_payload_h is not None
            assert kv_io_v_payload_h is not None
            assert kv_io_mode_h is not None
            assert kv_io_slot_count_h is not None
            assert kv_io_k_readback_h is not None
            assert kv_io_v_readback_h is not None
            assert full_wo_zero_h is not None
            assert dense_w_down_zero_h is not None

            direct_context_len = 65535
            direct_seq, _direct_pos, direct_table_flat, direct_slot = (
                _step_metadata(
                    context_lens=[direct_context_len],
                    num_blocks=args.num_blocks,
                    batch=BATCH,
                    active_rows=1,
                    physical_blocks=physical_blocks,
                    block_table_order="reverse",
                )
            )
            direct_table = direct_table_flat.reshape(
                BATCH,
                args.num_blocks,
            )
            direct_input = current_hidden.detach().clone()
            if any(
                not torch.equal(direct_input[rank], direct_input[0])
                for rank in range(1, tp)
            ):
                raise RuntimeError(
                    "SWA direct oracle requires TP-replicated hidden input",
                )

            rank_partials = []
            for rank in range(tp):
                rank_partials.append(
                    _torch_swa_local_partial_oracle(
                        hidden=direct_input[rank, :1],
                        input_rms_weight=init_h["input_rms"][rank, 1],
                        q_norm_weight=q_norm_publication_variants["swa"][
                            rank,
                            1,
                        ],
                        k_norm_weight=init_h["k_norm"][rank, 1],
                        wq=init_h["swa_wq"][rank],
                        wk=init_h["swa_wk"][rank],
                        wv=init_h["swa_wv"][rank],
                        wo=init_h["swa_wo"][rank],
                        w_g=init_h["swa_w_g"][rank],
                        gate_r=init_h["swa_gate_r"][rank],
                        seq_lens=direct_seq[:1],
                        block_table=direct_table[:1],
                        slot_mapping=direct_slot[:1],
                        rope_cos=init_h["rope_cos_swa"][rank],
                        rope_sin=init_h["rope_sin_swa"][rank],
                        k_cache_layer=k_cache_fixture[
                            layer_cache_rows:2 * layer_cache_rows
                        ],
                        v_cache_layer=v_cache_fixture[
                            layer_cache_rows:2 * layer_cache_rows
                        ],
                        eps=cfg.EPS,
                        block_size=BLOCK_SIZE,
                        sliding_window=cfg.SLIDING_WINDOW,
                        out_proj_k_chunk=cfg.OUT_PROJ_K_CHUNK,
                        out_proj_n_chunk=(
                            cfg.SWA_OUT_PROJ_MATMUL_N_CHUNK
                        ),
                    )
                )
            reduced_fp32 = torch.zeros_like(
                rank_partials[0],
                dtype=_F32,
            )
            for partial in rank_partials:
                reduced_fp32.add_(partial.float())
            reduced_bf16 = reduced_fp32.bfloat16()
            expected_output = (
                direct_input[0, :1].float() + reduced_bf16.float()
            ).bfloat16()

            direct_current_row = int(direct_slot[0].item())
            direct_cache_rows = [
                direct_current_row,
                layer_cache_rows + direct_current_row,
            ]

            def configure_direct_kv_io(
                *,
                mode: str,
                k_payload: torch.Tensor | None = None,
                v_payload: torch.Tensor | None = None,
            ) -> None:
                kv_io_rows_h.zero_()
                kv_io_write_mask_h.zero_()
                kv_io_k_payload_h.zero_()
                kv_io_v_payload_h.zero_()
                kv_io_mode_h.fill_(0 if mode == "read" else 1)
                kv_io_slot_count_h.fill_(len(direct_cache_rows))
                for rank in range(tp):
                    for slot_index, cache_row in enumerate(
                        direct_cache_rows
                    ):
                        kv_io_rows_h[rank, slot_index] = cache_row
                        if mode == "read":
                            continue
                        if (
                            mode != "restore"
                            or k_payload is None
                            or v_payload is None
                        ):
                            raise ValueError(
                                "SWA direct-oracle KV I/O accepts only "
                                "read or restore",
                            )
                        kv_io_write_mask_h[rank, slot_index] = 1
                        kv_io_k_payload_h[rank, slot_index].copy_(
                            k_payload[rank, slot_index],
                        )
                        kv_io_v_payload_h[rank, slot_index].copy_(
                            v_payload[rank, slot_index],
                        )

            configure_direct_kv_io(mode="read")
            rt.run(kv_io_compiled, *kv_io_args)
            direct_k_before = kv_io_k_readback_h[
                :, :len(direct_cache_rows)
            ].clone()
            direct_v_before = kv_io_v_readback_h[
                :, :len(direct_cache_rows)
            ].clone()

            num_tokens_h[:tp].fill_(1)
            seq_lens_h.copy_(
                direct_seq.unsqueeze(0).expand(tp, -1),
            )
            block_table_h.copy_(
                direct_table_flat.reshape(1, -1).expand(tp, -1),
            )
            slot_mapping_h.copy_(
                direct_slot.unsqueeze(0).expand(tp, -1),
            )
            actual_output = None
            restore_verified = False
            try:
                copy_audit_weights(restore=False)
                install_q_norm(q_norm_publication_variants["swa"])
                next_hidden_out[:, :1].fill_(float("nan"))
                rt.run(compiled, *arglist)
                actual_output = (
                    next_hidden_out[:, :1]
                    .detach()
                    .clone()
                    .contiguous()
                )
            finally:
                install_q_norm(init_h["q_norm"])
                copy_audit_weights(restore=True)
                configure_direct_kv_io(
                    mode="restore",
                    k_payload=direct_k_before,
                    v_payload=direct_v_before,
                )
                rt.run(kv_io_compiled, *kv_io_args)
                configure_direct_kv_io(mode="read")
                rt.run(kv_io_compiled, *kv_io_args)
                restore_verified = bool(
                    torch.equal(
                        kv_io_k_readback_h[
                            :, :len(direct_cache_rows)
                        ],
                        direct_k_before,
                    )
                    and torch.equal(
                        kv_io_v_readback_h[
                            :, :len(direct_cache_rows)
                        ],
                        direct_v_before,
                    )
                )
                current_hidden.copy_(direct_input)
            if not restore_verified:
                raise RuntimeError(
                    "SWA direct oracle failed to restore current-token "
                    "Full/SWA KV rows",
                )
            assert actual_output is not None
            if not bool(torch.isfinite(actual_output).all().item()):
                raise RuntimeError(
                    "SWA direct oracle found poisoned/non-finite output",
                )
            tp_bitwise_identical = bool(
                torch.equal(
                    actual_output,
                    actual_output[0:1].expand_as(actual_output),
                )
            )
            if not tp_bitwise_identical:
                raise RuntimeError(
                    "SWA direct oracle found TP-divergent final output",
                )
            direct_tensor_path = out / "swa_direct_oracle_tensors.pt"
            torch.save(
                {
                    "input": direct_input[:, :1],
                    "rank_partials": torch.stack(rank_partials),
                    "reduced_bf16": reduced_bf16,
                    "expected_output": expected_output,
                    "actual_output": actual_output,
                },
                direct_tensor_path,
            )
            numerical_reports = []
            for rank in range(tp):
                rank_report = _tilewise_numerical_report(
                    actual=actual_output[rank, 0],
                    expected=expected_output[0],
                    tile_width=64,
                    atol=0.05,
                    rtol=0.05,
                    max_bad_ratio=0.01,
                )
                numerical_reports.append({
                    "rank": rank,
                    **rank_report,
                })
                if not rank_report["passed"]:
                    actual_f = actual_output[rank, 0].float()
                    expected_f = expected_output[0].float()
                    raise RuntimeError(
                        "SWA direct-oracle numerical mismatch at "
                        f"rank={rank}: {rank_report}; "
                        "actual[min,max,mean]="
                        f"{actual_f.min().item(), actual_f.max().item(), actual_f.mean().item()} "
                        "expected[min,max,mean]="
                        f"{expected_f.min().item(), expected_f.max().item(), expected_f.mean().item()} "
                        f"tensors={direct_tensor_path}",
                    )

            window_start = direct_context_len - cfg.SLIDING_WINDOW
            first_block = window_start // BLOCK_SIZE
            end_block = (
                direct_context_len + BLOCK_SIZE - 1
            ) // BLOCK_SIZE
            swa_direct_oracle_audit = {
                "passed": True,
                "active_rows": 1,
                "context_len": direct_context_len,
                "block_table_order": "reverse",
                "window_start": window_start,
                "logical_blocks": list(range(first_block, end_block)),
                "physical_pages": [
                    int(direct_table[0, block].item())
                    for block in range(first_block, end_block)
                ],
                "valid_token_ranges": [
                    [
                        max(0, window_start - block * BLOCK_SIZE),
                        min(
                            BLOCK_SIZE,
                            direct_context_len - block * BLOCK_SIZE,
                        ),
                    ]
                    for block in range(first_block, end_block)
                ],
                "current_slot": direct_current_row,
                "full_and_dense_identity_isolation": {
                    "full_wo": "zero",
                    "dense_w_down_layers": [0, 1],
                    "swa_q_norm_effective_gamma": 0.0,
                },
                "rank_partial_sha256": [
                    _tensor_sha256(partial)
                    for partial in rank_partials
                ],
                "expected_output_sha256": _tensor_sha256(expected_output),
                "actual_output_sha256": _tensor_sha256(actual_output),
                "tensor_path": str(direct_tensor_path),
                "tp_bitwise_identical": tp_bitwise_identical,
                "kv_restore_verified": restore_verified,
                "torch_oracle_atol": 0.05,
                "torch_oracle_rtol": 0.05,
                "max_bad_ratio_per_rank_row_64_columns": 0.01,
                "numerical_reports": numerical_reports,
            }
            (out / "swa_direct_oracle_audit.json").write_text(
                json.dumps(swa_direct_oracle_audit, indent=2),
            )
            print(
                "[two-layer] reverse-table ctx65535 SWA direct oracle PASS",
                flush=True,
            )
        if args.alternating_input_audit_iters:
            audit_cases = [
                case
                for case in benchmark_cases
                if int(case["active_rows"]) > 0
            ]
            if not audit_cases:
                raise ValueError(
                    "alternating-input audit requires a positive runtime batch",
                )
            input_a = current_hidden.detach().clone()
            audit_generator = torch.Generator(device="cpu")
            audit_generator.manual_seed(args.seed + 0xA17E)
            input_b = torch.empty_like(current_hidden)
            input_b[0].normal_(
                0.0,
                0.02,
                generator=audit_generator,
            )
            for rank in range(1, tp):
                input_b[rank].copy_(input_b[0])
            alternating_input_audit = {"passed": True, "cases": []}
            for audit_case in audit_cases:
                set_step(audit_case)
                case_report = _run_alternating_input_audit(
                    run_once=lambda: rt.run(compiled, *arglist),
                    current_hidden=current_hidden,
                    next_hidden_out=next_hidden_out,
                    input_a=input_a,
                    input_b=input_b,
                    active_rows=int(audit_case["active_rows"]),
                    iterations=args.alternating_input_audit_iters,
                )
                case_report["case_id"] = audit_case["case_id"]
                alternating_input_audit["cases"].append(case_report)
                print(
                    "[two-layer] zero-warmup alternating-input audit PASS: "
                    f"case={audit_case['case_id']} "
                    f"iters={args.alternating_input_audit_iters}",
                    flush=True,
                )
            (out / "alternating_input_audit.json").write_text(
                json.dumps(alternating_input_audit, indent=2),
            )
        if args.inactive_row_audit_iters:
            inactive_cases = [
                case
                for case in benchmark_cases
                if 0 < int(case["active_rows"]) < STORAGE_BATCH
            ]
            if not inactive_cases:
                raise ValueError(
                    "inactive-row audit requires a runtime batch below "
                    f"capacity={STORAGE_BATCH}",
                )
            input_a = current_hidden.detach().clone()
            inactive_generator = torch.Generator(device="cpu")
            inactive_generator.manual_seed(args.seed + 0x1AC71E)
            inactive_row_audit = {"passed": True, "cases": []}
            for inactive_case in inactive_cases:
                set_step(inactive_case)
                active_rows = int(inactive_case["active_rows"])
                input_b = input_a.clone()
                input_b[0, active_rows:].normal_(
                    0.0,
                    0.5,
                    generator=inactive_generator,
                )
                for rank in range(1, tp):
                    input_b[rank].copy_(input_b[0])
                case_report = _run_alternating_input_audit(
                    run_once=lambda: rt.run(compiled, *arglist),
                    current_hidden=current_hidden,
                    next_hidden_out=next_hidden_out,
                    input_a=input_a,
                    input_b=input_b,
                    active_rows=active_rows,
                    iterations=args.inactive_row_audit_iters,
                    expected_output_relation="same",
                    zero_warmup=False,
                )
                case_report["case_id"] = inactive_case["case_id"]
                inactive_row_audit["cases"].append(case_report)
                print(
                    "[two-layer] inactive-row isolation audit PASS: "
                    f"case={inactive_case['case_id']} "
                    f"iters={args.inactive_row_audit_iters}",
                    flush=True,
                )
            (out / "inactive_row_audit.json").write_text(
                json.dumps(inactive_row_audit, indent=2),
            )
        if args.q_publication_audit_iters:
            fixed_input = current_hidden.detach().clone()
            q_publication_audit = {"passed": True, "cases": []}
            try:
                for attention_kind in ("full", "swa"):
                    variant_b = q_norm_publication_variants[attention_kind]
                    for audit_case in [
                        case
                        for case in benchmark_cases
                        if int(case["active_rows"]) > 0
                    ]:
                        set_step(audit_case)
                        invocation = [0]

                        def run_q_variant() -> None:
                            label = "A" if invocation[0] % 2 == 0 else "B"
                            install_q_norm(
                                init_h["q_norm"]
                                if label == "A"
                                else variant_b,
                            )
                            invocation[0] += 1
                            rt.run(compiled, *arglist)

                        case_report = _run_alternating_input_audit(
                            run_once=run_q_variant,
                            current_hidden=current_hidden,
                            next_hidden_out=next_hidden_out,
                            input_a=fixed_input,
                            input_b=fixed_input,
                            active_rows=int(audit_case["active_rows"]),
                            iterations=args.q_publication_audit_iters,
                            require_distinct_active_inputs=False,
                        )
                        case_report.update({
                            "case_id": audit_case["case_id"],
                            "attention_kind": attention_kind,
                            "variant": "q_norm_gamma_canonical_vs_minus_one",
                            "fixed_hidden_input": True,
                        })
                        q_publication_audit["cases"].append(case_report)
                        print(
                            "[two-layer] Q publication audit PASS: "
                            f"attention={attention_kind} "
                            f"case={audit_case['case_id']} "
                            f"iters={args.q_publication_audit_iters}",
                            flush=True,
                        )
            finally:
                install_q_norm(init_h["q_norm"])
                current_hidden.copy_(fixed_input)
            (out / "q_publication_audit.json").write_text(
                json.dumps(q_publication_audit, indent=2),
            )
        if args.kv_slot_audit_iters:
            assert kv_io_compiled is not None
            assert kv_io_args is not None
            assert kv_io_rows_h is not None
            assert kv_io_write_mask_h is not None
            assert kv_io_k_payload_h is not None
            assert kv_io_v_payload_h is not None
            assert kv_io_mode_h is not None
            assert kv_io_slot_count_h is not None
            assert kv_io_k_readback_h is not None
            assert kv_io_v_readback_h is not None
            assert full_wo_zero_h is not None
            assert dense_w_down_zero_h is not None
            reserve_blocks = physical_blocks - max_workload_blocks
            kv_audit_rows = min(
                7,
                STORAGE_BATCH - 1,
                reserve_blocks // 2,
            )
            if kv_audit_rows <= 0:
                raise ValueError(
                    "KV-slot audit requires two disjoint reserve-page ranges",
                )
            page_bases = {
                "A": int(max_workload_blocks),
                "B": int(max_workload_blocks + kv_audit_rows),
            }
            input_a = current_hidden.detach().clone()
            kv_generator = torch.Generator(device="cpu")
            kv_generator.manual_seed(args.seed + 0xCACE)
            input_b = torch.empty_like(current_hidden)
            input_b[0].normal_(
                0.0,
                0.02,
                generator=kv_generator,
            )
            for rank in range(1, tp):
                input_b[rank].copy_(input_b[0])
            variants = {"A": input_a, "B": input_b}
            expected = {}
            for label, hidden in variants.items():
                expected[label] = {
                    "full": _kv_slot_oracle(
                        hidden=hidden[:, :kv_audit_rows],
                        input_rms_weight=init_h["input_rms"][:, 0],
                        k_norm_weight=init_h["k_norm"][:, 0],
                        wk=init_h["full_wk"],
                        wv=init_h["full_wv"],
                        rope_cos=init_h["rope_cos_full"][:, 0],
                        rope_sin=init_h["rope_sin_full"][:, 0],
                        eps=cfg.EPS,
                    ),
                    # The audit temporarily zeros Full o_proj and both dense
                    # down projections, so layer 1 receives the original hidden
                    # tensor and its SWA K/V producer has an independent oracle.
                    "swa": _kv_slot_oracle(
                        hidden=hidden[:, :kv_audit_rows],
                        input_rms_weight=init_h["input_rms"][:, 1],
                        k_norm_weight=init_h["k_norm"][:, 1],
                        wk=init_h["swa_wk"],
                        wv=init_h["swa_wv"],
                        rope_cos=init_h["rope_cos_swa"][:, 0],
                        rope_sin=init_h["rope_sin_swa"][:, 0],
                        eps=cfg.EPS,
                    ),
                }

            def set_kv_audit_step(page_base: int) -> None:
                seq = torch.ones(BATCH, dtype=_I32)
                table = torch.zeros(BATCH, args.num_blocks, dtype=_I32)
                slot = torch.zeros(BATCH, dtype=_I32)
                for row in range(kv_audit_rows):
                    physical_page = page_base + row
                    table[row, 0] = physical_page
                    slot[row] = physical_page * BLOCK_SIZE
                num_tokens_h[:tp].fill_(kv_audit_rows)
                seq_lens_h.copy_(seq.unsqueeze(0).expand(tp, -1))
                block_table_h.copy_(
                    table.reshape(1, -1).expand(tp, -1),
                )
                slot_mapping_h.copy_(slot.unsqueeze(0).expand(tp, -1))

            def configure_kv_io(
                *,
                page_base: int,
                mode: str,
            ) -> list[dict]:
                layout = _kv_slot_audit_layout(
                    page_base=page_base,
                    active_rows=kv_audit_rows,
                    layer_cache_rows=layer_cache_rows,
                )
                if len(layout) > kv_io_rows_h.shape[1]:
                    raise RuntimeError(
                        "KV-slot audit layout exceeds the helper-program "
                        f"capacity: {len(layout)} > {kv_io_rows_h.shape[1]}",
                    )
                kv_io_rows_h.zero_()
                kv_io_write_mask_h.zero_()
                kv_io_k_payload_h.zero_()
                kv_io_v_payload_h.zero_()
                kv_io_mode_h.fill_(0 if mode == "read" else 1)
                kv_io_slot_count_h.fill_(len(layout))
                for rank in range(tp):
                    for slot_index, slot in enumerate(layout):
                        cache_row = int(slot["cache_row"])
                        kv_io_rows_h[rank, slot_index] = cache_row
                        if slot["role"] != "target":
                            continue
                        kv_io_write_mask_h[rank, slot_index] = 1
                        if mode == "poison":
                            kv_io_k_payload_h[rank, slot_index].fill_(
                                float("nan"),
                            )
                            kv_io_v_payload_h[rank, slot_index].fill_(
                                float("nan"),
                            )
                        elif mode == "restore":
                            kv_io_k_payload_h[rank, slot_index].copy_(
                                k_cache_fixture[cache_row],
                            )
                            kv_io_v_payload_h[rank, slot_index].copy_(
                                v_cache_fixture[cache_row],
                            )
                        elif mode != "read":
                            raise ValueError(f"unknown KV-slot I/O mode {mode!r}")
                return layout

            def run_kv_io(page_base: int, mode: str) -> list[dict]:
                layout = configure_kv_io(page_base=page_base, mode=mode)
                rt.run(kv_io_compiled, *kv_io_args)
                return layout

            iteration_results = []
            variant_hashes: dict[str, tuple[str, str]] = {}
            try:
                copy_audit_weights(restore=False)
                for iteration in range(args.kv_slot_audit_iters):
                    label = "A" if iteration % 2 == 0 else "B"
                    page_base = page_bases[label]
                    current_hidden.copy_(variants[label])
                    set_kv_audit_step(page_base)
                    layout = run_kv_io(page_base, "read")
                    canary_before = {
                        "k": kv_io_k_readback_h[:, :len(layout)].clone(),
                        "v": kv_io_v_readback_h[:, :len(layout)].clone(),
                    }
                    run_kv_io(page_base, "poison")
                    next_hidden_out[:, :kv_audit_rows].fill_(float("nan"))
                    rt.run(compiled, *arglist)
                    layout = run_kv_io(page_base, "read")
                    actual_by_tensor = {
                        "k": kv_io_k_readback_h[:, :len(layout)].clone(),
                        "v": kv_io_v_readback_h[:, :len(layout)].clone(),
                    }
                    tensor_reports = {
                        attention_kind: {"k": [], "v": []}
                        for attention_kind in ("full", "swa")
                    }
                    target_indices = []
                    for slot_index, slot in enumerate(layout):
                        if slot["role"] != "target":
                            for tensor_name, actual in actual_by_tensor.items():
                                for rank in range(tp):
                                    if not torch.equal(
                                        actual[rank, slot_index],
                                        canary_before[tensor_name][
                                            rank,
                                            slot_index,
                                        ],
                                    ):
                                        raise RuntimeError(
                                            "KV-slot audit adjacent-row canary "
                                            "changed at "
                                            f"iteration={iteration} "
                                            f"variant={label} tensor={tensor_name} "
                                            f"rank={rank} slot={slot}",
                                        )
                            continue
                        target_indices.append(slot_index)
                        attention_kind = str(slot["attention_kind"])
                        active_row = int(slot["active_row"])
                        for tensor_index, tensor_name in enumerate(("k", "v")):
                            actual = actual_by_tensor[tensor_name][
                                :,
                                slot_index,
                            ]
                            wanted = expected[label][attention_kind][
                                tensor_index
                            ][:, active_row]
                            if not bool(torch.isfinite(actual).all().item()):
                                raise RuntimeError(
                                    "KV-slot audit found a poisoned/unwritten "
                                    f"{attention_kind} {tensor_name.upper()} row "
                                    f"at iteration={iteration}",
                                )
                            for rank in range(tp):
                                report = _tilewise_numerical_report(
                                    actual=actual[rank],
                                    expected=wanted[rank],
                                    tile_width=64,
                                    atol=0.05,
                                    rtol=0.05,
                                    max_bad_ratio=0.01,
                                )
                                if not report["passed"]:
                                    raise RuntimeError(
                                        "KV-slot audit numerical mismatch at "
                                        f"iteration={iteration} variant={label} "
                                        f"attention={attention_kind} "
                                        f"tensor={tensor_name} rank={rank} "
                                        f"active_row={active_row}: {report}",
                                    )
                                tensor_reports[attention_kind][
                                    tensor_name
                                ].append({
                                    "rank": rank,
                                    "active_row": active_row,
                                    **report,
                                })
                    target_k = actual_by_tensor["k"][:, target_indices]
                    target_v = actual_by_tensor["v"][:, target_indices]
                    digests = (
                        _tensor_sha256(target_k),
                        _tensor_sha256(target_v),
                    )
                    digest_pair = (digests[0], digests[1])
                    previous = variant_hashes.setdefault(
                        label,
                        digest_pair,
                    )
                    if previous != digest_pair:
                        raise RuntimeError(
                            "KV-slot audit found intermittent cache "
                            f"publication for variant={label}: "
                            f"expected={previous}, actual={digest_pair}",
                        )
                    iteration_results.append({
                        "iteration": iteration,
                        "variant": label,
                        "physical_page_base": page_base,
                        "tensors": tensor_reports,
                    })
            finally:
                current_hidden.copy_(input_a)
                for page_base in page_bases.values():
                    run_kv_io(page_base, "restore")
                copy_audit_weights(restore=True)
            if variant_hashes.get("A") == variant_hashes.get("B"):
                raise RuntimeError(
                    "KV-slot audit variants did not produce distinct K/V rows",
                )
            kv_slot_audit = {
                "passed": True,
                "iterations": args.kv_slot_audit_iters,
                "active_rows": kv_audit_rows,
                "context_len": 1,
                "physical_page_bases": page_bases,
                "device_destination_poison": "nan",
                "device_io": "pypto_kv_slot_io_program",
                "adjacent_row_canaries": ["left", "right"],
                "attention_kinds": ["full", "swa"],
                "torch_oracle_atol": 0.05,
                "torch_oracle_rtol": 0.05,
                "max_bad_ratio_per_rank_row_64_columns": 0.01,
                "variant_hashes": variant_hashes,
                "iteration_results": iteration_results,
            }
            (out / "kv_slot_audit.json").write_text(
                json.dumps(kv_slot_audit, indent=2),
            )
            print(
                "[two-layer] disjoint-page Full/SWA K/V slot audit PASS: "
                f"rows={kv_audit_rows} iters={args.kv_slot_audit_iters}",
                flush=True,
            )
        results = []
        captured_outputs = {}
        for case in benchmark_cases:
            length = int(case["context_label"])
            active_rows = int(case["active_rows"])
            case_id = str(case["case_id"])
            set_step(case)
            for _ in range(max(0, args.warmup)):
                rt.run(compiled, *arglist)
            samples = []
            measured_output_sha256 = []
            measured_output_unique_paths = {}
            for iteration in range(max(1, args.iters)):
                t = time.time()
                rt.run(compiled, *arglist)
                samples.append((time.time() - t) * 1000.0)
                if args.audit_iteration_outputs:
                    iteration_output = (
                        next_hidden_out[:, :active_rows]
                        .detach()
                        .clone()
                        .contiguous()
                    )
                    output_digest = _tensor_sha256(iteration_output)
                    measured_output_sha256.append(output_digest)
                    if output_digest not in measured_output_unique_paths:
                        audit_dir = out / "audit_outputs" / case_id
                        audit_dir.mkdir(parents=True, exist_ok=True)
                        audit_path = (
                            audit_dir
                            / (
                                f"iter_{iteration:04d}_"
                                f"{output_digest[:16]}.pt"
                            )
                        )
                        torch.save(iteration_output, audit_path)
                        measured_output_unique_paths[output_digest] = str(
                            audit_path,
                        )
            ms = sorted(samples)
            n = len(ms)
            if active_rows:
                active_output = next_hidden_out[:, :active_rows].float()
                active_output_finite = bool(
                    torch.isfinite(active_output).all().item(),
                )
                hidden_tp_spread = float(
                    (active_output - active_output[0:1]).abs().max().item(),
                )
            else:
                active_output_finite = True
                hidden_tp_spread = 0.0
            if not active_output_finite:
                raise RuntimeError(
                    f"non-finite active output at {case_id}",
                )
            if hidden_tp_spread != 0.0:
                raise RuntimeError(
                    "TP-replicated output diverged at "
                    f"{case_id}: max_abs_spread={hidden_tp_spread}",
                )
            workload = workload_summaries[case_id]
            output_name = (
                f"bs_{active_rows}_context_{length}.pt"
                if matrix_mode
                else f"context_{length}.pt"
            )
            output_path = outputs_dir / output_name
            captured_output = (
                next_hidden_out[0, :active_rows]
                .detach()
                .clone()
                .contiguous()
            )
            torch.save(captured_output, output_path)
            output_sha256 = _tensor_sha256(captured_output)
            captured_outputs[case_id] = captured_output
            replacement = None
            if reference_root is not None:
                reference_path = reference_root / output_name
                if not reference_path.is_file():
                    raise FileNotFoundError(
                        f"missing replacement reference {reference_path}",
                    )
                reference = torch.load(
                    reference_path,
                    map_location="cpu",
                    weights_only=True,
                )
                if (
                    reference.dtype != captured_output.dtype
                    or tuple(reference.shape) != tuple(captured_output.shape)
                ):
                    raise ValueError(
                        f"replacement reference {reference_path} has "
                        f"shape={tuple(reference.shape)} dtype={reference.dtype}; "
                        f"expected shape={tuple(captured_output.shape)} "
                        f"dtype={captured_output.dtype}",
                    )
                got_f = captured_output.float()
                ref_f = reference.float()
                diff = (got_f - ref_f).abs()
                bad = diff > (
                    args.replacement_atol
                    + args.replacement_rtol * ref_f.abs()
                )
                bad_ratio = float(bad.float().mean().item())
                replacement = {
                    "reference_path": str(reference_path),
                    "reference_sha256": _tensor_sha256(reference),
                    "exact": bool(torch.equal(captured_output, reference)),
                    "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
                    "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
                    "bad_ratio": bad_ratio,
                    "atol": args.replacement_atol,
                    "rtol": args.replacement_rtol,
                    "max_bad_ratio": args.replacement_max_bad_ratio,
                    "passed": bad_ratio <= args.replacement_max_bad_ratio,
                }
            res = {
                **workload,
                "iters": n,
                "two_layer_ms_min": round(ms[0], 4),
                "two_layer_ms_mean": round(statistics.fmean(ms), 4),
                "two_layer_ms_p50": round(statistics.median(ms), 4),
                "two_layer_ms_p99": round(
                    _linear_percentile(ms, 0.99),
                    4,
                ),
                "two_layer_ms_max": round(ms[-1], 4),
                "active_output_finite": active_output_finite,
                "hidden_tp_spread": hidden_tp_spread,
                "active_output_path": str(output_path),
                "active_output_sha256": output_sha256,
                "measured_output_sha256": (
                    measured_output_sha256
                    if args.audit_iteration_outputs
                    else None
                ),
                "measured_output_unique_count": (
                    len(set(measured_output_sha256))
                    if args.audit_iteration_outputs
                    else None
                ),
                "measured_output_unique_paths": (
                    measured_output_unique_paths
                    if args.audit_iteration_outputs
                    else None
                ),
                "replacement_comparison": replacement,
            }
            results.append(res)
            print(json.dumps(res, sort_keys=True), flush=True)
            if (
                args.audit_iteration_outputs
                and len(set(measured_output_sha256)) != 1
            ):
                raise RuntimeError(
                    "prepared-program output changed across measured "
                    f"iterations at {case_id}: "
                    f"hashes={measured_output_sha256}, "
                    f"snapshots={measured_output_unique_paths}",
                )
            if replacement is not None and not replacement["passed"]:
                raise RuntimeError(
                    f"replacement comparison failed at {case_id}: "
                    f"{replacement}",
                )

        batch_prefix_consistency = []
        for length in ctx_lens:
            same_context_cases = [
                case
                for case in benchmark_cases
                if int(case["context_label"]) == int(length)
            ]
            reference_case = max(
                same_context_cases,
                key=lambda case: int(case["active_rows"]),
            )
            reference_output = captured_outputs[reference_case["case_id"]]
            for case in same_context_cases:
                active_rows = int(case["active_rows"])
                output = captured_outputs[case["case_id"]]
                reference_prefix = reference_output[:active_rows]
                diff = (output.float() - reference_prefix.float()).abs()
                exact = bool(torch.equal(output, reference_prefix))
                check = {
                    "case_id": case["case_id"],
                    "reference_case_id": reference_case["case_id"],
                    "exact": exact,
                    "max_abs_diff": (
                        float(diff.max().item())
                        if diff.numel()
                        else 0.0
                    ),
                }
                batch_prefix_consistency.append(check)
                if not exact:
                    raise RuntimeError(
                        "batch-prefix consistency failed: "
                        f"{check}",
                    )

        report = {
            "kind": "two_layer_attn_perf",
            "layers": ["L0_full_dense", "L1_swa_dense"],
            "seed": args.seed,
            "num_blocks": args.num_blocks,
            "block_table_blocks_per_row_capacity": args.num_blocks,
            "block_size": BLOCK_SIZE,
            "max_seq": cfg.MAX_SEQ_DEFAULT,
            "batch_capacity": BATCH,
            "active_rows": (
                active_row_counts[0]
                if len(active_row_counts) == 1
                else None
            ),
            "active_row_counts": active_row_counts,
            "active_context_lens": active_context_lens or None,
            "context_workloads": {
                case_id: summary
                for case_id, summary in workload_summaries.items()
            },
            "kv_layout": {
                "kind": "compact_active_row_pages",
                "block_table_order": args.block_table_order,
                "physical_blocks_per_layer": physical_blocks,
                "workload_blocks_per_layer_max": max_workload_blocks,
                "tail_reserve_blocks": STORAGE_BATCH - 1,
                "layer_cache_rows": layer_cache_rows,
                "initialized_layers": [0, 1],
                "initializer": "full_upload_deterministic_nonzero_mod255",
            },
            "batch_prefix_consistency": batch_prefix_consistency,
            "alternating_input_audit": alternating_input_audit,
            "inactive_row_audit": inactive_row_audit,
            "q_publication_audit": q_publication_audit,
            "kv_slot_audit": kv_slot_audit,
            "swa_direct_oracle_audit": swa_direct_oracle_audit,
            "warmup": args.warmup,
            "full_attn_task_layout": task_layout,
            "attention_task_profile": getattr(
                cfg,
                "ATTN_TASK_PROFILE",
                "legacy",
            ),
            "full_attn_blocks_per_task": task_grains,
            "full_attn_uniform_o1_mapping": uniform_o1_mapping,
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
            "full_attn_logical_tasks_by_case": {
                case["case_id"]: {
                    stage: _logical_task_count(
                        case["context_lens"],
                        int(grain),
                    )
                    for stage, grain in task_grains.items()
                }
                for case in benchmark_cases
            },
            "full_attn_logical_tasks_by_context": (
                {
                    str(case["context_label"]): {
                        stage: _logical_task_count(
                            case["context_lens"],
                            int(grain),
                        )
                        for stage, grain in task_grains.items()
                    }
                    for case in benchmark_cases
                }
                if len(active_row_counts) == 1
                else None
            ),
            "full_attn_sv_online_logical_tasks_by_case": {
                case["case_id"]: _sv_online_logical_task_count(
                    case["context_lens"],
                    online_blocks_per_task=int(task_grains["online_softmax"]),
                )
                for case in benchmark_cases
            },
            "full_attn_sv_online_logical_tasks_by_context": (
                {
                    str(case["context_label"]): _sv_online_logical_task_count(
                        case["context_lens"],
                        online_blocks_per_task=int(
                            task_grains["online_softmax"]
                        ),
                    )
                    for case in benchmark_cases
                }
                if len(active_row_counts) == 1
                else None
            ),
            "results": results,
        }
        (out / "itl_report.json").write_text(json.dumps(report, indent=2))
        print(f"ITL_REPORT={out / 'itl_report.json'}", flush=True)

        if args.dfx:
            from pypto.runtime.runner import RunConfig
            dfx_case = next(
                case
                for case in allocation_cases
                if case["context_label"] == dfx_ctx
                and case["active_rows"] == dfx_active_rows
            )
            set_step(dfx_case)
            # Both DFX iters run WARM (the timing loop above already ran), and
            # dep_gen / swimlane are kept in separate iters. See module docstring
            # for why co-running them (or profiling a cold iter) corrupts the
            # critical path with a multi-ms cold barrier.
            for _ in range(2):
                rt.run(compiled, *arglist)
            print(
                f"[two-layer] DFX dep_gen iter "
                f"(bs={dfx_active_rows}, ctx={dfx_ctx})",
                flush=True,
            )
            rt.run(compiled, *arglist,
                   config=RunConfig(platform=args.platform, enable_dep_gen=True))
            rt.run(compiled, *arglist)
            print(
                f"[two-layer] DFX l2_swimlane iter "
                f"(bs={dfx_active_rows}, ctx={dfx_ctx})",
                flush=True,
            )
            rt.run(
                compiled,
                *arglist,
                config=RunConfig(
                    platform=args.platform,
                    enable_l2_swimlane=True,
                    l2_swimlane_reuse_dep_gen=(
                        not args.platform.endswith("sim")
                    ),
                ),
            )
    finally:
        for st in owned:
            try:
                rt.free_stacked_tensor(st)
            except Exception as exc:  # noqa: BLE001
                print(f"[two-layer] warn: free failed: {exc}", file=sys.stderr)
        prepare_cm.__exit__(None, None, None)

    if args.dfx:
        dfx_context_lens = next(
            case["context_lens"]
            for case in allocation_cases
            if case["context_label"] == dfx_ctx
            and case["active_rows"] == dfx_active_rows
        )
        full_sv_online_logical_tasks = _sv_online_logical_task_count(
            dfx_context_lens,
            online_blocks_per_task=int(task_grains["online_softmax"]),
        )
        expected_logical_blocks = {
            "full_rope_q": dfx_active_rows,
            "full_rope_kv_cache": dfx_active_rows,
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
            "full_online_softmax_finalize": dfx_active_rows,
            "swa_rope_q": dfx_active_rows,
            "swa_rope_kv_cache": dfx_active_rows,
            "swa_qk_matmul": dfx_active_rows,
            "swa_softmax": dfx_active_rows,
            "swa_sv_matmul": dfx_active_rows,
            "swa_online_softmax": dfx_active_rows,
            "full_out_proj_matmul_aic": (BATCH // cfg.BATCH_TILE)
            * (
                (
                    HIDDEN // int(out_proj_grains["full"]["matmul"])
                    + int(out_proj_grains["full"]["matmul_tiles_per_task"])
                    - 1
                )
                // int(out_proj_grains["full"]["matmul_tiles_per_task"])
            ),
            "swa_out_proj_matmul_aic": (BATCH // cfg.BATCH_TILE)
            * (
                (
                    HIDDEN // int(out_proj_grains["swa"]["matmul"])
                    + int(out_proj_grains["swa"]["matmul_tiles_per_task"])
                    - 1
                )
                // int(out_proj_grains["swa"]["matmul_tiles_per_task"])
            ),
            "full_out_resid_add": (BATCH // cfg.BATCH_TILE)
            * (HIDDEN // int(out_proj_grains["full"]["vec"])),
            "swa_out_resid_add": (BATCH // cfg.BATCH_TILE)
            * (HIDDEN // int(out_proj_grains["swa"]["vec"])),
        }
        if not getattr(cfg, "FULL_ATTN_OUT_PROJ_FUSE_CAST", 0):
            expected_logical_blocks["full_out_proj_cast"] = (
                (BATCH // cfg.BATCH_TILE)
                * (HIDDEN // int(out_proj_grains["full"]["vec"]))
            )
        if not getattr(cfg, "SWA_OUT_PROJ_FUSE_CAST", 0):
            expected_logical_blocks["swa_out_proj_cast"] = (
                (BATCH // cfg.BATCH_TILE)
                * (HIDDEN // int(out_proj_grains["swa"]["vec"]))
            )
        _postprocess_dfx(
            Path(compiled.output_dir),
            out,
            expected_logical_blocks=expected_logical_blocks,
        )
    return 0


def _numeric_min_median_max(values: list[float]) -> dict[str, float]:
    return {
        "min": round(min(values), 4),
        "median": round(statistics.median(values), 4),
        "max": round(max(values), 4),
    }


def _aggregate_rank_uniformity(rank_reports: dict[str, dict]) -> dict:
    """Aggregate every rank instead of selecting a minimum-makespan rank."""
    reports = list(rank_reports.values())
    family_names = sorted(
        {
            family
            for report in reports
            for family in report["families"]
        },
    )
    family_metrics = (
        "busy_us",
        "logical_blocks",
        "resource_slices",
        "waves_at_full_resource",
        "invocation_count",
        "invocation_span_us_p50",
        "invocation_span_us_p99",
        "invocation_span_us_max",
        "slice_duration_us_p50",
        "slice_duration_us_p99",
        "slice_duration_us_max",
        "stage_span_us",
        "peak_concurrency",
        "average_concurrency",
        "full_resource_utilization",
        "packing_efficiency",
    )
    families = {}
    for family in family_names:
        present = [
            report["families"][family]
            for report in reports
            if family in report["families"]
        ]
        families[family] = {
            "present_rank_count": len(present),
            **{
                metric: _numeric_min_median_max(
                    [float(info[metric]) for info in present],
                )
                for metric in family_metrics
            },
        }
    return {
        "rank_count": len(rank_reports),
        "rank_tags": sorted(rank_reports),
        "makespan_us": _numeric_min_median_max(
            [float(report["makespan_us"]) for report in reports],
        ),
        "families": families,
    }


def _collective_low_wait_reference(
    rank_reports: dict[str, dict],
) -> dict | None:
    """Select a diagnostic rank with the least in-kernel collective wait.

    ``tp_all_reduce`` includes peer-arrival spin time, so the all-rank
    makespan median can be orders of magnitude larger than device compute
    during DFX capture.  This reference is only a low-wait heuristic; retain
    the all-rank aggregate alongside it.
    """
    candidates = []
    for tag, report in rank_reports.items():
        collective = report["families"].get("tp_all_reduce")
        if collective is None:
            continue
        candidates.append(
            (
                float(collective["stage_span_us"]),
                float(report["makespan_us"]),
                tag,
                report,
            ),
        )
    if not candidates:
        return None
    collective_us, makespan_us, tag, report = min(candidates)
    return {
        "rank_tag": tag,
        "tp_all_reduce_stage_span_us": round(collective_us, 4),
        "makespan_us": round(makespan_us, 4),
        "families": report["families"],
        "interpretation": (
            "diagnostic low-wait heuristic; not a replacement for all-rank "
            "correctness or wall-clock timing"
        ),
    }


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

    rank_summary = {}
    for rank_dir in sorted(dfx_root.rglob("l2_swimlane_records.json")):
        d = rank_dir.parent
        u = analyze_uniformity(
            d,
            expected_logical_blocks=expected_logical_blocks,
        )
        if u is None:
            continue
        rank_summary[str(d.relative_to(dfx_root))] = u
    if not rank_summary:
        return
    for tag, u in rank_summary.items():
        print_uniformity(tag, u)
    aggregate = _aggregate_rank_uniformity(rank_summary)
    low_wait = _collective_low_wait_reference(rank_summary)
    makespan = aggregate["makespan_us"]
    print(
        "\n[two-layer] ALL-RANK makespan "
        f"min/median/max={makespan['min'] / 1000:.3f}/"
        f"{makespan['median'] / 1000:.3f}/"
        f"{makespan['max'] / 1000:.3f} ms",
    )
    if low_wait is not None:
        print(
            "[two-layer] COLLECTIVE LOW-WAIT REFERENCE "
            f"{low_wait['rank_tag']}: makespan="
            f"{low_wait['makespan_us'] / 1000:.3f} ms, "
            "tp_all_reduce span-sum="
            f"{low_wait['tp_all_reduce_stage_span_us']:.3f} us "
            "(diagnostic heuristic)",
        )
    report = {
        "ranks": rank_summary,
        "all_rank_aggregate": aggregate,
        "collective_low_wait_reference": low_wait,
    }
    (out / "uniformity_report.json").write_text(json.dumps(report, indent=2))
    print(f"UNIFORMITY_REPORT={out / 'uniformity_report.json'}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
