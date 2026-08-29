# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Analyze L0-L4 Step3p5 MoE dependency and swimlane traces.

The focused graph has two MoE layers:

* L3: sliding-window attention + MoE
* L4: full attention + MoE

This analyzer keeps those layers separate, keeps AIC and AIV accounting
separate for mixed kernels, and compares profile-specific collective arrival
times across all ranks.  It intentionally treats long ``tp_all_reduce`` spans
as possible in-kernel peer wait rather than arithmetic time.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypto.runtime.runner import _CHIP_SWIMLANE_RECORDS_NAME
from tools.step3p5.five_layer_moe_golden_contract import (
    GOLDEN_SCHEMA as _GOLDEN_SCHEMA,
    LOCAL_OWNER_PROTOCOL_PROFILE,
    normalize_golden_protocol,
)


_LAYER_PREFIX = {
    "L3": "swa_moe_chip_orch_",
    "L4": "",
}
_STAGE_SUFFIXES = {
    "gate_init": ("gate_init",),
    "gate_fanout": ("gate_expert_fanout",),
    "gate_topk": ("gate_topk",),
    "shared_mlp": ("sh_mlp",),
    "shared_gate_up": ("sh_gate_up_mm",),
    "shared_gate_up_act": ("sh_gate_up_act",),
    "shared_down": ("sh_down",),
    "dispatch_meta": ("dispatch_meta",),
    "dispatch_push": ("dispatch_push",),
    "dispatch_wait": ("dispatch_wait",),
    "dispatch_gather": ("dispatch_gather",),
    "local_route_map_init": ("local_route_map_init",),
    "local_route_pack": ("local_route_pack",),
    "local_route_plan": ("local_route_plan",),
    "expert_gate_up": ("expert_gate_up",),
    "expert_gate": ("expert_gate_mm",),
    "expert_up": ("expert_up_mm",),
    "expert_gate_up_act": ("expert_gate_up_act",),
    "routed_h_quant": ("routed_h_quant",),
    "expert_down": ("expert_down",),
    "combine_scatter": ("combine_scatter",),
    "combine_wait": ("combine_wait",),
    "combine_reduce": ("combine_reduce",),
    "local_combine_reduce": ("local_combine_reduce",),
    "moe_residual_add": ("moe_residual_add",),
}
_PACKED_NZ_EXTERNAL_STAGES = {
    "expert_gate_up": "routed_nz_gmm1_swiglu_quant",
    "expert_down": "routed_nz_down",
}
_LEGACY_ARRIVAL_PAIRS = (
    ("dispatch", "dispatch_push", "dispatch_wait"),
    ("combine", "combine_scatter", "combine_wait"),
)
_LOCAL_EP_ARRIVAL_PAIRS = (
    ("moe_collective", "local_combine_reduce", "moe_all_reduce"),
)
_EXPECTED_RANKS = 8
_EXPECTED_RANK_TAGS = {f"rank{rank}/d0" for rank in range(_EXPECTED_RANKS)}
_EXPECTED_CORE_TYPES = ["aic"] * 24 + ["aiv"] * 48
_RECV_META_SIDECAR_SCHEMA = "step3p5.five-layer-moe-local-routes.v2"
_RECV_META_LAYERS = ("L3", "L4")
_RECV_META_AXES = (
    "layer",
    "owner_rank",
    "route_owner_rank",
    "local_expert_pad",
)
_RECV_META_SHAPE = (2, 8, 8, 40)
_LOCAL_EXPERT_COUNT_SHAPE = (2, 8, 36)
_RECV_META_WINDOW_SHAPE = (8, 40)
_RECV_META_WINDOW_BYTES = 8 * 40 * 4
# Canonical Step3p5 MoE dispatch emits eight routes per active token.
_MOE_TOPK = 8
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_IMAGE_DIGEST_PATTERN = re.compile(r".+@sha256:[0-9a-f]{64}")
_CHECKPOINT_SCHEMA = "step3p5.checkpoint-identity.v1"
_BASELINE_DECODE_SHA256 = (
    "3553664cbe5bba2453b17b992c9c8a5489deb0df8f88b98d4a93a1aa45544ff0"
)
# The route-sidecar release adds the explicit L3/L4 route-count outputs used by
# the formal route gate.  Its decode source is intentionally frozen separately
# from the earlier a17 candidate so a DFX capture cannot silently mix source
# generations.
_R6_ROUTE_DECODE_SHA256 = (
    "671a5df8a07e09303c398871fd1772f306b2998ea3e8168048588de6cc3fa323"
)
_PACKED_NZ_DECODE_SHA256 = (
    "da36c09dc275838ee364f76342d74717338ef313d912ba2b372808530489dd14"
)
_LOCAL_EP_DECODE_SHA256 = (
    "cdb2bb26ddc0ca773bcddd0629bfc7bdfa5c426a334e26dde4364aacd867f348"
)
# These are the only upper bounds carried from the release-qualified R5
# packed-fused analyzer.  R5 had a single mixed fused stage; the route-sidecar
# candidate splits the
# same work into AIC gate/up, AIV act/quant, and AIC down.  We therefore keep
# the proven upper scheduling bounds but do not invent a lower bound for the
# separately named gate/up stage.
_STAGED_FUSED_DURATION_LIMITS_US = {
    "p50_max": 200.0,
    "p90_max": 220.0,
    "p99_max": 320.0,
    "max": 500.0,
}
_FROZEN_SOURCE_POLICIES = {
    "baseline": {
        "policy_id": "campaign-baseline-56b3d477-row32-fused-v1",
        "frozen_ref": "stepfun/develop@56b3d477",
        "decode_sha256_prefix": "3553664c",
        "decode_sha256": _BASELINE_DECODE_SHA256,
        "source_role": "baseline",
        "storage_family": "row32_no_graph_wide_gate_up_scratch",
        "schedule_family": "fused_expert_gate_up",
        "task_partition": "tile_local_activation_quant_down",
        "experimental": False,
        "enforce_candidate_release_gate": False,
    },
    "candidate": {
        "policy_id": "campaign-candidate-671a5df8-route-sidecar-staged-fused-v1",
        "frozen_ref": "stepfun/develop@22492c2",
        "decode_sha256_prefix": "671a5df8",
        "decode_sha256": _R6_ROUTE_DECODE_SHA256,
        "source_role": "candidate",
        "storage_family": "row16_staged_fused_gate_up_local_tiles",
        "schedule_family": "staged_fused_gate_up_then_aiv_act_quant_down",
        "task_partition": "aic_gate_up_aiv_activation_quant_aic_down",
        "expert_release_family": "staged_fused_gate_up",
        "duration_limit_source": (
            "R5 packed-fused release-qualified upper bounds: "
            "p50<=200us,p90<=220us,p99<=320us,max<=500us; "
            "no lower bound is inferred for separately named gate_up."
        ),
        "experimental": True,
        "enforce_candidate_release_gate": True,
    },
    "packed-nz": {
        "policy_id": "release-packed-nz-da36c09d-mixed-fused-v1",
        "frozen_ref": "immutable source decode@da36c09d",
        "decode_sha256_prefix": "da36c09d",
        "decode_sha256": _PACKED_NZ_DECODE_SHA256,
        "source_role": "candidate",
        "storage_family": "packed_nz_w13_w2",
        "schedule_family": "mixed_fused_gmm1_swiglu_requant_then_down",
        "task_partition": (
            "one_mixed_aic_aiv_fused_task_then_one_mixed_aic_aiv_down_task"
        ),
        "expert_release_family": "packed_nz_mixed",
        "duration_limit_source": (
            "No new per-slice duration threshold is introduced here; timing "
            "qualification remains in the matched A/B/A and swimlane gates."
        ),
        "experimental": False,
        "enforce_candidate_release_gate": True,
    },
    "local-ep": {
        "policy_id": "release-local-ep-cdb2bb26-resident-dual-latch-22-v2",
        "frozen_ref": "immutable source decode@cdb2bb26",
        "decode_sha256_prefix": "cdb2bb26",
        "decode_sha256": _LOCAL_EP_DECODE_SHA256,
        "golden_protocol_profile": LOCAL_OWNER_PROTOCOL_PROFILE,
        "source_role": "candidate",
        "storage_family": "replicated_input_local_owner_packed_nz",
        "schedule_family": (
            "single_writer_route_map_owner_local_pack_plan_"
            "mixed_experts_local_combine_tp_all_reduce"
        ),
        "task_partition": (
            "one_route_metadata_task_owner_local_payload_grid_"
            "one_route_plan_task_mixed_expert_compute_local_combine_"
            "tp_all_reduce"
        ),
        "expert_release_family": "packed_nz_mixed",
        "mixed_resource_targets": {
            "expert_gate_up": {"aic": 22, "aiv": 44},
            "expert_down": {"aic": 23, "aiv": 46},
        },
        "duration_limit_source": (
            "No new per-slice duration threshold is introduced here; timing "
            "qualification remains in the matched A/B/A and swimlane gates."
        ),
        "experimental": True,
        "enforce_candidate_release_gate": True,
    },
    "row16": {
        "policy_id": "shared-experiment-reference-row16-v1",
        "frozen_ref": "immutable source decode@65b0b8bf",
        "decode_sha256_prefix": "65b0b8bf",
        "source_role": "reference",
        "storage_family": "row16_graph_wide_gate_up_int32_scratch",
        "schedule_family": "two_phase_split",
        "task_partition": (
            "graph_wide_gate_up_then_aiv_activation_quant_down"
        ),
        "experimental": False,
        "enforce_candidate_release_gate": True,
    },
    "shared-split": {
        "policy_id": "shared-experiment-5-5-16-v1",
        "frozen_ref": "immutable source decode@572ea2a2",
        "decode_sha256_prefix": "572ea2a2",
        "source_role": "candidate",
        "storage_family": (
            "row16_routed_scratch_plus_shared_fp32_gate_up_scratch"
        ),
        "schedule_family": (
            "routed_two_phase_plus_shared_gate_up_act_down_split"
        ),
        "task_partition": (
            "routed_row16_plus_shared_5_gate_up_5_activation_16_down"
        ),
        "experimental": True,
        "enforce_candidate_release_gate": True,
    },
}
_SOURCE_PROFILES = tuple(_FROZEN_SOURCE_POLICIES)
_ORIGIN_MAIN_COMPATIBILITY_REFERENCE = {
    "frozen_ref": "origin/main@1f48761c",
    "storage_family": "graph_wide_gate_up_int32_scratch",
    "schedule_family": "two_phase",
    "task_partition": "graph_wide",
    "campaign_baseline": False,
}
_ROUTED_PROFILE_STAGES = (
    "expert_gate_up",
    "expert_gate",
    "expert_up",
    "expert_gate_up_act",
    "expert_down",
)
_EXPERT_AIC_RELEASE_STAGES = {
    "staged_fused_gate_up": (
        "expert_gate_up",
        "expert_down",
    ),
    "split": (
        "expert_gate",
        "expert_up",
        "expert_down",
    ),
    "packed_nz_mixed": (
        "expert_gate_up",
        "expert_down",
    ),
}
_EXPERT_DURATION_LIMITS_US = {
    "staged_fused_gate_up": _STAGED_FUSED_DURATION_LIMITS_US,
    "split": {
        "p50_min": 10.0,
        "p50_max": 30.0,
        "p90_max": 30.0,
        "p99_max": 60.0,
        "max": 100.0,
    },
}
_PACKED_NZ_RESOURCE_TARGETS = {
    "expert_gate_up": {
        "aic": 24,
        "aiv": 48,
    },
    "expert_down": {
        "aic": 23,
        "aiv": 46,
    },
}


def _mixed_resource_grid_labels(
    targets: dict[str, dict[str, int]],
) -> dict[str, str]:
    return {
        stage: f"{int(values['aic'])} AIC/{int(values['aiv'])} AIV"
        for stage, values in targets.items()
    }


_LEGACY_DIAGNOSTIC_STAGE_RESOURCES = {
    "expert_gate_up": "aic",
    "expert_gate": "aic",
    "expert_up": "aic",
    "expert_gate_up_act": "aiv",
    "expert_down": "aic",
    "combine_scatter": "aiv",
    "combine_wait": "aiv",
    "combine_reduce": "aiv",
}
_LOCAL_EP_DIAGNOSTIC_STAGE_RESOURCES = {
    "local_route_map_init": "aiv",
    "local_route_pack": "aiv",
    "local_route_plan": "aiv",
    "expert_gate_up": "aic",
    "expert_down": "aic",
    "local_combine_reduce": "aiv",
    "moe_all_reduce": "aiv",
}
_LEGACY_TASK_TIMING_PROFILE_STAGES = (
    *_ROUTED_PROFILE_STAGES,
    "shared_mlp",
    "shared_gate_up",
    "shared_gate_up_act",
    "shared_down",
    "combine_scatter",
    "combine_wait",
    "combine_reduce",
)
_LOCAL_EP_TASK_TIMING_PROFILE_STAGES = (
    "local_route_map_init",
    "local_route_pack",
    "local_route_plan",
    *_ROUTED_PROFILE_STAGES,
    "shared_mlp",
    "shared_gate_up",
    "shared_gate_up_act",
    "shared_down",
    "local_combine_reduce",
    "moe_all_reduce",
)
_LEGACY_MARKDOWN_STAGE_ORDER = (
    "norm_quant",
    "gate_fanout",
    "gate_topk",
    "shared_mlp",
    "shared_gate_up",
    "shared_gate_up_act",
    "shared_down",
    "shared_split",
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
)
_LOCAL_EP_MARKDOWN_STAGE_ORDER = (
    "norm_quant",
    "gate_fanout",
    "gate_topk",
    "shared_mlp",
    "shared_gate_up",
    "shared_gate_up_act",
    "shared_down",
    "shared_split",
    "local_route_map_init",
    "local_route_pack",
    "local_route_plan",
    "expert_gate_up",
    "expert_down",
    "local_combine_reduce",
    "moe_all_reduce",
)
_CRITICAL_PATH_CONTRIBUTION_REASON = (
    "critical_path_report.md exposes aggregate totals and a name-only table; "
    "CPM_observed.json/CPM_static.json expose path membership but no "
    "machine-readable per-task contribution. A numeric contribution cannot "
    "be reconstructed without re-running the critical-path algorithm or a "
    "new structured sidecar."
)
_ROUTE_HISTOGRAM_REASON = (
    "Current DFX inputs contain dependency tasks and physical execution "
    "slices, but no token-to-expert assignments or per-expert routed-token "
    "counts. Routed task/tile counts are execution-shape metadata and are "
    "explicitly rejected as a route histogram proxy."
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
    swimlane_level: int | None = None
    predicated_skip_task_ids: tuple[str, ...] = ()

    @property
    def all_slices(self) -> list[Slice]:
        return [item for task_slices in self.slices_by_task.values() for item in task_slices]


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
    parser.add_argument(
        "--recv-meta-sidecar",
        help=(
            "optional read-only .pt/.pth/.json recv_meta sidecar from the "
            "independent instrumented L3/L4 program"
        ),
    )
    parser.add_argument(
        "--profile",
        choices=_SOURCE_PROFILES,
        help=(
            "source profile; defaults to PYPTO_MOE_DFX_PROFILE, then "
            "SOURCE_KIND, then candidate"
        ),
    )
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument(
        "--source-decode-sha256",
        help=(
            "exact models/step3p5/decode_fwd.py SHA256 for source-policy "
            "validation"
        ),
    )
    return parser.parse_args()


def _resolve_profile(profile: str | None) -> str:
    resolved = (
        profile
        or os.environ.get("PYPTO_MOE_DFX_PROFILE")
        or os.environ.get("SOURCE_KIND")
        or "candidate"
    )
    if resolved not in _FROZEN_SOURCE_POLICIES:
        raise ValueError(
            f"profile must be one of {_SOURCE_PROFILES}, got {resolved!r}"
        )
    return resolved


def _source_policy(profile: str | None) -> dict[str, Any]:
    resolved = _resolve_profile(profile)
    return {
        **_FROZEN_SOURCE_POLICIES[resolved],
        "policy_source": "frozen analyzer policy table",
        "selection_semantics": (
            "Selected only by --profile/PYPTO_MOE_DFX_PROFILE/SOURCE_KIND; "
            "source family fields are never inferred from dependency task "
            "names."
        ),
        "origin_main_compatibility_reference": (
            dict(_ORIGIN_MAIN_COMPATIBILITY_REFERENCE)
        ),
    }


def _arrival_pairs(
    profile: str | None = None,
) -> tuple[tuple[str, str, str], ...]:
    if _resolve_profile(profile) == "local-ep":
        return _LOCAL_EP_ARRIVAL_PAIRS
    return _LEGACY_ARRIVAL_PAIRS


def _diagnostic_stage_resources(
    profile: str | None = None,
) -> dict[str, str]:
    if _resolve_profile(profile) == "local-ep":
        return dict(_LOCAL_EP_DIAGNOSTIC_STAGE_RESOURCES)
    return dict(_LEGACY_DIAGNOSTIC_STAGE_RESOURCES)


def _timing_profile_stages(
    profile: str | None = None,
) -> tuple[str, ...]:
    if _resolve_profile(profile) == "local-ep":
        return _LOCAL_EP_TASK_TIMING_PROFILE_STAGES
    return _LEGACY_TASK_TIMING_PROFILE_STAGES


def _markdown_stage_order(
    profile: str | None = None,
) -> tuple[str, ...]:
    if _resolve_profile(profile) == "local-ep":
        return _LOCAL_EP_MARKDOWN_STAGE_ORDER
    return _LEGACY_MARKDOWN_STAGE_ORDER


def _source_identity_contract(
    profile: str | None,
    source_decode_sha256: str | None,
) -> dict[str, Any]:
    """Match explicit source provenance to the selected frozen policy."""
    policy = _source_policy(profile)
    expected_prefix = str(policy["decode_sha256_prefix"])
    expected_sha256 = policy.get("decode_sha256")
    if source_decode_sha256 is None:
        return {
            "available": False,
            "pass": None,
            "actual_decode_sha256": None,
            "expected_decode_sha256_prefix": expected_prefix,
            "expected_decode_sha256": expected_sha256,
            "policy_id": policy["policy_id"],
        }
    if not _SHA256_PATTERN.fullmatch(source_decode_sha256):
        raise ValueError("source_decode_sha256 must be a lowercase SHA256")
    if not isinstance(expected_sha256, str):
        return {
            "available": True,
            "pass": False,
            "actual_decode_sha256": source_decode_sha256,
            "expected_decode_sha256_prefix": expected_prefix,
            "expected_decode_sha256": None,
            "policy_id": policy["policy_id"],
            "reason": "selected source policy has no exact SHA256",
        }
    return {
        "available": True,
        "pass": source_decode_sha256 == expected_sha256,
        "actual_decode_sha256": source_decode_sha256,
        "expected_decode_sha256_prefix": expected_prefix,
        "expected_decode_sha256": expected_sha256,
        "policy_id": policy["policy_id"],
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")
    if not values:
        return 0.0
    ordered = sorted(values)
    # Conservative nearest-rank percentile: rank=ceil(q*n), 1-indexed.
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(quantile * len(ordered)) - 1),
    )
    return ordered[index]


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _duration_distribution(values: list[float]) -> dict[str, Any]:
    """Return exact physical-slice duration gates and summary statistics."""
    durations = [float(value) for value in values]
    count = len(durations)
    buckets = {
        "lt_10_us": sum(value < 10.0 for value in durations),
        "from_10_to_30_us": sum(10.0 <= value <= 30.0 for value in durations),
        "gt_30_us": sum(value > 30.0 for value in durations),
    }
    gates = {
        name: {
            "count": bucket_count,
            "ratio": _round(bucket_count / count, 4) if count else None,
        }
        for name, bucket_count in buckets.items()
    }
    if not durations:
        return {
            "available": False,
            "count": 0,
            "min_us": None,
            "p50_us": None,
            "p90_us": None,
            "p99_us": None,
            "max_us": None,
            "gates": gates,
            "gate_semantics": ("<10, 10<=duration<=30, and >30 microseconds"),
            "reason": "no physical slices for this resource",
        }
    return {
        "available": True,
        "count": count,
        "min_us": _round(min(durations)),
        "p50_us": _round(_percentile(durations, 0.50)),
        "p90_us": _round(_percentile(durations, 0.90)),
        "p99_us": _round(_percentile(durations, 0.99)),
        "max_us": _round(max(durations)),
        "gates": gates,
        "gate_semantics": "<10, 10<=duration<=30, and >30 microseconds",
    }


def _slice_record(trace: RankTrace, item: Slice) -> dict[str, Any]:
    return {
        "task_id": item.task_id,
        "core_id": item.core,
        "resource": item.resource,
        "start_tick": item.start,
        "end_tick": item.end,
        "service_span_us": _round((item.end - item.start) / trace.frequency_hz * 1e6),
    }


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


def _raw_swimlane_metadata(swim_path: Path) -> tuple[int, list[str]]:
    raw = json.loads(swim_path.read_text(encoding="utf-8"))
    metadata = raw.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{swim_path}: missing metadata object")
    try:
        level = int(raw["chip_swimlane_level"])
        num_cores = int(metadata["num_cores"])
        core_types = [str(value) for value in metadata["core_types"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{swim_path}: invalid chip swimlane metadata") from exc
    if num_cores != len(core_types):
        raise ValueError(f"{swim_path}: num_cores={num_cores} but core_types={len(core_types)}")
    if core_types != _EXPECTED_CORE_TYPES:
        raise ValueError(
            f"{swim_path}: expected 24 AIC + 48 AIV cores, got "
            f"{core_types.count('aic')} AIC + "
            f"{core_types.count('aiv')} AIV"
        )
    return level, core_types


def _predicated_skip_task_ids(swim: dict[str, Any]) -> tuple[str, ...]:
    """Return task IDs with explicit runtime predicate-retirement evidence."""
    return tuple(
        str(event["task_id"])
        for lane in swim.get("aicpu_scheduler_phases", [])
        for event in lane
        if event.get("phase") == "predicated_skip"
    )


def _load_rank(rank_dir: Path, dfx_root: Path) -> RankTrace | None:
    deps_path = rank_dir / "deps.json"
    names_path = rank_dir / "name_map.json"
    swim_path = rank_dir / _CHIP_SWIMLANE_RECORDS_NAME
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

    swimlane_level, core_types = _raw_swimlane_metadata(swim_path)
    swim = read_perf_data(swim_path)
    perf_tasks = list(swim.get("tasks", []))
    tasks = [
        Task(
            task_id=str(item["task_id"]),
            order=order,
            name=_callable_name(item, names),
            block_num=int(item.get("block_num", 0)),
            kernel_ids=tuple(int(value) if value is not None else -1 for value in item.get("kernel_ids", [])),
            early_dispatch=bool(item.get("early_dispatch", False)),
        )
        for order, item in enumerate(deps.get("tasks", []))
    ]
    task_by_id = {task.task_id: task for task in tasks}
    slices_by_task: dict[str, list[Slice]] = collections.defaultdict(list)
    for item in perf_tasks:
        core = int(item["core_id"])
        if not 0 <= core < len(core_types):
            raise ValueError(
                f"{swim_path}: observed core_id={core} outside metadata capacity {len(core_types)}"
            )
        resource = str(item.get("core_type", core_types[core]))
        if resource != core_types[core]:
            raise ValueError(
                f"{swim_path}: core_id={core} metadata={core_types[core]} but observed={resource}"
            )
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
                resource=resource,
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
        swimlane_level=swimlane_level,
        predicated_skip_task_ids=_predicated_skip_task_ids(swim),
    )


def _task_matches_layer(task: Task, layer: str, suffix: str) -> bool:
    prefix = _LAYER_PREFIX[layer]
    base = _strip_resource_suffix(task.name)
    expected = f"{prefix}{suffix}"
    if layer == "L3":
        return base == expected
    return base == expected and not base.startswith(_LAYER_PREFIX["L3"])


def _has_dependency_edge(trace: RankTrace, pred: str, succ: str) -> bool:
    return any(
        str(edge.get("pred")) == pred and str(edge.get("succ")) == succ
        for edge in trace.edges
    )


def _find_packed_nz_layer_tasks(
    trace: RankTrace,
    layer: str,
    stage_ids: dict[str, list[str]],
    profile: str = "candidate",
) -> dict[str, str]:
    """Map layer-agnostic packed-NZ extern tasks to one MoE layer."""
    resolved_profile = _resolve_profile(profile)
    packed_names = set(_PACKED_NZ_EXTERNAL_STAGES.values())
    if not any(
        _strip_resource_suffix(task.name) in packed_names
        for task in trace.tasks
    ):
        return {}

    if resolved_profile == "local-ep":
        start_stage = "local_route_plan"
        end_stage = "local_combine_reduce"
        window_name = "local_route_plan -> local_combine_reduce"
    else:
        start_stage = "dispatch_gather"
        end_stage = "combine_scatter"
        window_name = "dispatch_gather -> combine_scatter"
    start_ids = stage_ids[start_stage]
    end_ids = stage_ids[end_stage]
    if len(start_ids) != 1 or len(end_ids) != 1:
        raise RuntimeError(
            f"{trace.tag}/{layer}: packed-NZ mapping requires "
            f"exactly one {start_stage} and exactly one "
            f"{end_stage} task; "
            f"start={start_ids}, end={end_ids}"
        )
    start_tasks = [trace.task_by_id[task_id] for task_id in start_ids]
    start_task = start_tasks[0]
    end_task = trace.task_by_id[end_ids[0]]
    if start_task.order >= end_task.order:
        raise RuntimeError(
            f"{trace.tag}/{layer}: invalid packed-NZ task window; "
            f"start_order={start_task.order}, end_order={end_task.order}"
        )

    window = [
        task
        for task in trace.tasks
        if start_task.order < task.order < end_task.order
    ]
    mapped: dict[str, Task] = {}
    for stage, expected_name in _PACKED_NZ_EXTERNAL_STAGES.items():
        matches = [
            task
            for task in window
            if _strip_resource_suffix(task.name) == expected_name
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"{trace.tag}/{layer}: expected exactly one {expected_name!r} "
                f"dependency task inside the {window_name} window, got "
                f"{[(task.task_id, task.order, task.name) for task in matches]}"
            )
        mapped[stage] = matches[0]

    fused = mapped["expert_gate_up"]
    down = mapped["expert_down"]
    order_valid = (
        start_task.order < fused.order < down.order < end_task.order
    )
    if resolved_profile == "local-ep":
        if not order_valid:
            raise RuntimeError(
                f"{trace.tag}/{layer}: invalid packed-NZ task order inside "
                f"{window_name}; start_orders="
                f"{[task.order for task in start_tasks]}, "
                f"fused={fused.order}, down={down.order}, "
                f"end={end_task.order}"
            )
        return {stage: task.task_id for stage, task in mapped.items()}

    dependency_chain = {
        "gather_to_fused": _has_dependency_edge(
            trace,
            start_task.task_id,
            fused.task_id,
        ),
        "fused_to_down": _has_dependency_edge(
            trace,
            fused.task_id,
            down.task_id,
        ),
        "down_to_scatter": _has_dependency_edge(
            trace,
            down.task_id,
            end_task.task_id,
        ),
    }
    if not order_valid or not all(dependency_chain.values()):
        raise RuntimeError(
            f"{trace.tag}/{layer}: invalid packed-NZ dependency chain; "
            f"order_valid={order_valid}, edges={dependency_chain}"
        )
    return {stage: task.task_id for stage, task in mapped.items()}


def _find_layer_task_ids(
    trace: RankTrace,
    layer: str,
    profile: str = "candidate",
) -> dict[str, list[str]]:
    resolved_profile = _resolve_profile(profile)
    result: dict[str, list[str]] = {}
    for stage, suffixes in _STAGE_SUFFIXES.items():
        result[stage] = [
            task.task_id
            for task in trace.tasks
            if any(_task_matches_layer(task, layer, suffix) for suffix in suffixes)
        ]

    packed_nz_tasks = _find_packed_nz_layer_tasks(
        trace,
        layer,
        result,
        resolved_profile,
    )
    for stage, task_id in packed_nz_tasks.items():
        if result[stage]:
            raise RuntimeError(
                f"{trace.tag}/{layer}: both named and packed-NZ tasks map to "
                f"{stage}: named={result[stage]}, packed={task_id}"
            )
        result[stage] = [task_id]

    gate_tasks = [trace.task_by_id[task_id] for task_id in result["gate_init"]]
    if resolved_profile == "local-ep":
        result["norm_quant"] = [
            task.task_id
            for task in trace.tasks
            if _task_matches_layer(
                task,
                layer,
                "norm_quant_moe_input",
            )
        ]
    elif gate_tasks:
        gate_order = min(task.order for task in gate_tasks)
        prior_norm = [
            task for task in trace.tasks if task.name == "_norm_quant_moe_input" and task.order < gate_order
        ]
        if prior_norm:
            result["norm_quant"] = [max(prior_norm, key=lambda task: task.order).task_id]
        else:
            result["norm_quant"] = []
    else:
        result["norm_quant"] = []

    split_shared_stages = (
        "shared_gate_up",
        "shared_gate_up_act",
        "shared_down",
    )
    result["shared_split"] = [
        task_id
        for stage in split_shared_stages
        for task_id in result[stage]
    ]
    shared_task_ids = [
        *result["shared_mlp"],
        *result["shared_split"],
    ]
    shared_tasks = [
        trace.task_by_id[task_id]
        for task_id in shared_task_ids
    ]
    dispatch_tasks = [trace.task_by_id[task_id] for task_id in result["dispatch_meta"]]
    if shared_tasks and dispatch_tasks:
        shared_order = max(task.order for task in shared_tasks)
        dispatch_order = min(task.order for task in dispatch_tasks)
        all_reduces = [
            task
            for task in trace.tasks
            if task.name == "tp_all_reduce" and shared_order < task.order < dispatch_order
        ]
        result["shared_all_reduce"] = (
            [min(all_reduces, key=lambda task: task.order).task_id] if all_reduces else []
        )
    else:
        result["shared_all_reduce"] = []
    result["moe_all_reduce"] = []
    if resolved_profile == "local-ep":
        combine_ids = result["local_combine_reduce"]
        residual_ids = result["moe_residual_add"]
        if len(combine_ids) == 1 and len(residual_ids) == 1:
            combine = trace.task_by_id[combine_ids[0]]
            residual = trace.task_by_id[residual_ids[0]]
            all_reduces = [
                task
                for task in trace.tasks
                if (
                    task.name == "tp_all_reduce"
                    and combine.order < task.order < residual.order
                )
            ]
            if len(all_reduces) != 1:
                raise RuntimeError(
                    f"{trace.tag}/{layer}: local-EP mapping requires exactly "
                    "one tp_all_reduce between local_combine_reduce and "
                    f"moe_residual_add, got "
                    f"{[(task.task_id, task.order) for task in all_reduces]}"
                )
            result["moe_all_reduce"] = [all_reduces[0].task_id]
    return result


def _is_executable_task(task: Task) -> bool:
    return task.block_num > 0 and any(kernel_id >= 0 for kernel_id in task.kernel_ids)


def _task_id_contract(trace: RankTrace) -> dict[str, Any]:
    """Reconcile executable dependency task IDs with physical swim task IDs."""
    all_dep_ids = [task.task_id for task in trace.tasks]
    executable_dep_ids = [task.task_id for task in trace.tasks if _is_executable_task(task)]
    dep_id_set = set(executable_dep_ids)
    ignored_non_executable_ids = sorted(
        task.task_id for task in trace.tasks if not _is_executable_task(task)
    )
    swim_id_set = {str(task_id) for task_id, slices in trace.slices_by_task.items() if slices}
    duplicate_dep_ids = sorted(
        task_id for task_id, count in collections.Counter(all_dep_ids).items() if count > 1
    )
    invalid_physical_slices = [
        {
            "map_task_id": map_task_id,
            "slice_task_id": item.task_id,
            "core_id": item.core,
            "resource": item.resource,
            "start_tick": item.start,
            "end_tick": item.end,
        }
        for map_task_id, slices in trace.slices_by_task.items()
        for item in slices
        if (
            item.task_id != str(map_task_id)
            or item.end <= item.start
            or not 0 <= item.core < len(trace.core_types)
            or item.resource != trace.core_types[item.core]
        )
    ]
    predicated_skip_counts = collections.Counter(
        trace.predicated_skip_task_ids
    )
    predicated_skip_id_set = set(predicated_skip_counts)
    duplicate_predicated_skip_task_ids = sorted(
        task_id
        for task_id, count in predicated_skip_counts.items()
        if count > 1
    )
    missing_physical_ids = dep_id_set - swim_id_set
    predicated_skip_without_physical_slices = sorted(
        missing_physical_ids & predicated_skip_id_set
    )
    missing_on_swim = sorted(
        missing_physical_ids - predicated_skip_id_set
    )
    unknown_on_swim = sorted(swim_id_set - dep_id_set)
    unexpected_predicated_skip_task_ids = sorted(
        predicated_skip_id_set - dep_id_set
    )
    predicated_skip_with_physical_slices = sorted(
        predicated_skip_id_set & swim_id_set
    )
    exact = not (
        duplicate_dep_ids
        or invalid_physical_slices
        or missing_on_swim
        or unknown_on_swim
        or duplicate_predicated_skip_task_ids
        or unexpected_predicated_skip_task_ids
        or predicated_skip_with_physical_slices
    )
    return {
        "pass": exact,
        "all_dep_task_count": len(all_dep_ids),
        "dep_task_count": len(executable_dep_ids),
        "dep_unique_task_count": len(dep_id_set),
        "swim_task_count": len(swim_id_set),
        "dep_task_ids": sorted(dep_id_set),
        "swim_task_ids": sorted(swim_id_set),
        "ignored_non_executable_dep_task_ids": ignored_non_executable_ids,
        "duplicate_dep_task_ids": duplicate_dep_ids,
        "invalid_physical_slices": invalid_physical_slices,
        "missing_on_swim": missing_on_swim,
        "unknown_on_swim": unknown_on_swim,
        "predicated_skip_task_ids": sorted(predicated_skip_id_set),
        "predicated_skip_without_physical_slices": (
            predicated_skip_without_physical_slices
        ),
        "duplicate_predicated_skip_task_ids": (
            duplicate_predicated_skip_task_ids
        ),
        "unexpected_predicated_skip_task_ids": (
            unexpected_predicated_skip_task_ids
        ),
        "predicated_skip_with_physical_slices": (
            predicated_skip_with_physical_slices
        ),
        "interpretation": (
            "Every executable dependency task must have at least one physical "
            "swim slice or one explicit predicated_skip scheduler event. Every "
            "physical and skipped task ID must resolve to exactly one "
            "executable dependency task, and a skipped task cannot also have "
            "physical slices. Runtime/creator-only dependency records are "
            "listed separately and are not required to execute."
        ),
    }


def _edge_summary(edge: dict[str, Any]) -> dict[str, Any]:
    return {key: edge.get(key) for key in ("pred", "succ", "source", "arg", "tensor_id") if key in edge}


def _edges_between(
    trace: RankTrace,
    pred: str,
    succ: str,
) -> list[dict[str, Any]]:
    return [edge for edge in trace.edges if str(edge.get("pred")) == pred and str(edge.get("succ")) == succ]


def _combine_dependency_contract(trace: RankTrace) -> dict[str, Any]:
    """Validate every per-layer scatter -> wait -> reduce task chain."""
    stage_ids_by_layer = {layer: _find_layer_task_ids(trace, layer) for layer in _LAYER_PREFIX}
    combine_owner: dict[str, tuple[str, str]] = {}
    for layer, stage_ids in stage_ids_by_layer.items():
        for stage in (
            "combine_scatter",
            "combine_wait",
            "combine_reduce",
        ):
            for task_id in stage_ids[stage]:
                combine_owner[task_id] = (layer, stage)

    cross_layer_or_unexpected_explicit = []
    for edge in trace.edges:
        if edge.get("source") != "explicit":
            continue
        pred = str(edge.get("pred"))
        succ = str(edge.get("succ"))
        pred_owner = combine_owner.get(pred)
        succ_owner = combine_owner.get(succ)
        if pred_owner is None or succ_owner is None:
            continue
        expected_stage_pair = (
            pred_owner[0] == succ_owner[0]
            and (pred_owner[1], succ_owner[1])
            in {
                ("combine_scatter", "combine_wait"),
                ("combine_wait", "combine_reduce"),
            }
        )
        if expected_stage_pair:
            continue
        cross_layer_or_unexpected_explicit.append(
            {
                **_edge_summary(edge),
                "pred_layer": pred_owner[0] if pred_owner else None,
                "pred_stage": pred_owner[1] if pred_owner else None,
                "succ_layer": succ_owner[0] if succ_owner else None,
                "succ_stage": succ_owner[1] if succ_owner else None,
            }
        )

    layers: dict[str, Any] = {}
    all_errors: list[dict[str, Any]] = []
    for layer, stage_ids in stage_ids_by_layer.items():
        layer_errors: list[dict[str, Any]] = []
        task_ids = {
            stage: list(stage_ids[stage])
            for stage in (
                "combine_scatter",
                "combine_wait",
                "combine_reduce",
            )
        }
        stage_counts = {stage: len(ids) for stage, ids in task_ids.items()}
        if not all(stage_counts.values()) or len(set(stage_counts.values())) != 1:
            layer_errors.append(
                {
                    "code": "stage_task_count_mismatch",
                    "stage_counts": stage_counts,
                    "stage_task_ids": task_ids,
                    "expected": (
                        "each stage must be nonempty and scatter/wait/reduce "
                        "counts must match"
                    ),
                }
            )

        scatter_ids = set(task_ids["combine_scatter"])
        wait_ids = set(task_ids["combine_wait"])
        reduce_ids = set(task_ids["combine_reduce"])
        explicit_edges = [
            edge for edge in trace.edges if edge.get("source") == "explicit"
        ]
        scatter_wait_edges = [
            edge
            for edge in explicit_edges
            if str(edge.get("pred")) in scatter_ids
            and str(edge.get("succ")) in wait_ids
        ]
        wait_reduce_edges = [
            edge
            for edge in explicit_edges
            if str(edge.get("pred")) in wait_ids
            and str(edge.get("succ")) in reduce_ids
        ]

        def degree_errors(
            *,
            task_ids_to_check: set[str],
            edges: list[dict[str, Any]],
            endpoint: str,
            code: str,
        ) -> None:
            for task_id in sorted(
                task_ids_to_check,
                key=lambda value: trace.task_by_id[value].order,
            ):
                matches = [
                    edge
                    for edge in edges
                    if str(edge.get(endpoint)) == task_id
                ]
                if len(matches) == 1:
                    continue
                layer_errors.append(
                    {
                        "code": code,
                        "task_id": task_id,
                        "endpoint": endpoint,
                        "expected": 1,
                        "actual": len(matches),
                        "edges": [_edge_summary(edge) for edge in matches],
                    }
                )

        degree_errors(
            task_ids_to_check=scatter_ids,
            edges=scatter_wait_edges,
            endpoint="pred",
            code="scatter_to_wait_out_degree",
        )
        degree_errors(
            task_ids_to_check=wait_ids,
            edges=scatter_wait_edges,
            endpoint="succ",
            code="scatter_to_wait_in_degree",
        )
        degree_errors(
            task_ids_to_check=wait_ids,
            edges=wait_reduce_edges,
            endpoint="pred",
            code="wait_to_reduce_out_degree",
        )
        degree_errors(
            task_ids_to_check=reduce_ids,
            edges=wait_reduce_edges,
            endpoint="succ",
            code="wait_to_reduce_in_degree",
        )

        chains = []
        for scatter in sorted(
            scatter_ids,
            key=lambda value: trace.task_by_id[value].order,
        ):
            scatter_wait_matches = [
                edge
                for edge in scatter_wait_edges
                if str(edge.get("pred")) == scatter
            ]
            if len(scatter_wait_matches) != 1:
                continue
            wait = str(scatter_wait_matches[0].get("succ"))
            wait_reduce_matches = [
                edge
                for edge in wait_reduce_edges
                if str(edge.get("pred")) == wait
            ]
            if len(wait_reduce_matches) != 1:
                continue
            reduce = str(wait_reduce_matches[0].get("succ"))
            chain_errors = []
            task_ordered = (
                trace.task_by_id[scatter].order
                < trace.task_by_id[wait].order
                < trace.task_by_id[reduce].order
            )
            if not task_ordered:
                chain_errors.append(
                    {
                        "code": "task_order",
                        "scatter_task_id": scatter,
                        "wait_task_id": wait,
                        "reduce_task_id": reduce,
                        "scatter_order": trace.task_by_id[scatter].order,
                        "wait_order": trace.task_by_id[wait].order,
                        "reduce_order": trace.task_by_id[reduce].order,
                    }
                )

            stage_envelopes: dict[str, dict[str, int]] = {}
            missing_swim_timing = []
            for stage, task_id in (
                ("combine_scatter", scatter),
                ("combine_wait", wait),
                ("combine_reduce", reduce),
            ):
                slices = trace.slices_by_task.get(task_id, [])
                if not slices:
                    missing_swim_timing.append(task_id)
                    continue
                stage_envelopes[stage] = {
                    "start_tick": min(item.start for item in slices),
                    "end_tick": max(item.end for item in slices),
                }
            if missing_swim_timing:
                local_swim_order: dict[str, Any] = {
                    "available": False,
                    "pass": False,
                    "missing_task_ids": missing_swim_timing,
                    "reason": "one or more combine tasks have no physical swim timing",
                }
                chain_errors.append(
                    {
                        "code": "missing_swim_timing",
                        "task_ids": missing_swim_timing,
                    }
                )
            else:
                scatter_span = stage_envelopes["combine_scatter"]
                wait_span = stage_envelopes["combine_wait"]
                reduce_span = stage_envelopes["combine_reduce"]
                start_ordered = (
                    scatter_span["start_tick"]
                    <= wait_span["start_tick"]
                    <= reduce_span["start_tick"]
                )
                dependency_ordered = (
                    scatter_span["end_tick"] <= wait_span["start_tick"]
                    and wait_span["end_tick"] <= reduce_span["start_tick"]
                )
                local_swim_order = {
                    "available": True,
                    "pass": start_ordered and dependency_ordered,
                    "stage_envelopes": stage_envelopes,
                    "start_ordered": start_ordered,
                    "dependency_completion_ordered": dependency_ordered,
                    "semantics": (
                        "All ticks are compared only within this rank. The "
                        "scatter envelope must finish before wait starts, and "
                        "the wait envelope must finish before reduce starts."
                    ),
                }
                if not local_swim_order["pass"]:
                    chain_errors.append(
                        {
                            "code": "local_swim_execution_order",
                            "scatter_task_id": scatter,
                            "wait_task_id": wait,
                            "reduce_task_id": reduce,
                            "stage_envelopes": stage_envelopes,
                            "start_ordered": start_ordered,
                            "dependency_completion_ordered": dependency_ordered,
                        }
                    )

            scatter_reduce_data = [
                edge
                for edge in _edges_between(trace, scatter, reduce)
                if edge.get("source") == "tensormap"
            ]
            required_edges = {
                "scatter_to_wait_explicit": {
                    "pred": scatter,
                    "succ": wait,
                    "matches": [
                        _edge_summary(edge) for edge in scatter_wait_matches
                    ],
                    "pass": len(scatter_wait_matches) == 1,
                },
                "wait_to_reduce_explicit": {
                    "pred": wait,
                    "succ": reduce,
                    "matches": [
                        _edge_summary(edge) for edge in wait_reduce_matches
                    ],
                    "pass": len(wait_reduce_matches) == 1,
                },
                "scatter_to_reduce_data": {
                    "pred": scatter,
                    "succ": reduce,
                    "matches": [
                        _edge_summary(edge) for edge in scatter_reduce_data
                    ],
                    "pass": bool(scatter_reduce_data),
                },
            }
            if not scatter_reduce_data:
                chain_errors.append(
                    {
                        "code": "required_edge",
                        "edge": "scatter_to_reduce_data",
                        "pred": scatter,
                        "succ": reduce,
                        "match_count": 0,
                        "expected": "at_least_one",
                    }
                )
            chain = {
                "chain_id": f"{scatter}->{wait}->{reduce}",
                "pass": not chain_errors,
                "task_ids": {
                    "combine_scatter": scatter,
                    "combine_wait": wait,
                    "combine_reduce": reduce,
                },
                "task_ordered": task_ordered,
                "required_edges": required_edges,
                "local_swim_order": local_swim_order,
                "errors": chain_errors,
            }
            chains.append(chain)
            layer_errors.extend(
                {"chain_id": chain["chain_id"], **error}
                for error in chain_errors
            )

        layer_cross_edges = [
            edge
            for edge in cross_layer_or_unexpected_explicit
            if (edge["pred_layer"] == layer or edge["succ_layer"] == layer)
        ]
        if layer_cross_edges:
            layer_errors.append(
                {
                    "code": "unexpected_or_cross_layer_explicit_edge",
                    "edges": layer_cross_edges,
                }
            )
        expected_chain_count = len(scatter_ids)
        if len(chains) != expected_chain_count:
            layer_errors.append(
                {
                    "code": "resolved_chain_count",
                    "expected": expected_chain_count,
                    "actual": len(chains),
                }
            )
        local_swim_chains = [chain["local_swim_order"] for chain in chains]
        layers[layer] = {
            "pass": not layer_errors,
            "stage_task_ids": task_ids,
            "stage_task_counts": stage_counts,
            "chain_count": len(chains),
            "chains": chains,
            "local_swim_order": {
                "available": (
                    len(local_swim_chains) == expected_chain_count
                    and all(item["available"] for item in local_swim_chains)
                ),
                "pass": (
                    len(local_swim_chains) == expected_chain_count
                    and all(item["pass"] for item in local_swim_chains)
                ),
                "chain_count": len(local_swim_chains),
            },
            "unexpected_explicit_edges": layer_cross_edges,
            "errors": layer_errors,
        }
        all_errors.extend({"layer": layer, **error} for error in layer_errors)
    return {
        "pass": not all_errors,
        "layers": layers,
        "errors": all_errors,
        "interpretation": (
            "Every combine scatter must map one-to-one through explicit edges "
            "to one wait and one reduce task. Every reduce must consume its "
            "paired scatter output through a tensormap edge. The same order "
            "must be visible in local-rank swim timing for every chain, and "
            "unexpected explicit edges between combine-chain tasks are invalid."
        ),
    }


def _local_ep_dependency_contract(trace: RankTrace) -> dict[str, Any]:
    """Validate the replicated-input local-owner MoE dependency chain."""
    stage_ids_by_layer = {
        layer: _find_layer_task_ids(trace, layer, "local-ep")
        for layer in _LAYER_PREFIX
    }
    producer_stages = (
        "norm_quant",
        "gate_topk",
    )
    chain_stages = (
        "local_route_map_init",
        "local_route_pack",
        "local_route_plan",
        "expert_gate_up",
        "expert_down",
        "local_combine_reduce",
        "moe_all_reduce",
        "moe_residual_add",
    )
    mandatory_swim_stages = (
        *producer_stages,
        "local_route_map_init",
        "local_route_pack",
        "local_route_plan",
        "local_combine_reduce",
        "moe_all_reduce",
        "moe_residual_add",
    )
    expert_stages = ("expert_gate_up", "expert_down")
    required_edge_specs = (
        (
            "gate_to_map_data",
            "gate_topk",
            "local_route_map_init",
            ("tensormap",),
            "at_least_one",
        ),
        (
            "gate_to_pack_data",
            "gate_topk",
            "local_route_pack",
            ("tensormap",),
            "at_least_one",
        ),
        (
            "norm_to_pack_data",
            "norm_quant",
            "local_route_pack",
            ("tensormap",),
            "at_least_one",
        ),
        (
            "map_to_pack_explicit",
            "local_route_map_init",
            "local_route_pack",
            ("explicit",),
            "exactly_one",
        ),
        (
            "pack_to_plan_explicit",
            "local_route_pack",
            "local_route_plan",
            ("explicit",),
            "exactly_one",
        ),
        (
            "plan_to_expert_explicit",
            "local_route_plan",
            "expert_gate_up",
            ("explicit",),
            "exactly_one",
        ),
        (
            "plan_to_down_explicit",
            "local_route_plan",
            "expert_down",
            ("explicit",),
            "exactly_one",
        ),
        (
            "expert_to_down_explicit",
            "expert_gate_up",
            "expert_down",
            ("explicit",),
            "exactly_one",
        ),
        (
            "plan_to_combine_explicit",
            "local_route_plan",
            "local_combine_reduce",
            ("explicit",),
            "exactly_one",
        ),
        (
            "down_to_combine_explicit",
            "expert_down",
            "local_combine_reduce",
            ("explicit",),
            "exactly_one",
        ),
        (
            "map_to_pack_data",
            "local_route_map_init",
            "local_route_pack",
            ("tensormap",),
            "at_least_one",
        ),
        (
            "map_to_plan_data",
            "local_route_map_init",
            "local_route_plan",
            ("tensormap",),
            "at_least_one",
        ),
        (
            "map_to_combine_data",
            "local_route_map_init",
            "local_combine_reduce",
            ("tensormap",),
            "at_least_one",
        ),
        (
            "map_to_expert_count_data",
            "local_route_map_init",
            "expert_gate_up",
            ("tensormap",),
            "at_least_one",
        ),
        (
            "map_to_down_count_data",
            "local_route_map_init",
            "expert_down",
            ("tensormap",),
            "at_least_one",
        ),
        (
            "pack_to_expert_data",
            "local_route_pack",
            "expert_gate_up",
            ("tensormap",),
            "at_least_one",
        ),
        (
            "pack_to_down_data",
            "local_route_pack",
            "expert_down",
            ("tensormap",),
            "at_least_one",
        ),
        (
            "plan_to_expert_data",
            "local_route_plan",
            "expert_gate_up",
            ("tensormap",),
            "at_least_one",
        ),
        (
            "plan_to_down_data",
            "local_route_plan",
            "expert_down",
            ("tensormap",),
            "at_least_one",
        ),
        (
            "expert_to_down_data",
            "expert_gate_up",
            "expert_down",
            ("tensormap",),
            "at_least_one",
        ),
        (
            "down_to_combine_data",
            "expert_down",
            "local_combine_reduce",
            ("tensormap",),
            "at_least_one",
        ),
        (
            "shared_down_to_combine_data",
            "shared_output",
            "local_combine_reduce",
            ("tensormap",),
            "at_least_one",
        ),
        (
            "combine_to_all_reduce_data",
            "local_combine_reduce",
            "moe_all_reduce",
            ("tensormap",),
            "at_least_one",
        ),
        (
            "all_reduce_to_residual_data",
            "moe_all_reduce",
            "moe_residual_add",
            ("tensormap",),
            "at_least_one",
        ),
    )

    layers: dict[str, Any] = {}
    all_errors: list[dict[str, Any]] = []
    skipped_ids = set(trace.predicated_skip_task_ids)
    for layer, stage_ids in stage_ids_by_layer.items():
        layer_errors: list[dict[str, Any]] = []
        task_ids = {
            stage: list(stage_ids[stage])
            for stage in (*producer_stages, *chain_stages)
        }
        shared_candidates = [
            *stage_ids["shared_down"],
            *stage_ids["shared_mlp"],
        ]
        task_ids["shared_output"] = shared_candidates
        stage_counts = {
            stage: len(ids)
            for stage, ids in task_ids.items()
        }
        for stage, count in stage_counts.items():
            if count == 1:
                continue
            layer_errors.append(
                {
                    "code": "stage_task_count",
                    "stage": stage,
                    "expected": 1,
                    "actual": count,
                    "task_ids": task_ids[stage],
                }
            )

        resolved_ids = {
            stage: ids[0]
            for stage, ids in task_ids.items()
            if len(ids) == 1
        }
        task_order: dict[str, Any] = {
            "available": all(
                stage in resolved_ids
                for stage in (
                    *producer_stages,
                    *chain_stages,
                    "shared_output",
                )
            ),
            "pass": False,
        }
        if task_order["available"]:
            ordered = (
                trace.task_by_id[
                    resolved_ids["local_route_map_init"]
                ].order
                < trace.task_by_id[
                    resolved_ids["local_route_pack"]
                ].order
                < trace.task_by_id[
                    resolved_ids["local_route_plan"]
                ].order
                < trace.task_by_id[resolved_ids["expert_gate_up"]].order
                < trace.task_by_id[resolved_ids["expert_down"]].order
                < trace.task_by_id[
                    resolved_ids["local_combine_reduce"]
                ].order
                < trace.task_by_id[resolved_ids["moe_all_reduce"]].order
                < trace.task_by_id[resolved_ids["moe_residual_add"]].order
            )
            shared_before_combine = (
                trace.task_by_id[resolved_ids["shared_output"]].order
                < trace.task_by_id[
                    resolved_ids["local_combine_reduce"]
                ].order
            )
            gate_before_map = (
                trace.task_by_id[resolved_ids["gate_topk"]].order
                < trace.task_by_id[
                    resolved_ids["local_route_map_init"]
                ].order
            )
            gate_before_pack = (
                trace.task_by_id[resolved_ids["gate_topk"]].order
                < trace.task_by_id[
                    resolved_ids["local_route_pack"]
                ].order
            )
            norm_before_pack = (
                trace.task_by_id[resolved_ids["norm_quant"]].order
                < trace.task_by_id[
                    resolved_ids["local_route_pack"]
                ].order
            )
            producers_before_consumers = (
                gate_before_map
                and gate_before_pack
                and norm_before_pack
            )
            task_order = {
                "available": True,
                "pass": (
                    ordered
                    and shared_before_combine
                    and producers_before_consumers
                ),
                "local_owner_chain_ordered": ordered,
                "shared_down_before_combine": shared_before_combine,
                "gate_topk_before_map": gate_before_map,
                "gate_topk_before_pack": gate_before_pack,
                "norm_quant_before_pack": norm_before_pack,
                "orders": {
                    stage: trace.task_by_id[task_id].order
                    for stage, task_id in resolved_ids.items()
                },
            }
            if not task_order["pass"]:
                layer_errors.append(
                    {
                        "code": "task_order",
                        **task_order,
                    }
                )

        execution: dict[str, Any] = {}
        for stage in (
            "shared_output",
            *mandatory_swim_stages,
            *expert_stages,
        ):
            task_id = resolved_ids.get(stage)
            if task_id is None:
                continue
            has_slices = bool(trace.slices_by_task.get(task_id))
            predicated_skip = task_id in skipped_ids
            if (
                stage in mandatory_swim_stages
                or stage == "shared_output"
            ):
                passed = has_slices and not predicated_skip
                semantics = (
                    "This task must execute even when the rank owns zero "
                    "routed tokens."
                )
            else:
                passed = (
                    (has_slices and not predicated_skip)
                    or (not has_slices and predicated_skip)
                )
                semantics = (
                    "A local expert task may lack physical slices only with "
                    "an explicit predicated_skip scheduler event."
                )
            execution[stage] = {
                "pass": passed,
                "task_id": task_id,
                "task_count": 1,
                "has_physical_slices": has_slices,
                "predicated_skip": predicated_skip,
                "semantics": semantics,
            }
            if not passed:
                layer_errors.append(
                    {
                        "code": "physical_execution",
                        "stage": stage,
                        **execution[stage],
                    }
                )
        expected_block_nums = {
            "norm_quant": 2,
            "gate_topk": 1,
            "local_route_map_init": 1,
            "local_route_pack": 36,
            "local_route_plan": 1,
            "local_combine_reduce": 16,
        }
        for stage, expected_block_num in expected_block_nums.items():
            task_id = resolved_ids.get(stage)
            if task_id is None:
                continue
            block_num = trace.task_by_id[task_id].block_num
            block_num_pass = block_num == expected_block_num
            execution[stage].update(
                {
                    "pass": (
                        execution[stage]["pass"]
                        and block_num_pass
                    ),
                    "block_num": block_num,
                    "expected_block_num": expected_block_num,
                }
            )
            if not block_num_pass:
                layer_errors.append(
                    {
                        "code": "stage_block_num",
                        "stage": stage,
                        "expected": expected_block_num,
                        "actual": block_num,
                        "task_id": task_id,
                    }
                )

        required_edges: dict[str, Any] = {}
        local_swim_edges: dict[str, Any] = {}
        for (
            edge_name,
            pred_stage,
            succ_stage,
            allowed_sources,
            cardinality,
        ) in required_edge_specs:
            pred_ids = (
                [resolved_ids[pred_stage]]
                if pred_stage in resolved_ids
                else []
            )
            succ_ids = (
                [resolved_ids[succ_stage]]
                if succ_stage in resolved_ids
                else []
            )
            if not pred_ids or not succ_ids:
                required_edges[edge_name] = {
                    "pass": False,
                    "available": False,
                    "pred_stage": pred_stage,
                    "succ_stage": succ_stage,
                    "reason": "one or both stage task IDs are unavailable",
                }
                continue
            pair_edges = [
                edge
                for pred in pred_ids
                for succ in succ_ids
                for edge in _edges_between(trace, pred, succ)
            ]
            matching_edges = [
                edge
                for edge in pair_edges
                if str(edge.get("source")) in allowed_sources
            ]
            if cardinality == "exactly_one":
                edge_pass = len(matching_edges) == 1
            else:
                edge_pass = bool(matching_edges)
            required_edges[edge_name] = {
                "pass": edge_pass,
                "available": True,
                "pred_task_ids": pred_ids,
                "succ_task_ids": succ_ids,
                "pred_stage": pred_stage,
                "succ_stage": succ_stage,
                "allowed_sources": list(allowed_sources),
                "cardinality": cardinality,
                "matches": [
                    _edge_summary(edge)
                    for edge in matching_edges
                ],
                "all_pair_edges": [
                    _edge_summary(edge)
                    for edge in pair_edges
                ],
            }
            if not edge_pass:
                layer_errors.append(
                    {
                        "code": "required_edge",
                        "edge": edge_name,
                        **required_edges[edge_name],
                    }
                )

            pred_slices = [
                item
                for task_id in pred_ids
                for item in trace.slices_by_task.get(task_id, [])
            ]
            succ_slices = [
                item
                for task_id in succ_ids
                for item in trace.slices_by_task.get(task_id, [])
            ]
            if pred_slices and succ_slices:
                pred_end = max(item.end for item in pred_slices)
                succ_start = min(item.start for item in succ_slices)
                envelopes_overlap = pred_end > succ_start
                local_swim_edges[edge_name] = {
                    "available": True,
                    "pass": True,
                    "pred_end_tick": pred_end,
                    "succ_start_tick": succ_start,
                    "envelopes_overlap": envelopes_overlap,
                    "semantics": (
                        "Diagnostic only: allow_early_resolve permits safe "
                        "producer/consumer task-envelope overlap, so overlap "
                        "is not a structural failure."
                    ),
                }
            else:
                missing_stages = [
                    stage
                    for stage, stage_ids, slices in (
                        (pred_stage, pred_ids, pred_slices),
                        (succ_stage, succ_ids, succ_slices),
                    )
                    if not slices
                ]
                allowed_predicated_skip = all(
                    stage in expert_stages
                    and all(
                        task_id in skipped_ids
                        for task_id in stage_ids
                    )
                    for stage, stage_ids, slices in (
                        (pred_stage, pred_ids, pred_slices),
                        (succ_stage, succ_ids, succ_slices),
                    )
                    if not slices
                )
                local_swim_edges[edge_name] = {
                    "available": False,
                    "pass": allowed_predicated_skip,
                    "missing_stages": missing_stages,
                    "reason": (
                        "physical ordering is not observable for an explicitly "
                        "predicated local expert task"
                        if allowed_predicated_skip
                        else "one or both required task slices are missing"
                    ),
                }

        layers[layer] = {
            "pass": not layer_errors,
            "stage_task_ids": task_ids,
            "stage_task_counts": stage_counts,
            "task_order": task_order,
            "execution": execution,
            "required_edges": required_edges,
            "local_swim_order": {
                "pass": all(
                    edge["pass"]
                    for edge in local_swim_edges.values()
                ),
                "edges": local_swim_edges,
                "semantics": (
                    "Physical task envelopes are diagnostic only because "
                    "allow_early_resolve may expose safe overlap. Dependency "
                    "correctness is enforced by the task graph."
                ),
            },
            "errors": layer_errors,
        }
        all_errors.extend(
            {"layer": layer, **error}
            for error in layer_errors
        )
    return {
        "pass": not all_errors,
        "layers": layers,
        "errors": all_errors,
        "interpretation": (
            "Each layer must publish one norm/quant input and one complete "
            "gate-topk result into one route-map task, one 36-block local pack "
            "task, and one route-plan task before the owner-local expert "
            "tasks. Explicit scheduler dependencies and tensor-lineage edges "
            "must both preserve the frozen local-owner DAG through local "
            "combine, one TP all-reduce, and residual add. Only the two local "
            "expert tasks may retire through an explicit predicated_skip "
            "event on a zero-route rank."
        ),
    }


def _validate_structural_contracts(
    traces: list[RankTrace],
    profile: str = "candidate",
) -> dict[str, Any]:
    resolved_profile = _resolve_profile(profile)
    task_ids = {trace.tag: _task_id_contract(trace) for trace in traces}
    dependency_contract_name = (
        "local_ep_dependency"
        if resolved_profile == "local-ep"
        else "combine_dependency"
    )
    dependency_contracts = {
        trace.tag: (
            _local_ep_dependency_contract(trace)
            if resolved_profile == "local-ep"
            else _combine_dependency_contract(trace)
        )
        for trace in traces
    }
    errors = []
    for rank, contract in task_ids.items():
        if not contract["pass"]:
            errors.append(
                {
                    "rank": rank,
                    "contract": "task_id",
                    "missing_on_swim": contract["missing_on_swim"],
                    "unknown_on_swim": contract["unknown_on_swim"],
                    "duplicate_dep_task_ids": (contract["duplicate_dep_task_ids"]),
                    "invalid_physical_slices": (contract["invalid_physical_slices"][:8]),
                    "duplicate_predicated_skip_task_ids": (
                        contract["duplicate_predicated_skip_task_ids"]
                    ),
                    "unexpected_predicated_skip_task_ids": (
                        contract["unexpected_predicated_skip_task_ids"]
                    ),
                    "predicated_skip_with_physical_slices": (
                        contract["predicated_skip_with_physical_slices"]
                    ),
                }
            )
    for rank, contract in dependency_contracts.items():
        if not contract["pass"]:
            errors.append(
                {
                    "rank": rank,
                    "contract": dependency_contract_name,
                    "errors": contract["errors"],
                }
            )
    if errors:
        raise RuntimeError(
            "task-level structural DFX contract failed: " + json.dumps(errors[:8], sort_keys=True)
        )
    result = {
        "pass": True,
        "task_id": task_ids,
        dependency_contract_name: dependency_contracts,
    }
    if resolved_profile != "local-ep":
        result["combine_dependency"] = dependency_contracts
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
    span = max(end for _start, end in intervals) - min(start for start, _end in intervals)
    busy = sum(end - start for start, end in intervals)
    return peak, busy / span if span else 0.0


def _resource_metrics(
    trace: RankTrace,
    tasks: list[Task],
    resource: str,
) -> dict[str, Any]:
    slices = sorted(
        [
            item
            for task in tasks
            for item in trace.slices_by_task.get(task.task_id, [])
            if item.resource == resource
        ],
        key=lambda item: (
            item.start,
            item.end,
            item.core,
            item.task_id,
        ),
    )
    available_cores = sum(kind == resource for kind in trace.core_types)
    expected_slices = sum(task.block_num * task.resource_slices_per_block(resource) for task in tasks)
    durations = [(item.end - item.start) / trace.frequency_hz * 1e6 for item in slices]
    duration_distribution = _duration_distribution(durations)
    if not slices:
        return {
            "available": False,
            "reason": "no physical slices for this resource",
            "available_cores": available_cores,
            "expected_slices": expected_slices,
            "observed_slices": 0,
            "distinct_cores": 0,
            "waves_at_full_resource": (
                math.ceil(expected_slices / available_cores) if expected_slices and available_cores else 0
            ),
            "slice_duration_us_min": None,
            "slice_duration_us_p50": None,
            "slice_duration_us_p90": None,
            "slice_duration_us_p99": None,
            "slice_duration_us_max": None,
            "duration_distribution": duration_distribution,
            "physical_slices": [],
            "span_us": None,
            "busy_us": 0.0,
            "peak_concurrency": 0,
            "average_concurrency": 0.0,
            "unused_cores_at_peak": available_cores,
            "full_resource_utilization": 0.0,
            "packing_efficiency": 0.0,
        }
    frequency = trace.frequency_hz
    intervals = [(item.start, item.end) for item in slices]
    start = min(item.start for item in slices)
    end = max(item.end for item in slices)
    span_ticks = end - start
    busy_ticks = sum(item.end - item.start for item in slices)
    peak, average = _concurrency(intervals)
    schedulable = min(available_cores, expected_slices)
    return {
        "available": True,
        "available_cores": available_cores,
        "expected_slices": expected_slices,
        "observed_slices": len(slices),
        "distinct_cores": len({item.core for item in slices}),
        "waves_at_full_resource": (
            math.ceil(expected_slices / available_cores) if expected_slices and available_cores else 0
        ),
        "slice_duration_us_min": duration_distribution["min_us"],
        "slice_duration_us_p50": duration_distribution["p50_us"],
        "slice_duration_us_p90": duration_distribution["p90_us"],
        "slice_duration_us_p99": duration_distribution["p99_us"],
        "slice_duration_us_max": duration_distribution["max_us"],
        "duration_distribution": duration_distribution,
        "physical_slices": [_slice_record(trace, item) for item in slices],
        "span_us": _round(span_ticks / frequency * 1e6),
        "busy_us": _round(busy_ticks / frequency * 1e6),
        "peak_concurrency": peak,
        "average_concurrency": _round(average),
        "unused_cores_at_peak": max(0, available_cores - peak),
        "full_resource_utilization": (
            _round(busy_ticks / (span_ticks * available_cores), 4) if span_ticks and available_cores else 0.0
        ),
        "packing_efficiency": (_round(average / schedulable, 4) if schedulable else 0.0),
    }


def _task_timing_evidence(
    trace: RankTrace,
    task: Task,
) -> dict[str, Any]:
    """Report only timing fields supported by explicit local-rank evidence."""
    slices = trace.slices_by_task.get(task.task_id, [])
    if not slices:
        reason = "task has no physical swim slices"
        return {
            "queue_delay": {
                "available": False,
                "reason": reason,
            },
            "service_span": {
                "available": False,
                "reason": reason,
            },
            "dag_span": {
                "available": False,
                "reason": reason,
            },
            "critical_path_contribution": {
                "available": False,
                "reason": _CRITICAL_PATH_CONTRIBUTION_REASON,
            },
        }

    task_start = min(item.start for item in slices)
    task_end = max(item.end for item in slices)
    service_durations = [(item.end - item.start) / trace.frequency_hz * 1e6 for item in slices]
    predecessor_edges = [edge for edge in trace.edges if str(edge.get("succ")) == task.task_id]
    predecessor_ids = sorted({str(edge.get("pred")) for edge in predecessor_edges})
    predecessor_sources = {
        predecessor_id: sorted(
            {
                str(edge.get("source", "unknown"))
                for edge in predecessor_edges
                if str(edge.get("pred")) == predecessor_id
            }
        )
        for predecessor_id in predecessor_ids
    }
    unknown_predecessors = [
        predecessor_id for predecessor_id in predecessor_ids if predecessor_id not in trace.task_by_id
    ]
    missing_predecessor_timing = [
        predecessor_id for predecessor_id in predecessor_ids if not trace.slices_by_task.get(predecessor_id)
    ]
    predecessor_end_ticks = {
        predecessor_id: max(item.end for item in trace.slices_by_task[predecessor_id])
        for predecessor_id in predecessor_ids
        if trace.slices_by_task.get(predecessor_id)
    }
    overlapping_predecessors = [
        predecessor_id for predecessor_id, end_tick in predecessor_end_ticks.items() if end_tick > task_start
    ]
    queue_base = {
        "task_start_tick": task_start,
        "predecessor_task_ids": predecessor_ids,
        "predecessor_sources": predecessor_sources,
    }
    if not predecessor_ids:
        queue_delay: dict[str, Any] = {
            "available": False,
            "reason": ("task has no dependency predecessor; dependency-ready time is not defined"),
            **queue_base,
        }
    elif unknown_predecessors:
        queue_delay = {
            "available": False,
            "reason": "dependency graph contains unknown predecessor task IDs",
            "unknown_predecessor_task_ids": unknown_predecessors,
            **queue_base,
        }
    elif missing_predecessor_timing:
        queue_delay = {
            "available": False,
            "reason": ("one or more dependency predecessors have no physical timing"),
            "missing_predecessor_timing": missing_predecessor_timing,
            **queue_base,
        }
    elif overlapping_predecessors:
        queue_delay = {
            "available": False,
            "reason": (
                "aggregate predecessor execution overlaps task start; "
                "block-level dependency-ready time cannot be reconstructed "
                "from task envelopes"
            ),
            "overlapping_predecessor_task_ids": overlapping_predecessors,
            "predecessor_end_ticks": predecessor_end_ticks,
            **queue_base,
        }
    else:
        ready_tick = max(predecessor_end_ticks.values())
        queue_delay = {
            "available": True,
            "value_us": _round((task_start - ready_tick) / trace.frequency_hz * 1e6),
            "dependency_ready_tick": ready_tick,
            "predecessor_end_ticks": predecessor_end_ticks,
            "semantics": (
                "Local scheduler delay from the latest fully observed direct "
                "dependency completion to the first physical task slice."
            ),
            **queue_base,
        }
    return {
        "queue_delay": queue_delay,
        "service_span": {
            "available": True,
            "distribution_us": _duration_distribution(service_durations),
            "total_busy_us": _round(sum(service_durations)),
            "semantics": (
                "Per-physical-slice end-start service intervals. This is not the logical task wall span."
            ),
        },
        "dag_span": {
            "available": True,
            "value_us": _round((task_end - task_start) / trace.frequency_hz * 1e6),
            "start_tick": task_start,
            "end_tick": task_end,
            "semantics": (
                "Logical task envelope from its earliest physical slice start to latest physical slice end."
            ),
        },
        "critical_path_contribution": {
            "available": False,
            "reason": _CRITICAL_PATH_CONTRIBUTION_REASON,
        },
    }


def _stage_metrics(
    trace: RankTrace,
    task_ids: list[str],
) -> dict[str, Any] | None:
    tasks = [trace.task_by_id[task_id] for task_id in task_ids if task_id in trace.task_by_id]
    all_slices = [item for task in tasks for item in trace.slices_by_task.get(task.task_id, [])]
    if not tasks or not all_slices:
        return None
    frequency = trace.frequency_hz
    task_spans = []
    for task in tasks:
        slices = trace.slices_by_task.get(task.task_id, [])
        if slices:
            task_spans.append(
                (max(item.end for item in slices) - min(item.start for item in slices)) / frequency * 1e6,
            )
    start = min(item.start for item in all_slices)
    end = max(item.end for item in all_slices)
    task_details = []
    for task in tasks:
        task_slices = trace.slices_by_task.get(task.task_id, [])
        detail: dict[str, Any] = {
            "task_id": task.task_id,
            "name": task.name,
            "order": task.order,
            "block_num": task.block_num,
            "early_dispatch": task.early_dispatch,
            "resources": {},
            "timing_evidence": _task_timing_evidence(trace, task),
        }
        if task_slices:
            task_start = min(item.start for item in task_slices)
            task_end = max(item.end for item in task_slices)
            detail.update(
                {
                    "start_tick": task_start,
                    "end_tick": task_end,
                    "dag_task_span_us": _round((task_end - task_start) / frequency * 1e6),
                }
            )
        for resource in ("aic", "aiv"):
            resource_slices = [item for item in task_slices if item.resource == resource]
            durations = [(item.end - item.start) / frequency * 1e6 for item in resource_slices]
            duration_distribution = _duration_distribution(durations)
            detail["resources"][resource] = {
                "available": bool(resource_slices),
                "reason": (None if resource_slices else "no physical slices for this resource"),
                "expected_slices": (task.block_num * task.resource_slices_per_block(resource)),
                "observed_slices": len(resource_slices),
                "core_ids": sorted({item.core for item in resource_slices}),
                "slice_duration_us": [_round(value) for value in durations],
                "slice_duration_us_p50": (duration_distribution["p50_us"]),
                "slice_duration_us_p90": (duration_distribution["p90_us"]),
                "slice_duration_us_p99": (duration_distribution["p99_us"]),
                "slice_duration_us_max": (duration_distribution["max_us"]),
                "duration_distribution": duration_distribution,
            }
        task_details.append(detail)
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
        "task_instance_details": task_details,
        "timing_semantics": {
            "queue_delay": (
                "Only emitted when every direct dependency predecessor has "
                "complete local timing and all predecessor task envelopes "
                "finish before this task starts."
            ),
            "service_span": (
                "One physical AIC/AIV slice end-start interval. The requested "
                "10-30 us grain applies to this field."
            ),
            "dag_span": (
                "Wall envelope from the earliest to latest physical slice of "
                "one dependency-graph task instance."
            ),
            "critical_path_contribution": (
                "Unavailable unless a future structured critical-path "
                "sidecar exports a numeric contribution keyed by task ID."
            ),
        },
        "early_dispatch": all(task.early_dispatch for task in tasks),
        "resources": {},
    }
    for resource in ("aic", "aiv"):
        metrics = _resource_metrics(trace, tasks, resource)
        result["resources"][resource] = metrics
    return result


def _rank_metrics(
    trace: RankTrace,
    profile: str = "candidate",
) -> dict[str, Any]:
    all_slices = trace.all_slices
    if not all_slices:
        return {}
    frequency = trace.frequency_hz
    start = min(item.start for item in all_slices)
    end = max(item.end for item in all_slices)
    layers: dict[str, Any] = {}
    for layer in _LAYER_PREFIX:
        stage_ids = _find_layer_task_ids(trace, layer, profile)
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
                        (max(item.end for item in slices) - min(item.start for item in slices))
                        / frequency
                        * 1e6,
                    ),
                },
            )
    return {
        "hardware_capacity": {
            "source": f"raw {_CHIP_SWIMLANE_RECORDS_NAME} metadata.core_types",
            "swimlane_level": trace.swimlane_level,
            "num_cores": len(trace.core_types),
            "aic": trace.core_types.count("aic"),
            "aiv": trace.core_types.count("aiv"),
        },
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
    terminal_ticks = {trace.tag: max((item.end for item in trace.all_slices), default=0) for trace in traces}
    same_frequency = len(frequencies) == 1 and bool(frequencies)
    return {
        "frequencies_hz": frequencies,
        "same_frequency": same_frequency,
        "terminal_end_ticks": terminal_ticks,
        "terminal_end_skew_us": None,
        "terminal_end_skew_computed": False,
        "cross_rank_tick_math_enabled": False,
        "external_common_clock_anchor": None,
        "evidence_level": "per-rank normalized",
        "interpretation": (
            "The runtime reader normalizes every rank to a local origin. "
            "No cross-rank timestamp subtraction, ordering, skew, or remote "
            "arrival latency is computed without an external common-clock "
            "anchor."
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _nested_shape(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    if not value:
        return (0,)
    child_shapes = [_nested_shape(child, field) for child in value]
    if any(shape != child_shapes[0] for shape in child_shapes[1:]):
        raise ValueError(f"{field}: ragged nested array")
    return (len(value), *child_shapes[0])


def _as_nested_list(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_as_nested_list(child) for child in value]
    return value


def _nested_values(value: Any, field: str) -> tuple[tuple[int, ...], Any, str]:
    if hasattr(value, "detach") and hasattr(value, "tolist"):
        tensor = value.detach().cpu()
        return (
            tuple(int(size) for size in tensor.shape),
            tensor.tolist(),
            str(tensor.dtype),
        )
    if hasattr(value, "shape") and hasattr(value, "tolist"):
        return (
            tuple(int(size) for size in value.shape),
            value.tolist(),
            str(getattr(value, "dtype", "array")),
        )
    if isinstance(value, (list, tuple)):
        return (
            _nested_shape(value, field),
            _as_nested_list(value),
            "json-integer",
        )
    raise ValueError(f"{field}: expected a tensor or nested integer array")


def _flatten_nested(value: Any):
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _flatten_nested(child)
        return
    yield value


def _require_nonnegative_integers(value: Any, field: str) -> None:
    invalid = [
        item
        for item in _flatten_nested(value)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0
    ]
    if invalid:
        raise ValueError(
            f"{field}: expected nonnegative integer values, got {invalid[:8]}"
        )


def _load_recv_meta_payload(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        try:
            import torch  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                f"{path}: torch is required to load a non-JSON recv_meta sidecar"
            ) from exc
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: recv_meta sidecar root must be a mapping")
    return payload


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{field}: expected lowercase SHA256")
    return value


def _validated_sidecar_provenance(
    payload: dict[str, Any],
    *,
    expected_protocol_profile: str | None = None,
) -> dict[str, Any]:
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("recv_meta sidecar provenance must be a mapping")
    image_digest = provenance.get("image_digest")
    if not (
        isinstance(image_digest, str)
        and _IMAGE_DIGEST_PATTERN.fullmatch(image_digest)
    ):
        raise ValueError("recv_meta provenance image_digest is invalid")

    checkpoint = provenance.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("recv_meta provenance checkpoint is missing")
    if checkpoint.get("schema") != _CHECKPOINT_SCHEMA:
        raise ValueError("recv_meta checkpoint schema is invalid")
    logical_id = checkpoint.get("logical_id")
    if (
        not isinstance(logical_id, str)
        or not logical_id
        or "/" in logical_id
        or "\\" in logical_id
    ):
        raise ValueError("recv_meta checkpoint logical_id is invalid")
    identity_sha256 = _require_sha256(
        checkpoint.get("identity_sha256"),
        "recv_meta checkpoint identity_sha256",
    )
    index_file = checkpoint.get("index_file")
    if (
        not isinstance(index_file, str)
        or not index_file.endswith(".safetensors.index.json")
    ):
        raise ValueError("recv_meta checkpoint index_file is invalid")
    files = checkpoint.get("files")
    if not isinstance(files, dict) or len(files) < 3:
        raise ValueError(
            "recv_meta checkpoint identity must cover config, index, and shards"
        )
    if index_file not in files or "config.json" not in files:
        raise ValueError("recv_meta checkpoint metadata files are missing")
    for name, record in files.items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
        ):
            raise ValueError("recv_meta checkpoint file name is invalid")
        if not isinstance(record, dict):
            raise ValueError("recv_meta checkpoint file record is invalid")
        _require_sha256(
            record.get("sha256"),
            f"recv_meta checkpoint files.{name}.sha256",
        )
        if type(record.get("size_bytes")) is not int or (
            record["size_bytes"] <= 0
        ):
            raise ValueError("recv_meta checkpoint file size is invalid")
    if _json_sha256(files) != identity_sha256:
        raise ValueError("recv_meta checkpoint identity digest mismatch")
    for field in ("weight_tensor_count", "weight_shard_count"):
        if type(checkpoint.get(field)) is not int or checkpoint[field] <= 0:
            raise ValueError(f"recv_meta checkpoint {field} is invalid")
    if len(files) != checkpoint["weight_shard_count"] + 2:
        raise ValueError("recv_meta checkpoint shard count is inconsistent")
    authority_manifest_sha256 = checkpoint.get(
        "authority_manifest_sha256"
    )
    if authority_manifest_sha256 is not None:
        _require_sha256(
            authority_manifest_sha256,
            "recv_meta checkpoint authority_manifest_sha256",
        )

    source = provenance.get("source")
    if not isinstance(source, dict):
        raise ValueError("recv_meta source provenance is missing")
    for field in (
        "source_tree_manifest_sha256",
        "decode_fwd_sha256",
        "formal_program_sha256",
        "route_program_sha256",
        "route_holder_sha256",
        "route_stage_sha256",
    ):
        _require_sha256(source.get(field), f"recv_meta source.{field}")
    source_manifest_sha256 = _require_sha256(
        provenance.get("source_manifest_sha256"),
        "recv_meta source_manifest_sha256",
    )
    if _json_sha256(source) != source_manifest_sha256:
        raise ValueError("recv_meta source manifest digest mismatch")

    input_contract = provenance.get("input_contract")
    if not isinstance(input_contract, dict):
        raise ValueError("recv_meta input_contract is missing")
    workload = input_contract.get("workload")
    if not isinstance(workload, dict):
        raise ValueError("recv_meta workload is missing")
    active_batch = workload.get("active_batch")
    if type(active_batch) is not int or not 1 <= active_batch <= 16:
        raise ValueError("recv_meta active_batch is invalid")
    if workload.get("context_len") != 65536:
        raise ValueError("recv_meta context_len must be 65536 per sequence")
    if workload.get("context_semantics") != "per_active_sequence":
        raise ValueError("recv_meta context semantics are invalid")
    input_tokens = input_contract.get("input_tokens")
    if not (
        isinstance(input_tokens, list)
        and len(input_tokens) == active_batch
        and all(type(token) is int and token >= 0 for token in input_tokens)
    ):
        raise ValueError("recv_meta input token contract is invalid")
    tensor_sha256 = input_contract.get("tensor_sha256")
    if not isinstance(tensor_sha256, dict) or set(tensor_sha256) != {
        "active_hidden",
        "seq_lens",
        "positions",
        "block_table",
        "slot_mapping",
    }:
        raise ValueError("recv_meta input tensor provenance is incomplete")
    for field, digest in tensor_sha256.items():
        _require_sha256(digest, f"recv_meta input tensor.{field}")
    input_contract_sha256 = _require_sha256(
        provenance.get("input_contract_sha256"),
        "recv_meta input_contract_sha256",
    )
    if _json_sha256(input_contract) != input_contract_sha256:
        raise ValueError("recv_meta input contract digest mismatch")

    golden = provenance.get("formal_golden")
    if not isinstance(golden, dict):
        raise ValueError("recv_meta formal_golden provenance is missing")
    if golden.get("schema") != _GOLDEN_SCHEMA:
        raise ValueError("recv_meta formal_golden semantic contract is invalid")
    try:
        protocol = normalize_golden_protocol(
            golden,
            field="recv_meta formal_golden",
        )
    except ValueError as exc:
        raise ValueError(
            "recv_meta formal_golden semantic contract is invalid: "
            f"{exc}"
        ) from exc
    if (
        golden.get("active_batch") != active_batch
        or golden.get("context_len_per_sequence") != 65536
        or golden.get("image_ref") != image_digest
    ):
        raise ValueError("recv_meta formal_golden semantic contract is invalid")
    if (
        expected_protocol_profile is not None
        and protocol["protocol_profile"] != expected_protocol_profile
    ):
        raise ValueError(
            "recv_meta formal_golden protocol_profile="
            f"{protocol['protocol_profile']!r} does not match expected "
            f"{expected_protocol_profile!r}"
        )
    source_run = golden.get("source_run")
    if not isinstance(source_run, str) or not source_run:
        raise ValueError("recv_meta formal_golden source_run is missing")
    for field in (
        "manifest_sha256",
        "source_decode_fwd_sha256",
        "source_manifest_sha256",
    ):
        _require_sha256(golden.get(field), f"recv_meta formal_golden.{field}")
    if golden["source_decode_fwd_sha256"] != source["decode_fwd_sha256"]:
        raise ValueError(
            "recv_meta formal_golden decode SHA does not match route source"
        )
    golden_files = golden.get("files")
    if not isinstance(golden_files, dict) or set(golden_files) != {
        "hidden_l3.pt",
        "hidden_l4.pt",
    }:
        raise ValueError("recv_meta formal_golden files are incomplete")
    for name, digest in golden_files.items():
        _require_sha256(digest, f"recv_meta formal_golden files.{name}")
    return {
        "image_digest": image_digest,
        "checkpoint_identity_sha256": identity_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "decode_fwd_sha256": source["decode_fwd_sha256"],
        "source": dict(source),
        "input_contract_sha256": input_contract_sha256,
        "golden_manifest_sha256": golden["manifest_sha256"],
        "source_kind": protocol["source_kind"],
        "protocol_profile": protocol["protocol_profile"],
        "numeric_contract": dict(protocol["numeric_contract"]),
        "active_batch": active_batch,
        "context_len_per_sequence": 65536,
    }


def _validated_route_histogram(
    path: Path,
    *,
    profile: str | None = None,
    source_decode_sha256: str | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"recv_meta sidecar does not exist: {path}")
    payload = _load_recv_meta_payload(path)
    if payload.get("schema") != _RECV_META_SIDECAR_SCHEMA:
        raise ValueError(
            f"{path}: expected schema={_RECV_META_SIDECAR_SCHEMA}, "
            f"got {payload.get('schema')!r}"
        )
    if tuple(payload.get("layers", ())) != _RECV_META_LAYERS:
        raise ValueError(
            f"{path}: expected layers={list(_RECV_META_LAYERS)}, "
            f"got {payload.get('layers')!r}"
        )
    if tuple(payload.get("axes", ())) != _RECV_META_AXES:
        raise ValueError(
            f"{path}: expected axes={list(_RECV_META_AXES)}, "
            f"got {payload.get('axes')!r}"
        )
    policy = _source_policy(profile) if profile is not None else None
    expected_protocol_profile = (
        policy.get("golden_protocol_profile")
        if policy is not None
        else None
    )
    if (
        expected_protocol_profile is not None
        and not isinstance(expected_protocol_profile, str)
    ):
        raise ValueError(
            f"{path}: source policy golden_protocol_profile is invalid"
        )
    provenance = _validated_sidecar_provenance(
        payload,
        expected_protocol_profile=expected_protocol_profile,
    )
    if policy is not None:
        policy_id = str(policy["policy_id"])
        expected_prefix = str(policy["decode_sha256_prefix"])
        actual_decode = str(provenance["decode_fwd_sha256"])
        expected_decode = policy.get("decode_sha256")
        if not isinstance(expected_decode, str):
            raise ValueError(
                f"{path}: source policy {policy_id} has no exact decode SHA"
            )
        if actual_decode != expected_decode:
            raise ValueError(
                f"{path}: recv_meta decode_fwd_sha256={actual_decode} does "
                f"not match source policy {policy_id} "
                f"expected={expected_decode}"
            )
        if (
            source_decode_sha256 is not None
            and actual_decode != source_decode_sha256
        ):
            raise ValueError(
                f"{path}: recv_meta decode_fwd_sha256={actual_decode} does "
                f"not match live source_decode_sha256={source_decode_sha256}"
            )
        provenance = {
            **provenance,
            "source_policy_id": policy_id,
            "source_policy_decode_sha256_prefix": expected_prefix,
            "source_policy_decode_sha256": expected_decode,
        }

    recv_shape, recv_meta, recv_dtype = _nested_values(
        payload.get("owner_route_counts"),
        "owner_route_counts",
    )
    count_shape, local_expert_count, count_dtype = _nested_values(
        payload.get("local_expert_count"),
        "local_expert_count",
    )
    if recv_shape != _RECV_META_SHAPE:
        raise ValueError(
            f"{path}: owner_route_counts shape={recv_shape}, "
            f"expected={_RECV_META_SHAPE}"
        )
    if count_shape != _LOCAL_EXPERT_COUNT_SHAPE:
        raise ValueError(
            f"{path}: local_expert_count shape={count_shape}, "
            f"expected={_LOCAL_EXPERT_COUNT_SHAPE}"
        )
    if recv_dtype != "json-integer" and not recv_dtype.lower().endswith("int32"):
        raise ValueError(
            f"{path}: owner_route_counts dtype must be int32, "
            f"got {recv_dtype}"
        )
    if count_dtype != "json-integer" and not count_dtype.lower().endswith("int32"):
        raise ValueError(
            f"{path}: local_expert_count dtype must be int32, got {count_dtype}"
        )
    _require_nonnegative_integers(recv_meta, "owner_route_counts")
    _require_nonnegative_integers(local_expert_count, "local_expert_count")

    window_records = payload.get("snapshot_provenance")
    if not isinstance(window_records, list) or len(window_records) != 2:
        raise ValueError(
            f"{path}: snapshot_provenance must contain exactly L3 and L4 "
            "records"
        )
    windows_by_layer: dict[str, dict[str, Any]] = {}
    for record in window_records:
        if not isinstance(record, dict):
            raise ValueError(
                f"{path}: snapshot_provenance records must be mappings"
            )
        layer = str(record.get("layer"))
        if layer in windows_by_layer:
            raise ValueError(f"{path}: duplicate window provenance for {layer}")
        shape = tuple(record.get("shape", ()))
        window_id = record.get("snapshot_id")
        if layer not in _RECV_META_LAYERS:
            raise ValueError(f"{path}: unexpected window layer {layer!r}")
        if shape != _RECV_META_WINDOW_SHAPE:
            raise ValueError(
                f"{path}: {layer} snapshot shape={shape}, "
                f"expected={_RECV_META_WINDOW_SHAPE}"
            )
        if str(record.get("dtype")).lower() != "int32":
            raise ValueError(f"{path}: {layer} snapshot dtype must be int32")
        if int(record.get("byte_size", -1)) != _RECV_META_WINDOW_BYTES:
            raise ValueError(
                f"{path}: {layer} snapshot byte_size must be "
                f"{_RECV_META_WINDOW_BYTES}"
            )
        if not isinstance(window_id, (str, int)) or str(window_id) == "":
            raise ValueError(f"{path}: {layer} snapshot_id is missing")
        if record.get("source_tensor") != "local_expert_count":
            raise ValueError(
                f"{path}: {layer} source_tensor is invalid"
            )
        if (
            record.get("source_protocol")
            != "replicated_input_local_owner"
        ):
            raise ValueError(f"{path}: {layer} source_protocol is invalid")
        expected_capture = {
            "L3": "after_l3_before_l4",
            "L4": "after_l4",
        }[layer]
        if record.get("capture_point") != expected_capture:
            raise ValueError(f"{path}: {layer} capture_point is invalid")
        windows_by_layer[layer] = {
            "layer": layer,
            "snapshot_id": str(window_id),
            "shape": list(shape),
            "dtype": "int32",
            "byte_size": _RECV_META_WINDOW_BYTES,
            "source_tensor": "local_expert_count",
            "source_protocol": "replicated_input_local_owner",
            "capture_point": expected_capture,
        }
    if set(windows_by_layer) != set(_RECV_META_LAYERS):
        raise ValueError(
            f"{path}: window layers={sorted(windows_by_layer)}, "
            f"expected={list(_RECV_META_LAYERS)}"
        )
    window_ids = [
        windows_by_layer[layer]["snapshot_id"]
        for layer in _RECV_META_LAYERS
    ]
    if len(set(window_ids)) != len(window_ids):
        raise ValueError(
            f"{path}: L3/L4 must use distinct route snapshot IDs"
        )

    padding_errors = []
    derived_counts = []
    count_mismatches = []
    off_owner_errors = []
    per_layer_per_owner: list[list[int]] = []
    for layer_index, layer in enumerate(_RECV_META_LAYERS):
        layer_counts = []
        for owner_rank in range(_EXPECTED_RANKS):
            for route_owner_rank in range(_EXPECTED_RANKS):
                row = recv_meta[layer_index][owner_rank][route_owner_rank]
                padding = row[36:]
                if any(value != 0 for value in padding):
                    padding_errors.append(
                        {
                            "layer": layer,
                            "owner_rank": owner_rank,
                            "route_owner_rank": route_owner_rank,
                            "padding": padding,
                        }
                    )
                if (
                    route_owner_rank != owner_rank
                    and any(value != 0 for value in row[:36])
                ):
                    off_owner_errors.append(
                        {
                            "layer": layer,
                            "owner_rank": owner_rank,
                            "route_owner_rank": route_owner_rank,
                        }
                    )
            rank_counts = list(
                recv_meta[layer_index][owner_rank][owner_rank][:36]
            )
            layer_counts.append(rank_counts)
            expected = local_expert_count[layer_index][owner_rank]
            if rank_counts != expected:
                count_mismatches.append(
                    {
                        "layer": layer,
                        "owner_rank": owner_rank,
                        "derived": rank_counts,
                        "exported": expected,
                    }
                )
        derived_counts.append(layer_counts)
        per_layer_per_owner.append(
            [sum(rank_counts) for rank_counts in layer_counts]
        )
    if padding_errors:
        raise ValueError(
            f"{path}: owner_route_counts padding columns 36:40 must be zero; "
            f"errors={padding_errors[:8]}"
        )
    if off_owner_errors:
        raise ValueError(
            f"{path}: local-owner route rows must be diagonal; "
            f"errors={off_owner_errors[:8]}"
        )
    if count_mismatches:
        raise ValueError(
            f"{path}: local_expert_count != diagonal owner route row; "
            f"errors={count_mismatches[:8]}"
        )

    expected_global = provenance["active_batch"] * _MOE_TOPK
    global_per_layer = [
        sum(owner_totals) for owner_totals in per_layer_per_owner
    ]
    global_total_mismatches = [
        {
            "layer": layer,
            "actual": total,
            "expected": expected_global,
        }
        for layer, total in zip(_RECV_META_LAYERS, global_per_layer)
        if total != expected_global
    ]
    if global_total_mismatches:
        raise ValueError(
            f"{path}: route totals invalid; "
            "global per layer must equal active_batch * TOPK="
            f"{expected_global}, "
            f"mismatches={global_total_mismatches[:8]}"
        )

    source_sha256 = _sha256(path)
    result = {}
    for layer_index, layer in enumerate(_RECV_META_LAYERS):
        histogram = {
            f"rank{rank}/d0": list(derived_counts[layer_index][rank])
            for rank in range(_EXPECTED_RANKS)
        }
        totals = {
            rank: sum(counts) for rank, counts in histogram.items()
        }
        result[layer] = {
            "available": True,
            "blocking": False,
            "release_gate": False,
            "publication_evidence_required": True,
            "publication_evidence_ready": True,
            "histogram": histogram,
            "histogram_semantics": (
                "List index is local expert ID 0..35; each value is the exact "
                "route count computed by that rank's local expert owner."
            ),
            "total_routed_tokens_by_rank": totals,
            "route_totals_validated": True,
            "per_layer_per_owner": per_layer_per_owner[layer_index],
            "global_per_layer": global_per_layer,
            "expected_global_per_layer": expected_global,
            "owner_rows_diagonal": True,
            "zero_route_ranks": [
                rank for rank, total in totals.items() if total == 0
            ],
            "max_min_total_skew": max(totals.values()) - min(totals.values()),
            "source": path.name,
            "source_sha256": source_sha256,
            "source_schema": _RECV_META_SIDECAR_SCHEMA,
            "source_axes": list(_RECV_META_AXES),
            "owner_route_counts_shape": list(_RECV_META_SHAPE),
            "owner_route_counts_dtype": recv_dtype,
            "local_expert_count_shape": list(_LOCAL_EXPERT_COUNT_SHAPE),
            "local_expert_count_dtype": count_dtype,
            "snapshot_provenance": windows_by_layer[layer],
            "snapshot_independence_validated": True,
            "provenance": provenance,
            "expected_sidecar": "local_owner_route_counts",
            "reason": (
                "validated exact local-owner route-count histogram sidecar"
            ),
            "required_input": None,
            "proxy_fallback_allowed": False,
            "rejected_proxies": [
                "routed dependency task count",
                "logical block count",
                "physical AIC/AIV slice count",
                "gate/up/down tile count",
            ],
        }
    return result


def _route_histogram_contract(
    recv_meta_sidecar: Path | None = None,
    *,
    profile: str | None = None,
    source_decode_sha256: str | None = None,
) -> dict[str, Any]:
    """Load exact owner-count histograms or declare an analyzer limitation.

    Missing evidence never fails the analyzer gate, but it keeps candidate
    publication readiness in NOT_EVALUABLE state.
    """
    if recv_meta_sidecar is not None:
        return _validated_route_histogram(
            Path(recv_meta_sidecar),
            profile=profile,
            source_decode_sha256=source_decode_sha256,
        )
    return {
        layer: {
            "available": False,
            "blocking": False,
            "release_gate": False,
            "publication_evidence_required": True,
            "publication_evidence_ready": False,
            "histogram": None,
            "source": None,
            "expected_sidecar": "local_owner_route_counts",
            "expected_schema": _RECV_META_SIDECAR_SCHEMA,
            "expected_shape": list(_RECV_META_SHAPE),
            "reason": _ROUTE_HISTOGRAM_REASON,
            "required_input": (
                "Provide a local-owner route sidecar containing exact "
                "[2,8,8,40] diagonal owner counts, [2,8,36] "
                "local_expert_count, and distinct L3/L4 snapshot provenance."
            ),
            "proxy_fallback_allowed": False,
            "rejected_proxies": [
                "routed dependency task count",
                "logical block count",
                "physical AIC/AIV slice count",
                "gate/up/down tile count",
            ],
        }
        for layer in _LAYER_PREFIX
    }


def _local_ep_route_execution_contract(
    structural_contracts: dict[str, Any],
    route_histogram: dict[str, Any],
    profile: str = "candidate",
) -> dict[str, Any]:
    """Cross-check exact local route totals against expert execution."""
    if _resolve_profile(profile) != "local-ep":
        return {
            "applicable": False,
            "available": False,
            "pass": True,
            "layers": {},
            "errors": [],
            "reason": "selected profile does not use local-owner EP",
        }

    dependency_by_rank = structural_contracts.get(
        "local_ep_dependency",
        {},
    )
    unavailable_layers = [
        layer
        for layer in _LAYER_PREFIX
        if not route_histogram.get(layer, {}).get("available")
    ]
    if unavailable_layers:
        return {
            "applicable": True,
            "available": False,
            "pass": None,
            "layers": {
                layer: {
                    "available": False,
                    "pass": None,
                    "reason": (
                        route_histogram.get(layer, {}).get(
                            "reason",
                            "exact local route totals are unavailable",
                        )
                    ),
                }
                for layer in unavailable_layers
            },
            "errors": [],
            "reason": (
                "Exact local-owner route-count histogram evidence is required "
                "to correlate route emptiness with expert execution."
            ),
        }

    mandatory_stages = (
        "norm_quant",
        "gate_topk",
        "local_route_map_init",
        "local_route_pack",
        "local_route_plan",
        "local_combine_reduce",
        "moe_all_reduce",
        "moe_residual_add",
    )
    expert_stages = ("expert_gate_up", "expert_down")
    layers: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []
    for layer in _LAYER_PREFIX:
        histogram = route_histogram[layer]["histogram"]
        rank_results: dict[str, Any] = {}
        layer_errors: list[dict[str, Any]] = []
        for rank, local_counts in histogram.items():
            dependency = dependency_by_rank.get(rank)
            layer_dependency = (
                dependency.get("layers", {}).get(layer)
                if isinstance(dependency, dict)
                else None
            )
            if not isinstance(layer_dependency, dict):
                error = {
                    "rank": rank,
                    "layer": layer,
                    "code": "missing_structural_execution",
                }
                layer_errors.append(error)
                rank_results[rank] = {
                    "available": False,
                    "pass": False,
                    "errors": [error],
                }
                continue

            route_total = sum(int(value) for value in local_counts)
            expect_skip = route_total == 0
            execution = layer_dependency.get("execution", {})
            rank_errors: list[dict[str, Any]] = []
            expert_evidence: dict[str, Any] = {}
            for stage in expert_stages:
                stage_execution = execution.get(stage, {})
                has_slices = bool(
                    stage_execution.get("has_physical_slices")
                )
                predicated_skip = bool(
                    stage_execution.get("predicated_skip")
                )
                stage_pass = (
                    not has_slices and predicated_skip
                    if expect_skip
                    else has_slices and not predicated_skip
                )
                expert_evidence[stage] = {
                    "pass": stage_pass,
                    "has_physical_slices": has_slices,
                    "predicated_skip": predicated_skip,
                    "expected": (
                        "predicated_skip_without_physical_slices"
                        if expect_skip
                        else "physical_slices_without_predicated_skip"
                    ),
                }
                if not stage_pass:
                    rank_errors.append(
                        {
                            "rank": rank,
                            "layer": layer,
                            "code": "expert_execution_route_mismatch",
                            "stage": stage,
                            "route_total": route_total,
                            **expert_evidence[stage],
                        }
                    )

            mandatory_evidence: dict[str, Any] = {}
            for stage in mandatory_stages:
                stage_execution = execution.get(stage, {})
                has_slices = bool(
                    stage_execution.get("has_physical_slices")
                )
                predicated_skip = bool(
                    stage_execution.get("predicated_skip")
                )
                stage_pass = has_slices and not predicated_skip
                mandatory_evidence[stage] = {
                    "pass": stage_pass,
                    "has_physical_slices": has_slices,
                    "predicated_skip": predicated_skip,
                    "expected": "physical_slices_without_predicated_skip",
                }
                if not stage_pass:
                    rank_errors.append(
                        {
                            "rank": rank,
                            "layer": layer,
                            "code": "mandatory_stage_not_executed",
                            "stage": stage,
                            "route_total": route_total,
                            **mandatory_evidence[stage],
                        }
                    )

            rank_results[rank] = {
                "available": True,
                "pass": not rank_errors,
                "route_total": route_total,
                "expected_expert_state": (
                    "predicated_skip" if expect_skip else "executed"
                ),
                "expert_stages": expert_evidence,
                "mandatory_stages": mandatory_evidence,
                "errors": rank_errors,
            }
            layer_errors.extend(rank_errors)
        layers[layer] = {
            "available": True,
            "pass": not layer_errors,
            "ranks": rank_results,
            "errors": layer_errors,
        }
        errors.extend(layer_errors)
    return {
        "applicable": True,
        "available": True,
        "pass": not errors,
        "layers": layers,
        "errors": errors,
        "interpretation": (
            "An exact zero local route total requires both local expert tasks "
            "to retire through explicit predicate skips with no physical "
            "slices. A nonzero total requires both expert tasks to execute. "
            "Route pack, local combine, MoE TP all-reduce, and residual add "
            "must execute in either case."
        ),
    }


def _predecessors(trace: RankTrace, task_id: str) -> list[dict[str, Any]]:
    return [edge for edge in trace.edges if str(edge.get("succ")) == str(task_id)]


def _arrival_analysis(
    traces: list[RankTrace],
    ranks: dict[str, dict[str, Any]],
    clock_alignment: dict[str, Any],
    profile: str = "candidate",
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not traces:
        return result
    comparable_ticks = bool(clock_alignment.get("cross_rank_tick_math_enabled"))
    frequency = traces[0].frequency_hz
    for layer in _LAYER_PREFIX:
        layer_result: dict[str, Any] = {}
        for label, producer_stage, consumer_stage in _arrival_pairs(profile):
            producers = []
            consumers = []
            for trace in traces:
                rank_data = ranks[trace.tag]
                stage_data = rank_data.get("layers", {}).get(layer, {})
                producer = stage_data.get(producer_stage)
                consumer = stage_data.get(consumer_stage)
                if producer is not None:
                    producers.append(
                        {
                            "rank": trace.tag,
                            "start_tick": producer["start_tick"],
                            "end_tick": producer["end_tick"],
                            "span_us": producer["stage_span_us"],
                        },
                    )
                if consumer is not None:
                    consumer_task_id = consumer["task_ids"][0]
                    predecessors = _predecessors(trace, consumer_task_id)
                    explicit = [edge for edge in predecessors if edge.get("source") == "explicit"]
                    producer_task_ids = set(
                        producer.get("task_ids", [])
                        if producer is not None
                        else []
                    )
                    producer_predecessors = [
                        edge
                        for edge in predecessors
                        if str(edge.get("pred")) in producer_task_ids
                    ]
                    consumer_task = trace.task_by_id[consumer_task_id]
                    consumers.append(
                        {
                            "rank": trace.tag,
                            "task_id": consumer_task_id,
                            "stage": consumer_stage,
                            "start_tick": consumer["start_tick"],
                            "end_tick": consumer["end_tick"],
                            "span_us": consumer["stage_span_us"],
                            "has_producer_dependency": bool(
                                producer_predecessors
                            ),
                            "producer_dependency_sources": sorted(
                                {
                                    str(edge.get("source", "unknown"))
                                    for edge in producer_predecessors
                                }
                            ),
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
                            "timing_evidence": _task_timing_evidence(
                                trace,
                                consumer_task,
                            ),
                        },
                    )
            if not producers or not consumers:
                continue
            latest = max(producers, key=lambda item: item["end_tick"]) if comparable_ticks else None
            earliest = min(producers, key=lambda item: item["end_tick"]) if comparable_ticks else None
            producer_by_rank = {item["rank"]: item for item in producers}
            consumer_details = []
            for consumer in consumers:
                rank = consumer["rank"]
                local_producer = producer_by_rank.get(rank)
                peer_producers = [item for item in producers if item["rank"] != rank]
                latest_peer = (
                    max(peer_producers, key=lambda item: item["end_tick"])
                    if comparable_ticks and peer_producers
                    else None
                )
                detail = dict(consumer)
                if comparable_ticks and latest_peer is not None:
                    detail["latest_peer_producer_rank"] = latest_peer["rank"]
                    detail["remote_arrival_after_consumer_start_us"] = _round(
                        max(
                            0,
                            latest_peer["end_tick"] - consumer["start_tick"],
                        )
                        / frequency
                        * 1e6,
                    )
                    detail["remote_arrival_after_wait_start_us"] = detail[
                        "remote_arrival_after_consumer_start_us"
                    ]
                if comparable_ticks and local_producer is not None:
                    all_ready_tick = max(item["end_tick"] for item in producers)
                    detail["wait_overlap_completion_upper_bound_us"] = _round(
                        max(0, consumer["end_tick"] - all_ready_tick)
                        / frequency
                        * 1e6,
                    )
                consumer_details.append(detail)
            layer_result[label] = {
                "clock_domain_comparable": comparable_ticks,
                "clock_evidence_level": clock_alignment.get("evidence_level"),
                "producer_stage": producer_stage,
                "consumer_stage": consumer_stage,
                "producer_end_skew_us": (
                    _round(
                        (latest["end_tick"] - earliest["end_tick"]) / frequency * 1e6,
                    )
                    if latest is not None and earliest is not None
                    else None
                ),
                "earliest_producer_rank": (earliest["rank"] if earliest is not None else None),
                "latest_producer_rank": (latest["rank"] if latest is not None else None),
                "producer_ranks": producers,
                "consumer_ranks": consumer_details,
                # Retained for consumers of the v6 dispatch/combine schema.
                "wait_ranks": consumer_details,
            }
        result[layer] = layer_result
    return result


def _routed_slice_profile_contract(
    ranks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Prove that every routed stage exposes complete AIC/AIV slice data."""
    coverage: dict[str, Any] = {layer: {} for layer in _LAYER_PREFIX}
    errors: list[dict[str, Any]] = []
    for layer in _LAYER_PREFIX:
        for rank, rank_data in ranks.items():
            rank_coverage: dict[str, Any] = {}
            stages = rank_data.get("layers", {}).get(layer, {})
            for stage in _ROUTED_PROFILE_STAGES:
                stage_data = stages.get(stage)
                if stage_data is None:
                    rank_coverage[stage] = {
                        "present": False,
                        "resources": {},
                    }
                    continue
                resources: dict[str, Any] = {}
                for resource in ("aic", "aiv"):
                    metrics = stage_data.get("resources", {}).get(resource)
                    if metrics is None:
                        errors.append(
                            {
                                "rank": rank,
                                "layer": layer,
                                "stage": stage,
                                "resource": resource,
                                "code": "missing_resource_profile",
                            }
                        )
                        continue
                    distribution = metrics.get(
                        "duration_distribution",
                        {},
                    )
                    gates = distribution.get("gates", {})
                    gated_count = sum(
                        int(bucket.get("count", 0)) for bucket in gates.values() if isinstance(bucket, dict)
                    )
                    observed = int(metrics.get("observed_slices", 0))
                    record_count = len(metrics.get("physical_slices", []))
                    resource_errors = []
                    if record_count != observed:
                        resource_errors.append(
                            {
                                "code": "physical_record_count",
                                "expected": observed,
                                "actual": record_count,
                            }
                        )
                    if gated_count != observed:
                        resource_errors.append(
                            {
                                "code": "duration_gate_count",
                                "expected": observed,
                                "actual": gated_count,
                            }
                        )
                    errors.extend(
                        {
                            "rank": rank,
                            "layer": layer,
                            "stage": stage,
                            "resource": resource,
                            **error,
                        }
                        for error in resource_errors
                    )
                    resources[resource] = {
                        "available": bool(metrics.get("available")),
                        "expected_slices": metrics.get("expected_slices"),
                        "observed_slices": observed,
                        "physical_record_count": record_count,
                        "duration_gate_count": gated_count,
                        "duration_gates": gates,
                        "profile_path": (f"ranks.{rank}.layers.{layer}.{stage}.resources.{resource}"),
                    }
                rank_coverage[stage] = {
                    "present": True,
                    "task_instances": stage_data["task_instances"],
                    "resources": resources,
                }
            coverage[layer][rank] = rank_coverage
    return {
        "pass": not errors,
        "stages": list(_ROUTED_PROFILE_STAGES),
        "coverage": coverage,
        "errors": errors,
        "interpretation": (
            "Every present routed stage has explicit AIC and AIV profiles; "
            "each observed slice appears once in physical_slices and exactly "
            "one <10/10-30/>30us gate."
        ),
    }


def _expert_kernel_release_contract(
    ranks: dict[str, dict[str, Any]],
    profile: str = "candidate",
) -> dict[str, Any]:
    """Apply the per-nonempty-rank expert grain and activation release gates."""
    resolved_profile = _resolve_profile(profile)
    policy = _source_policy(resolved_profile)
    release_family = str(policy.get("expert_release_family", "split"))
    if release_family not in _EXPERT_AIC_RELEASE_STAGES:
        raise ValueError(
            f"unsupported expert release family {release_family!r} "
            f"for profile {resolved_profile!r}"
        )
    required_aic_stages = _EXPERT_AIC_RELEASE_STAGES[release_family]
    duration_limits = _EXPERT_DURATION_LIMITS_US.get(release_family, {})
    packed_nz_mixed = release_family == "packed_nz_mixed"
    mixed_resource_targets = (
        policy.get("mixed_resource_targets", _PACKED_NZ_RESOURCE_TARGETS)
        if packed_nz_mixed else {}
    )
    mixed_resource_labels = (
        _mixed_resource_grid_labels(mixed_resource_targets)
        if packed_nz_mixed else {}
    )
    coverage: dict[str, Any] = {layer: {} for layer in _LAYER_PREFIX}
    coverage_errors: list[dict[str, Any]] = []
    duration_errors: list[dict[str, Any]] = []
    activation_errors: list[dict[str, Any]] = []
    mixed_resource_errors: list[dict[str, Any]] = []
    for layer in _LAYER_PREFIX:
        for rank, rank_data in ranks.items():
            stages = rank_data.get("layers", {}).get(layer, {})
            observed_by_stage = {}
            for stage in _ROUTED_PROFILE_STAGES:
                stage_data = stages.get(stage, {})
                observed_by_stage[stage] = {
                    resource: int(
                        stage_data.get("resources", {})
                        .get(resource, {})
                        .get("observed_slices", 0)
                    )
                    for resource in ("aic", "aiv")
                }
            execution_nonempty = any(
                count > 0
                for resources in observed_by_stage.values()
                for count in resources.values()
            )
            rank_coverage: dict[str, Any] = {
                "execution_nonempty": execution_nonempty,
                "route_empty_inferred": False,
                "observed_slices_by_stage": observed_by_stage,
                "aic_duration_stages": {},
                "activation_aiv": {
                    "applicable": execution_nonempty and not packed_nz_mixed,
                    "pass": (
                        None
                        if not execution_nonempty or packed_nz_mixed
                        else False
                    ),
                },
                "mixed_resource_grid": {
                    "applicable": execution_nonempty and packed_nz_mixed,
                    "pass": (
                        None
                        if not execution_nonempty or not packed_nz_mixed
                        else False
                    ),
                },
            }
            if not execution_nonempty:
                rank_coverage["interpretation"] = (
                    "No routed physical compute was observed. This does not "
                    "prove route-empty because no routed-token histogram is "
                    "available."
                )
                coverage[layer][rank] = rank_coverage
                continue

            if packed_nz_mixed:
                stage_checks: dict[str, Any] = {}
                rank_errors: list[dict[str, Any]] = []
                for stage, targets in mixed_resource_targets.items():
                    stage_data = stages.get(stage)
                    resources = (
                        stage_data.get("resources", {})
                        if stage_data is not None
                        else {}
                    )
                    checks = {
                        "stage_present": stage_data is not None,
                        "one_task_instance": (
                            stage_data is not None
                            and int(stage_data.get("task_instances", 0)) == 1
                        ),
                        "blocks_match_aic_grid": (
                            stage_data is not None
                            and stage_data.get("blocks_per_task")
                            == [targets["aic"]]
                        ),
                    }
                    observed: dict[str, Any] = {}
                    for resource, target in targets.items():
                        metrics = resources.get(resource, {})
                        observed[resource] = {
                            "target": target,
                            "expected_slices": int(
                                metrics.get("expected_slices", 0)
                            ),
                            "observed_slices": int(
                                metrics.get("observed_slices", 0)
                            ),
                            "distinct_cores": int(
                                metrics.get("distinct_cores", 0)
                            ),
                        }
                        checks[f"{resource}_available"] = bool(
                            metrics.get("available")
                        )
                        checks[f"{resource}_expected_slices"] = (
                            observed[resource]["expected_slices"] == target
                        )
                        checks[f"{resource}_observed_slices"] = (
                            observed[resource]["observed_slices"] == target
                        )
                        checks[f"{resource}_distinct_cores"] = (
                            observed[resource]["distinct_cores"] == target
                        )
                    stage_pass = all(checks.values())
                    stage_checks[stage] = {
                        "pass": stage_pass,
                        "checks": checks,
                        "observed": observed,
                    }
                    if not stage_pass:
                        rank_errors.append(
                            {
                                "rank": rank,
                                "layer": layer,
                                "stage": stage,
                                "code": "packed_nz_mixed_resource_grid",
                                "failed_checks": [
                                    name
                                    for name, passed in checks.items()
                                    if not passed
                                ],
                                "observed": observed,
                            }
                        )
                mixed_resource_errors.extend(rank_errors)
                rank_coverage["mixed_resource_grid"] = {
                    "applicable": True,
                    "pass": not rank_errors,
                    "targets": mixed_resource_targets,
                    "stages": stage_checks,
                }
                rank_coverage["activation_aiv"] = {
                    "applicable": False,
                    "pass": None,
                    "reason": (
                        "Activation and requant run on the AIV slices of the "
                        "mixed expert_gate_up task."
                    ),
                }
                rank_coverage["interpretation"] = (
                    "Packed-NZ nonempty ranks require fused and down mixed "
                    "tasks at the policy-selected AIC/AIV resource grids. "
                    "A rank with no routed physical slices is accepted here; "
                    "the task-ID contract must prove explicit predicate skips "
                    "and the route sidecar must prove zero routes."
                )
                coverage[layer][rank] = rank_coverage
                continue

            for stage in required_aic_stages:
                stage_data = stages.get(stage)
                resource = (
                    stage_data.get("resources", {}).get("aic", {})
                    if stage_data is not None
                    else {}
                )
                distribution = resource.get("duration_distribution", {})
                values = {
                    "p50_us": distribution.get("p50_us"),
                    "p90_us": distribution.get("p90_us"),
                    "p99_us": distribution.get("p99_us"),
                    "max_us": distribution.get("max_us"),
                }
                checks = {
                    "p50_le_limit": (
                        values["p50_us"] is not None
                        and values["p50_us"] <= duration_limits["p50_max"]
                    ),
                    "p90_le_limit": (
                        values["p90_us"] is not None
                        and values["p90_us"] <= duration_limits["p90_max"]
                    ),
                    "p99_le_limit": (
                        values["p99_us"] is not None
                        and values["p99_us"] <= duration_limits["p99_max"]
                    ),
                    "max_le_limit": (
                        values["max_us"] is not None
                        and values["max_us"] <= duration_limits["max"]
                    ),
                }
                if "p50_min" in duration_limits:
                    checks["p50_ge_limit"] = (
                        values["p50_us"] is not None
                        and values["p50_us"] >= duration_limits["p50_min"]
                    )
                stage_pass = bool(resource.get("available")) and all(checks.values())
                rank_coverage["aic_duration_stages"][stage] = {
                    "present": stage_data is not None,
                    "available": bool(resource.get("available")),
                    "pass": stage_pass,
                    "values": values,
                    "checks": checks,
                    "physical_slice_count": int(resource.get("observed_slices", 0)),
                    "profile_path": (
                        f"ranks.{rank}.layers.{layer}.{stage}.resources.aic"
                    ),
                }
                if not stage_pass:
                    duration_errors.append(
                        {
                            "rank": rank,
                            "layer": layer,
                            "stage": stage,
                            "code": (
                                "missing_aic_stage"
                                if not resource.get("available")
                                else "duration_limit"
                            ),
                            "values": values,
                            "failed_checks": [
                                name for name, passed in checks.items() if not passed
                            ],
                        }
                    )

            activation_checks: dict[str, bool] = {}
            activation_errors_for_rank: list[dict[str, Any]] = []
            activation_stage_data: dict[str, dict[str, int]] = {}
            for activation_stage in (
                "expert_gate_up_act",
                "routed_h_quant",
            ):
                activation = stages.get(activation_stage)
                activation_resources = (
                    activation.get("resources", {}) if activation else {}
                )
                activation_aic = activation_resources.get("aic", {})
                activation_aiv = activation_resources.get("aiv", {})
                stage_checks = {
                    "stage_present": activation is not None,
                    "aiv_observed": (
                        bool(activation_aiv.get("available"))
                        and int(activation_aiv.get("observed_slices", 0)) > 0
                    ),
                    "aic_not_observed": (
                        int(activation_aic.get("observed_slices", 0)) == 0
                    ),
                }
                activation_checks[activation_stage] = all(
                    stage_checks.values()
                )
                activation_stage_data[activation_stage] = {
                    "aic_observed_slices": int(
                        activation_aic.get("observed_slices", 0)
                    ),
                    "aiv_observed_slices": int(
                        activation_aiv.get("observed_slices", 0)
                    ),
                }
                if not activation_checks[activation_stage]:
                    activation_errors_for_rank.append(
                        {
                            "stage": activation_stage,
                            "failed_checks": [
                                name
                                for name, passed in stage_checks.items()
                                if not passed
                            ],
                        }
                    )
            activation_pass = not activation_errors_for_rank
            rank_coverage["activation_aiv"] = {
                "applicable": True,
                "pass": activation_pass,
                "checks": activation_checks,
                "stages": activation_stage_data,
                "profile_path": (
                    f"ranks.{rank}.layers.{layer}.expert_gate_up_act.resources.aiv"
                ),
            }
            for activation_error in activation_errors_for_rank:
                activation_errors.append(
                    {
                        "rank": rank,
                        "layer": layer,
                        "stage": activation_error["stage"],
                        "code": "activation_quant_must_be_aiv_only",
                        "failed_checks": activation_error["failed_checks"],
                    }
                )
            rank_coverage["interpretation"] = (
                "execution_nonempty is based only on observed routed physical "
                "slices. It selects ranks to which the release gate applies; "
                "it is not a routed-token or route-empty inference."
            )
            coverage[layer][rank] = rank_coverage

        nonempty_ranks = [
            rank
            for rank, rank_coverage in coverage[layer].items()
            if rank_coverage["execution_nonempty"]
        ]
        if not nonempty_ranks:
            coverage_errors.append(
                {
                    "layer": layer,
                    "code": "no_routed_compute_observed",
                    "reason": (
                        "No rank exposes routed physical compute for this MoE "
                        "layer; the capture cannot satisfy the expert release "
                        "gate."
                    ),
                }
            )

    diagnostic_pass = (
        not coverage_errors
        and not duration_errors
        and not activation_errors
        and not mixed_resource_errors
    )
    release_enforced = policy["enforce_candidate_release_gate"]
    return {
        "pass": diagnostic_pass if release_enforced else None,
        "diagnostic_pass": diagnostic_pass,
        "release_gate_pass": diagnostic_pass if release_enforced else None,
        "release_gate_status": (
            "PASS"
            if release_enforced and diagnostic_pass
            else "BLOCKED"
            if release_enforced
            else "NOT_APPLICABLE"
        ),
        "coverage_pass": not coverage_errors,
        "duration_pass": not duration_errors,
        "activation_pass": not activation_errors,
        "mixed_resource_grid_pass": not mixed_resource_errors,
        "profile": resolved_profile,
        "source_policy": policy,
        "release_enforced": release_enforced,
        "release_family": release_family,
        "required_aic_stages": list(required_aic_stages),
        "duration_limits_us": dict(duration_limits),
        "mixed_resource_targets": mixed_resource_targets,
        "duration_limit_source": policy.get("duration_limit_source"),
        "coverage": coverage,
        "coverage_errors": coverage_errors,
        "duration_errors": duration_errors,
        "activation_errors": activation_errors,
        "mixed_resource_errors": mixed_resource_errors,
        "interpretation": (
            (
                f"The {policy['policy_id']} policy is selected by exact "
                "source SHA. Each execution-nonempty rank must expose one "
                "mixed fused task at "
                f"{mixed_resource_labels['expert_gate_up']} and one mixed "
                "down task at "
                f"{mixed_resource_labels['expert_down']}; independent "
                "activation/quant stages are not required."
            )
            if packed_nz_mixed
            else (
                "The 671a5df8 route-sidecar candidate is selected by exact "
                "source SHA, not by task-name inference. Its staged fused "
                "family requires AIC gate_up/down coverage and AIV-only "
                "activation and quant coverage on every execution-nonempty "
                "rank."
            )
        ),
    }


def _queue_delay_classification(stage_data: dict[str, Any]) -> dict[str, Any]:
    available_values = []
    unavailable = []
    for task in stage_data.get("task_instance_details", []):
        queue_delay = task.get("timing_evidence", {}).get("queue_delay", {})
        if queue_delay.get("available"):
            available_values.append(float(queue_delay["value_us"]))
        else:
            unavailable.append(
                {
                    "task_id": task.get("task_id"),
                    "reason": queue_delay.get("reason", "unspecified"),
                }
            )
    if not available_values:
        return {
            "classification": "unknown",
            "available": False,
            "reason": (
                unavailable[0]["reason"]
                if unavailable
                else "no task-level queue-delay evidence"
            ),
            "unavailable_tasks": unavailable,
        }
    max_delay = max(available_values)
    return {
        "classification": (
            "observed"
            if max_delay > 0.0
            else "not_observed"
            if not unavailable
            else "partial"
        ),
        "available": not unavailable,
        "max_us": _round(max_delay),
        "values_us": [_round(value) for value in available_values],
        "unavailable_tasks": unavailable,
        "semantics": (
            "Local-rank delay from the latest fully observed direct "
            "predecessor completion to the first physical task slice."
        ),
    }


def _parallelism_classification(resource: dict[str, Any]) -> dict[str, Any]:
    observed = int(resource.get("observed_slices", 0))
    available_cores = int(resource.get("available_cores", 0))
    peak = int(resource.get("peak_concurrency", 0))
    if not resource.get("available") or observed == 0:
        unknown = {
            "classification": "unknown",
            "available": False,
            "reason": "no physical slices for this stage resource",
        }
        return {
            "insufficient_ready_parallelism": dict(unknown),
            "scheduler_packing": dict(unknown),
        }

    full_width_target = min(observed, available_cores)
    if observed < available_cores:
        ready = {
            "classification": "observed_work_width_limit",
            "available": True,
            "observed_slice_count": observed,
            "available_cores": available_cores,
            "upper_bound_concurrency": full_width_target,
            "reason": (
                "Even if every observed physical slice were simultaneously "
                "ready, this stage could not occupy every resource core."
            ),
        }
    elif peak >= available_cores:
        ready = {
            "classification": "not_observed",
            "available": True,
            "observed_slice_count": observed,
            "available_cores": available_cores,
            "peak_concurrency": peak,
            "reason": "The stage reached full resource concurrency.",
        }
    else:
        ready = {
            "classification": "undetermined",
            "available": False,
            "observed_slice_count": observed,
            "available_cores": available_cores,
            "peak_concurrency": peak,
            "reason": (
                "Aggregate swim timing has enough total slices but no "
                "block-level ready timestamps, so readiness and scheduler "
                "packing cannot be separated."
            ),
            "required_input": "block-level ready/enqueue timestamps",
        }

    if peak < full_width_target:
        packing = {
            "classification": "suspected",
            "available": False,
            "peak_concurrency": peak,
            "full_width_target": full_width_target,
            "unused_offered_width_at_peak": full_width_target - peak,
            "reason": (
                "Observed peak concurrency is below the physical-slice width. "
                "Block-level ready/enqueue timestamps are required to prove "
                "scheduler packing rather than insufficient readiness."
            ),
            "required_input": "block-level ready/enqueue timestamps",
        }
    else:
        packing = {
            "classification": "not_observed",
            "available": True,
            "peak_concurrency": peak,
            "full_width_target": full_width_target,
            "reason": "Peak concurrency reached the observable work-width target.",
        }
    return {
        "insufficient_ready_parallelism": ready,
        "scheduler_packing": packing,
    }


def _execution_limit_classification(
    ranks: dict[str, dict[str, Any]],
    route_histogram: dict[str, Any],
    profile: str = "candidate",
) -> dict[str, Any]:
    """Keep route, readiness, queueing, and packing diagnoses separate."""
    coverage: dict[str, Any] = {layer: {} for layer in _LAYER_PREFIX}
    for layer in _LAYER_PREFIX:
        route_contract = route_histogram.get(layer, {})
        for rank, rank_data in ranks.items():
            if route_contract.get("available"):
                rank_histogram = route_contract["histogram"][rank]
                routed_tokens = sum(rank_histogram)
                route_empty = {
                    "classification": (
                        "observed" if routed_tokens == 0 else "not_observed"
                    ),
                    "available": True,
                    "routed_token_count": routed_tokens,
                    "local_expert_count": rank_histogram,
                    "source": route_contract["source"],
                    "semantics": (
                        "Exact sum over src_rank from the validated recv_meta "
                        "sidecar; no task/tile proxy is used."
                    ),
                }
            else:
                route_empty = {
                    "classification": "unknown",
                    "available": False,
                    "reason": route_contract.get(
                        "reason",
                        "no routed-token histogram is available",
                    ),
                    "rejected_proxies": route_contract.get(
                        "rejected_proxies",
                        [],
                    ),
                }
            rank_result: dict[str, Any] = {
                "route_empty": route_empty,
                "stages": {},
            }
            stages = rank_data.get("layers", {}).get(layer, {})
            for stage, resource_name in _diagnostic_stage_resources(
                profile
            ).items():
                stage_data = stages.get(stage)
                if stage_data is None:
                    rank_result["stages"][stage] = {
                        "execution_observed": False,
                        "resource": resource_name,
                        "queue_delay": {
                            "classification": "unknown",
                            "available": False,
                            "reason": "stage has no physical execution profile",
                        },
                        "insufficient_ready_parallelism": {
                            "classification": "unknown",
                            "available": False,
                            "reason": "stage has no physical execution profile",
                        },
                        "scheduler_packing": {
                            "classification": "unknown",
                            "available": False,
                            "reason": "stage has no physical execution profile",
                        },
                    }
                    continue
                resource = stage_data.get("resources", {}).get(resource_name, {})
                parallelism = _parallelism_classification(resource)
                rank_result["stages"][stage] = {
                    "execution_observed": bool(resource.get("available")),
                    "resource": resource_name,
                    "queue_delay": _queue_delay_classification(stage_data),
                    **parallelism,
                }
            coverage[layer][rank] = rank_result
    return {
        "coverage": coverage,
        "categories": [
            "route_empty",
            "insufficient_ready_parallelism",
            "queue_delay",
            "scheduler_packing",
        ],
        "interpretation": (
            "Route-empty requires routed-token evidence and is never inferred "
            "from task, block, tile, or physical-slice counts. Queue delay uses "
            "only local predecessor timing. Insufficient ready parallelism and "
            "scheduler packing remain distinct; ambiguous cases require "
            "block-level ready/enqueue timestamps."
        ),
    }


def _external_correctness_contract() -> dict[str, Any]:
    return {
        "hidden_state_bit_exact": {
            "enforced_here": False,
            "enforced_by": "outer five-layer correctness gate",
            "required_artifacts": [
                "hidden_l3.pt",
                "hidden_l4.pt",
            ],
            "comparison": (
                "bit-exact against the matching protocol-specific "
                "read-only frozen golden"
            ),
            "required_manifest_fields": [
                "source_kind",
                "bit_exact",
            ],
            "local_ep_required_manifest_fields": [
                "source_kind",
                "protocol_profile",
                "numeric_contract",
                "bit_exact",
            ],
            "interpretation": (
                "This DFX analyzer does not load or compare hidden states. "
                "DFX publication must be paired with the outer bit-exact gate "
                "for the same protocol_profile and numeric_contract; a "
                "tolerance-based comparison is not a substitute."
            ),
        }
    }


def _timing_evidence_contract(
    ranks: dict[str, dict[str, Any]],
    profile: str = "candidate",
) -> dict[str, Any]:
    fields = {
        field: {
            "available_task_count": 0,
            "unavailable_task_count": 0,
            "unavailable": [],
        }
        for field in (
            "queue_delay",
            "service_span",
            "dag_span",
            "critical_path_contribution",
        )
    }
    profiled_tasks = 0
    profiled_stages = _timing_profile_stages(profile)
    for rank, rank_data in ranks.items():
        for layer in _LAYER_PREFIX:
            stages = rank_data.get("layers", {}).get(layer, {})
            for stage in profiled_stages:
                stage_data = stages.get(stage)
                if stage_data is None:
                    continue
                for task in stage_data.get("task_instance_details", []):
                    profiled_tasks += 1
                    evidence = task.get("timing_evidence", {})
                    for field, summary in fields.items():
                        item = evidence.get(
                            field,
                            {
                                "available": False,
                                "reason": "field missing from analyzer output",
                            },
                        )
                        if item.get("available"):
                            summary["available_task_count"] += 1
                            continue
                        summary["unavailable_task_count"] += 1
                        summary["unavailable"].append(
                            {
                                "rank": rank,
                                "layer": layer,
                                "stage": stage,
                                "task_id": task["task_id"],
                                "reason": item.get(
                                    "reason",
                                    "unspecified",
                                ),
                            }
                        )
    unavailable_fields = [field for field, summary in fields.items() if summary["unavailable_task_count"]]
    return {
        "pass": not unavailable_fields,
        "profiled_task_count": profiled_tasks,
        "profiled_stages": list(profiled_stages),
        "fields": fields,
        "unavailable_fields": unavailable_fields,
        "interpretation": (
            "This is an instrumentation-completeness report, not a release "
            "gate. Missing evidence is never replaced by zero or an estimate."
        ),
    }


def _admission_contract(
    route_histogram: dict[str, Any],
    timing_evidence: dict[str, Any],
    routed_slice_profiles: dict[str, Any],
    expert_kernel_release: dict[str, Any],
) -> dict[str, Any]:
    blockers = []
    limitations = []
    profile = _resolve_profile(
        str(expert_kernel_release.get("profile") or "candidate")
    )
    source_policy = _source_policy(profile)
    expert_release_enforced = bool(
        source_policy["enforce_candidate_release_gate"]
    )
    packed_nz_mixed = (
        expert_kernel_release.get("release_family") == "packed_nz_mixed"
    )
    reported_release_enforced = expert_kernel_release.get("release_enforced")
    route_sidecar_ready = bool(route_histogram) and all(
        contract.get("available", False)
        and contract.get("publication_evidence_ready", False)
        for contract in route_histogram.values()
    )
    checks = {
        "routed_slice_profiles": bool(routed_slice_profiles["pass"]),
        "expert_routed_compute_coverage": (
            bool(expert_kernel_release["coverage_pass"])
            if expert_release_enforced
            else None
        ),
        "expert_aic_duration": (
            bool(expert_kernel_release["duration_pass"])
            if expert_release_enforced and not packed_nz_mixed
            else None
        ),
        "expert_activation_aiv": (
            bool(expert_kernel_release["activation_pass"])
            if expert_release_enforced and not packed_nz_mixed
            else None
        ),
        "expert_mixed_resource_grid": (
            bool(expert_kernel_release["mixed_resource_grid_pass"])
            if expert_release_enforced and packed_nz_mixed
            else None
        ),
        "route_histogram_proxy_prohibited": all(
            not contract.get("proxy_fallback_allowed", False)
            for contract in route_histogram.values()
        ),
        "recv_meta_publication_evidence_ready": route_sidecar_ready,
        "expert_release_policy_consistent": (
            reported_release_enforced is None
            or bool(reported_release_enforced) == expert_release_enforced
        ),
    }
    if not checks["expert_release_policy_consistent"]:
        blockers.append(
            {
                "code": "expert_release_policy_mismatch",
                "reason": (
                    "Candidate release enforcement is owned by the frozen "
                    "source policy and cannot be overridden by task-derived "
                    "or caller-supplied metadata."
                ),
                "policy_value": expert_release_enforced,
                "reported_value": bool(reported_release_enforced),
            }
        )
    if not routed_slice_profiles["pass"]:
        blockers.append(
            {
                "code": "routed_slice_profile_contract_failed",
                "errors": routed_slice_profiles["errors"][:8],
            }
        )
    if (
        expert_release_enforced
        and not packed_nz_mixed
        and not expert_kernel_release["duration_pass"]
    ):
        blockers.append(
            {
                "code": "expert_aic_duration_release_failed",
                "errors": expert_kernel_release["duration_errors"][:8],
            }
        )
    if expert_release_enforced and not expert_kernel_release["coverage_pass"]:
        blockers.append(
            {
                "code": "expert_routed_compute_coverage_failed",
                "errors": expert_kernel_release["coverage_errors"][:8],
            }
        )
    if (
        expert_release_enforced
        and not packed_nz_mixed
        and not expert_kernel_release["activation_pass"]
    ):
        blockers.append(
            {
                "code": "expert_activation_aiv_release_failed",
                "errors": expert_kernel_release["activation_errors"][:8],
            }
        )
    if (
        expert_release_enforced
        and packed_nz_mixed
        and not expert_kernel_release["mixed_resource_grid_pass"]
    ):
        blockers.append(
            {
                "code": "expert_mixed_resource_grid_release_failed",
                "errors": expert_kernel_release["mixed_resource_errors"][:8],
            }
        )
    if not expert_release_enforced:
        limitations.append(
            {
                "code": "baseline_expert_release_diagnostic_only",
                "analyzer_blocking": False,
                "release_readiness_blocking": False,
                "candidate_diagnostic_pass": expert_kernel_release.get(
                    "diagnostic_pass",
                    expert_kernel_release.get("pass"),
                ),
                "reason": (
                    "The frozen row32 fused baseline enforces structural DFX "
                    "contracts only. Split gate/up/down duration and AIV "
                    "activation are candidate-only release gates."
                ),
            }
        )
    if not checks["route_histogram_proxy_prohibited"]:
        blockers.append(
            {
                "code": "route_histogram_proxy_fallback_enabled",
                "reason": "tile/task/block counts must never substitute for recv_meta",
            }
        )
    for layer, contract in route_histogram.items():
        if not contract.get("available"):
            limitations.append(
                {
                    "code": "route_histogram_awaiting_recv_meta",
                    "layer": layer,
                    "analyzer_blocking": False,
                    "release_readiness_blocking": profile != "baseline",
                    "reason": contract["reason"],
                    "required_input": contract["required_input"],
                }
            )
    for field, summary in timing_evidence["fields"].items():
        if summary["unavailable_task_count"]:
            limitations.append(
                {
                    "code": f"{field}_incomplete",
                    "analyzer_blocking": False,
                    "release_readiness_blocking": False,
                    "unavailable_task_count": summary["unavailable_task_count"],
                    "reason": (
                        summary["unavailable"][0]["reason"]
                        if summary["unavailable"]
                        else "unspecified"
                    ),
                }
            )
    external = _external_correctness_contract()
    analyzer_gate_pass = not blockers
    release_reasons: list[dict[str, Any]]
    if not analyzer_gate_pass:
        release_status = "BLOCKED"
        release_reasons = [
            {
                "code": "analyzer_gate_failed",
                "analyzer_blockers": [blocker["code"] for blocker in blockers],
            }
        ]
    elif profile == "baseline":
        release_status = "DIAGNOSTIC_ONLY"
        release_reasons = [
            {
                "code": "baseline_not_a_release_candidate",
                "reason": (
                    "The frozen row32 fused baseline is analyzed for structural "
                    "and comparative diagnostics only."
                ),
            }
        ]
    elif not route_sidecar_ready:
        release_status = "NOT_EVALUABLE"
        release_reasons = [
            {
                "code": "recv_meta_publication_evidence_missing",
                "reason": (
                    "Exact L3/L4 owner-count histogram evidence is required "
                    "before candidate publication readiness can be evaluated."
                ),
            }
        ]
    else:
        release_status = "PENDING_EXTERNAL_GATE"
        release_reasons = [
            {
                "code": "hidden_state_bit_exact_required",
                "reason": (
                    "Analyzer-owned candidate gates and recv_meta evidence are "
                    "ready; the outer hidden-state bit-exact gate remains."
                ),
            }
        ]
    release_blocked = release_status in {
        "BLOCKED",
        "NOT_EVALUABLE",
        "PENDING_EXTERNAL_GATE",
    }
    return {
        "pass": analyzer_gate_pass,
        "analyzer_gate_pass": analyzer_gate_pass,
        "checks": checks,
        "profile": profile,
        "source_policy": source_policy,
        "expert_release_enforced": expert_release_enforced,
        "blockers": blockers,
        "non_blocking_limitations": limitations,
        "release_readiness": {
            "status": release_status,
            "blocked": release_blocked,
            "publication_allowed": False,
            "analyzer_ready": analyzer_gate_pass,
            "recv_meta_publication_evidence_ready": route_sidecar_ready,
            "external_hidden_state_gate_required": True,
            "reasons": release_reasons,
        },
        "external_required_gates": external,
        "interpretation": (
            "Structural task-ID/profile-dependency/slice corruption raises "
            "before report publication. This analyzer gate covers executable "
            "DFX contracts and policy-selected expert kernel release limits. "
            "Missing recv_meta does not fail the analyzer gate, but candidate "
            "release readiness is NOT_EVALUABLE and blocked until the exact "
            "sidecar is present. Unavailable critical-path contribution "
            "remains a non-blocking instrumentation limitation. Even with a "
            "valid sidecar, publication still requires the outer hidden-state "
            "bit-exact gate."
        ),
    }


def _aggregate_findings(
    ranks: dict[str, dict[str, Any]],
    arrivals: dict[str, Any],
    route_histogram: dict[str, Any],
    profile: str = "candidate",
) -> list[dict[str, str]]:
    local_ep = _resolve_profile(profile) == "local-ep"
    findings: list[dict[str, str]] = []
    for layer in _LAYER_PREFIX:
        fused_gate_rows = []
        split_gate_rows: dict[str, list[tuple[str, dict[str, Any]]]] = {
            "expert_gate": [],
            "expert_up": [],
            "expert_gate_up_act": [],
        }
        shared_rows = []
        split_shared_rows = []
        split_shared_stage_rows: dict[
            str,
            list[tuple[str, dict[str, Any]]],
        ] = {
            "shared_gate_up": [],
            "shared_gate_up_act": [],
            "shared_down": [],
        }
        scatter_rows = []
        wait_rows = []
        for rank, rank_data in ranks.items():
            stages = rank_data.get("layers", {}).get(layer, {})
            gate = stages.get("expert_gate_up")
            if gate and gate["resources"].get("aic", {}).get("available"):
                fused_gate_rows.append((rank, gate))
            for stage in split_gate_rows:
                split = stages.get(stage)
                if split:
                    split_gate_rows[stage].append((rank, split))
            shared = stages.get("shared_mlp")
            if shared and shared["resources"].get("aic", {}).get("available"):
                shared_rows.append((rank, shared))
            split_shared = stages.get("shared_split")
            if split_shared:
                split_shared_rows.append((rank, split_shared))
            for stage in split_shared_stage_rows:
                split_stage = stages.get(stage)
                if split_stage:
                    split_shared_stage_rows[stage].append(
                        (rank, split_stage),
                    )
            if not local_ep:
                scatter = stages.get("combine_scatter")
                if (
                    scatter
                    and scatter["resources"].get("aiv", {}).get("available")
                ):
                    scatter_rows.append((rank, scatter))
                wait = stages.get("combine_wait")
                if (
                    wait
                    and wait["resources"].get("aiv", {}).get("available")
                ):
                    wait_rows.append((rank, wait))

        route_contract = route_histogram.get(layer, {})
        if not route_contract.get("available"):
            findings.append(
                {
                    "layer": layer,
                    "severity": "high",
                    "finding": (
                        "True routed-token/expert histogram is unavailable. "
                        "No route imbalance value is published from task, "
                        "block, or physical-slice counts."
                    ),
                    "action": (
                        route_contract.get(
                            "required_input",
                            "Export per-expert routed-token counts.",
                        )
                    ),
                },
            )

        if any(split_gate_rows.values()):
            for stage, rows in split_gate_rows.items():
                if not rows:
                    continue
                resource = "aiv" if stage == "expert_gate_up_act" else "aic"
                resource_rows = [
                    (rank, row) for rank, row in rows if row["resources"].get(resource, {}).get("available")
                ]
                if not resource_rows:
                    continue
                p50 = statistics.median(
                    row["resources"][resource]["slice_duration_us_p50"] for _rank, row in resource_rows
                )
                blocks = sorted(
                    {block for _rank, row in resource_rows for block in row["blocks_per_task"]},
                )
                findings.append(
                    {
                        "layer": layer,
                        "severity": ("high" if p50 > 30.0 else "info" if p50 >= 10.0 else "medium"),
                        "finding": (
                            f"Routed {stage} {resource.upper()} slice p50 is "
                            f"{p50:.1f} us with {blocks} logical blocks per "
                            "dependency task instance."
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
                row["resources"]["aic"]["slice_duration_us_p50"] for _rank, row in fused_gate_rows
            )
            blocks = sorted(
                {block for _rank, row in fused_gate_rows for block in row["blocks_per_task"]},
            )
            findings.append(
                {
                    "layer": layer,
                    "severity": "high" if gate_p50 > 30.0 else "info",
                    "finding": (
                        f"Routed gate/up AIC slice p50 is {gate_p50:.1f} us "
                        f"with {blocks} logical blocks per dependency task "
                        "instance."
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
                        "No routed gate/up physical compute slices were found "
                        "for this layer. This is an execution-profile gap, not "
                        "a route-histogram inference."
                    ),
                    "action": (
                        (
                            "Use the local-combine to TP all-reduce arrival "
                            "analysis; do not treat a long collective span as "
                            "pure reduction arithmetic."
                        )
                        if local_ep
                        else (
                            "Use all-rank arrival analysis; do not attribute "
                            "the entire combine_wait span to the wait kernel "
                            "itself."
                        )
                    ),
                },
            )

        if local_ep:
            collective = arrivals.get(layer, {}).get("moe_collective")
            if (
                collective
                and (collective.get("producer_end_skew_us") or 0.0) > 100.0
            ):
                findings.append(
                    {
                        "layer": layer,
                        "severity": "high",
                        "finding": (
                            "Local-combine producer completion skew is "
                            f"{collective['producer_end_skew_us']:.1f} us; "
                            "latest producer is "
                            f"{collective['latest_producer_rank']}."
                        ),
                        "action": (
                            "Optimize the late local expert/shared partial "
                            "producer. The following TP all-reduce span may "
                            "include peer-arrival spin."
                        ),
                    },
                )
        else:
            combine = arrivals.get(layer, {}).get("combine")
            if (
                combine
                and (combine.get("producer_end_skew_us") or 0.0) > 100.0
            ):
                findings.append(
                    {
                        "layer": layer,
                        "severity": "high",
                        "finding": (
                            "Combine producer completion skew is "
                            f"{combine['producer_end_skew_us']:.1f} us; "
                            "latest producer is "
                            f"{combine['latest_producer_rank']}."
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
                if max_wait_us > 100.0:
                    findings.append(
                        {
                            "layer": layer,
                            "severity": "high",
                            "finding": (
                                f"Combine wait reaches {max_wait_us:.1f} us "
                                f"on {max_wait_rank}. Cross-rank completion "
                                "subtraction is unavailable for this capture, "
                                "and task counts are not used as route "
                                "evidence."
                            ),
                            "action": (
                                "Treat this as a remote-producer tail signal, "
                                "not wait-kernel arithmetic. Reduce routed "
                                "compute and scatter tails; separately check "
                                "whether local scatter overlaps the wait."
                            ),
                        },
                    )

        if shared_rows:
            max_peak = max(row["resources"]["aic"]["peak_concurrency"] for _rank, row in shared_rows)
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

        if split_shared_rows:
            observed_blocks = {
                stage: sorted(
                    {
                        block
                        for _rank, row in rows
                        for block in row["blocks_per_task"]
                    },
                )
                for stage, rows in split_shared_stage_rows.items()
            }
            expected_blocks = {
                "shared_gate_up": [5],
                "shared_gate_up_act": [5],
                "shared_down": [16],
            }
            max_span = max(
                row["stage_span_us"]
                for _rank, row in split_shared_rows
            )
            findings.append(
                {
                    "layer": layer,
                    "severity": (
                        "info"
                        if observed_blocks == expected_blocks
                        else "high"
                    ),
                    "finding": (
                        "Split shared-expert pipeline uses "
                        f"{observed_blocks} logical blocks; its end-to-end "
                        f"local envelope reaches {max_span:.1f} us."
                    ),
                    "action": (
                        "Require the 5/5/16 task-shape contract, bit-exact "
                        "hidden_l3/hidden_l4, at least 20% shared-envelope "
                        "reduction, and at least 2% paired wall-median gain "
                        "before promoting this experiment."
                    ),
                },
            )

        if scatter_rows:
            max_slice = max(row["resources"]["aiv"]["slice_duration_us_max"] for _rank, row in scatter_rows)
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
    output.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return output


def _expert_gate_markdown_lines(
    expert_release: dict[str, Any],
) -> list[str]:
    if expert_release["release_family"] == "packed_nz_mixed":
        labels = _mixed_resource_grid_labels(
            expert_release["mixed_resource_targets"]
        )
        return [
            "- Packed-NZ mixed resource grid gate: "
            f"`{expert_release['mixed_resource_grid_pass']}` "
            f"(fused {labels['expert_gate_up']}; "
            f"BS1 down {labels['expert_down']})",
            "- Independent activation/quant stage gate: `not applicable` "
            "(activation and requant execute on fused-task AIV slices)",
        ]
    return [
        "- Expert AIC duration gate: "
        f"`{expert_release['duration_pass']}` "
        "(raw diagnostic; enforced only for candidate)",
        "- Expert activation AIV gate: "
        f"`{expert_release['activation_pass']}` "
        "(raw diagnostic; enforced only for candidate)",
    ]


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    def duration_gate_text(resource: dict[str, Any]) -> str:
        if not resource.get("available"):
            return "-"
        gates = resource["duration_distribution"]["gates"]
        return "/".join(
            str(gates[name]["count"])
            for name in (
                "lt_10_us",
                "from_10_to_30_us",
                "gt_30_us",
            )
        )

    source_policy = report["source_policy"]
    profile = report["profile"]
    release_readiness = report["admission"]["release_readiness"]
    lines = [
        "# Step3p5 L0-L4 MoE DFX report",
        "",
        "This report keeps L3 and L4 separate and reports AIC/AIV resources "
        "independently. `tp_all_reduce` spans may include peer-arrival spin.",
        "",
        f"- LOW-WAIT reference: `{report['reference_rank']}`",
        f"- DFX root: `{report['dfx_root']}`",
        f"- Source profile: `{report['profile']}`",
        f"- Frozen source policy: `{source_policy['policy_id']}`",
        f"- Frozen ref/decode SHA prefix: `{source_policy['frozen_ref']}` / "
        f"`{source_policy['decode_sha256_prefix']}`",
        "- Source family: "
        f"`{source_policy['storage_family']}` / "
        f"`{source_policy['schedule_family']}` / "
        f"`{source_policy['task_partition']}`",
        "- Cross-rank tick math: "
        f"`{report['clock_alignment']['cross_rank_tick_math_enabled']}` "
        "(per-rank normalized clocks; no external common-clock anchor)",
        "- Cross-rank terminal skew: `not computed` "
        "(cross-rank timestamp subtraction is prohibited)",
        f"- Analyzer gate: `{report['admission']['pass']}`",
        "- Release readiness: "
        f"`{release_readiness['status']}`; "
        f"blocked=`{release_readiness['blocked']}`; "
        f"publication_allowed=`{release_readiness['publication_allowed']}`",
        "",
        "## Analyzer gate",
        "",
    ]
    if report["admission"]["blockers"]:
        lines.extend(
            f"- `{blocker['code']}`: {blocker.get('reason', blocker)}"
            for blocker in report["admission"]["blockers"]
        )
    else:
        lines.append("- No blockers.")
    lines.extend(["", "### Non-blocking instrumentation limitations", ""])
    if report["admission"]["non_blocking_limitations"]:
        lines.extend(
            f"- `{limitation['code']}`: {limitation.get('reason', limitation)}"
            for limitation in report["admission"]["non_blocking_limitations"]
        )
    else:
        lines.append("- None.")
    hidden_gate = report["external_correctness_contract"]["hidden_state_bit_exact"]
    expert_release = report["expert_kernel_release"]
    expert_gate_lines = _expert_gate_markdown_lines(expert_release)
    lines.extend(
        [
            "",
            "## Release contracts",
            "",
            "- Release readiness: "
            f"`{release_readiness['status']}` "
            f"(recv_meta ready="
            f"`{release_readiness['recv_meta_publication_evidence_ready']}`; "
            "outer hidden-state gate required="
            f"`{release_readiness['external_hidden_state_gate_required']}`)",
            "- Candidate expert gate enforcement: "
            f"`{report['expert_kernel_release']['release_enforced']}`",
            "- Candidate expert gate status: "
            f"`{report['expert_kernel_release']['release_gate_status']}`",
            *expert_gate_lines,
            "- Hidden-state bit-exact gate: "
            f"`external`, enforced by `{hidden_gate['enforced_by']}` for "
            f"`{hidden_gate['required_artifacts']}`",
        ]
    )
    lines.extend(
        [
            "",
            "## Rank overview",
            "",
        ]
    )
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
        route_contract = report["route_histogram"].get(layer, {})
        lines.extend(
            [
                "Route histogram: "
                f"`available={route_contract.get('available', False)}`; "
                "analyzer_blocking="
                f"`{route_contract.get('blocking', True)}`; "
                "publication_evidence_ready="
                f"`{route_contract.get('publication_evidence_ready', False)}`.",
                route_contract.get("reason", "No route-histogram evidence."),
                "",
            ],
        )
        stage_rows = []
        for rank, data in report["ranks"].items():
            stages = data.get("layers", {}).get(layer, {})
            for stage in _markdown_stage_order(profile):
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
                            f"{aic['slice_duration_us_p50']:.1f}/"
                            f"{aic['slice_duration_us_p90']:.1f}/"
                            f"{aic['slice_duration_us_p99']:.1f}/"
                            f"{aic['slice_duration_us_max']:.1f}"
                            if aic.get("available")
                            else "-"
                        ),
                        (
                            f"{aiv['slice_duration_us_p50']:.1f}/"
                            f"{aiv['slice_duration_us_p90']:.1f}/"
                            f"{aiv['slice_duration_us_p99']:.1f}/"
                            f"{aiv['slice_duration_us_max']:.1f}"
                            if aiv.get("available")
                            else "-"
                        ),
                        duration_gate_text(aic),
                        duration_gate_text(aiv),
                        (f"{aic.get('peak_concurrency', 0)}/{aiv.get('peak_concurrency', 0)}"),
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
                    "AIC p50/p90/p99/max us",
                    "AIV p50/p90/p99/max us",
                    "AIC <10/10-30/>30",
                    "AIV <10/10-30/>30",
                    "peak AIC/AIV",
                ],
            ),
        )

        lines.extend(["", f"### {layer} all-rank arrivals", ""])
        for label, _producer_stage, _consumer_stage in _arrival_pairs(
            profile
        ):
            arrival = report["arrivals"].get(layer, {}).get(label)
            if not arrival:
                continue
            producer_skew = arrival.get("producer_end_skew_us")
            producer_skew_text = f"{producer_skew:.1f} us" if producer_skew is not None else "not comparable"
            latest_producer = (
                f"`{arrival['latest_producer_rank']}`"
                if arrival.get("latest_producer_rank") is not None
                else "`not comparable`"
            )
            lines.append(
                f"- **{label}** producer end skew: "
                f"`{producer_skew_text}`; "
                f"latest producer: {latest_producer}; "
                f"consumer: `{arrival['consumer_stage']}`.",
            )
            arrival_rows = []
            producer_by_rank = {item["rank"]: item for item in arrival["producer_ranks"]}
            for consumer in arrival["consumer_ranks"]:
                producer = producer_by_rank.get(consumer["rank"], {})
                queue_delay = consumer["timing_evidence"]["queue_delay"]
                arrival_rows.append(
                    [
                        consumer["rank"],
                        f"{producer.get('span_us', 0.0):.1f}",
                        f"{consumer['span_us']:.1f}",
                        (
                            f"{consumer['remote_arrival_after_consumer_start_us']:.1f}"
                            if "remote_arrival_after_consumer_start_us"
                            in consumer
                            else "-"
                        ),
                        str(consumer["has_producer_dependency"]),
                        ",".join(consumer["producer_dependency_sources"])
                        or "-",
                        (f"{queue_delay['value_us']:.1f}" if queue_delay.get("available") else "blocked"),
                        (
                            f"{consumer['wait_overlap_completion_upper_bound_us']:.1f}"
                            if "wait_overlap_completion_upper_bound_us"
                            in consumer
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
                        "consumer span us",
                        "remote arrival after consumer start us",
                        "producer dep",
                        "dep sources",
                        "local queue delay us",
                        "completion saving upper bound us",
                    ],
                ),
            )
            lines.append("")

        lines.extend(["", f"### {layer} execution-limit classification", ""])
        classification_rows = []
        for rank, rank_data in report["execution_limit_classification"]["coverage"][
            layer
        ].items():
            route_empty = rank_data["route_empty"]["classification"]
            for stage, stage_data in rank_data["stages"].items():
                classification_rows.append(
                    [
                        rank,
                        stage,
                        stage_data["resource"],
                        route_empty,
                        stage_data["insufficient_ready_parallelism"][
                            "classification"
                        ],
                        stage_data["queue_delay"]["classification"],
                        stage_data["scheduler_packing"]["classification"],
                    ]
                )
        lines.extend(
            _markdown_table(
                classification_rows,
                [
                    "rank",
                    "stage",
                    "resource",
                    "route-empty",
                    "ready parallelism",
                    "queue delay",
                    "scheduler packing",
                ],
            )
        )

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
    recv_meta_sidecar: Path | None = None,
    profile: str | None = None,
    source_decode_sha256: str | None = None,
) -> dict[str, Any]:
    """Analyze one compiled L0-L4 DFX directory and write JSON/Markdown."""
    resolved_profile = _resolve_profile(profile)
    source_identity = _source_identity_contract(
        resolved_profile,
        source_decode_sha256,
    )
    if (
        source_identity["pass"] is not True
        and _source_policy(resolved_profile)["enforce_candidate_release_gate"]
    ):
        raise RuntimeError(
            "source policy identity unavailable or mismatched: "
            f"actual={source_identity['actual_decode_sha256']} "
            f"policy={source_identity['policy_id']} "
            "expected="
            f"{source_identity['expected_decode_sha256']}"
        )
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
            raise RuntimeError(
                "critical_path failed with return code "
                f"{process.returncode}; see "
                f"{out / 'critical_path_stdout.txt'}"
            )

    traces = [
        trace
        for path in sorted(dfx_root.rglob(_CHIP_SWIMLANE_RECORDS_NAME))
        if (trace := _load_rank(path.parent, dfx_root)) is not None
    ]
    if len(traces) != _EXPECTED_RANKS:
        raise RuntimeError(
            f"expected exactly {_EXPECTED_RANKS} complete rank traces under {dfx_root}, found {len(traces)}"
        )
    actual_rank_tags = {trace.tag for trace in traces}
    if actual_rank_tags != _EXPECTED_RANK_TAGS:
        raise RuntimeError(
            "rank trace set mismatch: "
            f"expected={sorted(_EXPECTED_RANK_TAGS)} "
            f"actual={sorted(actual_rank_tags)}"
        )
    missing_critical_path = [trace.tag for trace in traces if trace.critical_path.get("makespan_ms") is None]
    if missing_critical_path:
        raise RuntimeError(f"missing or malformed critical-path report for {missing_critical_path}")
    structural_contracts = _validate_structural_contracts(
        traces,
        resolved_profile,
    )
    ranks = {
        trace.tag: _rank_metrics(trace, resolved_profile)
        for trace in traces
    }
    slice_contract_errors = []
    for rank, rank_data in ranks.items():
        for layer, stages in rank_data.get("layers", {}).items():
            for stage, stage_data in stages.items():
                for resource, metrics in stage_data.get(
                    "resources",
                    {},
                ).items():
                    if metrics["expected_slices"] != metrics["observed_slices"]:
                        slice_contract_errors.append(
                            {
                                "rank": rank,
                                "layer": layer,
                                "stage": stage,
                                "resource": resource,
                                "expected": metrics["expected_slices"],
                                "observed": metrics["observed_slices"],
                            }
                        )
    if slice_contract_errors:
        raise RuntimeError(f"physical slice contract failed: {slice_contract_errors[:8]}")
    reference_rank = min(ranks, key=lambda tag: ranks[tag]["makespan_us"])
    clock_alignment = _clock_alignment(traces)
    arrivals = _arrival_analysis(
        traces,
        ranks,
        clock_alignment,
        resolved_profile,
    )
    route_histogram = _route_histogram_contract(
        recv_meta_sidecar,
        profile=resolved_profile,
        source_decode_sha256=source_decode_sha256,
    )
    local_ep_route_execution = _local_ep_route_execution_contract(
        structural_contracts,
        route_histogram,
        resolved_profile,
    )
    if (
        local_ep_route_execution["available"]
        and local_ep_route_execution["pass"] is not True
    ):
        raise RuntimeError(
            "local-EP route/execution contract failed: "
            + json.dumps(
                local_ep_route_execution["errors"][:8],
                sort_keys=True,
            )
        )
    routed_slice_profiles = _routed_slice_profile_contract(ranks)
    if not routed_slice_profiles["pass"]:
        raise RuntimeError(
            f"routed physical-slice profile contract failed: {routed_slice_profiles['errors'][:8]}"
        )
    expert_kernel_release = _expert_kernel_release_contract(
        ranks,
        resolved_profile,
    )
    execution_limits = _execution_limit_classification(
        ranks,
        route_histogram,
        resolved_profile,
    )
    timing_evidence = _timing_evidence_contract(
        ranks,
        resolved_profile,
    )
    admission = _admission_contract(
        route_histogram,
        timing_evidence,
        routed_slice_profiles,
        expert_kernel_release,
    )
    report = {
        "schema": "step3p5.five-layer-moe-dfx.v7",
        "build_dir": str(build_dir),
        "dfx_root": str(dfx_root),
        "profile": resolved_profile,
        "source_policy": _source_policy(resolved_profile),
        "source_identity_contract": source_identity,
        "recv_meta_sidecar": (
            str(recv_meta_sidecar) if recv_meta_sidecar is not None else None
        ),
        "reference_rank": reference_rank,
        "reference_note": (
            (
                "Minimum makespan is a LOW-WAIT heuristic only. Compare "
                "every rank from local_combine_reduce into moe_all_reduce; "
                "the collective span may include peer-arrival spin."
            )
            if resolved_profile == "local-ep"
            else (
                "Minimum makespan is a LOW-WAIT heuristic only. Compare "
                "every rank for dispatch/combine arrival and TP all-reduce "
                "spin."
            )
        ),
        "clock_alignment": clock_alignment,
        "rank_contract": {
            "expected": sorted(_EXPECTED_RANK_TAGS),
            "actual": sorted(actual_rank_tags),
            "exact": True,
        },
        "structural_contracts": structural_contracts,
        "slice_contract": {
            "expected_equals_observed": True,
            "errors": [],
        },
        "routed_slice_profiles": routed_slice_profiles,
        "expert_kernel_release": expert_kernel_release,
        "execution_limit_classification": execution_limits,
        "timing_evidence_contract": timing_evidence,
        "route_histogram": route_histogram,
        "local_ep_route_execution_contract": (
            local_ep_route_execution
        ),
        "external_correctness_contract": _external_correctness_contract(),
        "admission": admission,
        "ranks": ranks,
        "arrivals": arrivals,
        "findings": _aggregate_findings(
            ranks,
            arrivals,
            route_histogram,
            resolved_profile,
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
        recv_meta_sidecar=(
            Path(args.recv_meta_sidecar) if args.recv_meta_sidecar else None
        ),
        profile=args.profile,
        source_decode_sha256=args.source_decode_sha256,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
