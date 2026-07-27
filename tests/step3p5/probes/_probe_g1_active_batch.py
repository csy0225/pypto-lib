# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""PERF-G1 dynamic active-token / active-batch acceptance probe.

本探针严格区分两个概念：

* ``storage_batch``：canonical Main 的物理 storage shape，固定为 16；
* ``active_batch``：本次 invocation 的有效 token 数，运行时覆盖
  ``1/2/8/16``。

因此，本探针不会把 fixed-storage ABI 报告成 fixed effective batch。每个
device case 都在同一个 resident holder 内执行，复用同一份 compiled program
和 KV pool，只改变本次 step 的 active rows、metadata reserve rows 以及
``num_tokens_per_owner``。

模式：

* ``contract``：只做 card-free source contract，输出结构化 JSON；
* ``compile``：编译 canonical ``whole_decode_step3p5``，不 prepare、不运行；
* ``device``：启动 standalone Main KV exporters，resident prepare 一次，
  连续执行 active batch ``1/2/8/16``，检查 hidden、inactive rows、KV
  reserve metadata 和 logical-bound contract。

镜像内示例：

    python -m tests.step3p5.probes._probe_g1_active_batch \
        --mode compile --platform a2a3sim --out /tmp/g1-compile

设备示例：

    python -m tests.step3p5.probes._probe_g1_active_batch \
        --mode device --device 8,9,10,11,12,13,14,15 \
        --ckpt /data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp \
        --out /tmp/g1-device
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
CANONICAL = ROOT / "models" / "step3p5" / "decode_fwd.py"
HOLDER = ROOT / "tools" / "step3p5" / "whole_decode_holder.py"
CONFIG = ROOT / "models" / "step3p5" / "config.py"

SCHEMA = "step3p5.perf_g1.active_batch.v1"
TP = 8
STORAGE_BATCH = 16
TOPK = 8
BLOCK_SIZE = 128
HIDDEN_SIZE = 4096
ACTIVE_BATCHES = (1, 2, 8, 16)
HETEROGENEOUS_OWNER_COUNTS = (1, 2, 8, 1, 4, 0, 3, 2)
HETEROGENEOUS_OWNER_MAX = 8
KV_MAP_VERSION = 3
KV_MAP_LAYOUT = "flat_k_major_v_major_v1"
DEVICE_ROUTE_COUNTER_BLOCKER = (
    "真实 device route-counter telemetry 未提供；hidden rows、"
    "metadata、source contract 或 lowered active-bound 证据都不能证明 "
    "dispatch/combine route gate 的实际执行边界"
)
DEFAULT_CKPT = (
    "/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp"
)
DEFAULT_DEVICES = "8,9,10,11,12,13,14,15"
G1_MANIFEST_NAME = "g1_build_manifest.json"


def _function_source(source: str, tree: ast.AST, name: str) -> tuple[str, int]:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one function {name}, found {len(matches)}")
    node = matches[0]
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise ValueError(f"cannot recover source for {name}")
    return segment, int(node.lineno)


def _function_node(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one function {name}, found {len(matches)}")
    return matches[0]


def _parse_executable_pattern(needle: str) -> ast.stmt:
    """Parse one exact statement/header used by the source contract.

    Loop/branch patterns intentionally describe only the executable header.
    Their bodies are not part of the match; this prevents a parent ``For`` or
    ``If`` from passing merely because its body contains the requested code.
    """
    pattern = str(needle).strip()
    if pattern.startswith(("for ", "if ")) and pattern.endswith(":"):
        pattern = f"{pattern}\n    pass"
    parsed = ast.parse(pattern)
    if len(parsed.body) != 1:
        raise ValueError(f"expected one executable pattern: {needle!r}")
    return parsed.body[0]


def _same_ast(left: ast.AST, right: ast.AST) -> bool:
    return ast.dump(left, include_attributes=False) == ast.dump(
        right,
        include_attributes=False,
    )


def _node_matches_pattern(node: ast.AST, pattern: ast.stmt) -> bool:
    if isinstance(pattern, ast.Assign):
        return (
            isinstance(node, ast.Assign)
            and len(node.targets) == len(pattern.targets)
            and all(
                _same_ast(actual, expected)
                for actual, expected in zip(node.targets, pattern.targets)
            )
            and _same_ast(node.value, pattern.value)
        )
    if isinstance(pattern, ast.AnnAssign):
        return (
            isinstance(node, ast.AnnAssign)
            and _same_ast(node.target, pattern.target)
            and _same_ast(node.annotation, pattern.annotation)
            and _same_ast(node.value, pattern.value)
        )
    if isinstance(pattern, ast.AugAssign):
        return (
            isinstance(node, ast.AugAssign)
            and _same_ast(node.target, pattern.target)
            and type(node.op) is type(pattern.op)
            and _same_ast(node.value, pattern.value)
        )
    if isinstance(pattern, ast.For):
        return (
            isinstance(node, ast.For)
            and _same_ast(node.target, pattern.target)
            and _same_ast(node.iter, pattern.iter)
        )
    if isinstance(pattern, ast.If):
        return isinstance(node, ast.If) and _same_ast(node.test, pattern.test)
    if isinstance(pattern, ast.Expr) and isinstance(pattern.value, ast.Call):
        return isinstance(node, ast.Call) and _same_ast(node, pattern.value)
    return False


def _executable_nodes(function: ast.FunctionDef) -> list[ast.AST]:
    """Walk one function body without entering nested definitions."""

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.nodes: list[ast.AST] = []

        def generic_visit(self, node: ast.AST) -> None:
            self.nodes.append(node)
            super().generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            # The root FunctionDef is handled by the caller.  Any FunctionDef
            # reached here is nested and does not execute as part of the outer
            # function's data path.
            self.nodes.append(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.nodes.append(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.nodes.append(node)

        def visit_Lambda(self, node: ast.Lambda) -> None:
            self.nodes.append(node)

    visitor = _Visitor()
    for statement in function.body:
        visitor.visit(statement)
    return visitor.nodes


def _executable_match(
    function: ast.FunctionDef,
    needle: str,
) -> dict[str, Any]:
    """Match one exact executable AST node.

    Comments, docstrings, function annotations, and a parent node whose body
    happens to contain the pattern cannot satisfy this matcher.
    """
    try:
        pattern = _parse_executable_pattern(needle)
    except (SyntaxError, ValueError) as exc:
        return {
            "present": False,
            "line": None,
            "ast_kind": None,
            "code": None,
            "pattern_error": f"{type(exc).__name__}: {exc}",
        }
    for node in _executable_nodes(function):
        if not _node_matches_pattern(node, pattern):
            continue
        code = ast.unparse(node)
        return {
            "present": True,
            "line": int(getattr(node, "lineno", function.lineno)),
            "ast_kind": type(node).__name__,
            "code": code.splitlines()[0][:500],
        }
    return {
        "present": False,
        "line": None,
        "ast_kind": None,
        "code": None,
    }


def _line_for(segment: str, base_line: int, needle: str) -> int | None:
    for offset, line in enumerate(segment.splitlines()):
        if needle in line:
            return base_line + offset
    return None


def _config_constant(name: str, default: int) -> int:
    """读取 config.py 中的静态 BATCH，避免 compile mode 先 import runtime。"""
    try:
        tree = ast.parse(CONFIG.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                if len(node.targets) != 1:
                    continue
                target = node.targets[0]
                if (
                    isinstance(target, ast.Name)
                    and target.id == name
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, int)
                ):
                    return int(node.value.value)
    except (OSError, SyntaxError):
        pass
    return int(default)


def _owner_count_contract(
    owner_counts: tuple[int, ...] = HETEROGENEOUS_OWNER_COUNTS,
) -> dict[str, Any]:
    """验证异构 owner count 归约为全 rank 一致的 max(active_tokens)。

    这是 G1 的 card-free contract，不是设备 telemetry：canonical graph 会
    在 ``whole_chip_orch`` 中遍历 owner vector 并取 max；各 rank 不能各自
    使用不同 active bound，否则 EP dispatch/combine 的 token 数会失配。
    """
    values = [int(item) for item in owner_counts]
    valid_values = all(0 <= item <= STORAGE_BATCH for item in values)
    owner_count_ok = len(values) == TP
    observed_max = max(values) if values else 0
    return {
        "owner_counts": values,
        "owner_count": len(values),
        "expected_owner_count": TP,
        "valid_range": [0, STORAGE_BATCH],
        "values_in_storage_range": valid_values,
        "observed_max": observed_max,
        "expected_max": HETEROGENEOUS_OWNER_MAX,
        "max_reduction_is_correct": observed_max == HETEROGENEOUS_OWNER_MAX,
        "passed": bool(
            owner_count_ok
            and valid_values
            and observed_max == HETEROGENEOUS_OWNER_MAX
        ),
    }


def _synthetic_reserve_metadata(
    *,
    active_batch: int,
    scheduler_num_blocks: int = 32,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """构造 card-free fixed-storage metadata，专门验证 reserve/domain 隔离。"""
    import torch

    active_batch = int(active_batch)
    scheduler_num_blocks = int(scheduler_num_blocks)
    if not 1 <= active_batch <= STORAGE_BATCH:
        raise ValueError(
            f"active_batch must be in [1,{STORAGE_BATCH}], got {active_batch}"
        )
    physical_num_blocks = scheduler_num_blocks + STORAGE_BATCH - 1
    reserve = {
        "scheduler_num_blocks": scheduler_num_blocks,
        "physical_num_blocks": physical_num_blocks,
        "padding_block_ids": list(
            range(scheduler_num_blocks, physical_num_blocks)
        ),
        "block_size": BLOCK_SIZE,
    }
    seq = torch.ones(STORAGE_BATCH, dtype=torch.int32)
    pos = torch.zeros(STORAGE_BATCH, dtype=torch.int32)
    table = torch.zeros(
        STORAGE_BATCH,
        scheduler_num_blocks,
        dtype=torch.int32,
    )
    slots = torch.zeros(STORAGE_BATCH, dtype=torch.int32)

    active_blocks = torch.arange(active_batch, dtype=torch.int32)
    table[:active_batch, 0] = active_blocks
    slots[:active_batch] = active_blocks * BLOCK_SIZE
    for row, block_id in enumerate(
        reserve["padding_block_ids"][: STORAGE_BATCH - active_batch],
        start=active_batch,
    ):
        table[row, 0] = int(block_id)
        slots[row] = int(block_id) * BLOCK_SIZE

    metadata = _metadata_summary(
        seq_lens=seq,
        positions=pos,
        block_table=table,
        slot_mapping=slots,
        active_batch=active_batch,
        reserve=reserve,
    )
    return seq, pos, table, slots, metadata


def _inactive_reserve_domain_contract() -> dict[str, Any]:
    """对 active=1/2/8/16 做 card-free reserve/domain 检查。"""
    cases: list[dict[str, Any]] = []
    for active_batch in ACTIVE_BATCHES:
        _, _, _, _, metadata = _synthetic_reserve_metadata(
            active_batch=active_batch,
        )
        cases.append(
            {
                "active_batch": active_batch,
                "storage_batch": STORAGE_BATCH,
                "inactive_rows": STORAGE_BATCH - active_batch,
                "metadata": metadata,
                "passed": bool(metadata["passed"]),
            }
        )
    return {
        "storage_batch": STORAGE_BATCH,
        "active_batches": list(ACTIVE_BATCHES),
        "cases": cases,
        "passed": bool(cases) and all(item["passed"] for item in cases),
    }


def _source_contract() -> dict[str, Any]:
    """建立 G1 的静态逻辑边界合同，不把它误称为 device telemetry。"""
    canonical_source = CANONICAL.read_text(encoding="utf-8")
    canonical_tree = ast.parse(canonical_source)
    holder_source = HOLDER.read_text(encoding="utf-8")
    holder_tree = ast.parse(holder_source)

    method_needles: dict[str, tuple[str, ...]] = {
        "_gate": (
            "active_tokens = pl.cast(num_tokens, pl.INDEX)",
            "for tt in pl.range(active_tokens):",
        ),
        "_quant_moe_input": (
            "active_tokens = pl.cast(num_tokens, pl.INDEX)",
            "if active_tokens > BATCH:",
        ),
        "dispatch_step": (
            "active_tokens = pl.cast(num_tokens, pl.INDEX)",
            "for t in pl.range(active_tokens):",
            "for k in pl.range(TOPK):",
        ),
        "combine_step": (
            "active_tokens = pl.cast(num_tokens, pl.INDEX)",
            "if t < active_tokens:",
            "for k in pl.range(TOPK):",
        ),
        "whole_chip_orch": (
            "num_tokens = pl.cast(0, pl.INT32)",
            "for owner_rank in pl.range(n_ranks):",
            "pl.read(num_tokens_per_owner, [owner_rank])",
            "if num_tokens < 0:",
            "if num_tokens > BATCH:",
        ),
    }
    methods: dict[str, Any] = {}
    method_pass = True
    for name, needles in method_needles.items():
        function = _function_node(canonical_tree, name)
        segment, base_line = _function_source(
            canonical_source, canonical_tree, name
        )
        evidence = {
            needle: _executable_match(function, needle)
            for needle in needles
        }
        passed = all(item["present"] for item in evidence.values())
        method_pass = method_pass and passed
        methods[name] = {
            "passed": passed,
            "source_line": base_line,
            "evidence": evidence,
        }

    whole_segment, whole_line = _function_source(
        canonical_source, canonical_tree, "whole_chip_orch"
    )
    whole_function = _function_node(canonical_tree, "whole_chip_orch")
    holder_live, holder_live_line = _function_source(
        holder_source, holder_tree, "set_live_step"
    )
    holder_live_function = _function_node(holder_tree, "set_live_step")
    holder_enter, holder_enter_line = _function_source(
        holder_source, holder_tree, "__enter__"
    )
    holder_enter_function = _function_node(holder_tree, "__enter__")
    dispatch_segment, _ = _function_source(
        canonical_source, canonical_tree, "dispatch_step"
    )
    combine_segment, _ = _function_source(
        canonical_source, canonical_tree, "combine_step"
    )
    top_level_checks = {
        "storage_batch_is_16": _config_constant("BATCH", STORAGE_BATCH)
        == STORAGE_BATCH,
        "v4_arrival_thresholds": (
            "expected=moe_epoch" in dispatch_segment
            and "moe_epoch * n_local_experts" in dispatch_segment
            and "moe_epoch * n_local_experts" in combine_segment
            and "moe_epoch * 2" not in canonical_source
        ),
        "active_batch_cases_are_1_2_8_16": tuple(ACTIVE_BATCHES)
        == (1, 2, 8, 16),
        "active_batch_cases_fit_fixed_storage": all(
            1 <= int(active_batch) <= STORAGE_BATCH
            for active_batch in ACTIVE_BATCHES
        ),
        "canonical_has_runtime_owner_vector": (
            "num_tokens_per_owner" in whole_segment
            and "num_tokens_per_owner" in canonical_source
        ),
        "canonical_owner_vector_storage_is_not_signal_stride": (
            "NUM_TOKENS_STORAGE_I32 = COMM_SIGNAL_STRIDE_I32"
            not in canonical_source
        ),
        "canonical_owner_vector_has_padded_storage": (
            "NUM_TOKENS_STORAGE_I32 = 128" in canonical_source
        ),
        "holder_updates_owner_vector_from_valid_tokens": (
            _executable_match(
                holder_live_function,
                "self.num_tokens_per_owner[: self.tp].fill_(valid_tokens)",
            )["present"]
        ),
        "holder_keeps_fixed_storage_hidden": (
            _executable_match(
                holder_live_function,
                "self.current_hidden[:, :valid_tokens, :] = hidden",
            )["present"]
            and _executable_match(
                holder_live_function,
                "self.current_hidden.zero_()",
            )["present"]
        ),
        "holder_prepares_once_persistent": (
            _executable_match(
                holder_enter_function,
                "self.compiled.prepare(persistent=True)",
            )["present"]
        ),
        "whole_graph_clamps_owner_max_to_storage": (
            _executable_match(
                whole_function,
                "if num_tokens > BATCH:",
            )["present"]
            and _executable_match(
                whole_function,
                "num_tokens = pl.cast(BATCH, pl.INT32)",
            )["present"]
        ),
    }
    owner_count_contract = _owner_count_contract()
    inactive_reserve_contract = _inactive_reserve_domain_contract()
    contract_ok = (
        method_pass
        and all(top_level_checks.values())
        and owner_count_contract["passed"]
        and inactive_reserve_contract["passed"]
    )
    return {
        "passed": contract_ok,
        "classification": (
            "source_contract_only; not device telemetry"
        ),
        "storage_batch": STORAGE_BATCH,
        "effective_active_batches_required": list(ACTIVE_BATCHES),
        "heterogeneous_owner_count_contract": owner_count_contract,
        "inactive_reserve_domain_contract": inactive_reserve_contract,
        "logical_bound_semantics": {
            "gate": (
                "top-k and route metadata iterate active rows; gate expert "
                "column fan-out may retain static physical [16,*] tiles"
            ),
            "dispatch": (
                "histogram/pack/pull/inverse-map logical rows are active; "
                "send/recv windows remain fixed-capacity physical storage"
            ),
            "combine": (
                "route gather iterates active rows and active TOPK routes; "
                "inactive output rows are explicitly zeroed"
            ),
        },
        "top_level_checks": top_level_checks,
        "methods": methods,
        "source_locations": {
            "canonical": str(CANONICAL),
            "holder": str(HOLDER),
            "whole_chip_orch": whole_line,
            "holder_set_live_step": holder_live_line,
            "holder_enter": holder_enter_line,
        },
    }


def _expected_bounds(active_batch: int) -> dict[str, Any]:
    active_batch = int(active_batch)
    routes = active_batch * TOPK
    return {
        "storage_batch": STORAGE_BATCH,
        "active_tokens": active_batch,
        "inactive_storage_rows": STORAGE_BATCH - active_batch,
        "gate": {
            "logical_input_rows": active_batch,
            "topk_output_rows": active_batch,
            "route_records": routes,
            "physical_score_rows": STORAGE_BATCH,
        },
        "dispatch": {
            "logical_histogram_rows": active_batch,
            "logical_pack_rows": active_batch,
            "logical_route_records": routes,
            "logical_pull_route_records": routes,
            "logical_inverse_map_rows": active_batch,
            "physical_send_capacity_routes": STORAGE_BATCH * TOPK,
            "physical_recv_capacity_routes": STORAGE_BATCH * TOPK,
        },
        "combine": {
            "logical_output_rows": active_batch,
            "logical_route_reads": routes,
            "logical_staged_routed_rows_upper_bound": routes,
            "physical_output_rows": STORAGE_BATCH,
        },
    }


def _parse_devices(text: str) -> list[int]:
    values = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if len(values) != TP or len(set(values)) != TP:
        raise ValueError(f"expected {TP} distinct devices, got {values}")
    return values


def _reserve_dict(reserve: Any) -> dict[str, Any]:
    if hasattr(reserve, "as_dict"):
        return {
            str(key): value
            for key, value in dict(reserve.as_dict()).items()
        }
    if isinstance(reserve, dict):
        return dict(reserve)
    raise TypeError(f"unsupported reserve object {type(reserve)!r}")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_build_manifest(
    output_dir: Any,
    *,
    compile_started_at: float,
) -> Path:
    root = Path(str(output_dir)).resolve()
    manifest = {
        "schema": "step3p5.g1.build_provenance.v1",
        "artifact_root": str(root),
        "canonical_source": str(CANONICAL.resolve()),
        "canonical_source_sha256": _sha256_path(CANONICAL),
        "canonical_source_mtime": CANONICAL.stat().st_mtime,
        "compile_started_at": float(compile_started_at),
        "compile_finished_at": time.time(),
    }
    path = root / G1_MANIFEST_NAME
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _build_manifest_report(root: Path) -> dict[str, Any]:
    path = root / G1_MANIFEST_NAME
    if not path.is_file():
        return {
            "path": str(path),
            "available": False,
            "checks": {},
            "passed": False,
            "error": (
                "lowered artifact is not bound to the current canonical "
                "source; stale/mixed build directories are rejected"
            ),
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "path": str(path),
            "available": True,
            "checks": {},
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    checks = {
        "schema": data.get("schema")
        == "step3p5.g1.build_provenance.v1",
        "artifact_root": data.get("artifact_root") == str(root.resolve()),
        "canonical_source": data.get("canonical_source")
        == str(CANONICAL.resolve()),
        "canonical_source_sha256": data.get("canonical_source_sha256")
        == _sha256_path(CANONICAL),
        "compile_after_source": float(
            data.get("compile_finished_at", -1)
        )
        >= CANONICAL.stat().st_mtime,
    }
    return {
        "path": str(path),
        "available": True,
        "manifest": data,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _kv_map_report(
    out: Path,
    *,
    expected_reserve: Any | None = None,
    expected_scheduler_num_blocks: int | None = None,
) -> dict[str, Any]:
    """Fail-closed validation of every standalone Main KV exporter map.

    ``WholeDecodeHolder`` eventually validates these maps while importing
    them, but the holder's row inference intentionally treats a missing rank0
    map as "not an IPC run".  G1 must not inherit that fail-open behavior:
    every rank map is an explicit device-gate input and all reserve/schema
    fields must agree with one another and, when available, the holder's
    ``PaddingReserve``.
    """
    expected = (
        _reserve_dict(expected_reserve)
        if expected_reserve is not None
        else None
    )
    try:
        from tools.step3p5.pypto_kv_ipc import validate_pool_map
    except Exception as exc:  # noqa: BLE001
        return {
            "source": "standalone_device_kv_exporter_maps",
            "required_ranks": list(range(TP)),
            "missing_ranks": [],
            "rank_maps": [],
            "cross_rank_checks": {},
            "holder_reserve": expected,
            "passed": False,
            "error_type": type(exc).__name__,
            "error": f"cannot import strict KV map validator: {exc!r}",
        }
    rank_reports: list[dict[str, Any]] = []
    loaded: list[dict[str, Any]] = []
    missing_ranks: list[int] = []
    for rank in range(TP):
        path = out / f"pypto_kvpool_map.json.rank{rank}"
        item: dict[str, Any] = {
            "rank": rank,
            "path": str(path),
            "present": path.exists(),
        }
        if not path.exists():
            missing_ranks.append(rank)
            item.update(
                {
                    "passed": False,
                    "error": "required KV map is missing",
                }
            )
            rank_reports.append(item)
            continue
        try:
            map_obj = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(map_obj, dict):
                raise ValueError("map root must be a JSON object")
            validated = validate_pool_map(map_obj)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            item.update(
                {
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            rank_reports.append(item)
            continue

        fields = {
            "version": map_obj.get("version"),
            "layout": map_obj.get("layout"),
            "rank": map_obj.get("rank"),
            "tp_world_size": map_obj.get("tp_world_size"),
            "scheduler_num_blocks": map_obj.get("scheduler_num_blocks"),
            "physical_num_blocks": map_obj.get("physical_num_blocks"),
            "reserve_start": map_obj.get("reserve_start"),
            "padding_block_ids": map_obj.get("padding_block_ids"),
            "padding_block_count": map_obj.get("padding_block_count"),
            "block_size": map_obj.get("block_size"),
        }
        try:
            normalized = {
                "version": int(fields["version"]),
                "rank": int(fields["rank"]),
                "tp_world_size": int(fields["tp_world_size"]),
                "scheduler_num_blocks": int(
                    fields["scheduler_num_blocks"]
                ),
                "physical_num_blocks": int(fields["physical_num_blocks"]),
                "reserve_start": int(fields["reserve_start"]),
                "padding_block_count": int(fields["padding_block_count"]),
                "block_size": int(fields["block_size"]),
                "padding_block_ids": [
                    int(value) for value in fields["padding_block_ids"]
                ],
            }
        except (TypeError, ValueError, KeyError) as exc:
            item.update(
                {
                    "fields": fields,
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "error": (
                        "required schema/reserve field is missing or "
                        f"not integer-like: {exc}"
                    ),
                }
            )
            rank_reports.append(item)
            continue

        checks = {
            "strict_schema_validator": (
                int(validated.version) == KV_MAP_VERSION
                and validated.layout == KV_MAP_LAYOUT
            ),
            "schema_version": normalized["version"] == KV_MAP_VERSION,
            "layout": fields["layout"] == KV_MAP_LAYOUT,
            "rank": normalized["rank"] == rank,
            "tp_world_size": normalized["tp_world_size"] == TP,
            "block_size": normalized["block_size"] == BLOCK_SIZE,
            "reserve_start_matches_scheduler": (
                normalized["reserve_start"]
                == normalized["scheduler_num_blocks"]
            ),
            "padding_block_count": (
                normalized["padding_block_count"] == STORAGE_BATCH - 1
            ),
            "padding_block_ids_are_contiguous_tail": (
                normalized["padding_block_ids"]
                == list(
                    range(
                        normalized["scheduler_num_blocks"],
                        normalized["scheduler_num_blocks"]
                        + STORAGE_BATCH
                        - 1,
                    )
                )
            ),
            "physical_capacity_contains_reserve": (
                normalized["physical_num_blocks"]
                >= normalized["scheduler_num_blocks"] + STORAGE_BATCH - 1
            ),
        }
        if expected_scheduler_num_blocks is not None:
            requested_scheduler = int(expected_scheduler_num_blocks)
            checks.update(
                {
                    "scheduler_matches_requested_capacity": (
                        normalized["scheduler_num_blocks"]
                        == requested_scheduler
                    ),
                    "physical_matches_requested_capacity": (
                        normalized["physical_num_blocks"]
                        == requested_scheduler + STORAGE_BATCH - 1
                    ),
                }
            )
        if expected is not None:
            checks.update(
                {
                    "scheduler_matches_holder": (
                        normalized["scheduler_num_blocks"]
                        == int(expected["scheduler_num_blocks"])
                    ),
                    "physical_matches_holder": (
                        normalized["physical_num_blocks"]
                        == int(expected["physical_num_blocks"])
                    ),
                    "reserve_start_matches_holder": (
                        normalized["reserve_start"]
                        == int(expected["reserve_start"])
                    ),
                    "padding_ids_match_holder": (
                        normalized["padding_block_ids"]
                        == [
                            int(value)
                            for value in expected["padding_block_ids"]
                        ]
                    ),
                    "block_size_matches_holder": (
                        normalized["block_size"]
                        == int(expected["block_size"])
                    ),
                }
            )
        item.update(
            {
                "fields": fields,
                "normalized": normalized,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
        rank_reports.append(item)
        loaded.append({"rank": rank, "map": map_obj, "normalized": normalized})

    cross_rank_checks: dict[str, bool] = {}
    if loaded:
        first = loaded[0]["normalized"]
        cross_rank_checks = {
            "all_schema_versions_equal": all(
                item["normalized"]["version"] == first["version"]
                for item in loaded
            ),
            "all_layouts_equal": all(
                item["map"].get("layout") == loaded[0]["map"].get("layout")
                for item in loaded
            ),
            "all_scheduler_blocks_equal": all(
                item["normalized"]["scheduler_num_blocks"]
                == first["scheduler_num_blocks"]
                for item in loaded
            ),
            "all_physical_blocks_equal": all(
                item["normalized"]["physical_num_blocks"]
                == first["physical_num_blocks"]
                for item in loaded
            ),
            "all_reserve_starts_equal": all(
                item["normalized"]["reserve_start"]
                == first["reserve_start"]
                for item in loaded
            ),
            "all_padding_ids_equal": all(
                item["normalized"]["padding_block_ids"]
                == first["padding_block_ids"]
                for item in loaded
            ),
            "all_block_sizes_equal": all(
                item["normalized"]["block_size"] == first["block_size"]
                for item in loaded
            ),
        }
    else:
        cross_rank_checks = {
            "all_schema_versions_equal": False,
            "all_layouts_equal": False,
            "all_scheduler_blocks_equal": False,
            "all_physical_blocks_equal": False,
            "all_reserve_starts_equal": False,
            "all_padding_ids_equal": False,
            "all_block_sizes_equal": False,
        }

    all_checks = [
        item.get("passed", False) for item in rank_reports
    ] + list(cross_rank_checks.values())
    return {
        "source": "standalone_device_kv_exporter_maps",
        "required_ranks": list(range(TP)),
        "missing_ranks": missing_ranks,
        "rank_maps": rank_reports,
        "cross_rank_checks": cross_rank_checks,
        "holder_reserve": expected,
        "expected_scheduler_num_blocks": expected_scheduler_num_blocks,
        "passed": bool(
            len(rank_reports) == TP
            and len(loaded) == TP
            and not missing_ranks
            and all(all_checks)
        ),
    }


def _lowered_active_bound_report(output_dir: Any) -> dict[str, Any]:
    """Inspect compiler output without claiming runtime route telemetry.

    The generated whole-chip orchestration is useful evidence that the
    owner-vector value is read, clamped to fixed storage, and propagated as a
    task scalar.  It does *not* prove which device lanes executed how many
    routes; that limitation is kept explicit in the report.
    """
    root = Path(str(output_dir)) if output_dir else None
    report: dict[str, Any] = {
        "source": "lowered_whole_chip_orchestration",
        "artifact_root": str(root) if root is not None else None,
        "device_runtime_route_telemetry": False,
        "limitation": (
            "lowered IR proves scalar bound propagation only; this probe "
            "does not observe per-device route counters"
        ),
    }
    if root is None or not root.exists():
        report.update(
            {
                "artifact_present": False,
                "passed": False,
                "error": "compiled output directory is unavailable",
            }
        )
        return report
    manifest = _build_manifest_report(root)
    report["build_manifest"] = manifest
    candidates = sorted(root.rglob("whole_chip_orch.cpp"))
    if len(candidates) != 1:
        report.update(
            {
                "artifact_present": bool(candidates),
                "candidates": [str(path) for path in candidates],
                "passed": False,
                "error": (
                    "expected exactly one current whole_chip_orch.cpp, "
                    f"found {len(candidates)}"
                ),
            }
        )
        return report
    path = candidates[0]
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        report.update(
            {
                "artifact_present": False,
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return report

    owner_read = "num_tokens_per_owner" in text
    active_scalar = bool(
        re.search(r"add_scalar\([^;\n]*active_tokens", text)
    )
    clamp = bool(
        re.search(r"\(\s*16\s*<\s*active_tokens", text)
        and re.search(r"active_tokens[^;\n]*=\s*static_cast<int64_t>\(16\)", text)
    )
    active_bound = bool(
        re.search(
            r"active_tokens[^;\n]*static_cast<int64_t>\(num_tokens",
            text,
        )
    )
    evidence_lines = [
        line.strip()
        for line in text.splitlines()
        if (
            "num_tokens_per_owner" in line
            or "active_tokens" in line
            and (
                "static_cast<int64_t>(num_tokens" in line
                or "add_scalar(active_tokens" in line
                or "(16 < active_tokens" in line
            )
        )
    ][:20]
    checks = {
        "owner_vector_read_present": owner_read,
        "active_tokens_bound_from_owner_value": active_bound,
        "active_tokens_clamped_to_storage_16": clamp,
        "active_tokens_propagated_to_tasks": active_scalar,
    }
    report.update(
        {
            "artifact_present": True,
            "artifact": str(path),
            "checks": checks,
            "evidence_lines": evidence_lines,
            "passed": all(checks.values()) and manifest["passed"],
        }
    )
    return report


def _device_route_counter_report() -> dict[str, Any]:
    """声明当前 device probe 没有真实 route-counter telemetry。

    该探针目前只观测 holder 输出、固定 storage metadata 和 owner vector。
    这些证据不能升级为 device dispatch/combine route gate 证据，因此
    device mode 必须 fail-closed。不要用 source/lowered/synthetic 结果填充
    ``observed``，也不要把 hidden row 非零或 metadata 行数当作 route count。
    """
    return {
        "evidence_level": "device_route_counter_telemetry",
        "observed": False,
        "source": None,
        "rank_counter_observed": False,
        "per_kernel_counter_observed": False,
        "dispatch_route_counter_observed": False,
        "combine_route_counter_observed": False,
        "blocker": DEVICE_ROUTE_COUNTER_BLOCKER,
        "passed": False,
    }


def _device_acceptance_report(
    *,
    route_counter: dict[str, Any],
    compile_and_prepare_passed: bool = False,
    behavior_observations_passed: bool = False,
    cases_complete: bool = False,
) -> dict[str, Any]:
    """按证据等级计算 device acceptance，不允许低层证据替代 telemetry。"""
    route_counter_passed = bool(
        route_counter.get("observed")
        and route_counter.get("passed")
        and route_counter.get("dispatch_route_counter_observed")
        and route_counter.get("combine_route_counter_observed")
    )
    blockers: list[str] = []
    if not route_counter_passed:
        blockers.append(DEVICE_ROUTE_COUNTER_BLOCKER)
    if not compile_and_prepare_passed:
        blockers.append("compile/prepare/lowered artifact 前置证据未通过")
    if not behavior_observations_passed:
        blockers.append(
            "device hidden/metadata/owner behavior observations 未通过"
        )
    if not cases_complete:
        blockers.append(
            "active batch 1/2/8/16 device case 未全部完成"
        )
    ok = bool(
        route_counter_passed
        and compile_and_prepare_passed
        and behavior_observations_passed
        and cases_complete
    )
    return {
        "status": "PASS" if ok else "NO-GO",
        "ok": ok,
        "route_counter_telemetry": route_counter,
        "compile_and_prepare_passed": bool(compile_and_prepare_passed),
        "behavior_observations_passed": bool(
            behavior_observations_passed
        ),
        "cases_complete": bool(cases_complete),
        "blockers": blockers,
    }


def _metadata_summary(
    *,
    seq_lens: Any,
    positions: Any,
    block_table: Any,
    slot_mapping: Any,
    active_batch: int,
    reserve: Any,
) -> dict[str, Any]:
    """检查固定 storage metadata 中 active/domain 与 reserve/domain 的隔离。"""
    import torch

    active_batch = int(active_batch)
    reserve_obj = _reserve_dict(reserve)
    scheduler = int(reserve_obj["scheduler_num_blocks"])
    physical = int(reserve_obj["physical_num_blocks"])
    reserve_ids = {
        int(item) for item in reserve_obj["padding_block_ids"]
    }
    seq = seq_lens.to(torch.int32).flatten().cpu()
    pos = positions.to(torch.int32).flatten().cpu()
    table = block_table.to(torch.int32).cpu()
    slots = slot_mapping.to(torch.int32).flatten().cpu()
    active_positions = pos[:active_batch].tolist()
    active_seq = seq[:active_batch].tolist()
    active_block_ids: list[int] = []
    for row in range(active_batch):
        col = int(pos[row]) // BLOCK_SIZE
        active_block_ids.append(int(table[row, col]))
    inactive_block_ids = [
        int(item) for item in table[active_batch:, 0].tolist()
    ]
    active_slots = [int(item) for item in slots[:active_batch].tolist()]
    inactive_slots = [int(item) for item in slots[active_batch:].tolist()]
    active_scheduler_domain = all(
        0 <= block < scheduler for block in active_block_ids
    )
    active_slot_domain = all(
        0 <= slot < scheduler * BLOCK_SIZE for slot in active_slots
    )
    inactive_reserve_domain = all(
        block in reserve_ids for block in inactive_block_ids
    )
    inactive_slot_domain = all(
        slot // BLOCK_SIZE in reserve_ids
        and slot % BLOCK_SIZE == 0
        for slot in inactive_slots
    )
    reserve_domain_is_physical_tail = (
        reserve_ids == set(range(scheduler, physical))
    )
    no_active_inactive_alias = not (
        set(active_slots).intersection(inactive_slots)
        or set(active_block_ids).intersection(inactive_block_ids)
    )
    shape_ok = (
        tuple(seq.shape) == (STORAGE_BATCH,)
        and tuple(pos.shape) == (STORAGE_BATCH,)
        and tuple(slots.shape) == (STORAGE_BATCH,)
        and table.ndim == 2
        and table.shape[0] == STORAGE_BATCH
    )
    padding_seq_ok = bool(
        torch.equal(
            seq[active_batch:],
            torch.ones(STORAGE_BATCH - active_batch, dtype=torch.int32),
        )
    )
    padding_pos_ok = bool(torch.count_nonzero(pos[active_batch:]).item() == 0)
    return {
        "storage_shapes": {
            "seq_lens": list(seq.shape),
            "positions": list(pos.shape),
            "block_table": list(table.shape),
            "slot_mapping": list(slots.shape),
        },
        "active_rows": {
            "count": active_batch,
            "seq_lens": active_seq,
            "positions": active_positions,
            "block_ids_at_position": active_block_ids,
            "slot_mapping": active_slots,
            "scheduler_domain": active_scheduler_domain,
            "slot_scheduler_domain": active_slot_domain,
        },
        "inactive_rows": {
            "count": STORAGE_BATCH - active_batch,
            "seq_lens_are_one": padding_seq_ok,
            "positions_are_zero": padding_pos_ok,
            "reserve_block_ids": inactive_block_ids,
            "reserve_domain": inactive_reserve_domain,
            "slot_mapping": inactive_slots,
            "slot_reserve_domain": inactive_slot_domain,
        },
        "reserve": reserve_obj,
        "kv_capacity": {
            "scheduler_num_blocks": scheduler,
            "physical_num_blocks": physical,
            "reserve_block_count": len(reserve_ids),
            "physical_minus_scheduler": physical - scheduler,
            "reserve_is_contiguous_physical_tail": (
                reserve_domain_is_physical_tail
            ),
        },
        "no_active_inactive_alias": no_active_inactive_alias,
        "shape_ok": shape_ok,
        "passed": bool(
            shape_ok
            and active_scheduler_domain
            and active_slot_domain
            and inactive_reserve_domain
            and inactive_slot_domain
            and reserve_domain_is_physical_tail
            and no_active_inactive_alias
            and padding_seq_ok
            and padding_pos_ok
        ),
    }


def _owner_vector_summary(holder: Any, active_batch: int) -> dict[str, Any]:
    import torch

    values = holder.num_tokens_per_owner.detach().cpu().to(torch.int32)
    owners = [int(item) for item in values[:TP].tolist()]
    tail_nonzero = int(torch.count_nonzero(values[TP:]).item())
    return {
        "storage_shape": list(values.shape),
        "owner_count": TP,
        "owner_values": owners,
        "owner_max": max(owners) if owners else 0,
        "expected_owner_value": int(active_batch),
        "tail_nonzero_count": tail_nonzero,
        "passed": (
            len(values) >= TP
            and owners == [int(active_batch)] * TP
            and tail_nonzero == 0
        ),
    }


def _hidden_summary(hidden: Any, active_batch: int) -> dict[str, Any]:
    import torch

    active_batch = int(active_batch)
    tensor = hidden.detach().cpu().to(torch.bfloat16)
    shape = [int(item) for item in tensor.shape]
    exact_shape = shape == [TP, STORAGE_BATCH, HIDDEN_SIZE]
    if not exact_shape:
        return {
            "shape": shape,
            "storage_batch": (
                int(tensor.shape[1]) if tensor.ndim >= 2 else None
            ),
            "active_batch": active_batch,
            "hidden_shape_exact": exact_shape,
            "active_hidden_finite": False,
            "active_hidden_nonzero_rank_rows": None,
            "expected_active_nonzero_rank_rows": TP * active_batch,
            "active_hidden_abs_max": None,
            "inactive_hidden_finite": False,
            "inactive_hidden_nonzero_rank_rows": None,
            "inactive_hidden_abs_max": None,
            "inactive_rows_exact_zero": False,
            "hidden_tp_spread": None,
            "passed": False,
        }
    active = tensor[:, :active_batch].float()
    inactive = tensor[:, active_batch:].float()
    active_row_max = active.abs().amax(dim=-1)
    active_nonzero = int(torch.count_nonzero(active_row_max > 0).item())
    active_finite = bool(torch.isfinite(active).all().item())
    if inactive.numel():
        inactive_abs_max = float(inactive.abs().max().item())
        inactive_nonzero = int(
            torch.count_nonzero(inactive.abs().amax(dim=-1) > 0).item()
        )
        inactive_finite = bool(torch.isfinite(inactive).all().item())
    else:
        inactive_abs_max = 0.0
        inactive_nonzero = 0
        inactive_finite = True
    tp_spread = float(
        (
            active - active[0:1]
        ).abs().max().item()
    )
    return {
        "shape": shape,
        "storage_batch": int(tensor.shape[1]) if tensor.ndim >= 2 else None,
        "active_batch": active_batch,
        "hidden_shape_exact": exact_shape,
        "active_hidden_finite": active_finite,
        "active_hidden_nonzero_rank_rows": active_nonzero,
        "expected_active_nonzero_rank_rows": TP * active_batch,
        "active_hidden_abs_max": float(active.abs().max().item()),
        "inactive_hidden_finite": inactive_finite,
        "inactive_hidden_nonzero_rank_rows": inactive_nonzero,
        "inactive_hidden_abs_max": inactive_abs_max,
        "inactive_rows_exact_zero": inactive_abs_max == 0.0,
        "hidden_tp_spread": tp_spread,
        "passed": bool(
            exact_shape
            and active_finite
            and inactive_finite
            and active_nonzero == TP * active_batch
            and inactive_abs_max == 0.0
        ),
    }


def _write_report(out: Path, report: dict[str, Any]) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "g1_active_batch_report.json"
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    print(f"G1_REPORT={path}")
    return path


def _base_report(mode: str, contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "mode": mode,
        "product_entry": "models.step3p5.decode_fwd:whole_decode_step3p5",
        "storage_batch": STORAGE_BATCH,
        "effective_active_batches_required": list(ACTIVE_BATCHES),
        "fixed_storage_is_not_fixed_effective_batch": True,
        "evidence_policy": {
            "contract": (
                "source/synthetic contract only; may pass independently"
            ),
            "compile": (
                "compile/lowered artifact evidence only; may pass "
                "independently"
            ),
            "device_behavior": (
                "hidden/metadata/owner observations do not prove route "
                "execution bounds"
            ),
            "device_route_counter": (
                "required for device acceptance; missing means NO-GO"
            ),
        },
        "logical_bound_contract": contract,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _run_contract(args: argparse.Namespace) -> int:
    contract = _source_contract()
    report = _base_report("contract", contract)
    report.update(
        {
            "ok": bool(contract["passed"]),
            "execution": {
                "card_required": False,
                "device_invocations": 0,
                "compile": False,
            },
        }
    )
    _write_report(Path(args.out), report)
    return 0 if report["ok"] else 1


def _configure_compile_env(out: Path, num_blocks: int) -> None:
    physical_blocks = int(num_blocks) + (STORAGE_BATCH - 1)
    os.environ["PYPTO_STEP3P5_MAX_SEQ"] = str(int(num_blocks) * BLOCK_SIZE)
    os.environ["PYPTO_STEP3P5_BLOCK_TABLE_FLAT"] = str(
        STORAGE_BATCH * int(num_blocks)
    )
    os.environ["PYPTO_STEP3P5_KV_CACHE_ROWS"] = str(
        45 * physical_blocks * BLOCK_SIZE
    )
    os.environ["PYPTO_STEP3P5_ROPE_SEQ"] = str(int(num_blocks) * BLOCK_SIZE)
    os.environ["PYPTO_PROG_BUILD_DIR"] = str(out / "build_output")


def _run_compile(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    contract = _source_contract()
    report = _base_report("compile", contract)
    report["execution"] = {
        "card_required": False,
        "compile": True,
        "platform": args.platform,
        "device_ids": _parse_devices(args.device),
        "build_dir": str(out / "build_output"),
        "active_batch_runtime_cases": list(ACTIVE_BATCHES),
    }
    try:
        _configure_compile_env(out, args.num_blocks)
        from tools.step3p5.whole_decode_holder import WholeDecodeHolder

        compile_started_at = time.time()
        holder = WholeDecodeHolder(
            device_ids=_parse_devices(args.device),
            out_dir=str(out),
            ckpt=args.ckpt,
            platform=args.platform,
            kv_ipc=False,
        ).build()
        build_manifest = _write_build_manifest(
            holder.compiled.output_dir,
            compile_started_at=compile_started_at,
        )
        compiled_storage_batch = int(holder._consts["BATCH"])
        compiled_kv_rows = int(holder._consts["KVC"])
        expected_kv_rows = (
            45 * (int(args.num_blocks) + STORAGE_BATCH - 1) * BLOCK_SIZE
        )
        lowered_active_bound = _lowered_active_bound_report(
            holder.compiled.output_dir
        )
        compile_checks = {
            "canonical_program": (
                holder.program_name == "whole_decode_step3p5"
            ),
            "storage_batch": compiled_storage_batch == STORAGE_BATCH,
            "kv_rows": compiled_kv_rows == expected_kv_rows,
            "lowered_active_bound_propagation": bool(
                lowered_active_bound["passed"]
            ),
        }
        report["compile_result"] = {
            "passed": all(compile_checks.values()),
            "compiled_output_dir": str(holder.compiled.output_dir),
            "canonical_program": holder.program_name,
            "compiled_storage_batch": compiled_storage_batch,
            "compiled_kv_rows": compiled_kv_rows,
            "expected_kv_rows": expected_kv_rows,
            "build_manifest": str(build_manifest),
            "checks": compile_checks,
            "lowered_active_bound": lowered_active_bound,
        }
        report["ok"] = bool(
            contract["passed"]
            and report["compile_result"]["passed"]
        )
    except Exception as exc:  # noqa: BLE001
        report["compile_result"] = {
            "passed": False,
            "error_type": type(exc).__name__,
            "error": repr(exc),
        }
        report["ok"] = False
    _write_report(out, report)
    return 0 if report["ok"] else 1


def _run_device(args: argparse.Namespace) -> int:
    import torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    devices = _parse_devices(args.device)
    contract = _source_contract()
    report = _base_report("device", contract)
    report["execution"] = {
        "card_required": True,
        "compile": True,
        "platform": args.platform,
        "device_ids": devices,
        "resident_holder": True,
        "invocations": list(ACTIVE_BATCHES),
        "same_storage_batch_for_all_cases": STORAGE_BATCH,
        "metadata_step_per_case": {
            str(active): index
            for index, active in enumerate(ACTIVE_BATCHES)
        },
    }
    route_counter = _device_route_counter_report()
    report["device_acceptance"] = _device_acceptance_report(
        route_counter=route_counter,
    )
    reports: list[dict[str, Any]] = []
    procs: list[Any] = []
    try:
        _configure_compile_env(out, args.num_blocks)
        # 只复用现有诊断 exporter 的 graceful lifecycle，不在 probe 内
        # 自己 kill device 进程；device launch 前的卡清理仍由 runbook 负责。
        from tests.step3p5.harnesses._stage_main_hidden_only import (
            _start_exporters,
            _stop_exporters,
            _step_metadata,
            _load_embedding_row,
        )

        if not args.reuse_exporters:
            export_args = SimpleNamespace(
                out=str(out),
                ckpt=args.ckpt,
                num_blocks=args.num_blocks,
                kv_probe=False,
            )
            procs = _start_exporters(export_args, devices)
        else:
            ready = all(
                (out / f"ready.rank{rank}").exists()
                for rank in range(TP)
            )
            if not ready:
                raise RuntimeError(
                    "--reuse-exporters requested but exporter readiness "
                    "markers are incomplete"
                )

        initial_kv_map_report = _kv_map_report(
            out,
            expected_scheduler_num_blocks=args.num_blocks,
        )
        report["kv_maps_before_holder"] = initial_kv_map_report
        if not initial_kv_map_report["passed"]:
            raise RuntimeError(
                "required rank0..rank7 Main KV maps are missing or "
                "inconsistent; see kv_maps_before_holder"
            )

        from tools.step3p5.whole_decode_holder import WholeDecodeHolder

        holder = WholeDecodeHolder(
            device_ids=devices,
            out_dir=str(out),
            ckpt=args.ckpt,
            platform=args.platform,
            kv_ipc=True,
        ).build()
        build_manifest = _write_build_manifest(
            holder.compiled.output_dir,
            compile_started_at=time.time(),
        )
        report["build_manifest"] = str(build_manifest)
        embedding = _load_embedding_row(args.ckpt, args.seed_token)
        with holder:
            fixed_storage_ok = int(holder._consts["BATCH"]) == STORAGE_BATCH
            if not fixed_storage_ok:
                raise AssertionError(
                    "canonical compiled storage batch is not 16: "
                    f"{holder._consts['BATCH']}"
                )
            reserve = _reserve_dict(holder.padding_reserve)
            expected_kv_rows = (
                45 * int(reserve["physical_num_blocks"]) * BLOCK_SIZE
            )
            compiled_kv_rows = int(holder._consts["KVC"])
            map_summary = _kv_map_report(
                out,
                expected_reserve=reserve,
                expected_scheduler_num_blocks=args.num_blocks,
            )
            if not map_summary["passed"]:
                raise RuntimeError(
                    "rank0..rank7 Main KV maps do not match the resident "
                    "holder reserve; see kv_maps"
                )
            lowered_active_bound = _lowered_active_bound_report(
                holder.compiled.output_dir
            )

            for case_index, active_batch in enumerate(ACTIVE_BATCHES):
                seq, pos, table, slot = _step_metadata(
                    step=case_index,
                    scheduler_num_blocks=args.num_blocks,
                    valid_rows=active_batch,
                )
                holder.set_live_step(
                    embedding.unsqueeze(0).expand(
                        active_batch, -1
                    ).contiguous(),
                    seq_lens=seq,
                    positions=pos,
                    block_table=table,
                    slot_mapping=slot,
                )
                owner = _owner_vector_summary(holder, active_batch)
                metadata = _metadata_summary(
                    seq_lens=seq,
                    positions=pos,
                    block_table=table,
                    slot_mapping=slot,
                    active_batch=active_batch,
                    reserve=holder.padding_reserve,
                )
                started = time.time()
                result = holder.run()
                elapsed = time.time() - started
                hidden = result["next_hidden"]
                hidden_report = _hidden_summary(hidden, active_batch)
                bounds = _expected_bounds(active_batch)
                bounds["device_observed"] = {
                    "output_storage_shape": hidden_report["shape"],
                    "output_active_rows": active_batch,
                    "output_inactive_rows": STORAGE_BATCH - active_batch,
                    "owner_max": owner["owner_max"],
                }
                bounds["contract_derived"] = {
                    "dispatch_and_combine_route_upper_bound": (
                        active_batch * TOPK
                    ),
                    "not_device_route_telemetry": True,
                }
                bounds["evidence"] = {
                    "source_contract": (
                        "card-free AST/source contract; not device telemetry"
                    ),
                    "lowered_active_bound": lowered_active_bound,
                    "device_output": (
                        "active/inactive output row behavior is observed"
                    ),
                    "device_route_counter_observed": False,
                    "limitation": (
                        "no per-device gate/dispatch/combine route counter "
                        "is exposed; exact route iteration counts are not "
                        "reported as device telemetry"
                    ),
                }
                behavior_checks = {
                    "fixed_storage_shape": bool(
                        hidden_report["hidden_shape_exact"]
                    ),
                    "num_tokens_per_owner": bool(owner["passed"]),
                    "active_hidden": bool(
                        hidden_report["active_hidden_finite"]
                        and hidden_report[
                            "active_hidden_nonzero_rank_rows"
                        ]
                        == TP * active_batch
                    ),
                    "inactive_rows": bool(
                        hidden_report["inactive_hidden_finite"]
                        and hidden_report["inactive_rows_exact_zero"]
                    ),
                    "kv_reserve_metadata": bool(
                        metadata["passed"]
                        and compiled_kv_rows == expected_kv_rows
                        and map_summary["passed"]
                    ),
                }
                evidence_checks = {
                    "source_contract_available": bool(contract["passed"]),
                    "lowered_active_bound_artifact_available": bool(
                        lowered_active_bound["passed"]
                    ),
                    "device_route_counter_telemetry": bool(
                        route_counter["observed"]
                    ),
                }
                checks = {
                    **behavior_checks,
                    **evidence_checks,
                }
                case_report = {
                    "active_batch": active_batch,
                    "storage_batch": STORAGE_BATCH,
                    "run_sec": elapsed,
                    "num_tokens_per_owner": owner,
                    "metadata": metadata,
                    "kv_map": map_summary,
                    "logical_bounds": bounds,
                    "hidden": hidden_report,
                    "behavior_observations_passed": all(
                        behavior_checks.values()
                    ),
                    "evidence_checks": evidence_checks,
                    "acceptance_blockers": [DEVICE_ROUTE_COUNTER_BLOCKER],
                    "checks": checks,
                    "passed": all(checks.values()),
                }
                reports.append(case_report)
                print(
                    json.dumps(
                        {
                            "schema": SCHEMA,
                            "mode": "device",
                            "active_batch": active_batch,
                            "passed": case_report["passed"],
                            "checks": checks,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        report["cases"] = reports
        report["compile_and_prepare"] = {
            "passed": bool(
                int(holder._consts["BATCH"]) == STORAGE_BATCH
                and compiled_kv_rows == expected_kv_rows
                and map_summary["passed"]
                and lowered_active_bound["passed"]
            ),
            "compiled_storage_batch": STORAGE_BATCH,
            "compiled_kv_rows": compiled_kv_rows,
            "expected_kv_rows": expected_kv_rows,
            "kv_reserve": reserve,
            "kv_maps": map_summary,
            "lowered_active_bound": lowered_active_bound,
            "device_route_counter_observed": False,
            "limitation": (
                "device cases observe output rows and resident metadata, "
                "but the runtime exposes no per-kernel route counters"
            ),
        }
        cases_complete = len(reports) == len(ACTIVE_BATCHES)
        behavior_observations_passed = bool(
            cases_complete
            and all(
                item["behavior_observations_passed"]
                for item in reports
            )
        )
        report["device_acceptance"] = _device_acceptance_report(
            route_counter=route_counter,
            compile_and_prepare_passed=bool(
                contract["passed"]
                and report["compile_and_prepare"]["passed"]
            ),
            behavior_observations_passed=behavior_observations_passed,
            cases_complete=cases_complete,
        )
        report["ok"] = bool(report["device_acceptance"]["ok"])
    except Exception as exc:  # noqa: BLE001
        report["cases"] = reports
        report["ok"] = False
        report["error"] = {
            "error_type": type(exc).__name__,
            "error": repr(exc),
        }
    finally:
        if procs:
            from tests.step3p5.harnesses._stage_main_hidden_only import (
                _stop_exporters,
            )

            _stop_exporters(out, procs)
    _write_report(out, report)
    return 0 if report["ok"] else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PERF-G1: fixed storage batch=16 with dynamic active batch "
            "acceptance for 1/2/8/16"
        )
    )
    parser.add_argument(
        "--mode",
        choices=("contract", "compile", "device"),
        default="contract",
    )
    parser.add_argument("--out", default="/tmp/step3p5-g1-active-batch")
    parser.add_argument("--device", default=DEFAULT_DEVICES)
    parser.add_argument("--platform", choices=("a2a3", "a2a3sim"), default="a2a3")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--num-blocks", type=int, default=32)
    parser.add_argument("--seed-token", type=int, default=6127)
    parser.add_argument("--reuse-exporters", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.mode == "contract":
        return _run_contract(args)
    if args.mode == "compile":
        return _run_compile(args)
    return _run_device(args)


if __name__ == "__main__":
    raise SystemExit(main())
