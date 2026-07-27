# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""PERF-B3 resident KV-pool contract / compile / device probe.

本探针只验证 canonical Main 的 resident KV 生命周期和 KV 行证据，不改
production code：

* ``WholeDecodeHolder.build()`` / ``prepare(persistent=True)`` / KV IPC import /
  ``build_stacked_kv_pool()`` 只做一次；
* 同一个 holder 连续执行至少 6 次 invocation；
* exporter 侧在每次 invocation 前后分块 D2H 扫描完整 45 layers × 8
  ranks × K/V × all physical slots，并校验快照真实 SHA-256；
* 每轮只允许本轮完整 storage-batch ``slot_mapping`` 指向的物理行变化；
* invocation 0 写 slot0；invocation 1/2 写 slot1，随后切到 slot3+，
  显式证明 slot1 在不再映射时稳定；slot0 成为历史后不再映射，slot2
  始终不映射且保持未写；
* 相邻 invocation 之间也要求全池不发生异步漂移；
* 设备或 exporter 忙时只输出 ``NO-GO`` 报告并 graceful 停止，不使用
  ``SIGKILL`` / ``kill -9``。

``contract`` 是 card-free；``compile`` / ``device`` 只是提供交付入口，本次
开发验证不自动调用它们。设备忙时，``device`` 模式会在 exporter timeout 或
子进程提前退出后 graceful NO-GO。

示例：

    python -m tests.step3p5.probes._probe_b3_resident_kv \
        --mode contract --out /tmp/b3-contract

    python -m tests.step3p5.probes._probe_b3_resident_kv \
        --mode compile --platform a2a3sim --out /tmp/b3-compile

    python -m tests.step3p5.probes._probe_b3_resident_kv \
        --mode device --device 8,9,10,11,12,13,14,15 \
        --ckpt /data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp \
        --out /tmp/b3-device --invocations 6
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
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
CANONICAL = ROOT / "models" / "step3p5" / "decode_fwd.py"
HOLDER = ROOT / "tools" / "step3p5" / "whole_decode_holder.py"
HARNESS = ROOT / "tests" / "step3p5" / "harnesses" / "_stage_main_hidden_only.py"
EXPORTER = ROOT / "tools" / "step3p5" / "main_kv_exporter.py"

TP = 8
NUM_LAYERS = 45
BLOCK_SIZE = 128
DEFAULT_NUM_BLOCKS = 32
DEFAULT_SCAN_CHUNK_ROWS = 8192
MIN_INVOCATIONS = 6
KV_SECTIONS = ("K", "V")
KV_SLOTS = (0, 1, 2)
DEFAULT_CKPT = (
    "/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp"
)
DEFAULT_DEVICES = "8,9,10,11,12,13,14,15"
SCHEMA = "step3p5.perf_b3.resident_kv.v1"
FULL_SCAN_KIND = "full_pool_row_diff_v1"
ROW_BYTES = 128 * 2


def _parse_devices(text: str) -> list[int]:
    values = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if len(values) != TP or len(set(values)) != TP:
        raise ValueError(f"expected {TP} distinct devices, got {values}")
    return values


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _metadata_slot_values(metadata: tuple[Any, Any, Any, Any]) -> list[int]:
    """Extract the complete storage-batch slot_mapping used by the holder."""
    slot = metadata[3]
    try:
        values = slot.reshape(-1).tolist()
    except AttributeError as exc:
        raise ValueError("slot_mapping must be tensor-like") from exc
    result = [int(value) for value in values]
    if not result:
        raise ValueError("slot_mapping is empty")
    return result


def _invocation_metadata_step(invocation_index: int) -> int:
    """Return the explicit B3 watched-slot state-machine step.

    The first three invocations are intentionally ``0, 1, 1``: write slot0
    once, write slot1 once, then repeat the exact slot1 invocation to prove
    bitwise stability.  Later invocations move to slot3+ so slot1 becomes
    historical state and must remain immutable.  Slot2 is never mapped.
    """
    invocation_index = int(invocation_index)
    if invocation_index < 0:
        raise ValueError(
            f"invocation_index must be non-negative, got {invocation_index}"
        )
    if invocation_index == 0:
        return 0
    if invocation_index in (1, 2):
        return 1
    return invocation_index


def _pool_map_digest(pool_map: dict[str, Any]) -> str:
    payload = json.dumps(
        pool_map,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_pool_maps(out: Path, devices: list[int]) -> list[dict[str, Any]]:
    """Load the eight allocation-owner maps and reject incomplete ownership."""
    expected_entries = {
        f"L{layer}.{which}"
        for layer in range(NUM_LAYERS)
        for which in KV_SECTIONS
    }
    maps: list[dict[str, Any]] = []
    for rank in range(TP):
        path = out / f"pypto_kvpool_map.json.rank{rank}"
        if not path.is_file():
            raise RuntimeError(f"missing rank{rank} KV ownership map: {path}")
        try:
            pool_map = json.loads(_read(path))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid KV ownership map: {path}") from exc
        if not isinstance(pool_map, dict):
            raise RuntimeError(f"KV ownership map is not an object: {path}")
        entries = pool_map.get("map")
        if (
            int(pool_map.get("version", -1)) != 3
            or pool_map.get("layout") != "flat_k_major_v_major_v1"
            or int(pool_map.get("rank", -1)) != rank
            or int(pool_map.get("tp_world_size", -1)) != TP
            or int(pool_map.get("num_layers", -1)) != NUM_LAYERS
            or int(pool_map.get("block_size", -1)) != BLOCK_SIZE
            or not isinstance(entries, dict)
            or set(entries) != expected_entries
        ):
            raise RuntimeError(f"rank{rank} KV ownership contract mismatch: {path}")
        for key in sorted(expected_entries):
            entry = entries[key]
            if (
                not isinstance(entry, dict)
                or entry.get("dtype") != "bfloat16"
                or entry.get("flat_shape", [None, None])[1] != 128
                or int(entry.get("num_slots", -1)) <= 0
            ):
                raise RuntimeError(f"invalid rank{rank} KV entry {key}")
        maps.append(pool_map)

    num_slots = {
        int(pool_map["map"]["L0.K"]["num_slots"]) for pool_map in maps
    }
    pool_bytes = {int(pool_map.get("pool_bytes", -1)) for pool_map in maps}
    if len(num_slots) != 1 or len(pool_bytes) != 1:
        raise RuntimeError("rank KV maps disagree on physical allocation size")
    expected_pool_bytes = NUM_LAYERS * len(KV_SECTIONS) * next(iter(num_slots)) * ROW_BYTES
    if next(iter(pool_bytes)) != expected_pool_bytes:
        raise RuntimeError(
            "KV map pool_bytes does not cover exactly 45 layers × K/V × all rows"
        )
    if len(devices) != TP:
        raise ValueError(f"expected {TP} devices, got {devices}")
    return maps


def _physical_row_key(rank: int, row_index: int, num_slots: int) -> str:
    section_rows = NUM_LAYERS * int(num_slots)
    section, remainder = divmod(int(row_index), section_rows)
    layer, slot = divmod(remainder, int(num_slots))
    if section not in range(len(KV_SECTIONS)) or layer not in range(NUM_LAYERS):
        raise ValueError(f"physical row index out of range: {row_index}")
    return f"rank{rank}.L{layer}.{KV_SECTIONS[section]}.slot{slot}"


def _snapshot_file_digests(
    path: Path,
    *,
    row_count: int,
    chunk_rows: int,
) -> tuple[str, list[str]]:
    """Re-hash the owner sidecar; metadata-only digests are not trusted."""
    if chunk_rows <= 0:
        raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")
    expected_bytes = int(row_count) * ROW_BYTES
    whole = hashlib.sha256()
    chunks: list[str] = []
    with path.open("rb") as file:
        remaining = expected_bytes
        while remaining:
            raw = file.read(min(remaining, int(chunk_rows) * ROW_BYTES))
            if not raw:
                raise ValueError(f"snapshot is truncated: {path}")
            if len(raw) % ROW_BYTES:
                raise ValueError(f"snapshot chunk is not row aligned: {path}")
            whole.update(raw)
            chunks.append(hashlib.sha256(raw).hexdigest())
            remaining -= len(raw)
        if file.read(1):
            raise ValueError(f"snapshot has trailing data: {path}")
    return whole.hexdigest(), chunks


def _snapshot_paths(aggregate: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    ranks = aggregate.get("ranks")
    if not isinstance(ranks, list):
        return paths
    for item in ranks:
        scan = item.get("full_pool") if isinstance(item, dict) else None
        if isinstance(scan, dict) and scan.get("snapshot_path"):
            paths.append(Path(str(scan["snapshot_path"])))
    return paths


def _remove_snapshot_sidecars(aggregate: dict[str, Any]) -> None:
    for path in _snapshot_paths(aggregate):
        path.unlink(missing_ok=True)


def _function_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one function {name!r}, got {len(matches)}")
    segment = ast.get_source_segment(source, matches[0])
    if segment is None:
        raise ValueError(f"cannot recover source for {name!r}")
    return segment


def _class_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one class {name!r}, got {len(matches)}")
    segment = ast.get_source_segment(source, matches[0])
    if segment is None:
        raise ValueError(f"cannot recover source for class {name!r}")
    return segment


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _contract() -> dict[str, Any]:
    """建立 card-free B3 contract；不 import PyPTO、不 prepare device。"""
    probe_path = Path(__file__).resolve()
    canonical_source = _read(CANONICAL)
    holder_source = _read(HOLDER)
    harness_source = _read(HARNESS)
    exporter_source = _read(EXPORTER)
    probe_source = _read(probe_path)
    holder_class = _class_source(holder_source, "WholeDecodeHolder")
    holder_enter = _function_source(holder_class, "__enter__")
    holder_run = _function_source(holder_class, "run")
    harness_collect = _function_source(harness_source, "_collect_kv_probe")
    harness_summary = _function_source(harness_source, "_kv_probe_summary")
    exporter_snapshot_full_pool = _function_source(
        exporter_source, "snapshot_full_pool"
    )
    graceful_stop = _function_source(
        probe_source, "_graceful_stop_exporters"
    )
    device_entry = _function_source(probe_source, "_run_device")
    invocation_entry = _function_source(probe_source, "_run_one_invocation")
    full_pool_diff = _function_source(probe_source, "_full_pool_row_diff")
    allowed_rows = _function_source(probe_source, "_allowed_physical_rows")
    invocation_metadata_step = _function_source(
        probe_source, "_invocation_metadata_step"
    )
    active_slot_nonzero = _function_source(
        probe_source, "_active_slot_nonzero_report"
    )
    snapshot_contract = _function_source(probe_source, "_snapshot_contract")
    snapshot_cleanup = _function_source(
        probe_source, "_cleanup_full_pool_snapshots"
    )

    checks = [
        _check(
            "canonical_product_entry",
            "whole_decode_step3p5 = WholeDecodeStep3p5" in canonical_source,
            "canonical Main entry is whole_decode_step3p5",
        ),
        _check(
            "holder_uses_persistent_prepare",
            "self.compiled.prepare(persistent=True)" in holder_enter,
            "WholeDecodeHolder prepares one persistent runtime",
        ),
        _check(
            "holder_imports_kv_once",
            holder_enter.count("import_kv_all(") == 1,
            f"__enter__ import_kv_all count={holder_enter.count('import_kv_all(')}",
        ),
        _check(
            "holder_builds_stacked_kv_once",
            holder_enter.count("build_stacked_kv_pool(") == 1,
            (
                "whole_decode_holder.__enter__ builds one stacked K/V pool; "
                f"count={holder_enter.count('build_stacked_kv_pool(')}"
            ),
        ),
        _check(
            "holder_binds_inout_kv",
            "self.k_cache, self.v_cache = build_stacked_kv_pool(self._kv_maps)"
            in holder_enter
            and "args += [self.k_cache, self.v_cache]" in holder_source,
            "resident K/V objects are wired once as program InOut arguments",
        ),
        _check(
            "holder_run_reuses_rt",
            "self.rt.run(self.compiled" in holder_run
            and "self.compiled.prepare(" not in holder_run
            and "import_kv_all(" not in holder_run
            and "build_stacked_kv_pool(" not in holder_run,
            "run() reuses the resident runtime and does not re-import KV",
        ),
        _check(
            "kv_probe_covers_all_layers",
            '"layer_indices": list(range(45))' in harness_collect,
            "owner-side probe requests layers 0..44",
        ),
        _check(
            "kv_probe_covers_kv_sections",
            'for which in ("K", "V")' in harness_summary,
            "owner-side summary covers K and V",
        ),
        _check(
            "kv_probe_covers_slots_0_1_2",
            '"slots": [0, 1, 2]' in harness_collect
            and "slot0" in harness_summary
            and "slot1" in harness_summary
            and "slot2" in harness_summary,
            "owner-side probe covers slot0/slot1/slot2",
        ),
        _check(
            "kv_probe_requires_all_ranks",
            "if len(results) != TP:" in harness_collect
            and "range(TP)" in harness_collect,
            "owner-side probe waits for all eight rank responses",
        ),
        _check(
            "full_pool_exporter_scan_available",
            "for chunk_offset in range(0, self._pool_bytes, chunk_bytes)"
            in exporter_snapshot_full_pool
            and '"pool_sha256": pool_digest.hexdigest()'
            in exporter_snapshot_full_pool
            and '"snapshot_path": str(snapshot_path)'
            in exporter_snapshot_full_pool,
            "diagnostic exporter scans every physical pool byte and emits a raw sidecar digest",
        ),
        _check(
            "full_pool_harness_forwarding_available",
            "full_pool: bool = False" in harness_collect
            and '"full_pool": bool(full_pool)' in harness_collect
            and "snapshot_full_pool(" in harness_source,
            "harness forwards fail-closed full-pool scan requests to every owner",
        ),
        _check(
            "kv_probe_requests_full_pool_before_and_after",
            '"full_pool": bool(full_pool)' in harness_collect
            and invocation_entry.count("_collect_full_pool_probe(") == 2
            and 'phase="before"' in invocation_entry
            and 'phase="after"' in invocation_entry,
            "each invocation captures the complete physical pool before and after",
        ),
        _check(
            "full_pool_whitelist_uses_complete_slot_mapping",
            "_metadata_slot_values(metadata)" in invocation_entry
            and "slot_mapping=slot_mapping" in invocation_entry
            and "for slot in slots" in allowed_rows
            and "for layer in range(NUM_LAYERS)" in allowed_rows
            and "for section in range(len(KV_SECTIONS))" in allowed_rows,
            "row whitelist expands the invocation's complete slot_mapping",
        ),
        _check(
            "unauthorized_full_pool_rows_fail_closed",
            "unauthorized_rows += 1" in full_pool_diff
            and "unauthorized_rows == 0" in full_pool_diff
            and '"full_pool_row_diff_whitelist"' in invocation_entry,
            "any changed row outside the complete slot whitelist is a NO-GO",
        ),
        _check(
            "first_write_noop_fails_closed",
            "required_first_write_slot" in full_pool_diff
            and "first_write_unchanged" in full_pool_diff
            and "first_write_after_zero" in full_pool_diff
            and "first_write_slot=" in invocation_entry,
            "first use must leave the zero-initialized active KV rows; "
            "afterwards repeated identical invocations may be stable",
        ),
        _check(
            "pool_identity_is_checked_across_invocations",
            '"pool_base_stable"' in full_pool_diff
            and '"pool_map_stable"' in full_pool_diff
            and '"pool_shape_stable"' in full_pool_diff
            and "baseline_pool_bases" in invocation_entry
            and '"pool_base_stable_across_invocations"' in invocation_entry,
            "pool base/map/shape identity is checked before accepting reuse",
        ),
        _check(
            "full_pool_sidecars_are_removed",
            "finally:" in invocation_entry
            and "_cleanup_full_pool_snapshots(" in invocation_entry
            and "snapshot_path.unlink()" in snapshot_cleanup,
            "raw D2H sidecars are removed after exact row comparison",
        ),
        _check(
            "probe_requires_six_invocations",
            "MIN_INVOCATIONS = 6" in probe_source
            and "args.invocations < MIN_INVOCATIONS" in probe_source,
            "B3 device gate requires at least six invocations",
        ),
        _check(
            "probe_uses_same_holder",
            "with holder:" in device_entry
            and "_run_one_invocation(" in device_entry,
            "all invocations run inside one holder context",
        ),
        _check(
            "probe_full_pool_whitelist",
            "_full_pool_row_diff(" in device_entry
            or "_full_pool_row_diff(" in probe_source
            and "_allowed_physical_rows(" in probe_source
            and "unauthorized_changed_rows" in probe_source,
            "device gate compares all rows and rejects changes outside current slot_mapping",
        ),
        _check(
            "probe_inter_invocation_continuity",
            "previous_after_pool_sha256" in invocation_entry
            and '"inter_invocation_pool_continuity"' in invocation_entry
            and '"after_pool_sha256"' in full_pool_diff,
            "next invocation pre-scan must equal the previous post-scan",
        ),
        _check(
            "slot_state_machine",
            "if invocation_index in (1, 2):" in invocation_metadata_step
            and "return invocation_index" in invocation_metadata_step
            and "slot0_history_immutable" in snapshot_contract
            and "baseline_slot1_hashes" in snapshot_contract
            and "slot1_stable_after_first_write" in snapshot_contract
            and "slot2_untouched" in snapshot_contract,
            "device sequence is slot0, slot1, slot1, slot3+; watched history is checked independently of the general whitelist",
        ),
        _check(
            "active_mapped_slot_must_be_written",
            "_active_slot_nonzero_report(" in invocation_entry
            and "for section in range(len(KV_SECTIONS))"
            in active_slot_nonzero
            and "for layer in range(NUM_LAYERS)" in active_slot_nonzero
            and '"active_slot_rows_nonzero"' in invocation_entry,
            "every invocation proves its active slot contains nonzero K/V rows on all ranks and layers",
        ),
        _check(
            "graceful_device_failure",
            "\"status\": \"NO-GO\"" in device_entry
            and "proc.terminate()" in graceful_stop
            and "proc.kill(" not in graceful_stop
            and "signal.SIGKILL" not in graceful_stop,
            "busy-device path emits NO-GO and only uses graceful termination",
        ),
    ]
    passed = all(item["passed"] for item in checks)
    return {
        "schema": SCHEMA,
        "passed": passed,
        "classification": "source_contract_only; no compile/device execution",
        "requirements": {
            "resident_prepare_import_build_once": True,
            "minimum_invocations": MIN_INVOCATIONS,
            "coverage": {
                "layers": NUM_LAYERS,
                "ranks": TP,
                "sections": list(KV_SECTIONS),
                "physical_slots": "all rows from validated owner map",
                "watched_slots": list(KV_SLOTS),
            },
            "slot_semantics": {
                "slot0": "written once, then never mapped and immutable",
                "slot1": "written/reused, then immutable whenever not mapped",
                "slot2": "untouched; must remain zero",
            },
        },
        "checks": checks,
        "source_locations": {
            "canonical": str(CANONICAL),
            "holder": str(HOLDER),
            "harness": str(HARNESS),
            "exporter": str(EXPORTER),
        },
    }


def _configure_compile_env(out: Path, num_blocks: int) -> None:
    physical_blocks = int(num_blocks) + 15
    os.environ["PYPTO_STEP3P5_MAX_SEQ"] = str(int(num_blocks) * BLOCK_SIZE)
    os.environ["PYPTO_STEP3P5_BLOCK_TABLE_FLAT"] = str(
        16 * int(num_blocks)
    )
    os.environ["PYPTO_STEP3P5_KV_CACHE_ROWS"] = str(
        NUM_LAYERS * physical_blocks * BLOCK_SIZE
    )
    os.environ["PYPTO_STEP3P5_ROPE_SEQ"] = str(int(num_blocks) * BLOCK_SIZE)
    os.environ.setdefault("PYPTO_PROG_BUILD_DIR", str(out / "build_output"))


def _run_compile(args: argparse.Namespace) -> int:
    """Compile-only entry; this function is not called by card-free contract."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _configure_compile_env(out, args.num_blocks)
    from tools.step3p5.whole_decode_holder import WholeDecodeHolder  # noqa: PLC0415

    holder = WholeDecodeHolder(
        device_ids=_parse_devices(args.device),
        out_dir=str(out),
        ckpt=args.ckpt,
        platform=args.platform,
        kv_ipc=False,
    ).build()
    report = {
        "schema": SCHEMA,
        "mode": "compile",
        "status": "COMPILE-ONLY-PASS",
        "ok": True,
        "evidence_level": "compile_only",
        "program": holder.program_name,
        "compiled_output_dir": str(holder.compiled.output_dir),
        "note": (
            "compile mode validates canonical compilation only; resident KV "
            "import and device row evidence require device mode"
        ),
    }
    return _write_report(out, report)


def _ready(out: Path, rank: int) -> bool:
    return (
        (out / f"ready.rank{rank}").exists()
        and (out / f"pypto_weight_map.rank{rank}.json.done").exists()
        and (out / f"pypto_kvpool_map.json.rank{rank}.done").exists()
    )


def _graceful_stop_exporters(
    out: Path,
    processes: list[subprocess.Popen[Any]],
) -> None:
    """停止 exporter；只发 STOP/SIGTERM，不升级为强制终止。"""
    try:
        (out / "STOP").write_text("1", encoding="utf-8")
    except OSError:
        pass
    for proc in processes:
        try:
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # 不再升级为强制终止；报告交给上层标记 NO-GO。
                pass
        handle = getattr(proc, "_b3_log_handle", None)
        if handle is not None:
            handle.close()


def _start_exporters(
    args: argparse.Namespace,
    devices: list[int],
) -> list[subprocess.Popen[Any]]:
    """启动现有 Main hidden exporter，并提供可控 busy timeout。"""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in out.glob("ready.rank*"):
        path.unlink(missing_ok=True)
    for path in out.glob("*.done"):
        path.unlink(missing_ok=True)
    (out / "STOP").unlink(missing_ok=True)

    processes: list[subprocess.Popen[Any]] = []
    for rank, device in enumerate(devices):
        handle = open(out / f"b3_export_rank{rank}.log", "w", encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "tests.step3p5.harnesses._stage_main_hidden_only",
            "--export-rank",
            str(rank),
            "--dev",
            str(device),
            "--out",
            str(out),
            "--ckpt",
            args.ckpt,
            "--num-blocks",
            str(args.num_blocks),
            "--kv-probe",
        ]
        proc = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        setattr(proc, "_b3_log_handle", handle)
        processes.append(proc)

    deadline = time.monotonic() + float(args.exporter_timeout)
    while time.monotonic() < deadline:
        if all(_ready(out, rank) for rank in range(TP)):
            return processes
        failed = [proc.returncode for proc in processes if proc.poll() is not None]
        if failed:
            _graceful_stop_exporters(out, processes)
            raise RuntimeError(
                f"Main exporter exited before readiness, returncodes={failed}"
            )
        time.sleep(1.0)
    _graceful_stop_exporters(out, processes)
    raise TimeoutError(
        f"Main exporters were not ready within {args.exporter_timeout:.1f}s"
    )


def _load_embedding_row(ckpt: str, token: int) -> Any:
    from tests.step3p5.harnesses._stage_main_hidden_only import (  # noqa: PLC0415
        _load_embedding_row as load,
    )

    return load(ckpt, int(token))


def _step_metadata(
    *,
    step: int,
    num_blocks: int,
) -> tuple[Any, Any, Any, Any]:
    from tests.step3p5.harnesses._stage_main_hidden_only import (  # noqa: PLC0415
        _step_metadata as build,
    )

    return build(
        step=int(step),
        scheduler_num_blocks=int(num_blocks),
        valid_rows=1,
    )


def _collect_full_pool_probe(
    out: Path,
    *,
    step: int,
    phase: str,
    chunk_rows: int,
) -> dict[str, Any]:
    from tests.step3p5.harnesses._stage_main_hidden_only import (  # noqa: PLC0415
        _collect_kv_probe as collect,
    )

    return collect(
        out,
        step=int(step),
        timeout_sec=600.0,
        phase=str(phase),
        full_pool=True,
        full_pool_chunk_rows=int(chunk_rows),
    )


def _kv_summary(aggregate: dict[str, Any]) -> dict[str, Any]:
    from tests.step3p5.harnesses._stage_main_hidden_only import (  # noqa: PLC0415
        _kv_probe_summary as summarize,
    )

    return dict(summarize(aggregate))


def _validate_raw_aggregate(aggregate: dict[str, Any]) -> dict[str, Any]:
    """确认原始证据确实覆盖 45×8×K/V，而非只看汇总布尔值。"""
    expected_layers = list(range(NUM_LAYERS))
    expected_slots = list(KV_SLOTS)
    ranks = aggregate.get("ranks")
    checks: list[dict[str, Any]] = []
    checks.append(
        {
            "name": "all_layers",
            "passed": aggregate.get("layer_indices") == expected_layers,
            "detail": f"layer_indices={aggregate.get('layer_indices')!r}",
        }
    )
    checks.append(
        {
            "name": "all_slots",
            "passed": aggregate.get("slots") == expected_slots,
            "detail": f"slots={aggregate.get('slots')!r}",
        }
    )
    checks.append(
        {
            "name": "all_ranks",
            "passed": isinstance(ranks, list) and len(ranks) == TP,
            "detail": f"rank_results={len(ranks) if isinstance(ranks, list) else None}",
        }
    )
    required_keys = [
        f"L{layer}.{section}.slot{slot}"
        for layer in range(NUM_LAYERS)
        for section in KV_SECTIONS
        for slot in KV_SLOTS
    ]
    missing: list[str] = []
    if isinstance(ranks, list):
        for rank, item in enumerate(ranks):
            summary = item.get("summary") if isinstance(item, dict) else None
            if not isinstance(summary, dict):
                missing.append(f"rank{rank}:summary")
                continue
            for key in required_keys:
                if not isinstance(summary.get(key), dict):
                    missing.append(f"rank{rank}:{key}")
    else:
        missing.append("ranks")
    checks.append(
        {
            "name": "all_layer_rank_kv_slot_rows",
            "passed": not missing,
            "detail": f"missing={missing[:12]}",
        }
    )
    return {
        "layers": NUM_LAYERS,
        "ranks": TP,
        "sections": list(KV_SECTIONS),
        "slots": list(KV_SLOTS),
        "missing": missing,
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
    }


def _full_pool_rank_entries(
    aggregate: dict[str, Any],
    *,
    pool_maps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate and return all eight full-pool scan descriptors."""
    ranks = aggregate.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != TP:
        raise ValueError("full-pool snapshot must contain all eight ranks")
    entries: list[dict[str, Any]] = []
    for expected_rank, rank_result in enumerate(ranks):
        if not isinstance(rank_result, dict):
            raise ValueError(f"invalid full-pool rank result {expected_rank}")
        if int(rank_result.get("rank", -1)) != expected_rank:
            raise ValueError(
                f"full-pool result rank mismatch: expected {expected_rank}"
            )
        scan = rank_result.get("full_pool")
        if not isinstance(scan, dict):
            raise ValueError(f"rank{expected_rank} has no full-pool scan")
        pool_map = pool_maps[expected_rank]
        num_slots = int(pool_map["map"]["L0.K"]["num_slots"])
        row_count = NUM_LAYERS * len(KV_SECTIONS) * num_slots
        snapshot_path = Path(str(scan.get("snapshot_path", "")))
        checks = {
            "scan_kind": scan.get("scan_kind") == FULL_SCAN_KIND,
            "rank": int(scan.get("rank", -1)) == expected_rank,
            "pool_map_identity": (
                str(scan.get("pool_map_digest", ""))
                == _pool_map_digest(pool_map)
            ),
            "pool_bytes": (
                int(scan.get("pool_bytes", -1))
                == int(pool_map["pool_bytes"])
            ),
            "row_bytes": int(scan.get("row_bytes", -1)) == ROW_BYTES,
            "row_count": int(scan.get("row_count", -1)) == row_count,
            "num_slots": int(scan.get("num_slots", -1)) == num_slots,
            "snapshot_file_present": snapshot_path.is_file(),
            "snapshot_file_size": (
                snapshot_path.is_file()
                and snapshot_path.stat().st_size
                == row_count * ROW_BYTES
            ),
            "pool_base_present": int(scan.get("pool_base_debug", 0)) > 0,
            "pool_digest_present": bool(
                re.fullmatch(r"[0-9a-f]{64}", str(scan.get("pool_sha256", "")))
            ),
        }
        if not all(checks.values()):
            raise ValueError(
                f"rank{expected_rank} full-pool scan contract failed: {checks}"
            )
        entries.append(
            {
                "rank": expected_rank,
                "scan": scan,
                "snapshot_path": snapshot_path,
                "num_slots": num_slots,
                "row_count": row_count,
                "checks": checks,
            }
        )
    return entries


def _cleanup_full_pool_snapshots(
    *aggregates: dict[str, Any] | None,
) -> dict[str, Any]:
    """Remove diagnostic raw snapshots while preserving JSON summaries."""
    referenced: set[Path] = set()
    invalid_entries: list[str] = []
    for aggregate_index, aggregate in enumerate(aggregates):
        if aggregate is None:
            continue
        ranks = aggregate.get("ranks")
        if not isinstance(ranks, list):
            invalid_entries.append(f"aggregate{aggregate_index}:ranks")
            continue
        for rank_index, rank_result in enumerate(ranks):
            scan = (
                rank_result.get("full_pool")
                if isinstance(rank_result, dict)
                else None
            )
            raw_path = scan.get("snapshot_path") if isinstance(scan, dict) else None
            if not isinstance(raw_path, str) or not raw_path:
                invalid_entries.append(
                    f"aggregate{aggregate_index}:rank{rank_index}"
                )
                continue
            referenced.add(Path(raw_path))

    removed: list[str] = []
    missing: list[str] = []
    errors: list[dict[str, str]] = []
    for snapshot_path in sorted(referenced):
        try:
            snapshot_path.unlink()
            removed.append(str(snapshot_path))
        except FileNotFoundError:
            missing.append(str(snapshot_path))
        except OSError as exc:
            errors.append(
                {
                    "path": str(snapshot_path),
                    "error_type": type(exc).__name__,
                    "error": repr(exc),
                }
            )
    remaining = [
        str(snapshot_path)
        for snapshot_path in sorted(referenced)
        if snapshot_path.exists()
    ]
    return {
        "referenced": len(referenced),
        "removed": removed,
        "missing": missing,
        "invalid_entries": invalid_entries,
        "errors": errors,
        "remaining": remaining,
        "passed": not invalid_entries and not errors and not remaining,
    }


def _allowed_physical_rows(
    *,
    num_slots: int,
    slot_mapping: list[int],
) -> set[int]:
    slots = {int(slot) for slot in slot_mapping}
    invalid = sorted(slot for slot in slots if not 0 <= slot < num_slots)
    if invalid:
        raise ValueError(
            f"slot_mapping contains rows outside physical pool: {invalid}"
        )
    section_rows = NUM_LAYERS * num_slots
    return {
        section * section_rows + layer * num_slots + slot
        for section in range(len(KV_SECTIONS))
        for layer in range(NUM_LAYERS)
        for slot in slots
    }


def _full_pool_row_diff(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    pool_maps: list[dict[str, Any]],
    slot_mapping: list[int],
    required_first_write_slot: int | None = None,
) -> dict[str, Any]:
    """Prove mapped-row writes and reject a first-write no-op.

    The standalone exporter zero-initialises the entire pool.  A slot's first
    use must therefore show all 45-layer K/V rows as zero before the run,
    non-zero after it, and byte-different across the invocation.  Later
    repeated identical invocations may legitimately remain bitwise stable.
    """
    before_entries = _full_pool_rank_entries(
        before,
        pool_maps=pool_maps,
    )
    after_entries = _full_pool_rank_entries(
        after,
        pool_maps=pool_maps,
    )
    rank_reports: list[dict[str, Any]] = []
    for rank, (old, new) in enumerate(zip(before_entries, after_entries)):
        if old["row_count"] != new["row_count"]:
            raise ValueError(f"rank{rank} row count changed across invocation")
        if old["num_slots"] != new["num_slots"]:
            raise ValueError(f"rank{rank} num_slots changed across invocation")
        old_scan = old["scan"]
        new_scan = new["scan"]
        identity_checks = {
            "pool_base_stable": (
                int(old_scan["pool_base_debug"])
                == int(new_scan["pool_base_debug"])
            ),
            "pool_map_stable": (
                old_scan["pool_map_digest"] == new_scan["pool_map_digest"]
            ),
            "pool_shape_stable": (
                int(old_scan["pool_bytes"]) == int(new_scan["pool_bytes"])
                and int(old_scan["row_count"]) == int(new_scan["row_count"])
                and int(old_scan["row_bytes"]) == int(new_scan["row_bytes"])
            ),
        }
        allowed = _allowed_physical_rows(
            num_slots=int(old["num_slots"]),
            slot_mapping=slot_mapping,
        )
        first_write_rows = (
            _allowed_physical_rows(
                num_slots=int(old["num_slots"]),
                slot_mapping=[required_first_write_slot],
            )
            if required_first_write_slot is not None
            else set()
        )
        changed_rows = 0
        changed_allowed_rows = 0
        unauthorized_rows = 0
        first_write_before_nonzero = 0
        first_write_after_zero = 0
        first_write_unchanged = 0
        unauthorized_samples: list[str] = []
        old_digest = hashlib.sha256()
        new_digest = hashlib.sha256()
        with old["snapshot_path"].open("rb") as old_file, new[
            "snapshot_path"
        ].open("rb") as new_file:
            for row_index in range(int(old["row_count"])):
                old_row = old_file.read(ROW_BYTES)
                new_row = new_file.read(ROW_BYTES)
                if len(old_row) != ROW_BYTES or len(new_row) != ROW_BYTES:
                    raise ValueError(
                        f"rank{rank} snapshot file truncated at {row_index}"
                    )
                old_digest.update(old_row)
                new_digest.update(new_row)
                if row_index in first_write_rows:
                    if any(old_row):
                        first_write_before_nonzero += 1
                    if not any(new_row):
                        first_write_after_zero += 1
                    if old_row == new_row:
                        first_write_unchanged += 1
                if old_row == new_row:
                    continue
                changed_rows += 1
                if row_index in allowed:
                    changed_allowed_rows += 1
                else:
                    unauthorized_rows += 1
                    if len(unauthorized_samples) < 32:
                        unauthorized_samples.append(
                            _physical_row_key(
                                rank,
                                row_index,
                                int(old["num_slots"]),
                            )
                        )
            if old_file.read(1) or new_file.read(1):
                raise ValueError(f"rank{rank} snapshot file has trailing data")
        digest_checks = {
            "before_snapshot_digest_matches_descriptor": (
                old_digest.hexdigest() == str(old_scan["pool_sha256"])
            ),
            "after_snapshot_digest_matches_descriptor": (
                new_digest.hexdigest() == str(new_scan["pool_sha256"])
            ),
        }
        rank_reports.append(
            {
                "rank": rank,
                "row_count": int(old["row_count"]),
                "allowed_rows": len(allowed),
                "changed_rows": changed_rows,
                "changed_allowed_rows": changed_allowed_rows,
                "unauthorized_changed_rows": unauthorized_rows,
                "required_first_write_slot": required_first_write_slot,
                "first_write_rows": len(first_write_rows),
                "first_write_before_nonzero": first_write_before_nonzero,
                "first_write_after_zero": first_write_after_zero,
                "first_write_unchanged": first_write_unchanged,
                "unauthorized_samples": unauthorized_samples,
                "identity_checks": identity_checks,
                "digest_checks": digest_checks,
                "passed": (
                    all(identity_checks.values())
                    and all(digest_checks.values())
                    and unauthorized_rows == 0
                    and first_write_before_nonzero == 0
                    and first_write_after_zero == 0
                    and first_write_unchanged == 0
                ),
            }
        )
    return {
        "scan_kind": FULL_SCAN_KIND,
        "slot_mapping": list(slot_mapping),
        "unique_mapped_slots": sorted(set(slot_mapping)),
        "required_first_write_slot": required_first_write_slot,
        "pool_bases": [
            int(item["scan"]["pool_base_debug"]) for item in before_entries
        ],
        "before_pool_sha256": [
            str(item["scan"]["pool_sha256"]) for item in before_entries
        ],
        "after_pool_sha256": [
            str(item["scan"]["pool_sha256"]) for item in after_entries
        ],
        "ranks": rank_reports,
        "passed": all(item["passed"] for item in rank_reports),
    }


def _active_slot_nonzero_report(
    aggregate: dict[str, Any],
    *,
    pool_maps: list[dict[str, Any]],
    active_slot: int,
) -> dict[str, Any]:
    """Prove the active scheduler slot contains K/V data on every rank/layer."""
    entries = _full_pool_rank_entries(
        aggregate,
        pool_maps=pool_maps,
    )
    rank_reports: list[dict[str, Any]] = []
    expected_rows = NUM_LAYERS * len(KV_SECTIONS)
    for rank, entry in enumerate(entries):
        num_slots = int(entry["num_slots"])
        if not 0 <= int(active_slot) < num_slots:
            raise ValueError(
                f"active slot {active_slot} outside rank{rank} pool"
            )
        section_rows = NUM_LAYERS * num_slots
        nonzero_rows = 0
        zero_samples: list[str] = []
        with entry["snapshot_path"].open("rb") as snapshot:
            for section in range(len(KV_SECTIONS)):
                for layer in range(NUM_LAYERS):
                    row_index = (
                        section * section_rows
                        + layer * num_slots
                        + int(active_slot)
                    )
                    snapshot.seek(row_index * ROW_BYTES)
                    row = snapshot.read(ROW_BYTES)
                    if len(row) != ROW_BYTES:
                        raise ValueError(
                            f"rank{rank} active-slot snapshot truncated at "
                            f"row {row_index}"
                        )
                    if any(row):
                        nonzero_rows += 1
                    elif len(zero_samples) < 16:
                        zero_samples.append(
                            _physical_row_key(rank, row_index, num_slots)
                        )
        rank_reports.append(
            {
                "rank": rank,
                "active_slot": int(active_slot),
                "expected_rows": expected_rows,
                "nonzero_rows": nonzero_rows,
                "zero_samples": zero_samples,
                "passed": nonzero_rows == expected_rows,
            }
        )
    return {
        "active_slot": int(active_slot),
        "expected_rows_per_rank": expected_rows,
        "ranks": rank_reports,
        "passed": all(item["passed"] for item in rank_reports),
    }


def _inter_invocation_continuity(
    previous_after_pool_sha256: list[str] | None,
    current_before_pool_sha256: list[str],
) -> bool:
    """Require the next pre-scan to equal the previous post-scan exactly."""
    return (
        previous_after_pool_sha256 is None
        or list(current_before_pool_sha256)
        == list(previous_after_pool_sha256)
    )


def _snapshot_contract(
    *,
    summary: dict[str, Any],
    raw_contract: dict[str, Any],
    invocation_index: int,
    slot_mapping: list[int],
    baseline_slot0_hashes: dict[str, str] | None,
    baseline_slot1_hashes: dict[str, str] | None,
) -> dict[str, Any]:
    """检查 watched slot 的首次写入、重复写入和历史不变性。"""
    slot0_hashes = dict(summary.get("slot0_hashes", {}))
    slot1_hashes = dict(summary.get("slot1_hashes", {}))
    mapped_slots = {int(slot) for slot in slot_mapping}
    slot0_mapped = 0 in mapped_slots
    slot1_mapped = 1 in mapped_slots
    slot2_mapped = 2 in mapped_slots

    slot0_history_immutable = (
        slot0_mapped
        or baseline_slot0_hashes is None
        or slot0_hashes == baseline_slot0_hashes
    )
    slot0_written = (
        bool(summary.get("slot0_all_nonzero"))
        if slot0_mapped
        else (
            baseline_slot0_hashes is not None
            and slot0_hashes == baseline_slot0_hashes
        )
    )
    slot1_new_write = (
        bool(summary.get("slot1_all_nonzero"))
        if slot1_mapped
        else (
            not bool(summary.get("slot1_any_nonzero"))
            if baseline_slot1_hashes is None
            else slot1_hashes == baseline_slot1_hashes
        )
    )
    slot1_stable_after_first_write = (
        baseline_slot1_hashes is None
        or slot1_hashes == baseline_slot1_hashes
    )
    slot2_untouched = (
        not slot2_mapped
        and not bool(summary.get("slot2_any_nonzero"))
    )
    checks = {
        "raw_coverage": bool(raw_contract["passed"]),
        "slot0_written": slot0_written,
        "slot0_history_immutable": slot0_history_immutable,
        "slot1_new_write_or_initially_empty": slot1_new_write,
        "slot1_stable_after_first_write": slot1_stable_after_first_write,
        "slot2_not_mapped": not slot2_mapped,
        "slot2_untouched": slot2_untouched,
    }
    return {
        "invocation_index": invocation_index,
        "slot_mapping": list(slot_mapping),
        "mapped_slots": sorted(mapped_slots),
        "slot0_mapped": slot0_mapped,
        "slot1_mapped": slot1_mapped,
        "slot2_mapped": slot2_mapped,
        "observed_values": summary.get("observed_values"),
        "slot0_hashes": slot0_hashes,
        "slot0_history_immutable": slot0_history_immutable,
        "slot1_hashes": slot1_hashes,
        "slot1_all_nonzero": summary.get("slot1_all_nonzero"),
        "slot1_stable_after_first_write": slot1_stable_after_first_write,
        "slot2_any_nonzero": summary.get("slot2_any_nonzero"),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _run_one_invocation(
    holder: Any,
    *,
    embedding: Any,
    metadata: tuple[Any, Any, Any, Any],
    out: Path,
    invocation_index: int,
    metadata_step: int,
    first_write_slot: int | None,
    baseline_slot0_hashes: dict[str, str] | None,
    baseline_slot1_hashes: dict[str, str] | None,
    baseline_pool_bases: list[int] | None,
    previous_after_pool_sha256: list[str] | None,
    pool_maps: list[dict[str, Any]],
    scan_chunk_rows: int,
) -> tuple[
    dict[str, Any],
    dict[str, str],
    dict[str, str],
    list[int],
    list[str],
]:
    seq, pos, table, slot = metadata
    slot_mapping = _metadata_slot_values(metadata)
    holder.set_live_step(
        embedding.unsqueeze(0),
        seq_lens=seq,
        positions=pos,
        block_table=table,
        slot_mapping=slot,
    )
    before: dict[str, Any] | None = None
    aggregate: dict[str, Any] | None = None
    try:
        before = _collect_full_pool_probe(
            out,
            step=invocation_index,
            phase="before",
            chunk_rows=scan_chunk_rows,
        )
        started = time.monotonic()
        result = holder.run()
        elapsed = time.monotonic() - started
        hidden = result["next_hidden"]
        finite = bool(hidden.isfinite().all().item())
        aggregate = _collect_full_pool_probe(
            out,
            step=invocation_index,
            phase="after",
            chunk_rows=scan_chunk_rows,
        )
        raw_contract = _validate_raw_aggregate(aggregate)
        summary = _kv_summary(aggregate)
        full_pool_diff = _full_pool_row_diff(
            before,
            aggregate,
            pool_maps=pool_maps,
            slot_mapping=slot_mapping,
            required_first_write_slot=first_write_slot,
        )
        active_slot_nonzero = _active_slot_nonzero_report(
            aggregate,
            pool_maps=pool_maps,
            active_slot=slot_mapping[0],
        )
    finally:
        snapshot_cleanup = _cleanup_full_pool_snapshots(before, aggregate)

    pool_bases = [int(value) for value in full_pool_diff["pool_bases"]]
    before_pool_sha256 = [
        str(value) for value in full_pool_diff["before_pool_sha256"]
    ]
    after_pool_sha256 = [
        str(value) for value in full_pool_diff["after_pool_sha256"]
    ]
    evidence = _snapshot_contract(
        summary=summary,
        raw_contract=raw_contract,
        invocation_index=invocation_index,
        slot_mapping=slot_mapping,
        baseline_slot0_hashes=baseline_slot0_hashes,
        baseline_slot1_hashes=baseline_slot1_hashes,
    )
    checks = dict(evidence["checks"])
    checks["hidden_finite"] = finite
    checks["full_pool_row_diff_whitelist"] = bool(full_pool_diff["passed"])
    checks["active_slot_rows_nonzero"] = bool(active_slot_nonzero["passed"])
    checks["pool_base_stable_across_invocations"] = (
        baseline_pool_bases is None or pool_bases == baseline_pool_bases
    )
    checks["inter_invocation_pool_continuity"] = _inter_invocation_continuity(
        previous_after_pool_sha256,
        before_pool_sha256,
    )
    checks["snapshot_sidecars_removed"] = bool(snapshot_cleanup["passed"])
    report = {
        "invocation_index": invocation_index,
        "metadata_step": metadata_step,
        "first_write_slot": first_write_slot,
        "run_sec": elapsed,
        "holder_last_run_sec": float(getattr(holder, "_last_run_sec", 0.0)),
        "hidden_shape": list(hidden.shape),
        "hidden_finite": finite,
        "kv_probe_before_path": str(
            out / f"kv_probe_step{invocation_index}_before.json"
        ),
        "kv_probe_after_path": str(
            out / f"kv_probe_step{invocation_index}_after.json"
        ),
        "slot_mapping": slot_mapping,
        "coverage": raw_contract,
        "summary": summary,
        "full_pool_row_diff": full_pool_diff,
        "active_slot_nonzero": active_slot_nonzero,
        "inter_invocation_pool_continuity": {
            "previous_after_pool_sha256": previous_after_pool_sha256,
            "current_before_pool_sha256": before_pool_sha256,
            "passed": checks["inter_invocation_pool_continuity"],
        },
        "snapshot_cleanup": snapshot_cleanup,
        "evidence": evidence,
        "checks": checks,
        "passed": all(checks.values()),
    }
    return (
        report,
        dict(summary.get("slot0_hashes", {})),
        dict(summary.get("slot1_hashes", {})),
        pool_bases,
        after_pool_sha256,
    )


def _write_report(out: Path, report: dict[str, Any]) -> int:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "b3_resident_kv_report.json"
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    print(f"B3_REPORT={path}", flush=True)
    success = (
        bool(report["ok"])
        if "ok" in report
        else report.get("status") in {"PASS", "COMPILE-ONLY-PASS"}
    )
    return 0 if success else 1


def _run_device(args: argparse.Namespace) -> int:
    """设备入口；设备忙/初始化失败均转成 graceful NO-GO。"""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    devices = _parse_devices(args.device)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": "device",
        "status": "NO-GO",
        "ok": False,
        "program": "whole_decode_step3p5",
        "device_ids": devices,
        "requested_invocations": int(args.invocations),
        "coverage": {
            "layers": NUM_LAYERS,
            "ranks": TP,
            "sections": list(KV_SECTIONS),
            "slots": list(KV_SLOTS),
        },
        "resident_lifecycle": {
            "build_once": True,
            "prepare_once": True,
            "import_kv_once": True,
            "build_stacked_kv_once": True,
        },
        "invocations": [],
    }
    processes: list[subprocess.Popen[Any]] = []
    try:
        if int(args.invocations) < MIN_INVOCATIONS:
            raise ValueError(
                f"--invocations must be >= {MIN_INVOCATIONS}, got {args.invocations}"
            )
        if not args.reuse_exporters:
            processes = _start_exporters(args, devices)
        elif not all(_ready(out, rank) for rank in range(TP)):
            raise RuntimeError(
                "--reuse-exporters requested but all eight exporter markers "
                "are not ready"
            )

        _configure_compile_env(out, args.num_blocks)
        pool_maps = _load_pool_maps(out, devices)
        report["pool_identity"] = {
            "map_digests": [
                _pool_map_digest(pool_map) for pool_map in pool_maps
            ],
            "pool_bytes": [
                int(pool_map["pool_bytes"]) for pool_map in pool_maps
            ],
        }
    except Exception as exc:  # noqa: BLE001 - device busy is a release NO-GO
        report["error"] = {
            "stage": "exporter_or_prepare",
            "error_type": type(exc).__name__,
            "error": repr(exc),
        }
        if processes:
            _graceful_stop_exporters(out, processes)
        return _write_report(out, report)

    try:
        # Keep import local so contract/py_compile remain card-free.
        from tools.step3p5.whole_decode_holder import WholeDecodeHolder  # noqa: PLC0415

        holder = WholeDecodeHolder(
            device_ids=devices,
            out_dir=str(out),
            ckpt=args.ckpt,
            platform=args.platform,
            kv_ipc=True,
        ).build()
        embedding = _load_embedding_row(args.ckpt, args.seed_token)
        baseline_slot0_hashes: dict[str, str] | None = None
        baseline_slot1_hashes: dict[str, str] | None = None
        baseline_pool_bases: list[int] | None = None
        previous_after_pool_sha256: list[str] | None = None
        written_slots: set[int] = set()
        with holder:
            for invocation_index in range(int(args.invocations)):
                metadata_step = _invocation_metadata_step(invocation_index)
                metadata = _step_metadata(
                    step=metadata_step,
                    num_blocks=args.num_blocks,
                )
                (
                    invocation,
                    slot0_hashes,
                    slot1_hashes,
                    pool_bases,
                    after_pool_sha256,
                ) = _run_one_invocation(
                    holder,
                    embedding=embedding,
                    metadata=metadata,
                    out=out,
                    invocation_index=invocation_index,
                    metadata_step=metadata_step,
                    first_write_slot=(
                        metadata_step
                        if metadata_step not in written_slots
                        else None
                    ),
                    baseline_slot0_hashes=baseline_slot0_hashes,
                    baseline_slot1_hashes=baseline_slot1_hashes,
                    baseline_pool_bases=baseline_pool_bases,
                    previous_after_pool_sha256=previous_after_pool_sha256,
                    pool_maps=pool_maps,
                    scan_chunk_rows=args.scan_chunk_rows,
                )
                if baseline_slot0_hashes is None:
                    baseline_slot0_hashes = slot0_hashes
                if invocation_index == 1:
                    baseline_slot1_hashes = slot1_hashes
                written_slots.add(metadata_step)
                if baseline_pool_bases is None:
                    baseline_pool_bases = pool_bases
                previous_after_pool_sha256 = after_pool_sha256
                report["invocations"].append(invocation)
                print(
                    json.dumps(
                        {
                            "schema": SCHEMA,
                            "invocation": invocation_index,
                            "metadata_step": metadata_step,
                            "passed": invocation["passed"],
                            "checks": invocation["checks"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if not invocation["passed"]:
                    raise AssertionError(invocation)
        report["status"] = (
            "PASS"
            if len(report["invocations"]) >= MIN_INVOCATIONS
            else "NO-GO"
        )
        report["ok"] = report["status"] == "PASS"
    except Exception as exc:  # noqa: BLE001 - device busy/stall => NO-GO
        report["status"] = "NO-GO"
        report["ok"] = False
        report["error"] = {
            "stage": "resident_invocation",
            "error_type": type(exc).__name__,
            "error": repr(exc),
        }
    finally:
        if processes:
            _graceful_stop_exporters(out, processes)
    return _write_report(out, report)


def _run_contract(args: argparse.Namespace) -> int:
    report = _contract()
    report.update(
        {
            "mode": "contract",
            "status": "PASS" if report["passed"] else "NO-GO",
            "out": str(Path(args.out).resolve()),
        }
    )
    return _write_report(Path(args.out), report)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PERF-B3 resident K/V pool contract, compile, and device probe"
    )
    parser.add_argument(
        "--mode",
        choices=("contract", "compile", "device"),
        default="contract",
    )
    parser.add_argument("--out", default="/tmp/step3p5-b3-resident-kv")
    parser.add_argument("--device", default=DEFAULT_DEVICES)
    parser.add_argument("--platform", choices=("a2a3", "a2a3sim"), default="a2a3")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--num-blocks", type=int, default=DEFAULT_NUM_BLOCKS)
    parser.add_argument("--seed-token", type=int, default=6127)
    parser.add_argument("--invocations", type=int, default=MIN_INVOCATIONS)
    parser.add_argument("--exporter-timeout", type=float, default=2400.0)
    parser.add_argument(
        "--scan-chunk-rows",
        type=int,
        default=DEFAULT_SCAN_CHUNK_ROWS,
        help=(
            "diagnostic full-pool D2H rows per chunk; every physical row is "
            "still hashed and compared"
        ),
    )
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
