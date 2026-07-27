# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic fail-closed tests for the PERF-B3 full-pool row diff."""
from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tests.step3p5.probes._probe_b3_resident_kv import (
    FULL_SCAN_KIND,
    KV_SECTIONS,
    NUM_LAYERS,
    ROW_BYTES,
    TP,
    _active_slot_nonzero_report,
    _cleanup_full_pool_snapshots,
    _full_pool_row_diff,
    _inter_invocation_continuity,
    _invocation_metadata_step,
    _pool_map_digest,
    _snapshot_contract,
)
from tools.step3p5.main_kv_exporter import MainKvExporter


def _pool_maps(*, num_slots: int) -> list[dict[str, Any]]:
    pool_bytes = NUM_LAYERS * len(KV_SECTIONS) * num_slots * ROW_BYTES
    return [
        {
            "rank": rank,
            "pool_bytes": pool_bytes,
            "map": {"L0.K": {"num_slots": num_slots}},
        }
        for rank in range(TP)
    ]


def _aggregate(
    root: Path,
    *,
    phase: str,
    pool_maps: list[dict[str, Any]],
    changed_rows: dict[int, set[int]] | None = None,
    zero_slots: set[int] | None = None,
    trailing_rank: int | None = None,
    truncate_rank: int | None = None,
) -> dict[str, Any]:
    changed_rows = changed_rows or {}
    zero_slots = zero_slots or set()
    ranks: list[dict[str, Any]] = []
    for rank, pool_map in enumerate(pool_maps):
        num_slots = int(pool_map["map"]["L0.K"]["num_slots"])
        row_count = NUM_LAYERS * len(KV_SECTIONS) * num_slots
        rows = []
        for row_index in range(row_count):
            slot = row_index % num_slots
            fill = 0 if slot in zero_slots else rank
            rows.append(bytes([fill]) * ROW_BYTES)
        for row_index in changed_rows.get(rank, set()):
            rows[row_index] = bytes([(rank + 1) % 256]) * ROW_BYTES
        raw = b"".join(rows)
        if rank == truncate_rank:
            raw = raw[:-1]
        if rank == trailing_rank:
            raw += b"x"
        path = root / f"{phase}.rank{rank}.bin"
        path.write_bytes(raw)
        ranks.append(
            {
                "rank": rank,
                "full_pool": {
                    "scan_kind": FULL_SCAN_KIND,
                    "pool_map_digest": _pool_map_digest(pool_map),
                    "pool_sha256": hashlib.sha256(raw).hexdigest(),
                    "pool_bytes": int(pool_map["pool_bytes"]),
                    "row_bytes": ROW_BYTES,
                    "row_count": row_count,
                    "num_slots": num_slots,
                    "chunk_rows": row_count,
                    "chunk_sha256": [hashlib.sha256(raw).hexdigest()],
                    "snapshot_path": str(path),
                    "pool_base_debug": 0x100000 + rank * 0x10000,
                    "rank": rank,
                },
            }
        )
    return {
        "phase": phase,
        "ranks": ranks,
        "fixture_digest": hashlib.sha256(
            json.dumps(changed_rows, sort_keys=True, default=list).encode()
        ).hexdigest(),
    }


def test_full_pool_diff_accepts_only_rows_from_complete_slot_mapping(
    tmp_path: Path,
) -> None:
    pool_maps = _pool_maps(num_slots=3)
    before = _aggregate(
        tmp_path,
        phase="before",
        pool_maps=pool_maps,
        zero_slots={1},
    )
    # section K, layer 0, slot 1 is explicitly mapped by this invocation.
    after = _aggregate(
        tmp_path,
        phase="after",
        pool_maps=pool_maps,
        changed_rows={0: {1}},
    )

    report = _full_pool_row_diff(
        before,
        after,
        pool_maps=pool_maps,
        slot_mapping=[1, 1],
    )

    assert report["passed"]
    assert report["ranks"][0]["changed_allowed_rows"] == 1
    assert report["ranks"][0]["unauthorized_changed_rows"] == 0
    assert report["before_pool_sha256"] != report["after_pool_sha256"]


def test_first_write_rejects_active_slot_that_stays_zero(
    tmp_path: Path,
) -> None:
    pool_maps = _pool_maps(num_slots=3)
    before = _aggregate(
        tmp_path,
        phase="before",
        pool_maps=pool_maps,
        zero_slots={1},
    )
    after = _aggregate(
        tmp_path,
        phase="after",
        pool_maps=pool_maps,
        zero_slots={1},
    )

    report = _full_pool_row_diff(
        before,
        after,
        pool_maps=pool_maps,
        slot_mapping=[1],
        required_first_write_slot=1,
    )

    assert not report["passed"]
    assert report["ranks"][0]["first_write_unchanged"] == (
        NUM_LAYERS * len(KV_SECTIONS)
    )


def test_first_write_requires_all_active_layer_kv_rows_to_leave_zero(
    tmp_path: Path,
) -> None:
    pool_maps = _pool_maps(num_slots=3)
    before = _aggregate(
        tmp_path,
        phase="before",
        pool_maps=pool_maps,
        zero_slots={1},
    )
    changed = {
        rank: {
            section * NUM_LAYERS * 3 + layer * 3 + 1
            for section in range(len(KV_SECTIONS))
            for layer in range(NUM_LAYERS)
        }
        for rank in range(TP)
    }
    after = _aggregate(
        tmp_path,
        phase="after",
        pool_maps=pool_maps,
        changed_rows=changed,
        zero_slots={1},
    )

    report = _full_pool_row_diff(
        before,
        after,
        pool_maps=pool_maps,
        slot_mapping=[1],
        required_first_write_slot=1,
    )

    assert report["passed"]
    assert all(
        rank_report["first_write_unchanged"] == 0
        for rank_report in report["ranks"]
    )


def test_exporter_full_pool_snapshot_is_self_contained(
    tmp_path: Path,
) -> None:
    """Exercise the diagnostics path without an ACL device.

    Besides checking the emitted bytes and descriptor, this catches missing
    runtime imports in ``snapshot_full_pool`` that ``py_compile`` cannot find.
    """

    class _FakeAcl:
        @staticmethod
        def aclrtMemcpy(
            dst: ctypes.c_void_p,
            _dst_size: ctypes.c_size_t,
            src: ctypes.c_void_p,
            nbytes: ctypes.c_size_t,
            _kind: ctypes.c_int,
        ) -> int:
            ctypes.memmove(dst, src, int(nbytes.value))
            return 0

    raw = bytes(range(256)) * 2
    backing = ctypes.create_string_buffer(raw)
    exporter = MainKvExporter.__new__(MainKvExporter)
    exporter._acl = _FakeAcl()
    exporter._pool_ptr = ctypes.addressof(backing)
    exporter._pool_bytes = len(raw)
    exporter._pool_map = {
        "rank": 0,
        "pool_bytes": len(raw),
        "map": {"L0.K": {"num_slots": 2}},
    }

    descriptor = exporter.snapshot_full_pool(
        out_dir=str(tmp_path),
        probe_id="synthetic",
        chunk_rows=1,
    )

    snapshot = Path(descriptor["snapshot_path"])
    assert snapshot.read_bytes() == raw
    assert descriptor["pool_sha256"] == hashlib.sha256(raw).hexdigest()
    assert descriptor["row_count"] == 2
    assert descriptor["chunk_sha256"] == [
        hashlib.sha256(raw[:ROW_BYTES]).hexdigest(),
        hashlib.sha256(raw[ROW_BYTES:]).hexdigest(),
    ]


def test_full_pool_diff_rejects_changed_row_outside_slot_mapping(
    tmp_path: Path,
) -> None:
    pool_maps = _pool_maps(num_slots=3)
    before = _aggregate(tmp_path, phase="before", pool_maps=pool_maps)
    # Slot 2 changed while this invocation only owns slot 1.
    after = _aggregate(
        tmp_path,
        phase="after",
        pool_maps=pool_maps,
        changed_rows={0: {2}},
    )

    report = _full_pool_row_diff(
        before,
        after,
        pool_maps=pool_maps,
        slot_mapping=[1],
    )

    assert not report["passed"]
    assert report["ranks"][0]["unauthorized_changed_rows"] == 1
    assert report["ranks"][0]["unauthorized_samples"] == [
        "rank0.L0.K.slot2"
    ]


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("missing_rank", "all eight ranks"),
        ("truncated", "snapshot_file_size"),
        ("trailing", "snapshot_file_size"),
    ],
)
def test_full_pool_diff_rejects_incomplete_or_malformed_evidence(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    pool_maps = _pool_maps(num_slots=1)
    before = _aggregate(tmp_path, phase="before", pool_maps=pool_maps)
    after = _aggregate(
        tmp_path,
        phase="after",
        pool_maps=pool_maps,
        truncate_rank=0 if mutation == "truncated" else None,
        trailing_rank=0 if mutation == "trailing" else None,
    )
    if mutation == "missing_rank":
        after["ranks"].pop()

    with pytest.raises(ValueError, match=error):
        _full_pool_row_diff(
            before,
            after,
            pool_maps=pool_maps,
            slot_mapping=[0],
        )


def test_full_pool_snapshot_cleanup_removes_sidecars_and_preserves_summary(
    tmp_path: Path,
) -> None:
    pool_maps = _pool_maps(num_slots=1)
    before = _aggregate(tmp_path, phase="before", pool_maps=pool_maps)
    after = _aggregate(tmp_path, phase="after", pool_maps=pool_maps)

    cleanup = _cleanup_full_pool_snapshots(before, after)

    assert cleanup["passed"]
    assert cleanup["referenced"] == TP * 2
    assert len(cleanup["removed"]) == TP * 2
    assert not list(tmp_path.glob("*.bin"))
    assert before["ranks"][0]["full_pool"]["snapshot_path"].endswith(
        "before.rank0.bin"
    )


def test_full_pool_diff_rejects_descriptor_digest_not_matching_sidecar(
    tmp_path: Path,
) -> None:
    pool_maps = _pool_maps(num_slots=1)
    before = _aggregate(tmp_path, phase="before", pool_maps=pool_maps)
    after = _aggregate(tmp_path, phase="after", pool_maps=pool_maps)
    after["ranks"][0]["full_pool"]["pool_sha256"] = "0" * 64

    report = _full_pool_row_diff(
        before,
        after,
        pool_maps=pool_maps,
        slot_mapping=[0],
    )

    assert not report["passed"]
    assert not report["ranks"][0]["digest_checks"][
        "after_snapshot_digest_matches_descriptor"
    ]


def _summary(
    *,
    slot0: str,
    slot1: str,
    slot1_nonzero: bool,
    slot2_nonzero: bool = False,
) -> dict[str, Any]:
    return {
        "observed_values": 1,
        "slot0_all_nonzero": True,
        "slot0_hashes": {"rank0.L0.K.slot0": slot0},
        "slot1_any_nonzero": slot1_nonzero,
        "slot1_all_nonzero": slot1_nonzero,
        "slot1_hashes": {"rank0.L0.K.slot1": slot1},
        "slot2_any_nonzero": slot2_nonzero,
    }


def test_slot1_repeated_identical_invocation_must_preserve_first_write() -> None:
    report = _snapshot_contract(
        summary=_summary(
            slot0="slot0-stable",
            slot1="slot1-stable",
            slot1_nonzero=True,
        ),
        raw_contract={"passed": True},
        invocation_index=2,
        slot_mapping=[1],
        baseline_slot0_hashes={"rank0.L0.K.slot0": "slot0-stable"},
        baseline_slot1_hashes={"rank0.L0.K.slot1": "slot1-stable"},
    )

    assert report["passed"]
    assert report["checks"]["slot1_stable_after_first_write"]


def test_slot1_repeated_identical_invocation_rejects_history_drift() -> None:
    report = _snapshot_contract(
        summary=_summary(
            slot0="slot0-stable",
            slot1="slot1-drifted",
            slot1_nonzero=True,
        ),
        raw_contract={"passed": True},
        invocation_index=2,
        slot_mapping=[1],
        baseline_slot0_hashes={"rank0.L0.K.slot0": "slot0-stable"},
        baseline_slot1_hashes={"rank0.L0.K.slot1": "slot1-stable"},
    )

    assert not report["passed"]
    assert not report["checks"]["slot1_stable_after_first_write"]


def test_slot1_must_stabilize_after_it_is_unmapped() -> None:
    stable = _snapshot_contract(
        summary=_summary(
            slot0="slot0-stable",
            slot1="slot1-stable",
            slot1_nonzero=True,
        ),
        raw_contract={"passed": True},
        invocation_index=3,
        slot_mapping=[3],
        baseline_slot0_hashes={"rank0.L0.K.slot0": "slot0-stable"},
        baseline_slot1_hashes={"rank0.L0.K.slot1": "slot1-stable"},
    )
    drifted = _snapshot_contract(
        summary=_summary(
            slot0="slot0-stable",
            slot1="slot1-drifted",
            slot1_nonzero=True,
        ),
        raw_contract={"passed": True},
        invocation_index=3,
        slot_mapping=[3],
        baseline_slot0_hashes={"rank0.L0.K.slot0": "slot0-stable"},
        baseline_slot1_hashes={"rank0.L0.K.slot1": "slot1-stable"},
    )

    assert stable["passed"]
    assert not stable["slot1_mapped"]
    assert not drifted["passed"]


def test_slot2_is_never_mapped_or_written() -> None:
    report = _snapshot_contract(
        summary=_summary(
            slot0="slot0-stable",
            slot1="slot1-stable",
            slot1_nonzero=True,
            slot2_nonzero=True,
        ),
        raw_contract={"passed": True},
        invocation_index=3,
        slot_mapping=[3],
        baseline_slot0_hashes={"rank0.L0.K.slot0": "slot0-stable"},
        baseline_slot1_hashes={"rank0.L0.K.slot1": "slot1-stable"},
    )

    assert not report["passed"]
    assert not report["checks"]["slot2_untouched"]


def test_invocation_slot_state_machine_is_explicit() -> None:
    assert [_invocation_metadata_step(i) for i in range(6)] == [0, 1, 1, 3, 4, 5]


def test_pool_base_drift_is_rejected(tmp_path: Path) -> None:
    pool_maps = _pool_maps(num_slots=1)
    before = _aggregate(tmp_path, phase="before", pool_maps=pool_maps)
    after = _aggregate(tmp_path, phase="after", pool_maps=pool_maps)
    after["ranks"][0]["full_pool"]["pool_base_debug"] += ROW_BYTES

    report = _full_pool_row_diff(
        before,
        after,
        pool_maps=pool_maps,
        slot_mapping=[0],
    )

    assert not report["passed"]
    assert not report["ranks"][0]["identity_checks"]["pool_base_stable"]


def test_before_descriptor_digest_spoof_is_rejected(tmp_path: Path) -> None:
    pool_maps = _pool_maps(num_slots=1)
    before = _aggregate(tmp_path, phase="before", pool_maps=pool_maps)
    after = _aggregate(tmp_path, phase="after", pool_maps=pool_maps)
    before["ranks"][0]["full_pool"]["pool_sha256"] = "f" * 64

    report = _full_pool_row_diff(
        before,
        after,
        pool_maps=pool_maps,
        slot_mapping=[0],
    )

    assert not report["passed"]
    assert not report["ranks"][0]["digest_checks"][
        "before_snapshot_digest_matches_descriptor"
    ]


@pytest.mark.parametrize(
    "row_index",
    [
        # L44.K.slot1 and L0.V.slot1 when num_slots=2.
        44 * 2 + 1,
        NUM_LAYERS * 2 + 1,
    ],
)
def test_unauthorized_history_change_covers_non_l0_and_v(
    tmp_path: Path,
    row_index: int,
) -> None:
    pool_maps = _pool_maps(num_slots=2)
    before = _aggregate(tmp_path, phase="before", pool_maps=pool_maps)
    after = _aggregate(
        tmp_path,
        phase="after",
        pool_maps=pool_maps,
        changed_rows={0: {row_index}},
    )

    report = _full_pool_row_diff(
        before,
        after,
        pool_maps=pool_maps,
        slot_mapping=[0],
    )

    assert not report["passed"]
    assert report["ranks"][0]["unauthorized_changed_rows"] == 1


def test_inter_invocation_continuity_rejects_drift() -> None:
    previous = [f"rank{rank}-after" for rank in range(TP)]
    assert _inter_invocation_continuity(previous, list(previous))
    current = list(previous)
    current[3] = "rank3-drifted"
    assert not _inter_invocation_continuity(previous, current)


def test_active_slot_nonzero_requires_all_layer_kv_rows(
    tmp_path: Path,
) -> None:
    pool_maps = _pool_maps(num_slots=2)
    active_slot = 1
    active_rows = {
        section * NUM_LAYERS * 2 + layer * 2 + active_slot
        for section in range(len(KV_SECTIONS))
        for layer in range(NUM_LAYERS)
    }
    aggregate = _aggregate(
        tmp_path,
        phase="after",
        pool_maps=pool_maps,
        changed_rows={rank: set(active_rows) for rank in range(TP)},
    )

    report = _active_slot_nonzero_report(
        aggregate,
        pool_maps=pool_maps,
        active_slot=active_slot,
    )

    assert report["passed"]
    snapshot = Path(
        aggregate["ranks"][0]["full_pool"]["snapshot_path"]
    )
    raw = bytearray(snapshot.read_bytes())
    first_active_row = active_slot
    raw[first_active_row * ROW_BYTES : (first_active_row + 1) * ROW_BYTES] = (
        bytes(ROW_BYTES)
    )
    snapshot.write_bytes(raw)

    report = _active_slot_nonzero_report(
        aggregate,
        pool_maps=pool_maps,
        active_slot=active_slot,
    )

    assert not report["passed"]
    assert report["ranks"][0]["nonzero_rows"] == NUM_LAYERS * 2 - 1
