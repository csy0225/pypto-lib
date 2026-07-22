#!/usr/bin/env python3
"""Card-free tests for strict live IPC freshness and owner identity."""
from __future__ import annotations

import json
import os
import time

import pytest

from tools.step3p5.ipc_session import (
    IpcSessionError,
    attach_session,
    maybe_start_owner,
    validate_live_session,
    write_ready_manifest,
)


@pytest.fixture
def strict_session(monkeypatch):
    launch = time.time() - 0.1
    monkeypatch.setenv("PYPTO_LIVE_IPC_STRICT", "1")
    monkeypatch.setenv("PYPTO_IPC_SESSION_NONCE", "unit-session-a")
    monkeypatch.setenv("PYPTO_IPC_LAUNCH_EPOCH", str(launch))
    monkeypatch.setenv("PYPTO_IPC_HEARTBEAT_INTERVAL_SEC", "0.05")
    monkeypatch.setenv("PYPTO_IPC_HEARTBEAT_MAX_AGE_SEC", "1.0")
    return launch


def _artifact(tmp_path):
    owner = maybe_start_owner(
        str(tmp_path),
        role="weight",
        rank=0,
        device_id=8,
    )
    assert owner is not None
    pool_map = attach_session(
        {
            "version": 1,
            "rank": 0,
            "tp_world_size": 8,
            "pool_bytes": 512,
            "map": {"x": {"offset": 0, "shape": [256], "dtype": "bfloat16", "nbytes": 512}},
        },
        owner,
    )
    map_path = tmp_path / "map.json"
    key_path = tmp_path / "key.bin"
    ready_path = tmp_path / "map.json.done"
    map_path.write_text(json.dumps(pool_map), encoding="utf-8")
    key_path.write_bytes(b"k" * 256)
    write_ready_manifest(
        str(ready_path),
        map_path=str(map_path),
        key_path=str(key_path),
        owner=owner,
    )
    return owner, pool_map, map_path, key_path, ready_path


def _validate(pool_map, map_path, key_path, ready_path):
    validate_live_session(
        pool_map,
        expected_rank=0,
        expected_tp=8,
        expected_device_id=8,
        expected_role="weight",
        ready_path=str(ready_path),
        map_path=str(map_path),
        key_path=str(key_path),
    )


def test_strict_live_artifact_passes(strict_session, tmp_path):
    owner, pool_map, map_path, key_path, ready_path = _artifact(tmp_path)
    try:
        _validate(pool_map, map_path, key_path, ready_path)
    finally:
        owner.close()


def test_nonce_mismatch_fails_closed(strict_session, tmp_path, monkeypatch):
    owner, pool_map, map_path, key_path, ready_path = _artifact(tmp_path)
    try:
        monkeypatch.setenv("PYPTO_IPC_SESSION_NONCE", "unit-session-b")
        with pytest.raises(IpcSessionError, match="nonce"):
            _validate(pool_map, map_path, key_path, ready_path)
    finally:
        owner.close()


def test_pid_start_mismatch_fails_closed(strict_session, tmp_path):
    owner, pool_map, map_path, key_path, ready_path = _artifact(tmp_path)
    try:
        pool_map["ipc_session"]["producer_start_ticks"] += 1
        with pytest.raises(IpcSessionError, match="reused|restarted"):
            _validate(pool_map, map_path, key_path, ready_path)
    finally:
        owner.close()


def test_stale_heartbeat_fails_closed(strict_session, tmp_path):
    owner, pool_map, map_path, key_path, ready_path = _artifact(tmp_path)
    owner.close()
    heartbeat_path = pool_map["ipc_session"]["heartbeat_path"]
    heartbeat = json.loads(open(heartbeat_path, encoding="utf-8").read())
    heartbeat["heartbeat_at"] = time.time() - 5
    open(heartbeat_path, "w", encoding="utf-8").write(json.dumps(heartbeat))
    with pytest.raises(IpcSessionError, match="heartbeat"):
        _validate(pool_map, map_path, key_path, ready_path)


def test_ready_identity_mismatch_fails_closed(strict_session, tmp_path):
    owner, pool_map, map_path, key_path, ready_path = _artifact(tmp_path)
    try:
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        ready["ipc_session"]["producer_rank"] = 1
        ready_path.write_text(json.dumps(ready), encoding="utf-8")
        with pytest.raises(IpcSessionError, match="ready manifest"):
            _validate(pool_map, map_path, key_path, ready_path)
    finally:
        owner.close()


def test_offline_mode_accepts_legacy_map(monkeypatch):
    monkeypatch.delenv("PYPTO_LIVE_IPC_STRICT", raising=False)
    validate_live_session({"version": 1, "rank": 0, "tp_world_size": 8})

