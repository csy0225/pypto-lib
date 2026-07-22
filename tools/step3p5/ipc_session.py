#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed lifecycle contract for live Step3p5 IPC artifacts.

Offline/canonical harnesses may continue to consume the historical map
schemas.  A live vLLM/PyPTO session enables this contract with:

``PYPTO_LIVE_IPC_STRICT=1``
``PYPTO_IPC_SESSION_NONCE=<fresh opaque nonce>``
``PYPTO_IPC_LAUNCH_EPOCH=<unix timestamp created by the launcher>``

Every exporter then records the nonce, launch epoch, producer PID/start tick,
rank/device identity and a heartbeat path in its map and ready manifest.
Importers validate all artifacts *before* calling ``rt.import_ipc_all``.
This prevents an old ACL key/map pair from being imported after its producer
has exited or after a new 8001 session has started.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping

SESSION_SCHEMA_VERSION = 1
_DEFAULT_HEARTBEAT_INTERVAL = 2.0
_DEFAULT_HEARTBEAT_MAX_AGE = 30.0


class IpcSessionError(RuntimeError):
    """A live IPC artifact is stale, incomplete, or owned by another session."""


def _truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


def strict_live_enabled() -> bool:
    return _truthy("PYPTO_LIVE_IPC_STRICT")


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise IpcSessionError(f"strict live IPC requires {name}")
    return value


def current_session() -> tuple[str, float]:
    nonce = _required_env("PYPTO_IPC_SESSION_NONCE")
    raw_epoch = _required_env("PYPTO_IPC_LAUNCH_EPOCH")
    try:
        launch_epoch = float(raw_epoch)
    except ValueError as exc:
        raise IpcSessionError(
            f"PYPTO_IPC_LAUNCH_EPOCH must be a unix timestamp, got {raw_epoch!r}"
        ) from exc
    if launch_epoch <= 0:
        raise IpcSessionError("PYPTO_IPC_LAUNCH_EPOCH must be positive")
    return nonce, launch_epoch


def process_start_ticks(pid: int) -> int:
    """Return Linux ``/proc/<pid>/stat`` field 22 without parsing ``comm`` badly."""
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
    except OSError as exc:
        raise IpcSessionError(f"producer pid {pid} is not alive") from exc
    close = raw.rfind(")")
    if close < 0:
        raise IpcSessionError(f"cannot parse /proc/{pid}/stat")
    fields_after_comm = raw[close + 2 :].split()
    # The first element is field 3 (state); starttime is field 22.
    try:
        return int(fields_after_comm[19])
    except (IndexError, ValueError) as exc:
        raise IpcSessionError(f"cannot read start ticks for pid {pid}") from exc


def _atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(
        f".{target.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    with open(tmp, "wb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp, target)


def atomic_write_json(path: str | os.PathLike[str], obj: Mapping[str, Any]) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


class IpcSessionOwner:
    """Heartbeat owner retained by the exporting process for its lifetime."""

    def __init__(
        self,
        out_dir: str,
        *,
        role: str,
        rank: int,
        device_id: int,
        heartbeat_interval: float | None = None,
    ) -> None:
        nonce, launch_epoch = current_session()
        self.nonce = nonce
        self.launch_epoch = launch_epoch
        self.role = str(role)
        self.rank = int(rank)
        self.device_id = int(device_id)
        self.pid = os.getpid()
        self.start_ticks = process_start_ticks(self.pid)
        self.created_at = time.time()
        if self.created_at + 1e-6 < self.launch_epoch:
            raise IpcSessionError(
                "exporter clock/artifact predates PYPTO_IPC_LAUNCH_EPOCH: "
                f"created={self.created_at} launch={self.launch_epoch}"
            )
        safe_role = "".join(
            char if char.isalnum() or char in "-_" else "_" for char in self.role
        )
        self.heartbeat_path = os.path.abspath(
            os.path.join(
                out_dir,
                f"ipc_heartbeat.{safe_role}.rank{self.rank}.json",
            )
        )
        self._interval = float(
            heartbeat_interval
            if heartbeat_interval is not None
            else os.environ.get(
                "PYPTO_IPC_HEARTBEAT_INTERVAL_SEC",
                _DEFAULT_HEARTBEAT_INTERVAL,
            )
        )
        if self._interval <= 0:
            raise IpcSessionError("IPC heartbeat interval must be positive")
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"pypto-ipc-heartbeat-{safe_role}-r{self.rank}",
            daemon=True,
        )
        self._write_heartbeat()
        self._thread.start()

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": SESSION_SCHEMA_VERSION,
            "session_nonce": self.nonce,
            "launch_epoch": self.launch_epoch,
            "created_at": self.created_at,
            "producer_pid": self.pid,
            "producer_start_ticks": self.start_ticks,
            "producer_role": self.role,
            "producer_rank": self.rank,
            "producer_device_id": self.device_id,
            "heartbeat_path": self.heartbeat_path,
        }

    def _write_heartbeat(self) -> None:
        heartbeat = self.metadata()
        heartbeat["heartbeat_at"] = time.time()
        atomic_write_json(self.heartbeat_path, heartbeat)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._write_heartbeat()
            except Exception:
                # The map owner remains alive, but importers will fail closed
                # when the heartbeat becomes stale.  Do not silently fabricate
                # success after a filesystem failure.
                pass

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self._interval * 2))


def maybe_start_owner(
    out_dir: str,
    *,
    role: str,
    rank: int,
    device_id: int,
) -> IpcSessionOwner | None:
    if not strict_live_enabled():
        return None
    return IpcSessionOwner(
        out_dir,
        role=role,
        rank=rank,
        device_id=device_id,
    )


def attach_session(
    map_obj: dict[str, Any],
    owner: IpcSessionOwner | None,
) -> dict[str, Any]:
    if owner is not None:
        map_obj["ipc_session"] = owner.metadata()
    return map_obj


def write_ready_manifest(
    path: str,
    *,
    map_path: str,
    key_path: str,
    owner: IpcSessionOwner | None,
) -> None:
    if owner is None:
        _atomic_write_bytes(path, b"1\n")
        return
    manifest = {
        "ready_schema_version": 1,
        "ready_at": time.time(),
        "map_path": os.path.abspath(map_path),
        "key_path": os.path.abspath(key_path),
        "ipc_session": owner.metadata(),
    }
    atomic_write_json(path, manifest)


def _as_mapping(value: Any, *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IpcSessionError(f"{where} must be an object")
    return value


def _same_identity(
    session: Mapping[str, Any],
    other: Mapping[str, Any],
    *,
    where: str,
) -> None:
    fields = (
        "session_nonce",
        "launch_epoch",
        "producer_pid",
        "producer_start_ticks",
        "producer_role",
        "producer_rank",
        "producer_device_id",
        "heartbeat_path",
    )
    mismatch = [
        field for field in fields if session.get(field) != other.get(field)
    ]
    if mismatch:
        raise IpcSessionError(f"{where} owner identity mismatch: {mismatch}")


def validate_live_session(
    pool_map: Mapping[str, Any],
    *,
    expected_rank: int | None = None,
    expected_tp: int | None = None,
    expected_device_id: int | None = None,
    expected_role: str | None = None,
    ready_path: str | None = None,
    map_path: str | None = None,
    key_path: str | None = None,
) -> None:
    """Validate one live map/ready/owner tuple.

    No-op unless strict mode is enabled, preserving offline canonical maps.
    """
    if not strict_live_enabled():
        return
    nonce, launch_epoch = current_session()
    session = _as_mapping(pool_map.get("ipc_session"), where="map.ipc_session")
    if session.get("schema_version") != SESSION_SCHEMA_VERSION:
        raise IpcSessionError("unsupported map IPC session schema")
    if session.get("session_nonce") != nonce:
        raise IpcSessionError("map belongs to a different IPC session nonce")
    try:
        artifact_launch = float(session.get("launch_epoch"))
        created_at = float(session.get("created_at"))
        producer_pid = int(session.get("producer_pid"))
        producer_start = int(session.get("producer_start_ticks"))
        producer_rank = int(session.get("producer_rank"))
        producer_device = int(session.get("producer_device_id"))
    except (TypeError, ValueError) as exc:
        raise IpcSessionError("map IPC session has malformed numeric fields") from exc
    if abs(artifact_launch - launch_epoch) > 1e-6:
        raise IpcSessionError(
            "map launch epoch does not match the current launcher session"
        )
    if created_at + 1e-6 < launch_epoch:
        raise IpcSessionError("map was created before the current launch epoch")
    if expected_rank is not None and producer_rank != int(expected_rank):
        raise IpcSessionError(
            f"producer rank {producer_rank} != expected {expected_rank}"
        )
    if expected_device_id is not None and producer_device != int(expected_device_id):
        raise IpcSessionError(
            f"producer device {producer_device} != expected {expected_device_id}"
        )
    if expected_role is not None and session.get("producer_role") != expected_role:
        raise IpcSessionError(
            f"producer role {session.get('producer_role')!r} "
            f"!= expected {expected_role!r}"
        )
    if expected_tp is not None and int(pool_map.get("tp_world_size", -1)) != int(
        expected_tp
    ):
        raise IpcSessionError(
            f"map tp_world_size={pool_map.get('tp_world_size')} "
            f"!= expected {expected_tp}"
        )
    if int(pool_map.get("rank", -1)) != producer_rank:
        raise IpcSessionError("map rank and producer rank disagree")

    actual_start = process_start_ticks(producer_pid)
    if actual_start != producer_start:
        raise IpcSessionError(
            "producer PID was reused or restarted: "
            f"recorded_start={producer_start} actual_start={actual_start}"
        )

    heartbeat_path = str(session.get("heartbeat_path", ""))
    if not heartbeat_path:
        raise IpcSessionError("map IPC session has no heartbeat_path")
    try:
        heartbeat = _as_mapping(
            json.loads(Path(heartbeat_path).read_text(encoding="utf-8")),
            where="heartbeat",
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise IpcSessionError(f"cannot read heartbeat {heartbeat_path}") from exc
    _same_identity(session, heartbeat, where="heartbeat")
    try:
        heartbeat_at = float(heartbeat.get("heartbeat_at"))
    except (TypeError, ValueError) as exc:
        raise IpcSessionError("heartbeat_at is missing or malformed") from exc
    max_age = float(
        os.environ.get(
            "PYPTO_IPC_HEARTBEAT_MAX_AGE_SEC",
            _DEFAULT_HEARTBEAT_MAX_AGE,
        )
    )
    age = time.time() - heartbeat_at
    if age < -5.0 or age > max_age:
        raise IpcSessionError(
            f"producer heartbeat is stale/future: age={age:.3f}s max={max_age:.3f}s"
        )

    if ready_path is not None:
        ready_file = Path(ready_path)
        try:
            ready = _as_mapping(
                json.loads(ready_file.read_text(encoding="utf-8")),
                where="ready manifest",
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise IpcSessionError(
                f"strict live IPC requires a JSON ready manifest: {ready_path}"
            ) from exc
        if ready.get("ready_schema_version") != 1:
            raise IpcSessionError("unsupported ready manifest schema")
        ready_session = _as_mapping(
            ready.get("ipc_session"),
            where="ready.ipc_session",
        )
        _same_identity(session, ready_session, where="ready manifest")
        if map_path is not None and ready.get("map_path") != os.path.abspath(map_path):
            raise IpcSessionError("ready manifest map_path mismatch")
        if key_path is not None and ready.get("key_path") != os.path.abspath(key_path):
            raise IpcSessionError("ready manifest key_path mismatch")
        try:
            ready_at = float(ready.get("ready_at"))
        except (TypeError, ValueError) as exc:
            raise IpcSessionError("ready_at is missing or malformed") from exc
        if ready_at + 1e-6 < created_at:
            raise IpcSessionError("ready manifest predates its map")


def validate_key_file(path: str, *, expected_bytes: int = 256) -> bytes:
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise IpcSessionError(f"cannot read IPC key {path}") from exc
    if len(data) != expected_bytes:
        raise IpcSessionError(
            f"IPC key {path} has {len(data)} bytes, expected {expected_bytes}"
        )
    return data

