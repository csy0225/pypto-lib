# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared process, environment, and report helpers for whole-network CI.

This module deliberately contains no model or PyPTO kernel logic.  The
``run_whole_network_ci`` orchestration layer owns the model-specific gates,
while this module owns the failure-safe mechanics that must be identical for
every hardware run:

* child-process environment scrubbing;
* process-group based timeout and termination;
* per-stage log capture;
* JSON-safe stage results;
* exporter pool artifact cleanup.
"""
from __future__ import annotations

import datetime as _datetime
import json
import os
import signal
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class RunnerError(RuntimeError):
    """A deterministic preflight, stage, or cleanup failure."""


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with a ``Z`` suffix."""
    return (
        _datetime.datetime.now(_datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def parse_int_list(value: str | Iterable[int], *, name: str) -> tuple[int, ...]:
    """Parse a comma-separated list of non-negative integer IDs."""
    if isinstance(value, str):
        raw_items = value.split(",")
    else:
        raw_items = [str(item) for item in value]
    result: list[int] = []
    for raw in raw_items:
        item = raw.strip()
        if not item:
            continue
        try:
            parsed = int(item, 10)
        except ValueError as exc:
            raise RunnerError(f"{name} contains a non-integer value: {item!r}") from exc
        if parsed < 0:
            raise RunnerError(f"{name} contains a negative device ID: {parsed}")
        result.append(parsed)
    if not result:
        raise RunnerError(f"{name} must contain at least one device ID")
    return tuple(result)


def scrub_environment(
    *,
    repo_root: Path,
    base: Mapping[str, str] | None = None,
    overrides: Mapping[str, str | None] | None = None,
) -> dict[str, str]:
    """Build an isolated child environment for the back-8 PyPTO pipeline.

    The front-8 vLLM process is unrelated to the child process environment, but
    inheriting its process-global knobs can still change CANN/HCCL behavior in
    the back-8 workers.  Remove all known visibility, vLLM, HCCL, scheduler,
    affinity, and preload controls.  Runtime/compiler variables such as
    ``PTO_ISA_ROOT`` and ``PTO2_RING_*`` are intentionally preserved.
    """
    env = dict(os.environ if base is None else base)
    for key in list(env):
        if key.startswith("VLLM_"):
            env.pop(key, None)
    for key in (
        "ASCEND_RT_VISIBLE_DEVICES",
        "CPU_AFFINITY_CONF",
        "EPMOE_BYPASS_GATE",
        "HCCL_BUFFSIZE",
        "HCCL_OP_EXPANSION_MODE",
        "LD_PRELOAD",
        "P_DBG_STAGE",
        "P_FILL_BATCH",
        "PYPTO_MEM_PLANNER",
        "PYPTO_WEIGHT_IPC_VA_SHIFT_GB",
        "TASK_QUEUE_ENABLE",
        "SHM_BARRIER",
    ):
        env.pop(key, None)

    # An empty allocator setting is the isolated back-8 convention used by the
    # validated 0162 script.  The runner never exports a front-8 visibility
    # mask; every stage receives explicit physical device IDs.
    env["PYTORCH_NPU_ALLOC_CONF"] = ""
    env["PYTHONPATH"] = (
        str(repo_root)
        if not env.get("PYTHONPATH")
        else str(repo_root) + os.pathsep + env["PYTHONPATH"]
    )

    if overrides:
        for key, value in overrides.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = str(value)
    return env


def command_text(command: Sequence[str]) -> str:
    """Return a shell-safe human-readable command without using a shell."""
    return shlex.join([str(item) for item in command])


def _write_log_header(stream: Any, command: Sequence[str]) -> None:
    stream.write(f"# command: {command_text(command)}\n")
    stream.write(f"# started_utc: {utc_now()}\n")
    stream.flush()


def terminate_process_group(
    process: subprocess.Popen[Any],
    *,
    grace_seconds: float = 10.0,
) -> bool:
    """Terminate a POSIX process group, escalating to SIGKILL if needed."""
    group_id = process.pid

    def group_exists() -> bool:
        try:
            os.killpg(group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    if not group_exists():
        return True
    try:
        os.killpg(group_id, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except OSError:
        if process.poll() is None:
            process.terminate()

    deadline = time.monotonic() + grace_seconds
    while group_exists() and time.monotonic() < deadline:
        if process.poll() is None:
            try:
                process.wait(timeout=min(0.2, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        else:
            time.sleep(0.1)
    if group_exists():
        try:
            os.killpg(group_id, signal.SIGKILL)
        except ProcessLookupError:
            return True
        except OSError:
            if process.poll() is None:
                process.kill()
        deadline = time.monotonic() + grace_seconds
        while group_exists() and time.monotonic() < deadline:
            if process.poll() is None:
                try:
                    process.wait(timeout=min(0.2, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    pass
            else:
                time.sleep(0.1)
    if process.poll() is None:
        try:
            process.wait(timeout=max(0.1, grace_seconds))
        except subprocess.TimeoutExpired:
            return False
    return not group_exists()


@dataclass
class ManagedProcess:
    """A long-lived exporter process and its log stream."""

    name: str
    command: list[str]
    process: subprocess.Popen[Any]
    log_path: Path
    started_at: float
    _log_stream: Any = field(repr=False)

    @property
    def pid(self) -> int:
        return int(self.process.pid)

    def poll(self) -> int | None:
        return self.process.poll()

    def close_log(self) -> None:
        try:
            self._log_stream.close()
        except OSError:
            pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pid": self.pid,
            "command": self.command,
            "log": str(self.log_path),
            "returncode": self.poll(),
            "duration_sec": round(time.monotonic() - self.started_at, 3),
        }


def launch_logged_process(
    *,
    name: str,
    command: Sequence[str],
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
) -> ManagedProcess:
    """Launch one process in its own process group and capture all output."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("w", encoding="utf-8")
    normalized_command = [str(item) for item in command]
    _write_log_header(stream, normalized_command)
    try:
        process = subprocess.Popen(
            normalized_command,
            cwd=str(cwd),
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception:
        stream.close()
        raise
    return ManagedProcess(
        name=name,
        command=normalized_command,
        process=process,
        log_path=log_path,
        started_at=time.monotonic(),
        _log_stream=stream,
    )


@dataclass
class StageResult:
    """Result of a bounded foreground stage."""

    name: str
    command: list[str]
    log_path: Path
    returncode: int | None
    duration_sec: float
    timed_out: bool = False
    error: str = ""
    checks: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.error

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": self.command,
            "log": str(self.log_path),
            "returncode": self.returncode,
            "duration_sec": self.duration_sec,
            "timed_out": self.timed_out,
            "passed": self.passed,
            "error": self.error,
            "checks": self.checks,
        }


def run_logged_command(
    *,
    name: str,
    command: Sequence[str],
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
    timeout_seconds: float,
    watch_processes: Sequence[ManagedProcess] = (),
) -> StageResult:
    """Run a foreground stage with a hard timeout and a dedicated log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("w", encoding="utf-8")
    normalized_command = [str(item) for item in command]
    _write_log_header(stream, normalized_command)
    started = time.monotonic()
    process: subprocess.Popen[Any] | None = None
    timed_out = False
    error = ""
    returncode: int | None = None
    try:
        process = subprocess.Popen(
            normalized_command,
            cwd=str(cwd),
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout_seconds
        while True:
            dead = [
                managed
                for managed in watch_processes
                if managed.poll() is not None
            ]
            if dead:
                dead_text = ", ".join(
                    f"{managed.name}=rc{managed.poll()}" for managed in dead
                )
                error = f"watched exporter exited during stage: {dead_text}"
                terminate_process_group(process)
                returncode = process.poll()
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                error = f"stage exceeded timeout of {timeout_seconds:.1f}s"
                terminate_process_group(process)
                returncode = process.poll()
                break
            try:
                returncode = process.wait(timeout=min(2.0, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
        if not error:
            group_clean = terminate_process_group(process, grace_seconds=1.0)
            if not group_clean:
                error = "stage left a live child process group after the parent exited"
    except OSError as exc:
        error = f"failed to launch stage: {exc}"
    finally:
        stream.write(f"# finished_utc: {utc_now()}\n")
        stream.flush()
        stream.close()
    return StageResult(
        name=name,
        command=normalized_command,
        log_path=log_path,
        returncode=returncode,
        duration_sec=round(time.monotonic() - started, 3),
        timed_out=timed_out,
        error=error,
    )


def wait_for_processes(
    processes: Sequence[ManagedProcess],
    *,
    timeout_seconds: float,
    ready_files: Sequence[Path],
    poll_seconds: float = 2.0,
) -> None:
    """Wait for all ready sentinels while detecting an early child death."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        missing = [path for path in ready_files if not path.exists()]
        if not missing:
            return
        for managed in processes:
            returncode = managed.poll()
            if returncode is not None:
                raise RunnerError(
                    f"{managed.name} exited before readiness with rc={returncode}; "
                    f"log={managed.log_path}"
                )
        if time.monotonic() >= deadline:
            names = ", ".join(str(path) for path in missing[:4])
            suffix = "..." if len(missing) > 4 else ""
            raise RunnerError(
                f"exporters were not ready within {timeout_seconds:.1f}s; "
                f"missing={names}{suffix}"
            )
        time.sleep(poll_seconds)


def stop_processes(
    processes: Sequence[ManagedProcess],
    *,
    stop_file: Path,
    pool_dir: Path,
    timeout_seconds: float = 60.0,
    keep: bool = False,
) -> dict[str, Any]:
    """Stop exporters and remove IPC sentinels/maps only after they exit."""
    result: dict[str, Any] = {
        "requested": not keep,
        "kept": keep,
        "pids": [managed.pid for managed in processes],
        "stopped": False,
        "remaining_pids": [],
        "returncodes": {},
        "removed_files": [],
    }
    if keep:
        for managed in processes:
            managed.close_log()
        result["remaining_pids"] = [
            managed.pid for managed in processes if managed.poll() is None
        ]
        result["returncodes"] = {
            managed.name: managed.poll() for managed in processes
        }
        return result

    try:
        stop_file.write_text("1", encoding="utf-8")
    except OSError as exc:
        result["stop_file_error"] = str(exc)

    deadline = time.monotonic() + timeout_seconds
    for managed in processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            managed.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            terminate_process_group(
                managed.process,
                grace_seconds=min(10.0, max(1.0, remaining)),
            )

    for managed in processes:
        if managed.poll() is None:
            terminate_process_group(managed.process)
        else:
            # The exporter leader can exit while a forked child remains in the
            # same session.  Always probe and clean the process group.
            terminate_process_group(managed.process, grace_seconds=2.0)
        managed.close_log()

    remaining_pids = [managed.pid for managed in processes if managed.poll() is None]
    returncodes = {
        managed.name: managed.poll() for managed in processes
    }
    nonzero_returncodes = {
        name: returncode
        for name, returncode in returncodes.items()
        if returncode not in (None, 0)
    }
    result["remaining_pids"] = remaining_pids
    result["returncodes"] = returncodes
    result["nonzero_returncodes"] = nonzero_returncodes
    result["stopped"] = not remaining_pids and not nonzero_returncodes
    if not remaining_pids:
        result["removed_files"] = remove_pool_artifacts(pool_dir)
    return result


def remove_pool_artifacts(pool_dir: Path) -> list[str]:
    """Remove only exporter-owned sentinels, keys, and maps."""
    patterns = (
        "ready.rank*",
        "STOP",
        "pypto_weight.key.rank*",
        "pypto_weight_map.rank*.json",
        "pypto_weight_map.rank*.json.done",
    )
    removed: list[str] = []
    for pattern in patterns:
        for path in pool_dir.glob(pattern):
            if path.is_file() or path.is_symlink():
                try:
                    path.unlink()
                    removed.append(str(path))
                except OSError:
                    continue
    return removed


def active_exporter_pids(pool_dir: Path) -> list[dict[str, Any]]:
    """Find stage exporter processes that reference a specific pool directory."""
    try:
        completed = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    if completed.returncode != 0:
        return []

    needles = (
        "_stage_whole_mtp3_ipc",
        "_stage_whole_faithful_real_ipc",
    )
    pool_text = str(pool_dir.resolve())
    found: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        command = parts[1]
        if pid == os.getpid() or pool_text not in command:
            continue
        if any(needle in command for needle in needles):
            found.append({"pid": pid, "command": command})
    return found


def tail_text(path: Path, *, lines: int = 80) -> str:
    """Read the last ``lines`` lines of a stage log for failure diagnostics."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"<unable to read {path}: {exc}>"
    return "\n".join(content.splitlines()[-lines:])


def git_metadata(repo_root: Path) -> dict[str, Any]:
    """Collect source identity without making a dirty tree fail the run."""
    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""

    return {
        "sha": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(git("status", "--porcelain")),
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a UTF-8 indented JSON report atomically enough for CI artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def json_safe(value: Any) -> Any:
    """Convert common pathlib/dataclass values into JSON-safe values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return json_safe(value.__dict__)
    return value


__all__ = [
    "ManagedProcess",
    "RunnerError",
    "StageResult",
    "active_exporter_pids",
    "command_text",
    "git_metadata",
    "json_safe",
    "launch_logged_process",
    "parse_int_list",
    "remove_pool_artifacts",
    "run_logged_command",
    "scrub_environment",
    "stop_processes",
    "tail_text",
    "terminate_process_group",
    "utc_now",
    "wait_for_processes",
    "write_json",
]
