# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""PERF-C1 persistent canonical Main epoch/liveness probe.

The canonical Main graph has layer-local C1 epochs ``1..42``.  This probe
compiles that one canonical program, prepares one resident
``WholeDecodeHolder`` once, and sends at least six decode invocations through
the same prepared runtime.  A parent process owns the timeout watchdog while a
child process owns the persistent holder; this makes a stalled device dispatch
observable without pretending that a Python thread can safely interrupt an
in-flight device call.

The probe is deliberately liveness-only.  It does not alter
``models.step3p5.decode_fwd.py`` and does not claim numerical precision.

Examples::

    # Compile only, no exporter/device allocation:
    python -m tests.step3p5.probes._probe_c1_epoch_liveness \
        --compile-only --platform a2a3sim --out /tmp/c1-compile

    # Device run on cards 8..15.  Exporters are started once and the holder
    # remains resident for all six invocations:
    python -m tests.step3p5.probes._probe_c1_epoch_liveness \
        --platform a2a3 --devices 8,9,10,11,12,13,14,15 \
        --ckpt /data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp \
        --out /tmp/c1-live --invocations 6 --timeout 900

    # Reuse exporters that are already ready:
    python -m tests.step3p5.probes._probe_c1_epoch_liveness \
        --reuse-exporters --platform a2a3 \
        --devices 8,9,10,11,12,13,14,15 \
        --ckpt /data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp \
        --out /tmp/c1-live --invocations 6
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


_ROOT = Path(__file__).resolve().parents[3]
TP = 8
BATCH = 16
HIDDEN = 4096
BLOCK_SIZE = 128
DEFAULT_CKPT = "/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp"


def _ensure_repo_on_path() -> None:
    root = str(_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _parse_devices(value: str) -> list[int]:
    devices = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(devices) != TP or len(set(devices)) != TP:
        raise ValueError(f"expected {TP} distinct devices, got {devices}")
    return devices


def _configure_shape_env(*, num_blocks: int) -> None:
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    physical_blocks = int(num_blocks) + 15
    os.environ["PYPTO_STEP3P5_MAX_SEQ"] = str(int(num_blocks) * BLOCK_SIZE)
    os.environ["PYPTO_STEP3P5_BLOCK_TABLE_FLAT"] = str(BATCH * int(num_blocks))
    os.environ["PYPTO_STEP3P5_KV_CACHE_ROWS"] = str(
        45 * physical_blocks * BLOCK_SIZE
    )
    os.environ["PYPTO_STEP3P5_ROPE_SEQ"] = str(int(num_blocks) * BLOCK_SIZE)


def _source_epoch_contract() -> dict[str, Any]:
    """Reuse the lowered-IR probe's source contract without compiling twice."""
    _ensure_repo_on_path()
    from tests.step3p5.probes._inspect_c1_lowered_ir import (  # noqa: PLC0415
        _CANONICAL_SOURCE,
        _source_contract,
    )

    return _source_contract(_CANONICAL_SOURCE, canonical=True)


def _load_embedding_row(ckpt: str, token: int) -> torch.Tensor:
    import safetensors.torch as st  # noqa: PLC0415

    index_path = Path(ckpt) / "quant_model_weights.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard = index["weight_map"]["model.embed_tokens.weight"]
    with st.safe_open(str(Path(ckpt) / shard), framework="pt") as handle:
        row = handle.get_slice("model.embed_tokens.weight")[int(token), :]
    if tuple(row.shape) != (HIDDEN,):
        raise ValueError(f"embedding row shape={tuple(row.shape)}")
    return row.to(torch.bfloat16).contiguous()


def _ctx1_metadata(
    *,
    num_blocks: int,
    valid_rows: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build safe ctx=1 metadata, including allocator-owned padding rows."""
    if not 1 <= valid_rows <= BATCH:
        raise ValueError(f"valid_rows must be in [1,{BATCH}], got {valid_rows}")
    user_batch = BATCH
    seq = torch.ones(user_batch, dtype=torch.int32)
    pos = torch.zeros(user_batch, dtype=torch.int32)
    table = torch.zeros(user_batch, num_blocks, dtype=torch.int32)
    slots = torch.zeros(user_batch, dtype=torch.int32)
    from tools.step3p5.kv_padding import (  # noqa: PLC0415
        make_padding_reserve,
        validate_fixed_batch_metadata,
    )

    reserve = make_padding_reserve(num_blocks, num_blocks + 15)
    for row, block_id in enumerate(
        reserve.padding_block_ids[: BATCH - valid_rows],
        start=valid_rows,
    ):
        table[row, 0] = int(block_id)
        slots[row] = int(block_id) * BLOCK_SIZE
    validate_fixed_batch_metadata(
        seq_lens=seq,
        positions=pos,
        block_table=table,
        slot_mapping=slots,
        valid_rows=valid_rows,
        reserve=reserve,
        where="PERF-C1 liveness ctx=1",
    )
    return seq, pos, table, slots


def _holder_build(
    *,
    devices: list[int],
    out: Path,
    ckpt: str,
    platform: str,
    kv_ipc: bool,
) -> Any:
    _ensure_repo_on_path()
    from tools.step3p5.whole_decode_holder import (  # noqa: PLC0415
        WholeDecodeHolder,
    )

    return WholeDecodeHolder(
        device_ids=devices,
        out_dir=str(out),
        ckpt=ckpt,
        platform=platform,
        kv_ipc=kv_ipc,
    ).build()


def _compile_only(args: argparse.Namespace) -> int:
    """Compile canonical Main once without preparing a device runtime."""
    _configure_shape_env(num_blocks=args.num_blocks)
    os.environ.setdefault("PYPTO_PROG_BUILD_DIR", str(Path(args.out) / "build"))
    holder = _holder_build(
        devices=_parse_devices(args.devices),
        out=Path(args.out),
        ckpt=args.ckpt,
        platform=args.platform,
        kv_ipc=False,
    )
    source = _source_epoch_contract()
    report = {
        "kind": "PERF-C1-compile-only",
        "status": "PASS" if source["pass"] else "NO-GO",
        "program": holder.program_name,
        "compiled_output_dir": str(holder.compiled.output_dir),
        "epoch_contract": source,
        "note": "compile-only does not prove device liveness or task-DAG resource edges",
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if source["pass"] else 1


def _worker_send(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    sys.stdout.flush()


def _worker_main(args: argparse.Namespace) -> int:
    """Child process: build and retain exactly one resident holder."""
    _configure_shape_env(num_blocks=args.num_blocks)
    os.environ.setdefault("PYPTO_PROG_BUILD_DIR", str(Path(args.out) / "build"))
    devices = _parse_devices(args.devices)
    out = Path(args.out)
    source = _source_epoch_contract()
    if not source["pass"]:
        _worker_send({"status": "error", "stage": "source_contract", "detail": source})
        return 2

    # Compilation and runtime libraries are intentionally noisy.  Keep the
    # stdout channel machine-readable for the parent watchdog.
    with contextlib.redirect_stdout(sys.stderr):
        holder = _holder_build(
            devices=devices,
            out=out,
            ckpt=args.ckpt,
            platform=args.platform,
            kv_ipc=True,
        )
        context = holder
        context.__enter__()

    embedding = _load_embedding_row(args.ckpt, args.seed_token).unsqueeze(0)
    seq, pos, table, slots = _ctx1_metadata(
        num_blocks=args.num_blocks,
        valid_rows=args.active_rows,
    )
    _worker_send(
        {
            "status": "ready",
            "program": holder.program_name,
            "compiled_output_dir": str(holder.compiled.output_dir),
            "epoch_min": 1,
            "epoch_max": 42,
            "persistent_holder": True,
        }
    )
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            request = json.loads(line)
            if request.get("command") == "stop":
                _worker_send({"status": "stopped"})
                return 0
            if request.get("command") != "run":
                _worker_send(
                    {
                        "status": "error",
                        "stage": "protocol",
                        "detail": f"unknown command {request.get('command')!r}",
                    }
                )
                continue

            invocation = int(request.get("invocation", -1))
            started = time.monotonic()
            with contextlib.redirect_stdout(sys.stderr):
                holder.set_live_step(
                    embedding.expand(args.active_rows, -1).contiguous(),
                    seq_lens=seq,
                    positions=pos,
                    block_table=table,
                    slot_mapping=slots,
                )
                result = holder.run()
            elapsed = time.monotonic() - started
            hidden = result.get("next_hidden")
            if hidden is None:
                raise RuntimeError("canonical holder returned no next_hidden")
            active = hidden[:, : args.active_rows]
            finite = bool(torch.isfinite(active).all().item())
            nonzero = int(torch.count_nonzero(active.abs().amax(dim=-1) > 0).item())
            checksum = float(active.float().abs().sum().item())
            _worker_send(
                {
                    "status": "ok" if finite else "error",
                    "invocation": invocation,
                    "elapsed_sec": elapsed,
                    "holder_last_run_sec": float(holder._last_run_sec),
                    "shape": list(hidden.shape),
                    "active_finite": finite,
                    "active_nonzero_rows": nonzero,
                    "active_abs_sum": checksum,
                    "epoch_min": 1,
                    "epoch_max": 42,
                }
            )
    finally:
        with contextlib.redirect_stdout(sys.stderr):
            context.__exit__(None, None, None)
    return 0


def _readline_timeout(stream: Any, timeout: float) -> str | None:
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        readable, _, _ = select.select([stream], [], [], min(0.2, remaining))
        if readable:
            line = stream.readline()
            return line if line else None
    return None


def _terminate_gracefully(proc: subprocess.Popen[str], *, reason: str) -> dict[str, Any]:
    """Stop the watchdog child without using SIGKILL."""
    result: dict[str, Any] = {"reason": reason, "terminated": False}
    if proc.poll() is None:
        proc.terminate()  # SIGTERM; never use kill -9 for a device diagnostic.
        result["terminated"] = True
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            result["wait_timeout"] = True
    result["returncode"] = proc.returncode
    return result


def _start_exporters(args: argparse.Namespace, devices: list[int]) -> tuple[list[Any], Any]:
    """Use the existing exporter harness without changing it."""
    _ensure_repo_on_path()
    from tests.step3p5.harnesses._stage_main_hidden_only import (  # noqa: PLC0415
        _start_exporters as start,
    )

    harness_args = SimpleNamespace(
        out=str(args.out),
        ckpt=args.ckpt,
        num_blocks=args.num_blocks,
        kv_probe=False,
    )
    return start(harness_args, devices), harness_args


def _stop_exporters(out: Path, processes: list[Any]) -> None:
    _ensure_repo_on_path()
    from tests.step3p5.harnesses._stage_main_hidden_only import (  # noqa: PLC0415
        _stop_exporters as stop,
    )

    stop(out, processes)


def _device_main(args: argparse.Namespace) -> int:
    devices = _parse_devices(args.devices)
    if args.invocations < 6:
        raise ValueError("--invocations must be at least 6 for the C1 gate")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    _configure_shape_env(num_blocks=args.num_blocks)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYPTO_PROG_BUILD_DIR", str(out / "build"))
    source = _source_epoch_contract()
    if not source["pass"]:
        print(json.dumps({"status": "NO-GO", "epoch_contract": source}, indent=2))
        return 1

    exporters: list[Any] = []
    worker: subprocess.Popen[str] | None = None
    worker_log = open(out / "c1_worker.log", "w", encoding="utf-8")
    results: list[dict[str, Any]] = []
    failure: dict[str, Any] | None = None
    try:
        if not args.reuse_exporters:
            exporters, _ = _start_exporters(args, devices)
        else:
            missing = [
                rank
                for rank in range(TP)
                if not (out / f"ready.rank{rank}").exists()
            ]
            if missing:
                raise RuntimeError(
                    f"--reuse-exporters requested but missing ready ranks {missing}"
                )

        command = [
            sys.executable,
            "-m",
            "tests.step3p5.probes._probe_c1_epoch_liveness",
            "--worker",
            "--devices",
            args.devices,
            "--platform",
            args.platform,
            "--ckpt",
            args.ckpt,
            "--out",
            str(out),
            "--num-blocks",
            str(args.num_blocks),
            "--active-rows",
            str(args.active_rows),
            "--seed-token",
            str(args.seed_token),
        ]
        worker = subprocess.Popen(
            command,
            cwd=str(_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=worker_log,
            text=True,
            bufsize=1,
        )
        ready_line = _readline_timeout(worker.stdout, args.compile_timeout)
        if not ready_line:
            failure = {
                "stage": "compile_or_prepare",
                "reason": "persistent worker did not become ready before timeout",
            }
            failure.update(_terminate_gracefully(worker, reason=failure["reason"]))
        else:
            ready = json.loads(ready_line)
            if ready.get("status") != "ready":
                failure = {"stage": "compile_or_prepare", "response": ready}
            else:
                for invocation in range(args.invocations):
                    assert worker.stdin is not None
                    worker.stdin.write(
                        json.dumps(
                            {"command": "run", "invocation": invocation}
                        )
                        + "\n"
                    )
                    worker.stdin.flush()
                    line = _readline_timeout(worker.stdout, args.timeout)
                    if not line:
                        failure = {
                            "stage": "device_liveness",
                            "invocation": invocation,
                            "reason": (
                                f"persistent holder invocation exceeded {args.timeout}s "
                                "or worker exited without a response"
                            ),
                        }
                        failure.update(
                            _terminate_gracefully(
                                worker, reason=failure["reason"]
                            )
                        )
                        break
                    response = json.loads(line)
                    results.append(response)
                    if response.get("status") != "ok":
                        failure = {
                            "stage": "device_liveness",
                            "invocation": invocation,
                            "response": response,
                        }
                        break
                if failure is None and worker.poll() is None:
                    worker.stdin.write(json.dumps({"command": "stop"}) + "\n")
                    worker.stdin.flush()
                    _readline_timeout(worker.stdout, 30.0)
                    worker.wait(timeout=30)
    except Exception as exc:  # noqa: BLE001 - turn probe failures into evidence
        failure = {
            "stage": "probe_driver",
            "reason": repr(exc),
        }
        if worker is not None:
            failure.update(_terminate_gracefully(worker, reason=repr(exc)))
    finally:
        if worker is not None and worker.poll() is None:
            _terminate_gracefully(worker, reason="probe cleanup")
        worker_log.close()
        if exporters:
            _stop_exporters(out, exporters)

    passed = (
        failure is None
        and len(results) >= 6
        and all(
            item.get("status") == "ok"
            and item.get("active_finite") is True
            and item.get("epoch_min") == 1
            and item.get("epoch_max") == 42
            for item in results
        )
    )
    report = {
        "kind": "PERF-C1-persistent-epoch-liveness",
        "status": "PASS" if passed else "NO-GO",
        "program": "whole_decode_step3p5",
        "persistent_holder": True,
        "epoch_protocol": {
            "first": 1,
            "last": 42,
            "invocations_requested": args.invocations,
            "invocations_completed": len(results),
        },
        "devices": devices,
        "results": results,
        "failure": failure,
        "worker_log": str(out / "c1_worker.log"),
        "note": (
            "This is a liveness probe only; PASS does not imply B3/C1/G1 "
            "numerical precision or immutable-image release."
        ),
    }
    report_path = out / "c1_epoch_liveness.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"C1_LIVENESS_REPORT={report_path}", flush=True)
    return 0 if passed else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PERF-C1 canonical persistent epoch/liveness probe."
    )
    parser.add_argument("--platform", default="a2a3", choices=("a2a3", "a2a3sim"))
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-blocks", type=int, default=32)
    parser.add_argument("--active-rows", type=int, default=1)
    parser.add_argument("--seed-token", type=int, default=6127)
    parser.add_argument("--invocations", type=int, default=6)
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="per-invocation response timeout in seconds",
    )
    parser.add_argument(
        "--compile-timeout",
        type=float,
        default=3600.0,
        help="compile+prepare timeout in seconds",
    )
    parser.add_argument("--reuse-exporters", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.worker:
        return _worker_main(args)
    if args.compile_only:
        return _compile_only(args)
    return _device_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
