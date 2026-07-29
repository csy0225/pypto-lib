# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Canonical Step3p5 single-chip hidden-only CI.

The only device flow is:

1. Main 45-layer single-submit hidden-only for eight decode steps;
2. save Main step-0 hidden;
3. selected MTP45/46/47 hidden-only using that exact Main hidden;
4. optionally repeat the selected-MTP gate with fixed BATCH=16.

PyPTO never emits logits, token ids, sampling state, or acceptance state.
The harness uses a CPU tail only as a diagnostic comparison with the pinned
vLLM greedy oracle.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from tests.step3p5.ci._whole_network_ci_common import (
    RunnerError,
    StageResult,
    active_exporter_pids,
    git_metadata,
    json_safe,
    parse_int_list,
    run_logged_command,
    scrub_environment,
    tail_text,
    utc_now,
    write_json,
)


TP = 8
CANONICAL_DEVICES = tuple(range(8, 16))
PROTECTED_FRONT_DEVICES = tuple(range(8))
CANONICAL_HIDDEN_TOKEN = 6127
CANONICAL_FIRST_TOKEN = 303
CANONICAL_MAIN_TOKENS = (303, 1207, 19384, 872, 428, 6127, 4231, 2636)
CANONICAL_MTP_TOKENS = (6178, 410, 303)


@dataclass(frozen=True)
class WholeNetworkConfig:
    repo_root: Path
    ckpt: Path
    devices: tuple[int, ...] = CANONICAL_DEVICES
    protected_devices: tuple[int, ...] = PROTECTED_FRONT_DEVICES
    allow_protected_devices: bool = False
    out: Path = Path("/tmp/n1_single_chip_hidden_ci")
    artifact_dir: Path = Path("/tmp/n1_single_chip_hidden_ci_artifacts")
    active_batch: int = 1
    run_batch16: bool = True
    run_mtp: bool = True
    hidden_token: int = CANONICAL_HIDDEN_TOKEN
    first_token: int = CANONICAL_FIRST_TOKEN
    main_timeout: float = 1800.0
    mtp_timeout: float = 1200.0
    batch16_timeout: float = 1200.0
    dry_run: bool = False
    json_report: Path | None = None
    mtp_oracle_dir: Path | None = None

    @property
    def report_path(self) -> Path:
        return self.json_report or self.artifact_dir / "whole_network_report.json"


@dataclass
class WholeNetworkRun:
    config: WholeNetworkConfig
    env: dict[str, str] = field(default_factory=dict)
    stages: list[StageResult] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)


def _default_ckpt() -> Path:
    return Path(
        os.environ.get(
            "STEP3P5_CKPT_DIR",
            "/data/chensiyu/"
            "step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp",
        )
    )


def _default_out() -> Path:
    return Path(
        os.environ.get(
            "STEP3P5_CI_OUT",
            "/tmp/n1_single_chip_hidden_ci",
        )
    )


def _default_artifact_dir(out: Path) -> Path:
    configured = os.environ.get("STEP3P5_CI_ARTIFACT_DIR")
    if configured:
        return Path(configured)
    return out.parent / f"{out.name}_artifacts"


def config_from_environment() -> WholeNetworkConfig:
    out = _default_out().expanduser().resolve()
    return WholeNetworkConfig(
        repo_root=Path(__file__).resolve().parents[3],
        ckpt=_default_ckpt().expanduser().resolve(),
        devices=parse_int_list(
            os.environ.get(
                "STEP3P5_CI_DEVICES",
                ",".join(map(str, CANONICAL_DEVICES)),
            ),
            name="STEP3P5_CI_DEVICES",
        ),
        protected_devices=parse_int_list(
            os.environ.get(
                "STEP3P5_PROTECTED_DEVICES",
                ",".join(map(str, PROTECTED_FRONT_DEVICES)),
            ),
            name="STEP3P5_PROTECTED_DEVICES",
        ),
        allow_protected_devices=(
            os.environ.get("STEP3P5_ALLOW_PROTECTED_DEVICES") == "1"
        ),
        out=out,
        artifact_dir=_default_artifact_dir(out).expanduser().resolve(),
        active_batch=int(os.environ.get("STEP3P5_ACTIVE_BATCH", "1")),
        run_batch16=os.environ.get("STEP3P5_RUN_BATCH16", "1") != "0",
        hidden_token=int(
            os.environ.get(
                "STEP3P5_HIDDEN_TOKEN",
                str(CANONICAL_HIDDEN_TOKEN),
            )
        ),
        first_token=int(
            os.environ.get(
                "STEP3P5_FIRST_TOKEN",
                str(CANONICAL_FIRST_TOKEN),
            )
        ),
        main_timeout=float(os.environ.get("STEP3P5_MAIN_TIMEOUT", "1800")),
        mtp_timeout=float(os.environ.get("STEP3P5_MTP_TIMEOUT", "1200")),
        batch16_timeout=float(
            os.environ.get("STEP3P5_BATCH16_TIMEOUT", "1200")
        ),
        dry_run=os.environ.get("STEP3P5_DRY_RUN") == "1",
        json_report=(
            Path(os.environ["STEP3P5_JSON_REPORT"]).expanduser().resolve()
            if os.environ.get("STEP3P5_JSON_REPORT")
            else None
        ),
        mtp_oracle_dir=(
            Path(os.environ["STEP3P5_MTP_ORACLE_DIR"]).expanduser().resolve()
            if os.environ.get("STEP3P5_MTP_ORACLE_DIR")
            else None
        ),
    )


def _parse_args(argv: Sequence[str] | None = None) -> WholeNetworkConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=_default_ckpt())
    parser.add_argument(
        "--devices",
        default=",".join(map(str, CANONICAL_DEVICES)),
    )
    parser.add_argument(
        "--protected-devices",
        default=",".join(map(str, PROTECTED_FRONT_DEVICES)),
    )
    parser.add_argument("--allow-protected-devices", action="store_true")
    parser.add_argument("--out", type=Path, default=_default_out())
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--active-batch", type=int, default=1)
    parser.add_argument(
        "--run-batch16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--skip-mtp", action="store_true")
    parser.add_argument("--hidden-token", type=int, default=CANONICAL_HIDDEN_TOKEN)
    parser.add_argument("--first-token", type=int, default=CANONICAL_FIRST_TOKEN)
    parser.add_argument("--main-timeout", type=float, default=1800.0)
    parser.add_argument("--mtp-timeout", type=float, default=1200.0)
    parser.add_argument("--batch16-timeout", type=float, default=1200.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json-report", type=Path)
    parser.add_argument(
        "--mtp-oracle-dir",
        type=Path,
        default=(
            Path(os.environ["STEP3P5_MTP_ORACLE_DIR"])
            if os.environ.get("STEP3P5_MTP_ORACLE_DIR")
            else None
        ),
        help=(
            "Offline MTP oracle dir (dumps/single/mtp3_hidden.pt). Required to "
            "run the MTP stage; not baked into the image (no host paths)."
        ),
    )
    args = parser.parse_args(argv)

    out = args.out.expanduser().resolve()
    artifact_dir = (
        args.artifact_dir.expanduser().resolve()
        if args.artifact_dir is not None
        else _default_artifact_dir(out).expanduser().resolve()
    )
    return WholeNetworkConfig(
        repo_root=Path(__file__).resolve().parents[3],
        ckpt=args.ckpt.expanduser().resolve(),
        devices=parse_int_list(args.devices, name="devices"),
        protected_devices=parse_int_list(
            args.protected_devices,
            name="protected-devices",
        ),
        allow_protected_devices=args.allow_protected_devices,
        out=out,
        artifact_dir=artifact_dir,
        active_batch=args.active_batch,
        run_batch16=args.run_batch16,
        run_mtp=not args.skip_mtp,
        hidden_token=args.hidden_token,
        first_token=args.first_token,
        main_timeout=args.main_timeout,
        mtp_timeout=args.mtp_timeout,
        batch16_timeout=args.batch16_timeout,
        dry_run=args.dry_run,
        json_report=(
            args.json_report.expanduser().resolve()
            if args.json_report is not None
            else None
        ),
        mtp_oracle_dir=(
            args.mtp_oracle_dir.expanduser().resolve()
            if args.mtp_oracle_dir is not None
            else None
        ),
    )


def _validate_timeout(value: float, *, name: str) -> None:
    if not value > 0:
        raise RunnerError(f"{name} must be positive")


def _validate_checkpoint(ckpt: Path) -> dict[str, Any]:
    if not ckpt.is_dir():
        raise RunnerError(f"checkpoint directory does not exist: {ckpt}")
    indexes = (
        ckpt / "quant_model_weights.safetensors.index.json",
        ckpt / "model.safetensors.index.json",
    )
    index_path = next((path for path in indexes if path.is_file()), None)
    if index_path is None:
        raise RunnerError(
            "single-chip native-W8A8 CI requires an inspectable "
            f"safetensors index under {ckpt}"
        )
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot read checkpoint index {index_path}: {exc}") from exc
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RunnerError(f"checkpoint index has no weight_map: {index_path}")

    required_top = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    missing = sorted(required_top.difference(weight_map))
    if missing:
        raise RunnerError(f"checkpoint misses top-level tensors: {missing}")

    missing_w8a8: list[str] = []
    for layer in range(3, 45):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            weight = (
                f"model.layers.{layer}.moe.experts.0."
                f"{projection}.weight"
            )
            for key in (weight, f"{weight}_scale"):
                if key not in weight_map:
                    missing_w8a8.append(key)
    if missing_w8a8:
        raise RunnerError(
            "checkpoint misses native W8A8 routed weights: "
            f"{missing_w8a8[:8]}"
        )

    mtp_presence: dict[str, bool] = {}
    for layer in range(45, 48):
        required_mtp = (
            f"model.layers.{layer}.input_layernorm.weight",
            f"model.layers.{layer}.post_attention_layernorm.weight",
            f"model.layers.{layer}.self_attn.q_proj.weight",
            f"model.layers.{layer}.self_attn.k_proj.weight",
            f"model.layers.{layer}.self_attn.v_proj.weight",
            f"model.layers.{layer}.self_attn.o_proj.weight",
            f"model.layers.{layer}.self_attn.q_norm.weight",
            f"model.layers.{layer}.self_attn.k_norm.weight",
            f"model.layers.{layer}.self_attn.g_proj.weight",
            f"model.layers.{layer}.mlp.gate_proj.weight",
            f"model.layers.{layer}.mlp.up_proj.weight",
            f"model.layers.{layer}.mlp.down_proj.weight",
            f"model.layers.{layer}.enorm.weight",
            f"model.layers.{layer}.hnorm.weight",
            f"model.layers.{layer}.eh_proj.weight",
            f"model.layers.{layer}.transformer.shared_head.norm.weight",
            f"model.layers.{layer}.transformer.shared_head.output.weight",
        )
        absent = [key for key in required_mtp if key not in weight_map]
        mtp_presence[str(layer)] = not absent
        if absent:
            raise RunnerError(f"checkpoint misses MTP{layer} tensors: {absent}")

    missing_shards = sorted(
        {
            str(filename)
            for filename in weight_map.values()
            if not (ckpt / str(filename)).is_file()
        }
    )
    if missing_shards:
        raise RunnerError(
            f"checkpoint index references missing shards: {missing_shards[:8]}"
        )
    return {
        "path": str(ckpt),
        "index": str(index_path),
        "native_w8a8_index_pairs": 42,
        "mtp_layer_presence": mtp_presence,
    }


def preflight(config: WholeNetworkConfig) -> dict[str, Any]:
    if len(config.devices) != TP or len(set(config.devices)) != TP:
        raise RunnerError(f"devices must contain {TP} distinct IDs")
    expected = tuple(range(config.devices[0], config.devices[0] + TP))
    if config.devices != expected:
        raise RunnerError(
            f"devices must be ordered and contiguous: {config.devices}"
        )
    if not config.allow_protected_devices:
        overlap = sorted(set(config.devices).intersection(config.protected_devices))
        if overlap:
            raise RunnerError(
                f"devices {overlap} overlap protected devices "
                f"{config.protected_devices}"
            )
    if not config.repo_root.is_dir():
        raise RunnerError(f"repository root does not exist: {config.repo_root}")
    required_sources = (
        "models/step3p5/decode_fwd.py",
        "models/step3p5/dense_mlp.py",
        "models/step3p5/mtp_hidden_fwd.py",
        "tests/step3p5/harnesses/_stage_main_hidden_only.py",
        "tests/step3p5/harnesses/_stage_mtp_hidden_selected.py",
    )
    missing_sources = [
        relative
        for relative in required_sources
        if not (config.repo_root / relative).is_file()
    ]
    if missing_sources:
        raise RunnerError(f"repository misses canonical sources: {missing_sources}")
    if not 1 <= config.active_batch <= 16:
        raise RunnerError("--active-batch must be in [1,16]")
    if config.hidden_token != CANONICAL_HIDDEN_TOKEN:
        raise RunnerError(
            f"canonical Main input token must be {CANONICAL_HIDDEN_TOKEN}"
        )
    if config.first_token != CANONICAL_FIRST_TOKEN:
        raise RunnerError(
            f"canonical MTP first token must be {CANONICAL_FIRST_TOKEN}"
        )
    for value, name in (
        (config.main_timeout, "main-timeout"),
        (config.mtp_timeout, "mtp-timeout"),
        (config.batch16_timeout, "batch16-timeout"),
    ):
        _validate_timeout(value, name=name)
    if config.artifact_dir == config.out or config.out in config.artifact_dir.parents:
        raise RunnerError("artifact directory must not be inside the IPC directory")
    if config.report_path == config.out or config.out in config.report_path.parents:
        raise RunnerError("JSON report must not be inside the IPC directory")
    active = active_exporter_pids(config.out)
    if active:
        raise RunnerError(
            f"IPC directory already has active exporters: {active}"
        )
    pto_isa_root = os.environ.get("PTO_ISA_ROOT", "")
    if not pto_isa_root or not Path(pto_isa_root).is_dir():
        raise RunnerError("PTO_ISA_ROOT must point to the pinned pto-isa tree")
    return {
        "ok": True,
        "devices": list(config.devices),
        "protected_devices": list(config.protected_devices),
        "active_exporters_before_run": active,
        "checkpoint": _validate_checkpoint(config.ckpt),
        "canonical": {
            "main_input_token": CANONICAL_HIDDEN_TOKEN,
            "main_tokens": list(CANONICAL_MAIN_TOKENS),
            "mtp_first_token": CANONICAL_FIRST_TOKEN,
            "mtp_tokens": list(CANONICAL_MTP_TOKENS),
        },
    }


def _stage_command(module: str, *, args: Sequence[object]) -> list[str]:
    return [sys.executable, "-m", module, *[str(arg) for arg in args]]


def _read_stage_log(stage: StageResult) -> str:
    try:
        return stage.log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"<unable to read stage log: {exc}>"


def _set_stage_checks(stage: StageResult, checks: Mapping[str, Any]) -> None:
    stage.checks.update(checks)
    failures = [name for name, value in checks.items() if value is False]
    if stage.returncode != 0:
        failures.insert(0, f"returncode={stage.returncode}")
    if stage.timed_out:
        failures.insert(0, "timed_out")
    if failures and not stage.error:
        stage.error = "failed checks: " + ", ".join(failures)


def _run_stage(
    state: WholeNetworkRun,
    *,
    name: str,
    command: Sequence[str],
    log_path: Path,
    timeout_seconds: float,
) -> StageResult:
    stage = run_logged_command(
        name=name,
        command=command,
        cwd=state.config.repo_root,
        env=state.env,
        log_path=log_path,
        timeout_seconds=timeout_seconds,
        watch_processes=(),
    )
    state.stages.append(stage)
    return stage


def _require_stage(stage: StageResult) -> None:
    if stage.passed:
        return
    raise RunnerError(
        f"stage {stage.name} failed: {stage.error or stage.returncode}\n"
        f"log={stage.log_path}\n{tail_text(stage.log_path)}"
    )


def _load_json_report(path: Path, *, name: str, stage: StageResult) -> dict[str, Any]:
    if not path.is_file():
        stage.error = f"{name} did not write {path}"
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        stage.error = f"{name} report is invalid JSON: {exc}"
        return {}
    return value if isinstance(value, dict) else {}


def _run_main(state: WholeNetworkRun) -> dict[str, Any]:
    config = state.config
    out = config.out / "main"
    stage = _run_stage(
        state,
        name="main_hidden_8step",
        command=_stage_command(
            "tests.step3p5.harnesses._stage_main_hidden_only",
            args=(
                "--device",
                ",".join(map(str, config.devices)),
                "--ckpt",
                config.ckpt,
                "--out",
                out,
                "--steps",
                8,
            ),
        ),
        log_path=state.report["paths"]["logs"] / "main_hidden_8step.log",
        timeout_seconds=config.main_timeout,
    )
    text = _read_stage_log(stage)
    report_path = out / "main_hidden_only_report.json"
    payload = _load_json_report(report_path, name="Main", stage=stage)
    rows = payload.get("steps", [])
    tokens = [int(row.get("output_token", -1)) for row in rows]
    step0_hidden = out / "main_step00_hidden.pt"
    checks = {
        "process_rc_zero": stage.returncode == 0,
        "result_clean": "RESULT=MAIN_HIDDEN_ONLY_8STEP_TOKEN_EXACT" in text,
        "eight_steps": len(rows) == 8,
        "tokens_exact": tokens == list(CANONICAL_MAIN_TOKENS),
        "step0_hidden_saved": step0_hidden.is_file(),
        "pypto_hidden_only": (
            payload.get("ownership", {}).get("pypto_output")
            == "pre-final-norm BF16 next_hidden only"
        ),
    }
    _set_stage_checks(stage, checks)
    _require_stage(stage)
    return {
        "stage": stage.to_dict(),
        "report_path": str(report_path),
        "step0_hidden": str(step0_hidden),
        "tokens": tokens,
    }


def _run_mtp(
    state: WholeNetworkRun,
    *,
    name: str,
    active_batch: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    config = state.config
    out = config.out / name
    previous_hidden = config.out / "main" / "main_step00_hidden.pt"
    mtp_args = [
        "--device",
        ",".join(map(str, config.devices)),
        "--ckpt",
        config.ckpt,
        "--out",
        out,
        "--previous-hidden",
        previous_hidden,
        "--active-batch",
        active_batch,
    ]
    if config.mtp_oracle_dir is not None:
        mtp_args += ["--oracle-dir", config.mtp_oracle_dir]
    stage = _run_stage(
        state,
        name=name,
        command=_stage_command(
            "tests.step3p5.harnesses._stage_mtp_hidden_selected",
            args=tuple(mtp_args),
        ),
        log_path=state.report["paths"]["logs"] / f"{name}.log",
        timeout_seconds=timeout_seconds,
    )
    text = _read_stage_log(stage)
    report_path = out / "selected_mtp_hidden_report.json"
    payload = _load_json_report(report_path, name="MTP", stage=stage)
    rows = payload.get("reports", [])
    tokens = [int(row.get("output_token", -1)) for row in rows]
    checks = {
        "process_rc_zero": stage.returncode == 0,
        "result_clean": "RESULT=MTP_SELECTED_HIDDEN_DEVICE_PASS" in text,
        "three_layers": len(rows) == 3,
        "tokens_exact": tokens == list(CANONICAL_MTP_TOKENS),
        "main_step0_consumed": previous_hidden.is_file(),
        "pypto_hidden_only": (
            payload.get("ownership", {}).get("pypto_output")
            == "raw BF16 mtp_hidden only"
        ),
        "no_pypto_acceptance": (
            payload.get("ownership", {}).get("acceptance")
            == "not executed by PyPTO"
        ),
    }
    _set_stage_checks(stage, checks)
    _require_stage(stage)
    return {
        "stage": stage.to_dict(),
        "active_batch": active_batch,
        "report_path": str(report_path),
        "tokens": tokens,
    }


def _initial_report(config: WholeNetworkConfig) -> dict[str, Any]:
    logs = config.artifact_dir / "logs"
    return {
        "schema_version": 2,
        "ok": False,
        "started_utc": utc_now(),
        "finished_utc": None,
        "test": {
            "name": "step3p5_single_chip_hidden_only",
            "program_main": "whole_decode_step3p5",
            "program_mtp": "MTP_LAYER_HIDDEN_PROGRAMS",
            "canonical_document": "docs/step3p5/README.md",
            "run_id": time.strftime("%Y%m%d_%H%M%S"),
        },
        "source": git_metadata(config.repo_root),
        "machine": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
        },
        "config": {
            "repo_root": str(config.repo_root),
            "checkpoint": str(config.ckpt),
            "devices": list(config.devices),
            "active_batch": config.active_batch,
            "run_batch16": config.run_batch16,
            "run_mtp": config.run_mtp,
            "dry_run": config.dry_run,
            "timeouts_sec": {
                "main": config.main_timeout,
                "mtp": config.mtp_timeout,
                "batch16": config.batch16_timeout,
            },
        },
        "paths": {
            "out": config.out,
            "artifact_dir": config.artifact_dir,
            "logs": logs,
            "json_report": config.report_path,
        },
        "preflight": None,
        "main": None,
        "mtp_single": None,
        "mtp_batch16": None,
        "cleanup": None,
        "failure": None,
    }


def _install_signal_handlers() -> dict[int, Any]:
    previous: dict[int, Any] = {}

    def interrupt(signum: int, _frame: Any) -> None:
        raise RunnerError(f"received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupt)
        except ValueError:
            previous.clear()
            break
    return previous


def _restore_signal_handlers(previous: Mapping[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def run(config: WholeNetworkConfig) -> dict[str, Any]:
    state = WholeNetworkRun(config=config)
    state.report = _initial_report(config)
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    state.report["paths"]["logs"].mkdir(parents=True, exist_ok=True)
    previous_handlers = _install_signal_handlers()
    try:
        state.report["preflight"] = preflight(config)
        state.env = scrub_environment(
            repo_root=config.repo_root,
            overrides={
                "PTO2_RING_HEAP": os.environ.get(
                    "PTO2_RING_HEAP",
                    "4294967296",
                ),
                "PTO2_RING_TASK_WINDOW": os.environ.get(
                    "PTO2_RING_TASK_WINDOW",
                    "131072",
                ),
                "PTO2_RING_DEP_POOL": os.environ.get(
                    "PTO2_RING_DEP_POOL",
                    "131072",
                ),
            },
        )
        state.report["environment"] = {
            "scrubbed_vllm_front8_controls": True,
            "ascend_rt_visible_devices": state.env.get(
                "ASCEND_RT_VISIBLE_DEVICES"
            ),
            "vllm_keys_present": sorted(
                key for key in state.env if key.startswith("VLLM_")
            ),
        }
        if config.dry_run:
            state.report["dry_run"] = {
                "ok": True,
                "message": "preflight passed; no device stage launched",
            }
        else:
            state.report["main"] = _run_main(state)
            if config.run_mtp and config.mtp_oracle_dir is None:
                skip = {
                    "skipped": True,
                    "reason": (
                        "no --mtp-oracle-dir / STEP3P5_MTP_ORACLE_DIR; the MTP "
                        "oracle is not baked into the image"
                    ),
                }
                state.report["mtp_single"] = skip
                state.report["mtp_batch16"] = dict(skip)
            elif config.run_mtp:
                state.report["mtp_single"] = _run_mtp(
                    state,
                    name="mtp_hidden_single",
                    active_batch=config.active_batch,
                    timeout_seconds=config.mtp_timeout,
                )
                if config.run_batch16:
                    state.report["mtp_batch16"] = _run_mtp(
                        state,
                        name="mtp_hidden_batch16",
                        active_batch=16,
                        timeout_seconds=config.batch16_timeout,
                    )
            else:
                state.report["mtp_single"] = {
                    "skipped": True,
                    "reason": "--skip-mtp",
                }
                state.report["mtp_batch16"] = {
                    "skipped": True,
                    "reason": "--skip-mtp",
                }
    except Exception as exc:  # noqa: BLE001
        state.report["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        state.report["cleanup"] = {
            "requested": False,
            "skipped": True,
            "active_exporters_after_cleanup": active_exporter_pids(config.out),
        }
        state.report["finished_utc"] = utc_now()
        state.report["stages"] = [stage.to_dict() for stage in state.stages]
        state.report["ok"] = (
            state.report["failure"] is None
            and state.report["preflight"] is not None
            and all(stage.passed for stage in state.stages)
            and not state.report["cleanup"]["active_exporters_after_cleanup"]
        )
        write_json(config.report_path, json_safe(state.report))
        _restore_signal_handlers(previous_handlers)
    return json_safe(state.report)


def _print_summary(report: Mapping[str, Any]) -> None:
    print(f"SINGLE_CHIP_HIDDEN_CI={'PASS' if report.get('ok') else 'FAIL'}")
    print(f"report={report.get('paths', {}).get('json_report')}")
    if report.get("failure"):
        print(f"failure={report['failure'].get('message')}", file=sys.stderr)
    for stage in report.get("stages", []):
        print(
            f"stage={stage['name']} rc={stage['returncode']} "
            f"duration={stage['duration_sec']}s passed={stage['passed']}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    report = run(_parse_args(argv))
    _print_summary(report)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
