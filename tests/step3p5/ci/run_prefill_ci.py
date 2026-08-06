# Copyright (c) PyPTO Contributors.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Canonical Step3p5 single-chip prefill hidden-only CI.

This is the prefill dual of ``tests/step3p5/ci/run_whole_network_ci.py`` (the
decode CI runner).  The decode runner drives three stages (Main 8-step +
MTP single + MTP batch16); prefill has **no MTP**, so this runner drives a
single device stage:

1. ``tests.step3p5.harnesses._stage_prefill_hidden_only`` -- the prefill
   hidden-only harness, which runs the full 45-layer prefill forward via the
   canonical ``whole_prefill_step3p5`` program and returns the pre-final-norm
   BF16 hidden state (the hidden-only boundary, design §2.2 invariant 1).

The prefill gate for v1 is **hidden-only**: process rc, the
``RESULT=PREFILL_HIDDEN_DEVICE_PASS`` liveness marker, ownership
(``pypto_output == "pre-final-norm BF16 hidden only"``), hidden finiteness,
and hidden shape.  It is NOT token-exact -- there is no greedy oracle
comparison for prefill in v1 (design §5.1.3), so the canonical prompt tokens
are minimal placeholders.

IS_SCAFFOLD awareness
---------------------
``models/step3p5/prefill_layer_single_chip_hidden.py`` currently has
``IS_SCAFFOLD = True``: the ``whole_chip_orch`` body is a placeholder and
``WholePrefillHolder.build`` refuses to compile.  A full device run therefore
fails at the prefill stage until the P1 kernel work (token-tiling, W8A8,
attention fixes, dual-index) completes and ``IS_SCAFFOLD`` is flipped to
``False``.  The runner itself is card-free: ``--help`` and ``--dry-run``
(preflight only, no device stage launched) work without any NPU card.

active_exporter_pids limitation
-------------------------------
The shared ``active_exporter_pids`` helper in
``_whole_network_ci_common`` only recognizes the decode stage modules
(``_stage_main_hidden_only`` / ``_stage_mtp_hidden_selected``); it will not
catch prefill exporters (``_stage_prefill_hidden_only``) pointing at the same
IPC pool.  This runner supplements it with a prefill-specific scan
(``_active_prefill_exporter_pids``) so the preflight stale-exporter guard is
correct for the prefill track.  The common module is shared and must not be
edited.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import signal
import subprocess
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
# Prefill prompt length T = PREFILL_BATCH(1) * PREFILL_SEQ(128) = 128, mirroring
# models/step3p5/prefill_qkv_proj_rope.py:PREFILL_T.  The hidden-only gate
# checks shape [TP, PREFILL_T, HIDDEN] (or the unsliced [PREFILL_T, HIDDEN]).
PREFILL_T = 128
HIDDEN = 4096
# Minimal canonical prompt for v1.  The prefill gate is hidden-only (finite +
# shape + ownership), NOT token-exact -- no greedy oracle is compared for
# prefill in v1 (design §5.1.3) -- so these tokens are placeholders.  They are
# reused from the decode canonical Main tokens only because both tracks share
# the same checkpoint embedding; the harness pads/handles filling to PREFILL_T.
CANONICAL_PREFILL_PROMPT_TOKENS = (303, 1207, 19384, 872, 428, 6127, 4231, 2636)


@dataclass(frozen=True)
class PrefillCiConfig:
    repo_root: Path
    ckpt: Path
    devices: tuple[int, ...] = CANONICAL_DEVICES
    protected_devices: tuple[int, ...] = PROTECTED_FRONT_DEVICES
    allow_protected_devices: bool = False
    out: Path = Path("/tmp/n1_single_chip_prefill_hidden_ci")
    artifact_dir: Path = Path("/tmp/n1_single_chip_prefill_hidden_ci_artifacts")
    prefill_t: int = PREFILL_T
    prompt_tokens: tuple[int, ...] = CANONICAL_PREFILL_PROMPT_TOKENS
    prefill_timeout: float = 1800.0
    dry_run: bool = False
    json_report: Path | None = None

    @property
    def report_path(self) -> Path:
        return self.json_report or self.artifact_dir / "prefill_ci_report.json"


@dataclass
class PrefillCiRun:
    config: PrefillCiConfig
    env: dict[str, str] = field(default_factory=dict)
    stages: list[StageResult] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)


def _default_ckpt() -> Path:
    # No username path: this default mirrors tools/step3p5/whole_prefill_sidecar.py
    # and the probe/harness convention (no host-private info baked in).
    return Path(
        os.environ.get(
            "STEP3P5_CKPT_DIR",
            "/mnt/hw910test-jfs/models/"
            "step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp",
        )
    )


def _default_out() -> Path:
    return Path(
        os.environ.get(
            "STEP3P5_CI_OUT",
            "/tmp/n1_single_chip_prefill_hidden_ci",
        )
    )


def _default_artifact_dir(out: Path) -> Path:
    configured = os.environ.get("STEP3P5_CI_ARTIFACT_DIR")
    if configured:
        return Path(configured)
    return out.parent / f"{out.name}_artifacts"


def config_from_environment() -> PrefillCiConfig:
    out = _default_out().expanduser().resolve()
    return PrefillCiConfig(
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
        prefill_t=int(os.environ.get("STEP3P5_PREFILL_T", str(PREFILL_T))),
        prompt_tokens=parse_int_list(
            os.environ.get(
                "STEP3P5_PREFILL_PROMPT_TOKENS",
                ",".join(map(str, CANONICAL_PREFILL_PROMPT_TOKENS)),
            ),
            name="STEP3P5_PREFILL_PROMPT_TOKENS",
        ),
        prefill_timeout=float(os.environ.get("STEP3P5_PREFILL_TIMEOUT", "1800")),
        dry_run=os.environ.get("STEP3P5_DRY_RUN") == "1",
        json_report=(
            Path(os.environ["STEP3P5_JSON_REPORT"]).expanduser().resolve()
            if os.environ.get("STEP3P5_JSON_REPORT")
            else None
        ),
    )


def _parse_args(argv: Sequence[str] | None = None) -> PrefillCiConfig:
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
    parser.add_argument(
        "--prefill-t",
        type=int,
        default=PREFILL_T,
        help=(
            "prefill prompt length T (default 128 = PREFILL_BATCH*PREFILL_SEQ); "
            "the hidden-only gate checks shape [TP, T, HIDDEN] or [T, HIDDEN]"
        ),
    )
    parser.add_argument(
        "--prompt-tokens",
        default=",".join(map(str, CANONICAL_PREFILL_PROMPT_TOKENS)),
        help=(
            "comma list of canonical prompt token ids (v1 placeholders; the "
            "prefill gate is hidden-only, not token-exact)"
        ),
    )
    parser.add_argument("--prefill-timeout", type=float, default=1800.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json-report", type=Path)
    args = parser.parse_args(argv)

    out = args.out.expanduser().resolve()
    artifact_dir = (
        args.artifact_dir.expanduser().resolve()
        if args.artifact_dir is not None
        else _default_artifact_dir(out).expanduser().resolve()
    )
    return PrefillCiConfig(
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
        prefill_t=args.prefill_t,
        prompt_tokens=parse_int_list(args.prompt_tokens, name="prompt-tokens"),
        prefill_timeout=args.prefill_timeout,
        dry_run=args.dry_run,
        json_report=(
            args.json_report.expanduser().resolve()
            if args.json_report is not None
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
            "single-chip prefill hidden-only CI requires an inspectable "
            f"safetensors index under {ckpt}"
        )
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot read checkpoint index {index_path}: {exc}") from exc
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RunnerError(f"checkpoint index has no weight_map: {index_path}")

    # Prefill uses the SAME checkpoint as decode (same model).  The top-level
    # tensors (embed/norm/lm_head) and the native W8A8 routed expert weights
    # must be present: the scaffold is BF16 but the checkpoint still carries
    # the W8A8 weights that prefill will consume once P1 W8A8 lands.
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

    # MTP presence is INFORMATIONAL ONLY for prefill.  Prefill CI is Main-only
    # (no MTP stage), so MTP tensor absence must NOT fail the prefill gate.
    # The checkpoint is shared with the decode track, so the tensors are
    # expected to exist; decode raises on absence, prefill does not.
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
        "mtp_required_by_prefill_ci": False,
    }


def _active_prefill_exporter_pids(pool_dir: Path) -> list[dict[str, Any]]:
    """Prefill-specific supplement to the shared ``active_exporter_pids``.

    The common helper in ``_whole_network_ci_common`` filters by the decode
    stage module names (``_stage_main_hidden_only`` /
    ``_stage_mtp_hidden_selected``) and therefore cannot see prefill exporters
    (``_stage_prefill_hidden_only``) pointing at the same IPC pool.  This
    helper mirrors the same ``ps`` + ``--out`` pool-dir match for the prefill
    module only; it is intentionally narrow so it stays in sync with the
    common module's matching rules without editing that shared file.
    """
    try:
        completed = subprocess.run(
            ["ps", "-eo", "pid=,comm=,args="],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    if completed.returncode != 0:
        return []

    module = "tests.step3p5.harnesses._stage_prefill_hidden_only"
    pool_text = str(pool_dir.resolve())
    found: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=2)
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        comm = parts[1]
        command = parts[2]
        if pid == os.getpid() or not comm.startswith("python"):
            continue
        if module not in command:
            continue
        try:
            argv = shlex.split(command)
            out_index = argv.index("--out") + 1
        except (ValueError, IndexError):
            continue
        if out_index >= len(argv):
            continue
        try:
            command_pool = str(Path(argv[out_index]).resolve())
        except OSError:
            continue
        if command_pool == pool_text:
            found.append({"pid": pid, "command": command})
    return found


def preflight(config: PrefillCiConfig) -> dict[str, Any]:
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
    # Prefill CI is prefill-only; decode sources (decode_fwd.py / mtp_hidden_fwd.py
    # / _stage_main_hidden_only / _stage_mtp_hidden_selected) are a separate track
    # and are intentionally NOT required here.
    required_sources = (
        "models/step3p5/prefill_layer_single_chip_hidden.py",
        "models/step3p5/prefill_fwd.py",
        "tests/step3p5/harnesses/_stage_prefill_hidden_only.py",
    )
    missing_sources = [
        relative
        for relative in required_sources
        if not (config.repo_root / relative).is_file()
    ]
    if missing_sources:
        raise RunnerError(f"repository misses canonical sources: {missing_sources}")
    if config.prefill_t <= 0:
        raise RunnerError("--prefill-t must be positive")
    _validate_timeout(config.prefill_timeout, name="prefill-timeout")
    if config.artifact_dir == config.out or config.out in config.artifact_dir.parents:
        raise RunnerError("artifact directory must not be inside the IPC directory")
    if config.report_path == config.out or config.out in config.report_path.parents:
        raise RunnerError("JSON report must not be inside the IPC directory")
    # Stale-exporter guard: the common helper catches decode exporters; the
    # prefill-specific supplement catches prefill exporters (see limitation note
    # in the module docstring).
    active = active_exporter_pids(config.out)
    prefill_active = _active_prefill_exporter_pids(config.out)
    if active or prefill_active:
        raise RunnerError(
            f"IPC directory already has active exporters: {active + prefill_active}"
        )
    pto_isa_root = os.environ.get("PTO_ISA_ROOT", "")
    if not pto_isa_root or not Path(pto_isa_root).is_dir():
        raise RunnerError("PTO_ISA_ROOT must point to the pinned pto-isa tree")
    return {
        "ok": True,
        "devices": list(config.devices),
        "protected_devices": list(config.protected_devices),
        "active_exporters_before_run": active,
        "active_prefill_exporters_before_run": prefill_active,
        "checkpoint": _validate_checkpoint(config.ckpt),
        "canonical": {
            "prefill_prompt_tokens": list(config.prompt_tokens),
            "prefill_t": config.prefill_t,
            "hidden_dim": HIDDEN,
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
    state: PrefillCiRun,
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


def _run_prefill(state: PrefillCiRun) -> dict[str, Any]:
    config = state.config
    out = config.out / "prefill"
    stage = _run_stage(
        state,
        name="prefill_hidden",
        command=_stage_command(
            "tests.step3p5.harnesses._stage_prefill_hidden_only",
            args=(
                "--device",
                ",".join(map(str, config.devices)),
                "--ckpt",
                config.ckpt,
                "--out",
                out,
                "--prompt-t",
                config.prefill_t,
                "--prompt-tokens",
                ",".join(map(str, config.prompt_tokens)),
            ),
        ),
        log_path=state.report["paths"]["logs"] / "prefill_hidden.log",
        timeout_seconds=config.prefill_timeout,
    )
    text = _read_stage_log(stage)
    report_path = out / "prefill_hidden_only_report.json"
    payload = _load_json_report(report_path, name="Prefill", stage=stage)
    hidden_shape = payload.get("hidden_shape")
    hidden_finite = payload.get("hidden_finite")
    checks = {
        "process_rc_zero": stage.returncode == 0,
        "result_clean": "RESULT=PREFILL_HIDDEN_DEVICE_PASS" in text,
        "pypto_hidden_only": (
            payload.get("ownership", {}).get("pypto_output")
            == "pre-final-norm BF16 hidden only"
        ),
        "hidden_finite": bool(hidden_finite),
        "hidden_shape": hidden_shape
        in (
            [TP, config.prefill_t, HIDDEN],
            [config.prefill_t, HIDDEN],
        ),
    }
    _set_stage_checks(stage, checks)
    _require_stage(stage)
    return {
        "stage": stage.to_dict(),
        "report_path": str(report_path),
        "hidden_shape": hidden_shape,
        "hidden_finite": hidden_finite,
    }


def _initial_report(config: PrefillCiConfig) -> dict[str, Any]:
    logs = config.artifact_dir / "logs"
    return {
        "schema_version": 2,
        "ok": False,
        "started_utc": utc_now(),
        "finished_utc": None,
        "test": {
            "name": "step3p5_prefill_hidden_only",
            "program_main": "whole_prefill_step3p5",
            # Prefill has no MTP; program_mtp is null (decode sets it to
            # "MTP_LAYER_HIDDEN_PROGRAMS").  Kept as null to make the schema
            # difference explicit rather than omitting the key.
            "program_mtp": None,
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
            "prefill_t": config.prefill_t,
            "prompt_tokens": list(config.prompt_tokens),
            "dry_run": config.dry_run,
            "timeouts_sec": {
                "prefill": config.prefill_timeout,
            },
        },
        "paths": {
            "out": config.out,
            "artifact_dir": config.artifact_dir,
            "logs": logs,
            "json_report": config.report_path,
        },
        "preflight": None,
        "prefill": None,
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


def run(config: PrefillCiConfig) -> dict[str, Any]:
    state = PrefillCiRun(config=config)
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
            state.report["prefill"] = _run_prefill(state)
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
            "active_prefill_exporters_after_cleanup": _active_prefill_exporter_pids(
                config.out
            ),
        }
        state.report["finished_utc"] = utc_now()
        state.report["stages"] = [stage.to_dict() for stage in state.stages]
        state.report["ok"] = (
            state.report["failure"] is None
            and state.report["preflight"] is not None
            and all(stage.passed for stage in state.stages)
            and not state.report["cleanup"]["active_exporters_after_cleanup"]
            and not state.report["cleanup"]["active_prefill_exporters_after_cleanup"]
        )
        write_json(config.report_path, json_safe(state.report))
        _restore_signal_handlers(previous_handlers)
    return json_safe(state.report)


def _print_summary(report: Mapping[str, Any]) -> None:
    print(f"PREFILL_HIDDEN_CI={'PASS' if report.get('ok') else 'FAIL'}")
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
