# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the Step3p5 native-W8A8 main -> MTP3 whole-network CI gate.

The runner is intentionally an orchestration layer, not a second model
harness.  It launches the already-validated MTP3 exporter, invokes the
canonical main and MTP3 stage programs, then invokes the CPU reference.  Each
stage has an independent log and timeout, and the exporter pool is cleaned up
in ``finally`` unless the user explicitly asks to keep it for debugging.

The canonical default flow is:

1. fresh eight-rank exporter pool on the explicitly supplied devices;
2. main P42 with token 6127 and native W8A8/KV IPC;
3. MTP45 -> MTP46 -> MTP47 with the sampler handoff token 303;
4. optional fixed-BATCH=16 MTP validation;
5. optional CPU ctx=1 precision reference;
6. JSON report and exporter cleanup.

This command must run inside the pinned PyPTO/CANN environment.  It does not
source shell setup files itself; CI should source ``set_env.sh`` and the
workspace activation script before invoking it.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import platform
import re
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from tests.step3p5._whole_network_ci_common import (
    ManagedProcess,
    RunnerError,
    StageResult,
    active_exporter_pids,
    git_metadata,
    json_safe,
    launch_logged_process,
    parse_int_list,
    remove_pool_artifacts,
    run_logged_command,
    scrub_environment,
    stop_processes,
    tail_text,
    utc_now,
    wait_for_processes,
    write_json,
)


TP = 8
CANONICAL_DEVICES = tuple(range(8, 16))
PROTECTED_FRONT_DEVICES = tuple(range(8))
CANONICAL_HIDDEN_TOKEN = 6127
CANONICAL_FIRST_TOKEN = 303
CANONICAL_MAIN_ARGMAX = 303
CANONICAL_MTP_TOKENS = (6178, 410, 303)
CANONICAL_MOE_LAYERS = 42
IPC_ALIGNMENT = 512

MAIN_INT8_KEYS = (
    "moe_w_gate_r",
    "moe_w_up_r",
    "moe_w_down_r",
)
MAIN_SCALE_KEYS = (
    "moe_w_gate_r_scale",
    "moe_w_up_r_scale",
    "moe_w_down_r_scale",
)
MTP_FP32_KEYS = (
    "mtp_enorm_weight",
    "mtp_hnorm_weight",
    "mtp_input_rms_weight",
    "mtp_post_attn_rms_weight",
    "mtp_q_norm_weight",
    "mtp_k_norm_weight",
    "mtp_shared_head_norm_weight",
)
MTP_BF16_KEYS = (
    "embed_tokens",
    "mtp_eh_proj_weight",
    "mtp_wq_swa",
    "mtp_wk_swa",
    "mtp_wv_swa",
    "mtp_wo_swa",
    "mtp_w_g_swa",
    "mtp_dense_w_gate",
    "mtp_dense_w_up",
    "mtp_dense_w_down",
    "mtp_shared_head_output_weight",
    "mtp_k_cache",
    "mtp_v_cache",
)


@dataclass(frozen=True)
class WholeNetworkConfig:
    """Immutable configuration for one whole-network CI invocation."""

    repo_root: Path
    ckpt: Path
    devices: tuple[int, ...] = CANONICAL_DEVICES
    protected_devices: tuple[int, ...] = PROTECTED_FRONT_DEVICES
    allow_protected_devices: bool = False
    out: Path = Path("/tmp/n1_weight_ipc_mtp3_ci")
    artifact_dir: Path = Path("/tmp/n1_weight_ipc_mtp3_ci_artifacts")
    active_batch: int = 1
    run_batch16: bool = True
    run_reference: bool = True
    hidden_token: int = CANONICAL_HIDDEN_TOKEN
    first_token: int = CANONICAL_FIRST_TOKEN
    export_timeout: float = 2400.0
    main_timeout: float = 1800.0
    mtp_timeout: float = 1200.0
    batch16_timeout: float = 1200.0
    reference_timeout: float = 1200.0
    reference_threads: int = 64
    keep_exporters_on_failure: bool = False
    dry_run: bool = False
    json_report: Path | None = None

    @property
    def report_path(self) -> Path:
        return self.json_report or self.artifact_dir / "whole_network_report.json"

    @property
    def dump_dir(self) -> Path:
        return self.artifact_dir / "dumps"


@dataclass
class WholeNetworkRun:
    """Mutable execution state kept separate from the serializable report."""

    config: WholeNetworkConfig
    env: dict[str, str] = field(default_factory=dict)
    exporters: list[ManagedProcess] = field(default_factory=list)
    stages: list[StageResult] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)
    pool_owned: bool = False


def _default_ckpt() -> Path:
    return Path(
        os.environ.get(
            "STEP3P5_CKPT_DIR",
            "/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp",
        )
    )


def _default_out() -> Path:
    return Path(os.environ.get("STEP3P5_CI_OUT", "/tmp/n1_weight_ipc_mtp3_ci"))


def _default_artifact_dir(out: Path) -> Path:
    configured = os.environ.get("STEP3P5_CI_ARTIFACT_DIR")
    if configured:
        return Path(configured)
    return out.parent / f"{out.name}_artifacts"


def config_from_environment() -> WholeNetworkConfig:
    """Build the pytest/CI configuration from ``STEP3P5_*`` variables."""
    out = _default_out().expanduser().resolve()
    artifact_dir = _default_artifact_dir(out).expanduser().resolve()
    protected = parse_int_list(
        os.environ.get("STEP3P5_PROTECTED_DEVICES", "0,1,2,3,4,5,6,7"),
        name="STEP3P5_PROTECTED_DEVICES",
    )
    return WholeNetworkConfig(
        repo_root=Path(__file__).resolve().parents[2],
        ckpt=_default_ckpt().expanduser().resolve(),
        devices=parse_int_list(
            os.environ.get(
                "STEP3P5_CI_DEVICES",
                ",".join(map(str, CANONICAL_DEVICES)),
            ),
            name="STEP3P5_CI_DEVICES",
        ),
        protected_devices=protected,
        allow_protected_devices=os.environ.get("STEP3P5_ALLOW_PROTECTED_DEVICES") == "1",
        out=out,
        artifact_dir=artifact_dir,
        active_batch=int(os.environ.get("STEP3P5_ACTIVE_BATCH", "1")),
        run_batch16=os.environ.get("STEP3P5_RUN_BATCH16", "1") != "0",
        run_reference=os.environ.get("STEP3P5_RUN_REFERENCE", "1") != "0",
        hidden_token=int(os.environ.get("STEP3P5_HIDDEN_TOKEN", str(CANONICAL_HIDDEN_TOKEN))),
        first_token=int(os.environ.get("STEP3P5_FIRST_TOKEN", str(CANONICAL_FIRST_TOKEN))),
        export_timeout=float(os.environ.get("STEP3P5_EXPORT_TIMEOUT", "2400")),
        main_timeout=float(os.environ.get("STEP3P5_MAIN_TIMEOUT", "1800")),
        mtp_timeout=float(os.environ.get("STEP3P5_MTP_TIMEOUT", "1200")),
        batch16_timeout=float(os.environ.get("STEP3P5_BATCH16_TIMEOUT", "1200")),
        reference_timeout=float(os.environ.get("STEP3P5_REFERENCE_TIMEOUT", "1200")),
        reference_threads=int(os.environ.get("STEP3P5_REFERENCE_THREADS", "64")),
        keep_exporters_on_failure=os.environ.get("STEP3P5_KEEP_EXPORTERS_ON_FAILURE") == "1",
        dry_run=os.environ.get("STEP3P5_DRY_RUN") == "1",
        json_report=(
            Path(os.environ["STEP3P5_JSON_REPORT"]).expanduser().resolve()
            if os.environ.get("STEP3P5_JSON_REPORT")
            else None
        ),
    )


def _parse_args(argv: Sequence[str] | None = None) -> WholeNetworkConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=_default_ckpt())
    parser.add_argument(
        "--devices",
        default=os.environ.get("STEP3P5_CI_DEVICES", ",".join(map(str, CANONICAL_DEVICES))),
        help="exactly eight physical device IDs, for example 8,9,10,11,12,13,14,15",
    )
    parser.add_argument("--protected-devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--allow-protected-devices", action="store_true")
    parser.add_argument("--out", type=Path, default=_default_out())
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--active-batch", type=int, default=1)
    parser.add_argument(
        "--run-batch16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also run the fixed-BATCH=16 MTP harness gate",
    )
    parser.add_argument(
        "--run-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the CPU ctx=1 precision reference",
    )
    parser.add_argument("--hidden-token", type=int, default=CANONICAL_HIDDEN_TOKEN)
    parser.add_argument("--first-token", type=int, default=CANONICAL_FIRST_TOKEN)
    parser.add_argument("--export-timeout", type=float, default=2400.0)
    parser.add_argument("--main-timeout", type=float, default=1800.0)
    parser.add_argument("--mtp-timeout", type=float, default=1200.0)
    parser.add_argument("--batch16-timeout", type=float, default=1200.0)
    parser.add_argument("--reference-timeout", type=float, default=1200.0)
    parser.add_argument("--reference-threads", type=int, default=64)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run preflight and write the report without touching NPU devices",
    )
    parser.add_argument(
        "--keep-exporters-on-failure",
        action="store_true",
        help="preserve exporter children and pool sentinels for post-failure debugging",
    )
    parser.add_argument("--json-report", type=Path, default=None)
    args = parser.parse_args(argv)

    out = args.out.expanduser().resolve()
    artifact_dir = (
        args.artifact_dir.expanduser().resolve()
        if args.artifact_dir is not None
        else _default_artifact_dir(out).expanduser().resolve()
    )
    return WholeNetworkConfig(
        repo_root=Path(__file__).resolve().parents[2],
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
        run_reference=args.run_reference,
        hidden_token=args.hidden_token,
        first_token=args.first_token,
        export_timeout=args.export_timeout,
        main_timeout=args.main_timeout,
        mtp_timeout=args.mtp_timeout,
        batch16_timeout=args.batch16_timeout,
        reference_timeout=args.reference_timeout,
        reference_threads=args.reference_threads,
        keep_exporters_on_failure=args.keep_exporters_on_failure,
        dry_run=args.dry_run,
        json_report=(
            args.json_report.expanduser().resolve()
            if args.json_report is not None
            else None
        ),
    )


def _validate_timeout(value: float, *, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise RunnerError(f"{name} must be a positive finite number")


def _validate_checkpoint(ckpt: Path) -> dict[str, Any]:
    if not ckpt.is_dir():
        raise RunnerError(f"checkpoint directory does not exist: {ckpt}")
    index_files = [
        ckpt / "quant_model_weights.safetensors.index.json",
        ckpt / "model.safetensors.index.json",
    ]
    single_file = ckpt / "model.safetensors"
    existing_indexes = [path for path in index_files if path.is_file()]
    if not existing_indexes and not single_file.is_file():
        names = ", ".join(str(path.name) for path in index_files)
        raise RunnerError(
            f"checkpoint has no supported safetensors index ({names}) or model.safetensors: {ckpt}"
        )
    if not existing_indexes:
        raise RunnerError(
            "whole-network native-W8A8 CI requires an inspectable safetensors index; "
            f"single-file checkpoint is not accepted: {single_file}"
        )
    required_keys = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    quantized_expert_count = 0
    mtp_layer_presence: dict[str, bool] = {}
    if existing_indexes:
        try:
            index = json.loads(existing_indexes[0].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunnerError(f"cannot read checkpoint index {existing_indexes[0]}: {exc}") from exc
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise RunnerError(f"checkpoint index has no weight_map object: {existing_indexes[0]}")
        missing = sorted(required_keys.difference(weight_map))
        if missing:
            raise RunnerError(f"checkpoint misses required top-level tensors: {missing}")
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
            quantized_expert_count += 1
        if missing_w8a8:
            raise RunnerError(
                "checkpoint misses native W8A8 routed-expert tensors: "
                f"{missing_w8a8[:8]}"
                + ("..." if len(missing_w8a8) > 8 else "")
            )
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
            present = all(key in weight_map for key in required_mtp)
            mtp_layer_presence[str(layer)] = present
            if not present:
                missing_mtp = [key for key in required_mtp if key not in weight_map]
                raise RunnerError(f"checkpoint misses MTP{layer} tensors: {missing_mtp}")
        missing_shards = sorted(
            {
                str(filename)
                for filename in weight_map.values()
                if not (ckpt / str(filename)).is_file()
            }
        )
        if missing_shards:
            raise RunnerError(
                "checkpoint index references missing shard files: "
                f"{missing_shards[:8]}"
                + ("..." if len(missing_shards) > 8 else "")
            )
    return {
        "path": str(ckpt),
        "index_files": [str(path) for path in existing_indexes],
        "single_file": single_file.is_file(),
        "native_w8a8_index_pairs": quantized_expert_count,
        "mtp_layer_presence": mtp_layer_presence,
    }


def preflight(config: WholeNetworkConfig) -> dict[str, Any]:
    """Validate all non-device prerequisites before touching the IPC pool."""
    if len(config.devices) != TP:
        raise RunnerError(f"devices must contain exactly {TP} IDs, got {config.devices}")
    if len(set(config.devices)) != TP:
        raise RunnerError(f"devices must be distinct, got {config.devices}")
    expected_contiguous = tuple(range(config.devices[0], config.devices[0] + TP))
    if config.devices != expected_contiguous:
        raise RunnerError(
            "devices must be ordered and contiguous because the current IPC importer "
            f"binds dev_offset + rank; got {config.devices}, expected {expected_contiguous}"
        )
    if not config.allow_protected_devices:
        overlap = sorted(set(config.devices).intersection(config.protected_devices))
        if overlap:
            raise RunnerError(
                f"devices {overlap} overlap protected devices {config.protected_devices}; "
                "use the back-8 default or explicitly pass --allow-protected-devices"
            )
    if not config.repo_root.is_dir():
        raise RunnerError(f"repository root does not exist: {config.repo_root}")
    required_sources = (
        "tests/step3p5/_stage_whole_faithful_real_ipc.py",
        "tests/step3p5/_stage_whole_mtp3_ipc.py",
        "tools/step3p5/pypto_mtp3_ctx1_reference.py",
    )
    missing_sources = [
        relative
        for relative in required_sources
        if not (config.repo_root / relative).is_file()
    ]
    if missing_sources:
        raise RunnerError(f"repository misses whole-network test sources: {missing_sources}")
    if not 1 <= config.active_batch <= 16:
        raise RunnerError("--active-batch must be in [1,16]")
    if not 0 <= config.hidden_token < 128896:
        raise RunnerError(f"--hidden-token is outside the Step3p5 vocabulary: {config.hidden_token}")
    if not 0 <= config.first_token < 128896:
        raise RunnerError(f"--first-token is outside the Step3p5 vocabulary: {config.first_token}")
    if config.hidden_token != CANONICAL_HIDDEN_TOKEN:
        raise RunnerError(
            f"canonical N1 test requires hidden token {CANONICAL_HIDDEN_TOKEN}, "
            f"got {config.hidden_token}"
        )
    if config.first_token != CANONICAL_FIRST_TOKEN:
        raise RunnerError(
            f"canonical MTP sampler handoff requires token {CANONICAL_FIRST_TOKEN}, "
            f"got {config.first_token}"
        )
    if config.reference_threads < 0:
        raise RunnerError("--reference-threads must be non-negative")
    for value, name in (
        (config.export_timeout, "export-timeout"),
        (config.main_timeout, "main-timeout"),
        (config.mtp_timeout, "mtp-timeout"),
        (config.batch16_timeout, "batch16-timeout"),
        (config.reference_timeout, "reference-timeout"),
    ):
        _validate_timeout(value, name=name)

    out = config.out
    artifact_dir = config.artifact_dir
    if artifact_dir == out or out in artifact_dir.parents:
        raise RunnerError(
            f"artifact directory must not be inside IPC pool directory: out={out} artifacts={artifact_dir}"
        )
    if config.report_path == out or out in config.report_path.parents:
        raise RunnerError(
            f"JSON report must not be inside IPC pool directory: {config.report_path}"
        )
    if artifact_dir in out.parents:
        # This is safe, but a sibling is less surprising and avoids putting
        # large logs below a potentially cleaned parent directory.
        pass
    active = active_exporter_pids(out)
    if active:
        raise RunnerError(
            f"IPC pool already has active exporter processes for {out}: {active}"
        )
    pto_isa_root = os.environ.get("PTO_ISA_ROOT", "")
    if not pto_isa_root:
        raise RunnerError(
            "PTO_ISA_ROOT is not set; source the pinned workspace environment before running CI"
        )
    if not Path(pto_isa_root).is_dir():
        raise RunnerError(f"PTO_ISA_ROOT does not exist: {pto_isa_root}")
    checkpoint = _validate_checkpoint(config.ckpt)
    return {
        "ok": True,
        "devices": list(config.devices),
        "protected_devices": list(config.protected_devices),
        "protected_overlap": [],
        "checkpoint": checkpoint,
        "out": str(out),
        "artifact_dir": str(artifact_dir),
        "active_exporters_before_run": active,
        "pto_isa_root": pto_isa_root,
        "canonical": {
            "hidden_token": config.hidden_token,
            "main_argmax": CANONICAL_MAIN_ARGMAX,
            "first_token": config.first_token,
            "mtp_tokens": list(CANONICAL_MTP_TOKENS),
            "p_faithful_moe_layers": CANONICAL_MOE_LAYERS,
        },
    }


def _stage_command(
    module: str,
    *,
    args: Sequence[str],
) -> list[str]:
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


def _extract_tokens(text: str) -> list[int] | None:
    match = re.search(r"tokens_row0=(\[[^\]]*\])", text)
    if match is None:
        return None
    try:
        value = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        return None
    return value


def _extract_main_argmax(text: str) -> int | None:
    """Extract the executed main RUN result, not the golden text in TOP5."""
    match = re.search(
        r"^\[worker\] RUN done [^\n]*\bargmax=(\d+)\s*$",
        text,
        flags=re.MULTILINE,
    )
    return int(match.group(1)) if match is not None else None


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
        watch_processes=state.exporters,
    )
    state.stages.append(stage)
    return stage


def _require_stage(stage: StageResult, *, state: WholeNetworkRun) -> None:
    if stage.passed:
        return
    raise RunnerError(
        f"stage {stage.name} failed: {stage.error or f'rc={stage.returncode}'}\n"
        f"log={stage.log_path}\n{tail_text(stage.log_path)}"
    )


def _validate_pool_map(path: Path, *, rank: int) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot read exporter map rank{rank}: {path}: {exc}") from exc
    entries = payload.get("map")
    if not isinstance(entries, dict) or not entries:
        raise RunnerError(f"exporter map rank{rank} has no map entries: {path}")

    previous_end = 0
    offsets: dict[str, int] = {}
    for key, entry in entries.items():
        if not isinstance(entry, dict):
            raise RunnerError(f"rank{rank} map entry {key} is not an object")
        missing_fields = {"offset", "shape", "dtype", "nbytes"}.difference(entry)
        if missing_fields:
            raise RunnerError(
                f"rank{rank} map entry {key} misses fields={sorted(missing_fields)}"
            )
    ordered_entries = sorted(entries.items(), key=lambda item: int(item[1]["offset"]))
    for key, entry in ordered_entries:
        offset = int(entry["offset"])
        shape = tuple(int(item) for item in entry["shape"])
        dtype = str(entry["dtype"])
        nbytes = int(entry["nbytes"])
        if offset % IPC_ALIGNMENT:
            raise RunnerError(
                f"rank{rank} key={key} offset={offset} is not {IPC_ALIGNMENT}B aligned"
            )
        if offset < previous_end:
            raise RunnerError(f"rank{rank} key={key} overlaps an earlier IPC entry")
        expected_element_size = {
            "int8": 1,
            "bfloat16": 2,
            "float16": 2,
            "float32": 4,
        }.get(dtype)
        if expected_element_size is None:
            raise RunnerError(f"rank{rank} key={key} has unsupported dtype={dtype}")
        expected_nbytes = math.prod(shape) * expected_element_size
        if nbytes != expected_nbytes:
            raise RunnerError(
                f"rank{rank} key={key} nbytes={nbytes} != shape*dtype={expected_nbytes}"
            )
        if nbytes <= 0:
            raise RunnerError(f"rank{rank} key={key} has non-positive nbytes={nbytes}")
        previous_end = offset + nbytes
        offsets[key] = offset

    def require_dtype(keys: Sequence[str], expected: str) -> None:
        for key in keys:
            if key not in entries:
                raise RunnerError(f"rank{rank} exporter map misses required key={key}")
            actual = entries[key]["dtype"]
            if actual != expected:
                raise RunnerError(
                    f"rank{rank} key={key} must be {expected}, got {actual}"
                )

    require_dtype(MAIN_INT8_KEYS, "int8")
    require_dtype(MAIN_SCALE_KEYS, "float32")
    require_dtype(MTP_FP32_KEYS, "float32")
    require_dtype(MTP_BF16_KEYS, "bfloat16")
    require_dtype(("k_cache", "v_cache", "mtp_k_cache", "mtp_v_cache"), "bfloat16")
    if offsets["k_cache"] == offsets["mtp_k_cache"] or offsets["v_cache"] == offsets["mtp_v_cache"]:
        raise RunnerError(f"rank{rank} main and MTP KV entries must not alias")
    if offsets["mtp_k_cache"] == offsets["mtp_v_cache"]:
        raise RunnerError(f"rank{rank} MTP K/V entries must not alias")
    main_k_shape = tuple(int(item) for item in entries["k_cache"]["shape"])
    main_v_shape = tuple(int(item) for item in entries["v_cache"]["shape"])
    mtp_k_shape = tuple(int(item) for item in entries["mtp_k_cache"]["shape"])
    mtp_v_shape = tuple(int(item) for item in entries["mtp_v_cache"]["shape"])
    if main_k_shape != main_v_shape:
        raise RunnerError(f"rank{rank} main K/V shapes differ: {main_k_shape} vs {main_v_shape}")
    expected_mtp_shape = (3 * main_k_shape[0], *main_k_shape[1:])
    if mtp_k_shape != expected_mtp_shape or mtp_v_shape != expected_mtp_shape:
        raise RunnerError(
            f"rank{rank} MTP KV must contain three disjoint layer slices; "
            f"expected={expected_mtp_shape} k={mtp_k_shape} v={mtp_v_shape}"
        )
    pool_bytes = int(payload.get("pool_bytes", 0))
    if pool_bytes < previous_end:
        raise RunnerError(
            f"rank{rank} pool_bytes={pool_bytes} is smaller than final entry end={previous_end}"
        )

    return {
        "rank": rank,
        "path": str(path),
        "pool_bytes": pool_bytes,
        "entry_count": len(entries),
        "all_offsets_aligned": True,
        "native_main_routed_dtype": "int8",
        "native_main_scale_dtype": "float32",
        "mtp_norm_abi_dtype": "float32",
        "mtp_projection_dtype": "bfloat16",
        "main_mtp_kv_distinct": True,
        "entries": {
            key: {
                "offset": int(value["offset"]),
                "shape": list(value["shape"]),
                "dtype": value["dtype"],
                "nbytes": int(value["nbytes"]),
            }
            for key, value in entries.items()
            if key in ("k_cache", "v_cache", "mtp_k_cache", "mtp_v_cache")
        },
    }


def validate_pool_contract(config: WholeNetworkConfig) -> dict[str, Any]:
    """Validate all eight fresh maps independently of the worker harness."""
    ranks = [
        _validate_pool_map(
            config.out / f"pypto_weight_map.rank{rank}.json",
            rank=rank,
        )
        for rank in range(TP)
    ]
    return {
        "ok": True,
        "alignment_bytes": IPC_ALIGNMENT,
        "rank_count": len(ranks),
        "ranks": ranks,
        "native_w8a8": True,
        "mtp3_bf16_projection": True,
        "mtp_norm_fp32_abi": True,
        "kv_ipc": True,
    }


def _run_exporters(state: WholeNetworkRun) -> dict[str, Any]:
    config = state.config
    config.out.mkdir(parents=True, exist_ok=True)
    if active_exporter_pids(config.out):
        raise RunnerError(f"exporter appeared before launch: {config.out}")
    removed = remove_pool_artifacts(config.out)
    state.pool_owned = True

    for rank, device in enumerate(config.devices):
        command = _stage_command(
            "tests.step3p5._stage_whole_mtp3_ipc",
            args=[
                "--export-rank",
                rank,
                "--dev",
                device,
                "--out",
                config.out,
                "--ckpt",
                config.ckpt,
            ],
        )
        state.exporters.append(
            launch_logged_process(
                name=f"export-rank{rank}",
                command=command,
                cwd=config.repo_root,
                env=state.env,
                log_path=state.report["paths"]["logs"] / f"export_rank{rank}.log",
            )
        )

    wait_for_processes(
        state.exporters,
        timeout_seconds=config.export_timeout,
        ready_files=[
            config.out / f"ready.rank{rank}"
            for rank in range(TP)
        ],
    )
    contract = validate_pool_contract(config)
    for managed in state.exporters:
        if managed.poll() is not None:
            raise RunnerError(
                f"{managed.name} exited after readiness with rc={managed.poll()}; "
                f"log={managed.log_path}"
            )
    return {
        "launch_removed_stale_artifacts": removed,
        "ready": True,
        "processes": [process.to_dict() for process in state.exporters],
        "contract": contract,
    }


def _run_main(state: WholeNetworkRun) -> dict[str, Any]:
    config = state.config
    dump_dir = config.dump_dir
    dump_dir.mkdir(parents=True, exist_ok=True)
    stage = _run_stage(
        state,
        name="canonical_main_p42",
        command=_stage_command(
            "tests.step3p5._stage_whole_faithful_real_ipc",
            args=[
                "--device",
                ",".join(map(str, config.devices)),
                "--reuse-exporters",
                "--kv-ipc",
                "--hidden-token",
                config.hidden_token,
                "--out",
                config.out,
                "--ckpt",
                config.ckpt,
            ],
        ),
        log_path=state.report["paths"]["logs"] / "canonical_main_p42.log",
        timeout_seconds=config.main_timeout,
    )
    text = _read_stage_log(stage)
    hidden_path = dump_dir / f"P{CANONICAL_MOE_LAYERS}_nh_row0.pt"
    actual_argmax = _extract_main_argmax(text)
    checks = {
        "process_rc_zero": stage.returncode == 0,
        "run_done": "[worker] RUN done" in text,
        "result_clean": "RESULT=REAL_WEIGHT_IPC_RUN_CLEAN" in text,
        "argmax_303": actual_argmax == CANONICAL_MAIN_ARGMAX,
        "hidden_token_6127": f"embed(token={CANONICAL_HIDDEN_TOKEN})" in text,
        "kv_ipc": "KV cache via IPC" in text,
        "p42_hidden_dump": hidden_path.is_file(),
    }
    _set_stage_checks(stage, checks)
    _require_stage(stage, state=state)
    return {
        "stage": stage.to_dict(),
        "argmax": actual_argmax,
        "expected_argmax": CANONICAL_MAIN_ARGMAX,
        "hidden_dump": str(hidden_path),
        "hidden_dump_exists": hidden_path.is_file(),
    }


def _run_mtp(
    state: WholeNetworkRun,
    *,
    name: str,
    active_batch: int,
    timeout_seconds: float,
    dump_dir: Path,
) -> dict[str, Any]:
    config = state.config
    previous_hidden = config.dump_dir / f"P{CANONICAL_MOE_LAYERS}_nh_row0.pt"
    dump_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "mtp3_hidden.pt",
        "mtp3_logits_shards.pt",
        "mtp3_draft_token_ids.pt",
    ):
        (dump_dir / stale_name).unlink(missing_ok=True)
    stage = _run_stage(
        state,
        name=name,
        command=_stage_command(
            "tests.step3p5._stage_whole_mtp3_ipc",
            args=[
                "--device",
                ",".join(map(str, config.devices)),
                "--reuse-exporters",
                "--out",
                config.out,
                "--ckpt",
                config.ckpt,
                "--previous-hidden",
                previous_hidden,
                "--first-token",
                config.first_token,
                "--active-batch",
                active_batch,
                "--dump-dir",
                dump_dir,
            ],
        ),
        log_path=state.report["paths"]["logs"] / f"{name}.log",
        timeout_seconds=timeout_seconds,
    )
    text = _read_stage_log(stage)
    tokens = _extract_tokens(text)
    expected_dump_files = (
        dump_dir / "mtp3_hidden.pt",
        dump_dir / "mtp3_logits_shards.pt",
        dump_dir / "mtp3_draft_token_ids.pt",
    )
    checks = {
        "process_rc_zero": stage.returncode == 0,
        "run_done": "[worker] RUN done" in text,
        "result_clean": "RESULT=MTP3_IPC_RUN_CLEAN" in text,
        "tokens_present": tokens is not None,
        "tokens_exact": tokens == list(CANONICAL_MTP_TOKENS),
        "sampler_handoff_token_303": (
            f"active_batch={active_batch} first_token={CANONICAL_FIRST_TOKEN}" in text
        ),
        "tp_rank_and_full_logits_checks": stage.returncode == 0,
        "device_dumps_present": all(path.is_file() for path in expected_dump_files),
    }
    if active_batch < 16:
        checks["padding_hidden_logits_zero"] = "padding rows initialized" in text
    _set_stage_checks(stage, checks)
    _require_stage(stage, state=state)
    return {
        "stage": stage.to_dict(),
        "active_batch": active_batch,
        "tokens_row0": tokens,
        "expected_tokens_row0": list(CANONICAL_MTP_TOKENS),
        "dump_dir": str(dump_dir),
        "dump_files": [str(path) for path in expected_dump_files],
        "padding_checked": active_batch < 16,
        "full_logits_argmax_checked_by_harness": True,
    }


def _run_reference(state: WholeNetworkRun, *, dump_dir: Path) -> dict[str, Any]:
    config = state.config
    output = state.report["paths"]["reference_report"]
    output.unlink(missing_ok=True)
    stage = _run_stage(
        state,
        name="mtp3_cpu_reference",
        command=_stage_command(
            "tools.step3p5.pypto_mtp3_ctx1_reference",
            args=[
                "--ckpt",
                config.ckpt,
                "--previous-hidden",
                config.dump_dir / f"P{CANONICAL_MOE_LAYERS}_nh_row0.pt",
                "--device-dump",
                dump_dir,
                "--threads",
                config.reference_threads,
                "--out",
                output,
            ],
        ),
        log_path=state.report["paths"]["logs"] / "mtp3_cpu_reference.log",
        timeout_seconds=config.reference_timeout,
    )
    payload: dict[str, Any] = {}
    if output.is_file():
        try:
            payload = json.loads(output.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            stage.error = f"reference report is not valid JSON: {exc}"
    checks = {
        "process_rc_zero": stage.returncode == 0,
        "report_ok": payload.get("ok") is True,
        "reference_tokens_exact": payload.get("reference_tokens") == list(CANONICAL_MTP_TOKENS),
        "device_tokens_exact": payload.get("device_tokens_row0") == list(CANONICAL_MTP_TOKENS),
        "worst_numeric_pass_rate": float(payload.get("worst_numeric_pass_rate", 0.0)) >= 0.97,
    }
    _set_stage_checks(stage, checks)
    _require_stage(stage, state=state)
    return {
        "stage": stage.to_dict(),
        "report_path": str(output),
        "report": payload,
    }


def _initial_report(config: WholeNetworkConfig) -> dict[str, Any]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logs = config.artifact_dir / "logs"
    return {
        "schema_version": 1,
        "ok": False,
        "started_utc": utc_now(),
        "finished_utc": None,
        "test": {
            "name": "step3p5_whole_network_mtp3",
            "program_main": "whole_decode_faithful_real",
            "program_mtp": "whole_mtp3",
            "canonical_document": "pypto-project/N1-CANONICAL-TEST.md",
            "run_id": timestamp,
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
            "run_reference": config.run_reference,
            "dry_run": config.dry_run,
            "hidden_token": config.hidden_token,
            "first_token": config.first_token,
            "protected_devices": list(config.protected_devices),
            "allow_protected_devices": config.allow_protected_devices,
            "timeouts_sec": {
                "export": config.export_timeout,
                "main": config.main_timeout,
                "mtp": config.mtp_timeout,
                "batch16": config.batch16_timeout,
                "reference": config.reference_timeout,
            },
        },
        "paths": {
            "out": config.out,
            "artifact_dir": config.artifact_dir,
            "logs": logs,
            "dumps": config.dump_dir,
            "reference_report": config.artifact_dir / "mtp3_reference.json",
            "json_report": config.report_path,
        },
        "preflight": None,
        "exporters": None,
        "main": None,
        "mtp_single": None,
        "mtp_batch16": None,
        "reference": None,
        "cleanup": None,
        "failure": None,
    }


def _install_signal_handlers() -> dict[int, Any]:
    """Turn SIGINT/SIGTERM into catchable runner failures for cleanup."""
    previous: dict[int, Any] = {}

    def interrupt(signum: int, _frame: Any) -> None:
        raise RunnerError(f"received signal {signum}; aborting stages and cleaning exporters")

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupt)
        except ValueError:
            # ``run`` can be called from a non-main pytest thread.  Python only
            # permits signal handlers in the main interpreter thread.
            previous.clear()
            break
    return previous


def _restore_signal_handlers(previous: Mapping[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def run(config: WholeNetworkConfig) -> dict[str, Any]:
    """Execute one whole-network CI run and return its JSON-safe report."""
    state = WholeNetworkRun(config=config)
    state.report = _initial_report(config)
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    config.dump_dir.mkdir(parents=True, exist_ok=True)
    state.report["paths"]["logs"].mkdir(parents=True, exist_ok=True)
    previous_signal_handlers = _install_signal_handlers()

    try:
        # Never let a previous successful run satisfy a new run's dump gate.
        # The IPC pool itself is cleaned separately so a failed run cannot
        # accidentally reuse a stale map or a stale main-to-MTP hidden.
        for stale in (
            config.dump_dir / f"P{CANONICAL_MOE_LAYERS}_nh_row0.pt",
            config.dump_dir / f"P{CANONICAL_MOE_LAYERS}_hmid_row0.pt",
            config.dump_dir / f"P{CANONICAL_MOE_LAYERS}_S0_dbg_row0.pt",
        ):
            stale.unlink(missing_ok=True)
        state.report["preflight"] = preflight(config)
        state.env = scrub_environment(
            repo_root=config.repo_root,
            overrides={
                "P_FAITHFUL_MOE_LAYERS": str(CANONICAL_MOE_LAYERS),
                "N1_DUMP_DIR": str(config.dump_dir),
                "PTO2_RING_HEAP": os.environ.get("PTO2_RING_HEAP", "4294967296"),
                "PTO2_RING_TASK_WINDOW": os.environ.get("PTO2_RING_TASK_WINDOW", "131072"),
                "PTO2_RING_DEP_POOL": os.environ.get("PTO2_RING_DEP_POOL", "131072"),
            },
        )
        state.report["environment"] = {
            "scrubbed_vllm_front8_controls": True,
            "ascend_rt_visible_devices": state.env.get("ASCEND_RT_VISIBLE_DEVICES"),
            "vllm_keys_present": sorted(key for key in state.env if key.startswith("VLLM_")),
            "front8_hccl_keys_present": sorted(
                key
                for key in ("HCCL_BUFFSIZE", "HCCL_OP_EXPANSION_MODE")
                if key in state.env
            ),
            "p_faithful_moe_layers": state.env["P_FAITHFUL_MOE_LAYERS"],
        }

        if config.dry_run:
            state.report["dry_run"] = {
                "ok": True,
                "message": "preflight passed; no exporter or device stage was launched",
            }
        else:
            state.report["exporters"] = _run_exporters(state)
            state.report["main"] = _run_main(state)
            state.report["mtp_single"] = _run_mtp(
                state,
                name="mtp3_single",
                active_batch=config.active_batch,
                timeout_seconds=config.mtp_timeout,
                dump_dir=config.artifact_dir / "dumps" / "single",
            )
            if config.run_batch16:
                state.report["mtp_batch16"] = _run_mtp(
                    state,
                    name="mtp3_batch16",
                    active_batch=16,
                    timeout_seconds=config.batch16_timeout,
                    dump_dir=config.artifact_dir / "dumps" / "batch16",
                )
            if config.run_reference:
                state.report["reference"] = _run_reference(
                    state,
                    dump_dir=config.artifact_dir / "dumps" / "single",
                )
    except Exception as exc:  # noqa: BLE001 - report and cleanup must always run
        state.report["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        try:
            failed = state.report["failure"] is not None
            try:
                if state.pool_owned:
                    state.report["cleanup"] = stop_processes(
                        state.exporters,
                        stop_file=config.out / "STOP",
                        pool_dir=config.out,
                        keep=config.keep_exporters_on_failure and failed,
                    )
                    state.report["cleanup"]["active_exporters_after_cleanup"] = (
                        active_exporter_pids(config.out)
                    )
                else:
                    state.report["cleanup"] = {
                        "requested": False,
                        "skipped": True,
                        "active_exporters_after_cleanup": active_exporter_pids(config.out),
                    }
            except Exception as cleanup_exc:  # noqa: BLE001 - preserve the primary stage failure
                state.report["cleanup"] = {
                    "requested": state.pool_owned,
                    "stopped": False,
                    "error": f"{type(cleanup_exc).__name__}: {cleanup_exc}",
                    "active_exporters_after_cleanup": active_exporter_pids(config.out),
                }
                if state.report["failure"] is None:
                    state.report["failure"] = {
                        "type": type(cleanup_exc).__name__,
                        "message": f"cleanup failed: {cleanup_exc}",
                    }

            state.report["finished_utc"] = utc_now()
            stages_ok = all(stage.passed for stage in state.stages)
            active_after_cleanup = state.report["cleanup"].get(
                "active_exporters_after_cleanup",
                [],
            )
            cleanup_ok = bool(
                state.report["cleanup"].get("stopped")
                or state.report["cleanup"].get("skipped")
            ) and not state.report["cleanup"].get("remaining_pids") and not active_after_cleanup
            state.report["stages"] = [stage.to_dict() for stage in state.stages]
            state.report["ok"] = (
                state.report["failure"] is None
                and stages_ok
                and cleanup_ok
                and state.report["preflight"] is not None
            )
            write_json(config.report_path, json_safe(state.report))
        finally:
            _restore_signal_handlers(previous_signal_handlers)
    return json_safe(state.report)


def _print_summary(report: Mapping[str, Any]) -> None:
    status = "PASS" if report.get("ok") else "FAIL"
    print(f"WHOLE_NETWORK_MTP3={status}")
    print(f"report={report.get('paths', {}).get('json_report')}")
    if report.get("failure"):
        print(f"failure={report['failure'].get('message')}", file=sys.stderr)
    cleanup = report.get("cleanup") or {}
    print(
        "cleanup="
        f"stopped={cleanup.get('stopped', False)} "
        f"kept={cleanup.get('kept', False)} "
        f"remaining_pids={cleanup.get('remaining_pids', [])}"
    )
    for stage in report.get("stages", []):
        print(
            f"stage={stage['name']} rc={stage['returncode']} "
            f"duration={stage['duration_sec']}s passed={stage['passed']}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    config = _parse_args(argv)
    report = run(config)
    _print_summary(report)
    if not report.get("ok"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
