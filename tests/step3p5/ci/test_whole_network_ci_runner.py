# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Card-free tests for the Step3p5 whole-network CI orchestration layer."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.step3p5.ci._whole_network_ci_common import (
    RunnerError,
    active_exporter_pids,
    scrub_environment,
    terminate_process_group,
)
from tests.step3p5.ci.run_whole_network_ci import (
    WholeNetworkConfig,
    preflight,
    run,
)


def _write_checkpoint_index(ckpt: Path) -> None:
    weight_map = {
        "model.embed_tokens.weight": "weights.safetensors",
        "model.norm.weight": "weights.safetensors",
        "lm_head.weight": "weights.safetensors",
    }
    for layer in range(3, 45):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            weight = (
                f"model.layers.{layer}.moe.experts.0."
                f"{projection}.weight"
            )
            weight_map[weight] = "weights.safetensors"
            weight_map[f"{weight}_scale"] = "weights.safetensors"
    for layer in range(45, 48):
        weight_map.update(
            {
                f"model.layers.{layer}.input_layernorm.weight": "weights.safetensors",
                f"model.layers.{layer}.post_attention_layernorm.weight": "weights.safetensors",
                f"model.layers.{layer}.self_attn.q_proj.weight": "weights.safetensors",
                f"model.layers.{layer}.self_attn.k_proj.weight": "weights.safetensors",
                f"model.layers.{layer}.self_attn.v_proj.weight": "weights.safetensors",
                f"model.layers.{layer}.self_attn.o_proj.weight": "weights.safetensors",
                f"model.layers.{layer}.self_attn.q_norm.weight": "weights.safetensors",
                f"model.layers.{layer}.self_attn.k_norm.weight": "weights.safetensors",
                f"model.layers.{layer}.self_attn.g_proj.weight": "weights.safetensors",
                f"model.layers.{layer}.mlp.gate_proj.weight": "weights.safetensors",
                f"model.layers.{layer}.mlp.up_proj.weight": "weights.safetensors",
                f"model.layers.{layer}.mlp.down_proj.weight": "weights.safetensors",
                f"model.layers.{layer}.enorm.weight": "weights.safetensors",
                f"model.layers.{layer}.hnorm.weight": "weights.safetensors",
                f"model.layers.{layer}.eh_proj.weight": "weights.safetensors",
                f"model.layers.{layer}.transformer.shared_head.norm.weight": "weights.safetensors",
                f"model.layers.{layer}.transformer.shared_head.output.weight": "weights.safetensors",
            }
        )
    (ckpt / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )
    (ckpt / "weights.safetensors").write_bytes(b"test")


def _config(tmp_path: Path, ckpt: Path) -> WholeNetworkConfig:
    return WholeNetworkConfig(
        repo_root=Path(__file__).resolve().parents[3],
        ckpt=ckpt,
        out=tmp_path / "ipc",
        artifact_dir=tmp_path / "artifacts",
    )


def test_scrub_environment_removes_front8_controls(tmp_path: Path) -> None:
    env = scrub_environment(
        repo_root=tmp_path,
        base={
            "PATH": "/bin",
            "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "VLLM_USE_V1": "1",
            "HCCL_BUFFSIZE": "512",
            "HCCL_WHITELIST_DISABLE": "1",
            "LD_PRELOAD": "libjemalloc.so",
            "P_FILL_BATCH": "1",
            "PYPTO_MEM_PLANNER": "ptoas",
        },
    )
    assert "ASCEND_RT_VISIBLE_DEVICES" not in env
    assert "VLLM_USE_V1" not in env
    assert "HCCL_BUFFSIZE" not in env
    assert env["HCCL_WHITELIST_DISABLE"] == "1"
    assert "LD_PRELOAD" not in env
    assert "P_FILL_BATCH" not in env
    assert "PYPTO_MEM_PLANNER" not in env
    assert env["PYTORCH_NPU_ALLOC_CONF"] == ""
    assert env["PYTHONPATH"] == str(tmp_path)


def test_active_exporter_scan_ignores_shell_command_text(
    tmp_path: Path,
) -> None:
    pool = tmp_path / "ipc"
    output = "\n".join(
        [
            (
                "101 bash bash -c 'python -m "
                "tests.step3p5.harnesses._stage_mtp_hidden_selected "
                f"--out {pool}'"
            ),
            (
                "102 python python -m "
                "tests.step3p5.harnesses._stage_mtp_hidden_selected "
                f"--export-rank 0 --out {pool}"
            ),
            (
                "103 python python -m "
                "tests.step3p5.ci.run_whole_network_ci "
                f"--out {pool}"
            ),
        ]
    )
    completed = subprocess.CompletedProcess(
        args=["ps"],
        returncode=0,
        stdout=output,
        stderr="",
    )
    with patch(
        "tests.step3p5.ci._whole_network_ci_common.subprocess.run",
        return_value=completed,
    ):
        assert active_exporter_pids(pool) == [
            {
                "pid": 102,
                "command": (
                    "python -m "
                    "tests.step3p5.harnesses._stage_mtp_hidden_selected "
                    f"--export-rank 0 --out {pool}"
                ),
            }
        ]


def test_terminate_process_group_ignores_zombie_only_group() -> None:
    process = subprocess.Popen(["sleep", "0"])
    process.wait()
    ps_output = f"{process.pid} {process.pid} Z\n"
    completed = subprocess.CompletedProcess(
        args=["ps"],
        returncode=0,
        stdout=ps_output,
        stderr="",
    )
    with patch(
        "tests.step3p5.ci._whole_network_ci_common.subprocess.run",
        return_value=completed,
    ):
        assert terminate_process_group(process, grace_seconds=0.1)


def test_preflight_protects_front8_and_requires_contiguous_devices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    _write_checkpoint_index(ckpt)
    isa = tmp_path / "pto-isa"
    isa.mkdir()
    monkeypatch.setenv("PTO_ISA_ROOT", str(isa))

    with pytest.raises(RunnerError, match="overlap protected devices"):
        preflight(
            WholeNetworkConfig(
                **{
                    **_config(tmp_path, ckpt).__dict__,
                    "devices": tuple(range(8)),
                }
            )
        )

    with pytest.raises(RunnerError, match="ordered and contiguous"):
        preflight(
            WholeNetworkConfig(
                **{
                    **_config(tmp_path, ckpt).__dict__,
                    "devices": (8, 9, 10, 11, 12, 13, 14, 16),
                }
            )
        )

    report = preflight(_config(tmp_path, ckpt))
    assert report["ok"]
    assert report["checkpoint"]["native_w8a8_index_pairs"] == 42


def test_skip_mtp_dry_run_is_recorded_in_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    _write_checkpoint_index(ckpt)
    isa = tmp_path / "pto-isa"
    isa.mkdir()
    monkeypatch.setenv("PTO_ISA_ROOT", str(isa))

    config = WholeNetworkConfig(
        **{
            **_config(tmp_path, ckpt).__dict__,
            "dry_run": True,
            "run_mtp": False,
        }
    )
    report = run(config)
    assert report["ok"]
    assert report["config"]["run_mtp"] is False


def test_dry_run_writes_success_report_without_touching_devices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    _write_checkpoint_index(ckpt)
    isa = tmp_path / "pto-isa"
    isa.mkdir()
    monkeypatch.setenv("PTO_ISA_ROOT", str(isa))

    config = WholeNetworkConfig(
        **{
            **_config(tmp_path, ckpt).__dict__,
            "dry_run": True,
        }
    )
    report = run(config)
    assert report["ok"]
    assert report["dry_run"]["ok"]
    assert report["stages"] == []
    assert report["cleanup"]["skipped"]
    assert not config.out.exists()
    saved = json.loads(config.report_path.read_text(encoding="utf-8"))
    assert saved["ok"]
