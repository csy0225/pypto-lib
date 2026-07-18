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
)
from tests.step3p5.ci.run_whole_network_ci import (
    MAIN_INT8_KEYS,
    MAIN_SCALE_KEYS,
    MTP_BF16_KEYS,
    MTP_FP32_KEYS,
    WholeNetworkConfig,
    _extract_main_argmax,
    _extract_tokens,
    _validate_pool_map,
    preflight,
    run,
)


def _write_pool_map(path: Path) -> None:
    entries: dict[str, dict[str, object]] = {}
    offset = 0

    def add(key: str, dtype: str, shape: list[int]) -> None:
        nonlocal offset
        element_size = {
            "int8": 1,
            "bfloat16": 2,
            "float32": 4,
        }[dtype]
        offset = (offset + 511) // 512 * 512
        nbytes = element_size
        for item in shape:
            nbytes *= item
        entries[key] = {
            "offset": offset,
            "shape": shape,
            "dtype": dtype,
            "nbytes": nbytes,
        }
        offset += nbytes

    for key in MAIN_INT8_KEYS:
        add(key, "int8", [1])
    for key in MAIN_SCALE_KEYS:
        add(key, "float32", [1])
    for key in MTP_FP32_KEYS:
        add(key, "float32", [1])
    for key in MTP_BF16_KEYS:
        shape = [3, 1] if key in ("mtp_k_cache", "mtp_v_cache") else [1]
        add(key, "bfloat16", shape)
    for key in ("k_cache", "v_cache"):
        add(key, "bfloat16", [1, 1])
    path.write_text(
        json.dumps({"pool_bytes": offset, "map": entries}),
        encoding="utf-8",
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


def test_log_parsers_only_accept_executed_result() -> None:
    text = (
        "[worker] RUN done 2.61s max|logits|=12.0 argmax=303\n"
        "(vLLM golden next-token argmax=303)\n"
        "[worker] RUN done 1.94s tokens_row0=[6178, 410, 303]\n"
    )
    assert _extract_main_argmax(text) == 303
    assert _extract_tokens(text) == [6178, 410, 303]
    assert _extract_main_argmax("(vLLM golden next-token argmax=303)\n") is None


def test_active_exporter_scan_ignores_shell_command_text(
    tmp_path: Path,
) -> None:
    pool = tmp_path / "ipc"
    output = "\n".join(
        [
            (
                "101 bash bash -c 'python -m "
                "tests.step3p5.harnesses._stage_whole_mtp3_ipc "
                f"--out {pool}'"
            ),
            (
                "102 python python -m "
                "tests.step3p5.harnesses._stage_whole_mtp3_ipc "
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
                    "tests.step3p5.harnesses._stage_whole_mtp3_ipc "
                    f"--export-rank 0 --out {pool}"
                ),
            }
        ]


def test_pool_contract_rejects_bf16_main_routed_weight(tmp_path: Path) -> None:
    map_path = tmp_path / "pypto_weight_map.rank0.json"
    _write_pool_map(map_path)
    result = _validate_pool_map(map_path, rank=0)
    assert result["native_main_routed_dtype"] == "int8"

    payload = json.loads(map_path.read_text(encoding="utf-8"))
    payload["map"][MAIN_INT8_KEYS[0]]["dtype"] = "bfloat16"
    payload["map"][MAIN_INT8_KEYS[0]]["nbytes"] = 2
    map_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RunnerError, match="must be int8"):
        _validate_pool_map(map_path, rank=0)


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
