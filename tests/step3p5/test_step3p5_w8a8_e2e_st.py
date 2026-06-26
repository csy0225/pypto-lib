from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


DEFAULT_W8A8_ROOT = Path(
    "/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_v001/"
    "golden_step3p5_w8a8_vllm_20260626_004648"
)
DEFAULT_ALL_LAYERS_REPORT = Path(
    "/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_v001/"
    "pypto_all_layers_detail_compare_w8a8_beijing1_atol1_report.json"
)
DEFAULT_FINAL_LOGITS_REPORT = Path(
    "/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_v001/"
    "pypto_final_logits_from_vllm_w8a8/final_logits_report.json"
)
DEFAULT_ACCEPTANCE_REPORT = Path(
    "/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_v001/"
    "decode_acceptance_w8a8_rank0.json"
)


def _path_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default)))


def _load_or_skip(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"missing W8A8 artifact: {path}")
    return json.loads(path.read_text())


def test_step3p5_w8a8_vllm_golden_manifest_st() -> None:
    root = _path_env("STEP3P5_W8A8_GOLDEN_ROOT", DEFAULT_W8A8_ROOT)
    manifest = _load_or_skip(root / "manifest.json")

    assert manifest["quantization"] == "w8a8_dynamic"
    cases = {case["name"]: case for case in manifest["cases"]}
    assert "beijing_1tok" in cases
    case = cases["beijing_1tok"]
    assert case["response_summary"]["finish_reason"] == "length"
    assert case["num_dump_files"] >= 8 * 48
    names = {item["meta"]["name"] for item in case["dump_files"]}
    assert {"model_input", "layer_00_layer_input", "layer_44_ffn_out", "main_logits"} <= names


def test_step3p5_w8a8_weight_acceptance_st() -> None:
    report = _load_or_skip(_path_env("STEP3P5_W8A8_ACCEPTANCE_REPORT", DEFAULT_ACCEPTANCE_REPORT))

    assert report["ok"]
    assert report["checkpoint"]["is_w8a8_dynamic"]
    assert report["checkpoint"]["index_name"] == "quant_model_weights.safetensors.index.json"
    assert report["dispatcher"]["observed_layers"] == 48
    assert report["precision"]["worst_pass_rate"] == 1.0


def test_step3p5_w8a8_all_layers_detail_alignment_st() -> None:
    report = _load_or_skip(_path_env("STEP3P5_W8A8_ALL_LAYERS_REPORT", DEFAULT_ALL_LAYERS_REPORT))

    assert report["ok"]
    assert report["routed_w8a8_dynamic"]
    assert report["num_checks"] == 3960
    assert report["worst_pass_rate"] >= 0.999
    assert len(report["layers"]) == 45
    assert all(layer["ok"] for layer in report["layers"])


def test_step3p5_w8a8_final_logits_e2e_st() -> None:
    report = _load_or_skip(_path_env("STEP3P5_W8A8_FINAL_LOGITS_REPORT", DEFAULT_FINAL_LOGITS_REPORT))

    assert report["ok"]
    assert report["ckpt_dir"].endswith("step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp")
    assert report["cases"]
    for case in report["cases"]:
        assert case["pass"]
        for step in case["steps"]:
            assert step["pass"]
            assert step["full_logits"]["argmax_match"]
            assert step["full_logits"]["pass_rate"] >= 0.999
