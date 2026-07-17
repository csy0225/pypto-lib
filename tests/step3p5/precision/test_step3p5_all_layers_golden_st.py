from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

from tools.step3p5.all_layers_golden_scan import build_report


DEFAULT_GOLDEN_ROOT = Path(
    "/mnt/nvme1/chensiyu/logs/step3p5_910b_v017/"
    "golden_step3p5_vllm_20260625_150728"
)


def test_step3p5_vllm_golden_all_layers_st() -> None:
    golden_root = Path(os.environ.get("STEP3P5_VLLM_GOLDEN_ROOT", str(DEFAULT_GOLDEN_ROOT)))
    if not golden_root.exists():
        pytest.xfail(f"missing vLLM golden root: {golden_root}")

    report = build_report(argparse.Namespace(
        golden_root=str(golden_root),
        tp_world_size=8,
        num_layers=45,
    ))
    assert report["ok"]
    assert report["cases"]
    for case in report["cases"]:
        assert case["num_steps"] > 0
        for step in case["steps"]:
            assert step["ok"], f"{case['name']} step={step['step']}"
            assert len(step["layers"]) == 45
