from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

from tools.step3p5.pypto_layer0_detail_compare import compare


DEFAULT_DUMP_ROOT = Path(
    "/mnt/nvme1/chensiyu/logs/step3p5_910b_v017/"
    "vllm_tensor_dump_detail_layer0"
)
DEFAULT_CKPT_DIR = Path("/mnt/nvme1/chensiyu/step3p5_flash_release_hf_mtp3_bf16")


def test_step3p5_layer0_detail_tensor_input_st() -> None:
    dump_root = Path(os.environ.get("STEP3P5_VLLM_DETAIL_DUMP_ROOT", str(DEFAULT_DUMP_ROOT)))
    ckpt_dir = Path(os.environ.get("STEP3P5_CKPT_DIR", str(DEFAULT_CKPT_DIR)))
    if not dump_root.exists():
        pytest.xfail(f"missing vLLM detail dump root: {dump_root}")
    if not ckpt_dir.exists():
        pytest.xfail(f"missing Step3p5 checkpoint: {ckpt_dir}")

    report = compare(argparse.Namespace(
        dump_root=str(dump_root),
        ckpt_dir=str(ckpt_dir),
        occurrence=int(os.environ.get("STEP3P5_DETAIL_OCCURRENCE", "1")),
        tp_world_size=8,
        rtol=float(os.environ.get("STEP3P5_DETAIL_RTOL", "5e-3")),
        atol=float(os.environ.get("STEP3P5_DETAIL_ATOL", "5e-3")),
        mlp_rtol=float(os.environ.get("STEP3P5_DETAIL_MLP_RTOL", "8e-2")),
        mlp_atol=float(os.environ.get("STEP3P5_DETAIL_MLP_ATOL", "8e-2")),
        out=None,
    ))
    assert report["ok"]
    assert report["num_checks"] == 88
    assert report["worst_pass_rate"] >= 0.999
