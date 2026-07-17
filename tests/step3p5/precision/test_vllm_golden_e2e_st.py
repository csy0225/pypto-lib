from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

from tools.step3p5.vllm_golden_compare import build_report


DEFAULT_GOLDEN_ROOT = Path(
    "/mnt/nvme1/chensiyu/logs/step3p5_910b_v017/"
    "golden_step3p5_vllm_20260625_150728"
)
DEFAULT_CKPT_DIR = Path("/mnt/nvme1/chensiyu/step3p5_flash_release_hf_mtp3_bf16")


def _golden_root() -> Path:
    return Path(os.environ.get("STEP3P5_VLLM_GOLDEN_ROOT", str(DEFAULT_GOLDEN_ROOT)))


def _ckpt_dir() -> Path:
    return Path(os.environ.get("STEP3P5_CKPT_DIR", str(DEFAULT_CKPT_DIR)))


def _report(*, run_pypto_acceptance: bool = True) -> dict:
    golden_root = _golden_root()
    if not golden_root.exists():
        pytest.skip(f"Step3p5 vLLM golden root is not available: {golden_root}")
    ckpt_dir = _ckpt_dir()
    if run_pypto_acceptance and not ckpt_dir.exists():
        pytest.skip(f"Step3p5 checkpoint is not available: {ckpt_dir}")

    args = argparse.Namespace(
        golden_root=str(golden_root),
        ckpt_dir=str(ckpt_dir),
        max_steps=2,
        run_pypto_acceptance=run_pypto_acceptance,
        out=None,
    )
    return build_report(args)


def test_vllm_golden_sampler_and_pypto_readiness_st() -> None:
    """ST: vLLM tensor golden is readable and PyPTO real-ckpt readiness passes.

    This test is tokenizer-free: it consumes vLLM decoder tensor dumps and
    validates the temperature=0 sampler policy as argmax over dumped logits.
    """

    report = _report(run_pypto_acceptance=True)

    assert report["status"] == "GOLDEN_READY_PYPTO_E2E_BLOCKED"
    assert report["full_e2e"]["finish_reason"] == "stop"
    assert report["pypto_acceptance"]["returncode"] == 0
    assert report["pypto_acceptance"]["stdout_json"]["ok"]
    assert report["pypto_acceptance"]["stdout_json"]["checkpoint"]["ok"]
    assert report["pypto_acceptance"]["stdout_json"]["dispatcher"]["expected_layers"] == 48
    assert report["pypto_acceptance"]["stdout_json"]["dispatcher"]["observed_layers"] == 48

    cases = {case["name"]: case for case in report["cases"]}
    assert {"beijing_1tok", "beijing_4tok", "math_8tok"} <= set(cases)

    for case in cases.values():
        assert case["num_dump_files"] > 0
        assert case["tensor_counts"]["model_input"] >= 8
        assert case["tensor_counts"]["main_logits"] >= 8
        assert case["tensor_counts"]["layer_00_out"] >= 8

        first_input = case["first_meta"]["model_input"]
        assert first_input["hidden_states"]["shape"][-1] == 4096
        assert first_input["hidden_states"]["dtype"] == "torch.bfloat16"

        sampler_steps = case["sampler_temperature0"]
        assert sampler_steps, f"missing argmax sampler steps for {case['name']}"
        logits_checks = case["final_logits_correctness"]
        assert len(logits_checks) == len(sampler_steps)
        for step, logits_check in zip(sampler_steps, logits_checks, strict=True):
            assert step["num_ranks"] == 8
            assert step["rank_argmax_consensus"]
            assert step["rank0_token_id"] is not None
            rank0 = step["rank_results"]["0"]
            assert rank0["dtype"] == "torch.bfloat16"
            assert rank0["shape"][-1] == 128896
            assert rank0["finite"]
            assert rank0["row_index"] == rank0["shape"][0] - 1

            assert logits_check["num_ranks"] == 8
            assert logits_check["shape_consensus"]
            assert logits_check["all_finite"]
            assert logits_check["rank_exact_match"]
            for comparison in logits_check["rank_comparisons_to_rank0"].values():
                assert comparison["shape_match"]
                assert comparison["pass_rate"] == 1.0
                assert comparison["max_abs_diff"] == 0.0


@pytest.mark.xfail(
    reason=(
        "PyPTO Step3p5 tensor-input full-network decode runner is not wired yet; "
        "this is the final e2e precision gate once layer_XX_out/main_logits can be emitted."
    ),
    strict=True,
)
def test_pypto_tensor_input_decode_matches_vllm_golden_st() -> None:
    """ST gate for the future true vLLM-vs-PyPTO tensor-input e2e comparison."""

    report = _report(run_pypto_acceptance=False)
    assert report["pypto_tensor_input_decode_status"]["implemented"]
    assert not report["blockers"]
