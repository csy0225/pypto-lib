from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch


DEFAULT_GOLDEN_ROOT = Path(
    "/mnt/nvme1/chensiyu/logs/step3p5_910b_v017/"
    "golden_step3p5_vllm_20260625_150728"
)


def _golden_root() -> Path:
    return Path(os.environ.get("STEP3P5_VLLM_GOLDEN_ROOT", str(DEFAULT_GOLDEN_ROOT)))


def _pypto_output_root() -> Path | None:
    value = os.environ.get("STEP3P5_PYPTO_E2E_OUTPUT_ROOT")
    return Path(value) if value else None


def _max_steps() -> int | None:
    value = os.environ.get("STEP3P5_E2E_MAX_STEPS", "1")
    if value.lower() in {"", "all", "none"}:
        return None
    return int(value)


def _float_env(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _load_logits(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if "logits" in obj:
            return obj["logits"].float()
        if "main_logits" in obj:
            return obj["main_logits"].float()
    if torch.is_tensor(obj):
        return obj.float()
    raise AssertionError(f"Unsupported logits artifact format: {path}")


def _case_logits_files(case: dict) -> dict[tuple[int, int], Path]:
    files: dict[tuple[int, int], Path] = {}
    for item in case.get("dump_files", []):
        meta = item.get("meta", {})
        if meta.get("name") != "main_logits":
            continue
        path = Path(item["file"])
        prefix = int(path.name.split("_", 1)[0])
        rank = int(meta["rank"])
        files[(prefix, rank)] = path
    return files


def test_step3p5_pypto_vs_vllm_end_to_end_logits_st() -> None:
    """Final Step3p5 e2e ST: compare PyPTO logits against vLLM golden logits.

    This is the actual end-to-end precision gate. It intentionally does not
    pass with only vLLM golden data: PyPTO must emit comparable logits first.

    Expected PyPTO output layout once the tensor-input runner lands::

        $STEP3P5_PYPTO_E2E_OUTPUT_ROOT/
          beijing_1tok/main_logits_step000_rank0.pt
          beijing_1tok/main_logits_step000_rank1.pt
          ...
          beijing_4tok/main_logits_step000_rank0.pt
          ...

    Each file may be either a tensor, {"logits": tensor}, or
    {"main_logits": tensor}. The comparison checks shape, finite values,
    elementwise closeness, and temperature=0 argmax token equality.
    """

    golden_root = _golden_root()
    if not golden_root.exists():
        pytest.xfail(f"missing vLLM golden root: {golden_root}")
    pypto_root = _pypto_output_root()
    if pypto_root is None or not pypto_root.exists():
        pytest.xfail(
            "Set STEP3P5_PYPTO_E2E_OUTPUT_ROOT to PyPTO-produced logits "
            "artifacts to enable the real vLLM-vs-PyPTO E2E gate.",
        )

    manifest = json.load(open(golden_root / "manifest.json"))
    max_steps = _max_steps()
    rtol = _float_env("STEP3P5_E2E_RTOL", 5e-3)
    atol = _float_env("STEP3P5_E2E_ATOL", 5e-3)
    pass_rate_threshold = _float_env("STEP3P5_E2E_PASS_RATE", 0.999)
    compared = 0
    for case in manifest["cases"]:
        case_name = case["name"]
        if not (pypto_root / case_name).exists():
            continue
        golden_logits = _case_logits_files(case)
        steps = sorted({step for step, _rank in golden_logits})
        if max_steps is not None:
            steps = steps[:max_steps]
        for local_step, golden_step in enumerate(steps):
            for rank in range(8):
                golden_path = golden_logits[(golden_step, rank)]
                pypto_path = pypto_root / case_name / f"main_logits_step{local_step:03d}_rank{rank}.pt"
                assert pypto_path.exists(), f"missing PyPTO logits artifact: {pypto_path}"

                golden = _load_logits(golden_path)
                pypto = _load_logits(pypto_path)
                assert torch.isfinite(pypto).all()

                if tuple(pypto.shape) == tuple(golden.shape):
                    expected = golden
                    expected_argmax = int(golden[-1].argmax().item())
                    pypto_argmax = int(pypto[-1].argmax().item())
                else:
                    assert pypto.ndim == golden.ndim
                    assert pypto.shape[:-1] == golden.shape[:-1]
                    assert golden.shape[-1] % 8 == 0
                    vocab_local = golden.shape[-1] // 8
                    assert pypto.shape[-1] == vocab_local
                    lo = rank * vocab_local
                    hi = lo + vocab_local
                    expected = golden[:, lo:hi]
                    expected_argmax = int(expected[-1].argmax().item())
                    pypto_argmax = int(pypto[-1].argmax().item())

                close = torch.isclose(pypto, expected, rtol=rtol, atol=atol)
                pass_rate = float(close.float().mean().item())
                max_abs_diff = float((pypto - expected).abs().max().item())
                assert pass_rate >= pass_rate_threshold, (
                    f"{case_name} step={local_step} rank={rank} "
                    f"pass_rate={pass_rate:.6f} max_abs_diff={max_abs_diff:.6f}"
                )
                assert pypto_argmax == expected_argmax
                compared += 1

    if compared == 0:
        pytest.xfail(f"no comparable PyPTO logits artifacts under {pypto_root}")
