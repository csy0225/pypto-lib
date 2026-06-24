#!/usr/bin/env python3
"""Step3p5 decode e2e precision readiness preflight.

Host-only preflight for the final vLLM-vs-PyPTO end-to-end precision goal.
It checks external prerequisites and executes the existing torch-level whole-
decode smoke checks. It intentionally does not compile NPU kernels.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any


def _default_ckpt_dir() -> str:
    from models.step3p5.weight_loader import DEFAULT_CKPT_DIR

    return DEFAULT_CKPT_DIR


def _find_vllm(repo_root: Path, explicit: str | None) -> dict[str, Any]:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("STEP3P5_VLLM_ROOT") or os.environ.get("VLLM_ROOT")
    if env:
        candidates.append(Path(env))
    project_root = repo_root.parents[1]
    candidates.extend([
        project_root / "vllm",
        project_root / "vllm-ascend",
        project_root / "stepcast-vllm",
        project_root / "pypto-serving",
    ])

    checked = []
    for path in candidates:
        checked.append(str(path))
        if not path.exists():
            continue
        markers = [path / "vllm", path / "setup.py", path / "pyproject.toml"]
        if any(m.exists() for m in markers):
            return {"ok": True, "path": str(path), "checked": checked}
    spec = importlib.util.find_spec("vllm")
    if spec is not None:
        return {"ok": True, "path": "python:vllm", "checked": checked}
    return {"ok": False, "path": None, "checked": checked}


def _check_ckpt(path: Path) -> dict[str, Any]:
    index = path / "model.safetensors.index.json"
    single = path / "model.safetensors"
    shards = sorted(path.glob("*.safetensors")) if path.exists() else []
    return {
        "ok": path.exists() and (index.exists() or single.exists() or bool(shards)),
        "path": str(path),
        "exists": path.exists(),
        "has_index": index.exists(),
        "has_single": single.exists(),
        "num_safetensors": len(shards),
    }


def _check_decode_fwd_wiring() -> dict[str, Any]:
    from models.step3p5 import decode_fwd

    source = inspect.getsource(decode_fwd._build_decode_fwd_program)
    has_todo = "Phase 8" in source and "expected to wire" in source
    final_head_only = "The final RMSNorm + LM head per-rank shard is run here" in source
    has_layer_call = ".host_orch(" in source or "select_decode_layer(li" in source
    return {
        "ok": has_layer_call and not has_todo and not final_head_only,
        "has_select_decode_layer_reference": "select_decode_layer(" in source,
        "has_phase8_todo": has_todo,
        "final_head_only": final_head_only,
        "note": "Step3p5DecodeFwd.host_orch must call all 45 layer programs before final RMS+LM head.",
    }


def _run_host_smokes(batch: int, pass_rate: float) -> dict[str, Any]:
    from models.step3p5.decode_fwd import run_distributed_mock
    from models.step3p5.step3p5_decode import run_smoke

    decode_fwd = run_distributed_mock(batch=batch, pass_rate_threshold=pass_rate)
    decode_smoke = run_smoke(batch=batch, use_synthetic=True)
    return {
        "decode_fwd_mock": decode_fwd,
        "step3p5_decode_synthetic": decode_smoke,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("--vllm-root", default=None)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--pass-rate", type=float, default=0.97)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    ckpt_dir = Path(args.ckpt_dir or _default_ckpt_dir())
    report: dict[str, Any] = {
        "ckpt": _check_ckpt(ckpt_dir),
        "vllm": _find_vllm(repo_root, args.vllm_root),
        "decode_fwd_wiring": _check_decode_fwd_wiring(),
        "known_precision_policy": {
            "head_gate": "PyPTO currently bypasses head_gate (x1); vLLM parity must either patch vLLM the same way or accept this as an L1 blocker.",
            "moe8": "8-card MoE ST runtime passes, but golden precision for MoE is not yet implemented.",
            "split_dispatch": "Current split EP dispatch is correctness-first; non-split fusion is a Phase 22 perf item.",
        },
    }
    report["host_smokes"] = _run_host_smokes(args.batch, args.pass_rate)

    blockers = []
    if not report["ckpt"]["ok"]:
        blockers.append("checkpoint_unavailable")
    if not report["vllm"]["ok"]:
        blockers.append("vllm_unavailable")
    if not report["decode_fwd_wiring"]["ok"]:
        blockers.append("decode_fwd_45_layers_not_wired")
    if not report["host_smokes"]["decode_fwd_mock"]["ok"]:
        blockers.append("decode_fwd_mock_failed")
    if not report["host_smokes"]["step3p5_decode_synthetic"]["ok"]:
        blockers.append("step3p5_decode_synthetic_failed")
    blockers.append("head_gate_parity_policy_unresolved")
    blockers.append("moe8_golden_missing")
    report["ready_for_final_e2e_precision"] = not blockers
    report["blockers"] = blockers

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("Step3p5 decode e2e precision readiness")
        print("=" * 60)
        print(f"checkpoint : {'OK' if report['ckpt']['ok'] else 'MISSING'}  {report['ckpt']['path']}")
        print(f"vLLM       : {'OK' if report['vllm']['ok'] else 'MISSING'}  {report['vllm']['path']}")
        print(f"decode_fwd wiring: {'OK' if report['decode_fwd_wiring']['ok'] else 'INCOMPLETE'}")
        print(f"decode_fwd mock : {'OK' if report['host_smokes']['decode_fwd_mock']['ok'] else 'BAD'}  worst={report['host_smokes']['decode_fwd_mock']['worst_pass_rate']:.6f}")
        print(f"decode synthetic: {'OK' if report['host_smokes']['step3p5_decode_synthetic']['ok'] else 'BAD'}  pass={report['host_smokes']['step3p5_decode_synthetic']['pass_rate']:.6f}")
        print("blockers   : " + (", ".join(blockers) if blockers else "none"))
    return 0 if report["ready_for_final_e2e_precision"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
