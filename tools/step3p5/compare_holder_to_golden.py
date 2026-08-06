#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Tier-2 holder->golden comparator for Step3p5 W8A8 prefill numeric validation.

This closes the dataflow gap documented in section 4 blocker #1 of
``prefill_e2e_validation_toolchain.md``: no tool previously compared the
on-device ``WholePrefillHolder`` ``next_hidden`` output against a golden
vLLM hidden. This script does exactly that, reusing the Tier-1 comparison
primitives (``pypto_all_layers_detail_compare._compare`` for pass_rate and
``final_logits_from_vllm`` for the golden ``main_logits`` load + tail
RMSNorm/LM-head logits path) so the ``pass_rate >= 0.997`` gate is
identical to the Tier-1 suite (``prefill_precision_suite.py`` default).

Both the holder ``next_hidden`` and the golden ``main_logits.hidden_states``
are pre-final-norm post-45-layer hidden states ([T, HIDDEN] BF16) -- the
same numeric boundary -- so a direct element-wise comparison is valid when
the holder and golden run the same prompt at the same token positions.

Modes
-----
1. **Golden-present** (default when ``--golden-root`` exists): for each
   ``--seq-len`` case, load the holder-produced
   ``prefill_t{N:03d}_active_hidden.pt`` from ``--holder-dir`` (or produce
   it on-device first with ``--run-holder``) and the golden
   ``main_logits`` dump, then compare:
     (a) element-wise ``holder_next_hidden[0, :T, :]`` vs golden
         ``hidden_states`` (pass_rate gate >= ``--pass-rate``);
     (b) tail logits: holder hidden -> CPU final RMSNorm + LM head ->
         logits vs golden ``main_logits.logits`` (pass_rate + argmax_match
         gate, mirroring ``final_logits_from_vllm``).

2. **Self-consistency** (``--self-consistency``; no golden needed): runs
   ``models.step3p5.step3p5_prefill._torch_reference_prefill`` on the same
   input hidden. The reference walks all 45 layers with RMSNorm + dense
   MLP + shared-expert MLP + residual wiring, but the attention block is a
   no-op stand-in and routed-expert MoE is NOT modeled. So:
     - reference bit-determinism (run twice, rtol=0 / atol=0) is a REAL gate;
     - holder-vs-reference pass_rate is a DIAGNOSTIC ONLY (never a gate) --
       large diffs are expected because the reference omits attention and
       routed MoE. This validates only the dense+shared+residual path.

3. **Self-test** (``--self-test``): synthetic round-trip -- two random
   tensors (one = the other + tiny noise) -- to verify the pass_rate
   machinery and the >= ``--pass-rate`` gate. Card-free, no weights, no
   golden.

Golden-absent handling
----------------------
If ``--golden-root`` does not exist and neither ``--self-consistency`` nor
``--self-test`` is requested, the comparator prints a clear message
("golden not present at <path>; collect via
collect_w8a8_prefill_golden.py on a vLLM host first") and exits with code
3 -- distinct from pass=0 and numeric-fail=2, and NOT a crash or false pass.

Exit codes: 0 = pass, 2 = numeric fail, 3 = cannot run (golden absent /
required inputs missing).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Neutral /tmp default (no username path) -- matches collect_w8a8_prefill_golden.py.
DEFAULT_LOG_ROOT = Path(
    os.environ.get(
        "STEP3P5_W8A8_PREFILL_LOG_ROOT",
        "/tmp/step3p5_w8a8_prefill_logs",
    )
)
DEFAULT_GOLDEN_ROOT = DEFAULT_LOG_ROOT / "golden_step3p5_w8a8_prefill_vllm_sampled"
# Neutral /mnt default (matches tests/step3p5/harnesses/_stage_prefill_hidden_only.py).
DEFAULT_CKPT = os.environ.get(
    "STEP3P5_CKPT_DIR",
    "/mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp",
)

# Holder A/B target: the live holder gate only serves T <= PREFILL_T=128
# (classify_prefill_gate rejects T>128). Long golden cases (1k..128k) are
# sampled (max 128 rows/forward) and are NOT element-wise comparable to a
# 128-token holder run -- use the Tier-1 suite (prefill_precision_suite.py)
# for those. Default to the direct 128-token A/B.
DEFAULT_SEQ_LENS = (128,)

PASS_RATE_DEFAULT = 0.997
RTOL_DEFAULT = 5e-3
ATOL_DEFAULT = 5e-3
LOGITS_CHUNK_DEFAULT = 4096
# Match the production constants used by the holder / staging harness so
# shape checks line up with prefill_t{N:03d}_active_hidden.pt.
PREFILL_T = 128
HIDDEN = 4096

EXIT_PASS = 0
EXIT_FAIL = 2
EXIT_CANNOT_RUN = 3


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Golden-absent handling.
# --------------------------------------------------------------------------- #
def _golden_absent_exit(golden_root: Path) -> int:
    print(
        f"[compare] golden not present at {golden_root}\n"
        "    collect via collect_w8a8_prefill_golden.py on a vLLM host first "
        "(needs vanilla vLLM + debug-dump hook on port 8001;\n"
        "    add --seq-len 128 for the direct holder A/B target).\n"
        "    Alternatively run --self-consistency (torch reference, no golden) "
        "or --self-test (synthetic pass_rate check).",
        file=sys.stderr,
    )
    return EXIT_CANNOT_RUN


# --------------------------------------------------------------------------- #
# Holder output loading.
# --------------------------------------------------------------------------- #
def _holder_hidden_path(holder_dir: Path, seq_len: int) -> Path:
    return Path(holder_dir) / f"prefill_t{seq_len:03d}_active_hidden.pt"


def _load_holder_hidden(holder_dir: Path, seq_len: int):
    """Load ``prefill_t{N:03d}_active_hidden.pt`` -> [tp, N, HIDDEN] BF16."""
    import torch  # noqa: PLC0415

    path = _holder_hidden_path(holder_dir, seq_len)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


# --------------------------------------------------------------------------- #
# Golden case resolution + main_logits load (reuses final_logits_from_vllm).
# --------------------------------------------------------------------------- #
def _load_manifest(golden_root: Path) -> dict[str, Any]:
    manifest_path = golden_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing golden manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _match_golden_case_name(golden_root: Path, seq_len: int) -> str | None:
    """Find the golden case for ``seq_len`` (by manifest seq_len, then name)."""
    try:
        manifest = _load_manifest(golden_root)
    except FileNotFoundError:
        return None
    for case in manifest.get("cases", []):
        if int(case.get("seq_len", -1)) == int(seq_len):
            return case["name"]
    # Collector names cases prefill_{seq//1024}k; fall back to that.
    fallback = f"prefill_{seq_len // 1024}k"
    for case in manifest.get("cases", []):
        if case.get("name") == fallback:
            return fallback
    return None


def _resolve_cases(golden_root: Path, args: argparse.Namespace) -> list[tuple[int, str]]:
    """Build (seq_len, case_name) pairs from --case or --seq-len."""
    manifest = _load_manifest(golden_root)
    by_name = {c["name"]: c for c in manifest.get("cases", []) if c.get("dump_files")}
    pairs: list[tuple[int, str]] = []
    if args.case:
        for name in args.case:
            if name not in by_name:
                raise KeyError(
                    f"case {name!r} not in golden manifest "
                    f"{golden_root / 'manifest.json'}"
                )
            seq = int(by_name[name].get("seq_len", -1))
            if seq <= 0:
                raise KeyError(f"case {name!r} has no seq_len in manifest")
            pairs.append((seq, name))
        return pairs
    seq_lens = tuple(args.seq_len or DEFAULT_SEQ_LENS)
    for seq in seq_lens:
        name = _match_golden_case_name(golden_root, seq)
        if name is None:
            raise FileNotFoundError(
                f"no golden case matching seq_len={seq} in {golden_root}"
            )
        pairs.append((seq, name))
    return pairs


def _load_golden_hidden_logits(
    golden_root: Path, case_name: str
) -> tuple[Any, Any] | None:
    """Load golden ``main_logits`` rank0 step0 -> (hidden [T,H], logits [T,vocab])."""
    import torch  # noqa: PLC0415
    from tools.step3p5.final_logits_from_vllm import (  # noqa: PLC0415
        _load_vllm_main_logits_items,
        _rank0_per_step,
    )

    try:
        items = _rank0_per_step(
            _load_vllm_main_logits_items(golden_root, case_name)
        )
    except KeyError:
        return None
    if not items:
        return None
    obj = torch.load(items[0]["file"], map_location="cpu")
    hidden = obj["hidden_states"].bfloat16()
    logits = obj["logits"].float()
    return hidden, logits


# --------------------------------------------------------------------------- #
# Holder live run (delegates to the staging harness; needs 8 NPU devices).
# --------------------------------------------------------------------------- #
def _run_holder_live(
    args: argparse.Namespace, seq_lens: tuple[int, ...]
) -> bool:
    out = Path(args.holder_dir)
    out.mkdir(parents=True, exist_ok=True)
    for length in seq_lens:
        if not 1 <= length <= PREFILL_T:
            raise ValueError(
                f"--run-holder prompt len {length} out of [1,{PREFILL_T}] "
                "(PREFILL_T); the live holder gate rejects T>128"
            )
    command = [
        sys.executable,
        "-m",
        "tests.step3p5.harnesses._stage_prefill_hidden_only",
        "--device",
        args.device,
        "--ckpt",
        args.ckpt_dir,
        "--out",
        str(out),
        "--num-blocks",
        str(args.num_blocks),
        "--prompt-lens",
        ",".join(str(s) for s in seq_lens),
    ]
    print(
        f"[compare] running holder live (needs 8 NPU + PTO_ISA_ROOT): "
        f"{' '.join(command)}",
        file=sys.stderr,
    )
    proc = subprocess.run(command, cwd=str(_repo_root()))
    return proc.returncode == 0


# --------------------------------------------------------------------------- #
# Core comparison: holder next_hidden vs golden hidden + tail logits.
# --------------------------------------------------------------------------- #
def _compare_holder_to_golden(
    holder_hidden: Any,
    golden_hidden: Any,
    golden_logits: Any,
    *,
    ckpt_dir: str,
    rtol: float,
    atol: float,
    pass_rate_threshold: float,
    chunk_size: int,
) -> dict[str, Any]:
    from tools.step3p5.final_logits_from_vllm import (  # noqa: PLC0415
        _load_head_weights,
        _logits_from_full_head,
        _tensor_report,
        _zero_centered_rmsnorm,
    )
    from tools.step3p5.pypto_all_layers_detail_compare import _compare  # noqa: PLC0415

    # holder_hidden is [tp, T, HIDDEN]; the residual stream is replicated
    # across tp ranks, so rank0 [T, HIDDEN] is the canonical comparison
    # target. Report tp_spread as a replication sanity diagnostic.
    tp = int(holder_hidden.shape[0])
    holder_rank0 = holder_hidden[0].bfloat16().contiguous()
    if tp > 1:
        tp_spread = float(
            (holder_hidden.float() - holder_rank0.unsqueeze(0).float())
            .abs()
            .max()
            .item()
        )
    else:
        tp_spread = 0.0

    # (a) element-wise hidden comparison (same pre-final-norm boundary).
    hidden_report = _compare(
        "holder_next_hidden_vs_golden_hidden_states",
        holder_rank0,
        golden_hidden,
        rtol=rtol,
        atol=atol,
        pass_rate_threshold=pass_rate_threshold,
    )

    # (b) tail logits: holder hidden -> CPU final RMSNorm + LM head -> logits,
    # compared vs golden main_logits.logits (pass_rate + argmax_match gate).
    final_norm, lm_head = _load_head_weights(ckpt_dir)
    holder_normed = _zero_centered_rmsnorm(holder_rank0, final_norm).bfloat16()
    holder_logits = _logits_from_full_head(holder_normed, lm_head, chunk_size)
    logits_report = _tensor_report(
        holder_logits, golden_logits, rtol=rtol, atol=atol
    )
    logits_ok = (
        logits_report["shape_match"]
        and logits_report["finite"]
        and logits_report["pass_rate"] >= pass_rate_threshold
        and logits_report["argmax_match"]
    )
    return {
        "hidden": hidden_report,
        "logits": {
            **logits_report,
            "pass_rate_threshold": pass_rate_threshold,
            "ok": bool(logits_ok),
        },
        "tp": tp,
        "tp_spread": tp_spread,
        "ok": bool(hidden_report["ok"] and logits_ok),
    }


def _run_golden_present(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    golden_root = Path(args.golden_root)
    if not golden_root.exists():
        report = {
            "mode": "golden_absent",
            "golden_root": str(golden_root),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        return report, _golden_absent_exit(golden_root)

    if not args.holder_dir and not args.run_holder:
        print(
            "[compare] golden-present mode requires --holder-dir "
            "(pre-saved prefill_t{N:03d}_active_hidden.pt) or --run-holder",
            file=sys.stderr,
        )
        return (
            {
                "mode": "golden_present",
                "error": "missing --holder-dir / --run-holder",
                "golden_root": str(golden_root),
            },
            EXIT_CANNOT_RUN,
        )

    seq_lens = tuple(args.seq_len or DEFAULT_SEQ_LENS)
    if args.run_holder:
        if not _run_holder_live(args, seq_lens):
            return (
                {
                    "mode": "golden_present",
                    "error": "holder live run failed",
                    "golden_root": str(golden_root),
                },
                EXIT_FAIL,
            )

    pairs = _resolve_cases(golden_root, args)
    case_reports: list[dict[str, Any]] = []
    overall_ok = True
    for seq_len, case_name in pairs:
        entry: dict[str, Any] = {
            "case": case_name,
            "seq_len": seq_len,
            "ok": False,
        }
        holder_hidden = _load_holder_hidden(Path(args.holder_dir), seq_len)
        if holder_hidden is None:
            entry["error"] = (
                f"missing holder output {_holder_hidden_path(Path(args.holder_dir), seq_len)}"
            )
            entry["note"] = (
                "run _stage_prefill_hidden_only --prompt-lens "
                f"{seq_len} --out {args.holder_dir} (or pass --run-holder)"
            )
            case_reports.append(entry)
            overall_ok = False
            continue

        golden = _load_golden_hidden_logits(golden_root, case_name)
        if golden is None:
            entry["error"] = f"missing golden main_logits dump for case {case_name!r}"
            case_reports.append(entry)
            overall_ok = False
            continue
        golden_hidden, golden_logits = golden

        # Sampling guard: the golden collector samples max 128 rows/forward.
        # For seq_len > 128 the golden hidden is a subsample and is NOT
        # element-wise comparable to a holder run (which is capped at
        # PREFILL_T=128). The holder can also not produce seq_len > 128.
        if int(golden_hidden.shape[0]) != seq_len:
            entry["skipped"] = True
            entry["note"] = (
                f"golden hidden_states has {int(golden_hidden.shape[0])} rows "
                f"(sampled) but case seq_len={seq_len}; direct element-wise "
                "comparison requires the holder and golden to run the same "
                "prompt at the same token positions. For seq_len>128 golden "
                "cases use the Tier-1 suite (prefill_precision_suite.py)."
            )
            case_reports.append(entry)
            # Skip is not a failure of the comparator, but it is not a pass
            # either -- the case produced no numeric evidence.
            overall_ok = False
            continue

        result = _compare_holder_to_golden(
            holder_hidden,
            golden_hidden,
            golden_logits,
            ckpt_dir=args.ckpt_dir,
            rtol=args.rtol,
            atol=args.atol,
            pass_rate_threshold=args.pass_rate,
            chunk_size=args.logits_chunk_size,
        )
        entry.update(result)
        case_reports.append(entry)
        if not result["ok"]:
            overall_ok = False

    report = {
        "mode": "golden_present",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "golden_root": str(golden_root),
        "ckpt_dir": args.ckpt_dir,
        "holder_dir": str(args.holder_dir) if args.holder_dir else None,
        "tp_world_size": args.tp_world_size,
        "rtol": args.rtol,
        "atol": args.atol,
        "pass_rate_threshold": args.pass_rate,
        "cases": case_reports,
        "ok": bool(case_reports) and overall_ok,
    }
    return report, (EXIT_PASS if report["ok"] else EXIT_FAIL)


# --------------------------------------------------------------------------- #
# Self-consistency mode (no golden): torch BF16 reference forward.
# --------------------------------------------------------------------------- #
def _load_prompt_embedding(ckpt: str, prompt_len: int):
    """Load ``prompt_len`` contiguous token embedding rows (arange(T))."""
    import safetensors.torch as st  # noqa: PLC0415
    import torch  # noqa: PLC0415

    if not 1 <= prompt_len <= PREFILL_T:
        raise ValueError(
            f"prompt_len must be in [1,{PREFILL_T}] (PREFILL_T), got {prompt_len}"
        )
    index_path = Path(ckpt) / "quant_model_weights.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard = index["weight_map"]["model.embed_tokens.weight"]
    with st.safe_open(str(Path(ckpt) / shard), framework="pt") as handle:
        rows = handle.get_slice("model.embed_tokens.weight")[:prompt_len, :]
    if tuple(rows.shape) != (prompt_len, HIDDEN):
        raise ValueError(f"prompt embedding shape={tuple(rows.shape)}")
    return rows.to(torch.bfloat16).contiguous()


def _load_input_hidden(
    args: argparse.Namespace, seq_len: int, hidden_dim: int
):
    """Resolve the input hidden for the reference: --input-hidden, ckpt
    arange(T) embedding, or synthetic random aligned to ``hidden_dim``."""
    import torch  # noqa: PLC0415

    if args.input_hidden:
        hidden = torch.load(args.input_hidden, map_location="cpu").bfloat16()
        if hidden.shape[-1] != hidden_dim:
            raise ValueError(
                f"--input-hidden last dim {hidden.shape[-1]} != reference "
                f"hidden_dim {hidden_dim}"
            )
        return hidden
    # Real ckpt embeddings are HIDDEN-wide (4096); only load them when the
    # reference bundle matches that width (i.e. the real-ckpt bundle).
    if (
        args.ckpt_dir
        and Path(args.ckpt_dir).is_dir()
        and hidden_dim == HIDDEN
    ):
        return _load_prompt_embedding(args.ckpt_dir, seq_len)
    gen = torch.Generator().manual_seed(0)
    return (torch.rand(seq_len, hidden_dim, generator=gen) - 0.5).bfloat16()


def _run_self_consistency(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    from models.step3p5.step3p5_prefill import _torch_reference_prefill  # noqa: PLC0415
    from models.step3p5.weight_loader import (  # noqa: PLC0415
        KEY_INPUT_RMS,
        build_compact_shape_table,
        build_synthetic_bundle,
        load_step3p5_weights_for_rank,
    )
    from tools.step3p5.pypto_all_layers_detail_compare import _compare  # noqa: PLC0415

    seq_len = (args.seq_len or DEFAULT_SEQ_LENS)[0]
    if not 1 <= seq_len <= PREFILL_T:
        raise ValueError(
            f"--seq-len {seq_len} out of [1,{PREFILL_T}] for self-consistency"
        )
    tp = args.tp_world_size

    # Real ckpt -> 4096-wide bundle (matches holder output shape); else a
    # compact synthetic bundle (card-free, ~100 MB, hidden=256).
    if args.ckpt_dir and Path(args.ckpt_dir).is_dir():
        bundle = load_step3p5_weights_for_rank(
            args.ckpt_dir, 0, tp, int8_routed=False
        )
        bundle_mode = "ckpt_rank0"
    else:
        bundle = build_synthetic_bundle(
            0, tp, seed=0, shape_overrides=build_compact_shape_table(tp)
        )
        bundle_mode = "synthetic_compact"

    # Derive the hidden dim from the bundle so the synthetic-compact path
    # (hidden=256) gets a matching synthetic input, and the real-ckpt path
    # (hidden=4096) loads real arange(T) embeddings.
    hidden_dim = int(bundle[KEY_INPUT_RMS].shape[-1])
    input_hidden = _load_input_hidden(args, seq_len, hidden_dim)

    ref1 = _torch_reference_prefill(bundle, input_hidden)
    ref2 = _torch_reference_prefill(bundle, input_hidden)
    determinism = _compare(
        "reference_determinism",
        ref1,
        ref2,
        rtol=0.0,
        atol=0.0,
        pass_rate_threshold=1.0,
    )

    holder_diag: dict[str, Any] | None = None
    if args.holder_dir:
        holder_hidden = _load_holder_hidden(Path(args.holder_dir), seq_len)
        if holder_hidden is not None:
            holder_rank0 = holder_hidden[0].bfloat16().contiguous()
            if tuple(holder_rank0.shape) == tuple(ref1.shape):
                diag = _compare(
                    "holder_vs_reference_diagnostic",
                    holder_rank0,
                    ref1,
                    rtol=args.rtol,
                    atol=args.atol,
                    pass_rate_threshold=args.pass_rate,
                )
                # Deliberately NOT a gate: the reference omits attention
                # and routed MoE, so large diffs are expected. Reported for
                # diagnostic insight into the dense+shared+residual overlap.
                diag["ok"] = False
                diag["gate"] = "diagnostic_only"
                holder_diag = diag
            else:
                holder_diag = {
                    "name": "holder_vs_reference_diagnostic",
                    "shape": tuple(holder_rank0.shape),
                    "expected_shape": tuple(ref1.shape),
                    "shape_match": False,
                    "note": (
                        "holder/reference hidden shape mismatch (likely "
                        "synthetic-compact bundle vs 4096-wide holder); "
                        "pass --ckpt-dir to align shapes"
                    ),
                    "gate": "diagnostic_only",
                }

    report = {
        "mode": "self_consistency",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "bundle_mode": bundle_mode,
        "ckpt_dir": args.ckpt_dir,
        "seq_len": seq_len,
        "input_hidden_shape": tuple(input_hidden.shape),
        "reference_hidden_shape": tuple(ref1.shape),
        "determinism": determinism,
        "holder_vs_reference": holder_diag,
        "validated": (
            "RMSNorm + dense MLP + shared-expert MLP + residual wiring "
            "(45 layers). NOT validated: attention block (no-op stand-in "
            "in _torch_reference_prefill) and routed-expert MoE (shared-"
            "expert proxy only). Use golden-present mode for full numeric "
            "validation including attention + routed MoE."
        ),
        "ok": bool(determinism["ok"]),
    }
    return report, (EXIT_PASS if report["ok"] else EXIT_FAIL)


# --------------------------------------------------------------------------- #
# Self-test mode: synthetic round-trip pass_rate check.
# --------------------------------------------------------------------------- #
def _run_self_test(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    import torch  # noqa: PLC0415
    from tools.step3p5.pypto_all_layers_detail_compare import _compare  # noqa: PLC0415

    gen = torch.Generator().manual_seed(42)
    expected = (torch.rand(PREFILL_T, HIDDEN, generator=gen) - 0.5).bfloat16()
    # Small absolute offset (2e-3) that survives BF16 rounding yet stays
    # within rtol=5e-3 / atol=5e-3, so pass_rate ~ 1.0 with a nonzero diff.
    good_candidate = (expected.float() + 2e-3).bfloat16()
    good_report = _compare(
        "self_test_good_candidate",
        good_candidate,
        expected,
        rtol=args.rtol,
        atol=args.atol,
        pass_rate_threshold=args.pass_rate,
    )
    # Large offset -> should fail the >= --pass-rate gate.
    bad_candidate = (expected.float() + 1.0).bfloat16()
    bad_report = _compare(
        "self_test_bad_candidate",
        bad_candidate,
        expected,
        rtol=args.rtol,
        atol=args.atol,
        pass_rate_threshold=args.pass_rate,
    )
    ok = bool(good_report["ok"] and not bad_report["ok"])
    report = {
        "mode": "self_test",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rtol": args.rtol,
        "atol": args.atol,
        "pass_rate_threshold": args.pass_rate,
        "good": good_report,
        "bad": bad_report,
        "ok": ok,
    }
    return report, (EXIT_PASS if ok else EXIT_FAIL)


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    sys.path.insert(0, str(_repo_root()))

    if args.self_test:
        return _run_self_test(args)
    if args.self_consistency:
        return _run_self_consistency(args)
    return _run_golden_present(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--golden-root",
        default=str(DEFAULT_GOLDEN_ROOT),
        help=(
            "Golden root (collect_w8a8_prefill_golden.py output). Default "
            f"{DEFAULT_GOLDEN_ROOT}. If absent and no --self-consistency / "
            "--self-test, exits with code 3 (golden not present)."
        ),
    )
    parser.add_argument(
        "--holder-dir",
        default=None,
        help=(
            "Dir containing prefill_t{N:03d}_active_hidden.pt produced by "
            "_stage_prefill_hidden_only (or --run-holder). Required for "
            "golden-present mode unless --run-holder is set."
        ),
    )
    parser.add_argument(
        "--ckpt-dir",
        default=DEFAULT_CKPT,
        help=(
            "W8A8 checkpoint dir (for tail LM-head weights in golden-present "
            "mode and reference weights / arange(T) embedding in "
            f"--self-consistency). Default {DEFAULT_CKPT}."
        ),
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        action="append",
        default=None,
        help=(
            "Prompt length (repeatable). Default 128 (the holder A/B target; "
            "the live holder gate rejects T>128). Matched to a golden case "
            "by the manifest seq_len field."
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help=(
            "Explicit golden case name (repeatable, e.g. prefill_128k). "
            "Overrides --seq-len case matching."
        ),
    )
    parser.add_argument("--tp-world-size", type=int, default=8)
    parser.add_argument("--rtol", type=float, default=RTOL_DEFAULT)
    parser.add_argument("--atol", type=float, default=ATOL_DEFAULT)
    parser.add_argument(
        "--pass-rate",
        type=float,
        default=PASS_RATE_DEFAULT,
        help=(
            "pass_rate gate (torch.isclose(cand, exp, rtol, atol).mean()). "
            "Default 0.997 (matches prefill_precision_suite.py)."
        ),
    )
    parser.add_argument(
        "--logits-chunk-size",
        type=int,
        default=LOGITS_CHUNK_DEFAULT,
        help="LM-head matmul chunk size (matches final_logits_from_vllm).",
    )
    parser.add_argument(
        "--self-consistency",
        action="store_true",
        help=(
            "No-golden mode: run _torch_reference_prefill (dense+shared, NOT "
            "routed/attention) on the same input; gate on reference "
            "bit-determinism; report holder-vs-reference as diagnostic only."
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "Synthetic round-trip: verify pass_rate machinery and the "
            ">= --pass-rate gate with two random tensors + tiny noise. "
            "Card-free, no weights, no golden."
        ),
    )
    parser.add_argument(
        "--input-hidden",
        default=None,
        help=(
            "Path to a .pt file with the input hidden [T, HIDDEN] BF16 fed "
            "to the holder (for --self-consistency). If omitted, loads "
            "arange(T) token embeddings from --ckpt-dir, else synthetic."
        ),
    )
    parser.add_argument(
        "--run-holder",
        action="store_true",
        help=(
            "Produce holder output on-device first by invoking "
            "_stage_prefill_hidden_only (needs 8 NPU + PTO_ISA_ROOT + IPC "
            "pool), then compare. Writes prefill_t{N:03d}_active_hidden.pt "
            "into --holder-dir."
        ),
    )
    parser.add_argument(
        "--device",
        default="0,1,2,3,4,5,6,7",
        help="Comma list of 8 NPU device ids for --run-holder.",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=32,
        help="Scheduler KV blocks for --run-holder (matches staging harness).",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Write the JSON report to this path.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    report, exit_code = run(args)
    payload = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(payload, encoding="utf-8")
    print(payload)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
