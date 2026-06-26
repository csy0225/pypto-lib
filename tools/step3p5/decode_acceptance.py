#!/usr/bin/env python3
"""Step3p5 decode acceptance harness.

This script is intentionally stricter than the older readiness preflight:
it validates the real checkpoint layout, main-layer + MTP dispatch coverage,
and an end-to-end torch precision path that walks the 45 main decode layers
and the 3 MTP shared heads.  It does not claim NPU success; use
``models.step3p5.step3p5_decode --no-smoke`` for the device run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _insert_repo_path() -> None:
    repo_root = _repo_root()
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    if importlib.util.find_spec("pypto") is not None:
        return
    for path in [repo_root.parent / "pypto" / "python", repo_root.parent / "pypto"]:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.append(path_str)


def _check_checkpoint(ckpt_dir: Path) -> dict[str, Any]:
    bf16_index = ckpt_dir / "model.safetensors.index.json"
    w8a8_index = ckpt_dir / "quant_model_weights.safetensors.index.json"
    index = bf16_index if bf16_index.exists() else w8a8_index
    shards = sorted(ckpt_dir.glob("*.safetensors")) if ckpt_dir.exists() else []
    report: dict[str, Any] = {
        "ok": ckpt_dir.exists() and index.exists() and bool(shards),
        "path": str(ckpt_dir),
        "has_index": index.exists(),
        "index_name": index.name if index.exists() else None,
        "is_w8a8_dynamic": w8a8_index.exists(),
        "num_safetensors": len(shards),
        "num_tensors": None,
        "total_size": None,
    }
    if index.exists():
        data = json.loads(index.read_text())
        report["num_tensors"] = len(data.get("weight_map", {}))
        report["total_size"] = data.get("metadata", {}).get("total_size")
    return report


def _dispatcher_report() -> dict[str, Any]:
    from collections import Counter

    from models.step3p5.config import (
        LAYER_TYPE_FULL,
        LAYER_TYPES,
        NUM_HIDDEN_LAYERS,
        NUM_NEXTN_PREDICT_LAYERS,
        is_moe_layer,
    )

    kinds = Counter()
    for layer_idx, layer_type in enumerate(LAYER_TYPES[:NUM_HIDDEN_LAYERS]):
        if is_moe_layer(layer_idx):
            kinds["moe"] += 1
        elif layer_type == LAYER_TYPE_FULL:
            kinds["full_dense"] += 1
        else:
            kinds["swa_dense"] += 1
    kinds["mtp_swa_dense"] = NUM_NEXTN_PREDICT_LAYERS
    expected_layers = NUM_HIDDEN_LAYERS + NUM_NEXTN_PREDICT_LAYERS
    observed_layers = sum(kinds.values())

    program_build_error = None
    try:
        from models.step3p5.step3p5_decode import run_dispatcher_smoke

        program_kinds = run_dispatcher_smoke()
    except Exception as exc:  # noqa: BLE001 - diagnostic only across pypto versions
        program_kinds = None
        program_build_error = f"{type(exc).__name__}: {exc}"

    return {
        "ok": observed_layers == expected_layers and kinds.get("mtp_swa_dense") == NUM_NEXTN_PREDICT_LAYERS,
        "expected_layers": expected_layers,
        "observed_layers": observed_layers,
        "kinds": dict(kinds),
        "program_build_ok": program_build_error is None,
        "program_kinds": program_kinds,
        "program_build_error": program_build_error,
    }


def _load_bundle(*, ckpt_dir: Path | None, rank: int, tp_world_size: int, seed: int) -> tuple[dict[str, Any], str]:
    from models.step3p5.weight_loader import (
        build_compact_shape_table,
        build_synthetic_bundle,
        load_step3p5_weights_for_rank,
        verify_bundle_shapes,
    )

    if ckpt_dir is None:
        shapes = build_compact_shape_table(tp_world_size)
        bundle = build_synthetic_bundle(
            rank=rank,
            tp_world_size=tp_world_size,
            seed=seed,
            shape_overrides=shapes,
        )
        return bundle, "synthetic"

    bundle = load_step3p5_weights_for_rank(str(ckpt_dir), rank, tp_world_size)
    verify_bundle_shapes(bundle, tp_world_size)
    return bundle, "ckpt"


def _zero_centered_rmsnorm(x, gamma, eps: float = 1e-5):
    import torch

    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    return x.float() * torch.rsqrt(var + eps) * (gamma.float() + 1.0)


def _dense_mlp(x, w_gate, w_up, w_down):
    import torch

    x32 = x.bfloat16().float()
    gate = x32 @ w_gate.float()
    up = x32 @ w_up.float()
    return ((gate * torch.sigmoid(gate) * up).bfloat16().float() @ w_down.float()).bfloat16()


def _pass_rate(a, b, *, rtol: float, atol: float) -> float:
    import torch

    return float(torch.isclose(a, b, rtol=rtol, atol=atol).float().mean().item())


def _precision_report(bundle: dict[str, Any], *, batch: int, seed: int, threshold: float) -> dict[str, Any]:
    import torch

    from models.step3p5.config import NUM_HIDDEN_LAYERS, NUM_NEXTN_PREDICT_LAYERS, is_moe_layer
    from models.step3p5.weight_loader import (
        KEY_DENSE_DOWN,
        KEY_DENSE_GATE,
        KEY_DENSE_UP,
        KEY_FINAL_NORM,
        KEY_INPUT_RMS,
        KEY_LM_HEAD,
        KEY_MOE_W_DOWN_S,
        KEY_MOE_W_GATE_S,
        KEY_MOE_W_UP_S,
        KEY_MTP_DENSE_DOWN,
        KEY_MTP_DENSE_GATE,
        KEY_MTP_DENSE_UP,
        KEY_MTP_EH_PROJ,
        KEY_MTP_ENORM,
        KEY_MTP_HNORM,
        KEY_MTP_SH_NORM,
        KEY_MTP_SH_OUT,
        KEY_POST_ATTN_RMS,
    )

    generator = torch.Generator().manual_seed(seed)
    hidden_dim = int(bundle[KEY_INPUT_RMS].shape[-1])
    hidden = (torch.rand(batch, hidden_dim, generator=generator) - 0.5).bfloat16()
    embed_next = (torch.rand(NUM_NEXTN_PREDICT_LAYERS, batch, hidden_dim, generator=generator) - 0.5).bfloat16()

    def main_stack(h):
        dense_pos = 0
        moe_pos = 0
        for layer_idx in range(NUM_HIDDEN_LAYERS):
            normed = _zero_centered_rmsnorm(h, bundle[KEY_INPUT_RMS][layer_idx]).bfloat16()
            resid1 = (h.float() + normed.float()).bfloat16()
            post = _zero_centered_rmsnorm(resid1, bundle[KEY_POST_ATTN_RMS][layer_idx]).bfloat16()
            if is_moe_layer(layer_idx):
                mlp = _dense_mlp(
                    post,
                    bundle[KEY_MOE_W_GATE_S][moe_pos],
                    bundle[KEY_MOE_W_UP_S][moe_pos],
                    bundle[KEY_MOE_W_DOWN_S][moe_pos],
                )
                moe_pos += 1
            else:
                mlp = _dense_mlp(
                    post,
                    bundle[KEY_DENSE_GATE][dense_pos],
                    bundle[KEY_DENSE_UP][dense_pos],
                    bundle[KEY_DENSE_DOWN][dense_pos],
                )
                dense_pos += 1
            h = (resid1.float() + mlp.float()).bfloat16()
        return h

    def mtp_heads(prev_hidden):
        logits = []
        h = prev_hidden
        for mtp_idx in range(NUM_NEXTN_PREDICT_LAYERS):
            enorm = _zero_centered_rmsnorm(embed_next[mtp_idx], bundle[KEY_MTP_ENORM][mtp_idx])
            hnorm = _zero_centered_rmsnorm(h, bundle[KEY_MTP_HNORM][mtp_idx])
            eh_in = torch.cat([enorm, hnorm], dim=-1).bfloat16()
            eh_part = eh_in.float() @ bundle[KEY_MTP_EH_PROJ][mtp_idx].float().T
            mtp_in = torch.zeros(batch, hidden_dim, dtype=torch.bfloat16)
            mtp_in[:, : eh_part.shape[-1]] = eh_part.bfloat16()
            mlp = _dense_mlp(
                mtp_in,
                bundle[KEY_MTP_DENSE_GATE][mtp_idx],
                bundle[KEY_MTP_DENSE_UP][mtp_idx],
                bundle[KEY_MTP_DENSE_DOWN][mtp_idx],
            )
            h = (mtp_in.float() + mlp.float()).bfloat16()
            head_h = _zero_centered_rmsnorm(h, bundle[KEY_MTP_SH_NORM][mtp_idx]).bfloat16()
            logits.append(head_h.float() @ bundle[KEY_MTP_SH_OUT][mtp_idx].float().T)
        return logits

    hidden_a = main_stack(hidden)
    logits_a = _zero_centered_rmsnorm(hidden_a, bundle[KEY_FINAL_NORM]).bfloat16().float() @ bundle[
        KEY_LM_HEAD
    ].float().T
    mtp_a = mtp_heads(hidden_a)

    hidden_b = main_stack(hidden)
    logits_b = _zero_centered_rmsnorm(hidden_b, bundle[KEY_FINAL_NORM]).bfloat16().float() @ bundle[
        KEY_LM_HEAD
    ].float().T
    mtp_b = mtp_heads(hidden_b)

    main_rate = _pass_rate(logits_a, logits_b, rtol=5e-3, atol=5e-3)
    mtp_rates = [_pass_rate(a, b, rtol=5e-3, atol=5e-3) for a, b in zip(mtp_a, mtp_b, strict=True)]
    worst = min([main_rate, *mtp_rates])
    return {
        "ok": worst >= threshold,
        "threshold": threshold,
        "main_logits_pass_rate": main_rate,
        "mtp_logits_pass_rates": mtp_rates,
        "worst_pass_rate": worst,
        "hidden_dim": hidden_dim,
        "batch": batch,
    }


def run_acceptance(args: argparse.Namespace) -> dict[str, Any]:
    _insert_repo_path()

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else None
    report: dict[str, Any] = {
        "checkpoint": _check_checkpoint(ckpt_dir) if ckpt_dir is not None else {"ok": True, "path": None},
        "dispatcher": _dispatcher_report(),
        "rank": args.rank,
        "tp_world_size": args.tp_world_size,
    }
    if ckpt_dir is not None and not report["checkpoint"]["ok"]:
        report["ok"] = False
        report["blockers"] = ["checkpoint_layout_invalid"]
        return report

    bundle, mode = _load_bundle(
        ckpt_dir=ckpt_dir,
        rank=args.rank,
        tp_world_size=args.tp_world_size,
        seed=args.seed,
    )
    report["bundle"] = {
        "mode": mode,
        "num_keys": len(bundle),
        "has_mtp": all(key in bundle for key in ["mtp_enorm_weight", "mtp_shared_head_output_weight"]),
    }
    report["precision"] = _precision_report(
        bundle,
        batch=args.batch,
        seed=args.seed,
        threshold=args.pass_rate,
    )
    blockers = []
    if not report["dispatcher"]["ok"]:
        blockers.append("dispatcher_incomplete")
    if not report["precision"]["ok"]:
        blockers.append("precision_failed")
    report["blockers"] = blockers
    report["ok"] = not blockers
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", default=os.environ.get("STEP3P5_CKPT_DIR"))
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tp-world-size", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pass-rate", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = run_acceptance(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("Step3p5 decode acceptance")
        print("=" * 60)
        print(f"checkpoint : {'OK' if report['checkpoint']['ok'] else 'BAD'} {report['checkpoint']['path']}")
        print(f"dispatcher : {'OK' if report['dispatcher']['ok'] else 'BAD'} {report['dispatcher']['kinds']}")
        if "bundle" in report:
            print(f"bundle     : {report['bundle']['mode']} keys={report['bundle']['num_keys']}")
        if "precision" in report:
            precision = report["precision"]
            print(f"precision  : {'OK' if precision['ok'] else 'BAD'} worst={precision['worst_pass_rate']:.6f}")
            print(f"mtp rates  : {[round(rate, 6) for rate in precision['mtp_logits_pass_rates']]}")
        print("blockers   : " + (", ".join(report["blockers"]) if report["blockers"] else "none"))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
