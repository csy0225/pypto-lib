#!/usr/bin/env python3
"""Validate Step3p5 HF config against PyPTO compile-time constants.

This is the first Phase-20 production-backend guard: before translating vLLM
weights into PyPTO bundles, assert that the checkpoint/vLLM model config is
compatible with ``models.step3p5.config``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_hf_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    config_path = path / "config.json" if path.is_dir() else path
    if not config_path.exists():
        raise FileNotFoundError(f"missing config.json: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _parse_moe_layers(raw: Any) -> tuple[int, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    return tuple(int(x) for x in raw)


def _check(name: str, pypto_value: Any, hf_value: Any, errors: list[dict[str, Any]]) -> None:
    ok = pypto_value == hf_value
    if not ok:
        errors.append({"name": name, "pypto": pypto_value, "hf": hf_value})


def compare_config(hf_config: dict[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, str(_repo_root()))
    from models.step3p5 import config as c  # noqa: PLC0415

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    checks = {
        "hidden_size": c.HIDDEN,
        "intermediate_size": c.INTERMEDIATE,
        "vocab_size": c.VOCAB,
        "num_hidden_layers": c.NUM_HIDDEN_LAYERS,
        "num_nextn_predict_layers": c.NUM_NEXTN_PREDICT_LAYERS,
        "head_dim": c.HEAD_DIM,
        "num_attention_heads": c.NUM_HEADS_FULL,
        "num_attention_groups": c.NUM_KV_HEADS,
        "sliding_window": c.SLIDING_WINDOW,
        "moe_num_experts": c.MOE_NUM_EXPERTS,
        "moe_top_k": c.MOE_TOP_K,
        "moe_intermediate_size": c.MOE_INTERMEDIATE,
        "share_expert_dim": c.SHARE_EXPERT_DIM,
        "moe_router_scaling_factor": c.MOE_ROUTER_SCALING_FACTOR,
        "moe_router_activation": c.MOE_ROUTER_ACTIVATION,
        "norm_expert_weight": c.NORM_EXPERT_WEIGHT,
        "use_moe_router_bias": c.USE_MOE_ROUTER_BIAS,
        "use_head_wise_attn_gate": c.USE_HEAD_WISE_ATTN_GATE,
    }
    for key, pypto_value in checks.items():
        _check(key, pypto_value, hf_config.get(key), errors)

    # Attention layer-type pattern includes main + MTP layers in both configs.
    _check("layer_types", list(c.LAYER_TYPES), hf_config.get("layer_types"), errors)
    _check("moe_layers_enum", list(c.MOE_LAYER_INDICES), list(_parse_moe_layers(hf_config.get("moe_layers_enum"))), errors)

    rope_theta = hf_config.get("rope_theta")
    if rope_theta is not None:
        _check("rope_theta", list(c.LAYER_ROPE_THETA), [float(x) for x in rope_theta], errors)

    rope_scaling = hf_config.get("rope_scaling") or {}
    if rope_scaling:
        expected_rope_scaling = {
            "rope_type": c.ROPE_SCALING.get("rope_type", c.ROPE_SCALING.get("type")),
            "factor": c.ROPE_SCALING.get("factor"),
            "original_max_position_embeddings": c.ROPE_SCALING.get("original_max_position_embeddings"),
            "low_freq_factor": c.ROPE_SCALING.get("low_freq_factor"),
            "high_freq_factor": c.ROPE_SCALING.get("high_freq_factor"),
        }
        for key, value in expected_rope_scaling.items():
            if value is not None:
                _check(f"rope_scaling.{key}", value, rope_scaling.get(key), errors)

    # SWA heads are stored in Step3p5 config's attention_other_setting on some exports.
    attention_other = hf_config.get("attention_other_setting") or {}
    swa_heads = attention_other.get("sliding_attention", {}).get("num_attention_heads")
    if swa_heads is not None:
        _check("attention_other_setting.sliding_attention.num_attention_heads", c.NUM_HEADS_SWA, int(swa_heads), errors)
    else:
        warnings.append({
            "name": "attention_other_setting.sliding_attention.num_attention_heads",
            "message": "HF config omits SWA head override; PyPTO assumes NUM_HEADS_SWA=96.",
        })

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "hidden": c.HIDDEN,
            "layers": c.NUM_HIDDEN_LAYERS,
            "mtp_layers": c.NUM_NEXTN_PREDICT_LAYERS,
            "moe_layers": len(c.MOE_LAYER_INDICES),
            "full_layers": sum(1 for t in c.LAYER_TYPES[: c.NUM_HIDDEN_LAYERS] if t == c.LAYER_TYPE_FULL),
            "swa_layers": sum(1 for t in c.LAYER_TYPES[: c.NUM_HIDDEN_LAYERS] if t == c.LAYER_TYPE_SWA),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", required=True, help="Checkpoint directory or config.json path")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = compare_config(_load_hf_config(args.ckpt_dir))
    payload = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
    print(payload)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
