#!/usr/bin/env python3
"""Materialize/describe Step3p5 PyPTO per-rank weight bundles.

Phase-20 production backend will need a stable boundary between a vLLM-loaded
Step3p5 model and PyPTO runner inputs.  This tool defines that boundary in
terms of the existing ``weight_loader`` bundle keys/shapes and can export a
manifest (cheap) or actual per-rank ``.pt`` bundles (expensive, opt-in).

Current implementation supports checkpoint-backed translation by delegating to
``load_step3p5_weights_for_rank``.  vLLM ``nn.Module`` in-memory translation is
kept as the next step; the manifest emitted here is the contract it must match.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _tensor_meta(tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "numel": int(tensor.numel()),
    }


def _expected_meta(tp_world_size: int) -> dict[str, dict[str, Any]]:
    from models.step3p5.weight_loader import (  # noqa: PLC0415
        KEY_MOE_GATE_W,
        KEY_MOE_ROUTER_BIAS,
        expected_shapes,
    )

    fp32_keys = {KEY_MOE_GATE_W, KEY_MOE_ROUTER_BIAS}
    out: dict[str, dict[str, Any]] = {}
    for key, shape in expected_shapes(tp_world_size).items():
        out[key] = {
            "shape": list(shape),
            "dtype": "float32" if key in fp32_keys else "bfloat16",
            "numel": int(__import__("math").prod(shape)),
        }
    return out


def _bundle_meta(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {key: _tensor_meta(value) for key, value in sorted(bundle.items())}


def translate_rank_from_checkpoint(
    ckpt_dir: str,
    rank: int,
    tp_world_size: int,
    *,
    verify: bool = True,
) -> dict[str, Any]:
    from models.step3p5.weight_loader import (  # noqa: PLC0415
        load_step3p5_weights_for_rank,
        verify_bundle_shapes,
    )

    bundle = load_step3p5_weights_for_rank(ckpt_dir, rank, tp_world_size)
    if verify:
        verify_bundle_shapes(bundle, tp_world_size)
    return bundle


def build_manifest(
    *,
    source: str,
    ckpt_dir: str | None,
    ranks: list[int],
    tp_world_size: int,
    metadata_only: bool,
    rank_metadata: dict[int, dict[str, dict[str, Any]]],
    saved_files: dict[int, str],
) -> dict[str, Any]:
    return {
        "source": source,
        "ckpt_dir": ckpt_dir,
        "tp_world_size": tp_world_size,
        "ranks": ranks,
        "metadata_only": metadata_only,
        "bundle_contract": "models.step3p5.weight_loader per-rank bundle",
        "rank_metadata": {str(rank): rank_metadata[rank] for rank in ranks},
        "saved_files": {str(rank): saved_files[rank] for rank in sorted(saved_files)},
        "notes": (
            "metadata-only manifests are cheap and do not materialize checkpoint tensors; "
            "use --save-bundles explicitly when runner integration needs .pt inputs."
        ),
    }


def _expected_vllm_param_meta(tp_world_size: int) -> dict[str, dict[str, Any]]:
    """Expected local-rank vLLM Step3p5 parameter metadata.

    This describes the live vLLM module after tensor/expert parallel sharding,
    not the PyPTO bundle orientation.  It is the bridge contract for future
    ``nn.Module -> PyPTO bundle`` translation.
    """
    from models.step3p5.config import (  # noqa: PLC0415
        DENSE_LAYER_INDICES,
        HEAD_DIM,
        HIDDEN,
        INTERMEDIATE,
        LAYER_TYPE_FULL,
        LAYER_TYPES,
        MOE_INTERMEDIATE,
        MOE_LAYER_INDICES,
        MOE_NUM_EXPERTS,
        NUM_HEADS_FULL,
        NUM_HEADS_SWA,
        NUM_HIDDEN_LAYERS,
        NUM_KV_HEADS,
        SHARE_EXPERT_DIM,
        VOCAB,
        is_full_attention,
    )

    q_full_local = (NUM_HEADS_FULL // tp_world_size) * HEAD_DIM
    q_swa_local = (NUM_HEADS_SWA // tp_world_size) * HEAD_DIM
    kv_local = (NUM_KV_HEADS // tp_world_size) * HEAD_DIM
    inter_local = INTERMEDIATE // tp_world_size
    share_local = SHARE_EXPERT_DIM // tp_world_size
    vocab_local = VOCAB // tp_world_size
    experts_local = MOE_NUM_EXPERTS // tp_world_size

    meta: dict[str, dict[str, Any]] = {
        "model.embed_tokens.weight": {"shape": [vocab_local, HIDDEN], "dtype": "bfloat16"},
        "model.norm.weight": {"shape": [HIDDEN], "dtype": "bfloat16"},
        "lm_head.weight": {"shape": [vocab_local, HIDDEN], "dtype": "bfloat16"},
    }
    dense_layers = set(DENSE_LAYER_INDICES)
    moe_layers = set(MOE_LAYER_INDICES)
    for li in range(NUM_HIDDEN_LAYERS):
        q_local = q_full_local if is_full_attention(li) else q_swa_local
        meta[f"model.layers.{li}.self_attn.qkv_proj.weight"] = {
            "shape": [q_local + 2 * kv_local, HIDDEN], "dtype": "bfloat16",
        }
        meta[f"model.layers.{li}.self_attn.o_proj.weight"] = {
            "shape": [HIDDEN, q_local], "dtype": "bfloat16",
        }
        meta[f"model.layers.{li}.self_attn.q_norm.weight"] = {"shape": [HEAD_DIM], "dtype": "bfloat16"}
        meta[f"model.layers.{li}.self_attn.k_norm.weight"] = {"shape": [HEAD_DIM], "dtype": "bfloat16"}
        meta[f"model.layers.{li}.self_attn.g_proj.weight"] = {
            "shape": [q_local // HEAD_DIM, HIDDEN], "dtype": "bfloat16",
        }
        meta[f"model.layers.{li}.input_layernorm.weight"] = {"shape": [HIDDEN], "dtype": "bfloat16"}
        meta[f"model.layers.{li}.post_attention_layernorm.weight"] = {"shape": [HIDDEN], "dtype": "bfloat16"}
        if li in dense_layers:
            meta[f"model.layers.{li}.mlp.gate_up_proj.weight"] = {
                "shape": [2 * inter_local, HIDDEN], "dtype": "bfloat16",
            }
            meta[f"model.layers.{li}.mlp.down_proj.weight"] = {
                "shape": [HIDDEN, inter_local], "dtype": "bfloat16",
            }
        elif li in moe_layers:
            meta[f"model.layers.{li}.moe.router_bias"] = {"shape": [MOE_NUM_EXPERTS], "dtype": "bfloat16"}
            meta[f"model.layers.{li}.moe.gate.weight"] = {"shape": [MOE_NUM_EXPERTS, HIDDEN], "dtype": "float32"}
            meta[f"model.layers.{li}.moe.share_expert.gate_up_proj.weight"] = {
                "shape": [2 * share_local, HIDDEN], "dtype": "bfloat16",
            }
            meta[f"model.layers.{li}.moe.share_expert.down_proj.weight"] = {
                "shape": [HIDDEN, share_local], "dtype": "bfloat16",
            }
            meta[f"model.layers.{li}.moe.experts.w13_weight"] = {
                "shape": [experts_local, HIDDEN, 2 * MOE_INTERMEDIATE], "dtype": "int8",
            }
            meta[f"model.layers.{li}.moe.experts.w2_weight"] = {
                "shape": [experts_local, MOE_INTERMEDIATE, HIDDEN], "dtype": "int8",
            }
            meta[f"model.layers.{li}.moe.experts.w13_weight_scale"] = {
                "shape": [experts_local, 2 * MOE_INTERMEDIATE], "dtype": "bfloat16",
            }
            meta[f"model.layers.{li}.moe.experts.w13_weight_offset"] = {
                "shape": [experts_local, 2 * MOE_INTERMEDIATE], "dtype": "bfloat16",
            }
            meta[f"model.layers.{li}.moe.experts.w2_weight_scale"] = {
                "shape": [experts_local, HIDDEN], "dtype": "bfloat16",
            }
            meta[f"model.layers.{li}.moe.experts.w2_weight_offset"] = {
                "shape": [experts_local, HIDDEN], "dtype": "bfloat16",
            }
    return meta


def validate_vllm_param_meta(path: str | Path, tp_world_size: int) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    observed = payload.get("parameters", payload)
    expected = _expected_vllm_param_meta(tp_world_size)
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    mismatches = []
    for name, want in expected.items():
        if name not in observed:
            continue
        got = observed[name]
        if list(got.get("shape", [])) != list(want["shape"]) or got.get("dtype") != want["dtype"]:
            mismatches.append({"name": name, "expected": want, "observed": got})
    return {
        "ok": not missing and not mismatches,
        "num_expected": len(expected),
        "num_observed": len(observed),
        "missing": missing,
        "extra": extra,
        "mismatches": mismatches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", default=None, help="HF/W8A8 checkpoint directory")
    parser.add_argument("--out-dir", required=True, help="Output directory for manifest/bundles")
    parser.add_argument("--rank", type=int, action="append", default=None,
                        help="Rank to translate. Repeatable. Default: rank 0")
    parser.add_argument("--all-ranks", action="store_true")
    parser.add_argument("--tp-world-size", type=int, default=8)
    parser.add_argument("--metadata-only", action="store_true",
                        help="Only emit expected bundle key/shape/dtype metadata; do not load checkpoint tensors.")
    parser.add_argument("--save-bundles", action="store_true",
                        help="Save rankXX_bundle.pt files. Potentially very large; opt-in only.")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--vllm-param-meta", default=None, help="Validate parameter metadata dumped by vllm_monkey_patch.py")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(_repo_root()))

    if not args.metadata_only and not args.ckpt_dir:
        raise SystemExit("--ckpt-dir is required unless --metadata-only is set")
    if args.save_bundles and args.metadata_only:
        raise SystemExit("--save-bundles cannot be combined with --metadata-only")

    ranks = list(range(args.tp_world_size)) if args.all_ranks else (args.rank or [0])
    for rank in ranks:
        if not 0 <= rank < args.tp_world_size:
            raise SystemExit(f"rank {rank} out of range [0,{args.tp_world_size})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vllm_param_meta_report = None
    if args.vllm_param_meta:
        vllm_param_meta_report = validate_vllm_param_meta(args.vllm_param_meta, args.tp_world_size)
        (out_dir / "vllm_param_meta_report.json").write_text(
            json.dumps(vllm_param_meta_report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    rank_metadata: dict[int, dict[str, dict[str, Any]]] = {}
    saved_files: dict[int, str] = {}
    if args.metadata_only:
        expected = _expected_meta(args.tp_world_size)
        for rank in ranks:
            rank_metadata[rank] = expected
    else:
        for rank in ranks:
            bundle = translate_rank_from_checkpoint(
                args.ckpt_dir,
                rank,
                args.tp_world_size,
                verify=not args.no_verify,
            )
            rank_metadata[rank] = _bundle_meta(bundle)
            if args.save_bundles:
                import torch  # noqa: PLC0415

                path = out_dir / f"rank{rank:02d}_bundle.pt"
                torch.save(bundle, path)
                saved_files[rank] = str(path)

    manifest = build_manifest(
        source="checkpoint" if args.ckpt_dir else "expected_shapes",
        ckpt_dir=args.ckpt_dir,
        ranks=ranks,
        tp_world_size=args.tp_world_size,
        metadata_only=args.metadata_only,
        rank_metadata=rank_metadata,
        saved_files=saved_files,
    )
    manifest_path = out_dir / "weight_translate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "manifest": str(manifest_path),
        "ranks": ranks,
        "metadata_only": args.metadata_only,
        "saved_files": saved_files,
        "vllm_param_meta_report": vllm_param_meta_report,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
