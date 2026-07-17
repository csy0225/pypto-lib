#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Focused CPU reference for the canonical Step3p5 MTP3 ctx=1 chain.

This intentionally loads only checkpoint layers 45/46/47. It mirrors the
PyPTO TP=8 cast boundaries, including:

* vLLM's position-0 token-embedding mask;
* per-rank BF16 attention-o and dense-down partials before FP32 TP sums;
* zero-centred RMSNorm (``gamma + 1``);
* per-layer vocab-sharded shared heads and greedy handoff tokens.

At ctx=1 the attention softmax has a single valid key, so Q/K do not affect
the end-to-end result. Their projection/norm precision remains covered by the
existing 279-check vLLM detailed-dump test; this script validates the actual
canonical main-hidden -> MTP45 -> MTP46 -> MTP47 device outputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--previous-hidden", required=True)
    parser.add_argument("--device-dump", required=True)
    parser.add_argument("--tp-world-size", type=int, default=8)
    parser.add_argument("--pass-rate", type=float, default=0.97)
    parser.add_argument("--hidden-rtol", type=float, default=8e-2)
    parser.add_argument("--hidden-atol", type=float, default=2e-1)
    parser.add_argument("--logits-rtol", type=float, default=8e-2)
    parser.add_argument("--logits-atol", type=float, default=8e-2)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def _zc_rmsnorm(
    value: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    value_f = value.float()
    variance = value_f.square().mean(dim=-1, keepdim=True)
    return (
        value_f
        * torch.rsqrt(variance + eps)
        * (gamma.float() + 1.0)
    ).bfloat16()


def _chunked_mm(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    k_chunk: int,
) -> torch.Tensor:
    """Mirror a BF16-input, FP32-accumulated chunked matmul."""
    if left.shape[-1] != right.shape[0]:
        raise ValueError(
            f"matmul mismatch: left={tuple(left.shape)} "
            f"right={tuple(right.shape)}"
        )
    result = left[:, :k_chunk].float() @ right[:k_chunk].float()
    for start in range(k_chunk, left.shape[-1], k_chunk):
        result.add_(
            left[:, start : start + k_chunk].float()
            @ right[start : start + k_chunk].float()
        )
    return result


def _compare(
    name: str,
    candidate: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    pass_rate: float,
) -> dict[str, Any]:
    if tuple(candidate.shape) != tuple(expected.shape):
        return {
            "name": name,
            "ok": False,
            "shape": list(candidate.shape),
            "expected_shape": list(expected.shape),
            "pass_rate": 0.0,
        }
    candidate_f = candidate.float()
    expected_f = expected.float()
    finite = bool(torch.isfinite(candidate_f).all().item())
    diff = (candidate_f - expected_f).abs()
    close = torch.isclose(
        candidate_f,
        expected_f,
        rtol=rtol,
        atol=atol,
    )
    actual_rate = float(close.float().mean().item())
    return {
        "name": name,
        "ok": finite and actual_rate >= pass_rate,
        "finite": finite,
        "pass_rate": actual_rate,
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "candidate_max_abs": float(candidate_f.abs().max().item()),
        "expected_max_abs": float(expected_f.abs().max().item()),
    }


def _load_main_hidden(path: str, tp: int, hidden: int) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(value.shape) == (hidden,):
        value = value.unsqueeze(0).repeat(tp, 1)
    if tuple(value.shape) != (tp, hidden):
        raise ValueError(
            f"expected previous hidden {(tp, hidden)} or {(hidden,)}, "
            f"got {tuple(value.shape)}"
        )
    spread = (
        value.float() - value[0:1].float()
    ).abs().max().item()
    if spread != 0.0:
        raise ValueError(
            f"canonical previous hidden is not replicated across TP: {spread}"
        )
    return value[0:1].to(torch.bfloat16)


def _layer_reference(
    previous_hidden: torch.Tensor,
    *,
    cache,
    keys: dict[str, str],
    tp: int,
    eps: float,
    head_dim: int,
    num_heads_local: int,
    intermediate_local: int,
    vocab_local: int,
    input_k_chunk: int,
    kv_k_chunk: int,
    out_k_chunk: int,
    dense_k_chunk: int,
    dense_out_chunk: int,
    lm_head_k_chunk: int,
) -> tuple[torch.Tensor, list[torch.Tensor], int]:
    """Run one position-0 MTP layer with TP/cast boundaries."""
    from models.step3p5.weight_loader import (  # noqa: PLC0415
        _slice_g_proj,
        _slice_lm_head,
        _slice_mlp_col,
        _slice_mlp_row,
        _slice_o_proj,
        _slice_kv_proj,
    )

    # vLLM patch_deepseek_mtp.py: positions == 0 -> zero embedding.
    embed = torch.zeros_like(previous_hidden)
    enorm = _zc_rmsnorm(embed, cache.get(keys["enorm"]), eps)
    hnorm = _zc_rmsnorm(
        previous_hidden,
        cache.get(keys["hnorm"]),
        eps,
    )
    eh_input = torch.cat([enorm, hnorm], dim=-1)
    eh_weight = cache.get(keys["eh_proj"]).to(torch.bfloat16)
    mtp_in = _chunked_mm(
        eh_input,
        eh_weight.transpose(0, 1),
        k_chunk=dense_k_chunk,
    ).bfloat16()

    input_norm = _zc_rmsnorm(
        mtp_in,
        cache.get(keys["input_rms"]),
        eps,
    )
    v_full = cache.get(keys["v_proj"])
    o_full = cache.get(keys["o_proj"])
    g_full = cache.get(keys["g_proj"])
    partial_attention: list[torch.Tensor] = []
    for tp_rank in range(tp):
        v_weight = _slice_kv_proj(v_full, tp_rank, 1)
        v_local = _chunked_mm(
            input_norm,
            v_weight,
            k_chunk=kv_k_chunk,
        ).bfloat16()
        gate_weight = _slice_g_proj(
            g_full,
            tp_rank,
            num_heads_local,
            pad_to=16,
        )
        gate_logits = _chunked_mm(
            input_norm,
            gate_weight,
            k_chunk=input_k_chunk,
        )
        gate = torch.sigmoid(gate_logits).bfloat16()[
            :, :num_heads_local
        ]
        repeated_v = v_local.repeat(1, num_heads_local)
        expanded_gate = gate.repeat_interleave(head_dim, dim=-1)
        gated_attention = (
            repeated_v.float() * expanded_gate.float()
        ).bfloat16()
        o_weight = _slice_o_proj(
            o_full,
            tp_rank,
            num_heads_local,
        )
        partial_attention.append(
            _chunked_mm(
                gated_attention,
                o_weight,
                k_chunk=out_k_chunk,
            ).bfloat16()
        )
    attention_reduced = torch.stack(
        [partial.float() for partial in partial_attention],
        dim=0,
    ).sum(dim=0).bfloat16()
    resid1 = (
        mtp_in.float() + attention_reduced.float()
    ).bfloat16()

    post_norm = _zc_rmsnorm(
        resid1,
        cache.get(keys["post_attn_rms"]),
        eps,
    )
    gate_full_mlp = cache.get(keys["gate_proj"])
    up_full_mlp = cache.get(keys["up_proj"])
    down_full_mlp = cache.get(keys["down_proj"])
    partial_dense: list[torch.Tensor] = []
    for tp_rank in range(tp):
        gate_weight = _slice_mlp_col(
            gate_full_mlp,
            tp_rank,
            intermediate_local,
        )
        up_weight = _slice_mlp_col(
            up_full_mlp,
            tp_rank,
            intermediate_local,
        )
        down_weight = _slice_mlp_row(
            down_full_mlp,
            tp_rank,
            intermediate_local,
        )
        gate_value = _chunked_mm(
            post_norm,
            gate_weight,
            k_chunk=dense_k_chunk,
        )
        up_value = _chunked_mm(
            post_norm,
            up_weight,
            k_chunk=dense_k_chunk,
        )
        activated = (
            gate_value
            * torch.sigmoid(gate_value)
            * up_value
        ).bfloat16()
        partial_dense.append(
            _chunked_mm(
                activated,
                down_weight,
                k_chunk=dense_out_chunk,
            ).bfloat16()
        )
    dense_reduced = torch.stack(
        [partial.float() for partial in partial_dense],
        dim=0,
    ).sum(dim=0).bfloat16()
    hidden_out = (
        resid1.float() + dense_reduced.float()
    ).bfloat16()

    head_norm = _zc_rmsnorm(
        hidden_out,
        cache.get(keys["shared_head_norm"]),
        eps,
    )
    shared_head_full = cache.get(keys["shared_head_output"])
    logits_shards: list[torch.Tensor] = []
    for tp_rank in range(tp):
        head_weight = _slice_lm_head(
            shared_head_full,
            tp_rank,
            vocab_local,
        )
        logits_shards.append(
            _chunked_mm(
                head_norm,
                head_weight.transpose(0, 1),
                k_chunk=lm_head_k_chunk,
            )
        )
    token_id = int(
        torch.cat(logits_shards, dim=-1).argmax(dim=-1).item()
    )
    return hidden_out, logits_shards, token_id


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.tp_world_size != 8:
        raise ValueError("the canonical Step3p5 MTP program requires TP=8")
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")

    from models.step3p5.config import (  # noqa: PLC0415
        EPS,
        HEAD_DIM,
        HIDDEN,
        INPUT_PROJ_K_CHUNK,
        INTERMEDIATE_LOCAL,
        K_CHUNK,
        KV_PROJ_K_CHUNK_LOCAL,
        LM_HEAD_K_CHUNK,
        MLP_OUT_CHUNK,
        NUM_HEADS_SWA_LOCAL,
        NUM_HIDDEN_LAYERS,
        OUT_PROJ_K_CHUNK,
        VOCAB_LOCAL,
    )
    from models.step3p5.weight_loader import (  # noqa: PLC0415
        _ShardCache,
        _hf_mtp_keys,
        _read_index,
    )

    dump_dir = Path(args.device_dump)
    device_hidden = torch.load(
        dump_dir / "mtp3_hidden.pt",
        map_location="cpu",
        weights_only=True,
    )
    device_logits = torch.load(
        dump_dir / "mtp3_logits_shards.pt",
        map_location="cpu",
        weights_only=True,
    )
    device_tokens = torch.load(
        dump_dir / "mtp3_draft_token_ids.pt",
        map_location="cpu",
        weights_only=True,
    )
    previous_hidden = _load_main_hidden(
        args.previous_hidden,
        args.tp_world_size,
        HIDDEN,
    )

    if tuple(device_hidden.shape) != (
        args.tp_world_size,
        3,
        16,
        HIDDEN,
    ):
        raise ValueError(
            f"unexpected device hidden shape: {tuple(device_hidden.shape)}"
        )
    if tuple(device_logits.shape) != (
        args.tp_world_size,
        3,
        16,
        VOCAB_LOCAL,
    ):
        raise ValueError(
            f"unexpected device logits shape: {tuple(device_logits.shape)}"
        )

    reports: list[dict[str, Any]] = []
    reference_tokens: list[int] = []
    weight_map = _read_index(args.ckpt)
    with _ShardCache(args.ckpt, weight_map) as cache:
        for mtp_idx in range(3):
            global_layer = NUM_HIDDEN_LAYERS + mtp_idx
            keys = _hf_mtp_keys(global_layer)
            previous_hidden, logits_shards, token_id = _layer_reference(
                previous_hidden,
                cache=cache,
                keys=keys,
                tp=args.tp_world_size,
                eps=EPS,
                head_dim=HEAD_DIM,
                num_heads_local=NUM_HEADS_SWA_LOCAL,
                intermediate_local=INTERMEDIATE_LOCAL,
                vocab_local=VOCAB_LOCAL,
                input_k_chunk=INPUT_PROJ_K_CHUNK,
                kv_k_chunk=KV_PROJ_K_CHUNK_LOCAL,
                out_k_chunk=OUT_PROJ_K_CHUNK,
                dense_k_chunk=K_CHUNK,
                dense_out_chunk=MLP_OUT_CHUNK,
                lm_head_k_chunk=LM_HEAD_K_CHUNK,
            )
            reference_tokens.append(token_id)
            reference_logits = torch.stack(
                [shard[0] for shard in logits_shards],
                dim=0,
            )
            logits_report = _compare(
                f"mtp{global_layer}.logits",
                reference_logits,
                device_logits[:, mtp_idx, 0, :],
                rtol=args.logits_rtol,
                atol=args.logits_atol,
                pass_rate=args.pass_rate,
            )
            reports.append(logits_report)
            rank_tokens = device_tokens[:, mtp_idx, 0]
            device_token = int(rank_tokens[0])
            token_ok = bool(
                torch.equal(
                    rank_tokens,
                    rank_tokens[0].expand_as(rank_tokens),
                )
                and device_token == token_id
            )
            reports.append(
                {
                    "name": f"mtp{global_layer}.token",
                    "ok": token_ok,
                    "reference": token_id,
                    "device": rank_tokens.tolist(),
                }
            )
            hidden_report = _compare(
                f"mtp{global_layer}.hidden",
                previous_hidden[0],
                device_hidden[0, mtp_idx, 0],
                rtol=args.hidden_rtol,
                atol=args.hidden_atol,
                pass_rate=args.pass_rate,
            )
            reports.append(hidden_report)
            print(
                f"[reference] MTP{global_layer} token={token_id} "
                f"logits_pass_rate={logits_report['pass_rate']:.9f} "
                f"hidden_pass_rate={hidden_report['pass_rate']:.9f}",
                flush=True,
            )
            hidden_tp_spread = (
                device_hidden[:, mtp_idx, 0, :].float()
                - device_hidden[0:1, mtp_idx, 0, :].float()
            ).abs().max().item()
            reports.append(
                {
                    "name": f"mtp{global_layer}.hidden_tp_replication",
                    "ok": hidden_tp_spread == 0.0,
                    "max_abs_spread": hidden_tp_spread,
                }
            )

    report = {
        "ok": all(item["ok"] for item in reports),
        "checkpoint": args.ckpt,
        "previous_hidden": args.previous_hidden,
        "device_dump": args.device_dump,
        "position": 0,
        "embedding_masked": True,
        "reference_tokens": reference_tokens,
        "device_tokens_row0": device_tokens[0, :, 0].tolist(),
        "checks": reports,
        "worst_numeric_pass_rate": min(
            (
                item["pass_rate"]
                for item in reports
                if "pass_rate" in item
            ),
            default=0.0,
        ),
    }
    if args.out:
        Path(args.out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return report


def main() -> int:
    args = _parse_args()
    sys.path.insert(0, str(_repo_root()))
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
