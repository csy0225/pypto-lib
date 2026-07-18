# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Single-card precision ST for prefill QKV projection + RoPE.

Wraps ``_build_tp_prefill_qkv_proj_rope_full_program(tp_size=TP_WORLD_SIZE)``
- the production-proven ``@pl.program`` factory - and runs its
``host_orch`` with ``DistributedConfig(device_ids=[0])`` so the rank
loop executes exactly once on device 0.

Each rank produces its per-rank LOCAL shard (NUM_HEADS_FULL_LOCAL=8 Q
heads, KV_HEADS_LOCAL=1 KV head, hidden_q_local=1024, kv_hidden_local=128).
The world-level torch oracle ``_torch_prefill_qkv_proj_rope_full_oracle``
operates on full unsliced weights; we slice its output to rank-0 columns
to match what the kernel produced.

Why this is an ST (not a UT)
----------------------------
The QKV+RoPE block bundles five distinct kernels (input RMSNorm, three
matmuls for Q/K/V, Q/K head-wise RMSNorm, RoPE rotation, and a small
gate logits matmul). The five outputs ``normed_out / q_rot / k_rot /
v_proj / gate_logits`` cover every kernel in the block. No head_gate,
no collective - so single-card numerical correctness IS world-level
numerical correctness for this @pl.program.

Usage::

    cd /data/chensiyu/hw_project/pypto/workspace/pypto-lib
    python -m tests.step3p5.unit.test_prefill_qkv_proj_rope --smoke
    python -m tests.step3p5.unit.test_prefill_qkv_proj_rope -p a2a3 -d 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from models.step3p5.config import (
    HEAD_DIM,
    HIDDEN,
    HIDDEN_Q_FULL,
    HIDDEN_Q_FULL_LOCAL,
    KV_HEADS_LOCAL,
    KV_HIDDEN,
    KV_HIDDEN_LOCAL,
    LAYER_DYN,
    LAYER_HIDDEN_ROWS_DYN,
    NUM_HEADS_FULL,
    NUM_HEADS_FULL_LOCAL,
    ROPE_SEQ_DYN,
    TP_WORLD_SIZE,
)
from models.step3p5.prefill_qkv_proj_rope import (
    PREFILL_T,
    _build_tp_prefill_qkv_proj_rope_full_program,
    _torch_prefill_qkv_proj_rope_full_oracle,
)


# Constants matching the @pl.program signature for full attention path.
# rotary_dim = ROTARY_HALF_FULL * 2 = (HEAD_DIM//2//2) * 2 = HEAD_DIM // 2.
ROTARY_DIM_FULL = HEAD_DIM // 2  # 64


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--layer-idx", type=int, default=0,
                        help="Layer index (0..LAYER_DYN-1)")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _build_inputs(seed: int, layer_idx: int) -> dict[str, torch.Tensor]:
    """World-level random fixture; sliced per-rank for the kernel call.

    Shapes follow host_orch's ``[tp_size, ...]`` prefixing. Each rank's
    slot 0..tp_size-1 holds that rank's LOCAL shard; only slot 0 is
    actually consumed at world_size=1 dispatch.
    """
    g = torch.Generator().manual_seed(seed)
    tp = TP_WORLD_SIZE

    # World-level base tensors (single source of truth for the oracle).
    hidden_full = (
        torch.empty(PREFILL_T, HIDDEN, dtype=torch.float32)
        .normal_(mean=0.0, std=0.02, generator=g)
        .to(torch.bfloat16)
    )
    input_rms_per_layer = (
        torch.empty(LAYER_DYN, HIDDEN, dtype=torch.float32)
        .normal_(mean=0.0, std=0.05, generator=g)
    )
    wq_world = (
        torch.empty(LAYER_HIDDEN_ROWS_DYN, HIDDEN_Q_FULL, dtype=torch.float32)
        .normal_(mean=0.0, std=HIDDEN ** -0.5, generator=g)
        .to(torch.bfloat16)
    )
    wk_world = (
        torch.empty(LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN, dtype=torch.float32)
        .normal_(mean=0.0, std=HIDDEN ** -0.5, generator=g)
        .to(torch.bfloat16)
    )
    wv_world = (
        torch.empty(LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN, dtype=torch.float32)
        .normal_(mean=0.0, std=HIDDEN ** -0.5, generator=g)
        .to(torch.bfloat16)
    )
    q_norm_per_layer = (
        torch.empty(LAYER_DYN, HEAD_DIM, dtype=torch.float32)
        .normal_(mean=0.0, std=0.05, generator=g)
    )
    k_norm_per_layer = (
        torch.empty(LAYER_DYN, HEAD_DIM, dtype=torch.float32)
        .normal_(mean=0.0, std=0.05, generator=g)
    )
    w_g_world = (
        torch.empty(LAYER_HIDDEN_ROWS_DYN, NUM_HEADS_FULL, dtype=torch.float32)
        .normal_(mean=0.0, std=HIDDEN ** -0.5, generator=g)
        .to(torch.bfloat16)
    )
    rope_cos = (
        torch.empty(ROPE_SEQ_DYN, ROTARY_DIM_FULL, dtype=torch.float32)
        .normal_(mean=0.0, std=0.5, generator=g)
    )
    rope_sin = (
        torch.empty(ROPE_SEQ_DYN, ROTARY_DIM_FULL, dtype=torch.float32)
        .normal_(mean=0.0, std=0.5, generator=g)
    )
    positions = torch.arange(PREFILL_T, dtype=torch.int32)

    # Per-rank slicing for [tp_size, ...] inputs.
    def split_q(t):  # column-slice along TP into LOCAL Q heads
        return torch.stack([
            t[:, r * HIDDEN_Q_FULL_LOCAL:(r + 1) * HIDDEN_Q_FULL_LOCAL]
            for r in range(tp)
        ], dim=0).contiguous()

    def split_kv(t):
        return torch.stack([
            t[:, r * KV_HIDDEN_LOCAL:(r + 1) * KV_HIDDEN_LOCAL]
            for r in range(tp)
        ], dim=0).contiguous()

    def split_wg(t):
        return torch.stack([
            t[:, r * NUM_HEADS_FULL_LOCAL:(r + 1) * NUM_HEADS_FULL_LOCAL]
            for r in range(tp)
        ], dim=0).contiguous()

    def replicate(t):
        return t.unsqueeze(0).expand(tp, *t.shape).contiguous()

    inputs = {
        "current_hidden": replicate(hidden_full),
        "input_rms_weight": replicate(input_rms_per_layer),
        "wq": split_q(wq_world),
        "wk": split_kv(wk_world),
        "wv": split_kv(wv_world),
        "q_norm_weight": replicate(q_norm_per_layer),
        "k_norm_weight": replicate(k_norm_per_layer),
        "w_g": split_wg(w_g_world),
        "rope_cos": replicate(rope_cos),
        "rope_sin": replicate(rope_sin),
        "positions": replicate(positions),
        # World-level copies kept aside for the oracle.
        "_world_hidden": hidden_full,
        "_world_input_rms": input_rms_per_layer,
        "_world_wq": wq_world,
        "_world_wk": wk_world,
        "_world_wv": wv_world,
        "_world_q_norm": q_norm_per_layer,
        "_world_k_norm": k_norm_per_layer,
        "_world_w_g": w_g_world,
        "_world_rope_cos": rope_cos,
        "_world_rope_sin": rope_sin,
        "_world_positions": positions,
        "_layer_idx": layer_idx,
    }
    return inputs


def _build_specs(inputs: dict[str, torch.Tensor]) -> list:
    from golden import ScalarSpec, TensorSpec  # noqa: PLC0415

    tp = TP_WORLD_SIZE
    layer_idx_t = torch.tensor(inputs["_layer_idx"], dtype=torch.int32)

    return [
        TensorSpec("current_hidden", list(inputs["current_hidden"].shape),
                   torch.bfloat16, init_value=inputs["current_hidden"]),
        TensorSpec("input_rms_weight", list(inputs["input_rms_weight"].shape),
                   torch.float32, init_value=inputs["input_rms_weight"]),
        TensorSpec("wq", list(inputs["wq"].shape),
                   torch.bfloat16, init_value=inputs["wq"]),
        TensorSpec("wk", list(inputs["wk"].shape),
                   torch.bfloat16, init_value=inputs["wk"]),
        TensorSpec("wv", list(inputs["wv"].shape),
                   torch.bfloat16, init_value=inputs["wv"]),
        TensorSpec("q_norm_weight", list(inputs["q_norm_weight"].shape),
                   torch.float32, init_value=inputs["q_norm_weight"]),
        TensorSpec("k_norm_weight", list(inputs["k_norm_weight"].shape),
                   torch.float32, init_value=inputs["k_norm_weight"]),
        TensorSpec("w_g", list(inputs["w_g"].shape),
                   torch.bfloat16, init_value=inputs["w_g"]),
        TensorSpec("rope_cos", list(inputs["rope_cos"].shape),
                   torch.float32, init_value=inputs["rope_cos"]),
        TensorSpec("rope_sin", list(inputs["rope_sin"].shape),
                   torch.float32, init_value=inputs["rope_sin"]),
        TensorSpec("positions", list(inputs["positions"].shape),
                   torch.int32, init_value=inputs["positions"]),
        TensorSpec("normed_out", [tp, PREFILL_T, HIDDEN],
                   torch.bfloat16, is_output=True),
        TensorSpec("q_out", [tp, PREFILL_T, HIDDEN_Q_FULL_LOCAL],
                   torch.bfloat16, is_output=True),
        TensorSpec("k_out", [tp, PREFILL_T, KV_HIDDEN_LOCAL],
                   torch.bfloat16, is_output=True),
        TensorSpec("v_out", [tp, PREFILL_T, KV_HIDDEN_LOCAL],
                   torch.bfloat16, is_output=True),
        TensorSpec("gate_logits_out", [tp, PREFILL_T, NUM_HEADS_FULL_LOCAL],
                   torch.float32, is_output=True),
        ScalarSpec("layer_idx", torch.int32, value=layer_idx_t),
    ]


def _build_golden_fn(world: dict[str, torch.Tensor], layer_idx: int):
    """Run the world-level oracle once; slice outputs to rank 0."""

    def golden_fn(values: dict[str, torch.Tensor]) -> None:
        input_rms_layer = world["_world_input_rms"][layer_idx]
        q_norm_layer = world["_world_q_norm"][layer_idx]
        k_norm_layer = world["_world_k_norm"][layer_idx]
        row_lo = layer_idx * HIDDEN
        row_hi = row_lo + HIDDEN
        wq_layer = world["_world_wq"][row_lo:row_hi]
        wk_layer = world["_world_wk"][row_lo:row_hi]
        wv_layer = world["_world_wv"][row_lo:row_hi]
        w_g_layer = world["_world_w_g"][row_lo:row_hi]

        out = _torch_prefill_qkv_proj_rope_full_oracle(
            hidden=world["_world_hidden"],
            input_rms_weight=input_rms_layer,
            wq_full=wq_layer,
            wk_full=wk_layer,
            wv_full=wv_layer,
            q_norm_weight=q_norm_layer,
            k_norm_weight=k_norm_layer,
            w_g_full=w_g_layer,
            rope_cos=world["_world_rope_cos"],
            rope_sin=world["_world_rope_sin"],
            positions=world["_world_positions"],
        )
        normed_rank0 = out["normed"]
        q_rank0 = (
            out["q_rot"][:, :NUM_HEADS_FULL_LOCAL, :]
            .reshape(PREFILL_T, HIDDEN_Q_FULL_LOCAL)
        )
        k_rank0 = (
            out["k_rot"][:, :KV_HEADS_LOCAL, :]
            .reshape(PREFILL_T, KV_HIDDEN_LOCAL)
        )
        v_rank0 = (
            out["v_proj"][:, :KV_HEADS_LOCAL, :]
            .reshape(PREFILL_T, KV_HIDDEN_LOCAL)
        )
        gate_logits_rank0 = out["gate_logits"][:, :NUM_HEADS_FULL_LOCAL]

        # Other tp slots stay zero (kernel does not touch at world_size=1).
        values["normed_out"][:] = 0
        values["q_out"][:] = 0
        values["k_out"][:] = 0
        values["v_out"][:] = 0
        values["gate_logits_out"][:] = 0
        values["normed_out"][0] = normed_rank0
        values["q_out"][0] = q_rank0
        values["k_out"][0] = k_rank0
        values["v_out"][0] = v_rank0
        values["gate_logits_out"][0] = gate_logits_rank0

    return golden_fn


def main() -> int:
    args = _parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))

    from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415

    set_backend_type({
        "a2a3": BackendType.Ascend910B,
        "a2a3sim": BackendType.Ascend910B,
        "a5": BackendType.Ascend950,
        "a5sim": BackendType.Ascend950,
    }[args.platform])

    program_cls = _build_tp_prefill_qkv_proj_rope_full_program(
        tp_size=TP_WORLD_SIZE,
    )
    print(
        f"[test_prefill_qkv] LOCAL: PREFILL_T={PREFILL_T} HIDDEN={HIDDEN} "
        f"HIDDEN_Q_FULL_LOCAL={HIDDEN_Q_FULL_LOCAL} "
        f"KV_HIDDEN_LOCAL={KV_HIDDEN_LOCAL} layer={args.layer_idx}",
        flush=True,
    )

    inputs = _build_inputs(args.seed, args.layer_idx)
    specs = _build_specs(inputs)

    from golden import ratio_allclose, run  # noqa: PLC0415

    if args.smoke or args.platform.endswith("sim"):
        result = run(
            program=program_cls, specs=specs,
            runtime_cfg=dict(platform=args.platform, device_id=args.device),
            compile_only=True,
        )
        print(f"[test_prefill_qkv] SMOKE: {result}", flush=True)
        return 0 if result.passed else 1

    result = run(
        program=program_cls, specs=specs,
        golden_fn=_build_golden_fn(inputs, args.layer_idx),
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=6e-3,
        atol=6e-3,
        compare_fn={
            "normed_out": ratio_allclose(
                atol=6e-3, rtol=6e-3, max_error_ratio=0.06),
            "q_out": ratio_allclose(
                atol=6e-3, rtol=6e-3, max_error_ratio=0.06),
            "k_out": ratio_allclose(
                atol=6e-3, rtol=6e-3, max_error_ratio=0.06),
            "v_out": ratio_allclose(
                atol=6e-3, rtol=6e-3, max_error_ratio=0.06),
            "gate_logits_out": ratio_allclose(
                atol=6e-3, rtol=6e-3, max_error_ratio=0.06),
        },
    )
    print(f"[test_prefill_qkv] DEVICE: {result}", flush=True)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
