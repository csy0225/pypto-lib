# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Device probe for the full on-device head-gate (path (a) validation).

Replicates the head-gate that was moved worker-side, now that the N=16
matmul_acc bug is fixed (see _probe_matmul_acc_n16). Chain::

    gate_logits = normed_all @ w_g          # [T, NHF_PAD]  (K-chunked matmul_acc)
    gate_score  = sigmoid(gate_logits)       # [T, NHF_PAD]
    gate_exp    = gate_score @ R             # [T, HQ_LOCAL]  block-diag expand

R = block-diag ones [NHF_PAD, HQ_LOCAL]: R[h, h*HEAD_DIM + d] = 1 for
h < NUM_HEADS_FULL_LOCAL (real heads), rows for padded heads are zero. So
gate_exp[b, h*HEAD_DIM + d] = gate_score[b, h] = per-head sigmoid, broadcast
across HEAD_DIM — exactly modeling_step3p5 L527-531
(attn_out.view(.,num_heads,head_dim) * gate_states.unsqueeze(-1).sigmoid()).

If PASS, the on-device head-gate can be restored in attention_full/swa so the
monolithic whole-net self-computes per-layer gate_r (R is layer-independent).

Usage::
    cd /data/chensiyu/hw_project/pypto/workspace/pypto-lib
    python -m tests.step3p5.probes._probe_head_gate_full -p a2a3 -d 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pypto.language as pl  # noqa: E402
from models.step3p5.config import (  # noqa: E402
    HEAD_DIM,
    HIDDEN,
    HIDDEN_Q_FULL_LOCAL,
    NUM_HEADS_FULL_LOCAL,
    NUM_HEADS_FULL_LOCAL_PAD,
)

T = 16                              # BATCH
NHF_PAD = NUM_HEADS_FULL_LOCAL_PAD  # 16
NHF = NUM_HEADS_FULL_LOCAL          # 8
HQ = HIDDEN_Q_FULL_LOCAL            # 1024
KCHUNK = 512


@pl.jit
def head_gate_full(
    normed: pl.Tensor[[T, HIDDEN], pl.BF16],
    w_g: pl.Tensor[[HIDDEN, NHF_PAD], pl.BF16],
    gate_r: pl.Tensor[[NHF_PAD, HQ], pl.BF16],
    out: pl.Out[pl.Tensor[[T, HQ], pl.BF16]],
):
    gate_score_t = pl.create_tensor([T, NHF_PAD], dtype=pl.BF16)
    # --- Scope A: gate_logits = normed @ w_g (bf16) + sigmoid -----------------
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="head_gate_logits"):
        x0 = pl.slice(normed, [T, KCHUNK], [0, 0])
        wg0 = pl.slice(w_g, [KCHUNK, NHF_PAD], [0, 0])
        logits = pl.matmul(x0, wg0, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN // KCHUNK):
            k0 = kb * KCHUNK
            xk = pl.slice(normed, [T, KCHUNK], [0, k0])
            wgk = pl.slice(w_g, [KCHUNK, NHF_PAD], [k0, 0])
            logits = pl.matmul_acc(logits, xk, wgk)
        # gate_score = sigmoid(logits) = 1 / (1 + exp(-logits))
        gate_score = pl.recip(pl.add(pl.exp(pl.neg(logits)), 1.0))
        gate_score_t[:, :] = pl.cast(gate_score, target_type=pl.BF16)
    # --- Scope B: gate_exp = gate_score @ R (block-diag expand) ---------------
    # N-chunk the [T, HQ] output so the FP32 matmul result + bf16 cast stay in UB.
    NCHUNK = 256
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="head_gate_expand"):
        for nb in pl.range(0, HQ // NCHUNK):
            n0 = nb * NCHUNK
            r_chunk = pl.slice(gate_r, [NHF_PAD, NCHUNK], [0, n0])
            ge_chunk = pl.matmul(gate_score_t, r_chunk, out_dtype=pl.FP32)
            out[:, n0:n0 + NCHUNK] = pl.cast(ge_chunk, target_type=pl.BF16)
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-p", "--platform", default="a2a3",
                   choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    p.add_argument("-d", "--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    import torch  # noqa: PLC0415
    torch.manual_seed(args.seed)
    from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415
    set_backend_type({"a2a3": BackendType.Ascend910B, "a2a3sim": BackendType.Ascend910B,
                      "a5": BackendType.Ascend950, "a5sim": BackendType.Ascend950}[args.platform])
    from golden import TensorSpec, ratio_allclose, run_jit  # noqa: PLC0415

    _normed = (torch.randn(T, HIDDEN) * 0.5).bfloat16()
    _w_g = (torch.randn(HIDDEN, NHF_PAD) / HIDDEN ** 0.5).bfloat16()
    # Block-diag ones R: R[h, h*HEAD_DIM + d] = 1 for h < NHF (real heads).
    _R = torch.zeros(NHF_PAD, HQ)
    for h in range(NHF):
        _R[h, h * HEAD_DIM:(h + 1) * HEAD_DIM] = 1.0
    _R = _R.bfloat16()

    def golden(tensors):
        score = torch.sigmoid(tensors["normed"].float() @ tensors["w_g"].float())  # [T,NHF_PAD]
        tensors["out"][:] = (score @ tensors["gate_r"].float()).bfloat16()

    specs = [
        TensorSpec("normed", [T, HIDDEN], torch.bfloat16, init_value=lambda: _normed.clone()),
        TensorSpec("w_g", [HIDDEN, NHF_PAD], torch.bfloat16, init_value=lambda: _w_g.clone()),
        TensorSpec("gate_r", [NHF_PAD, HQ], torch.bfloat16, init_value=lambda: _R.clone()),
        TensorSpec("out", [T, HQ], torch.bfloat16, is_output=True),
    ]
    print(f"[head_gate_full] platform={args.platform} device={args.device} "
          f"T={T} NHF={NHF}/{NHF_PAD} HQ={HQ} K={HIDDEN}", flush=True)
    if args.smoke or args.platform.endswith("sim"):
        r = run_jit(fn=head_gate_full, specs=specs,
                    runtime_cfg=dict(platform=args.platform, device_id=args.device),
                    compile_only=True)
        print(f"[head_gate_full] SMOKE: {r}", flush=True)
        return 0 if r.passed else 1
    r = run_jit(
        fn=head_gate_full, specs=specs, golden_fn=golden,
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=3e-2, atol=3e-2,
        compare_fn={"out": ratio_allclose(atol=3e-2, rtol=3e-2, max_error_ratio=0.08)},
    )
    print(f"[head_gate_full] DEVICE RESULT: {r}", flush=True)
    return 0 if r.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
