# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Device probe for the N=16 matmul_acc codegen bug (head-gate gate_logits).

The head-gate gate_logits = RMSNorm(hidden) @ w_g has output width N=NUM_HEADS
(16). It was moved off-device because pl.matmul_acc with small N=16 was reported
to drop the K accumulation (gate_logits ~20x too small). This probe replicates
the exact K-chunked matmul_acc pattern (gate.py:138-149) with N=16 and compares
to torch x@w. If it PASSES on the current stack, the on-device head-gate can be
restored (path (a) to monolithic token-exact); if it FAILS (~20x off), the bug
is still present.

Usage::
    cd /data/chensiyu/hw_project/pypto/workspace/pypto-lib
    python -m tests.step3p5.probes._probe_matmul_acc_n16 -p a2a3 -d 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pypto.language as pl  # noqa: E402
from models.step3p5.config import BATCH, HIDDEN  # noqa: E402

T = BATCH           # 16
N = 16              # NUM_HEADS_FULL_LOCAL_PAD (head-gate gate_logits width)
KCHUNK = 512        # matches gate.py GATE_K_CHUNK


@pl.jit
def matmul_acc_n16(
    x: pl.Tensor[[T, HIDDEN], pl.BF16],
    w: pl.Tensor[[HIDDEN, N], pl.FP32],
    out: pl.Out[pl.Tensor[[T, N], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_matmul_n16"):
        # Cast per K-chunk (not the whole [T, HIDDEN] tile) so the FP32 working
        # set stays [T, KCHUNK] = 32 KB and does not overflow UB. This keeps the
        # matmul_acc(N=16) K-accumulation semantics identical to gate.py.
        x0 = pl.cast(pl.slice(x, [T, KCHUNK], [0, 0]), target_type=pl.FP32)
        w0 = pl.slice(w, [KCHUNK, N], [0, 0])
        acc = pl.matmul(x0, w0, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN // KCHUNK):
            k0 = kb * KCHUNK
            xk = pl.cast(pl.slice(x, [T, KCHUNK], [0, k0]), target_type=pl.FP32)
            wk = pl.slice(w, [KCHUNK, N], [k0, 0])
            acc = pl.matmul_acc(acc, xk, wk)
        out[:, :] = acc
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

    def init_x():
        return (torch.randn(T, HIDDEN) * 0.5).bfloat16()

    def init_w():
        return (torch.randn(HIDDEN, N) / HIDDEN ** 0.5).float()

    _x = init_x()
    _w = init_w()

    def golden(tensors):
        tensors["out"][:] = (tensors["x"].float() @ tensors["w"].float())

    specs = [
        TensorSpec("x", [T, HIDDEN], torch.bfloat16, init_value=lambda: _x.clone()),
        TensorSpec("w", [HIDDEN, N], torch.float32, init_value=lambda: _w.clone()),
        TensorSpec("out", [T, N], torch.float32, is_output=True),
    ]
    print(f"[matmul_acc_n16] platform={args.platform} device={args.device} T={T} N={N} K={HIDDEN}", flush=True)
    if args.smoke or args.platform.endswith("sim"):
        r = run_jit(fn=matmul_acc_n16, specs=specs,
                    runtime_cfg=dict(platform=args.platform, device_id=args.device),
                    compile_only=True)
        print(f"[matmul_acc_n16] SMOKE: {r}", flush=True)
        return 0 if r.passed else 1
    r = run_jit(
        fn=matmul_acc_n16, specs=specs, golden_fn=golden,
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=2e-2, atol=2e-2,
        compare_fn={"out": ratio_allclose(atol=2e-2, rtol=2e-2, max_error_ratio=0.06)},
    )
    print(f"[matmul_acc_n16] DEVICE RESULT: {r}", flush=True)
    return 0 if r.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
