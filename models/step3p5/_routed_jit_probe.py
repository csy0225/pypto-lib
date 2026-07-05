# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Probe: can the RECV-tiled routed-expert body compile as a plain @pl.jit?

If yes, the live worker (_stage_attn_worker.py's ChipWorker) can register it
like dense_swiglu_perrank / shared_swiglu_perrank / rms_lm_head_test — one
chip_process per card, no co-tenancy. If pl.parallel(N_LOCAL_EXPERTS) is
rejected at @pl.jit top level, fall back to the @pl.program path.

    python -m models.step3p5._routed_jit_probe --smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pypto.language as pl

from models.step3p5.vllm_routed_experts import (
    DOWN_K_CHUNK,
    DOWN_N_CHUNK,
    GATE_K_CHUNK,
    GATE_N_CHUNK,
    HIDDEN,
    INTER,
    LOCAL_RECV_MAX,
    N_LOCAL_EXPERTS,
    N_RECV_TILES,
    RECV_TILE,
)


@pl.jit.inline
def routed_experts_jit_body(
    local_routed_x: pl.Tensor[[LOCAL_RECV_MAX, HIDDEN], pl.BF16],
    local_expert_offset: pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
    local_expert_count: pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
    w_gate: pl.Tensor[[N_LOCAL_EXPERTS, HIDDEN, INTER], pl.BF16],
    w_up: pl.Tensor[[N_LOCAL_EXPERTS, HIDDEN, INTER], pl.BF16],
    w_down: pl.Tensor[[N_LOCAL_EXPERTS, INTER, HIDDEN], pl.BF16],
    local_routed_y: pl.Tensor[[LOCAL_RECV_MAX, HIDDEN], pl.BF16],
) -> pl.Tensor[[LOCAL_RECV_MAX, HIDDEN], pl.BF16]:
    for e in pl.parallel(N_LOCAL_EXPERTS):
        n_rows = pl.read(local_expert_count, [e])
        offset_i32 = pl.read(local_expert_offset, [e])
        offset = pl.cast(offset_i32, pl.INDEX)
        valid_rows = pl.cast(n_rows, pl.INDEX)
        for tile_idx in pl.range(N_RECV_TILES):
            tile_row0 = tile_idx * RECV_TILE
            tile_offset = offset + tile_row0
            tile_valid = pl.min(RECV_TILE, valid_rows - tile_row0)
            if tile_valid > 0:
                h_bf16 = pl.create_tensor([RECV_TILE, INTER], dtype=pl.BF16)
                for nb in pl.spmd(INTER // GATE_N_CHUNK, name_hint="rj_gate_up"):
                    n0 = nb * GATE_N_CHUNK
                    x0 = pl.slice(
                        local_routed_x, [RECV_TILE, GATE_K_CHUNK], [tile_offset, 0],
                        valid_shape=[tile_valid, GATE_K_CHUNK],
                    )
                    wg0 = pl.reshape(pl.slice(w_gate, [1, GATE_K_CHUNK, GATE_N_CHUNK], [e, 0, n0]), [GATE_K_CHUNK, GATE_N_CHUNK])
                    wu0 = pl.reshape(pl.slice(w_up, [1, GATE_K_CHUNK, GATE_N_CHUNK], [e, 0, n0]), [GATE_K_CHUNK, GATE_N_CHUNK])
                    gate_acc = pl.matmul(x0, wg0, out_dtype=pl.FP32)
                    up_acc = pl.matmul(x0, wu0, out_dtype=pl.FP32)
                    for kb in pl.range(1, HIDDEN // GATE_K_CHUNK):
                        k0 = kb * GATE_K_CHUNK
                        xk = pl.slice(local_routed_x, [RECV_TILE, GATE_K_CHUNK], [tile_offset, k0], valid_shape=[tile_valid, GATE_K_CHUNK])
                        wgk = pl.reshape(pl.slice(w_gate, [1, GATE_K_CHUNK, GATE_N_CHUNK], [e, k0, n0]), [GATE_K_CHUNK, GATE_N_CHUNK])
                        wuk = pl.reshape(pl.slice(w_up, [1, GATE_K_CHUNK, GATE_N_CHUNK], [e, k0, n0]), [GATE_K_CHUNK, GATE_N_CHUNK])
                        gate_acc = pl.matmul_acc(gate_acc, xk, wgk)
                        up_acc = pl.matmul_acc(up_acc, xk, wuk)
                    sig = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                    gated = pl.mul(pl.mul(gate_acc, sig), up_acc)
                    gv = pl.set_validshape(gated, tile_valid, GATE_N_CHUNK)
                    h_bf16[:, n0 : n0 + GATE_N_CHUNK] = pl.cast(gv, target_type=pl.BF16)
                for db in pl.spmd(HIDDEN // DOWN_N_CHUNK, name_hint="rj_down"):
                    d0 = db * DOWN_N_CHUNK
                    h0 = pl.slice(h_bf16, [RECV_TILE, DOWN_K_CHUNK], [0, 0], valid_shape=[tile_valid, DOWN_K_CHUNK])
                    wd0 = pl.reshape(pl.slice(w_down, [1, DOWN_K_CHUNK, DOWN_N_CHUNK], [e, 0, d0]), [DOWN_K_CHUNK, DOWN_N_CHUNK])
                    y_acc = pl.matmul(h0, wd0, out_dtype=pl.FP32)
                    for kb2 in pl.range(1, INTER // DOWN_K_CHUNK):
                        k0 = kb2 * DOWN_K_CHUNK
                        hk = pl.slice(h_bf16, [RECV_TILE, DOWN_K_CHUNK], [0, k0], valid_shape=[tile_valid, DOWN_K_CHUNK])
                        wdk = pl.reshape(pl.slice(w_down, [1, DOWN_K_CHUNK, DOWN_N_CHUNK], [e, k0, d0]), [DOWN_K_CHUNK, DOWN_N_CHUNK])
                        y_acc = pl.matmul_acc(y_acc, hk, wdk)
                    yv = pl.set_validshape(y_acc, tile_valid, DOWN_N_CHUNK)
                    ym = pl.fillpad(yv, pad_value=pl.PadValue.zero)
                    local_routed_y = pl.assemble(local_routed_y, pl.cast(ym, target_type=pl.BF16), [tile_offset, d0])
    return local_routed_y


@pl.jit
def routed_experts_jit(
    local_routed_x: pl.Tensor[[LOCAL_RECV_MAX, HIDDEN], pl.BF16],
    local_expert_offset: pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
    local_expert_count: pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
    w_gate: pl.Tensor[[N_LOCAL_EXPERTS, HIDDEN, INTER], pl.BF16],
    w_up: pl.Tensor[[N_LOCAL_EXPERTS, HIDDEN, INTER], pl.BF16],
    w_down: pl.Tensor[[N_LOCAL_EXPERTS, INTER, HIDDEN], pl.BF16],
    local_routed_y: pl.Out[pl.Tensor[[LOCAL_RECV_MAX, HIDDEN], pl.BF16]],
):
    routed_experts_jit_body(
        local_routed_x, local_expert_offset, local_expert_count,
        w_gate, w_up, w_down, local_routed_y,
    )
    return local_routed_y


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-p", "--platform", default="a2a3")
    p.add_argument("-d", "--device", type=int, default=8)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device-run", action="store_true",
                   help="compile + run on device + validate vs torch golden")
    p.add_argument("--real-weights", action="store_true")
    p.add_argument("--ckpt", default="/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp")
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--rank", type=int, default=0)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    import torch  # noqa: PLC0415
    from golden import TensorSpec, ratio_allclose, run_jit  # noqa: PLC0415
    from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415

    from models.step3p5.vllm_routed_experts import (  # noqa: PLC0415
        _balanced_csr,
        _real_weights,
        _synthetic_weights,
        golden_routed_experts_perrank,
    )

    set_backend_type(BackendType.Ascend910B)
    g = torch.Generator().manual_seed(0)
    n = LOCAL_RECV_MAX
    offs, counts = _balanced_csr(0)
    x = (torch.randn(n, HIDDEN, generator=g) * 0.3).bfloat16()
    w = _real_weights(a.ckpt, a.layer, a.rank) if a.real_weights else _synthetic_weights(a.rank)
    specs = [
        TensorSpec("local_routed_x", [n, HIDDEN], torch.bfloat16, init_value=x),
        TensorSpec("local_expert_offset", [N_LOCAL_EXPERTS], torch.int32, init_value=offs),
        TensorSpec("local_expert_count", [N_LOCAL_EXPERTS], torch.int32, init_value=counts),
        TensorSpec("w_gate", [N_LOCAL_EXPERTS, HIDDEN, INTER], torch.bfloat16, init_value=w["w_gate"]),
        TensorSpec("w_up", [N_LOCAL_EXPERTS, HIDDEN, INTER], torch.bfloat16, init_value=w["w_up"]),
        TensorSpec("w_down", [N_LOCAL_EXPERTS, INTER, HIDDEN], torch.bfloat16, init_value=w["w_down"]),
        TensorSpec("local_routed_y", [n, HIDDEN], torch.bfloat16, is_output=True),
    ]

    if a.device_run:
        def golden_fn(values):
            values["local_routed_y"] = golden_routed_experts_perrank(
                values["local_routed_x"], values["local_expert_offset"],
                values["local_expert_count"], values["w_gate"], values["w_up"], values["w_down"],
            )

        res = run_jit(
            fn=routed_experts_jit, specs=specs, golden_fn=golden_fn,
            runtime_cfg=dict(platform=a.platform, device_id=a.device),
            compile_only=False,
            compare_fn={"local_routed_y": ratio_allclose(atol=0.04, rtol=0.04, max_error_ratio=0.1)},
        )
        print(f"[routed_jit_probe] DEVICE-RUN real={a.real_weights}: passed={res.passed}", flush=True)
        return 0 if res.passed else 1

    res = run_jit(fn=routed_experts_jit, specs=specs, runtime_cfg=dict(platform=a.platform, device_id=a.device), compile_only=True)
    print(f"[routed_jit_probe] SMOKE: {res}", flush=True)
    return 0 if res.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
