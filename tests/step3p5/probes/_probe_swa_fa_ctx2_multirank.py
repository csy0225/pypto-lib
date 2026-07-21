"""TP=8 SWA decode attention ctx=2 stage-local numerical probe.

This is diagnostic-only.  It reproduces ``attention_swa.py`` Stage 1..4 with
the production storage shapes:

* 12 real Q heads padded to ``SWA_Q_PAD_ALIGNED=32``;
* one 128-token KV block;
* ctx_len=2;
* BF16 exp weights and FP32 online-softmax accumulation;
* the current ``ctx[32,128] -> BF16 -> reshape -> slice[1536]`` epilogue.

The program exports raw QK scores, BF16 exp weights, row-shaped context and
the flattened attention output independently.  This distinguishes:

1. QK;
2. mask/softmax;
3. SV/normalisation;
4. final padded-context flattening.

It intentionally does not include projection weights, collectives, MLP or
the production whole-net ABI.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--device",
        default="8,9,10,11,12,13,14,15",
    )
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--build-dir", default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))

    device_ids = [int(item) for item in args.device.split(",")]

    from pypto.backend import BackendType, set_backend_type

    set_backend_type(BackendType.Ascend910B)

    import pypto.language as pl
    import pypto.language.distributed as pld
    from pypto import ir
    from pypto.ir.distributed_compiled_program import DistributedConfig

    from models.step3p5.config import (
        ATTN_SCALE,
        BATCH,
        BLOCK_SIZE,
        HEAD_DIM,
        HIDDEN_Q_SWA_LOCAL,
        Q_HEAD_BATCH_SWA,
        TP_WORLD_SIZE,
    )

    if len(device_ids) != TP_WORLD_SIZE:
        raise ValueError(
            f"probe requires TP={TP_WORLD_SIZE}, got devices={device_ids}"
        )
    if HIDDEN_Q_SWA_LOCAL != Q_HEAD_BATCH_SWA * HEAD_DIM:
        raise ValueError("unexpected SWA local-head layout")

    q_real = Q_HEAD_BATCH_SWA
    q_pad = 32
    ctx_len = 2
    score_rows = BATCH * q_pad

    @pl.program
    class SwaFaCtx2Probe:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            q_padded: pl.Tensor[[score_rows, HEAD_DIM], pl.BF16],
            k_cache: pl.Tensor[[BLOCK_SIZE, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[BLOCK_SIZE, HEAD_DIM], pl.BF16],
            raw_out: pl.Out[
                pl.Tensor[[score_rows, BLOCK_SIZE], pl.FP32]
            ],
            exp_out: pl.Out[
                pl.Tensor[[score_rows, BLOCK_SIZE], pl.BF16]
            ],
            ctx_out: pl.Out[
                pl.Tensor[[score_rows, HEAD_DIM], pl.FP32]
            ],
            flat_out: pl.Out[
                pl.Tensor[[BATCH, HIDDEN_Q_SWA_LOCAL], pl.BF16]
            ],
        ):
            all_raw_scores = pl.create_tensor(
                [score_rows, BLOCK_SIZE],
                dtype=pl.FP32,
            )
            all_exp_padded = pl.create_tensor(
                [score_rows, BLOCK_SIZE],
                dtype=pl.BF16,
            )
            all_cur_mi = pl.create_tensor([score_rows, 1], dtype=pl.FP32)
            all_cur_li = pl.create_tensor([score_rows, 1], dtype=pl.FP32)
            all_oi_tmp = pl.create_tensor(
                [score_rows, HEAD_DIM],
                dtype=pl.FP32,
            )

            # Exact production Stage 1 shape and QK matmul.
            for fa_b in pl.spmd(BATCH, name_hint="probe_swa_qk_matmul"):
                row = fa_b * q_pad
                q_tile = pl.slice(
                    q_padded,
                    [q_pad, HEAD_DIM],
                    [row, 0],
                )
                k_tile = pl.slice(
                    k_cache,
                    [BLOCK_SIZE, HEAD_DIM],
                    [0, 0],
                )
                raw_scores = pl.matmul(
                    q_tile,
                    k_tile,
                    b_trans=True,
                    out_dtype=pl.FP32,
                )
                all_raw_scores = pl.assemble(
                    all_raw_scores,
                    raw_scores,
                    [row, 0],
                )
                raw_out = pl.assemble(raw_out, raw_scores, [row, 0])

            # Exact production Stage 2 valid-shape/fillpad/softmax path.
            for fa_b in pl.spmd(BATCH, name_hint="probe_swa_softmax"):
                row = fa_b * q_pad
                scores_valid = pl.slice(
                    all_raw_scores,
                    [q_pad, BLOCK_SIZE],
                    [row, 0],
                    valid_shape=[q_real, ctx_len],
                )
                scores_padded = pl.fillpad(
                    scores_valid,
                    pad_value=pl.PadValue.min,
                )
                scores = pl.mul(scores_padded, ATTN_SCALE)
                cur_mi = pl.row_max(scores)
                exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
                exp_scores_bf16 = pl.cast(
                    exp_scores,
                    target_type=pl.BF16,
                )
                exp_scores_fp32 = pl.cast(
                    exp_scores_bf16,
                    target_type=pl.FP32,
                )
                cur_li = pl.row_sum(exp_scores_fp32)
                all_exp_padded = pl.assemble(
                    all_exp_padded,
                    exp_scores_bf16,
                    [row, 0],
                )
                all_cur_mi = pl.assemble(
                    all_cur_mi,
                    cur_mi,
                    [row, 0],
                )
                all_cur_li = pl.assemble(
                    all_cur_li,
                    cur_li,
                    [row, 0],
                )
                exp_out = pl.assemble(
                    exp_out,
                    exp_scores_bf16,
                    [row, 0],
                )

            # Exact production Stage 3 BF16-exp x BF16-V matmul.
            for fa_b in pl.spmd(BATCH, name_hint="probe_swa_sv_matmul"):
                row = fa_b * q_pad
                exp_tile = pl.slice(
                    all_exp_padded,
                    [q_pad, BLOCK_SIZE],
                    [row, 0],
                )
                v_tile = pl.slice(
                    v_cache,
                    [BLOCK_SIZE, HEAD_DIM],
                    [0, 0],
                )
                oi_tmp = pl.matmul(
                    exp_tile,
                    v_tile,
                    out_dtype=pl.FP32,
                )
                all_oi_tmp = pl.assemble(
                    all_oi_tmp,
                    oi_tmp,
                    [row, 0],
                )

            # Exact one-block production Stage 4 and current flatten epilogue.
            for fa_b in pl.spmd(
                BATCH,
                name_hint="probe_swa_online_softmax",
            ):
                row = fa_b * q_pad
                oi = pl.slice(
                    all_oi_tmp,
                    [q_pad, HEAD_DIM],
                    [row, 0],
                )
                li = pl.slice(
                    all_cur_li,
                    [q_pad, 1],
                    [row, 0],
                )
                ctx = pl.row_expand_div(oi, li)
                ctx_out = pl.assemble(ctx_out, ctx, [row, 0])

                ctx_bf16 = pl.cast(ctx, target_type=pl.BF16)
                ctx_padded_flat = pl.reshape(
                    ctx_bf16,
                    [1, q_pad * HEAD_DIM],
                )
                ctx_flat_bf16 = pl.slice(
                    ctx_padded_flat,
                    [1, HIDDEN_Q_SWA_LOCAL],
                    [0, 0],
                )
                flat_out = pl.assemble(
                    flat_out,
                    ctx_flat_bf16,
                    [fa_b, 0],
                )

            return flat_out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            q_padded: pl.Tensor[
                [TP_WORLD_SIZE, score_rows, HEAD_DIM], pl.BF16
            ],
            k_cache: pl.Tensor[
                [TP_WORLD_SIZE, BLOCK_SIZE, HEAD_DIM], pl.BF16
            ],
            v_cache: pl.Tensor[
                [TP_WORLD_SIZE, BLOCK_SIZE, HEAD_DIM], pl.BF16
            ],
            raw_out: pl.Out[
                pl.Tensor[
                    [TP_WORLD_SIZE, score_rows, BLOCK_SIZE], pl.FP32
                ]
            ],
            exp_out: pl.Out[
                pl.Tensor[
                    [TP_WORLD_SIZE, score_rows, BLOCK_SIZE], pl.BF16
                ]
            ],
            ctx_out: pl.Out[
                pl.Tensor[
                    [TP_WORLD_SIZE, score_rows, HEAD_DIM], pl.FP32
                ]
            ],
            flat_out: pl.Out[
                pl.Tensor[
                    [TP_WORLD_SIZE, BATCH, HIDDEN_Q_SWA_LOCAL],
                    pl.BF16,
                ]
            ],
        ):
            for rank in pl.range(pld.world_size()):
                self.chip_orch(
                    q_padded[rank],
                    k_cache[rank],
                    v_cache[rank],
                    raw_out[rank],
                    exp_out[rank],
                    ctx_out[rank],
                    flat_out[rank],
                    device=rank,
                )

    generator = torch.Generator().manual_seed(args.seed)

    def randn(shape: tuple[int, ...], std: float = 0.2) -> torch.Tensor:
        return torch.empty(shape, dtype=torch.float32).normal_(
            0.0,
            std,
            generator=generator,
        )

    q = torch.zeros(
        TP_WORLD_SIZE,
        BATCH,
        q_pad,
        HEAD_DIM,
        dtype=torch.bfloat16,
    )
    q[:, :, :q_real, :] = randn(
        (TP_WORLD_SIZE, BATCH, q_real, HEAD_DIM)
    ).to(torch.bfloat16)
    q_flat = q.reshape(TP_WORLD_SIZE, score_rows, HEAD_DIM).contiguous()

    k = torch.zeros(
        TP_WORLD_SIZE,
        BLOCK_SIZE,
        HEAD_DIM,
        dtype=torch.bfloat16,
    )
    v = torch.zeros_like(k)
    k[:, :ctx_len, :] = randn(
        (TP_WORLD_SIZE, ctx_len, HEAD_DIM)
    ).to(torch.bfloat16)
    v[:, :ctx_len, :] = randn(
        (TP_WORLD_SIZE, ctx_len, HEAD_DIM)
    ).to(torch.bfloat16)

    raw_ref = torch.zeros(
        TP_WORLD_SIZE,
        BATCH,
        q_pad,
        BLOCK_SIZE,
    )
    exp_ref = torch.zeros_like(raw_ref)
    ctx_ref = torch.zeros(
        TP_WORLD_SIZE,
        BATCH,
        q_pad,
        HEAD_DIM,
    )
    for rank in range(TP_WORLD_SIZE):
        for batch_idx in range(BATCH):
            raw = q[rank, batch_idx].float() @ k[rank].float().T
            raw_ref[rank, batch_idx] = raw
            real_scores = raw[:q_real, :ctx_len] * (
                1.0 / math.sqrt(HEAD_DIM)
            )
            mi = real_scores.max(dim=-1, keepdim=True).values
            exp = torch.exp(real_scores - mi).to(torch.bfloat16)
            exp_ref[rank, batch_idx, :q_real, :ctx_len] = exp.float()
            li = exp.float().sum(dim=-1, keepdim=True)
            ctx = exp.float() @ v[rank, :ctx_len].float()
            ctx_ref[rank, batch_idx, :q_real] = ctx / li

    flat_ref = ctx_ref[:, :, :q_real, :].reshape(
        TP_WORLD_SIZE,
        BATCH,
        HIDDEN_Q_SWA_LOCAL,
    ).to(torch.bfloat16)

    raw_out = torch.zeros(
        TP_WORLD_SIZE,
        score_rows,
        BLOCK_SIZE,
        dtype=torch.float32,
    )
    exp_out = torch.zeros(
        TP_WORLD_SIZE,
        score_rows,
        BLOCK_SIZE,
        dtype=torch.bfloat16,
    )
    ctx_out = torch.zeros(
        TP_WORLD_SIZE,
        score_rows,
        HEAD_DIM,
        dtype=torch.float32,
    )
    flat_out = torch.zeros(
        TP_WORLD_SIZE,
        BATCH,
        HIDDEN_Q_SWA_LOCAL,
        dtype=torch.bfloat16,
    )

    if args.build_dir:
        os.environ["PYPTO_PROG_BUILD_DIR"] = args.build_dir
    compiled = ir.compile(
        SwaFaCtx2Probe,
        platform="a2a3",
        distributed_config=DistributedConfig(
            device_ids=device_ids,
            num_sub_workers=0,
        ),
        skip_ptoas=False,
        dump_passes=False,
    )
    print(f"[probe] compiled={compiled.output_dir}", flush=True)
    compiled(
        q_flat,
        k,
        v,
        raw_out,
        exp_out,
        ctx_out,
        flat_out,
    )
    print("[probe] run completed", flush=True)

    raw_out_v = raw_out.reshape(
        TP_WORLD_SIZE,
        BATCH,
        q_pad,
        BLOCK_SIZE,
    )
    exp_out_v = exp_out.reshape(
        TP_WORLD_SIZE,
        BATCH,
        q_pad,
        BLOCK_SIZE,
    )
    ctx_out_v = ctx_out.reshape(
        TP_WORLD_SIZE,
        BATCH,
        q_pad,
        HEAD_DIM,
    )

    def report(
        name: str,
        got: torch.Tensor,
        expected: torch.Tensor,
        *,
        atol: float,
        rtol: float,
    ) -> tuple[bool, float, float]:
        got_f = got.float()
        exp_f = expected.float()
        diff = (got_f - exp_f).abs()
        bad = diff > (atol + rtol * exp_f.abs())
        bad_ratio = float(bad.float().mean().item())
        max_abs = float(diff.max().item())
        ok = bad_ratio <= 0.02
        first = None
        if bad.any():
            index = bad.nonzero()[0].tolist()
            first = {
                "index": index,
                "got": float(got_f[tuple(index)].item()),
                "expected": float(exp_f[tuple(index)].item()),
            }
        print(
            f"[probe] {name} ok={ok} bad_ratio={bad_ratio:.6f} "
            f"max_abs={max_abs:.6f} first={first}",
            flush=True,
        )
        return ok, bad_ratio, max_abs

    raw_ok, _, _ = report(
        "raw_qk",
        raw_out_v[:, :, :q_real, :ctx_len],
        raw_ref[:, :, :q_real, :ctx_len],
        atol=0.05,
        rtol=0.05,
    )
    exp_ok, _, _ = report(
        "softmax_exp_bf16",
        exp_out_v[:, :, :q_real, :ctx_len],
        exp_ref[:, :, :q_real, :ctx_len],
        atol=0.02,
        rtol=0.02,
    )
    ctx_ok, _, _ = report(
        "sv_context_rows",
        ctx_out_v[:, :, :q_real, :],
        ctx_ref[:, :, :q_real, :],
        atol=0.04,
        rtol=0.04,
    )
    flat_ok, _, _ = report(
        "flattened_attn_out",
        flat_out,
        flat_ref,
        atol=0.04,
        rtol=0.04,
    )

    print(
        "[probe] classification "
        f"raw_qk={'PASS' if raw_ok else 'FAIL'} "
        f"softmax={'PASS' if exp_ok else 'FAIL'} "
        f"sv_context={'PASS' if ctx_ok else 'FAIL'} "
        f"flatten={'PASS' if flat_ok else 'FAIL'}",
        flush=True,
    )
    return 0 if raw_ok and exp_ok and ctx_ok and flat_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
