# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Standalone per-rank routed-expert grouped SwiGLU kernel for the vLLM-Ascend backend.

vLLM-Ascend's ``FusedMoE`` (Step3p5 ``SharedFusedMoE``) computes the routed
experts by: gate/topk -> all-to-all dispatch -> per-rank grouped GEMM over the
rank's local experts -> all-to-all combine. This kernel reproduces *only* the
per-rank grouped GEMM (the ``down(silu(gate(x)) * up(x))`` over the
``MOE_NUM_EXPERTS_LOCAL`` local experts), consuming CSR-packed dispatched token
rows and producing raw per-token expert outputs BEFORE the topk-weight combine.

It contains NO collective -- exactly the per-rank seam that vLLM's dispatch
(all-to-all in) and combine (topk-weight + all-to-all out) wrap around, mirroring
how ``vllm_dense_mlp`` / ``vllm_shared_mlp`` replace their per-rank SwiGLU compute
while vLLM keeps RMSNorm + the collectives.

The grouped-GEMM body is copied from ``expert_routed._build_expert_routed`` (plain
SiLU, ``swiglu_limit=0``); it is self-contained here (no cross-module inline) so a
thin same-module ``@pl.jit`` wrapper can drive it (closure-factory inline bodies are
not externally callable, see project memory).

Usage::

    cd /data/chensiyu/hw_project/pypto/workspace/pypto-lib
    python -m models.step3p5.vllm_routed_experts --smoke
    python -m models.step3p5.vllm_routed_experts -p a2a3 -d 8
    python -m models.step3p5.vllm_routed_experts -p a2a3 -d 8 \
        --real-weights --ckpt /mnt/nvme1/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp --layer 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from models.step3p5.config import (
    BATCH,
    EP_WORLD_SIZE,
    HIDDEN,
    MOE_INTERMEDIATE,
    MOE_NUM_EXPERTS_LOCAL,
    MOE_TOP_K,
)

T = BATCH
N_LOCAL_EXPERTS = MOE_NUM_EXPERTS_LOCAL  # 36
TOPK = MOE_TOP_K
INTER = MOE_INTERMEDIATE  # 1280
LOCAL_RECV_MAX = EP_WORLD_SIZE * T * TOPK  # 1024

GATE_K_CHUNK = 64
GATE_N_CHUNK = 64
DOWN_K_CHUNK = 64
DOWN_N_CHUNK = 128
# Per-tile row count: [RECV_TILE, INTER] BF16 keeps the working tile inside the
# ~192 KB Vec/UB budget ([1024, 1280] FP32 = 5.2 MB overflows by ~28x). Mirrors
# moe.py's production RECV-tiled routed body (the un-tiled expert_routed.py body
# only lives in GM inside the larger EpTpMoE scope and overflows UB standalone).
RECV_TILE = 32
N_RECV_TILES = LOCAL_RECV_MAX // RECV_TILE  # 32
assert LOCAL_RECV_MAX % RECV_TILE == 0
assert HIDDEN % GATE_K_CHUNK == 0
assert HIDDEN % DOWN_N_CHUNK == 0
assert INTER % GATE_N_CHUNK == 0
assert INTER % DOWN_K_CHUNK == 0


def _build_routed_experts_program(tp_size: int = 1):
    """Return a ``@pl.program`` wrapping the per-rank routed-expert body.

    The RECV-tiled routed FFN (plain SiLU) is inlined as a ``@pl.function(Inline)``
    method (mirroring ``moe.py``'s production ``_expert_routed``) so the per-tile
    ``[RECV_TILE, INTER]`` BF16 scratch fits UB. The routed FFN has NO collective,
    so a single-rank program == the per-rank vLLM hook (each vLLM TP rank drives
    its own worker); ``host_orch`` dispatches ``chip_orch`` per rank.
    """

    @pl.program
    class RoutedExperts:
        @pl.function(type=pl.FunctionType.Inline)
        def _expert_routed(
            self,
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
                    # Guard empty/tail tiles: submitting an expert kernel with
                    # tile_valid <= 0 triggers 507018 (matches decode_layer.py
                    # MoE fix). Balanced routing leaves ~31/32 tiles empty.
                    if tile_valid > 0:
                        h_bf16 = pl.create_tensor([RECV_TILE, INTER], dtype=pl.BF16)

                        for nb in pl.spmd(INTER // GATE_N_CHUNK, name_hint="vllm_routed_gate_up"):
                            n0 = nb * GATE_N_CHUNK
                            x0 = pl.slice(
                                local_routed_x, [RECV_TILE, GATE_K_CHUNK], [tile_offset, 0],
                                valid_shape=[tile_valid, GATE_K_CHUNK],
                            )
                            wg0_2d = pl.reshape(
                                pl.slice(w_gate, [1, GATE_K_CHUNK, GATE_N_CHUNK], [e, 0, n0]),
                                [GATE_K_CHUNK, GATE_N_CHUNK],
                            )
                            wu0_2d = pl.reshape(
                                pl.slice(w_up, [1, GATE_K_CHUNK, GATE_N_CHUNK], [e, 0, n0]),
                                [GATE_K_CHUNK, GATE_N_CHUNK],
                            )
                            gate_acc = pl.matmul(x0, wg0_2d, out_dtype=pl.FP32)
                            up_acc = pl.matmul(x0, wu0_2d, out_dtype=pl.FP32)
                            for kb in pl.range(1, HIDDEN // GATE_K_CHUNK):
                                k0 = kb * GATE_K_CHUNK
                                xk = pl.slice(
                                    local_routed_x, [RECV_TILE, GATE_K_CHUNK], [tile_offset, k0],
                                    valid_shape=[tile_valid, GATE_K_CHUNK],
                                )
                                wgk = pl.reshape(
                                    pl.slice(w_gate, [1, GATE_K_CHUNK, GATE_N_CHUNK], [e, k0, n0]),
                                    [GATE_K_CHUNK, GATE_N_CHUNK],
                                )
                                wuk = pl.reshape(
                                    pl.slice(w_up, [1, GATE_K_CHUNK, GATE_N_CHUNK], [e, k0, n0]),
                                    [GATE_K_CHUNK, GATE_N_CHUNK],
                                )
                                gate_acc = pl.matmul_acc(gate_acc, xk, wgk)
                                up_acc = pl.matmul_acc(up_acc, xk, wuk)

                            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                            silu = pl.mul(gate_acc, sigmoid)
                            gated = pl.mul(silu, up_acc)
                            gated_v = pl.set_validshape(gated, tile_valid, GATE_N_CHUNK)
                            h_bf16[:, n0 : n0 + GATE_N_CHUNK] = pl.cast(
                                gated_v, target_type=pl.BF16,
                            )

                        for db in pl.spmd(HIDDEN // DOWN_N_CHUNK, name_hint="vllm_routed_down"):
                            d0 = db * DOWN_N_CHUNK
                            h0 = pl.slice(
                                h_bf16, [RECV_TILE, DOWN_K_CHUNK], [0, 0],
                                valid_shape=[tile_valid, DOWN_K_CHUNK],
                            )
                            wd0 = pl.reshape(
                                pl.slice(w_down, [1, DOWN_K_CHUNK, DOWN_N_CHUNK], [e, 0, d0]),
                                [DOWN_K_CHUNK, DOWN_N_CHUNK],
                            )
                            y_acc = pl.matmul(h0, wd0, out_dtype=pl.FP32)
                            for kb2 in pl.range(1, INTER // DOWN_K_CHUNK):
                                k0 = kb2 * DOWN_K_CHUNK
                                hk = pl.slice(
                                    h_bf16, [RECV_TILE, DOWN_K_CHUNK], [0, k0],
                                    valid_shape=[tile_valid, DOWN_K_CHUNK],
                                )
                                wdk = pl.reshape(
                                    pl.slice(w_down, [1, DOWN_K_CHUNK, DOWN_N_CHUNK], [e, k0, d0]),
                                    [DOWN_K_CHUNK, DOWN_N_CHUNK],
                                )
                                y_acc = pl.matmul_acc(y_acc, hk, wdk)

                            y_v = pl.set_validshape(y_acc, tile_valid, DOWN_N_CHUNK)
                            y_m = pl.fillpad(y_v, pad_value=pl.PadValue.zero)
                            local_routed_y = pl.assemble(
                                local_routed_y, pl.cast(y_m, target_type=pl.BF16),
                                [tile_offset, d0],
                            )

            return local_routed_y

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            local_routed_x: pl.Tensor[[LOCAL_RECV_MAX, HIDDEN], pl.BF16],
            local_expert_offset: pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
            local_expert_count: pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
            w_gate: pl.Tensor[[N_LOCAL_EXPERTS, HIDDEN, INTER], pl.BF16],
            w_up: pl.Tensor[[N_LOCAL_EXPERTS, HIDDEN, INTER], pl.BF16],
            w_down: pl.Tensor[[N_LOCAL_EXPERTS, INTER, HIDDEN], pl.BF16],
            local_routed_y: pl.Out[pl.Tensor[[LOCAL_RECV_MAX, HIDDEN], pl.BF16]],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[LOCAL_RECV_MAX, HIDDEN], pl.BF16]:
            local_routed_y = self._expert_routed(
                local_routed_x, local_expert_offset, local_expert_count,
                w_gate, w_up, w_down, local_routed_y,
            )
            return local_routed_y

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            local_routed_x: pl.Tensor[[tp_size, LOCAL_RECV_MAX, HIDDEN], pl.BF16],
            local_expert_offset: pl.Tensor[[tp_size, N_LOCAL_EXPERTS], pl.INT32],
            local_expert_count: pl.Tensor[[tp_size, N_LOCAL_EXPERTS], pl.INT32],
            w_gate: pl.Tensor[[tp_size, N_LOCAL_EXPERTS, HIDDEN, INTER], pl.BF16],
            w_up: pl.Tensor[[tp_size, N_LOCAL_EXPERTS, HIDDEN, INTER], pl.BF16],
            w_down: pl.Tensor[[tp_size, N_LOCAL_EXPERTS, INTER, HIDDEN], pl.BF16],
            local_routed_y: pl.Out[
                pl.Tensor[[tp_size, LOCAL_RECV_MAX, HIDDEN], pl.BF16]
            ],
        ):
            for r in pl.range(pld.world_size()):
                self.chip_orch(
                    local_routed_x[r],
                    local_expert_offset[r],
                    local_expert_count[r],
                    w_gate[r], w_up[r], w_down[r],
                    local_routed_y[r],
                    r,
                    device=r,
                )

    return RoutedExperts


def golden_routed_experts_perrank(
    local_routed_x: torch.Tensor,
    offsets: torch.Tensor,
    counts: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Torch reference: per-expert silu(x@wg)*(x@wu) @ wd, CSR-packed rows."""
    out = torch.zeros(LOCAL_RECV_MAX, HIDDEN, dtype=torch.float32)
    for e in range(N_LOCAL_EXPERTS):
        n = int(counts[e].item())
        if n == 0:
            continue
        o = int(offsets[e].item())
        x_sub = local_routed_x[o : o + n, :].float()      # [n, HIDDEN]
        gate = x_sub @ w_gate[e].float()                  # [n, INTER]
        up = x_sub @ w_up[e].float()
        h = (gate * torch.sigmoid(gate)) * up             # SiluAndMul
        h_bf = h.to(torch.bfloat16).float()
        y = h_bf @ w_down[e].float()                      # [n, HIDDEN]
        out[o : o + n, :] = y
    return out.to(torch.bfloat16)


def _balanced_csr(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic balanced CSR: T*TOPK*EP routes spread across 36 local experts."""
    n_routes = T * TOPK * EP_WORLD_SIZE  # 1024 = LOCAL_RECV_MAX (worst case)
    per_expert = n_routes // N_LOCAL_EXPERTS
    counts = torch.full((N_LOCAL_EXPERTS,), per_expert, dtype=torch.int32)
    rem = n_routes - per_expert * N_LOCAL_EXPERTS
    for e in range(rem):
        counts[e] += 1
    offsets = torch.zeros(N_LOCAL_EXPERTS, dtype=torch.int32)
    running = 0
    for e in range(N_LOCAL_EXPERTS):
        offsets[e] = running
        running += int(counts[e].item())
    return offsets, counts


def _synthetic_weights(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    wg = (torch.randn(N_LOCAL_EXPERTS, HIDDEN, INTER, generator=g) / HIDDEN ** 0.5).to(torch.bfloat16)
    wu = (torch.randn(N_LOCAL_EXPERTS, HIDDEN, INTER, generator=g) / HIDDEN ** 0.5).to(torch.bfloat16)
    wd = (torch.randn(N_LOCAL_EXPERTS, INTER, HIDDEN, generator=g) / INTER ** 0.5).to(torch.bfloat16)
    return {"w_gate": wg, "w_up": wu, "w_down": wd}


def _real_weights(ckpt: str, layer_idx: int, rank: int) -> dict[str, torch.Tensor]:
    """Load real dequantized W8A8 routed weights for one rank's 36 local experts.

    Rank ``r`` owns global experts ``[r*36 .. (r+1)*36)`` (block-cyclic EP).
    HF expert weights: gate/up ``[INTER, HIDDEN]``, down ``[HIDDEN, INTER]``;
    transposed to pypto layout gate/up ``[HIDDEN, INTER]``, down ``[INTER, HIDDEN]``.
    """
    from models.step3p5.weight_loader import (  # noqa: PLC0415
        _ShardCache, _load_quantized_expert_projector, _read_index,
    )

    weight_map = _read_index(ckpt)
    wg = torch.empty(N_LOCAL_EXPERTS, HIDDEN, INTER, dtype=torch.bfloat16)
    wu = torch.empty(N_LOCAL_EXPERTS, HIDDEN, INTER, dtype=torch.bfloat16)
    wd = torch.empty(N_LOCAL_EXPERTS, INTER, HIDDEN, dtype=torch.bfloat16)
    with _ShardCache(ckpt, weight_map) as cache:
        for le in range(N_LOCAL_EXPERTS):
            ge = rank * N_LOCAL_EXPERTS + le
            g = _load_quantized_expert_projector(cache, layer_idx, ge, "gate_proj")
            u = _load_quantized_expert_projector(cache, layer_idx, ge, "up_proj")
            d = _load_quantized_expert_projector(cache, layer_idx, ge, "down_proj")
            wg[le] = g.t().contiguous().to(torch.bfloat16)   # [INTER,HIDDEN]->[HIDDEN,INTER]
            wu[le] = u.t().contiguous().to(torch.bfloat16)
            wd[le] = d.t().contiguous().to(torch.bfloat16)   # [HIDDEN,INTER]->[INTER,HIDDEN]
    return {"w_gate": wg, "w_up": wu, "w_down": wd}


def _serve(sock_path: str, device: int, ckpt: str, layer: int, rank: int) -> int:
    """Minimal per-rank UDS worker serving the routed-expert grouped GEMM.

    Wraps the validated `_build_routed_experts_program` (ir.compile, one process/
    card) with real dequantized W8A8 experts for `layer`. Protocol mirrors
    `_stage_attn_worker.py`: 4-byte header len + JSON header + body. Op `routed`:
    body carries BF16 hidden [LOCAL_RECV_MAX, HIDDEN]; header carries int32
    `offsets`/`counts` [N_LOCAL_EXPERTS]; returns BF16 y [LOCAL_RECV_MAX, HIDDEN].
    The backend converts vLLM's group_list -> offsets/counts and dequants hidden.
    """
    import json  # noqa: PLC0415
    import os as _os  # noqa: PLC0415
    import socket  # noqa: PLC0415
    import struct  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from pypto import ir  # noqa: PLC0415
    from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415

    prog = _build_routed_experts_program(tp_size=1)
    compiled = ir.compile(
        prog, platform="a2a3",
        distributed_config=DistributedConfig(device_ids=[device], num_sub_workers=0),
        skip_ptoas=False,
    )
    w = _real_weights(ckpt, layer, rank)
    wg, wu, wd = w["w_gate"].unsqueeze(0), w["w_up"].unsqueeze(0), w["w_down"].unsqueeze(0)
    print(f"[routed-worker] compiled + loaded layer={layer} rank={rank} dev={device}", flush=True)

    HDR = struct.Struct("<I")

    def _recv(conn, n):
        b = b""
        while len(b) < n:
            c = conn.recv(n - len(b))
            if not c:
                raise ConnectionError
            b += c
        return b

    if _os.path.exists(sock_path):
        _os.unlink(sock_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(4)
    print(f"[routed-worker] listening on {sock_path}", flush=True)
    while True:
        conn, _ = srv.accept()
        try:
            while True:
                (hlen,) = HDR.unpack(_recv(conn, 4))
                header = json.loads(_recv(conn, hlen).decode())
                blen = header.get("body_len", 0)
                body = _recv(conn, blen) if blen else b""
                op = header.get("op")
                if op == "ping":
                    hb = json.dumps({"ok": True}).encode()
                    conn.sendall(HDR.pack(len(hb)) + hb)
                    continue
                if op == "routed":
                    x = torch.frombuffer(bytearray(body), dtype=torch.bfloat16).reshape(LOCAL_RECV_MAX, HIDDEN)
                    offs = torch.tensor(header["offsets"], dtype=torch.int32)
                    cnts = torch.tensor(header["counts"], dtype=torch.int32)
                    y = torch.zeros(LOCAL_RECV_MAX, HIDDEN, dtype=torch.bfloat16)
                    x1 = x.unsqueeze(0).contiguous()
                    o1 = offs.unsqueeze(0).contiguous()
                    c1 = cnts.unsqueeze(0).contiguous()
                    y1 = y.unsqueeze(0).contiguous()
                    compiled(x1, o1, c1, wg, wu, wd, y1)
                    ob = y1[0].contiguous().numpy().tobytes()
                    hb = json.dumps({"ok": True, "body_len": len(ob)}).encode()
                    conn.sendall(HDR.pack(len(hb)) + hb + ob)
                    continue
                hb = json.dumps({"error": f"bad op {op}"}).encode()
                conn.sendall(HDR.pack(len(hb)) + hb)
        except (ConnectionError, OSError):
            conn.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-p", "--platform", default="a2a3",
                   choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    p.add_argument("-d", "--device", type=str, default="8",
                   help="comma-separated device ids; routed FFN has no "
                        "collective so a single rank validates per-rank compute")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--smoke", action="store_true",
                   help="host-only compile (no device run)")
    p.add_argument("--real-weights", action="store_true")
    p.add_argument("--ckpt", default="/mnt/nvme1/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp")
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--serve", type=str, default=None, help="UDS socket path to serve `routed` op")
    return p.parse_args()


def main() -> int:
    import os
    import time

    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415

    set_backend_type({
        "a2a3": BackendType.Ascend910B,
        "a2a3sim": BackendType.Ascend910B,
        "a5": BackendType.Ascend950,
        "a5sim": BackendType.Ascend950,
    }[args.platform])

    device_ids = [int(d) for d in args.device.split(",")]
    n_ranks = len(device_ids)

    if args.serve:
        return _serve(args.serve, device_ids[0], args.ckpt, args.layer, args.rank)

    print(
        f"[vllm_routed_experts] N_LOCAL_EXPERTS={N_LOCAL_EXPERTS} HIDDEN={HIDDEN} "
        f"INTER={INTER} LOCAL_RECV_MAX={LOCAL_RECV_MAX} n_ranks={n_ranks} "
        f"real_weights={args.real_weights} layer={args.layer} rank={args.rank}",
        flush=True,
    )

    # Per-rank inputs (same synthetic CSR + x; weights per rank's expert shard).
    g = torch.Generator().manual_seed(args.seed)
    offsets, counts = _balanced_csr(args.seed)
    x = (torch.randn(LOCAL_RECV_MAX, HIDDEN, generator=g) * 0.3).to(torch.bfloat16)

    def rep(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(0).expand(n_ranks, *t.shape).contiguous()

    per_rank_w = []
    for i in range(n_ranks):
        rk = args.rank + i
        w = (_real_weights(args.ckpt, args.layer, rk)
             if args.real_weights else _synthetic_weights(args.seed + rk))
        per_rank_w.append(w)

    wg = torch.stack([w["w_gate"] for w in per_rank_w], dim=0).contiguous()
    wu = torch.stack([w["w_up"] for w in per_rank_w], dim=0).contiguous()
    wd = torch.stack([w["w_down"] for w in per_rank_w], dim=0).contiguous()

    x_d = rep(x)
    off_d = rep(offsets)
    cnt_d = rep(counts)
    y_out = torch.zeros(n_ranks, LOCAL_RECV_MAX, HIDDEN, dtype=torch.bfloat16)

    inputs = [x_d, off_d, cnt_d, wg, wu, wd, y_out]

    prog = _build_routed_experts_program(tp_size=n_ranks)

    from pypto import ir  # noqa: PLC0415
    from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415

    build_dir = f"/tmp/p_routed_run_d{args.device.replace(',', '_')}"
    os.environ["PYPTO_PROG_BUILD_DIR"] = build_dir
    os.makedirs(build_dir, exist_ok=True)

    print(f"[vllm_routed_experts] compiling RoutedExperts tp={n_ranks} "
          f"device_ids={device_ids}", flush=True)
    compiled = ir.compile(
        prog, platform=args.platform,
        distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
        skip_ptoas=(args.smoke or args.platform.endswith("sim")),
        dump_passes=False,
    )
    print(f"[vllm_routed_experts] compile OK => {compiled.output_dir}", flush=True)
    if args.smoke or args.platform.endswith("sim"):
        print("[vllm_routed_experts] SMOKE: COMPILE OK", flush=True)
        return 0

    t0 = time.time()
    compiled(*inputs)
    print(f"[vllm_routed_experts] run {time.time()-t0:.1f}s", flush=True)

    atol = rtol = 4e-2
    all_pass = True
    for i in range(n_ranks):
        ref = golden_routed_experts_perrank(
            x, offsets, counts,
            per_rank_w[i]["w_gate"], per_rank_w[i]["w_up"], per_rank_w[i]["w_down"],
        ).float()
        valid = counts.sum().item()
        mask = torch.zeros(LOCAL_RECV_MAX, dtype=torch.bool)
        run = 0
        for e in range(N_LOCAL_EXPERTS):
            n = int(counts[e].item())
            mask[run : run + n] = True
            run += n
        out_r = y_out[i].float()
        d = (out_r - ref).abs()
        thr = atol + rtol * ref.abs()
        bad = ((d > thr) & mask.unsqueeze(-1)).float().sum().item() / (valid * HIDDEN)
        ok = bad <= 0.06
        all_pass = all_pass and ok
        print(f"  rank {i} (dev {device_ids[i]}): bad_ratio={bad:.4f} "
              f"max|out|={out_r.abs().max():.3f} max|ref|={ref.abs().max():.3f} "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
    print(f"[vllm_routed_experts] {'PASS' if all_pass else 'FAIL'}", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
