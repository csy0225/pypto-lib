# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Persistent PyPTO MLP/tail worker for the vLLM-Ascend backend (Phase 20).

Topology B (one worker process per TP rank): pinned to one physical NPU card,
shares it with the paired vLLM TP worker. Loads this rank's weight slices from a
HF safetensors checkpoint once, uploads them to warm ``ChipWorker`` handles, and
serves over a Unix domain socket on the shared mount.

Ops (length-prefixed: 4-byte BE header_len | json header | raw payload bytes):
  - ``mlp``        : dense-layer SwiGLU per-rank partial  -> bf16 [rows, HIDDEN]
  - ``shared_mlp`` : MoE shared-expert SwiGLU per-rank partial -> bf16 [rows, HIDDEN]
  - ``routed``     : MoE routed-expert per-rank grouped SwiGLU (CSR-packed,
                     header carries offsets/counts [N_LOCAL_EXPERTS]) -> bf16 [rows, HIDDEN]
  - ``tail``       : final RMSNorm + LM-head per-rank vocab shard
                     -> fp32 [rows, VOCAB_LOCAL]
  - ``ping`` / ``shutdown``
  bf16 payloads are sent as int16 (reinterpret); fp32 payloads as fp32.

Kernels compute per-rank results only; vLLM keeps RMSNorm-before-MLP, the TP
all_reduce (dense), the MoE dispatch/combine (routed), and the vocab all-gather (tail).
The ``routed`` op is the co-resident @pl.jit form of vllm_routed_experts (validated
via _routed_jit_probe --device-run); it registers on the SAME ChipWorker as dense/
shared/tail (one process/card) so there is no @pl.program co-tenancy with vLLM Worker_TP.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import socket
import struct
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from models.step3p5._routed_jit_probe import routed_experts_jit  # noqa: E402
from models.step3p5.config import (  # noqa: E402
    BATCH,
    HIDDEN,
    INTERMEDIATE,
    INTERMEDIATE_LOCAL,
    SHARE_EXPERT_DIM,
    SHARE_EXPERT_DIM_LOCAL,
    TP_WORLD_SIZE,
    USER_BATCH_DYN,
    VOCAB,
    VOCAB_LOCAL,
)
from models.step3p5.vllm_dense_mlp import INTER_LOCAL, dense_swiglu_perrank  # noqa: E402
from models.step3p5.vllm_routed_experts import LOCAL_RECV_MAX, N_LOCAL_EXPERTS, _real_weights  # noqa: E402
from models.step3p5.vllm_shared_mlp import SHARE_INTER_LOCAL, shared_swiglu_perrank  # noqa: E402

_HDR = struct.Struct(">I")


def _recv_exactly(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed mid-message")
        buf.extend(chunk)
    return bytes(buf)


def _recv_msg(conn: socket.socket) -> tuple[dict, bytes]:
    (hlen,) = _HDR.unpack(_recv_exactly(conn, 4))
    header = json.loads(_recv_exactly(conn, hlen).decode("utf-8"))
    body_len = int(header.get("nbytes", 0))
    body = _recv_exactly(conn, body_len) if body_len else b""
    return header, body


def _send_msg(conn: socket.socket, payload: bytes) -> None:
    conn.sendall(_HDR.pack(len(payload)))
    if payload:
        conn.sendall(payload)


def _open_ckpt(ckpt: str):
    index_files = glob.glob(os.path.join(ckpt, "*.index.json"))
    if not index_files:
        raise FileNotFoundError(f"no *.index.json under {ckpt}")
    weight_map = json.load(open(index_files[0]))["weight_map"]
    shard_cache: dict[str, object] = {}

    def _get(key: str) -> torch.Tensor:
        from safetensors import safe_open  # noqa: PLC0415
        if key not in weight_map:
            raise KeyError(f"{key} not in checkpoint index")
        shard = weight_map[key]
        f = shard_cache.get(shard)
        if f is None:
            f = safe_open(os.path.join(ckpt, shard), framework="pt", device="cpu")
            shard_cache[shard] = f
        return f.get_tensor(key)

    return _get


def _load_rank_dense_weights(ckpt, dense_layers, rank, tp):
    assert tp == TP_WORLD_SIZE
    assert INTER_LOCAL == INTERMEDIATE // tp == INTERMEDIATE_LOCAL
    _get = _open_ckpt(ckpt)
    lo, hi = rank * INTER_LOCAL, (rank + 1) * INTER_LOCAL
    out = {}
    for li in dense_layers:
        p = f"model.layers.{li}.mlp"
        gate = _get(f"{p}.gate_proj.weight"); up = _get(f"{p}.up_proj.weight")
        down = _get(f"{p}.down_proj.weight")
        out[li] = {
            "w_gate": gate[lo:hi, :].t().contiguous().to(torch.bfloat16),
            "w_up": up[lo:hi, :].t().contiguous().to(torch.bfloat16),
            "w_down": down[:, lo:hi].t().contiguous().to(torch.bfloat16),
        }
    return out


def _load_rank_shared_weights(ckpt, moe_layers, rank, tp):
    assert tp == TP_WORLD_SIZE
    _get = _open_ckpt(ckpt)
    nat, pad = SHARE_EXPERT_DIM_LOCAL, SHARE_INTER_LOCAL  # 160, 256
    lo, hi = rank * nat, (rank + 1) * nat
    out = {}
    for li in moe_layers:
        p = f"model.layers.{li}.share_expert"
        gate = _get(f"{p}.gate_proj.weight"); up = _get(f"{p}.up_proj.weight")
        down = _get(f"{p}.down_proj.weight")
        wg = gate[lo:hi, :].t().contiguous().to(torch.bfloat16)
        wu = up[lo:hi, :].t().contiguous().to(torch.bfloat16)
        wd = down[:, lo:hi].t().contiguous().to(torch.bfloat16)
        w_gate = torch.zeros(HIDDEN, pad, dtype=torch.bfloat16); w_gate[:, :nat] = wg
        w_up = torch.zeros(HIDDEN, pad, dtype=torch.bfloat16); w_up[:, :nat] = wu
        w_down = torch.zeros(pad, HIDDEN, dtype=torch.bfloat16); w_down[:nat, :] = wd
        out[li] = {"w_gate": w_gate, "w_up": w_up, "w_down": w_down}
    return out


def _load_rank_routed_weights(ckpt, routed_layers, rank):
    """Per-rank routed-expert weights (dequantized W8A8 if the ckpt is quantized).

    Reuses vllm_routed_experts._real_weights: rank r owns global experts
    [r*N_LOCAL_EXPERTS .. r*N_LOCAL_EXPERTS+N_LOCAL_EXPERTS), transposed to the
    RoutedExperts orientation (w_gate/w_up [N_LOCAL_EXPERTS, HIDDEN, INTER],
    w_down [N_LOCAL_EXPERTS, INTER, HIDDEN]).
    """
    out = {}
    for li in routed_layers:
        out[li] = _real_weights(ckpt, li, rank)
    return out


def _load_rank_tail_weights(ckpt, rank, tp):
    """final_norm (replicated [1,HIDDEN] fp32) + per-rank lm_head shard
    [VOCAB_LOCAL, HIDDEN] bf16 (vocab-parallel rows)."""
    assert tp == TP_WORLD_SIZE
    assert VOCAB_LOCAL == VOCAB // tp
    _get = _open_ckpt(ckpt)
    final_norm = _get("model.norm.weight").reshape(1, HIDDEN).to(torch.float32).contiguous()
    lm = _get("lm_head.weight")  # [VOCAB, HIDDEN]
    lo, hi = rank * VOCAB_LOCAL, (rank + 1) * VOCAB_LOCAL
    lm_shard = lm[lo:hi, :].contiguous().to(torch.bfloat16)  # [VOCAB_LOCAL, HIDDEN]
    return {"final_norm": final_norm, "lm_head": lm_shard}


class _MlpService:
    def __init__(self, device, platform, dense_weights, shared_weights, tail_weights,
                 routed_weights=None):
        from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415
        from pypto.runtime import ChipWorker, RunConfig  # noqa: PLC0415

        set_backend_type({"a2a3": BackendType.Ascend910B, "a2a3sim": BackendType.Ascend910B,
                          "a5": BackendType.Ascend950, "a5sim": BackendType.Ascend950}[platform])
        self._dense = dense_weights
        self._shared = shared_weights or {}
        self._tail = tail_weights
        self._routed = routed_weights or {}
        dummy_h = torch.zeros(BATCH, HIDDEN, dtype=torch.bfloat16)
        dummy_out = torch.zeros(BATCH, HIDDEN, dtype=torch.bfloat16)
        dw = next(iter(dense_weights.values()))
        compiled_dense = dense_swiglu_perrank.compile(
            dummy_h, dw["w_gate"], dw["w_up"], dw["w_down"], dummy_out,
            config=RunConfig(platform=platform))
        self._worker = ChipWorker(config=RunConfig(platform=platform, device_id=device),
                                  runtime=compiled_dense.runtime_name)
        self._dense_handle = self._worker.register(compiled_dense)
        self._shared_handle = None
        if self._shared:
            sw = next(iter(self._shared.values()))
            cs = shared_swiglu_perrank.compile(dummy_h, sw["w_gate"], sw["w_up"],
                                               sw["w_down"], dummy_out,
                                               config=RunConfig(platform=platform))
            self._shared_handle = self._worker.register(cs)
        self._routed_handle = None
        if self._routed:
            rw = next(iter(self._routed.values()))
            dummy_x = torch.zeros(LOCAL_RECV_MAX, HIDDEN, dtype=torch.bfloat16)
            dummy_off = torch.zeros(N_LOCAL_EXPERTS, dtype=torch.int32)
            dummy_cnt = torch.zeros(N_LOCAL_EXPERTS, dtype=torch.int32)
            dummy_ry = torch.zeros(LOCAL_RECV_MAX, HIDDEN, dtype=torch.bfloat16)
            cr = routed_experts_jit.compile(
                dummy_x, dummy_off, dummy_cnt, rw["w_gate"], rw["w_up"], rw["w_down"],
                dummy_ry, config=RunConfig(platform=platform))
            self._routed_handle = self._worker.register(cr)
        self._tail_handle = None
        self._tail_worker = None
        if self._tail:
            from tests.step3p5.test_rms_lm_head import rms_lm_head_test  # noqa: PLC0415
            seq_lens = torch.full((USER_BATCH_DYN,), USER_BATCH_DYN, dtype=torch.int32)
            logits_out = torch.zeros(USER_BATCH_DYN, VOCAB_LOCAL, dtype=torch.float32)
            ct = rms_lm_head_test.compile(
                dummy_h, self._tail["final_norm"], self._tail["lm_head"],
                seq_lens, logits_out, config=RunConfig(platform=platform))
            # Shared ChipWorker (one per card; two ChipWorkers on one device
            # conflict -> code -1). Correctness depends on warm-runtime buffers
            # being large enough for the tail's working set (PTO2_RING_HEAP env).
            self._tail_handle = self._worker.register(ct)
            self._tail_seq_lens = seq_lens
        print(f"[worker] registered dense+shared+routed+tail on device {device} "
              f"(shared={'y' if self._shared else 'n'} routed={'y' if self._routed else 'n'} "
              f"tail={'y' if self._tail else 'n'})",
              flush=True)

    def _run_mlp(self, handle, weights, layer_idx, hidden):
        w = weights[layer_idx]
        rows = hidden.shape[0]
        result = torch.empty(rows, HIDDEN, dtype=torch.bfloat16)
        for r0 in range(0, rows, BATCH):
            r1 = min(r0 + BATCH, rows)
            tile = torch.zeros(BATCH, HIDDEN, dtype=torch.bfloat16)
            tile[: r1 - r0] = hidden[r0:r1]
            out = torch.zeros(BATCH, HIDDEN, dtype=torch.bfloat16)
            handle(tile, w["w_gate"], w["w_up"], w["w_down"], out)
            result[r0:r1] = out[: r1 - r0]
        return result

    def mlp_partial(self, layer_idx, hidden):
        return self._run_mlp(self._dense_handle, self._dense, layer_idx, hidden)

    def shared_partial(self, layer_idx, hidden):
        if self._shared_handle is None:
            raise RuntimeError("shared kernel not loaded")
        return self._run_mlp(self._shared_handle, self._shared, layer_idx, hidden)

    def routed_partial(self, layer_idx, hidden, offsets, counts):
        """Per-rank routed grouped-GEMM. hidden [rows<=LOCAL_RECV_MAX, HIDDEN] is
        pre-dispatched (sorted-by-local-expert); offsets/counts are the CSR (int lists,
        len N_LOCAL_EXPERTS). Returns [rows, HIDDEN] bf16."""
        if self._routed_handle is None:
            raise RuntimeError("routed kernel not loaded")
        w = self._routed[layer_idx]
        num_recv = hidden.shape[0]
        if num_recv > LOCAL_RECV_MAX:
            raise ValueError(f"num_recv {num_recv} > LOCAL_RECV_MAX {LOCAL_RECV_MAX}")
        xp = torch.zeros(LOCAL_RECV_MAX, HIDDEN, dtype=torch.bfloat16)
        xp[:num_recv] = hidden
        off = torch.tensor(offsets, dtype=torch.int32)
        cnt = torch.tensor(counts, dtype=torch.int32)
        y = torch.zeros(LOCAL_RECV_MAX, HIDDEN, dtype=torch.bfloat16)
        self._routed_handle(xp, off, cnt, w["w_gate"], w["w_up"], w["w_down"], y)
        return y[:num_recv].contiguous()

    def tail_logits(self, hidden):
        if self._tail_handle is None:
            raise RuntimeError("tail kernel not loaded")
        rows = hidden.shape[0]
        result = torch.empty(rows, VOCAB_LOCAL, dtype=torch.float32)
        fn = self._tail["final_norm"]; lm = self._tail["lm_head"]
        for r0 in range(0, rows, BATCH):
            r1 = min(r0 + BATCH, rows)
            tile = torch.zeros(BATCH, HIDDEN, dtype=torch.bfloat16)
            tile[: r1 - r0] = hidden[r0:r1]
            out = torch.zeros(USER_BATCH_DYN, VOCAB_LOCAL, dtype=torch.float32)
            ret = self._tail_handle(tile, fn, lm, self._tail_seq_lens, out)
            src = ret if isinstance(ret, torch.Tensor) else out
            result[r0:r1] = src[: r1 - r0]
        return result

    def close(self):
        self._worker.close()
        if self._tail_worker is not None:
            self._tail_worker.close()


def _send_bf16(conn, t):
    _send_msg(conn, t.contiguous().view(torch.int16).view(-1).numpy().tobytes())


def _send_fp32(conn, t):
    _send_msg(conn, t.contiguous().to(torch.float32).view(-1).numpy().tobytes())


def _serve(sock_path, svc):
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path); srv.listen(8); os.chmod(sock_path, 0o777)
    print(f"[worker] listening on {sock_path}", flush=True)
    try:
        while True:
            conn, _ = srv.accept()
            try:
                while True:
                    try:
                        header, body = _recv_msg(conn)
                    except (ConnectionError, struct.error):
                        break
                    op = header.get("op")
                    if op == "ping":
                        _send_msg(conn, json.dumps({"ok": True}).encode()); continue
                    if op == "shutdown":
                        _send_msg(conn, json.dumps({"ok": True}).encode()); return
                    if op in ("mlp", "shared_mlp"):
                        rows = int(header["rows"]); layer = int(header["layer"])
                        hidden = torch.frombuffer(bytearray(body), dtype=torch.bfloat16).view(rows, HIDDEN)
                        part = svc.mlp_partial(layer, hidden) if op == "mlp" else svc.shared_partial(layer, hidden)
                        _send_bf16(conn, part); continue
                    if op == "routed":
                        rows = int(header["rows"]); layer = int(header["layer"])
                        hidden = torch.frombuffer(bytearray(body), dtype=torch.bfloat16).view(rows, HIDDEN)
                        part = svc.routed_partial(layer, hidden, header["offsets"], header["counts"])
                        _send_bf16(conn, part); continue
                    if op == "tail":
                        rows = int(header["rows"])
                        hidden = torch.frombuffer(bytearray(body), dtype=torch.bfloat16).view(rows, HIDDEN)
                        _send_fp32(conn, svc.tail_logits(hidden)); continue
                    _send_msg(conn, json.dumps({"error": f"bad op {op}"}).encode())
            finally:
                conn.close()
    finally:
        srv.close()
        if os.path.exists(sock_path):
            os.unlink(sock_path)
        svc.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=int, required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--tp", type=int, default=TP_WORLD_SIZE)
    ap.add_argument("--dense-layers", default="0,1,2")
    ap.add_argument("--moe-layers", default="")
    ap.add_argument("--routed-layers", default="",
                    help="MoE layers whose routed experts this worker serves (op=routed)")
    ap.add_argument("--tail", action="store_true")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--socket", required=True)
    ap.add_argument("--platform", default="a2a3")
    args = ap.parse_args()

    dense_layers = [int(x) for x in args.dense_layers.split(",") if x != ""]
    moe_layers = [int(x) for x in args.moe_layers.split(",") if x != ""]
    routed_layers = [int(x) for x in args.routed_layers.split(",") if x != ""]
    t0 = time.perf_counter()
    dense_w = _load_rank_dense_weights(args.ckpt, dense_layers, args.rank, args.tp)
    shared_w = (_load_rank_shared_weights(args.ckpt, moe_layers, args.rank, args.tp)
                if moe_layers else None)
    routed_w = (_load_rank_routed_weights(args.ckpt, routed_layers, args.rank)
                if routed_layers else None)
    tail_w = _load_rank_tail_weights(args.ckpt, args.rank, args.tp) if args.tail else None
    print(f"[worker] rank{args.rank} loaded dense{dense_layers} + shared({len(moe_layers)}) "
          f"+ routed({len(routed_layers)}) + tail({'y' if tail_w else 'n'}) "
          f"in {time.perf_counter()-t0:.1f}s", flush=True)
    svc = _MlpService(args.device, args.platform, dense_w, shared_w, tail_w, routed_w)
    _serve(args.socket, svc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
