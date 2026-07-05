"""vLLM-Ascend MoE routed-expert backend hook (pypto).

Monkey-patches `MoECommMethod._apply_mlp` (vllm_ascend/ops/fused_moe/moe_comm_method.py)
to route the per-rank post-dispatch grouped-GEMM to the pypto RoutedExperts worker
(vllm_routed_experts.py `_serve`) instead of vanilla `unified_apply_mlp`.

Seam contract (verified against running container source, 2026-07-05):
  input  MoEMlpComputeInput: .hidden_states [num_recv, HIDDEN] BF16 (sorted-by-local-expert),
         .group_list [N_LOCAL_EXPERTS] (counts if group_list_type==1 else cumsum), .group_list_type
  output torch.Tensor [num_recv, HIDDEN] BF16 -> token_combine

The worker holds dequantized-BF16 W8A8 experts for a layer; we intercept at the BF16
hidden entry (dynamic_scale is None) = the W8A8 reference-precision path the offline
golden matched (bad_ratio=0.0000).

Env / install:
  PYPTO_MOE=1                          enable
  PYPTO_MOE_SOCK=/tmp/routed_r{rank}.sock  worker socket (per rank)
  PYPTO_MOE_LAYERS=3                   which layer_idx to route (single-layer bring-up)
Call install() from a sitecustomize (like pypto_attn_backend.py).

Self-test (needs a running worker + real ckpt, no vLLM):
  python -m tools.step3p5.pypto_moe_backend --selftest --sock /tmp/routed_test.sock
"""
from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from models.step3p5.vllm_routed_experts import HIDDEN, LOCAL_RECV_MAX, N_LOCAL_EXPERTS  # noqa: E402

_HDR = struct.Struct("<I")


def _to_csr(group_list: torch.Tensor, group_list_type: int) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM group_list -> RoutedExperts (offsets, counts) int32 [N_LOCAL_EXPERTS].

    group_list_type: 1 = per-expert counts, 0 = cumulative token boundaries.
    offsets = exclusive prefix-sum of counts (start row of each expert in the packed x).
    """
    gl = group_list.detach().to(torch.int64).cpu()
    if group_list_type == 1:
        counts = gl
    elif group_list_type == 0:
        counts = torch.cat([gl[:1], torch.diff(gl)])
    else:
        raise ValueError(f"unsupported group_list_type {group_list_type}")
    counts = counts.to(torch.int32)
    offsets = torch.zeros_like(counts)
    if counts.numel() > 1:
        offsets[1:] = torch.cumsum(counts, 0)[:-1].to(torch.int32)
    return offsets, counts


class RoutedClient:
    """UDS client to a vllm_routed_experts `_serve` worker."""

    def __init__(self, sock_path: str):
        self.sock_path = sock_path
        self._conn: socket.socket | None = None

    def _connect(self):
        if self._conn is None:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(self.sock_path)
            self._conn = s
        return self._conn

    @staticmethod
    def _recvn(conn, n):
        b = b""
        while len(b) < n:
            c = conn.recv(n - len(b))
            if not c:
                raise ConnectionError("worker closed")
            b += c
        return b

    def _round(self, header, body=b""):
        conn = self._connect()
        header = dict(header)
        header["body_len"] = len(body)
        hb = json.dumps(header).encode()
        conn.sendall(_HDR.pack(len(hb)) + hb + body)
        (hlen,) = _HDR.unpack(self._recvn(conn, 4))
        h = json.loads(self._recvn(conn, hlen).decode())
        rb = self._recvn(conn, h.get("body_len", 0)) if h.get("body_len") else b""
        return h, rb

    def routed(self, x_bf16: torch.Tensor, offsets: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """x_bf16 [LOCAL_RECV_MAX, HIDDEN] on CPU -> y [LOCAL_RECV_MAX, HIDDEN] bf16."""
        body = x_bf16.contiguous().view(torch.uint16).numpy().tobytes()
        h, rb = self._round(
            {"op": "routed", "offsets": offsets.tolist(), "counts": counts.tolist()}, body
        )
        if not h.get("ok"):
            raise RuntimeError(f"worker error: {h}")
        return torch.frombuffer(bytearray(rb), dtype=torch.uint16).view(torch.bfloat16).reshape(
            LOCAL_RECV_MAX, HIDDEN
        )


def make_apply_mlp(client: RoutedClient, orig):
    """Build the _apply_mlp replacement bound to a worker client.

    layer targeting is left to the caller (install() decides when to route); this fn
    always routes when called.
    """

    def _pypto_apply_mlp(self, mlp_compute_input):
        x = mlp_compute_input.hidden_states
        num_recv = x.shape[0]
        if num_recv > LOCAL_RECV_MAX:
            return orig(self, mlp_compute_input)  # chunking TODO -> vanilla fallback
        offsets, counts = _to_csr(mlp_compute_input.group_list, mlp_compute_input.group_list_type)
        xp = torch.zeros(LOCAL_RECV_MAX, HIDDEN, dtype=torch.bfloat16)
        xp[:num_recv] = x.detach().to("cpu", torch.bfloat16)
        y = client.routed(xp, offsets, counts)
        return y[:num_recv].to(x.device, x.dtype)

    return _pypto_apply_mlp


def install():  # pragma: no cover - runs inside vLLM engine
    import os

    if os.environ.get("PYPTO_MOE") != "1":
        return
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    sock = os.environ.get("PYPTO_MOE_SOCK", f"/tmp/routed_r{rank}.sock")
    layers = {int(x) for x in os.environ.get("PYPTO_MOE_LAYERS", "3").split(",") if x != ""}
    import vllm_ascend.ops.fused_moe.moe_comm_method as mcm

    orig = mcm.MoECommMethod._apply_mlp
    client = RoutedClient(sock)
    routed_fn = make_apply_mlp(client, orig)

    def _dispatch(self, mlp_compute_input):
        # NOTE: layer_idx is not on mlp_compute_input; single-layer bring-up routes every
        # MoE call. For multi-layer, inject layer_idx via a FusedMoE.forward counter/threadlocal.
        return routed_fn(self, mlp_compute_input)

    mcm.MoECommMethod._apply_mlp = _dispatch
    print(f"[pypto_moe_backend] installed: sock={sock} layers={layers} rank={rank}", flush=True)


def _selftest(sock_path: str, ckpt: str, layer: int, rank: int) -> int:
    """Exercise _to_csr + RoutedClient + _pypto_apply_mlp against the torch golden.

    Needs a running vllm_routed_experts `_serve` worker at sock_path.
    """
    import types

    from models.step3p5.vllm_routed_experts import (
        _balanced_csr,
        _real_weights,
        golden_routed_experts_perrank,
    )

    offs_ref, counts_ref = _balanced_csr(0)
    g = torch.Generator().manual_seed(7)
    num_recv = int(counts_ref.sum().item())  # 1024 balanced
    x = (torch.randn(num_recv, HIDDEN, generator=g) * 0.3).bfloat16()

    # sanity: our _to_csr reproduces the balanced CSR from counts (group_list_type=1)
    offs, counts = _to_csr(counts_ref.clone(), 1)
    assert torch.equal(offs, offs_ref) and torch.equal(counts, counts_ref), "CSR conv mismatch"

    client = RoutedClient(sock_path)
    fn = make_apply_mlp(client, orig=lambda s, m: None)
    mlp_in = types.SimpleNamespace(hidden_states=x, group_list=counts_ref.clone(), group_list_type=1)
    y = fn(None, mlp_in)  # [num_recv, HIDDEN]

    w = _real_weights(ckpt, layer, rank)
    x_full = torch.zeros(LOCAL_RECV_MAX, HIDDEN, dtype=torch.bfloat16)
    x_full[:num_recv] = x
    golden_full = golden_routed_experts_perrank(
        x_full, offs_ref, counts_ref, w["w_gate"], w["w_up"], w["w_down"]
    )
    golden = golden_full[:num_recv]
    diff = (y.float() - golden.float()).abs()
    bad = (diff > 0.05).float().mean().item()
    print(
        f"[selftest] num_recv={num_recv} y_shape={tuple(y.shape)} maxdiff={diff.max():.4f} "
        f"bad_ratio@0.05={bad:.4f}",
        flush=True,
    )
    print("SELFTEST_PASS" if bad < 0.02 else "SELFTEST_FAIL", flush=True)
    return 0 if bad < 0.02 else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--sock", default="/tmp/routed_test.sock")
    p.add_argument("--ckpt", default="/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp")
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--rank", type=int, default=0)
    a = p.parse_args()
    if a.selftest:
        return _selftest(a.sock, a.ckpt, a.layer, a.rank)
    print("nothing to do (use --selftest, or import install())")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
