"""vLLM-Ascend MoE routed-expert backend hook (pypto).

Monkey-patches `MoECommMethod._apply_mlp` (vllm_ascend/ops/fused_moe/moe_comm_method.py)
to route the per-rank post-dispatch grouped-GEMM to the co-resident pypto routed worker
(tools/step3p5/pypto_mlp_worker.py `op=routed`) instead of vanilla `unified_apply_mlp`.

Wire protocol matches pypto_mlp_worker.py (co-resident @pl.jit worker, no @pl.program
co-tenancy): BE 4-byte header_len | json header (op/rows/layer/offsets/counts/nbytes) | body
(int16-reinterpreted bf16). Response = BE 4-byte payload_len | raw int16(bf16) bytes.

Seam contract (verified against running container source, 2026-07-05):
  input  MoEMlpComputeInput: .hidden_states [num_recv, HIDDEN] BF16 (sorted-by-local-expert),
         .group_list [N_LOCAL_EXPERTS] (counts if group_list_type==1 else cumsum), .group_list_type
  output torch.Tensor [num_recv, HIDDEN] BF16 -> token_combine

We intercept at the BF16 hidden entry (dynamic_scale None) = the W8A8 reference-precision path
the offline golden matched (bad_ratio=0.0000).

Env / install:
  PYPTO_MOE=1                          enable
  PYPTO_MOE_SOCK=/tmp/routed_r{rank}.sock  co-resident worker socket (per rank)
  PYPTO_MOE_LAYERS=3                   MoE layer_idx values to route (others stay vanilla)
Call install() from a sitecustomize (like pypto_attn_backend.py). install() also patches
FusedMoE.forward to track the current layer_idx (so only PYPTO_MOE_LAYERS route).

Self-test (needs a running pypto_mlp_worker --routed-layers L, no vLLM):
  python -m tools.step3p5.pypto_moe_backend --selftest --sock /tmp/mlpw_routed.sock --layer 3
"""
from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import threading
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

try:
    # host (has pypto) — authoritative constants
    from models.step3p5.vllm_routed_experts import HIDDEN, LOCAL_RECV_MAX, N_LOCAL_EXPERTS  # noqa: E402
except Exception:
    # vLLM engine container may lack pypto; these step3p5 constants are stable. install()/
    # RoutedClient/_apply_mlp only need the constants (no pypto). _selftest imports lazily.
    HIDDEN, LOCAL_RECV_MAX, N_LOCAL_EXPERTS = 4096, 1024, 36

_HDR = struct.Struct(">I")  # big-endian, matches pypto_mlp_worker.py

# --- current-layer tracking (so only target MoE layers route) ---
_TL = threading.local()


def set_current_layer(idx):
    _TL.layer = idx


def current_layer():
    return getattr(_TL, "layer", None)


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
    """UDS client to a co-resident pypto_mlp_worker `routed` op (BE / nbytes protocol)."""

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

    def _send(self, header, body=b""):
        conn = self._connect()
        header = dict(header)
        header["nbytes"] = len(body)
        hb = json.dumps(header).encode()
        conn.sendall(_HDR.pack(len(hb)) + hb + body)

    def _recv_payload(self):
        conn = self._connect()
        (plen,) = _HDR.unpack(self._recvn(conn, 4))
        return self._recvn(conn, plen)

    def routed(self, layer: int, x_bf16: torch.Tensor, offsets: torch.Tensor,
               counts: torch.Tensor) -> torch.Tensor:
        """x_bf16 [rows, HIDDEN] on CPU (rows<=LOCAL_RECV_MAX) -> y [rows, HIDDEN] bf16.
        Worker pads to LOCAL_RECV_MAX internally; offsets/counts describe the CSR."""
        rows = x_bf16.shape[0]
        body = x_bf16.contiguous().view(torch.int16).view(-1).numpy().tobytes()
        self._send({"op": "routed", "rows": rows, "layer": layer,
                    "offsets": offsets.tolist(), "counts": counts.tolist()}, body)
        rb = self._recv_payload()
        return torch.frombuffer(bytearray(rb), dtype=torch.int16).view(torch.bfloat16).reshape(rows, HIDDEN)


def make_apply_mlp(client: RoutedClient, orig, layers: set[int]):
    """Build the _apply_mlp replacement. Routes only when current_layer() is in *layers*
    (or, if layer tracking is unavailable and exactly one target layer is configured,
    routes every MoE call as a single-layer bring-up shortcut)."""

    single = len(layers) == 1
    only_layer = next(iter(layers)) if single else None

    def _pypto_apply_mlp(self, mlp_compute_input):
        cur = current_layer()
        if cur is not None:
            if cur not in layers:
                return orig(self, mlp_compute_input)
            layer = cur
        elif single:
            layer = only_layer  # bring-up shortcut: no tracking, single target layer
        else:
            return orig(self, mlp_compute_input)
        x = mlp_compute_input.hidden_states
        num_recv = x.shape[0]
        if num_recv > LOCAL_RECV_MAX:
            return orig(self, mlp_compute_input)  # chunking TODO -> vanilla fallback
        offsets, counts = _to_csr(mlp_compute_input.group_list, mlp_compute_input.group_list_type)
        xp = x.detach().to("cpu", torch.bfloat16)
        y = client.routed(layer, xp, offsets, counts)
        return y.to(x.device, x.dtype)

    return _pypto_apply_mlp


def _install_layer_tracking(layers):
    """Best-effort: wrap FusedMoE.forward to publish self.layer_idx into the threadlocal
    so _apply_mlp knows which layer it is serving. Guarded — if the attribute/class differ
    in this vLLM build, single-layer bring-up still works via the shortcut."""
    try:
        from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE as _FM  # type: ignore
    except Exception:
        try:
            from vllm.model_executor.layers.fused_moe.layer import FusedMoE as _FM  # type: ignore
        except Exception:
            print("[pypto_moe_backend] layer tracking unavailable (FusedMoE import failed)", flush=True)
            return
    if getattr(_FM, "_pypto_layer_wrapped", False):
        return
    _orig_fwd = _FM.forward

    def _tracked_forward(self, *a, **k):
        li = getattr(self, "layer_idx", None)
        if li is None:
            pfx = getattr(self, "prefix", "")
            for tok in str(pfx).split("."):
                if tok.isdigit():
                    li = int(tok); break
        set_current_layer(li)
        try:
            return _orig_fwd(self, *a, **k)
        finally:
            set_current_layer(None)

    _FM.forward = _tracked_forward
    _FM._pypto_layer_wrapped = True
    print("[pypto_moe_backend] FusedMoE.forward layer-tracking installed", flush=True)


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
    routed_fn = make_apply_mlp(client, orig, layers)
    mcm.MoECommMethod._apply_mlp = routed_fn
    _install_layer_tracking(layers)
    print(f"[pypto_moe_backend] installed: sock={sock} layers={layers} rank={rank}", flush=True)


def _selftest(sock_path: str, ckpt: str, layer: int, rank: int) -> int:
    """Exercise _to_csr + RoutedClient + _pypto_apply_mlp against the torch golden.
    Needs a running pypto_mlp_worker --routed-layers <layer> at sock_path."""
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

    offs, counts = _to_csr(counts_ref.clone(), 1)
    assert torch.equal(offs, offs_ref) and torch.equal(counts, counts_ref), "CSR conv mismatch"

    client = RoutedClient(sock_path)
    fn = make_apply_mlp(client, orig=lambda s, m: None, layers={layer})
    set_current_layer(layer)
    mlp_in = types.SimpleNamespace(hidden_states=x, group_list=counts_ref.clone(), group_list_type=1)
    y = fn(None, mlp_in)  # [num_recv, HIDDEN]
    set_current_layer(None)

    w = _real_weights(ckpt, layer, rank)
    x_full = torch.zeros(LOCAL_RECV_MAX, HIDDEN, dtype=torch.bfloat16)
    x_full[:num_recv] = x
    golden = golden_routed_experts_perrank(
        x_full, offs_ref, counts_ref, w["w_gate"], w["w_up"], w["w_down"]
    )[:num_recv]
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
    p.add_argument("--sock", default="/tmp/mlpw_routed.sock")
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
