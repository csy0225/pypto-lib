# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""N=1 whole-net prefill sidecar: resident holder behind an AF_UNIX socket.

This module is the prefill dual of ``tools/step3p5/whole_decode_sidecar.py``.
The framing protocol (``send_frame`` / ``recv_frame`` / ``_to_bytes`` /
``_from_bytes`` / ``_recvall`` / ``_validate_outgoing_tensors`` / ``_DT`` /
``_PROTOCOL_VERSION = 2``) is copied verbatim from the decode sidecar: it is
pure length-prefixed self-describing wire format and is holder-agnostic.

live single-handoff architecture (mirrors decode sidecar + phases/20 §G):
one resident pypto process (8 chip children, ``SIMPLER_COMM_NO_HCCL=1``
co-resident with vLLM) holds ``WholePrefillHolder`` (build+prepare once) and
exposes one AF_UNIX socket; the monkey-patched vLLM ``Step3p5Model.forward``
sends hidden + attn-meta for each single-sequence prefill step, and the
sidecar runs the whole-net prefill forward and returns next_hidden.

**Protocol** (length-prefixed self-describing, mirror of decode sidecar):
one frame = ``<I hdr_len>``(4B big-endian) + hdr_json(utf8) + tensor blobs
concatenated in order. hdr = {"version":2, "order":[name,...],
"tensors":{name:{shape,dtype,nbytes}}, "meta":{...}}. Tensor serialization
uses a uint8 byte-view (bf16-safe; numpy has no native bf16).

serve loop is decoupled into ``prefill_fn(meta, tensors) -> (out_meta,
out_tensors)``, so the protocol layer can be exercised with a stub echo
``prefill_fn`` for an offline round-trip self-test (``--selftest``; no device
/ no whole-net hang). The production ``prefill_fn`` is bound to the holder by
``run_sidecar``.

Prefill differs from decode in three load-bearing ways (see
``tools/step3p5/vllm_prefill_metadata.py``):

* the program token tensor is ``[PREFILL_T=128, HIDDEN=4096]`` — every row is
  a real token of one sequence, not one paged request row;
* ``positions`` and ``slot_mapping`` are per-token (length ``PREFILL_T``),
  while ``seq_lens`` and ``block_table`` are per-request (one prefill
  request, ``PREFILL_BATCH=1``);
* ``block_table`` is the single sequence's flat 1-D block list
  (``[max_blocks]`` INT32), not the decode ``[storage_batch, max_blocks]``
  2-D paged table.

Socket contract (the client/monkey-patch side reads the same env):
``PYPTO_WHOLE_PREFILL_SOCK`` (default ``/logs/pypto_whole_prefill.sock``).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import torch

from models.step3p5.prefill_qkv_proj_rope import PREFILL_T as _PREFILL_T
from tools.step3p5.kv_padding import (
    BLOCK_SIZE,
    PaddingReserve,
    PaddingReserveError,
    parse_padding_reserve,
)

_DT = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
}
_PROTOCOL_VERSION = 2
_MAX_HEADER_BYTES = 1 << 20
_MAX_TENSOR_BYTES = 1 << 30
_MAX_FRAME_BYTES = 2 << 30
_HIDDEN = 4096


class FrameProtocolError(ValueError):
    """Malformed or unsupported sidecar frame."""


class SidecarPrefillError(RuntimeError):
    """The sidecar returned a structured prefill failure."""


# ---- tensor <-> bytes (bf16-safe) ------------------------------------------

def _to_bytes(t: torch.Tensor) -> bytes:
    # numpy does not support bfloat16; route through a uint8 byte-view for
    # every dtype (correct for bf16 in particular).
    return t.detach().contiguous().cpu().flatten().view(torch.uint8).numpy().tobytes()


def _from_bytes(blob: bytes, dtype: torch.dtype, shape) -> torch.Tensor:
    # bytearray -> writable buffer (avoids frombuffer read-only warning);
    # clone owns the memory.
    return torch.frombuffer(bytearray(blob), dtype=dtype).reshape(shape).clone()


def _validate_outgoing_tensors(tensors: dict) -> None:
    if not isinstance(tensors, dict):
        raise FrameProtocolError("tensors must be a dictionary")
    for name, tensor in tensors.items():
        if not isinstance(name, str) or not name:
            raise FrameProtocolError("tensor names must be non-empty strings")
        if not isinstance(tensor, torch.Tensor):
            raise FrameProtocolError(f"{name}: expected torch.Tensor")
        dtype_name = str(tensor.dtype).removeprefix("torch.")
        if dtype_name not in _DT:
            raise FrameProtocolError(f"{name}: unsupported dtype {tensor.dtype}")
        nbytes = int(tensor.numel() * tensor.element_size())
        if nbytes > _MAX_TENSOR_BYTES:
            raise FrameProtocolError(f"{name}: tensor is too large ({nbytes} bytes)")


# ---- framing ----------------------------------------------------------------

def _recvall(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(f"peer closed after {len(buf)}/{n} bytes")
        buf += chunk
    return bytes(buf)


def send_frame(conn: socket.socket, tensors: dict, meta: dict | None = None) -> None:
    _validate_outgoing_tensors(tensors)
    order = list(tensors.keys())
    hdr = {
        "version": _PROTOCOL_VERSION,
        "order": order,
        "tensors": {
            n: {
                "shape": list(t.shape),
                "dtype": str(t.dtype).removeprefix("torch."),
                "nbytes": int(t.numel() * t.element_size()),
            }
            for n, t in tensors.items()
        },
        "meta": meta or {},
    }
    hbytes = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    if len(hbytes) > _MAX_HEADER_BYTES:
        raise FrameProtocolError(f"header is too large ({len(hbytes)} bytes)")
    conn.sendall(struct.pack(">I", len(hbytes)) + hbytes)
    for n in order:
        conn.sendall(_to_bytes(tensors[n]))


def recv_frame(conn: socket.socket):
    (hlen,) = struct.unpack(">I", _recvall(conn, 4))
    if hlen <= 0 or hlen > _MAX_HEADER_BYTES:
        raise FrameProtocolError(f"invalid header length {hlen}")
    hdr = json.loads(_recvall(conn, hlen).decode("utf-8"))
    if not isinstance(hdr, dict) or hdr.get("version") != _PROTOCOL_VERSION:
        raise FrameProtocolError(
            f"unsupported frame version {getattr(hdr, 'get', lambda *_: None)('version')}"
        )
    order = hdr.get("order")
    specs = hdr.get("tensors")
    if not isinstance(order, list) or not all(isinstance(name, str) and name for name in order):
        raise FrameProtocolError("header order must be a list of tensor names")
    if len(order) != len(set(order)):
        raise FrameProtocolError("header order contains duplicate tensor names")
    if not isinstance(specs, dict) or set(specs) != set(order):
        raise FrameProtocolError("header tensor specs do not match order")
    tensors = {}
    total_nbytes = 0
    for n in order:
        spec = specs[n]
        if not isinstance(spec, dict):
            raise FrameProtocolError(f"{n}: tensor spec must be an object")
        dtype_name = spec.get("dtype")
        if dtype_name not in _DT:
            raise FrameProtocolError(f"{n}: unsupported dtype {dtype_name!r}")
        shape = spec.get("shape")
        if not isinstance(shape, list) or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim < 0
            for dim in shape
        ):
            raise FrameProtocolError(f"{n}: invalid shape {shape!r}")
        nbytes = spec.get("nbytes")
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0:
            raise FrameProtocolError(f"{n}: invalid nbytes {nbytes!r}")
        expected_numel = 1
        for dim in shape:
            expected_numel *= dim
        expected_nbytes = expected_numel * torch.empty((), dtype=_DT[dtype_name]).element_size()
        if nbytes != expected_nbytes:
            raise FrameProtocolError(
                f"{n}: nbytes={nbytes} does not match shape/dtype bytes={expected_nbytes}"
            )
        if nbytes > _MAX_TENSOR_BYTES:
            raise FrameProtocolError(f"{n}: tensor is too large ({nbytes} bytes)")
        total_nbytes += nbytes
        if total_nbytes > _MAX_FRAME_BYTES:
            raise FrameProtocolError("frame tensor payload is too large")
        blob = _recvall(conn, nbytes)
        tensors[n] = _from_bytes(blob, _DT[dtype_name], shape)
    return hdr.get("meta", {}), tensors


@dataclass(frozen=True)
class PrefillRequest:
    """Validated single-sequence prefill request (prior-art ABI, PREFILL_BATCH=1).

    Field shapes differ from ``DecodeRequest`` because prefill is
    single-sequence, not paged: ``hidden`` is the full ``[PREFILL_T, HIDDEN]``
    capacity (active rows = ``valid_tokens``), ``seq_lens`` is per-request
    (length 1), ``positions`` / ``slot_mapping`` are per-token (length
    ``PREFILL_T``), and each group's ``block_table`` is the one sequence's
    flat 1-D block list.
    """

    valid_tokens: int
    valid_requests: int
    kv_group_ids: tuple[int, ...]
    hidden: torch.Tensor
    seq_lens: torch.Tensor
    positions: torch.Tensor
    block_tables: dict[int, torch.Tensor]
    slot_mappings: dict[int, torch.Tensor]
    padding_reserve: PaddingReserve


def _meta_int(meta: dict, name: str) -> int:
    value = meta.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise FrameProtocolError(f"prefill meta field {name!r} must be an integer")
    return value


def validate_prefill_request(meta: dict, tensors: dict) -> PrefillRequest:
    """Validate the stable single-sequence prefill request ABI and its padding.

    Mirrors :func:`validate_decode_request` structure but uses prefill shapes
    (per-token positions/slot_mapping, per-request seq_lens, flat 1-D
    block_table). All checks are fail-closed.
    """
    if not isinstance(meta, dict):
        raise FrameProtocolError("prefill meta must be an object")
    if meta.get("protocol_version") != _PROTOCOL_VERSION or meta.get("op") != "prefill":
        raise FrameProtocolError("request must be protocol v2 op=prefill")
    valid_tokens = _meta_int(meta, "valid_tokens")
    valid_requests = _meta_int(meta, "valid_requests")
    prefill_t = _meta_int(meta, "prefill_t")
    group_count = _meta_int(meta, "kv_group_count")
    if prefill_t != _PREFILL_T:
        raise FrameProtocolError(
            f"prefill_t={prefill_t} != compiled capacity {_PREFILL_T}"
        )
    if not (1 <= valid_tokens <= _PREFILL_T):
        raise FrameProtocolError(
            f"valid_tokens must be 1..{_PREFILL_T}, got {valid_tokens}"
        )
    # First version: single prefill request (PREFILL_BATCH == 1).
    if valid_requests != 1:
        raise FrameProtocolError(
            f"prefill first version supports one request (PREFILL_BATCH=1), "
            f"got valid_requests={valid_requests}"
        )
    if group_count <= 0:
        raise FrameProtocolError("kv_group_count must be positive")
    query_lengths = meta.get("query_lengths")
    if (
        not isinstance(query_lengths, list)
        or len(query_lengths) != valid_requests
        or any(
            isinstance(length, bool) or not isinstance(length, int) or length <= 0
            for length in query_lengths
        )
    ):
        raise FrameProtocolError(
            f"query_lengths must be a list of {valid_requests} positive ints"
        )
    # Prefill query is multi-token; sum(query_lengths) == valid_tokens.
    if sum(query_lengths) != valid_tokens:
        raise FrameProtocolError(
            f"query_lengths sum to {sum(query_lengths)}, "
            f"valid_tokens={valid_tokens}"
        )
    layer_to_group = meta.get("layer_to_group")
    if (
        not isinstance(layer_to_group, list)
        or len(layer_to_group) != 45
        or any(
            isinstance(group, bool) or not isinstance(group, int) or group < 0
            for group in layer_to_group
        )
    ):
        raise FrameProtocolError("layer_to_group must contain 45 non-negative group ids")
    group_ids = tuple(sorted(set(layer_to_group)))
    if len(group_ids) != group_count:
        raise FrameProtocolError("kv_group_count does not match layer_to_group")
    try:
        padding_reserve = parse_padding_reserve(
            meta.get("padding_reserve"),
            where="prefill padding_reserve",
        )
    except PaddingReserveError as exc:
        raise FrameProtocolError(str(exc)) from exc

    required = {"hidden", "meta_seq_lens", "meta_positions"}
    for group_id in group_ids:
        required.add(f"meta_block_table_g{group_id}")
        required.add(f"meta_slot_mapping_g{group_id}")
    if set(tensors) != required:
        raise FrameProtocolError(
            f"prefill tensor set mismatch; "
            f"missing={sorted(required - set(tensors))} "
            f"extra={sorted(set(tensors) - required)}"
        )

    hidden = tensors["hidden"]
    if hidden.dtype != torch.bfloat16 or tuple(hidden.shape) != (_PREFILL_T, _HIDDEN):
        raise FrameProtocolError(
            f"hidden must be BF16 [{_PREFILL_T},{_HIDDEN}], got "
            f"{hidden.dtype} {tuple(hidden.shape)}"
        )
    # Active rows must be finite; padding rows are zeroed by the holder.
    if not torch.isfinite(hidden[:valid_tokens].float()).all():
        raise FrameProtocolError("hidden active rows contain NaN/Inf")

    seq_lens = tensors["meta_seq_lens"]
    positions = tensors["meta_positions"]
    if seq_lens.dtype != torch.int32 or tuple(seq_lens.shape) != (1,):
        raise FrameProtocolError("meta_seq_lens must be INT32 [1]")
    if positions.dtype != torch.int32 or tuple(positions.shape) != (_PREFILL_T,):
        raise FrameProtocolError(f"meta_positions must be INT32 [{_PREFILL_T}]")
    seq_len = int(seq_lens[0])
    if seq_len <= 0:
        raise FrameProtocolError("seq_lens must be positive")
    if seq_len < valid_tokens:
        raise FrameProtocolError(
            f"seq_len={seq_len} < valid_tokens={valid_tokens}"
        )
    # positions[:valid_tokens] must be contiguous ending at seq_len-1.
    # A fresh prompt (seq_len == valid_tokens) yields arange(valid_tokens).
    expected_positions = torch.arange(
        seq_len - valid_tokens, seq_len, dtype=torch.int32
    )
    if not torch.equal(positions[:valid_tokens], expected_positions):
        raise FrameProtocolError(
            f"positions[:valid_tokens] must be contiguous "
            f"[seq_len-T, seq_len) = [{seq_len - valid_tokens}, {seq_len}); "
            f"got {positions[:valid_tokens].tolist()}"
        )
    if torch.count_nonzero(positions[valid_tokens:]).item() != 0:
        raise FrameProtocolError("positions padding must be zero")

    block_tables: dict[int, torch.Tensor] = {}
    slot_mappings: dict[int, torch.Tensor] = {}
    active_blocks_needed = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    for group_id in group_ids:
        block = tensors[f"meta_block_table_g{group_id}"]
        slot = tensors[f"meta_slot_mapping_g{group_id}"]
        if block.dtype != torch.int32 or block.ndim != 1 or block.numel() <= 0:
            raise FrameProtocolError(
                f"group {group_id}: block_table must be INT32 1-D flat [max_blocks]"
            )
        if block.numel() < active_blocks_needed:
            raise FrameProtocolError(
                f"group {group_id}: block_table width {block.numel()} cannot "
                f"cover seq_len={seq_len} (needs {active_blocks_needed} blocks)"
            )
        active_table = block[:active_blocks_needed]
        if torch.any(active_table < 0) or torch.any(
            active_table >= padding_reserve.scheduler_num_blocks
        ):
            raise FrameProtocolError(
                f"group {group_id}: active block_table ids must lie in "
                f"scheduler domain [0,{padding_reserve.scheduler_num_blocks})"
            )
        # Inactive block_table columns (beyond the sequence) must be zero.
        if torch.count_nonzero(block[active_blocks_needed:]).item() != 0:
            raise FrameProtocolError(
                f"group {group_id}: inactive block_table columns must be zero"
            )
        if slot.dtype != torch.int32 or tuple(slot.shape) != (_PREFILL_T,):
            raise FrameProtocolError(
                f"group {group_id}: slot_mapping must be INT32 [{_PREFILL_T}]"
            )
        # slot_mapping[t] == block_table[pos // BLOCK_SIZE] * BLOCK_SIZE +
        # pos % BLOCK_SIZE for every active token.
        pos_long = positions[:valid_tokens].to(torch.long)
        cols = (pos_long // BLOCK_SIZE).to(torch.long)
        expected_slot = (
            active_table.index_select(0, cols) * BLOCK_SIZE
            + (pos_long % BLOCK_SIZE).to(torch.int32)
        )
        if not torch.equal(slot[:valid_tokens], expected_slot):
            raise FrameProtocolError(
                f"group {group_id}: slot_mapping does not match "
                f"block_table*BLOCK_SIZE+pos%BLOCK_SIZE"
            )
        # Padding slot_mapping points at the allocator-owned reserve block.
        if valid_tokens < _PREFILL_T:
            if not padding_reserve.padding_block_ids:
                raise FrameProtocolError(
                    "padding reserve has no allocator-owned block for prefill padding"
                )
            padding_block = padding_reserve.padding_block_ids[0]
            expected_pad_slot = torch.full(
                (_PREFILL_T - valid_tokens,),
                padding_block * BLOCK_SIZE,
                dtype=torch.int32,
            )
            if not torch.equal(slot[valid_tokens:], expected_pad_slot):
                raise FrameProtocolError(
                    f"group {group_id}: padding slot_mapping must point at "
                    f"reserve block {padding_block}"
                )
        block_tables[group_id] = block
        slot_mappings[group_id] = slot

    return PrefillRequest(
        valid_tokens=valid_tokens,
        valid_requests=valid_requests,
        kv_group_ids=group_ids,
        hidden=hidden,
        seq_lens=seq_lens,
        positions=positions,
        block_tables=block_tables,
        slot_mappings=slot_mappings,
        padding_reserve=padding_reserve,
    )


# ---- server -----------------------------------------------------------------

class WholePrefillServer:
    """Resident AF_UNIX socket server.

    ``prefill_fn(meta, tensors) -> (out_meta, out_tensors)``. Mirrors
    ``WholeDecodeServer`` exactly (bind / serve_forever / close, same
    error-frame handling).
    """

    def __init__(self, sock_path: str, prefill_fn):
        self.sock_path = sock_path
        self.prefill_fn = prefill_fn
        self._srv = None
        self._stop = threading.Event()

    def _bind(self):
        try:
            os.unlink(self.sock_path)
        except OSError:
            pass
        Path(self.sock_path).parent.mkdir(parents=True, exist_ok=True)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.settimeout(0.2)
        srv.bind(self.sock_path)
        srv.listen(4)
        self._srv = srv
        print(f"[sidecar] listening on {self.sock_path}", flush=True)

    def serve_forever(self):
        self._bind()
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break  # server socket closed (graceful shutdown)
            try:
                while True:
                    try:
                        meta, tensors = recv_frame(conn)
                    except (ConnectionError, struct.error):
                        break  # client closed this connection
                    try:
                        out_meta, out_tensors = self.prefill_fn(meta, tensors)
                    except Exception as exc:  # noqa: BLE001
                        send_frame(
                            conn,
                            {},
                            {
                                "protocol_version": _PROTOCOL_VERSION,
                                "op": "error",
                                "ok": False,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            },
                        )
                        continue
                    response_meta = dict(out_meta)
                    response_meta.setdefault("protocol_version", _PROTOCOL_VERSION)
                    response_meta.setdefault("ok", True)
                    send_frame(conn, out_tensors, response_meta)
            finally:
                conn.close()

    def close(self):
        self._stop.set()
        if self._srv is not None:
            self._srv.close()
            self._srv = None
        try:
            os.unlink(self.sock_path)
        except OSError:
            pass


# ---- client -----------------------------------------------------------------

class WholePrefillClient:
    """Client for the monkey-patched vLLM prefill forward: connect to the
    sidecar, send hidden + meta, receive next_hidden. Mirrors
    ``WholeDecodeClient`` exactly (connect / ``prefill`` / close)."""

    def __init__(self, sock_path: str, connect_timeout: float = 30.0):
        self.sock_path = sock_path
        self.connect_timeout = connect_timeout
        self._conn = None

    def connect(self):
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.settimeout(self.connect_timeout)
        conn.connect(self.sock_path)
        conn.settimeout(None)
        self._conn = conn
        return self

    def prefill(self, tensors: dict, meta: dict | None = None):
        if self._conn is None:
            self.connect()
        send_frame(self._conn, tensors, meta or {})
        out_meta, out_tensors = recv_frame(self._conn)
        if out_meta.get("ok") is False:
            raise SidecarPrefillError(
                f"{out_meta.get('error_type', 'PrefillError')}: "
                f"{out_meta.get('error', '')}"
            )
        return out_meta, out_tensors

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ---- production prefill_fn (bound to a resident holder) ---------------------

def make_holder_prefill_fn(holder):
    """Map a socket request to the holder: set hidden(+meta) -> run -> hidden.

    Mirrors :func:`make_holder_decode_fn`. request tensors:
      - ``hidden``: ``[PREFILL_T, HIDDEN]`` BF16 (full capacity; active rows =
        ``valid_tokens``).
    response tensors:
      - ``next_hidden``: ``[valid_tokens, HIDDEN]`` BF16 (active slice of the
        holder's ``next_hidden_out``).
    response meta:
      - ``{"op": "prefill_result", "valid_tokens": int, "dt_sec": float,
         "program": str}``
    """
    if not getattr(holder, "hidden_only", False):
        raise ValueError(
            "live prefill sidecar requires the hidden-only whole-net program; "
            "the diagnostic LM-head program is not a production ABI"
        )

    def prefill_fn(meta, tensors):
        request = validate_prefill_request(meta, tensors)
        if request.padding_reserve != getattr(
            holder,
            "padding_reserve",
            None,
        ):
            raise FrameProtocolError(
                "prefill request padding reserve does not match the "
                "holder-imported Main KV allocation"
            )
        # The current release whole-net signature has one block table and one
        # slot mapping. The v1 production launch therefore explicitly disables
        # vLLM's hybrid KV manager and requires one resolved KV group. The
        # wire protocol stays group-aware so a future generator-owned ABI can
        # extend this without another socket-version change.
        if len(request.kv_group_ids) != 1:
            raise FrameProtocolError(
                "current whole-net ABI requires one KV group; start vLLM with "
                "--disable-hybrid-kv-cache-manager"
            )
        group_id = request.kv_group_ids[0]
        holder.set_live_prompt(
            request.hidden,
            seq_lens=request.seq_lens,
            positions=request.positions,
            block_table=request.block_tables[group_id],
            slot_mapping=request.slot_mappings[group_id],
        )
        res = holder.run()
        next_hidden = res["next_hidden"][0, : request.valid_tokens, :].clone()
        if not torch.isfinite(next_hidden.float()).all():
            raise RuntimeError("whole-net returned NaN/Inf next_hidden")
        return {
            "op": "prefill_result",
            "valid_tokens": request.valid_tokens,
            "dt_sec": float(getattr(holder, "_last_run_sec", 0.0)),
            "program": holder.program_name,
        }, {"next_hidden": next_hidden}

    return prefill_fn


def run_sidecar(args) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    import logging  # noqa: PLC0415
    logging.getLogger("simpler").setLevel(15)
    from tools.step3p5.whole_prefill_holder import WholePrefillHolder  # noqa: PLC0415
    device_ids = [int(d) for d in str(args.device).split(",")]
    holder = WholePrefillHolder(
        device_ids=device_ids, out_dir=args.out, ckpt=args.ckpt,
        platform=args.platform, kv_ipc=args.kv_ipc,
    ).build()
    with holder:
        server = WholePrefillServer(args.sock, make_holder_prefill_fn(holder))
        print(
            f"[sidecar] holder resident; serving whole-prefill on {args.sock}",
            flush=True,
        )
        try:
            server.serve_forever()
        finally:
            server.close()
    return 0


# ---- offline protocol self-test (no device / no holder / no hang) ----------

def _selftest() -> int:
    """In-process AF_UNIX round-trip with a stub echo prefill_fn.

    Verifies the protocol + bf16 encode/decode and the prefill ABI validation
    (mirrors decode ``_selftest`` structure, but single-sequence prefill
    shapes: per-token positions/slot_mapping, flat 1-D block_table).
    """
    import tempfile
    import time
    from tools.step3p5.kv_padding import make_padding_reserve

    sock_path = os.path.join(tempfile.mkdtemp(), "wp_selftest.sock")

    def echo_prefill_fn(meta, tensors):
        request = validate_prefill_request(meta, tensors)
        # Simulate whole-net: return the active hidden rows * 2 (BF16) so the
        # round-trip is exactly checkable. The production holder returns the
        # full [tp, PREFILL_T, HIDDEN] next_hidden_out and make_holder_prefill
        # _fn slices [0, :valid_tokens, :]; the echo mirrors the sliced shape.
        active = request.hidden[: request.valid_tokens]
        return {
            "op": "prefill_result",
            "valid_tokens": request.valid_tokens,
            "dt_sec": 0.0,
        }, {"next_hidden": (active.float() * 2).to(torch.bfloat16)}

    srv = WholePrefillServer(sock_path, echo_prefill_fn)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    for _ in range(50):
        if os.path.exists(sock_path):
            break
        time.sleep(0.05)

    ok = True
    cli = WholePrefillClient(sock_path).connect()
    reserve = make_padding_reserve(8, 23)
    t = 64  # active tokens; exercises padding (PREFILL_T - t = 64 padding rows)
    max_blocks = 4
    seq_len = t  # fresh prompt: seq_len == valid_tokens
    block_table = torch.zeros(max_blocks, dtype=torch.int32)
    block_table[0] = 5  # scheduler block id < scheduler_num_blocks=8
    positions = torch.zeros(_PREFILL_T, dtype=torch.int32)
    positions[:t] = torch.arange(t, dtype=torch.int32)  # fresh prompt arange(t)
    slot_mapping = torch.zeros(_PREFILL_T, dtype=torch.int32)
    # slot_mapping[t] = block_table[pos // BLOCK_SIZE] * BLOCK_SIZE + pos %
    # BLOCK_SIZE; for pos in [0, t) this is 5 * 128 + pos.
    slot_mapping[:t] = block_table[0] * BLOCK_SIZE + torch.arange(t, dtype=torch.int32)
    # Padding tokens map at the allocator-owned reserve block (8 * 128).
    slot_mapping[t:] = reserve.padding_block_ids[0] * BLOCK_SIZE
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)
    req_meta = {
        "protocol_version": _PROTOCOL_VERSION,
        "op": "prefill",
        "valid_tokens": t,
        "valid_requests": 1,
        "prefill_t": _PREFILL_T,
        "kv_group_count": 1,
        "layer_to_group": [0] * 45,
        "query_lengths": [t],
        "padding_reserve": reserve.as_dict(),
    }

    for trial in range(3):
        hidden = torch.randn(_PREFILL_T, _HIDDEN, dtype=torch.bfloat16)
        out_meta, out = cli.prefill(
            {
                "hidden": hidden,
                "meta_seq_lens": seq_lens,
                "meta_positions": positions,
                "meta_block_table_g0": block_table,
                "meta_slot_mapping_g0": slot_mapping,
            },
            req_meta,
        )
        got = out["next_hidden"]
        exp = (hidden[:t].float() * 2).to(torch.bfloat16)
        same_shape = tuple(got.shape) == (t, _HIDDEN)
        max_err = (got.float() - exp.float()).abs().max().item()
        hidden_only = "argmax_debug" not in out_meta
        trial_ok = same_shape and max_err == 0.0 and hidden_only
        ok = ok and trial_ok
        print(
            f"[selftest] trial {trial}: shape_ok={same_shape} max_err={max_err} "
            f"hidden_only_meta={hidden_only} -> "
            f"{'PASS' if trial_ok else 'FAIL'}",
            flush=True,
        )

    # The server must reject invalid padding without dying.
    bad_positions = positions.clone()
    bad_positions[t] = 1  # padding positions must be zero
    try:
        cli.prefill(
            {
                "hidden": hidden,
                "meta_seq_lens": seq_lens,
                "meta_positions": bad_positions,
                "meta_block_table_g0": block_table,
                "meta_slot_mapping_g0": slot_mapping,
            },
            req_meta,
        )
    except SidecarPrefillError:
        print("[selftest] invalid padding rejected -> PASS", flush=True)
    else:
        print("[selftest] invalid padding rejected -> FAIL", flush=True)
        ok = False

    cli.close()
    srv.close()
    th.join(timeout=2.0)
    print(
        f"[selftest] RESULT={'PROTOCOL_ROUND_TRIP_OK' if ok else 'FAIL'}",
        flush=True,
    )
    return 0 if ok else 1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--serve", action="store_true", help="build holder + serve on --sock")
    p.add_argument(
        "--selftest",
        action="store_true",
        help="offline protocol round-trip (no device)",
    )
    p.add_argument("--sock", default="/logs/pypto_whole_prefill.sock")
    p.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    p.add_argument("-d", "--device", default="0,1,2,3,4,5,6,7")
    p.add_argument(
        "--ckpt",
        default="/mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp",
    )
    p.add_argument("--out", default="/tmp/n1_weight_ipc")
    p.add_argument("--kv-ipc", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.selftest:
        return _selftest()
    if args.serve:
        return run_sidecar(args)
    print("nothing to do; pass --serve or --selftest", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
