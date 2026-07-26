# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""N=1 whole-net decode sidecar: resident holder behind an AF_UNIX socket.

live single-handoff 架构（对齐 phases/20 §G + SKILL §H N=1 唯一生产形态）：
一个常驻 pypto 进程（8 chip children，`SIMPLER_COMM_NO_HCCL=1` 与 vLLM 同卡
共存）持有 `WholeDecodeHolder`（build+prepare 一次），暴露一个 AF_UNIX socket；
被 monkey-patch 的 vLLM `Step3p5Model.forward` 每个 decode step 把 hidden +
attn-meta 发过来，sidecar 跑 whole-net 回 next_hidden。

**协议**（length-prefixed self-describing，镜像 phases/20 §G5b）：
一帧 = `<I hdr_len>`(4B big-endian) + hdr_json(utf8) + 各 tensor blob 顺序拼接。
hdr = {"version":2, "order":[name,...],
       "tensors":{name:{shape,dtype,nbytes}}, "meta":{...}}。
tensor 序列化用 uint8 byte-view（兼容 bf16——numpy 不支持 bf16）。

serve loop **解耦**为 `decode_fn(meta, tensors) -> (out_meta, out_tensors)`，
所以协议层可用 stub（echo）做 offline round-trip 单测（`--selftest`，不依赖
device / whole-net hang）。生产 decode_fn 由 `run_sidecar` 绑定到 holder。
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

from tools.step3p5.kv_padding import (
    PaddingReserve,
    PaddingReserveError,
    parse_padding_reserve,
    validate_fixed_batch_metadata,
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
_BATCH = 16
_HIDDEN = 4096
MAIN_PROGRAM = "whole_decode_faithful_real_single_chip_hidden_only"


class FrameProtocolError(ValueError):
    """Malformed or unsupported sidecar frame."""


class SidecarDecodeError(RuntimeError):
    """The sidecar returned a structured decode failure."""


# ---- tensor <-> bytes (bf16-safe) ------------------------------------------

def _to_bytes(t: torch.Tensor) -> bytes:
    # numpy 不支持 bfloat16 -> 统一走 uint8 byte-view（对所有 dtype 都对）。
    return t.detach().contiguous().cpu().flatten().view(torch.uint8).numpy().tobytes()


def _from_bytes(blob: bytes, dtype: torch.dtype, shape) -> torch.Tensor:
    # bytearray -> 可写 buffer，避免 frombuffer read-only warning；clone 拥有内存。
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
class DecodeRequest:
    valid_tokens: int
    valid_requests: int
    kv_group_ids: tuple[int, ...]
    hidden: torch.Tensor
    seq_lens: torch.Tensor
    positions: torch.Tensor
    block_tables: dict[int, torch.Tensor]
    slot_mappings: dict[int, torch.Tensor]
    padding_reserve: PaddingReserve


@dataclass(frozen=True)
class MtpLayerRequest:
    """Validated single-layer MTP request.

    The selected PyPTO program owns token embedding and the transformer body.
    Therefore the request carries token ids and previous hidden, while the
    response is only the raw hidden state consumed by vLLM's shared head.
    """

    layer_idx: int
    valid_tokens: int
    previous_hidden: torch.Tensor
    input_token_ids: torch.Tensor
    active_mask: torch.Tensor
    seq_lens: torch.Tensor
    positions: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    padding_reserve: PaddingReserve


def _meta_int(meta: dict, name: str) -> int:
    value = meta.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise FrameProtocolError(f"decode meta field {name!r} must be an integer")
    return value


def validate_decode_request(meta: dict, tensors: dict) -> DecodeRequest:
    """Validate the stable pure-decode request ABI and its initialized padding."""
    if not isinstance(meta, dict):
        raise FrameProtocolError("decode meta must be an object")
    if meta.get("protocol_version") != _PROTOCOL_VERSION or meta.get("op") != "decode":
        raise FrameProtocolError("request must be protocol v2 op=decode")
    valid_tokens = _meta_int(meta, "valid_tokens")
    valid_requests = _meta_int(meta, "valid_requests")
    storage_batch = _meta_int(meta, "storage_batch")
    group_count = _meta_int(meta, "kv_group_count")
    if storage_batch != _BATCH:
        raise FrameProtocolError(f"storage_batch={storage_batch} != {_BATCH}")
    if not (1 <= valid_tokens <= _BATCH) or valid_requests != valid_tokens:
        raise FrameProtocolError("pure decode requires 1..16 tokens and one token per request")
    if group_count <= 0:
        raise FrameProtocolError("kv_group_count must be positive")
    query_lengths = meta.get("query_lengths")
    if query_lengths != [1] * valid_requests:
        raise FrameProtocolError(f"query_lengths must be all one, got {query_lengths!r}")
    layer_to_group = meta.get("layer_to_group")
    if (
        not isinstance(layer_to_group, list)
        or len(layer_to_group) != 45
        or any(isinstance(group, bool) or not isinstance(group, int) or group < 0 for group in layer_to_group)
    ):
        raise FrameProtocolError("layer_to_group must contain 45 non-negative group ids")
    group_ids = tuple(sorted(set(layer_to_group)))
    if len(group_ids) != group_count:
        raise FrameProtocolError("kv_group_count does not match layer_to_group")
    try:
        padding_reserve = parse_padding_reserve(
            meta.get("padding_reserve"),
            where="decode padding_reserve",
        )
    except PaddingReserveError as exc:
        raise FrameProtocolError(str(exc)) from exc

    required = {"hidden", "meta_seq_lens", "meta_positions"}
    for group_id in group_ids:
        required.add(f"meta_block_table_g{group_id}")
        required.add(f"meta_slot_mapping_g{group_id}")
    if set(tensors) != required:
        raise FrameProtocolError(
            f"decode tensor set mismatch; missing={sorted(required - set(tensors))} "
            f"extra={sorted(set(tensors) - required)}"
        )

    hidden = tensors["hidden"]
    if hidden.dtype != torch.bfloat16 or tuple(hidden.shape) != (valid_tokens, _HIDDEN):
        raise FrameProtocolError(
            f"hidden must be BF16 [{valid_tokens},{_HIDDEN}], got {hidden.dtype} {tuple(hidden.shape)}"
        )
    seq_lens = tensors["meta_seq_lens"]
    positions = tensors["meta_positions"]
    if seq_lens.dtype != torch.int32 or tuple(seq_lens.shape) != (_BATCH,):
        raise FrameProtocolError("meta_seq_lens must be INT32 [16]")
    if positions.dtype != torch.int32 or tuple(positions.shape) != (_BATCH,):
        raise FrameProtocolError("meta_positions must be INT32 [16]")
    if torch.any(seq_lens[:valid_requests] <= 0):
        raise FrameProtocolError("valid seq_lens must be positive")
    if not torch.equal(positions[:valid_requests], seq_lens[:valid_requests] - 1):
        raise FrameProtocolError("valid positions must equal seq_lens-1")
    if not torch.equal(seq_lens[valid_requests:], torch.ones(_BATCH - valid_requests, dtype=torch.int32)):
        raise FrameProtocolError("seq_lens padding must be initialized to one")
    if torch.count_nonzero(positions[valid_requests:]).item() != 0:
        raise FrameProtocolError("positions padding must be initialized to zero")

    block_tables: dict[int, torch.Tensor] = {}
    slot_mappings: dict[int, torch.Tensor] = {}
    for group_id in group_ids:
        block = tensors[f"meta_block_table_g{group_id}"]
        slot = tensors[f"meta_slot_mapping_g{group_id}"]
        if block.dtype != torch.int32 or block.ndim != 2 or block.shape[0] != _BATCH or block.shape[1] <= 0:
            raise FrameProtocolError(f"group {group_id}: block table must be INT32 [16,max_blocks]")
        if slot.dtype != torch.int32 or tuple(slot.shape) != (_BATCH,):
            raise FrameProtocolError(f"group {group_id}: slot mapping must be INT32 [16]")
        try:
            validate_fixed_batch_metadata(
                seq_lens=seq_lens,
                positions=positions,
                block_table=block,
                slot_mapping=slot,
                valid_rows=valid_requests,
                reserve=padding_reserve,
                where=f"decode group {group_id}",
            )
        except PaddingReserveError as exc:
            raise FrameProtocolError(str(exc)) from exc
        block_tables[group_id] = block
        slot_mappings[group_id] = slot

    return DecodeRequest(
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


def validate_mtp_layer_request(meta: dict, tensors: dict) -> MtpLayerRequest:
    """Validate the stable selected-layer MTP ABI.

    This is intentionally independent from :func:`validate_decode_request`.
    Main decode has 45 layer-to-group entries; MTP has one selected draft
    attention layer and one independent MTP KV pool.  Mixing those validators
    would make a malformed MTP request look like a valid main request.
    """
    if not isinstance(meta, dict):
        raise FrameProtocolError("MTP meta must be an object")
    if meta.get("protocol_version") != _PROTOCOL_VERSION or meta.get("op") != "mtp_layer":
        raise FrameProtocolError("request must be protocol v2 op=mtp_layer")
    layer_idx = _meta_int(meta, "layer_idx")
    if layer_idx not in (0, 1, 2):
        raise FrameProtocolError(f"MTP layer_idx must be 0, 1 or 2, got {layer_idx}")
    valid_tokens = _meta_int(meta, "valid_tokens")
    valid_requests = _meta_int(meta, "valid_requests")
    storage_batch = _meta_int(meta, "storage_batch")
    if storage_batch != _BATCH:
        raise FrameProtocolError(f"MTP storage_batch={storage_batch} != {_BATCH}")
    if not (1 <= valid_tokens <= _BATCH) or valid_requests != valid_tokens:
        raise FrameProtocolError(
            "MTP pure decode requires 1..16 active rows and one token per request"
        )
    try:
        padding_reserve = parse_padding_reserve(
            meta.get("padding_reserve"),
            where="MTP padding_reserve",
        )
    except PaddingReserveError as exc:
        raise FrameProtocolError(str(exc)) from exc
    required = {
        "previous_hidden",
        "input_token_ids",
        "active_mask",
        "meta_seq_lens",
        "meta_positions",
        "meta_block_table",
        "meta_slot_mapping",
    }
    if set(tensors) != required:
        raise FrameProtocolError(
            "MTP tensor set mismatch; "
            f"missing={sorted(required - set(tensors))} "
            f"extra={sorted(set(tensors) - required)}"
        )

    previous_hidden = tensors["previous_hidden"]
    if (
        previous_hidden.dtype != torch.bfloat16
        or tuple(previous_hidden.shape) != (valid_tokens, _HIDDEN)
    ):
        raise FrameProtocolError(
            "previous_hidden must be BF16 "
            f"[{valid_tokens},{_HIDDEN}], got "
            f"{previous_hidden.dtype} {tuple(previous_hidden.shape)}"
        )
    input_token_ids = tensors["input_token_ids"]
    if input_token_ids.dtype != torch.int32 or tuple(input_token_ids.shape) != (_BATCH,):
        raise FrameProtocolError("input_token_ids must be INT32 [16]")
    active_mask = tensors["active_mask"]
    if active_mask.dtype != torch.int32 or tuple(active_mask.shape) != (_BATCH,):
        raise FrameProtocolError("active_mask must be INT32 [16]")
    expected_active = torch.zeros(_BATCH, dtype=torch.int32)
    expected_active[:valid_tokens] = 1
    if not torch.equal(active_mask, expected_active):
        raise FrameProtocolError("active_mask must be [1]*valid_tokens followed by zeros")
    if torch.count_nonzero(input_token_ids[valid_tokens:]).item() != 0:
        raise FrameProtocolError("input_token_ids padding must be zero")

    seq_lens = tensors["meta_seq_lens"]
    positions = tensors["meta_positions"]
    if seq_lens.dtype != torch.int32 or tuple(seq_lens.shape) != (_BATCH,):
        raise FrameProtocolError("MTP meta_seq_lens must be INT32 [16]")
    if positions.dtype != torch.int32 or tuple(positions.shape) != (_BATCH,):
        raise FrameProtocolError("MTP meta_positions must be INT32 [16]")
    if torch.any(seq_lens[:valid_tokens] <= 0):
        raise FrameProtocolError("MTP valid seq_lens must be positive")
    if not torch.equal(positions[:valid_tokens], seq_lens[:valid_tokens] - 1):
        raise FrameProtocolError("MTP valid positions must equal seq_lens-1")
    if not torch.equal(
        seq_lens[valid_tokens:], torch.ones(_BATCH - valid_tokens, dtype=torch.int32)
    ):
        raise FrameProtocolError("MTP seq_lens padding must be initialized to one")
    if torch.count_nonzero(positions[valid_tokens:]).item() != 0:
        raise FrameProtocolError("MTP positions padding must be initialized to zero")

    block_table = tensors["meta_block_table"]
    slot_mapping = tensors["meta_slot_mapping"]
    if (
        block_table.dtype != torch.int32
        or block_table.ndim != 2
        or block_table.shape[0] != _BATCH
        or block_table.shape[1] <= 0
    ):
        raise FrameProtocolError("MTP block table must be INT32 [16,max_blocks]")
    if slot_mapping.dtype != torch.int32 or tuple(slot_mapping.shape) != (_BATCH,):
        raise FrameProtocolError("MTP slot mapping must be INT32 [16]")
    try:
        validate_fixed_batch_metadata(
            seq_lens=seq_lens,
            positions=positions,
            block_table=block_table,
            slot_mapping=slot_mapping,
            valid_rows=valid_tokens,
            reserve=padding_reserve,
            where="MTP request",
        )
    except PaddingReserveError as exc:
        raise FrameProtocolError(str(exc)) from exc

    return MtpLayerRequest(
        layer_idx=layer_idx,
        valid_tokens=valid_tokens,
        previous_hidden=previous_hidden,
        input_token_ids=input_token_ids,
        active_mask=active_mask,
        seq_lens=seq_lens,
        positions=positions,
        block_table=block_table,
        slot_mapping=slot_mapping,
        padding_reserve=padding_reserve,
    )


# ---- server -----------------------------------------------------------------

class WholeDecodeServer:
    """常驻 socket server。decode_fn(meta, tensors) -> (out_meta, out_tensors)。"""

    def __init__(self, sock_path: str, decode_fn):
        self.sock_path = sock_path
        self.decode_fn = decode_fn
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
                        out_meta, out_tensors = self.decode_fn(meta, tensors)
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

class WholeDecodeClient:
    """monkey-patch `_pypto_full_forward` 用：连 sidecar，发 hidden+meta，收 next_hidden。"""

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

    def decode(self, tensors: dict, meta: dict | None = None):
        if self._conn is None:
            self.connect()
        send_frame(self._conn, tensors, meta or {})
        out_meta, out_tensors = recv_frame(self._conn)
        if out_meta.get("ok") is False:
            raise SidecarDecodeError(
                f"{out_meta.get('error_type', 'DecodeError')}: {out_meta.get('error', '')}"
            )
        return out_meta, out_tensors

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ---- production decode_fn (bound to a resident holder) ---------------------

def make_holder_decode_fn(holder):
    """把 socket 请求映射到 holder：set hidden(+meta) -> run -> 回 hidden。

    request tensors:
      - "hidden": [T, HIDDEN] bf16  (T<=BATCH; 放进 current_hidden row 0..T-1)
      - 可选 "meta_seq_lens"/"meta_block_table"/"meta_slot_mapping"/"meta_rope_*"
        （live decode 用；缺省保持 holder 现值 = ctx=1）
    response tensors:
      - "next_hidden": [T, HIDDEN] bf16  (holder.next_hidden_out rank0 row 0..T-1)
    response meta:
      - {"dt_sec": float, "program": str}
    """
    if not getattr(holder, "hidden_only", False):
        raise ValueError(
            "live sidecar requires the hidden-only whole-net program; "
            "the diagnostic LM-head program is not a production ABI"
        )

    def decode_fn(meta, tensors):
        request = validate_decode_request(meta, tensors)
        if request.padding_reserve != getattr(
            holder,
            "padding_reserve",
            None,
        ):
            raise FrameProtocolError(
                "decode request padding reserve does not match the "
                "holder-imported Main KV allocation"
            )
        # The current release whole-net signature has one block table and one
        # slot mapping.  The v1 production launch therefore explicitly disables
        # vLLM's hybrid KV manager and requires one resolved KV group.  Keep the
        # wire protocol group-aware so a future generator-owned ABI can extend
        # this without another socket-version change.
        if len(request.kv_group_ids) != 1:
            raise FrameProtocolError(
                "current whole-net ABI requires one KV group; start vLLM with "
                "--disable-hybrid-kv-cache-manager"
            )
        group_id = request.kv_group_ids[0]
        holder.set_live_step(
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
            "op": "decode_result",
            "valid_tokens": request.valid_tokens,
            "dt_sec": float(getattr(holder, "_last_run_sec", 0.0)),
            "program": holder.layer_name,
        }, {"next_hidden": next_hidden}
    return decode_fn


def make_holder_mtp_decode_fn(holder):
    """Bind an MTP holder to the hidden-only ``op=mtp_layer`` ABI."""
    if not getattr(holder, "hidden_only", True):
        raise ValueError("MTP sidecar holder must expose hidden-only output")

    def decode_fn(meta, tensors):
        request = validate_mtp_layer_request(meta, tensors)
        if request.padding_reserve != getattr(
            holder,
            "padding_reserve",
            None,
        ):
            raise FrameProtocolError(
                "MTP request padding reserve does not match the "
                "holder-imported MTP KV allocation"
            )
        holder.set_live_step(
            request.previous_hidden,
            input_token_ids=request.input_token_ids,
            active_mask=request.active_mask,
            seq_lens=request.seq_lens,
            positions=request.positions,
            block_table=request.block_table,
            slot_mapping=request.slot_mapping,
        )
        res = holder.run(request.layer_idx)
        mtp_hidden = res["mtp_hidden"]
        if tuple(mtp_hidden.shape) != tuple(request.previous_hidden.shape):
            raise RuntimeError(
                "MTP holder returned an unexpected hidden shape: "
                f"{tuple(mtp_hidden.shape)} != {tuple(request.previous_hidden.shape)}"
            )
        if not torch.isfinite(mtp_hidden.float()).all():
            raise RuntimeError("MTP holder returned NaN/Inf hidden")
        return {
            "op": "mtp_layer_result",
            "layer_idx": request.layer_idx,
            "valid_tokens": request.valid_tokens,
            "dt_sec": float(res["dt"]),
            "program": res["program"],
        }, {"mtp_hidden": mtp_hidden.clone()}

    return decode_fn


def make_combined_decode_fn(main_holder, mtp_holder):
    """Dispatch Main and MTP hidden-only requests on one serialized socket."""
    main_decode = make_holder_decode_fn(main_holder)
    mtp_decode = make_holder_mtp_decode_fn(mtp_holder)

    def decode_fn(meta, tensors):
        op = meta.get("op") if isinstance(meta, dict) else None
        if op == "decode":
            return main_decode(meta, tensors)
        if op == "mtp_layer":
            return mtp_decode(meta, tensors)
        raise FrameProtocolError(f"unsupported sidecar operation {op!r}")

    return decode_fn


def _main_program_kwargs(args) -> dict:
    """Return optional Main program selection for the resident holder.

    The canonical 0724 baseline remains the default.  Opt-in replacement
    programs are deliberately explicit and must provide both the importable
    module and the exported program symbol so a half-configured sidecar
    cannot silently fall back to a different implementation.
    """
    module = getattr(args, "layer_module", None)
    name = getattr(args, "layer_name", None)
    if (module is None) != (name is None):
        raise ValueError(
            "--layer-module and --layer-name must be provided together"
        )
    if module is None:
        return {}
    return {"layer_module": module, "program": name}


def run_sidecar(args) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    import logging  # noqa: PLC0415
    logging.getLogger("simpler").setLevel(15)
    from tools.step3p5.whole_decode_holder import WholeDecodeHolder  # noqa: PLC0415
    device_ids = [int(d) for d in str(args.device).split(",")]
    holder = WholeDecodeHolder(
        device_ids=device_ids, out_dir=args.out, ckpt=args.ckpt,
        platform=args.platform, kv_ipc=args.kv_ipc,
        **_main_program_kwargs(args),
    ).build()
    with holder:
        server = WholeDecodeServer(args.sock, make_holder_decode_fn(holder))
        print(f"[sidecar] holder resident; serving whole-net on {args.sock}", flush=True)
        try:
            server.serve_forever()
        finally:
            server.close()
    return 0


def run_mtp_sidecar(args) -> int:
    """Build the resident selected-layer MTP holder and serve ``mtp_layer``."""
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    import logging  # noqa: PLC0415

    logging.getLogger("simpler").setLevel(15)
    from tools.step3p5.mtp_layer_holder import MtpLayerHolder  # noqa: PLC0415

    device_ids = [int(d) for d in str(args.device).split(",")]
    holder = MtpLayerHolder(
        device_ids=device_ids,
        out_dir=args.out,
        mtp_kv_dir=args.mtp_kv_out or args.out,
        ckpt=args.ckpt,
        platform=args.platform,
    ).build()
    with holder:
        server = WholeDecodeServer(
            args.sock,
            make_holder_mtp_decode_fn(holder),
        )
        print(
            f"[sidecar] MTP holder resident; serving selected-layer body on "
            f"{args.sock}",
            flush=True,
        )
        try:
            server.serve_forever()
        finally:
            server.close()
    return 0


def run_combined_sidecar(args) -> int:
    """Serve Main decode and selected MTP layers from one resident process."""
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    import logging  # noqa: PLC0415

    logging.getLogger("simpler").setLevel(15)
    from tools.step3p5.mtp_layer_holder import MtpLayerHolder  # noqa: PLC0415
    from tools.step3p5.whole_decode_holder import WholeDecodeHolder  # noqa: PLC0415

    device_ids = [int(d) for d in str(args.device).split(",")]
    main_holder = WholeDecodeHolder(
        device_ids=device_ids,
        out_dir=args.out,
        ckpt=args.ckpt,
        platform=args.platform,
        kv_ipc=args.kv_ipc,
        **_main_program_kwargs(args),
    ).build()
    mtp_holder = MtpLayerHolder(
        device_ids=device_ids,
        out_dir=args.out,
        mtp_kv_dir=args.mtp_kv_out or args.out,
        ckpt=args.ckpt,
        platform=args.platform,
    ).build()
    with main_holder, mtp_holder:
        server = WholeDecodeServer(
            args.sock,
            make_combined_decode_fn(main_holder, mtp_holder),
        )
        print(
            f"[sidecar] Main+MTP holders resident; serving hidden-only ABI on "
            f"{args.sock}",
            flush=True,
        )
        try:
            server.serve_forever()
        finally:
            server.close()
    return 0


# ---- offline protocol self-test (no device / no holder / no hang) ----------

def _selftest() -> int:
    """in-process AF_UNIX round-trip with a stub echo decode_fn. 验证协议 + bf16 编解码。"""
    import tempfile
    import time
    from tools.step3p5.kv_padding import make_padding_reserve

    sock_path = os.path.join(tempfile.mkdtemp(), "wd_selftest.sock")

    def echo_decode_fn(meta, tensors):
        request = validate_decode_request(meta, tensors)
        h = request.hidden
        # 模拟 whole-net：只回同 shape 的 next_hidden（这里 *2 便于校验）。
        return {"op": "decode_result", "dt_sec": 0.0}, {
            "next_hidden": (h.float() * 2).to(torch.bfloat16)
        }

    srv = WholeDecodeServer(sock_path, echo_decode_fn)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    for _ in range(50):
        if os.path.exists(sock_path):
            break
        time.sleep(0.05)

    ok = True
    cli = WholeDecodeClient(sock_path).connect()
    reserve = make_padding_reserve(8, 23)
    for trial in range(3):
        hidden = torch.randn(4, 4096, dtype=torch.bfloat16)
        seq_lens = torch.ones(16, dtype=torch.int32)
        seq_lens[:4] = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        positions = torch.zeros(16, dtype=torch.int32)
        positions[:4] = seq_lens[:4] - 1
        block_table = torch.zeros(16, 8, dtype=torch.int32)
        block_table[:4, 0] = torch.arange(4, dtype=torch.int32)
        block_table[4:, 0] = torch.arange(8, 20, dtype=torch.int32)
        slot_mapping = torch.zeros(16, dtype=torch.int32)
        slot_mapping[:4] = torch.tensor([0, 129, 258, 387], dtype=torch.int32)
        slot_mapping[4:] = torch.arange(8, 20, dtype=torch.int32) * 128
        req_meta = {
            "protocol_version": _PROTOCOL_VERSION,
            "op": "decode",
            "valid_tokens": 4,
            "valid_requests": 4,
            "storage_batch": 16,
            "kv_group_count": 1,
            "layer_to_group": [0] * 45,
            "query_lengths": [1] * 4,
            "padding_reserve": reserve.as_dict(),
        }
        out_meta, out = cli.decode(
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
        exp = (hidden.float() * 2).to(torch.bfloat16)
        same_shape = tuple(got.shape) == tuple(hidden.shape)
        max_err = (got.float() - exp.float()).abs().max().item()
        no_postprocess_output = "argmax_debug" not in out_meta
        trial_ok = same_shape and max_err == 0.0 and no_postprocess_output
        ok = ok and trial_ok
        print(f"[selftest] trial {trial}: shape_ok={same_shape} max_err={max_err} "
              f"hidden_only_meta={no_postprocess_output} -> "
              f"{'PASS' if trial_ok else 'FAIL'}", flush=True)
    # The server must reject stale/non-initialized padding without dying.
    bad_seq = seq_lens.clone()
    bad_seq[8] = 0
    try:
        cli.decode(
            {
                "hidden": hidden,
                "meta_seq_lens": bad_seq,
                "meta_positions": positions,
                "meta_block_table_g0": block_table,
                "meta_slot_mapping_g0": slot_mapping,
            },
            req_meta,
        )
    except SidecarDecodeError:
        print("[selftest] invalid padding rejected -> PASS", flush=True)
    else:
        print("[selftest] invalid padding rejected -> FAIL", flush=True)
        ok = False
    cli.close()
    srv.close()
    th.join(timeout=2.0)
    print(f"[selftest] RESULT={'PROTOCOL_ROUND_TRIP_OK' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def _mtp_selftest() -> int:
    """Offline MTP framing/validation self-test; no holder or device."""
    import tempfile
    import time
    from tools.step3p5.kv_padding import make_padding_reserve

    sock_path = os.path.join(tempfile.mkdtemp(), "mtp_selftest.sock")

    def echo_decode_fn(meta, tensors):
        req = validate_mtp_layer_request(meta, tensors)
        return {
            "op": "mtp_layer_result",
            "layer_idx": req.layer_idx,
        }, {"mtp_hidden": req.previous_hidden.clone()}

    srv = WholeDecodeServer(sock_path, echo_decode_fn)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    for _ in range(50):
        if os.path.exists(sock_path):
            break
        time.sleep(0.05)

    cli = WholeDecodeClient(sock_path).connect()
    hidden = torch.randn(3, _HIDDEN, dtype=torch.bfloat16)
    reserve = make_padding_reserve(16, 31)
    ids = torch.tensor([6178, 410, 303] + [0] * 13, dtype=torch.int32)
    active = torch.tensor([1, 1, 1] + [0] * 13, dtype=torch.int32)
    seq = torch.ones(_BATCH, dtype=torch.int32)
    seq[:3] = torch.tensor([2, 3, 4], dtype=torch.int32)
    pos = torch.zeros(_BATCH, dtype=torch.int32)
    pos[:3] = seq[:3] - 1
    block = torch.zeros(_BATCH, 4, dtype=torch.int32)
    block[:3, 0] = torch.tensor([7, 8, 9], dtype=torch.int32)
    block[3:, 0] = torch.arange(16, 29, dtype=torch.int32)
    slots = torch.zeros(_BATCH, dtype=torch.int32)
    # slot_mapping is block_table[row, position] * BLOCK_SIZE + position.
    # The first three rows use seq_lens 2, 3, 4, hence positions 1, 2, 3.
    slots[:3] = torch.tensor([897, 1026, 1155], dtype=torch.int32)
    slots[3:] = torch.arange(16, 29, dtype=torch.int32) * 128
    meta = {
        "protocol_version": _PROTOCOL_VERSION,
        "op": "mtp_layer",
        "layer_idx": 1,
        "valid_tokens": 3,
        "valid_requests": 3,
        "storage_batch": _BATCH,
        "padding_reserve": reserve.as_dict(),
    }
    _, out = cli.decode(
        {
            "previous_hidden": hidden,
            "input_token_ids": ids,
            "active_mask": active,
            "meta_seq_lens": seq,
            "meta_positions": pos,
            "meta_block_table": block,
            "meta_slot_mapping": slots,
        },
        meta,
    )
    ok = torch.equal(out["mtp_hidden"], hidden)
    print(f"[mtp-selftest] hidden-only echo -> {'PASS' if ok else 'FAIL'}", flush=True)
    bad = dict(meta)
    bad["layer_idx"] = 3
    try:
        cli.decode(
            {
                "previous_hidden": hidden,
                "input_token_ids": ids,
                "active_mask": active,
                "meta_seq_lens": seq,
                "meta_positions": pos,
                "meta_block_table": block,
                "meta_slot_mapping": slots,
            },
            bad,
        )
    except SidecarDecodeError:
        print("[mtp-selftest] invalid layer rejected -> PASS", flush=True)
    else:
        print("[mtp-selftest] invalid layer rejected -> FAIL", flush=True)
        ok = False
    cli.close()
    srv.close()
    th.join(timeout=2.0)
    print(
        f"[mtp-selftest] RESULT={'MTP_PROTOCOL_ROUND_TRIP_OK' if ok else 'FAIL'}",
        flush=True,
    )
    return 0 if ok else 1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--serve", action="store_true", help="build holder + serve on --sock")
    p.add_argument(
        "--mtp-serve",
        action="store_true",
        help="build selected-layer MTP holder + serve op=mtp_layer",
    )
    p.add_argument(
        "--serve-all",
        action="store_true",
        help="serve Main decode and selected MTP layers on one socket",
    )
    p.add_argument("--selftest", action="store_true", help="offline protocol round-trip (no device)")
    p.add_argument(
        "--mtp-selftest",
        action="store_true",
        help="offline selected-layer MTP protocol round-trip",
    )
    p.add_argument("--sock", default="/logs/pypto_whole_decode.sock")
    p.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    p.add_argument("-d", "--device", default="0,1,2,3,4,5,6,7")
    p.add_argument("--ckpt", default="/mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp")
    p.add_argument("--out", default="/tmp/n1_weight_ipc")
    p.add_argument(
        "--layer-module",
        default=None,
        help=(
            "optional Main program module; must be paired with --layer-name. "
            "Default uses the canonical 0724 baseline Main"
        ),
    )
    p.add_argument(
        "--layer-name",
        default=None,
        help=(
            "optional Main program symbol; must be paired with "
            "--layer-module"
        ),
    )
    p.add_argument(
        "--mtp-kv-out",
        default=None,
        help="independent MTP KV IPC directory (defaults to --out)",
    )
    p.add_argument("--kv-ipc", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.selftest:
        return _selftest()
    if args.mtp_selftest:
        return _mtp_selftest()
    if args.serve:
        return run_sidecar(args)
    if args.mtp_serve:
        return run_mtp_sidecar(args)
    if args.serve_all:
        return run_combined_sidecar(args)
    print("nothing to do; pass --serve or --selftest", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
