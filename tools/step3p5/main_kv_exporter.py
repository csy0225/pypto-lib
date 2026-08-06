# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Standalone Main decoder KV IPC exporter.

The production Main KV allocation is owned by vLLM and is exported by the
vLLM KV-pool overlay.  This module is only the standalone device-gate owner:
it emits the *same* schema-v3 K-major/V-major map consumed by
``tools.step3p5.pypto_kv_ipc`` and keeps the ACL allocation alive while the
resident hidden-only holder runs.  Production defaults to all 45 layers;
focused diagnostics may request an explicit smaller contiguous layer prefix.

The exporter deliberately does not put K/V into the weight IPC pool.  Main
weights and Main paged KV are separate ownership domains, which is also the
live vLLM boundary.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from tools.step3p5.ipc_session import (
    atomic_write_json,
    attach_session,
    maybe_start_owner,
    write_ready_manifest,
)
from tools.step3p5.kv_padding import make_padding_reserve


KEY_BUF = 256
ALIGNMENT = 512
SCHEMA_VERSION = 3
LAYOUT = "flat_k_major_v_major_v1"
NUM_LAYERS = 45
HEAD_DIM = 128
NUM_KV_HEADS = 1
BLOCK_SIZE = 128
KV_DTYPE = "bfloat16"
KV_ITEMSIZE = 2
_HUGE_FIRST = 0
_D2H = 2


def _align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


def main_kv_layout(
    *,
    num_blocks: int,
    rank: int = 0,
    tp_world_size: int = 8,
    group_id: int = 0,
    num_layers: int = NUM_LAYERS,
) -> dict[str, Any]:
    """Create one validator-compatible Main KV map.

    ``num_blocks`` is the scheduler-visible capacity.  The map describes the
    physical allocation including the fifteen allocator-owned padding blocks.
    """
    num_layers = int(num_layers)
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    reserve = make_padding_reserve(
        int(num_blocks),
        int(num_blocks) + 15,
        block_size=BLOCK_SIZE,
    )
    physical_num_blocks = reserve.physical_num_blocks
    entry_bytes = (
        physical_num_blocks
        * BLOCK_SIZE
        * NUM_KV_HEADS
        * HEAD_DIM
        * KV_ITEMSIZE
    )
    if entry_bytes % ALIGNMENT:
        raise ValueError("one Main KV layer entry must be 512-byte aligned")
    k_section_bytes = num_layers * entry_bytes
    v_section_offset = _align_up(k_section_bytes)
    v_section_bytes = num_layers * entry_bytes
    pool_bytes = v_section_offset + v_section_bytes
    num_slots = physical_num_blocks * BLOCK_SIZE

    entries: dict[str, Any] = {}
    for which, section_offset in (("K", 0), ("V", v_section_offset)):
        for layer_idx in range(num_layers):
            offset = section_offset + layer_idx * entry_bytes
            entries[f"L{layer_idx}.{which}"] = {
                "layer_idx": layer_idx,
                "which": which,
                "group_id": int(group_id),
                "offset": offset,
                "nbytes": entry_bytes,
                "num_blocks": physical_num_blocks,
                "num_slots": num_slots,
                "shape": [
                    physical_num_blocks,
                    BLOCK_SIZE,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                ],
                "flat_shape": [num_slots, HEAD_DIM],
                "dtype": KV_DTYPE,
                "head_dim": HEAD_DIM,
                "num_kv_heads": NUM_KV_HEADS,
                "block_size": BLOCK_SIZE,
            }

    return {
        "version": SCHEMA_VERSION,
        "layout": LAYOUT,
        "rank": int(rank),
        "tp_world_size": int(tp_world_size),
        "pool_bytes": pool_bytes,
        "num_layers": num_layers,
        "head_dim": HEAD_DIM,
        "num_kv_heads": NUM_KV_HEADS,
        "block_size": BLOCK_SIZE,
        "dtype": KV_DTYPE,
        **reserve.as_dict(),
        "sections": {
            "K": {"offset": 0, "nbytes": k_section_bytes},
            "V": {"offset": v_section_offset, "nbytes": v_section_bytes},
        },
        "groups": [
            {
                "group_id": int(group_id),
                "layer_indices": list(range(num_layers)),
            }
        ],
        "map": entries,
    }


def main_kv_row_byte_offset(
    pool_map: dict[str, Any],
    *,
    layer_idx: int,
    which: str,
    slot: int,
) -> int:
    """Return the byte offset of one logical ``[HEAD_DIM]`` KV row.

    This helper is deliberately pure so the standalone diagnostic can prove
    that its owner-side D2H probe addresses the exact schema-v3 row described
    by the exported map.  ``slot`` is the scheduler/physical slot within one
    layer; the per-layer base remains map-owned and is never folded into the
    metadata ``slot_mapping``.
    """
    layer_idx = int(layer_idx)
    slot = int(slot)
    try:
        num_layers = int(pool_map["num_layers"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid Main KV map num_layers") from exc
    if layer_idx not in range(num_layers):
        raise ValueError(f"layer_idx must be 0..{num_layers - 1}")
    if which not in ("K", "V"):
        raise ValueError("which must be K or V")
    try:
        entry = pool_map["map"][f"L{layer_idx}.{which}"]
        num_slots = int(entry["num_slots"])
        offset = int(entry["offset"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid Main KV map entry L{layer_idx}.{which}"
        ) from exc
    if not 0 <= slot < num_slots:
        raise ValueError(
            f"slot={slot} outside L{layer_idx}.{which} num_slots={num_slots}"
        )
    return offset + slot * HEAD_DIM * KV_ITEMSIZE


class MainKvExporter:
    """Own and export one rank's standalone Main K/V allocation."""

    def __init__(self, dev: int) -> None:
        self.dev = int(dev)
        self._acl = ctypes.CDLL("libascendcl.so")
        self._acl.aclInit.argtypes = [ctypes.c_char_p]
        self._acl.aclrtSetDevice.argtypes = [ctypes.c_int]
        self._acl.aclrtIpcMemGetExportKey.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_uint64,
        ]
        self._acl.aclrtIpcMemClose.argtypes = [ctypes.c_char_p]
        self._acl.aclrtMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self._acl.aclrtMemset.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint8,
            ctypes.c_size_t,
        ]
        self._acl.aclrtMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self._acl.aclrtFree.argtypes = [ctypes.c_void_p]
        self._initialized = False
        self._pool_ptr: int | None = None
        self._pool_bytes = 0
        self._pool_map: dict[str, Any] | None = None
        self._export_key: ctypes.Array[ctypes.c_char] | None = None
        self._owner = None

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        rc = self._acl.aclInit(None)
        if rc not in (0, 100002):  # tolerate process-level already-init
            raise RuntimeError(f"aclInit rc={rc}")
        rc = self._acl.aclrtSetDevice(self.dev)
        if rc != 0:
            raise RuntimeError(f"aclrtSetDevice({self.dev}) rc={rc}")
        self._initialized = True

    def export(
        self,
        *,
        out_dir: str,
        rank: int,
        tp_world_size: int = 8,
        num_blocks: int,
        num_layers: int = NUM_LAYERS,
    ) -> dict[str, Any]:
        self._ensure_init()
        if self._pool_ptr is not None:
            raise RuntimeError("Main KV exporter already owns a pool")

        pool_map = main_kv_layout(
            num_blocks=num_blocks,
            rank=rank,
            tp_world_size=tp_world_size,
            num_layers=num_layers,
        )
        self._pool_map = pool_map
        pool_bytes = int(pool_map["pool_bytes"])
        dptr = ctypes.c_void_p()
        rc = self._acl.aclrtMalloc(
            ctypes.byref(dptr),
            ctypes.c_size_t(pool_bytes),
            ctypes.c_int(_HUGE_FIRST),
        )
        if rc != 0:
            raise RuntimeError(f"aclrtMalloc rc={rc} nbytes={pool_bytes}")
        self._pool_ptr = int(dptr.value or 0)
        self._pool_bytes = pool_bytes
        if not self._pool_ptr or self._pool_ptr % ALIGNMENT:
            self.teardown()
            raise RuntimeError(
                f"Main KV pool base is invalid or unaligned: {self._pool_ptr!r}"
            )

        rc = self._acl.aclrtMemset(
            ctypes.c_void_p(self._pool_ptr),
            ctypes.c_size_t(pool_bytes),
            ctypes.c_uint8(0),
            ctypes.c_size_t(pool_bytes),
        )
        if rc != 0:
            self.teardown()
            raise RuntimeError(f"aclrtMemset rc={rc} nbytes={pool_bytes}")

        key_buf = ctypes.create_string_buffer(KEY_BUF)
        rc = self._acl.aclrtIpcMemGetExportKey(
            ctypes.c_void_p(self._pool_ptr),
            ctypes.c_size_t(pool_bytes),
            key_buf,
            ctypes.c_size_t(KEY_BUF),
            ctypes.c_uint64(0x1),
        )
        if rc != 0:
            self.teardown()
            raise RuntimeError(
                f"aclrtIpcMemGetExportKey rc={rc} pool={self._pool_ptr:#x}"
            )
        self._export_key = key_buf

        out = os.path.abspath(out_dir)
        os.makedirs(out, exist_ok=True)
        key_path = os.path.join(out, f"pypto_kvpool.key.rank{rank}")
        map_path = os.path.join(out, f"pypto_kvpool_map.json.rank{rank}")
        ready_path = map_path + ".done"
        try:
            os.unlink(ready_path)
        except FileNotFoundError:
            pass

        self._owner = maybe_start_owner(
            out,
            role="kv",
            rank=rank,
            device_id=(
                int(os.environ["PYPTO_IPC_DEVICE_OFFSET"]) + int(rank)
                if "PYPTO_IPC_DEVICE_OFFSET" in os.environ
                else self.dev
            ),
        )
        pool_map = attach_session(pool_map, self._owner)
        key_tmp = f"{key_path}.tmp.{os.getpid()}"
        with open(key_tmp, "wb") as file:
            file.write(key_buf.raw)
            file.flush()
            os.fsync(file.fileno())
        os.replace(key_tmp, key_path)
        atomic_write_json(map_path, pool_map)
        write_ready_manifest(
            ready_path,
            map_path=map_path,
            key_path=key_path,
            owner=self._owner,
        )
        return {
            "ok": True,
            "rank": int(rank),
            "device": self.dev,
            "pool_bytes": pool_bytes,
            "num_layers": int(pool_map["num_layers"]),
            "scheduler_num_blocks": int(pool_map["scheduler_num_blocks"]),
            "physical_num_blocks": int(pool_map["physical_num_blocks"]),
            "padding_block_ids": list(pool_map["padding_block_ids"]),
            "key_path": key_path,
            "map_path": map_path,
            "ready_path": ready_path,
            "pool_base_debug": self._pool_ptr,
        }

    def snapshot_rows(
        self,
        *,
        layer_indices: list[int],
        slots: list[int],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """D2H-copy selected KV rows from the allocation owner.

        This is an opt-in standalone diagnostic and is not called by the live
        vLLM exporter path.  Capture happens only after the worker reports that
        ``rt.run()`` returned, so it does not alter the measured run interval
        or add a device operation inside the PyPTO program.

        Returns:
          ``(rows, summary)`` where ``rows`` contains CPU BF16 tensors suitable
          for ``torch.save`` and ``summary`` contains stable byte hashes and
          compact numerical observations.
        """
        if self._pool_ptr is None or self._pool_map is None:
            raise RuntimeError("Main KV pool is not exported")
        import torch  # noqa: PLC0415

        rows: dict[str, Any] = {}
        summary: dict[str, Any] = {}
        row_nbytes = HEAD_DIM * KV_ITEMSIZE
        for layer_idx in layer_indices:
            for which in ("K", "V"):
                for slot in slots:
                    key = f"L{int(layer_idx)}.{which}.slot{int(slot)}"
                    row_offset = main_kv_row_byte_offset(
                        self._pool_map,
                        layer_idx=int(layer_idx),
                        which=which,
                        slot=int(slot),
                    )
                    host = ctypes.create_string_buffer(row_nbytes)
                    rc = self._acl.aclrtMemcpy(
                        ctypes.cast(host, ctypes.c_void_p),
                        ctypes.c_size_t(row_nbytes),
                        ctypes.c_void_p(self._pool_ptr + row_offset),
                        ctypes.c_size_t(row_nbytes),
                        ctypes.c_int(_D2H),
                    )
                    if rc != 0:
                        raise RuntimeError(
                            f"aclrtMemcpy D2H rc={rc} key={key} "
                            f"offset={row_offset}"
                        )
                    raw = bytes(host.raw)
                    row = (
                        torch.frombuffer(
                            bytearray(raw),
                            dtype=torch.int16,
                        )
                        .view(torch.bfloat16)
                        .clone()
                    )
                    values = row.float()
                    rows[key] = row
                    summary[key] = {
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "nonzero": int(torch.count_nonzero(row).item()),
                        "finite": bool(torch.isfinite(values).all().item()),
                        "abs_max": float(values.abs().max().item()),
                        "abs_sum": float(values.abs().sum().item()),
                        "first8": [float(item) for item in values[:8]],
                        "byte_offset": int(row_offset),
                    }
        return rows, summary

    def snapshot_full_pool(
        self,
        *,
        out_dir: str,
        probe_id: str,
        chunk_rows: int = 8192,
    ) -> dict[str, Any]:
        """Hash every physical KV row without adding a production op.

        This is a diagnostics-only D2H scan used by the B3 acceptance probe.
        The pool is copied in contiguous chunks, while one SHA-256 digest is
        emitted for the whole pool and every transfer chunk.  The raw D2H
        bytes are written to a temporary diagnostic sidecar so the caller can
        compare all physical rows exactly without millions of Python hash
        calls.  The sidecar is outside the production program and is removed
        by the acceptance probe after each comparison.
        """
        if self._pool_ptr is None or self._pool_map is None:
            raise RuntimeError("Main KV pool is not exported")
        chunk_rows = int(chunk_rows)
        if chunk_rows <= 0:
            raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")

        row_bytes = HEAD_DIM * KV_ITEMSIZE
        if self._pool_bytes % row_bytes:
            raise RuntimeError(
                f"pool bytes {self._pool_bytes} is not row aligned"
            )
        total_rows = self._pool_bytes // row_bytes
        chunk_bytes = chunk_rows * row_bytes
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        snapshot_path = out / (
            f"kv_pool_snapshot_{probe_id}.rank{self._pool_map['rank']}.bin"
        )
        snapshot_tmp = snapshot_path.with_name(
            f"{snapshot_path.name}.tmp.{os.getpid()}"
        )
        pool_digest = hashlib.sha256()
        chunk_digests: list[str] = []
        row_count = 0
        try:
            with snapshot_tmp.open("wb") as snapshot_file:
                for chunk_offset in range(0, self._pool_bytes, chunk_bytes):
                    nbytes = min(chunk_bytes, self._pool_bytes - chunk_offset)
                    if nbytes % row_bytes:
                        raise RuntimeError(
                            f"scan chunk {nbytes} is not row aligned"
                        )
                    host = ctypes.create_string_buffer(nbytes)
                    rc = self._acl.aclrtMemcpy(
                        ctypes.cast(host, ctypes.c_void_p),
                        ctypes.c_size_t(nbytes),
                        ctypes.c_void_p(self._pool_ptr + chunk_offset),
                        ctypes.c_size_t(nbytes),
                        ctypes.c_int(_D2H),
                    )
                    if rc != 0:
                        raise RuntimeError(
                            f"aclrtMemcpy full-pool D2H rc={rc} "
                            f"offset={chunk_offset} nbytes={nbytes}"
                        )
                    raw = bytes(host.raw)
                    pool_digest.update(raw)
                    chunk_digests.append(hashlib.sha256(raw).hexdigest())
                    snapshot_file.write(raw)
                    row_count += nbytes // row_bytes
                snapshot_file.flush()
                os.fsync(snapshot_file.fileno())
            os.replace(snapshot_tmp, snapshot_path)
        except BaseException:
            try:
                snapshot_tmp.unlink()
            except FileNotFoundError:
                pass
            raise

        num_slots = int(self._pool_map["map"]["L0.K"]["num_slots"])
        return {
            "scan_kind": "full_pool_row_diff_v1",
            "pool_map_digest": hashlib.sha256(
                json.dumps(
                    self._pool_map,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "pool_sha256": pool_digest.hexdigest(),
            "pool_bytes": int(self._pool_bytes),
            "row_bytes": int(row_bytes),
            "row_count": int(row_count),
            "num_slots": num_slots,
            "chunk_rows": int(chunk_rows),
            "chunk_sha256": chunk_digests,
            "snapshot_path": str(snapshot_path),
            "pool_base_debug": int(self._pool_ptr),
            "rank": int(self._pool_map["rank"]),
        }

    def teardown(self) -> None:
        if self._export_key is not None:
            try:
                self._acl.aclrtIpcMemClose(self._export_key)
            finally:
                self._export_key = None
        if self._pool_ptr is not None:
            try:
                self._acl.aclrtFree(ctypes.c_void_p(self._pool_ptr))
            finally:
                self._pool_ptr = None
                self._pool_bytes = 0
                self._pool_map = None
        if self._owner is not None:
            self._owner.close()
            self._owner = None


__all__ = [
    "MainKvExporter",
    "main_kv_layout",
    "main_kv_row_byte_offset",
]
