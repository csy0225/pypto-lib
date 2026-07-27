# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""独立的 MTP45/46/47 KV IPC exporter。

该 exporter 只拥有 MTP 的 K/V allocation，不复用 vLLM 主模型 KV pool，也
它用于 selected-layer hidden-only program 的
standalone bring-up；live serving 时同一布局由 vLLM KV allocator seam 提供。

每个 rank 分配一块：

``K(MTP45)..K(MTP47), aligned gap, V(MTP45)..V(MTP47)``

exporter 进程必须在 holder 整个生命周期内存活。严格 live session 下，map、
ready manifest、heartbeat 和 ACL key 由 :mod:`ipc_session` 绑定到同一个
session nonce / producer identity。
"""
from __future__ import annotations

import ctypes
import os
from typing import Any

from tools.step3p5.ipc_session import (
    atomic_write_json,
    attach_session,
    maybe_start_owner,
    write_ready_manifest,
)

KEY_BUF = 256
ALIGNMENT = 512
SCHEMA_VERSION = 2
LAYOUT = "mtp_flat_k_major_v_major_v1"
NUM_LAYERS = 3
HEAD_DIM = 128
NUM_KV_HEADS = 1
BLOCK_SIZE = 128
KV_DTYPE = "bfloat16"
KV_ITEMSIZE = 2
_HUGE_FIRST = 0


def _align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


def mtp_kv_layout(
    *,
    num_blocks: int,
    rank: int = 0,
    tp_world_size: int = 8,
    group_id: int = 0,
) -> dict[str, Any]:
    """生成 validator 可消费的独立 MTP KV map。

    ``num_blocks`` 是 scheduler-visible capacity；exporter 会按当前编译
    storage capacity 另外分配 ``capacity - 1`` 个 allocator-owned reserve
    blocks，且 map entry 始终描述 physical capacity。
    """
    scheduler_num_blocks = int(num_blocks)
    if scheduler_num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    from tools.step3p5.kv_padding import (  # noqa: PLC0415
        STORAGE_BATCH,
        make_padding_reserve,
    )

    reserve = make_padding_reserve(
        scheduler_num_blocks,
        scheduler_num_blocks + STORAGE_BATCH - 1,
        block_size=BLOCK_SIZE,
        storage_capacity=STORAGE_BATCH,
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
        raise ValueError("one MTP KV layer entry must be 512-byte aligned")
    k_section_bytes = NUM_LAYERS * entry_bytes
    v_section_offset = _align_up(k_section_bytes)
    v_section_bytes = NUM_LAYERS * entry_bytes
    pool_bytes = v_section_offset + v_section_bytes
    num_slots = physical_num_blocks * BLOCK_SIZE
    entries: dict[str, Any] = {}
    for which, section_offset in (("K", 0), ("V", v_section_offset)):
        for layer_idx in range(NUM_LAYERS):
            offset = section_offset + layer_idx * entry_bytes
            entries[f"MTP{layer_idx}.{which}"] = {
                "layer_idx": layer_idx,
                "absolute_layer_idx": 45 + layer_idx,
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
            }
    return {
        "version": SCHEMA_VERSION,
        "layout": LAYOUT,
        "rank": int(rank),
        "tp_world_size": int(tp_world_size),
        "pool_bytes": pool_bytes,
        "num_layers": NUM_LAYERS,
        "head_dim": HEAD_DIM,
        "num_kv_heads": NUM_KV_HEADS,
        "block_size": BLOCK_SIZE,
        "dtype": KV_DTYPE,
        **reserve.as_dict(),
        "sections": {
            "K": {"offset": 0, "nbytes": k_section_bytes},
            "V": {"offset": v_section_offset, "nbytes": v_section_bytes},
        },
        "map": entries,
    }


class MtpKvExporter:
    """拥有一 rank MTP KV pool、ACL export key 和 live-session owner。"""

    def __init__(self, dev: int) -> None:
        self.dev = int(dev)
        self._acl = ctypes.CDLL("libascendcl.so")
        self._acl.aclrtIpcMemGetExportKey.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_uint64,
        ]
        # aclrtIpcMemClose consumes the opaque export key returned by
        # aclrtIpcMemGetExportKey, not the exported device pointer.  Passing
        # the pointer happens to reach the runtime but is an invalid cleanup
        # ABI and can poison the following session.
        self._acl.aclrtIpcMemClose.argtypes = [ctypes.c_char_p]
        self._acl.aclrtMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self._acl.aclrtFree.argtypes = [ctypes.c_void_p]
        self._acl.aclrtMemset.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint8,
            ctypes.c_size_t,
        ]
        self._initialized = False
        self._pool_ptr: int | None = None
        self._pool_bytes = 0
        self._export_key: ctypes.Array[ctypes.c_char] | None = None
        self._owner = None

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        # Weight and MTP-KV owners may share one exporter process during the
        # standalone gate.  Match the existing weight exporter and tolerate
        # an already-initialized process-level ACL runtime.
        self._acl.aclInit(None)
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
    ) -> dict[str, Any]:
        """分配、清零、导出一 rank MTP KV pool 并发布 map/ready。"""
        self._ensure_init()
        if self._pool_ptr is not None:
            raise RuntimeError("MTP KV exporter already exported a pool")
        pool_map = mtp_kv_layout(
            num_blocks=num_blocks,
            rank=rank,
            tp_world_size=tp_world_size,
        )
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
        if self._pool_ptr == 0 or self._pool_ptr % ALIGNMENT:
            self.teardown()
            raise RuntimeError(
                f"MTP KV pool base is invalid or unaligned: {self._pool_ptr:#x}"
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
        # Keep the key buffer alive until teardown.  The ACL close API takes
        # this key, whereas the pool pointer is only used for aclrtFree.
        self._export_key = key_buf

        os.makedirs(out_dir, exist_ok=True)
        key_path = os.path.abspath(
            os.path.join(out_dir, f"pypto_mtp_kvpool.key.rank{rank}")
        )
        map_path = os.path.abspath(
            os.path.join(out_dir, f"pypto_mtp_kvpool_map.json.rank{rank}")
        )
        done_path = map_path + ".done"
        try:
            os.unlink(done_path)
        except FileNotFoundError:
            pass

        self._owner = maybe_start_owner(
            out_dir,
            role="mtp_kv",
            rank=rank,
            device_id=(
                int(os.environ["PYPTO_IPC_DEVICE_OFFSET"]) + int(rank)
                if "PYPTO_IPC_DEVICE_OFFSET" in os.environ
                else self.dev
            ),
        )
        pool_map = attach_session(pool_map, self._owner)
        key_tmp = key_path + f".tmp.{os.getpid()}"
        with open(key_tmp, "wb") as file:
            file.write(key_buf.raw)
            file.flush()
            os.fsync(file.fileno())
        os.replace(key_tmp, key_path)
        atomic_write_json(map_path, pool_map)
        write_ready_manifest(
            done_path,
            map_path=map_path,
            key_path=key_path,
            owner=self._owner,
        )
        return {
            "ok": True,
            "rank": int(rank),
            "device": self.dev,
            "pool_bytes": pool_bytes,
            "num_blocks": int(pool_map["physical_num_blocks"]),
            "scheduler_num_blocks": int(
                pool_map["scheduler_num_blocks"]
            ),
            "physical_num_blocks": int(pool_map["physical_num_blocks"]),
            "padding_block_ids": list(pool_map["padding_block_ids"]),
            "key_path": key_path,
            "map_path": map_path,
            "ready_path": done_path,
            "pool_base_debug": self._pool_ptr,
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
        if self._owner is not None:
            self._owner.close()
            self._owner = None


__all__ = ["MtpKvExporter", "mtp_kv_layout"]
