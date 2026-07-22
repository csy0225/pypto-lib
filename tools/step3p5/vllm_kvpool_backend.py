# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""vLLM-Ascend Step3p5 K-major/V-major KV-pool overlay.

This module patches only ``NPUModelRunner._allocate_kv_cache_tensors``.  It
validates vLLM's resolved ``KVCacheConfig`` and replaces the 45 independent
raw K/V allocations with views into one rank-local allocation:

``K(L0)..K(L44), V(L0)..V(L44)``.

The views are returned to vLLM and are reshaped/bound by the unmodified
vLLM-Ascend path.  The same allocation is exported with one ACL IPC key and a
schema-v3 map consumed by ``pypto_kv_ipc.py``.

The first live whole-net ABI has one block table and one slot mapping, so this
overlay deliberately requires vLLM's hybrid KV manager to be disabled.  That
turns full and sliding attention into one KV group while preserving sliding
attention compute.  Multi-group metadata remains represented in the protocol
and can be enabled after the generator-owned whole-net signature is extended.

Environment:

``PYPTO_KVPOOL=1``
    Enable the overlay.
``PYPTO_KVPOOL_DIR=/logs``
    Directory for key/map/sentinel files.
"""
from __future__ import annotations

import ctypes
import json
import os
import re
from typing import Any, Mapping

import torch

KEY_BUF = 256
ALIGNMENT = 512
SCHEMA_VERSION = 3
LAYOUT = "flat_k_major_v_major_v1"
MTP_LAYOUT = "mtp_flat_k_major_v_major_v1"
MTP_SCHEMA_VERSION = 2
NUM_LAYERS = 45
MTP_START_LAYER = 45
NUM_MTP_LAYERS = 3
HEAD_DIM = 128
NUM_KV_HEADS = 1
BLOCK_SIZE = 128
KV_DTYPE = torch.bfloat16
KV_DTYPE_NAME = "bfloat16"
_LAYER_IDX_RE = re.compile(r"(?:^|[.]layers[.])(\d+)(?:[.]|$)")

_POOLS: list[Any] = []
_BIG: list[torch.Tensor] = []
_SESSION_OWNERS: list[Any] = []
_RESERVES: dict[str, Any] = {}
_PATCHED = False
_ORIGINAL = None


class KvPoolBackendError(RuntimeError):
    """Resolved vLLM configuration cannot satisfy the PyPTO KV ABI."""


def _dir() -> str:
    return os.environ.get("PYPTO_KVPOOL_DIR", "/logs")


def _align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _layer_idx(name: str) -> int | None:
    match = _LAYER_IDX_RE.search(str(name))
    return int(match.group(1)) if match else None


def _export_key(dptr: int, nbytes: int) -> bytes:
    acl = ctypes.CDLL("libascendcl.so")
    acl.aclrtIpcMemGetExportKey.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_uint64,
    ]
    key = ctypes.create_string_buffer(KEY_BUF)
    rc = acl.aclrtIpcMemGetExportKey(
        ctypes.c_void_p(dptr),
        ctypes.c_size_t(nbytes),
        key,
        KEY_BUF,
        ctypes.c_uint64(0x1),
    )
    if rc != 0:
        raise KvPoolBackendError(
            f"aclrtIpcMemGetExportKey rc={rc} dptr={hex(dptr)} nbytes={nbytes}"
        )
    return key.raw[:KEY_BUF]


def _get_layer_specs(kv_cache_config) -> dict[str, Any]:
    try:
        from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs  # noqa: PLC0415
    except ImportError:
        UniformTypeKVCacheSpecs = ()  # type: ignore[assignment]

    result: dict[str, Any] = {}
    for group in kv_cache_config.kv_cache_groups:
        group_spec = group.kv_cache_spec
        for layer_name in group.layer_names:
            if UniformTypeKVCacheSpecs and isinstance(group_spec, UniformTypeKVCacheSpecs):
                result[layer_name] = group_spec.kv_cache_specs[layer_name]
            else:
                result[layer_name] = group_spec
    return result


def _validate_config(self, kv_cache_config):
    from tools.step3p5.kv_padding import (  # noqa: PLC0415
        PADDING_BLOCK_COUNT,
        make_padding_reserve,
    )

    model_type = str(getattr(self.model_config.hf_text_config, "model_type", "")).lower()
    if model_type not in {"step3p5", "step3_5", "step3.5"}:
        raise KvPoolBackendError(
            f"PYPTO_KVPOOL is Step3p5-only, got model_type={model_type!r}"
        )
    groups = list(kv_cache_config.kv_cache_groups)
    if len(groups) != 1:
        raise KvPoolBackendError(
            "the current whole-net ABI requires one vLLM KV group; launch with "
            "--disable-hybrid-kv-cache-manager"
        )
    layer_specs = _get_layer_specs(kv_cache_config)
    by_index: dict[int, tuple[str, Any, Any]] = {}
    mtp_by_index: dict[int, tuple[str, Any, Any]] = {}
    for tensor_spec in kv_cache_config.kv_cache_tensors:
        if len(tensor_spec.shared_by) != 1:
            raise KvPoolBackendError(
                "single-group Step3p5 must allocate one KV tensor per layer; "
                f"got shared_by={tensor_spec.shared_by}"
            )
        layer_name = tensor_spec.shared_by[0]
        index = _layer_idx(layer_name)
        if index is None:
            continue
        if index < NUM_LAYERS:
            if index in by_index:
                raise KvPoolBackendError(f"duplicate Step3p5 KV layer index {index}")
            by_index[index] = (layer_name, tensor_spec, layer_specs[layer_name])
        elif MTP_START_LAYER <= index < MTP_START_LAYER + NUM_MTP_LAYERS:
            local_idx = index - MTP_START_LAYER
            if local_idx in mtp_by_index:
                raise KvPoolBackendError(f"duplicate Step3p5 MTP KV layer index {index}")
            if ".mtp_block.self_attn.attn" not in layer_name:
                raise KvPoolBackendError(
                    f"MTP KV layer {index} has unexpected key {layer_name!r}"
                )
            mtp_by_index[local_idx] = (
                layer_name,
                tensor_spec,
                layer_specs[layer_name],
            )
    if set(by_index) != set(range(NUM_LAYERS)):
        missing = sorted(set(range(NUM_LAYERS)) - set(by_index))
        raise KvPoolBackendError(
            f"KV allocation config does not cover 45 decoder layers: missing={missing[:8]}"
        )
    if set(mtp_by_index) != set(range(NUM_MTP_LAYERS)):
        missing = sorted(set(range(NUM_MTP_LAYERS)) - set(mtp_by_index))
        raise KvPoolBackendError(
            "KV allocation config does not cover MTP45/46/47: "
            f"missing_local_indices={missing}"
        )

    scheduler_num_blocks = int(kv_cache_config.num_blocks)
    configured_num_blocks = None
    ordered_specs = [
        by_index[index] for index in range(NUM_LAYERS)
    ] + [
        mtp_by_index[index] for index in range(NUM_MTP_LAYERS)
    ]
    for layer_name, tensor_spec, spec in ordered_specs:
        if int(spec.block_size) != BLOCK_SIZE:
            raise KvPoolBackendError(
                f"{layer_name}: block_size={spec.block_size} != {BLOCK_SIZE}"
            )
        if int(spec.num_kv_heads) != NUM_KV_HEADS or int(spec.head_size) != HEAD_DIM:
            raise KvPoolBackendError(
                f"{layer_name}: KV shape heads/head_dim="
                f"{spec.num_kv_heads}/{spec.head_size} != 1/128"
            )
        if spec.dtype != KV_DTYPE:
            raise KvPoolBackendError(
                f"{layer_name}: logical KV dtype={spec.dtype} != {KV_DTYPE}"
            )
        tensor_bytes = int(tensor_spec.size)
        if tensor_bytes <= 0 or tensor_bytes % 2:
            raise KvPoolBackendError(
                f"{layer_name}: total KV bytes must be positive and even, got {tensor_bytes}"
            )
        k_bytes = tensor_bytes // 2
        if k_bytes % ALIGNMENT:
            raise KvPoolBackendError(
                f"{layer_name}: K/V half size {k_bytes} is not {ALIGNMENT}-aligned"
            )
        logical_page_half = BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM * 2
        if k_bytes % logical_page_half:
            raise KvPoolBackendError(
                f"{layer_name}: raw K bytes={k_bytes} not divisible by "
                f"BF16 K-page bytes={logical_page_half}"
            )
        layer_num_blocks = k_bytes // logical_page_half
        if layer_num_blocks < scheduler_num_blocks:
            raise KvPoolBackendError(
                f"{layer_name}: physical blocks={layer_num_blocks} < "
                f"scheduler blocks={scheduler_num_blocks}"
            )
        if configured_num_blocks is None:
            configured_num_blocks = layer_num_blocks
        elif layer_num_blocks != configured_num_blocks:
            raise KvPoolBackendError("Step3p5 layers expose different KV capacities")
    assert configured_num_blocks is not None
    physical_num_blocks = max(
        int(configured_num_blocks),
        scheduler_num_blocks + PADDING_BLOCK_COUNT,
    )
    reserve = make_padding_reserve(
        scheduler_num_blocks,
        physical_num_blocks,
        block_size=BLOCK_SIZE,
    )
    entry_bytes = (
        physical_num_blocks
        * BLOCK_SIZE
        * NUM_KV_HEADS
        * HEAD_DIM
        * 2
    )
    return (
        by_index,
        mtp_by_index,
        reserve,
        int(entry_bytes),
    )


def _allocate_pool(total_bytes: int, device) -> tuple[Any, torch.Tensor]:
    pool = torch.npu.MemPool()
    with torch.npu.use_mem_pool(pool):
        big = torch.zeros(total_bytes, dtype=torch.int8, device=device)
    if int(big.data_ptr()) % ALIGNMENT:
        raise KvPoolBackendError(
            f"KV pool base {hex(int(big.data_ptr()))} is not {ALIGNMENT}-byte aligned"
        )
    return pool, big


def _write_export(
    *,
    rank: int,
    tp_world_size: int,
    base: int,
    total_bytes: int,
    scheduler_num_blocks: int,
    physical_num_blocks: int,
    entry_bytes: int,
    layer_to_group: Mapping[int, int],
) -> Any:
    out_dir = _dir()
    os.makedirs(out_dir, exist_ok=True)
    key_path = os.path.join(out_dir, f"pypto_kvpool.key.rank{rank}")
    map_path = os.path.join(out_dir, f"pypto_kvpool_map.json.rank{rank}")
    done_path = map_path + ".done"
    try:
        os.unlink(done_path)
    except FileNotFoundError:
        pass

    k_section_bytes = NUM_LAYERS * entry_bytes
    v_section_offset = _align_up(k_section_bytes)
    v_section_bytes = NUM_LAYERS * entry_bytes
    entries: dict[str, Any] = {}
    from tools.step3p5.kv_padding import make_padding_reserve  # noqa: PLC0415

    reserve = make_padding_reserve(
        scheduler_num_blocks,
        physical_num_blocks,
        block_size=BLOCK_SIZE,
    )
    num_slots = physical_num_blocks * BLOCK_SIZE
    for which, section_offset in (("K", 0), ("V", v_section_offset)):
        for layer in range(NUM_LAYERS):
            offset = section_offset + layer * entry_bytes
            entries[f"L{layer}.{which}"] = {
                "layer_idx": layer,
                "which": which,
                "group_id": int(layer_to_group[layer]),
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
                "dtype": KV_DTYPE_NAME,
                "head_dim": HEAD_DIM,
                "num_kv_heads": NUM_KV_HEADS,
                "block_size": BLOCK_SIZE,
            }
    from tools.step3p5.ipc_session import (  # noqa: PLC0415
        attach_session,
        maybe_start_owner,
        write_ready_manifest,
    )

    device_offset = int(os.environ.get("PYPTO_IPC_DEVICE_OFFSET", "0"))
    owner = maybe_start_owner(
        out_dir,
        role="kv",
        rank=rank,
        device_id=device_offset + rank,
    )
    map_obj = {
        "version": SCHEMA_VERSION,
        "layout": LAYOUT,
        "rank": rank,
        "tp_world_size": tp_world_size,
        "pool_base_debug": base,
        "pool_bytes": total_bytes,
        "num_layers": NUM_LAYERS,
        "head_dim": HEAD_DIM,
        "num_kv_heads": NUM_KV_HEADS,
        "block_size": BLOCK_SIZE,
        "dtype": KV_DTYPE_NAME,
        **reserve.as_dict(),
        "sections": {
            "K": {"offset": 0, "nbytes": k_section_bytes},
            "V": {"offset": v_section_offset, "nbytes": v_section_bytes},
        },
        "groups": [
            {
                "group_id": group_id,
                "layer_indices": [
                    layer for layer in range(NUM_LAYERS)
                    if layer_to_group[layer] == group_id
                ],
            }
            for group_id in sorted(set(layer_to_group.values()))
        ],
        "map": entries,
    }
    map_obj = attach_session(map_obj, owner)
    key = _export_key(base, total_bytes)
    tmp_key = key_path + ".tmp"
    tmp_map = map_path + ".tmp"
    with open(tmp_key, "wb") as file:
        file.write(key)
    with open(tmp_map, "w", encoding="utf-8") as file:
        json.dump(map_obj, file, indent=2, sort_keys=True)
    os.replace(tmp_key, key_path)
    os.replace(tmp_map, map_path)
    write_ready_manifest(
        done_path,
        map_path=map_path,
        key_path=key_path,
        owner=owner,
    )
    return owner


def _write_mtp_export(
    *,
    rank: int,
    tp_world_size: int,
    base: int,
    total_bytes: int,
    scheduler_num_blocks: int,
    physical_num_blocks: int,
    entry_bytes: int,
    layer_to_group: Mapping[int, int],
) -> Any:
    """Publish the independent MTP45/46/47 KV allocation."""
    out_dir = _dir()
    os.makedirs(out_dir, exist_ok=True)
    key_path = os.path.join(out_dir, f"pypto_mtp_kvpool.key.rank{rank}")
    map_path = os.path.join(out_dir, f"pypto_mtp_kvpool_map.json.rank{rank}")
    done_path = map_path + ".done"
    try:
        os.unlink(done_path)
    except FileNotFoundError:
        pass

    k_section_bytes = NUM_MTP_LAYERS * entry_bytes
    v_section_offset = _align_up(k_section_bytes)
    v_section_bytes = NUM_MTP_LAYERS * entry_bytes
    from tools.step3p5.kv_padding import make_padding_reserve  # noqa: PLC0415

    reserve = make_padding_reserve(
        scheduler_num_blocks,
        physical_num_blocks,
        block_size=BLOCK_SIZE,
    )
    num_slots = physical_num_blocks * BLOCK_SIZE
    entries: dict[str, Any] = {}
    for which, section_offset in (("K", 0), ("V", v_section_offset)):
        for local_idx in range(NUM_MTP_LAYERS):
            offset = section_offset + local_idx * entry_bytes
            entries[f"MTP{local_idx}.{which}"] = {
                "layer_idx": local_idx,
                "absolute_layer_idx": MTP_START_LAYER + local_idx,
                "which": which,
                "group_id": int(layer_to_group[local_idx]),
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
    from tools.step3p5.ipc_session import (  # noqa: PLC0415
        attach_session,
        maybe_start_owner,
        write_ready_manifest,
    )

    device_offset = int(os.environ.get("PYPTO_IPC_DEVICE_OFFSET", "0"))
    owner = maybe_start_owner(
        out_dir,
        role="mtp_kv",
        rank=rank,
        device_id=device_offset + rank,
    )
    map_obj = {
        "version": MTP_SCHEMA_VERSION,
        "layout": MTP_LAYOUT,
        "rank": rank,
        "tp_world_size": tp_world_size,
        "pool_base_debug": base,
        "pool_bytes": total_bytes,
        "num_layers": NUM_MTP_LAYERS,
        "head_dim": HEAD_DIM,
        "num_kv_heads": NUM_KV_HEADS,
        "block_size": BLOCK_SIZE,
        "dtype": KV_DTYPE_NAME,
        **reserve.as_dict(),
        "sections": {
            "K": {"offset": 0, "nbytes": k_section_bytes},
            "V": {"offset": v_section_offset, "nbytes": v_section_bytes},
        },
        "groups": [
            {
                "group_id": group_id,
                "layer_indices": [
                    local_idx
                    for local_idx in range(NUM_MTP_LAYERS)
                    if layer_to_group[local_idx] == group_id
                ],
            }
            for group_id in sorted(set(layer_to_group.values()))
        ],
        "map": entries,
    }
    map_obj = attach_session(map_obj, owner)
    key = _export_key(base, total_bytes)
    with open(key_path + ".tmp", "wb") as file:
        file.write(key)
    with open(map_path + ".tmp", "w", encoding="utf-8") as file:
        json.dump(map_obj, file, indent=2, sort_keys=True)
    os.replace(key_path + ".tmp", key_path)
    os.replace(map_path + ".tmp", map_path)
    write_ready_manifest(
        done_path,
        map_path=map_path,
        key_path=key_path,
        owner=owner,
    )
    return owner


def install() -> dict[str, Any]:
    global _PATCHED, _ORIGINAL  # noqa: PLW0603
    if _PATCHED:
        return {"ok": True, "already": True}
    import vllm_ascend.worker.model_runner_v1 as model_runner  # noqa: PLC0415
    from vllm.distributed import (  # noqa: PLC0415
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    runner_cls = model_runner.NPUModelRunner
    original = runner_cls._allocate_kv_cache_tensors
    _ORIGINAL = original

    def patched(self, kv_cache_config):
        by_index, mtp_by_index, reserve, entry_bytes = _validate_config(
            self, kv_cache_config
        )
        k_section_bytes = NUM_LAYERS * entry_bytes
        v_section_offset = _align_up(k_section_bytes)
        total_bytes = v_section_offset + NUM_LAYERS * entry_bytes
        pool, big = _allocate_pool(total_bytes, self.device)
        mtp_k_section_bytes = NUM_MTP_LAYERS * entry_bytes
        mtp_v_section_offset = _align_up(mtp_k_section_bytes)
        mtp_total_bytes = mtp_v_section_offset + NUM_MTP_LAYERS * entry_bytes
        mtp_pool, mtp_big = _allocate_pool(mtp_total_bytes, self.device)

        layer_to_group: dict[int, int] = {}
        mtp_layer_to_group: dict[int, int] = {}
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in group.layer_names:
                index = _layer_idx(layer_name)
                if index is not None and index < NUM_LAYERS:
                    layer_to_group[index] = group_id
                elif (
                    index is not None
                    and MTP_START_LAYER <= index < MTP_START_LAYER + NUM_MTP_LAYERS
                ):
                    mtp_layer_to_group[index - MTP_START_LAYER] = group_id
        if set(layer_to_group) != set(range(NUM_LAYERS)):
            raise KvPoolBackendError("KV groups do not cover all Step3p5 layers")
        if set(mtp_layer_to_group) != set(range(NUM_MTP_LAYERS)):
            raise KvPoolBackendError("KV groups do not cover MTP45/46/47")

        raw: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer in range(NUM_LAYERS):
            layer_name, _, _ = by_index[layer]
            k_offset = layer * entry_bytes
            v_offset = v_section_offset + layer * entry_bytes
            # Return raw byte views.  The unmodified vLLM-Ascend reshape path
            # applies the configured logical BF16 dtype and paged-cache shape.
            k_view = big[k_offset : k_offset + entry_bytes]
            v_view = big[v_offset : v_offset + entry_bytes]
            raw[layer_name] = (k_view, v_view)
        for local_idx in range(NUM_MTP_LAYERS):
            layer_name, _, _ = mtp_by_index[local_idx]
            k_offset = local_idx * entry_bytes
            v_offset = mtp_v_section_offset + local_idx * entry_bytes
            raw[layer_name] = (
                mtp_big[k_offset : k_offset + entry_bytes],
                mtp_big[v_offset : v_offset + entry_bytes],
            )

        _POOLS.extend((pool, mtp_pool))
        _BIG.extend((big, mtp_big))
        rank = get_tensor_model_parallel_rank()
        tp = get_tensor_model_parallel_world_size()
        owner = _write_export(
            rank=rank,
            tp_world_size=tp,
            base=int(big.data_ptr()),
            total_bytes=total_bytes,
            scheduler_num_blocks=reserve.scheduler_num_blocks,
            physical_num_blocks=reserve.physical_num_blocks,
            entry_bytes=entry_bytes,
            layer_to_group=layer_to_group,
        )
        if owner is not None:
            _SESSION_OWNERS.append(owner)
        mtp_owner = _write_mtp_export(
            rank=rank,
            tp_world_size=tp,
            base=int(mtp_big.data_ptr()),
            total_bytes=mtp_total_bytes,
            scheduler_num_blocks=reserve.scheduler_num_blocks,
            physical_num_blocks=reserve.physical_num_blocks,
            entry_bytes=entry_bytes,
            layer_to_group=mtp_layer_to_group,
        )
        if mtp_owner is not None:
            _SESSION_OWNERS.append(mtp_owner)
        _RESERVES["main"] = reserve
        _RESERVES["mtp"] = reserve
        print(
            f"[pypto-kvpool] rank={rank} main_bytes={total_bytes} "
            f"mtp_bytes={mtp_total_bytes} "
            f"scheduler_blocks={reserve.scheduler_num_blocks} "
            f"physical_blocks={reserve.physical_num_blocks} "
            f"entry_bytes={entry_bytes}; pools are independent -> {_dir()}",
            flush=True,
        )
        return raw

    runner_cls._allocate_kv_cache_tensors = patched
    _PATCHED = True
    return {"ok": True, "patched": True, "layout": LAYOUT}


def uninstall() -> dict[str, Any]:
    global _PATCHED, _ORIGINAL  # noqa: PLW0603
    if not _PATCHED:
        return {"ok": True, "installed": False}
    import vllm_ascend.worker.model_runner_v1 as model_runner  # noqa: PLC0415

    model_runner.NPUModelRunner._allocate_kv_cache_tensors = _ORIGINAL
    _PATCHED = False
    _ORIGINAL = None
    return {"ok": True, "uninstalled": True}


def maybe_autoload() -> dict[str, Any]:
    if os.environ.get("PYPTO_KVPOOL", "") == "1":
        return install()
    return {"ok": True, "skipped": "PYPTO_KVPOOL != 1"}


def status() -> dict[str, Any]:
    return {
        "installed": _PATCHED,
        "layout": LAYOUT,
        "dir": _dir(),
        "resident_pools": len(_BIG),
        "padding_reserves": {
            name: reserve.as_dict() for name, reserve in _RESERVES.items()
        },
    }


def get_padding_reserve(domain: str):
    """Return the allocator-owned Main or MTP reserve for this vLLM rank."""
    name = str(domain).lower()
    if name not in {"main", "mtp"}:
        raise KvPoolBackendError(
            f"padding reserve domain must be main or mtp, got {domain!r}"
        )
    try:
        return _RESERVES[name]
    except KeyError as exc:
        raise KvPoolBackendError(
            f"{name} padding reserve is unavailable; PYPTO_KVPOOL allocation "
            "must complete before live metadata extraction"
        ) from exc


def _selftest() -> int:
    """Validate the resolved-config seam without importing vLLM or touching NPU."""
    class _Spec:
        block_size = BLOCK_SIZE
        num_kv_heads = NUM_KV_HEADS
        head_size = HEAD_DIM
        dtype = KV_DTYPE

    class _Tensor:
        def __init__(self, layer_name: str, size: int):
            self.shared_by = [layer_name]
            self.size = size

    class _Group:
        def __init__(self, layer_names):
            self.layer_names = list(layer_names)
            self.kv_cache_spec = _Spec()

    class _Config:
        def __init__(self, groups, tensors, num_blocks):
            self.kv_cache_groups = groups
            self.kv_cache_tensors = tensors
            self.num_blocks = num_blocks

    class _TextConfig:
        model_type = "step3p5"

    class _ModelConfig:
        hf_text_config = _TextConfig()

    class _Runner:
        model_config = _ModelConfig()

    names = [
        f"model.layers.{idx}.self_attn.attn" for idx in range(NUM_LAYERS)
    ]
    mtp_names = [
        f"model.layers.{MTP_START_LAYER + idx}.mtp_block.self_attn.attn"
        for idx in range(NUM_MTP_LAYERS)
    ]
    all_names = names + mtp_names
    num_blocks = 3
    entry_bytes = BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM * 2 * num_blocks
    tensors = [_Tensor(name, entry_bytes * 2) for name in all_names]
    valid = _Config([_Group(all_names)], tensors, num_blocks)
    ok = True
    try:
        _, mtp, reserve, got_entry = _validate_config(_Runner(), valid)
        ok &= reserve.scheduler_num_blocks == num_blocks
        ok &= reserve.physical_num_blocks == num_blocks + 15
        ok &= got_entry == entry_bytes + 15 * BLOCK_SIZE * HEAD_DIM * 2
        ok &= len(mtp) == NUM_MTP_LAYERS
        print("[selftest] valid main+MTP one-group config -> PASS", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[selftest] valid one-group config -> FAIL: {exc}", flush=True)
        ok = False

    def reject(label: str, config) -> None:
        nonlocal ok
        try:
            _validate_config(_Runner(), config)
        except KvPoolBackendError:
            print(f"[selftest] reject {label} -> PASS", flush=True)
        else:
            print(f"[selftest] reject {label} -> FAIL", flush=True)
            ok = False

    reject(
        "multiple-groups",
        _Config(
            [_Group(all_names[:24]), _Group(all_names[24:])],
            tensors,
            num_blocks,
        ),
    )

    bad_dtype_spec = _Spec()
    bad_dtype_spec.dtype = torch.float16
    bad_dtype_group = _Group(all_names)
    bad_dtype_group.kv_cache_spec = bad_dtype_spec
    reject("dtype", _Config([bad_dtype_group], tensors, num_blocks))

    bad_shared = list(tensors)
    bad_shared[0] = _Tensor(all_names[0], entry_bytes * 2)
    bad_shared[0].shared_by = [all_names[0], all_names[1]]
    reject("shared-by", _Config([_Group(all_names)], bad_shared, num_blocks))

    print(
        f"[selftest] RESULT={'KVPOOL_CONFIG_SEAM_OK' if ok else 'FAIL'}",
        flush=True,
    )
    return 0 if ok else 1


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return _selftest()
    print("nothing to do; pass --selftest", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
