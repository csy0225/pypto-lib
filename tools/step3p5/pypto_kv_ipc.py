# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Strict vLLM paged-KV IPC map validator/importer.

The live bridge has one deliberately narrow ABI:

* one exported allocation per TP rank;
* K-major then V-major physical order;
* exactly 45 Step3p5 decoder layers;
* logical cache dtype ``bfloat16`` and head dimension 128;
* one local KV head per TP rank;
* vLLM's block size is 128;
* every entry and section starts on a 512-byte boundary.

The exporter may use a raw ``int8`` backing tensor because vLLM's Ascend
allocator reshapes raw bytes into the configured logical cache dtype later.
The map therefore describes the *logical* BF16 view, never the raw allocator
dtype.  Import is fail-closed when the map does not prove that view.

The validator is pure Python and can be used without a device.  Only
``kv_device_tensor`` and ``import_kv_all`` touch the PyPTO runtime.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple

_SCHEMA_VERSION = 3
_LAYOUT = "flat_k_major_v_major_v1"
_NUM_LAYERS = 45
_HEAD_DIM = 128
_NUM_KV_HEADS = 1
_KV_DTYPE = "bfloat16"
_KV_ITEMSIZE = 2
_BLOCK_SIZE = 128
_ALIGNMENT = 512
_WHICH = ("K", "V")


class KvIpcMapError(ValueError):
    """Raised when an IPC map violates the live KV ABI."""


def _kv_key(layer_idx: int, which: str) -> str:
    if which not in _WHICH:
        raise ValueError(f"which must be K or V, got {which!r}")
    return f"L{int(layer_idx)}.{which}"


def _align_up(value: int, alignment: int = _ALIGNMENT) -> int:
    if value < 0:
        raise KvIpcMapError(f"negative value cannot be aligned: {value}")
    return (value + alignment - 1) // alignment * alignment


def _as_int(obj: Mapping[str, Any], key: str, *, where: str) -> int:
    if key not in obj or isinstance(obj[key], bool):
        raise KvIpcMapError(f"{where}: missing integer field {key!r}")
    try:
        value = int(obj[key])
    except (TypeError, ValueError) as exc:
        raise KvIpcMapError(f"{where}: field {key!r} is not an integer") from exc
    return value


def _require_shape(value: Any, expected: Iterable[int], *, where: str) -> Tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise KvIpcMapError(f"{where}: shape must be a list")
    try:
        shape = tuple(int(x) for x in value)
    except (TypeError, ValueError) as exc:
        raise KvIpcMapError(f"{where}: shape contains a non-integer") from exc
    expected_tuple = tuple(int(x) for x in expected)
    if shape != expected_tuple:
        raise KvIpcMapError(f"{where}: shape {shape} != expected {expected_tuple}")
    return shape


@dataclass(frozen=True)
class KvEntry:
    layer_idx: int
    which: str
    group_id: int
    offset: int
    nbytes: int
    num_blocks: int
    num_slots: int
    shape: Tuple[int, int, int, int]
    flat_shape: Tuple[int, int]
    dtype: str
    head_dim: int
    num_kv_heads: int
    block_size: int


@dataclass(frozen=True)
class KvMapSummary:
    version: int
    layout: str
    rank: int
    tp_world_size: int
    pool_bytes: int
    num_layers: int
    head_dim: int
    num_kv_heads: int
    block_size: int
    dtype: str
    scheduler_num_blocks: int
    physical_num_blocks: int
    padding_block_ids: Tuple[int, ...]
    k_section: Tuple[int, int]
    v_section: Tuple[int, int]
    entries: Tuple[KvEntry, ...]


def _validate_range(offset: int, nbytes: int, pool_bytes: int, *, where: str) -> None:
    if offset % _ALIGNMENT:
        raise KvIpcMapError(f"{where}: offset {offset} is not {_ALIGNMENT}-byte aligned")
    if nbytes <= 0:
        raise KvIpcMapError(f"{where}: nbytes must be positive")
    if nbytes % _ALIGNMENT:
        raise KvIpcMapError(f"{where}: nbytes {nbytes} is not {_ALIGNMENT}-byte aligned")
    if offset < 0 or offset + nbytes > pool_bytes:
        raise KvIpcMapError(
            f"{where}: range [{offset}, {offset + nbytes}) exceeds pool_bytes={pool_bytes}"
        )


def validate_pool_map(pool_map: Mapping[str, Any]) -> KvMapSummary:
    """Validate and normalize a v2 K-major/V-major map.

    This function intentionally does not accept the historical interleaved
    v1 map.  It is safer to fail closed than to silently bind a layer's K/V
    view to the wrong physical range.
    """
    version = _as_int(pool_map, "version", where="map")
    if version != _SCHEMA_VERSION:
        raise KvIpcMapError(f"map: version {version} != supported {_SCHEMA_VERSION}")
    layout = pool_map.get("layout")
    if layout != _LAYOUT:
        raise KvIpcMapError(f"map: layout {layout!r} != supported {_LAYOUT!r}")

    rank = _as_int(pool_map, "rank", where="map")
    tp_world_size = _as_int(pool_map, "tp_world_size", where="map")
    pool_bytes = _as_int(pool_map, "pool_bytes", where="map")
    num_layers = _as_int(pool_map, "num_layers", where="map")
    head_dim = _as_int(pool_map, "head_dim", where="map")
    num_kv_heads = _as_int(pool_map, "num_kv_heads", where="map")
    block_size = _as_int(pool_map, "block_size", where="map")
    dtype = pool_map.get("dtype")

    if rank < 0 or rank >= tp_world_size:
        raise KvIpcMapError(f"map: rank={rank} outside tp_world_size={tp_world_size}")
    if pool_bytes <= 0 or pool_bytes % _ALIGNMENT:
        raise KvIpcMapError(f"map: pool_bytes={pool_bytes} is not a positive aligned size")
    if num_layers != _NUM_LAYERS:
        raise KvIpcMapError(f"map: num_layers={num_layers} != {_NUM_LAYERS}")
    if head_dim != _HEAD_DIM:
        raise KvIpcMapError(f"map: head_dim={head_dim} != {_HEAD_DIM}")
    if num_kv_heads != _NUM_KV_HEADS:
        raise KvIpcMapError(f"map: num_kv_heads={num_kv_heads} != {_NUM_KV_HEADS}")
    if block_size != _BLOCK_SIZE:
        raise KvIpcMapError(f"map: block_size={block_size} != {_BLOCK_SIZE}")
    if dtype != _KV_DTYPE:
        raise KvIpcMapError(f"map: dtype={dtype!r} != {_KV_DTYPE!r}")
    from tools.step3p5.kv_padding import (  # noqa: PLC0415
        PaddingReserveError,
        parse_padding_reserve,
    )

    try:
        reserve = parse_padding_reserve(pool_map, where="map")
    except PaddingReserveError as exc:
        raise KvIpcMapError(str(exc)) from exc

    sections = pool_map.get("sections")
    if not isinstance(sections, Mapping):
        raise KvIpcMapError("map: sections must be an object")
    section_ranges: Dict[str, Tuple[int, int]] = {}
    for which in _WHICH:
        section = sections.get(which)
        if not isinstance(section, Mapping):
            raise KvIpcMapError(f"map: missing sections[{which!r}]")
        off = _as_int(section, "offset", where=f"sections[{which}]")
        size = _as_int(section, "nbytes", where=f"sections[{which}]")
        _validate_range(off, size, pool_bytes, where=f"sections[{which}]")
        section_ranges[which] = (off, size)
    k_off, k_size = section_ranges["K"]
    v_off, v_size = section_ranges["V"]
    if k_off != 0:
        raise KvIpcMapError(f"map: K section must start at 0, got {k_off}")
    minimum_v_off = _align_up(k_off + k_size)
    if v_off < minimum_v_off or v_off % _ALIGNMENT:
        raise KvIpcMapError(
            f"map: V section offset {v_off} precedes aligned K end {minimum_v_off}"
        )
    if k_off + k_size > v_off:
        raise KvIpcMapError("map: K/V sections overlap")
    if v_off + v_size > pool_bytes:
        raise KvIpcMapError("map: V section exceeds pool")

    raw_entries = pool_map.get("map")
    if not isinstance(raw_entries, Mapping):
        raise KvIpcMapError("map: map must be an object")
    expected_keys = {_kv_key(layer, which) for layer in range(_NUM_LAYERS) for which in _WHICH}
    if set(raw_entries) != expected_keys:
        missing = sorted(expected_keys - set(raw_entries))
        extra = sorted(set(raw_entries) - expected_keys)
        raise KvIpcMapError(f"map: entries mismatch; missing={missing[:4]} extra={extra[:4]}")

    entries: List[KvEntry] = []
    ranges: List[Tuple[int, int, str]] = []
    for which in _WHICH:
        section_off, section_size = section_ranges[which]
        expected_offset = section_off
        previous_num_blocks = None
        previous_group_id = None
        for layer_idx in range(_NUM_LAYERS):
            key = _kv_key(layer_idx, which)
            raw = raw_entries[key]
            if not isinstance(raw, Mapping):
                raise KvIpcMapError(f"{key}: entry must be an object")
            where = f"map[{key}]"
            entry_layer = _as_int(raw, "layer_idx", where=where)
            entry_which = raw.get("which")
            group_id = _as_int(raw, "group_id", where=where)
            offset = _as_int(raw, "offset", where=where)
            nbytes = _as_int(raw, "nbytes", where=where)
            num_blocks = _as_int(raw, "num_blocks", where=where)
            num_slots = _as_int(raw, "num_slots", where=where)
            entry_head_dim = _as_int(raw, "head_dim", where=where)
            entry_kv_heads = _as_int(raw, "num_kv_heads", where=where)
            entry_block_size = _as_int(raw, "block_size", where=where)
            entry_dtype = raw.get("dtype")
            if entry_layer != layer_idx or entry_which != which:
                raise KvIpcMapError(f"{where}: layer/which identity is inconsistent")
            if group_id < 0:
                raise KvIpcMapError(f"{where}: group_id must be non-negative")
            if entry_dtype != _KV_DTYPE or entry_head_dim != _HEAD_DIM:
                raise KvIpcMapError(f"{where}: logical dtype/head_dim are not BF16/128")
            if entry_kv_heads != _NUM_KV_HEADS or entry_block_size != _BLOCK_SIZE:
                raise KvIpcMapError(f"{where}: KV heads/block size are not 1/128")
            if num_blocks <= 0 or num_slots != num_blocks * _BLOCK_SIZE:
                raise KvIpcMapError(f"{where}: num_slots must equal num_blocks*block_size")
            if num_blocks != reserve.physical_num_blocks:
                raise KvIpcMapError(
                    f"{where}: num_blocks={num_blocks} != "
                    f"physical_num_blocks={reserve.physical_num_blocks}"
                )
            expected_nbytes = num_slots * _NUM_KV_HEADS * _HEAD_DIM * _KV_ITEMSIZE
            if nbytes != expected_nbytes:
                raise KvIpcMapError(
                    f"{where}: nbytes={nbytes} != logical BF16 bytes={expected_nbytes}"
                )
            _require_shape(raw.get("shape"), (num_blocks, _BLOCK_SIZE, 1, _HEAD_DIM), where=where)
            _require_shape(raw.get("flat_shape"), (num_slots, _HEAD_DIM), where=where)
            _validate_range(offset, nbytes, pool_bytes, where=where)
            if offset != expected_offset:
                raise KvIpcMapError(
                    f"{where}: offset {offset} breaks contiguous {which} section; expected {expected_offset}"
                )
            if offset + nbytes > section_off + section_size:
                raise KvIpcMapError(f"{where}: entry exceeds {which} section")
            if previous_num_blocks is not None and num_blocks != previous_num_blocks:
                raise KvIpcMapError(f"{where}: num_blocks differs between layers")
            if previous_group_id is not None and group_id == previous_group_id:
                # Same group can contain several layers; this check is only to
                # retain the value in the normalized object, not to reject it.
                pass
            previous_num_blocks = num_blocks
            previous_group_id = group_id
            ranges.append((offset, offset + nbytes, key))
            entries.append(
                KvEntry(
                    layer_idx=layer_idx,
                    which=which,
                    group_id=group_id,
                    offset=offset,
                    nbytes=nbytes,
                    num_blocks=num_blocks,
                    num_slots=num_slots,
                    shape=(num_blocks, _BLOCK_SIZE, 1, _HEAD_DIM),
                    flat_shape=(num_slots, _HEAD_DIM),
                    dtype=_KV_DTYPE,
                    head_dim=_HEAD_DIM,
                    num_kv_heads=1,
                    block_size=_BLOCK_SIZE,
                )
            )
            expected_offset = _align_up(offset + nbytes)
        if expected_offset != section_off + section_size:
            raise KvIpcMapError(
                f"{which} section size {section_size} does not equal its contiguous entries"
            )

    for start, end, key in sorted(ranges):
        del end
        # The previous range in sorted order is checked below.  Keep this loop
        # explicit so the error identifies the two logical entries.
        for other_start, other_end, other_key in ranges:
            if key == other_key:
                continue
            if start < other_end and other_start < start:
                raise KvIpcMapError(f"map ranges overlap: {key} and {other_key}")

    return KvMapSummary(
        version=version,
        layout=layout,
        rank=rank,
        tp_world_size=tp_world_size,
        pool_bytes=pool_bytes,
        num_layers=num_layers,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        dtype=dtype,
        scheduler_num_blocks=reserve.scheduler_num_blocks,
        physical_num_blocks=reserve.physical_num_blocks,
        padding_block_ids=reserve.padding_block_ids,
        k_section=section_ranges["K"],
        v_section=section_ranges["V"],
        entries=tuple(sorted(entries, key=lambda e: (e.layer_idx, e.which))),
    )


class KvIpcMap:
    """One rank's validated imported KV pool."""

    def __init__(self, peer_base: int, pool_map: Mapping[str, Any]) -> None:
        self.peer_base = int(peer_base)
        self.pool_map = dict(pool_map)
        self.summary = validate_pool_map(pool_map)
        self._entries = {
            (entry.layer_idx, entry.which): entry for entry in self.summary.entries
        }

    @classmethod
    def from_map_obj(cls, peer_base: int, pool_map: Mapping[str, Any]) -> "KvIpcMap":
        return cls(peer_base, pool_map)

    def _entry(self, layer_idx: int, which: str) -> KvEntry:
        try:
            return self._entries[(int(layer_idx), which)]
        except KeyError as exc:
            raise KeyError(f"KV map missing L{layer_idx}.{which}") from exc

    def num_slots(self, layer_idx: int, which: str = "K") -> int:
        return self._entry(layer_idx, which).num_slots

    def kv_spec(self, layer_idx: int, which: str) -> Tuple[int, Tuple[int, int], str]:
        entry = self._entry(layer_idx, which)
        return entry.offset, entry.flat_shape, entry.dtype

    def kv_device_tensor(self, layer_idx: int, which: str):
        import torch  # noqa: PLC0415
        from pypto.runtime.device_tensor import DeviceTensor  # noqa: PLC0415

        entry = self._entry(layer_idx, which)
        return DeviceTensor(
            self.peer_base + entry.offset,
            entry.flat_shape,
            torch.bfloat16,
        )

    def kv_pair(self, layer_idx: int):
        return self.kv_device_tensor(layer_idx, "K"), self.kv_device_tensor(layer_idx, "V")

    def section_spec(self, which: str) -> Tuple[int, Tuple[int, int], str]:
        """Return one contiguous whole-net K or V section."""
        if which not in _WHICH:
            raise ValueError(f"which must be K or V, got {which!r}")
        offset, nbytes = self.summary.k_section if which == "K" else self.summary.v_section
        rows = nbytes // (_HEAD_DIM * _KV_ITEMSIZE)
        if rows != _NUM_LAYERS * self.num_slots(0, which):
            raise KvIpcMapError(
                f"{which} section rows={rows} do not equal "
                f"{_NUM_LAYERS}*slots={_NUM_LAYERS * self.num_slots(0, which)}"
            )
        return offset, (rows, _HEAD_DIM), _KV_DTYPE

    def section_device_tensor(self, which: str):
        import torch  # noqa: PLC0415
        from pypto.runtime.device_tensor import DeviceTensor  # noqa: PLC0415

        offset, shape, _ = self.section_spec(which)
        return DeviceTensor(self.peer_base + offset, shape, torch.bfloat16)


def import_kv_all(rt, out_dir: str, *, tp: int, dev_offset: int = 0) -> List[KvIpcMap]:
    """Import and validate every rank's one-key KV pool."""
    device_key_map: Dict[int, bytes] = {}
    maps_json: List[Mapping[str, Any]] = []
    from tools.step3p5.ipc_session import (  # noqa: PLC0415
        validate_key_file,
        validate_live_session,
    )

    for rank in range(tp):
        key_path = os.path.join(out_dir, f"pypto_kvpool.key.rank{rank}")
        map_path = os.path.join(out_dir, f"pypto_kvpool_map.json.rank{rank}")
        ready_path = map_path + ".done"
        key = validate_key_file(key_path)
        with open(map_path) as f:
            pool_map = json.load(f)
        validate_live_session(
            pool_map,
            expected_rank=rank,
            expected_tp=tp,
            expected_device_id=dev_offset + rank,
            expected_role="kv",
            ready_path=ready_path,
            map_path=map_path,
            key_path=key_path,
        )
        device_key_map[dev_offset + rank] = key
        maps_json.append(pool_map)
    for obj in maps_json:
        summary = validate_pool_map(obj)
        if summary.tp_world_size != tp:
            raise KvIpcMapError(
                f"rank {summary.rank}: tp_world_size={summary.tp_world_size} != requested {tp}"
            )
    first = validate_pool_map(maps_json[0])
    for obj in maps_json[1:]:
        summary = validate_pool_map(obj)
        if (
            summary.scheduler_num_blocks != first.scheduler_num_blocks
            or summary.physical_num_blocks != first.physical_num_blocks
            or summary.padding_block_ids != first.padding_block_ids
        ):
            raise KvIpcMapError(
                "TP ranks expose different Main padding reserves"
            )
    vas = rt.import_ipc_all(device_key_map)
    print(
        "[kv-ipc importer] import_ipc_all peer_bases="
        + str([hex(vas[dev_offset + rank]) for rank in range(tp)]),
        flush=True,
    )
    return [KvIpcMap(vas[dev_offset + rank], maps_json[rank]) for rank in range(tp)]


def build_stacked_kv(kv_maps: List[KvIpcMap], layer_idx: int):
    """Build a per-layer TP stacked K/V view from validated maps."""
    from pypto.runtime.device_tensor import StackedDeviceTensor  # noqa: PLC0415

    tp = len(kv_maps)
    k_shards = [item.kv_device_tensor(layer_idx, "K") for item in kv_maps]
    v_shards = [item.kv_device_tensor(layer_idx, "V") for item in kv_maps]
    shape = kv_maps[0]._entry(layer_idx, "K").flat_shape
    return (
        StackedDeviceTensor(k_shards, (tp, *shape), list(range(tp))),
        StackedDeviceTensor(v_shards, (tp, *shape), list(range(tp))),
    )


def build_stacked_kv_pool(kv_maps: List[KvIpcMap]):
    """Build whole-net flat K/V tensors ``[tp, 45*num_slots, 128]``."""
    if not kv_maps:
        raise KvIpcMapError("cannot stack an empty KV map list")
    from pypto.runtime.device_tensor import StackedDeviceTensor  # noqa: PLC0415

    tp = len(kv_maps)
    k_shape = kv_maps[0].section_spec("K")[1]
    v_shape = kv_maps[0].section_spec("V")[1]
    if k_shape != v_shape:
        raise KvIpcMapError(f"K/V section shapes differ: {k_shape} vs {v_shape}")
    for item in kv_maps[1:]:
        if item.section_spec("K")[1] != k_shape or item.section_spec("V")[1] != v_shape:
            raise KvIpcMapError("TP ranks expose different KV section shapes")
    k_shards = [item.section_device_tensor("K") for item in kv_maps]
    v_shards = [item.section_device_tensor("V") for item in kv_maps]
    return (
        StackedDeviceTensor(k_shards, (tp, *k_shape), list(range(tp))),
        StackedDeviceTensor(v_shards, (tp, *v_shape), list(range(tp))),
    )


def _synthetic_map(
    *,
    num_blocks: int = 32,
    scheduler_num_blocks: int = 17,
    group_count: int = 4,
    bad: str | None = None,
) -> Dict[str, Any]:
    per_entry = num_blocks * _BLOCK_SIZE * _NUM_KV_HEADS * _HEAD_DIM * _KV_ITEMSIZE
    entries: Dict[str, Any] = {}
    k_offset = 0
    for layer in range(_NUM_LAYERS):
        entries[_kv_key(layer, "K")] = {
            "layer_idx": layer,
            "which": "K",
            "group_id": layer % group_count,
            "offset": k_offset,
            "nbytes": per_entry,
            "num_blocks": num_blocks,
            "num_slots": num_blocks * _BLOCK_SIZE,
            "shape": [num_blocks, _BLOCK_SIZE, 1, _HEAD_DIM],
            "flat_shape": [num_blocks * _BLOCK_SIZE, _HEAD_DIM],
            "dtype": _KV_DTYPE,
            "head_dim": _HEAD_DIM,
            "num_kv_heads": 1,
            "block_size": _BLOCK_SIZE,
        }
        k_offset += per_entry
    v_offset = _align_up(k_offset)
    for layer in range(_NUM_LAYERS):
        entries[_kv_key(layer, "V")] = {
            "layer_idx": layer,
            "which": "V",
            "group_id": layer % group_count,
            "offset": v_offset,
            "nbytes": per_entry,
            "num_blocks": num_blocks,
            "num_slots": num_blocks * _BLOCK_SIZE,
            "shape": [num_blocks, _BLOCK_SIZE, 1, _HEAD_DIM],
            "flat_shape": [num_blocks * _BLOCK_SIZE, _HEAD_DIM],
            "dtype": _KV_DTYPE,
            "head_dim": _HEAD_DIM,
            "num_kv_heads": 1,
            "block_size": _BLOCK_SIZE,
        }
        v_offset += per_entry
    obj: Dict[str, Any] = {
        "version": _SCHEMA_VERSION,
        "layout": _LAYOUT,
        "rank": 0,
        "tp_world_size": 8,
        "pool_bytes": v_offset,
        "num_layers": _NUM_LAYERS,
        "head_dim": _HEAD_DIM,
        "num_kv_heads": 1,
        "block_size": _BLOCK_SIZE,
        "dtype": _KV_DTYPE,
        "scheduler_num_blocks": scheduler_num_blocks,
        "physical_num_blocks": num_blocks,
        "reserve_start": scheduler_num_blocks,
        "padding_block_ids": list(
            range(scheduler_num_blocks, scheduler_num_blocks + 15)
        ),
        "padding_block_count": 15,
        "sections": {
            "K": {"offset": 0, "nbytes": k_offset},
            "V": {"offset": v_offset - k_offset, "nbytes": k_offset},
        },
        "map": entries,
    }
    if bad == "interleaved":
        obj["layout"] = "interleaved_v1"
    elif bad == "overlap":
        obj["map"]["L1.K"]["offset"] = obj["map"]["L0.K"]["offset"]
    elif bad == "dtype":
        obj["dtype"] = "float16"
    elif bad == "alignment":
        obj["map"]["L0.K"]["offset"] = 1
    elif bad == "reserve_missing":
        del obj["padding_block_ids"]
    elif bad == "reserve_overlap":
        obj["reserve_start"] = scheduler_num_blocks - 1
    elif bad == "reserve_out_of_range":
        obj["physical_num_blocks"] = scheduler_num_blocks + 14
    return obj


def _selftest() -> int:
    ok = True
    valid = _synthetic_map()
    try:
        summary = validate_pool_map(valid)
        ok &= len(summary.entries) == 90
        ok &= summary.k_section[0] == 0
        ok &= summary.v_section[0] > summary.k_section[0]
        print("[selftest] valid K-major/V-major map -> PASS", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[selftest] valid map -> FAIL: {exc}", flush=True)
        ok = False

    for bad in (
        "interleaved",
        "overlap",
        "dtype",
        "alignment",
        "reserve_missing",
        "reserve_overlap",
        "reserve_out_of_range",
    ):
        try:
            validate_pool_map(_synthetic_map(bad=bad))
        except KvIpcMapError:
            print(f"[selftest] reject {bad} -> PASS", flush=True)
        else:
            print(f"[selftest] reject {bad} -> FAIL", flush=True)
            ok = False
    print(
        f"[selftest] RESULT={'KV_IPC_MAP_VALIDATOR_OK' if ok else 'FAIL'}",
        flush=True,
    )
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return _selftest()
    print("nothing to do; pass --selftest", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
