# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Strict IPC importer for the three vLLM-owned Step3p5 MTP KV pools.

Main decoder KV and MTP decoder KV are separate ownership domains.  The main
pool contains layers 0..44 and is imported by :mod:`pypto_kv_ipc`; this module
contains only layers 45..47 and uses a distinct ``mtp_kv`` IPC role.

The physical layout is deliberately simple and matches the main pool:

``K(MTP45)..K(MTP47), aligned gap, V(MTP45)..V(MTP47)``

The selected PyPTO MTP program consumes one flat view
``[3 * num_slots, 128]`` and uses a compile-time layer index to select its
slice.  No MTP shared-head, logits, token or acceptance object is part of this
ABI.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple

_SCHEMA_VERSION = 2
_LAYOUT = "mtp_flat_k_major_v_major_v1"
_NUM_LAYERS = 3
_HEAD_DIM = 128
_NUM_KV_HEADS = 1
_KV_DTYPE = "bfloat16"
_KV_ITEMSIZE = 2
_BLOCK_SIZE = 128
_ALIGNMENT = 512
_WHICH = ("K", "V")


class MtpKvIpcMapError(ValueError):
    """Raised when an MTP KV IPC map violates the live ABI."""


def _key(layer_idx: int, which: str) -> str:
    if which not in _WHICH:
        raise ValueError(f"which must be K or V, got {which!r}")
    return f"MTP{int(layer_idx)}.{which}"


def _align_up(value: int, alignment: int = _ALIGNMENT) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


def _as_int(obj: Mapping[str, Any], name: str, *, where: str) -> int:
    value = obj.get(name)
    if isinstance(value, bool):
        raise MtpKvIpcMapError(f"{where}: {name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise MtpKvIpcMapError(f"{where}: {name} is not an integer") from exc


def _shape(value: Any, expected: Iterable[int], *, where: str) -> Tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise MtpKvIpcMapError(f"{where}: shape must be a list")
    try:
        got = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise MtpKvIpcMapError(f"{where}: shape contains a non-integer") from exc
    want = tuple(int(item) for item in expected)
    if got != want:
        raise MtpKvIpcMapError(f"{where}: shape {got} != expected {want}")
    return got


def _validate_range(offset: int, nbytes: int, pool_bytes: int, *, where: str) -> None:
    if offset < 0 or offset % _ALIGNMENT:
        raise MtpKvIpcMapError(
            f"{where}: offset={offset} must be non-negative and {_ALIGNMENT}-aligned"
        )
    if nbytes <= 0 or nbytes % _ALIGNMENT:
        raise MtpKvIpcMapError(
            f"{where}: nbytes={nbytes} must be positive and {_ALIGNMENT}-aligned"
        )
    if offset + nbytes > pool_bytes:
        raise MtpKvIpcMapError(
            f"{where}: range [{offset},{offset+nbytes}) exceeds pool_bytes={pool_bytes}"
        )


@dataclass(frozen=True)
class MtpKvEntry:
    layer_idx: int
    which: str
    group_id: int
    offset: int
    nbytes: int
    num_blocks: int
    num_slots: int
    shape: Tuple[int, int, int, int]
    flat_shape: Tuple[int, int]


@dataclass(frozen=True)
class MtpKvMapSummary:
    version: int
    layout: str
    rank: int
    tp_world_size: int
    pool_bytes: int
    num_blocks: int
    scheduler_num_blocks: int
    physical_num_blocks: int
    padding_block_ids: Tuple[int, ...]
    k_section: Tuple[int, int]
    v_section: Tuple[int, int]
    entries: Tuple[MtpKvEntry, ...]


def validate_mtp_pool_map(pool_map: Mapping[str, Any]) -> MtpKvMapSummary:
    """Validate and normalize one rank's MTP KV map before ACL import."""
    if int(pool_map.get("version", -1)) != _SCHEMA_VERSION:
        raise MtpKvIpcMapError("unsupported MTP KV map version")
    if pool_map.get("layout") != _LAYOUT:
        raise MtpKvIpcMapError(
            f"layout {pool_map.get('layout')!r} != supported {_LAYOUT!r}"
        )

    rank = _as_int(pool_map, "rank", where="map")
    tp = _as_int(pool_map, "tp_world_size", where="map")
    pool_bytes = _as_int(pool_map, "pool_bytes", where="map")
    if rank < 0 or rank >= tp:
        raise MtpKvIpcMapError(f"rank={rank} outside tp_world_size={tp}")
    if pool_bytes <= 0 or pool_bytes % _ALIGNMENT:
        raise MtpKvIpcMapError("pool_bytes must be a positive aligned size")
    for name, expected in (
        ("num_layers", _NUM_LAYERS),
        ("head_dim", _HEAD_DIM),
        ("num_kv_heads", _NUM_KV_HEADS),
        ("block_size", _BLOCK_SIZE),
    ):
        if _as_int(pool_map, name, where="map") != expected:
            raise MtpKvIpcMapError(f"map {name} is not {expected}")
    if pool_map.get("dtype") != _KV_DTYPE:
        raise MtpKvIpcMapError("MTP KV map dtype must be bfloat16")
    from tools.step3p5.kv_padding import (  # noqa: PLC0415
        PaddingReserveError,
        parse_padding_reserve,
    )

    try:
        reserve = parse_padding_reserve(pool_map, where="map")
    except PaddingReserveError as exc:
        raise MtpKvIpcMapError(str(exc)) from exc

    sections = pool_map.get("sections")
    if not isinstance(sections, Mapping):
        raise MtpKvIpcMapError("map.sections must be an object")
    ranges: dict[str, tuple[int, int]] = {}
    for which in _WHICH:
        section = sections.get(which)
        if not isinstance(section, Mapping):
            raise MtpKvIpcMapError(f"missing sections[{which!r}]")
        off = _as_int(section, "offset", where=f"sections[{which}]")
        size = _as_int(section, "nbytes", where=f"sections[{which}]")
        _validate_range(off, size, pool_bytes, where=f"sections[{which}]")
        ranges[which] = (off, size)
    if ranges["K"][0] != 0:
        raise MtpKvIpcMapError("K section must start at offset zero")
    if ranges["K"][0] + ranges["K"][1] > ranges["V"][0]:
        raise MtpKvIpcMapError("K/V sections overlap")
    if ranges["V"][0] + ranges["V"][1] > pool_bytes:
        raise MtpKvIpcMapError("V section exceeds pool")

    raw_entries = pool_map.get("map")
    if not isinstance(raw_entries, Mapping):
        raise MtpKvIpcMapError("map.map must be an object")
    expected_keys = {_key(layer, which) for layer in range(_NUM_LAYERS) for which in _WHICH}
    if set(raw_entries) != expected_keys:
        raise MtpKvIpcMapError("MTP KV map entries do not cover exactly 3 K/V layers")

    entries: list[MtpKvEntry] = []
    spans: list[tuple[int, int, str]] = []
    num_blocks: int | None = None
    for which in _WHICH:
        section_off, section_bytes = ranges[which]
        expected_offset = section_off
        for layer in range(_NUM_LAYERS):
            name = _key(layer, which)
            raw = raw_entries[name]
            if not isinstance(raw, Mapping):
                raise MtpKvIpcMapError(f"{name}: entry must be an object")
            where = f"map[{name}]"
            entry_layer = _as_int(raw, "layer_idx", where=where)
            if entry_layer != layer or raw.get("which") != which:
                raise MtpKvIpcMapError(f"{where}: layer/which identity mismatch")
            group_id = _as_int(raw, "group_id", where=where)
            offset = _as_int(raw, "offset", where=where)
            nbytes = _as_int(raw, "nbytes", where=where)
            blocks = _as_int(raw, "num_blocks", where=where)
            slots = _as_int(raw, "num_slots", where=where)
            if blocks <= 0 or slots != blocks * _BLOCK_SIZE:
                raise MtpKvIpcMapError(f"{where}: invalid num_blocks/num_slots")
            if blocks != reserve.physical_num_blocks:
                raise MtpKvIpcMapError(
                    f"{where}: num_blocks={blocks} != "
                    f"physical_num_blocks={reserve.physical_num_blocks}"
                )
            if num_blocks is None:
                num_blocks = blocks
            elif blocks != num_blocks:
                raise MtpKvIpcMapError("MTP layers expose different block capacities")
            expected_nbytes = slots * _NUM_KV_HEADS * _HEAD_DIM * _KV_ITEMSIZE
            if nbytes != expected_nbytes:
                raise MtpKvIpcMapError(
                    f"{where}: nbytes={nbytes} != expected {expected_nbytes}"
                )
            _shape(raw.get("shape"), (blocks, _BLOCK_SIZE, 1, _HEAD_DIM), where=where)
            _shape(raw.get("flat_shape"), (slots, _HEAD_DIM), where=where)
            _validate_range(offset, nbytes, pool_bytes, where=where)
            if offset != expected_offset:
                raise MtpKvIpcMapError(
                    f"{where}: offset {offset} breaks contiguous {which} section"
                )
            if offset + nbytes > section_off + section_bytes:
                raise MtpKvIpcMapError(f"{where}: entry exceeds {which} section")
            spans.append((offset, offset + nbytes, name))
            entries.append(
                MtpKvEntry(
                    layer_idx=layer,
                    which=which,
                    group_id=group_id,
                    offset=offset,
                    nbytes=nbytes,
                    num_blocks=blocks,
                    num_slots=slots,
                    shape=(blocks, _BLOCK_SIZE, 1, _HEAD_DIM),
                    flat_shape=(slots, _HEAD_DIM),
                )
            )
            expected_offset = _align_up(offset + nbytes)
        if expected_offset != section_off + section_bytes:
            raise MtpKvIpcMapError(f"{which} section size is not contiguous")

    for left, right in zip(sorted(spans), sorted(spans)[1:]):
        if right[0] < left[1]:
            raise MtpKvIpcMapError(f"MTP KV entries overlap: {left[2]} and {right[2]}")

    assert num_blocks is not None
    return MtpKvMapSummary(
        version=_SCHEMA_VERSION,
        layout=_LAYOUT,
        rank=rank,
        tp_world_size=tp,
        pool_bytes=pool_bytes,
        num_blocks=num_blocks,
        scheduler_num_blocks=reserve.scheduler_num_blocks,
        physical_num_blocks=reserve.physical_num_blocks,
        padding_block_ids=reserve.padding_block_ids,
        k_section=ranges["K"],
        v_section=ranges["V"],
        entries=tuple(sorted(entries, key=lambda item: (item.layer_idx, item.which))),
    )


class MtpKvIpcMap:
    """One rank's validated MTP KV pool."""

    def __init__(self, peer_base: int, pool_map: Mapping[str, Any]):
        self.peer_base = int(peer_base)
        self.pool_map = dict(pool_map)
        self.summary = validate_mtp_pool_map(pool_map)
        self._entries = {
            (entry.layer_idx, entry.which): entry for entry in self.summary.entries
        }

    def _entry(self, layer_idx: int, which: str) -> MtpKvEntry:
        return self._entries[(int(layer_idx), which)]

    def num_slots(self, layer_idx: int = 0) -> int:
        return self._entry(layer_idx, "K").num_slots

    def section_spec(self, which: str) -> tuple[int, tuple[int, int], str]:
        if which not in _WHICH:
            raise ValueError(f"which must be K or V, got {which!r}")
        off, nbytes = self.summary.k_section if which == "K" else self.summary.v_section
        rows = nbytes // (_HEAD_DIM * _KV_ITEMSIZE)
        expected = _NUM_LAYERS * self.num_slots()
        if rows != expected:
            raise MtpKvIpcMapError(f"{which} rows={rows} != expected {expected}")
        return off, (rows, _HEAD_DIM), _KV_DTYPE

    def section_device_tensor(self, which: str):
        import torch  # noqa: PLC0415
        from pypto.runtime.device_tensor import DeviceTensor  # noqa: PLC0415

        off, shape, _ = self.section_spec(which)
        return DeviceTensor(self.peer_base + off, shape, torch.bfloat16)


def import_mtp_kv_all(rt, out_dir: str, *, tp: int, dev_offset: int = 0) -> List[MtpKvIpcMap]:
    """Validate and import all MTP KV pools before calling ``import_ipc_all``."""
    from tools.step3p5.ipc_session import (  # noqa: PLC0415
        validate_key_file,
        validate_live_session,
    )

    keys: Dict[int, bytes] = {}
    maps: list[Mapping[str, Any]] = []
    for rank in range(tp):
        key_path = os.path.join(out_dir, f"pypto_mtp_kvpool.key.rank{rank}")
        map_path = os.path.join(out_dir, f"pypto_mtp_kvpool_map.json.rank{rank}")
        ready_path = map_path + ".done"
        key = validate_key_file(key_path)
        with open(map_path, encoding="utf-8") as file:
            pool_map = json.load(file)
        validate_mtp_pool_map(pool_map)
        validate_live_session(
            pool_map,
            expected_rank=rank,
            expected_tp=tp,
            expected_device_id=dev_offset + rank,
            expected_role="mtp_kv",
            ready_path=ready_path,
            map_path=map_path,
            key_path=key_path,
        )
        keys[dev_offset + rank] = key
        maps.append(pool_map)
    first = validate_mtp_pool_map(maps[0])
    for obj in maps[1:]:
        summary = validate_mtp_pool_map(obj)
        if (
            summary.scheduler_num_blocks != first.scheduler_num_blocks
            or summary.physical_num_blocks != first.physical_num_blocks
            or summary.padding_block_ids != first.padding_block_ids
        ):
            raise MtpKvIpcMapError(
                "TP ranks expose different MTP padding reserves"
            )
    vas = rt.import_ipc_all(keys)
    return [
        MtpKvIpcMap(vas[dev_offset + rank], maps[rank])
        for rank in range(tp)
    ]


def build_stacked_mtp_kv_pool(maps: List[MtpKvIpcMap]):
    """Build the selected-program ABI ``[tp, 3*num_slots, 128]`` K/V views."""
    from pypto.runtime.device_tensor import StackedDeviceTensor  # noqa: PLC0415

    if not maps:
        raise MtpKvIpcMapError("cannot stack an empty MTP KV map list")
    tp = len(maps)
    k_shape = maps[0].section_spec("K")[1]
    v_shape = maps[0].section_spec("V")[1]
    if k_shape != v_shape:
        raise MtpKvIpcMapError("MTP K/V section shapes differ")
    for item in maps[1:]:
        if item.section_spec("K")[1] != k_shape or item.section_spec("V")[1] != v_shape:
            raise MtpKvIpcMapError("MTP KV capacities differ across TP ranks")
    return (
        StackedDeviceTensor(
            [item.section_device_tensor("K") for item in maps],
            (tp, *k_shape),
            list(range(tp)),
        ),
        StackedDeviceTensor(
            [item.section_device_tensor("V") for item in maps],
            (tp, *v_shape),
            list(range(tp)),
        ),
    )


def _synthetic_map(
    *,
    num_blocks: int = 32,
    scheduler_num_blocks: int = 17,
    bad: str | None = None,
) -> dict[str, Any]:
    entry_bytes = num_blocks * _BLOCK_SIZE * _NUM_KV_HEADS * _HEAD_DIM * _KV_ITEMSIZE
    entries: dict[str, Any] = {}
    k_bytes = _NUM_LAYERS * entry_bytes
    v_off = _align_up(k_bytes)
    for which, base in (("K", 0), ("V", v_off)):
        for layer in range(_NUM_LAYERS):
            off = base + layer * entry_bytes
            entries[_key(layer, which)] = {
                "layer_idx": layer,
                "which": which,
                "group_id": 0,
                "offset": off,
                "nbytes": entry_bytes,
                "num_blocks": num_blocks,
                "num_slots": num_blocks * _BLOCK_SIZE,
                "shape": [num_blocks, _BLOCK_SIZE, 1, _HEAD_DIM],
                "flat_shape": [num_blocks * _BLOCK_SIZE, _HEAD_DIM],
            }
    obj = {
        "version": _SCHEMA_VERSION,
        "layout": _LAYOUT,
        "rank": 0,
        "tp_world_size": 8,
        "pool_bytes": v_off + k_bytes,
        "num_layers": _NUM_LAYERS,
        "head_dim": _HEAD_DIM,
        "num_kv_heads": _NUM_KV_HEADS,
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
            "K": {"offset": 0, "nbytes": k_bytes},
            "V": {"offset": v_off, "nbytes": k_bytes},
        },
        "map": entries,
    }
    if bad == "overlap":
        obj["map"]["MTP1.K"]["offset"] = obj["map"]["MTP0.K"]["offset"]
    elif bad == "layout":
        obj["layout"] = "main_k_major_v_major_v1"
    elif bad == "reserve_missing":
        del obj["padding_block_ids"]
    elif bad == "reserve_overlap":
        obj["padding_block_ids"][0] = scheduler_num_blocks - 1
    elif bad == "reserve_out_of_range":
        obj["physical_num_blocks"] = scheduler_num_blocks + 14
    return obj


def _selftest() -> int:
    ok = True
    try:
        validate_mtp_pool_map(_synthetic_map())
        print("[selftest] valid three-layer MTP KV map -> PASS", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[selftest] valid map -> FAIL: {exc}", flush=True)
        ok = False
    for bad in (
        "overlap",
        "layout",
        "reserve_missing",
        "reserve_overlap",
        "reserve_out_of_range",
    ):
        try:
            validate_mtp_pool_map(_synthetic_map(bad=bad))
        except MtpKvIpcMapError:
            print(f"[selftest] reject {bad} -> PASS", flush=True)
        else:
            print(f"[selftest] reject {bad} -> FAIL", flush=True)
            ok = False
    print(
        f"[selftest] RESULT={'MTP_KV_IPC_MAP_VALIDATOR_OK' if ok else 'FAIL'}",
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
