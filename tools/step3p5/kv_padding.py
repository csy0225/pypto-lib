# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Main/MTP 共用的 fixed-batch KV padding reserve 契约。

PyPTO 的 Main 和 selected-MTP program 都固定执行 16 行。inactive row 仍会
执行 attention KV write，因此不能把 padding row 指向 scheduler block 0。
本模块只描述和校验 allocator-owned reserve：

```
scheduler domain: [0, scheduler_num_blocks)
padding reserve:  [scheduler_num_blocks, scheduler_num_blocks + 15)
```

Main/MTP 使用同一种语义，但各自由独立 KV allocation 拥有自己的 reserve。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

import torch

STORAGE_BATCH = 16
PADDING_BLOCK_COUNT = STORAGE_BATCH - 1
BLOCK_SIZE = 128


class PaddingReserveError(ValueError):
    """KV padding reserve 或 fixed-batch metadata 不满足生产 ABI。"""


def _integer(obj: Mapping[str, Any], name: str, *, where: str) -> int:
    value = obj.get(name)
    if isinstance(value, bool):
        raise PaddingReserveError(f"{where}: {name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise PaddingReserveError(
            f"{where}: {name} must be an integer"
        ) from exc


@dataclass(frozen=True)
class PaddingReserve:
    scheduler_num_blocks: int
    physical_num_blocks: int
    padding_block_ids: tuple[int, ...]
    block_size: int = BLOCK_SIZE

    @property
    def reserve_start(self) -> int:
        return self.scheduler_num_blocks

    def as_dict(self) -> dict[str, Any]:
        return {
            "scheduler_num_blocks": self.scheduler_num_blocks,
            "physical_num_blocks": self.physical_num_blocks,
            "reserve_start": self.reserve_start,
            "padding_block_ids": list(self.padding_block_ids),
            "padding_block_count": len(self.padding_block_ids),
            "block_size": self.block_size,
        }


def make_padding_reserve(
    scheduler_num_blocks: int,
    physical_num_blocks: int,
    *,
    block_size: int = BLOCK_SIZE,
) -> PaddingReserve:
    scheduler_num_blocks = int(scheduler_num_blocks)
    physical_num_blocks = int(physical_num_blocks)
    block_size = int(block_size)
    if scheduler_num_blocks <= 0:
        raise PaddingReserveError("scheduler_num_blocks must be positive")
    if block_size != BLOCK_SIZE:
        raise PaddingReserveError(
            f"block_size={block_size} != required {BLOCK_SIZE}"
        )
    minimum_physical = scheduler_num_blocks + PADDING_BLOCK_COUNT
    if physical_num_blocks < minimum_physical:
        raise PaddingReserveError(
            "physical KV capacity does not contain the 15-block padding "
            f"reserve: physical={physical_num_blocks}, "
            f"required>={minimum_physical}"
        )
    padding_block_ids = tuple(
        range(scheduler_num_blocks, minimum_physical)
    )
    return PaddingReserve(
        scheduler_num_blocks=scheduler_num_blocks,
        physical_num_blocks=physical_num_blocks,
        padding_block_ids=padding_block_ids,
        block_size=block_size,
    )


def parse_padding_reserve(
    obj: Mapping[str, Any],
    *,
    where: str = "padding_reserve",
) -> PaddingReserve:
    """从 IPC map 或 protocol meta 解析并严格校验 reserve。"""
    if not isinstance(obj, Mapping):
        raise PaddingReserveError(f"{where}: expected an object")
    scheduler = _integer(obj, "scheduler_num_blocks", where=where)
    physical = _integer(obj, "physical_num_blocks", where=where)
    reserve_start = _integer(obj, "reserve_start", where=where)
    block_size = _integer(obj, "block_size", where=where)
    count = _integer(obj, "padding_block_count", where=where)
    raw_ids = obj.get("padding_block_ids")
    if not isinstance(raw_ids, (list, tuple)):
        raise PaddingReserveError(
            f"{where}: padding_block_ids must be an ordered list"
        )
    if any(isinstance(value, bool) for value in raw_ids):
        raise PaddingReserveError(
            f"{where}: padding_block_ids contains a boolean"
        )
    try:
        ids = tuple(int(value) for value in raw_ids)
    except (TypeError, ValueError) as exc:
        raise PaddingReserveError(
            f"{where}: padding_block_ids contains a non-integer"
        ) from exc

    expected = make_padding_reserve(
        scheduler,
        physical,
        block_size=block_size,
    )
    if reserve_start != expected.reserve_start:
        raise PaddingReserveError(
            f"{where}: reserve_start={reserve_start} != "
            f"scheduler_num_blocks={expected.reserve_start}"
        )
    if count != PADDING_BLOCK_COUNT:
        raise PaddingReserveError(
            f"{where}: padding_block_count={count} != "
            f"{PADDING_BLOCK_COUNT}"
        )
    if ids != expected.padding_block_ids:
        raise PaddingReserveError(
            f"{where}: padding_block_ids={ids} != "
            f"allocator-owned contiguous reserve "
            f"{expected.padding_block_ids}"
        )
    return expected


def _validate_active_rows(
    *,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    valid_rows: int,
    reserve: PaddingReserve,
    where: str,
) -> None:
    """验证 active rows 完全位于 scheduler domain 且 slot/block 一致。"""
    for row in range(valid_rows):
        seq_len = int(seq_lens[row])
        if seq_len <= 0:
            raise PaddingReserveError(
                f"{where}: active row {row} has seq_len={seq_len}"
            )
        position = seq_len - 1
        table_index = position // reserve.block_size
        if table_index >= block_table.shape[1]:
            raise PaddingReserveError(
                f"{where}: active row {row} position={position} needs "
                f"block_table column {table_index}, width={block_table.shape[1]}"
            )
        used_blocks = table_index + 1
        active_ids = block_table[row, :used_blocks]
        if torch.any(active_ids < 0) or torch.any(
            active_ids >= reserve.scheduler_num_blocks
        ):
            raise PaddingReserveError(
                f"{where}: active row {row} uses a block outside scheduler "
                f"domain [0,{reserve.scheduler_num_blocks})"
            )
        expected_slot = (
            int(block_table[row, table_index]) * reserve.block_size
            + position % reserve.block_size
        )
        actual_slot = int(slot_mapping[row])
        if actual_slot != expected_slot:
            raise PaddingReserveError(
                f"{where}: active row {row} slot={actual_slot} does not "
                f"match block_table[{row},{table_index}]="
                f"{int(block_table[row, table_index])} at position={position}; "
                f"expected slot {expected_slot}"
            )


def pad_fixed_batch_metadata(
    *,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    valid_rows: int,
    reserve: PaddingReserve,
    storage_batch: int = STORAGE_BATCH,
    where: str = "metadata",
) -> tuple[torch.Tensor, torch.Tensor]:
    """保留 active rows，并用 reserve 初始化 fixed-batch padding rows。"""
    valid_rows = int(valid_rows)
    if storage_batch != STORAGE_BATCH:
        raise PaddingReserveError(
            f"{where}: storage_batch={storage_batch} != {STORAGE_BATCH}"
        )
    if not 1 <= valid_rows <= storage_batch:
        raise PaddingReserveError(
            f"{where}: valid_rows must be 1..{storage_batch}"
        )
    if seq_lens.dtype != torch.int32 or seq_lens.ndim != 1:
        raise PaddingReserveError(f"{where}: seq_lens must be INT32 [T]")
    if block_table.dtype != torch.int32 or block_table.ndim != 2:
        raise PaddingReserveError(
            f"{where}: block_table must be INT32 [T,max_blocks]"
        )
    if block_table.shape[1] <= 0 or block_table.shape[0] < valid_rows:
        raise PaddingReserveError(
            f"{where}: block_table does not cover {valid_rows} active rows"
        )
    if (
        slot_mapping.dtype != torch.int32
        or slot_mapping.ndim != 1
        or slot_mapping.numel() < valid_rows
    ):
        raise PaddingReserveError(
            f"{where}: slot_mapping must be INT32 and cover active rows"
        )
    if seq_lens.numel() < valid_rows:
        raise PaddingReserveError(
            f"{where}: seq_lens does not cover active rows"
        )

    active_seq = seq_lens[:valid_rows]
    active_table = block_table[:valid_rows]
    active_slots = slot_mapping[:valid_rows]
    _validate_active_rows(
        seq_lens=active_seq,
        block_table=active_table,
        slot_mapping=active_slots,
        valid_rows=valid_rows,
        reserve=reserve,
        where=where,
    )

    block_out = torch.zeros(
        (storage_batch, int(block_table.shape[1])),
        dtype=torch.int32,
    )
    slot_out = torch.zeros(storage_batch, dtype=torch.int32)
    block_out[:valid_rows] = active_table
    slot_out[:valid_rows] = active_slots
    padding_rows = storage_batch - valid_rows
    for offset in range(padding_rows):
        block_id = reserve.padding_block_ids[offset]
        block_out[valid_rows + offset, 0] = block_id
        slot_out[valid_rows + offset] = block_id * reserve.block_size
    return block_out, slot_out


def make_diagnostic_fixed_batch_metadata(
    *,
    valid_rows: int,
    reserve: PaddingReserve,
    max_blocks_per_row: int | None = None,
    storage_batch: int = STORAGE_BATCH,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造 standalone harness 使用的最小 ctx=1 fixed-batch metadata。

    该 helper 只用于没有 vLLM scheduler 的离线/真机诊断入口。active row ``i``
    使用 scheduler-owned block ``i``，padding row 使用 allocator reserve。
    生产 live 路径必须继续保留 vLLM 提供的 active block table/slot mapping，
    不得调用本 helper 重建 active metadata。
    """
    valid_rows = int(valid_rows)
    if not 1 <= valid_rows <= storage_batch:
        raise PaddingReserveError(
            f"diagnostic valid_rows must be 1..{storage_batch}"
        )
    if valid_rows > reserve.scheduler_num_blocks:
        raise PaddingReserveError(
            "diagnostic active rows need one scheduler block each: "
            f"valid_rows={valid_rows}, "
            f"scheduler_num_blocks={reserve.scheduler_num_blocks}"
        )
    if max_blocks_per_row is None:
        max_blocks_per_row = reserve.scheduler_num_blocks
    max_blocks_per_row = int(max_blocks_per_row)
    if max_blocks_per_row <= 0:
        raise PaddingReserveError(
            "diagnostic max_blocks_per_row must be positive"
        )

    active_seq = torch.ones(valid_rows, dtype=torch.int32)
    active_table = torch.zeros(
        (valid_rows, max_blocks_per_row),
        dtype=torch.int32,
    )
    active_ids = torch.arange(valid_rows, dtype=torch.int32)
    active_table[:, 0] = active_ids
    active_slots = active_ids * reserve.block_size
    block_table, slot_mapping = pad_fixed_batch_metadata(
        seq_lens=active_seq,
        block_table=active_table,
        slot_mapping=active_slots,
        valid_rows=valid_rows,
        reserve=reserve,
        storage_batch=storage_batch,
        where="diagnostic metadata",
    )
    seq_lens = torch.ones(storage_batch, dtype=torch.int32)
    positions = torch.zeros(storage_batch, dtype=torch.int32)
    validate_fixed_batch_metadata(
        seq_lens=seq_lens,
        positions=positions,
        block_table=block_table,
        slot_mapping=slot_mapping,
        valid_rows=valid_rows,
        reserve=reserve,
        storage_batch=storage_batch,
        where="diagnostic metadata",
    )
    return seq_lens, positions, block_table, slot_mapping


def diagnostic_reserve_from_compiled_shape(
    *,
    block_table_flat: int,
    storage_batch: int = STORAGE_BATCH,
    block_size: int = BLOCK_SIZE,
) -> PaddingReserve:
    """从 standalone program 的固定 block-table shape 推导诊断 reserve。"""
    block_table_flat = int(block_table_flat)
    storage_batch = int(storage_batch)
    if storage_batch != STORAGE_BATCH:
        raise PaddingReserveError(
            f"diagnostic storage_batch={storage_batch} != {STORAGE_BATCH}"
        )
    if block_table_flat <= 0 or block_table_flat % storage_batch:
        raise PaddingReserveError(
            "diagnostic block_table_flat must be positive and divisible by "
            f"storage batch {storage_batch}"
        )
    scheduler_num_blocks = block_table_flat // storage_batch
    return make_padding_reserve(
        scheduler_num_blocks,
        scheduler_num_blocks + PADDING_BLOCK_COUNT,
        block_size=block_size,
    )


def configure_standalone_main_storage_env(
    *,
    storage_batch: int = STORAGE_BATCH,
    num_layers: int = 45,
    block_size: int = BLOCK_SIZE,
) -> dict[str, int]:
    """Configure and validate Main standalone diagnostic storage shapes.

    This helper is intentionally *not* used by the live vLLM path.  Standalone
    harnesses do not have a scheduler-owned allocator, so they must reserve
    fifteen physical padding blocks themselves before importing ``config``.
    The compiled Main program stores one K/V section per decoder layer; its
    physical row capacity is therefore:

    ``num_layers * (scheduler_blocks + 15) * block_size``.

    Existing explicit environment values are checked rather than overwritten.
    A stale ``PYPTO_STEP3P5_KV_CACHE_ROWS`` must fail closed instead of causing
    a metadata reserve to point outside the allocated K/V tensor.
    """
    storage_batch = int(storage_batch)
    num_layers = int(num_layers)
    block_size = int(block_size)
    if storage_batch != STORAGE_BATCH:
        raise PaddingReserveError(
            f"standalone storage batch must be {STORAGE_BATCH}, got {storage_batch}"
        )
    if num_layers <= 0 or block_size <= 0:
        raise PaddingReserveError("standalone layer/block dimensions must be positive")

    max_seq = int(os.environ.get("PYPTO_STEP3P5_MAX_SEQ", "4096"))
    if max_seq <= 0 or max_seq % block_size:
        raise PaddingReserveError(
            "PYPTO_STEP3P5_MAX_SEQ must be a positive multiple of block size "
            f"{block_size}, got {max_seq}"
        )
    default_scheduler_blocks = max_seq // block_size

    raw_btf = os.environ.get("PYPTO_STEP3P5_BLOCK_TABLE_FLAT")
    if raw_btf is None:
        scheduler_blocks = default_scheduler_blocks
        expected_btf = scheduler_blocks * storage_batch
        os.environ["PYPTO_STEP3P5_BLOCK_TABLE_FLAT"] = str(expected_btf)
    else:
        try:
            block_table_flat = int(raw_btf)
        except (TypeError, ValueError) as exc:
            raise PaddingReserveError(
                f"PYPTO_STEP3P5_BLOCK_TABLE_FLAT is not an integer: {raw_btf!r}"
            ) from exc
        if block_table_flat <= 0 or block_table_flat % storage_batch:
            raise PaddingReserveError(
                "PYPTO_STEP3P5_BLOCK_TABLE_FLAT must be positive and divisible "
                f"by {storage_batch}, got {block_table_flat}"
            )
        scheduler_blocks = block_table_flat // storage_batch

    physical_blocks = scheduler_blocks + PADDING_BLOCK_COUNT
    expected_kv_rows = num_layers * physical_blocks * block_size
    raw_kv = os.environ.get("PYPTO_STEP3P5_KV_CACHE_ROWS")
    if raw_kv is not None:
        try:
            configured_kv_rows = int(raw_kv)
        except (TypeError, ValueError) as exc:
            raise PaddingReserveError(
                f"PYPTO_STEP3P5_KV_CACHE_ROWS is not an integer: {raw_kv!r}"
            ) from exc
        if configured_kv_rows != expected_kv_rows:
            raise PaddingReserveError(
                "standalone Main KV rows disagree with allocator-owned reserve: "
                f"configured={raw_kv}, expected={expected_kv_rows} "
                f"(45*({scheduler_blocks}+{PADDING_BLOCK_COUNT})*{block_size})"
            )
    os.environ["PYPTO_STEP3P5_KV_CACHE_ROWS"] = str(expected_kv_rows)

    rope_seq = max(max_seq, scheduler_blocks * block_size)
    raw_rope = os.environ.get("PYPTO_STEP3P5_ROPE_SEQ")
    if raw_rope is not None:
        try:
            configured_rope_seq = int(raw_rope)
        except (TypeError, ValueError) as exc:
            raise PaddingReserveError(
                f"PYPTO_STEP3P5_ROPE_SEQ is not an integer: {raw_rope!r}"
            ) from exc
        if configured_rope_seq < rope_seq:
            raise PaddingReserveError(
                "PYPTO_STEP3P5_ROPE_SEQ is shorter than standalone block-table "
                f"capacity: configured={raw_rope}, required>={rope_seq}"
            )
    os.environ.setdefault("PYPTO_STEP3P5_ROPE_SEQ", str(rope_seq))
    return {
        "scheduler_num_blocks": scheduler_blocks,
        "physical_num_blocks": physical_blocks,
        "kv_cache_rows": expected_kv_rows,
        "block_table_flat": scheduler_blocks * storage_batch,
        "rope_seq": int(os.environ["PYPTO_STEP3P5_ROPE_SEQ"]),
    }


def validate_fixed_batch_metadata(
    *,
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    valid_rows: int,
    reserve: PaddingReserve,
    storage_batch: int = STORAGE_BATCH,
    where: str = "metadata",
) -> None:
    """Fail-closed 校验 active scheduler rows 和 padding reserve rows。"""
    valid_rows = int(valid_rows)
    if storage_batch != STORAGE_BATCH:
        raise PaddingReserveError(
            f"{where}: storage_batch={storage_batch} != {STORAGE_BATCH}"
        )
    if not 1 <= valid_rows <= storage_batch:
        raise PaddingReserveError(
            f"{where}: valid_rows must be 1..{storage_batch}"
        )
    if seq_lens.dtype != torch.int32 or tuple(seq_lens.shape) != (
        storage_batch,
    ):
        raise PaddingReserveError(
            f"{where}: seq_lens must be INT32 [{storage_batch}]"
        )
    if positions.dtype != torch.int32 or tuple(positions.shape) != (
        storage_batch,
    ):
        raise PaddingReserveError(
            f"{where}: positions must be INT32 [{storage_batch}]"
        )
    if (
        block_table.dtype != torch.int32
        or block_table.ndim != 2
        or block_table.shape[0] != storage_batch
        or block_table.shape[1] <= 0
    ):
        raise PaddingReserveError(
            f"{where}: block_table must be INT32 "
            f"[{storage_batch},max_blocks]"
        )
    if slot_mapping.dtype != torch.int32 or tuple(slot_mapping.shape) != (
        storage_batch,
    ):
        raise PaddingReserveError(
            f"{where}: slot_mapping must be INT32 [{storage_batch}]"
        )
    if torch.any(seq_lens[:valid_rows] <= 0):
        raise PaddingReserveError(
            f"{where}: active seq_lens must be positive"
        )
    if not torch.equal(
        positions[:valid_rows],
        seq_lens[:valid_rows] - 1,
    ):
        raise PaddingReserveError(
            f"{where}: active positions must equal seq_lens-1"
        )
    _validate_active_rows(
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=slot_mapping,
        valid_rows=valid_rows,
        reserve=reserve,
        where=where,
    )

    padding_rows = storage_batch - valid_rows
    if not torch.equal(
        seq_lens[valid_rows:],
        torch.ones(padding_rows, dtype=torch.int32),
    ):
        raise PaddingReserveError(
            f"{where}: padding seq_lens must be one"
        )
    if torch.count_nonzero(positions[valid_rows:]).item():
        raise PaddingReserveError(
            f"{where}: padding positions must be zero"
        )
    for offset in range(padding_rows):
        row = valid_rows + offset
        block_id = reserve.padding_block_ids[offset]
        if int(block_table[row, 0]) != block_id:
            raise PaddingReserveError(
                f"{where}: padding row {row} block={int(block_table[row, 0])} "
                f"!= reserved block {block_id}"
            )
        if torch.count_nonzero(block_table[row, 1:]).item():
            raise PaddingReserveError(
                f"{where}: padding row {row} has nonzero unused block columns"
            )
        expected_slot = block_id * reserve.block_size
        if int(slot_mapping[row]) != expected_slot:
            raise PaddingReserveError(
                f"{where}: padding row {row} slot={int(slot_mapping[row])} "
                f"!= reserved slot {expected_slot}"
            )


__all__ = [
    "BLOCK_SIZE",
    "PADDING_BLOCK_COUNT",
    "STORAGE_BATCH",
    "PaddingReserve",
    "PaddingReserveError",
    "configure_standalone_main_storage_env",
    "diagnostic_reserve_from_compiled_shape",
    "make_padding_reserve",
    "make_diagnostic_fixed_batch_metadata",
    "pad_fixed_batch_metadata",
    "parse_padding_reserve",
    "validate_fixed_batch_metadata",
]
