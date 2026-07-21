from __future__ import annotations

import pytest

from tools.step3p5.main_kv_exporter import (
    main_kv_layout,
    main_kv_row_byte_offset,
)
from tools.step3p5.pypto_kv_ipc import validate_pool_map


def test_standalone_main_kv_layout_matches_schema_v3() -> None:
    pool_map = main_kv_layout(num_blocks=32, rank=3, tp_world_size=8)
    summary = validate_pool_map(pool_map)

    assert summary.rank == 3
    assert summary.tp_world_size == 8
    assert summary.scheduler_num_blocks == 32
    assert summary.physical_num_blocks == 47
    assert summary.padding_block_ids == tuple(range(32, 47))
    assert len(summary.entries) == 90
    assert summary.k_section[0] == 0
    assert summary.k_section[0] + summary.k_section[1] <= summary.v_section[0]
    assert all(entry.offset % 512 == 0 for entry in summary.entries)


def test_main_kv_probe_row_offset_keeps_layer_base_out_of_slot_mapping() -> None:
    pool_map = main_kv_layout(num_blocks=32, rank=0, tp_world_size=8)
    row_bytes = 128 * 2
    entry_bytes = (32 + 15) * 128 * row_bytes

    assert main_kv_row_byte_offset(
        pool_map,
        layer_idx=0,
        which="K",
        slot=0,
    ) == 0
    assert main_kv_row_byte_offset(
        pool_map,
        layer_idx=0,
        which="K",
        slot=1,
    ) == row_bytes
    assert main_kv_row_byte_offset(
        pool_map,
        layer_idx=44,
        which="K",
        slot=1,
    ) == 44 * entry_bytes + row_bytes
    assert main_kv_row_byte_offset(
        pool_map,
        layer_idx=0,
        which="V",
        slot=0,
    ) == 45 * entry_bytes


@pytest.mark.parametrize("num_blocks", [0, -1])
def test_standalone_main_kv_layout_rejects_invalid_capacity(num_blocks: int) -> None:
    with pytest.raises(ValueError, match="scheduler_num_blocks must be positive"):
        main_kv_layout(num_blocks=num_blocks)
