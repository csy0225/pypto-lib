from __future__ import annotations

import pytest

from tools.step3p5.mtp_kv_exporter import mtp_kv_layout
from tools.step3p5.pypto_mtp_kv_ipc import validate_mtp_pool_map


def test_standalone_mtp_kv_layout_matches_importer_contract() -> None:
    pool_map = mtp_kv_layout(
        num_blocks=32,
        rank=3,
        tp_world_size=8,
    )
    summary = validate_mtp_pool_map(pool_map)

    assert summary.rank == 3
    assert summary.tp_world_size == 8
    assert summary.scheduler_num_blocks == 32
    assert summary.physical_num_blocks == 47
    assert summary.num_blocks == 47
    assert summary.padding_block_ids == tuple(range(32, 47))
    assert len(summary.entries) == 6
    assert summary.k_section[0] == 0
    assert summary.k_section[0] + summary.k_section[1] <= summary.v_section[0]
    assert all(entry.offset % 512 == 0 for entry in summary.entries)
    assert {
        (entry.layer_idx, entry.which) for entry in summary.entries
    } == {
        (layer_idx, which)
        for layer_idx in range(3)
        for which in ("K", "V")
    }


@pytest.mark.parametrize("num_blocks", [0, -1])
def test_standalone_mtp_kv_layout_rejects_invalid_capacity(
    num_blocks: int,
) -> None:
    with pytest.raises(ValueError, match="num_blocks must be positive"):
        mtp_kv_layout(num_blocks=num_blocks)
