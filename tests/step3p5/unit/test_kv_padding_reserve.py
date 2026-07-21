from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tools.step3p5.kv_padding import (
    PaddingReserveError,
    configure_standalone_main_storage_env,
    make_padding_reserve,
    pad_fixed_batch_metadata,
    parse_padding_reserve,
    validate_fixed_batch_metadata,
)
from tools.step3p5.mtp_kv_exporter import mtp_kv_layout
from tools.step3p5.pypto_kv_ipc import (
    KvIpcMapError,
    _synthetic_map as synthetic_main_map,
    validate_pool_map,
)
from tools.step3p5.pypto_mtp_kv_ipc import (
    MtpKvIpcMapError,
    validate_mtp_pool_map,
)
from tools.step3p5.vllm_mtp_metadata import (
    extract_pypto_mtp_layer_meta,
    mtp_attention_key,
)


SCHEDULER_BLOCKS = 64
PHYSICAL_BLOCKS = SCHEDULER_BLOCKS + 15


def _active_metadata(valid: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_lens = torch.arange(1, valid + 1, dtype=torch.int32)
    block_table = torch.zeros((valid, 4), dtype=torch.int32)
    block_table[:, 0] = torch.arange(valid, dtype=torch.int32)
    slot_mapping = (
        torch.arange(valid, dtype=torch.int32) * 128
        + seq_lens
        - 1
    )
    return seq_lens, block_table, slot_mapping


@pytest.mark.parametrize("valid", [1, 2, 8, 16])
def test_main_and_mtp_share_fixed_batch_reserve_semantics(valid: int) -> None:
    reserve = make_padding_reserve(SCHEDULER_BLOCKS, PHYSICAL_BLOCKS)
    seq_lens, active_table, active_slots = _active_metadata(valid)
    block_table, slot_mapping = pad_fixed_batch_metadata(
        seq_lens=seq_lens,
        block_table=active_table,
        slot_mapping=active_slots,
        valid_rows=valid,
        reserve=reserve,
        where="unit",
    )
    padded_seq = torch.ones(16, dtype=torch.int32)
    padded_seq[:valid] = seq_lens
    positions = torch.zeros(16, dtype=torch.int32)
    positions[:valid] = seq_lens - 1

    assert torch.equal(block_table[:valid], active_table)
    assert torch.equal(slot_mapping[:valid], active_slots)
    expected_blocks = torch.arange(
        SCHEDULER_BLOCKS,
        SCHEDULER_BLOCKS + 16 - valid,
        dtype=torch.int32,
    )
    assert torch.equal(block_table[valid:, 0], expected_blocks)
    assert torch.count_nonzero(block_table[valid:, 1:]) == 0
    assert torch.equal(slot_mapping[valid:], expected_blocks * 128)
    assert len(torch.unique(block_table[valid:, 0])) == 16 - valid
    assert len(torch.unique(slot_mapping[valid:])) == 16 - valid

    validate_fixed_batch_metadata(
        seq_lens=padded_seq,
        positions=positions,
        block_table=block_table,
        slot_mapping=slot_mapping,
        valid_rows=valid,
        reserve=reserve,
        where="unit",
    )


@pytest.mark.parametrize(
    "mutation",
    ["missing", "overlap", "non_contiguous", "out_of_range"],
)
def test_reserve_parser_rejects_missing_overlap_or_out_of_range(
    mutation: str,
) -> None:
    obj = make_padding_reserve(
        SCHEDULER_BLOCKS,
        PHYSICAL_BLOCKS,
    ).as_dict()
    if mutation == "missing":
        del obj["padding_block_ids"]
    elif mutation == "overlap":
        obj["padding_block_ids"][0] = SCHEDULER_BLOCKS - 1
    elif mutation == "non_contiguous":
        obj["padding_block_ids"][1] += 1
    else:
        obj["physical_num_blocks"] = PHYSICAL_BLOCKS - 1
    with pytest.raises(PaddingReserveError):
        parse_padding_reserve(obj)


def test_padding_metadata_rejects_slot_zero_and_row_alias() -> None:
    reserve = make_padding_reserve(SCHEDULER_BLOCKS, PHYSICAL_BLOCKS)
    seq_lens, active_table, active_slots = _active_metadata(1)
    block_table, slot_mapping = pad_fixed_batch_metadata(
        seq_lens=seq_lens,
        block_table=active_table,
        slot_mapping=active_slots,
        valid_rows=1,
        reserve=reserve,
    )
    padded_seq = torch.ones(16, dtype=torch.int32)
    positions = torch.zeros(16, dtype=torch.int32)

    bad_slot = slot_mapping.clone()
    bad_slot[1] = 0
    with pytest.raises(PaddingReserveError):
        validate_fixed_batch_metadata(
            seq_lens=padded_seq,
            positions=positions,
            block_table=block_table,
            slot_mapping=bad_slot,
            valid_rows=1,
            reserve=reserve,
        )

    bad_alias = block_table.clone()
    bad_alias[2, 0] = bad_alias[1, 0]
    with pytest.raises(PaddingReserveError):
        validate_fixed_batch_metadata(
            seq_lens=padded_seq,
            positions=positions,
            block_table=bad_alias,
            slot_mapping=slot_mapping,
            valid_rows=1,
            reserve=reserve,
        )


def test_active_slot_must_match_block_table_and_position() -> None:
    reserve = make_padding_reserve(SCHEDULER_BLOCKS, PHYSICAL_BLOCKS)
    seq_lens = torch.tensor([129], dtype=torch.int32)
    active_table = torch.zeros((1, 4), dtype=torch.int32)
    active_table[0, :2] = torch.tensor([3, 9], dtype=torch.int32)
    expected_slot = 9 * 128
    block_table, slot_mapping = pad_fixed_batch_metadata(
        seq_lens=seq_lens,
        block_table=active_table,
        slot_mapping=torch.tensor([expected_slot], dtype=torch.int32),
        valid_rows=1,
        reserve=reserve,
    )
    padded_seq = torch.ones(16, dtype=torch.int32)
    padded_seq[0] = 129
    positions = torch.zeros(16, dtype=torch.int32)
    positions[0] = 128
    validate_fixed_batch_metadata(
        seq_lens=padded_seq,
        positions=positions,
        block_table=block_table,
        slot_mapping=slot_mapping,
        valid_rows=1,
        reserve=reserve,
    )

    bad_slot = slot_mapping.clone()
    bad_slot[0] = 3 * 128
    with pytest.raises(PaddingReserveError, match="does not match block_table"):
        validate_fixed_batch_metadata(
            seq_lens=padded_seq,
            positions=positions,
            block_table=block_table,
            slot_mapping=bad_slot,
            valid_rows=1,
            reserve=reserve,
        )


def test_standalone_main_storage_reserves_physical_kv_rows(monkeypatch) -> None:
    monkeypatch.delenv("PYPTO_STEP3P5_MAX_SEQ", raising=False)
    monkeypatch.delenv("PYPTO_STEP3P5_BLOCK_TABLE_FLAT", raising=False)
    monkeypatch.delenv("PYPTO_STEP3P5_KV_CACHE_ROWS", raising=False)
    monkeypatch.delenv("PYPTO_STEP3P5_ROPE_SEQ", raising=False)

    result = configure_standalone_main_storage_env()

    assert result == {
        "scheduler_num_blocks": 32,
        "physical_num_blocks": 47,
        "kv_cache_rows": 45 * 47 * 128,
        "block_table_flat": 32 * 16,
        "rope_seq": 4096,
    }
    assert int(__import__("os").environ["PYPTO_STEP3P5_KV_CACHE_ROWS"]) == (
        45 * 47 * 128
    )
    assert int(__import__("os").environ["PYPTO_STEP3P5_BLOCK_TABLE_FLAT"]) == (
        32 * 16
    )


def test_standalone_main_storage_rejects_stale_kv_capacity(monkeypatch) -> None:
    monkeypatch.setenv("PYPTO_STEP3P5_KV_CACHE_ROWS", "4096")
    monkeypatch.delenv("PYPTO_STEP3P5_MAX_SEQ", raising=False)
    monkeypatch.delenv("PYPTO_STEP3P5_BLOCK_TABLE_FLAT", raising=False)

    with pytest.raises(PaddingReserveError, match="KV rows disagree"):
        configure_standalone_main_storage_env()


@pytest.mark.parametrize(
    "bad",
    ["reserve_missing", "reserve_overlap", "reserve_out_of_range"],
)
def test_main_map_validator_rejects_invalid_reserve(bad: str) -> None:
    with pytest.raises(KvIpcMapError):
        validate_pool_map(synthetic_main_map(bad=bad))


@pytest.mark.parametrize("mutation", ["missing", "overlap", "out_of_range"])
def test_mtp_map_validator_rejects_invalid_reserve(mutation: str) -> None:
    obj = mtp_kv_layout(num_blocks=32)
    if mutation == "missing":
        del obj["padding_block_ids"]
    elif mutation == "overlap":
        obj["padding_block_ids"][0] = 31
    else:
        obj["physical_num_blocks"] = 46
    with pytest.raises(MtpKvIpcMapError):
        validate_mtp_pool_map(obj)


@pytest.mark.parametrize("valid", [1, 2, 8, 16])
def test_mtp_metadata_bridge_uses_the_same_reserve_semantics(valid: int) -> None:
    reserve = make_padding_reserve(SCHEDULER_BLOCKS, PHYSICAL_BLOCKS)
    seq_lens, block_table, slot_mapping = _active_metadata(valid)
    metadata = SimpleNamespace(
        num_prefills=0,
        num_actual_tokens=valid,
        num_decode_tokens=valid,
        decode_token_per_req=1,
        seq_lens=seq_lens,
        block_tables=block_table,
        slot_mapping=slot_mapping,
    )
    context = SimpleNamespace(
        attn_metadata={mtp_attention_key(1): metadata},
    )
    result = extract_pypto_mtp_layer_meta(
        context,
        layer_idx=1,
        positions=seq_lens - 1,
        valid_tokens=valid,
        padding_reserve=reserve,
    )

    assert torch.equal(result.block_table[:valid], block_table)
    assert torch.equal(result.slot_mapping[:valid], slot_mapping)
    expected_blocks = torch.arange(
        SCHEDULER_BLOCKS,
        SCHEDULER_BLOCKS + 16 - valid,
        dtype=torch.int32,
    )
    assert torch.equal(result.block_table[valid:, 0], expected_blocks)
    assert torch.equal(result.slot_mapping[valid:], expected_blocks * 128)
    assert result.protocol_meta()["padding_reserve"] == reserve.as_dict()
