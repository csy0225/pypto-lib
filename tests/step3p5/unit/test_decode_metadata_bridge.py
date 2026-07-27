#!/usr/bin/env python3
"""Card-free tests for the pure-decode metadata bridge (design §5).

Covers the batch/padding/eligibility contract the whole-net ABI depends on
(hard emphasis: dtype, padding, single- vs multi-batch, memory init):

  - storage_batch is a configured static capacity; active T may vary up to it;
  - seq_lens padding = 1 (so position = seq_len-1 never becomes -1);
  - positions padding = 0;
  - slot_mapping / block_table padding use allocator-owned reserve blocks;
  - speculative target verification is decomposed into ordered N=1 rounds;
  - reject prefill / PP>1 / profile / T=0 / T>storage capacity active requests /
    positions!=seq_lens-1 / missing decoder layer / non-positive seq_lens;
  - the KV group map covers all 45 decoder layers with no layer in two groups;
  - protocol tensor/meta shapes are stable.

Uses torch (available card-free); no vLLM, no device.

Run:
    python3 -m pytest tests/step3p5/unit/test_decode_metadata_bridge.py -q
"""
from __future__ import annotations

import os
import sys

import pytest

torch = pytest.importorskip("torch")

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.step3p5.vllm_decode_metadata import (  # noqa: E402
    DecodeMetadataError,
    extract_pypto_decode_meta,
    extract_pypto_decode_plan,
)
from tools.step3p5.kv_padding import (  # noqa: E402
    STORAGE_BATCH,
    make_padding_reserve,
)

_NUM_LAYERS = 45
_SCHEDULER_BLOCKS = 64
_DEFAULT_STORAGE_CAPACITY = STORAGE_BATCH
_PHYSICAL_BLOCKS = _SCHEDULER_BLOCKS + STORAGE_BATCH - 1


def _make_context(
    valid: int,
    *,
    num_groups: int = 2,
    num_reqs: int | None = None,
    seq_lens=None,
    positions=None,
    pipeline_parallel_size: int = 1,
    storage_capacity: int = _DEFAULT_STORAGE_CAPACITY,
    **meta_overrides,
):
    """Build a fake pure-decode ForwardContext for ``valid`` one-token requests.

    All 45 layers share one metadata object (mirrors the target builder that
    assigns one AscendMetadata per KV group); groups come from kv_cache_config.
    """
    n = max(valid, 1)
    if seq_lens is None:
        seq_lens = list(range(2, 2 + n))  # positive, distinct
    if positions is None:
        positions = [s - 1 for s in seq_lens]
    _num_reqs = num_reqs if num_reqs is not None else valid

    class Meta:
        num_actual_tokens = valid
        num_reqs = _num_reqs
        num_decode_tokens = valid
        num_prefills = 0
        num_decodes = valid
        decode_token_per_req = 1
        num_spec_decodes = 0
        attn_state = "DecodeOnly"

    meta = Meta()
    meta.seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
    meta.positions = torch.tensor(positions, dtype=torch.int32)
    meta.query_start_loc = torch.arange(n + 1, dtype=torch.int32)
    meta.block_tables = (
        torch.arange(n * 4, dtype=torch.int32).reshape(n, 4)
        % _SCHEDULER_BLOCKS
    )
    meta.slot_mapping = torch.tensor(
        [
            int(meta.block_tables[row, (int(seq_lens[row]) - 1) // 128])
            * 128
            + (int(seq_lens[row]) - 1) % 128
            for row in range(n)
        ],
        dtype=torch.int32,
    )
    for k, v in meta_overrides.items():
        setattr(meta, k, v)

    class Group:
        def __init__(self, layer_names):
            self.layer_names = layer_names

    groups = []
    for g in range(num_groups):
        names = [
            f"model.layers.{i}.self_attn.attn"
            for i in range(g, _NUM_LAYERS, num_groups)
        ]
        groups.append(Group(names))

    class Kvc:
        kv_cache_groups = groups

    class Parallel:
        pipeline_parallel_size = 1
        prefill_context_parallel_size = 1
        decode_context_parallel_size = 1

    Parallel.pipeline_parallel_size = pipeline_parallel_size

    class Cfg:
        kv_cache_config = Kvc()
        parallel_config = Parallel()
        speculative_config = None

    ctx = type("Ctx", (), {})()
    ctx.attn_metadata = {
        f"model.layers.{i}.self_attn.attn": meta for i in range(_NUM_LAYERS)
    }
    ctx.vllm_config = Cfg()
    ctx.pypto_padding_reserve = make_padding_reserve(
        _SCHEDULER_BLOCKS,
        _SCHEDULER_BLOCKS + storage_capacity - 1,
        storage_capacity=storage_capacity,
    )
    return ctx, Cfg()


def _make_spec_context(
    query_lengths,
    *,
    final_seq_lens,
    positions=None,
    num_groups: int = 2,
    num_speculative_tokens: int = 3,
    storage_capacity: int = _DEFAULT_STORAGE_CAPACITY,
):
    """Build request-major target-verification metadata.

    ``final_seq_lens`` mirrors vLLM's optimistic sequence lengths after all
    target + draft positions in this target forward have been scheduled.
    """
    query_lengths = [int(length) for length in query_lengths]
    final_seq_lens = [int(length) for length in final_seq_lens]
    valid_requests = len(query_lengths)
    valid_tokens = sum(query_lengths)
    if len(final_seq_lens) != valid_requests:
        raise ValueError("final_seq_lens must contain one value per request")

    starts = [0]
    derived_positions = []
    for query_len, final_seq_len in zip(query_lengths, final_seq_lens):
        starts.append(starts[-1] + query_len)
        first_seq_len = final_seq_len - query_len + 1
        derived_positions.extend(
            range(first_seq_len - 1, final_seq_len)
        )
    if positions is None:
        positions = derived_positions

    class Meta:
        num_actual_tokens = valid_tokens
        num_reqs = valid_requests
        num_decode_tokens = valid_tokens
        num_prefills = 0
        num_decodes = valid_requests
        decode_token_per_req = max(query_lengths)
        num_spec_decodes = max(query_lengths) - 1
        attn_state = "SpecDecoding"

    meta = Meta()
    meta.seq_lens = torch.tensor(final_seq_lens, dtype=torch.int32)
    meta.positions = torch.tensor(positions, dtype=torch.int32)
    meta.query_start_loc = torch.tensor(starts, dtype=torch.int32)
    meta.actual_seq_lengths_q = starts[1:]
    meta.block_tables = (
        torch.arange(valid_requests * 4, dtype=torch.int32)
        .reshape(valid_requests, 4)
    )
    slot_mapping = []
    for request_row, (query_len, final_seq_len) in enumerate(
        zip(query_lengths, final_seq_lens)
    ):
        first_seq_len = final_seq_len - query_len + 1
        for position in range(first_seq_len - 1, final_seq_len):
            slot_mapping.append(
                int(meta.block_tables[request_row, position // 128]) * 128
                + position % 128
            )
    meta.slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32)

    class Group:
        def __init__(self, layer_names):
            self.layer_names = layer_names

    groups = []
    for group_id in range(num_groups):
        groups.append(
            Group(
                [
                    f"model.layers.{index}.self_attn.attn"
                    for index in range(group_id, _NUM_LAYERS, num_groups)
                ]
            )
        )

    class Kvc:
        kv_cache_groups = groups

    class Parallel:
        pipeline_parallel_size = 1
        prefill_context_parallel_size = 1
        decode_context_parallel_size = 1

    class Speculative:
        pass

    Speculative.num_speculative_tokens = num_speculative_tokens

    class Cfg:
        kv_cache_config = Kvc()
        parallel_config = Parallel()
        speculative_config = Speculative()

    ctx = type("Ctx", (), {})()
    ctx.attn_metadata = {
        f"model.layers.{index}.self_attn.attn": meta
        for index in range(_NUM_LAYERS)
    }
    ctx.vllm_config = Cfg()
    ctx.pypto_padding_reserve = make_padding_reserve(
        _SCHEDULER_BLOCKS,
        _SCHEDULER_BLOCKS + storage_capacity - 1,
        storage_capacity=storage_capacity,
    )
    return ctx, Cfg()


class TestDecodeMetadataBatches:
    @pytest.mark.parametrize("valid", [1, 2, 8, min(16, STORAGE_BATCH)])
    def test_storage_batch_and_valid_requests(self, valid):
        ctx, cfg = _make_context(valid)
        meta = extract_pypto_decode_meta(ctx, vllm_config=cfg)
        assert meta.storage_batch == STORAGE_BATCH
        assert meta.valid_tokens == valid
        assert meta.valid_requests == valid
        assert meta.seq_lens.shape == (STORAGE_BATCH,)
        assert meta.positions.shape == (STORAGE_BATCH,)
        assert meta.seq_lens.dtype == torch.int32
        assert meta.positions.dtype == torch.int32

    @pytest.mark.parametrize("valid", [1, 2, 8, min(16, STORAGE_BATCH)])
    def test_padding_initialization(self, valid):
        ctx, cfg = _make_context(valid)
        meta = extract_pypto_decode_meta(ctx, vllm_config=cfg)
        # seq_lens padding == 1 (never 0 -> position -1); positions padding == 0.
        assert torch.all(meta.seq_lens[valid:] == 1)
        assert torch.all(meta.positions[valid:] == 0)
        for group in meta.groups:
            assert group.slot_mapping.shape == (STORAGE_BATCH,)
            expected_blocks = torch.arange(
                _SCHEDULER_BLOCKS,
                _SCHEDULER_BLOCKS + STORAGE_BATCH - valid,
                dtype=torch.int32,
            )
            assert torch.equal(
                group.block_table[valid:, 0],
                expected_blocks,
            )
            assert torch.count_nonzero(group.block_table[valid:, 1:]) == 0
            assert torch.equal(
                group.slot_mapping[valid:],
                expected_blocks * 128,
            )

    @pytest.mark.parametrize("valid", [1, 2, 8, min(16, STORAGE_BATCH)])
    def test_active_block_and_slot_rows_are_preserved(self, valid):
        ctx, cfg = _make_context(valid)
        source = next(iter(ctx.attn_metadata.values()))
        active_blocks = source.block_tables[:valid].clone()
        active_slots = source.slot_mapping[:valid].clone()
        meta = extract_pypto_decode_meta(ctx, vllm_config=cfg)
        for group in meta.groups:
            assert torch.equal(group.block_table[:valid], active_blocks)
            assert torch.equal(group.slot_mapping[:valid], active_slots)

    @pytest.mark.parametrize("valid", [1, min(16, STORAGE_BATCH)])
    def test_valid_rows_preserved(self, valid):
        seq = list(range(3, 3 + valid))
        ctx, cfg = _make_context(valid, seq_lens=seq)
        meta = extract_pypto_decode_meta(ctx, vllm_config=cfg)
        assert meta.seq_lens[:valid].tolist() == seq
        assert meta.positions[:valid].tolist() == [s - 1 for s in seq]


class TestDecodeMetadataGroups:
    @pytest.mark.parametrize("num_groups", [1, 2, 3, 5])
    def test_all_45_layers_grouped_uniquely(self, num_groups):
        ctx, cfg = _make_context(2, num_groups=num_groups)
        meta = extract_pypto_decode_meta(ctx, vllm_config=cfg)
        assert len(meta.layer_to_group) == _NUM_LAYERS
        assert all(g >= 0 for g in meta.layer_to_group)
        assert len(meta.groups) == num_groups
        covered = set()
        for group in meta.groups:
            assert not (covered & set(group.layer_indices)), "layer in two groups"
            covered |= set(group.layer_indices)
        assert covered == set(range(_NUM_LAYERS))

    def test_protocol_shapes(self):
        ctx, cfg = _make_context(4, num_groups=2)
        meta = extract_pypto_decode_meta(ctx, vllm_config=cfg)
        tensors = meta.protocol_tensors()
        assert tensors["meta_seq_lens"].shape == (STORAGE_BATCH,)
        assert tensors["meta_positions"].shape == (STORAGE_BATCH,)
        for g in range(2):
            assert tensors[f"meta_slot_mapping_g{g}"].shape == (STORAGE_BATCH,)
            assert tensors[f"meta_block_table_g{g}"].shape[0] == STORAGE_BATCH
        pmeta = meta.protocol_meta()
        assert pmeta["protocol_version"] == 2
        assert pmeta["storage_batch"] == STORAGE_BATCH
        assert pmeta["valid_tokens"] == 4
        assert pmeta["kv_group_count"] == 2
        assert len(pmeta["layer_to_group"]) == _NUM_LAYERS
        assert pmeta["padding_reserve"] == {
            "scheduler_num_blocks": _SCHEDULER_BLOCKS,
            "physical_num_blocks": _PHYSICAL_BLOCKS,
            "reserve_start": _SCHEDULER_BLOCKS,
            "padding_block_ids": list(
                range(_SCHEDULER_BLOCKS, _PHYSICAL_BLOCKS)
            ),
            "padding_block_count": STORAGE_BATCH - 1,
            "block_size": 128,
        }


class TestSpeculativeTargetVerification:
    def test_single_request_four_rounds(self):
        ctx, cfg = _make_spec_context(
            [4],
            final_seq_lens=[13],
        )
        source = next(iter(ctx.attn_metadata.values()))
        plan = extract_pypto_decode_plan(ctx, vllm_config=cfg)

        assert plan.valid_tokens == 4
        assert plan.valid_requests == 1
        assert plan.query_lengths == (4,)
        assert len(plan.steps) == 4
        for round_idx, step in enumerate(plan.steps):
            assert step.valid_tokens == 1
            assert step.valid_requests == 1
            assert step.token_indices == (round_idx,)
            assert step.query_lengths == (1,)
            assert step.seq_lens[0].item() == 10 + round_idx
            assert step.positions[0].item() == 9 + round_idx
            assert torch.all(step.seq_lens[1:] == 1)
            assert torch.count_nonzero(step.positions[1:]).item() == 0
            for group in step.groups:
                assert torch.equal(
                    group.block_table[0],
                    source.block_tables[0],
                )
                assert group.slot_mapping[0].item() == (
                    source.slot_mapping[round_idx].item()
                )

        # The compatibility helper must not silently collapse a multi-round
        # target verification into one incorrect whole-net call.
        with pytest.raises(DecodeMetadataError, match="multi-token"):
            extract_pypto_decode_meta(ctx, vllm_config=cfg)

    def test_uneven_requests_preserve_request_and_token_order(self):
        ctx, cfg = _make_spec_context(
            [4, 2, 1],
            final_seq_lens=[13, 8, 5],
        )
        source = next(iter(ctx.attn_metadata.values()))
        plan = extract_pypto_decode_plan(ctx, vllm_config=cfg)

        assert [step.token_indices for step in plan.steps] == [
            (0, 4, 6),
            (1, 5),
            (2,),
            (3,),
        ]
        assert [
            step.seq_lens[: step.valid_tokens].tolist()
            for step in plan.steps
        ] == [
            [10, 7, 5],
            [11, 8],
            [12],
            [13],
        ]
        assert [
            step.positions[: step.valid_tokens].tolist()
            for step in plan.steps
        ] == [
            [9, 6, 4],
            [10, 7],
            [11],
            [12],
        ]

        expected_requests = [
            (0, 1, 2),
            (0, 1),
            (0,),
            (0,),
        ]
        for step, request_indices in zip(plan.steps, expected_requests):
            for group in step.groups:
                expected_block = source.block_tables[
                    torch.tensor(request_indices, dtype=torch.long)
                ]
                expected_slot = source.slot_mapping[
                    torch.tensor(step.token_indices, dtype=torch.long)
                ]
                assert torch.equal(
                    group.block_table[: step.valid_tokens],
                    expected_block,
                )
                assert torch.equal(
                    group.slot_mapping[: step.valid_tokens],
                    expected_slot,
                )

                # Every inactive physical row owns a distinct reserve block
                # and slot; reserve rows are never aliased with active rows.
                padding_count = STORAGE_BATCH - step.valid_tokens
                expected_padding_blocks = torch.arange(
                    _SCHEDULER_BLOCKS,
                    _SCHEDULER_BLOCKS + padding_count,
                    dtype=torch.int32,
                )
                assert torch.equal(
                    group.block_table[step.valid_tokens:, 0],
                    expected_padding_blocks,
                )
                assert torch.equal(
                    group.slot_mapping[step.valid_tokens:],
                    expected_padding_blocks * 128,
                )

    def test_reject_query_wider_than_configured_speculation(self):
        ctx, cfg = _make_spec_context(
            [4],
            final_seq_lens=[13],
            num_speculative_tokens=2,
        )
        with pytest.raises(
            DecodeMetadataError,
            match="exceeds configured target verification width",
        ):
            extract_pypto_decode_plan(ctx, vllm_config=cfg)

    def test_reject_speculative_position_mismatch(self):
        ctx, cfg = _make_spec_context(
            [4, 2, 1],
            final_seq_lens=[13, 8, 5],
            positions=[9, 10, 999, 12, 6, 7, 4],
        )
        with pytest.raises(
            DecodeMetadataError,
            match="positions do not equal derived seq_lens-1",
        ):
            extract_pypto_decode_plan(ctx, vllm_config=cfg)


class TestDecodeMetadataCapacity:
    def test_configured_capacity_allows_runtime_active_batch_above_16(self):
        capacity = 32
        ctx, cfg = _make_context(17, storage_capacity=capacity)
        meta = extract_pypto_decode_meta(
            ctx, vllm_config=cfg, max_batch=capacity
        )
        assert meta.storage_batch == capacity
        assert meta.valid_tokens == 17
        assert meta.seq_lens.shape == (capacity,)
        assert meta.groups[0].block_table.shape[0] == capacity
        assert meta.protocol_meta()["storage_capacity"] == capacity

    def test_reserve_capacity_must_match_compiled_capacity(self):
        ctx, cfg = _make_context(2, storage_capacity=16)
        with pytest.raises(DecodeMetadataError, match="reserve capacity"):
            extract_pypto_decode_meta(ctx, vllm_config=cfg, max_batch=32)


class TestDecodeMetadataRejects:
    def test_reject_zero_tokens(self):
        ctx, cfg = _make_context(0)
        with pytest.raises(DecodeMetadataError):
            extract_pypto_decode_meta(ctx, vllm_config=cfg)

    def test_reject_over_max_batch(self):
        invalid = STORAGE_BATCH + 1
        ctx, cfg = _make_context(
            invalid,
            seq_lens=list(range(2, 2 + invalid)),
            storage_capacity=invalid,
        )
        with pytest.raises(DecodeMetadataError):
            extract_pypto_decode_meta(
                ctx, vllm_config=cfg, max_batch=STORAGE_BATCH
            )

    def test_reject_prefill(self):
        ctx, cfg = _make_context(2, num_prefills=1)
        with pytest.raises(DecodeMetadataError):
            extract_pypto_decode_meta(ctx, vllm_config=cfg)

    def test_reject_pipeline_parallel(self):
        ctx, cfg = _make_context(2, pipeline_parallel_size=2)
        with pytest.raises(DecodeMetadataError):
            extract_pypto_decode_meta(ctx, vllm_config=cfg)

    def test_reject_profile_run(self):
        ctx, cfg = _make_context(2)
        ctx.in_profile_run = True
        with pytest.raises(DecodeMetadataError):
            extract_pypto_decode_meta(ctx, vllm_config=cfg)

    def test_reject_positions_mismatch(self):
        # positions must equal seq_lens - 1 for pure decode.
        ctx, cfg = _make_context(2, seq_lens=[3, 5], positions=[0, 0])
        with pytest.raises(DecodeMetadataError):
            extract_pypto_decode_meta(ctx, vllm_config=cfg)

    def test_reject_nonpositive_seq_lens(self):
        ctx, cfg = _make_context(2, seq_lens=[3, 0], positions=[2, -1])
        with pytest.raises(DecodeMetadataError):
            extract_pypto_decode_meta(ctx, vllm_config=cfg)

    def test_reject_missing_decoder_layer(self):
        ctx, cfg = _make_context(2)
        del ctx.attn_metadata["model.layers.7.self_attn.attn"]
        with pytest.raises(DecodeMetadataError):
            extract_pypto_decode_meta(ctx, vllm_config=cfg)

    @pytest.mark.parametrize(
        "reserve",
        [
            {
                "scheduler_num_blocks": 64,
                "physical_num_blocks": 78,
                "reserve_start": 64,
                "padding_block_ids": list(range(64, 79)),
                "padding_block_count": 15,
                "block_size": 128,
            },
            {
                "scheduler_num_blocks": 64,
                "physical_num_blocks": 79,
                "reserve_start": 63,
                "padding_block_ids": list(range(64, 79)),
                "padding_block_count": 15,
                "block_size": 128,
            },
            {
                "scheduler_num_blocks": 64,
                "physical_num_blocks": 79,
                "reserve_start": 64,
                "padding_block_ids": [63] + list(range(65, 79)),
                "padding_block_count": 15,
                "block_size": 128,
            },
        ],
    )
    def test_reject_invalid_reserve(self, reserve):
        ctx, cfg = _make_context(2)
        ctx.pypto_padding_reserve = reserve
        with pytest.raises(DecodeMetadataError):
            extract_pypto_decode_meta(ctx, vllm_config=cfg)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
