#!/usr/bin/env python3
"""Card-free tests for the pure-prefill metadata bridge (design §5.1.2 + §3.1).

Dual of ``tests/step3p5/unit/test_decode_metadata_bridge.py``.  Covers the
single-sequence prefill contract the whole-prefill ABI depends on
(hard emphasis: per-token vs per-request shapes, variable T, multi-block
seq_lens, padding init, and fail-closed rejection of every unsupported
vLLM step):

  - ``positions`` and ``slot_mapping`` are per-token (length ``PREFILL_T``);
  - ``seq_lens`` and ``block_table`` are per-request (``PREFILL_BATCH == 1``);
  - ``block_table`` is the single sequence's flat 1-D block list;
  - a fresh prompt has ``positions == arange(T)`` and ``seq_lens == [T]``;
  - a continuation query has ``positions == [seq_len - T, ..., seq_len - 1]``
    and the block table covers ``ceil(seq_len / 128)`` scheduler blocks;
  - ``slot_mapping[t] == block_table[pos // 128] * 128 + pos % 128``;
  - padding tokens (``T .. PREFILL_T``) point at the allocator-owned reserve
    block; inactive block-table columns are zero;
  - reject decode-only / mixed batch / multi-prefill / chunked-prefill /
    CP / PCP / PP>1 / profile / T > PREFILL_T / multi KV group / missing
    layer / non-positive seq_len / non-contiguous positions / zero tokens /
    slot-mapping mismatch / bad padding reserve;
  - the KV group map covers all 45 decoder layers with no layer in two groups;
  - protocol tensor/meta shapes are stable.

Uses torch (available card-free); no vLLM, no device.

Run:
    python3 -m pytest tests/step3p5/unit/test_prefill_metadata_bridge.py -q
"""
from __future__ import annotations

import os
import sys

import pytest

torch = pytest.importorskip("torch")

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.step3p5.kv_padding import (  # noqa: E402
    STORAGE_BATCH,
    make_padding_reserve,
)
from tools.step3p5.vllm_prefill_metadata import (  # noqa: E402
    PREFILL_T,
    PrefillMetadataError,
    PrefillPlan,
    extract_pypto_prefill_meta,
)

_NUM_LAYERS = 45
_SCHEDULER_BLOCKS = 64
_STORAGE_CAPACITY = STORAGE_BATCH
_PHYSICAL_BLOCKS = _SCHEDULER_BLOCKS + _STORAGE_CAPACITY - 1
_RESERVE_BLOCK = _SCHEDULER_BLOCKS


def _make_prefill_context(
    t: int = PREFILL_T,
    *,
    seq_len: int | None = None,
    num_groups: int = 1,
    num_prefills: int = 1,
    num_decodes: int = 0,
    chunked_prefill: bool = False,
    pipeline_parallel_size: int = 1,
    prefill_context_parallel_size: int = 1,
    pcp_metadata: bool = False,
    positions=None,
    block_table_width: int = 4,
    **meta_overrides,
):
    """Build a fake single-prefill ForwardContext for ``t`` query tokens.

    Mirrors ``test_decode_metadata_bridge._make_context``: inner fake classes
    for Meta / Group / Kvc / Parallel / Sched / Cfg, one metadata object
    shared by all 45 layers, and a configured number of KV groups.  All 45
    layers share one metadata object (the target builder assigns one
    AscendMetadata per KV group); groups come from kv_cache_config.
    """
    if seq_len is None:
        seq_len = t
    if positions is None:
        # Contiguous query ending at seq_len - 1 (fresh prompt: arange(T)).
        positions = list(range(seq_len - t, seq_len))
    n = max(num_prefills, 1)
    block_tables = (
        torch.arange(n * block_table_width, dtype=torch.int32)
        .reshape(n, block_table_width)
        % _SCHEDULER_BLOCKS
    )
    slot_mapping = torch.tensor(
        [
            int(block_tables[0, pos // 128]) * 128 + pos % 128
            for pos in positions
        ],
        dtype=torch.int32,
    )

    # All metadata attributes are set on the instance. A class body cannot
    # reference enclosing function locals when the local shares the attr
    # name (e.g. ``num_prefills = num_prefills`` makes the RHS resolve to
    # the not-yet-assigned class local), so scalars are instance attrs too,
    # mirroring the decode test's fake-class + instance-attr pattern.
    class Meta:
        attn_state = "PrefillOnly"

    meta = Meta()
    meta.num_actual_tokens = t
    meta.num_reqs = num_prefills
    meta.num_prefills = num_prefills
    meta.num_decodes = num_decodes
    meta.num_decode_tokens = num_decodes
    meta.actual_seq_lengths_q = [t]
    meta.query_start_loc = torch.tensor([0, t], dtype=torch.int32)
    meta.seq_lens = torch.tensor([seq_len], dtype=torch.int32)
    meta.positions = torch.tensor(positions, dtype=torch.int32)
    meta.block_tables = block_tables
    meta.slot_mapping = slot_mapping
    if pcp_metadata:
        meta.prefill_context_parallel_metadata = object()
    for key, value in meta_overrides.items():
        setattr(meta, key, value)

    class Group:
        def __init__(self, layer_names):
            self.layer_names = layer_names

    groups = []
    for group_id in range(num_groups):
        groups.append(
            Group(
                [
                    f"model.layers.{i}.self_attn.attn"
                    for i in range(group_id, _NUM_LAYERS, num_groups)
                ]
            )
        )

    class Kvc:
        kv_cache_groups = groups

    class Parallel:
        pipeline_parallel_size = 1
        prefill_context_parallel_size = 1
        decode_context_parallel_size = 1

    Parallel.pipeline_parallel_size = pipeline_parallel_size
    Parallel.prefill_context_parallel_size = prefill_context_parallel_size

    class Sched:
        chunked_prefill_enabled = False
        enable_chunked_prefill = False

    Sched.chunked_prefill_enabled = chunked_prefill
    Sched.enable_chunked_prefill = chunked_prefill

    class Cfg:
        kv_cache_config = Kvc()
        parallel_config = Parallel()
        scheduler_config = Sched()
        speculative_config = None

    ctx = type("Ctx", (), {})()
    ctx.attn_metadata = {
        f"model.layers.{i}.self_attn.attn": meta for i in range(_NUM_LAYERS)
    }
    ctx.vllm_config = Cfg()
    ctx.pypto_padding_reserve = make_padding_reserve(
        _SCHEDULER_BLOCKS,
        _PHYSICAL_BLOCKS,
        storage_capacity=_STORAGE_CAPACITY,
    )
    return ctx, Cfg()


class TestPrefillMetadataHappyPath:
    def test_single_fresh_prefill_full(self):
        ctx, cfg = _make_prefill_context(PREFILL_T)
        plan = extract_pypto_prefill_meta(ctx, vllm_config=cfg)
        assert isinstance(plan, PrefillPlan)
        assert plan.valid_tokens == PREFILL_T
        assert plan.prefill_t == PREFILL_T
        assert plan.valid_requests == 1
        assert plan.seq_lens.dtype == torch.int32
        assert plan.positions.dtype == torch.int32
        assert plan.seq_lens.tolist() == [PREFILL_T]
        assert torch.equal(
            plan.positions[: plan.valid_tokens],
            torch.arange(PREFILL_T, dtype=torch.int32),
        )
        # Fresh prompt: T == PREFILL_T, so there is no padding tail.
        assert tuple(plan.positions.shape) == (PREFILL_T,)

    @pytest.mark.parametrize("t", [1, 32, 64, 128])
    def test_variable_T(self, t):
        ctx, cfg = _make_prefill_context(t)
        plan = extract_pypto_prefill_meta(ctx, vllm_config=cfg)
        assert plan.valid_tokens == t
        assert plan.seq_lens.tolist() == [t]
        assert torch.equal(
            plan.positions[:t], torch.arange(t, dtype=torch.int32)
        )
        assert tuple(plan.positions.shape) == (PREFILL_T,)
        # Padding tail (only when T < PREFILL_T): positions == 0 and slots
        # point at the allocator-owned reserve block.
        if t < PREFILL_T:
            assert torch.all(plan.positions[t:] == 0)
            for group in plan.groups:
                assert torch.all(
                    group.slot_mapping[t:] == _RESERVE_BLOCK * 128
                )

    def test_slot_mapping_matches_block_table(self):
        ctx, cfg = _make_prefill_context(100)
        source = next(iter(ctx.attn_metadata.values()))
        plan = extract_pypto_prefill_meta(ctx, vllm_config=cfg)
        for group in plan.groups:
            for token in range(plan.valid_tokens):
                pos = int(source.positions[token])
                col = pos // 128
                expected = int(source.block_tables[0, col]) * 128 + pos % 128
                assert int(group.slot_mapping[token]) == expected

    def test_active_block_table_preserved(self):
        ctx, cfg = _make_prefill_context(100, seq_len=300)
        source = next(iter(ctx.attn_metadata.values()))
        seq_len = int(source.seq_lens[0])
        active_blocks = (seq_len + 127) // 128
        plan = extract_pypto_prefill_meta(ctx, vllm_config=cfg)
        for group in plan.groups:
            assert torch.equal(
                group.block_table[:active_blocks],
                source.block_tables[0, :active_blocks],
            )
            assert torch.count_nonzero(group.block_table[active_blocks:]) == 0

    def test_padding_initialization(self):
        ctx, cfg = _make_prefill_context(50, seq_len=50)
        plan = extract_pypto_prefill_meta(ctx, vllm_config=cfg)
        assert torch.all(plan.positions[50:] == 0)
        for group in plan.groups:
            assert tuple(group.slot_mapping.shape) == (PREFILL_T,)
            assert torch.all(group.slot_mapping[50:] == _RESERVE_BLOCK * 128)
            # Inactive block-table columns (beyond the one block used by a
            # 50-token sequence) are zero.
            assert int(group.block_table[1]) == 0


class TestPrefillMetadataMultiBlock:
    def test_multi_block_prior_context(self):
        # seq_len=256 with a T=128 query lands entirely in the second block.
        ctx, cfg = _make_prefill_context(128, seq_len=256)
        source = next(iter(ctx.attn_metadata.values()))
        plan = extract_pypto_prefill_meta(ctx, vllm_config=cfg)
        assert plan.valid_tokens == 128
        assert plan.seq_lens.tolist() == [256]
        assert torch.equal(
            plan.positions[:128],
            torch.arange(128, 256, dtype=torch.int32),
        )
        # block_table covers ceil(256/128)=2 scheduler blocks; tail is zero.
        for group in plan.groups:
            assert torch.equal(
                group.block_table[:2], source.block_tables[0, :2]
            )
            assert torch.count_nonzero(group.block_table[2:]) == 0
            # Every query token's slot is block_table[1]*128 + pos%128.
            for token in range(128):
                pos = 128 + token
                assert int(group.slot_mapping[token]) == (
                    int(source.block_tables[0, 1]) * 128 + pos % 128
                )


class TestPrefillMetadataGroups:
    def test_single_kv_group_covers_45_layers(self):
        ctx, cfg = _make_prefill_context(64, num_groups=1)
        plan = extract_pypto_prefill_meta(ctx, vllm_config=cfg)
        assert len(plan.layer_to_group) == _NUM_LAYERS
        assert all(g == 0 for g in plan.layer_to_group)
        assert len(plan.groups) == 1
        assert plan.groups[0].layer_indices == tuple(range(_NUM_LAYERS))

    def test_protocol_tensors_and_meta(self):
        ctx, cfg = _make_prefill_context(96)
        plan = extract_pypto_prefill_meta(ctx, vllm_config=cfg)
        tensors = plan.protocol_tensors()
        assert set(tensors) == {
            "meta_seq_lens",
            "meta_positions",
            "meta_block_table_g0",
            "meta_slot_mapping_g0",
        }
        assert tuple(tensors["meta_seq_lens"].shape) == (1,)
        assert tuple(tensors["meta_positions"].shape) == (PREFILL_T,)
        assert tensors["meta_block_table_g0"].ndim == 1
        assert tuple(tensors["meta_slot_mapping_g0"].shape) == (PREFILL_T,)
        pmeta = plan.protocol_meta()
        assert pmeta["protocol_version"] == 2
        assert pmeta["op"] == "prefill"
        assert pmeta["valid_tokens"] == 96
        assert pmeta["valid_requests"] == 1
        assert pmeta["prefill_t"] == PREFILL_T
        assert pmeta["kv_group_count"] == 1
        assert len(pmeta["layer_to_group"]) == _NUM_LAYERS
        assert pmeta["query_lengths"] == [96]
        assert pmeta["padding_reserve"]["reserve_start"] == _SCHEDULER_BLOCKS


class TestPrefillMetadataRejects:
    def test_reject_decode_only(self):
        ctx, cfg = _make_prefill_context(
            2, num_prefills=0, num_decodes=2, attn_state="DecodeOnly"
        )
        with pytest.raises(PrefillMetadataError, match="num_prefills > 0"):
            extract_pypto_prefill_meta(ctx, vllm_config=cfg)

    def test_reject_mixed_batch(self):
        ctx, cfg = _make_prefill_context(64, num_decodes=1)
        with pytest.raises(PrefillMetadataError, match="num_decodes == 0"):
            extract_pypto_prefill_meta(ctx, vllm_config=cfg)

    def test_reject_multi_prefill(self):
        ctx, cfg = _make_prefill_context(128, num_prefills=2)
        with pytest.raises(PrefillMetadataError, match="single prefill request"):
            extract_pypto_prefill_meta(ctx, vllm_config=cfg)

    def test_reject_chunked_prefill(self):
        ctx, cfg = _make_prefill_context(64, chunked_prefill=True)
        with pytest.raises(PrefillMetadataError, match="chunked prefill"):
            extract_pypto_prefill_meta(ctx, vllm_config=cfg)

    def test_reject_prefill_context_parallel(self):
        ctx, cfg = _make_prefill_context(64, prefill_context_parallel_size=2)
        with pytest.raises(
            PrefillMetadataError, match="prefill context parallelism"
        ):
            extract_pypto_prefill_meta(ctx, vllm_config=cfg)

    def test_reject_pcp_metadata(self):
        ctx, cfg = _make_prefill_context(64, pcp_metadata=True)
        with pytest.raises(PrefillMetadataError, match="PCP metadata"):
            extract_pypto_prefill_meta(ctx, vllm_config=cfg)

    def test_reject_pipeline_parallel(self):
        ctx, cfg = _make_prefill_context(64, pipeline_parallel_size=2)
        with pytest.raises(PrefillMetadataError, match="pipeline parallelism"):
            extract_pypto_prefill_meta(ctx, vllm_config=cfg)

    def test_reject_profile_run(self):
        ctx, cfg = _make_prefill_context(64)
        ctx.in_profile_run = True
        with pytest.raises(PrefillMetadataError, match="profile"):
            extract_pypto_prefill_meta(ctx, vllm_config=cfg)

    def test_reject_T_exceeds_prefill_t(self):
        ctx, cfg = _make_prefill_context(PREFILL_T + 1, seq_len=PREFILL_T + 1)
        with pytest.raises(PrefillMetadataError, match="exceeds compiled"):
            extract_pypto_prefill_meta(ctx, vllm_config=cfg)

    def test_reject_multi_kv_group(self):
        ctx, cfg = _make_prefill_context(64, num_groups=2)
        with pytest.raises(PrefillMetadataError, match="single KV group"):
            extract_pypto_prefill_meta(ctx, vllm_config=cfg)

    def test_reject_missing_decoder_layer(self):
        ctx, cfg = _make_prefill_context(64)
        del ctx.attn_metadata["model.layers.7.self_attn.attn"]
        with pytest.raises(PrefillMetadataError, match="missing decoder layers"):
            extract_pypto_prefill_meta(ctx, vllm_config=cfg)

    def test_reject_nonpositive_seq_len(self):
        ctx, cfg = _make_prefill_context(
            64, seq_lens=torch.tensor([0], dtype=torch.int32)
        )
        with pytest.raises(PrefillMetadataError, match="seq_lens must be positive"):
            extract_pypto_prefill_meta(ctx, vllm_config=cfg)

    def test_reject_positions_not_contiguous(self):
        # Off-by-one query: positions == [1, 2, ..., 128] instead of arange(T).
        ctx, cfg = _make_prefill_context(64, positions=list(range(1, 65)))
        with pytest.raises(PrefillMetadataError, match="contiguous"):
            extract_pypto_prefill_meta(ctx, vllm_config=cfg)

    def test_reject_zero_tokens(self):
        ctx, cfg = _make_prefill_context(0)
        with pytest.raises(PrefillMetadataError, match="positive actual tokens"):
            extract_pypto_prefill_meta(ctx, vllm_config=cfg)

    def test_reject_slot_mapping_mismatch(self):
        ctx, cfg = _make_prefill_context(64)
        ctx.attn_metadata["model.layers.0.self_attn.attn"].slot_mapping = (
            torch.zeros(PREFILL_T, dtype=torch.int32)
        )
        with pytest.raises(PrefillMetadataError, match="slot_mapping does not match"):
            extract_pypto_prefill_meta(ctx, vllm_config=cfg)

    @pytest.mark.parametrize(
        "reserve",
        [
            # Malformed reserve map (missing required integer fields).
            {"scheduler_num_blocks": 64},
            # Valid structure but no allocator-owned padding block.
            make_padding_reserve(64, 64, storage_capacity=1).as_dict(),
        ],
    )
    def test_reject_invalid_reserve(self, reserve):
        ctx, cfg = _make_prefill_context(64)
        with pytest.raises(PrefillMetadataError):
            extract_pypto_prefill_meta(
                ctx, vllm_config=cfg, padding_reserve=reserve
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
