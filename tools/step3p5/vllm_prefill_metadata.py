# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Pure-prefill metadata bridge from vLLM-Ascend to the PyPTO ABI.

This module is the prefill dual of ``tools/step3p5/vllm_decode_metadata.py``.
The decode bridge accepts a ``ForwardContext``-like object describing a
pure-decode step and emits the fixed-batch paged tensors the 45-layer
``whole_decode_step3p5`` program consumes.  The prefill bridge does the same
for a single-sequence prefill step targeting the ``whole_prefill_step3p5``
program (``models/step3p5/prefill_layer_single_chip_hidden.py``).

Prefill semantics differ from decode in three load-bearing ways, so this
module does **not** reuse decode's fixed-batch paged ABI
(``pad_fixed_batch_metadata`` / the ``[storage_batch=16]`` tensor shape):

* the program token tensor is ``[PREFILL_T=128, HIDDEN=4096]`` — every row is
  a real token of one sequence, not one paged request row;
* ``positions`` and ``slot_mapping`` are per-token (length ``PREFILL_T``),
  while ``seq_lens`` and ``block_table`` are per-request (one prefill
  request, ``PREFILL_BATCH=1``);
* ``block_table`` is the single sequence's flat 1-D block list
  (``[BLOCK_TABLE_FLAT_DYN]`` in the program signature), not the decode
  ``[storage_batch, max_blocks]`` 2-D paged table.

A normal prefill step has one request (``num_prefills == 1``) carrying a
contiguous query of ``T`` tokens (``1 <= T <= PREFILL_T``).  The query may
land anywhere in the sequence: ``positions == [seq_len - T, ..., seq_len -
1]``.  A fresh prompt has ``seq_len == T`` so ``positions == arange(T)``.
The block table covers ``ceil(seq_len / BLOCK_SIZE)`` scheduler-owned blocks
(possibly several when the sequence has prior context).

Requirements (all fail-closed — this bridge never silently falls back):

* ``num_prefills == 1`` and ``num_decodes == 0`` (pure single-prefill step);
* ``1 <= T <= PREFILL_T``;
* no chunked-prefill, context-parallel (CP/PCP), or pipeline-parallel handoff;
* a single KV group (the first version requires
  ``--disable-hybrid-kv-cache-manager``);
* every decoder layer has complete group metadata and the per-layer counts
  agree;
* positions are contiguous and end at ``seq_len - 1``;
* ``slot_mapping[t] == block_table[pos // BLOCK_SIZE] * BLOCK_SIZE + pos %
  BLOCK_SIZE`` for every active token.

Padding is initialized here, not by callers:

* ``positions[T:PREFILL_T] == 0`` and ``slot_mapping[T:PREFILL_T]`` points at
  the allocator-owned reserve block (one block is enough — padding KV is
  never read, the tail hidden rows are zeroed by the holder);
* inactive ``block_table`` columns (beyond ``ceil(seq_len / BLOCK_SIZE)``)
  are zero.

The frozen canonical constants shared by every prefill bridge file:

* ``PREFILL_T = 128``, ``PREFILL_BATCH = 1``, ``HIDDEN = 4096`` (from
  ``models.step3p5.prefill_qkv_proj_rope`` and ``models.step3p5.config``);
* socket path default ``/logs/pypto_prefill.sock``
  (``DEFAULT_PREFILL_SOCKET``).

The decode metadata module ``extract_pypto_decode_plan`` explicitly rejects
prefill (raises when ``num_prefills != 0``); this module is the dual — it
requires ``num_prefills > 0`` and rejects decode-only steps.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch

from tools.step3p5.kv_padding import (
    PaddingReserve,
    PaddingReserveError,
    make_padding_reserve,
    parse_padding_reserve,
)

_LAYER_RE = re.compile(r"(?:^|[.]layers[.])(\d+)(?:[.]|$)")

# Constants are hardcoded here (not imported from ``models.step3p5.*``) so the
# bridge stays importable in the card-free unit-test environment, exactly like
# ``vllm_decode_metadata.py``.  Source of truth:
#   models/step3p5/config.py            -> HIDDEN=4096, BLOCK_SIZE=128,
#                                          NUM_HIDDEN_LAYERS=45
#   models/step3p5/prefill_qkv_proj_rope.py -> PREFILL_BATCH=1, PREFILL_SEQ=128
_NUM_LAYERS = 45
PREFILL_T = 128                       # PREFILL_BATCH * PREFILL_SEQ
PREFILL_BATCH = 1
BLOCK_SIZE = 128                      # paged-cache block (also K/V SEQ_TILE)
_MAX_PREFILL_T = PREFILL_T
DEFAULT_PREFILL_SOCKET = "/logs/pypto_prefill.sock"


class PrefillMetadataError(ValueError):
    """Raised when the current vLLM step is not supported by the prefill ABI."""


def _tensor(value: Any, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise PrefillMetadataError(
            f"{name}: expected torch.Tensor, got {type(value).__name__}"
        )
    return value.detach().to("cpu").contiguous()


def _i32(value: Any, *, name: str) -> torch.Tensor:
    result = _tensor(value, name=name)
    if result.dtype not in (torch.int32, torch.int64, torch.long):
        raise PrefillMetadataError(
            f"{name}: expected integer dtype, got {result.dtype}"
        )
    return result.to(torch.int32)


def _attr(
    obj: Any,
    names: Sequence[str],
    *,
    name: str,
    required: bool = True,
) -> Any:
    for candidate in names:
        value = getattr(obj, candidate, None)
        if value is not None:
            return value
    if required:
        raise PrefillMetadataError(f"{name}: missing any of {tuple(names)}")
    return None


def _int_attr(
    obj: Any,
    names: Sequence[str],
    *,
    name: str,
    default: int | None = None,
) -> int | None:
    value = _attr(obj, names, name=name, required=default is None)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise PrefillMetadataError(
            f"{name}: expected integer, got {value!r}"
        ) from exc


def _layer_index(name: str) -> int | None:
    match = _LAYER_RE.search(str(name))
    return int(match.group(1)) if match else None


def _enum_name(value: Any) -> str:
    if value is None:
        return ""
    return (
        str(getattr(value, "name", value))
        .rsplit(".", 1)[-1]
        .lower()
        .replace("_", "")
    )


def _chunked_prefill_enabled(scheduler_config: Any) -> bool:
    """Return True if vLLM chunked-prefill scheduling is active."""
    if scheduler_config is None:
        return False
    for candidate in ("chunked_prefill_enabled", "enable_chunked_prefill"):
        value = getattr(scheduler_config, candidate, None)
        if value is not None and bool(value):
            return True
    return False


def _prefill_query_lengths(
    metadata: Any,
    valid_tokens: int,
    num_prefills: int,
) -> list[int]:
    """Derive the per-request prefill query lengths.

    Mirrors decode's ``_query_lengths`` but does not fall back to "one token
    per request" — a prefill request always carries a multi-token query.
    """
    starts = _attr(
        metadata,
        ("query_start_loc_cpu", "query_start_loc"),
        name="query_start_loc",
        required=False,
    )
    if starts is not None:
        starts_cpu = _i32(starts, name="query_start_loc").flatten()
        if starts_cpu.numel() >= num_prefills + 1:
            starts_cpu = starts_cpu[: num_prefills + 1]
            lengths = (starts_cpu[1:] - starts_cpu[:-1]).tolist()
            if sum(lengths) == valid_tokens:
                return [int(x) for x in lengths]

    actual = getattr(metadata, "actual_seq_lengths_q", None)
    if isinstance(actual, (list, tuple)) and len(actual) >= num_prefills:
        # vLLM-Ascend stores query_start_loc[1:] in this field.
        ends = [int(x) for x in actual[:num_prefills]]
        starts_list = [0] + ends[:-1]
        lengths = [end - start for start, end in zip(starts_list, ends)]
        if sum(lengths) == valid_tokens:
            return lengths

    # A single-prefill fixture without query_start_loc: all tokens belong to
    # the one request.  This path is intentionally only valid for num_prefills
    # == 1 (PREFILL_BATCH == 1); multi-prefill callers must supply
    # query_start_loc so each request's boundary is unambiguous.
    if num_prefills == 1:
        return [valid_tokens]
    raise PrefillMetadataError("cannot derive per-request prefill query lengths")


def _group_layer_indices(
    vllm_config: Any,
    layer_names: Mapping[int, Any],
) -> list[tuple[int, tuple[int, ...]]]:
    groups = getattr(
        getattr(vllm_config, "kv_cache_config", None),
        "kv_cache_groups",
        None,
    )
    if groups:
        result: list[tuple[int, tuple[int, ...]]] = []
        seen: set[int] = set()
        for group_id, group in enumerate(groups):
            indices = []
            for layer_name in getattr(group, "layer_names", ()):
                index = _layer_index(layer_name)
                if index is not None and index < _NUM_LAYERS:
                    indices.append(index)
                    seen.add(index)
            if indices:
                result.append((group_id, tuple(sorted(set(indices)))))
        if result and len(seen) == _NUM_LAYERS:
            return result
        raise PrefillMetadataError(
            f"kv_cache_config groups cover {len(seen)}/{_NUM_LAYERS} decoder layers"
        )

    # Offline fallback: the target builder assigns one metadata object to all
    # layers in a group.  Not used for production vLLM; keeps the bridge
    # unit-testable without constructing VllmConfig.
    by_identity: dict[int, list[int]] = {}
    for index, metadata in layer_names.items():
        by_identity.setdefault(id(metadata), []).append(index)
    return [
        (group_id, tuple(sorted(indices)))
        for group_id, indices in enumerate(by_identity.values())
    ]


@dataclass(frozen=True)
class PyPtoPrefillKvGroupMeta:
    group_id: int
    layer_indices: tuple[int, ...]
    block_table: torch.Tensor   # [max_blocks] flat INT32 — single sequence
    slot_mapping: torch.Tensor  # [PREFILL_T] INT32 — per-token


@dataclass(frozen=True)
class PrefillPlan:
    """Validated single-prefill metadata plan for the 45-layer PyPTO program.

    Field shape differs from ``PyPtoDecodeMeta`` because prefill is
    single-sequence, not paged: ``seq_lens`` is per-request (length
    ``num_prefills``), ``positions`` / ``slot_mapping`` are per-token (length
    ``prefill_t``), and each group's ``block_table`` is the one sequence's
    flat block list.
    """

    valid_tokens: int
    prefill_t: int
    seq_lens: torch.Tensor        # [num_prefills], INT32
    positions: torch.Tensor       # [prefill_t], INT32
    groups: tuple[PyPtoPrefillKvGroupMeta, ...]
    layer_to_group: tuple[int, ...]
    query_lengths: tuple[int, ...]
    padding_reserve: PaddingReserve

    @property
    def valid_requests(self) -> int:
        return len(self.query_lengths)

    def protocol_tensors(self) -> dict[str, torch.Tensor]:
        """Return fixed-shape tensors suitable for the socket frame."""
        result: dict[str, torch.Tensor] = {
            "meta_seq_lens": self.seq_lens,
            "meta_positions": self.positions,
        }
        for group in self.groups:
            result[f"meta_block_table_g{group.group_id}"] = group.block_table
            result[f"meta_slot_mapping_g{group.group_id}"] = group.slot_mapping
        return result

    def protocol_meta(self) -> dict[str, Any]:
        return {
            "protocol_version": 2,
            "op": "prefill",
            "valid_tokens": self.valid_tokens,
            "valid_requests": self.valid_requests,
            "prefill_t": self.prefill_t,
            "kv_group_count": len(self.groups),
            "layer_to_group": list(self.layer_to_group),
            "query_lengths": list(self.query_lengths),
            "padding_reserve": self.padding_reserve.as_dict(),
        }


def _resolve_padding_reserve(
    forward_context: Any,
    *,
    padding_reserve: PaddingReserve | Mapping[str, Any] | None = None,
) -> PaddingReserve:
    if padding_reserve is None:
        padding_reserve = getattr(
            forward_context,
            "pypto_padding_reserve",
            None,
        )
    if padding_reserve is None:
        try:
            from tools.step3p5.vllm_kvpool_backend import (  # noqa: PLC0415
                get_padding_reserve,
            )

            padding_reserve = get_padding_reserve("main")
        except Exception as exc:  # noqa: BLE001
            raise PrefillMetadataError(
                "Main allocator-owned padding reserve is unavailable; "
                "PYPTO_KVPOOL must publish scheduler/physical reserve before "
                "prefill metadata extraction"
            ) from exc
    if isinstance(padding_reserve, PaddingReserve):
        return padding_reserve
    try:
        return parse_padding_reserve(
            padding_reserve,
            where="Main padding reserve",
        )
    except PaddingReserveError as exc:
        raise PrefillMetadataError(str(exc)) from exc


def _extract_prefill_group_metadata(
    metadata_by_layer: Mapping[int, Any],
    indices: Iterable[int],
    *,
    num_prefills: int,
    valid_tokens: int,
    seq_len: int,
    positions: torch.Tensor,
    reserve: PaddingReserve,
    prefill_t: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (flat block_table [max_blocks], slot_mapping [prefill_t])."""
    representatives = [
        metadata_by_layer[index]
        for index in indices
        if index in metadata_by_layer
    ]
    if not representatives:
        raise PrefillMetadataError("KV group has no layer metadata")
    metadata = representatives[0]
    raw_block = _attr(
        metadata,
        ("block_tables", "block_table_tensor", "block_table"),
        name="block_table",
    )
    block = _i32(raw_block, name="block_table")
    if (
        block.ndim != 2
        or block.shape[0] < num_prefills
        or block.shape[1] <= 0
    ):
        raise PrefillMetadataError(
            f"block_table: expected [requests, blocks], got {tuple(block.shape)}"
        )
    active_blocks_needed = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    if block.shape[1] < active_blocks_needed:
        raise PrefillMetadataError(
            f"block_table width {block.shape[1]} cannot cover seq_len={seq_len} "
            f"(needs {active_blocks_needed} blocks of size {BLOCK_SIZE})"
        )
    active_table = block[0, :active_blocks_needed].to(torch.int32)
    if torch.any(active_table < 0) or torch.any(
        active_table >= reserve.scheduler_num_blocks
    ):
        raise PrefillMetadataError(
            f"active block_table ids must lie in scheduler domain "
            f"[0,{reserve.scheduler_num_blocks}), got {active_table.tolist()}"
        )
    # Flat 1-D block table for the single prefill sequence; trailing columns
    # (beyond the sequence) are zeroed.
    block_flat = torch.zeros(int(block.shape[1]), dtype=torch.int32)
    block_flat[:active_blocks_needed] = active_table

    raw_slot = _attr(metadata, ("slot_mapping",), name="slot_mapping")
    slot = _i32(raw_slot, name="slot_mapping").flatten()
    if slot.numel() < valid_tokens:
        raise PrefillMetadataError(
            f"slot_mapping has {slot.numel()} entries, need {valid_tokens}"
        )
    active_slot = slot[:valid_tokens].to(torch.int32)
    # Each query token's slot must equal block_table[pos // BLOCK] * BLOCK +
    # pos % BLOCK.  ``positions`` carries absolute cache row per token.
    pos_long = positions.to(torch.long)
    cols = (pos_long // BLOCK_SIZE).to(torch.long)
    expected_slot = (
        active_table.index_select(0, cols) * BLOCK_SIZE
        + (pos_long % BLOCK_SIZE).to(torch.int32)
    )
    if not torch.equal(active_slot, expected_slot):
        raise PrefillMetadataError(
            "slot_mapping does not match block_table*BLOCK_SIZE+pos%BLOCK_SIZE: "
            f"slot={active_slot.tolist()}, expected={expected_slot.tolist()}"
        )

    slot_out = torch.zeros(prefill_t, dtype=torch.int32)
    slot_out[:valid_tokens] = active_slot
    if valid_tokens < prefill_t:
        if not reserve.padding_block_ids:
            raise PrefillMetadataError(
                "padding reserve has no allocator-owned block for prefill padding"
            )
        padding_block = reserve.padding_block_ids[0]
        slot_out[valid_tokens:] = padding_block * BLOCK_SIZE
    return block_flat, slot_out


def extract_pypto_prefill_meta(
    forward_context: Any,
    *,
    vllm_config: Any = None,
    positions: Any = None,
    padding_reserve: PaddingReserve | Mapping[str, Any] | None = None,
    prefill_t: int = _MAX_PREFILL_T,
) -> PrefillPlan:
    """Build the validated prefill plan for one vLLM prefill forward.

    ``prefill_t`` is the compile-time static token capacity (``PREFILL_T``),
    not the runtime active token count.  The latter is derived from the input
    metadata and must be no greater than this capacity.
    """
    prefill_t = int(prefill_t)
    if prefill_t <= 0:
        raise PrefillMetadataError(
            f"prefill_t must be positive, got {prefill_t}"
        )
    if bool(getattr(forward_context, "in_profile_run", False)):
        raise PrefillMetadataError("profile/dummy runs are not supported")
    if vllm_config is not None:
        parallel = getattr(vllm_config, "parallel_config", None)
        if int(getattr(parallel, "pipeline_parallel_size", 1)) != 1:
            raise PrefillMetadataError("pipeline parallelism is unsupported")
        if int(getattr(parallel, "prefill_context_parallel_size", 1)) != 1:
            raise PrefillMetadataError("prefill context parallelism is unsupported")
        if int(getattr(parallel, "decode_context_parallel_size", 1)) != 1:
            raise PrefillMetadataError("decode context parallelism is unsupported")
        if _chunked_prefill_enabled(
            getattr(vllm_config, "scheduler_config", None)
        ):
            raise PrefillMetadataError("chunked prefill is unsupported")

    attn_metadata = getattr(forward_context, "attn_metadata", None)
    if not isinstance(attn_metadata, Mapping) or not attn_metadata:
        raise PrefillMetadataError(
            "forward context has no layer attention metadata"
        )

    metadata_by_layer: dict[int, Any] = {}
    for layer_name, metadata in attn_metadata.items():
        index = _layer_index(str(layer_name))
        if index is None or index >= _NUM_LAYERS:
            continue
        metadata_by_layer[index] = metadata
    if set(metadata_by_layer) != set(range(_NUM_LAYERS)):
        missing = sorted(set(range(_NUM_LAYERS)) - set(metadata_by_layer))
        raise PrefillMetadataError(
            f"attention metadata missing decoder layers: {missing[:8]}"
        )

    representative = metadata_by_layer[0]
    valid_tokens = _int_attr(
        representative,
        ("num_actual_tokens",),
        name="num_actual_tokens",
    )
    num_prefills = _int_attr(
        representative,
        ("num_prefills",),
        name="num_prefills",
    )
    num_decodes = _int_attr(
        representative,
        ("num_decodes",),
        name="num_decodes",
        default=0,
    )
    reported_requests_raw = getattr(representative, "num_reqs", None)
    try:
        reported_requests = (
            None if reported_requests_raw is None
            else int(reported_requests_raw)
        )
    except (TypeError, ValueError) as exc:
        raise PrefillMetadataError(
            f"num_reqs: expected integer, got {reported_requests_raw!r}"
        ) from exc
    assert valid_tokens is not None
    assert num_prefills is not None

    # Dual of decode's prefill reject: decode rejects num_prefills != 0;
    # prefill requires num_prefills > 0 and rejects decode-only steps.
    if num_prefills <= 0:
        raise PrefillMetadataError(
            "prefill metadata bridge requires num_prefills > 0; "
            "decode-only steps are rejected"
        )
    # PREFILL_BATCH == 1: the program token tensor holds one sequence.
    if num_prefills != 1:
        raise PrefillMetadataError(
            f"prefill first version supports a single prefill request "
            f"(PREFILL_BATCH={PREFILL_BATCH}), got num_prefills={num_prefills}"
        )
    # Mixed prefill+decode steps (chunked-prefill mixed batches included)
    # are not supported by the single-sequence ABI.
    if num_decodes != 0:
        raise PrefillMetadataError(
            f"mixed prefill+decode steps are unsupported; require "
            f"num_decodes == 0, got {num_decodes}"
        )
    if valid_tokens <= 0:
        raise PrefillMetadataError(
            f"prefill requires positive actual tokens, got {valid_tokens}"
        )
    if valid_tokens > prefill_t:
        raise PrefillMetadataError(
            f"prefill T={valid_tokens} exceeds compiled "
            f"PREFILL_T={prefill_t}"
        )
    if reported_requests is not None and reported_requests < num_prefills:
        raise PrefillMetadataError(
            f"metadata reports num_reqs={reported_requests} "
            f"< num_prefills={num_prefills}"
        )

    # Per-layer consistency + attention state + PCP metadata.
    for index, metadata in metadata_by_layer.items():
        layer_prefills = _int_attr(
            metadata,
            ("num_prefills",),
            name=f"layer{index}.num_prefills",
            default=num_prefills,
        )
        layer_decodes = _int_attr(
            metadata,
            ("num_decodes",),
            name=f"layer{index}.num_decodes",
            default=num_decodes,
        )
        layer_tokens = _int_attr(
            metadata,
            ("num_actual_tokens",),
            name=f"layer{index}.num_actual_tokens",
            default=valid_tokens,
        )
        if (
            layer_prefills != num_prefills
            or layer_decodes != num_decodes
            or layer_tokens != valid_tokens
        ):
            raise PrefillMetadataError(
                f"layer{index}: inconsistent prefill counts"
            )
        if _enum_name(getattr(metadata, "attn_state", None)) not in (
            "",
            "prefillonly",
            "prefill",
        ):
            raise PrefillMetadataError(
                f"layer{index}: unsupported attention state "
                f"{metadata.attn_state!r}"
            )
        if getattr(metadata, "prefill_context_parallel_metadata", None) is not None:
            raise PrefillMetadataError(f"layer{index}: PCP metadata is unsupported")

    query_lengths = _prefill_query_lengths(
        representative, valid_tokens, num_prefills
    )
    if len(query_lengths) != num_prefills or any(
        length <= 0 for length in query_lengths
    ):
        raise PrefillMetadataError(
            f"invalid prefill query lengths: {query_lengths}"
        )
    if sum(query_lengths) != valid_tokens:
        raise PrefillMetadataError(
            f"query lengths sum to {sum(query_lengths)}, "
            f"num_actual_tokens={valid_tokens}"
        )
    if max(query_lengths) > prefill_t:
        raise PrefillMetadataError(
            f"prefill query T={max(query_lengths)} exceeds compiled "
            f"PREFILL_T={prefill_t}"
        )
    query_t = int(query_lengths[0])

    raw_seq = _attr(
        representative,
        ("seq_lens_cpu", "_seq_lens_cpu", "seq_lens"),
        name="seq_lens",
    )
    seq = _i32(raw_seq, name="seq_lens").flatten()
    if seq.numel() < num_prefills:
        raise PrefillMetadataError(
            "seq_lens is shorter than prefill request count"
        )
    final_seq_lens = seq[:num_prefills].to(torch.int32)
    if torch.any(final_seq_lens <= 0):
        raise PrefillMetadataError("valid seq_lens must be positive")
    seq_len = int(final_seq_lens[0])
    if seq_len < query_t:
        raise PrefillMetadataError(
            f"seq_len={seq_len} < query T={query_t}"
        )

    raw_positions = positions
    if raw_positions is None:
        raw_positions = getattr(representative, "positions", None)
    if raw_positions is None:
        raise PrefillMetadataError(
            "prefill positions are required for the per-token RoPE/cache row"
        )
    positions_flat = _i32(raw_positions, name="positions").flatten()
    if positions_flat.numel() < valid_tokens:
        raise PrefillMetadataError(
            "positions is shorter than actual token count"
        )
    positions_flat = positions_flat[:valid_tokens].to(torch.int32)
    # The query must be a contiguous block ending at seq_len - 1
    # (positions[t] = seq_len - T + t).  A fresh prompt has seq_len == T so
    # positions == arange(T).
    expected_positions = torch.arange(
        seq_len - query_t, seq_len, dtype=torch.int32
    )
    if not torch.equal(positions_flat, expected_positions):
        raise PrefillMetadataError(
            f"prefill positions must be contiguous [seq_len-T, seq_len) = "
            f"[{seq_len - query_t}, {seq_len}); got {positions_flat.tolist()}"
        )

    if vllm_config is None:
        vllm_config = getattr(forward_context, "vllm_config", None)
    reserve = _resolve_padding_reserve(
        forward_context,
        padding_reserve=padding_reserve,
    )
    if not reserve.padding_block_ids:
        raise PrefillMetadataError(
            "padding reserve has no allocator-owned block for prefill padding"
        )

    group_specs = _group_layer_indices(vllm_config, metadata_by_layer)
    # First version requires a single KV group (--disable-hybrid-kv-cache-
    # manager).  The hybrid allocator interleaves full/SWA groups, which the
    # single-sequence prefill program does not yet partition.
    if len(group_specs) != 1:
        raise PrefillMetadataError(
            f"prefill first version requires a single KV group "
            f"(--disable-hybrid-kv-cache-manager), got {len(group_specs)}"
        )
    layer_to_group = [-1] * _NUM_LAYERS
    for group_id, indices in group_specs:
        if not indices:
            continue
        for index in indices:
            if layer_to_group[index] != -1:
                raise PrefillMetadataError(
                    f"layer {index} belongs to multiple KV groups"
                )
            layer_to_group[index] = group_id
    if any(group_id < 0 for group_id in layer_to_group):
        raise PrefillMetadataError(
            "KV groups do not cover all 45 decoder layers"
        )

    # All groups must agree on each request's sequence length.
    for index, metadata in metadata_by_layer.items():
        layer_seq = _i32(
            _attr(
                metadata,
                ("seq_lens_cpu", "_seq_lens_cpu", "seq_lens"),
                name=f"layer{index}.seq_lens",
            ),
            name=f"layer{index}.seq_lens",
        ).flatten()[:num_prefills]
        if not torch.equal(layer_seq, final_seq_lens):
            raise PrefillMetadataError(
                f"layer{index}: seq_lens disagree across KV groups"
            )

    groups: list[PyPtoPrefillKvGroupMeta] = []
    for group_id, indices in group_specs:
        if not indices:
            continue
        block_flat, slot_out = _extract_prefill_group_metadata(
            metadata_by_layer,
            indices,
            num_prefills=num_prefills,
            valid_tokens=valid_tokens,
            seq_len=seq_len,
            positions=positions_flat,
            reserve=reserve,
            prefill_t=prefill_t,
        )
        groups.append(PyPtoPrefillKvGroupMeta(group_id, indices, block_flat, slot_out))

    pos_out = torch.zeros(prefill_t, dtype=torch.int32)
    pos_out[:valid_tokens] = positions_flat

    return PrefillPlan(
        valid_tokens=valid_tokens,
        prefill_t=prefill_t,
        seq_lens=final_seq_lens.clone().contiguous(),
        positions=pos_out,
        groups=tuple(sorted(groups, key=lambda group: group.group_id)),
        layer_to_group=tuple(layer_to_group),
        query_lengths=tuple(query_lengths),
        padding_reserve=reserve,
    )


def _fixture_context(
    group_count: int = 1,
    t: int = PREFILL_T,
    seq_len: int | None = None,
):
    """Build a fake single-prefill ForwardContext for the selftest.

    Mirrors ``vllm_decode_metadata._fixture_context``: inner fake classes for
    Meta / Group / Kvc / Parallel / Sched / Cfg, one metadata object shared
    by all 45 layers, and a configured number of KV groups.
    """
    if seq_len is None:
        seq_len = t
    positions = list(range(seq_len - t, seq_len))
    max_blocks = 4
    block_row = torch.arange(max_blocks, dtype=torch.int32)
    slot_mapping = torch.tensor(
        [
            int(block_row[pos // BLOCK_SIZE]) * BLOCK_SIZE + pos % BLOCK_SIZE
            for pos in positions
        ],
        dtype=torch.int32,
    )

    # Scalar attributes are class defaults (literals only — a class body
    # cannot reference enclosing function locals); tensor attributes are
    # set on the instance below, mirroring ``vllm_decode_metadata._fixture``.
    class Meta:
        num_actual_tokens = t
        num_reqs = 1
        num_prefills = 1
        num_decodes = 0
        num_decode_tokens = 0
        actual_seq_lengths_q = [t]
        attn_state = "PrefillOnly"

    meta = Meta()
    meta.query_start_loc = torch.tensor([0, t], dtype=torch.int32)
    meta.seq_lens = torch.tensor([seq_len], dtype=torch.int32)
    meta.positions = torch.tensor(positions, dtype=torch.int32)
    meta.block_tables = (
        block_row.unsqueeze(0).expand(1, max_blocks).contiguous()
    )
    meta.slot_mapping = slot_mapping

    class Group:
        def __init__(self, layer_names):
            self.layer_names = layer_names

    class Kvc:
        kv_cache_groups = []

    class Parallel:
        pipeline_parallel_size = 1
        prefill_context_parallel_size = 1
        decode_context_parallel_size = 1

    class Sched:
        chunked_prefill_enabled = False

    class Cfg:
        kv_cache_config = Kvc()
        parallel_config = Parallel()
        scheduler_config = Sched()
        speculative_config = None

    for group_id in range(group_count):
        indices = tuple(range(group_id, _NUM_LAYERS, group_count))
        Cfg.kv_cache_config.kv_cache_groups.append(
            Group(
                [
                    f"model.layers.{index}.self_attn.attn"
                    for index in indices
                ]
            )
        )
    context = type("Context", (), {})()
    context.attn_metadata = {
        f"model.layers.{index}.self_attn.attn": meta
        for index in range(_NUM_LAYERS)
    }
    context.vllm_config = Cfg()
    context.pypto_padding_reserve = make_padding_reserve(64, 79)
    return context


def _selftest() -> int:
    ok = True
    plan = extract_pypto_prefill_meta(_fixture_context())
    ok &= plan.valid_tokens == PREFILL_T and plan.prefill_t == PREFILL_T
    ok &= plan.valid_requests == 1 and len(plan.groups) == 1
    ok &= torch.equal(
        plan.positions[: plan.valid_tokens],
        torch.arange(PREFILL_T, dtype=torch.int32),
    )
    ok &= plan.seq_lens.tolist() == [PREFILL_T]
    ok &= torch.equal(
        plan.groups[0].slot_mapping[: plan.valid_tokens],
        torch.arange(PREFILL_T, dtype=torch.int32),
    )
    print(
        f"[selftest] single-group T={PREFILL_T} extraction -> "
        f"{'PASS' if ok else 'FAIL'}",
        flush=True,
    )

    multi_block = extract_pypto_prefill_meta(
        _fixture_context(t=64, seq_len=256)
    )
    mb_ok = (
        multi_block.valid_tokens == 64
        and multi_block.seq_lens.tolist() == [256]
        and torch.equal(
            multi_block.positions[:64],
            torch.arange(192, 256, dtype=torch.int32),
        )
        and int(multi_block.groups[0].block_table[1]) == 1
    )
    ok &= mb_ok
    print(
        f"[selftest] multi-block seq_len=256 T=64 -> "
        f"{'PASS' if mb_ok else 'FAIL'}",
        flush=True,
    )

    decode_only = _fixture_context()
    for item in set(decode_only.attn_metadata.values()):
        item.num_prefills = 0
        item.num_decodes = 2
        item.num_actual_tokens = 2
        item.num_decode_tokens = 2
        item.attn_state = "DecodeOnly"
    try:
        extract_pypto_prefill_meta(decode_only)
    except PrefillMetadataError:
        print("[selftest] reject decode-only -> PASS", flush=True)
    else:
        print("[selftest] reject decode-only -> FAIL", flush=True)
        ok = False

    multi_group = _fixture_context(group_count=2)
    try:
        extract_pypto_prefill_meta(multi_group)
    except PrefillMetadataError:
        print("[selftest] reject multi KV group -> PASS", flush=True)
    else:
        print("[selftest] reject multi KV group -> FAIL", flush=True)
        ok = False

    oversize = _fixture_context(t=PREFILL_T + 1, seq_len=PREFILL_T + 1)
    for item in set(oversize.attn_metadata.values()):
        item.num_actual_tokens = PREFILL_T + 1
        item.query_start_loc = torch.tensor(
            [0, PREFILL_T + 1], dtype=torch.int32
        )
        item.actual_seq_lengths_q = [PREFILL_T + 1]
    try:
        extract_pypto_prefill_meta(oversize)
    except PrefillMetadataError:
        print("[selftest] reject T > PREFILL_T -> PASS", flush=True)
    else:
        print("[selftest] reject T > PREFILL_T -> FAIL", flush=True)
        ok = False

    profile = _fixture_context()
    profile.in_profile_run = True
    try:
        extract_pypto_prefill_meta(profile)
    except PrefillMetadataError:
        print("[selftest] reject profile run -> PASS", flush=True)
    else:
        print("[selftest] reject profile run -> FAIL", flush=True)
        ok = False

    print(
        f"[selftest] RESULT="
        f"{'PREFILL_METADATA_BRIDGE_OK' if ok else 'FAIL'}",
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
