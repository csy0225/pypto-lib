# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Pure-decode metadata bridge from vLLM-Ascend to the PyPTO ABI.

The vLLM internals intentionally do not cross the sidecar boundary.  This
module accepts a ``ForwardContext``-like object and emits plain tensors plus a
stable layer-to-KV-group map.

The target vLLM-Ascend version builds one ``AscendMetadata`` object per KV
group and assigns it to every layer in that group.  A Step3p5 hybrid model can
therefore have more than two groups (the default hybrid allocator commonly
interleaves the full/SWA pattern).  The extractor never infers a group from
``full`` versus ``SWA``; it uses ``vllm_config.kv_cache_config.kv_cache_groups``
when available and falls back to metadata object identity only for offline
fixtures.

Only decode is accepted.  A normal step has one token per request.  During
vLLM speculative verification a request may contain target + draft tokens;
the bridge decomposes that flattened query into ordered one-token rounds:

```
round 0: every request's first target token
round 1: every request that has a second token
...
```

Each round is one complete 45-layer PyPTO invocation with at most 16 rows.
Earlier rounds therefore publish their per-layer KV before a later speculative
position consumes it.  vLLM still owns the final target logits and
acceptance/rejection.

Requirements:

* 1 <= active requests <= configured storage capacity;
* every query length is in ``[1, num_speculative_tokens + 1]``;
* no prefill, chunked-prefill, CP/PCP, or PP handoff;
* every layer has complete group metadata;
* sequence lengths and positions agree across all groups.

Padding is initialized here, not by callers:

* hidden padding is handled by the holder;
* seq_lens padding = 1 and positions padding = 0;
* block-table/slot padding use the allocator-owned 15-block reserve.
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
    STORAGE_BATCH,
    pad_fixed_batch_metadata,
)

_LAYER_RE = re.compile(r"(?:^|[.]layers[.])(\d+)(?:[.]|$)")
_MAX_BATCH = STORAGE_BATCH
_NUM_LAYERS = 45


class DecodeMetadataError(ValueError):
    """Raised when the current vLLM step is not supported by the PyPTO ABI."""


def _tensor(value: Any, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise DecodeMetadataError(f"{name}: expected torch.Tensor, got {type(value).__name__}")
    return value.detach().to("cpu").contiguous()


def _i32(value: Any, *, name: str) -> torch.Tensor:
    result = _tensor(value, name=name)
    if result.dtype not in (torch.int32, torch.int64, torch.long):
        raise DecodeMetadataError(f"{name}: expected integer dtype, got {result.dtype}")
    return result.to(torch.int32)


def _attr(obj: Any, names: Sequence[str], *, name: str, required: bool = True) -> Any:
    for candidate in names:
        value = getattr(obj, candidate, None)
        if value is not None:
            return value
    if required:
        raise DecodeMetadataError(f"{name}: missing any of {tuple(names)}")
    return None


def _int_attr(obj: Any, names: Sequence[str], *, name: str, default: int | None = None) -> int | None:
    value = _attr(obj, names, name=name, required=default is None)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise DecodeMetadataError(f"{name}: expected integer, got {value!r}") from exc


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


def _query_lengths(metadata: Any, valid_tokens: int, valid_requests: int) -> list[int]:
    starts = _attr(metadata, ("query_start_loc_cpu", "query_start_loc"), name="query_start_loc", required=False)
    if starts is not None:
        starts_cpu = _i32(starts, name="query_start_loc").flatten()
        if starts_cpu.numel() >= valid_requests + 1:
            starts_cpu = starts_cpu[: valid_requests + 1]
            lengths = (starts_cpu[1:] - starts_cpu[:-1]).tolist()
            if sum(lengths) == valid_tokens:
                return [int(x) for x in lengths]

    actual = getattr(metadata, "actual_seq_lengths_q", None)
    if isinstance(actual, (list, tuple)) and len(actual) >= valid_requests:
        # vLLM-Ascend stores query_start_loc[1:] in this field.
        ends = [int(x) for x in actual[:valid_requests]]
        starts_list = [0] + ends[:-1]
        lengths = [end - start for start, end in zip(starts_list, ends)]
        if sum(lengths) == valid_tokens:
            return lengths

    # A decode-only fixture may not expose query_start_loc.  num_decodes and
    # num_decode_tokens are sufficient to prove the normal one-token path.
    num_decode_tokens = _int_attr(
        metadata, ("num_decode_tokens",), name="num_decode_tokens", default=None
    )
    num_decodes = _int_attr(metadata, ("num_decodes",), name="num_decodes", default=None)
    if num_decode_tokens == valid_tokens and num_decodes == valid_requests:
        return [1] * valid_requests
    raise DecodeMetadataError("cannot derive per-request query lengths")


def _group_layer_indices(vllm_config: Any, layer_names: Mapping[int, Any]) -> list[tuple[int, tuple[int, ...]]]:
    groups = getattr(getattr(vllm_config, "kv_cache_config", None), "kv_cache_groups", None)
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
        raise DecodeMetadataError(
            f"kv_cache_config groups cover {len(seen)}/{_NUM_LAYERS} decoder layers"
        )

    # Offline fallback: the target builder assigns one metadata object to all
    # layers in a group.  This path is intentionally not used for production
    # vLLM; it keeps the bridge unit-testable without constructing VllmConfig.
    by_identity: dict[int, list[int]] = {}
    for index, metadata in layer_names.items():
        by_identity.setdefault(id(metadata), []).append(index)
    return [
        (group_id, tuple(sorted(indices)))
        for group_id, indices in enumerate(by_identity.values())
    ]


@dataclass(frozen=True)
class PyPtoKvGroupMeta:
    group_id: int
    layer_indices: tuple[int, ...]
    block_table: torch.Tensor  # [storage_batch, max_blocks]
    slot_mapping: torch.Tensor  # [storage_batch]


@dataclass(frozen=True)
class PyPtoDecodeMeta:
    valid_tokens: int
    storage_batch: int
    seq_lens: torch.Tensor  # [storage_batch], INT32
    positions: torch.Tensor  # [storage_batch], INT32
    groups: tuple[PyPtoKvGroupMeta, ...]
    layer_to_group: tuple[int, ...]
    query_lengths: tuple[int, ...]
    padding_reserve: PaddingReserve
    token_indices: tuple[int, ...] = ()

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
            "op": "decode",
            "valid_tokens": self.valid_tokens,
            "valid_requests": self.valid_requests,
            "storage_batch": self.storage_batch,
            "storage_capacity": self.storage_batch,
            "kv_group_count": len(self.groups),
            "layer_to_group": list(self.layer_to_group),
            "query_lengths": list(self.query_lengths),
            "padding_reserve": self.padding_reserve.as_dict(),
        }


@dataclass(frozen=True)
class PyPtoDecodePlan:
    """One vLLM target forward decomposed into ordered PyPTO decode rounds."""

    valid_tokens: int
    valid_requests: int
    query_lengths: tuple[int, ...]
    steps: tuple[PyPtoDecodeMeta, ...]
    padding_reserve: PaddingReserve


def _extract_group_metadata(metadata_by_layer: Mapping[int, Any], indices: Iterable[int], *,
                            valid_requests: int, storage_batch: int,
                            reserve: PaddingReserve) -> tuple[torch.Tensor, torch.Tensor]:
    representatives = [metadata_by_layer[index] for index in indices if index in metadata_by_layer]
    if not representatives:
        raise DecodeMetadataError("KV group has no layer metadata")
    metadata = representatives[0]
    raw_block = _attr(metadata, ("block_tables", "block_table_tensor", "block_table"), name="block_table")
    block = _i32(raw_block, name="block_table")
    if block.ndim != 2 or block.shape[0] < valid_requests:
        raise DecodeMetadataError(f"block_table: expected [requests, blocks], got {tuple(block.shape)}")
    raw_slot = _attr(metadata, ("slot_mapping",), name="slot_mapping")
    slot = _i32(raw_slot, name="slot_mapping").flatten()
    if slot.numel() < valid_requests:
        raise DecodeMetadataError(f"slot_mapping has {slot.numel()} entries, need {valid_requests}")
    # For pure decode, query order and token order coincide.  Active rows are
    # preserved byte-for-byte; only fixed-batch padding rows are synthesized
    # from the allocator-owned reserve.
    try:
        return pad_fixed_batch_metadata(
            seq_lens=_i32(
                _attr(
                    metadata,
                    ("seq_lens_cpu", "_seq_lens_cpu", "seq_lens"),
                    name="seq_lens",
                ),
                name="seq_lens",
            ).flatten()[:valid_requests],
            block_table=block[:valid_requests],
            slot_mapping=slot[:valid_requests],
            valid_rows=valid_requests,
            reserve=reserve,
            storage_batch=storage_batch,
            where="decode metadata",
        )
    except PaddingReserveError as exc:
        raise DecodeMetadataError(str(exc)) from exc


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
            raise DecodeMetadataError(
                "Main allocator-owned padding reserve is unavailable; "
                "PYPTO_KVPOOL must publish scheduler/physical reserve before "
                "metadata extraction"
            ) from exc
    if isinstance(padding_reserve, PaddingReserve):
        return padding_reserve
    try:
        from tools.step3p5.kv_padding import parse_padding_reserve

        return parse_padding_reserve(
            padding_reserve,
            where="Main padding reserve",
        )
    except PaddingReserveError as exc:
        raise DecodeMetadataError(str(exc)) from exc


def _extract_group_step_metadata(
    metadata_by_layer: Mapping[int, Any],
    indices: Iterable[int],
    *,
    request_indices: Sequence[int],
    token_indices: Sequence[int],
    seq_lens: torch.Tensor,
    reserve: PaddingReserve,
    storage_batch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    representatives = [
        metadata_by_layer[index]
        for index in indices
        if index in metadata_by_layer
    ]
    if not representatives:
        raise DecodeMetadataError("KV group has no layer metadata")
    metadata = representatives[0]
    raw_block = _attr(
        metadata,
        ("block_tables", "block_table_tensor", "block_table"),
        name="block_table",
    )
    block = _i32(raw_block, name="block_table")
    if (
        block.ndim != 2
        or not request_indices
        or block.shape[0] <= max(request_indices)
    ):
        raise DecodeMetadataError(
            "block_table does not cover selected requests: "
            f"shape={tuple(block.shape)}, requests={tuple(request_indices)}"
        )
    raw_slot = _attr(metadata, ("slot_mapping",), name="slot_mapping")
    slot = _i32(raw_slot, name="slot_mapping").flatten()
    if not token_indices or slot.numel() <= max(token_indices):
        raise DecodeMetadataError(
            "slot_mapping does not cover selected tokens: "
            f"size={slot.numel()}, tokens={tuple(token_indices)}"
        )
    request_index = torch.tensor(request_indices, dtype=torch.long)
    token_index = torch.tensor(token_indices, dtype=torch.long)
    try:
        return pad_fixed_batch_metadata(
            seq_lens=seq_lens,
            block_table=block.index_select(0, request_index),
            slot_mapping=slot.index_select(0, token_index),
            valid_rows=len(request_indices),
            reserve=reserve,
            storage_batch=storage_batch,
            where="decode metadata step",
        )
    except PaddingReserveError as exc:
        raise DecodeMetadataError(str(exc)) from exc


def extract_pypto_decode_plan(
    forward_context: Any,
    *,
    vllm_config: Any = None,
    positions: Any = None,
    max_batch: int = _MAX_BATCH,
    padding_reserve: PaddingReserve | Mapping[str, Any] | None = None,
) -> PyPtoDecodePlan:
    """Build ordered rounds for one vLLM target forward.

    ``max_batch`` is the compile-time static storage capacity, not the
    runtime active batch.  The latter is derived from each input metadata and
    must be no greater than this capacity.
    """
    max_batch = int(max_batch)
    if max_batch <= 0:
        raise DecodeMetadataError(
            f"storage capacity must be positive, got {max_batch}"
        )
    if bool(getattr(forward_context, "in_profile_run", False)):
        raise DecodeMetadataError("profile/dummy runs are not supported")
    if vllm_config is not None:
        parallel = getattr(vllm_config, "parallel_config", None)
        if int(getattr(parallel, "pipeline_parallel_size", 1)) != 1:
            raise DecodeMetadataError("pipeline parallelism is unsupported")
        if int(getattr(parallel, "prefill_context_parallel_size", 1)) != 1:
            raise DecodeMetadataError("prefill context parallelism is unsupported")
        if int(getattr(parallel, "decode_context_parallel_size", 1)) != 1:
            raise DecodeMetadataError("decode context parallelism is unsupported")
    attn_metadata = getattr(forward_context, "attn_metadata", None)
    if not isinstance(attn_metadata, Mapping) or not attn_metadata:
        raise DecodeMetadataError("forward context has no layer attention metadata")

    metadata_by_layer: dict[int, Any] = {}
    for layer_name, metadata in attn_metadata.items():
        index = _layer_index(str(layer_name))
        if index is None or index >= _NUM_LAYERS:
            continue
        metadata_by_layer[index] = metadata
    if set(metadata_by_layer) != set(range(_NUM_LAYERS)):
        missing = sorted(set(range(_NUM_LAYERS)) - set(metadata_by_layer))
        raise DecodeMetadataError(f"attention metadata missing decoder layers: {missing[:8]}")

    representative = metadata_by_layer[0]
    valid_tokens = _int_attr(
        representative,
        ("num_actual_tokens", "num_decode_tokens"),
        name="num_actual_tokens",
    )
    valid_requests = _int_attr(
        representative,
        ("num_decodes",),
        name="num_decodes",
    )
    reported_requests_raw = getattr(representative, "num_reqs", None)
    try:
        reported_requests = (
            None if reported_requests_raw is None else int(reported_requests_raw)
        )
    except (TypeError, ValueError) as exc:
        raise DecodeMetadataError(
            f"num_reqs: expected integer, got {reported_requests_raw!r}"
        ) from exc
    assert valid_tokens is not None
    assert valid_requests is not None
    if valid_tokens <= 0:
        raise DecodeMetadataError(
            f"decode requires positive actual tokens, got {valid_tokens}"
        )
    if not (1 <= valid_requests <= max_batch):
        raise DecodeMetadataError(
            f"decode requires 1..{max_batch} active requests for the "
            f"configured storage capacity, got {valid_requests}"
        )
    if reported_requests is not None and reported_requests < valid_requests:
        raise DecodeMetadataError(
            f"metadata reports num_reqs={reported_requests} "
            f"< active requests={valid_requests}"
        )

    for index, metadata in metadata_by_layer.items():
        num_prefills = _int_attr(
            metadata,
            ("num_prefills",),
            name=f"layer{index}.num_prefills",
            default=0,
        )
        num_decode_tokens = _int_attr(
            metadata,
            ("num_decode_tokens",),
            name=f"layer{index}.num_decode_tokens",
            default=valid_tokens,
        )
        layer_requests = _int_attr(
            metadata,
            ("num_decodes",),
            name=f"layer{index}.num_decodes",
            default=valid_requests,
        )
        if (
            num_prefills != 0
            or num_decode_tokens != valid_tokens
            or layer_requests != valid_requests
        ):
            raise DecodeMetadataError(f"layer{index}: inconsistent decode counts")
        if _enum_name(getattr(metadata, "attn_state", None)) not in (
            "",
            "decodeonly",
            "specdecoding",
        ):
            raise DecodeMetadataError(
                f"layer{index}: unsupported attention state "
                f"{metadata.attn_state!r}"
            )
        if getattr(metadata, "prefill_context_parallel_metadata", None) is not None:
            raise DecodeMetadataError(f"layer{index}: PCP metadata is unsupported")

    query_lengths = _query_lengths(representative, valid_tokens, valid_requests)
    if len(query_lengths) != valid_requests or any(
        length <= 0 for length in query_lengths
    ):
        raise DecodeMetadataError(
            f"invalid decode query lengths: {query_lengths}"
        )
    speculative_config = (
        getattr(vllm_config, "speculative_config", None)
        if vllm_config is not None
        else None
    )
    max_query_len = 1
    if speculative_config is not None:
        max_query_len += int(
            getattr(speculative_config, "num_speculative_tokens", 0)
        )
    if max(query_lengths) > max_query_len:
        raise DecodeMetadataError(
            "decode query exceeds configured target verification width: "
            f"query_lengths={query_lengths}, max={max_query_len}"
        )
    if sum(query_lengths) != valid_tokens:
        raise DecodeMetadataError(
            f"query lengths sum to {sum(query_lengths)}, "
            f"num_actual_tokens={valid_tokens}"
        )

    raw_seq = _attr(
        representative,
        ("seq_lens_cpu", "_seq_lens_cpu", "seq_lens"),
        name="seq_lens",
    )
    seq = _i32(raw_seq, name="seq_lens").flatten()
    if seq.numel() < valid_requests:
        raise DecodeMetadataError("seq_lens is shorter than valid request count")
    final_seq_lens = seq[:valid_requests]
    if torch.any(final_seq_lens <= 0):
        raise DecodeMetadataError("valid seq_lens must be positive")

    raw_positions = positions
    if raw_positions is None:
        raw_positions = getattr(representative, "positions", None)
    if raw_positions is None:
        raise DecodeMetadataError(
            "decode positions are required for ordered target verification"
        )
    else:
        positions_flat = _i32(raw_positions, name="positions").flatten()
        if positions_flat.numel() < valid_tokens:
            raise DecodeMetadataError(
                "positions is shorter than actual token count"
            )
        positions_flat = positions_flat[:valid_tokens]

    if vllm_config is None:
        vllm_config = getattr(forward_context, "vllm_config", None)
    reserve = _resolve_padding_reserve(
        forward_context,
        padding_reserve=padding_reserve,
    )
    if reserve.storage_capacity != max_batch:
        raise DecodeMetadataError(
            f"padding reserve capacity={reserve.storage_capacity} does not "
            f"match compiled storage capacity={max_batch}"
        )
    group_specs = _group_layer_indices(vllm_config, metadata_by_layer)
    layer_to_group = [-1] * _NUM_LAYERS
    for group_id, indices in group_specs:
        if not indices:
            continue
        for index in indices:
            if layer_to_group[index] != -1:
                raise DecodeMetadataError(f"layer {index} belongs to multiple KV groups")
            layer_to_group[index] = group_id
    if any(group_id < 0 for group_id in layer_to_group):
        raise DecodeMetadataError("KV groups do not cover all 45 decoder layers")

    # All groups must agree on each request's final sequence length.
    for index, metadata in metadata_by_layer.items():
        layer_seq = _i32(
            _attr(
                metadata,
                ("seq_lens_cpu", "_seq_lens_cpu", "seq_lens"),
                name=f"layer{index}.seq_lens",
            ),
            name=f"layer{index}.seq_lens",
        ).flatten()[:valid_requests]
        if not torch.equal(layer_seq, final_seq_lens):
            raise DecodeMetadataError(f"layer{index}: seq_lens disagree across KV groups")

    query_starts = [0]
    for length in query_lengths:
        query_starts.append(query_starts[-1] + int(length))

    steps: list[PyPtoDecodeMeta] = []
    for round_idx in range(max(query_lengths)):
        request_indices = [
            request_idx
            for request_idx, length in enumerate(query_lengths)
            if round_idx < length
        ]
        token_indices = [
            query_starts[request_idx] + round_idx
            for request_idx in request_indices
        ]
        step_seq = torch.tensor(
            [
                int(final_seq_lens[request_idx])
                - int(query_lengths[request_idx])
                + round_idx
                + 1
                for request_idx in request_indices
            ],
            dtype=torch.int32,
        )
        if torch.any(step_seq <= 0):
            raise DecodeMetadataError(
                f"round {round_idx}: derived non-positive sequence length"
            )
        token_index = torch.tensor(token_indices, dtype=torch.long)
        step_positions = positions_flat.index_select(0, token_index)
        if not torch.equal(step_positions, step_seq - 1):
            raise DecodeMetadataError(
                f"round {round_idx}: positions do not equal derived seq_lens-1"
            )
        seq_out = torch.ones(max_batch, dtype=torch.int32)
        pos_out = torch.zeros(max_batch, dtype=torch.int32)
        seq_out[: len(request_indices)] = step_seq
        pos_out[: len(request_indices)] = step_positions

        groups: list[PyPtoKvGroupMeta] = []
        for group_id, indices in group_specs:
            if not indices:
                continue
            block, slot = _extract_group_step_metadata(
                metadata_by_layer,
                indices,
                request_indices=request_indices,
                token_indices=token_indices,
                seq_lens=step_seq,
                reserve=reserve,
                storage_batch=max_batch,
            )
            groups.append(
                PyPtoKvGroupMeta(group_id, indices, block, slot)
            )
        steps.append(
            PyPtoDecodeMeta(
                valid_tokens=len(request_indices),
                storage_batch=max_batch,
                seq_lens=seq_out,
                positions=pos_out,
                groups=tuple(
                    sorted(groups, key=lambda group: group.group_id)
                ),
                layer_to_group=tuple(layer_to_group),
                query_lengths=(1,) * len(request_indices),
                padding_reserve=reserve,
                token_indices=tuple(token_indices),
            )
        )

    return PyPtoDecodePlan(
        valid_tokens=valid_tokens,
        valid_requests=valid_requests,
        query_lengths=tuple(query_lengths),
        steps=tuple(steps),
        padding_reserve=reserve,
    )


def extract_pypto_decode_meta(
    forward_context: Any,
    *,
    vllm_config: Any = None,
    positions: Any = None,
    max_batch: int = _MAX_BATCH,
    padding_reserve: PaddingReserve | Mapping[str, Any] | None = None,
) -> PyPtoDecodeMeta:
    """Backward-compatible one-round pure-decode extractor."""
    plan = extract_pypto_decode_plan(
        forward_context,
        vllm_config=vllm_config,
        positions=positions,
        max_batch=max_batch,
        padding_reserve=padding_reserve,
    )
    if len(plan.steps) != 1:
        raise DecodeMetadataError(
            "multi-token target verification requires "
            "extract_pypto_decode_plan()"
        )
    return plan.steps[0]


def _fixture_context(group_count: int = 4, valid: int = 2):
    class Group:
        def __init__(self, layer_names):
            self.layer_names = layer_names

    class Cfg:
        class Kvc:
            kv_cache_groups = []

        kv_cache_config = Kvc()

    class Meta:
        num_actual_tokens = valid
        num_reqs = valid
        num_decode_tokens = valid
        num_prefills = 0
        num_decodes = valid
        decode_token_per_req = 1
        num_spec_decodes = 0
        attn_state = "DecodeOnly"
        seq_lens = torch.tensor([3, 5, 0], dtype=torch.int32)
        positions = torch.tensor([2, 4, 0], dtype=torch.int32)
        query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
        block_tables = torch.arange(valid * 4, dtype=torch.int32).reshape(valid, 4)
        slot_mapping = torch.tensor([10, 20], dtype=torch.int32)

    layer_meta = {index: Meta() for index in range(_NUM_LAYERS)}
    for group_id in range(group_count):
        indices = tuple(range(group_id, _NUM_LAYERS, group_count))
        group = Group([f"model.layers.{index}.self_attn.attn" for index in indices])
        Cfg.kv_cache_config.kv_cache_groups.append(group)
    # Make metadata identity match the configured groups.
    for group_id, group in enumerate(Cfg.kv_cache_config.kv_cache_groups):
        meta = Meta()
        for layer_name in group.layer_names:
            layer_meta[_layer_index(layer_name)] = meta
    context = type("Context", (), {})()
    context.attn_metadata = {
        f"model.layers.{index}.self_attn.attn": layer_meta[index]
        for index in range(_NUM_LAYERS)
    }
    context.vllm_config = Cfg()
    return context


def _selftest() -> int:
    ok = True
    meta = extract_pypto_decode_meta(_fixture_context())
    ok &= meta.valid_tokens == 2 and meta.storage_batch == 16
    ok &= len(meta.groups) == 4 and all(group.block_table.shape == (16, 4) for group in meta.groups)
    ok &= tuple(meta.seq_lens[:3].tolist()) == (3, 5, 1)
    ok &= tuple(meta.positions[:3].tolist()) == (2, 4, 0)
    print(f"[selftest] 4-group T=2 extraction -> {'PASS' if ok else 'FAIL'}", flush=True)

    bad = _fixture_context()
    bad.attn_metadata["model.layers.0.self_attn.attn"].num_prefills = 1
    try:
        extract_pypto_decode_meta(bad)
    except DecodeMetadataError:
        print("[selftest] reject prefill -> PASS", flush=True)
    else:
        print("[selftest] reject prefill -> FAIL", flush=True)
        ok = False

    padded = _fixture_context(valid=2)
    for item in set(padded.attn_metadata.values()):
        item.num_reqs = 16
        item.seq_lens = torch.tensor([3, 5] + [1] * 14, dtype=torch.int32)
    meta = extract_pypto_decode_meta(
        padded,
        positions=torch.tensor([2, 4] + [0] * 14, dtype=torch.int32),
    )
    padded_ok = meta.valid_requests == 2 and tuple(meta.seq_lens[:3].tolist()) == (3, 5, 1)
    ok &= padded_ok
    print(
        f"[selftest] padded storage num_reqs=16, valid=2 -> "
        f"{'PASS' if padded_ok else 'FAIL'}",
        flush=True,
    )

    profile = _fixture_context()
    profile.in_profile_run = True
    try:
        extract_pypto_decode_meta(profile)
    except DecodeMetadataError:
        print("[selftest] reject profile run -> PASS", flush=True)
    else:
        print("[selftest] reject profile run -> FAIL", flush=True)
        ok = False
    print(
        f"[selftest] RESULT={'DECODE_METADATA_BRIDGE_OK' if ok else 'FAIL'}",
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
