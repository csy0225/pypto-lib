# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Extract one selected Step3p5 MTP layer's vLLM attention metadata.

The target-model metadata bridge covers layers 0..44 and rejects speculative
decode by design.  MTP is itself the speculative model, so it has a separate
extractor keyed by the exact draft attention names:

``model.layers.45.mtp_block.self_attn.attn`` through layer 47.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from tools.step3p5.kv_padding import (
    PaddingReserve,
    PaddingReserveError,
    pad_fixed_batch_metadata,
    STORAGE_BATCH,
)

# Compiled physical capacity; runtime valid tokens are supplied per invocation.
_BATCH = STORAGE_BATCH
_MTP_START = 45
_MTP_LAYERS = 3


class MtpMetadataError(ValueError):
    """The current draft forward cannot satisfy the selected-layer ABI."""


def mtp_attention_key(layer_idx: int) -> str:
    if int(layer_idx) not in range(_MTP_LAYERS):
        raise MtpMetadataError(f"layer_idx must be 0..{_MTP_LAYERS - 1}")
    return (
        f"model.layers.{_MTP_START + int(layer_idx)}."
        "mtp_block.self_attn.attn"
    )


def _tensor(value: Any, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise MtpMetadataError(
            f"{name}: expected torch.Tensor, got {type(value).__name__}"
        )
    return value.detach().to("cpu").contiguous()


def _i32(value: Any, *, name: str) -> torch.Tensor:
    result = _tensor(value, name=name)
    if result.dtype not in (torch.int32, torch.int64, torch.long):
        raise MtpMetadataError(f"{name}: expected integer dtype, got {result.dtype}")
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
        raise MtpMetadataError(f"{name}: missing any of {tuple(names)}")
    return None


def _integer(
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
        raise MtpMetadataError(f"{name}: expected integer, got {value!r}") from exc


@dataclass(frozen=True)
class PyPtoMtpLayerMeta:
    layer_idx: int
    valid_tokens: int
    seq_lens: torch.Tensor
    positions: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    padding_reserve: PaddingReserve

    def protocol_meta(self) -> dict[str, Any]:
        return {
            "protocol_version": 2,
            "op": "mtp_layer",
            "layer_idx": self.layer_idx,
            "valid_tokens": self.valid_tokens,
            "valid_requests": self.valid_tokens,
            "storage_batch": _BATCH,
            "padding_reserve": self.padding_reserve.as_dict(),
        }

    def protocol_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "meta_seq_lens": self.seq_lens,
            "meta_positions": self.positions,
            "meta_block_table": self.block_table,
            "meta_slot_mapping": self.slot_mapping,
        }


def extract_pypto_mtp_layer_meta(
    forward_context: Any,
    *,
    layer_idx: int,
    positions: torch.Tensor,
    valid_tokens: int,
    padding_reserve: PaddingReserve | Mapping[str, Any] | None = None,
) -> PyPtoMtpLayerMeta:
    """Extract one pure-decode MTP layer and initialize protocol padding."""
    layer_idx = int(layer_idx)
    if layer_idx not in range(_MTP_LAYERS):
        raise MtpMetadataError(f"layer_idx must be 0..{_MTP_LAYERS - 1}")
    if not 1 <= int(valid_tokens) <= _BATCH:
        raise MtpMetadataError(f"MTP valid_tokens must be 1..{_BATCH}")
    if padding_reserve is None:
        try:
            from tools.step3p5.vllm_kvpool_backend import (  # noqa: PLC0415
                get_padding_reserve,
            )

            padding_reserve = get_padding_reserve("mtp")
        except Exception as exc:  # noqa: BLE001
            raise MtpMetadataError(
                "MTP allocator-owned padding reserve is unavailable; "
                "PYPTO_KVPOOL must publish the MTP reserve before metadata "
                "extraction"
            ) from exc
    if isinstance(padding_reserve, PaddingReserve):
        reserve = padding_reserve
    else:
        try:
            from tools.step3p5.kv_padding import parse_padding_reserve

            reserve = parse_padding_reserve(
                padding_reserve,
                where="MTP padding reserve",
            )
        except PaddingReserveError as exc:
            raise MtpMetadataError(str(exc)) from exc

    attn_metadata = getattr(forward_context, "attn_metadata", None)
    if not isinstance(attn_metadata, Mapping) or not attn_metadata:
        raise MtpMetadataError("forward context has no MTP attention metadata")
    key = mtp_attention_key(layer_idx)
    metadata = attn_metadata.get(key)
    if metadata is None:
        # vLLM-Ascend may assign the same object to every draft layer.  Do not
        # guess a shortened key; only accept an unambiguous exact-layer suffix.
        matches = [
            value
            for name, value in attn_metadata.items()
            if str(name).endswith(
                f"layers.{_MTP_START + layer_idx}.mtp_block.self_attn.attn"
            )
        ]
        if len(matches) != 1:
            raise MtpMetadataError(f"MTP attention metadata missing exact key {key}")
        metadata = matches[0]

    num_prefills = _integer(
        metadata, ("num_prefills",), name="num_prefills", default=0
    )
    if num_prefills:
        raise MtpMetadataError("MTP selected-layer sidecar supports decode only")
    num_actual = _integer(
        metadata,
        ("num_actual_tokens", "num_decode_tokens"),
        name="num_actual_tokens",
        default=valid_tokens,
    )
    if num_actual is not None and num_actual < valid_tokens:
        raise MtpMetadataError(
            f"MTP metadata has {num_actual} actual tokens, need {valid_tokens}"
        )
    decode_per_req = _integer(
        metadata,
        ("decode_token_per_req",),
        name="decode_token_per_req",
        default=1,
    )
    if decode_per_req != 1:
        raise MtpMetadataError("MTP selected-layer ABI requires one token per request")

    raw_seq = _attr(
        metadata,
        ("seq_lens_cpu", "_seq_lens_cpu", "seq_lens"),
        name="seq_lens",
    )
    seq = _i32(raw_seq, name="seq_lens").flatten()
    pos = _i32(positions, name="positions").flatten()
    if seq.numel() < valid_tokens or pos.numel() < valid_tokens:
        raise MtpMetadataError("MTP seq_lens/positions are shorter than hidden rows")
    seq_valid = seq[:valid_tokens]
    pos_valid = pos[:valid_tokens]
    if torch.any(seq_valid <= 0):
        raise MtpMetadataError("MTP valid seq_lens must be positive")
    if not torch.equal(pos_valid, seq_valid - 1):
        raise MtpMetadataError("MTP positions must equal seq_lens-1")

    raw_block = _attr(
        metadata,
        ("block_tables", "block_table_tensor", "block_table"),
        name="block_table",
    )
    block = _i32(raw_block, name="block_table")
    if block.ndim != 2 or block.shape[0] < valid_tokens or block.shape[1] <= 0:
        raise MtpMetadataError(
            f"MTP block_table must cover active rows, got {tuple(block.shape)}"
        )
    raw_slot = _attr(metadata, ("slot_mapping",), name="slot_mapping")
    slots = _i32(raw_slot, name="slot_mapping").flatten()
    if slots.numel() < valid_tokens:
        raise MtpMetadataError("MTP slot_mapping is shorter than active rows")

    seq_out = torch.ones(_BATCH, dtype=torch.int32)
    pos_out = torch.zeros(_BATCH, dtype=torch.int32)
    seq_out[:valid_tokens] = seq_valid
    pos_out[:valid_tokens] = pos_valid
    try:
        block_out, slot_out = pad_fixed_batch_metadata(
            seq_lens=seq_valid,
            block_table=block[:valid_tokens],
            slot_mapping=slots[:valid_tokens],
            valid_rows=valid_tokens,
            reserve=reserve,
            storage_batch=_BATCH,
            where="MTP metadata",
        )
    except PaddingReserveError as exc:
        raise MtpMetadataError(str(exc)) from exc

    return PyPtoMtpLayerMeta(
        layer_idx=layer_idx,
        valid_tokens=valid_tokens,
        seq_lens=seq_out,
        positions=pos_out,
        block_table=block_out,
        slot_mapping=slot_out,
        padding_reserve=reserve,
    )


__all__ = [
    "MtpMetadataError",
    "PyPtoMtpLayerMeta",
    "extract_pypto_mtp_layer_meta",
    "mtp_attention_key",
]
