# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Shared protocol contract for Step3p5 five-layer MoE goldens."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any


GOLDEN_SCHEMA = "step3p5.five-layer-moe-golden.v3"

BASELINE_SOURCE_KIND = "baseline"
LOCAL_EP_SOURCE_KIND = "local-ep"

LEGACY_PROTOCOL_PROFILE = "legacy_distributed_ep"
LOCAL_OWNER_PROTOCOL_PROFILE = "replicated_input_local_owner"

NUMERIC_COMPARISON = "bit_exact_to_protocol_golden"
LEGACY_NUMERIC_CONTRACT_NAME = "legacy_baseline_bit_exact_v1"
LOCAL_OWNER_NUMERIC_CONTRACT_NAME = (
    "local_owner_partial_tp_all_reduce_bf16_v1"
)
CANONICAL_HIDDEN_ONLY_MOE_PROTOCOL_PROFILE = LOCAL_OWNER_PROTOCOL_PROFILE
SOURCE_PROTOCOL_BINDING_SCHEMA = (
    "step3p5.canonical-hidden-only-moe-source-protocol.v1"
)

_PROTOCOL_CONTRACTS = {
    LEGACY_PROTOCOL_PROFILE: {
        "source_kind": BASELINE_SOURCE_KIND,
        "numeric_contract": {
            "name": LEGACY_NUMERIC_CONTRACT_NAME,
            "comparison": NUMERIC_COMPARISON,
            "bit_exact": True,
        },
    },
    LOCAL_OWNER_PROTOCOL_PROFILE: {
        "source_kind": LOCAL_EP_SOURCE_KIND,
        "numeric_contract": {
            "name": LOCAL_OWNER_NUMERIC_CONTRACT_NAME,
            "comparison": NUMERIC_COMPARISON,
            "bit_exact": True,
        },
    },
}


def golden_protocol_fields(protocol_profile: str) -> dict[str, Any]:
    """Return a fresh manifest fragment for a supported protocol profile."""
    expected = _PROTOCOL_CONTRACTS.get(protocol_profile)
    if expected is None:
        raise ValueError(
            f"unsupported golden protocol_profile={protocol_profile!r}"
        )
    return {
        "source_kind": expected["source_kind"],
        "protocol_profile": protocol_profile,
        "numeric_contract": dict(expected["numeric_contract"]),
        "bit_exact": True,
    }


def canonical_hidden_only_moe_protocol_fields() -> dict[str, Any]:
    """Return the protocol implemented by canonical hidden-only decode MoE."""
    return golden_protocol_fields(
        CANONICAL_HIDDEN_ONLY_MOE_PROTOCOL_PROFILE
    )


def source_protocol_binding_fields(
    *,
    source_manifest_sha256: str,
    decode_fwd_sha256: str,
    moe_protocol_contract_sha256: str,
    protocol_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a canonical, hash-addressed source-to-protocol binding."""
    hashes = {
        "source_manifest_sha256": source_manifest_sha256,
        "decode_fwd_sha256": decode_fwd_sha256,
        "moe_protocol_contract_sha256": moe_protocol_contract_sha256,
    }
    for name, value in hashes.items():
        if not isinstance(value, str) or re.fullmatch(
            r"[0-9a-f]{64}",
            value,
        ) is None:
            raise ValueError(f"{name} must be a lowercase SHA256")
    normalized = normalize_golden_protocol(
        protocol_contract,
        field="source protocol binding contract",
    )
    binding = {
        "schema": SOURCE_PROTOCOL_BINDING_SCHEMA,
        **hashes,
        "protocol_contract": normalized,
    }
    digest = hashlib.sha256(
        json.dumps(
            binding,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    return {
        "source_protocol_binding": binding,
        "source_protocol_binding_sha256": digest,
    }


def normalize_golden_protocol(
    manifest: Mapping[str, Any],
    *,
    field: str = "golden",
) -> dict[str, Any]:
    """Validate and normalize legacy or explicit protocol-golden metadata."""
    if manifest.get("bit_exact") is not True:
        raise ValueError(f"{field}.bit_exact must be true")

    source_kind = manifest.get("source_kind")
    has_protocol = "protocol_profile" in manifest
    has_numeric = "numeric_contract" in manifest
    if has_protocol != has_numeric:
        raise ValueError(
            f"{field} must declare protocol_profile and numeric_contract "
            "together"
        )

    if not has_protocol:
        if source_kind != BASELINE_SOURCE_KIND:
            raise ValueError(
                f"{field} without protocol metadata must be a legacy baseline"
            )
        return golden_protocol_fields(LEGACY_PROTOCOL_PROFILE)

    protocol_profile = manifest.get("protocol_profile")
    if not isinstance(protocol_profile, str):
        raise ValueError(f"{field}.protocol_profile must be a string")
    expected = golden_protocol_fields(protocol_profile)
    if source_kind != expected["source_kind"]:
        raise ValueError(
            f"{field}.source_kind={source_kind!r} is invalid for "
            f"protocol_profile={protocol_profile!r}"
        )

    numeric_contract = manifest.get("numeric_contract")
    if not isinstance(numeric_contract, Mapping):
        raise ValueError(f"{field}.numeric_contract must be a mapping")
    expected_numeric = expected["numeric_contract"]
    if set(numeric_contract) != set(expected_numeric):
        raise ValueError(
            f"{field}.numeric_contract fields must be "
            f"{sorted(expected_numeric)}"
        )
    for name, value in expected_numeric.items():
        actual = numeric_contract.get(name)
        matches = actual is True if value is True else actual == value
        if not matches:
            raise ValueError(
                f"{field}.numeric_contract.{name} must be {value!r}"
            )
    return expected


def normalize_source_protocol_binding(
    value: Mapping[str, Any],
    *,
    field: str = "source protocol binding",
) -> dict[str, Any]:
    """Strictly validate a persisted source-to-protocol binding."""
    binding = value.get("source_protocol_binding")
    digest = value.get("source_protocol_binding_sha256")
    if not isinstance(binding, Mapping):
        raise ValueError(f"{field}.source_protocol_binding must be a mapping")
    expected_keys = {
        "schema",
        "source_manifest_sha256",
        "decode_fwd_sha256",
        "moe_protocol_contract_sha256",
        "protocol_contract",
    }
    if set(binding) != expected_keys:
        raise ValueError(
            f"{field}.source_protocol_binding fields must be "
            f"{sorted(expected_keys)}"
        )
    if binding.get("schema") != SOURCE_PROTOCOL_BINDING_SCHEMA:
        raise ValueError(f"{field}.source_protocol_binding schema is invalid")
    expected = source_protocol_binding_fields(
        source_manifest_sha256=str(
            binding.get("source_manifest_sha256", "")
        ),
        decode_fwd_sha256=str(binding.get("decode_fwd_sha256", "")),
        moe_protocol_contract_sha256=str(
            binding.get("moe_protocol_contract_sha256", "")
        ),
        protocol_contract=(
            binding["protocol_contract"]
            if isinstance(binding.get("protocol_contract"), Mapping)
            else {}
        ),
    )
    if digest != expected["source_protocol_binding_sha256"]:
        raise ValueError(f"{field}.source_protocol_binding_sha256 mismatch")
    return expected


def normalize_golden_source_binding(
    manifest: Mapping[str, Any],
    *,
    field: str = "golden",
) -> dict[str, Any]:
    """Bind golden protocol metadata to its immutable producer source."""
    protocol = normalize_golden_protocol(manifest, field=field)
    has_binding = "source_protocol_binding" in manifest
    has_binding_sha = "source_protocol_binding_sha256" in manifest
    if has_binding != has_binding_sha:
        raise ValueError(
            f"{field} source protocol binding fields must appear together"
        )
    if (
        protocol["protocol_profile"] == LOCAL_OWNER_PROTOCOL_PROFILE
        and not has_binding
    ):
        raise ValueError(
            f"{field} local-owner golden is missing source protocol binding"
        )
    if not has_binding:
        return {
            "protocol_contract": protocol,
            "source_protocol_binding": None,
            "source_protocol_binding_sha256": None,
        }
    normalized = normalize_source_protocol_binding(
        manifest,
        field=field,
    )
    binding = normalized["source_protocol_binding"]
    if (
        binding["source_manifest_sha256"]
        != manifest.get("source_manifest_sha256")
        or binding["decode_fwd_sha256"]
        != manifest.get("source_decode_fwd_sha256")
        or binding["protocol_contract"] != protocol
    ):
        raise ValueError(f"{field} source protocol binding mismatch")
    return {
        "protocol_contract": protocol,
        **normalized,
    }
