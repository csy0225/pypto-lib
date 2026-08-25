# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

import pytest
import torch

from models.step3p5.config import (
    HIDDEN,
    MOE_NUM_EXPERTS,
    SHARE_EXPERT_DIM,
    TP_WORLD_SIZE,
)
from models.step3p5.weight_loader import (
    KEY_MOE_GATE_W,
    KEY_MOE_GATE_W_NK,
    KEY_MOE_ROUTER_BIAS,
    KEY_MOE_W_GATE_S,
    KEY_MOE_W_GATE_S_NK,
    KEY_MOE_W_UP_S,
    KEY_MOE_W_UP_S_NK,
    NUM_MOE_LAYERS,
    _slice_mlp_col,
    _slice_mlp_col_native,
    build_compact_shape_table,
    build_synthetic_bundle,
    expected_shapes,
    load_step3p5_weights_for_rank,
)


_LEGACY_MOE_KEYS = {
    KEY_MOE_GATE_W,
    KEY_MOE_W_GATE_S,
    KEY_MOE_W_UP_S,
}
_NATIVE_MOE_KEYS = {
    KEY_MOE_GATE_W_NK,
    KEY_MOE_W_GATE_S_NK,
    KEY_MOE_W_UP_S_NK,
}


def test_decode_native_moe_contract_replaces_legacy_layouts() -> None:
    legacy = expected_shapes()
    native = expected_shapes(decode_native_moe=True)

    assert _LEGACY_MOE_KEYS <= legacy.keys()
    assert not (_NATIVE_MOE_KEYS & legacy.keys())
    assert _NATIVE_MOE_KEYS <= native.keys()
    assert not (_LEGACY_MOE_KEYS & native.keys())
    assert native.keys() == (
        legacy.keys() - _LEGACY_MOE_KEYS | _NATIVE_MOE_KEYS
    )

    shared_local = SHARE_EXPERT_DIM // TP_WORLD_SIZE
    assert native[KEY_MOE_GATE_W_NK] == (
        NUM_MOE_LAYERS,
        MOE_NUM_EXPERTS,
        HIDDEN,
    )
    assert native[KEY_MOE_W_GATE_S_NK] == (
        NUM_MOE_LAYERS,
        shared_local,
        HIDDEN,
    )
    assert native[KEY_MOE_W_UP_S_NK] == (
        NUM_MOE_LAYERS,
        shared_local,
        HIDDEN,
    )


def test_default_compact_contract_remains_legacy() -> None:
    legacy = build_compact_shape_table()
    native = build_compact_shape_table(decode_native_moe=True)

    assert _LEGACY_MOE_KEYS <= legacy.keys()
    assert not (_NATIVE_MOE_KEYS & legacy.keys())
    assert _NATIVE_MOE_KEYS <= native.keys()
    assert not (_LEGACY_MOE_KEYS & native.keys())


def test_native_moe_synthetic_dtypes_and_no_duplicate_layouts() -> None:
    shapes = {
        KEY_MOE_GATE_W_NK: (2, 3, 4),
        KEY_MOE_ROUTER_BIAS: (2, 3),
        KEY_MOE_W_GATE_S_NK: (2, 5, 4),
        KEY_MOE_W_UP_S_NK: (2, 5, 4),
    }
    bundle = build_synthetic_bundle(
        rank=0,
        shape_overrides=shapes,
        decode_native_moe=True,
    )

    assert set(bundle) == set(shapes)
    assert not (_LEGACY_MOE_KEYS & bundle.keys())
    assert bundle[KEY_MOE_GATE_W_NK].dtype == torch.float32
    assert bundle[KEY_MOE_ROUTER_BIAS].dtype == torch.float32
    assert bundle[KEY_MOE_W_GATE_S_NK].dtype == torch.bfloat16
    assert bundle[KEY_MOE_W_UP_S_NK].dtype == torch.bfloat16
    assert all(tensor.is_contiguous() for tensor in bundle.values())


def test_native_shared_slice_preserves_checkpoint_nk_layout() -> None:
    weight = torch.arange(48, dtype=torch.float32).reshape(12, 4)

    native = _slice_mlp_col_native(weight, rank=1, dim_local=3)
    legacy = _slice_mlp_col(weight, rank=1, dim_local=3)
    expected = weight[3:6].to(torch.bfloat16)

    torch.testing.assert_close(native, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(legacy, expected.T, rtol=0.0, atol=0.0)
    assert native.is_contiguous()
    assert legacy.is_contiguous()


@pytest.mark.parametrize("tp_world_size", [1, 4, 16])
def test_weight_loader_rejects_noncanonical_local_owner_world_size(
    tp_world_size: int,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "canonical Step3p5 deployment requires "
            "co-located TP=EP=8"
        ),
    ):
        load_step3p5_weights_for_rank(
            "/checkpoint-is-not-read",
            rank=0,
            tp_world_size=tp_world_size,
        )


@pytest.mark.parametrize("rank", [-1, TP_WORLD_SIZE])
def test_weight_loader_rejects_rank_outside_canonical_world(rank: int) -> None:
    with pytest.raises(
        ValueError,
        match=rf"rank {rank} out of range \[0, {TP_WORLD_SIZE}\)",
    ):
        load_step3p5_weights_for_rank(
            "/checkpoint-is-not-read",
            rank=rank,
            tp_world_size=TP_WORLD_SIZE,
        )
