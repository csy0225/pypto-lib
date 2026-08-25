# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Card-free byte-layout tests for native routed-expert FRACTAL_NZ weights."""

from __future__ import annotations

import torch

from models.step3p5.weight_loader import (
    KEY_MOE_W13_R,
    KEY_MOE_W_DOWN_R,
    build_compact_shape_table,
    expected_shapes,
    pack_int8_fractal_nz,
    unpack_int8_fractal_nz,
)


def test_pack_matches_fractal_nz_address_equation() -> None:
    logical = torch.arange(2 * 32 * 64, dtype=torch.int64)
    logical = logical.remainder(255).sub(127).to(torch.int8).reshape(2, 32, 64)
    packed = pack_int8_fractal_nz(logical)

    assert tuple(packed.shape) == (2, 2, 2, 16, 32)
    for expert in range(2):
        for n1 in range(2):
            for m1 in range(2):
                for m0 in range(16):
                    for n0 in range(32):
                        assert packed[expert, n1, m1, m0, n0] == logical[
                            expert,
                            m1 * 16 + m0,
                            n1 * 32 + n0,
                        ]


def test_pack_round_trip_is_byte_exact() -> None:
    generator = torch.Generator().manual_seed(20260817)
    logical = torch.randint(
        -128,
        128,
        (3, 2, 64, 128),
        dtype=torch.int8,
        generator=generator,
    )
    packed = pack_int8_fractal_nz(logical)
    restored = unpack_int8_fractal_nz(packed, tuple(logical.shape))
    assert torch.equal(restored, logical)


def test_native_production_shapes_are_physical_nz() -> None:
    shapes = expected_shapes(8, int8_routed=True)
    assert shapes[KEY_MOE_W13_R] == (42, 36, 80, 256, 16, 32)
    assert shapes[KEY_MOE_W_DOWN_R] == (42, 36, 128, 80, 16, 32)


def test_reference_and_native_shape_contracts_do_not_alias() -> None:
    reference = build_compact_shape_table(8, int8_routed=False)
    native = build_compact_shape_table(8, int8_routed=True)

    assert reference[KEY_MOE_W13_R] == (42, 2, 256, 128)
    assert reference[KEY_MOE_W_DOWN_R] == (42, 2, 64, 256)
    assert native[KEY_MOE_W13_R] == (42, 2, 4, 16, 16, 32)
    assert native[KEY_MOE_W_DOWN_R] == (42, 2, 8, 4, 16, 32)
