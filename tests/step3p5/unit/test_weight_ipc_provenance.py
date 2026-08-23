# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

import torch

from pypto.runtime.device_tensor import DeviceTensor
from tools.step3p5.pypto_weight_ipc import WeightIpcMap


@dataclass(frozen=True)
class _FakeBuffer:
    base: int

    def tensor(self, shapes, dtype):
        return shapes, dtype


class _FakeRuntime:
    def __init__(self) -> None:
        self._device_buffers: dict[tuple[int, int], _FakeBuffer] = {}
        self.imported: list[tuple[int, tuple[int, ...], torch.dtype, int]] = []

    def imported_tensor(
        self,
        ptr: int,
        shape,
        dtype: torch.dtype,
        *,
        worker_id: int = 0,
    ) -> DeviceTensor:
        shape = tuple(shape)
        buffer = _FakeBuffer(ptr)
        self._device_buffers[(worker_id, ptr)] = buffer
        self.imported.append((ptr, shape, dtype, worker_id))
        return DeviceTensor(ptr, shape, dtype, buffer=buffer)


def test_device_tensor_slice_remints_registered_owner_buffer() -> None:
    runtime = _FakeRuntime()
    peer_base = 0x100000
    key_offset = 0x200
    pool_map = {
        "map": {
            "weight": {
                "offset": key_offset,
                "shape": [4, 3, 2],
                "dtype": "bfloat16",
                "nbytes": 48,
            }
        }
    }
    weight_map = WeightIpcMap(
        peer_base, pool_map, runtime=runtime, worker_id=7
    )

    sliced = weight_map.device_tensor_slice("weight", 1, 3)

    expected_ptr = peer_base + key_offset + 3 * 2 * 2
    assert sliced.data_ptr == expected_ptr
    assert sliced.shape == (2, 3, 2)
    assert sliced.dtype is torch.bfloat16
    assert sliced.buffer is runtime._device_buffers[(7, expected_ptr)]
    assert runtime.imported[-1] == (
        expected_ptr, (2, 3, 2), torch.bfloat16, 7
    )
