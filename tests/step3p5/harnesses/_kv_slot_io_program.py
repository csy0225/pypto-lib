# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Small device-side scatter/gather helper for resident KV-cache audits.

The public runtime copy API intentionally validates device-allocation
provenance. A KV row is an interior view of a multi-gigabyte resident cache, so
the focused attention harness uses this normal PyPTO program instead of
bypassing the runtime guard with raw pointer arithmetic.
"""
from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld

from models.step3p5 import config as cfg

TP_SIZE = cfg.TP_WORLD_SIZE
KV_CACHE_ROWS = cfg.KV_CACHE_ROWS_DYN
HEAD_DIM = cfg.HEAD_DIM

# The release audit uses at most seven active rows. For every target row it
# gathers the target plus one canary on each side, for both Full and SWA slabs.
KV_AUDIT_MAX_ACTIVE_ROWS = 7
KV_SLOT_IO_CAPACITY = 2 * 3 * KV_AUDIT_MAX_ACTIVE_ROWS


@pl.program
class KVSlotIO:
    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch(
        self,
        k_cache: pl.InOut[pl.Tensor[[KV_CACHE_ROWS, HEAD_DIM], pl.BF16]],
        v_cache: pl.InOut[pl.Tensor[[KV_CACHE_ROWS, HEAD_DIM], pl.BF16]],
        cache_rows: pl.Tensor[[KV_SLOT_IO_CAPACITY], pl.INT32],
        write_mask: pl.Tensor[[KV_SLOT_IO_CAPACITY], pl.INT32],
        k_payload: pl.Tensor[[KV_SLOT_IO_CAPACITY, HEAD_DIM], pl.BF16],
        v_payload: pl.Tensor[[KV_SLOT_IO_CAPACITY, HEAD_DIM], pl.BF16],
        mode: pl.Tensor[[1], pl.INT32],
        slot_count: pl.Tensor[[1], pl.INT32],
        k_readback: pl.Out[
            pl.Tensor[[KV_SLOT_IO_CAPACITY, HEAD_DIM], pl.BF16]
        ],
        v_readback: pl.Out[
            pl.Tensor[[KV_SLOT_IO_CAPACITY, HEAD_DIM], pl.BF16]
        ],
    ):
        active_slots = pl.cast(pl.read(slot_count, [0]), pl.INDEX)
        write_mode = pl.read(mode, [0])
        for slot in pl.spmd(
            KV_SLOT_IO_CAPACITY,
            name_hint="kv_slot_audit_io",
        ):
            if slot < active_slots:
                cache_row = pl.cast(pl.read(cache_rows, [slot]), pl.INDEX)
                if write_mode != 0 and pl.read(write_mask, [slot]) != 0:
                    k_tile = pl.slice(k_payload, [1, HEAD_DIM], [slot, 0])
                    v_tile = pl.slice(v_payload, [1, HEAD_DIM], [slot, 0])
                    k_cache = pl.assemble(k_cache, k_tile, [cache_row, 0])
                    v_cache = pl.assemble(v_cache, v_tile, [cache_row, 0])
                    k_readback = pl.assemble(
                        k_readback,
                        k_tile,
                        [slot, 0],
                    )
                    v_readback = pl.assemble(
                        v_readback,
                        v_tile,
                        [slot, 0],
                    )
                else:
                    k_readback = pl.assemble(
                        k_readback,
                        pl.slice(k_cache, [1, HEAD_DIM], [cache_row, 0]),
                        [slot, 0],
                    )
                    v_readback = pl.assemble(
                        v_readback,
                        pl.slice(v_cache, [1, HEAD_DIM], [cache_row, 0]),
                        [slot, 0],
                    )
        return k_cache, v_cache, k_readback, v_readback

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        k_cache: pl.InOut[
            pl.Tensor[[TP_SIZE, KV_CACHE_ROWS, HEAD_DIM], pl.BF16]
        ],
        v_cache: pl.InOut[
            pl.Tensor[[TP_SIZE, KV_CACHE_ROWS, HEAD_DIM], pl.BF16]
        ],
        cache_rows: pl.Tensor[[TP_SIZE, KV_SLOT_IO_CAPACITY], pl.INT32],
        write_mask: pl.Tensor[[TP_SIZE, KV_SLOT_IO_CAPACITY], pl.INT32],
        k_payload: pl.Tensor[
            [TP_SIZE, KV_SLOT_IO_CAPACITY, HEAD_DIM],
            pl.BF16,
        ],
        v_payload: pl.Tensor[
            [TP_SIZE, KV_SLOT_IO_CAPACITY, HEAD_DIM],
            pl.BF16,
        ],
        mode: pl.Tensor[[TP_SIZE, 1], pl.INT32],
        slot_count: pl.Tensor[[TP_SIZE, 1], pl.INT32],
        k_readback: pl.Out[
            pl.Tensor[[TP_SIZE, KV_SLOT_IO_CAPACITY, HEAD_DIM], pl.BF16]
        ],
        v_readback: pl.Out[
            pl.Tensor[[TP_SIZE, KV_SLOT_IO_CAPACITY, HEAD_DIM], pl.BF16]
        ],
    ):
        for rank in pl.range(pld.world_size()):
            self.chip_orch(
                k_cache[rank],
                v_cache[rank],
                cache_rows[rank],
                write_mask[rank],
                k_payload[rank],
                v_payload[rank],
                mode[rank],
                slot_count[rank],
                k_readback[rank],
                v_readback[rank],
                device=rank,
            )


kv_slot_io = KVSlotIO
