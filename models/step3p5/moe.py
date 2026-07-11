# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""[中文摘要] `EpTpMoE` 类:把 gate → dispatch → expert_routed → expert_shared(+末尾
tp_all_reduce)→ combine 这 6 个子阶段串成一个独立的多卡 @pl.program;按
(routed_swiglu_limit, shared_swiglu_limit) 在工厂函数里编译期烤出 3 个特化
(SiLU/SiLU、SwiGLU7/SiLU、SwiGLU7/SwiGLU16),`select_moe_block(layer_idx)`
按层选一个。
[关键装饰器] @pl.program +
   @pl.function(level=HOST, role=Orchestrator)  ← host_orch
   @pl.function(type=Orchestration)             ← chip_orch
   @pl.function(type=InCore)                    ← gate / dispatch / expert_*
   @pl.function(type=Inline)                    ← 小段 helper
另含同名 `self.tp_all_reduce(...)` 方法(collective 在 @pl.program 内的 self-method 复制,见指南 §4.5)。
[SPMD 角色] 跨卡(EP all-to-all + TP all-reduce 共享专家)+ 片上多核 SPMD(每段 stage 用 pl.spmd)。
[详见] 中文架构指南 §3, §4.2, §4.5, §10

────── 以下为英文原 docstring ──────

Step3p5 MoE block orchestration (decode, TP=EP=8 BF16 — single ``@pl.program``).

Wires the 6 sub-stages into one ``@pl.program`` class ``EpTpMoE``:

    gate (local FP32, replicated)
      └─► dispatch (EP all-to-all)
           └─► expert_routed (local, 36 experts/card)
                └─► combine (EP all-to-all back + weighted gather + sh_y add)
    expert_shared (TP-sliced 160-lane FFN + tp_all_reduce) ──┐
                                                              ├─► combine
    shared_y_tp_reduced ─────────────────────────────────────┘

Per-layer activation choices are compile-time baked at factory time
(matches the single-card factory pattern from Phase 4):

  - ``SWIGLU_LIMITS[layer_idx]``: routed-expert activation (0 / 7).
  - ``SWIGLU_LIMITS_SHARED[layer_idx]``: shared-expert activation
    (0 / 16; only layer 44 uses 16).

Three specialisations are pre-built (the only combinations actually
used by step3p5):

  - ``EpTpMoE_silu_silu``         layers 3..42  (silu / silu)
  - ``EpTpMoE_swiglu7_silu``      layer 43      (swiglu7 / silu)
  - ``EpTpMoE_swiglu7_swiglu16``  layer 44      (swiglu7 / swiglu16)

``select_moe_block(layer_idx)`` returns the right specialisation for
the caller (``decode_layer.py`` / prefill caller in Wave 3).

Distributed topology (locked Phase 9 Wave 2):

  * Single-node, 8 cards, ``world_size == TP_WORLD_SIZE == EP_WORLD_SIZE == 8``.
  * TP and EP groups co-located on the same 8 ranks.

Per-card weight bundle (host weight loader contract):

  Replicated on every rank:
    * ``gate_w[HIDDEN, MOE_NUM_EXPERTS=288]`` FP32
    * ``router_bias[MOE_NUM_EXPERTS=288]`` FP32

  EP-sliced (rank ``r`` gets global expert ids
  ``[r * 36 .. (r + 1) * 36)``):
    * ``w_gate_r[MOE_NUM_EXPERTS_LOCAL=36, HIDDEN, MOE_INTERMEDIATE=1280]`` BF16
    * ``w_up_r  [MOE_NUM_EXPERTS_LOCAL=36, HIDDEN, MOE_INTERMEDIATE=1280]`` BF16
    * ``w_down_r[MOE_NUM_EXPERTS_LOCAL=36, MOE_INTERMEDIATE=1280, HIDDEN]`` BF16

  TP-sliced (rank ``r`` gets intermediate lanes
  ``[r * 160 .. (r + 1) * 160)``):
    * ``w_gate_s[HIDDEN, SHARE_EXPERT_DIM_LOCAL=160]`` BF16
    * ``w_up_s  [HIDDEN, SHARE_EXPERT_DIM_LOCAL=160]`` BF16
    * ``w_down_s[SHARE_EXPERT_DIM_LOCAL=160, HIDDEN]`` BF16

Wave-3 wire-in contract:

  ``decode_layer.py`` (Wave 3) calls ``select_moe_block(layer_idx)`` and
  passes the same 9 weight bundles + ``my_rank`` scalar + the per-collective
  windows. The block does not own a residual add; the caller (the
  ``_moe_mlp_body`` in ``decode_layer.py``) adds the post-attention
  residual back to ``moe_out``.

Cross-rank windows allocated by ``host_orch`` (one per call site to keep
the lifetime contract clean):

  * ``pub_counts``         : ``[N_RANKS * N_RANKS, N_LOCAL_EXPERTS]`` INT32
                             (dispatch publishes counts here)
  * ``count_done_sig``     : ``[N_RANKS, 1]`` INT32 (dispatch count barrier)
  * ``recv_x``             : ``[LOCAL_RECV_MAX, HIDDEN]`` BF16
                             (dispatch payload window)
  * ``data_done_sig``      : ``[N_RANKS, 1]`` INT32 (dispatch payload barrier)
  * ``sh_tmp_window``      : ``[T, HIDDEN // TP_WORLD_SIZE]`` BF16
                             (tp_all_reduce scratch for shared expert)
  * ``sh_signal_window``   : ``[N_RANKS, 1]`` INT32 (tp_all_reduce barrier)
  * ``src_route_table``    : ``[N_RANKS, N_LOCAL_EXPERTS, T*TOPK]`` INT32
                             (combine's r_route publication window)
  * ``route_pub_sig``      : ``[N_RANKS, 1]`` INT32 (route-table barrier)
  * ``routed_y_buf``       : ``[T*TOPK, HIDDEN]`` BF16 (combine push window)
  * ``combine_done_sig``   : ``[N_RANKS, 1]`` INT32 (combine push barrier)
"""

# pyright: reportUndefinedVariable=false

from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld
import os as _os
# Build-time flag: EpTpMoE gate_step takes vLLM fused-router topk (injected
# via repurposed gate_w) and SKIPS the on-device gate top-k (TSORT32/TMRGSORT
# hangs on tied logits = PTOAS Blocker 1). Read at import (per fresh process).
_EPMOE_BYPASS_GATE = _os.environ.get("EPMOE_BYPASS_GATE") == "1"

from .config import (
    BATCH,
    EP_WORLD_SIZE,
    HIDDEN,
    MOE_INTERMEDIATE,
    MOE_NUM_EXPERTS,
    MOE_NUM_EXPERTS_LOCAL,
    MOE_ROUTER_SCALING_FACTOR,
    MOE_TOP_K,
    SHARE_EXPERT_DIM_LOCAL,
    SWIGLU_LIMITS,
    SWIGLU_LIMITS_SHARED,
    TP_WORLD_SIZE,
)
from .dispatch import (
    LOCAL_RECV_MAX,
    N_RANKS_PAD,
    PER_RANK_BUCKETS,
)


T = BATCH
N_RANKS = EP_WORLD_SIZE                     # 8
N_LOCAL_EXPERTS = MOE_NUM_EXPERTS_LOCAL      # 36
N_EXPERTS_GLOBAL = MOE_NUM_EXPERTS           # 288
TOPK = MOE_TOP_K                             # 8
N_EXPERTS = N_EXPERTS_GLOBAL                 # convenience alias for sigs
INTER = MOE_INTERMEDIATE                     # 1280
SH_INTER_LOCAL = SHARE_EXPERT_DIM_LOCAL      # 160
N_ROUTES_PER_RANK = T * TOPK                 # 128
SH_TP_CHUNK = HIDDEN // TP_WORLD_SIZE        # tp_all_reduce per-step chunk

# -----------------------------------------------------------------------------
# Kernel-internal constants for the LIFTED sub-bodies (Phase X.3).
#
# pypto frontend rejects naked @pl.jit.inline calls from inside @pl.function
# bodies, so we lift the kernel bodies (gate / dispatch helpers / expert_routed
# / expert_shared / combine helpers) into class methods on the enclosing
# @pl.program. Their tiling / sort widths / activation thresholds live here as
# module-level constants so the lifted method bodies can reference them via
# closure capture without depending on the source-of-truth modules at parse
# time.
# -----------------------------------------------------------------------------

# Router (gate) kernel constants — mirrors gate.py.
ROUTER_SCORE_PAD = 512        # next pow-of-two over N_EXPERTS=288
ROUTER_TOPK_PAD = 16          # 32B-aligned width for the (val, idx) slice
ROUTER_SORT_PAD = ROUTER_TOPK_PAD * 2  # interleaved (value, index) pair width
ROUTER_GATE_K_CHUNK = 256     # K-loop step over HIDDEN for the gate matmul (256: gate_matmul UB fits 188416 for standalone EpTpMoE; 512 overflowed by ~10KB)
ROUTER_GATE_N_CHUNK = 32      # N-chunk for gate matmul; [K=512,N=32] FP32 = 65536 B (L0B)
ROUTER_FP32_NEG_INF = -3.4028235e38
ROUTER_SCALE = MOE_ROUTER_SCALING_FACTOR  # 3.0
assert TOPK <= ROUTER_TOPK_PAD
assert HIDDEN % ROUTER_GATE_K_CHUNK == 0
assert N_EXPERTS % ROUTER_GATE_N_CHUNK == 0

# Routed-expert kernel constants — mirrors expert_routed.py.
ROUTED_GATE_K_CHUNK = 64    # A-tile K dim; [32,64] BF16 = 4096 B (L0A)
ROUTED_GATE_N_CHUNK = 64    # B-tile N dim; [64,64] BF16 = 8192 B (L0B)
ROUTED_DOWN_K_CHUNK = 64    # A-tile K dim; [32,64] BF16 = 4096 B (L0A)
ROUTED_DOWN_N_CHUNK = 128   # B-tile N dim; [64,128] BF16 = 16384 B (L0B)
ROUTED_MAX_TILE = LOCAL_RECV_MAX

# Per-tile row count for the routed-expert compute body.  PTOAS Vec UB on
# A2/A3 caps tiles at 192 KB; allocating [ROUTED_MAX_TILE=1024, INTER=1280]
# FP32 = 5.2 MB blows past that by ~28×.  Row-tiling: RECV_TILE rows per
# inner pass, outer ``for tile_idx in pl.range(N_RECV_TILES)`` loop.
# 32 * 1280 * 4 = 160 KB keeps headroom for intermediates.
RECV_TILE = 32
ROUTED_QUANT_T_TILE = 16    # per-token dyn-quant token-tile (RECV_TILE % == 0)
MOE_IN_QUANT_T_TILE = 16   # per-token input dyn-quant token-tile (T % == 0)
# INT8-native W8A8: per-token activation dequant scale carried alongside the
# INT8 recv_x through dispatch.  Padded to 8 FP32 cols (=32B, the min UB/GM
# tile row) so the scale window's remote_load tiles are 32B-aligned, mirroring
# DeepSeek v4 dispatch W_PAD.  Column 0 holds the scale; columns 1..7 are pad.
SCALE_W_PAD = 8
assert ROUTED_MAX_TILE % RECV_TILE == 0, (
    f"ROUTED_MAX_TILE ({ROUTED_MAX_TILE}) must be divisible by RECV_TILE ({RECV_TILE})"
)
N_RECV_TILES = ROUTED_MAX_TILE // RECV_TILE  # 32 outer iterations
assert HIDDEN % ROUTED_GATE_K_CHUNK == 0
assert HIDDEN % ROUTED_DOWN_N_CHUNK == 0
assert INTER % ROUTED_GATE_N_CHUNK == 0
assert INTER % ROUTED_DOWN_K_CHUNK == 0

# Shared-expert kernel constants — mirrors expert_shared.py.
# INTER=160 (SH_INTER_LOCAL) is below GATE_N_CHUNK=256, so the N-tile equals
# the full per-card intermediate slice and the down-direction K-tile equals
# INTER in a single chunk.
SHARED_GATE_K_CHUNK = 256
SHARED_GATE_N_CHUNK = SH_INTER_LOCAL  # 160 — one N tile covers the slice
SHARED_DOWN_K_CHUNK = SH_INTER_LOCAL  # 160 — one K tile covers the slice
SHARED_DOWN_N_CHUNK = 256
# Narrow swiglu N-chunk for the shared-expert clamp: the full [T,160] Vec-tile
# clamp is miscompiled (wide-tile tmov misprune -> ~45% wrong). Run the clamp
# on [T,32] chunks (mirrors the working routed path).
SHARED_SWIGLU_N_CHUNK = 32
assert SH_INTER_LOCAL == SHARED_SWIGLU_N_CHUNK * 5
assert HIDDEN % SHARED_GATE_K_CHUNK == 0
assert HIDDEN % SHARED_DOWN_N_CHUNK == 0


# =============================================================================
# Program factory
# =============================================================================


def _build_ep_tp_moe_program(
    routed_swiglu_limit: float,
    shared_swiglu_limit: float,
):
    """Build a ``@pl.program`` class with the two activation choices baked in.

    Mirrors the in-tree TP+EP MoE reference's deferred-build pattern; keeps
    the module importable even if the embedded body trips the parser at
    collection time.
    """
    if routed_swiglu_limit == 0.0:
        routed_swiglu_step = False
    elif routed_swiglu_limit == 7.0:
        routed_swiglu_step = True
    else:
        raise ValueError(
            f"routed_swiglu_limit must be 0.0 or 7.0, got "
            f"{routed_swiglu_limit}",
        )

    if shared_swiglu_limit == 0.0:
        shared_swiglu_step = False
    elif shared_swiglu_limit == 16.0:
        shared_swiglu_step = True
    else:
        raise ValueError(
            f"shared_swiglu_limit must be 0.0 or 16.0, got "
            f"{shared_swiglu_limit}",
        )

    # Closure-captured Python constants used by the lifted expert_routed /
    # expert_shared method bodies. The activation choice is baked at factory
    # build time (compile-time constant), not selected at runtime.
    _routed_swiglu_limit = routed_swiglu_limit
    _shared_swiglu_limit = shared_swiglu_limit
    _routed_swiglu_step = routed_swiglu_step
    _shared_swiglu_step = shared_swiglu_step

    @pl.program
    class EpTpMoE:
        # ---------- Collective: TP all_reduce (lifted from collectives.py) ----
        # NOTE: Phase X.2 lifted the pull-side ring all-reduce body from the
        # @pl.jit.inline wrapper in collectives.py into a class method, per
        # the canonical pattern in
        # tests/st/distributed/test_l3_allreduce.py. The body is identical
        # in semantics to collectives.tp_all_reduce; only the call form
        # changes (self.tp_all_reduce(...) inside @pl.function(InCore)).
        @pl.function(type=pl.FunctionType.InCore)
        def tp_all_reduce(
            self,
            local: pl.Tensor[[T, HIDDEN], pl.BF16],
            tmp_window: pld.DistributedTensor[[T, HIDDEN], pl.BF16],
            signal_window: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[T, HIDDEN], pl.BF16]:
            # Barrier-style all-reduce (mirrors decode_layer.py DenseFull +
            # tests/st/distributed/test_l3_allreduce.py, PASS TP=8). Replaces the
            # RING version which deadlocked at TP=8 (lone unconverted survivor of
            # the dense ring->barrier migration; project_barrier_allreduce_fixes_multicard_507018).
            group_size = N_RANKS
            ar_chunk = HIDDEN // 8
            for k0 in pl.range(0, HIDDEN, ar_chunk):
                stage_tile = pl.load(local, [0, k0], [T, ar_chunk])
                pl.store(stage_tile, [0, k0], tmp_window)
            for peer in pl.range(group_size):
                if peer != my_rank:
                    pld.system.notify(
                        target=signal_window, peer=peer,
                        offsets=[my_rank, 0], value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for src in pl.range(group_size):
                if src != my_rank:
                    pld.system.wait(
                        signal=signal_window, offsets=[src, 0],
                        expected=1, cmp=pld.WaitCmp.Ge,
                    )
            for k0 in pl.range(0, HIDDEN, ar_chunk):
                own_tile = pl.load(tmp_window, [0, k0], [T, ar_chunk])
                acc = pl.cast(own_tile, target_type=pl.FP32)
                for peer in pl.range(group_size):
                    if peer != my_rank:
                        recv = pld.tile.remote_load(
                            tmp_window, peer=peer,
                            offsets=[0, k0], shape=[T, ar_chunk],
                        )
                        acc = pl.add(acc, pl.cast(recv, target_type=pl.FP32))
                pl.store(
                    pl.cast(acc, target_type=pl.BF16), [0, k0], local,
                )
            return local

        # ---------- Collective: EP all_to_all (lifted from collectives.py) ----
        @pl.function(type=pl.FunctionType.Inline)
        def ep_all_to_all(
            self,
            send: pld.DistributedTensor[[LOCAL_RECV_MAX, HIDDEN], pl.INT8],
            recv: pld.DistributedTensor[[LOCAL_RECV_MAX, HIDDEN], pl.INT8],
            send_scale: pld.DistributedTensor[
                [LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
            ],
            recv_scale: pld.DistributedTensor[
                [LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
            ],
            send_counts: pl.Tensor[[N_RANKS], pl.INT32],
            recv_counts: pl.Tensor[[N_RANKS], pl.INT32],
            send_offsets: pl.Tensor[[N_RANKS], pl.INT32],
            recv_offsets: pl.Tensor[[N_RANKS], pl.INT32],
            read_offsets: pl.Tensor[[N_RANKS], pl.INT32],
            signal_window: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pld.DistributedTensor[[LOCAL_RECV_MAX, HIDDEN], pl.INT8]:
            """Pull-side variable-length token-level all-to-all over EP.

            Static dims (group_size=EP_WORLD_SIZE, d_cols=HIDDEN) baked
            from module-scope constants.  The per-token FP32 dequant scale
            (``send_scale`` / ``recv_scale``) is fused into the SAME loops
            as the INT8 activation so no extra notify/wait barrier is
            needed for the scale movement.
            """
            group_size = EP_WORLD_SIZE
            d_cols = HIDDEN

            # 1) Local self-bucket copy (symmetric fixed slot my_rank*MAX).
            PER_PEER_BOUND = T * TOPK  # 128 = MAX_PER_RANK per (src,dst) slot
            _self_base = pl.cast(my_rank * PER_PEER_BOUND, pl.INDEX)
            for r in pl.range(PER_PEER_BOUND):
                self_tile = pl.load(
                    send, [_self_base + r, 0], [1, d_cols],
                )
                pl.store(self_tile, [_self_base + r, 0], recv)
                # Fused scale self-copy: same row index, SCALE_W_PAD-wide tile.
                s_self = pl.load(
                    send_scale, [_self_base + r, 0], [1, SCALE_W_PAD],
                )
                pl.store(s_self, [_self_base + r, 0], recv_scale)

            # 2) Set(1) notify every peer.
            for peer in pl.range(group_size):
                if peer != my_rank:
                    pld.system.notify(
                        target=signal_window,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )

            # 3) Ge(1) wait for every peer.
            for src in pl.range(group_size):
                if src != my_rank:
                    pld.system.wait(
                        signal=signal_window,
                        offsets=[src, 0],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )

            # 4) Pull every peer's bucket-for-me (symmetric fixed slots).
            # My block in peer's send_buf is at my_rank*MAX; store into
            # peer's block in my recv at peer*MAX. Read full MAX rows (gap
            # rows past the real count are ignored by the re-pack, which
            # uses the available column-my_rank counts). Indices are
            # compound-scalar (my_rank*MAX) or loop-var (peer*MAX) -> no
            # bare [my_rank] index and no cross-rank offset data.
            _my_base = pl.cast(my_rank * (T * TOPK), pl.INDEX)
            for peer in pl.range(group_size):
                if peer != my_rank:
                    _peer_base = pl.cast(peer * (T * TOPK), pl.INDEX)
                    for r in pl.range(T * TOPK):
                        peer_tile = pld.tile.remote_load(
                            send,
                            peer=peer,
                            offsets=[_my_base + r, 0],
                            shape=[1, d_cols],
                        )
                        pl.store(peer_tile, [_peer_base + r, 0], recv)
                        # Fused scale pull: same peer/row, SCALE_W_PAD-wide tile.
                        s_peer = pld.tile.remote_load(
                            send_scale,
                            peer=peer,
                            offsets=[_my_base + r, 0],
                            shape=[1, SCALE_W_PAD],
                        )
                        pl.store(s_peer, [_peer_base + r, 0], recv_scale)

            return recv

        # ---------- Stage 1: gate (local, replicated) ----------
        #
        # Phase X.3 lift: the gate body (gate.py's @pl.jit.inline) is
        # inlined verbatim here as a class-method @pl.function(Inline) so
        # the pypto frontend can resolve it from inside gate_step. The
        # original @pl.jit.inline ``gate`` in gate.py remains intact for
        # unit-test independence.
        @pl.function(type=pl.FunctionType.Inline)
        def _gate(
            self,
            x: pl.Tensor[[T, HIDDEN], pl.BF16],
            gate_w: pl.Tensor[[HIDDEN, N_EXPERTS], pl.FP32],
            router_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
            expert_indices: pl.Tensor[[T, TOPK], pl.INT32],
            expert_weights: pl.Tensor[[T, TOPK], pl.FP32],
        ):
            score_buf = pl.create_tensor(
                [T, ROUTER_SCORE_PAD], dtype=pl.FP32,
            )
            biased_buf = pl.create_tensor(
                [T, ROUTER_SCORE_PAD], dtype=pl.FP32,
            )

            # Stage 1: gate matmul + sigmoid + bias.
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_matmul"):
                # Initialise output pads once — columns beyond N_EXPERTS
                # keep 0 / NEG_INF so topk is not tricked by uninitialised.
                score_buf[:, :] = pl.full(
                    [T, ROUTER_SCORE_PAD], dtype=pl.FP32, value=0.0,
                )
                biased_buf[:, :] = pl.full(
                    [T, ROUTER_SCORE_PAD],
                    dtype=pl.FP32, value=ROUTER_FP32_NEG_INF,
                )
                for nb in pl.range(N_EXPERTS // ROUTER_GATE_N_CHUNK):
                    n0 = nb * ROUTER_GATE_N_CHUNK
                    # Cast x per K-chunk → [T,K] FP32 = small tile
                    # (avoids full [T,HIDDEN] FP32 Vec buffer overflow).
                    x0 = pl.cast(
                        pl.slice(x, [T, ROUTER_GATE_K_CHUNK], [0, 0]),
                        target_type=pl.FP32,
                    )
                    w0 = pl.slice(
                        gate_w,
                        [ROUTER_GATE_K_CHUNK, ROUTER_GATE_N_CHUNK],
                        [0, n0],
                    )
                    logits_n = pl.matmul(x0, w0, out_dtype=pl.FP32)
                    for kb in pl.range(1, HIDDEN // ROUTER_GATE_K_CHUNK):
                        k0 = kb * ROUTER_GATE_K_CHUNK
                        xk = pl.cast(
                            pl.slice(
                                x, [T, ROUTER_GATE_K_CHUNK], [0, k0],
                            ),
                            target_type=pl.FP32,
                        )
                        wk = pl.slice(
                            gate_w,
                            [ROUTER_GATE_K_CHUNK, ROUTER_GATE_N_CHUNK],
                            [k0, n0],
                        )
                        logits_n = pl.matmul_acc(logits_n, xk, wk)
                    # Apply sigmoid per N-chunk — vec ops convert cube→vec
                    # block layout, avoiding blayout mismatch when storing
                    # into pre-created score_buf / biased_buf (vec layout).
                    score_n_chunk = pl.recip(
                        pl.add(pl.exp(pl.neg(logits_n)), 1.0),
                    )
                    bias_chunk = pl.slice(
                        router_bias, [ROUTER_GATE_N_CHUNK], [n0],
                    )
                    bias_row_chunk = pl.reshape(
                        bias_chunk, [1, ROUTER_GATE_N_CHUNK],
                    )
                    # ROUTER-BIAS-BF16: vLLM runs router_bias in BF16; the loader
                    # gives FP32. BF16-round it (bias~4.79 dominates; its ~0.015
                    # rounding decides the top-8 tail — wrong tail vs vLLM without
                    # this). Cast the [1,32] reshaped chunk (2D-safe).
                    bias_row_chunk = pl.cast(
                        pl.cast(bias_row_chunk, target_type=pl.BF16),
                        target_type=pl.FP32,
                    )
                    biased_n_chunk = pl.add(
                        score_n_chunk,
                        pl.col_expand_mul(
                            pl.full(
                                [T, ROUTER_GATE_N_CHUNK],
                                dtype=pl.FP32, value=1.0,
                            ),
                            bias_row_chunk,
                        ),
                    )
                    score_buf[:, n0 : n0 + ROUTER_GATE_N_CHUNK] = score_n_chunk
                    biased_buf[:, n0 : n0 + ROUTER_GATE_N_CHUNK] = biased_n_chunk

            # Stage 2: per-row top-K via sort32 + mrgsort.
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_topk"):
                topk_idx_tile = pl.create_tensor(
                    [T, ROUTER_TOPK_PAD], dtype=pl.INT32,
                )
                for tt in pl.range(T):
                    row = biased_buf[tt : tt + 1, :]
                    idx_init = pl.arange(
                        0, [1, ROUTER_SCORE_PAD], dtype=pl.UINT32,
                    )
                    srt = pl.sort32(row, idx_init)
                    # DeepSeek-style format1 progressive merge (v3_2:
                    # 64->256->1024->4096, 4-way per stage: block_len N -> 4N).
                    # The prior format2 2-way merge of two NOT-fully-sorted
                    # 512-halves hung on distinct scores. srt is 1024 wide
                    # (512 (val,idx) pairs); sort32->64-runs, mrgsort(64)->256,
                    # mrgsort(256)->1024 = one fully-sorted run.
                    srt = pl.mrgsort(srt, block_len=64)
                    srt = pl.mrgsort(srt, block_len=256)
                    pairs = srt[:, 0:ROUTER_SORT_PAD]
                    top_idx = pl.gather(
                        pairs, mask_pattern=pl.tile.MaskPattern.P1010,
                        output_dtype=pl.INT32,
                    )
                    topk_idx_tile[tt : tt + 1, :] = top_idx

                gather_all = pl.gather(
                    score_buf, dim=-1, index=topk_idx_tile,
                )
                gather_valid = pl.set_validshape(gather_all, T, TOPK)
                topk_vals_pad = pl.fillpad(
                    gather_valid, pad_value=pl.PadValue.zero,
                )

                denom = pl.reshape(pl.row_sum(topk_vals_pad), [T, 1])
                weights_pad = pl.mul(
                    pl.row_expand_div(topk_vals_pad, denom),
                    ROUTER_SCALE,
                )

                for tt in pl.range(T):
                    for k in pl.range(TOPK):
                        pl.write(
                            expert_indices, [tt, k],
                            pl.read(topk_idx_tile, [tt, k]),
                        )
                        pl.write(
                            expert_weights, [tt, k],
                            pl.read(weights_pad, [tt, k]),
                        )

            return expert_weights

        @pl.function(type=pl.FunctionType.Inline)
        def _gate_bypass(
            self,
            gate_w: pl.Tensor[[HIDDEN, N_EXPERTS], pl.FP32],
            expert_indices: pl.Tensor[[T, TOPK], pl.INT32],
            expert_weights: pl.Tensor[[T, TOPK], pl.FP32],
        ):
            # vLLM fused-router topk injected via repurposed gate_w buffer:
            # gate_w[tt,k]=float(topk_id), gate_w[tt,TOPK+k]=topk_weight*ROUTE_SCALE.
            # Skips the hanging TSORT32/TMRGSORT gate top-k (PTOAS Blocker 1).
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_bypass"):
                for tt in pl.range(T):
                    for k in pl.range(TOPK):
                        pl.write(
                            expert_indices, [tt, k],
                            pl.cast(pl.read(gate_w, [tt, k]), pl.INT32),
                        )
                        pl.write(
                            expert_weights, [tt, k],
                            pl.read(gate_w, [tt, TOPK + k]),
                        )
            return expert_weights

        @pl.function(type=pl.FunctionType.Inline)
        def gate_step(
            self,
            x: pl.Tensor[[T, HIDDEN], pl.BF16],
            gate_w: pl.Tensor[[HIDDEN, N_EXPERTS], pl.FP32],
            router_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
            expert_indices: pl.Out[pl.Tensor[[T, TOPK], pl.INT32]],
            expert_weights: pl.Out[pl.Tensor[[T, TOPK], pl.FP32]],
        ) -> tuple[
            pl.Tensor[[T, TOPK], pl.INT32],
            pl.Tensor[[T, TOPK], pl.FP32],
        ]:
            if _EPMOE_BYPASS_GATE:
                self._gate_bypass(
                    gate_w, expert_indices, expert_weights,
                )
            else:
                self._gate(
                    x, gate_w, router_bias, expert_indices, expert_weights,
                )
            return expert_indices, expert_weights

        # ---------- Stage 2: dispatch (EP all-to-all) ----------
        #
        # Phase X.3 lift: the four dispatch helpers (histogram_and_prefix_sum,
        # pack_send_payload, build_local_expert_csr, build_inverse_map) are
        # lifted from dispatch.py as @pl.function(Inline) class methods so
        # the pypto frontend resolves them from inside dispatch_step. The
        # originals in dispatch.py remain intact for unit-test independence.

        @pl.function(type=pl.FunctionType.Inline)
        def _histogram_and_prefix_sum(
            self,
            indices: pl.Tensor[[T, TOPK], pl.INT32],
            send_counts_per_bucket: pl.Tensor[[PER_RANK_BUCKETS], pl.INT32],
            send_counts_per_rank: pl.Tensor[[N_RANKS], pl.INT32],
            send_offsets_per_rank: pl.Tensor[[N_RANKS], pl.INT32],
        ):
            """Local histogram + per-rank prefix-sum prelude."""
            for bkt in pl.range(PER_RANK_BUCKETS):
                pl.write(
                    send_counts_per_bucket, [bkt], pl.cast(0, pl.INT32),
                )
            for r in pl.range(N_RANKS):
                pl.write(
                    send_counts_per_rank, [r], pl.cast(0, pl.INT32),
                )

            for t in pl.range(T):
                for k in pl.range(TOPK):
                    eid = pl.read(indices, [t, k])
                    dst = eid // N_LOCAL_EXPERTS
                    loc_e = eid - dst * N_LOCAL_EXPERTS
                    bkt = dst * N_LOCAL_EXPERTS + loc_e
                    cur = pl.read(send_counts_per_bucket, [bkt])
                    pl.write(
                        send_counts_per_bucket, [bkt],
                        pl.cast(cur + 1, pl.INT32),
                    )
                    r_cur = pl.read(send_counts_per_rank, [dst])
                    pl.write(
                        send_counts_per_rank, [dst],
                        pl.cast(r_cur + 1, pl.INT32),
                    )

            pl.write(send_offsets_per_rank, [0], pl.cast(0, pl.INT32))
            for r in pl.range(1, N_RANKS):
                prev_off = pl.read(send_offsets_per_rank, [r - 1])
                prev_cnt = pl.read(send_counts_per_rank, [r - 1])
                pl.write(
                    send_offsets_per_rank, [r],
                    pl.cast(prev_off + prev_cnt, pl.INT32),
                )

        @pl.function(type=pl.FunctionType.Inline)
        def _pack_send_payload(
            self,
            x: pl.Tensor[[T, HIDDEN], pl.INT8],
            x_scale: pl.Tensor[[T, SCALE_W_PAD], pl.FP32],
            indices: pl.Tensor[[T, TOPK], pl.INT32],
            send_counts_per_bucket: pl.Tensor[[PER_RANK_BUCKETS], pl.INT32],
            send_offsets_per_rank: pl.Tensor[[N_RANKS], pl.INT32],
            send_buf: pld.DistributedTensor[[LOCAL_RECV_MAX, HIDDEN], pl.INT8],
            send_scale: pld.DistributedTensor[
                [LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
            ],
            cursor_per_bucket: pl.Tensor[[PER_RANK_BUCKETS], pl.INT32],
            bucket_offset: pl.Tensor[[PER_RANK_BUCKETS], pl.INT32],
        ):
            """Pack outgoing tokens into ``send_buf`` ordered by (dst, loc_e)."""
            for r in pl.range(N_RANKS):
                rank_off = pl.cast(r * (T * TOPK), pl.INT32)  # symmetric dst-block base
                pl.write(
                    bucket_offset, [r * N_LOCAL_EXPERTS],
                    pl.cast(rank_off, pl.INT32),
                )
                pl.write(
                    cursor_per_bucket, [r * N_LOCAL_EXPERTS],
                    pl.cast(rank_off, pl.INT32),
                )
                for e in pl.range(1, N_LOCAL_EXPERTS):
                    prev_off = pl.read(
                        bucket_offset, [r * N_LOCAL_EXPERTS + e - 1],
                    )
                    prev_cnt = pl.read(
                        send_counts_per_bucket,
                        [r * N_LOCAL_EXPERTS + e - 1],
                    )
                    new_off = pl.cast(prev_off + prev_cnt, pl.INT32)
                    pl.write(
                        bucket_offset, [r * N_LOCAL_EXPERTS + e], new_off,
                    )
                    pl.write(
                        cursor_per_bucket, [r * N_LOCAL_EXPERTS + e],
                        new_off,
                    )

            for t in pl.range(T):
                for k in pl.range(TOPK):
                    eid = pl.read(indices, [t, k])
                    dst = eid // N_LOCAL_EXPERTS
                    loc_e = eid - dst * N_LOCAL_EXPERTS
                    bkt = dst * N_LOCAL_EXPERTS + loc_e
                    slot_i32 = pl.read(cursor_per_bucket, [bkt])
                    slot = pl.cast(slot_i32, pl.INDEX)
                    x_tile = pl.load(x, [t, 0], [1, HIDDEN])
                    pl.store(x_tile, [slot, 0], send_buf)
                    # Carry the per-token dequant scale alongside the INT8
                    # activation so the receiver can dequantize in the routed
                    # expert.  Full SCALE_W_PAD-wide tile (col 0 = scale,
                    # cols 1..7 = pad) keeps the remote_load tile 32B-aligned.
                    s_tile = pl.load(x_scale, [t, 0], [1, SCALE_W_PAD])
                    pl.store(s_tile, [slot, 0], send_scale)
                    pl.write(
                        cursor_per_bucket, [bkt],
                        pl.cast(slot_i32 + 1, pl.INT32),
                    )

        @pl.function(type=pl.FunctionType.Inline)
        def _build_local_expert_csr(
            self,
            pub_counts: pld.DistributedTensor[
                [N_RANKS * N_RANKS, N_LOCAL_EXPERTS], pl.INT32
            ],
            local_expert_offset: pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
            local_expert_count: pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ):
            """Receiver-side CSR: scan pub_counts for my dst slot."""
            for e in pl.range(N_LOCAL_EXPERTS):
                acc = pl.cast(0, pl.INT32)
                for s in pl.range(N_RANKS):
                    acc = acc + pl.read(
                        pub_counts, [s * N_RANKS + my_rank, e],
                    )
                pl.write(local_expert_count, [e], pl.cast(acc, pl.INT32))

            pl.write(local_expert_offset, [0], pl.cast(0, pl.INT32))
            for e in pl.range(1, N_LOCAL_EXPERTS):
                prev_off = pl.read(local_expert_offset, [e - 1])
                prev_cnt = pl.read(local_expert_count, [e - 1])
                pl.write(
                    local_expert_offset, [e],
                    pl.cast(prev_off + prev_cnt, pl.INT32),
                )

        @pl.function(type=pl.FunctionType.Inline)
        def _build_inverse_map(
            self,
            indices: pl.Tensor[[T, TOPK], pl.INT32],
            pub_counts: pld.DistributedTensor[
                [N_RANKS * N_RANKS, N_LOCAL_EXPERTS], pl.INT32
            ],
            inverse_map: pl.Tensor[[T, TOPK], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ):
            """Encode (dst_rank, dst_row_in_recv_buf) into one INT32 per (t,k).

            Packed as ``dst_rank * LOCAL_RECV_MAX + dst_row``.
            """
            cursor = pl.create_tensor(
                [PER_RANK_BUCKETS], dtype=pl.INT32,
            )
            for bkt in pl.range(PER_RANK_BUCKETS):
                pl.write(cursor, [bkt], pl.cast(0, pl.INT32))

            for t in pl.range(T):
                for k in pl.range(TOPK):
                    eid = pl.read(indices, [t, k])
                    dst = eid // N_LOCAL_EXPERTS
                    loc_e = eid - dst * N_LOCAL_EXPERTS
                    bkt = dst * N_LOCAL_EXPERTS + loc_e

                    src_off = pl.cast(0, pl.INT32)
                    for s in pl.range(N_RANKS):
                        if s < my_rank:
                            src_off = src_off + pl.read(
                                pub_counts, [s * N_RANKS + dst, loc_e],
                            )

                    loc_e_off = pl.cast(0, pl.INT32)
                    for prev_e in pl.range(N_LOCAL_EXPERTS):
                        if prev_e < loc_e:
                            for s in pl.range(N_RANKS):
                                loc_e_off = loc_e_off + pl.read(
                                    pub_counts,
                                    [s * N_RANKS + dst, prev_e],
                                )

                    my_cursor_val = pl.read(cursor, [bkt])
                    dst_row = loc_e_off + src_off + my_cursor_val
                    packed = (
                        dst * pl.cast(LOCAL_RECV_MAX, pl.INT32) + dst_row
                    )
                    pl.write(inverse_map, [t, k], pl.cast(packed, pl.INT32))
                    pl.write(
                        cursor, [bkt],
                        pl.cast(my_cursor_val + 1, pl.INT32),
                    )

        @pl.function(type=pl.FunctionType.InCore)
        def dispatch_step(  # noqa: PLR0913
            self,
            x: pl.Tensor[[T, HIDDEN], pl.INT8],
            x_scale: pl.Tensor[[T, SCALE_W_PAD], pl.FP32],
            expert_indices: pl.Tensor[[T, TOPK], pl.INT32],
            local_routed_x_out: pl.Out[
                pl.Tensor[[LOCAL_RECV_MAX, HIDDEN], pl.INT8]
            ],
            local_routed_x_scale_out: pl.Out[
                pl.Tensor[[LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32]
            ],
            local_expert_offset: pl.Out[
                pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32]
            ],
            local_expert_count: pl.Out[
                pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32]
            ],
            inverse_map: pl.Out[pl.Tensor[[T, TOPK], pl.INT32]],
            # ``send_buf`` is a DistributedTensor window allocated by the
            # orchestration caller (DDR-backed, ~4 MB INT8).  Using
            # DistributedTensor allows ep_all_to_all's pld.tile.remote_load
            # on it for cross-rank pull.
            send_buf: pld.DistributedTensor[[LOCAL_RECV_MAX, HIDDEN], pl.INT8],
            send_scale: pld.DistributedTensor[
                [LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
            ],
            # Windows:
            pub_counts: pld.DistributedTensor[
                [N_RANKS * N_RANKS, N_LOCAL_EXPERTS], pl.INT32
            ],
            count_done_sig: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
            recv_x: pld.DistributedTensor[
                [LOCAL_RECV_MAX, HIDDEN], pl.INT8
            ],
            recv_scale: pld.DistributedTensor[
                [LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
            ],
            data_done_sig: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> tuple[
            pl.Tensor[[LOCAL_RECV_MAX, HIDDEN], pl.INT8],
            pl.Tensor[[LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32],
            pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
            pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
            pl.Tensor[[T, TOPK], pl.INT32],
        ]:
            # ---- Prelude: local histogram + per-rank prefix-sum ----
            send_counts_bkt = pl.create_tensor(
                [PER_RANK_BUCKETS], dtype=pl.INT32,
            )
            send_counts_rank = pl.create_tensor([N_RANKS_PAD], dtype=pl.INT32)
            send_offsets_rank = pl.create_tensor([N_RANKS_PAD], dtype=pl.INT32)
            self._histogram_and_prefix_sum(
                expert_indices,
                send_counts_bkt, send_counts_rank, send_offsets_rank,
            )

            # ---- Publish per-rank counts (Set: single-writer per cell) ----
            for peer in pl.range(N_RANKS):
                for e in pl.range(N_LOCAL_EXPERTS):
                    v = pl.read(
                        send_counts_bkt, [peer * N_LOCAL_EXPERTS + e],
                    )
                    if peer == my_rank:
                        pl.write(
                            pub_counts,
                            [my_rank * N_RANKS + my_rank, e],
                            v,
                        )
                    else:
                        pld.system.notify(
                            target=pub_counts,
                            peer=peer,
                            offsets=[my_rank * N_RANKS + peer, e],
                            value=v,
                            op=pld.NotifyOp.Set,
                        )

            # ---- Count-done barrier ----
            for peer in pl.range(N_RANKS):
                if peer != my_rank:
                    pld.system.notify(
                        target=count_done_sig,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for src in pl.range(N_RANKS):
                if src != my_rank:
                    pld.system.wait(
                        signal=count_done_sig,
                        offsets=[src, 0],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )

            # ---- Pack outgoing payload into the send_buf window ----
            cursor_bkt = pl.create_tensor(
                [PER_RANK_BUCKETS], dtype=pl.INT32,
            )
            bucket_offset = pl.create_tensor(
                [PER_RANK_BUCKETS], dtype=pl.INT32,
            )
            self._pack_send_payload(
                x, x_scale, expert_indices,
                send_counts_bkt, send_offsets_rank,
                send_buf, send_scale, cursor_bkt, bucket_offset,
            )

            # ---- Build send/recv counts/offsets for ep_all_to_all ----
            recv_counts = pl.create_tensor([N_RANKS_PAD], dtype=pl.INT32)
            for src in pl.range(N_RANKS):
                acc = pl.cast(0, pl.INT32)
                for e in pl.range(N_LOCAL_EXPERTS):
                    acc = acc + pl.read(
                        pub_counts, [src * N_RANKS + my_rank, e],
                    )
                pl.write(recv_counts, [src], pl.cast(acc, pl.INT32))
            recv_offsets = pl.create_tensor([N_RANKS_PAD], dtype=pl.INT32)
            pl.write(recv_offsets, [0], pl.cast(0, pl.INT32))
            for r in pl.range(1, N_RANKS):
                prev_off = pl.read(recv_offsets, [r - 1])
                prev_cnt = pl.read(recv_counts, [r - 1])
                pl.write(
                    recv_offsets, [r],
                    pl.cast(prev_off + prev_cnt, pl.INT32),
                )

            # read_offsets[peer] = offset in PEER's send_buf where peer put
            # MY tokens = sum over dst<my_rank of peer's per-expert counts.
            # pub_counts is fully published (count_done barrier), so local.
            read_offsets = pl.create_tensor([N_RANKS_PAD], dtype=pl.INT32)
            for peer in pl.range(N_RANKS):
                racc = pl.cast(0, pl.INT32)
                for d in pl.range(N_RANKS):
                    if d < my_rank:
                        for e in pl.range(N_LOCAL_EXPERTS):
                            racc = racc + pl.read(
                                pub_counts, [peer * N_RANKS + d, e],
                            )
                pl.write(read_offsets, [peer], pl.cast(racc, pl.INT32))

            # ---- EP all-to-all push of payload ----
            self.ep_all_to_all(
                send_buf, recv_x, send_scale, recv_scale,
                send_counts_rank, recv_counts,
                send_offsets_rank, recv_offsets, read_offsets,
                data_done_sig, my_rank,
            )

            # ---- Stage out: build per-local-expert CSR view + re-pack ----
            self._build_local_expert_csr(
                pub_counts,
                local_expert_offset, local_expert_count,
                my_rank,
            )
            # Re-pack recv_x (src-rank-major) into local_routed_x_out
            # (loc_e-major, src-rank-secondary) so expert_routed sees
            # contiguous CSR rows per local expert.  The matching scale row
            # is re-packed in lockstep so the per-token dequant scale stays
            # aligned with its INT8 activation row.
            running = pl.cast(0, pl.INT32)
            for e in pl.range(N_LOCAL_EXPERTS):
                for src in pl.range(N_RANKS):
                    n = pl.cast(
                        pl.read(pub_counts, [src * N_RANKS + my_rank, e]),
                        pl.INDEX,
                    )
                    src_base = pl.cast(src * (T * TOPK), pl.INDEX)  # symmetric src-block base
                    src_e_off = pl.cast(0, pl.INT32)
                    for prev_e in pl.range(N_LOCAL_EXPERTS):
                        if prev_e < e:
                            src_e_off = src_e_off + pl.read(
                                pub_counts,
                                [src * N_RANKS + my_rank, prev_e],
                            )
                    for row in pl.range(n):
                        src_row = (
                            src_base
                            + pl.cast(src_e_off, pl.INDEX) + row
                        )
                        dst_row = pl.cast(running, pl.INDEX) + row
                        tile = pl.load(recv_x, [src_row, 0], [1, HIDDEN])
                        pl.store(tile, [dst_row, 0], local_routed_x_out)
                        s = pl.load(
                            recv_scale, [src_row, 0], [1, SCALE_W_PAD],
                        )
                        pl.store(
                            s, [dst_row, 0], local_routed_x_scale_out,
                        )
                    running = running + pl.cast(n, pl.INT32)

            # ---- Inverse-map for combine ----
            self._build_inverse_map(
                expert_indices, pub_counts, inverse_map, my_rank,
            )

            return (
                local_routed_x_out,
                local_routed_x_scale_out,
                local_expert_offset,
                local_expert_count,
                inverse_map,
            )

        # ---------- Stage 3a: expert_routed (local 36 experts) ----------
        #
        # Phase X.3 lift: the expert_routed factory body is inlined here as
        # a class-method @pl.function(Inline).  Row-tiling (RECV_TILE=32)
        # prevents the [ROUTED_MAX_TILE, INTER] FP32 accumulator from
        # exceeding the 192 KB Vec/UB budget.  pl.spmd dispatches the
        # gate/up and down projections into cube kernels.
        @pl.function(type=pl.FunctionType.Inline)
        def _expert_routed(  # noqa: PLR0913, PLR0915
            self,
            # INT8-native routed experts (true W8A8, math mirrors DeepSeek v4
            # expert_routed.py and matches the vLLM-ascend W8A8 oracle):
            #   * local_routed_x arrives INT8 (per-token quantised in dispatch),
            #     paired with local_routed_x_scale (per-token FP32 dequant scale).
            #   * gate/up/down weights are INT8 with per-output-channel FP32
            #     scale (step3p5 [K,N] layout -> scale on trailing N dim, applied
            #     via col_expand_mul; the per-token act scale via row_expand_mul).
            local_routed_x: pl.Tensor[[LOCAL_RECV_MAX, HIDDEN], pl.INT8],
            local_routed_x_scale: pl.Tensor[
                [LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
            ],
            local_expert_offset: pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
            local_expert_count: pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
            w_gate: pl.Tensor[
                [N_LOCAL_EXPERTS, HIDDEN, INTER], pl.INT8
            ],
            w_gate_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
            w_up: pl.Tensor[
                [N_LOCAL_EXPERTS, HIDDEN, INTER], pl.INT8
            ],
            w_up_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
            w_down: pl.Tensor[
                [N_LOCAL_EXPERTS, INTER, HIDDEN], pl.INT8
            ],
            w_down_scale: pl.Tensor[[N_LOCAL_EXPERTS, HIDDEN], pl.FP32],
            local_routed_y: pl.Tensor[
                [LOCAL_RECV_MAX, HIDDEN], pl.BF16
            ],
        ):
            for e in pl.parallel(N_LOCAL_EXPERTS):
                n_rows = pl.read(local_expert_count, [e])
                offset_i32 = pl.read(local_expert_offset, [e])
                offset = pl.cast(offset_i32, pl.INDEX)
                valid_rows = pl.cast(n_rows, pl.INDEX)

                # Row-tile loop — process RECV_TILE rows per outer iteration.
                # Bridge-tensor pattern (h_bf16 at tile_idx loop level, outside
                # both expert_gate_up and expert_down pl.spmd scopes) eliminates
                # the [32, INTER] FP32 h_tile accumulator from each scope's live
                # set and partitions gate/up from down-proj allocations.
                #
                # ``tile_valid`` is min(RECV_TILE, n_rows - tile_row0) so the
                # trailing tile correctly masks out padding rows past
                # ``offset + n_rows``.
                for tile_idx in pl.range(N_RECV_TILES):
                    tile_row0 = tile_idx * RECV_TILE
                    tile_offset = offset + tile_row0
                    # Per-tile valid rows (scalar) — clamps trailing tile to
                    # the active row span.  Use ``pl.min`` for scalar min/max
                    # (``pl.minimum`` is the tensor variant).
                    tile_valid = pl.min(RECV_TILE, valid_rows - tile_row0)
                    if tile_row0 < valid_rows:

                        # Bridge tensor — lives at tile_idx loop level, shared
                        # between expert_gate_up and expert_down SPMD dispatches.
                        # As an Inline-level create_tensor it is in vec (UB) space;
                        # pl.slice of it in expert_down gives tmov vec→left ✓.
                        h_bf16 = pl.create_tensor(
                            [RECV_TILE, INTER], dtype=pl.BF16,
                        )

                        # Gate+up projection: each SPMD block handles one N-chunk
                        # of the INTER dimension.
                        for nb in pl.spmd(
                            INTER // ROUTED_GATE_N_CHUNK,
                            name_hint="expert_gate_up",
                        ):
                            n0 = nb * ROUTED_GATE_N_CHUNK
                            x0 = pl.slice(
                                local_routed_x,
                                [RECV_TILE, ROUTED_GATE_K_CHUNK],
                                [tile_offset, 0],
                            )
                            wg0_2d = pl.reshape(
                                pl.slice(
                                    w_gate,
                                    [1, ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                                    [e, 0, n0],
                                ),
                                [ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                            )
                            wu0_2d = pl.reshape(
                                pl.slice(
                                    w_up,
                                    [1, ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                                    [e, 0, n0],
                                ),
                                [ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                            )
                            gate_acc = pl.matmul(x0, wg0_2d, out_dtype=pl.INT32)
                            up_acc = pl.matmul(x0, wu0_2d, out_dtype=pl.INT32)
                            for kb in pl.range(1, HIDDEN // ROUTED_GATE_K_CHUNK):
                                k0 = kb * ROUTED_GATE_K_CHUNK
                                xk = pl.slice(
                                    local_routed_x,
                                    [RECV_TILE, ROUTED_GATE_K_CHUNK],
                                    [tile_offset, k0],
                                )
                                wgk = pl.reshape(
                                    pl.slice(
                                        w_gate,
                                        [
                                            1,
                                            ROUTED_GATE_K_CHUNK,
                                            ROUTED_GATE_N_CHUNK,
                                        ],
                                        [e, k0, n0],
                                    ),
                                    [ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                                )
                                wuk = pl.reshape(
                                    pl.slice(
                                        w_up,
                                        [
                                            1,
                                            ROUTED_GATE_K_CHUNK,
                                            ROUTED_GATE_N_CHUNK,
                                        ],
                                        [e, k0, n0],
                                    ),
                                    [ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                                )
                                gate_acc = pl.matmul_acc(gate_acc, xk, wgk)
                                up_acc = pl.matmul_acc(up_acc, xk, wuk)

                            # Dequant INT32 -> FP32: per-token activation scale
                            # (row_expand_mul) x per-output-channel weight scale
                            # (col_expand_mul), matching DeepSeek v4 expert_routed
                            # and the vLLM-ascend W8A8 oracle.  step3p5 weight is
                            # [K,N] so the output channel is the trailing N dim.
                            x_scale_col = pl.slice(
                                local_routed_x_scale,
                                [RECV_TILE, 1], [tile_offset, 0],
                            )
                            wg_scale_row = pl.slice(
                                w_gate_scale, [1, ROUTED_GATE_N_CHUNK], [e, n0],
                            )
                            wu_scale_row = pl.slice(
                                w_up_scale, [1, ROUTED_GATE_N_CHUNK], [e, n0],
                            )
                            gate_2d = pl.col_expand_mul(
                                pl.row_expand_mul(
                                    pl.cast(
                                        gate_acc, target_type=pl.FP32, mode="none",
                                    ),
                                    x_scale_col,
                                ),
                                wg_scale_row,
                            )
                            up_2d = pl.col_expand_mul(
                                pl.row_expand_mul(
                                    pl.cast(
                                        up_acc, target_type=pl.FP32, mode="none",
                                    ),
                                    x_scale_col,
                                ),
                                wu_scale_row,
                            )
                            sigmoid = pl.recip(
                                pl.add(pl.exp(pl.neg(gate_2d)), 1.0),
                            )
                            silu = pl.mul(gate_2d, sigmoid)
                            # Compile-time const baked at factory time: only one
                            # branch is emitted per specialisation.
                            if _routed_swiglu_step:
                                silu_c = pl.minimum(silu, _routed_swiglu_limit)
                                up_c = pl.maximum(
                                    pl.minimum(up_2d, _routed_swiglu_limit),
                                    -_routed_swiglu_limit,
                                )
                                gated = pl.mul(silu_c, up_c)
                            else:
                                gated = pl.mul(silu, up_2d)

                            gated_v = pl.set_validshape(
                                gated, tile_valid, ROUTED_GATE_N_CHUNK,
                            )
                            # DeepSeek-aligned (v4 expert_routed fillpad): zero the
                            # gated tile so h_bf16 rows [tile_valid:32] are ZERO not
                            # uninitialised UB (partial-tile grouped-GEMM fix). New var
                            # (fillpad resets valid_shape -> cannot reassign gated_v).
                            gated_m = pl.fillpad(gated_v, pad_value=pl.PadValue.zero)
                            # No fillpad — gated_v (none-pad mode) matches
                            # uninitialised h_bf16 subview (none-pad mode);
                            # expert_down reads h_bf16 with valid_shape= so
                            # padding rows beyond tile_valid are not used.
                            h_bf16[
                                :, n0 : n0 + ROUTED_GATE_N_CHUNK
                            ] = pl.cast(gated_m, target_type=pl.BF16)

                        # --- Per-token INT8 requant of the swiglu intermediate.
                        # Produces an INT8 h tile fed straight into the INT8
                        # down-proj (true W8A8, matches the vLLM-ascend oracle's
                        # npu_dynamic_quant before the INT8 down-proj).  Math +
                        # cast chain copy DeepSeek v4 expert_routed h_tile_i8
                        # (FP32 -> INT32 rint -> FP16 round -> INT8 trunc inside a
                        # pl.at(CORE_GROUP) scope) so the in-kernel INT8 tile is
                        # cube-consumable (avoids the gap-5 tcvt->cube layout bug).
                        # UNCONDITIONAL: every routed layer is W8A8; the swiglu
                        # clamp above (gated by _routed_swiglu_step) is orthogonal.
                        # h_scale_dq (per-token FP32 dequant scale) is an SSA value
                        # produced here and consumed in the down spmd below, exactly
                        # like DeepSeek's h_tile_scale_dq (exp_h_q -> exp_w2).
                        h_i8 = pl.create_tensor(
                            [RECV_TILE, INTER], dtype=pl.INT8,
                        )
                        with pl.at(
                            level=pl.Level.CORE_GROUP, name_hint="routed_h_quant",
                        ):
                            eh_amax = pl.full(
                                [1, RECV_TILE], dtype=pl.FP32, value=1e-4,
                            )
                            for hqa in pl.range(INTER // ROUTED_GATE_N_CHUNK):
                                hqa0 = hqa * ROUTED_GATE_N_CHUNK
                                eh_a = pl.cast(
                                    pl.slice(
                                        h_bf16,
                                        [RECV_TILE, ROUTED_GATE_N_CHUNK],
                                        [0, hqa0],
                                    ),
                                    target_type=pl.FP32,
                                )
                                eh_amax = pl.maximum(
                                    eh_amax,
                                    pl.reshape(
                                        pl.row_max(
                                            pl.maximum(eh_a, pl.neg(eh_a)),
                                        ),
                                        [1, RECV_TILE],
                                    ),
                                )
                            eh_sq_row = pl.div(
                                pl.full(
                                    [1, RECV_TILE], dtype=pl.FP32, value=127.0,
                                ),
                                eh_amax,
                            )
                            # Per-token dequant scale (recip of quant scale) for
                            # the down-proj dequant.  SSA value carried into the
                            # down spmd (DeepSeek h_tile_scale_dq pattern).
                            h_scale_dq = pl.reshape(
                                pl.recip(eh_sq_row), [RECV_TILE, 1],
                            )
                            eh_sq_col = pl.reshape(eh_sq_row, [RECV_TILE, 1])
                            for hqn in pl.range(INTER // ROUTED_GATE_N_CHUNK):
                                hqn0 = hqn * ROUTED_GATE_N_CHUNK
                                eh_q = pl.cast(
                                    pl.slice(
                                        h_bf16,
                                        [RECV_TILE, ROUTED_GATE_N_CHUNK],
                                        [0, hqn0],
                                    ),
                                    target_type=pl.FP32,
                                )
                                eh_scaled = pl.row_expand_mul(eh_q, eh_sq_col)
                                eh_i32 = pl.cast(
                                    eh_scaled, target_type=pl.INT32, mode="rint",
                                )
                                eh_half = pl.cast(
                                    eh_i32, target_type=pl.FP16, mode="round",
                                )
                                h_i8[
                                    :, hqn0 : hqn0 + ROUTED_GATE_N_CHUNK
                                ] = pl.cast(
                                    eh_half, target_type=pl.INT8, mode="trunc",
                                )

                        # Down projection: each SPMD block handles one D-chunk of
                        # the HIDDEN output dimension.  h_bf16 is vec (UB) space so
                        # pl.slice of it gives tmov vec→left ✓.
                        for db in pl.spmd(
                            HIDDEN // ROUTED_DOWN_N_CHUNK,
                            name_hint="expert_down",
                        ):
                            d0 = db * ROUTED_DOWN_N_CHUNK
                            h0 = pl.slice(
                                h_i8,
                                [RECV_TILE, ROUTED_DOWN_K_CHUNK],
                                [0, 0],
                            )
                            wd0 = pl.reshape(
                                pl.slice(
                                    w_down,
                                    [
                                        1,
                                        ROUTED_DOWN_K_CHUNK,
                                        ROUTED_DOWN_N_CHUNK,
                                    ],
                                    [e, 0, d0],
                                ),
                                [ROUTED_DOWN_K_CHUNK, ROUTED_DOWN_N_CHUNK],
                            )
                            y_acc = pl.matmul(h0, wd0, out_dtype=pl.INT32)
                            for kb2 in pl.range(1, INTER // ROUTED_DOWN_K_CHUNK):
                                k0 = kb2 * ROUTED_DOWN_K_CHUNK
                                hk = pl.slice(
                                    h_i8,
                                    [RECV_TILE, ROUTED_DOWN_K_CHUNK],
                                    [0, k0],
                                )
                                wdk = pl.reshape(
                                    pl.slice(
                                        w_down,
                                        [
                                            1,
                                            ROUTED_DOWN_K_CHUNK,
                                            ROUTED_DOWN_N_CHUNK,
                                        ],
                                        [e, k0, d0],
                                    ),
                                    [
                                        ROUTED_DOWN_K_CHUNK,
                                        ROUTED_DOWN_N_CHUNK,
                                    ],
                                )
                                y_acc = pl.matmul_acc(y_acc, hk, wdk)

                            # Dequant INT32 -> FP32: per-token h scale
                            # (row_expand_mul) x per-output-channel down-weight
                            # scale (col_expand_mul).  Routing weight is NOT
                            # applied here (step3p5 applies it in combine reduce,
                            # unlike DeepSeek which folds it into row_scale).
                            wd_scale_row = pl.slice(
                                w_down_scale, [1, ROUTED_DOWN_N_CHUNK], [e, d0],
                            )
                            y_2d = pl.col_expand_mul(
                                pl.row_expand_mul(
                                    pl.cast(
                                        y_acc, target_type=pl.FP32, mode="none",
                                    ),
                                    h_scale_dq,
                                ),
                                wd_scale_row,
                            )
                            y_v = pl.set_validshape(
                                y_2d, tile_valid, ROUTED_DOWN_N_CHUNK,
                            )
                            y_m = pl.fillpad(
                                y_v, pad_value=pl.PadValue.zero,
                            )
                            local_routed_y = pl.assemble(
                                local_routed_y,
                                pl.cast(y_m, target_type=pl.BF16),
                                [tile_offset, d0],
                            )

            return local_routed_y

        @pl.function(type=pl.FunctionType.Inline)
        def expert_routed_step(
            self,
            local_routed_x: pl.Tensor[[LOCAL_RECV_MAX, HIDDEN], pl.INT8],
            local_routed_x_scale: pl.Tensor[
                [LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
            ],
            local_expert_offset: pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
            local_expert_count: pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
            w_gate_r: pl.Tensor[
                [N_LOCAL_EXPERTS, HIDDEN, INTER], pl.INT8
            ],
            w_gate_r_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
            w_up_r: pl.Tensor[
                [N_LOCAL_EXPERTS, HIDDEN, INTER], pl.INT8
            ],
            w_up_r_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
            w_down_r: pl.Tensor[
                [N_LOCAL_EXPERTS, INTER, HIDDEN], pl.INT8
            ],
            w_down_r_scale: pl.Tensor[[N_LOCAL_EXPERTS, HIDDEN], pl.FP32],
            local_routed_y: pl.Out[
                pl.Tensor[[LOCAL_RECV_MAX, HIDDEN], pl.BF16]
            ],
        ) -> pl.Tensor[[LOCAL_RECV_MAX, HIDDEN], pl.BF16]:
            local_routed_y = self._expert_routed(
                local_routed_x, local_routed_x_scale,
                local_expert_offset, local_expert_count,
                w_gate_r, w_gate_r_scale,
                w_up_r, w_up_r_scale,
                w_down_r, w_down_r_scale,
                local_routed_y,
            )
            return local_routed_y

        # ---------- Stage 3b: expert_shared (TP-sliced + tp_all_reduce) ----
        #
        # Phase X.3 lift: the expert_shared local FFN body
        # (expert_shared.py's @pl.jit.inline expert_shared_local) is
        # inlined here as a class-method @pl.function(Inline). The body
        # uses only ``pl.at(level=pl.Level.CORE_GROUP)`` cube-level
        # matmuls — no parallel/spmd/distributed primitives — so Inline
        # is sufficient. Activation choice (plain SiLU vs. SwigluStep@16)
        # is baked at factory build time via the ``_shared_swiglu_step``
        # / ``_shared_swiglu_limit`` Python closure constants. The
        # originals (``expert_shared_silu`` / ``expert_shared_swiglu16``)
        # in expert_shared.py remain intact for unit-test independence.
        @pl.function(type=pl.FunctionType.Inline)
        def _expert_shared_local(
            self,
            x: pl.Tensor[[T, HIDDEN], pl.BF16],
            w_gate: pl.Tensor[[HIDDEN, SH_INTER_LOCAL], pl.BF16],
            w_up: pl.Tensor[[HIDDEN, SH_INTER_LOCAL], pl.BF16],
            w_down: pl.Tensor[[SH_INTER_LOCAL, HIDDEN], pl.BF16],
            sh_y_shard: pl.Tensor[[T, HIDDEN], pl.BF16],
        ):
            # Gate+up and down projections merged into one InCore kernel so
            # that h_tile (Vec SRAM) is live across both projections in the
            # same kernel dispatch.  Two separate pl.at(CORE_GROUP) scopes
            # become two separate InCore kernel dispatches when
            # expert_shared_step is Inline, and h_tile cannot cross that
            # dispatch boundary.
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_mlp"):
                # NCHUNK-CLAMP-FIX v4: swiglu16 clamp on the FULL [T,160] Vec tile is
                # miscompiled (wide-tile tmov misprune ~45%). Run the clamp on 5 narrow
                # [T,32] chunks (mirror working routed path) as separate tiles h_c0..h_c4;
                # they feed the down-proj K-loop directly (cannot recombine into a wide
                # Vec tile: A2/A3 tmov needs matching src/dst shapes). Straight-line
                # (parser rejects list.append / bare-for).

                x0_0 = pl.slice(x, [T, SHARED_GATE_K_CHUNK], [0, 0])
                wg0_0 = pl.slice(w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 0])
                wu0_0 = pl.slice(w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 0])
                gate_acc_0 = pl.matmul(x0_0, wg0_0, out_dtype=pl.FP32)
                up_acc_0 = pl.matmul(x0_0, wu0_0, out_dtype=pl.FP32)
                for kb in pl.range(1, HIDDEN // SHARED_GATE_K_CHUNK):
                    k0 = kb * SHARED_GATE_K_CHUNK
                    xk_0 = pl.slice(x, [T, SHARED_GATE_K_CHUNK], [0, k0])
                    wgk_0 = pl.slice(w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 0])
                    wuk_0 = pl.slice(w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 0])
                    gate_acc_0 = pl.matmul_acc(gate_acc_0, xk_0, wgk_0)
                    up_acc_0 = pl.matmul_acc(up_acc_0, xk_0, wuk_0)
                sigmoid_0 = pl.recip(pl.add(pl.exp(pl.neg(gate_acc_0)), 1.0))
                silu_0 = pl.mul(gate_acc_0, sigmoid_0)
                if _shared_swiglu_step:
                    silu_c_0 = pl.minimum(silu_0, _shared_swiglu_limit)
                    up_c_0 = pl.maximum(pl.minimum(up_acc_0, _shared_swiglu_limit), -_shared_swiglu_limit)
                    gated_0 = pl.mul(silu_c_0, up_c_0)
                else:
                    gated_0 = pl.mul(silu_0, up_acc_0)
                h_c0 = pl.cast(gated_0, target_type=pl.BF16)
                x0_1 = pl.slice(x, [T, SHARED_GATE_K_CHUNK], [0, 0])
                wg0_1 = pl.slice(w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 32])
                wu0_1 = pl.slice(w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 32])
                gate_acc_1 = pl.matmul(x0_1, wg0_1, out_dtype=pl.FP32)
                up_acc_1 = pl.matmul(x0_1, wu0_1, out_dtype=pl.FP32)
                for kb in pl.range(1, HIDDEN // SHARED_GATE_K_CHUNK):
                    k0 = kb * SHARED_GATE_K_CHUNK
                    xk_1 = pl.slice(x, [T, SHARED_GATE_K_CHUNK], [0, k0])
                    wgk_1 = pl.slice(w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 32])
                    wuk_1 = pl.slice(w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 32])
                    gate_acc_1 = pl.matmul_acc(gate_acc_1, xk_1, wgk_1)
                    up_acc_1 = pl.matmul_acc(up_acc_1, xk_1, wuk_1)
                sigmoid_1 = pl.recip(pl.add(pl.exp(pl.neg(gate_acc_1)), 1.0))
                silu_1 = pl.mul(gate_acc_1, sigmoid_1)
                if _shared_swiglu_step:
                    silu_c_1 = pl.minimum(silu_1, _shared_swiglu_limit)
                    up_c_1 = pl.maximum(pl.minimum(up_acc_1, _shared_swiglu_limit), -_shared_swiglu_limit)
                    gated_1 = pl.mul(silu_c_1, up_c_1)
                else:
                    gated_1 = pl.mul(silu_1, up_acc_1)
                h_c1 = pl.cast(gated_1, target_type=pl.BF16)
                x0_2 = pl.slice(x, [T, SHARED_GATE_K_CHUNK], [0, 0])
                wg0_2 = pl.slice(w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 64])
                wu0_2 = pl.slice(w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 64])
                gate_acc_2 = pl.matmul(x0_2, wg0_2, out_dtype=pl.FP32)
                up_acc_2 = pl.matmul(x0_2, wu0_2, out_dtype=pl.FP32)
                for kb in pl.range(1, HIDDEN // SHARED_GATE_K_CHUNK):
                    k0 = kb * SHARED_GATE_K_CHUNK
                    xk_2 = pl.slice(x, [T, SHARED_GATE_K_CHUNK], [0, k0])
                    wgk_2 = pl.slice(w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 64])
                    wuk_2 = pl.slice(w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 64])
                    gate_acc_2 = pl.matmul_acc(gate_acc_2, xk_2, wgk_2)
                    up_acc_2 = pl.matmul_acc(up_acc_2, xk_2, wuk_2)
                sigmoid_2 = pl.recip(pl.add(pl.exp(pl.neg(gate_acc_2)), 1.0))
                silu_2 = pl.mul(gate_acc_2, sigmoid_2)
                if _shared_swiglu_step:
                    silu_c_2 = pl.minimum(silu_2, _shared_swiglu_limit)
                    up_c_2 = pl.maximum(pl.minimum(up_acc_2, _shared_swiglu_limit), -_shared_swiglu_limit)
                    gated_2 = pl.mul(silu_c_2, up_c_2)
                else:
                    gated_2 = pl.mul(silu_2, up_acc_2)
                h_c2 = pl.cast(gated_2, target_type=pl.BF16)
                x0_3 = pl.slice(x, [T, SHARED_GATE_K_CHUNK], [0, 0])
                wg0_3 = pl.slice(w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 96])
                wu0_3 = pl.slice(w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 96])
                gate_acc_3 = pl.matmul(x0_3, wg0_3, out_dtype=pl.FP32)
                up_acc_3 = pl.matmul(x0_3, wu0_3, out_dtype=pl.FP32)
                for kb in pl.range(1, HIDDEN // SHARED_GATE_K_CHUNK):
                    k0 = kb * SHARED_GATE_K_CHUNK
                    xk_3 = pl.slice(x, [T, SHARED_GATE_K_CHUNK], [0, k0])
                    wgk_3 = pl.slice(w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 96])
                    wuk_3 = pl.slice(w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 96])
                    gate_acc_3 = pl.matmul_acc(gate_acc_3, xk_3, wgk_3)
                    up_acc_3 = pl.matmul_acc(up_acc_3, xk_3, wuk_3)
                sigmoid_3 = pl.recip(pl.add(pl.exp(pl.neg(gate_acc_3)), 1.0))
                silu_3 = pl.mul(gate_acc_3, sigmoid_3)
                if _shared_swiglu_step:
                    silu_c_3 = pl.minimum(silu_3, _shared_swiglu_limit)
                    up_c_3 = pl.maximum(pl.minimum(up_acc_3, _shared_swiglu_limit), -_shared_swiglu_limit)
                    gated_3 = pl.mul(silu_c_3, up_c_3)
                else:
                    gated_3 = pl.mul(silu_3, up_acc_3)
                h_c3 = pl.cast(gated_3, target_type=pl.BF16)
                x0_4 = pl.slice(x, [T, SHARED_GATE_K_CHUNK], [0, 0])
                wg0_4 = pl.slice(w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 128])
                wu0_4 = pl.slice(w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [0, 128])
                gate_acc_4 = pl.matmul(x0_4, wg0_4, out_dtype=pl.FP32)
                up_acc_4 = pl.matmul(x0_4, wu0_4, out_dtype=pl.FP32)
                for kb in pl.range(1, HIDDEN // SHARED_GATE_K_CHUNK):
                    k0 = kb * SHARED_GATE_K_CHUNK
                    xk_4 = pl.slice(x, [T, SHARED_GATE_K_CHUNK], [0, k0])
                    wgk_4 = pl.slice(w_gate, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 128])
                    wuk_4 = pl.slice(w_up, [SHARED_GATE_K_CHUNK, SHARED_SWIGLU_N_CHUNK], [k0, 128])
                    gate_acc_4 = pl.matmul_acc(gate_acc_4, xk_4, wgk_4)
                    up_acc_4 = pl.matmul_acc(up_acc_4, xk_4, wuk_4)
                sigmoid_4 = pl.recip(pl.add(pl.exp(pl.neg(gate_acc_4)), 1.0))
                silu_4 = pl.mul(gate_acc_4, sigmoid_4)
                if _shared_swiglu_step:
                    silu_c_4 = pl.minimum(silu_4, _shared_swiglu_limit)
                    up_c_4 = pl.maximum(pl.minimum(up_acc_4, _shared_swiglu_limit), -_shared_swiglu_limit)
                    gated_4 = pl.mul(silu_c_4, up_c_4)
                else:
                    gated_4 = pl.mul(silu_4, up_acc_4)
                h_c4 = pl.cast(gated_4, target_type=pl.BF16)

                # Explicit K-chunking for the down projection.
                # K=160 (SH_INTER_LOCAL) would trigger backend auto-K-split
                # which emits an invalid ``tmov acc→acc``.  Expose the K-loop
                # at the user level using K_INNER=32 (5 passes of 32) so the
                # compiler sees a structured pl.matmul + pl.matmul_acc chain
                # (same pattern as gate/up above) and never needs the copy.
                # Down projection: 5 narrow [T,32] chunks are the K-tiles
                # (no wide Vec h_tile). K accumulation straight-line.
                for db in pl.range(HIDDEN // SHARED_DOWN_N_CHUNK):
                    d0 = db * SHARED_DOWN_N_CHUNK
                    wd_c0 = pl.slice(w_down, [SHARED_SWIGLU_N_CHUNK, SHARED_DOWN_N_CHUNK], [0, d0])
                    y_acc = pl.matmul(h_c0, wd_c0, out_dtype=pl.FP32)
                    wd_c1 = pl.slice(w_down, [SHARED_SWIGLU_N_CHUNK, SHARED_DOWN_N_CHUNK], [32, d0])
                    y_acc = pl.matmul_acc(y_acc, h_c1, wd_c1)
                    wd_c2 = pl.slice(w_down, [SHARED_SWIGLU_N_CHUNK, SHARED_DOWN_N_CHUNK], [64, d0])
                    y_acc = pl.matmul_acc(y_acc, h_c2, wd_c2)
                    wd_c3 = pl.slice(w_down, [SHARED_SWIGLU_N_CHUNK, SHARED_DOWN_N_CHUNK], [96, d0])
                    y_acc = pl.matmul_acc(y_acc, h_c3, wd_c3)
                    wd_c4 = pl.slice(w_down, [SHARED_SWIGLU_N_CHUNK, SHARED_DOWN_N_CHUNK], [128, d0])
                    y_acc = pl.matmul_acc(y_acc, h_c4, wd_c4)
                    sh_y_shard = pl.assemble(
                        sh_y_shard, pl.cast(y_acc, target_type=pl.BF16), [0, d0],
                    )

            return sh_y_shard

        @pl.function(type=pl.FunctionType.Inline)
        def expert_shared_step(  # noqa: PLR0913
            self,
            x: pl.Tensor[[T, HIDDEN], pl.BF16],
            w_gate_s: pl.Tensor[[HIDDEN, SH_INTER_LOCAL], pl.BF16],
            w_up_s: pl.Tensor[[HIDDEN, SH_INTER_LOCAL], pl.BF16],
            w_down_s: pl.Tensor[[SH_INTER_LOCAL, HIDDEN], pl.BF16],
            sh_y: pl.Out[pl.Tensor[[T, HIDDEN], pl.BF16]],
            sh_tmp_window: pld.DistributedTensor[
                [T, HIDDEN], pl.BF16
            ],
            sh_signal_window: pld.DistributedTensor[
                [N_RANKS, 1], pl.INT32
            ],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[T, HIDDEN], pl.BF16]:
            sh_y = self._expert_shared_local(
                x, w_gate_s, w_up_s, w_down_s, sh_y,
            )
            self.tp_all_reduce(
                sh_y, sh_tmp_window, sh_signal_window, my_rank,
            )
            return sh_y

        # ---------- Stage 4: combine (EP a2a back + weighted gather) ------
        #
        # Phase X.3 lift: the three combine helpers (publish_src_route_table,
        # push_routed_y_to_sources, weighted_gather_and_add) are lifted from
        # combine.py as class methods. Phase X.4 rewrites the two cross-rank
        # push sites — the deprecated ``pld.tile.remote_store(target=...,
        # peer=..., offsets=...)`` API is gone; site 1 (route-table publish)
        # uses ``pld.system.notify(op=NotifyOp.Set)`` for the per-cell scalar
        # scatter (canonical, matches dispatch_step's ``pub_counts`` publish);
        # site 2 (routed-y row push) uses the canonical
        # ``pl.store -> pld.tensor.put`` recipe from
        # ``tests/st/distributed/test_l3_put.py`` with a Phase-X.5 FIXME
        # documenting the host-window restructuring the per-row offset
        # semantics needs. The weighted gather is a CORE_GROUP cube body
        # -> Inline. The originals in combine.py remain intact for
        # unit-test independence.
        @pl.function(type=pl.FunctionType.InCore)
        def _publish_src_route_table(
            self,
            indices: pl.Tensor[[T, TOPK], pl.INT32],
            src_route_table: pld.DistributedTensor[
                [N_RANKS, N_LOCAL_EXPERTS, N_ROUTES_PER_RANK], pl.INT32
            ],
            my_rank: pl.Scalar[pl.INT32],
        ):
            cursor = pl.create_tensor(
                [PER_RANK_BUCKETS], dtype=pl.INT32,
            )
            for i in pl.range(PER_RANK_BUCKETS):
                pl.write(cursor, [i], pl.cast(0, pl.INT32))

            for t in pl.range(T):
                for k in pl.range(TOPK):
                    eid = pl.read(indices, [t, k])
                    dst = eid // N_LOCAL_EXPERTS
                    loc_e = eid - dst * N_LOCAL_EXPERTS
                    bkt = dst * N_LOCAL_EXPERTS + loc_e
                    idx = pl.read(cursor, [bkt])
                    r_route = pl.cast(t * TOPK + k, pl.INT32)

                    if dst == my_rank:
                        # Self-rank publish: write the scalar r_route
                        # directly into the local view of the
                        # DistributedTensor. Mirrors the self-branch
                        # used for ``pub_counts`` above; avoids the
                        # ``tmp = create_tensor; load; store`` dance
                        # that the InCore tile verifier rejects with
                        # "tile.load requires TensorType ... got TileType".
                        pl.write(
                            src_route_table,
                            [my_rank, loc_e, pl.cast(idx, pl.INDEX)],
                            r_route,
                        )
                    else:
                        # Cross-rank scalar scatter — replaces the deprecated
                        # ``pld.tile.remote_store(tile, target=..., peer=...,
                        # offsets=...)`` API. The canonical bulk-write
                        # primitive ``pld.tensor.put(dst, peer, src)`` is
                        # whole-tensor only (no offsets), so it cannot express
                        # a per-cell scatter at varying
                        # ``[my_rank, loc_e, idx]`` cells.
                        # ``pld.system.notify(op=NotifyOp.Set)`` IS the
                        # canonical cross-rank scalar-cell write — the same
                        # primitive ``dispatch_step`` already uses to publish
                        # per-(src, dst, e) send counts into ``pub_counts``.
                        # The verifier only requires that ``target`` be a
                        # window-bound DistributedTensor (it is).
                        pld.system.notify(
                            target=src_route_table,
                            peer=dst,
                            offsets=[
                                my_rank, loc_e, pl.cast(idx, pl.INDEX),
                            ],
                            value=r_route,
                            op=pld.NotifyOp.Set,
                        )
                    pl.write(cursor, [bkt], pl.cast(idx + 1, pl.INT32))

        @pl.function(type=pl.FunctionType.InCore)
        def _push_routed_y_to_sources(  # noqa: PLR0913
            self,
            local_routed_y: pl.Tensor[
                [LOCAL_RECV_MAX, HIDDEN], pl.BF16
            ],
            pub_counts: pld.DistributedTensor[
                [N_RANKS * N_RANKS, N_LOCAL_EXPERTS], pl.INT32
            ],
            routed_y_buf: pld.DistributedTensor[
                [N_ROUTES_PER_RANK, HIDDEN], pl.BF16
            ],
            combine_done: pld.DistributedTensor[
                [N_RANKS, 1], pl.INT32
            ],
            src_route_table: pld.DistributedTensor[
                [N_RANKS, N_LOCAL_EXPERTS, N_ROUTES_PER_RANK], pl.INT32
            ],
            my_rank: pl.Scalar[pl.INT32],
        ):
            e_cursor = pl.cast(0, pl.INT32)
            for e in pl.range(N_LOCAL_EXPERTS):
                src_off = pl.cast(0, pl.INT32)
                for src in pl.range(N_RANKS):
                    n = pl.cast(
                        pl.read(
                            pub_counts, [src * N_RANKS + my_rank, e],
                        ),
                        pl.INDEX,
                    )
                    for row in pl.range(n):
                        r_route = pl.read(
                            src_route_table,
                            [src, e, pl.cast(row, pl.INDEX)],
                        )
                        local_row = (
                            pl.cast(e_cursor, pl.INDEX)
                            + pl.cast(src_off, pl.INDEX) + row
                        )
                        if src == my_rank:
                            tile = pl.load(
                                local_routed_y,
                                [local_row, 0], [1, HIDDEN],
                            )
                            pl.store(tile, [r_route, 0], routed_y_buf)
                        else:
                            # DeepSeek-style push (combine_ep): tensor.put
                            # GM->peer-GM establishes a RAW dep into the gather
                            # (remote_store is fire-and-forget -> gather races
                            # push -> partial sums / sign flips).
                            pld.tensor.put(
                                dst=routed_y_buf,
                                peer=src,
                                src=local_routed_y,
                                dst_offsets=[r_route, 0],
                                src_offsets=[local_row, 0],
                                shape=[1, HIDDEN],
                            )
                    src_off = src_off + pl.cast(n, pl.INT32)
                total_e = pl.cast(0, pl.INT32)
                for src2 in pl.range(N_RANKS):
                    total_e = total_e + pl.read(
                        pub_counts, [src2 * N_RANKS + my_rank, e],
                    )
                e_cursor = e_cursor + total_e

            # combine DMA fence (mirrors combine.cpp:177 pipe_barrier between
            # last TPUT and TNOTIFY): remote_store (TPUT) has no trailing dsb
            # (pto-isa TPut.hpp), so combine_done can outrun the push DMA ->
            # consumer gathers stale routed_y_buf. A notify emits
            # dsb(DSB_DDR)+pipe_barrier(PIPE_ALL) (TNotify.hpp) draining the
            # TPUTs. combine_done[my_rank,0] is never waited (waiters use
            # src!=my_rank); AtomicAdd+0 is a pure no-op fence write.
            pld.system.notify(
                target=combine_done,
                peer=my_rank,
                offsets=[my_rank, 0],
                value=0,
                op=pld.NotifyOp.AtomicAdd,
            )
            for peer in pl.range(N_RANKS):
                if peer != my_rank:
                    pld.system.notify(
                        target=combine_done,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for src in pl.range(N_RANKS):
                if src != my_rank:
                    pld.system.wait(
                        signal=combine_done,
                        offsets=[src, 0],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )

        @pl.function(type=pl.FunctionType.Inline)
        def _weighted_gather_and_add(
            self,
            routed_y_buf: pld.DistributedTensor[
                [N_ROUTES_PER_RANK, HIDDEN], pl.BF16
            ],
            expert_weights: pl.Tensor[[T, TOPK], pl.FP32],
            sh_y: pl.Tensor[[T, HIDDEN], pl.BF16],
            moe_out: pl.Tensor[[T, HIDDEN], pl.BF16],
        ):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_combine"):
                for b in pl.range(T):
                    # CORE_GROUP cube body works in Tile-level. Use pl.load
                    # (returns Tile) rather than pl.slice (returns Tensor)
                    # so acc / row_fp32 / weighted all live at the same
                    # type level — pl.add refuses mixed Tensor+Tile ops.
                    acc = pl.cast(
                        pl.load(sh_y, [b, 0], [1, HIDDEN]),
                        target_type=pl.FP32,
                    )
                    for k in pl.range(TOPK):
                        w_fp = pl.read(expert_weights, [b, k])

                        r_route = b * TOPK + k
                        # ``routed_y_buf`` is a ``pld.DistributedTensor``
                        # window — ``pl.slice`` is rejected by the verifier
                        # for window sources (it requires a ``TensorType``).
                        # Use ``pl.load(window, offsets, shape)``, which is
                        # the canonical window->tile lift documented in
                        # ``tests/st/distributed/test_l3_allreduce.py`` (see
                        # the ``acc = pl.load(data, [0, 0], [1, SIZE])``
                        # line in the compute phase). The cube/vec lowering
                        # hoists this per-row load into the surrounding
                        # CORE_GROUP tile when possible.
                        row_fp32 = pl.cast(
                            pl.load(
                                routed_y_buf, [r_route, 0], [1, HIDDEN],
                            ),
                            target_type=pl.FP32,
                        )
                        weighted = pl.mul(row_fp32, w_fp)
                        acc = pl.add(acc, weighted)

                    # acc is a CORE_GROUP Tile; pl.store is the canonical
                    # Tile→Tensor write back into the moe_out output tensor.
                    # pl.assemble is for Tensor→Tensor copy and rejects a
                    # Tile second argument.
                    pl.store(
                        pl.cast(acc, target_type=pl.BF16),
                        [b, 0],
                        moe_out,
                    )

            return moe_out

        @pl.function(type=pl.FunctionType.InCore)
        def _zero_routed_y_buf(
            self,
            routed_y_buf: pld.DistributedTensor[
                [N_ROUTES_PER_RANK, HIDDEN], pl.BF16
            ],
        ):
            # Zero-init: data windows are NOT auto-zeroed (only signal windows).
            # _weighted_gather_and_add reads all N_ROUTES_PER_RANK cells; any slot
            # the combine push misses would read uninitialized ~5e4 garbage.
            for r in pl.range(N_ROUTES_PER_RANK):
                routed_y_buf[r : r + 1, :] = pl.full(
                    [1, HIDDEN], dtype=pl.BF16, value=0.0,
                )

        @pl.function(type=pl.FunctionType.Inline)
        def combine_step(  # noqa: PLR0913
            self,
            local_routed_y: pl.Tensor[
                [LOCAL_RECV_MAX, HIDDEN], pl.BF16
            ],
            expert_indices: pl.Tensor[[T, TOPK], pl.INT32],
            expert_weights: pl.Tensor[[T, TOPK], pl.FP32],
            sh_y: pl.Tensor[[T, HIDDEN], pl.BF16],
            moe_out: pl.Out[pl.Tensor[[T, HIDDEN], pl.BF16]],
            pub_counts: pld.DistributedTensor[
                [N_RANKS * N_RANKS, N_LOCAL_EXPERTS], pl.INT32
            ],
            src_route_table: pld.DistributedTensor[
                [N_RANKS, N_LOCAL_EXPERTS, N_ROUTES_PER_RANK], pl.INT32
            ],
            route_pub_sig: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
            routed_y_buf: pld.DistributedTensor[
                [N_ROUTES_PER_RANK, HIDDEN], pl.BF16
            ],
            combine_done_sig: pld.DistributedTensor[
                [N_RANKS, 1], pl.INT32
            ],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[T, HIDDEN], pl.BF16]:
            # Step A: publish my r_route table to every dst peer.
            # zero routed_y_buf before pushes (unwritten cells else = garbage)
            self._zero_routed_y_buf(routed_y_buf)
            self._publish_src_route_table(
                expert_indices, src_route_table, my_rank,
            )
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="pub_route_barrier"):
                for peer in pl.range(N_RANKS):
                    if peer != my_rank:
                        pld.system.notify(
                            target=route_pub_sig,
                            peer=peer,
                            offsets=[my_rank, 0],
                            value=1,
                            op=pld.NotifyOp.AtomicAdd,
                        )
                for src in pl.range(N_RANKS):
                    if src != my_rank:
                        pld.system.wait(
                            signal=route_pub_sig,
                            offsets=[src, 0],
                            expected=1,
                            cmp=pld.WaitCmp.Ge,
                        )

            # Step B: push my local_routed_y rows back to source ranks.
            self._push_routed_y_to_sources(
                local_routed_y,
                pub_counts,
                routed_y_buf,
                combine_done_sig,
                src_route_table,
                my_rank,
            )

            # Step C: weighted gather + sh_y add (local).
            moe_out = self._weighted_gather_and_add(
                routed_y_buf, expert_weights, sh_y, moe_out,
            )
            return moe_out

        @pl.function(type=pl.FunctionType.InCore)
        def _serialize_after_shared(
            self,
            x: pl.Tensor[[T, HIDDEN], pl.BF16],
            sh_y: pl.Tensor[[T, HIDDEN], pl.BF16],
            x_out: pl.Out[pl.Tensor[[T, HIDDEN], pl.BF16]],
        ):
            # SERIALIZE: numerically identity (x + 0*sh_y) but declares sh_y as
            # an input so the orchestration forces the shared-expert
            # tp_all_reduce (produces sh_y) to COMPLETE before the routed
            # dispatch/combine cross-rank collectives -> breaks the overlapping
            # cross-rank collective deadlock (func5 tp_all_reduce + func13 _push).
            for b in pl.range(T):
                xr = pl.cast(
                    pl.load(x, [b, 0], [1, HIDDEN]), target_type=pl.FP32,
                )
                yr = pl.cast(
                    pl.load(sh_y, [b, 0], [1, HIDDEN]), target_type=pl.FP32,
                )
                z = pl.add(xr, pl.mul(yr, 0.0))
                pl.store(
                    pl.cast(z, target_type=pl.BF16), [b, 0], x_out,
                )
            return x_out

        # InCore scheduled kernel (mirrors _serialize_after_shared: InCore +
        # pl.range + Out params).  It is NOT inlined into the Orchestration
        # caller, so its two Out buffers (x_i8_out, x_scale_out) are written and
        # read by the caller directly.  Uses pl.range (NOT pl.spmd): a top-level
        # pl.spmd carries its own InCore scope, and an InCore function wrapping a
        # top-level spmd leaves a nested InCore ScopeStmt that the SplitIncoreOrch
        # precondition (pypto #1828) rejects.
        @pl.function(type=pl.FunctionType.InCore)
        def _quant_moe_input(
            self,
            x: pl.Tensor[[T, HIDDEN], pl.BF16],
            x_i8_out: pl.Out[pl.Tensor[[T, HIDDEN], pl.INT8]],
            x_scale_out: pl.Out[
                pl.Tensor[[T, SCALE_W_PAD], pl.FP32]
            ],
        ):
            # Per-token INT8 dynamic-quant of the routed-expert INPUT (oracle
            # npu_dynamic_quant(x) before gate_up). Placed AFTER gate+shared,
            # BEFORE dispatch, so router + unquantized shared see the ORIGINAL x
            # and the routed experts consume pre-quantized tokens (gate_up reads
            # a normal GM slice, NOT a vec-quantized cube-matmul input). Tensor
            # pl.slice + spmd-over-tokens, mirroring the proven interm-quant.
            # Per-token over the FULL HIDDEN; INT32-rint; symmetric
            # scale=amax/127 (dequant scale = recip(q_inv) = amax/127).
            #
            # Output: x_i8_out holds INT8 quantized activations, x_scale_out
            # holds the per-token dequant scale (amax/127) in column 0 of the
            # SCALE_W_PAD-wide FP32 window (cols 1..7 are pad, mirror DeepSeek
            # v4 dispatch W_PAD).
            for qtg in pl.range(T // MOE_IN_QUANT_T_TILE):
                qt0 = qtg * MOE_IN_QUANT_T_TILE
                q_amax = pl.full(
                    [1, MOE_IN_QUANT_T_TILE], dtype=pl.FP32, value=1e-4,
                )
                for qab in pl.range(HIDDEN // ROUTED_GATE_K_CHUNK):
                    qa0 = qab * ROUTED_GATE_K_CHUNK
                    qac = pl.cast(
                        pl.slice(
                            x,
                            [MOE_IN_QUANT_T_TILE, ROUTED_GATE_K_CHUNK],
                            [qt0, qa0],
                        ),
                        target_type=pl.FP32,
                    )
                    q_amax = pl.maximum(
                        q_amax,
                        pl.reshape(
                            pl.row_max(pl.maximum(qac, pl.neg(qac))),
                            [1, MOE_IN_QUANT_T_TILE],
                        ),
                    )
                q_inv_row = pl.div(
                    pl.full([1, MOE_IN_QUANT_T_TILE], dtype=pl.FP32, value=127.0),
                    q_amax,
                )
                q_inv = pl.reshape(q_inv_row, [MOE_IN_QUANT_T_TILE, 1])
                q_scale = pl.reshape(
                    pl.recip(q_inv_row), [MOE_IN_QUANT_T_TILE, 1],
                )
                # Write per-token dequant scale ONCE per T-tile into col 0 of
                # the SCALE_W_PAD-wide window (cols 1..7 stay zero-init pad).
                x_scale_out[
                    qt0 : qt0 + MOE_IN_QUANT_T_TILE, 0:1
                ] = q_scale
                for qnb in pl.range(HIDDEN // ROUTED_GATE_K_CHUNK):
                    qn0 = qnb * ROUTED_GATE_K_CHUNK
                    qch = pl.cast(
                        pl.slice(
                            x,
                            [MOE_IN_QUANT_T_TILE, ROUTED_GATE_K_CHUNK],
                            [qt0, qn0],
                        ),
                        target_type=pl.FP32,
                    )
                    qq = pl.cast(
                        pl.row_expand_mul(qch, q_inv),
                        target_type=pl.INT32, mode="rint",
                    )
                    # INT8 cast chain mirrors the oracle npu_dynamic_quant:
                    # rint INT32 -> round FP16 -> trunc INT8.
                    qf = pl.cast(qq, target_type=pl.FP16, mode="round")
                    qi8 = pl.cast(qf, target_type=pl.INT8, mode="trunc")
                    x_i8_out[
                        qt0 : qt0 + MOE_IN_QUANT_T_TILE,
                        qn0 : qn0 + ROUTED_GATE_K_CHUNK,
                    ] = qi8
            # Scheduled InCore kernel with two Out buffers -> two return values,
            # captured by the caller via tuple assignment (InOut discipline
            # requires reading the post-call return, not the pre-call variable).
            return x_i8_out, x_scale_out

        # ---------- DEBUG per-stage dump copy (localization only) ----------
        @pl.function(type=pl.FunctionType.InCore)
        def _dbg_copy(
            self,
            src: pl.Tensor[[LOCAL_RECV_MAX, HIDDEN], pl.BF16],
            dst: pl.Out[pl.Tensor[[LOCAL_RECV_MAX, HIDDEN], pl.BF16]],
        ):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="dbg_copy"):
                for r in pl.range(LOCAL_RECV_MAX):
                    tile = pl.load(src, [r, 0], [1, HIDDEN])
                    pl.store(tile, [r, 0], dst)
            return dst

        @pl.function(type=pl.FunctionType.InCore)
        def _dbg_copy_ybuf(
            self,
            src: pld.DistributedTensor[[N_ROUTES_PER_RANK, HIDDEN], pl.BF16],
            dst: pl.Out[pl.Tensor[[N_ROUTES_PER_RANK, HIDDEN], pl.BF16]],
        ):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="dbg_copy_ybuf"):
                for r in pl.range(N_ROUTES_PER_RANK):
                    tile = pl.load(src, [r, 0], [1, HIDDEN])
                    pl.store(tile, [r, 0], dst)
            return dst

        # ---------- Per-rank orchestration ----------
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(  # noqa: PLR0913
            self,
            # Inputs
            x: pl.Tensor[[T, HIDDEN], pl.BF16],
            gate_w: pl.Tensor[[HIDDEN, N_EXPERTS], pl.FP32],
            router_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
            w_gate_r: pl.Tensor[
                [N_LOCAL_EXPERTS, HIDDEN, INTER], pl.INT8
            ],
            w_gate_r_scale: pl.Tensor[
                [N_LOCAL_EXPERTS, INTER], pl.FP32
            ],
            w_up_r: pl.Tensor[
                [N_LOCAL_EXPERTS, HIDDEN, INTER], pl.INT8
            ],
            w_up_r_scale: pl.Tensor[
                [N_LOCAL_EXPERTS, INTER], pl.FP32
            ],
            w_down_r: pl.Tensor[
                [N_LOCAL_EXPERTS, INTER, HIDDEN], pl.INT8
            ],
            w_down_r_scale: pl.Tensor[
                [N_LOCAL_EXPERTS, HIDDEN], pl.FP32
            ],
            w_gate_s: pl.Tensor[[HIDDEN, SH_INTER_LOCAL], pl.BF16],
            w_up_s: pl.Tensor[[HIDDEN, SH_INTER_LOCAL], pl.BF16],
            w_down_s: pl.Tensor[[SH_INTER_LOCAL, HIDDEN], pl.BF16],
            # Output
            moe_out: pl.Out[pl.Tensor[[T, HIDDEN], pl.BF16]],
            # Windows
            pub_counts: pld.DistributedTensor[
                [N_RANKS * N_RANKS, N_LOCAL_EXPERTS], pl.INT32
            ],
            count_done_sig: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
            recv_x: pld.DistributedTensor[
                [LOCAL_RECV_MAX, HIDDEN], pl.INT8
            ],
            recv_scale: pld.DistributedTensor[
                [LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
            ],
            data_done_sig: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
            send_buf: pld.DistributedTensor[
                [LOCAL_RECV_MAX, HIDDEN], pl.INT8
            ],
            send_scale: pld.DistributedTensor[
                [LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
            ],
            sh_tmp_window: pld.DistributedTensor[
                [T, HIDDEN], pl.BF16
            ],
            sh_signal_window: pld.DistributedTensor[
                [N_RANKS, 1], pl.INT32
            ],
            src_route_table: pld.DistributedTensor[
                [N_RANKS, N_LOCAL_EXPERTS, N_ROUTES_PER_RANK], pl.INT32
            ],
            route_pub_sig: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
            routed_y_buf: pld.DistributedTensor[
                [N_ROUTES_PER_RANK, HIDDEN], pl.BF16
            ],
            combine_done_sig: pld.DistributedTensor[
                [N_RANKS, 1], pl.INT32
            ],
            # Per-rank scalar (trails all tensor args).
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[T, HIDDEN], pl.BF16]:
            # 1) Gate (local, replicated).
            expert_indices = pl.create_tensor([T, TOPK], dtype=pl.INT32)
            expert_weights = pl.create_tensor([T, TOPK], dtype=pl.FP32)
            expert_indices, expert_weights = self.gate_step(
                x, gate_w, router_bias,
                expert_indices, expert_weights,
            )

            # 2) Shared-expert lane (TP-sliced + tp_all_reduce).
            #    Logically parallel to dispatch+expert_routed; the
            #    compiler is free to overlap. Returns the
            #    fully-reduced sh_y.
            sh_y = pl.create_tensor([T, HIDDEN], dtype=pl.BF16)
            sh_y = self.expert_shared_step(
                x, w_gate_s, w_up_s, w_down_s, sh_y,
                sh_tmp_window, sh_signal_window, my_rank,
            )

            # SERIALIZE shared tp_all_reduce -> routed dispatch (break overlap deadlock)
            x_ser_buf = pl.create_tensor([T, HIDDEN], dtype=pl.BF16)
            x = self._serialize_after_shared(x, sh_y, x_ser_buf)

            # Per-token INT8 input-quant for ROUTED experts (all routed layers
            # are W8A8). Router (gate_step) + unquantized shared expert above
            # already used ORIGINAL x.  Outputs INT8 activations + a per-token
            # FP32 dequant scale carried through dispatch alongside the INT8
            # recv_x.
            x_disp_i8 = pl.create_tensor([T, HIDDEN], dtype=pl.INT8)
            x_disp_scale = pl.create_tensor(
                [T, SCALE_W_PAD], dtype=pl.FP32,
            )
            (x_disp_i8, x_disp_scale) = self._quant_moe_input(
                x, x_disp_i8, x_disp_scale,
            )

            # 3) Dispatch (EP all-to-all).
            local_routed_x = pl.create_tensor(
                [LOCAL_RECV_MAX, HIDDEN], dtype=pl.INT8,
            )
            local_routed_x_scale = pl.create_tensor(
                [LOCAL_RECV_MAX, SCALE_W_PAD], dtype=pl.FP32,
            )
            local_expert_offset = pl.create_tensor(
                [N_LOCAL_EXPERTS], dtype=pl.INT32,
            )
            local_expert_count = pl.create_tensor(
                [N_LOCAL_EXPERTS], dtype=pl.INT32,
            )
            inverse_map = pl.create_tensor([T, TOPK], dtype=pl.INT32)
            (
                local_routed_x,
                local_routed_x_scale,
                local_expert_offset,
                local_expert_count,
                inverse_map,
            ) = self.dispatch_step(
                x_disp_i8, x_disp_scale, expert_indices,
                local_routed_x, local_routed_x_scale,
                local_expert_offset, local_expert_count, inverse_map,
                send_buf, send_scale,
                pub_counts, count_done_sig, recv_x, recv_scale, data_done_sig,
                my_rank,
            )

            # 4) Routed experts (local 36).
            local_routed_y = pl.create_tensor(
                [LOCAL_RECV_MAX, HIDDEN], dtype=pl.BF16,
            )
            local_routed_y = self.expert_routed_step(
                local_routed_x,
                local_routed_x_scale,
                local_expert_offset, local_expert_count,
                w_gate_r, w_gate_r_scale,
                w_up_r, w_up_r_scale,
                w_down_r, w_down_r_scale,
                local_routed_y,
            )

            # 5) Combine (EP a2a back + weighted gather + sh_y add).
            moe_out = self.combine_step(
                local_routed_y,
                expert_indices, expert_weights, sh_y,
                moe_out,
                pub_counts, src_route_table, route_pub_sig,
                routed_y_buf, combine_done_sig,
                my_rank,
            )
            return moe_out

        # ---------- Host orchestrator ----------
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(  # noqa: PLR0913, PLR0915
            self,
            x: pl.Tensor[[N_RANKS, T, HIDDEN], pl.BF16],
            gate_w: pl.Tensor[[N_RANKS, HIDDEN, N_EXPERTS], pl.FP32],
            router_bias: pl.Tensor[[N_RANKS, N_EXPERTS], pl.FP32],
            w_gate_r: pl.Tensor[
                [N_RANKS, N_LOCAL_EXPERTS, HIDDEN, INTER], pl.INT8
            ],
            w_gate_r_scale: pl.Tensor[
                [N_RANKS, N_LOCAL_EXPERTS, INTER], pl.FP32
            ],
            w_up_r: pl.Tensor[
                [N_RANKS, N_LOCAL_EXPERTS, HIDDEN, INTER], pl.INT8
            ],
            w_up_r_scale: pl.Tensor[
                [N_RANKS, N_LOCAL_EXPERTS, INTER], pl.FP32
            ],
            w_down_r: pl.Tensor[
                [N_RANKS, N_LOCAL_EXPERTS, INTER, HIDDEN], pl.INT8
            ],
            w_down_r_scale: pl.Tensor[
                [N_RANKS, N_LOCAL_EXPERTS, HIDDEN], pl.FP32
            ],
            w_gate_s: pl.Tensor[
                [N_RANKS, HIDDEN, SH_INTER_LOCAL], pl.BF16
            ],
            w_up_s: pl.Tensor[
                [N_RANKS, HIDDEN, SH_INTER_LOCAL], pl.BF16
            ],
            w_down_s: pl.Tensor[
                [N_RANKS, SH_INTER_LOCAL, HIDDEN], pl.BF16
            ],
            moe_out: pl.Out[pl.Tensor[[N_RANKS, T, HIDDEN], pl.BF16]],
        ):
            # Per-collective-call-site window allocations (team-lead policy:
            # "Each collective call site allocates its own [group_size, 1]
            # INT32 signal_window via pld.alloc_window_buffer").
            #
            # Wave-3 integration hoist contract: when the full decode wires
            # the 45 layers + 3 MTP layers, the TP-AR signal/scratch windows
            # (``sh_tmp_buf`` / ``sh_sig_buf``) MUST be allocated once at the
            # top-level decode ``host_orch`` and threaded down — re-allocating
            # per layer × per call site burns window-buffer budget at
            # 45-layer scale. The allocations below are scoped to the
            # standalone MoE-only harness in this module's ``__main__``
            # path; the Wave-3 integration-author swaps these for handles
            # threaded through the chip_orch signature.
            pub_counts_buf = pld.alloc_window_buffer(
                N_RANKS * N_RANKS * N_LOCAL_EXPERTS * 4,
            )
            count_done_buf = pld.alloc_window_buffer(N_RANKS * 4)
            recv_x_buf = pld.alloc_window_buffer(
                LOCAL_RECV_MAX * HIDDEN * 1,  # INT8
            )
            recv_scale_buf = pld.alloc_window_buffer(
                LOCAL_RECV_MAX * SCALE_W_PAD * 4,  # FP32
            )
            data_done_buf = pld.alloc_window_buffer(N_RANKS * 4)
            send_buf_buf = pld.alloc_window_buffer(
                LOCAL_RECV_MAX * HIDDEN * 1,  # INT8
            )
            send_scale_buf = pld.alloc_window_buffer(
                LOCAL_RECV_MAX * SCALE_W_PAD * 4,  # FP32
            )
            sh_tmp_buf = pld.alloc_window_buffer(T * HIDDEN * 2)
            sh_sig_buf = pld.alloc_window_buffer(N_RANKS * 4)
            src_route_buf = pld.alloc_window_buffer(
                N_RANKS * N_LOCAL_EXPERTS * N_ROUTES_PER_RANK * 4,
            )
            route_pub_buf = pld.alloc_window_buffer(N_RANKS * 4)
            routed_y_window_buf = pld.alloc_window_buffer(
                N_ROUTES_PER_RANK * HIDDEN * 2,
            )
            combine_done_buf = pld.alloc_window_buffer(N_RANKS * 4)

            for r in pl.range(pld.world_size()):
                pub_counts = pld.window(
                    pub_counts_buf,
                    [N_RANKS * N_RANKS, N_LOCAL_EXPERTS], dtype=pl.INT32,
                )
                count_done_sig = pld.window(
                    count_done_buf, [N_RANKS, 1], dtype=pl.INT32,
                )
                recv_x = pld.window(
                    recv_x_buf, [LOCAL_RECV_MAX, HIDDEN], dtype=pl.INT8,
                )
                recv_scale = pld.window(
                    recv_scale_buf,
                    [LOCAL_RECV_MAX, SCALE_W_PAD], dtype=pl.FP32,
                )
                data_done_sig = pld.window(
                    data_done_buf, [N_RANKS, 1], dtype=pl.INT32,
                )
                send_x = pld.window(
                    send_buf_buf, [LOCAL_RECV_MAX, HIDDEN], dtype=pl.INT8,
                )
                send_scale = pld.window(
                    send_scale_buf,
                    [LOCAL_RECV_MAX, SCALE_W_PAD], dtype=pl.FP32,
                )
                sh_tmp_window = pld.window(
                    sh_tmp_buf, [T, HIDDEN], dtype=pl.BF16,
                )
                sh_signal_window = pld.window(
                    sh_sig_buf, [N_RANKS, 1], dtype=pl.INT32,
                )
                src_route_table = pld.window(
                    src_route_buf,
                    [N_RANKS, N_LOCAL_EXPERTS, N_ROUTES_PER_RANK],
                    dtype=pl.INT32,
                )
                route_pub_sig = pld.window(
                    route_pub_buf, [N_RANKS, 1], dtype=pl.INT32,
                )
                routed_y_buf = pld.window(
                    routed_y_window_buf,
                    [N_ROUTES_PER_RANK, HIDDEN], dtype=pl.BF16,
                )
                combine_done_sig = pld.window(
                    combine_done_buf, [N_RANKS, 1], dtype=pl.INT32,
                )
                self.chip_orch(
                    x[r], gate_w[r], router_bias[r],
                    w_gate_r[r], w_gate_r_scale[r],
                    w_up_r[r], w_up_r_scale[r],
                    w_down_r[r], w_down_r_scale[r],
                    w_gate_s[r], w_up_s[r], w_down_s[r],
                    moe_out[r],
                    pub_counts, count_done_sig,
                    recv_x, recv_scale, data_done_sig,
                    send_x, send_scale,
                    sh_tmp_window, sh_signal_window,
                    src_route_table, route_pub_sig,
                    routed_y_buf, combine_done_sig,
                    r,
                    device=r,
                )

    return EpTpMoE


# Lazy specialisation cache.
#
# WHY LAZY: instantiating the @pl.program at module-import time triggers the
# pypto frontend AST parser on the class body. That parser only succeeds when
# the call sites it observes are themselves @pl.* registered callables.
# Building eagerly at import time fails before any caller has a chance to
# wire up the runtime context. The in-tree TP+EP MoE reference defers its own
# program build to ``__main__`` for the same reason; we follow that pattern.
#
# The three module-level names ``EpTpMoE_silu_silu`` /
# ``EpTpMoE_swiglu7_silu`` / ``EpTpMoE_swiglu7_swiglu16`` remain importable
# (PEP 562 ``__getattr__``). On first attribute access we build the matching
# class and cache it; repeat access returns the same class.
_EPTPMOE_CACHE: dict[tuple[float, float], object] = {}

_LAZY_NAMES = {
    "EpTpMoE_silu_silu": (0.0, 0.0),         # layers 3..42
    "EpTpMoE_swiglu7_silu": (7.0, 0.0),      # layer 43
    "EpTpMoE_swiglu7_swiglu16": (7.0, 16.0), # layer 44
}


def _get_eptp_moe(routed_lim: float, shared_lim: float):
    key = (routed_lim, shared_lim)
    prog = _EPTPMOE_CACHE.get(key)
    if prog is None:
        prog = _build_ep_tp_moe_program(routed_lim, shared_lim)
        _EPTPMOE_CACHE[key] = prog
    return prog


def __getattr__(name: str):  # PEP 562
    if name in _LAZY_NAMES:
        return _get_eptp_moe(*_LAZY_NAMES[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def select_moe_block(layer_idx: int):
    """Pick the ``EpTpMoE`` program for ``layer_idx``.

    Returns the matching ``@pl.program`` class; raises ``ValueError``
    for layers outside the MoE range or with unsupported swiglu limits.
    Wave-3 ``decode_layer.py`` calls this and constructs/invokes the
    program inside its per-layer loop.

    The underlying ``@pl.program`` class is built lazily on first access
    and cached — so repeat lookups for the same layer specialisation
    return the same class.
    """
    routed_lim = float(SWIGLU_LIMITS[layer_idx])
    shared_lim = float(SWIGLU_LIMITS_SHARED[layer_idx])
    if routed_lim == 0.0 and shared_lim == 0.0:
        return _get_eptp_moe(0.0, 0.0)          # layers 3..42
    if routed_lim == 7.0 and shared_lim == 0.0:
        return _get_eptp_moe(7.0, 0.0)          # layer 43
    if routed_lim == 7.0 and shared_lim == 16.0:
        return _get_eptp_moe(7.0, 16.0)         # layer 44
    raise ValueError(
        f"Unsupported (routed={routed_lim}, shared={shared_lim}) for "
        f"layer {layer_idx}; expected one of (0, 0), (7, 0), (7, 16).",
    )


# =============================================================================
# Torch end-to-end reference + distributed-mock harness.
#
# Replays the full TP+EP MoE on host torch using world-wide weights, then
# compares against a per-rank loop where each rank holds only its EP/TP
# shard. The harness exercises layers 3, 4, and 44 (the three SwigluStep
# combinations) and reports the worst-element pass_rate.
# =============================================================================


def _torch_moe_ref(
    routed_swiglu_limit: float,
    shared_swiglu_limit: float,
    x,
    gate_w_full,
    router_bias_full,
    w_gate_r_full,
    w_up_r_full,
    w_down_r_full,
    w_gate_s_full,
    w_up_s_full,
    w_down_s_full,
):
    """Global single-card reference."""
    import torch
    import torch.nn.functional as F

    # 1. Router.
    logits = x.float() @ gate_w_full.float()
    score = torch.sigmoid(logits)
    biased = score + router_bias_full.float().view(1, -1)
    indices = torch.argsort(-biased, dim=-1, stable=True)[:, :TOPK]
    topk_vals = torch.gather(score, dim=-1, index=indices.long())
    weights = (topk_vals / topk_vals.sum(dim=-1, keepdim=True)) * 3.0
    weights_bf = weights.to(torch.bfloat16).float()

    # 2. Routed experts (global).
    routed_acc = torch.zeros(T, HIDDEN, dtype=torch.float32)
    for t in range(T):
        for k in range(TOPK):
            eid = int(indices[t, k].item())
            x_row = x[t : t + 1, :].float()
            gate_a = x_row @ w_gate_r_full[eid].float()
            up_a = x_row @ w_up_r_full[eid].float()
            if routed_swiglu_limit > 0.0:
                silu_g = F.silu(gate_a).clamp(max=routed_swiglu_limit)
                up_c = up_a.clamp(
                    min=-routed_swiglu_limit, max=routed_swiglu_limit,
                )
                h = silu_g * up_c
            else:
                h = F.silu(gate_a) * up_a
            h_bf = h.to(torch.bfloat16).float()
            y = h_bf @ w_down_r_full[eid].float()
            routed_acc[t, :] += weights_bf[t, k] * y[0]

    # 3. Shared expert.
    sh_gate = x.float() @ w_gate_s_full.float()
    sh_up = x.float() @ w_up_s_full.float()
    if shared_swiglu_limit > 0.0:
        sh_silu = F.silu(sh_gate).clamp(max=shared_swiglu_limit)
        sh_up_c = sh_up.clamp(
            min=-shared_swiglu_limit, max=shared_swiglu_limit,
        )
        sh_h = sh_silu * sh_up_c
    else:
        sh_h = F.silu(sh_gate) * sh_up
    sh_h_bf = sh_h.to(torch.bfloat16).float()
    sh_out = sh_h_bf @ w_down_s_full.float()

    return (sh_out + routed_acc).to(torch.bfloat16)


def _distributed_mock_check(layer_idx: int, seed: int = 0) -> float:
    """Mock 8-rank EpTpMoE end-to-end; return per-element pass_rate.

    Builds world-wide weights, then runs a torch loop where each rank
    holds only its EP/TP shard. The per-rank outputs are aggregated
    (each rank produces the same ``moe_out`` after combine) and
    compared to the global reference.

    Uses a *tiny* per-rank routed INTER (= 256) and a tiny shared lane
    (= 64) to keep the harness fast — the real run uses INTER=1280 and
    SH_INTER_LOCAL=160. The contract check is purely shape/semantic.
    """
    import torch
    import torch.nn.functional as F

    routed_lim = float(SWIGLU_LIMITS[layer_idx])
    shared_lim = float(SWIGLU_LIMITS_SHARED[layer_idx])

    gen = torch.Generator().manual_seed(seed)
    inter_mock = 256
    sh_lane_mock = 64
    sh_total = sh_lane_mock * TP_WORLD_SIZE

    x = (torch.randn(T, HIDDEN, generator=gen) * 0.3).to(torch.bfloat16)
    gate_w = (
        torch.randn(HIDDEN, N_EXPERTS_GLOBAL, generator=gen) / HIDDEN ** 0.5
    ).float()
    router_bias = (
        torch.randn(N_EXPERTS_GLOBAL, generator=gen) * 0.05
    ).float()

    w_gate_r = (
        torch.randn(N_EXPERTS_GLOBAL, HIDDEN, inter_mock, generator=gen)
        / HIDDEN ** 0.5
    ).to(torch.bfloat16)
    w_up_r = (
        torch.randn(N_EXPERTS_GLOBAL, HIDDEN, inter_mock, generator=gen)
        / HIDDEN ** 0.5
    ).to(torch.bfloat16)
    w_down_r = (
        torch.randn(N_EXPERTS_GLOBAL, inter_mock, HIDDEN, generator=gen)
        / inter_mock ** 0.5
    ).to(torch.bfloat16)

    w_gate_s = (
        torch.randn(HIDDEN, sh_total, generator=gen) / HIDDEN ** 0.5
    ).to(torch.bfloat16)
    w_up_s = (
        torch.randn(HIDDEN, sh_total, generator=gen) / HIDDEN ** 0.5
    ).to(torch.bfloat16)
    w_down_s = (
        torch.randn(sh_total, HIDDEN, generator=gen) / sh_total ** 0.5
    ).to(torch.bfloat16)

    moe_ref = _torch_moe_ref(
        routed_lim, shared_lim,
        x, gate_w, router_bias,
        w_gate_r, w_up_r, w_down_r,
        w_gate_s, w_up_s, w_down_s,
    )

    # Per-rank mock: each rank holds its EP/TP shard, then the cross-rank
    # reductions are simulated by summing the contributions.
    logits = x.float() @ gate_w
    score = torch.sigmoid(logits)
    biased = score + router_bias.view(1, -1)
    indices = torch.argsort(-biased, dim=-1, stable=True)[:, :TOPK]
    topk_vals = torch.gather(score, dim=-1, index=indices.long())
    weights = (topk_vals / topk_vals.sum(dim=-1, keepdim=True)) * 3.0
    weights_bf = weights.to(torch.bfloat16).float()

    routed_y_per_route = torch.zeros(T, TOPK, HIDDEN, dtype=torch.float32)
    for t in range(T):
        for k in range(TOPK):
            eid = int(indices[t, k].item())
            x_row = x[t : t + 1, :].float()
            gate_a = x_row @ w_gate_r[eid].float()
            up_a = x_row @ w_up_r[eid].float()
            if routed_lim > 0.0:
                silu_g = F.silu(gate_a).clamp(max=routed_lim)
                up_c = up_a.clamp(min=-routed_lim, max=routed_lim)
                h = silu_g * up_c
            else:
                h = F.silu(gate_a) * up_a
            h_bf = h.to(torch.bfloat16).float()
            y = h_bf @ w_down_r[eid].float()
            routed_y_per_route[t, k, :] = y[0]

    # Shared: TP-sliced + summed (the tp_all_reduce homogenises across ranks).
    sh_reduced = torch.zeros(T, HIDDEN, dtype=torch.float32)
    for r in range(TP_WORLD_SIZE):
        col_lo = r * sh_lane_mock
        col_hi = (r + 1) * sh_lane_mock
        wgs = w_gate_s[:, col_lo:col_hi]
        wus = w_up_s[:, col_lo:col_hi]
        wds = w_down_s[col_lo:col_hi, :]
        gate_l = x.float() @ wgs.float()
        up_l = x.float() @ wus.float()
        if shared_lim > 0.0:
            silu_l = F.silu(gate_l).clamp(max=shared_lim)
            up_lc = up_l.clamp(min=-shared_lim, max=shared_lim)
            h_l = silu_l * up_lc
        else:
            h_l = F.silu(gate_l) * up_l
        h_lbf = h_l.to(torch.bfloat16).float()
        sh_shard = h_lbf @ wds.float()
        sh_reduced += sh_shard

    moe_per_rank = sh_reduced.clone()
    for t in range(T):
        for k in range(TOPK):
            moe_per_rank[t, :] += weights_bf[t, k] * routed_y_per_route[t, k, :]
    moe_per_rank_bf = moe_per_rank.to(torch.bfloat16)

    diff = (moe_per_rank_bf.float() - moe_ref.float()).abs()
    tol = 5e-2 + 5e-2 * moe_ref.float().abs()
    matches = int((diff <= tol).sum().item())
    return matches / (T * HIDDEN)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Step3p5 MoE block end-to-end (TP=EP=8) — gate → dispatch → "
            "expert_routed (36/card) → combine (with TP-reduced shared). "
            "Exercises layer 3 (silu/silu), layer 4 (silu/silu), and "
            "layer 44 (swiglu7/swiglu16) via the distributed-mock "
            "harness."
        ),
    )
    parser.add_argument("--layers", default="3,4,44",
                        help="Comma-separated layer indices to run.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    layers = [int(s) for s in args.layers.split(",") if s.strip()]
    worst = 1.0
    for layer_idx in layers:
        # Sanity: program builder accepts the layer (validates the
        # specialisation table).
        _ = select_moe_block(layer_idx)
        pass_rate = _distributed_mock_check(
            layer_idx, seed=args.seed + layer_idx,
        )
        print(
            f"[moe.py] layer {layer_idx}: "
            f"routed={SWIGLU_LIMITS[layer_idx]}, "
            f"shared={SWIGLU_LIMITS_SHARED[layer_idx]}, "
            f"pass_rate={pass_rate:.4f}",
        )
        worst = min(worst, pass_rate)
    if worst < 0.97:
        raise SystemExit(1)


__all__ = [
    "EpTpMoE_silu_silu",
    "EpTpMoE_swiglu7_silu",
    "EpTpMoE_swiglu7_swiglu16",
    "select_moe_block",
    "T",
    "HIDDEN",
    "N_EXPERTS",
    "N_EXPERTS_GLOBAL",
    "N_LOCAL_EXPERTS",
    "TOPK",
    "INTER",
    "SH_INTER_LOCAL",
    "N_RANKS",
    "N_ROUTES_PER_RANK",
    "LOCAL_RECV_MAX",
    "SH_TP_CHUNK",
]
