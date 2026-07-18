#!/usr/bin/env python3
"""Range-scoped INT8-native W8A8 transform of the base-faithful inlined routed
MoE in models/step3p5/decode_layer.py (whole_decode_faithful program), mirroring
the validated moe.py cd3ef0d port. Operates ONLY on the
_build_whole_decode_faithful_program range so the other inlined programs
(DecodeLayerMoE / prefill / etc.) are untouched. Asserts each replacement hits
exactly once. Run on 0162 (python), then _gen_faithful_real +
_probe_whole_faithful_canonical to compile.
"""
import pathlib
import sys

P = pathlib.Path("models/step3p5/decode_layer.py")
s = P.read_text()

B0 = s.index("def _build_whole_decode_faithful_program(")
B1 = s.index("\nwhole_decode_faithful = _build_whole_decode_faithful_program()")
head, base, tail = s[:B0], s[B0:B1], s[B1:]


def repl(old: str, new: str, n: int = 1) -> None:
    global base
    c = base.count(old)
    if c != n:
        sys.exit(f"FAIL: expected {n} match, got {c} for:\n{old[:200]}")
    base = base.replace(old, new, n)


# ---- 1. _expert_routed signature: INT8 weights + per-channel FP32 scales ----
repl(
    """            w_gate: pl.Tensor[
                [n_local_experts, HIDDEN, inter], pl.BF16
            ],
            w_up: pl.Tensor[
                [n_local_experts, HIDDEN, inter], pl.BF16
            ],
            w_down: pl.Tensor[
                [n_local_experts, inter, HIDDEN], pl.BF16
            ],
            local_routed_y: pl.Tensor[
                [local_recv_max, HIDDEN], pl.BF16
            ],
        ):""",
    """            w_gate: pl.Tensor[
                [n_local_experts, HIDDEN, inter], pl.INT8
            ],
            w_gate_scale: pl.Tensor[[n_local_experts, inter], pl.FP32],
            w_up: pl.Tensor[
                [n_local_experts, HIDDEN, inter], pl.INT8
            ],
            w_up_scale: pl.Tensor[[n_local_experts, inter], pl.FP32],
            w_down: pl.Tensor[
                [n_local_experts, inter, HIDDEN], pl.INT8
            ],
            w_down_scale: pl.Tensor[[n_local_experts, HIDDEN], pl.FP32],
            local_routed_y: pl.Tensor[
                [local_recv_max, HIDDEN], pl.BF16
            ],
        ):""",
)

# ---- 2. In-kernel per-token INT8 quant of the input tile (DeepSeek cast chain),
#         inserted right after the h_bf16 bridge create. ----
repl(
    """                        h_bf16 = pl.create_tensor(
                            [RECV_TILE, inter], dtype=pl.BF16,
                        )

                        # Gate+up projection: each SPMD block handles one N-chunk""",
    """                        h_bf16 = pl.create_tensor(
                            [RECV_TILE, inter], dtype=pl.BF16,
                        )

                        # In-kernel per-token INT8 quant of the routed input tile
                        # (DeepSeek v4 cast chain FP32->INT32 rint->FP16 round->
                        # INT8 trunc, pl.at CORE_GROUP).  x_scale_dq (per-token
                        # dequant scale) SSA-carried into the gate/up dequant.
                        x_i8 = pl.create_tensor(
                            [RECV_TILE, HIDDEN], dtype=pl.INT8,
                        )
                        with pl.at(
                            level=pl.Level.CORE_GROUP, name_hint="routed_x_quant",
                        ):
                            xe_amax = pl.full(
                                [1, RECV_TILE], dtype=pl.FP32, value=1e-4,
                            )
                            for xka in pl.range(HIDDEN // ROUTED_GATE_K_CHUNK):
                                xka0 = xka * ROUTED_GATE_K_CHUNK
                                xe_a = pl.cast(
                                    pl.slice(
                                        local_routed_x,
                                        [RECV_TILE, ROUTED_GATE_K_CHUNK],
                                        [tile_offset, xka0],
                                    ),
                                    target_type=pl.FP32,
                                )
                                xe_amax = pl.maximum(
                                    xe_amax,
                                    pl.reshape(
                                        pl.row_max(
                                            pl.maximum(xe_a, pl.neg(xe_a)),
                                        ),
                                        [1, RECV_TILE],
                                    ),
                                )
                            xe_sq_row = pl.div(
                                pl.full(
                                    [1, RECV_TILE], dtype=pl.FP32, value=127.0,
                                ),
                                xe_amax,
                            )
                            x_scale_dq = pl.reshape(
                                pl.recip(xe_sq_row), [RECV_TILE, 1],
                            )
                            xe_sq_col = pl.reshape(xe_sq_row, [RECV_TILE, 1])
                            for xkn in pl.range(HIDDEN // ROUTED_GATE_K_CHUNK):
                                xkn0 = xkn * ROUTED_GATE_K_CHUNK
                                xe_q = pl.cast(
                                    pl.slice(
                                        local_routed_x,
                                        [RECV_TILE, ROUTED_GATE_K_CHUNK],
                                        [tile_offset, xkn0],
                                    ),
                                    target_type=pl.FP32,
                                )
                                xe_scaled = pl.row_expand_mul(xe_q, xe_sq_col)
                                xe_i32 = pl.cast(
                                    xe_scaled, target_type=pl.INT32, mode="rint",
                                )
                                xe_half = pl.cast(
                                    xe_i32, target_type=pl.FP16, mode="round",
                                )
                                x_i8[
                                    :, xkn0 : xkn0 + ROUTED_GATE_K_CHUNK
                                ] = pl.cast(
                                    xe_half, target_type=pl.INT8, mode="trunc",
                                )

                        # Gate+up projection: each SPMD block handles one N-chunk""",
)

# ---- 3. gate/up: read x_i8 (tile-local), INT8xINT8 matmul, dequant ----
repl(
    """                            n0 = nb * ROUTED_GATE_N_CHUNK
                            x0 = pl.slice(
                                local_routed_x,
                                [RECV_TILE, ROUTED_GATE_K_CHUNK],
                                [tile_offset, 0],
                                valid_shape=[tile_valid, ROUTED_GATE_K_CHUNK],
                            )""",
    """                            n0 = nb * ROUTED_GATE_N_CHUNK
                            x0 = pl.slice(
                                x_i8,
                                [RECV_TILE, ROUTED_GATE_K_CHUNK],
                                [0, 0],
                                valid_shape=[tile_valid, ROUTED_GATE_K_CHUNK],
                            )""",
)
repl(
    """                                xk = pl.slice(
                                    local_routed_x,
                                    [RECV_TILE, ROUTED_GATE_K_CHUNK],
                                    [tile_offset, k0],
                                    valid_shape=[
                                        tile_valid, ROUTED_GATE_K_CHUNK,
                                    ],
                                )""",
    """                                xk = pl.slice(
                                    x_i8,
                                    [RECV_TILE, ROUTED_GATE_K_CHUNK],
                                    [0, k0],
                                    valid_shape=[
                                        tile_valid, ROUTED_GATE_K_CHUNK,
                                    ],
                                )""",
)
repl(
    """                            gate_acc = pl.matmul(x0, wg0_2d, out_dtype=pl.FP32)
                            up_acc = pl.matmul(x0, wu0_2d, out_dtype=pl.FP32)""",
    """                            gate_acc = pl.matmul(x0, wg0_2d, out_dtype=pl.INT32)
                            up_acc = pl.matmul(x0, wu0_2d, out_dtype=pl.INT32)""",
)
repl(
    """                                gate_acc = pl.matmul_acc(gate_acc, xk, wgk)
                                up_acc = pl.matmul_acc(up_acc, xk, wuk)

                            sigmoid = pl.recip(
                                pl.add(pl.exp(pl.neg(gate_acc)), 1.0),
                            )
                            silu = pl.mul(gate_acc, sigmoid)
                            if _routed_swiglu_step:
                                silu_c = pl.minimum(silu, _routed_swiglu_limit)
                                up_c = pl.maximum(
                                    pl.minimum(up_acc, _routed_swiglu_limit),
                                    -_routed_swiglu_limit,
                                )
                                gated = pl.mul(silu_c, up_c)
                            else:
                                gated = pl.mul(silu, up_acc)""",
    """                                gate_acc = pl.matmul_acc(gate_acc, xk, wgk)
                                up_acc = pl.matmul_acc(up_acc, xk, wuk)

                            # Dequant INT32 -> FP32: per-token act scale (row) x
                            # per-output-channel weight scale (col).
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
                                    x_scale_dq,
                                ),
                                wg_scale_row,
                            )
                            up_2d = pl.col_expand_mul(
                                pl.row_expand_mul(
                                    pl.cast(
                                        up_acc, target_type=pl.FP32, mode="none",
                                    ),
                                    x_scale_dq,
                                ),
                                wu_scale_row,
                            )
                            sigmoid = pl.recip(
                                pl.add(pl.exp(pl.neg(gate_2d)), 1.0),
                            )
                            silu = pl.mul(gate_2d, sigmoid)
                            if _routed_swiglu_step:
                                silu_c = pl.minimum(silu, _routed_swiglu_limit)
                                up_c = pl.maximum(
                                    pl.minimum(up_2d, _routed_swiglu_limit),
                                    -_routed_swiglu_limit,
                                )
                                gated = pl.mul(silu_c, up_c)
                            else:
                                gated = pl.mul(silu, up_2d)""",
)

# ---- 4. h_i8 requant between gate/up and down ----
repl(
    """                            h_bf16[
                                :, n0 : n0 + ROUTED_GATE_N_CHUNK
                            ] = pl.cast(gated_v, target_type=pl.BF16)

                        # Down projection: each SPMD block handles one D-chunk of""",
    """                            h_bf16[
                                :, n0 : n0 + ROUTED_GATE_N_CHUNK
                            ] = pl.cast(gated_v, target_type=pl.BF16)

                        # Per-token INT8 requant of the swiglu intermediate for the
                        # INT8 down-proj (DeepSeek v4 h_tile_i8 cast chain).
                        h_i8 = pl.create_tensor(
                            [RECV_TILE, inter], dtype=pl.INT8,
                        )
                        with pl.at(
                            level=pl.Level.CORE_GROUP, name_hint="routed_h_quant",
                        ):
                            eh_amax = pl.full(
                                [1, RECV_TILE], dtype=pl.FP32, value=1e-4,
                            )
                            for hqa in pl.range(inter // ROUTED_GATE_N_CHUNK):
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
                            h_scale_dq = pl.reshape(
                                pl.recip(eh_sq_row), [RECV_TILE, 1],
                            )
                            eh_sq_col = pl.reshape(eh_sq_row, [RECV_TILE, 1])
                            for hqn in pl.range(inter // ROUTED_GATE_N_CHUNK):
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

                        # Down projection: each SPMD block handles one D-chunk of""",
)

# ---- 5. down: read h_i8, INT8xINT8 matmul, dequant ----
repl(
    """                            d0 = db * ROUTED_DOWN_N_CHUNK
                            h0 = pl.slice(
                                h_bf16,
                                [RECV_TILE, ROUTED_DOWN_K_CHUNK],
                                [0, 0],
                                valid_shape=[
                                    tile_valid, ROUTED_DOWN_K_CHUNK,
                                ],
                            )""",
    """                            d0 = db * ROUTED_DOWN_N_CHUNK
                            h0 = pl.slice(
                                h_i8,
                                [RECV_TILE, ROUTED_DOWN_K_CHUNK],
                                [0, 0],
                                valid_shape=[
                                    tile_valid, ROUTED_DOWN_K_CHUNK,
                                ],
                            )""",
)
repl(
    """                                hk = pl.slice(
                                    h_bf16,
                                    [RECV_TILE, ROUTED_DOWN_K_CHUNK],
                                    [0, k0],
                                    valid_shape=[
                                        tile_valid, ROUTED_DOWN_K_CHUNK,
                                    ],
                                )""",
    """                                hk = pl.slice(
                                    h_i8,
                                    [RECV_TILE, ROUTED_DOWN_K_CHUNK],
                                    [0, k0],
                                    valid_shape=[
                                        tile_valid, ROUTED_DOWN_K_CHUNK,
                                    ],
                                )""",
)
repl(
    """                            y_acc = pl.matmul(h0, wd0, out_dtype=pl.FP32)""",
    """                            y_acc = pl.matmul(h0, wd0, out_dtype=pl.INT32)""",
)
repl(
    """                                y_acc = pl.matmul_acc(y_acc, hk, wdk)

                            y_v = pl.set_validshape(
                                y_acc, tile_valid, ROUTED_DOWN_N_CHUNK,
                            )""",
    """                                y_acc = pl.matmul_acc(y_acc, hk, wdk)

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
                            )""",
)

# ---- 6. expert_routed_step: INT8 weights + scales, thread to _expert_routed ----
repl(
    """            w_gate_r: pl.Tensor[
                [n_local_experts, HIDDEN, inter], pl.BF16
            ],
            w_up_r: pl.Tensor[
                [n_local_experts, HIDDEN, inter], pl.BF16
            ],
            w_down_r: pl.Tensor[
                [n_local_experts, inter, HIDDEN], pl.BF16
            ],
            local_routed_y: pl.Out[
                pl.Tensor[[local_recv_max, HIDDEN], pl.BF16]
            ],
        ) -> pl.Tensor[[local_recv_max, HIDDEN], pl.BF16]:
            local_routed_y = self._expert_routed(
                local_routed_x,
                local_expert_offset, local_expert_count,
                w_gate_r, w_up_r, w_down_r,
                local_routed_y,
            )""",
    """            w_gate_r: pl.Tensor[
                [n_local_experts, HIDDEN, inter], pl.INT8
            ],
            w_gate_r_scale: pl.Tensor[[n_local_experts, inter], pl.FP32],
            w_up_r: pl.Tensor[
                [n_local_experts, HIDDEN, inter], pl.INT8
            ],
            w_up_r_scale: pl.Tensor[[n_local_experts, inter], pl.FP32],
            w_down_r: pl.Tensor[
                [n_local_experts, inter, HIDDEN], pl.INT8
            ],
            w_down_r_scale: pl.Tensor[[n_local_experts, HIDDEN], pl.FP32],
            local_routed_y: pl.Out[
                pl.Tensor[[local_recv_max, HIDDEN], pl.BF16]
            ],
        ) -> pl.Tensor[[local_recv_max, HIDDEN], pl.BF16]:
            local_routed_y = self._expert_routed(
                local_routed_x,
                local_expert_offset, local_expert_count,
                w_gate_r, w_gate_r_scale,
                w_up_r, w_up_r_scale,
                w_down_r, w_down_r_scale,
                local_routed_y,
            )""",
)

# ---- 7. base chip_orch signature: routed weights INT8 + interleaved scales ----
repl(
    """            w_gate_r: pl.Tensor[[n_local_experts, HIDDEN, inter], pl.BF16],
            w_up_r: pl.Tensor[[n_local_experts, HIDDEN, inter], pl.BF16],
            w_down_r: pl.Tensor[[n_local_experts, inter, HIDDEN], pl.BF16],""",
    """            w_gate_r: pl.Tensor[[n_local_experts, HIDDEN, inter], pl.INT8],
            w_gate_r_scale: pl.Tensor[[n_local_experts, inter], pl.FP32],
            w_up_r: pl.Tensor[[n_local_experts, HIDDEN, inter], pl.INT8],
            w_up_r_scale: pl.Tensor[[n_local_experts, inter], pl.FP32],
            w_down_r: pl.Tensor[[n_local_experts, inter, HIDDEN], pl.INT8],
            w_down_r_scale: pl.Tensor[[n_local_experts, HIDDEN], pl.FP32],""",
)

# ---- 8. base chip_orch -> expert_routed_step call: interleave scales ----
repl(
    """                w_gate_r, w_up_r, w_down_r,""",
    """                w_gate_r, w_gate_r_scale, w_up_r, w_up_r_scale,
                w_down_r, w_down_r_scale,""",
)

# ---- 9. base host_orch m_w_*_r decls: INT8 + interleaved scales ----
repl(
    """            m_w_gate_r: pl.Tensor[[tp_size, n_local_experts, HIDDEN, inter], pl.BF16],
            m_w_up_r: pl.Tensor[[tp_size, n_local_experts, HIDDEN, inter], pl.BF16],
            m_w_down_r: pl.Tensor[[tp_size, n_local_experts, inter, HIDDEN], pl.BF16],""",
    """            m_w_gate_r: pl.Tensor[[tp_size, n_local_experts, HIDDEN, inter], pl.INT8],
            m_w_gate_r_scale: pl.Tensor[[tp_size, n_local_experts, inter], pl.FP32],
            m_w_up_r: pl.Tensor[[tp_size, n_local_experts, HIDDEN, inter], pl.INT8],
            m_w_up_r_scale: pl.Tensor[[tp_size, n_local_experts, inter], pl.FP32],
            m_w_down_r: pl.Tensor[[tp_size, n_local_experts, inter, HIDDEN], pl.INT8],
            m_w_down_r_scale: pl.Tensor[[tp_size, n_local_experts, HIDDEN], pl.FP32],""",
)

# ---- 10. base host_orch -> chip_orch calls (all): interleave m_w_*_r_scale[r] ----
n_calls = base.count("                        m_w_gate_r[r], m_w_up_r[r], m_w_down_r[r],")
if n_calls == 0:
    sys.exit("FAIL: no host_orch chip_orch routed-weight call sites found")
base = base.replace(
    "                        m_w_gate_r[r], m_w_up_r[r], m_w_down_r[r],",
    "                        m_w_gate_r[r], m_w_gate_r_scale[r], "
    "m_w_up_r[r], m_w_up_r_scale[r], m_w_down_r[r], m_w_down_r_scale[r],",
    n_calls,
)
print(f"  host_orch chip_orch call sites patched: {n_calls}")

s2 = head + base + tail
P.write_text(s2)
print("OK: base-faithful inlined _expert_routed + expert_routed_step -> INT8")
