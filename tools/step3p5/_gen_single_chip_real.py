# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Generate the experimental single-chip-launch whole-decode builder.

The canonical ``WholeDecodeFaithfulReal`` program is one Python/PyPTO program,
but its host orchestrator calls one chip orchestration per transformer layer
plus one LM-head orchestration.  Distributed lowering therefore emits 46
rank-local ``_submit_chip`` calls.

This generator derives ``WholeDecodeFaithfulRealSingleChip`` from the validated
builder without changing any compute/collective body:

* the five layer/tail helpers remain CHIP Orchestration and are marked ``inline_orchestration``; the compiler expands them only after outlining;
* one new ``whole_chip_orch`` source-unrolls all 45 layers and the tail;
* the host orchestrator allocates the same per-layer-distinct communication
  storage, but stacks buffers by category and submits ``whole_chip_orch`` once
  per rank;
* every control-signal slice retains a 512-byte stride, so no two layers share
  a cache line;
* native INT8 routed-expert weights/scales and the pull dispatch/combine path
  are inherited byte-for-byte from the validated builder.

The generated builder is written to ``decode_layer_single_chip.py`` from the
tracked ``decode_layer.py`` source of truth. Generation is deterministic and
round-tripped before the output is accepted. The canonical default remains
42 MoE layers; ``SINGLE_CHIP_MOE_LAYERS`` is diagnostic-only.
"""
from __future__ import annotations

import re
import os
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "models" / "step3p5" / "decode_layer.py"
OUTPUT = REPO / "models" / "step3p5" / "decode_layer_single_chip.py"

# Keep only shared imports/constants/helpers in the generated module. The
# generated single-submit module must not re-export the legacy 46-submit
# whole-net builders from decode_layer.py.
HEADER_STOP_DEF = "def _build_decode_layer_dense_program("

SOURCE_DEF = "def _build_whole_decode_faithful_real_program("
SOURCE_BIND = (
    "\nwhole_decode_faithful_real = "
    "_build_whole_decode_faithful_real_program()"
)
GENERATED_DEF = "def _build_whole_decode_faithful_real_single_chip_program("
GENERATED_BIND = (
    "\nwhole_decode_faithful_real_single_chip = "
    "_build_whole_decode_faithful_real_single_chip_program()"
)

N_DENSE = 3
N_MOE = 42
SIGNAL_STRIDE_I32 = "COMM_SIGNAL_STRIDE_I32"


def _requested_moe_layers() -> int:
    value = int(os.environ.get("SINGLE_CHIP_MOE_LAYERS", str(N_MOE)))
    if not 0 <= value <= N_MOE:
        raise ValueError(
            f"SINGLE_CHIP_MOE_LAYERS must be in [0, {N_MOE}], got {value}"
        )
    return value


WINDOWS = [
    # name, type shape, dtype, allocation bytes
    (
        "dense_attn_tmp_stack",
        "[WHOLE_CHIP_DENSE_LAYERS * BATCH, HIDDEN]",
        "pl.BF16",
        "WHOLE_CHIP_DENSE_LAYERS * BATCH * HIDDEN * 2",
    ),
    (
        "dense_attn_signal_stack",
        "[WHOLE_CHIP_DENSE_LAYERS * COMM_SIGNAL_STRIDE_I32, 1]",
        "pl.INT32",
        "WHOLE_CHIP_DENSE_LAYERS * COMM_CONTROL_SIGNAL_BYTES",
    ),
    (
        "dense_mlp_tmp_stack",
        "[WHOLE_CHIP_DENSE_LAYERS * BATCH, HIDDEN]",
        "pl.BF16",
        "WHOLE_CHIP_DENSE_LAYERS * BATCH * HIDDEN * 2",
    ),
    (
        "dense_mlp_signal_stack",
        "[WHOLE_CHIP_DENSE_LAYERS * COMM_SIGNAL_STRIDE_I32, 1]",
        "pl.INT32",
        "WHOLE_CHIP_DENSE_LAYERS * COMM_CONTROL_SIGNAL_BYTES",
    ),
    (
        "moe_attn_tmp_stack",
        "[WHOLE_CHIP_MOE_LAYERS * BATCH, HIDDEN]",
        "pl.BF16",
        "WHOLE_CHIP_MOE_LAYERS * BATCH * HIDDEN * 2",
    ),
    (
        "moe_attn_signal_stack",
        "[WHOLE_CHIP_MOE_LAYERS * COMM_SIGNAL_STRIDE_I32, 1]",
        "pl.INT32",
        "WHOLE_CHIP_MOE_LAYERS * COMM_CONTROL_SIGNAL_BYTES",
    ),
    (
        "moe_pub_counts_stack",
        "[WHOLE_CHIP_MOE_LAYERS * n_ranks * n_ranks, "
        "n_local_experts_pad]",
        "pl.INT32",
        "WHOLE_CHIP_MOE_LAYERS * n_ranks * n_ranks * "
        "n_local_experts_pad * 4",
    ),
    (
        "moe_count_done_stack",
        "[WHOLE_CHIP_MOE_LAYERS * COMM_SIGNAL_STRIDE_I32, 1]",
        "pl.INT32",
        "WHOLE_CHIP_MOE_LAYERS * COMM_CONTROL_SIGNAL_BYTES",
    ),
    (
        "moe_recv_x_stack",
        "[WHOLE_CHIP_MOE_LAYERS * local_recv_max, HIDDEN]",
        "pl.INT8",
        "WHOLE_CHIP_MOE_LAYERS * local_recv_max * HIDDEN",
    ),
    (
        "moe_recv_scale_stack",
        "[WHOLE_CHIP_MOE_LAYERS * local_recv_max, 8]",
        "pl.FP32",
        "WHOLE_CHIP_MOE_LAYERS * local_recv_max * 8 * 4",
    ),
    (
        "moe_recv_route_stack",
        "[WHOLE_CHIP_MOE_LAYERS * local_recv_max, idx_pad]",
        "pl.INT32",
        "WHOLE_CHIP_MOE_LAYERS * local_recv_max * idx_pad * 4",
    ),
    (
        "moe_data_done_stack",
        "[WHOLE_CHIP_MOE_LAYERS * COMM_SIGNAL_STRIDE_I32, 1]",
        "pl.INT32",
        "WHOLE_CHIP_MOE_LAYERS * COMM_CONTROL_SIGNAL_BYTES",
    ),
    (
        "moe_send_x_stack",
        "[WHOLE_CHIP_MOE_LAYERS * local_recv_max, HIDDEN]",
        "pl.INT8",
        "WHOLE_CHIP_MOE_LAYERS * local_recv_max * HIDDEN",
    ),
    (
        "moe_send_scale_stack",
        "[WHOLE_CHIP_MOE_LAYERS * local_recv_max, 8]",
        "pl.FP32",
        "WHOLE_CHIP_MOE_LAYERS * local_recv_max * 8 * 4",
    ),
    (
        "moe_send_route_stack",
        "[WHOLE_CHIP_MOE_LAYERS * local_recv_max, idx_pad]",
        "pl.INT32",
        "WHOLE_CHIP_MOE_LAYERS * local_recv_max * idx_pad * 4",
    ),
    (
        "moe_shared_tmp_stack",
        "[WHOLE_CHIP_MOE_LAYERS * BATCH, HIDDEN]",
        "pl.BF16",
        "WHOLE_CHIP_MOE_LAYERS * BATCH * HIDDEN * 2",
    ),
    (
        "moe_shared_signal_stack",
        "[WHOLE_CHIP_MOE_LAYERS * COMM_SIGNAL_STRIDE_I32, 1]",
        "pl.INT32",
        "WHOLE_CHIP_MOE_LAYERS * COMM_CONTROL_SIGNAL_BYTES",
    ),
    (
        "moe_routed_y_stack",
        "[WHOLE_CHIP_MOE_LAYERS * n_routes_per_rank, HIDDEN]",
        "pl.BF16",
        "WHOLE_CHIP_MOE_LAYERS * n_routes_per_rank * HIDDEN * 2",
    ),
    (
        "moe_combine_done_stack",
        "[WHOLE_CHIP_MOE_LAYERS * COMM_SIGNAL_STRIDE_I32, 1]",
        "pl.INT32",
        "WHOLE_CHIP_MOE_LAYERS * COMM_CONTROL_SIGNAL_BYTES",
    ),
    (
        "moe_routed_src_stack",
        "[WHOLE_CHIP_MOE_LAYERS * local_recv_max, HIDDEN]",
        "pl.BF16",
        "WHOLE_CHIP_MOE_LAYERS * local_recv_max * HIDDEN * 2",
    ),
]


# Whole-chip parameters must be at most 2-D at the orchestration boundary.
# Scalar-indexing an N-D tensor inside an inlined layer creates an N-D
# ``tile.slice`` after tensor-to-tile conversion; the 0162 compiler rejects
# that in ``FlattenTileNdTo2D``.  The host therefore passes zero-copy flattened
# views, and the whole-chip body takes explicit 1-D/2-D layer views.
FLAT_PARAM_SHAPES = {
    "full_wq": "[12 * HIDDEN, hidden_q_full]",
    "full_wk": "[12 * HIDDEN, KV_HIDDEN_LOCAL]",
    "full_wv": "[12 * HIDDEN, KV_HIDDEN_LOCAL]",
    "full_wo": "[12 * hidden_q_full, HIDDEN]",
    "full_w_g": "[12 * HIDDEN, nh_full_pad]",
    "full_gate_r": "[12 * nh_full_pad, hidden_q_full]",
    "swa_wq": "[33 * HIDDEN, hidden_q_swa]",
    "swa_wk": "[33 * HIDDEN, KV_HIDDEN_LOCAL]",
    "swa_wv": "[33 * HIDDEN, KV_HIDDEN_LOCAL]",
    "swa_wo": "[33 * hidden_q_swa, HIDDEN]",
    "swa_w_g": "[33 * HIDDEN, nh_swa_pad]",
    "swa_gate_r": "[33 * nh_swa_pad, hidden_q_swa]",
    "dense_w_gate": "[3 * HIDDEN, INTER_LOCAL]",
    "dense_w_up": "[3 * HIDDEN, INTER_LOCAL]",
    "dense_w_down": "[3 * INTER_LOCAL, HIDDEN]",
    "moe_gate_w": "[42 * HIDDEN, N_EXPERTS]",
    "moe_router_bias": "[42 * N_EXPERTS]",
    "moe_w_gate_r": "[42 * n_local_experts * HIDDEN, inter]",
    "moe_w_gate_r_scale": "[42 * n_local_experts, inter]",
    "moe_w_up_r": "[42 * n_local_experts * HIDDEN, inter]",
    "moe_w_up_r_scale": "[42 * n_local_experts, inter]",
    "moe_w_down_r": "[42 * n_local_experts * inter, HIDDEN]",
    "moe_w_down_r_scale": "[42 * n_local_experts, HIDDEN]",
    "moe_w_gate_s": "[42 * HIDDEN, sh_inter_local]",
    "moe_w_up_s": "[42 * HIDDEN, sh_inter_local]",
    "moe_w_down_s": "[42 * sh_inter_local, HIDDEN]",
}


def _flatten_signature_params(signature: str) -> str:
    for name, shape in FLAT_PARAM_SHAPES.items():
        pattern = re.compile(
            rf"^(            {name}: pl\.Tensor)\[\[[^\n]+\], "
            rf"(pl\.[A-Z0-9]+)\],$",
            re.M,
        )
        signature, count = pattern.subn(
            rf"\1[{shape}, \2],",
            signature,
        )
        if count != 1:
            raise RuntimeError(
                f"expected one per-rank signature line for {name}, got {count}"
            )
    return signature


def _host_param_expr(name: str) -> str:
    shape = FLAT_PARAM_SHAPES.get(name)
    if shape is None:
        return f"{name}[r]"
    return f"pl.reshape({name}[r], {shape})"


def _slice_2d(
    name: str,
    rows: str,
    cols: str,
    pos: int,
) -> str:
    return (
        f"pl.slice({name}, [{rows}, {cols}], "
        f"[{pos} * ({rows}), 0])"
    )


def _slice_1d(name: str, size: str, pos: int) -> str:
    return f"pl.slice({name}, [{size}], [{pos} * ({size})])"


def _layer_param(name: str, pos: int) -> str:
    """Build the rank-local view consumed by one inlined layer helper."""
    simple_2d = {
        "full_wq": ("HIDDEN", "hidden_q_full"),
        "full_wk": ("HIDDEN", "KV_HIDDEN_LOCAL"),
        "full_wv": ("HIDDEN", "KV_HIDDEN_LOCAL"),
        "full_wo": ("hidden_q_full", "HIDDEN"),
        "full_w_g": ("HIDDEN", "nh_full_pad"),
        "full_gate_r": ("nh_full_pad", "hidden_q_full"),
        "swa_wq": ("HIDDEN", "hidden_q_swa"),
        "swa_wk": ("HIDDEN", "KV_HIDDEN_LOCAL"),
        "swa_wv": ("HIDDEN", "KV_HIDDEN_LOCAL"),
        "swa_wo": ("hidden_q_swa", "HIDDEN"),
        "swa_w_g": ("HIDDEN", "nh_swa_pad"),
        "swa_gate_r": ("nh_swa_pad", "hidden_q_swa"),
        "dense_w_gate": ("HIDDEN", "INTER_LOCAL"),
        "dense_w_up": ("HIDDEN", "INTER_LOCAL"),
        "dense_w_down": ("INTER_LOCAL", "HIDDEN"),
        "moe_gate_w": ("HIDDEN", "N_EXPERTS"),
        "moe_w_gate_r_scale": ("n_local_experts", "inter"),
        "moe_w_up_r_scale": ("n_local_experts", "inter"),
        "moe_w_down_r_scale": ("n_local_experts", "HIDDEN"),
        "moe_w_gate_s": ("HIDDEN", "sh_inter_local"),
        "moe_w_up_s": ("HIDDEN", "sh_inter_local"),
        "moe_w_down_s": ("sh_inter_local", "HIDDEN"),
    }
    if name in simple_2d:
        rows, cols = simple_2d[name]
        return _slice_2d(name, rows, cols, pos)
    if name == "moe_router_bias":
        return _slice_1d(name, "N_EXPERTS", pos)
    routed_3d = {
        "moe_w_gate_r": ("HIDDEN", "inter"),
        "moe_w_up_r": ("HIDDEN", "inter"),
        "moe_w_down_r": ("inter", "HIDDEN"),
    }
    if name in routed_3d:
        inner, cols = routed_3d[name]
        view = _slice_2d(
            name,
            f"n_local_experts * {inner}",
            cols,
            pos,
        )
        return (
            f"pl.reshape({view}, "
            f"[n_local_experts, {inner}, {cols}])"
        )
    raise KeyError(f"no single-layer view rule for {name}")


def _method_decorator_to_helper(builder: str, name: str) -> str:
    old = (
        "        @pl.function(type=pl.FunctionType.Orchestration)\n"
        f"        def {name}("
    )
    new = (
        "        @pl.function(\n"
        "            type=pl.FunctionType.Orchestration,\n"
        '            attrs={"inline_orchestration": True},\n'
        "        )\n"
        f"        def {name}("
    )
    if builder.count(old) != 1:
        raise RuntimeError(
        f"expected one orchestration decorator for {name}, "
        f"found {builder.count(old)}"
        )
    return builder.replace(old, new, 1)


def _host_signature(builder: str) -> tuple[str, list[str], int]:
    marker = (
        "        @pl.function(level=pl.Level.HOST, "
        "role=pl.Role.Orchestrator)\n"
        "        def host_orch("
    )
    start = builder.rfind(marker)
    if start < 0:
        raise RuntimeError("active real-builder host_orch not found")
    def_start = builder.index("        def host_orch(", start)
    close = builder.index("\n        ):", def_start) + len("\n        ):")
    signature = builder[def_start:close]
    names = re.findall(r"^            ([A-Za-z_][A-Za-z0-9_]*):", signature, re.M)
    if not names:
        raise RuntimeError("failed to parse host_orch parameter names")
    return signature, names, start


def _per_rank_signature(host_signature: str) -> str:
    signature = host_signature.replace(
        "def host_orch(", "def whole_chip_orch(", 1
    )
    signature = signature.replace("[[tp_size, ", "[[")
    signature = _flatten_signature_params(signature)
    lines = signature.splitlines()
    if lines[-1] != "        ):":
        raise RuntimeError("unexpected host signature terminator")
    lines.pop()
    for name, shape, dtype, _ in WINDOWS:
        lines.extend(
            [
                f"            {name}: pld.DistributedTensor[",
                f"                {shape}, {dtype}",
                "            ],",
            ]
        )
    lines.append("            my_rank: pl.Scalar[pl.INT32],")
    lines.append("        ):")
    return "\n".join(lines)


def _slice(
    stack: str,
    shape: str,
    offset: str,
) -> str:
    return f"pl.slice({stack}, {shape}, {offset})"


def _dense_windows(pos: int) -> tuple[str, str, str, str]:
    return (
        _slice(
            "dense_attn_tmp_stack",
            "[BATCH, HIDDEN]",
            f"[{pos} * BATCH, 0]",
        ),
        _slice(
            "dense_attn_signal_stack",
            "[tp_size, 1]",
            f"[{pos} * {SIGNAL_STRIDE_I32}, 0]",
        ),
        _slice(
            "dense_mlp_tmp_stack",
            "[BATCH, HIDDEN]",
            f"[{pos} * BATCH, 0]",
        ),
        _slice(
            "dense_mlp_signal_stack",
            "[tp_size, 1]",
            f"[{pos} * {SIGNAL_STRIDE_I32}, 0]",
        ),
    )


def _moe_windows(pos: int) -> list[str]:
    signal_off = f"{pos} * {SIGNAL_STRIDE_I32}"
    recv_off = f"{pos} * local_recv_max"
    return [
        _slice("moe_attn_tmp_stack", "[BATCH, HIDDEN]", f"[{pos} * BATCH, 0]"),
        _slice("moe_attn_signal_stack", "[tp_size, 1]", f"[{signal_off}, 0]"),
        _slice(
            "moe_pub_counts_stack",
            "[n_ranks * n_ranks, n_local_experts_pad]",
            f"[{pos} * n_ranks * n_ranks, 0]",
        ),
        _slice("moe_count_done_stack", "[n_ranks, 1]", f"[{signal_off}, 0]"),
        _slice("moe_recv_x_stack", "[local_recv_max, HIDDEN]", f"[{recv_off}, 0]"),
        _slice("moe_recv_scale_stack", "[local_recv_max, 8]", f"[{recv_off}, 0]"),
        _slice("moe_data_done_stack", "[n_ranks, 1]", f"[{signal_off}, 0]"),
        _slice(
            "moe_recv_route_stack",
            "[local_recv_max, idx_pad]",
            f"[{recv_off}, 0]",
        ),
        _slice("moe_send_x_stack", "[local_recv_max, HIDDEN]", f"[{recv_off}, 0]"),
        _slice("moe_send_scale_stack", "[local_recv_max, 8]", f"[{recv_off}, 0]"),
        _slice(
            "moe_send_route_stack",
            "[local_recv_max, idx_pad]",
            f"[{recv_off}, 0]",
        ),
        _slice(
            "moe_shared_tmp_stack",
            "[BATCH, HIDDEN]",
            f"[{pos} * BATCH, 0]",
        ),
        _slice("moe_shared_signal_stack", "[n_ranks, 1]", f"[{signal_off}, 0]"),
        _slice(
            "moe_routed_y_stack",
            "[n_routes_per_rank, HIDDEN]",
            f"[{pos} * n_routes_per_rank, 0]",
        ),
        _slice("moe_combine_done_stack", "[n_ranks, 1]", f"[{signal_off}, 0]"),
        _slice(
            "moe_routed_src_stack",
            "[local_recv_max, HIDDEN]",
            f"[{recv_off}, 0]",
        ),
    ]


def _emit_call(lines: list[str], indent: str, text: str) -> None:
    lines.append(f"{indent}{text}")


def _whole_chip_body(signature: str, moe_layers: int) -> str:
    lines = [
        "        @pl.function(type=pl.FunctionType.Orchestration)",
        signature,
        "            # One rank-local task graph. Every communication argument below",
        "            # is a distinct window allocation; signal allocations are 512 B",
        "            # so AtomicAdd/TWAIT traffic cannot alias another layer.",
        "            h_layer_0 = pl.create_tensor(",
        "                [BATCH, HIDDEN], dtype=pl.BF16",
        "            )",
    ]

    a_tmp, a_sig, m_tmp, m_sig = _dense_windows(0)
    _emit_call(lines, "            ", "h_layer_0 = self.full_chip_orch(")
    for arg in [
        "current_hidden",
        "input_rms",
        _layer_param("full_wq", 0),
        _layer_param("full_wk", 0),
        _layer_param("full_wv", 0),
        "q_norm",
        "k_norm",
        "seq_lens",
        "block_table",
        "slot_mapping",
        "rope_cos_full",
        "rope_sin_full",
        "k_cache",
        "v_cache",
        _layer_param("full_wo", 0),
        _layer_param("full_w_g", 0),
        _layer_param("full_gate_r", 0),
        "post_rms",
        _layer_param("dense_w_gate", 0),
        _layer_param("dense_w_up", 0),
        _layer_param("dense_w_down", 0),
        "h_layer_0",
        a_tmp,
        a_sig,
        m_tmp,
        m_sig,
        "0",
        "0",
        "0",
        "my_rank",
    ]:
        _emit_call(lines, "                ", f"{arg},")
    _emit_call(lines, "            ", ")")

    a_tmp, a_sig, m_tmp, m_sig = _dense_windows(1)
    _emit_call(lines, "            ", "h_mid_out = self.swa_chip_orch(")
    for arg in [
        "h_layer_0",
        "input_rms",
        _layer_param("swa_wq", 0),
        _layer_param("swa_wk", 0),
        _layer_param("swa_wv", 0),
        "q_norm",
        "k_norm",
        "seq_lens",
        "block_table",
        "slot_mapping",
        "rope_cos_swa",
        "rope_sin_swa",
        "k_cache",
        "v_cache",
        _layer_param("swa_wo", 0),
        _layer_param("swa_w_g", 0),
        _layer_param("swa_gate_r", 0),
        "post_rms",
        _layer_param("dense_w_gate", 1),
        _layer_param("dense_w_up", 1),
        _layer_param("dense_w_down", 1),
        "h_mid_out",
        a_tmp,
        a_sig,
        m_tmp,
        m_sig,
        "1",
        "0",
        "0",
        "my_rank",
    ]:
        _emit_call(lines, "                ", f"{arg},")
    _emit_call(lines, "            ", ")")

    lines.extend(
        [
            "            h_layer_2 = pl.create_tensor(",
            "                [BATCH, HIDDEN], dtype=pl.BF16",
            "            )",
        ]
    )
    a_tmp, a_sig, m_tmp, m_sig = _dense_windows(2)
    _emit_call(lines, "            ", "h_layer_2 = self.swa_chip_orch(")
    for arg in [
        "h_mid_out",
        "input_rms",
        _layer_param("swa_wq", 1),
        _layer_param("swa_wk", 1),
        _layer_param("swa_wv", 1),
        "q_norm",
        "k_norm",
        "seq_lens",
        "block_table",
        "slot_mapping",
        "rope_cos_swa",
        "rope_sin_swa",
        "k_cache",
        "v_cache",
        _layer_param("swa_wo", 1),
        _layer_param("swa_w_g", 1),
        _layer_param("swa_gate_r", 1),
        "post_rms",
        _layer_param("dense_w_gate", 2),
        _layer_param("dense_w_up", 2),
        _layer_param("dense_w_down", 2),
        "h_layer_2",
        a_tmp,
        a_sig,
        m_tmp,
        m_sig,
        "2",
        "0",
        "0",
        "my_rank",
    ]:
        _emit_call(lines, "                ", f"{arg},")
    _emit_call(lines, "            ", ")")

    previous = "h_layer_2"
    for layer in range(3, 3 + moe_layers):
        pos = layer - 3
        is_full = layer % 4 == 0
        attn_pos = layer // 4 if is_full else layer - (layer // 4) - 1
        prefix = "full" if is_full else "swa"
        method = f"{prefix}_moe_chip_orch"
        if layer == 44:
            destination = "next_hidden_out"
            debug_destination = "dbg_out"
        else:
            destination = f"h_layer_{layer}"
            debug_destination = f"dbg_layer_{layer}"
            lines.extend(
                [
                    f"            {destination} = pl.create_tensor(",
                    "                [BATCH, HIDDEN], dtype=pl.BF16",
                    "            )",
                    f"            {debug_destination} = pl.create_tensor(",
                    "                [BATCH, HIDDEN], dtype=pl.BF16",
                    "            )",
                ]
            )
        resid = f"resid_hold_layer_{layer}"
        lines.extend(
            [
                f"            {resid} = pl.create_tensor(",
                "                [BATCH, HIDDEN], dtype=pl.BF16",
                "            )",
            ]
        )
        args = [
            previous,
            "input_rms",
            _layer_param(f"{prefix}_wq", attn_pos),
            _layer_param(f"{prefix}_wk", attn_pos),
            _layer_param(f"{prefix}_wv", attn_pos),
            "q_norm",
            "k_norm",
            "seq_lens",
            "block_table",
            "slot_mapping",
            f"rope_cos_{prefix}",
            f"rope_sin_{prefix}",
            "k_cache",
            "v_cache",
            _layer_param(f"{prefix}_wo", attn_pos),
            _layer_param(f"{prefix}_w_g", attn_pos),
            _layer_param(f"{prefix}_gate_r", attn_pos),
            "post_rms",
            _layer_param("moe_gate_w", pos),
            _layer_param("moe_router_bias", pos),
            _layer_param("moe_w_gate_r", pos),
            _layer_param("moe_w_gate_r_scale", pos),
            _layer_param("moe_w_up_r", pos),
            _layer_param("moe_w_up_r_scale", pos),
            _layer_param("moe_w_down_r", pos),
            _layer_param("moe_w_down_r_scale", pos),
            _layer_param("moe_w_gate_s", pos),
            _layer_param("moe_w_up_s", pos),
            _layer_param("moe_w_down_s", pos),
            destination,
            debug_destination,
            resid,
            *_moe_windows(pos),
            str(layer),
            "0",
            "my_rank",
        ]
        _emit_call(lines, "            ", f"{destination} = self.{method}(")
        for arg in args:
            _emit_call(lines, "                ", f"{arg},")
        _emit_call(lines, "            ", ")")
        previous = destination

    lines.extend(
        [
            "            logits_shard_out = self.lm_head_orch(",
            f"                {previous}, final_norm_weight,",
            "                lm_head_weight, seq_lens, logits_shard_out,",
            "            )",
            "            return logits_shard_out",
        ]
    )
    return "\n".join(lines)


def _host_body(host_signature: str, param_names: list[str]) -> str:
    lines = [
        "        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)",
        host_signature,
        "            # Buffers are stacked by semantic category, not reused by",
        "            # layer. Every per-layer slice therefore has a distinct",
        "            # address. All category sizes are multiples of 512 bytes,",
        "            # preserving both category-base and signal-slot alignment.",
    ]
    for name, _, _, nbytes in WINDOWS:
        lines.append(
            f"            {name}_buf = pld.alloc_window_buffer({nbytes})"
        )
    lines.append("            for r in pl.range(pld.world_size()):")
    lines.append("                self.whole_chip_orch(")
    for name in param_names:
        lines.append(f"                    {_host_param_expr(name)},")
    for name, shape, dtype, _ in WINDOWS:
        lines.extend(
            [
                f"                    pld.window({name}_buf, {shape},",
                f"                               dtype={dtype}),",
            ]
        )
    lines.extend(
        [
            "                    r,",
            "                    device=r,",
            "                )",
        ]
    )
    return "\n".join(lines)


def _count_base_params(signature: str) -> tuple[int, int, int]:
    """Return (total, tensor, scalar) for non-window parameters."""
    total = tensor = scalar = 0
    for line in signature.splitlines():
        if ": pl.Scalar[" in line:
            total += 1
            scalar += 1
        elif ": pl.Tensor[" in line or ": pl.Out[pl.Tensor[" in line:
            total += 1
            tensor += 1
    return total, tensor, scalar


def _audit_layout(moe_layers: int) -> None:
    """Audit arena physical sizes and 512-byte storage alignment."""
    values = {
        "BATCH": 16,
        "HIDDEN": 4096,
        "tp_size": 8,
        "n_ranks": 8,
        "n_local_experts_pad": 40,
        "local_recv_max": 1024,
        "idx_pad": 8,
        "n_routes_per_rank": 128,
        "COMM_CONTROL_SIGNAL_BYTES": 512,
        "COMM_SIGNAL_STRIDE_I32": 128,
        "WHOLE_CHIP_DENSE_LAYERS": N_DENSE,
        # Diagnostic layer-count reduction must not change the physical ABI or
        # arena ledger. Only the emitted layer calls change.
        "WHOLE_CHIP_MOE_LAYERS": N_MOE,
    }
    for name, shape, dtype, alloc in WINDOWS:
        dims = [x.strip() for x in shape.strip("[]").replace("\n", " ").split(",") if x.strip()]
        if len(dims) != 2:
            raise RuntimeError(f"arena {name} must expose a 2-D storage shape: {shape}")
        def ev(expr: str) -> int:
            return int(eval(expr, {"__builtins__": {}}, values))
        rows, cols = ev(dims[0]), ev(dims[1])
        item_bytes = {"pl.BF16": 2, "pl.FP32": 4, "pl.INT32": 4, "pl.INT8": 1}[dtype]
        physical = rows * cols * item_bytes
        allocation = ev(alloc)
        if physical != allocation:
            raise RuntimeError(
                f"arena {name} allocation mismatch: shape={rows}x{cols}, "
                f"dtype={dtype}, physical={physical}, allocation={allocation}"
            )
        if physical % 512 != 0:
            raise RuntimeError(f"arena {name} is not 512-byte aligned: {physical} B")
    for name, shape, _, _ in WINDOWS:
        if "signal" in name or "done" in name:
            if "COMM_SIGNAL_STRIDE_I32" not in shape:
                raise RuntimeError(f"control arena {name} lacks 512-B signal stride")


def _abi_gate(host_signature: str, whole_signature: str, window_count: int) -> None:
    """Reject tensor/scalar ABI sizes outside the runtime envelope."""
    host_total, _, _ = _count_base_params(host_signature)
    whole_total, whole_tensor, whole_scalar = _count_base_params(whole_signature)
    # ``whole_signature`` already contains the explicit ``my_rank`` scalar,
    # while ``host_signature`` does not. DistributedTensor declarations are
    # intentionally counted below because they are multi-line annotations.
    if whole_total != host_total + 1:
        raise RuntimeError(
            f"single-chip base ABI mismatch: host={host_total}, whole={whole_total}"
        )
    actual_total = host_total + window_count + 1  # windows + my_rank
    formal_total = whole_total + window_count
    if actual_total != formal_total:
        raise RuntimeError(
            f"single-chip ABI mismatch: host actual={actual_total}, "
            f"whole-chip formal={formal_total}"
        )
    whole_tensor += window_count
    whole_scalar += window_count  # one CommCtx per DistributedTensor; my_rank already counted
    if whole_tensor > 128 or whole_scalar > 128:
        raise RuntimeError(
            f"single-chip ABI exceeds runtime limit: tensor={whole_tensor}, "
            f"scalar={whole_scalar}, limits=(128,128)"
        )


def generate() -> str:
    text = BASE.read_text()
    if GENERATED_DEF in text:
        raise RuntimeError("tracked base unexpectedly contains the generated builder")
    moe_layers = _requested_moe_layers()
    _audit_layout(moe_layers)
    source_start = text.index(SOURCE_DEF)
    source_end = text.index(SOURCE_BIND, source_start)
    builder = text[source_start:source_end]

    builder = builder.replace(SOURCE_DEF, GENERATED_DEF, 1)
    builder = builder.replace(
        "class WholeDecodeFaithfulReal:",
        "class WholeDecodeFaithfulRealSingleChip:",
        1,
    )
    builder = builder.replace(
        "return WholeDecodeFaithfulReal",
        "return WholeDecodeFaithfulRealSingleChip",
        1,
    )
    for method in (
        "lm_head_orch",
        "full_chip_orch",
        "swa_chip_orch",
        "full_moe_chip_orch",
        "swa_moe_chip_orch",
    ):
        builder = _method_decorator_to_helper(builder, method)

    const_anchor = "    COMM_CONTROL_SIGNAL_BYTES = 512\n"
    if builder.count(const_anchor) != 1:
        raise RuntimeError("COMM_CONTROL_SIGNAL_BYTES anchor mismatch")
    const_replacement = (
        const_anchor
        + "    COMM_SIGNAL_STRIDE_I32 = COMM_CONTROL_SIGNAL_BYTES // 4\n"
        + f"    WHOLE_CHIP_DENSE_LAYERS = {N_DENSE}\n"
        + f"    WHOLE_CHIP_MOE_LAYERS = {N_MOE}\n"
    )
    builder = builder.replace(const_anchor, const_replacement, 1)

    host_signature, param_names, host_start = _host_signature(builder)
    per_rank = _per_rank_signature(host_signature)
    _abi_gate(host_signature, per_rank, len(WINDOWS))
    builder = (
        builder[:host_start]
        + _whole_chip_body(per_rank, moe_layers)
        + "\n"
        + _host_body(host_signature, param_names)
        + "\n\n"
        + "    return WholeDecodeFaithfulRealSingleChip\n"
    )

    header_end = text.index(HEADER_STOP_DEF)
    header = text[:header_end].rstrip() + "\n\n"
    note = '''
# N1 whole-net single-submit entry. This generated module intentionally keeps
# only the final single-submit implementation and shared helpers; legacy
# 46-submit whole-net builders remain in decode_layer.py for generator input /
# historical baseline only and are not re-exported here.
'''
    return header + note + builder + GENERATED_BIND + "\n"


def main() -> int:
    generated = generate()
    OUTPUT.write_text(generated)
    if generate() != generated:
        raise RuntimeError("single-chip generator is not deterministic")
    print(f"[gen-single-chip] wrote {OUTPUT}")
    print(f"[gen-single-chip] round-trip byte check: PASS ({len(generated)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
