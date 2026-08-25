# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""[中文摘要] step3p5 模型常量与拓扑(45 主层 + 3 MTP、TP=EP=8、按层 attention/MLP
类别表),以及 `is_full_attention` / `is_moe_layer` / `ep_expert_owner` 等
查询函数;是全模块共享的事实源。
[关键装饰器] 无(纯 Python 常量与函数)。
[SPMD 角色] 不参与 SPMD 自身,但提供 TP_WORLD_SIZE / EP_WORLD_SIZE / 各种
LOCAL 维度,所有 @pl.program kernel 的 shape 决策都靠这里。
[详见] 中文架构指南 §1, §9

────── 以下为英文原 docstring ──────

Step3p5 model configuration.

Source of truth for the
``step3p5_flash_release_hf_mtp3_bf16`` checkpoint shipped under
``/mnt/chensiyu-jfs/multi-hardware/models/``.

Step3p5 has the following distinguishing features:
- 45 main layers + 3 MTP next-n-predict layers
- mixed full-attention (64 heads) and sliding-attention (96 heads, win=512)
- per-layer RoPE theta and partial rotary factor (0.5 / 1.0)
- llama3 yarn rope scaling on full-attention layers only
- zero-centered q/k RMSNorm (effective gamma = stored_gamma + 1.0)
- head-wise attention gate (g_proj per-head sigmoid)
- MoE on layers 3..44 (288 experts, top-8, sigmoid routing + router_bias,
  renormalize, 1280-dim shared expert); dense MLP on layers 0..2
- SwigluStep with limit=7 on the routed-MoE active path of two specific layers
  and limit=16 on layer 44 share-expert; plain SiLU elsewhere

The active integration contract is documented in ``docs/step3p5/README.md``.
"""

from __future__ import annotations

import os

import pypto.language as pl

# -----------------------------------------------------------------------------
# Model-bound shape constants (formerly ``pl.dynamic(...)``).
#
# Step3p5's context length, layer count, and KV-cache layout are fixed by the
# model checkpoint, so these dimensions belong in the program signature as
# integer constants — not symbolic ``pl.dynamic`` placeholders. We also hit
# two pypto codegen bugs when they were dyn:
#   * ``OptimizeOrchTensors::ComputeRowMajorStrides`` returned empty for any
#     parent shape with a dyn dim, so cross-function slices lost their
#     ``TensorView(stride=…)`` annotation.
#   * The codegen added a phantom trailing ``int32_t`` per dyn-dim Var to the
#     kernel signature, which the dispatch did not pass.
# Numeric values are the static ABI constants consumed by the single-chip
# hidden-only Main and selected-MTP programs.
# -----------------------------------------------------------------------------
DEFAULT_STORAGE_BATCH_CAPACITY = 16
STORAGE_BATCH_CAPACITY = int(
    os.environ.get(
        "PYPTO_STEP3P5_STORAGE_BATCH_CAPACITY",
        str(DEFAULT_STORAGE_BATCH_CAPACITY),
    ),
)
# Current attention/LM-head kernels tile the static storage rows in groups of
# 16.  The capacity is configurable at compile time, while each invocation
# supplies its own runtime active-row count.
if STORAGE_BATCH_CAPACITY <= 0 or STORAGE_BATCH_CAPACITY % 16 != 0:
    raise ValueError(
        "PYPTO_STEP3P5_STORAGE_BATCH_CAPACITY must be a positive multiple "
        f"of current batch tile 16, got {STORAGE_BATCH_CAPACITY}",
    )
USER_BATCH_DYN = STORAGE_BATCH_CAPACITY
_LIVE_MAX_SEQ = int(os.environ.get("PYPTO_STEP3P5_MAX_SEQ", "4096"))
if _LIVE_MAX_SEQ <= 0 or _LIVE_MAX_SEQ % 128 != 0:
    raise ValueError(
        "PYPTO_STEP3P5_MAX_SEQ must be a positive multiple of block size 128, "
        f"got {_LIVE_MAX_SEQ}",
    )
_live_kv_rows_env = os.environ.get("PYPTO_STEP3P5_KV_CACHE_ROWS")
KV_NUM_LAYERS = int(os.environ.get("PYPTO_STEP3P5_KV_NUM_LAYERS", "45"))
if KV_NUM_LAYERS <= 0:
    raise ValueError(
        "PYPTO_STEP3P5_KV_NUM_LAYERS must be positive, "
        f"got {KV_NUM_LAYERS}",
    )
KV_CACHE_ROWS_DYN = int(_live_kv_rows_env or "4096")
if KV_CACHE_ROWS_DYN <= 0:
    raise ValueError(
        "PYPTO_STEP3P5_KV_CACHE_ROWS must be positive, "
        f"got {KV_CACHE_ROWS_DYN}",
    )
if (
    _live_kv_rows_env is not None
    and KV_CACHE_ROWS_DYN % (KV_NUM_LAYERS * 128) != 0
):
    raise ValueError(
        "live PYPTO_STEP3P5_KV_CACHE_ROWS must be divisible by "
        f"{KV_NUM_LAYERS}*128 rows, got {KV_CACHE_ROWS_DYN}",
    )
_live_mtp_kv_rows_env = os.environ.get("PYPTO_STEP3P5_MTP_KV_CACHE_ROWS")
if _live_mtp_kv_rows_env is not None:
    MTP_KV_CACHE_ROWS_DYN = int(_live_mtp_kv_rows_env)
elif _live_kv_rows_env is not None:
    # Main uses one flat K/V section containing all 45 layers, while the MTP
    # selected-layer program's dynamic value is the capacity of ONE MTP
    # layer.  Derive it from the same vLLM allocator only when live main rows
    # are explicitly configured; the standalone default keeps its historical
    # per-layer 4096-row diagnostic capacity.
    MTP_KV_CACHE_ROWS_DYN = KV_CACHE_ROWS_DYN // KV_NUM_LAYERS
else:
    MTP_KV_CACHE_ROWS_DYN = KV_CACHE_ROWS_DYN
if MTP_KV_CACHE_ROWS_DYN <= 0 or MTP_KV_CACHE_ROWS_DYN % 128 != 0:
    raise ValueError(
        "PYPTO_STEP3P5_MTP_KV_CACHE_ROWS must be a positive multiple of "
        f"block size 128, got {MTP_KV_CACHE_ROWS_DYN}",
    )
BLOCK_TABLE_FLAT_DYN = int(
    os.environ.get(
        "PYPTO_STEP3P5_BLOCK_TABLE_FLAT",
        str(((_LIVE_MAX_SEQ + 127) // 128) * USER_BATCH_DYN),
    ),
)
if BLOCK_TABLE_FLAT_DYN <= 0 or BLOCK_TABLE_FLAT_DYN % USER_BATCH_DYN != 0:
    raise ValueError(
        "PYPTO_STEP3P5_BLOCK_TABLE_FLAT must be positive and divisible by "
        f"storage batch {USER_BATCH_DYN}, got {BLOCK_TABLE_FLAT_DYN}",
    )
ROPE_SEQ_DYN = int(
    os.environ.get("PYPTO_STEP3P5_ROPE_SEQ", str(_LIVE_MAX_SEQ)),
)
if ROPE_SEQ_DYN < _LIVE_MAX_SEQ:
    raise ValueError(
        "PYPTO_STEP3P5_ROPE_SEQ must cover PYPTO_STEP3P5_MAX_SEQ, got "
        f"{ROPE_SEQ_DYN} < {_LIVE_MAX_SEQ}",
    )
# Decoder norm/QK-norm leading rows must match the flat KV layer count.
# Production keeps the default 45; focused L0-L4 diagnostics set both to 5.
LAYER_DYN = KV_NUM_LAYERS
LAYER_HIDDEN_ROWS_DYN = 49152              # = n_full_attn_layers * HIDDEN = 12 * 4096
LAYER_INTER_ROWS_DYN = 4224                # = n_dense_mlp_layers * INTERMEDIATE_LOCAL = 3 * 1408
# MoE-only dims that are declared in __all__ but not yet referenced by any
# tensor signature; left dyn until DecodeLayerMoE wires them up.
LAYER_EXPERTS_DYN = pl.dynamic("LAYER_EXPERTS_DYN")  # n_layers * num_experts
LAYER_EXPERT_ROWS_DYN = pl.dynamic("LAYER_EXPERT_ROWS_DYN")
LAYER_SHARE_ROWS_DYN = pl.dynamic("LAYER_SHARE_ROWS_DYN")

# -----------------------------------------------------------------------------
# Top-level model shape (matches checkpoint config.json verbatim).
# -----------------------------------------------------------------------------
HIDDEN = 4096
INTERMEDIATE = 11264                  # dense MLP hidden
VOCAB = 128896
NUM_HIDDEN_LAYERS = 45
NUM_NEXTN_PREDICT_LAYERS = 3          # MTP layers (indices 45..47 in the ckpt)
NUM_TOTAL_LAYERS = NUM_HIDDEN_LAYERS + NUM_NEXTN_PREDICT_LAYERS

MAX_POSITION_EMBEDDINGS = 262144
MAX_SEQ_DEFAULT = _LIVE_MAX_SEQ       # default for kernel-level golden harness
                                      # (the ckpt supports up to 262144)

# -----------------------------------------------------------------------------
# Attention shape — two variants, selected per-layer by ``LAYER_TYPES``.
# Both variants share the same kv-head count, kv hidden, and head dim.
# -----------------------------------------------------------------------------
HEAD_DIM = 128
NUM_KV_HEADS = 8                       # ``num_attention_groups``
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM    # 1024

# Full attention (``layer_type == "full_attention"``):
NUM_HEADS_FULL = 64                    # 64 heads * 128 = 8192 q hidden
HIDDEN_Q_FULL = NUM_HEADS_FULL * HEAD_DIM
Q_PER_KV_FULL = NUM_HEADS_FULL // NUM_KV_HEADS  # 8

# Sliding attention (``layer_type == "sliding_attention"``):
# attention_other_setting overrides the head count for SWA layers.
NUM_HEADS_SWA = 96                     # 96 heads * 128 = 12288 q hidden
HIDDEN_Q_SWA = NUM_HEADS_SWA * HEAD_DIM
Q_PER_KV_SWA = NUM_HEADS_SWA // NUM_KV_HEADS  # 12

SLIDING_WINDOW = 512                   # SWA context window (in tokens)

# -----------------------------------------------------------------------------
# MoE shape.
# -----------------------------------------------------------------------------
MOE_NUM_EXPERTS = 288
MOE_TOP_K = 8
MOE_INTERMEDIATE = 1280                # routed expert hidden
SHARE_EXPERT_DIM = 1280                # shared expert hidden
MOE_ROUTER_SCALING_FACTOR = 3.0
MOE_ROUTER_ACTIVATION = "sigmoid"      # sigmoid-gated routing with learned bias
NORM_EXPERT_WEIGHT = True              # renormalize top-k weights
USE_MOE_ROUTER_BIAS = True             # additive learned bias on router logits
NEED_FP32_GATE = True                  # gate matmul runs in FP32

# -----------------------------------------------------------------------------
# Step3p5-specific scalars.
# -----------------------------------------------------------------------------
ZERO_CENTERED_NORM = True              # RMSNorm: gamma_eff = stored_gamma + 1.0
USE_HEAD_WISE_ATTN_GATE = True         # per-head sigmoid gate (g_proj)
USE_QK_NORM = True                     # per-head q_norm / k_norm; treat as
                                       # always-on regardless of the
                                       # ``use_qk_norm`` json flag (vllm reads
                                       # the per-head norm weights anyway when
                                       # ``use_optimus_qknorm`` is true).

# -----------------------------------------------------------------------------
# Numeric constants.
# -----------------------------------------------------------------------------
EPS = 1e-5
HIDDEN_INV = 1.0 / HIDDEN
HEAD_DIM_INV = 1.0 / HEAD_DIM
ATTN_SCALE = 1.0 / (HEAD_DIM ** 0.5)
HALF_DIM = HEAD_DIM // 2

# Partial-rotary half-dim for the two layer types. The "rotary_dim" is
# ``head_dim * partial_rotary_factor``; the half is what gets split into
# (lo, hi) for the RoPE rotation, the rest of the head_dim is pass-through.
ROTARY_HALF_FULL = (HEAD_DIM // 2) // 2     # partial = 0.5 -> rotary_dim=64 -> half=32
ROTARY_HALF_SWA = HEAD_DIM // 2             # partial = 1.0 -> rotary_dim=128 -> half=64

# -----------------------------------------------------------------------------
# RoPE per-layer tables (extracted from the checkpoint config.json).
#
# Pattern repeats every 4 layers: [full, sliding, sliding, sliding].
# Full-attention layers use theta=5e6 with llama3 yarn rope scaling and
# partial_rotary_factor=0.5; sliding layers use theta=1e4, no scaling, and
# partial_rotary_factor=1.0. ``yarn_only_types=["full_attention"]`` means
# scaling is only applied on full-attention layers.
#
# These tables are 48-long (45 main + 3 MTP). The 4-cycle pattern
# [full, sliding, sliding, sliding] continues uninterrupted through the MTP
# layers: index 44 is full (cycle start), so 45/46/47 are sliding/sliding/
# sliding. This matches the ckpt's partial_rotary_factors[45..47] == 1.0,
# 1.0, 1.0 (verified 2026-06-03 against config.json on the
# step3p5_flash_release_hf_mtp3_bf16 checkpoint). The numeric LAYER_TYPES /
# LAYER_ROPE_THETA / LAYER_PARTIAL_ROTARY_FACTOR tables below evaluate to
# exactly that.
# -----------------------------------------------------------------------------
LAYER_TYPE_FULL = "full_attention"
LAYER_TYPE_SWA = "sliding_attention"

# fmt: off
LAYER_TYPES: tuple[str, ...] = (
    LAYER_TYPE_FULL, LAYER_TYPE_SWA, LAYER_TYPE_SWA, LAYER_TYPE_SWA,
    LAYER_TYPE_FULL, LAYER_TYPE_SWA, LAYER_TYPE_SWA, LAYER_TYPE_SWA,
    LAYER_TYPE_FULL, LAYER_TYPE_SWA, LAYER_TYPE_SWA, LAYER_TYPE_SWA,
    LAYER_TYPE_FULL, LAYER_TYPE_SWA, LAYER_TYPE_SWA, LAYER_TYPE_SWA,
    LAYER_TYPE_FULL, LAYER_TYPE_SWA, LAYER_TYPE_SWA, LAYER_TYPE_SWA,
    LAYER_TYPE_FULL, LAYER_TYPE_SWA, LAYER_TYPE_SWA, LAYER_TYPE_SWA,
    LAYER_TYPE_FULL, LAYER_TYPE_SWA, LAYER_TYPE_SWA, LAYER_TYPE_SWA,
    LAYER_TYPE_FULL, LAYER_TYPE_SWA, LAYER_TYPE_SWA, LAYER_TYPE_SWA,
    LAYER_TYPE_FULL, LAYER_TYPE_SWA, LAYER_TYPE_SWA, LAYER_TYPE_SWA,
    LAYER_TYPE_FULL, LAYER_TYPE_SWA, LAYER_TYPE_SWA, LAYER_TYPE_SWA,
    LAYER_TYPE_FULL, LAYER_TYPE_SWA, LAYER_TYPE_SWA, LAYER_TYPE_SWA,
    LAYER_TYPE_FULL, LAYER_TYPE_SWA, LAYER_TYPE_SWA, LAYER_TYPE_SWA,
)
assert len(LAYER_TYPES) == NUM_TOTAL_LAYERS, (
    f"LAYER_TYPES has {len(LAYER_TYPES)} entries, expected {NUM_TOTAL_LAYERS}"
)

# Per-layer RoPE theta. Full-attention layers use 5e6, sliding use 1e4.
LAYER_ROPE_THETA: tuple[float, ...] = tuple(
    5_000_000.0 if t == LAYER_TYPE_FULL else 10_000.0 for t in LAYER_TYPES
)

# Per-layer partial rotary factor. Full-attention layers use 0.5, sliding 1.0.
LAYER_PARTIAL_ROTARY_FACTOR: tuple[float, ...] = tuple(
    0.5 if t == LAYER_TYPE_FULL else 1.0 for t in LAYER_TYPES
)
# fmt: on

# yarn rope scaling parameters (only applied on full-attention layers per
# ``yarn_only_types=["full_attention"]``).
ROPE_SCALING = {
    "rope_type": "llama3",
    "factor": 2.0,
    "original_max_position_embeddings": 131072,
    "low_freq_factor": 1.0,
    "high_freq_factor": 32.0,
}
YARN_ONLY_TYPES = (LAYER_TYPE_FULL,)

# -----------------------------------------------------------------------------
# MoE / dense MLP layer membership.
# -----------------------------------------------------------------------------
# moe_layers_enum = "3,4,...,44" -- layers 0,1,2 are dense MLP (the "use_mfa"
# / "moe_layer_offset" knobs are not used by step3p5 once moe_layers_enum is
# present, see vllm modeling_step3p5).
MOE_LAYER_INDICES: tuple[int, ...] = tuple(range(3, NUM_HIDDEN_LAYERS))  # 3..44
DENSE_LAYER_INDICES: tuple[int, ...] = tuple(
    i for i in range(NUM_HIDDEN_LAYERS) if i not in MOE_LAYER_INDICES
)  # (0, 1, 2)

# -----------------------------------------------------------------------------
# SwigluStep tables (per-layer activation limit). 0.0 means plain SiLU.
# Only the routed-MoE active path uses ``swiglu_limits``; the share/dense MLP
# uses ``swiglu_limits_shared``. Lengths cover the 48 total layers.
# -----------------------------------------------------------------------------
# fmt: off
SWIGLU_LIMITS: tuple[float, ...] = (
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 7.0, 7.0, 0.0, 0.0, 0.0,
)
assert len(SWIGLU_LIMITS) == NUM_TOTAL_LAYERS

SWIGLU_LIMITS_SHARED: tuple[float, ...] = (
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 16.0, 0.0, 0.0, 0.0,
)
assert len(SWIGLU_LIMITS_SHARED) == NUM_TOTAL_LAYERS
# fmt: on


# -----------------------------------------------------------------------------
# Helpers for per-layer compile-time selection.
# -----------------------------------------------------------------------------
def is_full_attention(layer_idx: int) -> bool:
    return LAYER_TYPES[layer_idx] == LAYER_TYPE_FULL


def is_moe_layer(layer_idx: int) -> bool:
    return layer_idx in MOE_LAYER_INDICES


def num_heads_for_layer(layer_idx: int) -> int:
    return NUM_HEADS_FULL if is_full_attention(layer_idx) else NUM_HEADS_SWA


def hidden_q_for_layer(layer_idx: int) -> int:
    return HIDDEN_Q_FULL if is_full_attention(layer_idx) else HIDDEN_Q_SWA


def rotary_half_for_layer(layer_idx: int) -> int:
    """Half-dim of the rotary slice for partial RoPE.

    Full-attention layers rotate the leading 0.5 * head_dim lanes, so the
    sin/cos pair operates on (head_dim*0.5)/2 = 32 lanes.
    Sliding layers rotate the full head_dim, so the half is 64.
    """
    return ROTARY_HALF_FULL if is_full_attention(layer_idx) else ROTARY_HALF_SWA


# -----------------------------------------------------------------------------
# Tiling defaults shared across all step3p5 kernels.
# Per-kernel kernels may override locally; these are the safe starting points.
# -----------------------------------------------------------------------------
# Static tensor-storage capacity.  Runtime active batch/token count is carried
# separately by host metadata and must be <= this capacity.
BATCH = STORAGE_BATCH_CAPACITY
BATCH_TILE = 16
BLOCK_SIZE = 128                        # paged-cache block (also K/V SEQ_TILE)
SEQ_TILE = 128

# Scope 1 tiling (input proj).
INPUT_PROJ_K_CHUNK = 256

# SWA input RMSNorm assigns independent storage rows to logical SPMD tasks.
# The runtime, rather than this model constant, maps those workload-derived
# logical tasks onto the target's available vector cores.  Two rows per task
# is the A2A3 release point validated on 0162.  The environment entry is
# fail-closed to that profile until another task grain passes the same gates.
SWA_RMSNORM_ROWS_PER_TASK = int(
    os.environ.get("PYPTO_STEP3P5_SWA_RMSNORM_ROWS_PER_TASK", "2"),
)
if (
    SWA_RMSNORM_ROWS_PER_TASK <= 0
    or SWA_RMSNORM_ROWS_PER_TASK > BATCH
    or BATCH % SWA_RMSNORM_ROWS_PER_TASK != 0
):
    raise ValueError(
        "PYPTO_STEP3P5_SWA_RMSNORM_ROWS_PER_TASK must be positive, not "
        f"greater than BATCH={BATCH}, and divide BATCH exactly; got "
        f"{SWA_RMSNORM_ROWS_PER_TASK}",
    )
if SWA_RMSNORM_ROWS_PER_TASK != 2:
    raise ValueError(
        "PYPTO_STEP3P5_SWA_RMSNORM_ROWS_PER_TASK currently supports only "
        f"the calibrated value 2, got {SWA_RMSNORM_ROWS_PER_TASK}",
    )

KV_PROJ_K_CHUNK = INPUT_PROJ_K_CHUNK // 2  # 128 — keeps K/V L0B within 512 KB at TP=1
Q_OUT_CHUNK = 256
KV_OUT_CHUNK = 256

# Scope 3 tiling (output proj + MLP / MoE).
K_CHUNK = 256
OUT_PROJ_K_CHUNK = 256
# Keep decode full-attention and SWA output-projection grains independent:
# their local-Q widths and surrounding stage timing differ, so architecture
# calibration may select different N tiles.  Matmul and vector epilogues are
# also independent: on 910B, an N=128 out-proj matmul currently fails target
# lowering while the cast/residual vector kernels can still benefit from a
# wider tile and fewer logical tasks.
#
# The defaults below are the calibrated A2A3 release profile from the 0162
# bs=1/ctx=64k sweep: grouping=3 leaves 22 AIC logical tasks (one wave on
# 24 AICs), while vec N=128 leaves 32 AIV tasks (one wave on 48 AIVs).
# They are scheduling defaults, not semantic constants: another architecture
# should override full/SWA independently from its own resource/task sweep.
OUT_PROJ_N_CHUNK = 64  # Backward-compatible tile for non-attention callers.
FULL_ATTN_OUT_PROJ_N_CHUNK = int(
    os.environ.get("PYPTO_STEP3P5_FULL_ATTN_OUT_PROJ_N_CHUNK", "64"),
)
SWA_OUT_PROJ_N_CHUNK = int(
    os.environ.get("PYPTO_STEP3P5_SWA_OUT_PROJ_N_CHUNK", "64"),
)
FULL_ATTN_OUT_PROJ_MATMUL_N_CHUNK = int(
    os.environ.get(
        "PYPTO_STEP3P5_FULL_ATTN_OUT_PROJ_MATMUL_N_CHUNK",
        str(FULL_ATTN_OUT_PROJ_N_CHUNK),
    ),
)
FULL_ATTN_OUT_PROJ_VEC_N_CHUNK = int(
    os.environ.get(
        "PYPTO_STEP3P5_FULL_ATTN_OUT_PROJ_VEC_N_CHUNK",
        "128",
    ),
)
SWA_OUT_PROJ_MATMUL_N_CHUNK = int(
    os.environ.get(
        "PYPTO_STEP3P5_SWA_OUT_PROJ_MATMUL_N_CHUNK",
        str(SWA_OUT_PROJ_N_CHUNK),
    ),
)
SWA_OUT_PROJ_VEC_N_CHUNK = int(
    os.environ.get(
        "PYPTO_STEP3P5_SWA_OUT_PROJ_VEC_N_CHUNK",
        "128",
    ),
)
FULL_ATTN_OUT_PROJ_MATMUL_TILES_PER_TASK = int(
    os.environ.get(
        "PYPTO_STEP3P5_FULL_ATTN_OUT_PROJ_MATMUL_TILES_PER_TASK",
        "3",
    ),
)
SWA_OUT_PROJ_MATMUL_TILES_PER_TASK = int(
    os.environ.get(
        "PYPTO_STEP3P5_SWA_OUT_PROJ_MATMUL_TILES_PER_TASK",
        "3",
    ),
)
FULL_ATTN_OUT_PROJ_FUSE_CAST = int(
    os.environ.get(
        "PYPTO_STEP3P5_FULL_ATTN_OUT_PROJ_FUSE_CAST",
        "1",
    ),
)
SWA_OUT_PROJ_FUSE_CAST = int(
    os.environ.get(
        "PYPTO_STEP3P5_SWA_OUT_PROJ_FUSE_CAST",
        "1",
    ),
)
for _name, _value in (
    ("PYPTO_STEP3P5_FULL_ATTN_OUT_PROJ_N_CHUNK", FULL_ATTN_OUT_PROJ_N_CHUNK),
    ("PYPTO_STEP3P5_SWA_OUT_PROJ_N_CHUNK", SWA_OUT_PROJ_N_CHUNK),
    (
        "PYPTO_STEP3P5_FULL_ATTN_OUT_PROJ_MATMUL_N_CHUNK",
        FULL_ATTN_OUT_PROJ_MATMUL_N_CHUNK,
    ),
    (
        "PYPTO_STEP3P5_FULL_ATTN_OUT_PROJ_VEC_N_CHUNK",
        FULL_ATTN_OUT_PROJ_VEC_N_CHUNK,
    ),
    (
        "PYPTO_STEP3P5_SWA_OUT_PROJ_MATMUL_N_CHUNK",
        SWA_OUT_PROJ_MATMUL_N_CHUNK,
    ),
    (
        "PYPTO_STEP3P5_SWA_OUT_PROJ_VEC_N_CHUNK",
        SWA_OUT_PROJ_VEC_N_CHUNK,
    ),
):
    if _value <= 0 or HIDDEN % _value != 0:
        raise ValueError(
            f"{_name} must be positive and divide HIDDEN={HIDDEN}, "
            f"got {_value}",
        )
for _name, _value, _n_chunk in (
    (
        "PYPTO_STEP3P5_FULL_ATTN_OUT_PROJ_MATMUL_TILES_PER_TASK",
        FULL_ATTN_OUT_PROJ_MATMUL_TILES_PER_TASK,
        FULL_ATTN_OUT_PROJ_MATMUL_N_CHUNK,
    ),
    (
        "PYPTO_STEP3P5_SWA_OUT_PROJ_MATMUL_TILES_PER_TASK",
        SWA_OUT_PROJ_MATMUL_TILES_PER_TASK,
        SWA_OUT_PROJ_MATMUL_N_CHUNK,
    ),
):
    if _value <= 0:
        raise ValueError(f"{_name} must be positive, got {_value}")
    _out_proj_n_tiles = HIDDEN // _n_chunk
    if _value > _out_proj_n_tiles:
        raise ValueError(
            f"{_name}={_value} exceeds the {_out_proj_n_tiles} legal "
            f"N={_n_chunk} tiles in HIDDEN={HIDDEN}",
        )
for _name, _value in (
    ("PYPTO_STEP3P5_FULL_ATTN_OUT_PROJ_FUSE_CAST", FULL_ATTN_OUT_PROJ_FUSE_CAST),
    ("PYPTO_STEP3P5_SWA_OUT_PROJ_FUSE_CAST", SWA_OUT_PROJ_FUSE_CAST),
):
    if _value not in (0, 1):
        raise ValueError(f"{_name} must be 0 or 1, got {_value}")
# MLP_OUT_CHUNK must divide BOTH the world-level INTERMEDIATE (11264, used
# by the historical single-card drafts) AND the per-card TP-sliced
# INTERMEDIATE_LOCAL=1408 (used by the hidden-only dense MLP body).
# gcd(11264, 1408) = 1408 with many divisors; 128 is the largest power
# of 2 that divides 1408 (1408 = 128 * 11) while still aligning with
# the cube's 128B / 16-row friendly tiling.
MLP_OUT_CHUNK = 128

# TP all-reduce transfer tile width.  This controls TPUT staging and the final
# local copy; reduce-scatter ownership remains HIDDEN / TP.  It does not change
# peer order, FP32 accumulation order, or either selected branch's protocol.
# Keep the A2A3 release default at 512 while allowing platform calibration.
TP_ALL_REDUCE_CHUNK = int(
    os.environ.get("PYPTO_STEP3P5_TP_ALL_REDUCE_CHUNK", "512"),
)
if TP_ALL_REDUCE_CHUNK <= 0 or HIDDEN % TP_ALL_REDUCE_CHUNK != 0:
    raise ValueError(
        "PYPTO_STEP3P5_TP_ALL_REDUCE_CHUNK must be positive and divide "
        f"HIDDEN={HIDDEN}, got {TP_ALL_REDUCE_CHUNK}",
    )

MLP_SPMD_INNER = 2
MLP_GROUP_CHUNK = MLP_SPMD_INNER * MLP_OUT_CHUNK
DOWN_MLP_CHUNK = 256
DOWN_OUT_CHUNK = 256
FINAL_RMS_K_CHUNK = 128
LM_HEAD_K_CHUNK = 128
# VOCAB_CHUNK must divide BOTH the world-level VOCAB (128896, used by the
# historical single-card drafts) AND the per-card TP-sliced
# VOCAB_LOCAL = 128896 // 8 = 16112 (used by rms_lm_head's vocab-sliced
# matmul). 16112 = 16 * 19 * 53, so the largest power-of-2 divisor is 16,
# matching the cube's 32B BF16 row alignment. 16 also divides 128896 cleanly
# (128896 = 16 * 8056).
VOCAB_CHUNK = 16

# fa_fused decode tiling for the full-attention path. Step3p5 pairs Q heads in
# batches of (Q_PER_KV) per KV head: q_per_kv = 8 for full-attention layers and
# 12 for sliding layers — both clean factors of the kv-head count and >= the
# cube's minimum row count, so the fa_fused pattern fits without re-tuning.
# Q_HEAD_BATCH = q_per_kv keeps one Q-group per KV head.
Q_HEAD_BATCH_FULL = Q_PER_KV_FULL       # 8
Q_HEAD_BATCH_SWA = Q_PER_KV_SWA         # 12
# Q_HEAD_PAD: padded Q row count fa_fused operates on; needs to be a multiple
# of 4 with Q_HEAD_PAD//2 >= Q_HEAD_BATCH.
Q_HEAD_PAD_FULL = 16                    # 16 % 4 == 0, 16/2 == 8 >= 8 (full)
Q_HEAD_PAD_SWA = 24                     # 24 % 4 == 0, 24/2 == 12 >= 12 (swa)

MAX_BLOCKS_PER_SEQ = (MAX_SEQ_DEFAULT + BLOCK_SIZE - 1) // BLOCK_SIZE

# Decode full-attention work granularity. Each logical SPMD block processes
# this many paged-cache blocks; the runtime maps logical blocks onto the
# architecture's available physical cores and dispatches extra blocks in waves.
#
# A profile names a set of compile-time tuning defaults; it never names or
# fixes a physical core count. The runtime maps workload-derived logical tasks
# onto the target's available resources. ``portable`` preserves the proven
# release fallback. ``a2a3`` records the all-rank profile validated across
# per-request 64K batch sizes on the 0162 A2A3 stack; launchers must select it
# explicitly.
# Environment overrides remain highest priority for single-variable sweeps.
PTO2_LOGICAL_BLOCK_LIMIT = 2**15 - 1
ATTN_TASK_PROFILE = os.environ.get(
    "PYPTO_STEP3P5_ATTN_TASK_PROFILE",
    "portable",
)
_ATTN_TASK_PROFILES = {
    "portable": {
        "qk_blocks_per_task": 22,
        "softmax_blocks_per_task": 12,
        "online_blocks_per_task": 16,
        "online_reduce_fan_in": 8,
        "qk_uniform_o1": 0,
        "softmax_uniform_o1": 0,
        "online_uniform_o1": 0,
        "online_reduce_uniform_o1": 0,
    },
    "a2a3": {
        "qk_blocks_per_task": 22,
        "softmax_blocks_per_task": 16,
        "online_blocks_per_task": 22,
        "online_reduce_fan_in": 8,
        "qk_uniform_o1": 1,
        "softmax_uniform_o1": 1,
        "online_uniform_o1": 1,
        "online_reduce_uniform_o1": 1,
    },
}
if ATTN_TASK_PROFILE not in _ATTN_TASK_PROFILES:
    raise ValueError(
        "PYPTO_STEP3P5_ATTN_TASK_PROFILE must be one of "
        f"{sorted(_ATTN_TASK_PROFILES)}, got {ATTN_TASK_PROFILE!r}",
    )
_attn_task_profile = _ATTN_TASK_PROFILES[ATTN_TASK_PROFILE]
FULL_ATTN_QK_BLOCKS_PER_TASK = int(
    os.environ.get(
        "PYPTO_STEP3P5_FULL_ATTN_QK_BLOCKS_PER_TASK",
        str(_attn_task_profile["qk_blocks_per_task"]),
    ),
)
FULL_ATTN_SOFTMAX_BLOCKS_PER_TASK = int(
    os.environ.get(
        "PYPTO_STEP3P5_FULL_ATTN_SOFTMAX_BLOCKS_PER_TASK",
        str(_attn_task_profile["softmax_blocks_per_task"]),
    ),
)
FULL_ATTN_ONLINE_SOFTMAX_BLOCKS_PER_TASK = int(
    os.environ.get(
        "PYPTO_STEP3P5_FULL_ATTN_ONLINE_SOFTMAX_BLOCKS_PER_TASK",
        str(_attn_task_profile["online_blocks_per_task"]),
    ),
)
FULL_ATTN_ONLINE_SOFTMAX_PARTIALS_PER_REDUCE_TASK = int(
    os.environ.get(
        "PYPTO_STEP3P5_FULL_ATTN_ONLINE_SOFTMAX_PARTIALS_PER_REDUCE_TASK",
        str(_attn_task_profile["online_reduce_fan_in"]),
    ),
)
FULL_ATTN_QK_UNIFORM_O1 = int(
    os.environ.get(
        "PYPTO_STEP3P5_FULL_ATTN_QK_UNIFORM_O1",
        str(_attn_task_profile["qk_uniform_o1"]),
    ),
)
FULL_ATTN_SOFTMAX_UNIFORM_O1 = int(
    os.environ.get(
        "PYPTO_STEP3P5_FULL_ATTN_SOFTMAX_UNIFORM_O1",
        str(_attn_task_profile["softmax_uniform_o1"]),
    ),
)
FULL_ATTN_ONLINE_SOFTMAX_UNIFORM_O1 = int(
    os.environ.get(
        "PYPTO_STEP3P5_FULL_ATTN_ONLINE_SOFTMAX_UNIFORM_O1",
        str(_attn_task_profile["online_uniform_o1"]),
    ),
)
FULL_ATTN_ONLINE_SOFTMAX_REDUCE_UNIFORM_O1 = int(
    os.environ.get(
        "PYPTO_STEP3P5_FULL_ATTN_ONLINE_SOFTMAX_REDUCE_UNIFORM_O1",
        str(_attn_task_profile["online_reduce_uniform_o1"]),
    ),
)
for _name, _value in (
    ("PYPTO_STEP3P5_FULL_ATTN_QK_BLOCKS_PER_TASK", FULL_ATTN_QK_BLOCKS_PER_TASK),
    (
        "PYPTO_STEP3P5_FULL_ATTN_SOFTMAX_BLOCKS_PER_TASK",
        FULL_ATTN_SOFTMAX_BLOCKS_PER_TASK,
    ),
    (
        "PYPTO_STEP3P5_FULL_ATTN_ONLINE_SOFTMAX_BLOCKS_PER_TASK",
        FULL_ATTN_ONLINE_SOFTMAX_BLOCKS_PER_TASK,
    ),
):
    if _value <= 0:
        raise ValueError(f"{_name} must be positive, got {_value}")
    _max_logical_tasks = STORAGE_BATCH_CAPACITY * (
        (MAX_BLOCKS_PER_SEQ + _value - 1) // _value
    )
    if _max_logical_tasks > PTO2_LOGICAL_BLOCK_LIMIT:
        raise ValueError(
            f"{_name}={_value} can launch {_max_logical_tasks} logical "
            f"blocks at capacity, exceeding the PTO2 int16 limit "
            f"{PTO2_LOGICAL_BLOCK_LIMIT}; increase task grain or widen "
            "runtime launch counters",
        )
if FULL_ATTN_ONLINE_SOFTMAX_PARTIALS_PER_REDUCE_TASK <= 0:
    raise ValueError(
        "PYPTO_STEP3P5_FULL_ATTN_ONLINE_SOFTMAX_PARTIALS_PER_REDUCE_TASK "
        "must be positive, got "
        f"{FULL_ATTN_ONLINE_SOFTMAX_PARTIALS_PER_REDUCE_TASK}",
    )
for _name, _value in (
    ("PYPTO_STEP3P5_FULL_ATTN_QK_UNIFORM_O1", FULL_ATTN_QK_UNIFORM_O1),
    (
        "PYPTO_STEP3P5_FULL_ATTN_SOFTMAX_UNIFORM_O1",
        FULL_ATTN_SOFTMAX_UNIFORM_O1,
    ),
    (
        "PYPTO_STEP3P5_FULL_ATTN_ONLINE_SOFTMAX_UNIFORM_O1",
        FULL_ATTN_ONLINE_SOFTMAX_UNIFORM_O1,
    ),
    (
        "PYPTO_STEP3P5_FULL_ATTN_ONLINE_SOFTMAX_REDUCE_UNIFORM_O1",
        FULL_ATTN_ONLINE_SOFTMAX_REDUCE_UNIFORM_O1,
    ),
):
    if _value not in (0, 1):
        raise ValueError(f"{_name} must be 0 or 1, got {_value}")


# -----------------------------------------------------------------------------
# Distributed topology (Phase 9 — single-node, 8 cards, one process per card).
#
# Step3p5 inference deployment is a single 8-card node. The TP and EP groups
# are co-located on the same world (typical inference layout — one process
# per card, world_size = TP_WORLD_SIZE = EP_WORLD_SIZE).
#
# Tensor-Parallel (TP) sharding axes:
#   - Attention Q/K/V/O      sliced by HEAD count
#                            (NUM_HEADS_FULL / NUM_HEADS_SWA / NUM_KV_HEADS)
#   - lm_head                sliced by VOCAB
#   - shared_expert / dense  sliced by INTERMEDIATE / SHARE_EXPERT_DIM
#     MLP
#
# Expert-Parallel (EP) sharding:
#   - Routed experts         288 experts evenly partitioned, 36 per card
#   - Token routing          replicated input, owner-local route packing,
#                            then TP all-reduce of local output partials
#
# IMPORTANT for downstream code in this package:
#   The single-card constants HIDDEN_Q_FULL, HIDDEN_Q_SWA, KV_HIDDEN,
#   INTERMEDIATE, SHARE_EXPERT_DIM, VOCAB, MOE_NUM_EXPERTS defined above are
#   now WORLD-LEVEL totals. Per-card kernel shapes must use the *_LOCAL
#   forms below (Phase 9 Wave 2 refactor). The world-level totals stay
#   defined for host-side weight-loading and golden references.
# -----------------------------------------------------------------------------
TP_WORLD_SIZE = 8
EP_WORLD_SIZE = 8

# Per-card attention head / feature counts (sliced by TP_WORLD_SIZE).
NUM_HEADS_FULL_LOCAL = NUM_HEADS_FULL // TP_WORLD_SIZE   # 64 // 8 == 8
NUM_HEADS_SWA_LOCAL = NUM_HEADS_SWA // TP_WORLD_SIZE     # 96 // 8 == 12
KV_HEADS_LOCAL = NUM_KV_HEADS // TP_WORLD_SIZE           # 8  // 8 == 1

HIDDEN_Q_FULL_LOCAL = NUM_HEADS_FULL_LOCAL * HEAD_DIM    # 8  * 128 == 1024
HIDDEN_Q_SWA_LOCAL = NUM_HEADS_SWA_LOCAL * HEAD_DIM      # 12 * 128 == 1536
KV_HIDDEN_LOCAL = KV_HEADS_LOCAL * HEAD_DIM              # 1  * 128 ==  128

# Adaptive K-chunk for K/V projection: at TP=1 KV_HIDDEN_LOCAL (1024) exceeds
# INPUT_PROJ_K_CHUNK (256), which would make L0B = 1024*256*2 = 512 KB — at the
# limit. Use the halved KV_PROJ_K_CHUNK (128) in that case.
# At TP=8 KV_HIDDEN_LOCAL=128 ≤ 256 → falls back to INPUT_PROJ_K_CHUNK=256.
if KV_HIDDEN_LOCAL > INPUT_PROJ_K_CHUNK:
    KV_PROJ_K_CHUNK_LOCAL = KV_PROJ_K_CHUNK      # 128 — TP=1 / large KV_HIDDEN_LOCAL
else:
    KV_PROJ_K_CHUNK_LOCAL = INPUT_PROJ_K_CHUNK   # 256 — TP=8 / normal KV_HIDDEN_LOCAL

# PTOAS A2/A3 cube unit requires bf16 matmul N (output cols) to be a
# multiple of 16. NUM_HEADS_*_LOCAL after TP=8 sharding falls below that
# threshold (8 / 12), so the per-head gate weight ``w_g`` is padded out
# to NUM_HEADS_*_LOCAL_PAD on its column axis. The host weight loader
# zero-pads the upper columns; downstream consumers only index the first
# NUM_HEADS_*_LOCAL columns so the pad is read-only ignored.
NUM_HEADS_FULL_LOCAL_PAD = 16   # ceil(8  / 16) * 16 == 16
NUM_HEADS_SWA_LOCAL_PAD = 16    # ceil(12 / 16) * 16 == 16

# Per-card MLP / shared-expert / lm_head dims (sliced by TP_WORLD_SIZE).
INTERMEDIATE_LOCAL = INTERMEDIATE // TP_WORLD_SIZE       # 11264 // 8 == 1408
SHARE_EXPERT_DIM_LOCAL = SHARE_EXPERT_DIM // TP_WORLD_SIZE  # 1280 // 8 == 160
VOCAB_LOCAL = VOCAB // TP_WORLD_SIZE                     # 128896 // 8 == 16112

# Per-card routed-expert count (sliced by EP_WORLD_SIZE).
MOE_NUM_EXPERTS_LOCAL = MOE_NUM_EXPERTS // EP_WORLD_SIZE  # 288 // 8 == 36

# Note on Q_HEAD_BATCH / Q_HEAD_PAD:
#   Q_HEAD_BATCH_FULL/SWA == Q_PER_KV_FULL/SWA, and Q_PER_KV is invariant
#   under TP (numerator and denominator are sliced by the same factor:
#   NUM_HEADS_FULL/NUM_KV_HEADS == NUM_HEADS_FULL_LOCAL/KV_HEADS_LOCAL).
#   With TP=8 each rank sees KV_HEADS_LOCAL=1 KV bucket × Q_PER_KV Q-rows,
#   and the fa_fused tile constraints stay the same. The Q_HEAD_PAD values
#   (16 for full, 24 for SWA) likewise stay valid.

# Sanity: every TP / EP sliced dim must divide cleanly.
if (TP_WORLD_SIZE, EP_WORLD_SIZE) != (8, 8):
    raise ValueError(
        "replicated-input local-owner MoE requires co-located TP=EP=8, "
        f"got TP_WORLD_SIZE={TP_WORLD_SIZE} and "
        f"EP_WORLD_SIZE={EP_WORLD_SIZE}"
    )
assert NUM_HEADS_FULL % TP_WORLD_SIZE == 0, (
    f"NUM_HEADS_FULL={NUM_HEADS_FULL} must be a multiple of "
    f"TP_WORLD_SIZE={TP_WORLD_SIZE}"
)
assert NUM_HEADS_SWA % TP_WORLD_SIZE == 0, (
    f"NUM_HEADS_SWA={NUM_HEADS_SWA} must be a multiple of "
    f"TP_WORLD_SIZE={TP_WORLD_SIZE}"
)
assert NUM_KV_HEADS % TP_WORLD_SIZE == 0, (
    f"NUM_KV_HEADS={NUM_KV_HEADS} must be a multiple of "
    f"TP_WORLD_SIZE={TP_WORLD_SIZE}"
)
assert INTERMEDIATE % TP_WORLD_SIZE == 0, (
    f"INTERMEDIATE={INTERMEDIATE} must be a multiple of "
    f"TP_WORLD_SIZE={TP_WORLD_SIZE}"
)
assert SHARE_EXPERT_DIM % TP_WORLD_SIZE == 0, (
    f"SHARE_EXPERT_DIM={SHARE_EXPERT_DIM} must be a multiple of "
    f"TP_WORLD_SIZE={TP_WORLD_SIZE}"
)
assert VOCAB % TP_WORLD_SIZE == 0, (
    f"VOCAB={VOCAB} must be a multiple of TP_WORLD_SIZE={TP_WORLD_SIZE}"
)
assert MOE_NUM_EXPERTS % EP_WORLD_SIZE == 0, (
    f"MOE_NUM_EXPERTS={MOE_NUM_EXPERTS} must be a multiple of "
    f"EP_WORLD_SIZE={EP_WORLD_SIZE}"
)
# The Q_PER_KV invariant.
assert NUM_HEADS_FULL_LOCAL // KV_HEADS_LOCAL == Q_PER_KV_FULL
assert NUM_HEADS_SWA_LOCAL // KV_HEADS_LOCAL == Q_PER_KV_SWA


def ep_expert_owner(expert_id: int) -> int:
    """Return the EP rank that hosts the given global routed-expert id.

    Experts ``0..MOE_NUM_EXPERTS_LOCAL-1`` belong to rank 0,
    ``MOE_NUM_EXPERTS_LOCAL..2*MOE_NUM_EXPERTS_LOCAL-1`` to rank 1, and so
    on. Mirrors vLLM's contiguous expert-block sharding.
    """
    if not 0 <= expert_id < MOE_NUM_EXPERTS:
        raise ValueError(
            f"expert_id {expert_id} out of range [0, {MOE_NUM_EXPERTS})"
        )
    return expert_id // MOE_NUM_EXPERTS_LOCAL


def ep_local_expert_id(expert_id: int) -> int:
    """Return the local index within the owning card for a global expert id.

    The owner rank is :func:`ep_expert_owner`; the local index is the
    in-shard slot ``[0, MOE_NUM_EXPERTS_LOCAL)``.
    """
    if not 0 <= expert_id < MOE_NUM_EXPERTS:
        raise ValueError(
            f"expert_id {expert_id} out of range [0, {MOE_NUM_EXPERTS})"
        )
    return expert_id % MOE_NUM_EXPERTS_LOCAL


def ep_global_expert_id(rank: int, local_id: int) -> int:
    """Inverse of (:func:`ep_expert_owner`, :func:`ep_local_expert_id`)."""
    if not 0 <= rank < EP_WORLD_SIZE:
        raise ValueError(f"rank {rank} out of range [0, {EP_WORLD_SIZE})")
    if not 0 <= local_id < MOE_NUM_EXPERTS_LOCAL:
        raise ValueError(
            f"local_id {local_id} out of range [0, {MOE_NUM_EXPERTS_LOCAL})"
        )
    return rank * MOE_NUM_EXPERTS_LOCAL + local_id


__all__ = [
    # dynamic dims
    "DEFAULT_STORAGE_BATCH_CAPACITY",
    "STORAGE_BATCH_CAPACITY",
    "USER_BATCH_DYN",
    "KV_CACHE_ROWS_DYN",
    "KV_NUM_LAYERS",
    "MTP_KV_CACHE_ROWS_DYN",
    "BLOCK_TABLE_FLAT_DYN",
    "ROPE_SEQ_DYN",
    "LAYER_DYN",
    "LAYER_HIDDEN_ROWS_DYN",
    "LAYER_INTER_ROWS_DYN",
    "LAYER_EXPERTS_DYN",
    "LAYER_EXPERT_ROWS_DYN",
    "LAYER_SHARE_ROWS_DYN",
    # top-level shape
    "HIDDEN",
    "INTERMEDIATE",
    "VOCAB",
    "NUM_HIDDEN_LAYERS",
    "NUM_NEXTN_PREDICT_LAYERS",
    "NUM_TOTAL_LAYERS",
    "MAX_POSITION_EMBEDDINGS",
    "MAX_SEQ_DEFAULT",
    # attention
    "HEAD_DIM",
    "NUM_KV_HEADS",
    "KV_HIDDEN",
    "NUM_HEADS_FULL",
    "HIDDEN_Q_FULL",
    "Q_PER_KV_FULL",
    "NUM_HEADS_SWA",
    "HIDDEN_Q_SWA",
    "Q_PER_KV_SWA",
    "SLIDING_WINDOW",
    "HALF_DIM",
    "ROTARY_HALF_FULL",
    "ROTARY_HALF_SWA",
    "ATTN_SCALE",
    # moe
    "MOE_NUM_EXPERTS",
    "MOE_TOP_K",
    "MOE_INTERMEDIATE",
    "SHARE_EXPERT_DIM",
    "MOE_ROUTER_SCALING_FACTOR",
    "MOE_ROUTER_ACTIVATION",
    "NORM_EXPERT_WEIGHT",
    "USE_MOE_ROUTER_BIAS",
    "NEED_FP32_GATE",
    # step3p5 flags
    "ZERO_CENTERED_NORM",
    "USE_HEAD_WISE_ATTN_GATE",
    "USE_QK_NORM",
    # numeric
    "EPS",
    "HIDDEN_INV",
    "HEAD_DIM_INV",
    # per-layer tables
    "LAYER_TYPE_FULL",
    "LAYER_TYPE_SWA",
    "LAYER_TYPES",
    "LAYER_ROPE_THETA",
    "LAYER_PARTIAL_ROTARY_FACTOR",
    "ROPE_SCALING",
    "YARN_ONLY_TYPES",
    "MOE_LAYER_INDICES",
    "DENSE_LAYER_INDICES",
    "SWIGLU_LIMITS",
    "SWIGLU_LIMITS_SHARED",
    # helpers
    "is_full_attention",
    "is_moe_layer",
    "num_heads_for_layer",
    "hidden_q_for_layer",
    "rotary_half_for_layer",
    # tiling
    "BATCH",
    "BATCH_TILE",
    "BLOCK_SIZE",
    "SEQ_TILE",
    "INPUT_PROJ_K_CHUNK",
    "SWA_RMSNORM_ROWS_PER_TASK",
    "KV_PROJ_K_CHUNK",
    "KV_PROJ_K_CHUNK_LOCAL",
    "Q_OUT_CHUNK",
    "KV_OUT_CHUNK",
    "K_CHUNK",
    "OUT_PROJ_K_CHUNK",
    "OUT_PROJ_N_CHUNK",
    "FULL_ATTN_OUT_PROJ_N_CHUNK",
    "SWA_OUT_PROJ_N_CHUNK",
    "FULL_ATTN_OUT_PROJ_MATMUL_N_CHUNK",
    "FULL_ATTN_OUT_PROJ_VEC_N_CHUNK",
    "FULL_ATTN_OUT_PROJ_MATMUL_TILES_PER_TASK",
    "FULL_ATTN_OUT_PROJ_FUSE_CAST",
    "SWA_OUT_PROJ_MATMUL_N_CHUNK",
    "SWA_OUT_PROJ_VEC_N_CHUNK",
    "SWA_OUT_PROJ_MATMUL_TILES_PER_TASK",
    "SWA_OUT_PROJ_FUSE_CAST",
    "MLP_OUT_CHUNK",
    "MLP_SPMD_INNER",
    "MLP_GROUP_CHUNK",
    "DOWN_MLP_CHUNK",
    "DOWN_OUT_CHUNK",
    "FINAL_RMS_K_CHUNK",
    "LM_HEAD_K_CHUNK",
    "VOCAB_CHUNK",
    "Q_HEAD_BATCH_FULL",
    "Q_HEAD_BATCH_SWA",
    "Q_HEAD_PAD_FULL",
    "Q_HEAD_PAD_SWA",
    "MAX_BLOCKS_PER_SEQ",
    "PTO2_LOGICAL_BLOCK_LIMIT",
    "ATTN_TASK_PROFILE",
    "FULL_ATTN_QK_BLOCKS_PER_TASK",
    "FULL_ATTN_SOFTMAX_BLOCKS_PER_TASK",
    "FULL_ATTN_ONLINE_SOFTMAX_BLOCKS_PER_TASK",
    "FULL_ATTN_ONLINE_SOFTMAX_PARTIALS_PER_REDUCE_TASK",
    "FULL_ATTN_QK_UNIFORM_O1",
    "FULL_ATTN_SOFTMAX_UNIFORM_O1",
    "FULL_ATTN_ONLINE_SOFTMAX_UNIFORM_O1",
    "FULL_ATTN_ONLINE_SOFTMAX_REDUCE_UNIFORM_O1",
    # distributed topology
    "TP_WORLD_SIZE",
    "EP_WORLD_SIZE",
    "NUM_HEADS_FULL_LOCAL",
    "NUM_HEADS_SWA_LOCAL",
    "NUM_HEADS_FULL_LOCAL_PAD",
    "NUM_HEADS_SWA_LOCAL_PAD",
    "KV_HEADS_LOCAL",
    "HIDDEN_Q_FULL_LOCAL",
    "HIDDEN_Q_SWA_LOCAL",
    "KV_HIDDEN_LOCAL",
    "INTERMEDIATE_LOCAL",
    "SHARE_EXPERT_DIM_LOCAL",
    "VOCAB_LOCAL",
    "MOE_NUM_EXPERTS_LOCAL",
    "ep_expert_owner",
    "ep_local_expert_id",
    "ep_global_expert_id",
]
