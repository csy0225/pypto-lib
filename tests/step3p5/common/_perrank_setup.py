# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""TP=1/EP=1 codegen patch that PRESERVES TP=8 per-rank slice widths.

This is the canonical helper for **single-card kernel-level ST/UT** —
each rank carries its own TP=8 slice (NUM_HEADS_FULL_LOCAL=8,
INTERMEDIATE_LOCAL=1408, SHARE_EXPERT_DIM_LOCAL=160,
MOE_NUM_EXPERTS_LOCAL=36, KV_HEADS_LOCAL=1, etc.). TP_WORLD_SIZE and
EP_WORLD_SIZE are flipped to 1 so codegen elides the tp_all_reduce /
ep_all_to_all collectives (1 rank cannot communicate). All other
constants stay at their canonical TP=8 per-rank values.

CONTRAST with ``_tp1_setup.py::apply_tp1_patch()`` which UN-slices the
``*_LOCAL`` constants to their full unsliced values (1280 / 11264 /
288 / 8 / 64) for Phase 15 e2e — putting 8-card aggregate work onto
1 rank. That helper is **wrong** for ST/UT: kernel chunks like
``SHARED_GATE_N_CHUNK = INTER_S_LOCAL`` follow the slice and overflow
L1/UB when the slice grows.

See the iron-rule section in CLAUDE.md (single-card ST/UT shape) for
the recurring-mistake context.
"""

from __future__ import annotations

import importlib
from types import ModuleType


def apply_perrank_patch(*, reload_modules: list[str] | None = None) -> dict[str, int]:
    """Patch ``models.step3p5.config`` for TP=1/EP=1 codegen + TP=8 widths.

    Sets ``TP_WORLD_SIZE = EP_WORLD_SIZE = 1`` so distributed codegen
    skips collectives (1 rank, no peers). Leaves every ``*_LOCAL``
    constant and every ``LAYER_*_DYN`` value at its canonical TP=8
    per-rank slice width. The reloaded kernel modules pick up the new
    world-size attributes but keep canonical slice widths for tile /
    chunk sizing.

    Args:
        reload_modules: Fully-qualified module names to ``importlib.reload``
            after patching. Order matters when modules cross-reference
            each other (``attention_full`` before ``decode_layer``, etc.).

    Returns:
        Summary dict for log/debug.
    """
    cfg: ModuleType = importlib.import_module("models.step3p5.config")

    cfg.TP_WORLD_SIZE = 1
    cfg.EP_WORLD_SIZE = 1
    # NOTE: do NOT touch *_LOCAL or LAYER_*_DYN. They stay at their
    # canonical TP=8 per-rank slice values:
    #   NUM_HEADS_FULL_LOCAL = 8       (= 64 // 8)
    #   NUM_HEADS_SWA_LOCAL = 12       (= 96 // 8)
    #   KV_HEADS_LOCAL = 1             (= 8 // 8)
    #   HIDDEN_Q_FULL_LOCAL = 1024     (= 8 * 128)
    #   HIDDEN_Q_SWA_LOCAL = 1536      (= 12 * 128)
    #   KV_HIDDEN_LOCAL = 128          (= 1 * 128)
    #   INTERMEDIATE_LOCAL = 1408      (= 11264 // 8)
    #   SHARE_EXPERT_DIM_LOCAL = 160   (= 1280 // 8)
    #   VOCAB_LOCAL = 16112            (= 128896 // 8)
    #   MOE_NUM_EXPERTS_LOCAL = 36     (= 288 // 8)
    #   LAYER_INTER_ROWS_DYN = 4224    (= 3 * 1408)
    #   LAYER_HIDDEN_ROWS_DYN = 49152  (= 12 * 4096)
    summary: dict[str, int] = {
        "TP_WORLD_SIZE": cfg.TP_WORLD_SIZE,
        "EP_WORLD_SIZE": cfg.EP_WORLD_SIZE,
        "NUM_HEADS_FULL_LOCAL": cfg.NUM_HEADS_FULL_LOCAL,
        "NUM_HEADS_SWA_LOCAL": cfg.NUM_HEADS_SWA_LOCAL,
        "KV_HEADS_LOCAL": cfg.KV_HEADS_LOCAL,
        "INTERMEDIATE_LOCAL": cfg.INTERMEDIATE_LOCAL,
        "SHARE_EXPERT_DIM_LOCAL": cfg.SHARE_EXPERT_DIM_LOCAL,
        "MOE_NUM_EXPERTS_LOCAL": cfg.MOE_NUM_EXPERTS_LOCAL,
        "LAYER_INTER_ROWS_DYN": cfg.LAYER_INTER_ROWS_DYN,
    }

    if reload_modules:
        for mod_name in reload_modules:
            mod = importlib.import_module(mod_name)
            importlib.reload(mod)

    return summary
