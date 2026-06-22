# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""TP=1/EP=1 monkey-patch helper for per-operator precision tests.

Mirrors the proven Phase 15 single-rank patch in
``models.step3p5.step3p5_decode.run_real_npu`` (lines 369-419). Applies
to ``models.step3p5.config`` and reloads the kernel-module(s) the caller
asks for so they re-bake the single-rank constants into their tensor
shape annotations and ``pl.parallel`` block counts.

The canonical step3p5/* source stays TP=8/EP=8 — these patches are
transient, applied per-test-process only.
"""

from __future__ import annotations

import importlib
import math
from types import ModuleType


def apply_tp1_patch(*, reload_modules: list[str] | None = None) -> dict[str, int]:
    """Patch ``models.step3p5.config`` for a TP=1/EP=1 single-card run.

    Patches the per-rank sliced constants (``*_LOCAL`` family) to equal
    their unsliced counterparts, then reloads the modules in
    ``reload_modules`` so they re-evaluate their tensor-shape annotations
    against the patched config. The ``LAYER_*_DYN`` overrides match the
    in-tree run_real_npu values (3 dense MLP * INTERMEDIATE_LOCAL,
    12 full-attn * HIDDEN_Q_FULL_LOCAL).

    Args:
        reload_modules: Fully-qualified module names to ``importlib.reload``
            after patching. Order matters when modules cross-reference each
            other - pass dependency order (``attention_full`` before
            ``decode_layer``, etc.).

    Returns:
        A dict of patched constants for log/debug, e.g.
        ``{"TP_WORLD_SIZE": 1, "VOCAB_LOCAL": 128896, ...}``.
    """
    cfg: ModuleType = importlib.import_module("models.step3p5.config")

    cfg.TP_WORLD_SIZE = 1
    cfg.EP_WORLD_SIZE = 1

    # Per-rank widths collapse to their unsliced counterparts.
    cfg.NUM_HEADS_FULL_LOCAL = cfg.NUM_HEADS_FULL
    cfg.NUM_HEADS_SWA_LOCAL = cfg.NUM_HEADS_SWA
    cfg.KV_HEADS_LOCAL = cfg.NUM_KV_HEADS
    cfg.HIDDEN_Q_FULL_LOCAL = cfg.HIDDEN_Q_FULL
    cfg.HIDDEN_Q_SWA_LOCAL = cfg.HIDDEN_Q_SWA
    cfg.KV_HIDDEN_LOCAL = cfg.KV_HIDDEN
    cfg.INTERMEDIATE_LOCAL = cfg.INTERMEDIATE
    cfg.SHARE_EXPERT_DIM_LOCAL = cfg.SHARE_EXPERT_DIM
    cfg.VOCAB_LOCAL = cfg.VOCAB
    cfg.MOE_NUM_EXPERTS_LOCAL = cfg.MOE_NUM_EXPERTS

    # 16-aligned head-count pads (used by attention front-matter).
    cfg.NUM_HEADS_FULL_LOCAL_PAD = math.ceil(cfg.NUM_HEADS_FULL / 16) * 16
    cfg.NUM_HEADS_SWA_LOCAL_PAD = math.ceil(cfg.NUM_HEADS_SWA / 16) * 16

    # KV-projection K-chunk: TP=1 KV_HIDDEN_LOCAL (1024) > the canonical
    # INPUT_PROJ_K_CHUNK=256, drop to 128 to fit the L0 buffer budget.
    cfg.KV_PROJ_K_CHUNK_LOCAL = cfg.KV_PROJ_K_CHUNK

    # ``LAYER_INTER_ROWS_DYN`` is a static-baked dyn dim in config.py
    # (workaround for upstream pypto bugs #3/#4 - see
    # docs/known-pypto-pitfalls.md). It is sized at TP=8 by default; under
    # TP=1 the per-rank widths multiply 8x, so the bound must rise too.
    n_dense_mlp_layers = 3
    cfg.LAYER_INTER_ROWS_DYN = n_dense_mlp_layers * cfg.INTERMEDIATE_LOCAL

    summary: dict[str, int] = {
        "TP_WORLD_SIZE": cfg.TP_WORLD_SIZE,
        "EP_WORLD_SIZE": cfg.EP_WORLD_SIZE,
        "VOCAB_LOCAL": cfg.VOCAB_LOCAL,
        "INTERMEDIATE_LOCAL": cfg.INTERMEDIATE_LOCAL,
        "KV_HIDDEN_LOCAL": cfg.KV_HIDDEN_LOCAL,
        "NUM_HEADS_FULL_LOCAL": cfg.NUM_HEADS_FULL_LOCAL,
        "NUM_HEADS_SWA_LOCAL": cfg.NUM_HEADS_SWA_LOCAL,
        "MOE_NUM_EXPERTS_LOCAL": cfg.MOE_NUM_EXPERTS_LOCAL,
        "LAYER_INTER_ROWS_DYN": cfg.LAYER_INTER_ROWS_DYN,
    }

    if reload_modules:
        for mod_name in reload_modules:
            mod = importlib.import_module(mod_name)
            reloaded = importlib.reload(mod)
            # ``LAYER_QHIDDEN_ROWS_DYN`` lives on attention_full / attention_swa,
            # not config. When either is reloaded, patch the per-module dyn
            # dim AFTER the reload so the kernel's tensor shape annotations
            # pick it up on first re-import by downstream modules
            # (decode_layer / prefill_layer).
            if mod_name.endswith("attention_full"):
                n_full_attn_layers = 12
                reloaded.LAYER_QHIDDEN_ROWS_DYN = (
                    n_full_attn_layers * cfg.HIDDEN_Q_FULL_LOCAL
                )
                summary["LAYER_QHIDDEN_ROWS_DYN_FULL"] = (
                    reloaded.LAYER_QHIDDEN_ROWS_DYN
                )
            if mod_name.endswith("attention_swa"):
                n_swa_attn_layers = 33  # SWA count in LAYER_TYPES[0..44]
                reloaded.LAYER_QHIDDEN_ROWS_DYN = (
                    n_swa_attn_layers * cfg.HIDDEN_Q_SWA_LOCAL
                )
                summary["LAYER_QHIDDEN_ROWS_DYN_SWA"] = (
                    reloaded.LAYER_QHIDDEN_ROWS_DYN
                )

    return summary
