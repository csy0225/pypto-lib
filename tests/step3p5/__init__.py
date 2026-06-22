# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Per-operator and per-layer precision tests for the step3p5 model.

Each ``test_<op>.py`` is a runnable script:

    cd /data/chensiyu/hw_project/pypto/workspace/pypto-lib
    python -m tests.step3p5.test_rms_lm_head --smoke -p a2a3sim
    python -m tests.step3p5.test_rms_lm_head -p a2a3 -d 0

The shared ``_tp1_setup`` helper monkey-patches ``models.step3p5.config``
to a single-rank topology BEFORE the kernel modules are imported, so the
canonical ``models/step3p5/*.py`` sources stay TP=8/EP=8 — single-card
runs are an isolated test-only context. This mirrors the proven
``models.step3p5.step3p5_decode.run_real_npu`` Phase 15 path.
"""
