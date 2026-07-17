# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Categorized tests and execution harnesses for the Step3p5 model.

The package is organized by responsibility:

* ``common``: topology/configuration patch helpers;
* ``probes``: focused compiler and runtime diagnostics;
* ``harnesses``: device, real-weight, IPC, and canonical runners;
* ``unit``: card-free and focused contract tests;
* ``system``: layer-level and multi-device tests;
* ``precision``: golden and end-to-end precision tests;
* ``ci``: whole-network CI orchestration and reports.

Each categorized ``test_<op>.py`` remains a runnable script:

    cd /data/chensiyu/hw_project/pypto/workspace/pypto-lib
    python -m tests.step3p5.unit.test_rms_lm_head --smoke -p a2a3sim
    python -m tests.step3p5.unit.test_rms_lm_head -p a2a3 -d 0

The shared ``_tp1_setup`` helper monkey-patches ``models.step3p5.config``
to a single-rank topology BEFORE the kernel modules are imported, so the
canonical ``models/step3p5/*.py`` sources stay TP=8/EP=8 — single-card
runs are an isolated test-only context. This mirrors the proven
``models.step3p5.step3p5_decode.run_real_npu`` Phase 15 path.
"""
