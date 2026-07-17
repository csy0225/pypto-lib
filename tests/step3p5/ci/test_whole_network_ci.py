# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pytest gate for the Step3p5 whole-network hardware CI runner.

This test is skipped by default.  A dedicated eight-device CI job enables it
with ``STEP3P5_WHOLE_NET_CI=1`` and supplies the checkpoint/device variables.
Keeping the default skip is important: ordinary CPU or small-device pytest
jobs must not allocate the 25+ GB/rank IPC pool or consume NPU cards.
"""
from __future__ import annotations

import os

import pytest

from tests.step3p5.ci.run_whole_network_ci import (
    config_from_environment,
    run,
)


def test_step3p5_whole_network_ci() -> None:
    """Run the canonical main -> sampler -> MTP3 hardware gate when enabled."""
    if os.environ.get("STEP3P5_WHOLE_NET_CI") != "1":
        pytest.skip("set STEP3P5_WHOLE_NET_CI=1 on a dedicated 8-device runner")
    report = run(config_from_environment())
    assert report["ok"], (
        "Step3p5 whole-network CI failed; "
        f"report={report.get('paths', {}).get('json_report')} "
        f"failure={report.get('failure')}"
    )
