# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Source contracts for SWA runtime task publication."""
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[3]
_SOURCE = (
    _ROOT / "models" / "step3p5" / "attention_swa.py"
).read_text(encoding="utf-8")


def test_swa_attention_stages_capture_and_chain_runtime_tasks() -> None:
    assert "name_hint=\"swa_rope_q\"" in _SOURCE
    assert ") as swa_rope_q_tid:" in _SOURCE
    assert "name_hint=\"swa_rope_kv_cache\"" in _SOURCE
    assert ") as swa_rope_kv_tid:" in _SOURCE
    assert "with pl.spmd(\n        swa_active_tasks,\n        name_hint=\"swa_qk_matmul\"" in _SOURCE
    assert (
        "name_hint=\"swa_qk_matmul\",\n"
        "        deps=[swa_rope_q_tid, swa_rope_kv_tid],"
    ) in _SOURCE
    assert ") as swa_qk_tid:" in _SOURCE
    assert "name_hint=\"swa_softmax\",\n        deps=[swa_qk_tid]," in _SOURCE
    assert ") as swa_softmax_tid:" in _SOURCE
    assert "name_hint=\"swa_sv_matmul\",\n        deps=[swa_softmax_tid]," in _SOURCE
    assert ") as swa_sv_tid:" in _SOURCE
    assert "name_hint=\"swa_online_softmax\",\n        deps=[swa_sv_tid]," in _SOURCE
