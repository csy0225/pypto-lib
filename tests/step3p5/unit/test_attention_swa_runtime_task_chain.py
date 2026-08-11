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
    assert 'name_hint="swa_qkv_proj"' in _SOURCE
    assert ") as swa_qkv_proj_tid:" in _SOURCE
    assert 'name_hint="swa_qkv_split_qknorm_rope"' in _SOURCE
    assert (
        'name_hint="swa_qkv_split_qknorm_rope",\n'
        "        deps=[swa_qkv_proj_tid],"
    ) in _SOURCE
    assert ") as swa_qkv_prerope_tid:" in _SOURCE
    for retired_hint in (
        "swa_q_proj",
        "swa_kv_proj",
        "swa_qk_norm_zc",
        "swa_rope_q",
        "swa_rope_kv_cache",
    ):
        assert f'name_hint="{retired_hint}"' not in _SOURCE
    assert (
        "with pl.spmd(\n"
        "        swa_active_tasks,\n"
        '        name_hint="swa_attn_mix"'
    ) in _SOURCE
    assert (
        'name_hint="swa_attn_mix",\n'
        "        deps=[swa_qkv_prerope_tid],"
    ) in _SOURCE
    for old_hint in (
        "swa_qk_matmul",
        "swa_softmax",
        "swa_sv_matmul",
        "swa_online_softmax",
    ):
        assert f'name_hint="{old_hint}"' not in _SOURCE
    for scratch in (
        "all_raw_scores",
        "all_exp_padded",
        "all_cur_mi",
        "all_cur_li",
        "all_oi_tmp",
    ):
        assert scratch not in _SOURCE
    assert "mi_new = pl.maximum(mi, cur_mi)" in _SOURCE
    assert "ctx = pl.row_expand_div(oi, li)" in _SOURCE
