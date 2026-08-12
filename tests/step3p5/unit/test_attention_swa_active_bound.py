"""SWA decode attention contracts for workload-derived RoPE/KV tasks."""
from __future__ import annotations

import ast
from pathlib import Path

import torch


_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_PATH = _ROOT / "models" / "step3p5" / "attention_swa.py"


def _source() -> str:
    return _SOURCE_PATH.read_text(encoding="utf-8")


def _attention_function() -> ast.FunctionDef:
    tree = ast.parse(_source())
    matches = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "attention_swa"
    ]
    assert len(matches) == 1
    return matches[0]


def _call_name(node: ast.AST) -> str:
    if not isinstance(node, ast.Call):
        return ""
    parts: list[str] = []
    value: ast.AST = node.func
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


def _spmd_scope(function: ast.FunctionDef, hint: str) -> ast.With:
    matches = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.With)
        and len(node.items) == 1
        and _call_name(node.items[0].context_expr) == "pl.spmd"
        and any(
            keyword.arg == "name_hint"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == hint
            for keyword in node.items[0].context_expr.keywords
        )
    ]
    assert len(matches) == 1, f"expected one {hint!r} SPMD scope"
    return matches[0]


def _guarded_source(source: str, scope: ast.With) -> str:
    guards = [
        node
        for node in scope.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "b < active_tokens"
    ]
    assert len(guards) == 1
    guarded = ast.get_source_segment(source, guards[0])
    assert guarded is not None
    return guarded


def _window_block_spans(
    context_len: int,
    *,
    window: int = 512,
    block_size: int = 128,
) -> list[tuple[int, int, int]]:
    """Return ``(logical_block, valid_lo, valid_hi)`` for an SWA window."""
    window_start = max(0, context_len - window)
    first_block = window_start // block_size
    end_block = (context_len + block_size - 1) // block_size
    return [
        (
            block,
            max(window_start, block * block_size) - block * block_size,
            min(context_len, (block + 1) * block_size) - block * block_size,
        )
        for block in range(first_block, end_block)
    ]


def test_swa_formal_keeps_capacity_and_runtime_num_tokens() -> None:
    function = _attention_function()
    args = {arg.arg: ast.unparse(arg.annotation) for arg in function.args.args}
    assert "num_tokens" in args
    assert "BATCH" in args["current_hidden"]
    assert "USER_BATCH_DYN" in args["seq_lens"]
    assert "USER_BATCH_DYN" in args["slot_mapping"]
    assert ast.unparse(function.returns) == "pl.Tensor[[BATCH, HIDDEN], pl.BF16]"


def test_swa_packs_qkv_projection_and_fuses_active_prerope_task() -> None:
    source = _source()
    function = _attention_function()

    qkv_proj_scope = _spmd_scope(function, "swa_qkv_proj")
    qkv_proj_call = qkv_proj_scope.items[0].context_expr
    assert (
        ast.unparse(qkv_proj_call.args[0])
        == "BATCH // BATCH_TILE * SWA_QKV_PROJ_BLOCKS"
    )
    assert any(
        keyword.arg == "allow_early_resolve"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in qkv_proj_call.keywords
    )
    assert (
        isinstance(qkv_proj_scope.items[0].optional_vars, ast.Name)
        and qkv_proj_scope.items[0].optional_vars.id == "swa_qkv_proj_tid"
    )
    qkv_proj_source = ast.get_source_segment(source, qkv_proj_scope)
    assert qkv_proj_source is not None
    assert "qkv_spmd_idx = pl.tile.get_block_idx()" in qkv_proj_source
    assert "if qkv_ob < SWA_QKV_Q_BLOCKS:" in qkv_proj_source
    assert "qkv_kind = qkv_ob - SWA_QKV_Q_BLOCKS" in qkv_proj_source
    assert "[qkv_b0, SWA_QKV_K_OFFSET]" in qkv_proj_source
    assert "[qkv_b0, SWA_QKV_V_OFFSET]" in qkv_proj_source
    for weight in ("wq", "wk", "wv"):
        assert weight in qkv_proj_source

    head_logits_scope = _spmd_scope(function, "swa_head_gate_logits_mm")
    head_logits_call = head_logits_scope.items[0].context_expr
    assert any(
        keyword.arg == "deps"
        and ast.unparse(keyword.value) == "[swa_attn_out_zero_tid]"
        for keyword in head_logits_call.keywords
    )
    assert "hg_part = pl.tile.get_block_idx()" in (
        ast.get_source_segment(source, head_logits_scope) or ""
    )
    assert source.index('name_hint="swa_attn_out_zero"') < source.index(
        'name_hint="swa_head_gate_logits_mm"'
    )
    assert source.index('name_hint="swa_qkv_proj"') < source.index(
        'name_hint="swa_head_gate_expand"'
    ) < source.index('name_hint="swa_qkv_split_qknorm_rope"')

    prerope_scope = _spmd_scope(function, "swa_qkv_split_qknorm_rope")
    prerope_call = prerope_scope.items[0].context_expr
    assert ast.unparse(prerope_call.args[0]) == "active_tokens"
    assert any(
        keyword.arg == "allow_early_resolve"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in prerope_call.keywords
    )
    assert (
        isinstance(prerope_scope.items[0].optional_vars, ast.Name)
        and prerope_scope.items[0].optional_vars.id == "swa_qkv_prerope_tid"
    )
    assert any(
        keyword.arg == "deps"
        and ast.unparse(keyword.value) == "[swa_qkv_proj_tid]"
        for keyword in prerope_call.keywords
    )
    prerope_source = _guarded_source(source, prerope_scope)
    assert "pl.tensor.read(seq_lens, [b])" in prerope_source
    assert "pl.tensor.read(slot_mapping, [b])" in prerope_source
    assert "qkv_proj" in prerope_source
    assert "qk_sq = pl.row_sum(pl.mul(qk_chunk, qk_chunk))" in prerope_source
    assert "pl.concat(pl.concat(q_flat, k_chunk), qk_zero_pad)" in prerope_source
    assert "k_scaled = pl.slice(" in prerope_source
    assert "[Q_HEAD_BATCH_SWA, 0]" in prerope_source
    assert "k_chunk_8" not in prerope_source
    assert "q_pad = pl.slice" not in prerope_source
    assert "[16, HEAD_DIM]" in prerope_source
    assert "q_lo = pl.slice(" in prerope_source
    assert "q_hi = pl.slice(" in prerope_source
    assert "[16, ROTARY_HALF_SWA]" in prerope_source
    assert "[pad_row_base, 0]" in prerope_source
    assert "[pad_row_base, ROTARY_HALF_SWA]" in prerope_source
    assert "for qh in pl.range(Q_HEAD_BATCH_SWA)" not in prerope_source
    assert "q_head = pl.slice" not in prerope_source
    assert "all_q_padded = pl.assemble(" in prerope_source
    assert "k_cache = pl.assemble(" in prerope_source
    assert "v_cache = pl.assemble(" in prerope_source

    function_source = ast.unparse(function)
    for retired_hint in (
        "swa_q_proj",
        "swa_kv_proj",
        "swa_qk_norm_zc",
        "swa_rope_q",
        "swa_rope_kv_cache",
    ):
        assert f"name_hint='{retired_hint}'" not in function_source
    for retired_tensor in (
        "q_proj",
        "k_proj",
        "v_proj",
        "q_proj_norm",
        "k_proj_norm",
    ):
        assert f"    {retired_tensor} = pl.create_tensor" not in source
    assert "swa_rope_stage = pl.create_tensor" not in function_source
    assert "swa_k_rope_stage = pl.create_tensor" not in function_source
    assert "swa_v_stage = pl.create_tensor" not in function_source

    mix_call = _spmd_scope(function, "swa_attn_mix").items[0].context_expr
    assert any(
        keyword.arg == "deps"
        and ast.unparse(keyword.value) == "[swa_qkv_prerope_tid]"
        for keyword in mix_call.keywords
    )


def test_swa_has_no_padding_slot_fallback_in_decode_rope_kv_path() -> None:
    source = _source()
    start = source.index("    all_q_padded = pl.create_tensor(")
    end = source.index("    # ----- Mixed attention core:", start)
    scope2 = source[start:end]
    assert "b_safe" not in scope2
    assert "slot_mapping, [b]" in scope2
    assert "slot_mapping, [b_safe]" not in scope2
    assert scope2.count("        active_tokens,") == 1
    assert "if b < active_tokens:" in scope2


def test_swa_tail_window_uses_chronological_trailing_blocks() -> None:
    aligned = _window_block_spans(65536)
    assert aligned == [
        (508, 0, 128),
        (509, 0, 128),
        (510, 0, 128),
        (511, 0, 128),
    ]

    unaligned = _window_block_spans(65535)
    assert unaligned == [
        (507, 127, 128),
        (508, 0, 128),
        (509, 0, 128),
        (510, 0, 128),
        (511, 0, 127),
    ]
    assert sum(valid_hi - valid_lo for _, valid_lo, valid_hi in unaligned) == 512


def test_swa_source_indexes_tail_window_and_masks_both_edges() -> None:
    source = _source()
    function_source = ast.unparse(_attention_function())

    assert function_source.count(
        "fa_window_start = pl.max(0, fa_ctx_len - SLIDING_WINDOW)"
    ) == 1
    assert function_source.count(
        "fa_first_block = fa_window_start // BLOCK_SIZE"
    ) == 1
    assert function_source.count(
        "fa_end_block = (fa_ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE"
    ) == 1
    assert source.count("[fa_block_table_base + fa_block]") == 1
    assert "[fa_block_table_base + sb]" not in source
    assert "SWA_STORAGE_BLOCKS" not in function_source
    assert "valid_lo = pl.max(" in source
    assert "valid_hi = pl.min(" in source
    assert "[fa_cache_row + valid_lo, 0]" not in source
    assert source.count("[fa_cache_row, 0]") == 2
    assert "zero_i32 = pl.const(0, pl.INT32)" in source
    assert "one_i32 = pl.const(1, pl.INT32)" in source
    assert "valid_from_i32 = pl.minimum(" in source
    assert "valid_to_i32 = pl.minimum(" in source
    assert "pl.neg(" in source
    assert "pl.cmp(" not in source
    assert "valid_mask = pl.cast(" in source
    assert "scores = pl.col_expand_add(scores, invalid_bias)" in source
    assert "all_raw_scores" not in function_source
    assert "all_exp_padded" not in function_source
    assert "all_oi_tmp" not in function_source
    assert "mi_new = pl.maximum(mi, cur_mi)" in function_source
    assert "ctx = pl.row_expand_div(oi, li)" in function_source

def _swa_reverse_oracle_fixture(
    *,
    reverse: bool,
    perturb_masked_edges: bool = False,
    perturb_valid_block: int | None = None,
) -> tuple[dict, torch.Tensor]:
    """Build logically identical linear/reverse 64K KV page layouts."""
    context_len = 65535
    block_size = 128
    num_blocks = 512
    head_dim = 8
    eps = 1.0e-6
    hidden = torch.tensor(
        [[1.0, -0.5, 0.75, -1.25, 0.375, 1.5, -0.875, 0.625]],
        dtype=torch.bfloat16,
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(0x5A17)
    logical_k = (
        torch.randn(
            num_blocks,
            block_size,
            head_dim,
            generator=generator,
        )
        * 0.2
    )
    logical_v = (
        torch.randn(
            num_blocks,
            block_size,
            head_dim,
            generator=generator,
        )
        * 0.2
    )
    # Give every logical block a non-periodic identity in addition to random
    # token data, so a wrong five-block window cannot hide inside tolerance.
    block_ids = (
        torch.arange(num_blocks, dtype=torch.float32).reshape(-1, 1, 1)
        - (num_blocks - 1) / 2
    )
    dim_sign = torch.tensor(
        [1.0, -1.0, 0.75, -0.75, 0.5, -0.5, 0.25, -0.25],
    ).reshape(1, 1, -1)
    logical_k = (
        logical_k + block_ids * 0.002 * torch.flip(dim_sign, dims=(-1,))
    ).to(torch.bfloat16)
    logical_v = (
        logical_v + block_ids * 0.01 * dim_sign
    ).to(torch.bfloat16)
    if perturb_masked_edges:
        # ctx=65535 selects block507[127:] ... block511[:127].
        logical_k[507, 126] = torch.tensor(
            [48.0, -48.0, 40.0, -40.0, 32.0, -32.0, 24.0, -24.0],
        )
        logical_k[511, 127] = -logical_k[507, 126]
        logical_v[507, 126] = torch.tensor(
            [64.0, -64.0, 56.0, -56.0, 48.0, -48.0, 40.0, -40.0],
        )
        logical_v[511, 127] = -logical_v[507, 126]
    if perturb_valid_block is not None:
        if perturb_valid_block not in range(507, 512):
            raise ValueError("valid SWA perturb block must be in [507,511]")
        valid_offset = 127 if perturb_valid_block == 507 else 64
        logical_k[perturb_valid_block, valid_offset] = torch.tensor(
            [24.0, -24.0, 20.0, -20.0, 16.0, -16.0, 12.0, -12.0],
        )
        logical_v[perturb_valid_block, valid_offset] = torch.tensor(
            [-32.0, 32.0, -28.0, 28.0, -24.0, 24.0, -20.0, 20.0],
        )

    logical_to_physical = (
        torch.arange(num_blocks - 1, -1, -1, dtype=torch.int32)
        if reverse
        else torch.arange(num_blocks, dtype=torch.int32)
    )
    if reverse:
        assert int(logical_to_physical[507].item()) == 4
        assert int(logical_to_physical[511].item()) == 0
    physical_k = torch.empty_like(logical_k)
    physical_v = torch.empty_like(logical_v)
    for logical_block, physical_block in enumerate(
        logical_to_physical.tolist(),
    ):
        physical_k[physical_block].copy_(logical_k[logical_block])
        physical_v[physical_block].copy_(logical_v[logical_block])

    current_block = (context_len - 1) // block_size
    current_offset = (context_len - 1) % block_size
    current_physical = int(logical_to_physical[current_block].item())
    slot_mapping = torch.tensor(
        [current_physical * block_size + current_offset],
        dtype=torch.int32,
    )
    identity = torch.eye(head_dim, dtype=torch.float32).bfloat16()
    rope_cos = torch.ones(context_len, head_dim, dtype=torch.float32)
    rope_sin = torch.zeros_like(rope_cos)
    kwargs = {
        "hidden_states": hidden,
        "input_rms_weight": torch.zeros(1, head_dim),
        "wq_full": identity,
        "wk_full": identity,
        "wv_full": identity,
        "q_norm_weight": torch.zeros(1, head_dim),
        "k_norm_weight": torch.zeros(1, head_dim),
        "wo_full": identity,
        "w_g_full": torch.tensor(
            [[1.0], [-0.75], [0.5], [-0.25], [0.875], [-1.0], [0.625], [-0.5]],
            dtype=torch.bfloat16,
        ),
        "seq_lens": torch.tensor([context_len], dtype=torch.int32),
        "block_table": logical_to_physical,
        "slot_mapping": slot_mapping,
        "rope_cos": rope_cos,
        "rope_sin": rope_sin,
        "k_cache_full": physical_k.reshape(-1, head_dim),
        "v_cache_full": physical_v.reshape(-1, head_dim),
        "num_heads_full": 1,
        "num_kv_heads_full": 1,
        "head_dim": head_dim,
        "rotary_half": head_dim // 2,
        "q_per_kv": 1,
        "eps": eps,
        "block_size": block_size,
        "sliding_window": 512,
    }

    x = hidden.float()
    normed = (
        x
        * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    ).bfloat16()
    q_normed = (
        normed.float()
        * torch.rsqrt(
            normed.float().pow(2).mean(dim=-1, keepdim=True) + eps,
        )
    ).bfloat16()
    k_current = q_normed
    visible_k = logical_k.reshape(-1, head_dim)[
        context_len - 512:context_len
    ].clone()
    visible_v = logical_v.reshape(-1, head_dim)[
        context_len - 512:context_len
    ].clone()
    visible_k[-1] = k_current
    visible_v[-1] = normed
    scores = (
        q_normed.float()
        @ visible_k.float().T
        / head_dim ** 0.5
    )
    context = (torch.softmax(scores, dim=-1) @ visible_v.float()).bfloat16()[0]
    gate = (
        torch.sigmoid(
            normed.float()
            @ kwargs["w_g_full"].float(),
        )
        .bfloat16()
        .float()[0, 0]
    )
    gated = (context.float() * gate).bfloat16()
    expected = (gated.float() + hidden[0].float()).bfloat16().reshape(1, -1)
    return kwargs, expected


def test_swa_unaligned_reverse_window_matches_independent_direct_oracle() -> None:
    from models.step3p5.attention_swa import (
        _torch_single_card_attention_swa,
    )

    linear, expected = _swa_reverse_oracle_fixture(reverse=False)
    reverse, _ = _swa_reverse_oracle_fixture(reverse=True)
    linear_output = _torch_single_card_attention_swa(**linear)
    reverse_output = _torch_single_card_attention_swa(**reverse)

    torch.testing.assert_close(linear_output, expected, rtol=0.005, atol=0.005)
    torch.testing.assert_close(reverse_output, expected, rtol=0.005, atol=0.005)
    assert torch.equal(reverse_output, linear_output)

    wrong = dict(reverse)
    wrong_table = reverse["block_table"].clone()
    wrong_table[507:512] = torch.tensor([507, 0, 1, 2, 3])
    wrong["block_table"] = wrong_table
    wrong_output = _torch_single_card_attention_swa(**wrong)
    assert float((wrong_output.float() - expected.float()).abs().max()) > 0.01


def test_swa_unaligned_window_masks_perturbed_prefix_and_suffix_tokens() -> None:
    from models.step3p5.attention_swa import (
        _torch_single_card_attention_swa,
    )

    reverse, _ = _swa_reverse_oracle_fixture(reverse=True)
    perturbed, _ = _swa_reverse_oracle_fixture(
        reverse=True,
        perturb_masked_edges=True,
    )
    baseline_output = _torch_single_card_attention_swa(**reverse)
    perturbed_output = _torch_single_card_attention_swa(**perturbed)

    assert torch.equal(perturbed_output, baseline_output)

    for valid_block in range(507, 512):
        valid_perturbed, _ = _swa_reverse_oracle_fixture(
            reverse=True,
            perturb_valid_block=valid_block,
        )
        valid_perturbed_output = _torch_single_card_attention_swa(
            **valid_perturbed,
        )
        assert not torch.equal(valid_perturbed_output, baseline_output)


def test_decode_torch_oracles_compute_head_gate_from_normalized_hidden() -> None:
    full_source = (
        _ROOT / "models" / "step3p5" / "attention_full.py"
    ).read_text(encoding="utf-8")
    swa_source = _source()

    assert full_source.count(
        "torch.sigmoid(normed_bf16.float() @ w_g_"
    ) == 2
    assert "torch.sigmoid(hidden_states.float() @ w_g_" not in full_source
    assert (
        "gate_logits = normed_bf16.float() @ w_g_full.float()"
        in swa_source
    )
    assert (
        "torch.sigmoid(gate_logits).bfloat16().float()"
        in swa_source
    )
    assert (
        "torch.sigmoid(normed_bf16.float() @ w_g_local.float())"
        in swa_source
    )
