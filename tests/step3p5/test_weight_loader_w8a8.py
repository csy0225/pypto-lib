from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.step3p5.weight_loader import (
    _dequant_w8a8_dynamic_weight,
    _has_quantized_routed_experts,
    _read_index,
)


def test_w8a8_quant_index_is_supported(tmp_path) -> None:
    index = {
        "metadata": {"total_size": 3},
        "weight_map": {
            "model.embed_tokens.weight": "quant_model_weights-00001-of-00001.safetensors",
            "model.layers.3.moe.experts.0.gate_proj.weight": "quant_model_weights-00001-of-00001.safetensors",
        },
    }
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(json.dumps(index))

    weight_map = _read_index(str(tmp_path))

    assert weight_map == index["weight_map"]
    assert _has_quantized_routed_experts(weight_map, 3)
    assert not _has_quantized_routed_experts(weight_map, 4)


def test_w8a8_dynamic_dequant_applies_scale_and_offset() -> None:
    weight = torch.tensor([[2, -4, 8], [10, 20, -30]], dtype=torch.int8)
    scale = torch.tensor([[0.5], [0.25]], dtype=torch.float32)
    offset = torch.tensor([[1.0], [-2.0]], dtype=torch.float32)

    got = _dequant_w8a8_dynamic_weight(weight, scale, offset)
    expected = torch.tensor(
        [[0.5, -2.5, 3.5], [3.0, 5.5, -7.0]],
        dtype=torch.bfloat16,
    )

    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)
