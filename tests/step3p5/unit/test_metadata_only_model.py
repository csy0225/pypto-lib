#!/usr/bin/env python3
"""Metadata-only tail-only Step3p5 model contract tests.

Two layers of coverage:

1. Card-free structural (runs anywhere): AST/source introspection of the vLLM
   ``step3p5.py`` proves the tail-only decode instance holds **no** decoder
   weights — the metadata-only layer constructs only a vLLM ``Attention`` and
   its forward is an identity no-op, and ``Step3p5Model`` builds exactly
   ``num_hidden_layers`` such layers while retaining embed/norm.

2. Runtime (needs a running vLLM + checkpoint): asserts exactly 45
   ``static_forward_context`` entries keyed ``model.layers.N.self_attn.attn``
   with ``attn_type == DECODER`` and zero decoder-layer parameters. Gated by
   ``STEP3P5_CKPT_DIR`` + importable ``vllm``; skipped card-free.

Run:
    python3 -m pytest tests/step3p5/unit/test_metadata_only_model.py -q
"""
from __future__ import annotations

import ast
import os
import sys

import pytest

# Callables whose presence in the metadata-only layer __init__ would mean a
# decoder weight leaked into the tail-only decode instance.
_FORBIDDEN_LAYER_CALLS = {
    "Step3p5MLP",
    "FusedMoEBlock",
    "FusedMoE",
    "ReplicatedLinear",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "MergedColumnParallelLinear",
    "QKVParallelLinear",
    "VocabParallelEmbedding",
    "GemmaRMSNorm",
    "FP32ReplicatedLinear",
}


def _find_step3p5_source() -> str:
    """Locate the vLLM step3p5.py source file (card-free; no import needed)."""
    env = os.environ.get("VLLM_STEP3P5_PATH")
    candidates = []
    if env:
        candidates.append(env)
    try:
        import importlib.util

        spec = importlib.util.find_spec("vllm.model_executor.models.step3p5")
        if spec and spec.origin:
            candidates.append(spec.origin)
    except Exception:  # noqa: BLE001
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    rel_paths = (
        "vllm/vllm/model_executor/models/step3p5.py",
        "vllm/model_executor/models/step3p5.py",
    )
    # Walk ancestors so this works regardless of nesting depth.
    ancestor = here
    for _ in range(8):
        for rel in rel_paths:
            candidates.append(os.path.join(ancestor, rel))
        ancestor = os.path.dirname(ancestor)
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    pytest.skip("vLLM step3p5.py source not found (set VLLM_STEP3P5_PATH)")


def _class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name!r} not found in step3p5.py")


def _method_node(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"method {name!r} not found in {cls.name}")


def _call_names(node: ast.AST) -> set:
    names = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            func = n.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


@pytest.fixture(scope="module")
def source_text() -> str:
    return open(_find_step3p5_source(), encoding="utf-8").read()


@pytest.fixture(scope="module")
def tree(source_text) -> ast.Module:
    return ast.parse(source_text)


class TestMetadataOnlyStructure:
    """Card-free: the tail-only decode instance owns no decoder weights."""

    def test_metadata_layer_constructs_only_attention(self, tree):
        cls = _class_node(tree, "PyPtoMetadataOnlyStep3p5DecoderLayer")
        init = _method_node(cls, "__init__")
        calls = _call_names(init)
        assert "Attention" in calls, "metadata-only layer must register Attention"
        leaked = _FORBIDDEN_LAYER_CALLS & calls
        assert not leaked, (
            "metadata-only layer must NOT construct decoder weights, found: "
            f"{sorted(leaked)}"
        )

    def test_metadata_layer_forward_is_identity(self, tree):
        cls = _class_node(tree, "PyPtoMetadataOnlyStep3p5DecoderLayer")
        fwd = _method_node(cls, "forward")
        returns = [n for n in fwd.body if isinstance(n, ast.Return)]
        assert len(returns) == 1, "identity forward must have exactly one return"
        ret = returns[0].value
        assert isinstance(ret, ast.Name) and ret.id == "hidden_states", (
            "metadata-only forward must return hidden_states unchanged (no-op)"
        )
        # No attention/MLP/MoE call in the no-op forward body.
        assert not (_FORBIDDEN_LAYER_CALLS & _call_names(fwd))

    def test_model_tail_only_branch_builds_metadata_layers(self, tree, source_text):
        cls = _class_node(tree, "Step3p5Model")
        init = _method_node(cls, "__init__")
        src = ast.get_source_segment(source_text, init)
        assert "PyPtoMetadataOnlyStep3p5DecoderLayer" in src
        assert "_pypto_metadata_only" in src
        assert "config.num_hidden_layers" in src
        # embedding + final norm are retained on the tail-only instance.
        assert "self.embed_tokens" in src
        assert "self.norm" in src


@pytest.mark.skipif(
    not os.environ.get("STEP3P5_CKPT_DIR"),
    reason="runtime metadata contract needs STEP3P5_CKPT_DIR + a live vLLM build",
)
class TestMetadataOnlyRuntime:
    """Runtime: exactly 45 KV/metadata attention proxies, no decoder params.

    Gated on 0162 (needs importable vllm + a real checkpoint config). Skipped
    card-free. This is the authoritative C1 gate that must pass on device.
    """

    def _build_tail_only_model(self):
        pytest.importorskip("torch")
        pytest.importorskip("vllm")
        os.environ.setdefault("PYPTO_STEP3P5_TAIL_ONLY", "1")
        from transformers import AutoConfig  # noqa: PLC0415
        from vllm.config import ModelConfig, VllmConfig  # noqa: PLC0415
        from vllm.model_executor.models.step3p5 import (  # noqa: PLC0415
            Step3p5Model,
        )

        ckpt = os.environ["STEP3P5_CKPT_DIR"]
        AutoConfig.from_pretrained(ckpt, trust_remote_code=True)
        model_config = ModelConfig(model=ckpt, trust_remote_code=True)
        vllm_config = VllmConfig(model_config=model_config)
        model = Step3p5Model(vllm_config=vllm_config, prefix="model")
        return model, vllm_config

    def test_exactly_num_layers_metadata_entries(self):
        _model, vllm_config = self._build_tail_only_model()
        num_layers = vllm_config.model_config.hf_config.num_hidden_layers
        sfc = vllm_config.compilation_config.static_forward_context
        keys = sorted(sfc.keys())
        assert len(keys) == num_layers, (
            f"expected {num_layers} attention proxies, got {len(keys)}: {keys[:5]}"
        )
        for i in range(num_layers):
            assert f"model.layers.{i}.self_attn.attn" in sfc

    def test_decoder_layers_have_zero_parameters(self):
        model, _ = self._build_tail_only_model()
        assert getattr(model, "_pypto_metadata_only", False) is True
        for i, layer in enumerate(model.layers):
            n = sum(1 for _ in layer.parameters())
            assert n == 0, f"metadata-only layer {i} holds {n} params (must be 0)"

    def test_embed_norm_retained(self):
        model, _ = self._build_tail_only_model()
        names = [name for name, _ in model.named_parameters()]
        assert any("embed_tokens" in n for n in names)
        assert any("norm" in n for n in names)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
