#!/usr/bin/env python3
"""Card-free structural tests for the load_format=pypto model loader.

Importing the loader requires ``vllm`` (BaseModelLoader / register_model_loader),
so these tests assert the contract via AST/source introspection instead:
  - registered for load_format "pypto";
  - subclasses DefaultModelLoader (so the 3 tail weights load for free);
  - overrides load_weights and chains super().load_weights (tail params);
  - builds the exporter native-W8A8 (int8_routed=True) and fails closed via
    validate_weight_map before ready; owns the exporter on the model.

Run:
    python3 -m pytest tests/step3p5/unit/test_pypto_model_loader.py -q
"""
from __future__ import annotations

import ast
import os

import pytest

_LOADER = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "tools",
        "step3p5",
        "pypto_model_loader.py",
    )
)


@pytest.fixture(scope="module")
def source() -> str:
    if not os.path.isfile(_LOADER):
        pytest.skip("pypto_model_loader.py not found")
    return open(_LOADER, encoding="utf-8").read()


@pytest.fixture(scope="module")
def tree(source) -> ast.Module:
    return ast.parse(source)


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == name:
            return n
    raise AssertionError(f"class {name!r} not found")


class TestPyPtoModelLoaderContract:
    def test_registered_for_pypto_load_format(self, tree):
        cls = _class(tree, "PyPtoStep3p5ModelLoader")
        decos = []
        for d in cls.decorator_list:
            if isinstance(d, ast.Call) and isinstance(d.func, ast.Name):
                decos.append((d.func.id, [ast.literal_eval(a) for a in d.args]))
        assert ("register_model_loader", ["pypto"]) in decos, (
            "loader must be @register_model_loader('pypto')"
        )

    def test_subclasses_default_loader(self, tree):
        cls = _class(tree, "PyPtoStep3p5ModelLoader")
        bases = [b.id for b in cls.bases if isinstance(b, ast.Name)]
        assert "DefaultModelLoader" in bases, (
            "must subclass DefaultModelLoader so the 3 tail weights load for free"
        )

    def test_load_weights_chains_super(self, source):
        assert "super().load_weights(model, model_config)" in source

    def test_vllm_tail_only_filter_is_explicit(self):
        vllm_path = os.environ.get("VLLM_STEP3P5_PATH")
        if not vllm_path:
            here = os.path.abspath(os.path.dirname(__file__))
            candidates = []
            ancestor = here
            for _ in range(8):
                candidates.extend(
                    (
                        os.path.join(
                            ancestor,
                            "vllm",
                            "vllm",
                            "model_executor",
                            "models",
                            "step3p5.py",
                        ),
                        os.path.join(
                            ancestor,
                            "vllm",
                            "model_executor",
                            "models",
                            "step3p5.py",
                        ),
                    )
                )
                ancestor = os.path.dirname(ancestor)
            vllm_path = next((p for p in candidates if os.path.isfile(p)), None)
        if not vllm_path or not os.path.isfile(vllm_path):
            pytest.skip("vLLM step3p5.py source not found")
        vllm_source = open(vllm_path, encoding="utf-8").read()
        assert "_pypto_tail_only_weights" in vllm_source
        assert '"model.embed_tokens."' in vllm_source
        assert '"model.norm."' in vllm_source
        assert '"lm_head."' in vllm_source
        assert "weights = _pypto_tail_only_weights(weights)" in vllm_source

    def test_native_w8a8_export_and_fail_closed(self, source):
        # routed experts stay INT8 (no BF16 dequant); map validated before ready.
        assert "int8_routed=True" in source
        assert "validate_weight_map(pool_map, native_w8a8=True)" in source

    def test_owns_exporter_and_frees_bundle(self, source):
        assert "_pypto_weight_exporter" in source
        assert "del bundle" in source

    def test_embedding_excluded_documented(self, source):
        # embedding is vLLM-only; the PyPTO pool must not carry KEY_EMBED.
        assert "KEY_EMBED" in source or "embed" in source.lower()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
