#!/usr/bin/env python3
"""Card-free tests for the generated vLLM sitecustomize load order.

The sitecustomize shim must install, in order:
  1. the load_format=pypto model loader registration (before model build);
  2. the KV allocator overlay;
  3. the Step3p5 forward patch.

``make_vllm_sitecustomize`` is pure stdlib, so this runs card-free.

Run:
    python3 -m pytest tests/step3p5/unit/test_sitecustomize_order.py -q
"""
from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.step3p5.make_vllm_sitecustomize import SITE_TEMPLATE  # noqa: E402


class TestSitecustomizeOrder:
    def test_all_three_stages_present(self):
        assert "tools.step3p5.pypto_model_loader" in SITE_TEMPLATE
        assert "vllm_kvpool_backend" in SITE_TEMPLATE
        assert "tools.step3p5.vllm_monkey_patch" in SITE_TEMPLATE

    def test_loader_before_kv_before_patch(self):
        i_loader = SITE_TEMPLATE.index("tools.step3p5.pypto_model_loader")
        i_kv = SITE_TEMPLATE.index("vllm_kvpool_backend")
        i_patch = SITE_TEMPLATE.index("tools.step3p5.vllm_monkey_patch")
        assert i_loader < i_kv < i_patch, (
            "sitecustomize must register loader, then KV overlay, then patch"
        )

    def test_loader_is_opt_in_and_strict_by_default(self):
        # gated by PYPTO_STEP3P5_LOADER; strict default so misconfig fails loud.
        assert "PYPTO_STEP3P5_LOADER" in SITE_TEMPLATE
        assert "PYPTO_STEP3P5_LOADER_STRICT" in SITE_TEMPLATE

    def test_template_generates_valid_python(self):
        import ast

        rendered = SITE_TEMPLATE.replace(
            "__REPO_ROOT_REPR__", repr("/x")
        ).replace("__MODE_REPR__", repr("full"))
        ast.parse(rendered)  # must be syntactically valid


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
