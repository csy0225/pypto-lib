# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Card-free contracts for the canonical Main hidden-only product boundary."""
from __future__ import annotations

import ast
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[3]
_CANONICAL = _ROOT / "models" / "step3p5" / "decode_fwd.py"
_DENSE_MLP = _ROOT / "models" / "step3p5" / "dense_mlp.py"
_RETIRED_MAIN = (
    _ROOT
    / "models"
    / "step3p5"
    / ("decode_layer_single_chip" + "_hidden.py")
)
_RETIRED_SINGLE_LAYER_DRAFTS = (
    _ROOT
    / "models"
    / "step3p5"
    / ("single_layer_decode_full" + "_draft.py"),
    _ROOT
    / "models"
    / "step3p5"
    / ("single_layer_decode_swa" + "_draft.py"),
)
_RETIRED_SIDECARS_AND_PROBES = (
    _ROOT / "models" / "step3p5" / ("_repro_1738" + "_followup.py"),
    _ROOT / "models" / "step3p5" / ("_routed_jit" + "_probe.py"),
    _ROOT / "models" / "step3p5" / ("vllm_routed" + "_experts.py"),
    _ROOT / "tools" / "step3p5" / ("_dense_golden" + "_ctx1.py"),
    _ROOT / "tools" / "step3p5" / ("pypto_mlp" + "_worker.py"),
    _ROOT / "tools" / "step3p5" / ("pypto_moe" + "_backend.py"),
)
_REMOVED_OPT_PACKAGE = _ROOT / "models" / "step3p5_opt"
_HOLDER = _ROOT / "tools" / "step3p5" / "whole_decode_holder.py"
_SIDECAR = _ROOT / "tools" / "step3p5" / "whole_decode_sidecar.py"
_HARNESS = (
    _ROOT
    / "tests"
    / "step3p5"
    / "harnesses"
    / "_stage_main_hidden_only.py"
)
_CI = _ROOT / "tests" / "step3p5" / "ci" / "run_whole_network_ci.py"


def _parse(path: Path) -> tuple[str, ast.Module]:
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source)


def _method(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, f"expected one {name}, found {len(matches)}"
    return matches[0]


def test_retired_main_and_opt_package_are_removed() -> None:
    assert not _RETIRED_MAIN.exists()
    assert all(not path.exists() for path in _RETIRED_SINGLE_LAYER_DRAFTS)
    assert all(not path.exists() for path in _RETIRED_SIDECARS_AND_PROBES)
    assert not (_REMOVED_OPT_PACKAGE / "__init__.py").exists()
    assert not (_REMOVED_OPT_PACKAGE / "decode_fwd.py").exists()


def test_decode_fwd_is_the_only_main_program_entry() -> None:
    source, tree = _parse(_CANONICAL)
    program_classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(ast.unparse(item) == "pl.program" for item in node.decorator_list)
    ]
    assert [node.name for node in program_classes] == ["WholeDecodeStep3p5"]
    assert "whole_decode_step3p5 = WholeDecodeStep3p5" in source
    assert "whole_decode" + "_opt" not in source
    assert "WholeDecode" + "Opt" not in source
    assert "for layer_idx in pl.range" in source


def test_canonical_main_is_hidden_only_and_owns_persistent_kv() -> None:
    source, tree = _parse(_CANONICAL)
    forbidden = {
        "final_norm_weight",
        "lm_head_weight",
        "logits_shard_out",
        "draft_token_ids",
    }
    for function_name in ("whole_chip_orch", "host_orch"):
        function = _method(tree, function_name)
        arguments = {arg.arg for arg in function.args.args}
        assert not arguments.intersection(forbidden)
        annotations = {
            arg.arg: ast.unparse(arg.annotation)
            for arg in function.args.args
            if arg.annotation is not None
        }
        for cache_name in ("k_cache", "v_cache"):
            assert annotations[cache_name].startswith("pl.InOut[")

    host_arguments = {
        arg.arg for arg in _method(tree, "host_orch").args.args
    }
    assert "next_hidden_out" in host_arguments
    assert "per_layer_hidden" not in host_arguments
    assert "return next_hidden_out" in source


def test_dense_mlp_is_shared_compute_only_not_an_entrypoint() -> None:
    source, tree = _parse(_DENSE_MLP)
    assert "def dense_mlp_body_tp(" in source
    assert not any(
        isinstance(node, ast.ClassDef)
        and any(ast.unparse(item) == "pl.program" for item in node.decorator_list)
        for node in tree.body
    )
    assert not any(isinstance(node, ast.ClassDef) for node in tree.body)
    assert "host_orch" not in source
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and "debug" in node.name.lower()
        for node in ast.walk(tree)
    )


def test_holder_compiles_only_the_canonical_symbol() -> None:
    source, tree = _parse(_HOLDER)
    init = _method(tree, "__init__")
    init_args = {arg.arg for arg in init.args.args}
    assert "program" not in init_args
    assert "layer_module" not in init_args
    assert "layer_name" not in init_args
    assert 'MAIN_PROGRAM = "whole_decode_step3p5"' in source
    assert "dl.whole_decode_step3p5" in source
    assert "getattr(dl, self.program_name)" not in source


def test_sidecar_harness_and_ci_offer_no_main_rollback_selector() -> None:
    sources = {
        path: path.read_text(encoding="utf-8")
        for path in (_HOLDER, _SIDECAR, _HARNESS, _CI)
    }
    forbidden_options = (
        "--baseline" + "-main",
        "baseline" + "_main",
        "--layer" + "-module",
        "--layer" + "-name",
        "_main_program" + "_kwargs",
    )
    for path, source in sources.items():
        for forbidden in forbidden_options:
            assert forbidden not in source, f"{path.name} retains {forbidden}"

    assert '"program_main": "whole_decode_step3p5"' in sources[_CI]
    assert '"models/step3p5/decode_fwd.py"' in sources[_CI]


def test_holder_run_exposes_only_raw_hidden() -> None:
    source, tree = _parse(_HOLDER)
    run = _method(tree, "run")
    run_source = ast.get_source_segment(source, run) or ""
    assert 'result = {"next_hidden": self._next_hidden_out}' in run_source
    assert "per_layer_hidden" not in run_source
    assert "h_mid" not in run_source
    assert "dbg" not in run_source
