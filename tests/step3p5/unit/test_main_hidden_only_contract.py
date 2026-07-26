# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Static gates for the Main live hidden-only ownership boundary."""
from __future__ import annotations

import ast
from types import SimpleNamespace
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[3]
_HIDDEN_PROGRAM = (
    _ROOT / "models" / "step3p5" / "decode_layer_single_chip_hidden.py"
)
_CANONICAL_PROGRAM = _ROOT / "models" / "step3p5" / "decode_fwd.py"
_REMOVED_OPT_PACKAGE = _ROOT / "models" / "step3p5_opt"
_HOLDER = _ROOT / "tools" / "step3p5" / "whole_decode_holder.py"
_SIDECAR = _ROOT / "tools" / "step3p5" / "whole_decode_sidecar.py"


def _class_source(path: Path, class_name: str) -> str:
    source = path.read_text()
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name
        == "_build_whole_decode_faithful_real_single_chip_hidden_only_program"
    )
    program_class = next(
        item
        for item in node.body
        if isinstance(item, ast.ClassDef) and item.name == class_name
    )
    return ast.get_source_segment(source, program_class) or ""


def test_hidden_only_program_is_the_only_main_production_entry():
    source = _HIDDEN_PROGRAM.read_text()
    assert "whole_decode_faithful_real_single_chip_hidden_only" in source
    assert "whole_decode_faithful_real_single_chip =" not in source


def test_canonical_loop_form_owns_the_default_main_entry():
    source = _CANONICAL_PROGRAM.read_text()
    assert "@pl.program" in source
    assert "class WholeDecodeStep3p5" in source
    assert "whole_decode_step3p5 = WholeDecodeStep3p5" in source
    assert "WholeDecodeOpt" not in source
    assert "whole_decode_opt" not in source
    assert "for layer_idx in pl.range" in source
    assert "whole_decode_faithful_real_single_chip_hidden_only" not in source


def test_step3p5_opt_compatibility_package_is_removed():
    assert not (_REMOVED_OPT_PACKAGE / "__init__.py").exists()
    assert not (_REMOVED_OPT_PACKAGE / "decode_fwd.py").exists()


def test_hidden_only_program_selects_swiglu_by_physical_layer():
    source = _HIDDEN_PROGRAM.read_text()
    assert "MOE_ACTIVATION_BY_LAYER" in source
    assert '("swa", 7.0, 0.0): "swa_moe_chip_orch_swiglu7_silu"' in source
    assert (
        '("full", 7.0, 16.0): '
        '"full_moe_chip_orch_swiglu7_swiglu16"'
    ) in source

    factory_class = _class_source(
        _HIDDEN_PROGRAM,
        "WholeDecodeFaithfulRealSingleChipHiddenOnly",
    )
    tree = ast.parse(factory_class)
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    for name in (
        "_expert_routed_swiglu7",
        "expert_routed_step_swiglu7",
        "_expert_shared_local_swiglu16",
        "expert_shared_step_swiglu16",
        "swa_moe_chip_orch_swiglu7_silu",
        "full_moe_chip_orch_swiglu7_swiglu16",
    ):
        assert name in functions

    for name in (
        "_expert_routed",
        "_expert_routed_swiglu7",
        "_expert_shared_local",
        "_expert_shared_local_swiglu16",
    ):
        function_source = ast.get_source_segment(factory_class, functions[name])
        assert function_source is not None
        assert "layer_idx" not in function_source

    assert "self.swa_moe_chip_orch_swiglu7_silu(" in factory_class
    assert "self.full_moe_chip_orch_swiglu7_swiglu16(" in factory_class

def test_hidden_only_program_has_no_vllm_owned_tail():
    source = _class_source(
        _HIDDEN_PROGRAM,
        "WholeDecodeFaithfulRealSingleChipHiddenOnly",
    )
    tree = ast.parse(source)
    names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    function_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    argument_names = {
        arg.arg
        for node in ast.walk(tree)
        if isinstance(node, ast.arguments)
        for arg in (*node.posonlyargs, *node.args, *node.kwonlyargs)
    }
    assert "lm_head_orch" not in function_names
    for forbidden in (
        "rms_lm_head",
        "final_norm_weight",
        "lm_head_weight",
        "logits_shard_out",
    ):
        assert forbidden not in names
        assert forbidden not in argument_names
    assert "return next_hidden_out" in source


def test_hidden_only_program_declares_persistent_kv_inout():
    source = _class_source(
        _HIDDEN_PROGRAM,
        "WholeDecodeFaithfulRealSingleChipHiddenOnly",
    )
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name in {"whole_chip_orch", "host_orch"}
    }
    assert set(functions) == {"whole_chip_orch", "host_orch"}
    for function_name, node in functions.items():
        annotations = {
            arg.arg: ast.unparse(arg.annotation)
            for arg in node.args.args
            if arg.annotation is not None
        }
        for cache_name in ("k_cache", "v_cache"):
            assert annotations[cache_name].startswith("pl.InOut["), (
                f"{function_name}.{cache_name} must preserve paged KV across "
                "decode dispatches"
            )


def test_live_holder_and_sidecar_default_to_current_hidden_only():
    holder = _HOLDER.read_text()
    sidecar = _SIDECAR.read_text()
    symbol = "whole_decode_faithful_real_single_chip_hidden_only"
    assert symbol in holder
    assert symbol in sidecar
    assert "layer_name=" not in holder
    assert "--layer-module" in sidecar
    assert "--layer-name" in sidecar
    assert "--baseline-main" in sidecar
    assert "CURRENT_MAIN_MODULE = \"models.step3p5.decode_fwd\"" in sidecar
    assert "CURRENT_MAIN_PROGRAM = \"whole_decode_step3p5\"" in sidecar
    assert "decode_layer_single_chip as dl" not in holder
    assert "KEY_FINAL_NORM" not in holder
    assert "KEY_LM_HEAD" not in holder
    assert '"argmax_debug": int(res["argmax"])' not in sidecar
    assert "live sidecar requires the hidden-only whole-net program" in sidecar


def test_sidecar_main_program_selection_defaults_to_canonical_loop_form():
    import tools.step3p5.whole_decode_sidecar as sidecar

    assert sidecar._main_program_kwargs(SimpleNamespace(
        baseline_main=False,
        layer_module=None,
        layer_name=None,
    )) == {
        "layer_module": "models.step3p5.decode_fwd",
        "program": "whole_decode_step3p5",
    }
    assert sidecar._main_program_kwargs(SimpleNamespace(
        baseline_main=True,
        layer_module=None,
        layer_name=None,
    )) == {}
    assert sidecar._main_program_kwargs(SimpleNamespace(
        baseline_main=False,
        layer_module="models.step3p5.decode_fwd",
        layer_name="whole_decode_step3p5",
    )) == {
        "layer_module": "models.step3p5.decode_fwd",
        "program": "whole_decode_step3p5",
    }

    for kwargs in (
        {
            "baseline_main": False,
            "layer_module": "models.step3p5.decode_fwd",
            "layer_name": None,
        },
        {
            "baseline_main": False,
            "layer_module": None,
            "layer_name": "whole_decode_step3p5",
        },
    ):
        try:
            sidecar._main_program_kwargs(SimpleNamespace(**kwargs))
        except ValueError as exc:
            assert "must be provided together" in str(exc)
        else:
            raise AssertionError("half-configured Main program was accepted")

    try:
        sidecar._main_program_kwargs(SimpleNamespace(
            baseline_main=True,
            layer_module="models.step3p5.decode_fwd",
            layer_name="whole_decode_step3p5",
        ))
    except ValueError as exc:
        assert "cannot be combined" in str(exc)
    else:
        raise AssertionError("baseline/custom Main combination was accepted")


def test_sidecar_forwards_main_program_selection_to_both_main_entrypoints():
    source = _SIDECAR.read_text()
    tree = ast.parse(source)
    helper_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_main_program_kwargs"
    ]
    assert len(helper_calls) == 2


def test_holder_run_exposes_only_raw_hidden():
    holder = _HOLDER.read_text()
    assert 'return {"next_hidden": self._next_hidden_out,' in holder
    assert '"per_layer_hidden": self._per_layer_hidden' in holder
    assert "h_mid=self." not in holder
    assert "dbg=self." not in holder
    assert "nh_row0_max=" not in holder
