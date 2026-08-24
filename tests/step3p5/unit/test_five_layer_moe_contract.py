# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Card-free contracts for the focused Step3p5 L0-L4 MoE graph."""
from __future__ import annotations

import ast
import json
from argparse import Namespace
from pathlib import Path

import torch

from tests.step3p5.harnesses import _stage_five_layer_moe as stage


_ROOT = Path(__file__).resolve().parents[3]
_PROGRAM = (
    _ROOT
    / "tests"
    / "step3p5"
    / "harnesses"
    / "_five_layer_moe_program.py"
)
_HOLDER = _ROOT / "tools" / "step3p5" / "five_layer_moe_holder.py"
_CONFIG = _ROOT / "models" / "step3p5" / "config.py"
_STAGE = (
    _ROOT
    / "tests"
    / "step3p5"
    / "harnesses"
    / "_stage_five_layer_moe.py"
)
_SWIMLANE_GATE = (
    _ROOT
    / "deployment"
    / "docker"
    / "run_swimlane_gate.sh"
)


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


def _segment(source: str, node: ast.AST) -> str:
    result = ast.get_source_segment(source, node)
    assert result is not None
    return result


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _calls(function: ast.FunctionDef, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and _call_name(node) == name
    ]


def test_focused_graph_is_exactly_l0_through_l4() -> None:
    source, tree = _parse(_PROGRAM)
    function = _method(tree, "five_layer_chip_orch")
    body = _segment(source, function)

    assert body.count("full_chip_orch(") == 1
    assert body.count("swa_chip_orch(") == 2
    assert body.count("swa_moe_chip_orch(") == 1
    assert body.count("full_moe_chip_orch(") == 1
    assert body.index("# L0:") < body.index("# L1:")
    assert body.index("# L1:") < body.index("# L2:")
    assert body.index("# L2:") < body.index("# L3:")
    assert body.index("# L3:") < body.index("# L4:")

    returned = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Return)
    ]
    assert len(returned) == 1
    assert ast.unparse(returned[0].value) == "(hidden_l3, hidden_l4)"


def test_l4_consumes_the_actual_l3_output_and_both_are_host_outputs() -> None:
    _, tree = _parse(_PROGRAM)
    chip = _method(tree, "five_layer_chip_orch")
    full_moe_calls = _calls(chip, "full_moe_chip_orch")
    assert len(full_moe_calls) == 1
    assert ast.unparse(full_moe_calls[0].args[0]) == "hidden_l3"

    for function_name in ("five_layer_chip_orch", "five_layer_host_orch"):
        function = _method(tree, function_name)
        annotations = {
            arg.arg: ast.unparse(arg.annotation)
            for arg in function.args.args
            if arg.annotation is not None
        }
        assert annotations["hidden_l3"].startswith("pl.Out[")
        assert annotations["hidden_l4"].startswith("pl.Out[")


def test_layer_slots_indices_and_epochs_match_canonical_l0_l4() -> None:
    _, tree = _parse(_PROGRAM)
    function = _method(tree, "five_layer_chip_orch")

    calls = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        target = ast.unparse(node.targets[0])
        name = _call_name(node.value)
        if target in {"h0", "h1", "h2", "hidden_l3", "hidden_l4"}:
            calls.append((target, name, node.value))
    by_target = {target: (name, call) for target, name, call in calls}

    assert by_target["h0"][0] == "full_chip_orch"
    assert by_target["h1"][0] == "swa_chip_orch"
    assert by_target["h2"][0] == "swa_chip_orch"
    assert by_target["hidden_l3"][0] == "swa_moe_chip_orch"
    assert by_target["hidden_l4"][0] == "full_moe_chip_orch"

    assert [ast.unparse(arg) for arg in by_target["h0"][1].args[-5:]] == [
        "0", "0", "0", "num_tokens", "my_rank",
    ]
    assert [ast.unparse(arg) for arg in by_target["h1"][1].args[-5:]] == [
        "1", "0", "0", "num_tokens", "my_rank",
    ]
    assert [ast.unparse(arg) for arg in by_target["h2"][1].args[-5:]] == [
        "2", "0", "0", "num_tokens", "my_rank",
    ]
    assert [ast.unparse(arg) for arg in by_target["hidden_l3"][1].args[-5:]] == [
        "3", "0", "num_tokens", "my_rank", "1",
    ]
    assert [ast.unparse(arg) for arg in by_target["hidden_l4"][1].args[-5:]] == [
        "4", "0", "num_tokens", "my_rank", "2",
    ]

    holder = _HOLDER.read_text(encoding="utf-8")
    assert "_FULL_SLOTS = (0, 1)  # L0, L4" in holder
    assert "_SWA_SLOTS = (0, 1, 2)  # L1, L2, L3" in holder
    assert "_MOE_SLOTS = (0, 1)  # L3, L4" in holder
    assert "_NORM_SLOTS = (0, 1, 2, 3, 4)" in holder
    assert "_KV_NUM_LAYERS = 5" in holder


def test_focused_holder_uses_only_five_kv_and_norm_layers() -> None:
    source, tree = _parse(_HOLDER)
    build = _method(tree, "build")
    build_body = _segment(source, build)
    wrapper = _method(tree, "__enter__")
    wrapper_body = _segment(source, wrapper)
    enter = _method(tree, "_enter_impl")
    body = _segment(source, enter)

    assert "return self._enter_impl()" in wrapper_body
    assert "config.KV_NUM_LAYERS" in build_body
    assert "canonical.LAYER_DYN" in build_body
    assert "focused.LAYER_DYN" in build_body
    assert "set(layer_contract.values()) != {_KV_NUM_LAYERS}" in build_body
    assert body.count("expected_num_layers=_KV_NUM_LAYERS") == 1
    for key in (
        "KEY_INPUT_RMS",
        "KEY_POST_ATTN_RMS",
        "KEY_Q_NORM",
        "KEY_K_NORM",
    ):
        assert f"weight_slots(keys.{key}, _NORM_SLOTS)" in body

    configure = _method(
        ast.parse(
            (
                _ROOT
                / "tests"
                / "step3p5"
                / "harnesses"
                / "_stage_five_layer_moe.py"
            ).read_text(encoding="utf-8")
        ),
        "_configure",
    )
    configure_body = ast.unparse(configure)
    assert "PYPTO_STEP3P5_KV_NUM_LAYERS" in configure_body
    assert "5 * layout['physical_num_blocks'] * BLOCK_SIZE" in configure_body

    config = _CONFIG.read_text(encoding="utf-8")
    assert "LAYER_DYN = KV_NUM_LAYERS" in config
    program = _PROGRAM.read_text(encoding="utf-8")
    assert "LAYER_DYN = _canonical.LAYER_DYN" in program


def test_focused_holder_preserves_ipc_provenance_for_weight_slices() -> None:
    source, tree = _parse(_HOLDER)
    enter = _method(tree, "_enter_impl")
    body = _segment(source, enter)

    assert ".device_tensor_slice(key, start, stop)" in body
    assert ".device_tensor(key)[start:stop]" not in body


def test_holder_reuses_prepared_dep_gen_for_swimlane_capture() -> None:
    _, tree = _parse(_HOLDER)
    run = _method(tree, "run")
    run_config_calls = _calls(run, "RunConfig")
    assert len(run_config_calls) == 1

    keywords = {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in run_config_calls[0].keywords
        if keyword.arg is not None
    }
    swim_modes = "dfx in {'swim', 'l2'}"
    assert keywords["enable_dep_gen"] == "dfx == 'dep'"
    assert keywords["enable_l2_swimlane"] == swim_modes
    assert keywords["l2_swimlane_reuse_dep_gen"] == swim_modes


def test_dfx_wait_uses_the_runtime_chip_swimlane_artifact_name() -> None:
    source, tree = _parse(_STAGE)
    main = _method(tree, "main")
    body = _segment(source, main)

    assert "PYPTO_RECV_META_SIDECAR" in source
    assert "source_decode_sha256=" in body
    assert (
        "from pypto.runtime.runner import _CHIP_SWIMLANE_RECORDS_NAME"
        in ast.unparse(main)
    )
    assert "_wait_for_artifacts" in body
    assert "_CHIP_SWIMLANE_RECORDS_NAME" in body
    assert '"l2_swimlane_records.json"' not in source


def test_formal_golden_requires_digest_bound_bit_exact_provenance() -> None:
    source, tree = _parse(_STAGE)
    assert "--image-digest" in source
    assert "--source-run" in source
    assert "GOLDEN_SCHEMA" in source
    assert "bit_exact" in source

    # The context-length restriction is enforced at the two validation
    # boundaries, rather than in ``main`` itself: ``_configure`` rejects a
    # non-64K producer invocation and ``_write_golden`` revalidates the
    # workload recorded in the manifest before writing a baseline. Keep the
    # contract test aligned with those actual guards instead of depending on
    # an implementation detail of ``main``'s call graph.
    configure = _segment(source, _method(tree, "_configure"))
    write_golden = _segment(source, _method(tree, "_write_golden"))
    assert "args.context_len != 65536" in configure
    assert "context_len != 65536" in write_golden
    assert "image_digest=args.image_digest" in source
    assert "source_run=args.source_run" in source


def test_swimlane_gate_uses_official_schema_and_propagates_failure() -> None:
    source = _SWIMLANE_GATE.read_text(encoding="utf-8")

    assert (
        "SWIMLANE_RECORDS_NAME=chip_swimlane_records.json"
        in source
    )
    assert '"analyzer_profile":"$PROFILE"' in source
    assert "PYPTO_RECV_META_SIDECAR" in source
    assert 'exit "$gate_rc"' in source
    assert "SWIMLANE_GATE_RC_NONZERO" in source
    assert "l2_swimlane_records.json" not in source


def test_golden_writer_emits_route_compatible_v3_manifest(
    tmp_path: Path,
) -> None:
    from tests.step3p5.harnesses._stage_five_layer_moe_route import (
        _load_golden_contract,
    )

    decode_sha = "a" * 64
    source = {
        "decode_fwd_sha256": decode_sha,
        "program_sha256": "b" * 64,
        "holder_sha256": "c" * 64,
        "harness_sha256": "d" * 64,
    }
    producer = {
        "workload": {
            "active_batch": 1,
            "context_len": 65536,
        },
        "source": source,
    }
    hidden_l3 = torch.zeros((8, 1, 4096), dtype=torch.bfloat16)
    hidden_l4 = torch.ones((8, 1, 4096), dtype=torch.bfloat16)
    image = "image@sha256:" + "e" * 64

    stage._write_golden(
        tmp_path,
        hidden_l3=hidden_l3,
        hidden_l4=hidden_l4,
        manifest=producer,
        image_digest=image,
        source_run="final-image-formal-bs1-64k",
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["schema"] == "step3p5.five-layer-moe-golden.v3"
    assert manifest["bit_exact"] is True
    assert manifest["source_kind"] == "baseline"
    assert manifest["source_decode_fwd_sha256"] == decode_sha
    contract, tensors = _load_golden_contract(
        tmp_path,
        active_batch=1,
        context_len=65536,
        image_digest=image,
        source_decode_sha256=decode_sha,
    )
    assert contract["bit_exact"]
    assert torch.equal(tensors["hidden_l4"], hidden_l4)


def test_64k_workload_is_per_sequence_for_every_required_batch() -> None:
    for active_batch in (1, 2, 4, 7, 8, 16):
        args = Namespace(
            active_batch=active_batch,
            context_len=65536,
            num_blocks=512,
        )
        layout = stage._workload_layout(args)
        assert layout == {
            "blocks_per_sequence": 512,
            "block_table_blocks_per_row_capacity": 512,
            "scheduler_num_blocks": active_batch * 512,
            "physical_num_blocks": active_batch * 512 + 15,
            "max_sequence_tokens": 65536,
        }


def test_bs7_64k_metadata_uses_disjoint_full_length_contexts() -> None:
    seq, pos, table, slot = stage._step_metadata(
        context_len=65536,
        active_batch=7,
        blocks_per_row_capacity=512,
        scheduler_num_blocks=7 * 512,
    )
    assert tuple(table.shape) == (16, 512)
    assert torch.equal(seq[:7], torch.full((7,), 65536, dtype=torch.int32))
    assert torch.equal(pos[:7], torch.full((7,), 65535, dtype=torch.int32))
    for row in range(7):
        first = row * 512
        assert torch.equal(
            table[row],
            torch.arange(first, first + 512, dtype=torch.int32),
        )
        assert int(slot[row]) == (first + 511) * 128 + 127
    assert torch.equal(
        table[7:, 0],
        torch.arange(7 * 512, 7 * 512 + 9, dtype=torch.int32),
    )


def test_focused_program_reuses_canonical_compute_functions() -> None:
    source, tree = _parse(_PROGRAM)
    assert "import models.step3p5.decode_fwd as _canonical" in source
    assert "_CANONICAL_PROGRAM = _canonical.whole_decode_step3p5" in source
    assert "ir.Program(" in source

    required = {
        "tp_all_reduce",
        "full_chip_orch",
        "swa_chip_orch",
        "_gate",
        "gate_step",
        "_norm_quant_moe_input",
        "dispatch_step",
        "_expert_routed",
        "expert_routed_step",
        "_expert_shared_local",
        "expert_shared_step",
        "combine_step",
        "full_moe_chip_orch",
        "swa_moe_chip_orch",
    }
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_REQUIRED_CANONICAL"
            for target in node.targets
        )
    )
    assert isinstance(assignment.value, ast.Tuple)
    assert {ast.literal_eval(item) for item in assignment.value.elts} == required

    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FiveLayerMoeFocused"
    )
    focused_methods = {
        node.name for node in class_node.body if isinstance(node, ast.FunctionDef)
    }
    assert focused_methods == {"five_layer_chip_orch", "five_layer_host_orch"}
    assert not required.intersection(focused_methods)


def test_focused_program_registers_optional_fused_all_reduce_before_build() -> None:
    _, tree = _parse(_PROGRAM)
    symbol = "tp_all_reduce_residual_bs1"

    optional_assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and [ast.unparse(target) for target in node.targets] == ["_optional"]
    ]
    assert len(optional_assignments) == 1
    lookup = optional_assignments[0].value
    assert isinstance(lookup, ast.Call)
    assert ast.unparse(lookup.func) == "_CANONICAL_PROGRAM.get_function"
    assert [ast.literal_eval(arg) for arg in lookup.args] == [symbol]
    assert lookup.keywords == []

    conditionals = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "_optional is not None"
    ]
    assert len(conditionals) == 1
    conditional = conditionals[0]
    assert [ast.unparse(node) for node in conditional.body] == [
        "_FUNCTIONS[_optional.name] = _optional"
    ]
    assert conditional.orelse == []

    program_build = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == "five_layer_moe" for target in node.targets)
    )
    assert tree.body.index(optional_assignments[0]) < tree.body.index(conditional)
    assert tree.body.index(conditional) < tree.body.index(program_build)
