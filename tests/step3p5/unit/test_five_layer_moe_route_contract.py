# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Card-free contracts for the focused L0-L4 route sidecar."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest
import torch

from tests.step3p5.harnesses._stage_five_layer_moe_route import (
    _checkpoint_identity,
    _json_sha256,
    _load_golden_contract,
    _sidecar_payload,
    _validate_route_totals,
)
from tools.step3p5.analyze_five_layer_moe_dfx import (
    _route_histogram_contract,
)
from tools.step3p5.five_layer_moe_route_holder import (
    FiveLayerMoeRouteHolder,
    assemble_route_outputs,
)
from tools.step3p5.five_layer_moe_golden_contract import (
    source_protocol_binding_fields,
)


_ROOT = Path(__file__).resolve().parents[3]
_PROGRAM = (
    _ROOT
    / "tests"
    / "step3p5"
    / "harnesses"
    / "_five_layer_moe_route_program.py"
)
_DECODE = _ROOT / "models" / "step3p5" / "decode_fwd.py"
_HOLDER = _ROOT / "tools" / "step3p5" / "five_layer_moe_route_holder.py"
_STAGE = (
    _ROOT
    / "tests"
    / "step3p5"
    / "harnesses"
    / "_stage_five_layer_moe_route.py"
)
_IMAGE = "image@sha256:" + "a" * 64


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


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _calls(function: ast.FunctionDef, name: str) -> list[ast.Call]:
    return sorted(
        [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and _call_name(node) == name
        ],
        key=lambda node: node.lineno,
    )


def _call_assignment(function: ast.FunctionDef, name: str) -> ast.Assign:
    matches = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and _call_name(node.value) == name
    ]
    assert len(matches) == 1, f"expected one assignment to {name}"
    return matches[0]


def _fake_provenance(*, active_batch: int = 1) -> dict[str, object]:
    checkpoint_files = {
        "config.json": {
            "size_bytes": 1,
            "sha256": "1" * 64,
        },
        "quant_model_weights.safetensors.index.json": {
            "size_bytes": 2,
            "sha256": "2" * 64,
        },
        "weights.safetensors": {
            "size_bytes": 3,
            "sha256": "3" * 64,
        },
    }
    source = {
        "source_tree_manifest_sha256": "4" * 64,
        "decode_fwd_sha256": "5" * 64,
        "formal_program_sha256": "6" * 64,
        "route_program_sha256": "7" * 64,
        "route_holder_sha256": "8" * 64,
        "route_stage_sha256": "9" * 64,
    }
    input_contract = {
        "workload": {
            "active_batch": active_batch,
            "context_len": 65536,
            "context_semantics": "per_active_sequence",
        },
        "input_tokens": list(range(active_batch)),
        "tensor_sha256": {
            "active_hidden": "a" * 64,
            "seq_lens": "b" * 64,
            "positions": "c" * 64,
            "block_table": "d" * 64,
            "slot_mapping": "e" * 64,
        },
    }
    return {
        "image_digest": _IMAGE,
        "checkpoint": {
            "schema": "step3p5.checkpoint-identity.v1",
            "logical_id": "checkpoint",
            "index_file": (
                "quant_model_weights.safetensors.index.json"
            ),
            "weight_tensor_count": 10,
            "weight_shard_count": 1,
            "files": checkpoint_files,
            "identity_sha256": _json_sha256(checkpoint_files),
        },
        "source": source,
        "source_manifest_sha256": _json_sha256(source),
        "input_contract": input_contract,
        "input_contract_sha256": _json_sha256(input_contract),
        "formal_golden": {
            "schema": "step3p5.five-layer-moe-golden.v3",
            "manifest_sha256": "f" * 64,
            "source_run": "baseline-r1-normal-bs1-64k",
            "source_kind": "baseline",
            "source_decode_fwd_sha256": "5" * 64,
            "source_manifest_sha256": "4" * 64,
            "active_batch": active_batch,
            "context_len_per_sequence": 65536,
            "image_ref": _IMAGE,
            "files": {
                "hidden_l3.pt": "1" * 64,
                "hidden_l4.pt": "2" * 64,
            },
            "bit_exact": True,
        },
    }


def test_route_program_is_additive_and_reuses_canonical_compute() -> None:
    source, tree = _parse(_PROGRAM)
    assert "import models.step3p5.decode_fwd as _canonical" in source
    assert "_five_layer_moe_program" not in source
    assert "models.step3p5.decode_fwd" in source
    assert "five_layer_moe_route = ir.Program(" in source

    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "FiveLayerMoeRouteInstrumented"
    )
    methods = {
        node.name for node in class_node.body if isinstance(node, ast.FunctionDef)
    }
    assert methods == {
        "snapshot_local_routes_and_hidden",
        "five_layer_route_chip_orch",
        "five_layer_route_host_orch",
    }


def test_route_stage_requires_a_local_owner_protocol_golden() -> None:
    source, tree = _parse(_STAGE)
    main = _method(tree, "main")
    body = ast.get_source_segment(source, main)

    assert body is not None
    assert (
        "expected_protocol_profile=LOCAL_OWNER_PROTOCOL_PROFILE"
        in body
    )


def test_route_program_stays_count_only_without_attention_replay() -> None:
    source, _ = _parse(_PROGRAM)

    assert "proves exact local-owner count histograms only" in source
    assert "(token, topk-slot) -> expert/packed-row" in source
    assert "replay_local_routes" not in source
    assert "replay_attn" not in source
    assert "attention_swa_inline" not in source
    assert "attention_full_inline" not in source


def test_l3_snapshot_fences_local_route_counts_before_l4() -> None:
    _, tree = _parse(_PROGRAM)
    chip = _method(tree, "five_layer_route_chip_orch")
    l3_call = _calls(chip, "swa_moe_chip_orch")
    l4_call = _calls(chip, "full_moe_chip_orch")
    snapshots = _calls(chip, "snapshot_local_routes_and_hidden")
    assert len(l3_call) == len(l4_call) == 1
    assert len(snapshots) == 2
    assert (
        l3_call[0].lineno
        < snapshots[0].lineno
        < l4_call[0].lineno
        < snapshots[1].lineno
    )

    assert [ast.unparse(arg) for arg in snapshots[0].args] == [
        "local_expert_count_l3",
        "my_rank",
        "hidden_l3_raw",
        "recv_meta_l3",
        "hidden_l3",
    ]
    assert ast.unparse(l4_call[0].args[0]) == "hidden_l3"
    assert [ast.unparse(arg) for arg in snapshots[1].args] == [
        "local_expert_count_l4",
        "my_rank",
        "hidden_l4_raw",
        "recv_meta_l4",
        "hidden_l4",
    ]

    returned = [
        node
        for node in ast.walk(chip)
        if isinstance(node, ast.Return)
    ]
    assert len(returned) == 1
    assert ast.unparse(returned[0].value) == (
        "(hidden_l3, hidden_l4, recv_meta_l3, recv_meta_l4)"
    )


def test_l3_l4_rebind_post_call_local_expert_count() -> None:
    _, route_tree = _parse(_PROGRAM)
    _, canonical_tree = _parse(_DECODE)
    chip = _method(route_tree, "five_layer_route_chip_orch")

    for callee, hidden, count in (
        ("swa_moe_chip_orch", "hidden_l3_raw", "local_expert_count_l3"),
        ("full_moe_chip_orch", "hidden_l4_raw", "local_expert_count_l4"),
    ):
        canonical = _method(canonical_tree, callee)
        assert ast.unparse(canonical.returns) == (
            "tuple[pl.Tensor[[BATCH, HIDDEN], pl.BF16], "
            "pl.Tensor[[n_local_experts_pad], pl.INT32]]"
        )
        returns = [
            node
            for node in ast.walk(canonical)
            if isinstance(node, ast.Return)
        ]
        assert len(returns) == 1
        assert ast.unparse(returns[0].value) == (
            "(next_hidden_out, local_expert_count)"
        )

        assignment = _call_assignment(chip, callee)
        assert ast.unparse(assignment.targets[0]) == f"({hidden}, {count})"
        call = assignment.value
        parameters = [
            arg.arg for arg in canonical.args.args if arg.arg != "self"
        ]
        bound = {
            parameter: ast.unparse(argument)
            for parameter, argument in zip(
                parameters, call.args, strict=True
            )
        }
        assert bound["next_hidden_out"] == hidden
        assert bound["local_expert_count"] == count


def test_snapshot_is_one_incore_body_without_nested_task_scope() -> None:
    _, tree = _parse(_PROGRAM)
    snapshot = _method(tree, "snapshot_local_routes_and_hidden")
    annotations = {
        arg.arg: ast.unparse(arg.annotation)
        for arg in snapshot.args.args
        if arg.annotation is not None
    }
    assert annotations["local_expert_count"].startswith("pl.Tensor[")
    assert "n_local_experts_pad" in annotations["local_expert_count"]
    assert annotations["recv_meta_out"].startswith("pl.Out[")
    assert annotations["hidden_out"].startswith("pl.Out[")

    calls = {_call_name(call) for call in ast.walk(snapshot) if isinstance(call, ast.Call)}
    assert {"full", "read", "store", "range"}.issubset(calls)
    assert "at" not in calls
    body = ast.unparse(snapshot)
    assert "[n_ranks, n_local_experts_pad]" in body
    assert "SNAPSHOT_HIDDEN_CHUNK" in body
    assert "[my_rank, expert]" in body
    assert "for expert in pl.range(n_local_experts_pad):" in body
    assert "pl.read(local_expert_count, [expert])" in body


def test_host_and_holder_expose_both_route_snapshots() -> None:
    _, tree = _parse(_PROGRAM)
    host = _method(tree, "five_layer_route_host_orch")
    annotations = {
        arg.arg: ast.unparse(arg.annotation)
        for arg in host.args.args
        if arg.annotation is not None
    }
    for name in (
        "hidden_l3",
        "hidden_l4",
        "local_expert_count_l3",
        "local_expert_count_l4",
        "recv_meta_l3",
        "recv_meta_l4",
    ):
        assert annotations[name].startswith("pl.Out[")
    for name in ("local_expert_count_l3", "local_expert_count_l4"):
        assert "n_local_experts_pad" in annotations[name]

    route_calls = _calls(host, "five_layer_route_chip_orch")
    assert len(route_calls) == 1
    rendered_args = [ast.unparse(arg) for arg in route_calls[0].args]
    assert "recv_meta_l3[rank]" in rendered_args
    assert "recv_meta_l4[rank]" in rendered_args

    holder = _HOLDER.read_text(encoding="utf-8")
    assert "class FiveLayerMoeRouteHolder" in holder
    assert "_five_layer_moe_route_program as focused" in holder
    assert "focused.five_layer_moe_route" in holder
    assert "self._focused.n_local_experts_pad" in holder
    assert '"recv_meta": recv_meta' in holder
    assert '"local_expert_count": local_expert_count' in holder


def test_canonical_moe_helpers_export_count_as_out_tensor() -> None:
    _, tree = _parse(_DECODE)
    for name in ("full_moe_chip_orch", "swa_moe_chip_orch"):
        fn = _method(tree, name)
        args = [arg.arg for arg in fn.args.args]
        index = args.index("local_expert_count")
        annotation = ast.unparse(fn.args.args[index].annotation)
        assert annotation.startswith("pl.Out[")
        assert "n_local_experts_pad" in annotation
        assert "local_expert_count = pl.create_tensor" not in ast.unparse(fn)


def test_route_program_registers_optional_fused_all_reduce() -> None:
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

    conditionals = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "_optional is not None"
    ]
    assert len(conditionals) == 1
    assert "_FUNCTIONS[_optional.name] = _optional" in ast.unparse(
        conditionals[0]
    )


def test_explicit_counts_match_diagonal_owner_rows() -> None:
    l3 = torch.zeros((8, 8, 40), dtype=torch.int32)
    l4 = torch.zeros((8, 8, 40), dtype=torch.int32)
    count_l3 = torch.zeros((8, 40), dtype=torch.int32)
    count_l4 = torch.zeros((8, 40), dtype=torch.int32)
    rank = 3
    expert = 4
    l3[rank, rank, expert] = 5
    l4[rank, rank, expert] = 7
    count_l3[rank, expert] = 5
    count_l4[rank, expert] = 7

    recv_meta, counts = assemble_route_outputs(
        l3,
        l4,
        local_expert_count_l3=count_l3,
        local_expert_count_l4=count_l4,
    )

    assert int(recv_meta[rank, 0, rank, expert]) == 5
    assert int(recv_meta[rank, 1, rank, expert]) == 7
    assert tuple(counts.shape) == (8, 2, 36)
    assert torch.equal(counts[:, 0], count_l3[:, :36])
    assert torch.equal(counts[:, 1], count_l4[:, :36])
    assert not bool(torch.any(recv_meta[:, :, :, 36:]))


def test_route_output_assembly_rejects_off_owner_rows() -> None:
    l3 = torch.zeros((8, 8, 40), dtype=torch.int32)
    l4 = torch.zeros((8, 8, 40), dtype=torch.int32)
    count_l3 = torch.zeros((8, 40), dtype=torch.int32)
    count_l4 = torch.zeros((8, 40), dtype=torch.int32)
    l3[0, 1, 0] = 2

    with pytest.raises(ValueError, match="off-owner"):
        assemble_route_outputs(
            l3,
            l4,
            local_expert_count_l3=count_l3,
            local_expert_count_l4=count_l4,
        )


def test_route_output_assembly_rejects_diagonal_count_mismatch() -> None:
    l3 = torch.zeros((8, 8, 40), dtype=torch.int32)
    l4 = torch.zeros((8, 8, 40), dtype=torch.int32)
    count_l3 = torch.zeros((8, 40), dtype=torch.int32)
    count_l4 = torch.zeros((8, 40), dtype=torch.int32)
    l3[3, 3, 4] = 5
    count_l3[3, 4] = 4

    with pytest.raises(ValueError, match="diagonal owner route row"):
        assemble_route_outputs(
            l3,
            l4,
            local_expert_count_l3=count_l3,
            local_expert_count_l4=count_l4,
        )


def test_route_output_assembly_rejects_nonzero_count_padding() -> None:
    l3 = torch.zeros((8, 8, 40), dtype=torch.int32)
    l4 = torch.zeros((8, 8, 40), dtype=torch.int32)
    count_l3 = torch.zeros((8, 40), dtype=torch.int32)
    count_l4 = torch.zeros((8, 40), dtype=torch.int32)
    count_l3[0, 36] = 1

    with pytest.raises(ValueError, match="padding 36:40 is non-zero"):
        assemble_route_outputs(
            l3,
            l4,
            local_expert_count_l3=count_l3,
            local_expert_count_l4=count_l4,
        )


def test_route_holder_preserves_ipc_provenance_for_weight_slices() -> None:
    source, tree = _parse(_HOLDER)
    wrapper = _method(tree, "__enter__")
    wrapper_body = ast.get_source_segment(source, wrapper)
    assert wrapper_body is not None
    assert "return self._enter_impl()" in wrapper_body
    enter = _method(tree, "_enter_impl")
    body = ast.get_source_segment(source, enter)
    assert body is not None

    assert ".device_tensor_slice(key, start, stop)" in body
    assert ".device_tensor(key)[start:stop]" not in body


def test_route_output_assembly_is_exact_and_device_ordered() -> None:
    l3 = torch.zeros((8, 8, 40), dtype=torch.int32)
    l4 = torch.zeros((8, 8, 40), dtype=torch.int32)
    l3[0, 0, 0] = 5
    l3[2, 2, 2] = 3
    l4[2, 2, 6] = 4
    l4[7, 7, 35] = 4

    recv_meta, local_expert_count = assemble_route_outputs(l3, l4)
    assert tuple(recv_meta.shape) == (8, 2, 8, 40)
    assert tuple(local_expert_count.shape) == (8, 2, 36)
    assert recv_meta.dtype == torch.int32
    assert local_expert_count.dtype == torch.int32
    assert int(local_expert_count[0, 0, 0]) == 5
    assert int(local_expert_count[7, 1, 35]) == 4
    validation = _validate_route_totals(recv_meta, active_batch=1)
    assert validation["per_layer_per_owner"] == [
        [5, 0, 3, 0, 0, 0, 0, 0],
        [0, 0, 4, 0, 0, 0, 0, 4],
    ]
    assert validation["global_per_layer"] == [8, 8]
    assert validation["expected_global_per_layer"] == 8
    assert validation["owner_rows_diagonal"]


def test_route_sidecar_is_analyzer_compatible(tmp_path: Path) -> None:
    recv_meta = torch.zeros((8, 2, 8, 40), dtype=torch.int32)
    recv_meta[0, 0, 0, 0] = 5
    recv_meta[2, 0, 2, 2] = 3
    recv_meta[2, 1, 2, 6] = 4
    recv_meta[7, 1, 7, 35] = 4
    local_expert_count = torch.stack(
        [
            recv_meta[rank, :, rank, :36]
            for rank in range(8)
        ],
        dim=0,
    )

    validation = _validate_route_totals(recv_meta, active_batch=1)
    assert validation["global_per_layer"] == [8, 8]
    payload = _sidecar_payload(
        recv_meta_device=recv_meta,
        local_expert_count_device=local_expert_count,
        provenance=_fake_provenance(),
        window_id_prefix="route-test",
    )
    assert tuple(payload["owner_route_counts"].shape) == (2, 8, 8, 40)
    assert tuple(payload["local_expert_count"].shape) == (2, 8, 36)
    assert payload["snapshot_provenance"][0]["source_tensor"] == (
        "local_expert_count"
    )

    sidecar = tmp_path / "recv_meta_sidecar.pt"
    torch.save(payload, sidecar)
    analyzed = _route_histogram_contract(sidecar)
    assert analyzed["L3"]["available"]
    assert analyzed["L3"]["total_routed_tokens_by_rank"]["rank0/d0"] == 5
    assert analyzed["L4"]["snapshot_independence_validated"]
    assert analyzed["L3"]["owner_rows_diagonal"]
    assert analyzed["L3"]["global_per_layer"] == [8, 8]
    assert analyzed["L3"]["source"] == sidecar.name
    assert analyzed["L3"]["provenance"]["active_batch"] == 1


def test_route_sidecar_rejects_incomplete_publication_provenance(
    tmp_path: Path,
) -> None:
    recv_meta = torch.zeros((8, 2, 8, 40), dtype=torch.int32)
    local_expert_count = torch.zeros((8, 2, 36), dtype=torch.int32)
    provenance = _fake_provenance()
    del provenance["checkpoint"]
    payload = _sidecar_payload(
        recv_meta_device=recv_meta,
        local_expert_count_device=local_expert_count,
        provenance=provenance,
        window_id_prefix="route-test",
    )
    sidecar = tmp_path / "recv_meta_sidecar.pt"
    torch.save(payload, sidecar)
    with pytest.raises(ValueError, match="checkpoint"):
        _route_histogram_contract(sidecar)


def test_route_sidecar_rejects_golden_source_sha_mismatch(
    tmp_path: Path,
) -> None:
    recv_meta = torch.zeros((8, 2, 8, 40), dtype=torch.int32)
    local_expert_count = torch.zeros((8, 2, 36), dtype=torch.int32)
    provenance = _fake_provenance()
    provenance["formal_golden"]["source_decode_fwd_sha256"] = "0" * 64
    payload = _sidecar_payload(
        recv_meta_device=recv_meta,
        local_expert_count_device=local_expert_count,
        provenance=provenance,
        window_id_prefix="route-test",
    )
    sidecar = tmp_path / "recv_meta_sidecar.pt"
    torch.save(payload, sidecar)
    with pytest.raises(ValueError, match="does not match route source"):
        _route_histogram_contract(sidecar)


def test_golden_contract_is_validated_before_device_use(
    tmp_path: Path,
) -> None:
    hidden_l3 = torch.zeros((8, 1, 4096), dtype=torch.bfloat16)
    hidden_l4 = torch.ones((8, 1, 4096), dtype=torch.bfloat16)
    torch.save(hidden_l3, tmp_path / "hidden_l3.pt")
    torch.save(hidden_l4, tmp_path / "hidden_l4.pt")
    files = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in ("hidden_l3.pt", "hidden_l4.pt")
    }
    manifest = {
        "schema": "step3p5.five-layer-moe-golden.v3",
        "source_run": "baseline-r1-normal-bs1-64k",
        "source_kind": "baseline",
        "source_decode_fwd_sha256": "1" * 64,
        "source_manifest_sha256": "2" * 64,
        "active_batch": 1,
        "context_len_per_sequence": 65536,
        "image_ref": _IMAGE,
        "files": files,
        "bit_exact": True,
    }
    manifest.update(
        source_protocol_binding_fields(
            source_manifest_sha256="2" * 64,
            decode_fwd_sha256="1" * 64,
            moe_protocol_contract_sha256="3" * 64,
            protocol_contract=manifest,
        )
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    contract, tensors = _load_golden_contract(
        tmp_path,
        active_batch=1,
        context_len=65536,
        image_digest=_IMAGE,
        source_decode_sha256="1" * 64,
    )

    assert contract["source_kind"] == "baseline"
    assert contract["protocol_profile"] == "legacy_distributed_ep"
    assert contract["numeric_contract"] == {
        "name": "legacy_baseline_bit_exact_v1",
        "comparison": "bit_exact_to_protocol_golden",
        "bit_exact": True,
    }
    assert contract["files"] == files
    assert torch.equal(tensors["hidden_l4"], hidden_l4)

    with pytest.raises(ValueError, match="does not match expected"):
        _load_golden_contract(
            tmp_path,
            active_batch=1,
            context_len=65536,
            image_digest=_IMAGE,
            source_decode_sha256="1" * 64,
            expected_protocol_profile="replicated_input_local_owner",
        )

    manifest["files"]["hidden_l4.pt"] = "0" * 64
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        _load_golden_contract(
            tmp_path,
            active_batch=1,
            context_len=65536,
            image_digest=_IMAGE,
            source_decode_sha256="1" * 64,
        )

    manifest["files"]["hidden_l4.pt"] = hashlib.sha256(
        (tmp_path / "hidden_l4.pt").read_bytes()
    ).hexdigest()
    manifest["source_decode_fwd_sha256"] = "0" * 64
    manifest.update(
        source_protocol_binding_fields(
            source_manifest_sha256="2" * 64,
            decode_fwd_sha256="0" * 64,
            moe_protocol_contract_sha256="3" * 64,
            protocol_contract=manifest,
        )
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match live source"):
        _load_golden_contract(
            tmp_path,
            active_batch=1,
            context_len=65536,
            image_digest=_IMAGE,
            source_decode_sha256="1" * 64,
        )


def test_local_ep_golden_contract_requires_exact_protocol_metadata(
    tmp_path: Path,
) -> None:
    hidden = torch.zeros((8, 1, 4096), dtype=torch.bfloat16)
    for name in ("hidden_l3.pt", "hidden_l4.pt"):
        torch.save(hidden, tmp_path / name)
    files = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in ("hidden_l3.pt", "hidden_l4.pt")
    }
    manifest = {
        "schema": "step3p5.five-layer-moe-golden.v3",
        "source_run": "local-ep-formal-bs1-64k",
        "source_kind": "local-ep",
        "protocol_profile": "replicated_input_local_owner",
        "numeric_contract": {
            "name": "local_owner_partial_tp_all_reduce_bf16_v1",
            "comparison": "bit_exact_to_protocol_golden",
            "bit_exact": True,
        },
        "source_decode_fwd_sha256": "1" * 64,
        "source_manifest_sha256": "2" * 64,
        "active_batch": 1,
        "context_len_per_sequence": 65536,
        "image_ref": _IMAGE,
        "files": files,
        "bit_exact": True,
    }
    manifest.update(
        source_protocol_binding_fields(
            source_manifest_sha256="2" * 64,
            decode_fwd_sha256="1" * 64,
            moe_protocol_contract_sha256="3" * 64,
            protocol_contract=manifest,
        )
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    contract, _ = _load_golden_contract(
        tmp_path,
        active_batch=1,
        context_len=65536,
        image_digest=_IMAGE,
        source_decode_sha256="1" * 64,
        expected_protocol_profile="replicated_input_local_owner",
    )

    assert contract["source_kind"] == "local-ep"
    assert contract["protocol_profile"] == "replicated_input_local_owner"
    assert contract["numeric_contract"] == manifest["numeric_contract"]

    invalid_cases = [
        (
            lambda value: value.pop("numeric_contract"),
            "declare protocol_profile and numeric_contract together",
        ),
        (
            lambda value: value.pop("protocol_profile"),
            "declare protocol_profile and numeric_contract together",
        ),
        (
            lambda value: value.__setitem__(
                "protocol_profile",
                "unknown_ep",
            ),
            "unsupported golden protocol_profile",
        ),
        (
            lambda value: value.__setitem__("source_kind", "baseline"),
            "source_kind=.*invalid",
        ),
        (
            lambda value: value["numeric_contract"].__setitem__(
                "name",
                "legacy_baseline_bit_exact_v1",
            ),
            "numeric_contract.name",
        ),
        (
            lambda value: value["numeric_contract"].__setitem__(
                "comparison",
                "bit_exact_to_legacy_baseline",
            ),
            "numeric_contract.comparison",
        ),
        (
            lambda value: value["numeric_contract"].__setitem__(
                "bit_exact",
                1,
            ),
            "numeric_contract.bit_exact",
        ),
    ]
    for mutate, match in invalid_cases:
        candidate = json.loads(json.dumps(manifest))
        mutate(candidate)
        manifest_path.write_text(json.dumps(candidate), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            _load_golden_contract(
                tmp_path,
                active_batch=1,
                context_len=65536,
                image_digest=_IMAGE,
                source_decode_sha256="1" * 64,
                expected_protocol_profile="replicated_input_local_owner",
            )


def test_golden_contract_requires_explicit_bit_exact_manifest(
    tmp_path: Path,
) -> None:
    hidden = torch.zeros((8, 1, 4096), dtype=torch.bfloat16)
    for name in ("hidden_l3.pt", "hidden_l4.pt"):
        torch.save(hidden, tmp_path / name)
    files = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in ("hidden_l3.pt", "hidden_l4.pt")
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "step3p5.five-layer-moe-golden.v3",
                "source_run": "formal-bs1-64k",
                "source_kind": "baseline",
                "source_decode_fwd_sha256": "1" * 64,
                "source_manifest_sha256": "2" * 64,
                "active_batch": 1,
                "context_len_per_sequence": 65536,
                "image_ref": _IMAGE,
                "files": files,
                "bit_exact": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="bit_exact must be true"):
        _load_golden_contract(
            tmp_path,
            active_batch=1,
            context_len=65536,
            image_digest=_IMAGE,
            source_decode_sha256="1" * 64,
        )


def test_checkpoint_identity_covers_shards_without_absolute_path(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "weights.safetensors").write_bytes(b"weights")
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "weights.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    identity = _checkpoint_identity(tmp_path)

    assert "path" not in identity
    assert identity["logical_id"] == tmp_path.name
    assert identity["weight_shard_count"] == 1
    assert set(identity["files"]) == {
        "config.json",
        "quant_model_weights.safetensors.index.json",
        "weights.safetensors",
    }


def test_checkpoint_manifest_rejects_same_size_shard_mutation(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.json"
    shard = tmp_path / "weights.safetensors"
    index = tmp_path / "quant_model_weights.safetensors.index.json"
    config.write_text("{}", encoding="utf-8")
    shard.write_bytes(b"AAAA")
    index.write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": shard.name,
                }
            }
        ),
        encoding="utf-8",
    )
    files = {
        path.name: {
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in (config, index, shard)
    }
    manifest = tmp_path / "checkpoint_identity.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "step3p5.checkpoint-identity.v1",
                "logical_id": tmp_path.name,
                "index_file": index.name,
                "weight_tensor_count": 1,
                "weight_shard_count": 1,
                "files": files,
                "identity_sha256": _json_sha256(files),
            }
        ),
        encoding="utf-8",
    )

    identity = _checkpoint_identity(tmp_path, manifest)
    assert identity["files"][shard.name]["sha256"] == files[shard.name]["sha256"]

    shard.write_bytes(b"BBBB")
    with pytest.raises(ValueError, match="checkpoint file hash mismatch"):
        _checkpoint_identity(tmp_path, manifest)


def test_exporters_are_cleaned_when_holder_build_fails() -> None:
    _, tree = _parse(_STAGE)
    main = _method(tree, "main")
    try_nodes = [node for node in main.body if isinstance(node, ast.Try)]
    cleanup_try = next(
        node
        for node in try_nodes
        if any(
            _call_name(call) == "_stop_exporters"
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
        )
    )
    protected_calls = {
        _call_name(call)
        for statement in cleanup_try.body
        for call in ast.walk(statement)
        if isinstance(call, ast.Call)
    }
    assert "_start_exporters" in protected_calls
    assert "build" in protected_calls


def test_route_total_validation_rejects_missing_global_routes() -> None:
    recv_meta = torch.zeros((8, 2, 8, 40), dtype=torch.int32)
    with pytest.raises(ValueError, match="global route totals"):
        _validate_route_totals(recv_meta, active_batch=1)


@pytest.mark.parametrize("active_batch", [1, 2, 4, 7, 8, 16])
def test_route_total_validation_counts_one_global_topk_set(
    active_batch: int,
) -> None:
    recv_meta = torch.zeros((8, 2, 8, 40), dtype=torch.int32)
    expected = active_batch * 8
    recv_meta[0, 0, 0, 0] = expected
    recv_meta[7, 1, 7, 35] = expected

    validation = _validate_route_totals(
        recv_meta,
        active_batch=active_batch,
    )
    assert validation["global_per_layer"] == [expected, expected]
    assert validation["expected_global_per_layer"] == expected


def test_holder_rejects_heterogeneous_owner_counts() -> None:
    holder = object.__new__(FiveLayerMoeRouteHolder)
    holder.tp = 8
    holder._consts = {"BATCH": 16}
    holder.num_tokens_per_owner = torch.zeros(128, dtype=torch.int32)
    holder.num_tokens_per_owner[:8] = torch.tensor(
        [4, 4, 4, 3, 4, 4, 4, 4],
        dtype=torch.int32,
    )

    with pytest.raises(ValueError, match="requires identical"):
        holder._validate_replicated_owner_counts()


@pytest.mark.parametrize("active_batch", [1, 2, 4, 7, 8, 16])
def test_holder_accepts_replicated_owner_counts(
    active_batch: int,
) -> None:
    holder = object.__new__(FiveLayerMoeRouteHolder)
    holder.tp = 8
    holder._consts = {"BATCH": 16}
    holder.num_tokens_per_owner = torch.zeros(128, dtype=torch.int32)
    holder.num_tokens_per_owner[:8].fill_(active_batch)

    assert holder._validate_replicated_owner_counts() == active_batch


@pytest.mark.parametrize("failure", ["shape", "dtype", "negative", "padding"])
def test_route_output_assembly_rejects_invalid_device_evidence(
    failure: str,
) -> None:
    l3 = torch.zeros((8, 8, 40), dtype=torch.int32)
    l4 = torch.zeros((8, 8, 40), dtype=torch.int32)
    expected = "recv_meta"
    if failure == "shape":
        l3 = l3[:, :, :39]
        expected = "shape"
    elif failure == "dtype":
        l3 = l3.to(torch.int64)
        expected = "dtype"
    elif failure == "negative":
        l3[0, 0, 0] = -1
        expected = "negative"
    else:
        l3[0, 0, 36] = 1
        expected = "padding"

    with pytest.raises((ValueError, OverflowError), match=expected):
        assemble_route_outputs(l3, l4)
