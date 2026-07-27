# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""C3 peer-fanout architecture contract and compile probe.

This probe is intentionally independent from the canonical Step3.5 decode
program.  It builds one minimal ``@pl.program`` that demonstrates the C3
shape without editing production code:

* peer work fans out with orchestration-level ``pl.spmd``;
* every peer writes one disjoint, peer-major slab;
* the peer grid TaskId is joined explicitly before consumption;
* combine first pulls every peer slab, then reduces by token;
* no InCore function contains ``pl.parallel``;
* peer tasks never write the shared ``moe_out`` directly;
* there is no shared ``[1, HIDDEN]`` route staging row and no ``token * 0``
  pseudo-dependency.

``contract`` is card-free and does not import PyPTO.  ``compile`` first runs
the same fail-closed contract, then compiles the probe.  A real compilation
failure is reported as ``NO-GO`` and returns a non-zero exit status.
``audit`` is also card-free, but audits the current canonical decode source
and returns ``NO-GO`` until the production graph has orchestration-level peer
fan-out, non-aliasing peer-owned slabs, and an explicit grid TaskId join.

The modes are deliberately separate: a probe contract or compile pass is only
evidence that the DSL shape is expressible.  It is not a production C3 pass.

Examples::

    python -m tests.step3p5.probes._probe_c3_peer_fanout --mode contract

    python -m tests.step3p5.probes._probe_c3_peer_fanout --mode audit

    python -m tests.step3p5.probes._probe_c3_peer_fanout \
        --mode compile --platform a2a3sim --out /tmp/c3-peer-fanout

    # Frontend/lowering-only diagnostic; not a real backend compile:
    python -m tests.step3p5.probes._probe_c3_peer_fanout \
        --mode compile --skip-ptoas --out /tmp/c3-peer-fanout-frontend
"""

import argparse
import ast
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
CANONICAL = ROOT / "models" / "step3p5" / "decode_fwd.py"
RETIRED_SINGLE_CHIP = (
    ROOT / "models" / "step3p5" / "decode_layer_single_chip_hidden.py"
)
REFERENCE_MOE = ROOT / "models" / "deepseek" / "v4" / "moe.py"
PYPTO_WORKSPACE = ROOT.parent / "pypto"
SCHEMA = "step3p5.c3.peer_fanout.v1"

# Keep the graph deliberately small while preserving a real multi-peer,
# multi-token, 512-byte storage-aligned BF16 transfer.
N_RANKS = 2
TOKENS = 8
HIDDEN = 32


def _dotted_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _decorator_is(node: ast.expr, dotted: str) -> bool:
    return _dotted_name(node) == dotted or (
        isinstance(node, ast.Call) and _dotted_name(node.func) == dotted
    )


def _function_type(function: ast.FunctionDef) -> str | None:
    for decorator in function.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        if _dotted_name(decorator.func) != "pl.function":
            continue
        for keyword in decorator.keywords:
            if keyword.arg == "type":
                return _dotted_name(keyword.value)
    return None


def _calls(node: ast.AST, dotted: str) -> list[ast.Call]:
    return [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Call) and _dotted_name(item.func) == dotted
    ]


def _name_list(node: ast.AST | None) -> list[str] | None:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    result: list[str] = []
    for item in node.elts:
        name = _dotted_name(item)
        if name is None:
            return None
        result.append(name)
    return result


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _spmd_with(
    function: ast.FunctionDef,
    *,
    name_hint: str,
) -> tuple[ast.With, ast.withitem] | None:
    for node in ast.walk(function):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            expression = item.context_expr
            if not isinstance(expression, ast.Call):
                continue
            if _dotted_name(expression.func) != "pl.spmd":
                continue
            hint = _keyword(expression, "name_hint")
            if isinstance(hint, ast.Constant) and hint.value == name_hint:
                return node, item
    return None


def _spmd_parallel_lines(function: ast.FunctionDef) -> list[int]:
    """Find illegal ``pl.parallel`` calls inside implicit InCore SPMD bodies."""
    lines: set[int] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.With):
            continue
        if not any(
            isinstance(item.context_expr, ast.Call)
            and _dotted_name(item.context_expr.func) == "pl.spmd"
            for item in node.items
        ):
            continue
        lines.update(call.lineno for call in _calls(node, "pl.parallel"))
    return sorted(lines)


def _assigned_call(
    function: ast.FunctionDef,
    *,
    target: str,
    dotted: str,
) -> ast.Call | None:
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        if not isinstance(node.targets[0], ast.Name):
            continue
        if node.targets[0].id != target or not isinstance(node.value, ast.Call):
            continue
        if _dotted_name(node.value.func) == dotted:
            return node.value
    return None


def _is_peer_major_base(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Mult)
        and isinstance(node.left, ast.Name)
        and node.left.id == "peer"
        and isinstance(node.right, ast.Name)
        and node.right.id == "TOKENS"
    )


def _is_peer_major_row(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Add)
        and _is_peer_major_base(node.left)
        and isinstance(node.right, ast.Name)
        and node.right.id == "token"
    )


def _offset_first(call: ast.Call) -> ast.AST | None:
    offsets = _keyword(call, "dst_offsets")
    if isinstance(offsets, (ast.List, ast.Tuple)) and offsets.elts:
        return offsets.elts[0]
    if _dotted_name(call.func) == "pl.store" and len(call.args) >= 2:
        offsets = call.args[1]
        if isinstance(offsets, (ast.List, ast.Tuple)) and offsets.elts:
            return offsets.elts[0]
    return None


def _store_target(call: ast.Call) -> str | None:
    if _dotted_name(call.func) != "pl.store" or len(call.args) < 3:
        return None
    return _dotted_name(call.args[2])


def _writes_symbol(node: ast.AST, symbol: str) -> bool:
    """Conservatively detect direct writes to a named tensor.

    The audit must fail closed if a future worker uses ``pl.write`` or a
    subscript assignment instead of ``pl.store``.  Looking only for one DSL
    spelling would otherwise create a false PASS for a peer worker that still
    writes the shared output.
    """
    for call in _calls(node, "pl.store"):
        if _store_target(call) == symbol:
            return True
    for call in _calls(node, "pl.write"):
        if call.args and _dotted_name(call.args[0]) == symbol:
            return True
    for call in _calls(node, "pld.tensor.put"):
        dst = _keyword(call, "dst")
        if _dotted_name(dst) == symbol:
            return True
    for call in _calls(node, "pld.tensor.get"):
        if call.args and _dotted_name(call.args[0]) == symbol:
            return True
    for item in ast.walk(node):
        if not isinstance(item, ast.Assign):
            continue
        for target in item.targets:
            if (
                isinstance(target, ast.Subscript)
                and _dotted_name(target.value) == symbol
            ):
                return True
    return False


def _write_sites(node: ast.AST, symbol: str) -> list[int]:
    """Return source lines for direct writes to ``symbol``."""
    lines: list[int] = []
    for item in ast.walk(node):
        if not isinstance(item, ast.Call):
            continue
        dotted = _dotted_name(item.func)
        writes = False
        if dotted == "pl.store":
            writes = _store_target(item) == symbol
        elif dotted == "pl.write":
            writes = bool(item.args) and _dotted_name(item.args[0]) == symbol
        elif dotted == "pld.tensor.put":
            writes = _dotted_name(_keyword(item, "dst")) == symbol
        elif dotted == "pld.tensor.get":
            writes = bool(item.args) and _dotted_name(item.args[0]) == symbol
        if writes:
            lines.append(item.lineno)
    for item in ast.walk(node):
        if not isinstance(item, ast.Assign):
            continue
        for target in item.targets:
            if (
                isinstance(target, ast.Subscript)
                and _dotted_name(target.value) == symbol
            ):
                lines.append(item.lineno)
    return sorted(lines)


def _annotation_contains_shape(
    function: ast.FunctionDef,
    *,
    shape_fragment: str,
) -> bool:
    """Match a shape in a formal annotation, independent of its argument name."""
    needle = "".join(shape_fragment.split())
    for argument in (*function.args.posonlyargs, *function.args.args):
        annotation = argument.annotation
        if annotation is None:
            continue
        rendered = "".join(ast.unparse(annotation).split())
        if needle in rendered:
            return True
    return False


def _block_index_used_in_transfer(
    node: ast.With,
    block_index_names: set[str],
) -> bool:
    """Require a peer grid index to affect a transfer or slab address.

    Merely finding ``pl.tile.get_block_idx()`` is insufficient: an unrelated
    SPMD grid with a misleading ``name_hint`` must not be reported as the C3
    peer grid.  This check intentionally accepts only uses that can affect a
    transfer operand or a store address.
    """
    if not block_index_names:
        return False

    transfer_calls = []
    for dotted in (
        "pld.tensor.get",
        "pld.tensor.put",
        "pld.tile.remote_load",
        "pld.tile.remote_store",
        "pl.store",
        "pl.write",
    ):
        transfer_calls.extend(_calls(node, dotted))
    return any(
        any(
            isinstance(item, ast.Name) and item.id in block_index_names
            for item in ast.walk(call)
        )
        for call in transfer_calls
    )


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _source_contract() -> dict[str, Any]:
    """Audit this probe's architecture without importing PyPTO."""
    source_path = Path(__file__).resolve()
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))

    program_classes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and any(_decorator_is(dec, "pl.program") for dec in node.decorator_list)
    ]
    checks: list[dict[str, Any]] = []
    checks.append(
        _check(
            "single_program",
            len(program_classes) == 1,
            f"@pl.program class count={len(program_classes)}",
        )
    )
    if len(program_classes) != 1:
        return {
            "passed": False,
            "checks": checks,
            "source": str(source_path),
        }

    program = program_classes[0]
    methods = {
        node.name: node
        for node in program.body
        if isinstance(node, ast.FunctionDef)
    }
    chip_orch = methods.get("chip_orch")
    host_orch = methods.get("host_orch")
    checks.append(
        _check(
            "orchestration_entrypoints",
            chip_orch is not None
            and host_orch is not None
            and _function_type(chip_orch) == "pl.FunctionType.Orchestration",
            "chip_orch is orchestration and host_orch is present",
        )
    )
    if chip_orch is None or host_orch is None:
        return {
            "passed": False,
            "checks": checks,
            "source": str(source_path),
        }

    incore_parallel_sites: list[str] = []
    for method in methods.values():
        if _function_type(method) != "pl.FunctionType.InCore":
            continue
        for call in _calls(method, "pl.parallel"):
            incore_parallel_sites.append(f"{method.name}:{call.lineno}")
    for method in methods.values():
        incore_parallel_sites.extend(
            f"{method.name}:{line}:inside_spmd"
            for line in _spmd_parallel_lines(method)
        )
    checks.append(
        _check(
            "no_incore_parallel",
            not incore_parallel_sites,
            (
                "no pl.parallel call occurs in an InCore function"
                if not incore_parallel_sites
                else f"illegal sites={incore_parallel_sites}"
            ),
        )
    )

    peer_with = _spmd_with(chip_orch, name_hint="c3_peer_pull")
    peer_with_node = peer_with[0] if peer_with else None
    peer_with_item = peer_with[1] if peer_with else None
    peer_call = (
        peer_with_item.context_expr
        if peer_with_item is not None
        and isinstance(peer_with_item.context_expr, ast.Call)
        else None
    )
    peer_task_name = (
        peer_with_item.optional_vars.id
        if peer_with_item is not None
        and isinstance(peer_with_item.optional_vars, ast.Name)
        else None
    )
    peer_extent_ok = bool(
        peer_call
        and peer_call.args
        and isinstance(peer_call.args[0], ast.Name)
        and peer_call.args[0].id == "N_RANKS"
    )
    has_peer_block_id = bool(
        peer_with_node
        and any(
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "peer"
            and isinstance(node.value, ast.Call)
            and _dotted_name(node.value.func) == "pl.tile.get_block_idx"
            for node in ast.walk(peer_with_node)
        )
    )
    checks.append(
        _check(
            "orchestration_spmd_peer_fanout",
            peer_extent_ok
            and peer_task_name == "peer_pull_tid"
            and has_peer_block_id,
            (
                "c3_peer_pull is an N_RANKS grid captured as peer_pull_tid; "
                "peer comes from pl.tile.get_block_idx()"
            ),
        )
    )

    slab_call = _assigned_call(
        chip_orch,
        target="peer_slab",
        dotted="pl.create_tensor",
    )
    slab_shape_ok = False
    if slab_call and slab_call.args:
        shape = slab_call.args[0]
        slab_shape_ok = (
            isinstance(shape, ast.List)
            and len(shape.elts) == 2
            and isinstance(shape.elts[0], ast.BinOp)
            and isinstance(shape.elts[0].op, ast.Mult)
            and _dotted_name(shape.elts[0].left) == "N_RANKS"
            and _dotted_name(shape.elts[0].right) == "TOKENS"
            and _dotted_name(shape.elts[1]) == "HIDDEN"
        )
    checks.append(
        _check(
            "peer_major_slab",
            slab_shape_ok,
            "peer_slab shape is [N_RANKS * TOKENS, HIDDEN]",
        )
    )

    peer_gets = (
        _calls(peer_with_node, "pld.tensor.get")
        if peer_with_node is not None
        else []
    )
    peer_stores = (
        _calls(peer_with_node, "pl.store")
        if peer_with_node is not None
        else []
    )
    remote_disjoint = any(
        call.args
        and _dotted_name(call.args[0]) == "peer_slab"
        and _is_peer_major_base(_offset_first(call))
        for call in peer_gets
        if _offset_first(call) is not None
    )
    local_disjoint = any(
        _store_target(call) == "peer_slab"
        and _is_peer_major_row(_offset_first(call))
        for call in peer_stores
        if _offset_first(call) is not None
    )
    peer_writes_moe_out = bool(
        peer_with_node and _writes_symbol(peer_with_node, "moe_out")
    )
    checks.append(
        _check(
            "write_disjoint_peer_tasks",
            remote_disjoint and local_disjoint and not peer_writes_moe_out,
            (
                "remote and self-peer paths write only "
                "peer_slab[peer * TOKENS + token, :]; peer tasks do not "
                "write moe_out"
            ),
        )
    )

    join_call = _assigned_call(
        chip_orch,
        target="peer_pull_join",
        dotted="pl.system.task_dummy",
    )
    join_deps = _name_list(_keyword(join_call, "deps")) if join_call else None
    reduce_with = _spmd_with(chip_orch, name_hint="c3_token_reduce")
    reduce_node = reduce_with[0] if reduce_with else None
    reduce_item = reduce_with[1] if reduce_with else None
    reduce_call = (
        reduce_item.context_expr
        if reduce_item is not None
        and isinstance(reduce_item.context_expr, ast.Call)
        else None
    )
    reduce_deps = _name_list(_keyword(reduce_call, "deps")) if reduce_call else None
    reduce_extent_ok = bool(
        reduce_call
        and reduce_call.args
        and _dotted_name(reduce_call.args[0]) == "TOKENS"
    )
    checks.append(
        _check(
            "grid_taskid_explicit_join",
            join_deps == ["peer_pull_tid"]
            and reduce_deps == ["peer_pull_join"]
            and reduce_extent_ok,
            (
                "peer_pull_tid -> task_dummy peer_pull_join -> "
                "TOKENS reduction grid"
            ),
        )
    )

    reduce_stores = _calls(reduce_node, "pl.store") if reduce_node else []
    output_write_sites = _write_sites(chip_orch, "moe_out")
    reduction_reads_slab = bool(
        reduce_node
        and any(
            call.args
            and _dotted_name(call.args[0]) == "peer_slab"
            for call in _calls(reduce_node, "pl.load")
        )
    )
    token_output_write = any(
        _store_target(call) == "moe_out"
        and isinstance(_offset_first(call), ast.Name)
        and _offset_first(call).id == "token"
        for call in reduce_stores
        if _offset_first(call) is not None
    )
    checks.append(
        _check(
            "peer_pull_then_token_reduction",
            reduction_reads_slab
            and token_output_write
            and len(output_write_sites) == 1,
            (
                "the joined token grid reads peer_slab and is the sole "
                "writer of moe_out[token, :]"
            ),
        )
    )

    create_one_hidden = False
    for call in _calls(program, "pl.create_tensor"):
        if not call.args or not isinstance(call.args[0], ast.List):
            continue
        shape = call.args[0]
        if (
            len(shape.elts) == 2
            and isinstance(shape.elts[0], ast.Constant)
            and shape.elts[0].value == 1
            and _dotted_name(shape.elts[1]) == "HIDDEN"
        ):
            create_one_hidden = True
    route_stage_names = [
        node.lineno
        for node in ast.walk(program)
        if isinstance(node, ast.Name) and node.id == "route_stage"
    ]
    checks.append(
        _check(
            "no_shared_route_stage",
            not route_stage_names and not create_one_hidden,
            (
                "no route-stage symbol and no "
                "pl.create_tensor([1, HIDDEN], ...)"
                if not route_stage_names
                else f"route-stage symbol lines={route_stage_names}"
            ),
        )
    )

    token_zero_products: list[int] = []
    for node in ast.walk(program):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
            continue
        sides = (node.left, node.right)
        has_token = any(
            isinstance(side, ast.Name) and side.id == "token" for side in sides
        )
        has_zero = any(
            isinstance(side, ast.Constant) and side.value == 0 for side in sides
        )
        if has_token and has_zero:
            token_zero_products.append(node.lineno)
    checks.append(
        _check(
            "no_token_zero_dependency",
            not token_zero_products,
            (
                "no token * 0 expression"
                if not token_zero_products
                else f"token * 0 lines={token_zero_products}"
            ),
        )
    )

    host_allocates_windows = (
        len(_calls(host_orch, "pld.alloc_window_buffer")) == 2
        and len(_calls(host_orch, "pld.window")) == 2
    )
    checks.append(
        _check(
            "distributed_host_wiring",
            host_allocates_windows,
            "host_orch allocates payload and readiness windows exactly once",
        )
    )

    reference_source = REFERENCE_MOE.read_text(encoding="utf-8")
    reference_patterns = {
        "captured_spmd_taskid": "with pl.spmd(" in reference_source
        and " as _cscatter_tid:" in reference_source,
        "grid_deps": "deps=[_cwait_tid]" in reference_source,
        "distributed_window": "pld.alloc_window_buffer(" in reference_source
        and "pld.window(" in reference_source,
    }
    checks.append(
        _check(
            "deepseek_v4_moe_pattern_provenance",
            all(reference_patterns.values()),
            f"reference patterns={reference_patterns}",
        )
    )

    passed = all(item["passed"] for item in checks)
    return {
        "passed": passed,
        "evidence_class": "source_contract",
        "production_claim": False,
        "checks": checks,
        "source": str(source_path),
        "reference": str(REFERENCE_MOE),
        "architecture": {
            "program_count": 1,
            "ranks": N_RANKS,
            "tokens": TOKENS,
            "hidden": HIDDEN,
            "peer_slab_shape": [N_RANKS * TOKENS, HIDDEN],
            "peer_writer": "peer_slab[peer * TOKENS:(peer + 1) * TOKENS, :]",
            "join": "peer_pull_tid -> peer_pull_join -> token_reduce",
            "output_writer": "one reduction task per token",
        },
    }


def _canonical_c3_audit() -> dict[str, Any]:
    """Audit production C3 readiness without importing or compiling PyPTO."""
    source = CANONICAL.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(CANONICAL))
    programs = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(_decorator_is(dec, "pl.program") for dec in node.decorator_list)
    ]
    checks: list[dict[str, Any]] = []
    checks.append(
        _check(
            "single_canonical_program",
            len(programs) == 1,
            f"@pl.program class count={len(programs)}",
        )
    )
    checks.append(
        _check(
            "retired_single_chip_file_absent",
            not RETIRED_SINGLE_CHIP.exists(),
            (
                "retired decode_layer_single_chip_hidden.py remains absent"
                if not RETIRED_SINGLE_CHIP.exists()
                else f"retired file unexpectedly exists: {RETIRED_SINGLE_CHIP}"
            ),
        )
    )
    if len(programs) != 1:
        return {
            "decision": "NO-GO",
            "production_ready": False,
            "evidence_class": "source_contract",
            "checks": checks,
            "source": str(CANONICAL),
            "blockers": ["canonical @pl.program identity is ambiguous"],
        }

    program = programs[0]
    methods = {
        node.name: node
        for node in program.body
        if isinstance(node, ast.FunctionDef)
    }

    # C3 does not own tp_all_reduce. Keep the audit target narrow instead of
    # turning a peer-loop search into an unrelated collective rewrite request.
    c3_methods = {
        name: methods[name]
        for name in (
            "_dispatch_pack_publish",
            "_dispatch_pull",
            "_dispatch_stage",
            "_stage_routed_src",
            "_pull_routed_y",
        )
        if name in methods
    }
    missing_methods = sorted(
        {
            "_dispatch_pack_publish",
            "_dispatch_pull",
            "_dispatch_stage",
            "_stage_routed_src",
            "_pull_routed_y",
        }
        - set(c3_methods)
    )
    checks.append(
        _check(
            "canonical_c3_methods_present",
            not missing_methods,
            (
                "all audited dispatch/combine methods are present"
                if not missing_methods
                else f"missing methods={missing_methods}"
            ),
        )
    )

    incore_parallel_sites: list[str] = []
    for method in methods.values():
        if _function_type(method) != "pl.FunctionType.InCore":
            continue
        for call in _calls(method, "pl.parallel"):
            incore_parallel_sites.append(f"{method.name}:{call.lineno}")
    checks.append(
        _check(
            "no_incore_parallel",
            not incore_parallel_sites,
            (
                "no InCore function contains pl.parallel"
                if not incore_parallel_sites
                else f"illegal sites={incore_parallel_sites}"
            ),
        )
    )

    # C3 targets bulk peer reads. Notify/wait control loops and post-join
    # token/expert reductions may remain sequential inside one worker; only
    # the remote data movement itself must move to the orchestration SPMD
    # boundary. Flagging every loop variable named ``peer`` would incorrectly
    # report completion-notify loops as a C3 blocker.
    incore_remote_transfer_sites: list[str] = []
    for method_name in ("_dispatch_pull", "_pull_routed_y"):
        method = c3_methods.get(method_name)
        if method is None:
            continue
        for dotted in (
            "pld.tensor.get",
            "pld.tile.remote_load",
            "pld.tile.remote_store",
        ):
            incore_remote_transfer_sites.extend(
                f"{method_name}:{call.lineno}:{dotted}"
                for call in _calls(method, dotted)
            )
    checks.append(
        _check(
            "peer_work_at_orchestration_spmd_boundary",
            not incore_remote_transfer_sites,
            (
                "no audited remote data movement remains in InCore workers"
                if not incore_remote_transfer_sites
                else (
                    "InCore remote-transfer sites="
                    f"{incore_remote_transfer_sites}"
                )
            ),
        )
    )

    peer_formals: list[str] = []
    for method_name, method in methods.items():
        formals = {arg.arg for arg in method.args.args}
        has_peer_formal = bool(
            formals
            & {
                "peer",
                "peer_id",
                "source_peer",
                "target_peer",
            }
        )
        has_remote_transfer = any(
            _calls(method, dotted)
            for dotted in (
                "pld.tensor.get",
                "pld.tensor.put",
                "pld.tile.remote_load",
                "pld.tile.remote_store",
            )
        )
        if has_peer_formal and has_remote_transfer:
            peer_formals.append(method_name)
    c3_peer_grids: list[dict[str, Any]] = []
    for method_name, method in methods.items():
        for node in ast.walk(method):
            if not isinstance(node, ast.With):
                continue
            for item in node.items:
                call = item.context_expr
                if (
                    not isinstance(call, ast.Call)
                    or _dotted_name(call.func) != "pl.spmd"
                ):
                    continue
                hint = _keyword(call, "name_hint")
                if (
                    not isinstance(hint, ast.Constant)
                    or not isinstance(hint.value, str)
                    or not any(
                        word in hint.value
                        for word in ("peer", "dispatch_pull", "combine_pull")
                    )
                ):
                    continue
                tid = (
                    item.optional_vars.id
                    if isinstance(item.optional_vars, ast.Name)
                    else None
                )
                block_index_names = {
                    assign.targets[0].id
                    for assign in ast.walk(node)
                    if isinstance(assign, ast.Assign)
                    and len(assign.targets) == 1
                    and isinstance(assign.targets[0], ast.Name)
                    and isinstance(assign.value, ast.Call)
                    and _dotted_name(assign.value.func)
                    == "pl.tile.get_block_idx"
                }
                block_index_used = _block_index_used_in_transfer(
                    node, block_index_names
                )
                c3_peer_grids.append(
                    {
                        "method": method_name,
                        "name_hint": hint.value,
                        "tid": tid,
                        "block_index_names": sorted(block_index_names),
                        "block_index_used": block_index_used,
                        "writes_moe_out": _writes_symbol(node, "moe_out"),
                        "node": node,
                    }
                )
    c3_spmd_names = [
        item["name_hint"] for item in c3_peer_grids
    ]
    captured_peer_tids = {
        item["tid"]
        for item in c3_peer_grids
        if item["tid"] is not None
        and item["block_index_names"]
        and item["block_index_used"]
        and not item["writes_moe_out"]
    }
    checks.append(
        _check(
            "dedicated_peer_worker_abi",
            bool(peer_formals or captured_peer_tids),
            (
                f"peer formals={sorted(peer_formals)}, "
                f"captured peer grids={sorted(captured_peer_tids)}"
                if peer_formals or captured_peer_tids
                else (
                    "no dedicated peer formal or block-indexed peer SPMD "
                    "worker exists"
                )
            ),
        )
    )
    checks.append(
        _check(
            "canonical_peer_spmd_grid",
            bool(captured_peer_tids),
            (
                f"C3 peer SPMD grids={sorted(c3_spmd_names)}, "
                f"captured tids={sorted(captured_peer_tids)}"
                if captured_peer_tids
                else (
                    "no captured C3 peer SPMD grid uses its block index in "
                    "remote/slab movement without writing moe_out"
                )
            ),
        )
    )

    peer_join_names: set[tuple[str, str]] = set()
    downstream_join_consumers: list[str] = []
    for method_name, method in methods.items():
        method_peer_tids = {
            item["tid"]
            for item in c3_peer_grids
            if item["method"] == method_name
            and item["tid"] in captured_peer_tids
        }
        method_join_names: set[str] = set()
        for node in ast.walk(method):
            if (
                not isinstance(node, ast.Assign)
                or len(node.targets) != 1
                or not isinstance(node.targets[0], ast.Name)
                or not isinstance(node.value, ast.Call)
                or _dotted_name(node.value.func)
                != "pl.system.task_dummy"
            ):
                continue
            deps = set(_name_list(_keyword(node.value, "deps")) or [])
            if deps & method_peer_tids:
                method_join_names.add(node.targets[0].id)
                peer_join_names.add((method_name, node.targets[0].id))
        for call in ast.walk(method):
            if not isinstance(call, ast.Call):
                continue
            if _dotted_name(call.func) not in {
                "pl.spmd",
                "pl.spmd_submit",
                "pl.submit",
                "pl.at",
            }:
                continue
            deps = set(_name_list(_keyword(call, "deps")) or [])
            if deps & method_join_names:
                downstream_join_consumers.append(
                    f"{method_name}:{call.lineno}"
                )
    checks.append(
        _check(
            "explicit_peer_grid_join",
            bool(peer_join_names and downstream_join_consumers),
            (
                f"peer joins={sorted(peer_join_names)}, "
                f"downstream consumers={downstream_join_consumers}"
                if peer_join_names and downstream_join_consumers
                else (
                    "no task_dummy tied to a captured peer-grid TaskId and "
                    "then consumed by a downstream task exists"
                )
            ),
        )
    )

    dispatch_pull = methods.get("_dispatch_pull")
    dispatch_full_output_abi = False
    if dispatch_pull is not None:
        dispatch_formals = {arg.arg for arg in dispatch_pull.args.args}
        dispatch_full_output_abi = {
            "recv_x",
            "recv_scale",
            "recv_counts",
            "inverse_map_out",
            "local_expert_offset",
            "local_expert_count",
        }.issubset(dispatch_formals)
    checks.append(
        _check(
            "non_aliasing_dispatch_peer_abi",
            not dispatch_full_output_abi
            and bool(peer_formals or captured_peer_tids),
            (
                "dispatch peer workers expose peer-owned output slabs"
                if not dispatch_full_output_abi
                and (peer_formals or captured_peer_tids)
                else (
                    "_dispatch_pull still owns full shared recv/count/map "
                    "outputs; duplicating it per peer would alias"
                )
            ),
        )
    )

    pull_routed = methods.get("_pull_routed_y")
    route_stage_present = False
    pull_writes_moe_out = False
    if pull_routed is not None:
        route_stage_present = _annotation_contains_shape(
            pull_routed,
            shape_fragment="[1, HIDDEN]",
        )
        pull_writes_moe_out = _writes_symbol(pull_routed, "moe_out")
    checks.append(
        _check(
            "non_aliasing_combine_peer_abi",
            not route_stage_present
            and not pull_writes_moe_out
            and bool(captured_peer_tids),
            (
                "peer workers stage disjoint route rows and do not write moe_out"
                if not route_stage_present
                and not pull_writes_moe_out
                and captured_peer_tids
                else (
                    "_pull_routed_y still combines runtime-peer TGET, shared "
                    "[1,HIDDEN] route_stage, FP32 accumulation, and moe_out "
                    "writes in one InCore worker"
                )
            ),
        )
    )

    fake_zero_dependencies: list[int] = []
    for node in ast.walk(program):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
            continue
        sides = (node.left, node.right)
        has_zero = any(
            isinstance(side, ast.Constant) and side.value == 0 for side in sides
        )
        has_token_like = any(
            isinstance(side, ast.Name)
            and side.id in {
                "token",
                "active_tokens",
                "active_count",
                "notify_token",
            }
            for side in sides
        )
        if has_zero and has_token_like:
            fake_zero_dependencies.append(node.lineno)
    checks.append(
        _check(
            "no_fake_zero_dependency",
            not fake_zero_dependencies,
            (
                "no token/control scalar multiplied by zero"
                if not fake_zero_dependencies
                else f"fake dependency lines={fake_zero_dependencies}"
            ),
        )
    )

    # C3 does not prescribe a universal 512-byte signal ABI. Only record the
    # canonical reused epoch slots that already participate in notify/wait.
    def _notify_targets(name: str) -> bool:
        return any(
            _dotted_name(_keyword(call, "target")) == name
            and _dotted_name(_keyword(call, "op"))
            == "pld.NotifyOp.AtomicAdd"
            for call in _calls(program, "pld.system.notify")
        )

    def _waits_on(name: str) -> bool:
        return any(
            _dotted_name(_keyword(call, "signal")) == name
            and _dotted_name(_keyword(call, "cmp")) == "pld.WaitCmp.Ge"
            for call in _calls(program, "pld.system.wait")
        )

    reused_epoch_signal_contract = {
        "count_done": (
            "moe_count_done_sig_stack_buf = "
            "pld.alloc_window_buffer(COMM_CONTROL_SIGNAL_BYTES)"
            in source
            and _notify_targets("count_done_sig")
            and _waits_on("count_done_sig")
            and "moe_epoch * 2" in source
        ),
        "combine_done": (
            "moe_combine_done_sig_stack_buf = "
            "pld.alloc_window_buffer(COMM_CONTROL_SIGNAL_BYTES)"
            in source
            and (
                _notify_targets("combine_done")
                or _notify_targets("combine_done_sig")
            )
            and (
                _waits_on("combine_done")
                or _waits_on("combine_done_sig")
            )
            and "moe_epoch * 2" in source
        ),
        "stride_constant": (
            "COMM_CONTROL_SIGNAL_BYTES = 512" in source
            and "COMM_SIGNAL_STRIDE_I32 = COMM_CONTROL_SIGNAL_BYTES // 4"
            in source
        ),
    }
    checks.append(
        _check(
            "stacked_reused_control_signal_scope_only",
            all(reused_epoch_signal_contract.values()),
            (
                "512-byte evidence is limited to canonical stacked/reused "
                f"notify/wait epoch slots: {reused_epoch_signal_contract}"
            ),
        )
    )

    source_gate_passed = all(item["passed"] for item in checks)
    blockers = [
        item["detail"]
        for item in checks
        if not item["passed"]
        and item["name"]
        not in {
            "single_canonical_program",
            "retired_single_chip_file_absent",
            "canonical_c3_methods_present",
            "no_incore_parallel",
            "no_fake_zero_dependency",
            "stacked_reused_control_signal_scope_only",
        }
    ]
    if source_gate_passed:
        blockers.extend(
            [
                "source checks cannot prove lowered TaskId/DAG ordering",
                "no 2-rank/8-rank non-aliasing device evidence is attached",
                "no multi-epoch liveness or vLLM semantic evidence is attached",
            ]
        )
    return {
        "decision": "NO-GO",
        "source_gate": "PASS" if source_gate_passed else "NO-GO",
        "production_ready": False,
        "evidence_class": "source_contract",
        "checks": checks,
        "blockers": blockers,
        "required_abi": {
            "dispatch": (
                "orchestration-level peer SPMD grid; each block owns fixed "
                "peer-major recv/count slabs; compact/map consumer runs after "
                "a grid-wide TaskId join"
            ),
            "combine": (
                "peer-pull grid writes only a peer-major route slab; explicit "
                "join; one reducer task owns each moe_out token and preserves "
                "the existing FP32 top-k accumulation order"
            ),
            "completion": (
                "epoch completion notify occurs after the final compact/token "
                "consumer, not merely after remote transfer"
            ),
        },
        "protected_scope": {
            "audit_is_read_only": True,
            "canonical_modified_by_probe": False,
            "tp_all_reduce_audited_for_rewrite": False,
            "retired_file_restored": False,
            "generic_512_signal_requirement": False,
        },
        "source": str(CANONICAL),
    }


def _build_program() -> Any:
    """Build the one minimal C3 program; called only in compile mode."""
    _ensure_pypto_path()

    from pypto.backend import BackendType, set_backend_type

    set_backend_type(BackendType.Ascend910B)

    import pypto.language as pl
    import pypto.language.distributed as pld

    @pl.program
    class C3PeerFanoutProbe:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            local_routes: pl.Tensor[[TOKENS, HIDDEN], pl.BF16],
            moe_out: pl.Out[pl.Tensor[[TOKENS, HIDDEN], pl.BF16]],
            peer_routes: pld.DistributedTensor[
                [TOKENS, HIDDEN], pl.BF16
            ],
            ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[TOKENS, HIDDEN], pl.BF16]:
            # Publish this rank's complete token slab. The captured grid
            # TaskId is a real data-production edge into the readiness task.
            with pl.spmd(
                TOKENS,
                name_hint="c3_publish_local_routes",
            ) as publish_tid:
                token = pl.tile.get_block_idx()
                route_row = pl.load(
                    local_routes, [token, 0], [1, HIDDEN],
                )
                pl.store(route_row, [token, 0], peer_routes)

            # Cross-rank readiness is distinct from the grid join. It ensures
            # every peer window is populated before any TGET starts.
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="c3_peer_ready",
                deps=[publish_tid],
            ) as ready_tid:
                for peer in pl.range(N_RANKS):
                    if peer != my_rank:
                        pld.system.notify(
                            target=ready,
                            peer=peer,
                            offsets=[my_rank, 0],
                            value=1,
                            op=pld.NotifyOp.AtomicAdd,
                        )
                for source_peer in pl.range(N_RANKS):
                    if source_peer != my_rank:
                        pld.system.wait(
                            signal=ready,
                            offsets=[source_peer, 0],
                            expected=1,
                            cmp=pld.WaitCmp.Ge,
                        )

            # Peer-major local GM slab. Grid block `peer` owns exactly rows
            # [peer * TOKENS, (peer + 1) * TOKENS), so all writers are
            # disjoint even though the peer pulls execute concurrently.
            peer_slab = pl.create_tensor(
                [N_RANKS * TOKENS, HIDDEN],
                dtype=pl.BF16,
            )
            with pl.spmd(
                N_RANKS,
                name_hint="c3_peer_pull",
                deps=[ready_tid],
            ) as peer_pull_tid:
                peer = pl.tile.get_block_idx()
                if peer == my_rank:
                    for token in pl.range(TOKENS):
                        self_row = pl.load(
                            peer_routes, [token, 0], [1, HIDDEN],
                        )
                        pl.store(
                            self_row,
                            [peer * TOKENS + token, 0],
                            peer_slab,
                        )
                else:
                    pld.tensor.get(
                        peer_slab,
                        peer=peer,
                        src=peer_routes,
                        dst_offsets=[peer * TOKENS, 0],
                        src_offsets=[0, 0],
                        shape=[TOKENS, HIDDEN],
                    )

            # A grid TaskId represents the whole peer fan-out. Materialize an
            # explicit join before the token consumers; no fake tensor edge.
            peer_pull_join = pl.system.task_dummy(deps=[peer_pull_tid])

            # One task owns one output token. Peer rows are consumed only
            # after the explicit grid join, so peer tasks never race on
            # moe_out and no shared [1,HIDDEN] staging row is needed.
            with pl.spmd(
                TOKENS,
                name_hint="c3_token_reduce",
                deps=[peer_pull_join],
            ) as token_reduce_tid:
                token = pl.tile.get_block_idx()
                acc = pl.cast(
                    pl.load(peer_slab, [token, 0], [1, HIDDEN]),
                    target_type=pl.FP32,
                )
                for peer in pl.range(1, N_RANKS):
                    peer_row = pl.load(
                        peer_slab,
                        [peer * TOKENS + token, 0],
                        [1, HIDDEN],
                    )
                    acc = pl.add(
                        acc,
                        pl.cast(peer_row, target_type=pl.FP32),
                    )
                pl.store(
                    pl.cast(
                        acc,
                        target_type=pl.BF16,
                        mode="rint",
                    ),
                    [token, 0],
                    moe_out,
                )
            return moe_out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            local_routes: pl.Tensor[
                [N_RANKS, TOKENS, HIDDEN], pl.BF16
            ],
            moe_out: pl.Out[
                pl.Tensor[[N_RANKS, TOKENS, HIDDEN], pl.BF16]
            ],
        ):
            peer_routes_buf = pld.alloc_window_buffer(
                TOKENS * HIDDEN * 2,
            )
            ready_buf = pld.alloc_window_buffer(N_RANKS * 4)
            for rank in pl.range(pld.world_size()):
                peer_routes = pld.window(
                    peer_routes_buf,
                    [TOKENS, HIDDEN],
                    dtype=pl.BF16,
                )
                ready = pld.window(
                    ready_buf,
                    [N_RANKS, 1],
                    dtype=pl.INT32,
                )
                self.chip_orch(
                    local_routes[rank],
                    moe_out[rank],
                    peer_routes,
                    ready,
                    rank,
                    device=rank,
                )

    return C3PeerFanoutProbe


def _ensure_pypto_path() -> None:
    pypto_python = PYPTO_WORKSPACE / "python"
    if pypto_python.is_dir() and str(pypto_python) not in sys.path:
        sys.path.insert(0, str(pypto_python))


def _parse_devices(value: str) -> list[int]:
    devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(devices) != N_RANKS or len(set(devices)) != N_RANKS:
        raise ValueError(
            f"expected {N_RANKS} distinct device ids, got {devices}"
        )
    return devices


def _base_report(mode: str, contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "mode": mode,
        "status": "GO" if contract.get("passed") else "NO-GO",
        "evidence_class": "source_contract",
        "production_claim": False,
        "contract": contract,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _write_report(out: str | None, report: dict[str, Any]) -> None:
    rendered = json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    )
    print(rendered, flush=True)
    if out is None:
        return
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "c3_peer_fanout_report.json"
    path.write_text(rendered + "\n", encoding="utf-8")
    print(f"C3_REPORT={path}", flush=True)


def _run_contract(args: argparse.Namespace) -> int:
    try:
        contract = _source_contract()
        report = _base_report("contract", contract)
    except Exception as exc:  # Fail closed on parser/source drift.
        report = {
            "schema": SCHEMA,
            "mode": "contract",
            "status": "NO-GO",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
    _write_report(args.out, report)
    return 0 if report["status"] == "GO" else 1


def _run_audit(args: argparse.Namespace) -> int:
    try:
        audit = _canonical_c3_audit()
        report = {
            "schema": SCHEMA,
            "mode": "audit",
            "status": audit["decision"],
            "evidence_class": "source_contract",
            "production_claim": True,
            "canonical_audit": audit,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    except Exception as exc:
        report = {
            "schema": SCHEMA,
            "mode": "audit",
            "status": "NO-GO",
            "evidence_class": "source_contract",
            "production_claim": True,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
    _write_report(args.out, report)
    return 0 if report["status"] == "GO" else 1


def _run_compile(args: argparse.Namespace) -> int:
    try:
        contract = _source_contract()
    except Exception as exc:
        report = {
            "schema": SCHEMA,
            "mode": "compile",
            "status": "NO-GO",
            "stage": "contract",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        _write_report(args.out, report)
        return 1

    report = _base_report("compile", contract)
    report["evidence_class"] = "compile"
    report["production_claim"] = False
    report["compile"] = {
        "attempted": False,
        "backend_compile": not args.skip_ptoas,
        "platform": args.platform,
        "device_ids": None,
    }
    if not contract["passed"]:
        report["status"] = "NO-GO"
        report["stage"] = "contract"
        _write_report(args.out, report)
        return 1

    try:
        devices = _parse_devices(args.device)
        report["compile"]["device_ids"] = devices
        if args.out:
            build_dir = Path(args.out) / "build_output"
            build_dir.mkdir(parents=True, exist_ok=True)
            import os

            os.environ["PYPTO_PROG_BUILD_DIR"] = str(build_dir)

        _ensure_pypto_path()
        from pypto import ir
        from pypto.ir.distributed_compiled_program import DistributedConfig

        program = _build_program()
        report["compile"]["attempted"] = True
        compiled = ir.compile(
            program,
            platform=args.platform,
            distributed_config=DistributedConfig(
                device_ids=devices,
                num_sub_workers=0,
            ),
            skip_ptoas=args.skip_ptoas,
            dump_passes=args.dump_passes,
        )
        report["compile"].update(
            {
                "passed": True,
                "output_dir": str(compiled.output_dir),
                "classification": (
                    "frontend_lowering_only"
                    if args.skip_ptoas
                    else "real_backend_compile"
                ),
            }
        )
        report["status"] = "GO"
    except Exception as exc:  # A compile exception is always an explicit NO-GO.
        report["status"] = "NO-GO"
        report["stage"] = "compile"
        report["compile"].update(
            {
                "passed": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        )

    _write_report(args.out, report)
    return 0 if report["status"] == "GO" else 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("contract", "audit", "compile"),
        default="contract",
    )
    parser.add_argument(
        "--platform",
        choices=("a2a3", "a2a3sim"),
        default="a2a3sim",
    )
    parser.add_argument(
        "--device",
        default="0,1",
        help=f"exactly {N_RANKS} logical compile device ids",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="optional report/build directory",
    )
    parser.add_argument(
        "--skip-ptoas",
        action="store_true",
        help="frontend/lowering diagnostic only; not a real backend compile",
    )
    parser.add_argument("--dump-passes", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.mode == "contract":
        return _run_contract(args)
    if args.mode == "audit":
        return _run_audit(args)
    return _run_compile(args)


if __name__ == "__main__":
    sys.exit(main())
