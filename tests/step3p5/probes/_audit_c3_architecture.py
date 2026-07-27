# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""C3 architecture audit for the canonical Step3.5 decode graph.

This probe is intentionally static and does not import ``pypto``.  It records
the current C3 release decision without changing the canonical production
program:

* exactly one ``@pl.program`` is present;
* ``pl.parallel`` is not placed in an InCore function;
* dispatch/combine peer work has not been mistaken for legal InCore
  parallelism;
* a non-aliasing per-peer task ABI and an explicit TaskId join are required
  before C3 can move from NO-GO to an implementation.

The expected result for the current canonical source is ``NO-GO``.  The
pytest entry point asserts that this NO-GO is explicit and auditable; the
command-line entry point returns a non-zero status so it can also be used as a
release gate.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[3]
_CANONICAL = _ROOT / "models" / "step3p5" / "decode_fwd.py"
_C3_METHODS = {
    "_dispatch_pack_publish",
    "_dispatch_pull",
    "_dispatch_stage",
    "_pull_routed_y",
    "_stage_routed_src",
}
_PEER_NAMES = {"peer", "src", "dst", "source_peer", "target_peer"}


@dataclass(frozen=True)
class Finding:
    status: str
    check: str
    evidence: str


def _dotted_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _decorator_is(node: ast.expr, dotted: str) -> bool:
    if _dotted_name(node) == dotted:
        return True
    return (
        isinstance(node, ast.Call)
        and _dotted_name(node.func) == dotted
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


def _source_line(source_lines: list[str], line: int) -> str:
    if 1 <= line <= len(source_lines):
        return source_lines[line - 1].strip()
    return ""


def _calls(function: ast.AST, attr: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
    ]


def _peer_loops(function: ast.FunctionDef) -> list[tuple[str, int, str]]:
    result: list[tuple[str, int, str]] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.For):
            continue
        if not isinstance(node.target, ast.Name):
            continue
        if not isinstance(node.iter, ast.Call):
            continue
        if _dotted_name(node.iter.func) != "pl.range":
            continue
        if node.target.id not in _PEER_NAMES:
            continue
        result.append((node.target.id, node.lineno, ast.unparse(node.iter)))
    return result


def _method_map(program_class: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in program_class.body
        if isinstance(node, ast.FunctionDef)
    }


def audit_c3() -> tuple[str, list[Finding]]:
    source = _CANONICAL.read_text(encoding="utf-8")
    source_lines = source.splitlines()
    tree = ast.parse(source, filename=str(_CANONICAL))

    program_classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(_decorator_is(dec, "pl.program") for dec in node.decorator_list)
    ]
    findings: list[Finding] = []

    if len(program_classes) == 1:
        program_class = program_classes[0]
        findings.append(
            Finding(
                "PASS",
                "single-canonical-program",
                f"{program_class.name} at line {program_class.lineno} is "
                "the only class decorated with @pl.program",
            )
        )
    else:
        names = ", ".join(node.name for node in program_classes) or "<none>"
        findings.append(
            Finding(
                "NO-GO",
                "single-canonical-program",
                f"expected exactly one @pl.program, found {len(program_classes)}: {names}",
            )
        )
        return "NO-GO", findings

    methods = _method_map(program_class)

    incore_parallel_sites: list[str] = []
    for function in methods.values():
        if _function_type(function) != "pl.FunctionType.InCore":
            continue
        for call in _calls(function, "parallel"):
            incore_parallel_sites.append(
                f"{function.name}:{call.lineno}: "
                f"{_source_line(source_lines, call.lineno)}"
            )
    if incore_parallel_sites:
        findings.append(
            Finding(
                "NO-GO",
                "no-incore-parallel-pseudocompletion",
                "pl.parallel found in InCore: " + "; ".join(incore_parallel_sites),
            )
        )
    else:
        findings.append(
            Finding(
                "PASS",
                "no-incore-parallel-pseudocompletion",
                "no pl.parallel call occurs in any InCore function",
            )
        )

    peer_sites: list[str] = []
    for name in sorted(_C3_METHODS):
        function = methods.get(name)
        if function is None:
            continue
        for variable, line, iterator in _peer_loops(function):
            peer_sites.append(
                f"{name}:{line}: for {variable} in {iterator} "
                f"({_source_line(source_lines, line)})"
            )
    pull_function = methods.get("_pull_routed_y")
    has_dynamic_peer_get = bool(
        pull_function
        and _calls(pull_function, "get")
        and any(
            isinstance(node, ast.Name) and node.id == "dst"
            for node in ast.walk(pull_function)
        )
    )
    if peer_sites or has_dynamic_peer_get:
        extra = (
            "; _pull_routed_y uses runtime dst + pld.tensor.get inside the "
            "token/top-k accumulation loop"
            if has_dynamic_peer_get
            else ""
        )
        findings.append(
            Finding(
                "NO-GO",
                "peer-work-at-orchestration-boundary",
                "peer work remains in sequential InCore code: "
                + "; ".join(peer_sites)
                + extra,
            )
        )
    else:
        findings.append(
            Finding(
                "PASS",
                "peer-work-at-orchestration-boundary",
                "no peer work was found in the audited dispatch/combine helpers",
            )
        )

    peer_formal_methods: list[str] = []
    for function in methods.values():
        formal_names = {arg.arg for arg in function.args.args}
        if formal_names & {"peer", "peer_id", "source_peer", "target_peer"}:
            peer_formal_methods.append(
                f"{function.name}:{function.lineno}"
            )
    if peer_formal_methods:
        findings.append(
            Finding(
                "PASS",
                "non-aliasing-per-peer-abi",
                "peer formal found in: " + ", ".join(peer_formal_methods),
            )
        )
    else:
        findings.append(
            Finding(
                "NO-GO",
                "non-aliasing-per-peer-abi",
                "no dedicated task/function accepts a peer identifier; "
                "current ABI keeps peer as a loop-local scalar",
            )
        )

    submit_calls = [
        call
        for call in ast.walk(program_class)
        if isinstance(call, ast.Call)
        and _dotted_name(call.func) == "pl.submit"
    ]
    spmd_submit_calls = [
        call
        for call in ast.walk(program_class)
        if isinstance(call, ast.Call)
        and _dotted_name(call.func) == "pl.spmd_submit"
    ]
    task_dummy_calls = [
        call
        for call in ast.walk(program_class)
        if isinstance(call, ast.Call)
        and _dotted_name(call.func) == "pl.system.task_dummy"
    ]
    task_id_array_calls = [
        call
        for call in ast.walk(program_class)
        if isinstance(call, ast.Call)
        and _dotted_name(call.func) == "pl.array.create"
        and len(call.args) >= 2
        and _dotted_name(call.args[1]) == "pl.TASK_ID"
    ]
    explicit_join = bool(task_dummy_calls or task_id_array_calls)
    if explicit_join:
        evidence = (
            f"task_dummy={len(task_dummy_calls)}, "
            f"task_id_arrays={len(task_id_array_calls)}"
        )
        findings.append(Finding("PASS", "explicit-peer-join", evidence))
    else:
        findings.append(
            Finding(
                "NO-GO",
                "explicit-peer-join",
                "no task_dummy or TASK_ID array fan-in exists; "
                f"submit={len(submit_calls)}, "
                f"spmd_submit={len(spmd_submit_calls)}",
            )
        )

    if pull_function is not None:
        has_accumulator = any(
            isinstance(node, ast.Name) and node.id == "acc"
            for node in ast.walk(pull_function)
        )
        stores_moe_out = any(
            isinstance(node, ast.Name) and node.id == "moe_out"
            for node in ast.walk(pull_function)
        )
        if has_accumulator and stores_moe_out:
            findings.append(
                Finding(
                    "NO-GO",
                    "combine-write-alias-safety",
                    "_pull_routed_y carries acc across token/top-k routes and "
                    "stores into shared moe_out; combine cannot be peer-parallel "
                    "without a route-stage plus explicit reduction/join",
                )
            )
        else:
            findings.append(
                Finding(
                    "PASS",
                    "combine-write-alias-safety",
                    "no shared weighted-gather accumulator was detected",
                )
            )

    overall = (
        "NO-GO"
        if any(finding.status == "NO-GO" for finding in findings)
        else "PASS"
    )
    return overall, findings


def format_report(overall: str, findings: list[Finding]) -> str:
    lines = [
        "C3_ARCHITECTURE_AUDIT",
        f"source={_CANONICAL}",
        f"overall={overall}",
    ]
    lines.extend(
        f"[{finding.status}] {finding.check}: {finding.evidence}"
        for finding in findings
    )
    return "\n".join(lines)


def test_c3_audit_reports_explicit_no_go() -> None:
    overall, findings = audit_c3()
    report = format_report(overall, findings)
    print(report)
    assert overall == "NO-GO", report
    checks = {finding.check: finding.status for finding in findings}
    assert checks["single-canonical-program"] == "PASS", report
    assert checks["no-incore-parallel-pseudocompletion"] == "PASS", report
    assert checks["peer-work-at-orchestration-boundary"] == "NO-GO", report
    assert checks["non-aliasing-per-peer-abi"] == "NO-GO", report
    assert checks["explicit-peer-join"] == "NO-GO", report
    assert checks["combine-write-alias-safety"] == "NO-GO", report


def main() -> int:
    overall, findings = audit_c3()
    print(format_report(overall, findings))
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
