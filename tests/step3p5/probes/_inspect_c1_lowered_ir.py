# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""PERF-C1 canonical lowering and task-DAG inspection probe.

This probe is diagnostic-only.  It never changes the canonical model source or
patches compiler/runtime state.  It can:

* compile ``models.step3p5.decode_fwd:whole_decode_step3p5`` with explicit
  pass dumps;
* compile the selected MTP hidden-only variants;
* verify canonical host-side ``[128, 1]`` signal-stack views plus lowered
  one-row control pviews with physical stride 128, while independent MTP
  signal windows retain the compact physical stride 8;
* inspect generated kernel sources and, when supplied, a runtime ``deps.json``
  task graph;
* check that wait tasks carry only control/token resources and that a token
  produced by a wait task has a real RAW edge to a consumer.

The runtime task graph is produced by a separate device run with
``RunConfig(enable_dep_gen=True)``.  Compilation alone cannot manufacture
device task instances, so the probe reports that part as unavailable unless a
``--deps`` file is supplied.  Use ``--require-task-dag`` in a release gate.

Examples::

    python -m tests.step3p5.probes._inspect_c1_lowered_ir \
        --platform a2a3sim \
        --devices 0,1,2,3,4,5,6,7 \
        --build-dir /tmp/c1-lowered-ir \
        --skip-ptoas

    python -m tests.step3p5.probes._inspect_c1_lowered_ir \
        --no-compile \
        --main-build-dir /tmp/c1-lowered-ir/main \
        --mtp-build-dir /tmp/c1-lowered-ir/mtp_0 \
        --deps /tmp/c1-liveness/dfx/deps.json \
        --require-task-dag
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable


_ROOT = Path(__file__).resolve().parents[3]
_CANONICAL_SOURCE = _ROOT / "models" / "step3p5" / "decode_fwd.py"
_MTP_SOURCE = _ROOT / "models" / "step3p5" / "mtp_hidden_fwd.py"

COMM_SIGNAL_ROWS = 128
MTP_SIGNAL_ROWS = 8
C1_MANIFEST_NAME = "c1_build_manifest.json"
_SIGNAL_WORDS = (
    "signal",
    "sig",
    "meta_arrived",
    "data_arrived",
    "combine_arrived",
    "arrived",
)
_TOKEN_WORDS = ("active_token", "notify_token", "reuse_token", "token")
_EP_DATA_WORDS = (
    "pub_counts",
    "recv_x",
    "recv_scale",
    "send_x",
    "send_scale",
    "routed_src",
    "routed_y",
    "routed_x",
    "local_routed",
    "route_stage",
    "expert_indices",
    "expert_weights",
    "inverse_map",
)
_WAIT_WORDS = (
    "_wait_",
    "wait_previous",
    "wait_dispatch",
    "wait_combine",
    "twait",
    "t_wait",
)
_CANONICAL_SIGNAL_SLOT_SPECS = {
    "dense_attn_signal_stack": {
        "backing": "dense_attn_signal_stack_buf", "slot_kind": "stacked",
        "extent": "NUM_DENSE_LAYERS", "protocol": "tp_all_reduce",
        "consumers": ("full_chip_orch", "swa_chip_orch"),
        "binding_needles": ("attn_tmp_window, attn_signal_window, my_rank",),
    },
    "dense_mlp_signal_stack": {
        "backing": "dense_mlp_signal_stack_buf", "slot_kind": "stacked",
        "extent": "NUM_DENSE_LAYERS", "protocol": "tp_all_reduce",
        "consumers": ("full_chip_orch", "swa_chip_orch"),
        "binding_needles": ("mlp_tmp_window, mlp_signal_window, my_rank",),
    },
    "moe_attn_signal_stack": {
        "backing": "moe_attn_signal_stack_buf", "slot_kind": "stacked",
        "extent": "NUM_MOE_LAYERS_TOTAL", "protocol": "tp_all_reduce",
        "consumers": ("full_moe_chip_orch", "swa_moe_chip_orch"),
        "binding_needles": ("attn_tmp_window, attn_signal_window, my_rank",),
    },
    "moe_meta_arrived_stack": {
        "backing": "moe_meta_arrived_stack_buf", "slot_kind": "reused",
        "extent": None, "protocol": "dispatch_arrival",
        "consumers": ("dispatch_step",),
        "binding_needles": ("recv_meta, meta_arrived, recv_x, recv_aux",),
    },
    "moe_data_arrived_stack": {
        "backing": "moe_data_arrived_stack_buf", "slot_kind": "reused",
        "extent": None, "protocol": "dispatch_arrival",
        "consumers": ("dispatch_step",),
        "binding_needles": ("recv_route, data_arrived",),
    },
    "moe_sh_signal_stack": {
        "backing": "moe_sh_signal_stack_buf", "slot_kind": "stacked",
        "extent": "NUM_MOE_LAYERS_TOTAL", "protocol": "tp_all_reduce",
        "consumers": ("full_moe_chip_orch", "swa_moe_chip_orch"),
        "binding_needles": ("sh_y, sh_tmp_window, sh_signal_window, my_rank",),
    },
    "moe_combine_arrived_stack": {
        "backing": "moe_combine_arrived_stack_buf", "slot_kind": "reused",
        "extent": None, "protocol": "combine_arrival",
        "consumers": ("combine_step",),
        "binding_needles": ("combine_arrived", "recv_route_compact, routed_y_buf"),
    },
}
_CANONICAL_SIGNAL_PROTOCOL_FUNCTIONS = {
    "tp_all_reduce": ("tp_all_reduce",),
    "dispatch_arrival": ("dispatch_step",),
    "combine_arrival": ("combine_step",),
}



def _ensure_repo_on_path() -> None:
    root = str(_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _parse_devices(value: str) -> list[int]:
    devices = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(devices) != 8 or len(set(devices)) != 8:
        raise ValueError(f"expected eight distinct device ids, got {devices}")
    return devices


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_version(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable:{type(exc).__name__}"
    output = (result.stdout or result.stderr).strip()
    return output.splitlines()[0][:500] if output else f"exit={result.returncode}"


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return {"commit": "", "dirty": None, "status": []}
    return {
        "commit": commit,
        "dirty": bool(status),
        "status": status[:200],
    }


def _write_build_manifest(
    root: Path,
    *,
    source_path: Path,
    program: str,
    compile_started_at: float,
    compile_finished_at: float,
) -> Path:
    """Bind one lowered artifact to the exact source used to compile it."""
    root = root.resolve()
    source_path = source_path.resolve()
    manifest = {
        "schema": "step3p5.c1.build_provenance.v1",
        "artifact_root": str(root),
        "program": str(program),
        "source_path": str(source_path),
        "source_sha256": _sha256_path(source_path),
        "source_mtime": source_path.stat().st_mtime,
        "compile_started_at": float(compile_started_at),
        "compile_finished_at": float(compile_finished_at),
        "git": _git_state(),
        "compiler": {
            "python": sys.version.splitlines()[0],
            "ptoas": _command_version(["ptoas", "--version"]),
        },
    }
    path = root / C1_MANIFEST_NAME
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _artifact_provenance_contract(
    root: Path,
    *,
    canonical: bool,
) -> dict[str, Any]:
    manifest_path = root / C1_MANIFEST_NAME
    source_path = _CANONICAL_SOURCE if canonical else _MTP_SOURCE
    if not manifest_path.is_file():
        return {
            "manifest": str(manifest_path),
            "available": False,
            "source": str(source_path),
            "pass": False,
            "checks": {
                "manifest_present": False,
                "source_path": False,
                "source_sha256": False,
                "artifact_root": False,
                "compile_after_source": False,
                "git_commit_present": False,
                "compiler_version_present": False,
                "ptoas_version_present": False,
            },
            "error": (
                "lowered artifact has no current-source provenance manifest; "
                "shape/stride may be inspected but cannot be attributed to "
                "the current canonical/MTP source"
            ),
        }
    try:
        data = json.loads(_read_text(manifest_path))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "manifest": str(manifest_path),
            "available": True,
            "source": str(source_path),
            "pass": False,
            "checks": {},
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    compile_finished = float(data.get("compile_finished_at", -1))
    checks = {
        "manifest_present": data.get("schema")
        == "step3p5.c1.build_provenance.v1",
        "source_path": data.get("source_path") == str(source_path.resolve()),
        "source_sha256": data.get("source_sha256") == _sha256_path(source_path),
        "artifact_root": data.get("artifact_root") == str(root.resolve()),
        "compile_after_source": compile_finished >= source_path.stat().st_mtime,
        "git_commit_present": bool(data.get("git", {}).get("commit")),
        "compiler_version_present": bool(
            data.get("compiler", {}).get("python")
        ),
        "ptoas_version_present": bool(
            data.get("compiler", {}).get("ptoas")
        ),
    }
    return {
        "manifest": str(manifest_path),
        "available": True,
        "source": str(source_path),
        "manifest_data": data,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _jsonable(value: Any) -> Any:
    """Convert the small report objects used by this probe to JSON values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _eval_static_int(node: ast.AST, values: dict[str, int]) -> int | None:
    """Evaluate the small integer-expression subset used by model constants."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.Name):
        return values.get(node.id)
    if isinstance(node, ast.UnaryOp):
        value = _eval_static_int(node.operand, values)
        if value is None:
            return None
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        return None
    if isinstance(node, ast.BinOp):
        left = _eval_static_int(node.left, values)
        right = _eval_static_int(node.right, values)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, (ast.FloorDiv, ast.Div)):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
    return None


def _static_ints(
    source: str,
    *,
    seed: dict[str, int] | None = None,
) -> dict[str, int]:
    """Resolve module-level integer assignments without importing PyPTO."""
    values = dict(seed or {})
    tree = ast.parse(source)
    # Constants in these modules are topologically ordered.  Repeat a few
    # times so a harmless forward alias still resolves without executing code.
    assignments: list[tuple[str, ast.AST]] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                assignments.append((target.id, node.value))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assignments.append((node.target.id, node.value))
    for _ in range(len(assignments) + 1):
        changed = False
        for name, value_node in assignments:
            value = _eval_static_int(value_node, values)
            if value is not None and values.get(name) != value:
                values[name] = value
                changed = True
        if not changed:
            break
    return values


def _function_segment(source: str, name: str) -> str:
    tree = ast.parse(source)
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        return ""
    return ast.get_source_segment(source, matches[0]) or ""


def _canonical_signal_scope_contract(source: str) -> dict[str, Any]:
    """Check the local 512B predicate instead of a generic signal ABI.

    A physical stride of 128 INT32 rows is accepted only for the named
    canonical control slots that are stacked/reused and participate in the
    notify/wait/AtomicAdd protocols.  The protocol bodies must still address
    only ``n_ranks``/``group_size`` logical peers.
    """
    expected_backings = {
        str(spec["backing"]) for spec in _CANONICAL_SIGNAL_SLOT_SPECS.values()
    }
    allocations = {
        match.group("name"): _normalise_whitespace(match.group("expr"))
        for match in re.finditer(
            r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
            r"pld\.alloc_window_buffer\((?P<expr>.*?)\)",
            source,
            flags=re.DOTALL,
        )
    }
    stride_backings = {
        name
        for name, expr in allocations.items()
        if "COMM_CONTROL_SIGNAL_BYTES" in expr
    }
    slot_rows: list[dict[str, Any]] = []
    for slot_name, spec in _CANONICAL_SIGNAL_SLOT_SPECS.items():
        backing = str(spec["backing"])
        extent = spec["extent"]
        allocation = allocations.get(backing, "")
        allocation_pass = (
            "COMM_CONTROL_SIGNAL_BYTES" in allocation
            and (
                extent is None
                or str(extent) in allocation
            )
        )
        protocol = str(spec["protocol"])
        function_rows: list[dict[str, Any]] = []
        for function_name in _CANONICAL_SIGNAL_PROTOCOL_FUNCTIONS[protocol]:
            segment = _function_segment(source, function_name)
            has_control_op = (
                "pld.system.notify(" in segment
                or "pld.system.wait(" in segment
            )
            logical_bound = (
                (
                    "for peer in pl.range(n_ranks):" in segment
                    or "for src in pl.range(n_ranks):" in segment
                )
                if protocol != "tp_all_reduce"
                else (
                    "group_size = tp_size" in segment
                    and "for peer in pl.range(group_size):" in segment
                    and "for src in pl.range(group_size):" in segment
                )
            )
            function_rows.append(
                {
                    "function": function_name,
                    "control_op": has_control_op,
                    "logical_peer_bound": logical_bound,
                    "pass": has_control_op and logical_bound,
                }
            )
        binding_pass = all(
            needle in source for needle in spec["binding_needles"]
        )
        row_pass = (
            spec["slot_kind"] in {"stacked", "reused"}
            and allocation_pass
            and binding_pass
            and all(row["pass"] for row in function_rows)
        )
        slot_rows.append(
            {
                "slot": slot_name,
                "backing": backing,
                "slot_kind": spec["slot_kind"],
                "protocol": protocol,
                "allocation": allocation,
                "allocation_pass": allocation_pass,
                "binding_pass": binding_pass,
                "functions": function_rows,
                "pass": row_pass,
            }
        )
    unexpected_stride_backings = sorted(
        stride_backings - expected_backings
    )
    missing_stride_backings = sorted(
        expected_backings - stride_backings
    )
    passed = (
        all(row["pass"] for row in slot_rows)
        and not unexpected_stride_backings
        and not missing_stride_backings
    )
    return {
        "pass": passed,
        "slots": slot_rows,
        "expected_stride_backings": sorted(expected_backings),
        "actual_stride_backings": sorted(stride_backings),
        "unexpected_stride_backings": unexpected_stride_backings,
        "missing_stride_backings": missing_stride_backings,
        "generic_window_abi": False,
        "logical_rows": "n_ranks/group_size only",
    }


def _source_contract(path: Path, *, canonical: bool) -> dict[str, Any]:
    source = _read_text(path)
    compact = _normalise_whitespace(source)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "pass": bool(passed), "detail": detail})

    if canonical:
        values = _static_ints(source)
        scope_contract = _canonical_signal_scope_contract(source)
        control_bytes = values.get("COMM_CONTROL_SIGNAL_BYTES")
        stride = values.get("COMM_SIGNAL_STRIDE_I32")
        check(
            "canonical_control_signal_bytes",
            control_bytes == 512,
            f"COMM_CONTROL_SIGNAL_BYTES={control_bytes!r}, expected 512",
        )
        check(
            "canonical_signal_stride_i32",
            stride == COMM_SIGNAL_ROWS,
            f"COMM_SIGNAL_STRIDE_I32={stride!r}, expected {COMM_SIGNAL_ROWS}",
        )
        formal_hits = len(
            re.findall(
                r"DistributedTensor\s*\[\[\s*COMM_SIGNAL_STRIDE_I32\s*,\s*1\s*\]",
                compact,
            )
        )
        window_hits = len(
            re.findall(
                r"pld\.window\([^)]*signal[^)]*\[\s*NUM_[A-Z_]*COMM_SIGNAL_STRIDE_I32\s*,\s*1\s*\]",
                compact,
                flags=re.IGNORECASE,
            )
        )
        slice_hits = len(
            re.findall(
                r"pl\.slice\([^)]*signal[^)]*\[\s*COMM_SIGNAL_STRIDE_I32\s*,\s*1\s*\]",
                compact,
                flags=re.IGNORECASE,
            )
        )
        check(
            "canonical_signal_formals_use_stride",
            formal_hits >= 8,
            f"found {formal_hits} DistributedTensor formal(s) using COMM_SIGNAL_STRIDE_I32",
        )
        check(
            "canonical_signal_window_views_use_stride",
            "COMM_SIGNAL_STRIDE_I32, 1" in compact
            and "dense_attn_signal_stack_buf" in compact
            and "moe_attn_signal_stack_buf" in compact,
            f"signal window source contains stacked [COMM_SIGNAL_STRIDE_I32, 1] views; "
            f"regex window_hits={window_hits}",
        )
        check(
            "canonical_signal_slices_use_stride",
            "pl.slice(moe_attn_signal_stack" in compact
            and "COMM_SIGNAL_STRIDE_I32, 1" in compact,
            f"canonical signal slice sites present; regex slice_hits={slice_hits}",
        )
        epoch_literals = sorted(
            int(value)
            for value in re.findall(r"moe_epoch(?:_\d+)?\s*=\s*pl\.cast\((\d+)", source)
        )
        loop_epochs = values.get("NUM_MOE_LAYERS")
        has_loop_epoch = (
            "for layer_idx in pl.range(NUM_MOE_LAYERS):" in source
            and "moe_epoch = pl.cast(layer_idx + 1, pl.INT32)" in source
        )
        check(
            "canonical_epoch_1_to_42",
            has_loop_epoch
            and loop_epochs == 40
            and epoch_literals == [41, 42]
            and values.get("NUM_MOE_LAYERS_TOTAL") == 42,
            (
                f"runtime loop epochs=1..{loop_epochs!r}, explicit epochs="
                f"{epoch_literals}, total={values.get('NUM_MOE_LAYERS_TOTAL')!r}"
            ),
        )
        check(
            "canonical_three_arrival_lineages",
            all(name in source for name in (
                "meta_arrived", "data_arrived", "combine_arrived",
            ))
            and "expected=moe_epoch" in source
            and source.count("moe_epoch * n_local_experts") >= 2
            and "moe_epoch * 2" not in source,
            "meta waits for epoch; payload/combine wait for epoch*N_LOCAL",
        )
        check(
            "canonical_512b_stacked_reused_control_scope",
            bool(scope_contract["pass"]),
            (
                "512B stride is limited to named stacked/reused "
                "notify/wait/AtomicAdd control slots; "
                f"unexpected={scope_contract['unexpected_stride_backings']}, "
                f"missing={scope_contract['missing_stride_backings']}"
            ),
        )
    else:
        values = _static_ints(source, seed={"TP_WORLD_SIZE": MTP_SIGNAL_ROWS})
        rows = values.get("SIGNAL_WINDOW_ROWS")
        check(
            "mtp_signal_rows_compact",
            rows == MTP_SIGNAL_ROWS,
            f"MTP SIGNAL_WINDOW_ROWS={rows!r}, expected compact TP=8",
        )
        check(
            "mtp_signal_allocations_compact",
            bool(
                re.search(
                    r"(?:eh_sig|attn_sig|mlp_sig)\s*=\s*pld\.alloc_window_buffer\(\s*tp_size\s*\*\s*4\s*\)",
                    compact,
                )
            ),
            "MTP signal backing allocations use tp_size * 4",
        )
        check(
            "mtp_signal_views_compact",
            bool(
                re.search(
                    r"pld\.window\([^)]*(?:eh_sig|attn_sig|mlp_sig)[^)]*\[\s*tp_size\s*,\s*1\s*\]",
                    compact,
                    flags=re.IGNORECASE,
                )
            ),
            "MTP signal window views use [tp_size, 1]",
        )
        check(
            "mtp_does_not_inherit_stacked_signal_stride",
            "COMM_SIGNAL_STRIDE_I32" not in source
            and "COMM_CONTROL_SIGNAL_BYTES" not in source,
            "MTP source does not import canonical stacked signal constants",
        )

    result = {
        "path": str(path),
        "kind": "canonical" if canonical else "mtp",
        "pass": all(item["pass"] for item in checks),
        "checks": checks,
    }
    if canonical:
        result["signal_scope_contract"] = scope_contract
    return result


_LOWERED_SUFFIXES = {
    ".py",
    ".cpp",
    ".cc",
    ".c",
    ".h",
    ".hpp",
    ".pto",
    ".mlir",
    ".ll",
    ".ir",
}


def _artifact_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _LOWERED_SUFFIXES
    )


def _is_lowered_root(root: Path) -> bool:
    """Return whether *root* looks like one compiler output directory.

    A source checkout can contain plenty of Python files, but that is not a
    lowered build.  Require at least one compiler-owned marker plus one
    lowered source/artifact file so source text is never counted as a lowered
    artifact by accident.
    """
    if not root.is_dir():
        return False
    markers = (
        root / "kernel_config.py",
        root / "distributed_meta.json",
    )
    marker_dirs = (
        root / "passes_dump",
        root / "ptoas",
        root / "kernels",
        root / "orchestration",
    )
    return bool(
        _artifact_files(root)
        and (
            any(path.is_file() for path in markers)
            or any(path.is_dir() for path in marker_dirs)
        )
    )


def _missing_artifact_contract(
    root: Path | None,
    *,
    kind: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "path": str(root) if root is not None else "",
        "kind": kind,
        "available": False,
        "artifact_kind": "missing-or-invalid",
        "files_scanned": 0,
        "evidence_files": [],
        "pass": False,
        "checks": [
            {
                "name": f"{kind}_lowered_artifact_available",
                "pass": False,
                "detail": reason,
            }
        ],
    }


def _discover_main_build_dir(root: Path) -> tuple[Path | None, list[Path], list[str]]:
    """Resolve one Main lowered root from an explicit directory.

    The argument may be the final compiler directory itself or a parent that
    contains a ``whole_decode_step3p5_*`` output.  Ambiguous parent
    directories fail closed instead of selecting an unrelated historical
    build.
    """
    root = root.expanduser().resolve()
    if _is_lowered_root(root):
        return root, [root], []
    if not root.is_dir():
        return None, [], [f"Main build directory does not exist: {root}"]

    candidates = _dedupe_paths(
        path.parent
        for path in root.rglob("kernel_config.py")
        if path.is_file() and _is_lowered_root(path.parent)
    )
    if not candidates:
        candidates = _dedupe_paths(
            path
            for path in root.rglob("*")
            if path.is_dir() and _is_lowered_root(path)
        )
    if not candidates:
        return None, [], [
            f"no lowered Main artifact found below explicit directory: {root}"
        ]

    named = [
        path
        for path in candidates
        if "whole_decode_step3p5" in path.name.lower()
        or "wholedecodestep3p5" in path.name.lower()
    ]
    if len(named) == 1:
        return named[0], sorted(candidates), []
    if len(candidates) == 1:
        return candidates[0], candidates, []
    return None, sorted(candidates), [
        "ambiguous Main lowered artifacts; pass --main-build-dir as the "
        "final compiler output directory: "
        + ", ".join(str(path) for path in sorted(candidates))
    ]


def _path_key(path: Path) -> str:
    """Return a stable path key without requiring the path to exist."""
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        key = _path_key(resolved)
        if key in seen:
            continue
        seen.add(key)
        result.append(resolved)
    return result


def _discover_mtp_build_dirs(root: Path) -> list[Path]:
    """Discover real MTP lowered roots below an explicit directory.

    The compiler output is not required to be laid out as ``mtp_0/mtp_1/...``.
    Recent builds put each generated program below a directory named
    ``MtpLayerHidden_*``.  When the explicitly supplied path is itself one of
    those directories, it is also a valid root.  A direct lowered root remains
    supported as a compatibility fallback, but an empty directory is never
    reported as a successful artifact.
    """
    root = root.expanduser().resolve()
    if not root.is_dir():
        return []

    candidates: list[Path] = []
    if root.name.startswith("MtpLayerHidden_"):
        candidates.append(root)
    candidates.extend(
        path
        for path in root.rglob("MtpLayerHidden_*")
        if path.is_dir()
    )
    # Current direct compile callers may choose ``mtp_0``/``mtp_1``/``mtp_2``
    # as output_dir while the generated program name is only present inside
    # the artifact metadata.  Include those real compiler roots too.
    candidates.extend(
        path
        for path in root.rglob("mtp_*")
        if path.is_dir() and _is_lowered_root(path)
    )
    candidates = _dedupe_paths(candidates)
    candidates = [
        path for path in candidates
        if _is_lowered_root(path)
    ]
    if candidates:
        return sorted(candidates, key=lambda path: (str(path).lower(), str(path)))

    # Some compile callers pass the final output directory directly and the
    # compiler does not preserve the program name in the directory name.
    # Accept it only when it contains lowered source/artifact files.
    return [root] if _is_lowered_root(root) else []


def _discover_deps_files(
    explicit_paths: Iterable[Path],
    *,
    artifact_roots: Iterable[Path],
) -> list[Path]:
    """Resolve explicit deps files/directories and recursively discover deps.json.

    A task-DAG result is valid only when a real ``deps.json`` is found.  Build
    metadata or source text must not be silently substituted for the runtime
    resource DAG.
    """
    explicit_paths = list(explicit_paths)
    candidates: list[Path] = []
    for raw_path in explicit_paths:
        path = raw_path.expanduser().resolve()
        if path.is_file():
            if path.name == "deps.json":
                candidates.append(path)
            continue
        if path.is_dir():
            candidates.extend(
                item
                for item in path.rglob("deps.json")
                if item.is_file()
            )

    # An invalid/missing explicit --deps must not silently fall through to an
    # unrelated historical deps.json below a build directory.
    if not candidates and not explicit_paths:
        for root in artifact_roots:
            root = root.expanduser().resolve()
            if root.is_dir():
                candidates.extend(
                    item
                    for item in root.rglob("deps.json")
                    if item.is_file()
                )
    return sorted(_dedupe_paths(candidates), key=lambda path: (str(path).lower(), str(path)))


def _artifact_root_for_deps(
    deps_path: Path,
    artifact_roots: Iterable[Path],
) -> Path:
    """Choose the lowered root whose kernel catalog explains one deps DAG."""
    deps_path = deps_path.resolve()
    artifact_roots = [
        root.resolve()
        for root in artifact_roots
        if _is_lowered_root(root.resolve())
    ]
    containing = [
        root
        for root in artifact_roots
        if root == deps_path.parent or root in deps_path.parents
    ]
    if containing:
        return max(containing, key=lambda path: len(path.parts))

    # Device DFX output is commonly outside the compile directory (for
    # example ``/tmp/c1-liveness/dfx/deps.json``).  In that case select the
    # unique lowered root that actually contains C1 wait kernels.  Never pick
    # an arbitrary MTP/historical build when more than one catalog matches.
    wait_roots = [
        root
        for root in artifact_roots
        if any(_is_wait_kernel(entry) for entry in _kernel_catalog(root))
    ]
    if len(wait_roots) == 1:
        return wait_roots[0]
    if len(artifact_roots) == 1:
        return artifact_roots[0]
    return deps_path.parent


def _signal_contexts(text: str, radius: int = 220) -> list[str]:
    contexts: list[str] = []
    lower = text.lower()
    for match in re.finditer("|".join(re.escape(item) for item in _SIGNAL_WORDS), lower):
        start = max(0, match.start() - radius)
        end = min(len(text), match.end() + radius)
        contexts.append(text[start:end])
    return contexts


_CANONICAL_SIGNAL_STACK_SYMBOLS = {
    "ext_dense_attn_signal_stack",
    "ext_dense_mlp_signal_stack",
    "ext_moe_attn_signal_stack",
    "ext_moe_meta_arrived_stack",
    "ext_moe_data_arrived_stack",
    "ext_moe_sh_signal_stack",
    "ext_moe_combine_arrived_stack",
}
_MTP_SIGNAL_WINDOW_SYMBOLS = {
    "ext_eh_signal_window",
    "ext_attn_signal_window",
    "ext_mlp_signal_window",
}
_ORCH_TENSOR_ARG_RE = re.compile(
    r"\b(?P<params>[A-Za-z_][A-Za-z0-9_]*)"
    r"\.add_(?P<kind>input|inout|output)\s*"
    r"\(\s*(?P<value>[A-Za-z_][A-Za-z0-9_]*)\s*\)\s*;"
)
_PTO_VIEW_RE_TEMPLATE = (
    r"pto\.make_tensor_view\s+%arg{arg}\s*,\s*"
    r"shape\s*=\s*\[(?P<shape>[^\]]+)\]\s*,\s*"
    r"strides\s*=\s*\[(?P<strides>[^\]]+)\]"
)


def _cpp_kernel_catalog(files: Iterable[Path]) -> dict[int, dict[str, Any]]:
    """Build the func-id to generated kernel/PTO catalog for one artifact."""
    result: dict[int, dict[str, Any]] = {}
    for config in files:
        if config.name != "kernel_config.py":
            continue
        text = _read_text(config)
        for match in re.finditer(
            r'\{\s*"func_id"\s*:\s*(?P<id>-?\d+)\s*,\s*'
            r'"name"\s*:\s*"(?P<name>[^"]+)"',
            text,
        ):
            func_id = int(match.group("id"))
            name = match.group("name")
            pto_path = config.parent / "ptoas" / f"{name}.pto"
            source_paths = sorted(
                config.parent.glob(f"kernels/*/{name}.cpp")
            )
            result[func_id] = {
                "func_id": func_id,
                "name": name,
                "config": str(config),
                "pto_path": str(pto_path),
                "pto_text": _read_text(pto_path) if pto_path.is_file() else "",
                "source_paths": [str(path) for path in source_paths],
            }
    return result


def _orchestration_tasks(text: str) -> list[dict[str, Any]]:
    """Parse generated L0 task tensor arguments and concrete func ids."""
    tasks: list[dict[str, Any]] = []
    declarations = list(
        re.finditer(
            r"\bL0TaskArgs\s+(?P<params>[A-Za-z_][A-Za-z0-9_]*)\s*;",
            text,
        )
    )
    for index, declaration in enumerate(declarations):
        params = declaration.group("params")
        end = (
            declarations[index + 1].start()
            if index + 1 < len(declarations)
            else len(text)
        )
        block = text[declaration.end():end]
        submit = re.search(
            rf"\brt_submit_(?:aiv|aic)_task\s*"
            rf"\(\s*(?P<func_id>-?\d+)\s*,\s*{re.escape(params)}\s*\)\s*;",
            block,
        )
        if submit is None:
            continue
        tensor_args = [
            {
                "index": arg_index,
                "kind": match.group("kind"),
                "value": match.group("value"),
            }
            for arg_index, match in enumerate(
                item
                for item in _ORCH_TENSOR_ARG_RE.finditer(
                    block[:submit.start()]
                )
                if item.group("params") == params
            )
        ]
        tasks.append(
            {
                "params": params,
                "func_id": int(submit.group("func_id")),
                "tensor_args": tensor_args,
            }
        )
    return tasks


def _cpp_symbol_views(
    text: str,
    *,
    symbol: str,
    rows: int,
) -> list[dict[str, Any]]:
    """Resolve exact host-side ``symbol.view([rows, 1])`` variables."""
    escaped = re.escape(symbol)
    views: list[dict[str, Any]] = []
    for match in re.finditer(
        rf"\bTensor\s+(?P<view>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
        rf"{escaped}\s*\.\s*view\s*"
        rf"\(\s*(?P<shape>[A-Za-z_][A-Za-z0-9_]*)\s*,",
        text,
        flags=re.IGNORECASE,
    ):
        shape_var = match.group("shape")
        declarations = list(
            re.finditer(
                rf"\buint32_t\s+{re.escape(shape_var)}\s*"
                rf"\[\s*2\s*\]\s*=\s*\{{(?P<body>.*?)\}}\s*;",
                text[:match.start()],
                flags=re.IGNORECASE | re.DOTALL,
            )
        )
        shape_body = declarations[-1].group("body") if declarations else ""
        row_min = bool(
            re.search(
                rf"std::min\s*<\s*uint32_t\s*>\s*"
                rf"\(\s*{int(rows)}\s*,\s*{escaped}\s*\.\s*shapes"
                rf"\s*\[\s*0\s*\]",
                shape_body,
                flags=re.IGNORECASE,
            )
        )
        col_min = bool(
            re.search(
                rf"std::min\s*<\s*uint32_t\s*>\s*"
                rf"\(\s*1\s*,\s*{escaped}\s*\.\s*shapes"
                rf"\s*\[\s*1\s*\]",
                shape_body,
                flags=re.IGNORECASE,
            )
        )
        views.append(
            {
                "view": match.group("view"),
                "shape_var": shape_var,
                "row_min": row_min,
                "col_min": col_min,
                "host_shape": [rows, 1] if row_min and col_min else None,
                "pass": row_min and col_min,
            }
        )
    return views


def _pto_const_value(token: str) -> int | None:
    """Resolve a generated PTO constant token such as ``%c128_index``."""
    token = token.strip()
    match = re.fullmatch(r"%c(?P<value>-?\d+)(?:_[A-Za-z0-9_]+)?", token)
    if match:
        return int(match.group("value"))
    if re.fullmatch(r"-?\d+", token):
        return int(token)
    return None


def _pto_arg_shape_stride(
    kernel: dict[str, Any] | None,
    *,
    arg_index: int,
    rows: int,
    forbidden_rows: int | None,
) -> dict[str, Any]:
    """Inspect one exact kernel formal, never a neighbouring resource."""
    if kernel is None:
        return {
            "available": False,
            "arg_index": arg_index,
            "matches": [],
            "expected_shape_stride": False,
            "forbidden_shape_stride": False,
            "pass": False,
        }
    text = str(kernel.get("pto_text", ""))
    matches: list[dict[str, Any]] = []
    for match in re.finditer(
        _PTO_VIEW_RE_TEMPLATE.format(arg=int(arg_index)),
        text,
        flags=re.IGNORECASE,
    ):
        shape = [
            _pto_const_value(token)
            for token in match.group("shape").split(",")
        ]
        strides = [
            _pto_const_value(token)
            for token in match.group("strides").split(",")
        ]
        matches.append(
            {
                "shape": shape,
                "strides": strides,
                "expected": shape == [rows, 1] and strides == [1, rows],
                "forbidden": (
                    forbidden_rows is not None
                    and shape == [forbidden_rows, 1]
                    and strides == [1, forbidden_rows]
                ),
            }
        )
    expected = any(item["expected"] for item in matches)
    forbidden = any(item["forbidden"] for item in matches)
    return {
        "available": bool(text),
        "kernel": kernel.get("name"),
        "func_id": kernel.get("func_id"),
        "pto_path": kernel.get("pto_path"),
        "arg_index": arg_index,
        "matches": matches,
        "expected_shape_stride": expected,
        "forbidden_shape_stride": forbidden,
        "pass": expected and not forbidden,
    }


def _cpp_signal_evidence(
    files: Iterable[Path],
    *,
    rows: int,
    canonical: bool,
) -> dict[str, Any]:
    """Prove each external symbol's exact host-view and kernel-formal ABI.

    Evidence is accepted only through this chain:

    ``ext symbol -> Tensor view -> task tensor argument index -> func_id ->
    kernel PTO %argN make_tensor_view``.

    A Shape/Stride belonging to an unrelated signal or model tensor cannot
    satisfy another external symbol.
    """
    files = list(files)
    cpp_files = [
        path
        for path in files
        if path.suffix.lower() in {".cpp", ".cc", ".c", ".h", ".hpp"}
    ]
    orchestration_files = [
        path
        for path in cpp_files
        if "orchestration" in {part.lower() for part in path.parts}
    ]
    catalog = _cpp_kernel_catalog(files)
    required_symbols = (
        _CANONICAL_SIGNAL_STACK_SYMBOLS
        if canonical
        else _MTP_SIGNAL_WINDOW_SYMBOLS
    )
    symbol_evidence: dict[str, dict[str, Any]] = {}
    for symbol in sorted(required_symbols):
        host_rows: list[dict[str, Any]] = []
        chains: list[dict[str, Any]] = []
        for path in orchestration_files:
            text = _read_text(path)
            views = _cpp_symbol_views(text, symbol=symbol, rows=rows)
            if not canonical and re.search(
                rf"\bconst\s+Tensor&\s+{re.escape(symbol)}\s*=",
                text,
                flags=re.IGNORECASE,
            ):
                # MTP signal windows are independent compact external tensors;
                # unlike canonical stacked slots they are passed directly and
                # do not require a host-side per-layer .view().
                views.append(
                    {
                        "view": symbol,
                        "shape_var": None,
                        "row_min": None,
                        "col_min": None,
                        "host_shape": None,
                        "source": "direct_external_tensor",
                        "pass": True,
                    }
                )
            tasks = _orchestration_tasks(text)
            for view in views:
                host_rows.append({"file": str(path), **view})
                for task in tasks:
                    for arg in task["tensor_args"]:
                        if arg["value"] != view["view"]:
                            continue
                        kernel = catalog.get(task["func_id"])
                        formal = _pto_arg_shape_stride(
                            kernel,
                            arg_index=arg["index"],
                            rows=rows,
                            forbidden_rows=(
                                None if canonical else COMM_SIGNAL_ROWS
                            ),
                        )
                        chains.append(
                            {
                                "file": str(path),
                                "view": view["view"],
                                "host_shape_pass": view["pass"],
                                "params": task["params"],
                                "tensor_arg_index": arg["index"],
                                "tensor_arg_kind": arg["kind"],
                                "func_id": task["func_id"],
                                "kernel": (
                                    kernel.get("name")
                                    if kernel is not None
                                    else None
                                ),
                                "formal": formal,
                                "pass": view["pass"] and formal["pass"],
                            }
                        )
        host_pass = bool(host_rows) and any(
            item["pass"] for item in host_rows
        )
        chain_pass = bool(chains) and any(item["pass"] for item in chains)
        forbidden_chains = [
            item
            for item in chains
            if item["formal"]["forbidden_shape_stride"]
        ]
        symbol_evidence[symbol] = {
            "host_views": host_rows,
            "host_shape_view_pass": host_pass,
            "task_kernel_chains": chains,
            "task_kernel_chain_pass": chain_pass,
            "forbidden_stride_128_chains": forbidden_chains,
            "pass": host_pass and chain_pass and not forbidden_chains,
        }
    missing_required_symbols = sorted(
        symbol
        for symbol, evidence in symbol_evidence.items()
        if not evidence["host_views"]
    )
    mtp_any_wide_signal_stride = bool(
        not canonical
        and any(
            item["formal"]["forbidden_shape_stride"]
            for evidence in symbol_evidence.values()
            for item in evidence["task_kernel_chains"]
        )
    )
    passed = (
        bool(orchestration_files)
        and bool(catalog)
        and not missing_required_symbols
        and all(item["pass"] for item in symbol_evidence.values())
        and not mtp_any_wide_signal_stride
    )
    return {
        "cpp_files": len(cpp_files),
        "orchestration_files": [str(path) for path in orchestration_files],
        "kernel_catalog_size": len(catalog),
        "required_signal_symbols": sorted(required_symbols),
        "missing_required_signal_symbols": missing_required_symbols,
        "symbol_evidence": symbol_evidence,
        "mtp_any_wide_signal_stride": mtp_any_wide_signal_stride,
        "pass": passed,
    }


def _artifact_shape_contract(root: Path, *, canonical: bool) -> dict[str, Any]:
    if not _is_lowered_root(root):
        return _missing_artifact_contract(
            root,
            kind="canonical" if canonical else "mtp",
            reason=(
                "explicit path is not a compiler lowered artifact root; "
                "expected kernel_config.py/distributed_meta.json or "
                "passes_dump/ptoas/kernels/orchestration with lowered files"
            ),
        )
    files = _artifact_files(root)
    checks: list[dict[str, Any]] = []
    all_text = ""
    evidence_files: list[str] = []
    for path in files:
        try:
            text = _read_text(path)
        except OSError:
            continue
        all_text += "\n" + text
    contexts = _signal_contexts(all_text)
    normal_contexts = [_normalise_whitespace(item) for item in contexts]
    cpp_evidence = _cpp_signal_evidence(
        files,
        rows=COMM_SIGNAL_ROWS if canonical else MTP_SIGNAL_ROWS,
        canonical=canonical,
    )
    provenance = _artifact_provenance_contract(
        root,
        canonical=canonical,
    )
    if canonical:
        wide_hits = sum(
            len(re.findall(r"\[\s*128\s*,\s*1\s*\]", item))
            for item in normal_contexts
        )
        compact_conflict_hits = sum(
            len(re.findall(r"\[\s*8\s*,\s*1\s*\]", item))
            for item in normal_contexts
        )
        wide_named_hits = sum(
            1
            for item in normal_contexts
            if re.search(r"\[\s*128\s*,\s*1\s*\]", item)
        )
        for path in files:
            text = _read_text(path)
            if re.search(r"\[\s*128\s*,\s*1\s*\]", text) and any(
                word in text.lower() for word in _SIGNAL_WORDS
            ):
                evidence_files.append(str(path))
        legacy_pass = wide_hits >= 2 and wide_named_hits >= 1
        # If generated C++ is present, it is the authoritative lowered
        # evidence.  Do not let an unrelated .pto/[128,1] string hide a
        # missing ext_moe_* Shape/Stride/min bound in the real C++ artifact.
        cpp_authoritative = bool(cpp_evidence["cpp_files"])
        passed = cpp_evidence["pass"] if cpp_authoritative else legacy_pass
        cpp_pass = cpp_evidence["pass"]
        checks.append(
            {
                "name": "canonical_lowered_signal_shape_128x1",
                "pass": cpp_pass if cpp_authoritative else legacy_pass,
                "detail": (
                    f"signal-context literal [128,1] hits={wide_hits}, "
                    f"conflicting [8,1] hits={compact_conflict_hits}, "
                    f"files={len(set(evidence_files))}; "
                    f"cpp_authoritative={cpp_authoritative}; "
                    "unrelated legacy text is ignored when exact symbol chains "
                    "are available"
                ),
            }
        )
        checks.append(
            {
                "name": "canonical_lowered_cpp_signal_shape_stride",
                "pass": cpp_evidence["pass"],
                "detail": (
                    "requires ext_moe_*_(signal|sig)_stack + exact "
                    "std::min<uint32_t>(128, ext_moe_*_stack.shapes[...]) "
                    "+ exact task argument -> kernel PTO %argN "
                    "shape=[128,1], strides=[1,128]; "
                    f"symbols={len(cpp_evidence['symbol_evidence'])}, "
                    f"catalog={cpp_evidence['kernel_catalog_size']}"
                ),
            }
        )
    else:
        compact_hits = sum(
            len(re.findall(r"\[\s*8\s*,\s*1\s*\]", item))
            for item in normal_contexts
        )
        wide_hits = sum(
            len(re.findall(r"\[\s*128\s*,\s*1\s*\]", item))
            for item in normal_contexts
        )
        for path in files:
            text = _read_text(path)
            if re.search(r"\[\s*8\s*,\s*1\s*\]", text) and any(
                word in text.lower() for word in _SIGNAL_WORDS
            ):
                evidence_files.append(str(path))
        legacy_pass = compact_hits >= 1 and wide_hits == 0
        cpp_authoritative = bool(cpp_evidence["cpp_files"])
        passed = cpp_evidence["pass"] if cpp_authoritative else legacy_pass
        cpp_pass = cpp_evidence["pass"]
        checks.append(
            {
                "name": "mtp_lowered_signal_shape_8x1",
                "pass": cpp_pass if cpp_authoritative else legacy_pass,
                "detail": (
                    f"signal-context literal [8,1] hits={compact_hits}, "
                    f"canonical-width hits={wide_hits}, files={len(set(evidence_files))}; "
                    f"cpp_authoritative={cpp_authoritative}; "
                    "only exact symbol-to-formal chains decide when C++ exists"
                ),
            }
        )
        checks.append(
            {
                "name": "mtp_lowered_cpp_signal_shape_stride",
                "pass": cpp_evidence["pass"],
                "detail": (
                    "requires ext_{eh,attn,mlp}_signal_window + "
                    "exact task argument -> kernel PTO %argN "
                    "shape=[8,1], strides=[1,8], with no exact stride 128; "
                    f"symbols={len(cpp_evidence['symbol_evidence'])}, "
                    f"wide_conflict={cpp_evidence['mtp_any_wide_signal_stride']}"
                ),
            }
        )
    return {
        "path": str(root),
        "files_scanned": len(files),
        "evidence_files": sorted(set(evidence_files))[:40],
        "cpp_evidence": cpp_evidence,
        "provenance": provenance,
        "pass": (
            passed
            and all(item["pass"] for item in checks)
            and provenance["pass"]
        ),
        "checks": checks,
    }


def _write_synthetic_signal_artifact(
    root: Path,
    *,
    canonical: bool,
    add_mtp_stride_128_conflict: bool = False,
    add_legacy_wide_conflict: bool = False,
) -> None:
    """Create the smallest lowered-artifact fixture understood by this probe."""
    root.mkdir(parents=True, exist_ok=True)
    orchestration_dir = root / "orchestration"
    pto_dir = root / "ptoas"
    orchestration_dir.mkdir()
    pto_dir.mkdir()
    symbols = (
        sorted(_CANONICAL_SIGNAL_STACK_SYMBOLS)
        if canonical
        else sorted(_MTP_SIGNAL_WINDOW_SYMBOLS)
    )
    rows = COMM_SIGNAL_ROWS if canonical else MTP_SIGNAL_ROWS
    orchestration_lines = [
        "// synthetic orchestration",
        "void synthetic_entry(const L2TaskArgs& orch_args) {",
    ]
    config_rows: list[str] = []
    next_func_id = 1
    for index, symbol in enumerate(symbols):
        orchestration_lines.append(
            f"  const Tensor& {symbol} = orch_args.tensor({index}).ref();"
        )
        value = symbol
        if canonical:
            value = f"signal_view_{index}"
            orchestration_lines.extend(
                [
                    f"  uint32_t signal_offsets_{index}[2] = {{0, 0}};",
                    (
                        f"  uint32_t signal_shapes_{index}[2] = {{"
                        f"std::min<uint32_t>({rows}, {symbol}.shapes[0]), "
                        f"std::min<uint32_t>(1, {symbol}.shapes[1])}};"
                    ),
                    (
                        f"  Tensor {value} = {symbol}.view("
                        f"signal_shapes_{index}, signal_offsets_{index});"
                    ),
                ]
            )
        params = f"params_t{index}"
        kernel = f"signal_kernel_{index}"
        orchestration_lines.extend(
            [
                f"  L0TaskArgs {params};",
                f"  {params}.add_input({value});",
                f"  rt_submit_aiv_task({next_func_id}, {params});",
            ]
        )
        config_rows.append(
            f'{{"func_id": {next_func_id}, "name": "{kernel}"}}'
        )
        (pto_dir / f"{kernel}.pto").write_text(
            (
                "module {\n"
                f"  func.func @{kernel}(%arg0: !pto.ptr<i32>) {{\n"
                "    %signal = pto.make_tensor_view %arg0, "
                f"shape = [%c{rows}_index, %c1_index], "
                f"strides = [%c1_index, %c{rows}_index] "
                "{layout = #pto.layout<dn>}: "
                "!pto.tensor_view<?x?xi32>\n"
                "  }\n"
                "}\n"
            ),
            encoding="utf-8",
        )
        next_func_id += 1

    if add_mtp_stride_128_conflict:
        if canonical:
            raise ValueError("MTP stride conflict is only valid for MTP fixtures")
        symbol = sorted(_MTP_SIGNAL_WINDOW_SYMBOLS)[0]
        params = "params_t_mtp_conflict"
        kernel = "signal_kernel_mtp_conflict"
        orchestration_lines.extend(
            [
                f"  L0TaskArgs {params};",
                f"  {params}.add_input({symbol});",
                f"  rt_submit_aiv_task({next_func_id}, {params});",
            ]
        )
        config_rows.append(
            f'{{"func_id": {next_func_id}, "name": "{kernel}"}}'
        )
        (pto_dir / f"{kernel}.pto").write_text(
            (
                "module {\n"
                f"  func.func @{kernel}(%arg0: !pto.ptr<i32>) {{\n"
                "    %signal = pto.make_tensor_view %arg0, "
                "shape = [%c128_index, %c1_index], "
                "strides = [%c1_index, %c128_index] "
                "{layout = #pto.layout<dn>}: "
                "!pto.tensor_view<?x?xi32>\n"
                "  }\n"
                "}\n"
            ),
            encoding="utf-8",
        )

    orchestration_lines.append("}")
    (orchestration_dir / "synthetic.cpp").write_text(
        "\n".join(orchestration_lines) + "\n",
        encoding="utf-8",
    )
    (root / "kernel_config.py").write_text(
        "KERNELS = [\n  " + ",\n  ".join(config_rows) + "\n]\n",
        encoding="utf-8",
    )
    if add_legacy_wide_conflict:
        (root / "legacy_signal.pto").write_text(
            "// legacy signal metadata: [128, 1]\n",
            encoding="utf-8",
        )
    source_path = _CANONICAL_SOURCE if canonical else _MTP_SOURCE
    now = time.time()
    _write_build_manifest(
        root,
        source_path=source_path,
        program="synthetic_canonical" if canonical else "synthetic_mtp",
        compile_started_at=now,
        compile_finished_at=max(now, source_path.stat().st_mtime),
    )


def _write_synthetic_task_dag_artifact(root: Path) -> Path:
    """Create two independent wait->consumer token RAW chains."""
    root.mkdir(parents=True, exist_ok=True)
    kernel_dir = root / "kernels" / "aiv"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernels = {
        101: (
            "wait_dispatch_ready",
            (
                "// Unpack tensor: count_done_sig\n"
                "// Unpack tensor: dispatch_active_token\n"
            ),
        ),
        102: (
            "wait_combine_ready",
            (
                "// Unpack tensor: combine_done_sig\n"
                "// Unpack tensor: combine_active_token\n"
            ),
        ),
        201: ("dispatch_consumer", "// Unpack tensor: dispatch_active_token\n"),
        202: ("combine_consumer", "// Unpack tensor: combine_active_token\n"),
    }
    config_rows: list[str] = []
    for func_id, (name, source) in kernels.items():
        config_rows.append(f'{{"func_id": {func_id}, "name": "{name}"}}')
        (kernel_dir / f"{name}.cpp").write_text(source, encoding="utf-8")
    (root / "kernel_config.py").write_text(
        "KERNELS = [\n  " + ",\n  ".join(config_rows) + "\n]\n",
        encoding="utf-8",
    )
    deps = {
        "tasks": [
            {
                "task_id": 1,
                "name": "wait_dispatch_ready",
                "kernel_ids": [101],
                "args": [
                    {
                        "idx": 0,
                        "type": "INPUT_EXISTING",
                        "dtype": "INT32",
                        "shape": [COMM_SIGNAL_ROWS, 1],
                        "tensor_id": 301,
                    },
                    {
                        "idx": 1,
                        "type": "OUTPUT",
                        "dtype": "INT32",
                        "shape": [COMM_SIGNAL_ROWS],
                        "tensor_id": 401,
                    },
                ],
            },
            {
                "task_id": 2,
                "name": "wait_combine_ready",
                "kernel_ids": [102],
                "args": [
                    {
                        "idx": 0,
                        "type": "INPUT_EXISTING",
                        "dtype": "INT32",
                        "shape": [COMM_SIGNAL_ROWS, 1],
                        "tensor_id": 302,
                    },
                    {
                        "idx": 1,
                        "type": "OUTPUT",
                        "dtype": "INT32",
                        "shape": [COMM_SIGNAL_ROWS],
                        "tensor_id": 402,
                    },
                ],
            },
            {
                "task_id": 3,
                "name": "dispatch_consumer",
                "kernel_ids": [201],
                "args": [
                    {
                        "idx": 0,
                        "type": "INPUT",
                        "dtype": "INT32",
                        "shape": [COMM_SIGNAL_ROWS],
                        "tensor_id": 401,
                    }
                ],
            },
            {
                "task_id": 4,
                "name": "combine_consumer",
                "kernel_ids": [202],
                "args": [
                    {
                        "idx": 2,
                        "type": "INOUT",
                        "dtype": "INT32",
                        "shape": [COMM_SIGNAL_ROWS],
                        "tensor_id": 402,
                    }
                ],
            },
        ],
        "tensors": [
            {"tensor_id": tensor_id}
            for tensor_id in (301, 302, 401, 402)
        ],
        "edges": [
            {
                "pred": 1,
                "succ": 3,
                "arg": 0,
                "tensor_id": 401,
                "source": "creator",
                "consumer_dtype": "INT32",
                "consumer_shape": [COMM_SIGNAL_ROWS],
            },
            {
                "pred": 2,
                "succ": 4,
                "arg": 2,
                "tensor_id": 402,
                "source": "tensormap",
                "consumer_dtype": "INT32",
                "consumer_shape": [COMM_SIGNAL_ROWS],
            },
        ],
    }
    deps_path = root / "deps.json"
    deps_path.write_text(
        json.dumps(deps, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return deps_path


def _mutate_synthetic_deps(
    source: Path,
    target: Path,
    mutation: str,
) -> Path:
    data = json.loads(_read_text(source))
    if mutation == "missing_second_wait_raw":
        data["edges"] = [data["edges"][0]]
    elif mutation == "wrong_successor_arg":
        data["edges"][1]["arg"] = 0
    elif mutation == "wrong_consumer_dtype":
        data["tasks"][3]["args"][0]["dtype"] = "FLOAT32"
    elif mutation == "wrong_consumer_shape":
        data["tasks"][3]["args"][0]["shape"] = [MTP_SIGNAL_ROWS]
    elif mutation == "wait_owns_ep_data":
        data["tasks"][0]["args"].append(
            {
                "idx": 2,
                "type": "INPUT",
                "dtype": "BFLOAT16",
                "shape": [16, 4096],
                "tensor_id": 501,
            }
        )
        data["tensors"].append({"tensor_id": 501})
        wait_source = target.parent / "kernels" / "aiv" / "wait_dispatch_ready.cpp"
        wait_source.write_text(
            _read_text(wait_source) + "// Unpack tensor: recv_x\n",
            encoding="utf-8",
        )
    else:
        raise ValueError(f"unknown synthetic deps mutation {mutation!r}")
    target.write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


def _synthetic_contract_report() -> dict[str, Any]:
    """Exercise positive and fail-closed C1 artifact contracts."""
    with tempfile.TemporaryDirectory(prefix="pypto-c1-inspector-") as raw_dir:
        base = Path(raw_dir)
        canonical_ok = base / "canonical_ok"
        mtp_ok = base / "mtp_ok"
        mtp_mixed = base / "mtp_mixed_stride"
        mtp_legacy_conflict = base / "mtp_legacy_conflict"
        dag_root = base / "task_dag"
        _write_synthetic_signal_artifact(canonical_ok, canonical=True)
        _write_synthetic_signal_artifact(mtp_ok, canonical=False)
        _write_synthetic_signal_artifact(
            mtp_mixed,
            canonical=False,
            add_mtp_stride_128_conflict=True,
        )
        _write_synthetic_signal_artifact(
            mtp_legacy_conflict,
            canonical=False,
            add_legacy_wide_conflict=True,
        )
        dag_path = _write_synthetic_task_dag_artifact(dag_root)

        canonical_report = _artifact_shape_contract(
            canonical_ok,
            canonical=True,
        )
        mtp_report = _artifact_shape_contract(mtp_ok, canonical=False)
        mixed_report = _artifact_shape_contract(
            mtp_mixed,
            canonical=False,
        )
        legacy_report = _artifact_shape_contract(
            mtp_legacy_conflict,
            canonical=False,
        )
        missing_report = _artifact_shape_contract(
            base / "missing",
            canonical=True,
        )
        dag_positive = _wait_task_report(
            dag_path,
            artifact_root=dag_root,
        )
        dag_negative_reports: dict[str, dict[str, Any]] = {}
        for mutation in (
            "missing_second_wait_raw",
            "wrong_successor_arg",
            "wrong_consumer_dtype",
            "wrong_consumer_shape",
            "wait_owns_ep_data",
        ):
            mutation_root = base / f"dag_{mutation}"
            mutation_path = _write_synthetic_task_dag_artifact(mutation_root)
            mutation_path = _mutate_synthetic_deps(
                mutation_path,
                mutation_path,
                mutation,
            )
            dag_negative_reports[mutation] = _wait_task_report(
                mutation_path,
                artifact_root=mutation_root,
            )

        canonical_source = _read_text(_CANONICAL_SOURCE)
        signal_scope = _canonical_signal_scope_contract(canonical_source)
        generic_stride_source = (
            canonical_source
            + "\nrogue_signal_buf = "
            "pld.alloc_window_buffer(COMM_CONTROL_SIGNAL_BYTES)\n"
        )
        generic_stride_scope = _canonical_signal_scope_contract(
            generic_stride_source
        )
        stale_manifest = canonical_ok / C1_MANIFEST_NAME
        stale_data = json.loads(_read_text(stale_manifest))
        stale_data["source_sha256"] = "0" * 64
        stale_manifest.write_text(
            json.dumps(stale_data, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        stale_report = _artifact_shape_contract(
            canonical_ok,
            canonical=True,
        )
        cases = [
            {
                "name": "canonical_six_stacked_reused_symbols_positive",
                "expected": True,
                "actual": canonical_report["pass"],
            },
            {
                "name": "mtp_three_symbols_stride8_positive",
                "expected": True,
                "actual": mtp_report["pass"],
            },
            {
                "name": "mtp_stride8_and_stride128_negative",
                "expected": False,
                "actual": mixed_report["pass"],
                "wide_conflict": mixed_report.get(
                    "cpp_evidence", {}
                ).get("mtp_any_wide_signal_stride"),
            },
            {
                "name": "unrelated_legacy_text_does_not_override_cpp",
                "expected": True,
                "actual": legacy_report["pass"],
            },
            {
                "name": "missing_artifact_fail_closed",
                "expected": False,
                "actual": missing_report["pass"],
            },
            {
                "name": "task_dag_two_waits_per_task_raw_positive",
                "expected": True,
                "actual": dag_positive["pass"],
            },
            *[
                {
                    "name": f"task_dag_{mutation}_negative",
                    "expected": False,
                    "actual": report["pass"],
                    "failures": report["failures"],
                }
                for mutation, report in dag_negative_reports.items()
            ],
            {
                "name": "stacked_reused_control_signal_scope_positive",
                "expected": True,
                "actual": signal_scope["pass"],
            },
            {
                "name": "generic_512b_signal_backing_negative",
                "expected": False,
                "actual": generic_stride_scope["pass"],
                "unexpected": generic_stride_scope[
                    "unexpected_stride_backings"
                ],
            },
            {
                "name": "stale_source_manifest_negative",
                "expected": False,
                "actual": stale_report["pass"],
            },
        ]
        passed = all(
            bool(case["actual"]) is bool(case["expected"])
            for case in cases
        )
        return {
            "kind": "PERF-C1-inspector-synthetic-contracts",
            "status": "PASS" if passed else "NO-GO",
            "pass": passed,
            "cases": cases,
        }


def _compile_program(
    program: Any,
    *,
    output_dir: Path,
    platform: str,
    devices: list[int],
    skip_ptoas: bool,
    source_path: Path,
) -> dict[str, Any]:
    """Compile one distributed program and preserve the output path."""
    _ensure_repo_on_path()
    from pypto import ir  # noqa: PLC0415
    from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415
    from pypto.ir.distributed_compiled_program import (  # noqa: PLC0415
        DistributedConfig,
    )
    from pypto.ir.pass_manager import PassDumpLevel  # noqa: PLC0415

    set_backend_type(BackendType.Ascend910B)
    output_dir.mkdir(parents=True, exist_ok=True)
    compile_started_at = time.time()
    compiled = ir.compile(
        program,
        output_dir=str(output_dir),
        platform=platform,
        distributed_config=DistributedConfig(
            device_ids=devices,
            num_sub_workers=0,
        ),
        skip_ptoas=skip_ptoas,
        dump_passes=PassDumpLevel.EXPLICIT,
    )
    compile_finished_at = time.time()
    manifest_path = _write_build_manifest(
        Path(compiled.output_dir),
        source_path=source_path,
        program=getattr(program, "name", repr(program)),
        compile_started_at=compile_started_at,
        compile_finished_at=compile_finished_at,
    )
    return {
        "output_dir": str(compiled.output_dir),
        "program": getattr(program, "name", repr(program)),
        "pass_dump_dir": str(compiled.output_dir / "passes_dump"),
        "build_manifest": str(manifest_path),
    }


def compile_canonical_and_mtp(
    *,
    main_build_dir: Path,
    mtp_build_dir: Path,
    platform: str,
    devices: list[int],
    skip_ptoas: bool,
    compile_mtp: bool = True,
) -> dict[str, Any]:
    """Compile canonical Main and all selected MTP variants."""
    _ensure_repo_on_path()
    from models.step3p5.decode_fwd import whole_decode_step3p5  # noqa: PLC0415

    result: dict[str, Any] = {
        "main": _compile_program(
            whole_decode_step3p5,
            output_dir=main_build_dir,
            platform=platform,
            devices=devices,
            skip_ptoas=skip_ptoas,
            source_path=_CANONICAL_SOURCE,
        ),
        "mtp": [],
    }
    if compile_mtp:
        from models.step3p5.mtp_hidden_fwd import (  # noqa: PLC0415
            MTP_LAYER_HIDDEN_PROGRAMS,
        )

        for layer_idx, program in enumerate(MTP_LAYER_HIDDEN_PROGRAMS):
            result["mtp"].append(
                _compile_program(
                    program,
                    output_dir=mtp_build_dir / f"mtp_{layer_idx}",
                    platform=platform,
                    devices=devices,
                    skip_ptoas=skip_ptoas,
                    source_path=_MTP_SOURCE,
                )
            )
    return result


def _normalise_id(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _shape_from_arg(arg: dict[str, Any]) -> list[int] | None:
    raw = arg.get("shape")
    if not isinstance(raw, list):
        return None
    result: list[int] = []
    for value in raw:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            return None
    return result


def _arg_type(arg: dict[str, Any]) -> str:
    return str(arg.get("type", "")).strip().upper()


def _is_output_arg(arg: dict[str, Any]) -> bool:
    return _arg_type(arg) in {"OUTPUT", "OUTPUT_EXISTING", "INOUT"}


def _is_input_arg(arg: dict[str, Any]) -> bool:
    return _arg_type(arg) in {
        "INPUT",
        "INPUT_EXISTING",
        "INOUT",
    }


def _kernel_catalog(root: Path) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for config_path in root.rglob("kernel_config.py"):
        text = _read_text(config_path)
        entries = re.findall(
            r'\{\s*"func_id"\s*:\s*(-?\d+)\s*,\s*"name"\s*:\s*"([^"]+)"'
            r'(?P<tail>[^}]*)\}',
            text,
        )
        for func_id_text, name, _tail in entries:
            func_id = int(func_id_text)
            sources = list(
                config_path.parent.glob(f"kernels/*/{name}.cpp")
            )
            catalog.append(
                {
                    "config": str(config_path),
                    "func_id": func_id,
                    "name": name,
                    "source_paths": [str(path) for path in sources],
                    "source": "\n".join(_read_text(path) for path in sources),
                }
            )
    return catalog


def _is_wait_kernel(entry: dict[str, Any]) -> bool:
    haystack = f"{entry.get('name', '')}\n{entry.get('source', '')}".lower()
    return any(word in haystack for word in _WAIT_WORDS)


def _tensor_unpack_names(source: str) -> list[str]:
    return re.findall(r"//\s*Unpack tensor:\s*([A-Za-z_][A-Za-z0-9_]*)", source)


def _is_large_ep_data_arg(arg: dict[str, Any]) -> bool:
    shape = _shape_from_arg(arg)
    dtype = str(arg.get("dtype", "")).upper()
    if not shape:
        return False
    elements = 1
    for dim in shape:
        elements *= max(1, dim)
    # Wait tasks may carry an implementation-private pipe workspace.  Do not
    # reject an opaque 1-D workspace; reject the recognizable EP payloads.
    if len(shape) >= 2 and dtype in {"INT8", "BFLOAT16", "FLOAT16"}:
        return shape[-1] >= 1024 or elements >= 16 * 4096
    if len(shape) >= 2 and dtype in {"INT32", "FLOAT32"}:
        return elements > COMM_SIGNAL_ROWS * 2 and shape != [COMM_SIGNAL_ROWS, 1]
    return False


def _wait_task_report(
    deps_path: Path,
    *,
    artifact_root: Path,
) -> dict[str, Any]:
    """Inspect wait-task resources and token RAW edges in one deps.json."""
    data = json.loads(_read_text(deps_path))
    tasks = data.get("tasks", [])
    edges = data.get("edges", [])
    tensors = data.get("tensors", [])
    if (
        not isinstance(tasks, list)
        or not isinstance(edges, list)
        or not isinstance(tensors, list)
    ):
        raise ValueError(
            "deps.json must contain list-valued tasks, tensors, and edges"
        )
    if not tasks or not edges:
        return {
            "path": str(deps_path),
            "artifact_root": str(artifact_root),
            "tasks": len(tasks) if isinstance(tasks, list) else 0,
            "tensors": len(tensors) if isinstance(tensors, list) else 0,
            "edges": len(edges) if isinstance(edges, list) else 0,
            "wait_tasks": [],
            "unknown_wait_tasks": [],
            "token_raw_edges": [],
            "checks": {
                "deps_schema_complete": False,
                "wait_task_resource_set_excludes_ep_data": False,
                "token_raw_edge_present": False,
            },
            "pass": False,
            "failures": [
                "deps.json has no task/resource DAG: "
                f"tasks={len(tasks)}, tensors={len(tensors)}, edges={len(edges)}"
            ],
        }

    catalog = _kernel_catalog(artifact_root)
    dag_failures: list[str] = []
    if not catalog:
        dag_failures.append(
            "no lowered kernel catalog was found for deps.json at "
            f"artifact_root={artifact_root}"
        )
    by_func_id: dict[int, list[dict[str, Any]]] = {}
    for entry in catalog:
        by_func_id.setdefault(int(entry["func_id"]), []).append(entry)

    wait_tasks: list[dict[str, Any]] = []
    unknown_wait_tasks: list[dict[str, Any]] = []
    for raw_task in tasks:
        if not isinstance(raw_task, dict):
            continue
        task_id = _normalise_id(raw_task.get("task_id"))
        kernel_ids = raw_task.get("kernel_ids", [])
        if task_id is None:
            continue
        candidate_entries: list[dict[str, Any]] = []
        if isinstance(kernel_ids, list):
            for func_id in kernel_ids:
                normalized = _normalise_id(func_id)
                if normalized is not None:
                    candidate_entries.extend(by_func_id.get(normalized, []))
        names = [str(entry["name"]) for entry in candidate_entries]
        is_wait = any(_is_wait_kernel(entry) for entry in candidate_entries)
        if not is_wait:
            # Some older deps schemas carry a human-readable function name.
            task_name = str(raw_task.get("name", ""))
            is_wait = any(word in task_name.lower() for word in _WAIT_WORDS)
            if is_wait and not candidate_entries:
                unknown_wait_tasks.append(
                    {"task_id": task_id, "name": task_name}
                )
        if is_wait:
            wait_tasks.append(
                {
                    "task_id": task_id,
                    "task": raw_task,
                    "kernel_names": names,
                    "entries": candidate_entries,
                }
            )
    if not wait_tasks:
        dag_failures.append(
            "deps.json contains no task that resolves to a lowered C1 "
            "wait kernel; task-DAG evidence is unavailable"
        )

    resource_rows: list[dict[str, Any]] = []
    raw_failures: list[str] = []
    tensor_ids = {
        tensor_id
        for raw_tensor in tensors
        if isinstance(raw_tensor, dict)
        for tensor_id in [_normalise_id(raw_tensor.get("tensor_id"))]
        if tensor_id is not None
    }
    task_by_id = {
        task_id: raw_task
        for raw_task in tasks
        if isinstance(raw_task, dict)
        for task_id in [_normalise_id(raw_task.get("task_id"))]
        if task_id is not None
    }
    for item in wait_tasks:
        task = item["task"]
        args = task.get("args", [])
        if not isinstance(args, list):
            args = []
        shapes = [
            {
                "idx": arg.get("idx"),
                "type": arg.get("type"),
                "dtype": str(arg.get("dtype", "")).upper(),
                "shape": _shape_from_arg(arg),
                "tensor_id": _normalise_id(arg.get("tensor_id")),
            }
            for arg in args
            if isinstance(arg, dict)
        ]
        has_signal = any(
            row["dtype"] == "INT32"
            and row["shape"] == [COMM_SIGNAL_ROWS, 1]
            for row in shapes
        )
        token_rows = [
            row
            for row in shapes
            if row["dtype"] == "INT32"
            and row["shape"] == [COMM_SIGNAL_ROWS]
            and _is_output_arg(row)
        ]
        token_output_ids = {
            row["tensor_id"]
            for row in token_rows
            if row["tensor_id"] is not None
        }
        missing_token_tensors = sorted(token_output_ids - tensor_ids)
        bad_shape_rows = [
            row for row in shapes if _is_large_ep_data_arg(
                {
                    "dtype": row["dtype"],
                    "shape": row["shape"],
                }
            )
        ]
        source_data_names: set[str] = set()
        resource_names: set[str] = set()
        for entry in item["entries"]:
            source = str(entry.get("source", ""))
            names = _tensor_unpack_names(source)
            resource_names.update(names)
            source_data_names.update(
                name for name in names
                if any(word in name.lower() for word in _EP_DATA_WORDS)
            )
        # A valid wait task must be identifiable from the lowered resource
        # set, not only from a kernel name.  The source-side unpack list is
        # supplementary evidence; if it is absent in an old artifact, the
        # shape/resource rows still have to prove control-only ownership.
        unknown_large_args = [
            row for row in shapes
            if row["shape"] is not None and _is_large_ep_data_arg(
                {"dtype": row["dtype"], "shape": row["shape"]}
            )
        ]
        resource_pass = (
            has_signal
            and bool(token_rows)
            and bool(resource_names or shapes)
            and not bad_shape_rows
            and not source_data_names
            and not unknown_large_args
            and not missing_token_tensors
        )
        if not resource_pass:
            raw_failures.append(
                f"task {item['task_id']} resources: signal={has_signal}, "
                f"token={bool(token_rows)}, bad_shapes={bad_shape_rows}, "
                f"data_names={sorted(source_data_names)}, "
                f"unknown_large_args={unknown_large_args}, "
                f"missing_token_tensors={missing_token_tensors}"
            )
        resource_rows.append(
            {
                "task_id": item["task_id"],
                "kernel_names": item["kernel_names"],
                "tensor_args": shapes,
                "resource_names": sorted(resource_names),
                "source_data_names": sorted(source_data_names),
                "token_output_args": token_rows,
                "token_output_tensor_ids": sorted(
                    tensor_id
                    for tensor_id in token_output_ids
                    if tensor_id is not None
                ),
                "missing_token_tensor_ids": missing_token_tensors,
                "resource_pass": resource_pass,
            }
        )

    # Associate token RAW evidence independently with every wait task.  A
    # token produced by wait task A must be the tensor_id of an OUTPUT arg on
    # A and an INPUT/INOUT arg on the concrete successor.  A token from task
    # B must not satisfy task A, even if both tasks use the same [128] shape.
    token_rows_by_task = {
        row["task_id"]: row
        for row in resource_rows
    }
    raw_edges_by_task: dict[int, list[dict[str, Any]]] = {}
    token_failures: list[str] = []
    for raw_edge in edges:
        if not isinstance(raw_edge, dict):
            continue
        pred = _normalise_id(raw_edge.get("pred"))
        tensor_id = _normalise_id(raw_edge.get("tensor_id"))
        if pred not in token_rows_by_task:
            continue
        output_ids = set(
            token_rows_by_task[pred]["token_output_tensor_ids"]
        )
        if tensor_id not in output_ids:
            continue
        succ = _normalise_id(raw_edge.get("succ"))
        edge_arg_idx = _normalise_id(raw_edge.get("arg"))
        succ_task = task_by_id.get(succ)
        succ_args = (
            succ_task.get("args", [])
            if isinstance(succ_task, dict)
            else []
        )
        edge_source = str(raw_edge.get("source", "")).strip().lower()
        edge_consumer_dtype = str(
            raw_edge.get("consumer_dtype", "")
        ).strip().upper()
        edge_consumer_shape = _shape_from_arg(
            {"shape": raw_edge.get("consumer_shape")}
        )
        consumer_token_args = [
            arg
            for arg in succ_args
            if isinstance(arg, dict)
            and edge_arg_idx is not None
            and edge_arg_idx >= 0
            and _normalise_id(arg.get("idx")) == edge_arg_idx
            and _normalise_id(arg.get("tensor_id")) == tensor_id
            and str(arg.get("dtype", "")).strip().upper() == "INT32"
            and _shape_from_arg(arg) == [COMM_SIGNAL_ROWS]
            and _is_input_arg(arg)
        ]
        annotation_pass = (
            edge_source in {"creator", "tensormap"}
            and edge_consumer_dtype == "INT32"
            and edge_consumer_shape == [COMM_SIGNAL_ROWS]
        )
        evidence = {
            "pred": pred,
            "succ": succ,
            "arg": edge_arg_idx,
            "tensor_id": tensor_id,
            "source": edge_source,
            "consumer_dtype": edge_consumer_dtype,
            "consumer_shape": edge_consumer_shape,
            "consumer_token_args": consumer_token_args,
            "annotation_pass": annotation_pass,
        }
        if (
            succ is None
            or tensor_id not in tensor_ids
            or not consumer_token_args
            or not annotation_pass
        ):
            token_failures.append(
                f"wait task {pred} token tensor {tensor_id} has no exact "
                f"creator/tensormap RAW match on successor {succ} "
                f"arg {edge_arg_idx}: "
                f"tensor_known={tensor_id in tensor_ids}, "
                f"consumer_arg={bool(consumer_token_args)}, "
                f"annotation={annotation_pass}"
            )
            raw_edges_by_task.setdefault(pred, []).append(
                {**evidence, "pass": False}
            )
            continue
        raw_edges_by_task.setdefault(pred, []).append(
            {**evidence, "pass": True}
        )

    token_task_rows: list[dict[str, Any]] = []
    for row in resource_rows:
        task_id = int(row["task_id"])
        task_edges = raw_edges_by_task.get(task_id, [])
        valid_edges = [
            edge for edge in task_edges if edge.get("pass") is True
        ]
        token_pass = bool(row["token_output_tensor_ids"]) and bool(valid_edges)
        if not token_pass:
            token_failures.append(
                f"wait task {task_id} has no per-task token OUTPUT -> "
                "successor INPUT/INOUT RAW association"
            )
        token_task_rows.append(
            {
                "task_id": task_id,
                "output_tensor_ids": row["token_output_tensor_ids"],
                "edges": task_edges,
                "pass": token_pass,
            }
        )

    dag_available = (
        bool(tasks)
        and bool(tensors)
        and bool(edges)
        and bool(wait_tasks)
    )
    resource_pass = (
        dag_available
        and not unknown_wait_tasks
        and not raw_failures
        and all(
            bool(row.get("resource_pass"))
            for row in resource_rows
        )
    )
    token_raw_pass = (
        dag_available
        and bool(token_task_rows)
        and all(row["pass"] for row in token_task_rows)
    )
    return {
        "path": str(deps_path),
        "artifact_root": str(artifact_root),
        "tasks": len(tasks),
        "tensors": len(tensors),
        "edges": len(edges),
        "wait_tasks": resource_rows,
        "unknown_wait_tasks": unknown_wait_tasks,
        "token_raw_by_wait_task": token_task_rows,
        "token_raw_edges": [
            edge
            for rows in raw_edges_by_task.values()
            for edge in rows
            if edge.get("pass") is True
        ],
        "checks": {
            "deps_schema_complete": bool(tasks) and bool(tensors) and bool(edges),
            "wait_task_resource_set_excludes_ep_data": resource_pass,
            "token_raw_edge_present": token_raw_pass,
        },
        "pass": (
            bool(tasks)
            and bool(tensors)
            and bool(edges)
            and not unknown_wait_tasks
            and resource_pass
            and token_raw_pass
        ),
        "failures": (
            dag_failures
            + raw_failures
            + token_failures
            + (
                [
                    "wait-like task(s) could not be resolved to lowered "
                    f"kernel catalog entries: {unknown_wait_tasks}"
                ]
                if unknown_wait_tasks
                else []
            )
        ),
    }


def inspect(
    *,
    main_build_dir: Path | None,
    mtp_build_dirs: Iterable[Path],
    deps_path: Path | None,
    deps_paths: Iterable[Path] = (),
    require_task_dag: bool,
    expect_mtp: bool = True,
) -> dict[str, Any]:
    """Run source, lowered artifact, and optional DAG checks."""
    mtp_build_dirs = list(mtp_build_dirs)
    explicit_deps = list(deps_paths)
    if deps_path is not None:
        explicit_deps.append(deps_path)
    artifact_roots = [
        path for path in [main_build_dir, *mtp_build_dirs]
        if path is not None
    ]
    report: dict[str, Any] = {
        "kind": "PERF-C1-lowered-IR-inspection",
        "source": {
            "canonical": _source_contract(_CANONICAL_SOURCE, canonical=True),
            "mtp": _source_contract(_MTP_SOURCE, canonical=False),
        },
        "lowered": {
            "discovery_roots": [str(path) for path in artifact_roots],
        },
        "task_dag": [],
    }
    if main_build_dir is not None:
        report["lowered"]["main"] = _artifact_shape_contract(
            main_build_dir, canonical=True
        )
    else:
        report["lowered"]["main"] = _missing_artifact_contract(
            None,
            kind="canonical",
            reason="no lowered Main artifact was discovered",
        )
    if mtp_build_dirs:
        report["lowered"]["mtp"] = [
            _artifact_shape_contract(path, canonical=False)
            for path in mtp_build_dirs
        ]
    elif expect_mtp:
        report["lowered"]["mtp"] = [
            _missing_artifact_contract(
                None,
                kind="mtp",
                reason="no lowered MTP MtpLayerHidden_* artifact was discovered",
            )
        ]
    else:
        report["lowered"]["mtp"] = []

    discovered_deps = _discover_deps_files(
        explicit_deps,
        artifact_roots=artifact_roots,
    )
    if discovered_deps:
        for discovered in discovered_deps:
            report["task_dag"].append(
                _wait_task_report(
                    discovered,
                    artifact_root=_artifact_root_for_deps(
                        discovered,
                        artifact_roots,
                    ),
                )
            )
    else:
        report["task_dag"] = [
            {
                "path": "",
                "artifact_root": str(main_build_dir or ""),
                "tasks": 0,
                "tensors": 0,
                "edges": 0,
                "pass": False,
                "checks": {
                    "deps_schema_complete": False,
                    "wait_task_resource_set_excludes_ep_data": False,
                    "token_raw_edge_present": False,
                },
                "failures": [
                    "NO-GO: no runtime deps.json/resource DAG was supplied "
                    "or discovered; source/lowered contracts cannot prove "
                    "task liveness or resource ownership"
                ],
            }
        ]

    source_pass = bool(report["source"]["canonical"]["pass"]) and bool(
        report["source"]["mtp"]["pass"]
    )
    lowered_pass = (
        bool(report["lowered"].get("main", {}).get("pass", False))
        and (
            not expect_mtp
            or bool(report["lowered"].get("mtp"))
            and all(
                bool(item.get("pass", False))
                for item in report["lowered"].get("mtp", [])
            )
        )
    )
    dag_pass = bool(report["task_dag"]) and all(
        bool(item.get("pass", False)) for item in report["task_dag"]
    )
    report["pass"] = source_pass and lowered_pass and dag_pass
    report["status"] = "PASS" if report["pass"] else "NO-GO"
    report["gates"] = {
        "source_contract": "PASS" if source_pass else "NO-GO",
        "lowered_artifact": "PASS" if lowered_pass else "NO-GO",
        "deps_resource_dag": "PASS" if dag_pass else "NO-GO",
        "require_task_dag": bool(require_task_dag),
    }
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile and inspect PERF-C1 canonical/MTP lowering and task DAG."
    )
    parser.add_argument("--platform", default="a2a3sim", choices=("a2a3sim", "a2a3"))
    parser.add_argument(
        "--devices",
        default="0,1,2,3,4,5,6,7",
        help="eight device ids used by distributed compile",
    )
    parser.add_argument("--build-dir", default="/tmp/step3p5-c1-lowered-ir")
    parser.add_argument("--main-build-dir", default="")
    parser.add_argument("--mtp-build-dir", default="")
    parser.add_argument("--deps", default="", help="device-captured deps.json")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument("--skip-ptoas", action="store_true")
    parser.add_argument("--require-task-dag", action="store_true")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run synthetic positive/negative inspector contracts only",
    )
    parser.add_argument("--report", default="", help="optional JSON report path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.self_test:
        report = _synthetic_contract_report()
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 0 if report["pass"] else 1
    _ensure_repo_on_path()
    devices = _parse_devices(args.devices)
    build_root = Path(args.build_dir).resolve()
    main_input_dir = (
        Path(args.main_build_dir).resolve()
        if args.main_build_dir
        else build_root / "main"
    )
    mtp_input_dir = (
        Path(args.mtp_build_dir).resolve()
        if args.mtp_build_dir
        else build_root / "mtp"
    )

    compile_report: dict[str, Any] | None = None
    discovery_errors: list[str] = []
    if not args.no_compile:
        compile_report = compile_canonical_and_mtp(
            main_build_dir=main_input_dir,
            mtp_build_dir=mtp_input_dir,
            platform=args.platform,
            devices=devices,
            skip_ptoas=args.skip_ptoas,
            compile_mtp=not args.no_mtp,
        )
    main_build_dir, main_candidates, main_errors = _discover_main_build_dir(
        main_input_dir
    )
    discovery_errors.extend(main_errors)
    mtp_build_dirs = (
        _discover_mtp_build_dirs(mtp_input_dir)
        if not args.no_mtp
        else []
    )
    if not args.no_mtp and not mtp_build_dirs:
        discovery_errors.append(
            "no lowered MTP MtpLayerHidden_* artifact found below "
            f"explicit directory: {mtp_input_dir}"
        )
    artifact_roots = [
        path for path in [main_build_dir, *mtp_build_dirs]
        if path is not None
    ]
    deps_candidates = _discover_deps_files(
        [Path(args.deps).resolve()] if args.deps else [],
        artifact_roots=artifact_roots,
    )
    report = inspect(
        main_build_dir=main_build_dir,
        mtp_build_dirs=mtp_build_dirs,
        deps_path=None,
        deps_paths=deps_candidates,
        require_task_dag=args.require_task_dag,
        expect_mtp=not args.no_mtp,
    )
    report["discovery"] = {
        "main_input": str(main_input_dir),
        "main_candidates": [str(path) for path in main_candidates],
        "mtp_input": str(mtp_input_dir),
        "mtp_candidates": [str(path) for path in mtp_build_dirs],
        "deps_candidates": [str(path) for path in deps_candidates],
        "errors": discovery_errors,
    }
    if discovery_errors:
        report["pass"] = False
        report["status"] = "NO-GO"
    if compile_report is not None:
        report["compile"] = compile_report
    print(json.dumps(_jsonable(report), indent=2, sort_keys=True), flush=True)
    if args.report:
        report_path = Path(args.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(_jsonable(report), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"C1_IR_REPORT={report_path}", flush=True)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
