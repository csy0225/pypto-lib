# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""PERF-C1 canonical lowering and task-DAG inspection probe.

This probe is diagnostic-only.  It never changes the canonical model source or
patches compiler/runtime state.  It can:

* compile ``models.step3p5.decode_fwd:whole_decode_step3p5`` with explicit
  pass dumps;
* compile the selected MTP hidden-only variants;
* verify that canonical stacked/reused control-signal views lower to
  ``[128, 1]`` while the independent MTP signal views remain ``[8, 1]``;
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
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable


_ROOT = Path(__file__).resolve().parents[3]
_CANONICAL_SOURCE = _ROOT / "models" / "step3p5" / "decode_fwd.py"
_MTP_SOURCE = _ROOT / "models" / "step3p5" / "mtp_hidden_fwd.py"

COMM_SIGNAL_ROWS = 128
MTP_SIGNAL_ROWS = 8
_SIGNAL_WORDS = (
    "signal",
    "sig",
    "count_done",
    "combine_done",
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


def _source_contract(path: Path, *, canonical: bool) -> dict[str, Any]:
    source = _read_text(path)
    compact = _normalise_whitespace(source)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "pass": bool(passed), "detail": detail})

    if canonical:
        values = _static_ints(source)
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
            "canonical_wait_control_only_source",
            all(
                name in source
                for name in (
                    "_wait_previous_dispatch",
                    "_wait_dispatch_ready",
                    "_wait_previous_combine",
                    "_wait_combine_ready",
                )
            ),
            "all four C1 wait helpers are present",
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

    return {
        "path": str(path),
        "kind": "canonical" if canonical else "mtp",
        "pass": all(item["pass"] for item in checks),
        "checks": checks,
    }


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

    if not candidates:
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
    """Choose the deepest lowered root containing one deps.json."""
    deps_path = deps_path.resolve()
    containing = [
        root.resolve()
        for root in artifact_roots
        if root.resolve() == deps_path.parent
        or root.resolve() in deps_path.parents
    ]
    if containing:
        return max(containing, key=lambda path: len(path.parts))
    return deps_path.parent


def _signal_contexts(text: str, radius: int = 220) -> list[str]:
    contexts: list[str] = []
    lower = text.lower()
    for match in re.finditer("|".join(re.escape(item) for item in _SIGNAL_WORDS), lower):
        start = max(0, match.start() - radius)
        end = min(len(text), match.end() + radius)
        contexts.append(text[start:end])
    return contexts


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
    if canonical:
        wide_hits = sum(
            len(re.findall(r"\[\s*128\s*,\s*1\s*\]", item))
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
        passed = wide_hits >= 2 and wide_named_hits >= 1
        checks.append(
            {
                "name": "canonical_lowered_signal_shape_128x1",
                "pass": passed,
                "detail": (
                    f"signal-context literal [128,1] hits={wide_hits}, "
                    f"files={len(set(evidence_files))}"
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
        passed = compact_hits >= 1 and wide_hits == 0
        checks.append(
            {
                "name": "mtp_lowered_signal_shape_8x1",
                "pass": passed,
                "detail": (
                    f"signal-context literal [8,1] hits={compact_hits}, "
                    f"canonical-width hits={wide_hits}, files={len(set(evidence_files))}"
                ),
            }
        )
    return {
        "path": str(root),
        "files_scanned": len(files),
        "evidence_files": sorted(set(evidence_files))[:40],
        "pass": all(item["pass"] for item in checks),
        "checks": checks,
    }


def _compile_program(
    program: Any,
    *,
    output_dir: Path,
    platform: str,
    devices: list[int],
    skip_ptoas: bool,
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
    return {
        "output_dir": str(compiled.output_dir),
        "program": getattr(program, "name", repr(program)),
        "pass_dump_dir": str(compiled.output_dir / "passes_dump"),
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

    resource_rows: list[dict[str, Any]] = []
    raw_failures: list[str] = []
    token_tensor_ids: set[int] = set()
    for item in wait_tasks:
        task = item["task"]
        args = task.get("args", [])
        if not isinstance(args, list):
            args = []
        shapes = [
            {
                "idx": arg.get("idx"),
                "type": arg.get("type"),
                "dtype": arg.get("dtype"),
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
            if row["dtype"] == "INT32" and row["shape"] == [COMM_SIGNAL_ROWS]
        ]
        for row in token_rows:
            if row["tensor_id"] is not None:
                token_tensor_ids.add(row["tensor_id"])
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
        passed = (
            has_signal
            and bool(token_rows)
            and bool(resource_names or shapes)
            and not bad_shape_rows
            and not source_data_names
            and not unknown_large_args
        )
        if not passed:
            raw_failures.append(
                f"task {item['task_id']} resources: signal={has_signal}, "
                f"token={bool(token_rows)}, bad_shapes={bad_shape_rows}, "
                f"data_names={sorted(source_data_names)}, "
                f"unknown_large_args={unknown_large_args}"
            )
        resource_rows.append(
            {
                "task_id": item["task_id"],
                "kernel_names": item["kernel_names"],
                "tensor_args": shapes,
                "resource_names": sorted(resource_names),
                "source_data_names": sorted(source_data_names),
                "pass": passed,
            }
        )

    wait_task_ids = {item["task_id"] for item in wait_tasks}
    raw_edges: list[dict[str, Any]] = []
    for raw_edge in edges:
        if not isinstance(raw_edge, dict):
            continue
        pred = _normalise_id(raw_edge.get("pred"))
        tensor_id = _normalise_id(raw_edge.get("tensor_id"))
        if pred in wait_task_ids and (
            tensor_id in token_tensor_ids
            or (
                str(raw_edge.get("consumer_dtype", "")).upper() == "INT32"
                and raw_edge.get("consumer_shape") == [COMM_SIGNAL_ROWS]
            )
        ):
            raw_edges.append(
                {
                    "pred": pred,
                    "succ": _normalise_id(raw_edge.get("succ")),
                    "arg": raw_edge.get("arg"),
                    "tensor_id": tensor_id,
                    "source": raw_edge.get("source"),
                    "consumer_shape": raw_edge.get("consumer_shape"),
                }
            )

    dag_available = (
        bool(tasks)
        and bool(tensors)
        and bool(edges)
        and bool(wait_tasks)
    )
    resource_pass = dag_available and not raw_failures
    token_raw_pass = bool(raw_edges)
    return {
        "path": str(deps_path),
        "artifact_root": str(artifact_root),
        "tasks": len(tasks),
        "tensors": len(tensors),
        "edges": len(edges),
        "wait_tasks": resource_rows,
        "unknown_wait_tasks": unknown_wait_tasks,
        "token_raw_edges": raw_edges,
        "checks": {
            "deps_schema_complete": bool(tasks) and bool(tensors) and bool(edges),
            "wait_task_resource_set_excludes_ep_data": resource_pass,
            "token_raw_edge_present": token_raw_pass,
        },
        "pass": (
            bool(tasks)
            and bool(tensors)
            and bool(edges)
            and resource_pass
            and token_raw_pass
        ),
        "failures": raw_failures,
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
    parser.add_argument("--report", default="", help="optional JSON report path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
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
