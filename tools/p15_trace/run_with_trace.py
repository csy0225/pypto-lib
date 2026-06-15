"""Phase 15 / bug #3 diagnostic harness — instrument chip_orch with
per-task dispatch traces, then run step3p5 single-card decode.

What this does
--------------
Monkey-patches ``pypto.runtime.device_runner.compile_single_orchestration``
*before* ``ir.compile()`` reaches the orchestration-cpp build step. The patch:

  1. Reads the generated ``chip_orch.cpp`` source.
  2. Inserts a ``fprintf(stderr, ...)``+``fflush(stderr)`` line before every
     ``rt_submit_(aic|aiv)_task(K, params_tK);`` site, tagged with the task
     index ``K``, the AIC/AIV variant, and the kernel name parsed out of the
     adjacent ``// Task K: <name>`` comment (when present).
  3. Optionally truncates dispatch — if env ``P15_DISPATCH_LIMIT`` is set,
     comments out every ``rt_submit_*_task(K, ...)`` whose K exceeds the
     limit. Lets us bisect "which downstream task interferes with the
     pre-rope path".
  4. Writes the patched source back, deletes any pre-cached ``.so`` next to
     it, and invokes the original compiler. Pypto's compile-cache layer
     therefore picks up the patched file.

Usage
-----
    cd /data/chensiyu/hw_project/pypto/workspace/pypto-lib
    python tools/p15_trace/run_with_trace.py            # full e2e w/ trace
    P15_DISPATCH_LIMIT=6 python tools/p15_trace/run_with_trace.py
                                                          # rope-only (cut at task 6)

The chip_process child writes traces to its own stderr, which surfaces in
the parent log. Look for ``[P15_TRACE]`` lines; the last one before the
RuntimeError is the closest dispatch to the fault.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Pypto must be importable; we patch its compile entrypoint *before*
# step3p5_decode triggers ir.compile().
from pypto.runtime import device_runner  # type: ignore[import-not-found]

_RT_SUBMIT_RE = re.compile(
    r"^(?P<lead>\s*)rt_submit_(?P<kind>aic|aiv)_task\(\s*(?P<id>\d+)\s*,\s*(?P<arg>params_t\d+)\s*\)\s*;\s*$"
)
# A "// Task N: kernel_name" comment usually precedes the dispatch (within ~6 lines).
_TASK_COMMENT_RE = re.compile(r"//\s*Task\s+(?P<id>\d+):\s*(?P<name>\S+)")

# Workaround for the chip_orch codegen bug pinned in
# project_p15_fault_is_full_head_gate.md: ``full_head_gate`` is the only
# SPMD task whose ``params_tN.add_*`` block is missing its iteration-
# index scalar. Kernel signature has ``int32_t v4`` and uses it as a
# row-offset multiplier (``v21 = v4 * 16`` → row addressing). With nothing
# dispatched, v4 reads uninitialised garbage from the param block and the
# AICore ends up at a random UB address, surfacing as 507018.
#
# At the model's BATCH=BATCH_TILE=16, ``hg_spmd_idx`` ∈ {0} (single iter),
# so the correct value is the constant 0. Setting env
# ``P15_INJECT_HEAD_GATE_SCALAR=1`` makes the harness insert
# ``params_t<N>.add_scalar(0);`` before the head_gate ``rt_submit_aiv_task``
# line. If that flips the run from FAIL→PASS, the missing dispatch scalar
# IS the root cause of bug #3.


def _instrument_orchestration_source(src: str) -> str:
    """Return ``src`` with one trace line inserted before each rt_submit.

    Also obeys ``P15_DISPATCH_LIMIT`` to comment-out late dispatches for
    bisect.
    """
    lines = src.splitlines(keepends=True)

    # Map task-id -> kernel name from the "// Task N: name" comments
    # that the codegen emits next to each rt_submit. We scan once.
    task_names: dict[int, str] = {}
    for ln in lines:
        m = _TASK_COMMENT_RE.search(ln)
        if m:
            task_names[int(m.group("id"))] = m.group("name")

    limit_env = os.environ.get("P15_DISPATCH_LIMIT")
    limit = int(limit_env) if limit_env is not None else None

    # Pre-pass: locate the head_gate SPMD dispatch by scanning for the
    # ``// Spmd full_head_gate_spmd: full_head_gate`` comment and the
    # ``params_tN`` immediately following it.
    head_gate_task_id: int | None = None
    for i, ln in enumerate(lines):
        if "Spmd full_head_gate_spmd: full_head_gate" in ln:
            for j in range(i + 1, min(i + 12, len(lines))):
                m_inner = re.match(r"\s*params_t(\d+)\.", lines[j])
                if m_inner:
                    head_gate_task_id = int(m_inner.group(1))
                    break
            break
    inject_head_gate = (
        os.environ.get("P15_INJECT_HEAD_GATE_SCALAR", "0") == "1"
        and head_gate_task_id is not None
    )

    out: list[str] = []
    # Inject a sanity-check banner at the very first line of
    # aicpu_orchestration_entry — if THIS one does not surface in the
    # host log, the AICPU log channel itself is the problem and per-task
    # traces are useless.
    inserted_banner = False
    for ln in lines:
        if (
            not inserted_banner
            and ln.lstrip().startswith("void aicpu_orchestration_entry(")
        ):
            out.append(ln)
            out.append(
                '    LOG_WARN("%s", "[P15_TRACE] aicpu_orchestration_entry HIT");\n'
            )
            inserted_banner = True
            continue

        # head_gate scalar injection: every other SPMD dispatch puts
        # ``add_scalar(...)`` BEFORE ``launch_spec.set_block_num`` and the
        # ``Arg`` packer treats launch_spec as a barrier marker. Inject
        # exactly there for params_t<head_gate_task_id> when the workaround
        # env is on.
        if inject_head_gate and head_gate_task_id is not None:
            m_ls = re.match(
                rf"^(\s*)params_t{head_gate_task_id}\.launch_spec\.",
                ln,
            )
            if m_ls:
                lead_ls = m_ls.group(1)
                out.append(
                    f"{lead_ls}// [P15_TRACE injected] missing SPMD iter scalar(s) for full_head_gate\n"
                )
                # Kernel signature has int32_t v4 (used as row-offset
                # multiplier) and int32_t v5 (currently unreferenced in
                # the body). Pass both as 0 so the param block matches the
                # kernel ABI byte-for-byte.
                out.append(f"{lead_ls}params_t{head_gate_task_id}.add_scalar(0);\n")
                out.append(f"{lead_ls}params_t{head_gate_task_id}.add_scalar(0);\n")
                out.append(ln)
                continue

        m = _RT_SUBMIT_RE.match(ln)
        if not m:
            out.append(ln)
            continue
        lead = m.group("lead")
        kind = m.group("kind").upper()
        tid = int(m.group("id"))
        name = task_names.get(tid, "<unknown>")

        # Trace line — AICPU-side LOG_WARN routes through
        # current_runtime()->ops->log_warn into the host's unified log
        # channel (which surfaces in the chip_process child's stderr).
        # Plain `fprintf(stderr, …)` does NOT work here: chip_orch.cpp runs
        # on the on-chip AICPU, whose libc stderr is not wired to host fd 2.
        trace = (
            f'{lead}LOG_WARN("[P15_TRACE] dispatch task=%d kind={kind} '
            f'name={name}", {tid});\n'
        )
        out.append(trace)

        if limit is not None and tid > limit:
            # Comment out the dispatch; record which task got dropped.
            out.append(f"{lead}// [P15_TRACE_DROPPED tid={tid}] {ln.lstrip()}")
            out.append(
                f'{lead}LOG_WARN("[P15_TRACE] DROPPED task=%d (limit={limit})", {tid});\n'
            )
        else:
            out.append(ln)

    return "".join(out)


def install_patch() -> None:
    """Wrap ``compile_single_orchestration`` with the trace-injecting patch."""
    if getattr(device_runner.compile_single_orchestration, "_p15_patched", False):
        return  # idempotent

    original = device_runner.compile_single_orchestration

    def patched(source, compiler, runtime_name, cache_dir=None):
        source_path = Path(source)
        if source_path.name == "chip_orch.cpp":
            text = source_path.read_text()
            patched_text = _instrument_orchestration_source(text)
            source_path.write_text(patched_text)
            # Force a fresh build: drop any pre-cached .so / cache_dir entry.
            so = source_path.with_suffix(".so")
            if so.exists():
                so.unlink()
            if cache_dir is not None:
                cached_bin = Path(cache_dir) / f"orch_{source_path.stem}.bin"
                if cached_bin.exists():
                    cached_bin.unlink()
            limit = os.environ.get("P15_DISPATCH_LIMIT", "<none>")
            sys.stderr.write(
                f"[P15_TRACE] instrumented {source_path} "
                f"(P15_DISPATCH_LIMIT={limit})\n"
            )
            sys.stderr.flush()
        return original(source, compiler, runtime_name, cache_dir=cache_dir)

    patched._p15_patched = True  # type: ignore[attr-defined]
    device_runner.compile_single_orchestration = patched
    sys.stderr.write("[P15_TRACE] compile_single_orchestration patched\n")
    sys.stderr.flush()


def main() -> int:
    install_patch()
    # Forward to step3p5_decode entry point.
    from models.step3p5 import step3p5_decode  # type: ignore[import-not-found]
    # ``--tp-world-size 1`` activates the TP=1 monkey-patch in
    # ``run_real_npu`` — config is reloaded with TP_WORLD_SIZE=1 so
    # ``tp_all_reduce`` becomes a no-op (no peers to sum with). Without
    # this flag the canonical TP=8 codegen runs on a single card and
    # tp_all_reduce expects 7 peer ranks → MPU fault on missing
    # signal-window pointers.
    argv = [
        "step3p5_decode", "-p", "a2a3", "-d", "0",
        "--tp-world-size", "1",
        "--no-smoke", "--dummy-weights",
    ]
    return step3p5_decode.main(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
