"""Minimal compile-only reproducer for hw-native-sys/pypto#1738 (follow-up of #1702 / PR #1718).

Builds a single fanned-out ``pl.spmd`` scope that writes TWO same-shape GM
outputs and returns the SECOND one.  The PR #1718 fix added SSA-base
disambiguation in ``GenerateSingleReturnAlias`` /
``VarLineageCollector::VisitStmt_(AssignStmt)``; the PR commit
``0f4881cb`` explicitly notes that ``tuple`` / ``__ssa_v1`` / ``__phi``
sibling lineage paths are *untouched*.

Run (compile-only, a2a3sim simulator, no NPU device):

    cd workspace/pypto-lib
    PYPTO_PROG_BUILD_DIR=/tmp/repro_1738 \
        python -m models.step3p5._repro_1738_followup -p a2a3sim

Inspect the dumped passes to see which Out arg the orchestration aliases
the returned value to.  In a healthy compile, ``returned``'s downstream
read alias should resolve to the second Out arg (``real_out``); a
mis-alias to the first Out arg (``scratch_out``) reproduces the family
that PR #1718 fixed for one path.

NOT a regression test for #1702 itself — the on-board AICPU 507018 only
triggers through the larger ``jit.inline + tail-appended create_tensor``
path that the dsv4 ``hc_pre + mix_x`` kernel exercises (per PR #1718
commit ``80845cdf``).  This file just shrinks the *codegen-side* shape
of the bug family for triage.
"""
from __future__ import annotations

import argparse
import sys

import pypto.language as pl
from pypto import ir


BATCH = 16
DIM = 128
SCRATCH_COLS = 8           # first  Out: [BATCH, 8]   FP32
NRANKS = 4                 # spmd fan-out


@pl.jit.inline
def dual_out_spmd(
    x: pl.Tensor[[BATCH, DIM], pl.BF16],
    scratch_out: pl.Tensor[[BATCH, SCRATCH_COLS], pl.FP32],
    real_out: pl.Tensor[[BATCH, DIM], pl.BF16],
):
    """Single fanned-out spmd scope writing TWO Out args; return the second.

    The body:
      * computes a small per-row reduction into ``scratch_out``
      * passes ``x`` through into ``real_out`` unchanged

    Returns ``real_out`` (the SECOND Out arg).  In the orchestration
    codegen ``GenerateSingleReturnAlias`` resolves the call's single SSA
    return — the result Var's SSA base must uniquely match ``real_out``
    (not ``scratch_out``) for the alias to be correct; the PR #1718 fix
    ensures this for the canonical pattern, but the commit explicitly
    leaves ``tuple`` / ``__ssa_v1`` / ``__phi`` sibling paths untouched.
    """
    for r in pl.spmd(NRANKS, name_hint="dual_out_repro_spmd"):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="dual_out_repro_inner"):
            x_tile = pl.slice(x, [BATCH, DIM], [0, 0])
            # First Out (scratch): small reduction, [BATCH, SCRATCH_COLS] FP32.
            row_sum = pl.row_sum(x_tile)                            # [BATCH, 1] FP32
            sum_padded = pl.create_tensor([BATCH, SCRATCH_COLS], dtype=pl.FP32)
            sum_padded = pl.assemble(sum_padded, row_sum, [0, 0])
            scratch_out = pl.assemble(scratch_out, sum_padded, [0, 0])
            # Second Out (real): full passthrough, [BATCH, DIM] BF16.
            real_out = pl.assemble(real_out, x_tile, [0, 0])
    return real_out  # SECOND Out arg.


@pl.jit
def repro_root(
    x: pl.Tensor[[BATCH, DIM], pl.BF16],
    scratch_final: pl.Out[pl.Tensor[[BATCH, SCRATCH_COLS], pl.FP32]],
    real_final: pl.Out[pl.Tensor[[BATCH, DIM], pl.BF16]],
):
    """Drive ``dual_out_spmd`` and exercise the lineage tracer downstream.

    The host-side wrapper:
      * allocates intermediate scratch + real tensors (the ``create_tensor``
        scratch tail mirrors what dsv4 ``hc_pre`` does — PR #1718 commit
        ``80845cdf`` calls this out as the trigger context);
      * calls ``dual_out_spmd`` and binds its return value to ``y``;
      * uses ``y`` in a downstream ``pl.assemble`` so
        ``VarLineageCollector::VisitStmt_(AssignStmt)`` (the path patched
        by commit ``0f4881cb``) is exercised.
    """
    scratch_intermediate = pl.create_tensor([BATCH, SCRATCH_COLS], dtype=pl.FP32)
    real_intermediate = pl.create_tensor([BATCH, DIM], dtype=pl.BF16)

    y = dual_out_spmd(x, scratch_intermediate, real_intermediate)

    # Downstream consumer of ``y`` — lineage tracer must resolve y to
    # ``real_intermediate`` (the second Out of dual_out_spmd), NOT to
    # ``scratch_intermediate``.  A mis-resolution would emit a
    # ``Tensor::reshape`` / ``ext_*`` mapping against the small scratch
    # buffer, which is exactly the on-board 507018 trigger described in
    # PR #1718 commit ``80845cdf``.
    real_final = pl.assemble(real_final, y, [0, 0])

    # Also surface scratch through the second Out, so the orchestration
    # has a non-trivial multi-Out signature on the *outer* call too.
    scratch_final = pl.assemble(scratch_final, scratch_intermediate, [0, 0])
    return scratch_final, real_final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", default="a2a3sim",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    args = parser.parse_args()

    print(f"[1738-repro] compiling repro_root on {args.platform}", flush=True)

    compiled = ir.compile(
        repro_root,
        platform=args.platform,
        skip_ptoas=False,
        dump_passes=True,
    )
    print(f"[1738-repro] OK output_dir={compiled.output_dir}", flush=True)
    print(
        "[1738-repro] inspect "
        f"{compiled.output_dir}/passes/  AND "
        f"{compiled.output_dir}/.../orchestration/*.cpp  "
        "for the return-alias decision on the second Out arg of "
        "dual_out_spmd (line: 'real_out' should be aliased; if the "
        "generated chip code aliases 'scratch_out' for the returned "
        "value, the family fault is reproduced).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
