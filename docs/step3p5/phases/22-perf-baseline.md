# Phase 22 — Perf Baseline + Tuning

> **Component pin snapshot (at doc creation, 2026-06-22)**
>
> | Repo | Branch | Pin | Notes |
> |------|--------|-----|-------|
> | pypto-lib | `stepfun/develop` | `a6b5faa` (pre-commit) | Will become the commit that adds this file |
> | pypto | `stepfun/develop` | `b00c8b23` | DFX env hooks: `PYPTO_DISTRIBUTED_DEP_GEN`, `PYPTO_DISTRIBUTED_L2_SWIMLANE` |
> | pto-isa | `stepfun/develop` | `e25732f0` | |
> | PTOAS | `stepfun/develop` | `da011a3d` | binary v0.45 |
> | simpler (submodule) | — | `a6e06406` | |

## Goal

Produce **publishable** decode performance numbers for step3p5 served
through the pypto kernel + vLLM scheduling stack:

1. Single-card token/s, TTFT, ITL across canonical workload mix.
2. Bottleneck identification via PMU + L2 swimlane.
3. Two rounds of optimization (tile sizing, buffer reuse, scheduling).
4. TP=8 multi-card scaling (gated on barrier all_reduce UB fix +
   MoE 507018 fix).
5. vs upstream vLLM Ascend backend / vLLM CUDA / native eager baseline.

**Phase 22 ENTRY criterion**: Phase 21 PASS (precision parity proven).
Running perf without precision is reporting numbers from an unverified
configuration.

## Scope

**In:**
- Benchmark script `bench_vllm_backend_perf.py` with canonical workload
  matrix (input length, output length, batch size).
- DFX trace integration (`PYPTO_DISTRIBUTED_DEP_GEN=1`,
  `PYPTO_DISTRIBUTED_L2_SWIMLANE=1`) via env hook landed in pypto
  `03136bf6`.
- Bottleneck attribution per kernel segment (matmul / attention /
  collective / lm_head).
- Tile-size and chunk-constant tuning passes.
- TP=8 multi-card scaling (gated, see below).
- Comparison report.

**Out:**
- Continuous batching tuning (vLLM scheduler is upstream; not changed
  in this phase).
- Quantization (bf16 only; no fp8 / int8 in this phase).
- KV cache compression / sparsity.

## Hard gates before Phase 22 multi-card section

| Gate | Source | Status as of 2026-06-22 | Doc |
|------|--------|-------------------------|-----|
| Barrier all_reduce UB-friendly rewrite | csy0225/pypto-lib `wip/step3p5-barrier-allreduce-20260622` HEAD `b5bb6ee` (regresses dense ST device with UB overflow) | NOT STARTED; rewrite needed | [`docs/known-pypto-pitfalls.md §7`](../../known-pypto-pitfalls.md) |
| MoE 507018 device runtime fix | Phase 19 milestone | NOT STARTED; needs `P19_DISPATCH_LIMIT` dispatch-cut bisect tool | (no doc yet, sits under TASK-30) |

Single-card Phase 22 work proceeds with mixed-mode MoE (Phase 20 default)
even while these gates are open. Multi-card sections wait.

## Benchmark matrix

| Axis | Values | Notes |
|------|--------|-------|
| Input (prompt) length | 128, 1024, 4096 | covers short-Q&A, mid-context, long-context |
| Output (max_tokens) | 16, 64, 256 | covers short-reply, dialogue turn, long generation |
| Batch size | 1, 4, 16 | single, small, full tile-specialised batch |
| TP world size | 1, 8 | single-card and production |
| MoE mode | mixed, full-pypto | full-pypto gated on MoE 507018 fix |

Metrics per (config) run:
- **TTFT** (time-to-first-token, prefill-dominated; if prefill blocked
  on Phase 17 use synthetic prefill — see "Prefill workaround" below)
- **ITL** (inter-token latency, decode-only, ms/token)
- **TPS** (throughput, tokens/sec across batch)
- **NPU utilization** (AICore%, AIV%, HBM bandwidth%)
- **Per-kernel wallclock breakdown** (from L2 swimlane)

## Prefill workaround (Phase 17 blocked)

Phase 17 prefill MoE L1 overflow (TASK-29) is independently blocking.
For Phase 22 **decode-only** perf, prefill is bypassed:

1. Pre-populate KV cache with synthetic data sized to target input
   length.
2. Set initial `seq_lens = input_length` per batch.
3. Run decode loop for `max_tokens` steps.
4. Report decode-only metrics (TPS, ITL); TTFT marked N/A or reported
   as "synthetic prefill" with a separate measurement.

This is the standard approach when prefill is gated; the decode-side
perf number is what users care about for serving steady-state throughput.

## Deliverables

```
pypto-lib/tests/step3p5/
└── bench_vllm_backend_perf.py    # canonical workload runner with metrics

pypto-lib/docs/step3p5/
├── phases/
│   └── 22-perf-baseline.md       # this doc
└── perf-reports/
    ├── single-card-baseline-<date>.md     # first numbers
    ├── single-card-opt-round1-<date>.md
    ├── single-card-opt-round2-<date>.md
    └── multi-card-tp8-<date>.md           # gated
```

## Tasks

| # | Task | Output | Estimate |
|---|------|--------|----------|
| 3.1 | `bench_vllm_backend_perf.py` — canonical workload runner; sweeps matrix; emits CSV + JSON metrics | bench script | 2 d |
| 3.2 | Single-card dense+mixed-MoE baseline run; first numbers table | `single-card-baseline-<date>.md` | 1 d |
| 3.3 | DFX trace capture: enable env hooks, run 10-step decode, dump swimlane + dep-graph | trace files | 1 d |
| 3.4 | Bottleneck attribution — segment wallclock by kernel: gate_up matmul, fa_fused, out_proj, tp_all_reduce, dispatch, combine, lm_head; identify top-3 | analysis report | 2 d |
| 3.5 | Optimization round 1 — tile-size tuning (`MLP_OUT_CHUNK`, `OUT_PROJ_K_CHUNK`, `KV_OUT_CHUNK`, others); measure delta per change | `single-card-opt-round1-<date>.md` | 1-2 w |
| 3.6 | Optimization round 2 — cross-kernel L2 reuse, PSC (pipeline schedule), cube/vector parallelism balance | `single-card-opt-round2-<date>.md` | 1-2 w |
| 3.7 | (Gated) TP=8 multi-card baseline — barrier all_reduce fix + MoE 507018 fix landed, then run on 8 cards | `multi-card-tp8-<date>.md` | 2-3 w (post-gate) |
| 3.8 | Final report vs upstream vLLM (eager torch) + vs Ascend official vLLM backend if available | comparison doc | 1 w |

## Exit criteria

**Single-card baseline (gate-free):**

`docs/step3p5/perf-reports/single-card-baseline-<date>.md` published with:

- TPS / ITL / TTFT-or-synthetic across `(prompt_len, output_len, batch)` in matrix
- NPU utilization snapshots
- Per-kernel wallclock pie chart
- Comparison vs vLLM Ascend backend if available, else vs upstream torch eager

**Optimization rounds:**

Round 1 and Round 2 reports each show **≥ X% speedup** vs immediate
predecessor (X TBD after round 1 baseline; typical first-round wins
on Ascend kernels are 20-40%).

**Multi-card baseline (gated):**

After barrier all_reduce + MoE 507018 fixes:
`docs/step3p5/perf-reports/multi-card-tp8-<date>.md` with TP=8 numbers,
scaling efficiency vs single-card, and the same matrix.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Multi-card gate (barrier all_reduce + MoE 507018) takes 4-6 weeks to clear | High | Single-card Phase 22 work (tasks 3.1-3.6) proceeds independently |
| Mixed-mode MoE makes single-card numbers unrepresentative (21/45 layers in vLLM eager not pypto) | High | Report results in two columns: total token/s AND dense-layer-only token/s; the latter is the "pypto kernel" number |
| Optimization deltas swamped by measurement noise (first-token compile, page-fault, IO) | Medium | Warm up 16 steps before measure; pin frequency where possible; report median + p95 |
| L2 swimlane dump too large to upload as report artifact | Low | Aggregate at kernel-segment level, summary plots only; raw traces archived on persistent NVMe |
| TP=8 perf scaling poor due to allreduce comm overhead | Medium-High | Phase 22 explicitly measures comm time as a separate metric; informs next-phase compute/comm overlap work |

## Status

- 2026-06-22: design landed (this doc).
- All tasks NOT STARTED. Gated on Phase 21 PASS.
- Multi-card portion additionally gated on barrier all_reduce UB fix + MoE 507018 fix.
- Estimate ~6-8 weeks for full Phase 22 (single-card 3-4 weeks +
  multi-card 3-4 weeks post-gate).

## References

- [`20-vllm-backend-monkey-patch.md`](20-vllm-backend-monkey-patch.md) — Phase 20 (e2e)
- [`21-precision-validation.md`](21-precision-validation.md) — Phase 21 (precision gate)
- [`../../known-pypto-pitfalls.md`](../../known-pypto-pitfalls.md) §7 — `pl.range(constant)` UB overflow (barrier all_reduce gate)
- pypto DFX env hooks: `pypto/python/pypto/runtime/distributed_runner.py:399-405`
- `tools/p15_trace/run_with_trace.py` — existing single-rank trace runner
