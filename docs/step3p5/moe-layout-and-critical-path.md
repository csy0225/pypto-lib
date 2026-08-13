# Step3p5 MoE layout and critical-path optimization record

This document records the reusable conclusions from the Step3p5 decode MoE
router and shared-expert optimization admitted in August 2026. It is both a
change record and the source material for a future model-development skill.

The change targets pure decode with a fixed storage capacity and a runtime
active-row count. It does not add a BS1 GEMV path. Cube kernels keep their
minimum 16-row tile while task grids and row-wise vector work scale with the
runtime batch.

## 1. Problem statement

Two five-layer L2 swimlane symptoms motivated the work:

1. `swa_moe_chip_orch_gate_expert_fanout` took about 33 us for BS1.
2. `swa_moe_chip_orch_sh_gate_up_act` took about 87 us and visually extended
   across the routed-expert schedule.

The first issue was a real router critical-path cost. The second was a real
local kernel cost, but it was initially misclassified as a whole-layer
critical-path dependency. The distinction matters: reducing a long task does
not reduce ITL when another branch already determines the join time.

## 2. Keep a layout ledger across every boundary

A matrix layout is an end-to-end contract, not a local implementation detail.
Audit all of these boundaries together:

```text
checkpoint -> loader -> resident bundle key -> IPC map -> program annotation
           -> pl.slice shape -> matmul transpose flag -> generated PTO view
```

The admitted decode layout ledger is:

| Weight | Checkpoint layout | Legacy loader layout | Hidden-only decode layout | Matmul |
|---|---|---|---|---|
| Router gate | `[experts, hidden]` | `[hidden, experts]` | `[experts, hidden]` | `b_trans=True` |
| Shared gate | `[inter_local, hidden]` after TP slice | `[hidden, inter_local]` | `[inter_local, hidden]` | `b_trans=True` |
| Shared up | `[inter_local, hidden]` after TP slice | `[hidden, inter_local]` | `[inter_local, hidden]` | `b_trans=True` |
| Shared down | `[inter_local, hidden]` | unchanged | `[inter_local, hidden]` | `b_trans=False` |

The checkpoint-native `[N, K]` layout matches the reference backend's
`MatMulV2(false, true)` contract. It also gives a worker a contiguous K row
instead of retaining a transposed `[K, N]` representation whose K-side access
is strided for this fanout.

### 2.1 Use semantic ABI keys

The hidden-only decode path uses explicit keys:

```text
moe_gate_w_nk
moe_w_gate_s_nk
moe_w_up_s_nk
```

The suffix is part of the ABI: it prevents a tensor with the right element
count but the wrong axis meaning from silently crossing the loader/program
boundary. The default loader remains on the legacy layout for callers that
have not opted into the decode-native contract.

A bundle must contain exactly one representation for each matrix. Do not keep
both `[K, N]` and `[N, K]` copies in a resident weight pool as a compatibility
shortcut. Duplicate layouts increase memory, hide ownership mistakes, and
make it unclear which representation a program actually consumed.

### 2.2 Required layout checks

Before trusting a performance result:

1. Assert the loader key set and exact shapes for native and legacy modes.
2. Assert the native TP slice is contiguous and numerically equals the
   checkpoint slice.
3. Assert native and legacy modes do not emit duplicate layout keys.
4. Inspect the generated `.pto` file. The tensor view and `b_trans` flag are
   authoritative; source annotations alone are not enough.
5. Compare against the reference backend at the operator boundary, not only at
   the model output.

The focused contract lives in
`tests/step3p5/unit/test_weight_loader_native_moe.py`.

## 3. Fixed storage is not fixed work

For decode, the program storage capacity is 16 rows, while the runtime active
batch is:

```text
T = clamp(num_tokens, 0, storage_batch)
M_tiles = ceil(T / 16)
```

Cube still executes an M=16 tile for BS1. The optimization changes task
ownership and avoids work on inactive rows; it does not claim that the cube
executes a physical M=1 GEMM.

The admitted grids are:

| Stage | Runtime grid |
|---|---:|
| Router `x * gamma` precompute | `T` |
| Router expert fanout | `M_tiles * 18` |
| Shared gate MM | `M_tiles * 5` |
| Shared up MM | `M_tiles * 5` |
| Shared activation | `M_tiles * 5` |
| Shared down workers | `M_tiles * 2` |

The multipliers come from model dimensions and the selected output tile or
worker grain. They are not a fixed global core count. For a capacity-32 audit,
BS1--16 uses `18/5/2` workers and BS17--32 uses `36/10/4` workers for router
fanout, shared gate/up/activation, and shared down respectively.

Do not implement dynamic batch support by choosing one fixed task count, such
as 24, for every BS. Instead:

1. derive the number of active M tiles from the runtime row count;
2. multiply by the static N-tile or worker count;
3. decompose the block index with static divisors; and
4. keep inactive rows out of vector, routing, and selection loops.

## 4. Router restructuring

The old fanout rebuilt `x * (gamma + 1)` in every expert-column worker. The
new route is:

```text
active-row xg precompute
  -> checkpoint-native [N, K] cube fanout with b_trans=True
  -> active-row sigmoid, bias, top-k, and normalization
```

The precompute is intentional. It adds one stage but removes repeated FP32
conversion and gamma multiplication from every expert-column worker.

At BS1, the representative five-layer medians were:

| Router component | Before | After |
|---|---:|---:|
| Expert fanout | 32.68 us | 9.21 us |
| Active-row xg precompute | n/a | 5.10 us |
| Top-k | 9.92 us | 12.63 us |

Do not report `32.68 -> 9.21 us` as the layer-level gain. The complete router
critical span, including dependencies and scheduling gaps, changed by only
about 9--10 us per layer in the representative trace (`49.84 -> 40.54 us`).
The new precompute, slower top-k, and the wait for the norm result account for
the difference between the local fanout reduction and the critical-span
reduction.

## 5. Shared gate/up split and the critical-path correction

Shared gate and up are independent matrix products. They now run as separate
cube fanouts, and the vector SwiGLU activation explicitly depends on both:

```text
              +-> shared gate MM -+
post-norm ----|                     +-> activation -> shared down
              +-> shared up MM ---+
```

At BS1, the gate/up/activation path changed from 87.16 us to 11.62 us. This is
a valid local optimization and creates substantial scheduling headroom.
However, the old shared branch was already parallel with the routed branch.
At the actual layer join, the baseline shared-down branch completed before
routed `expert_down`:

| Layer kind | Baseline shared completion relative to routed completion |
|---|---:|
| SWA MoE | 65.5 us earlier |
| Full MoE | 23.9 us earlier |

Therefore most of the 75 us local reduction was hidden and could not be added
once per MoE layer to predict whole-network ITL.

The initial analysis compared shared completion with `dispatch_gather`. That
comparison showed the shared path finishing later than gather, but gather is
not the branch join. Routed gate/up, quantization, and expert down continue
after gather. Always compare both branch endpoints at their first common
consumer or completion barrier.

## 6. Reference-backend comparison

The matching vLLM-Ascend trace used:

```text
checkpoint-native [N, K]
F.linear -> MatMulV2(false, true)
fused MoeGatingTopK
```

Its measured router components were approximately:

| Component | vLLM-Ascend |
|---|---:|
| Matmul | 12.68 us |
| Top-k | 4.84 us |
| Router span | 19.25 us |

After this change, PyPTO fanout is no longer the main gap. PyPTO top-k remains
about 12.63 us, so selection/fusion is a higher-value follow-up than further
BS1 fanout specialization.

## 7. Admission evidence

### 7.1 Focused correctness and build gates

- Five-layer L3/L4 hidden outputs were BF16 byte-exact (`max_abs=0`).
- Capacity-16 and capacity-32 compile gates passed.
- The dynamic-grid audit covered BS1 through BS32.
- The full Step3p5 unit suite passed; the host run reported 376 passed and
  7 environment-gated skips.
- Modified-file Ruff and `git diff --check` passed.

### 7.2 Matched whole-network A/B/A

The authoritative whole-network performance comparison used the same base
source, image, devices, checkpoint, BS1/context-65536 workload, ten warmups,
and 100 measured iterations per arm. Each arm ran in a fresh container.

| Arm | P50 | Mean |
|---|---:|---:|
| Baseline A1 | 29.559 ms | 29.892 ms |
| Candidate B | 29.326 ms | 29.506 ms |
| Baseline A2 | 29.604 ms | 29.785 ms |
| Baseline midpoint | 29.5815 ms | 29.8385 ms |
| Candidate delta | **-0.2555 ms (-0.86%)** | **-0.3325 ms (-1.11%)** |

All three arms produced the same hidden SHA-256 and the same expected tail
token. The baseline P50 half-range was 0.0225 ms, so the candidate improvement
was larger than the measured A/A bracket.

A single absolute run is not an A/B result. In particular, do not compare a
candidate against an older intermediate artifact from a different Git base and
attribute the difference to this change.

## 8. Reusable model-development workflow

This case should be extracted into a model-development skill. Until that skill
exists, use this document as the checklist.

### Inputs the future skill should require

- exact source revision and dirty diff;
- model, BS, context, storage capacity, and active-row contract;
- checkpoint weight shapes and dtypes;
- reference-backend operator trace;
- PyPTO L2 swimlane and dependency graph; and
- correctness oracle and matched performance baseline.

### Workflow

1. **Freeze provenance.** Snapshot parent and candidate sources and record the
   image, checkpoint identity, devices, and workload.
2. **Build a layout ledger.** Track every important matrix from checkpoint to
   generated PTO view, including transpose flags and contiguity.
3. **Separate storage from work.** Derive runtime grids from active rows and
   static tile counts; audit boundary BS values.
4. **Draw the dependency DAG.** Identify the first common consumer of parallel
   branches before calling a task critical.
5. **Measure complete spans.** Include prerequisite stages, gaps, and any
   slower successor introduced by the rewrite.
6. **Use compile-only sweeps first.** Reject illegal tile/layout variants before
   reserving devices.
7. **Gate precision independently.** Local speedups do not waive byte-exact or
   model-level correctness requirements.
8. **Run matched A/B/A.** Use fresh containers and identical inputs, bracket the
   candidate with the same baseline, and report both P50 and mean.
9. **Publish local and model-level results separately.** Never multiply a local
   kernel delta by layer count unless that kernel is proven to remain on the
   global critical path.

### Required outputs

A reusable skill should emit:

- the layout ledger;
- runtime-grid formulas and boundary audit;
- before/after dependency endpoints;
- local kernel and complete critical-span metrics;
- precision results;
- matched whole-network A/B/A results; and
- remaining bottlenecks ranked by critical-path impact.

## 9. Follow-up priorities

1. Reduce or fuse router top-k toward the reference backend's selection cost.
2. Re-profile the complete router-to-dispatch span after each top-k change.
3. Optimize routed-expert and collective stages that actually determine the
   layer join.
4. Treat further shared gate/up work as throughput or larger-BS work unless a
   new trace proves that branch has re-entered the critical path.
