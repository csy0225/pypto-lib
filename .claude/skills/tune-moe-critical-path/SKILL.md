---
name: tune-moe-critical-path
description: Audit and optimize PyPTO MoE decode critical paths, including checkpoint-to-kernel layouts, active-batch task grids, physical reduction carriers, dependency DAGs, router and TopK preprocessing, dispatch quantization ownership, and matched DFX or whole-network validation. Use when a MoE swimlane is slow, a task appears incorrectly serialized, a layout or transpose change is proposed, BS-dependent work must replace a fixed task count, or a local optimization must be proven at model level.
---

# Tune MoE Critical Paths

Optimize the dependency path rather than the longest visible task. Preserve
layout, active-row, reduction, quantization, and precision contracts while
changing task ownership.

## 1. Freeze the comparison

Record before editing:

- source revision and dirty diff;
- image digest, checkpoint identity, devices, and runtime versions;
- model layers, active BS, storage capacity, context, warmup, and iterations;
- correctness oracle; and
- baseline generated dependencies and L2 swimlanes for every rank.

Use fresh source snapshots for A/B/A. Never compare artifacts from different
Git bases or workloads.

## 2. Build a layout ledger

Trace each matrix through the complete boundary chain:

```text
checkpoint -> loader -> semantic bundle key -> IPC map -> annotation
           -> slice shape -> transpose flag -> generated PTO tensor view
```

Record shape, dtype, contiguity, and axis meaning at every boundary. Treat a
semantic key such as an `_nk` suffix as an ABI. Avoid duplicate transposed
copies as a compatibility shortcut. Inspect generated PTO views; source
annotations alone are not authoritative.

## 3. Separate storage, logical work, and physical alignment

Clamp the runtime row count first:

```text
T = clamp(num_tokens, 0, storage_batch)
M_tiles = ceil(T / cube_m_tile)
```

Retain the cube's legal minimum M tile, but scale its grid with
`M_tiles * static_n_fragments`. For row-wise work, assign logical ownership
from `T`, for example one worker per token or a measured grid-stride plan.
Never use one fixed count such as 24 for every BS.

Treat aligned scratch shapes independently from logical rows. An `[8, K]`
carrier can make a single-row FP32 reduction legal without representing eight
tokens. Load and transform only the active row, then pad the reduction result
without changing the validated addition tree.

Audit `T=0`, `T=1`, every tile boundary, and full capacity. Do not add a BS1
GEMV path unless profiling proves that a separate implementation is required.

## 4. Draw the real dependency DAG

Read generated dependency edges and physical swimlanes. For each task, record:

- explicit and inferred predecessors;
- earliest predecessor completion;
- physical start and finish;
- queue delay after readiness; and
- the first common consumer of parallel branches.

Call a task globally critical only when its completion moves that join. Compare
shared and routed branches at their common consumer, not at an intermediate
stage such as dispatch gather.

## 5. Choose ownership before choosing fusion

Give token-level statistics and quantized payloads one token owner. Let TopK
routes consume that result; do not recompute amax or quantization once per
route. A `T * TOPK` decomposition can multiply reads, writes, and AIV pressure
while appearing more parallel locally.

Split an early control result from a later payload only after measuring the new
task wave. Check whether the split:

1. lets the early consumer start sooner;
2. adds scheduler queueing or synchronization;
3. competes with routed or shared work for the same execution resource; and
4. improves the complete layer span across all ranks.

Prefer the original ABI and DAG when active-row ownership removes wasted work
without adding a wave. Reject a fusion or split that shortens one producer but
regresses the branch join or whole-network ITL.

## 6. Preserve reduction and quantization arithmetic

Keep the validated FP32 reduction order. For a backend-illegal `[1, 1]` FP32
result, assemble one active row into row zero of an aligned zero carrier and
apply the original K-wide reduction. Do not reshape K into several lanes and
finish with a butterfly unless the complete whole-network precision gate
passes.

Preserve dynamic-quantization operations and association exactly, including:

```text
FP32 -> INT32(rint) -> FP16(round) -> INT8(trunc)
scale = inv_rms * (amax / 127)
```

Inspect generated PTO arithmetic. Distinguish vector tile division from scalar
lowering, and avoid algebraic reassociation that changes FP32 rounding.

## 7. Compare equivalent reference-backend boundaries

Map functions, shapes, dtypes, and fusion boundaries before comparing time. A
reference backend may fuse dynamic quantization into routing initialization or
dispatch, so its RMSNorm task is not equivalent to a producer that also emits
post-norm, INT8 data, and scale. Use the reference to discover layout and
fusion opportunities, not to compare unrelated task names.

## 8. Validate in increasing cost order

Run these gates in order:

1. AST or source contracts for grids, task names, ownership, and dependencies.
2. Whole-decode compilation at default capacity and at least one larger
   capacity.
3. Focused multi-rank execution with byte-exact intermediate outputs.
4. DFX capture for every rank; report service time, complete span, readiness
   delay, and overlap.
5. Matched whole-network A/B/A with fresh containers and identical inputs.
6. Ruff, headers, English-only checks, skill validation, and
   `git diff --check`.

Treat token equality as insufficient when the hidden-state hash differs.
Report local gains separately from router-ready, layer-join, and ITL gains.
Never multiply a local delta by layer count without proving that the stage
remains globally critical.

## 9. Publish the evidence

Publish:

- the layout ledger;
- active-grid formulas and boundary audit;
- logical-row and physical-carrier ownership;
- before/after dependency endpoints;
- rejected variants and the lesson from each;
- local service, router-ready, branch-join, and ITL metrics;
- focused precision and compile results;
- matched whole-network A/B/A results; and
- artifact paths for source snapshots, reports, and representative swimlanes.
