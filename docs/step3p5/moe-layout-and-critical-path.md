# Step3p5 MoE layout and critical-path optimization record

This document records reusable conclusions from the Step3p5 decode MoE router,
shared-expert, TopK, and input-preprocessing work admitted in August 2026. Use
`.claude/skills/tune-moe-critical-path/SKILL.md` as the executable workflow.

The work targets pure decode with fixed storage capacity and a runtime active
row count. It does not add a BS1 GEMV path. Cube kernels retain their legal
minimum 16-row tile, while task grids and row-wise vector work scale with the
active batch.

## 1. Problem statement

Four five-layer L2 swimlane symptoms motivated the work:

1. `swa_moe_chip_orch_gate_expert_fanout` took about 33 us for BS1.
2. `swa_moe_chip_orch_sh_gate_up_act` took about 87 us and visually extended
   across the routed-expert schedule.
3. Router TopK took about 12.6 us after the fanout layout change.
4. `norm_quant_moe_input` took about 24 us and made TopK wait for a producer
   that also generated the shared-expert and routed-dispatch payloads.

The fanout, TopK, and preprocessing stages affected router readiness. The long
shared-expert task was a real local cost but was initially misclassified as a
whole-layer critical-path dependency. A shorter visible task does not reduce
ITL when another branch determines the first common join.

## 2. Keep a layout ledger across every boundary

A matrix layout is an end-to-end contract, not a local implementation detail.
Audit every boundary together:

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
`MatMulV2(false, true)` contract. It also gives each fanout worker a contiguous
K row instead of a transposed `[K, N]` representation with strided K access.

### 2.1 Use semantic ABI keys

The hidden-only decode path uses explicit keys:

```text
moe_gate_w_nk
moe_w_gate_s_nk
moe_w_up_s_nk
```

The suffix is part of the ABI. It prevents a tensor with the correct element
count but the wrong axis meaning from crossing the loader/program boundary.
The default loader remains on the legacy layout for callers that have not
selected the decode-native contract.

Keep exactly one representation of each matrix in a bundle. Do not retain both
`[K, N]` and `[N, K]` copies as a compatibility shortcut. Duplicate layouts
increase memory, hide ownership mistakes, and obscure which representation the
program consumed.

### 2.2 Required layout checks

Before trusting a performance result:

1. Assert loader keys and exact shapes in native and legacy modes.
2. Assert the native TP slice is contiguous and equals the checkpoint slice.
3. Assert neither mode emits duplicate layout keys.
4. Inspect generated `.pto` files. Tensor views and transpose flags are
   authoritative; source annotations alone are insufficient.
5. Compare against the reference backend at an equivalent operator boundary.

The focused contract lives in
`tests/step3p5/unit/test_weight_loader_native_moe.py`.

## 3. Fixed storage is not fixed work

For decode, storage capacity is static while the active batch is dynamic:

```text
T = clamp(num_tokens, 0, storage_batch)
M_tiles = ceil(T / 16)
```

Cube still executes an M=16 tile for BS1. The optimization changes task
ownership and removes inactive-row vector work; it does not claim that cube
executes a physical M=1 GEMM.

The admitted cube and shared-expert grids are:

| Stage | Runtime grid |
|---|---:|
| Router `x * gamma` precompute | `T` |
| Router expert fanout | `M_tiles * 18` |
| Router TopK | `T` |
| Shared gate MM | `M_tiles * 5` |
| Shared up MM | `M_tiles * 5` |
| Shared activation | `M_tiles * 5` |
| Shared down workers | `M_tiles * 2` |
| MoE norm/quant producer | `T` |

The static multipliers come from model dimensions and the selected tile or
worker grain. They are not a fixed global core count. For capacity 32, BS1--16
uses `18/5/2` workers for router fanout, shared gate/up/activation, and shared
down; BS17--32 uses `36/10/4`.

Do not choose one fixed task count, such as 24, for every BS. Derive active M
tiles or logical rows from `T`, multiply only by static N fragments where
required, and audit zero, one, tile-boundary, and full-capacity batches.

### 3.1 Distinguish logical rows from physical carriers

An aligned reduction carrier is not a logical token tile. In the admitted
norm/quant producer, one active token owns one worker:

```text
pl.spmd(T) -> worker t owns logical row t
```

The backend-safe FP32 reduction carrier has eight lanes. For RMS, the worker
squares one `[1, K_chunk]` active row, assembles it into row zero of a zero
`[8, K_chunk]` carrier, and performs the validated K-wide `row_sum`. For amax,
the same active row is partitioned into eight reduction lanes and then reduced
to lane zero. The other lanes satisfy reduction/store alignment; they do not
cause BS1 to process eight tokens.

Do not remove a physical carrier solely because active BS is one. Remove
inactive logical work while preserving the backend alignment and floating-point
reduction contracts.

## 4. Router fanout restructuring

The old fanout rebuilt `x * (gamma + 1)` in every expert-column worker. The
admitted route is:

```text
active-row xg precompute
  -> checkpoint-native [N, K] cube fanout with b_trans=True
  -> active-row sigmoid, bias, TopK, and normalization
```

The precompute adds one stage but removes repeated FP32 conversion and gamma
multiplication from every expert-column worker.

At BS1, representative five-layer medians were:

| Router component | Before | After |
|---|---:|---:|
| Expert fanout | 32.68 us | 9.21 us |
| Active-row xg precompute | n/a | 5.10 us |
| TopK in that revision | 9.92 us | 12.63 us |

Do not report `32.68 -> 9.21 us` as a layer-level gain. The complete router
critical span, including prerequisites and scheduling gaps, changed by about
9--10 us per layer (`49.84 -> 40.54 us`). The new precompute, slower TopK, and
wait for the norm result explain the smaller end-to-end gain.

### 4.1 Active-row TopK

TopK was subsequently changed from a serial active-row loop to one runtime task
per logical row. Each task sorts exactly the 288-expert domain using a
256-expert head, a 32-expert tail, and a final merge of their TopK candidates.
Scores and selected indices stay task-local instead of materializing padded
512-element score and bias rows in GM.

Representative TopK service time fell from about 12.6 us to 2.5--2.7 us. The
five-layer L3/L4 outputs remained BF16 byte-exact. This is faster than the
approximately 4.84 us `MoeGatingTopK` measured in the matching vLLM-Ascend
trace, so further router work should include readiness and scheduling gaps
rather than targeting selection service time alone.

## 5. Shared gate/up split and critical-path correction

Shared gate and up are independent matrix products. They run as separate cube
fanouts, and the vector SwiGLU activation explicitly depends on both:

```text
              +-> shared gate MM -+
post-norm ----|                     +-> activation -> shared down
              +-> shared up MM ---+
```

At BS1, the gate/up/activation path changed from 87.16 us to 11.62 us. This is a
valid local optimization and creates scheduling headroom. However, the old
shared branch was already parallel with the routed branch. At the actual layer
join, baseline shared down completed before routed `expert_down`:

| Layer kind | Baseline shared completion relative to routed completion |
|---|---:|
| SWA MoE | 65.5 us earlier |
| Full MoE | 23.9 us earlier |

The initial analysis compared shared completion with `dispatch_gather`, but
gather is not the branch join. Routed gate/up, quantization, and expert down
continue afterward. Compare branch endpoints at their first common consumer or
completion barrier.

## 6. Compare equivalent reference-backend stages

The matching vLLM-Ascend route uses:

```text
checkpoint-native [N, K]
F.linear -> MatMulV2(false, true)
fused MoeGatingTopK
...
MoeInitRoutingCustom
```

Representative components were:

| Component | vLLM-Ascend |
|---|---:|
| Router matmul | 12.68 us |
| TopK | 4.84 us |
| Router span | 19.25 us |
| Gemma RMSNorm | 3.98 us |
| MoeInitRoutingCustom | 11.52 us |

`MoeInitRoutingCustom` includes routing initialization and dynamic
quantization; the trace does not show a separate per-layer `DynamicQuant` task.
Its 3.98 us RMSNorm is therefore not equivalent to a PyPTO task that also
produces BF16 post-norm, INT8 data, and scale. Use the reference trace to find
layout and fusion opportunities, but compare functions, shapes, dtypes, and
fusion boundaries before comparing durations.

## 7. MoE input preprocessing

### 7.1 Original coupling

`norm_quant_moe_input` produces three logically different outputs:

1. BF16 post-norm for the shared expert;
2. `inv_rms` for router-logit normalization; and
3. per-token INT8 data and scale for routed dispatch.

It scans H=4096 twice, rebuilds `resid * (gamma + 1)` during emission, and
preserves the validated conversion sequence:

```text
FP32 -> INT32(rint) -> FP16(round) -> INT8(trunc)
scale = inv_rms * (amax / 127)
```

This makes splitting the early `inv_rms` result from the later payload look
attractive. Whole-network results showed that task-wave and ownership costs
must be considered before changing that dependency graph.

### 7.2 Rejected decompositions

Two dispatch-fusion variants improved selected local spans but were not
admissible:

| Variant | Decomposition | Five-layer result | Whole-network result |
|---|---|---|---|
| V1 | Move quantization into a single-token dispatch-push wave | L3 router-ready improved, but five-MoE span regressed 12.24 us and L4 span regressed 12.12 us | Rejected before admission |
| V2 | Use `T * TOPK` route workers; BS1 launches eight workers | Five-MoE span regressed 3.87 us | P50 regressed 1.2235 ms (4.259%); hidden SHA differed |

V2 repeated token-level amax/quantization once per route. At BS1 that caused
roughly eight times the quantization reads and writes, increased AIV and task
competition, and changed model-level determinism despite matching the final
token. Assign token-level payload preparation to one token owner; let routes
consume the result instead of recomputing it.

V1 demonstrates a separate trap: a local producer or router-ready interval can
improve while a new scheduling wave or downstream resource collision makes the
five-layer span worse. Do not admit a split or fusion from one task duration.

### 7.3 Admitted active-row producer

The admitted V3 keeps the original single `norm_quant_moe_input` producer,
dispatch ABI, and dependency DAG. It changes only logical ownership:

```text
T active tokens -> T workers -> one complete row per worker
```

RMS, amax, BF16 post-norm, INT8 quantization, and scale remain in one task. The
implementation still performs the second scan and xg recomputation because
removing them with additional global task waves was slower at five-layer and
whole-network scope. The eight-lane FP32 tensors are physical reduction
carriers, not eight-row logical tiles.

This choice preserves the exact reduction and quantization sequence while
removing work on inactive storage rows. It is a BS-dynamic decomposition, not a
BS1-specialized GEMV path and not a fixed worker count.

### 7.4 V3 measured result

Medians across eight ranks relative to the active-row TopK baseline were:

| Metric | Change |
|---|---:|
| `norm_quant_moe_input` service | about 24.0 us -> 17.3--17.6 us |
| L3 router-ready | -5.39 us |
| L4 router-ready | -5.05 us |
| L3 MoE span | -5.90 us |
| L4 MoE span | -1.74 us |
| Five-MoE span | -7.05 us (-1.00%) |

The matched fresh-container BS1/context-65536 A/B/A result was:

| Arm | P50 | Mean |
|---|---:|---:|
| Baseline A1 | 28.723 ms | 28.850 ms |
| Candidate | 28.512 ms | 28.640 ms |
| Baseline A2 | 28.769 ms | 29.035 ms |
| Baseline midpoint | 28.746 ms | 28.9425 ms |
| Candidate delta | **-0.234 ms (-0.814%)** | **-0.3025 ms (-1.045%)** |

The P50 improvement exceeded the 0.023 ms A/A half-range. All three arms
produced the same expected token and hidden SHA-256.

## 8. Admission evidence

### 8.1 Router and shared-expert round

- Five-layer L3/L4 hidden outputs were BF16 byte-exact.
- Capacity-16 and capacity-32 compile gates passed.
- The dynamic-grid audit covered BS1 through BS32.
- The matched whole-network A/B/A result was -0.2555 ms P50 (-0.86%) against
  its baseline midpoint, beyond the 0.0225 ms A/A half-range.

### 8.2 TopK and preprocessing round

- Active-row TopK reduced service time to about 2.5--2.7 us and remained
  byte-exact in the five-layer harness.
- The focused performance-contract suite passed 38 tests; the complete
  Step3p5 unit suite passed 379 tests with 4 environment-gated skips.
- Capacity-16 and capacity-32 whole-decode compile gates passed.
- The final five-layer run captured all eight ranks; L3/L4 outputs were
  byte-exact with `max_abs=0`.
- The final fresh-container A/B/A precision gate passed with identical hidden
  SHA-256 across all arms.
- The representative campaign names are
  `moe-active-row-norm-quant-five-layer-dfx-20260814-013120` and
  `whole-aba-moe-active-row-norm-quant-20260814-013634`.

A single absolute run is not an A/B result. Do not compare a candidate with an
older intermediate artifact from a different Git base or workload and
attribute the difference to the current change.

## 9. Reusable model-development workflow

Use `.claude/skills/tune-moe-critical-path/SKILL.md`. The minimum workflow is:

1. Freeze parent and candidate source, image, checkpoint, devices, and workload.
2. Build a checkpoint-to-PTO layout ledger.
3. Separate storage capacity, logical active work, and physical alignment
   carriers; audit boundary BS values.
4. Draw generated dependencies and identify the first common branch consumer.
5. Give token-level payloads one owner; avoid route-level recomputation.
6. Evaluate any new task wave for queueing and resource contention, not only
   producer service time.
7. Reject illegal layout and tile variants with compile-only gates first.
8. Require byte-exact focused outputs before trusting timing.
9. Compare all ranks and report service, span, readiness delay, and overlap.
10. Run matched whole-network A/B/A in fresh containers.
11. Publish local, router-ready, layer-join, and ITL results separately.

## 10. Follow-up priorities

1. Reduce the remaining two-pass norm/quant work only if the design avoids a
   slower task wave and preserves exact reduction/quantization arithmetic.
2. Reuse router `gate_xg` only after proving its lifetime, traffic, and
   dependency costs at five-layer and whole-network scope.
3. Profile larger active batches before selecting a worker cap or a different
   token-to-core ownership formula.
4. Optimize routed-expert and collective stages that determine the layer join.
5. Treat further shared gate/up work as throughput or larger-BS work unless a
   new trace proves that branch has re-entered the critical path.
