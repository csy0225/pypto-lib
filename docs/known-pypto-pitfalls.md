# Known PyPTO Pitfalls and Hard Limits

This document catalogs concrete defects, undocumented constraints, and
**hard limits** of the pypto / pto-isa / simpler stack that have bitten
upper-layer model code. Each entry has the same shape:

- **Symptom** — what you see at compile or runtime
- **Trigger** — the smallest model-side construct that produces it
- **Source location** — pypto / pto-isa / simpler file and line
- **Avoidance recipe** — the model-side pattern to use instead
- **Reproducer** — when applicable, a script you can run

Keep this list current. When you spend more than ~15 minutes debugging
something that turns out to be a known hard limit (or a documented bug),
add it here in the same shape.

The original investigation that surfaced the four entries below is in
[CLAUDE.md](../../CLAUDE.md) "Phase 15 单卡 = `full_fa_fused` VEC UB
对齐 bug" / TASK-30 and the project-memory note
`project_p15_fault_is_full_head_gate.md`.

---

## 1. `[N, 1]` (1-column FP32) intra-UB VEC tile — runtime "UB not aligned"

**Symptom** — runtime fault on AIV:

```
errcode:(0, 0x800, 0) errorStr: The UB address accessed by the VEC
instruction is not aligned. ... subErrType:4
```

surfaced as `RuntimeError: run_prepared failed with code 507018`. The
chip_process child gets force-reset by simpler.

**Trigger** — slicing a single column out of a wider 2D tile and feeding
it to `pl.row_expand_mul`:

```python
gate_row_fp32 = pl.slice(gate_logits, [BATCH_TILE, NUM_HEADS_PAD], [b0, 0])
sigmoid_all = pl.recip(pl.add(pl.exp(pl.neg(gate_row_fp32)), 1.0))
for hg_h in pl.range(NUM_HEADS):
    head_slice = pl.slice(attn_out, [BATCH_TILE, HEAD_DIM], [b0, hg_h * HEAD_DIM])
    hg_gate = pl.slice(sigmoid_all, [BATCH_TILE, 1], [0, hg_h])  # ← BAD
    gated = pl.row_expand_mul(pl.cast(head_slice, target_type=pl.FP32), hg_gate)
```

`pl.slice(..., [N, 1], ...)` lowers to a tile descriptor

```cpp
Tile<TileType::Vec, float, N, K_parent, BLayout::RowMajor, N, 1, ...> v;
```

with `valid_shape=[N, 1]` whose **valid row byte size** is `1 col ×
sizeof(FP32) = 4 bytes`. pto-isa requires that to be 32-byte aligned
(see entry §2 below). The static check on `pto.alloc_tile` catches
the mis-alignment for `pl.full([1, 1], ...)` / `pl.full([N, 1], ...)`
but **NOT** for `pl.slice`, so the kernel compiles cleanly and only
faults at runtime.

**Source location** — pto-isa AIV vec micro-instructions; the matching
static checker is `pto.alloc_tile` op verifier in ptoas and
`pto-isa` `isSameLayout` static-assert at `TLoad.hpp:459` (already
covers GM↔UB load; **does not** cover intra-UB VEC tiles).

**Avoidance recipe** — never produce a `[N, 1]` (or `[1, 1]`) FP32 tile
via slicing. Build per-row gate tiles via **reduction** (`pl.row_sum`,
`pl.row_max`) or **reshape** of a 1-D vector — those tile descriptors
have `tile_shape == valid_shape` and skip the alignment check:

```python
# ✅ Reduction-built [B, 1] (qwen3-14b RMSNorm pattern)
inv_rms = pl.rsqrt(pl.add(pl.mul(pl.row_sum(pl.mul(x, x)), HIDDEN_INV), EPS))
normed  = pl.col_expand_mul(pl.row_expand_mul(x, inv_rms), gamma)

# ✅ Reshape-built [B, 1] (deepseek-v3.2 / qwen3-14b L3 pattern)
gate_b1 = pl.reshape(per_row_scalar_vec, [BATCH_TILE, 1])
gated   = pl.row_expand_mul(x, gate_b1)
```

If the gate inherently lives as a column of a 2D tile (head-wise gate),
no clean model-side rewrite exists today — see the **status** below.

**Reproducer** — `pypto-lib/tools/p15_trace/run_with_trace.py`,
`P15_DISPATCH_LIMIT=10` PASS / `=11` FAIL. The single delta is
dispatching `full_head_gate`, which generates the broken `[N, 1]` tile.

**Status** — open upstream: pto-isa needs to extend the alignment guard
to intra-UB VEC tiles (the `pto.alloc_tile` rule already exists for
`pl.full`-allocated tiles; the `pl.slice` lowering path needs the same
check). Until that lands, **`step3p5`'s `full_head_gate`** keeps the
known-broken shape with an explicit comment block; Phase 15 single-card
decode is gated on the upstream fix.

---

## 2. Vec / none_box tile row byte size must be ≥ 32 B and 32-B aligned

**Symptom** — codegen error at compile time:

```
'pto.alloc_tile' op expects result row-major none_box tile row byte
size (cols * sizeof(dtype)) to be 32-byte aligned, but got 4 bytes
```

**Trigger** — explicit `pl.full` with too-narrow column count:

```python
ones_11 = pl.full([1, 1],  dtype=pl.FP32, value=1.0)   # 4 B → reject
ones_18 = pl.full([1, 8],  dtype=pl.FP32, value=1.0)   # 32 B → OK
ones_b1 = pl.full([16, 1], dtype=pl.FP32, value=0.0)   # 4 B / row → reject
ones_b8 = pl.full([16, 8], dtype=pl.FP32, value=0.0)   # 32 B / row → OK
```

This is the **AIV vector lane width** — the AICore VEC pipe fetches
32 bytes (256 bits) per micro-op and any narrower contiguous span
crosses the boundary unaligned.

| dtype | sizeof | min cols for legal Vec tile |
|---|---|---|
| FP32 | 4 B | **8** |
| INT32 | 4 B | **8** |
| FP16 / BF16 | 2 B | **16** |
| INT8 / U8 | 1 B | **32** |

**Source location** — emitted by `pto.alloc_tile` op verifier in
ptoas. Search for the literal phrase
`"to be 32-byte aligned, but got"` for the exact rule.

**Avoidance recipe** — choose tile column counts that satisfy the table.
Where the math wants a `[N, 1]` shape, build it via reduction (entry §1),
not `pl.full`.

---

## 3. `pl.dynamic("dim")` on a tensor's **leading** dim drops cross-function slice strides

**Symptom** — generated kernel `.cpp` emits the wrong stride template
for a tensor parameter:

```cpp
// expected for rope_cos: [4096, 64] FP32 → row stride 64
GlobalTensor<float, ..., pto::Stride<64, 64, 64, 64, 1>, ...> v51 = ...;
// what we get when rope_cos is [ROPE_SEQ_DYN, 64]:
GlobalTensor<float, ..., pto::Stride<32, 32, 32, 32, 1>, ...> v51 = ...;
```

For `pos = 0` the bad stride does not change the address (offset is
0 either way) so the kernel APPEARS to work; for `pos > 0` the kernel
reads from row `pos × 32 elem` instead of `pos × 64 elem` and the GM
load lands on misaligned data. **Latent**.

**Trigger** — a parameter tensor whose **leading** dim is `pl.dynamic(...)`,
sliced with a constant column shape, then passed as `In` arg to a
different `@pl.function(type=pl.FunctionType.InCore)` callee:

```python
ROPE_SEQ_DYN = pl.dynamic("ROPE_SEQ_DYN")

@pl.function(type=pl.FunctionType.Orchestration)
def chip_orch(
    rope_cos: pl.Tensor[[ROPE_SEQ_DYN, 64], pl.FP32],
    ...
):
    cos_lo = pl.slice(rope_cos, [1, 32], [pos, 0])  # ← stride lost on cross-fn pass
    # passing cos_lo to an InCore function loses the parent stride view
```

**Source location** —
`pypto/src/ir/transforms/optimize_orch_tensors_pass.cpp:75-91`
`ComputeRowMajorStrides` returns empty when **any** shape dim is dyn.
Then `SliceInputStridesOptimizer::Apply` (line 1813) sees `empty`
and `continue`s — the `In` parameter loses its
`TensorView(stride=[…])` annotation and downstream codegen falls back
to contiguous stride.

The fix in upstream is straightforward: compute only the *trailing*
strides the slice actually needs. The function already takes
`full_strides.end() - in_rank` (line 1824) so the leading-dim's stride
is never read. A trailing-only helper would not need the leading dim
to be static.

**Avoidance recipe** — for **model-bound** dims (context length, layer
count, KV-cache layout, MLP intermediate, etc.) declare integer
constants in `config.py`, **not** `pl.dynamic`. step3p5 uses this style
already — see `models/step3p5/config.py` "Model-bound shape constants"
block. `pl.dynamic` is correct **only** for genuinely per-request
dimensions (live batch size that varies across calls); even then,
slices off it should ideally not cross InCore boundaries.

```python
# ✅ Model-bound — static int
ROPE_SEQ_DEFAULT = 4096
KV_CACHE_ROWS   = 4096
NUM_HIDDEN_LAYERS = 45

@pl.function(...)
def chip_orch(
    rope_cos: pl.Tensor[[ROPE_SEQ_DEFAULT, 64], pl.FP32],
    ...
):

# ❌ Don't — pypto codegen drops parent stride on the slice
ROPE_SEQ_DYN = pl.dynamic("ROPE_SEQ_DYN")

@pl.function(...)
def chip_orch(rope_cos: pl.Tensor[[ROPE_SEQ_DYN, 64], pl.FP32], ...):
    cos_lo = pl.slice(rope_cos, [1, 32], [pos, 0])  # → InCore arg
```

**Reproducer** — see project-memory `project_p15_rope_bisect_ladder.md`
and the IR-dump pair at
`.build-cache/p15_real_passes3/passes_dump/14_after_OptimizeOrchTensors.py`
(failing real, no `TensorView` on `cos_lo`) vs
`build_output/FullRopeTrueCacheId6_*/passes_dump/14_after_OptimizeOrchTensors.py`
(passing repro, `TensorView(stride=[64, 1])` annotated).

**Status** — open upstream. Step3p5 worked around at the model side by
staticising the dyn dims in `config.py`; **the upstream bug remains**.

---

## 4. `pl.dynamic` dims add phantom unreferenced `int32_t` params to kernels

**Symptom** — generated kernel `.cpp` signature has more trailing
`int32_t` parameters than the chip_orch `params_tN.add_scalar(...)`
dispatch supplies. Body may reference them in row-offset arithmetic
(e.g. `v21 = v4 * 16` when `v4` is one of the phantom params).

**Trigger** — same as §3: any tensor parameter shape that contains a
`pl.dynamic(...)` Var causes the codegen to thread that Var as a
trailing int param of every kernel that touches the tensor.

**Source location** —
`pypto/src/codegen/pto/pto_codegen.cpp` `CollectTensorShapeDynVars`
(line 175). Comments in that function explicitly note the trailing
`%argN` parameters: "trailing index params on the emitted func.func
signature".

**Why it usually doesn't fault** — the AICore parameter block is
zero-initialised by the runtime, so an unreferenced phantom reads `0`.
Code like `v21 = v4 * 16` then evaluates to `0`, indexes row 0,
addresses match. **It is fragile, not benign**: a future codegen change
that drops the zero-init or re-orders args breaks every dyn-dim kernel.

For `full_head_gate` specifically, the phantom `v4` is read at
`v21 = v4 * v15` and used as a row-offset multiplier. Empirically the
zero-init has held; the head_gate fault is entry §1, not §4.

**Avoidance recipe** — same as §3: use static integer constants for
model-bound dims so no phantom int32 is emitted.

**Status** — open upstream. The codegen needs to either (a) skip
emitting the phantom param when no kernel body references it, or
(b) wire the dispatch to fill it explicitly via
`params_tN.add_scalar(extracted_dyn_dim_value)`.

---

## 5. AICPU `aicpu_orchestration_entry` cannot `fprintf(stderr)`

**Symptom** — `fprintf(stderr, ...)` injected into the generated
`chip_orch.cpp` (the AICPU-side orchestration body) compiles into the
`.so` (you can `strings` for the format string and find it) but the
output never reaches the parent process's stderr, even though
"[chip_process pid=…] ready" from the same child does.

**Trigger** — any plain `fprintf` / `printf` inside
`aicpu_orchestration_entry`.

**Why** — `chip_orch.cpp` runs on the on-chip AICPU (an ARM core
inside the Ascend chip), not the host x86 process. Its libc `stderr`
is not wired to host `fd 2`.

**Avoidance recipe** — use the `LOG_*` macros from
`pto_orchestration_api.h` (route through
`current_runtime()->ops->log_*` into simpler's unified log channel
and surface in the chip_process child's stderr):

```cpp
LOG_WARN("[my_trace] something interesting %d", value);
LOG_INFO_V5("[my_trace] %s", "verbose info");
```

Note the AICPU log is gated by CANN's `ASCEND_GLOBAL_LOG_LEVEL` (and
simpler's HostLogger level) — even `LOG_WARN` may be silenced by
default. Set `ASCEND_GLOBAL_LOG_LEVEL=1` before launch if you need
the warn / info tier to show.

**Reproducer / harness** — for kernel-launch trace, the simplest
approach is to monkey-patch
`pypto.runtime.device_runner.compile_single_orchestration`
**before** `ir.compile` runs and rewrite the generated
`chip_orch.cpp` to insert traces. See
`pypto-lib/tools/p15_trace/run_with_trace.py` for a working
implementation that:

1. Inserts `LOG_WARN("[P15_TRACE] dispatch task=%d ...", N);` before
   each `rt_submit_(aic|aiv)_task(N, params_tN)`.
2. Optionally comments out late dispatches via
   `P15_DISPATCH_LIMIT=N` for "skip post-task-N" bisect.
3. Optionally injects extra `params_tN.add_scalar(0);` for testing
   phantom-int-param hypotheses.

---

## 6. `for x in <iterable>` inside a `@pl.jit` / `@pl.function` body

**Symptom** — frontend error:

```
Error: For loop must use pl.range(), pl.parallel(), pl.unroll(),
pl.pipeline(), pl.while_(), or pl.spmd()
```

**Trigger** — Python `for x in range(N):` or any non-pypto iterator
inside a kernel body. **Note:** `pl.unroll(N)` is the explicit
compile-time-unroll variant — it expands at parse time so the loop
variable is a Python int, not a runtime scalar.

**Avoidance recipe** — pick the right loop primitive:

| Construct | Body sees iter as | Placement |
|---|---|---|
| `pl.range(N)` | runtime `Scalar` | inside or outside `pl.at` |
| `pl.parallel(N)` | runtime `Scalar`, distributed | outside `pl.at` only |
| `pl.spmd(N)` | runtime `Scalar`, distributed (each iter is an InCore region) | outside `pl.at` only |
| `pl.pipeline(N, stage=K)` | runtime `Scalar`, software-pipelined | inside `pl.at` only |
| `pl.unroll(N)` | **Python int** (compile-time unrolled) | inside or outside `pl.at` |
| `pl.while_(cond)` | runtime, while-loop | per case |

Use `pl.unroll` when constant-folding the iter index is necessary —
e.g. when slice offsets `[..., hg_h]` must lower to compile-time
constant addresses. (Note: per §1 above, `pl.unroll` does not by
itself rescue a `[N, 1]` slice — the slice still emits the broken
tile.)

**Source location** — `pypto/python/pypto/language/loop_compatibility.py`
(or similar) — search for the error message string.

---

## 7. `pl.range(constant)` unrolls without SSA buffer reuse → UB overflow at compile time

**Symptom** — compile-time fault at `AllocateMemoryAddr` pass:

```
Verification failed after 'AllocateMemoryAddr' for properties {AllocatedMemoryAddr}:
[1] ERROR - AllocatedMemoryAddr
  Message: Function 'tp_all_reduce': Vec buffer usage (655360 bytes)
           exceeds platform limit (188416 bytes)
  Location: <kernel>.py:<line>
```

**Trigger** — a `pl.range(N)` whose bound `N` is a Python int (factory
closure constant, module global, captured `tp_size=8`, etc.) where each
iteration of the body creates fresh tile SSA values that depend on the
iteration index, including a loop-carried accumulator. Concretely:

```python
group_size = tp_size            # Python int, e.g. 8
acc = pl.cast(own_tile, target_type=pl.FP32)
for peer in pl.range(group_size):
    if peer != my_rank:
        recv      = pld.tile.remote_load(window, peer=peer, ...)  # BF16 [B, CHUNK]
        recv_fp32 = pl.cast(recv, target_type=pl.FP32)             # FP32 [B, CHUNK]
        acc       = pl.add(acc, recv_fp32)                         # NEW FP32 [B, CHUNK]
```

The compiler **fully unrolls** the loop because `group_size` is a
compile-time constant. It then treats each iteration's `recv`,
`recv_fp32`, and `acc` as **distinct SSA values** and allocates UB
slots for all of them simultaneously. Tile-reuse / liveness analysis
across unrolled iterations is **not implemented** for the loop-carried
`acc`. UB cost = `(group_size - 1) × per_iter_tile_bytes`, easily
blowing the 184 KB Vec UB budget on A2A3 once `B × CHUNK × 4` per tile
exceeds ~25 KB.

**Source location** — pypto compiler `AllocateMemoryAddr` pass; visible
in `Vec buffer usage` overflow messages from `MaterializeTensorStrides`
and downstream codegen.

**Avoidance recipes**:

A. **Make the loop bound runtime-dynamic** so the compiler emits a real
   loop and allocates UB once for the iteration body. Mirror the
   canonical `pypto/tests/st/distributed/test_l3_allreduce.py`:

   ```python
   ctx    = pld.get_comm_ctx(data)
   nranks = pld.nranks(ctx)        # runtime Scalar, NOT a Python int
   for peer in pl.range(nranks):
       ...
   ```

B. **Don't carry the accumulator across iterations** — write the
   partial result back to a host-visible tensor (`local`) at the end of
   each peer iteration and re-load it at the start of the next. Per-
   iteration working set is `cur + recv + cur_fp32 + recv_fp32 + sum +
   sum_bf16` ≈ `6 × B × CHUNK × 2..4` ≈ 144 KB at `B=16, CHUNK=512`
   which fits 184 KB:

   ```python
   for peer in pl.range(group_size):
       if peer != my_rank:
           cur  = pl.load(local, [0, k0], [B, CHUNK])
           recv = pld.tile.remote_load(window, peer=peer, offsets=[0, k0],
                                       shape=[B, CHUNK])
           summed = pl.add(pl.cast(cur,  target_type=pl.FP32),
                           pl.cast(recv, target_type=pl.FP32))
           pl.store(pl.cast(summed, target_type=pl.BF16), [0, k0], local)
   ```

   This trades one extra DDR round-trip per peer for predictable UB.

C. **Shrink CHUNK** so `(group_size - 1) × per_iter_tile_bytes ≤ UB
   limit`. Quick and degrades DMA efficiency; only useful when the
   kernel cannot tolerate (A) or (B).

**Reproducer** — observed 2026-06-22 on csy0225/pypto-lib branch
`wip/step3p5-barrier-allreduce-20260622` HEAD `b5bb6ee`, in
`models/step3p5/decode_layer.py:487` `_dense_mlp_body_tp.tp_all_reduce`.
The body matches pattern (A) above but with `group_size = tp_size = 8`
(Python int from factory closure), and trips the overflow because the
compiler sees seven distinct unrolled `acc` SSA values plus their FP32
casts. The HEAD-of-`stepfun/develop` ring all_reduce avoids the issue
by storing each chunk back to `local` immediately (pattern B applied
naturally to the ring shape).

**Counter-example — unrolling alone does NOT inflate UB** (measured
2026-08-10, `stepfun/develop@d13b2ca6`). `decode_fwd.py`'s
`expert_gate_up` has `for kb in pl.range(1, HIDDEN // 256)` — a
constant-bound, fully-unrolled 15-iteration loop — and its AIV kernel
allocates only **two** `pl.tile.alloc(pl.Mem.Vec, 65536)` at offsets
`0` and `65536`; all 16 K-iterations rewrite the same two offsets.
Reuse works fine.

The discriminator is therefore **not** "is the bound a Python int" but:

```text
does any Vec-resident value stay live ACROSS iterations?
  yes -> every iteration's copy is simultaneously live -> UB = N x per_iter
  no  -> MemoryReuse assigns the same offsets each iteration -> UB = per_iter
```

In §7's `tp_all_reduce` the loop-carried `acc` forces every unrolled
copy live at once. In `expert_gate_up` the accumulator lives on the
**cube** side (`pl.matmul` / `pl.matmul_acc` produce INT32 in L0C, not
UB), so the AIV side is a pure staging shuttle with nothing carried.

Diagnostic: don't reason about it — read
`passes_dump/33_after_AllocateMemoryAddr.py` and count
`pl.tile.alloc(pl.Mem.Vec, N)` per function (see
[dev-workflow-gotchas.md](dev-workflow-gotchas.md) §7).

**Cross-reference** — this is **distinct** from §6 ("kernel body must
use `pl.range/parallel/unroll/...`"). §6 is a frontend rejection of
raw `for x in range(N):`. §7 is a back-end UB-budget defect that bites
you even when you correctly use `pl.range`, but the bound is a compile-
time int **and** something is loop-carried in UB. See §8 for the UB
budget model itself.

---

## 8. AIV Vec (UB) budget is per-kernel-per-core; fusion is super-additive; `pl.pipeline` buys overlap **with** buffer

Three separate facts that together decide every tiling / fusion /
software-pipelining question. All measured 2026-08-10 on
`stepfun/develop@d13b2ca6` (A2A3, `Vec` limit `188416 B`); scripts and
JSON at
`0162:/mnt/persist/chensiyu/workspace/perf-2026q3/ub-scope-20260810/`.

### 8.1 The budget is per-kernel-per-core, not a shared pool

| Evidence | Observation |
|---|---|
| Sum over all kernels far exceeds the limit, yet compiles | 149 AIV functions allocate `4676512 B` total = **24.8x** the `188416 B` limit; `COMPILE_OK` |
| Each function restarts at offset 0 | every top consumer reports `min_offset = 0`; `mem_vec_*` numbering is per-function |
| SPMD grid does not share either | `combine_reduce` has `core_num=16` and `40960 B`; a shared pool would need `655360 B` and fail — it compiles |

UB is a per-AIV-core scratchpad laid out at kernel entry from the static
offsets `AllocateMemoryAddr` computed, and dead at kernel exit. Kernel
boundaries isolate it for free — **no task dependency is required to
make two kernels' UB coexist, because they never coexist.**

The load-bearing consequence: **a kernel cannot leave an intermediate in
UB for the next kernel to consume.** Any hand-off goes through GM/DDR or
a c2v/v2c pipe. This is why "fusing two tasks" forces both stagings to
be simultaneously live.

So a per-kernel usage ranking (e.g. `swa_qk_norm_zc` 148352 /
`full_rmsnorm_zc` 135488 / `attn_residual_hold` 131072 /
`expert_gate_up_aiv` 131072 / `tp_all_reduce` 98304 /
`combine_reduce` 40960) is a list of **independent** occupancies, not a
sum. Read it as "how much headroom does *this* kernel have".

### 8.2 Fusion cost is super-additive: it adds c2v pipe slots

Merging `expert_gate_up_act` into `expert_gate_up` did **not** cost
`sum(parts)`. Diffing the pass dumps, the fused AIV kernel gains:

```text
aiv_initialize_pipe dir_mask   2 (v2c only)  ->  3 (bidirectional)
                    + a locally allocated c2v_slot_buffer
kernel body                    + tpop_from_aic x2, + cast x4
```

Accounting: `2 x 65536` (original v2c staging, charged to AIC via
`import_peer_buffer`) `+ 4 x 65536` (new local c2v slots) `= 393216 B`
— which is exactly the number in the overflow message. The extra copies
are **new pipe slots**, not load/compute/store replicas.

Arithmetic model (verified by prediction, see §8.4):

```text
slot_size            = K_CHUNK x N_CHUNK        (INT8 operand)
baseline expert_gate_up  Vec = 512  x K_CHUNK
fused  gate_up+act       Vec = 1536 x K_CHUNK
```

**Fusion is not a capability limit.** The same tree already contains a
fused path that compiles — `full_moe_chip_orch_swiglu7_swiglu16_expert_gate_up_aiv`
uses `4 x 8192 = 32768 B` by running `K=64 / N=64 /
RECV_SPECIAL_TILE=32`. Before declaring "the platform can't do X",
grep the generated kernel names and the other model paths for an
existing instance of X.

### 8.3 `pl.pipeline(stage=N)` spends buffer to buy overlap

Easy to get backwards. `pl.pipeline` **replicates the loop body `stage`
times** to enable ping-pong, so both copies must be live at once:

```text
Vec = stage x n_staging_buffers x K_CHUNK x N_CHUNK
```

Measured on `expert_gate_up`'s K-reduction loop rewritten in the
`models/deepseek/v4/expert_routed.py::exp_gate_mm` shape
(`pl.create_tensor` accumulator + `pl.pipeline(0, HIDDEN, KC, stage=2)`
+ `k0 == 0 ? pl.matmul : pl.matmul_acc`):

| variant | predicted `stage x 2 x KC x 256` | measured |
|---|---:|---|
| `KC=256, stage=2` | 262144 B | **FAIL, reports 262144 B** |
| `KC=128, stage=2` | 131072 B | **COMPILE_OK** |

And the dump confirms it lowered rather than being silently ignored:
staging goes `2 x 65536` -> **`4 x 32768`**, `aiv_initialize_pipe` shows
`pipe: (32768, 4)`, and `27_after_LowerPipelineLoops.py` contains the
pipeline structure.

Practical rule: **to add `stage=2` you must first roughly halve the
tile.** A kernel already above `94208 B` cannot take `stage=2` at its
current tiling, no matter what the other kernels do (§8.1).

### 8.4 Method: fit a law, then predict a number

Every wrong root cause in this area was a narrative without a number;
every correct one came from a prediction that matched. The loop that
worked:

1. Sweep the one parameter you suspect and record the overflow byte
   count — the number in `Vec buffer usage (X bytes)` **is a
   measurement**, not just an error.
   K_CHUNK sweep gave `256 -> 393216`, `512 -> 786432`,
   `1024 -> 1572864` — linear in `KC`, which already falsifies
   "unrolling replicates buffers".
2. Fit the coefficient (`1536 x KC` fused vs `512 x KC` baseline).
3. Write the prediction down **before** running: `KC=128 -> 196608
   FAIL`, `KC=64 -> 98304 PASS`.
4. Run the no-card codegen gate (~14 s, see
   [dev-workflow-gotchas.md](dev-workflow-gotchas.md) §6). Both hit
   exactly.
5. Only then read the pass dump to confirm the *mechanism*.

---

## 9. Cross-references and further reading

- [pypto-coding-style.md](pypto-coding-style.md) — the canonical happy-
  path API (broadcast ops, slicing, loop primitives, `pl.at` scopes).
- [performance-tuning.md](performance-tuning.md) — L2 (inter-kernel
  schedule) and L1/L0 (intra-kernel) tuning rules. §8 above is the
  **UB budget model** those rules have to fit inside.
- [dynamic-shape-guidelines.md](dynamic-shape-guidelines.md) — the
  *correct* way to use `pl.dynamic` when you really need it.
- [debugging.md](debugging.md) — runtime / precision symptom triage.
- [compile-runtime-workflow.md](compile-runtime-workflow.md) — what the
  pypto compile + simpler dispatch pipeline does at each stage.
- [dev-workflow-gotchas.md](dev-workflow-gotchas.md) — operational
  pitfalls outside pypto itself (stale `__pycache__` after monkey-
  patching, environment activation, git/SSH/PAT auth on netboot
  hosts) — separate from this file because they are *workflow* bugs,
  not *pypto* bugs, but burn the same kind of debugging time.
- `../models/step3p5/CLAUDE.md` — the project-level tracker that
  references this file from §"已知风险".
