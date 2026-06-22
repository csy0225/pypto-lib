# Ring `tp_all_reduce` hits 507018 on multi-rank; resolved with a barrier-mesh all-reduce

**Status:** Worked around in step3p5 (barrier-mesh all-reduce replaces the ring).
The underlying pypto codegen bug for the ring pattern is still open upstream.

**Affected:** `models/step3p5/{decode_layer,attention_full,attention_swa}.py`
`tp_all_reduce` (the TP-group sum-reduce after attention out-proj and after the
dense MLP down-proj).

## Symptom

A canonical TP=8 dense decode layer compiled cleanly but, when dispatched across
8 ranks, every `chip_process` aborted in the drain phase:

```
RuntimeError: LocalMailboxEndpoint::run: child failed (endpoint=N, code=1):
  chip_process dev=N: RuntimeError: run_prepared failed with code 507018
```

507018 = `aclrtSynchronizeStreamWithTimeout (AICPU) failed` — an AICore/AICPU
execution fault, value-independent (reproduced with zero/dummy inputs). The fault
synchronised across all 8 ranks at the all-reduce task.

## Root cause (ring pattern)

The original `tp_all_reduce` was a pull-side **ring** reduce-scatter + all-gather
(mirrors `pypto/tests/st/distributed/test_l3_ring_allreduce.py`): `group_size-1`
steps, each step stores one chunk into a `[BATCH, tp_chunk]` window, issues a
monotonic `AtomicAdd` notify and a `Ge(step+1)` wait, then `remote_load`s the
previous rank's chunk. When this multi-step + monotonic-AtomicAdd + per-step
window-reuse pattern is lowered for multi-rank, pypto codegen mishandles the
multibuffer comm context, producing a kernel that faults at runtime with 507018.
The ring is UB-cheap (O(N) bytes/rank) but does not survive multi-rank codegen.

## A/B evidence (gpu-a910x-0162, 2026-06-22)

Same harness (`step3p5_decode.run_real_npu --no-smoke -p a2a3 -d 0,1,2,3,4,5,6,7
--tp-world-size 8 --dummy-weights`), same zero inputs; only the all-reduce
implementation differs:

| all_reduce | 8-card TP=8 dense layer-0 | evidence |
|---|---|---|
| ring (committed `stepfun/develop`) | 507018, 3/3 runs | `/tmp/tp8_e2e{,_v2,_v3}.log` (2026-06-20) |
| barrier-mesh (this branch) | clean, rc=0, 6.78s | run 2026-06-22 |

## Resolution (barrier-mesh)

Replace the ring with the canonical **barrier mesh** all-reduce
(mirrors `pypto/tests/st/distributed/test_l3_allreduce.py`, verified by pypto at
P=2/4):

1. **Stage-in** — store the full local `[BATCH, HIDDEN]` into this rank's window
   slot.
2. **Barrier** — `pld.system.notify` every peer (AtomicAdd 1), then
   `pld.system.wait(Ge 1)` on every peer slot. Fixed `expected=1`; no monotonic
   step counter.
3. **Reduce** — for each peer, `pld.tile.remote_load` its slice and `pl.add`
   (through FP32; PTOAS A2/A3 has no bf16 `tadd`).

Bandwidth tradeoff: O(N^2) bytes/rank vs ring's O(N), but at N=8 / HIDDEN=4096
BF16 that is ~56 KB cross-rank read per rank — negligible vs the dense gate_up
matmul traffic.

### UB caveat that gated the barrier (chunk-follows-slice)

The first barrier draft tiled HIDDEN with `tp_chunk = HIDDEN // tp_size`. Under
`apply_tp1_patch` (single-card e2e / dense ST, `tp_size=1`) that collapses to
`tp_chunk = HIDDEN = 4096`, so the phase-3 `[BATCH, 4096]` FP32 accumulator tile
(256 KB) blows the 188 KB UB limit:

```
Function 'tp_all_reduce': Vec buffer usage (655360 bytes) exceeds platform limit (188416 bytes)
```

Fix: tile HIDDEN with a **fixed** width independent of `tp_size`,
`ar_chunk = HIDDEN // 8` (= 512, the canonical TP=8 chunk; HIDDEN divisible by
it). At TP=8 unchanged; at TP=1 bounded. See known-pypto-pitfalls.md §7
("collective HIDDEN tiling must not follow tp_size").

## Remaining

- `DecodeLayerMoE.tp_all_reduce` is still the ring form (separate MoE 507018
  blocker; convert when the MoE device path is unblocked).
- Multi-card numerical/golden validation (non-zero inputs) still pending — the
  A/B run above only proves fault-free execution (zero input → 0.0 output).
- Upstream pypto fix for the ring multibuffer-ctx lowering not yet filed.
