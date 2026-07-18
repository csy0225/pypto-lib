# step3p5 barrier-mesh `tp_all_reduce` — change record

> **Why this record exists.** Before rewriting the step3p5 MoE expert-parallel (EP)
> dispatch/combine in the DeepSeek-v4 *push* style, this captures the barrier-mesh
> all-reduce work already on `wip/moe-barrier-allreduce` so it is not lost or
> re-litigated. The barrier all-reduce is a **TP collective** used by attention and
> the shared expert; it is **orthogonal to the EP dispatch/combine** being rewritten
> and **must be carried forward unchanged** (it is verified on 8-card dense).

**Branch:** `wip/moe-barrier-allreduce` (pypto-lib), tip `fa60514`.
**Machine:** gpu-a910x-0162 (driver 25.5.2 / firmware 7.8.0.7.220 / CANN 9.0.0-beta.1, ptoas v0.45).
**Date recorded:** 2026-06-24.

---

## 1. What problem it solves

The original `tp_all_reduce` was a pull-side **ring** reduce-scatter + all-gather
(`group_size-1` steps; each step stored one `[BATCH, tp_chunk]` chunk into a window,
issued a monotonic `AtomicAdd` notify + `Ge(step+1)` wait, then `remote_load`ed the
previous rank's chunk). That multi-step + monotonic-AtomicAdd + per-step-window
pattern triggered a **pypto codegen multibuffer-ctx bug**: a canonical TP=8 dense
decode layer compiled cleanly but, dispatched on 8 ranks, every `chip_process`
aborted with **507018** (`aclrtSynchronizeStreamWithTimeout (AICPU) failed`),
synchronized across all 8 ranks at the all-reduce task (task 13 of the decode layer),
value-independent.

Root-cause detail (codegen side) is documented in
[`pypto-codegen-tp-all-reduce-multibuffer-ctx.md`](pypto-codegen-tp-all-reduce-multibuffer-ctx.md)
and [`../known-pypto-pitfalls.md` §7](../known-pypto-pitfalls.md). That upstream bug
is still open; step3p5 works around it with the barrier-mesh form below.

## 2. The design — barrier-mesh all-reduce

Mirrors `pypto/tests/st/distributed/test_l3_allreduce.py` (verified PASS at
TP=2/4/8 on real NPU). Three phases, no ring stepping:

1. **stage-in** — each rank copies its whole `local[BATCH, HIDDEN]` into its own
   `tmp_window` slot, tiled by a **fixed `ar_chunk = HIDDEN // 8`**.
2. **barrier** — each rank `notify(AtomicAdd, value=1)` to every peer's
   `signal_window[my_rank]`, then `wait(Ge 1)` on every peer's cell.
3. **reduce** — each rank, per `ar_chunk` tile, casts its own tile to FP32 and
   `remote_load`s + adds every peer's same tile (accumulate in FP32), then casts
   back to BF16 into `local`.

Canonical body (`models/step3p5/attention_full.py:815-865`, mirrored in
`attention_swa.py` and `decode_layer.py`):

```python
ar_chunk = HIDDEN // 8                       # fixed, INDEPENDENT of tp_size
for k0 in pl.range(0, HIDDEN, ar_chunk):     # phase 1: stage-in
    stage_tile = pl.load(local, [0, k0], [BATCH, ar_chunk])
    pl.store(stage_tile, [0, k0], tmp_window)
for peer in pl.range(group_size):            # phase 2: barrier notify
    if peer != my_rank:
        pld.system.notify(target=signal_window, peer=peer,
                          offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
for src in pl.range(group_size):             # phase 2: barrier wait
    if src != my_rank:
        pld.system.wait(signal=signal_window, offsets=[src, 0],
                        expected=1, cmp=pld.WaitCmp.Ge)
for k0 in pl.range(0, HIDDEN, ar_chunk):     # phase 3: mesh reduce (fp32 acc)
    own_tile = pl.load(tmp_window, [0, k0], [BATCH, ar_chunk])
    acc = pl.cast(own_tile, target_type=pl.FP32)
    for peer in pl.range(group_size):
        if peer != my_rank:
            recv = pld.tile.remote_load(tmp_window, peer=peer,
                                        offsets=[0, k0], shape=[BATCH, ar_chunk])
            acc = pl.add(acc, pl.cast(recv, target_type=pl.FP32))
    pl.store(pl.cast(acc, target_type=pl.BF16), [0, k0], local)
```

### Two load-bearing details

- **`ar_chunk = HIDDEN // 8` is fixed, NOT `HIDDEN // tp_size`.** With
  `tp_chunk = HIDDEN // tp_size`, single-card (`apply_tp1_patch`, `tp_size=1`)
  collapses to `tp_chunk = HIDDEN = 4096`, so the phase-3 `[BATCH, 4096]` FP32 acc
  tile (256 KB) overflows the 188 KB UB limit (`Vec buffer usage 655360 > 188416`)
  at `AllocateMemoryAddr`. The fixed `512` (= canonical TP=8 chunk; HIDDEN is
  divisible by it) keeps the per-iteration working set bounded for every `tp_size`.
  (commit `4f3605e`)
- **The `tmp_window` is full `[BATCH, HIDDEN]`, not `[BATCH, tp_chunk]`.** The
  inlined barrier form stages the whole HIDDEN; the old 1/8-width window overran on
  multi-card. `attn_tmp_buf`/`sh_tmp_buf` in `DecodeLayerMoE` were widened to
  `BATCH*HIDDEN` accordingly. (commit `fa60514`)

## 3. Commit set (since merge-base `9c4773f` on `stepfun/develop`)

| commit | summary | key change |
|--------|---------|------------|
| `b5bb6ee` | barrier-style tp_all_reduce + per_rank multirank inputs | ring → barrier-mesh in attention_full/swa + decode_layer; `per_rank()` broadcast of host inputs to `[N_RANKS,...]` |
| `4f3605e` | decouple barrier all_reduce HIDDEN tiling from tp_size | `ar_chunk = HIDDEN // 8` (fixed); +codegen doc + known-pitfalls §7 |
| `862c273` | multi-rank TP=8 numerical golden test | `tests/step3p5/system/test_decode_layer_full_dense_multirank_st.py` |
| `c5911d1` | merge `stepfun/develop` | — |
| `fa60514` | barrier all_reduce + HIDDEN-width windows for `DecodeLayerMoE` | widen `attn_tmp`/`sh_tmp` to HIDDEN; shared-expert ring → barrier-mesh |

**Files touched:** `models/step3p5/{attention_full,attention_swa,decode_layer,step3p5_decode}.py`,
`tests/step3p5/system/test_decode_layer_full_dense_multirank_st.py`,
`docs/known-pypto-pitfalls.md`, `docs/upstream-issues/pypto-codegen-tp-all-reduce-multibuffer-ctx.md`.

## 4. Verification status

- **8-card TP=8 dense decode layer-0 e2e:** clean `rc=0`, 6.78 s (vs ring → 507018 on
  3/3 prior runs). Barrier-mesh resolves the multi-card 507018 for the TP collective.
- **Single-card dense ST (device 0):** `full_dense` 7.91 s / `swa_dense` 14.96 s — PASS,
  no regression vs ring.
- **MoE 8-card:** the barrier all_reduce + 1 GB heap let it compile and launch all 8
  chip processes with `orch_error_code=0` (heap deadlock gone), **but it still stalls
  later** at the EP `dispatch_step` `count_done` barrier (`sched=100`). That stall is
  **not** an all_reduce problem — it is the EP dispatch/combine path, which is exactly
  what the DeepSeek-style push rewrite targets. See
  [`step3p5-moe-multicard-comm-window-setup.md`](step3p5-moe-multicard-comm-window-setup.md) §9.

## 5. What carries forward into the MoE push rewrite

- **KEEP the barrier-mesh `tp_all_reduce` verbatim** (attention o-proj reduce +
  shared-expert reduce). It is verified on 8-card dense and is independent of EP
  dispatch/combine.
- **KEEP** the widened `attn_tmp`/`sh_tmp` windows (`[BATCH, HIDDEN]`) and their
  `attn_sig`/`sh_sig` barrier windows.
- **KEEP** the `per_rank()` host-input broadcast to `[N_RANKS, ...]` in
  `step3p5_decode.py` (the MoE ST harness depends on `tensors[name][r_idx, ...]`).
- The MoE rewrite replaces only the **EP windows + dispatch/combine** (`recv_x`,
  `send_x`, `pub_counts`, `src_route_table`, `routed_y`, the barriers) — not the TP
  all_reduce above.

## 6. Cross-links

- [`pypto-codegen-tp-all-reduce-multibuffer-ctx.md`](pypto-codegen-tp-all-reduce-multibuffer-ctx.md) — the ring-pattern codegen root cause (upstream-open).
- [`../known-pypto-pitfalls.md` §7](../known-pypto-pitfalls.md) — `pl.range(constant)` UB-overflow / chunk-follows-slice pitfall.
- [`step3p5-moe-multicard-comm-window-setup.md`](step3p5-moe-multicard-comm-window-setup.md) — the EP-MoE multi-card blockers (heap-ring deadlock fixed; `count_done` stall = next target of the push rewrite).
