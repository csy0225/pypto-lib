# step3p5 MoE 8-card EP — scheduler timeout debug log (superseded)

> **Superseded status (2026-06-24):** The earlier "missing in-kernel fence" hypothesis
> was useful but incomplete. A deeper cut matrix showed dispatch-only passes and
> dispatch+routed-expert fails; the confirmed root cause is routed-expert launching
> expert tiles with `tile_valid <= 0`. Guarding the routed expert tile body with
> `if tile_valid > 0` makes the full 8-card DeepSeek-style MoE path pass on
> gpu-a910x-0162 under CANN 9.0.0 non-GA. This file is retained as investigation
> history; use the 2026-06-24 commit and pypto-project status as the current
> source of truth.


## 0. Current validation status (2026-06-24 evening)

The 8-card communication/scheduler issue is no longer a global blocker on
`gpu-a910x-0162` with the current CANN 9.0.0 non-GA environment and rebuilt
runtime/native artifacts:

- `test_decode_layer_full_dense_multirank_st` passes 8-card numerical golden
  validation on devices `0..7` (`bad_ratio=0.0004` on every rank).
- `test_decode_layer_moe_st --smoke` passes for all six MoE variants.
- The MoE ST harness now validates `world_size=8` numerically instead of
  skipping golden validation: each rank computes its local attention `o_proj`
  partial, TP all-reduce is modeled by summing those partials, the replicated
  residual is added once, and zero expert weights make dispatch/combine/shared
  execute while contributing zero to the final tensor.
- With that golden, all MoE variants that occur in the real Step3p5 layer
  table pass on real devices: `full_silu_silu`, `full_swiglu7_silu`,
  `full_swiglu7_swiglu16`, `swa_silu_silu`, and `swa_swiglu7_silu`.


## 1. Goal (this work item)

Refactor the step3p5 MoE expert-parallel (EP) dispatch/combine to the DeepSeek **push**
style — eliminate the `send_x` pack buffer + extra comm windows (overhead) AND fix the
8-card multi-card stall.

## 2. What landed (valid, on the branch)

- **Push dispatch rewrite** (`decode_layer.py`): tokens are `pld.tensor.put` directly into
  the destination peer's `recv_x` at the final expert-major CSR row; r_route rides with the
  token via `pld.tile.remote_store` into `recv_r_route`. Dropped `pack_send_payload`,
  `ep_all_to_all` (pull), the receiver gather, and `build_inverse_map`.
- **Combine simplified**: `r_route` read from `recv_r_route_out` (no `src_route_table`, no
  `route_pub` barrier); routing weight applied source-side in `_weighted_gather_and_add`.
- **Windows**: dropped `send_x` (8 MB), `src_route_table` (~144 KB), `route_pub`; added
  `recv_r_route` (~32 KB); `pub_counts` cols padded to `n_local_experts_pad=40`.
  Net **13 → 11 windows, ~17.4 MB → ~9.4 MB per rank**. Compiles clean (~0.85 s).
- **Test fix** (`tests/step3p5/test_decode_layer_moe_st.py`): for `world_size>1`, set
  `router_bias = 0` so routing spreads across all `MOE_NUM_EXPERTS` (real load-balanced
  all-to-all, `dst_rank = eid // N_LOC_E` covers ranks 0..7). The old
  `[0, N_LOC_E)` mask forced every token to rank 0 (degenerate all-to-one). `world_size==1`
  keeps the restriction (no push to a non-existent peer).

## 3. The blocker

8-card compiles, all 8 `chip_process` reach "ready", then every rank aborts:
```
sync_run_streams: aclrtSynchronizeStreamWithTimeout (AICPU) failed: 507018
validate_runtime_impl: PTO2 runtime failed: orch_error_code=0 sched_error_code=100 runtime_status=-100
```
Device-log stall snapshot (AICPU `scheduler_cold_path.cpp`), symmetric on all 8 ranks:
```
TASK ring=1 state=RUNNING fanin_refcount=4/4 kernels=[aic:-1 aiv0:21 aiv1:-1]
   running_on=[core=N(aiv0) busy kernel=21 cond_reg_state=ack]
SUMMARY completed=35/38 scan_ready=0 scan_waiting=0 scan_running=1
```
AIV kernel **slot 21 = the dispatch kernel**, `state=RUNNING` with `cond_reg_state=ack` =
busy-waiting **inside a cross-rank op** (the in-kernel `pto::comm::TWAIT`/TPUT ack spin) for
a peer acknowledgment that never arrives → **symmetric cross-rank comm deadlock**.

## 4. Decisive evidence: the runtime DOES support 8-card EP

Bumped the hand-written reference `runtime/examples/workers/l3/ep_dispatch_combine` to N=8
(3 edits: `main.py N_RANKS=8`, `dispatch.cpp`/`combine.cpp` `static constexpr int N=8`) and ran
`-d 0-7`:
```
[ep_dispatch] all ranks matched golden ✅
```
So 8-card EP all-to-all works. The reference's `dispatch.cpp` carries **`pipe_barrier(PIPE_ALL)`
at 5 sites** — most critically (a) between the `pub_counts` notify group and the `count_done`
barrier, and (b) between the payload TPUT loop and `stage_out`. That fence is what lets the
cross-rank ops drain so the barrier/reads see complete data.

## 5. Exhaustive DSL-side attempts — all deadlock at 8-card

| # | attempt | result |
|---|---------|--------|
| 1 | EP barriers `Set` -> `AtomicAdd` (count_done/data_done/combine_done) | `sched=100` (no change) |
| 2 | Burst-free count exchange (own-row write + barrier + `remote_load`, "Option B") | `sched=100` (failure flipped to all-cores-IDLE) |
| 3 | DeepSeek-aligned count exchange (`AtomicAdd` notify-publish) + push | `sched=100` |
| 4 | Balanced random routing (`router_bias=0`) — rule out test data | `sched=100` (unchanged) |
| 5 | **Task-split** dispatch into 3 InCore tasks (publish, count_done+push, stage_out) so the task boundary drains, per doc option 2 | smaller heap -> `orch_error=2` HEAP_RING_DEADLOCK; with `PTO2_RING_HEAP=4GB` -> `sched=100` returns |

Attempt 5 is the key refutation: **a pypto InCore-task boundary does NOT drain cross-rank
TPUT/notify** — it is not a `pipe_barrier` equivalent. (It only added tasks, raising heap
pressure; with enough heap the underlying deadlock reappears unchanged.)

Also ruled out: rolling `pto-isa` back to `ddafa8da` — incompatible (current pypto codegen
calls `GetValidRow`/`GetValidCol`, added after `ddafa8da`; `TMov.hpp` won't compile). The
four repos are version-locked; a lone pto-isa downgrade is not viable.

## 6. Root cause

The pypto **DSL exposes no in-kernel memory-fence primitive**. `pld.system` offers only
`notify`/`wait`; `pipe_barrier(PIPE_ALL)` exists only in C++ kernel templates
(`pypto/runtime/builtins/.../kernel.cpp.in`), never as a callable DSL op. The codegen does
not auto-insert a fence between consecutive cross-rank op groups. So any DSL EP-dispatch that
issues a burst of cross-rank notifies/TPUTs followed by a barrier (or a TPUT followed by a
local read) cannot order them on the comm engine, and at 8 ranks all peers busy-wait on each
other's acks -> symmetric deadlock. (At 2 ranks the single-peer bursts are tiny and never
starve, which is why DeepSeek-v4's DSL dispatch only ever validated 2-card.)

## 7. Required fix (upstream pypto)

Two equivalent options, both in the **pypto framework** (not pypto-lib):
1. **Codegen auto-fence**: emit a pipe/memory barrier (`pipe_barrier(PIPE_ALL)` equivalent)
   between consecutive `pld.system.notify` / `pld.tensor.put` / `pld.tile.remote_store` groups
   that target different windows, so a barrier-notify or a local read issued after a
   cross-rank burst is ordered after it on the comm engine.
2. **Expose `pld.system.fence()`** DSL primitive and call it in `dispatch` at the two
   load-bearing points the reference kernel fences (publish/count_done, push/stage_out).

Necessary **and** sufficient evidence: the hand-written C++ `ep_dispatch_combine` passes
8-card *because* it has those `pipe_barrier`s; every fence-less DSL variant deadlocks.

## 8. Current code state (branch `wip/moe-barrier-allreduce`, uncommitted)

- `models/step3p5/decode_layer.py`: `DecodeLayerMoE` push rewrite + 3-task dispatch split +
  AtomicAdd EP barriers + `pub_counts` pad. Dead-but-defined (uncalled): `ep_all_to_all`,
  `_pack_send_payload`, `_build_inverse_map`, `_publish_src_route_table` (remove on cleanup).
- `tests/step3p5/test_decode_layer_moe_st.py`: multi-card balanced-routing `router_bias`.
- Single-card MoE device run also hits 507018 (same EP-runtime family; not separately triaged).
- Mirror to `moe.py` `EpTpMoE` + `dispatch.py`/`combine.py` reference bodies + goldens: NOT done.
- Window-overhead measurement + commit: NOT done.

## 9. Cross-links

- [`step3p5-moe-multicard-comm-window-setup.md`](step3p5-moe-multicard-comm-window-setup.md) — earlier blocker history (heap-ring deadlock fix; the `count_done` localization).
- [`step3p5-barrier-allreduce-change-record.md`](step3p5-barrier-allreduce-change-record.md) — the barrier-mesh all_reduce (TP collective) that is kept.
- `pypto/runtime/examples/workers/l3/ep_dispatch_combine/` — the working hand-written 8-card EP reference (N=2 default; N=8 verified passing).
