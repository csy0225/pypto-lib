# step3p5 MoE decode layer — multi-card (TP=8/EP=8) debug record

**Status (2026-06-24): resolved for ST runtime.** Two setup/runtime issues were
separated during debug: (1) PTO2 heap-ring pressure is avoided by using the larger
`PTO2_RING_HEAP/PTO2_RING_TASK_WINDOW/PTO2_RING_DEP_POOL` settings; (2) the
remaining `sched_error_code=100` was caused by routed-expert kernels being launched
for empty/past-tail tiles (`tile_valid <= 0`). The final fix guards the routed expert
tile body with `if tile_valid > 0`; full 8-card DeepSeek-style MoE ST now passes
on gpu-a910x-0162.


## 1. Symptom

8-card run compiles, all 8 `chip_process` reach "ready", then every rank aborts:

```
[ERROR] sync_run_streams: aclrtSynchronizeStreamWithTimeout (AICPU) failed: 507018
[ERROR] validate_runtime_impl: PTO2 runtime failed:
        orch_error_code=0 sched_error_code=100 runtime_status=-100
RuntimeError: chip_process dev=N: run_prepared failed with code 507018
```

`finalize()` force-resets all 8 cards (recovers cleanly). The fault is fast
(~1 s after "ready"), value-independent.

---

## 2. The MoE decode layer — forward structure

One `DecodeLayerMoE` layer (full-attention flavour) runs, per rank, this task
chain (8-card task ids from the generated `chip_orch.cpp`):

| task | kernel | kind | notes |
|------|--------|------|-------|
| 0 | `full_rmsnorm_zc` | AIV | input RMSNorm (zero-centred) |
| 1 | `full_q_proj` | AIC | per-rank Q proj (8 heads) |
| 2 | `full_k_proj` | AIC | per-rank K proj (1 KV head) |
| 3 | `full_v_proj` | AIC | per-rank V proj (1 KV head) |
| 4 | `full_qk_norm_zc` | AIV | Q/K RMSNorm |
| 5 | `full_gate_proj` | AIC | head-gate weight proj (gate bypassed) |
| 6 | `full_rope_kv_cache` | AIV | RoPE + KV-cache write |
| 7 | `full_qk_matmul` | AIC | attention scores |
| 8 | `full_softmax` | AIV | |
| 9 | `full_sv_matmul` | AIC | |
| 10 | `full_online_softmax` | AIV | flash-attn online softmax |
| 11 | `full_out_proj_matmul` | AIC | per-rank o-proj partial |
| 12 | `full_out_proj_cast` | AIV | |
| **13** | **`tp_all_reduce`** | AIV | **TP all-reduce of attention o-proj (barrier-mesh)** |
| 14 | `full_out_resid_add` | AIV | resid1 = current_hidden + o |
| 15 | `moe_post_rmsnorm_zc` | AIV | post-attn RMSNorm |
| 16 | `gate_topk` | AIV | router: top-K expert selection |
| 17 | (gate post) | AIV | |
| **19** | **`tp_all_reduce`** | AIV | (gate/route reduce) |
| **20** | **`dispatch_step`** | AIV | **EP all-to-all: send tokens to expert-owning ranks** |
| ... | routed-expert gate/up/silu/down | AIC/AIV | per-rank 36 local experts |
| 23 | `_publish_src_route_table` | AIV | combine bookkeeping |
| 24 | `pub_route_barrier` | AIV | |
| **25** | **`_push_routed_y_to_sources`** | AIV | **EP all-to-all back** |
| 26 | `moe_combine` | AIV | weighted gather of expert outputs |
| 27 | `moe_residual_add` | AIV | next_hidden = resid1 + moe_out + shared_expert |

Collectives: **two TP `tp_all_reduce`** (attention + shared-expert/gate) and **two
EP all-to-all** (dispatch + combine).

---

## 3. Kernel (chip_orch) inputs / outputs — shapes

Model constants: `HIDDEN=4096`, `HEAD_DIM=128`, `BATCH=16`,
`NUM_HEADS_FULL=64`, `NUM_KV_HEADS=8`, `Q_PER_KV_FULL=8`,
`MOE_NUM_EXPERTS=288`, `MOE_TOP_K(TOPK)=8`, `MOE_INTERMEDIATE(routed)=1280`,
`SHARE_EXPERT_DIM=1280`, `NUM_HIDDEN_LAYERS=45`.

TP=8 / EP=8 per-rank derived widths:
`tp_size = n_ranks = 8`, `NUM_HEADS_FULL_LOCAL=8`, `KV_HEADS_LOCAL=1`,
`hidden_q_local = 8*128 = 1024`, `KV_HIDDEN_LOCAL = 1*128 = 128`,
`N_LOCAL_EXPERTS = 288/8 = 36`, `INT_R(routed local) = 1280`,
`sh_inter_local = INTER_S_LOCAL = 1280/8 = 160`,
`n_full (full-attn layers) = 12`, `num_heads_local_pad = 16`,
`LOCAL_RECV_MAX = n_ranks*BATCH*TOPK = 8*16*8 = 1024`,
`n_routes_per_rank = BATCH*TOPK = 16*8 = 128`.

### Host data tensors (passed as `tensors[name][r_idx, ...]`, leading dim = N_RANKS=8)

| # | name | per-rank shape | dtype | sharding |
|---|------|----------------|-------|----------|
| 1 | `current_hidden` | [16, 4096] | BF16 | replicated |
| 2 | `input_rms_weight` | [45, 4096] | FP32 | replicated (per-layer rows) |
| 3 | `wq` | [49152, 1024] | BF16 | TP head-sliced (n_full*HIDDEN x H_Q_local) |
| 4 | `wk` | [49152, 128] | BF16 | TP KV-head-sliced |
| 5 | `wv` | [49152, 128] | BF16 | TP KV-head-sliced |
| 6 | `q_norm_weight` | [45, 128] | FP32 | replicated |
| 7 | `k_norm_weight` | [45, 128] | FP32 | replicated |
| 8 | `seq_lens` | [16] | INT32 | replicated |
| 9 | `block_table` | [512] | INT32 | replicated |
| 10 | `slot_mapping` | [16] | INT32 | replicated |
| 11 | `rope_cos` | [4096, 64] | FP32 | replicated |
| 12 | `rope_sin` | [4096, 64] | FP32 | replicated |
| 13 | `k_cache` | [4096, 128] | BF16 | per-rank (1 KV head) |
| 14 | `v_cache` | [4096, 128] | BF16 | per-rank (1 KV head) |
| 15 | `wo` | [12288, 4096] | BF16 | TP head-sliced (n_full*H_Q_local x HIDDEN) |
| 16 | `w_g` | [49152, 16] | BF16 | head-gate weight (bypassed; PAD=16) |
| 17 | `post_rms_weight` | [45, 4096] | FP32 | replicated |
| 18 | `gate_w` | [4096, 288] | FP32 | replicated (router) |
| 19 | `router_bias` | [288] | FP32 | replicated |
| 20 | `w_gate_r` | [36, 4096, 1280] | BF16 | EP expert-sliced (36 local experts) |
| 21 | `w_up_r` | [36, 4096, 1280] | BF16 | EP expert-sliced |
| 22 | `w_down_r` | [36, 1280, 4096] | BF16 | EP expert-sliced |
| 23 | `w_gate_s` | [4096, 160] | BF16 | TP intermediate-sliced (shared expert) |
| 24 | `w_up_s` | [4096, 160] | BF16 | TP intermediate-sliced |
| 25 | `w_down_s` | [160, 4096] | BF16 | TP intermediate-sliced |
| 26 | `next_hidden_out` | [16, 4096] | BF16 | **OUTPUT** (replicated after all-reduce) |
| - | `layer_idx` | scalar | INT32 | not per-rank |

### Cross-rank comm windows (allocated per rank via `pld.alloc_window_buffer` + `pld.window`)

This is the suspected root cause — `DecodeLayerMoE` allocates **13** windows
(vs the dense layer's **4**), two of which are 8 MB:

| window | shape | dtype | bytes/rank | role |
|--------|-------|-------|-----------:|------|
| `attn_tmp` | [16, 4096] | BF16 | 128 KB | attention all-reduce staging (fixed: was [16, tp_chunk]) |
| `attn_sig` | [8, 1] | INT32 | 32 B | attention all-reduce barrier |
| `pub_counts` | [64, 36] | INT32 | 72 KB | dispatch per-(src,dst) token counts (n_ranks^2 x local_experts) |
| `count_done` | [8, 1] | INT32 | 32 B | dispatch count barrier |
| `recv_x` | [1024, 4096] | BF16 | **8 MB** | dispatch recv buffer (LOCAL_RECV_MAX x HIDDEN) |
| `data_done` | [8, 1] | INT32 | 32 B | dispatch data barrier |
| `send_x` | [1024, 4096] | BF16 | **8 MB** | dispatch send buffer |
| `sh_tmp` | [16, 4096] | BF16 | 128 KB | shared-expert all-reduce staging (fixed: was [16, sh_tp_chunk]) |
| `sh_sig` | [8, 1] | INT32 | 32 B | shared-expert all-reduce barrier |
| `src_route` | [8, 36, 128] | INT32 | 144 KB | combine route table (n_ranks x local_experts x routes) |
| `route_pub` | [8, 1] | INT32 | 32 B | route publish barrier |
| `routed_y` | [128, 4096] | BF16 | 1 MB | combine routed-output buffer (routes x HIDDEN) |
| `combine_done` | [8, 1] | INT32 | 32 B | combine barrier |

**Total ~ 17.4 MB of comm windows per rank**, dominated by `recv_x`+`send_x` (16 MB).
Compare the dense layer: only `attn_tmp`/`attn_sig`/`mlp_tmp`/`mlp_sig`
(~256 KB total), which maps fine on 8 cards.

---

## 4. Fault localization (dispatch-cut bisect)

Using `P15_DISPATCH_LIMIT` (tools/p15_trace/run_with_trace.py) to comment out
`rt_submit_*_task(K)` for K > limit, then observe whether 507018 still appears.
All runs on 8 real cards (`--world-size 8 -p a2a3`):

| LIMIT | tasks kept | 507018? |
|------:|------------|---------|
| 17 | through gate_topk | YES |
| 14 | through out_resid_add (incl. attention all-reduce 13) | YES |
| 12 | through out_proj_cast (no all-reduce) | YES |
| **0** | **only rmsnorm (task 0); everything else cut** | **YES** |

**Conclusion: faults even at LIMIT=0** -> the fault is NOT in any compute kernel,
NOT in a collective, NOT input-dependent. It is at the **comm-domain /
window-allocation setup**, which happens before/independent of task dispatch.

Corroborating facts:
- Attention weights are std=0.02 (same careful scaling as the **passing** dense
  8-card golden) -> not BF16 overflow.
- The dense 8-card golden (`test_decode_layer_full_dense_multirank_st`,
  same per-rank attention kernel) PASSES (`bad_ratio 0.0004`) — it has only 4 small
  windows.
- `~/ascend/log/run/plog/plog-*.log` carries no per-window fault detail (init-stage
  TDT lines only).

---

## 5. Why "per-rank sliced inputs" (option B) does NOT help

The setup fault is input-value- and input-layout-independent (reproduces at
LIMIT=0 with only rmsnorm). Slicing/scaling the host inputs cannot change a
comm-window allocation fault. The comm windows are defined by the compiled
program, identical regardless of input data.

---

## 6. Prime suspects (for the follow-up investigation)

The cross-rank `allocate_domain` mapping of the 13 MoE windows on 8 cards. Most
likely one of:
1. **Total window size / count limit** — 17.4 MB x 8 ranks of symmetric windows
   may exceed a comm-domain pool/registration limit (dense's 256 KB is fine).
2. **One specific EP window's mapping** — `recv_x`/`send_x` (8 MB each) or the
   routing windows (`pub_counts` [64,36], `src_route` [8,36,128]) may have a
   shape/alignment the cross-rank mapping rejects.
3. A symmetric-window requirement violated by one of the MoE windows.

To isolate: temporarily reduce/stub MoE windows (invasive) or add simpler-side
`allocate_domain` instrumentation to log which window's cross-rank map returns the
error. This is a **simpler (PTO runtime) comm-domain** layer issue, likely upstream.

---

## 7. What is fixed vs. what remains

**Fixed (commit `fa60514`, branch `wip/moe-barrier-allreduce`):**
- `attn_tmp_window`: `[BATCH, tp_chunk]` -> `[BATCH, HIDDEN]` (the inlined barrier
  attention all-reduce tiles full HIDDEN; the old 1/8-width window overran on
  multi-card).
- shared-expert `tp_all_reduce`: ring -> barrier-mesh (the ring form hit the
  multi-card 507018 fixed for the dense layer).
- Both `sh_tmp`/`attn_tmp` windows widened to `BATCH*HIDDEN`.

**Test-harness scaffolding (uncommitted, on 0162 working tree):**
- `test_decode_layer_moe_st.py`: relaxed the `world_size=8 requires a2a3sim` guard
  (0162 has 16 NPUs); broadcast inputs to `[N_RANKS,...]` (host_orch indexes
  `tensors[name][r_idx]`, was `[1,...]` -> IndexError); skip the single-rank numerical
  golden at `world_size>1`.

**Remaining blocker:** the comm-window setup-level 507018 (section 6).

---

## 8. Reproduction

```bash
# on gpu-a910x-0162, env activated (CANN set_env + activate.sh + PTO_ISA_ROOT)
cd /data/chensiyu/hw_project/pypto/workspace/pypto-lib
python -m tests.step3p5.test_decode_layer_moe_st \
    --variant full_silu_silu --world-size 8 -p a2a3 -d 0
# -> compiles, 8 chip_process ready, then 507018 on all ranks; cards force-reset.

# dispatch-cut to confirm setup-level:
P15_DISPATCH_LIMIT=0 python /tmp/moe_trace8.py   # still 507018
```
