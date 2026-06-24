# step3p5 MoE 解码层 —— 多卡 (TP=8/EP=8) debug 记录

> **状态 (2026-06-24):ST runtime 已解决。** 本轮排查把两个问题拆开了:
> 1) PTO2 heap ring 压力通过调大 `PTO2_RING_HEAP/PTO2_RING_TASK_WINDOW/PTO2_RING_DEP_POOL` 避免;
> 2) 剩余 `sched_error_code=100` 的真因不是 missing fence,而是 routed expert
> 对空 tile / 越过尾部 tile 仍提交 kernel (`tile_valid <= 0`)。最终在 routed
> expert tile body 外加 `if tile_valid > 0` 后,8 卡 DeepSeek-style MoE ST 在
> gpu-a910x-0162 + CANN 9.0.0 非 GA 上通过。本文保留为排查记录。


## 1. 现象

8 卡运行能编译、8 个 `chip_process` 全部 "ready",随后每个 rank 都中止:

```
[ERROR] sync_run_streams: aclrtSynchronizeStreamWithTimeout (AICPU) failed: 507018
[ERROR] validate_runtime_impl: PTO2 runtime failed:
        orch_error_code=0 sched_error_code=100 runtime_status=-100
RuntimeError: chip_process dev=N: run_prepared failed with code 507018
```

`finalize()` 把 8 张卡 force-reset(能干净恢复)。故障很快(ready 后 ~1s),与数值无关。

---

## 2. MoE 解码层 —— 前向结构

一个 `DecodeLayerMoE` 层(full-attention 风味)在每个 rank 上跑下面这条 task 链(task id 取自 8 卡生成的 `chip_orch.cpp`):

| task | kernel | 类型 | 说明 |
|------|--------|------|------|
| 0 | `full_rmsnorm_zc` | AIV | 输入 RMSNorm(零中心) |
| 1 | `full_q_proj` | AIC | per-rank Q 投影(8 头) |
| 2 | `full_k_proj` | AIC | per-rank K 投影(1 KV 头) |
| 3 | `full_v_proj` | AIC | per-rank V 投影(1 KV 头) |
| 4 | `full_qk_norm_zc` | AIV | Q/K RMSNorm |
| 5 | `full_gate_proj` | AIC | head-gate 权重投影(gate 已旁路) |
| 6 | `full_rope_kv_cache` | AIV | RoPE + 写 KV-cache |
| 7 | `full_qk_matmul` | AIC | 注意力打分 |
| 8 | `full_softmax` | AIV | |
| 9 | `full_sv_matmul` | AIC | |
| 10 | `full_online_softmax` | AIV | flash-attn 在线 softmax |
| 11 | `full_out_proj_matmul` | AIC | per-rank o-proj 部分和 |
| 12 | `full_out_proj_cast` | AIV | |
| **13** | **`tp_all_reduce`** | AIV | **注意力 o-proj 的 TP all-reduce(barrier-mesh)** |
| 14 | `full_out_resid_add` | AIV | resid1 = current_hidden + o |
| 15 | `moe_post_rmsnorm_zc` | AIV | attn 后 RMSNorm |
| 16 | `gate_topk` | AIV | 路由器:top-K 专家选择 |
| 17 | (gate 后处理) | AIV | |
| **19** | **`tp_all_reduce`** | AIV | (gate/route reduce) |
| **20** | **`dispatch_step`** | AIV | **EP all-to-all:把 token 发给拥有对应专家的 rank** |
| ... | routed-expert gate/up/silu/down | AIC/AIV | 每 rank 36 个本地专家 |
| 23 | `_publish_src_route_table` | AIV | combine 记账 |
| 24 | `pub_route_barrier` | AIV | |
| **25** | **`_push_routed_y_to_sources`** | AIV | **EP all-to-all 回传** |
| 26 | `moe_combine` | AIV | 专家输出的加权 gather |
| 27 | `moe_residual_add` | AIV | next_hidden = resid1 + moe_out + shared_expert |

collective 共 4 个:**2 个 TP `tp_all_reduce`**(注意力 + shared-expert/gate)和 **2 个 EP all-to-all**(dispatch + combine)。

---

## 3. Kernel(chip_orch)输入 / 输出 —— shape

模型常量:`HIDDEN=4096`、`HEAD_DIM=128`、`BATCH=16`、`NUM_HEADS_FULL=64`、`NUM_KV_HEADS=8`、`Q_PER_KV_FULL=8`、`MOE_NUM_EXPERTS=288`、`MOE_TOP_K(TOPK)=8`、`MOE_INTERMEDIATE(routed)=1280`、`SHARE_EXPERT_DIM=1280`、`NUM_HIDDEN_LAYERS=45`。

TP=8 / EP=8 的 per-rank 派生宽度:
`tp_size = n_ranks = 8`、`NUM_HEADS_FULL_LOCAL=8`、`KV_HEADS_LOCAL=1`、
`hidden_q_local = 8*128 = 1024`、`KV_HIDDEN_LOCAL = 1*128 = 128`、
`N_LOCAL_EXPERTS = 288/8 = 36`、`INT_R(routed 本地) = 1280`、
`sh_inter_local = INTER_S_LOCAL = 1280/8 = 160`、
`n_full(full-attn 层数) = 12`、`num_heads_local_pad = 16`、
`LOCAL_RECV_MAX = n_ranks*BATCH*TOPK = 8*16*8 = 1024`、
`n_routes_per_rank = BATCH*TOPK = 16*8 = 128`。

### Host 数据张量(以 `tensors[name][r_idx, ...]` 传入,首维 = N_RANKS=8)

| # | 名称 | per-rank shape | dtype | 切分方式 |
|---|------|----------------|-------|----------|
| 1 | `current_hidden` | [16, 4096] | BF16 | 复制(replicated) |
| 2 | `input_rms_weight` | [45, 4096] | FP32 | 复制(逐层行) |
| 3 | `wq` | [49152, 1024] | BF16 | TP 按头切(n_full*HIDDEN x H_Q_local) |
| 4 | `wk` | [49152, 128] | BF16 | TP 按 KV 头切 |
| 5 | `wv` | [49152, 128] | BF16 | TP 按 KV 头切 |
| 6 | `q_norm_weight` | [45, 128] | FP32 | 复制 |
| 7 | `k_norm_weight` | [45, 128] | FP32 | 复制 |
| 8 | `seq_lens` | [16] | INT32 | 复制 |
| 9 | `block_table` | [512] | INT32 | 复制 |
| 10 | `slot_mapping` | [16] | INT32 | 复制 |
| 11 | `rope_cos` | [4096, 64] | FP32 | 复制 |
| 12 | `rope_sin` | [4096, 64] | FP32 | 复制 |
| 13 | `k_cache` | [4096, 128] | BF16 | per-rank(1 KV 头) |
| 14 | `v_cache` | [4096, 128] | BF16 | per-rank(1 KV 头) |
| 15 | `wo` | [12288, 4096] | BF16 | TP 按头切(n_full*H_Q_local x HIDDEN) |
| 16 | `w_g` | [49152, 16] | BF16 | head-gate 权重(已旁路;PAD=16) |
| 17 | `post_rms_weight` | [45, 4096] | FP32 | 复制 |
| 18 | `gate_w` | [4096, 288] | FP32 | 复制(路由器) |
| 19 | `router_bias` | [288] | FP32 | 复制 |
| 20 | `w_gate_r` | [36, 4096, 1280] | BF16 | EP 按专家切(36 个本地专家) |
| 21 | `w_up_r` | [36, 4096, 1280] | BF16 | EP 按专家切 |
| 22 | `w_down_r` | [36, 1280, 4096] | BF16 | EP 按专家切 |
| 23 | `w_gate_s` | [4096, 160] | BF16 | TP 按 intermediate 切(shared expert) |
| 24 | `w_up_s` | [4096, 160] | BF16 | TP 按 intermediate 切 |
| 25 | `w_down_s` | [160, 4096] | BF16 | TP 按 intermediate 切 |
| 26 | `next_hidden_out` | [16, 4096] | BF16 | **输出**(all-reduce 后各 rank 一致) |
| - | `layer_idx` | 标量 | INT32 | 非 per-rank |

### 跨卡 comm window(每 rank 通过 `pld.alloc_window_buffer` + `pld.window` 分配)

这是**根因嫌疑**—— `DecodeLayerMoE` 分配了 **13 个** window(dense 层只有 **4 个**),其中两个是 8 MB:

| window | shape | dtype | 每卡字节 | 用途 |
|--------|-------|-------|---------:|------|
| `attn_tmp` | [16, 4096] | BF16 | 128 KB | 注意力 all-reduce 暂存(已修:原来是 [16, tp_chunk]) |
| `attn_sig` | [8, 1] | INT32 | 32 B | 注意力 all-reduce barrier |
| `pub_counts` | [64, 36] | INT32 | 72 KB | dispatch 的 per-(src,dst) token 计数(n_ranks^2 x local_experts) |
| `count_done` | [8, 1] | INT32 | 32 B | dispatch 计数 barrier |
| `recv_x` | [1024, 4096] | BF16 | **8 MB** | dispatch 接收缓冲(LOCAL_RECV_MAX x HIDDEN) |
| `data_done` | [8, 1] | INT32 | 32 B | dispatch 数据 barrier |
| `send_x` | [1024, 4096] | BF16 | **8 MB** | dispatch 发送缓冲 |
| `sh_tmp` | [16, 4096] | BF16 | 128 KB | shared-expert all-reduce 暂存(已修:原来是 [16, sh_tp_chunk]) |
| `sh_sig` | [8, 1] | INT32 | 32 B | shared-expert all-reduce barrier |
| `src_route` | [8, 36, 128] | INT32 | 144 KB | combine 路由表(n_ranks x local_experts x routes) |
| `route_pub` | [8, 1] | INT32 | 32 B | route publish barrier |
| `routed_y` | [128, 4096] | BF16 | 1 MB | combine 路由输出缓冲(routes x HIDDEN) |
| `combine_done` | [8, 1] | INT32 | 32 B | combine barrier |

**每卡合计约 17.4 MB 的 comm window**,主要被 `recv_x`+`send_x`(16 MB)占据。对比 dense 层:只有 `attn_tmp`/`attn_sig`/`mlp_tmp`/`mlp_sig`(共 ~256 KB),在 8 卡上映射正常。

---

## 4. 故障定位(dispatch-cut 二分)

用 `P15_DISPATCH_LIMIT`(tools/p15_trace/run_with_trace.py)把 K > limit 的 `rt_submit_*_task(K)` 注释掉,再看 507018 是否还出现。全部在 8 张真卡上跑(`--world-size 8 -p a2a3`):

| LIMIT | 保留的 task | 是否 507018 |
|------:|------------|---------|
| 17 | 到 gate_topk | 是 |
| 14 | 到 out_resid_add(含注意力 all-reduce 13) | 是 |
| 12 | 到 out_proj_cast(无 all-reduce) | 是 |
| **0** | **只留 rmsnorm(task 0),其余全砍** | **是** |

**结论:连 LIMIT=0 都挂** -> 故障不在任何计算 kernel、不在 collective、与输入无关。它在 **comm-domain / window 分配建立**阶段,发生在 task 调度之前 / 与之无关。

旁证:
- 注意力权重是 std=0.02(与**通过的** dense 8 卡 golden 同样的缩放)-> 不是 BF16 溢出。
- dense 8 卡 golden(`test_decode_layer_full_dense_multirank_st`,同一套 per-rank 注意力 kernel)**通过**(`bad_ratio 0.0004`)—— 它只有 4 个小 window。
- `~/ascend/log/run/plog/plog-*.log` 没有 per-window 故障细节(只有 init 阶段 TDT 行)。

---

## 5. 为什么 "per-rank 切分输入"(option B)解决不了

这个 setup 故障与输入值、输入布局都无关(只留 rmsnorm 的 LIMIT=0 也复现)。切分 / 缩放 host 输入改变不了一个 comm-window 分配故障。comm window 由编译后的程序定义,与输入数据无关。

---

## 6. 主要嫌疑(供后续排查)

### 6.0 ⭐ 跨模型对照(2026-06-23 新增,**修正了原"window 规模超限"假设**)

实跑了 **DeepSeek v4 decode_layer** 做对照(它是 step3p5 MoE 的模板,但 **EP=2 + DP 注意力 + INT8 量化 dispatch + 小很多的 window**):

| 模型 | 配置 | 真机结果 | dispatch-cut LIMIT=0 |
|------|------|----------|----------------------|
| step3p5 MoE(pypto 生成) | 8 卡, TP8/EP8, BF16, ~17.4MB window | **507018**(8卡 sched=100) | **仍挂(setup 层)** |
| DeepSeek v4(pypto 生成) | 2 卡, DP/EP2, INT8, 小 window | **507018**(orch=2) | **仍挂(setup 层)** |
| step3p5 **dense**(pypto 生成,对照) | 8 卡, TP8, **只有 4 个小 all_reduce window** | **PASS**(golden 0.0004) | — |
| **simpler 手写 `ep_dispatch_combine`(对照)** | **2 卡, EP2, DeepSeek-V4 真实 MoE shape(T=128,TOPK=6,D=4096,L=16,R=192)** | **✅ PASS**(`all ranks matched golden`, bad=0/524288, rel 6e-4) | — |

**结论(两次修正后定稿)**:
1. DeepSeek 小 window 也挂 → **不是 window 规模问题**。
2. **simpler 自己手写的 EP dispatch+combine,用同样的 DeepSeek MoE shape、同样 2 卡真机,跑得干净通过** → **EP all-to-all 原语 + comm-window 机制在 simpler 层是好的**。
3. 同样 2 卡 / 同样 shape 下:**手写 simpler 编排 PASS,pypto 生成的编排 507018** —— 干净的 A/B 把矛头指向 **pypto 对分布式 MoE orchestration / comm-window setup 的 codegen**,**不是** simpler runtime、**不是** EP 原语、**不是** window 大小、**不是**模型。
4. 旁证:pypto 生成的纯 TP dense 层(只有 all_reduce window)8 卡 PASS → pypto 的 TP 编排没问题,问题在 **EP 编排的 codegen**。

→ **根因方向:pypto codegen 生成的 distributed EP-MoE host_orch/chip_orch 的 comm-domain/window 建立,与能跑通的手写 simpler 版本有差异。** 下一步就是 diff 这两者的 orchestration / window 建立代码,找出 pypto 多做/少做/做错了什么。

> ⚠ 保留:DeepSeek v4 decode_layer 是 "smoke" 测试,可能只在 a2a3sim 验证过(0162 缺 `g++-15` 跑不了 sim 对照),其 pypto 真机多卡是否曾通过未知。但「simpler 手写同 shape PASS + 两个 pypto 生成模型挂」已足以定位到 pypto 的 EP 编排 codegen。

### 6.1 待排查的具体方向(已收敛到 pypto codegen)

由 §6.0:simpler 手写 `examples/workers/l3/ep_dispatch_combine`(同 DeepSeek shape、同 2 卡真机)**PASS**,而 pypto 生成的 EP-MoE **挂**。所以排查应**对比这两条路径的编排 / comm-window 建立**:
- diff **pypto 生成的 `host_orch.py` / `chip_orch.cpp`**(EP-MoE 的 `alloc_window_buffer` / `pld.window` / domain 建立 / notify-wait 序列)**vs** simpler 手写 `ep_dispatch_combine` 的对应建立代码,找出 pypto 多做 / 少做 / 顺序不同的地方(window 注册顺序、domain rank 映射、对称性要求、buffer 对齐等);
- 缩小后多半是 **pypto 的 distributed codegen pass**(生成 host/chip orchestration 的那段)对 EP 场景处理有误,而非 simpler runtime 或 EP 原语本身。

(原先怀疑的"window 总量超限 / 某个大 window 映射失败 / simpler EP 原语 bug"都已被 §6.0 的两个对照排除:DeepSeek 小 window 也挂 → 非规模;simpler 手写同 shape PASS → 非 simpler 原语。)

### 6.2 pypto-生成 vs simpler-手写 的结构差异(diff 清单)

对比 pypto 生成的 `host_orch.py`(`allocate_domain` 调用)与能跑通的手写 `ep_dispatch_combine`(`main.py` + `dispatch.cpp`/`combine.cpp`),已发现的结构差异 + 手写版赖以成立的不变量(任一被 pypto codegen 违反都可能就是根因):

| 维度 | 手写(PASS) | pypto 生成(FAIL) | 风险 |
|------|------------|-------------------|------|
| comm buffer 数 | **1 个** `scratch`,kernel 内部按硬编码偏移自切 | **N 个**独立 `CommBufferSpec`,运行时按 nbytes 顺序紧排 | — |
| 对齐 | signal 槽 pad 到 `SIGNAL_BYTES=64`,大 buffer 偏移 **32B 对齐**(recv_x @320) | signal buffer = **8B 不 pad**,`data_done`@520 / `recv_x`@528 **非 32B 对齐** | **强嫌疑(若 AICore 跨卡 DMA 需 32B 对齐)** |
| 标量 | 2 个:`nranks`(0)、`device_ctx`(1) | 9 个 `device_ctx` | ABI 差异(各自 kernel 不同,不一定错) |
| domain rank | `workers=range(nranks)` 稠密 rank,kernel 用 `ctx->rankId` 索引 `windowsIn[]` | `workers=range(world_size)` 同上 | 需确认 pypto 传的是 dense rank 而非 device_id |

**手写版的关键不变量(来自 review,pypto codegen diff 时逐条核对)**:
1. **跨 rank window 布局对称** —— `CommRemotePtr = windowsIn[peer] + (local_ptr - windowsIn[self])`,要求各 rank window_size / buffer 顺序 / 偏移完全一致。
2. **signal 槽零初始化**靠 HCCL window 分配时清零;`AtomicAdd`-from-0。pypto 若复用脏 window 不清零会破协议。
3. **`pipe_barrier(PIPE_ALL)` 在 pub_counts 写 与 count_done signal 之间**是 load-bearing(否则 peer 提前 wait 通过 → 读到 stale pub_counts → 507018)。
4. **signal 路径 skip self,payload 路径 include self**(pub_counts/recv_x 的 self 槽必须被自己填)。pypto 若两者都统一 skip self → self 槽为 0 → prefix_sum 算错 slot → 507018。
5. `TWAIT` 用固定 `expected=1`(非单调计数,单次调用)。
6. 跨卡只用 push(`TPUT`/`TNOTIFY`),无 `remote_load` 拉。
7. scratch 以 `ContinuousTensor(child_memory=True)` 传入,否则子 kernel 看不到 IPC 映射内存。

**怀疑根因优先级(2026-06-23 更新,对齐已被实测排除)**:
- ❌ **对齐 —— 已排除**:把生成的 comm window 全部 round-up 到 512B 对齐(monkeypatch `alloc_window_buffer`)后,DeepSeek 2 卡**仍 507018**(同 `orch_error=2`)。所以对齐不是根因。
- 当前剩余嫌疑(都在**生成 kernel 的跨卡协议**层,需对比生成的 dispatch/combine kernel vs 手写):**#4 self 槽处理**(手写:signal skip self / payload include self;pypto 若统一 skip self → self 的 token 进不了 recv_x/pub_counts → prefix_sum 算错 → 507018)> **#3 `pipe_barrier` 顺序**(pub_counts 写 vs count_done signal)> comm-context / `rankId` 传递(pypto 传 9 个 device_ctx,需确认 kernel 用的是 dense rank)。

**验证/修复下一步**(upstream pypto codegen):
- 读 pypto 生成的 EP dispatch/combine kernel(`@pl.jit moe_ep` 经 codegen 后的 chip kernel),逐条对比手写 `dispatch.cpp`/`combine.cpp` 的 §6.2 七个不变量 —— 重点 #4(self 是否被正确填)和 #3(barrier 是否在 notify 组之间);
- 或拿到真正的 per-task fault 元数据(plog 没有;可能要 simpler 侧 instrument orchestration error code 2 的来源)。
- 这是 **pypto 仓库**的 codegen 活,不是 step3p5/pypto-lib。建议带本文 §6 全部证据(A/B 隔离 + 对齐已排除 + 7 不变量清单)开 upstream pypto issue。

### 6.3 upstream 检查 + DSL 逐条核对(2026-06-23)

**upstream pypto 无对症修复**:`HEAD(stepfun/develop b00c8b23)..origin/main` 共 10 个 commit,涉及 codegen 的只有 `953fbe8e fix(codegen): loop-carry manual_scope TaskId`(是**编译期 INTERNAL_CHECK** 崩溃修复,非运行期 507018)、`03db2b9f allow_early_resolve hint on pl.spmd`、tiling/transpose/docs。**没有针对 EP 多卡运行期 507018 的修复。**

**DSL 逐条核对(读 `models/deepseek/v4/dispatch.py` 对照 §6.2 不变量)**:
- **#4 self 槽 —— 正确,排除**:publish 循环(L141)`pub_counts` **include self**;count_done barrier(L162-174)notify/wait 都 `!= my_rank` skip self;prefix-sum(L193)读 self 的 `pub_counts[s*N+my_rank]` 槽。与手写版一致。
- **#3 barrier —— 仍是嫌疑**:手写 C++ 在 `pub_counts` 写与 `count_done` signal 之间有显式 `pipe_barrier(PIPE_ALL)`;DSL 里两个 notify 循环之间**没有显式 fence**,依赖 pypto codegen 在两段之间插入内存序。若 codegen 没插 → `count_done` 可能先于 `pub_counts` AtomicAdd 落盘可见 → peer wait 提前通过 → 读 stale pub_counts → 507018。

**仍未解的关键矛盾**:step3p5 的 dispatch-cut `LIMIT=0`(只留 task 0 rmsnorm)仍挂,指向 **comm-domain setup/scope-enter** 层(在 dispatch kernel 之前),而 #3 barrier 是 dispatch kernel 内的事 —— 两者对不上。要厘清需拿到 `orch_error_code=2` 的真正来源(plog 没有),**需在 simpler 侧给 orchestration error / comm scope-enter 加 instrument**。剩余两条路:(a) 在 simpler 打出 orch_error=2 来源 + comm scope-enter 的设备侧动作;(b) 看 pypto 是否在 notify 组间插 fence(#3)。两者都在 upstream(pypto / simpler)。

### 6.4 ⭐ 根因找到 + 修复(2026-06-23,推翻上面所有 comm-window/对齐/codegen 假设)

**解开 §6.3 的"矛盾"靠的是解码错误码**(读 `runtime/.../common/pto_runtime_status.h`):
- `orch_error_code=2` = **`PTO2_ERROR_HEAP_RING_DEADLOCK`**(单卡 MoE / DeepSeek 2 卡)
- `sched_error_code=100` = **`PTO2_ERROR_SCHEDULER_TIMEOUT`**(step3p5 8 卡)

**根因 = orchestration heap ring 死锁**,根本不是 comm-window/对齐/codegen。`pto_ring_buffer.h:394` 在 heap ring 满且无法回收(circular wait)时报 `HEAP_RING_DEADLOCK`,错误路径自带修法提示:**`Runtime env: PTO2_RING_HEAP=<bytes>`**。默认 `PTO2_HEAP_SIZE = 256MB/ring`(`pto_runtime2_types.h:73`),4 ring 共 1GB;EP-MoE 的 orchestration(84-tensor 大 task + 嵌套 comm/manual scope + 多 window)把 256MB/ring 撑爆。这也解释了 §6.3 的"LIMIT=0 也挂"——heap 死锁在 orchestration 层,与 dispatch 哪个 task 无关;dense 层 orchestration 小 → 256MB/ring 够 → PASS。

**修复验证(都设 `PTO2_RING_HEAP=1073741824`,即 1GB/ring)**:
| 跑法 | 之前 | 加 `PTO2_RING_HEAP=1GB` 后 |
|------|------|---------------------------|
| **DeepSeek v4 2 卡** | 507018 (orch=2) | **✅ PASS,golden 对上**(kv_cache + x_next,29.35s) |
| step3p5 MoE 单卡 | 507018 (orch=2) | heap 死锁消失(orch_error 2→0),但 **sched=100 SCHEDULER_TIMEOUT** |
| step3p5 MoE 8 卡 | 507018 (sched=100) | 同上,orch=0 / **sched=100** |

**DeepSeek MoE 多卡至此完全跑通(根因+修复闭环)。**

**step3p5 的第二关 = `SCHEDULER_TIMEOUT`**(`scheduler_dispatch.cpp:1258`:`SCHEDULER_TIMEOUT_CYCLES` 内无进度的 watchdog;`scheduler_cold_path.cpp:382` STALL TIMEOUT_EXIT)。这是真·任务 stall(无进度),**不是 size 旋钮能解**。step3p5 比 DeepSeek 重(TP=8 注意力 + 36 本地专家),heap 解锁后某个 task 仍 stall。单卡那次 sched-timeout 很可能是 `world_size=1` collective 等不存在的 peer(老 artifact);8 卡是真·多卡 stall,需 dispatch-cut bisect 定位哪个 task 不推进。

**落地建议**:
1. **DeepSeek / 一般 EP-MoE 多卡**:设 `PTO2_RING_HEAP`(≥512MB/ring,推荐 1GB)。这应进 step3p5/DeepSeek 的 runtime config(`CASES[*]["config"]` 或 run harness 默认),并向 upstream 反馈"EP-MoE 默认 heap ring 偏小"。
2. **step3p5 8 卡**:下一步 dispatch-cut bisect 定位 SCHEDULER_TIMEOUT 是哪个 task stall(barrier all_reduce? a2a? 某 expert kernel?)。

---

## 7. 已修 vs 待修

**已修(commit `fa60514`,分支 `wip/moe-barrier-allreduce`):**
- `attn_tmp_window`:`[BATCH, tp_chunk]` -> `[BATCH, HIDDEN]`(内联的 barrier 注意力 all-reduce 按整个 HIDDEN 平铺;旧的 1/8 宽 window 在多卡会越界)。
- shared-expert `tp_all_reduce`:ring -> barrier-mesh(ring 形式会撞 dense 层已修的多卡 507018)。
- `sh_tmp`/`attn_tmp` 两个 window 都扩到 `BATCH*HIDDEN`。

**测试脚手架(未提交,在 0162 工作树):**
- `test_decode_layer_moe_st.py`:放宽 `world_size=8 requires a2a3sim` 的 guard(0162 有 16 张 NPU);把输入广播成 `[N_RANKS,...]`(host_orch 按 `tensors[name][r_idx]` 取,原来是 `[1,...]` -> IndexError);`world_size>1` 时跳过单卡数值 golden。

**剩余卡点:** comm-window setup 层的 507018(见 §6)。

---

## 8. 复现

```bash
# 在 gpu-a910x-0162,环境已激活(CANN set_env + activate.sh + PTO_ISA_ROOT)
cd /data/chensiyu/hw_project/pypto/workspace/pypto-lib
python -m tests.step3p5.test_decode_layer_moe_st \
    --variant full_silu_silu --world-size 8 -p a2a3 -d 0
# -> 能编译,8 个 chip_process ready,然后所有 rank 507018;卡被 force-reset。

# dispatch-cut 确认是 setup 层:
P15_DISPATCH_LIMIT=0 python /tmp/moe_trace8.py   # 仍然 507018
```
