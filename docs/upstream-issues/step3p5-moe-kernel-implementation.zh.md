# step3p5 MoE 解码层 —— kernel 实现、融合结构与硬件要求

> 配套文档:[step3p5-moe-multicard-comm-window-setup.zh.md](step3p5-moe-multicard-comm-window-setup.zh.md)(多卡卡点)。
> 本文讲 **kernel 怎么实现的、哪些 math 被融合进同一个 kernel、每个 kernel 的硬件 buffer 预算和对应的 tiling 参数**。
> 说明:本会话**没有改动融合结构**(融合是 step3p5 既有设计,Phase 12 把 EpTpMoE 的子函数 inline 进 InCore kernel);本会话只改了 all_reduce(ring→barrier)和它的 window 尺寸。下面描述的是**当前代码里的实现**。

---

## 1. 硬件 buffer 层级(Ascend 910B / a2a3)

| buffer | 别名 | 容量 | 用途 |
|--------|------|-----:|------|
| **UB**(Unified Buffer) | Vec | 192 KB(**可用上限 188416 B ≈ 184 KB**) | 所有 AIV(向量)算子的工作内存:load/store tile、cast、add、softmax、rmsnorm… |
| **L1** | cube 输入缓存 | **512 KB** | cube(矩阵)算子从 GM 搬进来的 A/B 大块 |
| **L0A** | Left | **64 KB** | matmul 的 A-tile(左矩阵) |
| **L0B** | Right | **64 KB** | matmul 的 B-tile(右矩阵) |
| **L0C** | Acc | **128 KB** | matmul 的累加器(FP32 输出) |

**两类计算单元**:
- **AIC**(AI Cube):矩阵乘(q/k/v proj、qk/sv matmul、expert gate_up/down)。受 L1/L0A/L0B/L0C 约束。
- **AIV**(AI Vector):逐元素 / 规约(rmsnorm、rope、softmax、silu、cast、add、topk、collective)。受 **UB 188416 B** 约束。

> codegen 的 `AllocateMemoryAddr` pass 会静态校验每个 kernel 的 buffer 占用;超了就报
> `Vec buffer usage (N bytes) exceeds platform limit (188416 bytes)`(UB)或对应 L0/L1 报错。
> tiling 参数(下面各 `*_CHUNK`)的唯一目的就是把每个 kernel 的 **单次迭代工作集**压在这些上限内。

---

## 2. 一个 MoE 解码层的 kernel 序列(= 设备 task)

> **⚠ 先澄清一个常见误解:这一层不是「一个大 kernel」。**
> attention 和 MoE **没有**被融合进同一个 compute kernel。一层 = **~28 个各自独立下发的设备 task**,每个 task 都已经 tile 到很小(§3 的 `*_CHUNK` 就是把每个 kernel 的单次迭代工作集压在 §1 buffer 上限内)。`chip_orch` 只是**编排(orchestration)**,按顺序把这些小 kernel 依次下发。
> 所以**当前的卡点不是「kernel 太大」**——没有任何单个 compute kernel 过大。卡点是 orchestration 在跑 task 之前给这一层分配的**跨卡 comm-window**(见 §2.0)。

### 2.0 🟥 当前故障点 —— 在所有算子「之前」,不对应表里任何一行

dispatch-cut 二分显示:**LIMIT=0(只保留 task 0 rmsnorm,其余全砍)在 8 卡上仍然 507018**。说明故障**不对应下表任何一个算子**,而是发生在 task 调度**之前**的 **comm-window 分配 / 跨卡映射(allocate_domain)阶段** —— orchestration 要为这一层建立 13 个跨卡 window(尤其下表 🟧 标记的 EP dispatch/combine 用的两块 8MB `recv_x`/`send_x`,共 16MB)。这一步在 8 卡规模上挂,与具体跑哪个算子无关。详见 [step3p5-moe-multicard-comm-window-setup.zh.md](step3p5-moe-multicard-comm-window-setup.zh.md)。

### 2.1 算子序列(按颜色分类)

颜色图例:🟩 AIV 向量/规约 · 🟦 AIC cube 矩阵乘 · 🟪 TP collective(all_reduce)· 🟧 EP collective(dispatch/combine a2a,**用到那两块 8MB 大 window,是 §2.0 故障的嫌疑窗口来源**)

| task | 类别 | kernel | 融合进这个 kernel 的 math |
|------|------|--------|--------------------------|
| 0 | 🟩 | `full_rmsnorm_zc` | 输入 RMSNorm:`x*rsqrt(mean(x²)+eps)*(w+1)`(零中心) |
| 1-3 | 🟦 | `full_{q,k,v}_proj` | 三个独立 proj matmul:`normed @ w{q,k,v}` |
| 4 | 🟩 | `full_qk_norm_zc` | Q、K 各自 head 维 RMSNorm,**两个 norm 融合在一个 kernel** |
| 5 | 🟦 | `full_gate_proj` | head-gate 权重投影(gate 本身已旁路为 ×1) |
| 6 | 🟩 | `full_rope_kv_cache` | **RoPE 旋转 + 写 KV-cache 融合**:cos/sin 旋转 Q/K,再把 K/V 写进 paged cache 槽位 |
| 7 | 🟦 | `full_qk_matmul` | `Q @ Kᵀ` 注意力打分 |
| 8 | 🟩 | `full_softmax` | softmax(max 归约 + exp) |
| 9 | 🟦 | `full_sv_matmul` | `softmax @ V` |
| 10 | 🟩 | `full_online_softmax` | **flash-attn 在线 softmax**:跨 KV block 的 running max/sum/output 重缩放融合 |
| 11-12 | 🟦🟩 | `full_out_proj_matmul` + `_cast` | o-proj `attn @ wo`(per-rank 部分和)+ BF16 cast |
| **13** | 🟪 | **`tp_all_reduce`** | **注意力 TP 组求和**(barrier-mesh:stage→notify/wait→remote_load+加;本会话 ring→barrier 改这里) |
| 14 | 🟩 | `full_out_resid_add` | `resid1 = current_hidden + o`(残差) |
| 15 | 🟩 | `moe_post_rmsnorm_zc` | attn 后 RMSNorm |
| 16 | 🟩 | `gate_topk` | **路由器融合**:`gate matmul(x@gate_w)` + sigmoid + router_bias + **top-8 选择(sort32)** |
| **20** | 🟧 | `dispatch_step` | **dispatch 融合**:直方图 + 前缀和 + pack + **EP all-to-all 发送(用 `send_x`/`recv_x` 8MB×2)** + 本地专家 CSR/逆映射 |
| (routed) | 🟦🟩 | routed expert MLP | **每个本地专家融合 gate_up matmul + SiLU + down matmul**(36 个本地专家循环) |
| **19** | 🟪 | `tp_all_reduce` | shared-expert / gate 的 TP 求和(本会话 ring→barrier 改这里) |
| (shared)| 🟦🟩 | shared expert MLP | shared expert `gate_up + SiLU + down`(TP 切分)+ all_reduce |
| 23-24 | 🟩 | `_publish_src_route_table` + `pub_route_barrier` | combine 记账 + barrier |
| **25** | 🟧 | `_push_routed_y_to_sources` | **EP all-to-all 回传**(把专家输出送回 token 源 rank;用 `routed_y`/`src_route` window) |
| 26 | 🟩 | `moe_combine` | **加权 gather**:按 top-K 权重把各专家输出累加 |
| 27 | 🟩 | `moe_residual_add` | `next_hidden = resid1 + routed_out + shared_out` |

**pypto 融合机制(澄清「融合」的粒度)**:`@pl.function(type=Inline)` 的子函数(`_gate` / `_histogram_and_prefix_sum` / `_pack_send_payload` / `ep_all_to_all` / `_expert_routed` / `_expert_shared_local` / `_weighted_gather_and_add` …)在 build 时被**拍扁(inline)进调用它的 `InCore` kernel**,共享同一块 UB —— 这是 kernel **内部**的算子融合(比如 rope+kv_cache 写进一个 kernel)。`@pl.function(type=InCore)` 的才是一个独立下发的设备 task。**attention 与 MoE 之间是 orchestration 编排,不是 kernel 内融合** —— 它们始终是分开的 task。

### 2.2 DeepSeek v4 也是这样吗?(以及它会不会也卡)

**结构一样**:DeepSeek v4 的 `decode_layer`(`models/deepseek/v4/decode_layer.py:99`)也是先调 `attention_*(...)` 再调 `moe_ep(...)` —— 同样是 orchestration 把 attn + MoE 编排在一层,**也没有把两者融进同一个 compute kernel**。结构差异如下:

| 维度 | DeepSeek v4 | step3p5 |
|------|-------------|---------|
| 并行规模 | **EP=2**(`assert N_RANKS == 2`) | **TP=8 / EP=8** |
| 注意力 | **DP**(数据并行,**无 TP all_reduce window**) | **TP=8**(有 attn/sh all_reduce window) |
| dispatch 精度 | **INT8/FP8 量化**(`recv_x` INT8 + 单独 scale/weight 小 buffer,**无独立 `send_x`**) | **BF16**(`recv_x` + `send_x` 各 8MB) |
| 跨卡 window 足迹 | 小、少 | **~17.4MB/卡,13 个** |

**⭐ 但实测对照(2026-06-23)推翻了"DeepSeek 因窗口小所以没事"的猜测:**
实跑 `python models/deepseek/v4/decode_layer.py -p a2a3 -d 0,1`(2 卡真机)→ **同样 507018**,且 dispatch-cut **LIMIT=0 仍挂**(setup 层),和 step3p5 同一家族。也就是说 **DeepSeek 这套小很多的 EP-MoE 窗口、2 卡、INT8,照样在真机多卡 setup 挂**。

→ **结论改写**:这**不是 window 大小/规模问题**,而是 **EP all-to-all(dispatch/combine)的 comm 路径在真机多卡 setup 阶段对两个模型都挂**的共性问题(很可能 simpler 上游);而**纯 TP 的 dense 层(只有 all_reduce window)在 8 卡 PASS**。所以"改 INT8 / 缩窗口"**不会**解决——DeepSeek 已经是 INT8 小窗口还是挂。真正要查的是 simpler 的 **EP comm 原语**在真机多卡。详见 [step3p5-moe-multicard-comm-window-setup.zh.md §6.0](step3p5-moe-multicard-comm-window-setup.zh.md)。

---

## 3. 各 kernel 的 tiling 参数 + 硬件预算(标注)

下面每个 `*_CHUNK` 都是为了把对应 buffer 压在 §1 的上限内。**K**=matmul 收缩维,**N**=输出维。

### 3.1 注意力(AIC matmul)

| 参数 | 值 | 约束的 buffer | 计算 |
|------|---:|--------------|------|
| `INPUT_PROJ_K_CHUNK` | 256 | L0B | q/k/v proj 的 K 分块 |
| `KV_PROJ_K_CHUNK(_LOCAL)` | 128 (TP=1) / 256 (TP=8) | L1 | TP=1 时 `KV_HIDDEN_LOCAL=1024`,用 256 会让 cube 输入 `1024*256*2=512KB` 顶到 L1 上限 → 折半到 128 |
| `Q_OUT_CHUNK` / `K_CHUNK` | 256 / 256 | L0C / L0B | q 输出分块 / FA 的 KV SEQ_TILE(=BLOCK_SIZE 128 的倍数) |
| `OUT_PROJ_K_CHUNK` / `_N_CHUNK` | 256 / 256 | L0B / L0C | o-proj 分块 |

### 3.2 路由器 gate_topk(小 matmul + topk,在 AIV)

| 参数 | 值 | 约束 | 计算 |
|------|---:|------|------|
| `ROUTER_GATE_K_CHUNK` | 256 | UB | gate matmul 的 K 分块;`x0/xk [16,256] FP32 = 16384 B` 压在 Vec 预算内(512→256 就是为此) |
| `ROUTER_GATE_N_CHUNK` | 32 | L0B | gate matmul 的 N 分块;`[K=256,N=32] FP32 = 32768 B` 在 L0B(64KB)内 |

(gate 输出 288 个专家 logits → top-8 选择用 sort32 硬件原语)

### 3.3 routed expert MLP(AIC matmul,每个本地专家)

| 参数 | 值 | 约束 | 计算 |
|------|---:|------|------|
| `ROUTED_GATE_K_CHUNK` | 64 | L0A | A-tile K 维;`[32,64] BF16 = 4096 B` |
| `ROUTED_GATE_N_CHUNK` | 64 | L0B | B-tile N 维;`[64,64] BF16 = 8192 B` |
| `ROUTED_DOWN_K_CHUNK` | 64 | L0A | down 的 A-tile K |
| `ROUTED_DOWN_N_CHUNK` | 128 | L0B | down 的 B-tile N;`[64,128] BF16 = 16384 B` |

(routed 专家 hidden = `MOE_INTERMEDIATE = 1280`;每 rank 36 个本地专家)

### 3.4 shared expert MLP(AIC matmul,TP 切分)

| 参数 | 值 | 约束 | 计算 |
|------|---:|------|------|
| `SHARED_GATE_K_CHUNK` | 256 | L0B | |
| `SHARED_GATE_N_CHUNK` | `INTER_S_LOCAL = 160` | L0C | **⚠ chunk 跟 slice 走**:一个 N tile 覆盖整个 `160` 切片;TP=1 时会变 1280 → 爆 L0C(见 §4 警告) |
| `SHARED_DOWN_K_CHUNK` | `INTER_S_LOCAL = 160` | L0B | 同上,也是 chunk 跟 slice 走 |
| `SHARED_DOWN_N_CHUNK` | 256 | L0C | |

(shared expert hidden = `SHARE_EXPERT_DIM = 1280`,TP=8 切到 `INTER_S_LOCAL = 160`)

### 3.5 all_reduce(AIV,本会话改过)

| 参数 | 值 | 约束 | 计算 |
|------|---:|------|------|
| `ar_chunk` | `HIDDEN // 8 = 512`(固定,**不跟 tp_size 走**) | UB | barrier-mesh 把 `[BATCH,HIDDEN]` 按 512 列分块;`acc [16,512] FP32 = 32768 B` 在 UB 内。**本会话的修复点**:原来用 `tp_chunk=HIDDEN//tp_size`,TP=1 时塌成 4096 → `[16,4096] FP32 = 256KB` 爆 UB |

---

## 4. ⚠ tiling 的两种风格(写新 kernel 必读)

step3p5 里 chunk 常量有两种风格,踩过坑:

| 风格 | 写法 | TP=8 切片(如 160) | TP=1 unslice(如 1280) | 例子 |
|------|------|-----|-----|------|
| **固定 chunk(推荐)** | `MLP_OUT_CHUNK = 128`(常量) | 不爆 | 不爆(仍按 128 迭代) | dense MLP、`ar_chunk` |
| **chunk 跟 slice 走(坑)** | `SHARED_GATE_N_CHUNK = INTER_S_LOCAL` | `160` 一 tile 覆盖 | `1280` 一 tile → **爆 L0C** | shared expert gate/down |

本会话修的 all_reduce 就是从"跟 slice 走"改成了固定 `ar_chunk = HIDDEN//8`。shared expert 的 `SHARED_GATE_N_CHUNK / SHARED_DOWN_K_CHUNK` 仍是"跟 slice 走"风格,在 TP=8 单卡(切片 160)没问题,但 TP=1 unslice(1280)会爆 —— 这是 Phase 17 prefill MoE L1 overflow 的同类隐患。

---

## 5. 关键 shape / 容量参数汇总

| 量 | 值 | 来源 |
|----|---:|------|
| HIDDEN | 4096 | 模型 |
| HEAD_DIM | 128 | 模型 |
| BATCH(kernel 级) | 16 | 模型 |
| NUM_HEADS_FULL / NUM_KV_HEADS | 64 / 8 | 模型 |
| 每 rank(TP=8):Q 头 / KV 头 | 8 / 1 | NUM_HEADS_FULL_LOCAL / KV_HEADS_LOCAL |
| MOE_NUM_EXPERTS / 每 rank 本地专家 | 288 / 36 | EP=8 切分 |
| MOE_TOP_K | 8 | 路由 top-K |
| MOE_INTERMEDIATE(routed expert hidden) | 1280 | 模型 |
| SHARE_EXPERT_DIM / 每 rank 切片 | 1280 / 160 | TP=8 切分 |
| dispatch 收发缓冲行数 `LOCAL_RECV_MAX` | `n_ranks*BATCH*TOPK = 1024` | 最坏情况收 token 数 |
| combine 路由数 `n_routes_per_rank` | `BATCH*TOPK = 128` | |

> kernel 完整 I/O(26 个 host 张量 + 13 个跨卡 window)的逐个 shape/dtype/切分,见配套文档
> [step3p5-moe-multicard-comm-window-setup.zh.md §3](step3p5-moe-multicard-comm-window-setup.zh.md)。
