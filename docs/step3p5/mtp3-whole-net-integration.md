# Step3p5 MTP3 接入 N=1 整网：架构与 buffer 设计

> 基线：`whole_decode_faithful_real` @ `0e7a0fdd` 已在 0162
> devices 8..15 完成 P42 / native-W8A8 / token-6127 / argmax-303
> canonical closure。本文只在该基线上增加 MTP3，不改动已发布的 45 层
> main-network program 和它的 dispatch/combine 协议。

## 1. 不能跨越的模型边界

MTP3 不是把 layer 45/46/47 无条件接到 main layer 44 后面即可：

```text
main hidden
  -> main lm_head
  -> target sampler 产生第一个 token
  -> MTP layer 45(previous_hidden=main hidden, embed=first token)
  -> MTP45 shared_head argmax
  -> MTP layer 46(previous_hidden=MTP45 hidden, embed=MTP45 token)
  -> MTP46 shared_head argmax
  -> MTP layer 47(previous_hidden=MTP46 hidden, embed=MTP46 token)
  -> MTP47 shared_head argmax
```

主模型 logits 与第一个 MTP layer 之间存在 vLLM target sampler 边界。请求可能是
temperature/top-k/top-p，而不是永远 greedy；因此不能在 main program 内偷偷用
argmax 代替 target sampler。生产接法是：

1. `whole_decode_faithful_real` 保持一个 N=1 main-network program；
2. vLLM sampler 得到首 token；
3. 常驻的 `whole_mtp3` N=1 program 一次完成三个 MTP layer；
4. `whole_mtp3` 内部的 layer45→46→47 与 vLLM
   `enable_multi_layers_mtp` 一致，draft token 使用各 MTP shared-head 的 greedy
   argmax。

两个 program 属于同一 whole-model serving pipeline，但 sampler 语义边界明确，
不能为了字面上的“一个 launch”破坏模型定义。

## 2. MTP layer 边界

每个 MTP layer 是一个完整的 SWA+dense decoder block：

```text
previous_hidden [B,4096] BF16
draft_token_ids [B] INT32
  -> replicated embedding lookup
  -> enorm(embedding) + hnorm(previous_hidden)
  -> concat [B,8192]
  -> TP-row-sliced eh_proj + TP all-reduce
  -> SWA attention + TP all-reduce + residual
  -> dense MLP + TP all-reduce + residual
  -> shared_head norm + vocab-sharded logits
  -> TP global argmax
```

层间只交接：

- `previous_hidden [B,4096] BF16`
- `draft_token_ids [B] INT32`

attention 的 `resid1`、norm scratch、QKV、MLP 中间量都留在单层
Orchestration method 内，不跨 layer。

position metadata 以 attention 已使用的 `seq_lens` 为唯一真源：

```text
position[b] = seq_lens[b] - 1
```

vLLM Ascend 的 Step3p5 MTP patch 会在 `position == 0` 时把 token embedding
置零。因此 `whole_mtp3` 在 `active_mask[b] != 0 && seq_lens[b] == 1` 时不读取
token embedding，直接写零；不能把 canonical ctx=1 的 token embedding
错误地送进 `enorm`。

## 3. 权重与 dtype

0162 W8A8 checkpoint 的 `quant_model_description.json` 明确把
`model.layers.45..47.*` 标为 `FLOAT`，而 `mtp_layers.safetensors` 中 51 个
MTP tensor 都是 BF16。故：

- main-network routed MoE 继续使用原生 INT8 weight + FP32 scale，禁止 BF16
  dequant fallback；
- MTP attention/dense/shared-head 使用 checkpoint 原生 BF16，不是历史
  BF16-dequant 路径；
- MTP norm weight 在 IPC pool 中按 program ABI materialize 为 FP32；
- 所有 pool entry 仍保持 512-byte 起始地址对齐。

MTP 权重已包含在现有 per-rank IPC weight bundle 中，不复制第二份权重。
program 只为连续的 `[3,...]` tensor 建 flatten view：

| bundle shape | program view |
|---|---|
| `eh_proj [3,512,8192]` | `[1536,8192]` |
| `wq [3,4096,1536]` | `[12288,1536]` |
| `wo [3,1536,4096]` | `[4608,4096]` |
| `dense_gate/up [3,4096,1408]` | `[12288,1408]` |
| `dense_down [3,1408,4096]` | `[4224,4096]` |
| `shared_head [3,16112,4096]` | `[48336,4096]` |

flatten 只改变 `DeviceTensor` view，不搬运数据。

## 4. persistent buffer 与生命周期

### 4.1 hidden

```text
main_previous_hidden (input)
  -> mtp_hidden_out[:,0]   (layer45 output / layer46 input)
  -> mtp_hidden_out[:,1]   (layer46 output / layer47 input)
  -> mtp_hidden_out[:,2]   (layer47 final output)
```

输出 ABI 为 `mtp_hidden_out [TP,3,B,HIDDEN] BF16`。这三块空间同时承担
层间 handoff 和逐层精度观测，不再另分配内部 ping/pong，因此与“两个内部
buffer + 一个 final output”相比不增加 hidden 总占用。

### 4.2 logits / token

- `mtp_logits_out [TP,3,B,VOCAB_LOCAL] FP32`：保留三层 logits shard，供精度
  和 serving sampler/debug 使用；
- `draft_token_ids_out [TP,3,B] INT32`：每层 global argmax 后在所有 rank
  上复制；layer46/47 直接读取前一层 token slice。

### 4.3 KV

MTP 三层使用 distinct KV slice：

```text
mtp_k_cache [TP,3,KV_CACHE_ROWS,HEAD_DIM] BF16
mtp_v_cache [TP,3,KV_CACHE_ROWS,HEAD_DIM] BF16
```

层间绝不别名。standalone/canonical 先由 exporter 在同一个 IPC pool 中 carve；
live serving 再把三个 slice 映射到 vLLM MTP KV pool。

### 4.4 collective windows

每层有三次 TP all-reduce（eh_proj / attention / dense MLP）：

- scratch：`[B,4096] BF16`，每个 call-site 独立；
- signal：逻辑 `[8,1] INT32`，物理各占 512B；
- 三层合计 9 组 scratch + 9 个 512B signal。

每层 global-argmax 再分配：

- sparse rank-column max value window：逻辑 `[B,TP] FP32`，物理 512B；
- sparse rank-column token id window：逻辑 `[B,TP] INT32`，物理 512B；
- ready signal：逻辑 `[8,1] INT32`，物理 512B。

不跨层复用 control window，避免 AtomicAdd counter、remote read 和后继写发生
别名/竞态。

## 5. batch / pad / 初始化

- kernel 固定 `BATCH=16`；
- active row 由上层 metadata 决定，padding row 的 hidden/token/cache 初始值必须
  明确写零，不能依赖 allocator 残值；
- `active_mask [TP,16]` 是显式 ABI；embedding input、previous hidden 和每一层
  dense 输出都会在 layer 边界按该 mask 归零，避免 padding 值进入 shared-head
  或下一 MTP layer；
- attention 仍按固定 16 行执行，所以 runtime 必须为 padding row 提供
  `seq_lens >= 1`、独立且不与 active row 冲突的 `slot_mapping/block_table`
  cache 区域；禁止多个 row 共写同一 slot；
- embedding lookup、hidden、logits 和 token output 都覆盖全部 16 行；
- 单 batch 是 `row0 active + row1..15 zero-pad` 的特例；
- 多 batch 使用相同 ABI，不改变 weight/cache/window 地址布局。

## 6. 分层 TP global argmax

`VOCAB_LOCAL=16112` 的 `[16,16112]` logits 不能直接做一次
`row_argmax`：其 Vec 峰值超过当前约 188 KiB 上限；而 row-major
`[16,1]` 的 FP32/INT32 tile 每行只有 4B，也不满足 PTOAS 32B 对齐规则。
实现采用三层 reduction：

```text
16112 local logits
  -> 1007 x 16-token chunk candidates（pad 到 1008）
  -> 63 x 16-chunk group candidates（pad 到 64）
  -> 每 rank 每 row 一个 (value, global_token_id)
  -> 8-rank candidate gather
  -> global winner
```

每个 MTP layer 使用独立的 candidate value/id/ready window，三个 window
物理空间均为 512B。每个 rank 只写 `[B,TP]` candidate matrix 的自身列，
rank 间以 full-tile remote load + value maximum / id add 合并；这避免任何
不受 PTOAS 支持的 `[16,1]` ND↔ColMajor TLOAD。不跨层复用 counter 或 data
slot。`row_argmax/row_max` 产生的 `[B,1]` tile 也禁止直接用 `[b,0]`
scalar-read；必须先 reshape 成对齐的 `[1,B]` row tile，再从 `[0,b]`
读取。写入 row-major candidate table 的单列时同样不能直接 tile-store；
必须由 `[1,B]` 对齐 row 逐 batch scalar-write，避免当前 lowering 对 row>0
的 ColMajor 地址误判。

## 7. 验证顺序

1. `py_compile`、`git diff --check`；
2. MTP3 program TP=8 compile；
3. native checkpoint + IPC weight/KV device run，无 stall；
4. CPU torch ctx=1 MTP3 reference：逐层 hidden/logits/token 对齐；
5. 重新执行 `N1-CANONICAL-TEST.md` P42，确认 main-network 仍
   `rc=0 / RUN done / argmax=303`；
6. fresh exporter pool 下做 main P42 → sampler token 303 → MTP3 的整链 smoke；
7. 最终 live vLLM `step3p5_mtp + enable_multi_layers_mtp` A/B。

## 8. 统一整网 CI 入口

整网测试编排已收敛到：

```text
tests/step3p5/run_whole_network_ci.py
tests/step3p5/test_whole_network_ci.py
```

runner 不复制 main/MTP kernel harness，只负责 fresh exporter 生命周期、前后 8 卡
环境隔离、canonical main → sampler token → MTP3 顺序、batch1/batch16、CPU
reference、超时、日志、JSON report 与 finally cleanup。详细使用和 CI job 示例见：

```text
tests/step3p5/WHOLE_NETWORK_CI.md
```
