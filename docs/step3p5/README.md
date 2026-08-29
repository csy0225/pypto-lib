# Step3p5 single-chip hidden-only canonical 入口

本文是 Step3p5 vLLM + PyPTO 集成的唯一 active 设计、实现和验证入口。
旧 per-layer decode、旧 whole-MTP3、PyPTO 内部 logits/token 输出、多套
generator 和历史进度文档均已移除，不能再作为方案或定位依据。

## 1. 唯一生产方案

### Main 45 层

```text
vLLM token embedding + paged-KV metadata
  -> PyPTO single-chip 45-layer whole program
  -> pre-final-norm BF16 hidden
  -> vLLM final RMSNorm + LM head + target sampling
```

唯一 program：

```text
models/step3p5/decode_fwd.py
whole_decode_step3p5
```

历史 45× unroll Main、rollback selector 和自定义 Main module/name 参数均已
删除。holder、sidecar、harness 和 CI 只允许上述 canonical symbol。

### MTP45/46/47

```text
Main hidden + vLLM sampled token
  -> PyPTO selected MTP45 hidden
  -> vLLM shared head + draft sampling
  -> PyPTO selected MTP46 hidden
  -> vLLM shared head + draft sampling
  -> PyPTO selected MTP47 hidden
  -> vLLM shared head + draft sampling
  -> vLLM acceptance/rejection
```

唯一 program family：

```text
models/step3p5/mtp_hidden_fwd.py
MTP_LAYER_HIDDEN_PROGRAMS
```

Main/MTP 的后处理边界完全一致：PyPTO 只返回 raw hidden。final norm、
LM head、target/draft sampling、accept/reject 和用户可见后处理全部由 vLLM
完成。MTP 请求携带 vLLM 已采样的 token id，selected MTP body 内只执行
该层规定的 embedding、input projection、attention 和 dense MLP。

## 2. 生产依赖闭包

```text
models/step3p5/
  decode_fwd.py
  dense_mlp.py
  mtp_hidden_fwd.py
  attention_full.py
  attention_swa.py
  moe.py
  gate.py
  dispatch.py
  expert_routed.py
  expert_shared.py
  combine.py
  collectives.py
  _ops.py
  config.py
  weight_loader.py

tools/step3p5/
  whole_decode_holder.py
  mtp_layer_holder.py
  whole_decode_sidecar.py
  vllm_monkey_patch.py
  pypto_model_loader.py
  vllm_decode_metadata.py
  vllm_mtp_metadata.py
  kv_padding.py
  ipc_session.py
  pypto_weight_ipc.py
  pypto_kv_ipc.py
  pypto_mtp_kv_ipc.py
  main_kv_exporter.py
  mtp_kv_exporter.py
```

安装器只接受 `PYPTO_STEP3P5_PATCH_MODE=full`。real decode 请求若不满足
canonical ABI 必须 fail-closed；禁止 vanilla、per-layer 或 silent fallback。
只有明确识别的 profile/dummy/warmup call 可以执行 harmless no-op。

## 3. 设计不变量

### Native W8A8

- routed expert weight 为 INT8，scale 为 FP32；
- routed input 和 clamped SwiGLU 中间激活使用 per-token INT8
  quant/requant；
- shared expert 保持 BF16；
- hidden-only decode 的 router、shared gate 和 shared up 权重保持 checkpoint
  原生 `[N, K]` layout，并以 `b_trans=True` 执行；
- native decode ABI 使用 `moe_gate_w_nk`、`moe_w_gate_s_nk` 和
  `moe_w_up_s_nk`，同一个 bundle 不得同时保存 legacy/native 两套 layout；
- default/prefill loader 仍保持 legacy layout，只有 production hidden-only
  decode 显式选择 native contract；
- 禁止 BF16 dequant/fallback；
- RMSNorm EPS 为 `1e-5`；
- router bias 按 vLLM 的 BF16 round 语义对齐。

### KV ownership 和 fixed-storage batch ABI

这里的“固定 16”只指 **program tensor 的 storage shape / kernel tile ABI**，
不表示 vLLM scheduler 的实际请求数固定为 16，也不表示每一步都按 16 个
有效 token 做完整的 MoE routing。pure-decode 请求每个 request 贡献一个
当前 token，因此一次调用的逻辑有效行数为：

```text
1 <= valid_tokens == valid_requests <= 16
storage_batch = 16
```

holder 保留前 `valid_tokens` 行的真实 hidden 和 scheduler metadata，并执行：

```text
current_hidden[valid_tokens:16] = 0
num_tokens_per_owner[0:tp] = valid_tokens
```

当前 G1 实现已让 gate、local route pack、routed combine 等 MoE row-wise 阶段只处理
前 `valid_tokens` 行；但 attention、dense/TP scratch 和部分固定 tile 仍保留
`[16, ...]` storage/compute 形状。尤其 inactive row 当前仍可能执行 attention
的 KV write，所以 padding metadata 不能省略，也不能指向 scheduler 正在管理
的 block。sidecar 最终只返回输出的前 `valid_tokens` 行。

#### Replicated-input local-owner MoE scheduling

Each MoE layer starts after attention has already reduced the TP partials.
Every TP rank therefore has the same complete hidden state and runs the same
router. The decode graph does not copy that activation through an EP dispatch
exchange. Each rank keeps only routes whose global expert belongs to its
contiguous 36-expert shard:

```text
owner(eid) = eid // 36
local_e    = eid - owner(eid) * 36
out_row    = local_e * 128 + cursor[local_e]
```

Let:

```text
T = clamp(num_tokens, 0, 16)
M = ceil(T / 16)
R = T * TOPK
A = number of local experts that receive at least one route on this rank
E = n_local_experts = 36
```

The 128-row slab for each local expert is the worst-case `16 * TOPK` route
capacity. It preserves the packed-NZ external-kernel ABI while ensuring each
`(token, topk-slot)` route maps to exactly one owner row. Non-owner ranks leave
that route's `local_route_row` entry at `-1`.

Router and shared-expert compute scale with active M tiles rather than a fixed
chip-wide task count:

| Compute stage | Runtime grid |
|---|---:|
| router active-row xg precompute | `T` |
| router expert fanout | `M * 18` |
| shared gate / up / activation | `M * 5` each |
| shared down workers | `M * 2` |

Cube still executes its minimum 16-row tile for BS1; this is dynamic task
ownership, not a separate GEMV specialization. The layout, dependency, and
critical-path rationale is recorded in
[moe-layout-and-critical-path.md](moe-layout-and-critical-path.md).

The active per-rank MoE chain is:

Gate routes are published as complete `[16, 8]` INT32/FP32 tiles. Local expert
counts use a 40-entry physical INT32 ABI, while the route plan uses 96 entries.
Only count entries `0:36` and plan entries `0:38` are semantic. Plan entries
`64` and `80` are the gate/up-ready and hidden-ready counters; the other slots
in `38:96` are zero padding that keeps metadata and both counter cache lines
disjoint for every 4-byte-aligned base. The producer zeroes both complete
tensors.

For `A = min(active_local_experts, 36)`, the gate producer count is
`G = min(22, 10A)`, the activation producer count is `H = min(22, 5A)`, and
the quant consumer count is `Q = min(22, A)`. Only AIC blocks `b < G` publish
gate/up completion; only lane-0 AIV blocks `b < H` await gate/up and publish
hidden completion; only lane-0 AIV blocks `b < Q` await hidden completion.

| Swimlane task | Runtime shape | Function |
|---|---:|---|
| `local_route_map_init` | one CORE_GROUP task | Build route sentinels, owner-local fixed-slab row mappings, and counts with a single GM metadata writer. |
| `local_route_pack` | owner-local expert grid | Copy payload, scale, and route weight into disjoint fixed expert slabs. |
| `local_route_plan` | one CORE_GROUP task | Build the compact active-expert plan after all local payload writers complete. |
| `routed_nz_gmm1_swiglu_quant` | 22 AIC / 44 AIV resident mixed grid | Compute gate/up, activation, and requantization; only workers with active expert work participate in the two producer latches. |
| `routed_nz_down` | 23 workers at BS1, 22 otherwise | Compute the owner-local weighted routed partial. |
| `local_combine_reduce` | fixed storage grid of 16 | Add the TP-sharded shared partial and this rank's routed rows in FP32. |
| `tp_all_reduce` | one TP collective | Sum every rank's shared and routed partial into the complete MoE output. |
| `moe_residual_add` | one control task | Add the replicated post-attention residual after the collective. |

The layer invariant is:

```text
rank_partial = shared_tp_partial + sum(owner-local routed outputs)
moe_output   = TP_AllReduce(rank_partial)
next_hidden  = post_attention_residual + moe_output
```

A rank with no routed rows still computes its shared TP partial and executes
`local_combine_reduce`, `tp_all_reduce`, and `moe_residual_add`. All ranks
therefore enter the same final collective even under maximally skewed routing.
The old EP payload/meta/combine windows, completion credits, and `moe_epoch`
are not part of the canonical decode ABI.

This protocol is scoped to the canonical hidden-only decode graph
(`WholeDecodeStep3p5`) and the focused five-layer programs composed from it.
Prefill, the standalone `EpTpMoE` path, and `_compile_moe.py` retain their
existing distributed-EP behavior and are not covered by this contract.

The local-owner numerical contract is also distinct from the legacy r10
distributed-EP reduction tree. Each rank casts its local shared+routed partial
to BF16 before the final TP all-reduce, so a legacy golden is not a bit-exact
oracle for this graph even though the mathematical sum is equivalent. Formal
goldens for this path must declare:

```text
source_kind = local-ep
protocol_profile = replicated_input_local_owner
numeric_contract.name = local_owner_partial_tp_all_reduce_bf16_v1
numeric_contract.comparison = bit_exact_to_protocol_golden
numeric_contract.bit_exact = true
```

#### KV ownership 与物理布局

live 路径的 KV allocation 由 vLLM allocator 拥有。Main holder 在
`prepare` 阶段只做一次 IPC import 和 stacked view 构造，之后把 K/V 以
`pl.InOut` 传给 repeated `rt.run()`；每步不得重建、整池拷贝或清空历史 KV。
standalone device gate 使用 exporter 代替 vLLM 成为 allocation owner，但
消费的 map/layout 契约相同。

每个 TP rank 有两个相互独立的 allocation domain：

```text
Main: K-major/V-major，45 个 decoder layer
MTP:  K-major/V-major，MTP45/46/47 三个 selected layer
```

Main 与 MTP 不共享 pool base、IPC key、map 或 padding reserve。`slot_mapping`
只表示**单层 KV section 内**的逻辑 slot，不包含 layer base；当前 layer 的
K/V section offset 由 imported map/stacked view 提供，不能把 layer offset
再次编码进 `slot_mapping`。

#### Scheduler block 与 padding reserve

设 vLLM scheduler 可见 block 数为 `S=scheduler_num_blocks`，实际 allocation
容量为 `P=physical_num_blocks`。产品 ABI 要求：

```text
scheduler-owned block ids: [0, S)
padding reserve block ids: [S, S + 15)
physical capacity:         P >= S + 15
```

这里的 15 是 `storage_batch - 1`：因为一次合法调用至少有一个 active row，
最多只需要 15 个 inactive-row slot。若 `valid_tokens=16`，本轮不消费 reserve；
若物理 allocation 大于 `S+15`，多出的尾部容量也不属于 scheduler domain，
不能据此改变上述固定 reserve ID。

对每个 active row，vLLM 提供的 `seq_lens`、`block_table` 和 `slot_mapping`
原样保留并严格校验：

```text
position = seq_len - 1
table_col = position // 128
block_id = block_table[row, table_col]       # 必须位于 [0, S)
slot_mapping[row] = block_id * 128 + position % 128
```

对 `row >= valid_tokens` 的 inactive row，control plane 必须设置：

```text
seq_len = 1
position = 0
block_table[row, 0] = 一个本轮未重复使用的 reserve block id
slot_mapping[row] = block_table[row, 0] * 128
```

因此 reserve 是为 fixed-storage attention 写入提供的地址隔离，不是 active
request 的 KV 容量，也不是“把有效 batch 固定为 16”的手段。Main/MTP 虽使用
同一套 metadata 语义，但必须在各自独立的 KV allocation 中拥有对应 reserve。

### 两个独立 gate

数值正确和无 stall 必须分别通过。`RUN_CLEAN`、一次 token 正确、
compile-only、P1/P20、随机输入或 BF16 fallback 都不能替代完整准出。

#### Five-layer chip-swimlane gate

The focused L0-L4 DFX gate is launched from the repository root:

```bash
PYPTO_UPGRADE_WORKSPACE=/path/to/run-root \
CKPT=/path/to/checkpoint \
bash deployment/docker/run_swimlane_gate.sh <image-id-or-digest>
```

A complete capture contains exactly one raw record for every rank:

```text
dfx_outputs/rank{0..7}/d0/chip_swimlane_records.json
```

Both `chip_swimlane_records.json` and its top-level
`chip_swimlane_level` field are runtime-owned schema names.

Candidate publication also requires the exact local-route sidecar produced
by `tests.step3p5.harnesses._stage_five_layer_moe_route`. Supply it through
`PYPTO_RECV_META_SIDECAR`; task, block, and physical-slice counts are not valid
route-histogram substitutes. The analyzer checks both the live source hash and
the sidecar source hash against the selected frozen profile. The active
`local-ep` profile is bound to the complete `decode_fwd.py` SHA in the analyzer
policy; matching only an abbreviated prefix is rejected.

The active expert family is `packed_nz_mixed`. DFX maps the two layer-local
external tasks through the `local_route_pack .. local_combine_reduce` window
and validates the chain through the per-layer MoE `tp_all_reduce` and residual
task. The analyzer does not require or infer the retired dispatch/combine wait
stages for this profile.

Generate the formal golden and route sidecar from the same immutable final
image before running the swimlane gate. The formal writer requires
`--context-len=65536`, a digest-qualified image reference, a non-empty run ID,
and bit-exact output:

```bash
PYPTO_IMAGE_DIGEST='registry/image@sha256:<64-hex>' \
PYPTO_SOURCE_RUN='final-image-formal-bs1-64k' \
python -m tests.step3p5.harnesses._stage_five_layer_moe \
  ... --active-batch 1 --context-len 65536 --num-blocks 512 \
  --write-golden /out/formal-golden

python -m tests.step3p5.harnesses._stage_five_layer_moe_route \
  ... --active-batch 1 --context-len 65536 --num-blocks 512 \
  --image-digest 'registry/image@sha256:<64-hex>' \
  --golden-dir /out/formal-golden
```

The route reader rejects the golden before device exporters start unless its
manifest is `step3p5.five-layer-moe-golden.v3`, `bit_exact=true`, bound to the
same immutable image, and its full decode SHA equals the live route source.
The analyzer again requires the sidecar route source, formal-golden source,
live DFX source, and frozen `local-ep` SHA to agree exactly. A golden from an
older image digest or source tree must not be relabeled or reused.

The gate uses separate dependency-generation and timing submissions. Its
instrumented makespans explain critical-path composition but are not clean ITL
measurements; use the independent ITL gate for absolute latency. The analyzer
requires every executable dependency task to have physical slices or one
explicit runtime `predicated_skip` event; duplicate, unknown, or conflicting
skip evidence still fails closed. Dependency-chain, physical-slice, and
admission contract violations also fail closed. Eight raw records with a
nonzero process status are
evidence from a failed gate and must never be relabeled as an RC0 result.

### 512B control-signal stride 的作用域

DeepSeek v4 的 512B 主要用于 data tile、L2 cache line 和 MTE 性能对齐，
不是通用 control-signal ABI。step3p5 只对 canonical 中满足以下条件的
control signal slot 做 512B 物理 stride 隔离：

1. 被 `notify` / `wait` / `AtomicAdd` 使用；
2. 多个 layer/slot 位于同一个 backing buffer，或同一个 slot 跨 repeated
   collective invocation 复用。

这些 slot 使用：

```text
COMM_CONTROL_SIGNAL_BYTES = 512
COMM_SIGNAL_STRIDE_I32 = 128
formal/window/slice shape = [128, 1] INT32
```

通信 loop 仍只访问前 `n_ranks` 行。普通 data window 和 MTP 每次调用独立
分配的 compact signal 不按该本地 false-sharing 约束机械扩成 512B。

## 4. single-step 与 multi-step

| 状态 | single-step | multi-step 中的变化 |
|---|---|---|
| token/embedding | 可固定输入 | 上一步 vLLM sampler 决定下一步 token/embedding |
| `seq_lens` | 固定一次 | 正常 decode 每轮递增 |
| `positions` | `seq_len-1` | 跟随当前 token position |
| `block_table` | 当前已分配 blocks | 跨 block boundary 时变化 |
| `slot_mapping` | 当前 token 写入 slot | 每轮写入不同 slot |
| KV | 首次写 K/V | 下一轮读取历史 K/V 并追加当前 K/V |
| weights | resident | 不应随 step 改变 |
| collective signal | 一次执行 | repeated `rt.run()` 必须证明每轮 fresh/zero |
| sampling 参数 | 不进入 PyPTO ABI | 通过 sampled token 间接改变下一步输入 |

canonical diagnostic sampling：

```json
{
  "temperature": 0,
  "top_p": 1,
  "top_k": -1,
  "min_p": 0,
  "seed": 0
}
```

若 step1 正确、step2 错误，按以下顺序定位：

1. 对齐 step2 sampled token、embedding 和 sampling 参数；
2. 对齐 `seq_lens`、`positions`、request/token ordering；
3. 对齐 step1 KV 写入 slot 与 step2 KV 读取 block；
4. 检查 resident temporary buffer、IPC lifetime、collective signal；
5. 最后检查 layer-specific weight row 和计算。

step1 使用相同 resident weights 正确时，通用权重加载错误优先级较低；
仍需排除 MTP45/46/47 row 选择、KV slab、RoPE/position 分支和 buffer alias。

## 5. 0162 标准环境

保护 cards 0–7 的 vanilla vLLM oracle，只使用 cards 8–15：

```bash
set +u
source /usr/local/Ascend/cann/set_env.sh
source /data/chensiyu/hw_project/pypto/workspace/activate.sh
set -u

export PTO_ISA_ROOT=/data/chensiyu/hw_project/pypto/workspace/pto-isa
export PTO2_RING_HEAP=4294967296
export PTO2_RING_TASK_WINDOW=131072
export PTO2_RING_DEP_POOL=131072

cd /data/chensiyu/hw_project/pypto/workspace/vllm-pypto
export CKPT=/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp
```

## 6. Canonical device gate

Main 8-step：

```bash
python -m tests.step3p5.harnesses._stage_main_hidden_only \
  --device 8,9,10,11,12,13,14,15 \
  --out /data/chensiyu/hw_project/pypto/workspace/logs_n1_0162/main-hidden-final \
  --ckpt "$CKPT" \
  --steps 8
```

固定 diagnostic tokens：

```text
[303, 1207, 19384, 872, 428, 6127, 4231, 2636]
```

必须检查：8-step token exact、每步 hidden artifact、finite、TP spread=0、
无 stall/507018/dmesg fault，且 ownership 为
`pre-final-norm BF16 next_hidden only`。

MTP batch1 和 batch16：

```bash
python -m tests.step3p5.harnesses._stage_mtp_hidden_selected \
  --device 8,9,10,11,12,13,14,15 \
  --out /data/chensiyu/hw_project/pypto/workspace/logs_n1_0162/mtp-hidden-final \
  --ckpt "$CKPT" \
  --previous-hidden /data/chensiyu/hw_project/pypto/workspace/logs_n1_0162/main-hidden-final/main_step00_hidden.pt \
  --active-batch 1

python -m tests.step3p5.harnesses._stage_mtp_hidden_selected \
  --device 8,9,10,11,12,13,14,15 \
  --out /data/chensiyu/hw_project/pypto/workspace/logs_n1_0162/mtp-hidden-batch16 \
  --ckpt "$CKPT" \
  --previous-hidden /data/chensiyu/hw_project/pypto/workspace/logs_n1_0162/main-hidden-final/main_step00_hidden.pt \
  --active-batch 16
```

固定 MTP diagnostic tokens：

```text
[6178, 410, 303]
```

每层检查 raw hidden pass rate、TP spread、absolute layer 45/46/47 的
weight row 和 KV slab。PyPTO 不得返回 draft token 或 acceptance。

一键 runner：

```bash
scripts/run_pypto_mtp3_back8.sh
```

权威 Python 入口：

```text
tests/step3p5/ci/run_whole_network_ci.py
```

## 7. 当前 canonical 状态（2026-08-07）

loop-form Main 已正式位于
`models/step3p5/decode_fwd.py`，生产 holder 直接编译：

```text
models.step3p5.decode_fwd:whole_decode_step3p5
```

当前 release 不提供 rollback、自定义 Main module/name 或第二个 import
compatibility 入口。Main 与 MTP 共用的 dense MLP kernel 已独立到
`models/step3p5/dense_mlp.py`；该模块不包含 `@pl.program`、host ABI 或
历史 whole-net 入口。

在 0162 的 256-step replacement 回归中，迁移前后：

```text
token:  256/256 exact
hidden: 256/256 exact
max_abs_diff: 0.0
TP spread: 0.0
```

因此 replacement regression 通过。相同 vanilla oracle 的 raw 对齐为
`240/256 = 93.75%`；这低于历史 `>=95%` vanilla raw gate，不能把 raw
结果标记为无条件 PASS。详细数据见
`tests/step3p5/ci/LIVE_PRECISION_AB.md`。

### 7.1 canonical rename 回归

pre-canonical loop-form artifact 正式迁移为
`models.step3p5.decode_fwd:whole_decode_step3p5` 后，在相同的
`stepfun-develop-20260726-opt-b2@sha256:0b22fcef...` 镜像环境、相同
checkpoint、相同 256-step oracle 和 8-15 卡上重新运行默认入口：

```text
canonical raw alignment:          240/256 = 93.75%
pre-rename ↔ canonical token:     256/256 exact
pre-rename ↔ canonical hidden:    256/256 exact
max_abs_diff:                     0.0
TP spread max:                    0.0
step 127 / 128 / 255:             PASS
```

0162 artifact：

```text
/tmp/canonical_step3p5_n256_20260726_1900/
  main_hidden_only_report.json
  canonical_vs_pre_rename.json
```

因此本次正式化只改变 canonical module/program 名称，不改变已验证的数学
实现。2026-07-27 的清理进一步删除了 retired unroll source、rollback
selector 和自定义 Main 入口；后续实现与验收统一以 canonical 为 base。

### 7.2 Historical V4-Flash active-route MoE admission

This section records the superseded push/gather implementation and its
admission evidence. It is not the active canonical decode protocol.

The runtime-active dispatch/combine rewrite was integrated into
`stepfun/develop` as a fast-forward change:

```text
base:   63814d4ae62718b3c0721834878e4b4af4e7ac1b
change: cd19fe6b80f90e27f576091b05753d166a77507c
```

Admission evidence collected on 0162:

| Gate | Configuration | Result |
|---|---|---|
| Focused contracts | `test_attention_swa_active_bound.py` and `test_performance_bc_contract.py` | `30 passed` |
| Five-layer correctness | matched base/change, BS1, context 65536, layers L3/L4 | BF16 bit-exact, `max_abs_diff = 0` |
| Five-layer correctness | matched base/change, BS16, context 65536, layers L3/L4 | BF16 bit-exact, `max_abs_diff = 0` |
| Whole-net smoke/ITL | change only, BS1, context 65536, warmup 5, measured 50 | PASS; finite hidden; mean 38.297 ms, p50 38.244 ms, p99 39.697 ms |

The whole-net ITL entry is a candidate-only absolute measurement. No matched
whole-net performance A/B was run for this admission, so these numbers must
not be reported as a speedup or regression percentage. The five-layer runs
establish numerical non-regression relative to the stated base; they do not
replace the independent live-oracle gate.

### 7.3 Native-layout router and shared-expert admission

The August 2026 decode MoE change keeps router/shared gate/up weights in
checkpoint-native `[N, K]`, uses runtime-BS task grids, and runs shared gate and
up independently. Matched BS1/context-65536 whole-net A/B/A measured a P50
change from a 29.5815 ms baseline midpoint to 29.326 ms (`-0.2555 ms`,
`-0.86%`) with identical hidden hashes and token output across all arms.

The detailed layout ledger, local kernel metrics, critical-path correction,
vLLM-Ascend comparison, validation gates, and future skill-extraction workflow
are in [moe-layout-and-critical-path.md](moe-layout-and-critical-path.md).

## 8. 已证实的精度根因

L7 首个 decode step 的错误已定位到 shared expert 宽 Vec tile：

```text
旧：gate/up/SwiGLU -> [BATCH,160] -> down projection
新：gate/up/SwiGLU -> 5 x [BATCH,32] -> down projection 累加
```

0162 决定性证据：

```text
修复前 step0 shared abs_max = 0.51171875
修复后 step0 shared abs_max = 32.75
L7 step0/step1 FFN bad_ratio = 0
L7 step0/step1 layer_output bad_ratio = 0
L3-L8 cumulative 两步 token exact，duplicate/tp spread = 0
```

该结论只覆盖 shared-expert wide-tile miscompile。完整 Main 8-step、
MTP batch16 和 live vLLM A/B 仍需各自通过。

## 8. 定位纪律

遇到精度、507018、running-stalled 或 timeout，必须先读取：

```text
pypto-project/.claude/skills/pypto-whole-net-hang-debug/SKILL.md
pypto-project/.claude/skills/pypto-dev-constraints/SKILL.md
```

每次运行保存独立 host/device/dmesg/build/source-hash 证据包；精度与
无 stall 分开记录。
