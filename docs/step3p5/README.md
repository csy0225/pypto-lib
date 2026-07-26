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
models/step3p5/decode_layer_single_chip_hidden.py
whole_decode_faithful_real_single_chip_hidden_only
```

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
  decode_layer_single_chip_hidden.py
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
- 禁止 BF16 dequant/fallback；
- RMSNorm EPS 为 `1e-5`；
- router bias 按 vLLM 的 BF16 round 语义对齐。

### KV 和 fixed batch

物理 batch 固定为 16。active row 保留 vLLM scheduler metadata；
padding row 使用 allocator-owned 的 15 个 reserve blocks：

```text
scheduler domain: [0, scheduler_num_blocks)
padding reserve:  [scheduler_num_blocks, scheduler_num_blocks + 15)
```

每个 active row 必须满足：

```text
position = seq_len - 1
slot_mapping[row] =
  block_table[row, position // 128] * 128 + position % 128
```

padding row 必须使用 `seq_len=1`、`position=0` 和互不重叠的 reserve
block/slot。Main 与 MTP 使用独立 KV pool、map 和 reserve。

### 两个独立 gate

数值正确和无 stall 必须分别通过。`RUN_CLEAN`、一次 token 正确、
compile-only、P1/P20、随机输入或 BF16 fallback 都不能替代完整准出。

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

## 7. 当前 B2 replacement 状态（2026-07-26）

当前实现位于 `perf/step3p5-bc-v2`。`models/step3p5_opt/decode_fwd.py`
提供 loop-form Main opt，生产 sidecar 通过显式的：

```text
--layer-module models.step3p5_opt.decode_fwd
--layer-name whole_decode_opt
```

选择自定义实现。当前 release 不传参数时默认使用
`models.step3p5_opt.decode_fwd:whole_decode_opt`；只有显式传
`--baseline-main` 时才回退到 canonical 0724 hidden-only baseline。

在 0162 的 256-step 回归中，opt 与 current baseline：

```text
token:  256/256 exact
hidden: 256/256 exact
max_abs_diff: 0.0
TP spread: 0.0
```

因此 replacement regression 通过。相同 vanilla oracle 的 raw 对齐为
`240/256 = 93.75%`，baseline 也完全复现该结果；这低于历史 `>=95%`
vanilla raw gate，不能把 raw 结果标记为无条件 PASS。详细数据见
`tests/step3p5/ci/LIVE_PRECISION_AB.md`。

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
