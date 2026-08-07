# Live token-alignment A/B（主网整网精度准出）

pypto 整网 decode 与 **live vanilla vLLM W8A8 oracle** 在同一 prompt 上逐 token 对齐，
统计 top-1 一致率。这是主网 hidden-only decode 的**在线精度准出口径**（Phase 21 L3：
top-1 ≥ 95%）。

## 为什么不再用硬编码 oracle

旧 harness 用写死的 `DEFAULT_ORACLE_TOKENS = [303,1207,19384,...]`。其中 position 2 的
`19384(题目)` 是**错的**：它来自"一次性生成完整文本再 `tokenizer.encode(text)`"——该
re-tokenization 在 merge 边界会串位。对相同 no-BOS 显式 id 上下文 `[6127,303,1207]`，
**vanilla vLLM 自己**和 pypto **都**输出 `6127(北京)`，`19384` 只是 vanilla 的第 2 名。
硬编码常量导致 harness 在 step2 误报 FAIL。**正确做法 = step-by-step 显式 id、每步单
token encode、无 BOS**（`gen_vanilla_oracle.py`）。

## 两阶段（0162）

pypto `.venv311` 没有 transformers，vanilla oracle 在独立容器里；oracle 端口 8000
host-networked（host 可直连）。

- Stage 1（oracle 环境，有 transformers + 能连 8000）：`gen_vanilla_oracle.py` 从 seed
  逐步贪心生成 → `ORACLE_IDS_JSON`。请求显式设置
  `return_tokens_as_token_ids=true`，并且只接受 `logprobs.tokens` 中的
  `token_id:<id>`；若服务端未返回原生 ID，oracle 直接失败，不把返回文本重新
  tokenize 成 ID。
- Stage 2（pypto host，cards 8-15）：`_stage_main_hidden_only --teacher-forced
  --seed-token <seed> --oracle-token <id>...`，每步喂 oracle 正确 token（解耦 token
  链，避免一次翻转污染后续），比 pypto argmax == oracle 下一 token。

## 跑法

先确保 vanilla oracle 起在 cards 0-7 / 8000（`/logs/start_8000_oracle.sh`，容器内）。
然后：

```bash
ORACLE_PYTHON="/path/to/oracle-python-wrapper" \
CHECKPOINT_MANIFEST="/path/to/trusted/checkpoint_identity.json" \
N=128 SEED=6127 \
bash tests/step3p5/ci/run_live_precision_ab.sh
```

`ORACLE_PYTHON` 必须是单个可执行文件路径。若 oracle 位于独立 namespace，
先准备一个固定 wrapper 可执行文件，再传入该路径；脚本不解析 shell 命令字符串。

默认是 release gate：严格要求 `N=128`、阈值 `>=95%`、当前 checkout clean，
并从脚本自身路径定位被测 pypto-lib，而不是隐式切到相邻旧工作树。若只做诊断，可显式
设置 `LIVE_PRECISION_RELEASE_GATE=0`；诊断模式只输出
`LIVE_AB_DIAGNOSTIC_ALIGNED`，不能作为 release PASS。

两个阶段还必须使用同一份可信的 full-shard checkpoint identity manifest。
脚本会分别在 oracle 和 PyPTO namespace 中重新计算 config、index 和所有 weight
shard 的 SHA256，再比较两份 checkpoint evidence；路径可以不同，但 logical ID、
manifest hash 和 full-shard identity 必须完全一致。任何同尺寸 shard 篡改都会在
启动模型前 fail closed。

准出：`LIVE_AB_ALIGNED >= 95%`。脚本要求恰好生成 N 个非负 oracle token ID、
PyPTO 日志包含连续的 `step=0..N-1`，并独立核对每步 `expected_token` 与 frozen
oracle、`input_token` teacher-forced 链及 `token_exact` 一致性。N 必须为正数，
门限必须是 `[0,100]` 内的有限值；对齐率低于门限时以非零状态退出。仅打印对齐率
不构成 PASS。

## 已验证结果（2026-07-23, 0162, stepfun/develop a632c42e）

- seed=6127(北京)，N=128，**ALIGNED = 124/128 = 96.9%**（≥95% PASS）。
- 生成文本连贯（"，但北京是直辖市，不是省。所以..."）。
- 4 个 miss（step 43/70/98/104）全是 vanilla **自己**的 near/dead-tie（gap 0.0–0.25
  logprob；43/98 gap=0.0000），且 pypto 的选择 = vanilla **fresh 查询的 #1**（rank 0）
  → 属 vanilla 自身 tie-break 非确定性，非 pypto 精度缺陷。
- 结论：主网 multi-decode 精度**正常**，pypto ≈ vanilla 逐 token 对齐。

> 边界：本 gate 覆盖主网 45 层 hidden→vLLM tail 的 decode 对齐。MTP45/46/47 端到端对齐
> 是独立 gate（另见 MTP CI）。

## 2026-07-26 当前 B2 release 回归（0162，N=256）

本次回归使用固定的 0724 镜像：

```text
image:  hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260724
digest: sha256:2b0dc4612796a34bea6720ccb4bf8fa3af4ea406cdd0f12add34586ca860d7e0
seed:   6127
steps:  256
oracle: vanilla vLLM cards 0-7
test:   current baseline/opt cards 8-15, teacher-forced
```

### 两个必须分开判定的 gate

| gate | 结果 | 判定 |
|------|------|------|
| vanilla raw live alignment | opt `240/256 = 93.75%`；baseline `240/256 = 93.75%` | **raw 95% gate 未通过** |
| replacement equivalence | opt vs baseline token `256/256`；hidden `256/256` exact；`max_abs_diff=0` | **通过** |

这里的 `240/256` 不是 opt 相对 baseline 的回归损失：在同一 checkpoint、
同一 0724 runtime、同一 oracle 和同一 8-15 设备集合上，canonical baseline
逐步复现了完全相同的 240/256 结果。opt 与 baseline 每一步 token 完全一致，
每一步主 hidden 逐字节一致，两个版本的 TP hidden spread 均为 `0.0`。

跨 KV block 边界的 `step127`、`step128` 以及长序列末尾的 `step255` 均
`token_exact=true`。所有 256 步 hidden 均 finite；16 个 raw miss 是：

```text
2, 20, 49, 52, 57, 62, 125, 131,
151, 153, 161, 162, 187, 221, 231, 252
```

对这些 miss 使用相同显式 token 上下文对 vanilla 重复查询，观察到 fresh
top-1 在多个位置切换或出现近似 tie。因此当前发布结论必须写成：

> **B2 replacement regression PASS：current opt 不改变 0724-derived
> canonical baseline 的 256-step 行为。vanilla raw 95% gate 仍是
> 93.75%，不能写成无条件的 vanilla precision PASS。**

回归产物（0162）：

```text
/tmp/live_ab_opt_ad478abb_n256_20260726/raw_alignment_summary.json
/tmp/live_ab_opt_ad478abb_n256_20260726/baseline_opt_n256_compare.json
/tmp/live_ab_opt_ad478abb_n256_20260726/vanilla_miss_requery.json
```

### 2026-07-26 canonical rename regression

正式入口从 pre-canonical loop-form artifact 迁移到
`models.step3p5.decode_fwd:whole_decode_step3p5` 后，使用同一镜像、
checkpoint、oracle 和设备集合重新执行 N=256：

| 比较项 | 结果 |
|---|---:|
| canonical 对 vanilla raw | `240/256 = 93.75%` |
| pre-rename ↔ canonical token | `256/256 exact` |
| pre-rename ↔ canonical hidden | `256/256 exact` |
| `max_abs_diff` | `0.0` |
| `TP spread max` | `0.0` |
| step 127 / 128 / 255 | `PASS` |

artifact：

```text
/tmp/canonical_step3p5_n256_20260726_1900/main_hidden_only_report.json
/tmp/canonical_step3p5_n256_20260726_1900/canonical_vs_pre_rename.json
```

这证明正式化仅改变 module/program 名称和默认选择，不改变已验证的
loop-form 数学实现。2026-07-27 已删除 retired unroll source、rollback
selector 和自定义 Main module/name 参数，后续只验收 canonical。
