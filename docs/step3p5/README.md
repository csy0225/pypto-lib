# Step3p5 N=1 整网与 MTP3 统一入口

本文是当前 `feat/whole-net-n1-fusion` 分支上 Step3p5 N=1 整网程序的
**总入口文档**。如果你只想知道“从哪里启动、使用哪条命令、成功应该看到
什么”，以本文为准。

更细的设计和验证材料：

| 内容 | 文档 |
|---|---|
| MTP3 接入架构、main/sampler/MTP 边界和 MTP buffer 设计 | [`mtp3-whole-net-integration.md`](mtp3-whole-net-integration.md) |
| 0162 机器上的 MTP3 验证记录 | [`mtp3-validation-0162-20260717.md`](mtp3-validation-0162-20260717.md) |
| CI runner 的详细约束、报告和 self-hosted 示例 | [`../../tests/step3p5/ci/WHOLE_NETWORK_CI.md`](../../tests/step3p5/ci/WHOLE_NETWORK_CI.md) |
| N=1 唯一准出标准 | `/data/chensiyu/hw_project/pypto/pypto-project/N1-CANONICAL-TEST.md` |

## 1. 当前代码入口

当前代码已经合并到：

```text
仓库：/data/chensiyu/hw_project/pypto/workspace/pypto-lib
分支：feat/whole-net-n1-fusion
```

模型 program 的入口如下：

```text
N=1 main 45 层：
  models/step3p5/decode_layer.py
  symbol: whole_decode_faithful_real

MTP3：
  models/step3p5/mtp_fwd.py
  symbol: whole_mtp3
```

测试和执行入口如下：

```text
推荐的整网 shell 入口：
  scripts/run_pypto_mtp3_back8.sh

权威的整网 Python runner：
  python -m tests.step3p5.ci.run_whole_network_ci

main device harness：
  tests/step3p5/harnesses/_stage_whole_faithful_real_ipc.py

MTP3 device harness：
  tests/step3p5/harnesses/_stage_whole_mtp3_ipc.py

vLLM 前 8 卡参考启动脚本：
  scripts/run_vllm_mtp3_front8.sh
```

新自动化应使用分类后的 `harnesses` 和 `ci` 路径。顶层的以下模块只是
兼容入口，不要在新代码中继续复制一套编排逻辑：

```text
tests.step3p5._stage_whole_faithful_real_ipc
tests.step3p5._stage_whole_mtp3_ipc
tests.step3p5.run_whole_network_ci
```

## 2. 整网执行边界

当前整网不是把 target sampler 偷塞进 PyPTO program，而是明确保持下面的
模型边界：

```text
embed(token=6127)
    |
    v
whole_decode_faithful_real
    | 45 layers, P42, native W8A8, KV IPC
    | main argmax = 303
    v
target sampler
    | first token = 303
    v
whole_mtp3
    | MTP45 -> MTP46 -> MTP47
    | one PyPTO MTP3 program dispatch
    v
draft tokens = [6178, 410, 303]
```

需要区分两个概念：

1. `whole_decode_faithful_real` 和 `whole_mtp3` 各自都是一次
   `runtime.run()` 的 program-level dispatch；
2. main 和 MTP3 之间仍然存在 target sampler 边界，所以完整链路不是一次
   host API 调用，也不是一个把 sampler 融进 device program 的大 program。

MTP3 program 内部的 MTP45/46/47 层间 hidden 交接在 device program 内完成；
main 的最终 hidden、sampler token 和 MTP3 的输入输出由整网 runner 按显式边界
管理。

## 3. 0162 标准环境

下面是 0162 上经过验证的环境初始化。vendor 环境脚本可能访问未定义变量，
因此保留 `set +u` 包围环境加载：

```bash
set +u
source /usr/local/Ascend/cann/set_env.sh
source /data/chensiyu/hw_project/pypto/workspace/activate.sh
set -u

export PTO_ISA_ROOT=/data/chensiyu/hw_project/pypto/workspace/pto-isa
export PTO2_RING_HEAP=4294967296
export PTO2_RING_TASK_WINDOW=131072
export PTO2_RING_DEP_POOL=131072

cd /data/chensiyu/hw_project/pypto/workspace/pypto-lib

export CKPT=/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp
export DEVICES=8,9,10,11,12,13,14,15
```

当前 canonical N=1/MTP3 默认使用：

```text
main hidden token = 6127
target sampler first token = 303
main P_FAITHFUL_MOE_LAYERS = 42
MTP active batch = 1
MTP fixed batch = 16
MTP expected tokens = [6178, 410, 303]
```

不要把下面这些历史路径、输入或模式当作当前准出对象：

```text
/mnt/hw910test-jfs/...
随机 hidden
P1/P20/P31 代替 P42
BF16 dequant 的历史权重路径
只 compile 不 run
只有 RESULT=...RUN_CLEAN、没有 argmax/逐层检查
```

如果 checkpoint 不在上述路径，必须显式覆盖 `CKPT`，并确认它是原生
Step3p5 W8A8 MTP3 checkpoint。

## 4. 推荐：一条命令跑完整 main → sampler → MTP3 整网

这是 operator 和专用 CI runner 最推荐的入口：

```bash
CKPT=/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp \
DEVICES=8,9,10,11,12,13,14,15 \
OUT=/tmp/n1_weight_ipc_mtp3_ci \
ARTIFACT_DIR=/data/chensiyu/hw_project/pypto/workspace/logs_n1/mtp3_ci \
scripts/run_pypto_mtp3_back8.sh
```

该 shell 文件只负责加载环境和转发参数，真正的编排入口是：

```bash
python -m tests.step3p5.ci.run_whole_network_ci
```

runner 会依次执行：

1. 检查 8 张连续物理卡，并默认保护设备 `0..7`；
2. 在设备 `8..15` 启动 fresh native-W8A8/MTP3 IPC exporter；
3. 检查每个 rank 的 512B 对齐、dtype 和不重叠 map；
4. 跑 main P42，要求 `argmax=303` 并保存 `P42_nh_row0.pt`；
5. 在 sampler 边界使用 first token `303`；
6. 跑 MTP3 active-batch=1，要求 `[6178,410,303]`；
7. 跑 MTP3 active-batch=16；
8. 跑 CPU ctx=1 reference；
9. finally 清理 exporter、pool map、sentinel 和残留 PID。

成功时至少应看到：

```text
WHOLE_NETWORK_MTP3=PASS
stage=canonical_main_p42 rc=0 ... passed=True
stage=mtp3_single rc=0 ... passed=True
stage=mtp3_batch16 rc=0 ... passed=True
stage=mtp3_cpu_reference rc=0 ... passed=True
cleanup=stopped=True ... remaining_pids=[]
```

报告默认在：

```text
<ARTIFACT_DIR>/whole_network_report.json
```

### 4.1 只做 preflight，不启动 exporter 或 NPU stage

第一次接入机器、切换设备号或检查环境时，先运行：

```bash
CKPT=/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp \
DEVICES=8,9,10,11,12,13,14,15 \
scripts/run_pypto_mtp3_back8.sh --dry-run
```

`--dry-run` 只做 checkpoint、设备和环境 preflight，不启动 exporter，不执行
device program。

### 4.2 只关闭非核心附加检查

完整准出不建议关闭 batch16 或 CPU reference。调试时可以临时使用：

```bash
scripts/run_pypto_mtp3_back8.sh --no-run-batch16
scripts/run_pypto_mtp3_back8.sh --no-run-reference
```

这两种结果只能作为缩短调试周期的结果，不能替代完整整网 PASS。

## 5. CI 接入入口

### 5.1 硬件 pytest gate

普通 pytest 默认跳过硬件 gate。专用、独占 8-NPU runner 上使用：

```bash
export STEP3P5_WHOLE_NET_CI=1
export STEP3P5_CKPT_DIR=/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp
export STEP3P5_CI_DEVICES=8,9,10,11,12,13,14,15
export STEP3P5_CI_OUT=/tmp/n1_weight_ipc_mtp3_ci
export STEP3P5_CI_ARTIFACT_DIR="$PWD/whole-network-artifacts"

pytest -q tests/step3p5/ci/test_whole_network_ci.py -s
```

对应的 pytest 文件是：

```text
tests/step3p5/ci/test_whole_network_ci.py
```

不要把该 gate 放入普通的两卡 NPU job。它需要：

```text
独占 8 张连续设备
后 8 卡不被其它作业占用
足够的每 rank IPC pool 空间
与 canonical CANN/PyPTO/PTOAS/pto-isa 环境一致
```

### 5.2 无卡静态/runner 测试

提交或 CI 的普通阶段可以先跑：

```bash
python -m compileall -q \
  models/step3p5/mtp.py \
  models/step3p5/mtp_fwd.py \
  tests/step3p5 \
  tools/step3p5/pypto_mtp3_ctx1_reference.py

bash -n \
  scripts/run_pypto_mtp3_back8.sh \
  scripts/run_vllm_mtp3_front8.sh

pytest -q \
  tests/step3p5/ci/test_test_layout.py \
  tests/step3p5/ci/test_whole_network_ci_runner.py \
  tests/step3p5/ci/test_whole_network_ci.py
```

在没有硬件 gate 环境时，最后一个测试会 skip；这不是整网硬件 PASS。

## 6. 前 8 卡 vLLM 参考入口

前 8 卡的 vLLM MTP3 参考启动脚本是：

```text
scripts/run_vllm_mtp3_front8.sh
```

它在子 shell 中设置：

```text
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
VLLM_USE_V1=1
VLLM_ASCEND_ENABLE_PREFETCH_MLP=0
HCCL_OP_EXPANSION_MODE=AIV
HCCL_BUFFSIZE=512
TASK_QUEUE_ENABLE=0
SHM_BARRIER=true
speculative_config = step3p5_mtp, num_speculative_tokens=3,
                     enable_multi_layers_mtp=true
```

使用时显式指定 checkpoint 和日志：

```bash
CKPT=/mnt/hw910test/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp \
PORT=8000 \
LOG=/data/LY/logs/step3p5_910b_v017/precision/mtp3.log \
scripts/run_vllm_mtp3_front8.sh
```

**注意：** 0162 的设备 `0..7` 可能已有 root-owned vLLM 服务。已有服务时不要
重复启动该脚本，不要对前 8 卡执行 kill/stop，也不要把前 8 卡的
`ASCEND_RT_VISIBLE_DEVICES` 继承到 PyPTO 后 8 卡进程。

`run_pypto_mtp3_back8.sh` 的 runner 会主动清理影响后 8 卡的 vLLM/front-8
变量，但不会粗暴删除 pinned CANN 环境所需的全部 HCCL 变量。

## 7. 只跑某个 stage：调试入口

整网验证优先使用第 4 节的 runner。只有定位问题时才直接调用 stage harness。
下面的流程使用同一个 MTP3 exporter pool，因此 main harness 可以访问 main
权重、KV 和 MTP3 相关 map。

### 7.1 启动共享 exporter pool

```bash
export OUT=/tmp/n1_weight_ipc_mtp3_debug
mkdir -p "$OUT"
rm -f \
  "$OUT"/ready.rank* \
  "$OUT"/STOP \
  "$OUT"/pypto_weight.key.rank* \
  "$OUT"/pypto_weight_map.rank*.json \
  "$OUT"/pypto_weight_map.rank*.json.done \
  2>/dev/null || true

exporter_pids=()
for r in $(seq 0 7); do
  dev=$((8 + r))
  python -m tests.step3p5.harnesses._stage_whole_mtp3_ipc \
    --export-rank "$r" \
    --dev "$dev" \
    --out "$OUT" \
    --ckpt "$CKPT" \
    > "/tmp/step3p5_mtp3_export_rank${r}.log" 2>&1 &
  exporter_pids+=("$!")
done

while [ "$(find "$OUT" -maxdepth 1 -name 'ready.rank*' | wc -l)" -lt 8 ]; do
  sleep 15
done
```

### 7.2 只跑 N=1 main P42

```bash
export P_FAITHFUL_MOE_LAYERS=42
export N1_DUMP_DIR=/tmp/n1_mtp3_debug/main_dump
mkdir -p "$N1_DUMP_DIR"

python -m tests.step3p5.harnesses._stage_whole_faithful_real_ipc \
  --device "$DEVICES" \
  --reuse-exporters \
  --kv-ipc \
  --hidden-token 6127 \
  --out "$OUT" \
  --ckpt "$CKPT"
```

成功条件：

```text
rc=0
[worker] RUN done
argmax=303
${N1_DUMP_DIR}/P42_nh_row0.pt 存在
```

### 7.3 只跑 MTP3

```bash
export MTP_DUMP_DIR=/tmp/n1_mtp3_debug/mtp3_single
mkdir -p "$MTP_DUMP_DIR"

python -m tests.step3p5.harnesses._stage_whole_mtp3_ipc \
  --device "$DEVICES" \
  --reuse-exporters \
  --out "$OUT" \
  --ckpt "$CKPT" \
  --previous-hidden "$N1_DUMP_DIR/P42_nh_row0.pt" \
  --first-token 303 \
  --active-batch 1 \
  --dump-dir "$MTP_DUMP_DIR"
```

成功条件：

```text
rc=0
[worker] RUN done
tokens=[6178, 410, 303]
RESULT=MTP3_IPC_RUN_CLEAN
```

### 7.4 清理 debug exporter

无论 stage 成功还是失败，都必须先让 exporter 看到 `STOP`，再删除 pool：

```bash
touch "$OUT/STOP"
for pid in "${exporter_pids[@]}"; do
  wait "$pid" || true
done
rm -f \
  "$OUT"/ready.rank* \
  "$OUT"/STOP \
  "$OUT"/pypto_weight.key.rank* \
  "$OUT"/pypto_weight_map.rank*.json \
  "$OUT"/pypto_weight_map.rank*.json.done \
  2>/dev/null || true
```

不要在 exporter 进程仍存活时删除 `OUT`，否则可能留下 stale IPC map 或误判
后续运行。

## 8. 失败时按整体边界排查

建议顺序固定为：

1. 先确认是否违反框架约束：TP/EP 是否为 8、设备是否连续、IPC offset 是否
   512B 对齐、dtype 是否匹配、buffer 是否重叠、MTP KV 是否切成 3 个独立
   slice、host tensor 是否在 `prepare()` 前 shared；
2. 再确认环境隔离：后 8 卡是否误继承 `ASCEND_RT_VISIBLE_DEVICES`、vLLM/HCCL
   变量是否污染、前 8 卡是否被误操作；
3. 再检查程序边界：main hidden 是否来自 P42、sampler token 是否为 303、
   MTP 是否从 `P42_nh_row0.pt` 开始；
4. 最后才进入 kernel 内部逻辑和局部数值分析。

典型结果与含义：

| 现象 | 优先检查 |
|---|---|
| preflight 失败 | checkpoint index、设备连续性、保护卡配置、环境 |
| exporter 未 ready | checkpoint 读取、rank/device 映射、pool 路径权限 |
| main `argmax != 303` | P42、native W8A8 map、KV IPC、hidden token 6127 |
| MTP token 不一致 | main→sampler hidden/token 边界、MTP KV slice、active mask |
| batch1 通过、batch16 失败 | padding 初始化、active mask、固定 16 行 buffer/slot |
| CPU reference 失败 | dtype/ABI、层间 hidden、shared-head logits、token 链 |
| cleanup 失败 | exporter PID、STOP sentinel、旧 pool map、残留进程 |

禁止用以下现象直接宣布通过：

```text
compile OK
RUN_CLEAN
单层或 P1 通过
随机输入通过
只看某一个 rank
只看局部 hidden，不看 main argmax/MTP tokens/CPU reference
```

## 9. 入口速查表

| 目的 | 命令 |
|---|---|
| 完整整网复现 | `scripts/run_pypto_mtp3_back8.sh` |
| 直接调用完整 runner | `python -m tests.step3p5.ci.run_whole_network_ci` |
| 只做 preflight | `scripts/run_pypto_mtp3_back8.sh --dry-run` |
| main P42 调试 | `python -m tests.step3p5.harnesses._stage_whole_faithful_real_ipc ...` |
| MTP3 调试 | `python -m tests.step3p5.harnesses._stage_whole_mtp3_ipc ...` |
| 硬件 pytest gate | `pytest -q tests/step3p5/ci/test_whole_network_ci.py -s` |
| 无卡 runner 测试 | `pytest -q tests/step3p5/ci/test_whole_network_ci_runner.py` |
| 前 8 卡 vLLM 参考 | `scripts/run_vllm_mtp3_front8.sh` |
