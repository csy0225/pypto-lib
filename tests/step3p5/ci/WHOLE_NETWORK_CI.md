# Step3p5 single-chip hidden-only CI

唯一权威 runner：

```text
tests/step3p5/ci/run_whole_network_ci.py
```

执行：

1. preflight checkpoint、PTO-ISA、cards 8–15 和 protected cards 0–7；
2. Main 45-layer hidden-only 8-step；
3. MTP45/46/47 selected hidden-only active-batch=1；
4. MTP45/46/47 selected hidden-only active-batch=16；
5. 清理 exporter 并检查残余 PID。

PyPTO live ABI 只允许：

```text
Main -> pre-final-norm BF16 next_hidden
MTP  -> raw BF16 mtp_hidden
```

CI 的 CPU tail 只做 standalone token diagnostic，不属于 PyPTO 输出。

```bash
python -m tests.step3p5.ci.run_whole_network_ci \
  --ckpt /data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp \
  --devices 8,9,10,11,12,13,14,15 \
  --out /tmp/n1_single_chip_hidden_ci \
  --artifact-dir /data/chensiyu/hw_project/pypto/workspace/logs_n1_0162/single_chip_hidden_ci
```

Main tokens：

```text
[303, 1207, 19384, 872, 428, 6127, 4231, 2636]
```

MTP tokens：

```text
[6178, 410, 303]
```

数值正确与无 stall 是两个独立 gate。step1 正确、step2 错误时先验证
token/embedding、metadata、KV 地址和 repeated-run state，再检查 weights
和算子。
