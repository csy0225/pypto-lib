# Step3p5 canonical test layout

当前测试树只服务于 single-chip hidden-only Main/MTP：

```text
harnesses/
  _stage_main_hidden_only.py
  _stage_mtp_hidden_selected.py
unit/
  ABI、metadata、KV reserve、IPC、loader、control-plane contracts
ci/
  run_whole_network_ci.py
  test_whole_network_ci_runner.py
  WHOLE_NETWORK_CI.md
```

推荐入口：

```bash
python -m tests.step3p5.ci.run_whole_network_ci \
  --ckpt /data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp \
  --devices 8,9,10,11,12,13,14,15 \
  --out /tmp/n1_single_chip_hidden_ci \
  --artifact-dir /data/chensiyu/hw_project/pypto/workspace/logs_n1_0162/single_chip_hidden_ci
```

旧 per-layer decode、旧 whole-net IPC wrapper、旧 whole-MTP3 和旧
generator 不允许重新添加兼容入口。
