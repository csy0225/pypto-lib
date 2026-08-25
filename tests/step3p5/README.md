# Step3p5 canonical test layout

The active test tree covers single-chip hidden-only Main/MTP plus the
focused five-layer MoE admission harness:

```text
harnesses/
  _stage_main_hidden_only.py
  _stage_mtp_hidden_selected.py
  _stage_five_layer_moe.py
unit/
  ABI、metadata、KV reserve、IPC、loader、control-plane contracts
  test_performance_bc_contract.py
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

## MoE runtime-active scheduling contracts

`tests/step3p5/unit/test_performance_bc_contract.py` is the static admission
gate for the replicated-input local-owner decode schedule. Together with
`test_local_ep_protocol.py`, it
checks that:

1. every rank consumes the same gate result without EP activation exchange;
2. each `(token, top-k)` route is packed only by its unique contiguous expert
   owner and stays within the fixed 128-row expert slab;
3. local combine adds the TP-sharded shared partial and owner-local routed
   rows in FP32;
4. every MoE orchestrator performs one TP all-reduce after local combine and
   adds the residual only after that collective;
5. zero-route ranks still execute local combine and the final collective;
6. routed external kernels use fixed workspace strides and never issue a
   zero-valid GM load for an empty 8-row part.

Run the focused contracts with:

```bash
python -m pytest -q -p no:cacheprovider \
  tests/step3p5/unit/test_attention_swa_active_bound.py \
  tests/step3p5/unit/test_local_ep_protocol.py \
  tests/step3p5/unit/test_performance_bc_contract.py
```
