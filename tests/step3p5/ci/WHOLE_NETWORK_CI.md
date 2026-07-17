# Step3p5 Whole-Network MTP3 CI

## 1. Scope

The authoritative CI entry point is:

```text
tests/step3p5/ci/run_whole_network_ci.py
```

The pre-categorization module remains as a compatibility entry point:

```text
tests.step3p5.run_whole_network_ci
```

New CI jobs should use the categorized module. Existing pinned 0162 commands
may keep using the compatibility module while downstream automation migrates.

It is an orchestration runner, not another model harness.  The runner reuses:

```text
canonical main  tests/step3p5/harnesses/_stage_whole_faithful_real_ipc.py
MTP3 device     tests/step3p5/harnesses/_stage_whole_mtp3_ipc.py
CPU reference   tools/step3p5/pypto_mtp3_ctx1_reference.py
```

The canonical main and MTP harnesses also retain their historical module names
as thin compatibility entry points:

```text
tests.step3p5._stage_whole_faithful_real_ipc
tests.step3p5._stage_whole_mtp3_ipc
```

This keeps the already-validated model, buffer, and layer boundaries in one
place.  In particular, the target sampler boundary remains:

```text
main P42 hidden/logits
  -> target token 303
  -> MTP45
  -> MTP46
  -> MTP47
```

The runner does not fuse the target sampler into either program and does not
introduce a second weight-loading or kernel execution path.

## 2. Default gate

The default invocation performs all of the following:

1. preflight for exactly eight ordered, contiguous physical devices;
2. protection of devices 0..7 unless explicitly overridden;
3. fresh MTP-capable IPC exporters, one per rank;
4. independent validation of all eight exporter maps:
   - every entry is 512-byte aligned and non-overlapping;
   - main routed MoE weights are INT8;
   - main routed scales are FP32;
   - MTP projection/attention/MLP/shared-head weights are BF16;
   - MTP norm ABI tensors are FP32;
   - main and MTP KV entries do not alias;
   - MTP KV contains three layer slices;
5. canonical main P42, token 6127, native W8A8, and KV IPC;
6. main `rc=0`, `RUN done`, hidden dump, and `argmax=303`;
7. MTP active-batch=1:
   - exact tokens `[6178, 410, 303]`;
   - TP-rank token consistency;
   - program token equals concatenated full-logits argmax;
   - inactive hidden/logits rows are zero;
8. MTP active-batch=16 with the same per-row token/logits checks;
9. CPU ctx=1 MTP3 reference:
   - exact reference/device tokens;
   - `ok=true`;
   - worst numeric pass rate at least 0.97;
10. exporter shutdown, sentinel/map cleanup, and residual PID check.

A `RESULT=...RUN_CLEAN` line alone never satisfies the gate.

## 3. Direct invocation on 0162

Enter the pinned environment first:

```bash
set +u
source /usr/local/Ascend/cann/set_env.sh
source /data/chensiyu/hw_project/pypto/workspace/activate.sh
set -u

export PTO_ISA_ROOT=/data/chensiyu/hw_project/pypto/workspace/pto-isa
export PTO2_RING_HEAP=4294967296
export PTO2_RING_TASK_WINDOW=131072
export PTO2_RING_DEP_POOL=131072

cd /data/chensiyu/hw_project/pypto/workspace/pypto-lib-mtp3
python -m tests.step3p5.ci.run_whole_network_ci \
  --ckpt /data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp \
  --devices 8,9,10,11,12,13,14,15 \
  --out /tmp/n1_weight_ipc_mtp3_ci \
  --artifact-dir /data/chensiyu/hw_project/pypto/workspace/logs_n1/mtp3_ci
```

The compatibility shell entry point performs only environment activation and
then delegates to the same Python runner:

```bash
scripts/run_pypto_mtp3_back8.sh
```

Use preflight-only mode before assigning a new machine:

```bash
python -m tests.step3p5.ci.run_whole_network_ci \
  --ckpt "$STEP3P5_CKPT_DIR" \
  --devices 8,9,10,11,12,13,14,15 \
  --dry-run
```

## 4. Pytest/CI invocation

The hardware pytest gate is skipped unless explicitly enabled:

```bash
STEP3P5_WHOLE_NET_CI=1 \
STEP3P5_CKPT_DIR=/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp \
STEP3P5_CI_DEVICES=8,9,10,11,12,13,14,15 \
STEP3P5_CI_OUT=/tmp/n1_weight_ipc_mtp3_ci \
STEP3P5_CI_ARTIFACT_DIR="$PWD/whole-network-artifacts" \
pytest -q tests/step3p5/ci/test_whole_network_ci.py -s
```

Recommended dedicated self-hosted job:

```yaml
step3p5-whole-network:
  runs-on: [self-hosted, linux, npu, npu-8, step3p5]
  timeout-minutes: 120
  env:
    STEP3P5_WHOLE_NET_CI: "1"
    STEP3P5_CKPT_DIR: /path/to/pinned/checkpoint
    STEP3P5_CI_DEVICES: 8,9,10,11,12,13,14,15
    STEP3P5_CI_OUT: /tmp/n1_weight_ipc_mtp3_ci
    STEP3P5_CI_ARTIFACT_DIR: ${{ github.workspace }}/whole-network-artifacts
  steps:
    - uses: actions/checkout@v4
    - name: Run Step3p5 whole-network gate
      shell: bash
      run: |
        source /path/to/pinned/cann/set_env.sh
        source /path/to/pinned/workspace/activate.sh
        export PTO_ISA_ROOT=/path/to/pinned/pto-isa
        pytest -q tests/step3p5/ci/test_whole_network_ci.py -s
    - name: Upload diagnostics
      if: always()
      uses: actions/upload-artifact@v4
      with:
        name: step3p5-whole-network
        path: whole-network-artifacts
```

Do not add this test to the existing generic two-device NPU job.  The job must
have exclusive ownership of eight contiguous devices and enough memory for the
per-rank IPC pool.

## 5. Environment isolation

Every child stage receives explicit physical device IDs.  The runner removes
the front-8 settings that can alter back-8 behavior:

```text
ASCEND_RT_VISIBLE_DEVICES
VLLM_*
HCCL_BUFFSIZE
HCCL_OP_EXPANSION_MODE
TASK_QUEUE_ENABLE
SHM_BARRIER
CPU_AFFINITY_CONF
LD_PRELOAD
EPMOE_BYPASS_GATE
P_DBG_STAGE
P_FILL_BATCH
PYPTO_MEM_PLANNER
PYPTO_WEIGHT_IPC_VA_SHIFT_GB
```

Other CANN/HCCL variables from the pinned machine environment remain available.
The runner never starts, stops, or signals the vLLM process on devices 0..7.

## 6. Artifacts and failure behavior

The report path defaults to:

```text
<artifact-dir>/whole_network_report.json
```

Per-rank exporter logs, stage logs, main hidden, MTP device dumps, and the CPU
reference report are retained below the artifact directory.  Console output is
kept short; a failed stage prints the tail of its dedicated log.

On normal success or failure, `finally` writes the IPC `STOP` sentinel, waits
for all exporter process groups, removes exporter-owned sentinels/keys/maps,
and verifies that no exporter command still references the pool directory.

`--keep-exporters-on-failure` is a manual debugging option.  It intentionally
leaves the run in a failed state and must not be enabled in unattended CI.

## 7. Card-free runner tests

The orchestration code has device-free coverage for environment scrubbing, log
parsing, IPC map dtype/alignment gates, checkpoint preflight, and protected
device handling:

```bash
pytest -q tests/step3p5/ci/test_whole_network_ci_runner.py
```

The hardware gate itself remains skipped in ordinary pytest:

```text
tests/step3p5/ci/test_whole_network_ci.py
```
