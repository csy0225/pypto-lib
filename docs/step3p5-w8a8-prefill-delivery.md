# Step3p5 W8A8 prefill precision delivery

## Scope

This delivery extends the completed W8A8 decode precision closure to prefill.
The validation flow mirrors decode: vLLM eager detail dumps are used as the
oracle, PyPTO host/reference math replays the W8A8 checkpoint bundle, and the
suite checks per-layer detail tensors plus final RMSNorm/LM-head logits.

Required prompt lengths are covered: `1k`, `4k`, `8k`, `32k`, `64k`, `128k`.

## Code delivered

- `tools/step3p5/collect_w8a8_prefill_golden.py`
  - collects sampled W8A8 vLLM prefill detail golden cases;
  - prunes dump files to tensors required by PyPTO detail/final-logits compare.
- `tools/step3p5/prefill_precision_suite.py`
  - runs multi-length detail comparison and final-logits comparison;
  - emits `STEP3P5_W8A8_PREFILL_REPORT.{json,md}`;
  - can package the report/artifacts into a tar file.
- `tests/step3p5/precision/test_step3p5_w8a8_prefill_st.py`
  - pytest ST wrapper for either precomputed reports or live golden replay.

The W8A8 checkpoint support used by this flow is the same loader path already
landed for decode (`quant_model_weights.safetensors.index.json` and dequantized
W8A8_DYNAMIC routed experts).

## 0162 validation result

Host: `gpu-a910x-0162.host.platform.shaipower.com`

- Checkpoint: `/mnt/nvme1/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp`
- Golden root: `/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_prefill_v001/golden_step3p5_w8a8_prefill_vllm_sampled`
- Report root: `/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_prefill_v001/pypto_prefill_precision`
- Acceptance: W8A8 prefill sampled detail pass rate >= `0.997`; final logits all PASS.

| Case | Seq len | Detail | Final logits | Worst pass rate |
|---|---:|---|---|---:|
| `prefill_1k` | 1024 | PASS | PASS | 0.999349 |
| `prefill_4k` | 4096 | PASS | PASS | 0.998698 |
| `prefill_8k` | 8192 | PASS | PASS | 0.999023 |
| `prefill_32k` | 32768 | PASS | PASS | 0.999349 |
| `prefill_64k` | 65536 | PASS | PASS | 0.999756 |
| `prefill_128k` | 131072 | PASS | PASS | 0.997559 |

Pytest gate:

```bash
STEP3P5_PREFILL_REPORT_ROOT=/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_prefill_v001/pypto_prefill_precision \
PYTHONPATH=. pytest -q tests/step3p5/precision/test_step3p5_w8a8_prefill_st.py
# 1 passed in 0.01s
```

## Artifact package

```text
/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_prefill_v001/step3p5_w8a8_prefill_regression_20260626.tar
/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_prefill_v001/step3p5_w8a8_prefill_regression_20260626.tar.sha256
```

SHA256:

```text
cd34f034e017c68437547e5f7f453a2f6b481a1e97e162a89ac21c422fe76b6e  step3p5_w8a8_prefill_regression_20260626.tar
```

## Notes

- Full 128k prefill detail dumps are prohibitively large, so the golden capture
  uses a vLLM debug-dump sampling patch: per forward, rows are evenly sampled
  up to 128 tokens, and only tensors needed by the PyPTO comparator are kept.
- Final logits are compared for every case and all pass.
