# Step3p5 single-chip hidden-only model

## Production programs

```text
Main:
  decode_fwd.py
  whole_decode_step3p5

Rollback baseline:
  decode_layer_single_chip_hidden.py
  whole_decode_faithful_real_single_chip_hidden_only

MTP45/46/47:
  mtp_hidden_fwd.py
  MTP_LAYER_HIDDEN_PROGRAMS
```

canonical Main 返回 pre-final-norm BF16 hidden；MTP 返回 raw pre-shared-head BF16
hidden。final norm、LM head、sampling 和 acceptance/rejection 不属于本目录
的 production program。

## Required modules

```text
config.py / _ops.py
attention_full.py / attention_swa.py
gate.py / dispatch.py / combine.py
expert_routed.py / expert_shared.py / moe.py
collectives.py
weight_loader.py
```

`step3p5_prefill.py` 和 `prefill_*` 是独立 prefill 工作，不是 decode
fallback。旧 per-layer decode、whole-MTP3 和 generator 已删除。

完整边界、0162 环境、single/multi-step 差异和 canonical 命令见：

```text
docs/step3p5/README.md
```
