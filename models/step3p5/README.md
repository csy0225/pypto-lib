# Step3p5 single-chip hidden-only model

## Production programs

```text
Main:
  decode_fwd.py
  whole_decode_step3p5

MTP45/46/47:
  mtp_hidden_fwd.py
  MTP_LAYER_HIDDEN_PROGRAMS
```

canonical Main 返回 pre-final-norm BF16 hidden；MTP 返回 raw pre-shared-head BF16
hidden。`decode_fwd.py` 是唯一 Main 产品入口；历史 unroll Main 已删除。
final norm、LM head、sampling 和 acceptance/rejection 不属于本目录的
production program。

## Required modules

```text
config.py / _ops.py / dense_mlp.py
attention_full.py / attention_swa.py
gate.py / dispatch.py / combine.py
expert_routed.py / expert_shared.py / moe.py
collectives.py
weight_loader.py
```

`step3p5_prefill.py` 和 `prefill_*` 是独立 prefill 工作，不是 decode
fallback。旧 per-layer decode、whole-MTP3、generator、Phase-2
single-layer draft，以及已经断链的 per-layer MoE socket worker 均已删除。

完整边界、0162 环境、single/multi-step 差异和 canonical 命令见：

```text
docs/step3p5/README.md
```
