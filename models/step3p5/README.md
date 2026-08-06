# Step3p5 single-chip hidden-only model

## Production programs

```text
Main (decode):
  decode_fwd.py
  whole_decode_step3p5

Main (prefill):
  prefill_layer_single_chip_hidden.py
  whole_prefill_step3p5

MTP45/46/47:
  mtp_hidden_fwd.py
  MTP_LAYER_HIDDEN_PROGRAMS
```

canonical Main 返回 pre-final-norm BF16 hidden；MTP 返回 raw pre-shared-head BF16
hidden。`decode_fwd.py` 是 decode 唯一 Main 产品入口；
`prefill_layer_single_chip_hidden.py` 是 prefill 唯一 Main 产品入口（与 decode
对偶的 hidden-only 整网 single `@pl.program`）。历史 unroll Main 已删除。
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

prefill (canonical 轨道，与 decode 对偶):
  prefill_layer_single_chip_hidden.py   # whole_prefill_step3p5
  prefill_fwd.py / prefill_moe.py
  prefill_attention_full.py / prefill_attention_swa.py
  prefill_qkv_proj_rope.py
```

prefill 是与 decode 对偶的 canonical 生产轨道，不是 decode fallback：它镜像
decode 的 single-chip hidden-only 边界（vLLM embed+paged-KV → PyPTO 45 层
whole program → pre-final-norm BF16 hidden → vLLM LM head/sampling），复用同一
weight pool / KV pool / IPC 契约 / fail-closed 纪律，差异仅在 token 维度
（`PREFILL_T=128`）、per-token positions/slot_mapping、resident `position_ids`
与 KV 写入语义（prefill 写 paged cache，decode 读）。旧 per-layer decode、
whole-MTP3、generator、Phase-2 single-layer draft，以及已经断链的 per-layer
MoE socket worker 均已删除。

> prefill 整网 program 当前为契约 scaffold（`IS_SCAFFOLD=True`），真实 45 层
> body 与 P1 内核 codegen 改造仍在推进；桥接层已落地。详见
> `/data/jhj/pypto_step3p5/RECOVERY_PROGRESS.md`。

完整边界、0162 环境、single/multi-step 差异和 canonical 命令见：

```text
docs/step3p5/README.md
```
