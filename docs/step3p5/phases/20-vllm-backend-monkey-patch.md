# Phase 20 — vLLM Backend Monkey-Patch (e2e flow)

> **Component pin snapshot (at doc creation, 2026-06-22)**
>
> | Repo | Branch | Pin | Notes |
> |------|--------|-----|-------|
> | pypto-lib | `stepfun/develop` | `a6b5faa` (pre-commit) | Will become the commit that adds this file |
> | pypto | `stepfun/develop` | `b00c8b23` | DFX env hooks + repros |
> | pto-isa | `stepfun/develop` | `e25732f0` | = origin/main |
> | PTOAS | `stepfun/develop` | `da011a3d` | binary v0.45 |
> | simpler (submodule) | — | `a6e06406` | Phase 16 4 patches |
> | vLLM (reference) | stepcast/develop | `0e0901376` | path `<workspace>/.../ascend/vllm/` |

## Goal

End-to-end decode flow: a vLLM offline_inference call goes through the
**pypto step3p5 decoder kernel** (compiled via `pypto.ir.compile`) instead
of the upstream torch eager `Step3p5DecoderLayer.forward`. The user
opt-in surface is a single Python call:

```python
import pypto.step3p5.vllm_backend as B
B.install()
from vllm import LLM, SamplingParams
out = LLM(model="<jfs>/step3p5_flash_release_hf_mtp3_bf16/").generate(
    "Hello", SamplingParams(max_tokens=16)
)
```

Phase 20 makes this **not crash** with any sequence of 16 tokens out.
Precision parity vs upstream vLLM is Phase 21; perf measurement is
Phase 22.

## Scope

**In:**
- Whole-model monkey-patch on `Step3p5Model.forward` (one fused
  `decode_fwd` kernel call per forward, not 45 per-layer calls).
- Single-rank (TP=1) path with `apply_tp1_patch`-style unsliced
  widths — Phase 15 e2e proved this works for dense layers.
- Mixed-mode MoE: dense layers go through pypto; MoE layers (21/45)
  fall back to vLLM's `FusedMoEBlock` (host-side switch).
- Real weight load from HF safetensors at
  `<jfs>/step3p5_flash_release_hf_mtp3_bf16/`.
- KV cache and attention metadata bridged from vLLM forward context.
- `install()` / `uninstall()` / `is_installed()` public API.

**Out (deferred):**
- Multi-rank canonical TP=8 (Phase 22 + barrier all_reduce gate).
- Full MoE through pypto (Phase 22 + MoE 507018 gate).
- Per-layer monkey-patch granularity (kept as escape hatch for Phase 21
  precision-diff harness, see "Per-layer escape hatch" below).
- MTP integration (Phase 23+).
- Tokenizer / sampler — already in vLLM, untouched.

## Key decisions

| # | Decision | Reason |
|---|----------|--------|
| D1 | **Whole-model patch** on `Step3p5Model.forward`, not per-layer | pypto `decode_fwd` is a single fused 45-layer + lm_head program; 45 launches would erase the fusion benefit |
| D2 | **Comm option A**: pypto kernel uses simpler shmem-IPC comm internally; vLLM `tp_group` is unused inside pypto kernel | Avoids writing a simpler↔HCCL bridge; pypto kernel is self-contained |
| D3 | **Mixed-mode MoE** in Phase 20 | MoE device 507018 is a separate hard blocker; not on the e2e critical path |
| D4 | Code lives in `pypto-lib/models/step3p5/vllm_backend/`; vLLM repo unchanged | Reversible, no fork pollution |
| D5 | Per-layer hook surface kept available for Phase 21 (not used in Phase 20) | Enables layer-by-layer hidden_states diff in Phase 21 without re-architecting |

## Deliverables

```
pypto-lib/models/step3p5/vllm_backend/
├── __init__.py              # public API: install(), uninstall(), is_installed()
├── install.py               # monkey-patch dispatcher, stashes originals
├── weight_translate.py      # vLLM nn.Module -> pypto bundle dict
├── kv_bridge.py             # vLLM kv_cache layout -> pypto k_cache/v_cache view
├── attn_meta_bridge.py      # vLLM AttentionMetadata -> pypto seq_lens/block_table/slot_mapping
├── compile_cache.py         # rank-aware compiled kernel cache (compile once, reuse via runtime_dir)
├── mixed_moe.py             # MoE-layer fallback to vLLM's FusedMoEBlock
├── config_align.py          # assert vLLM hf_config matches pypto config.py constants
└── README.md
```

Plus exit test: `pypto-lib/tests/step3p5/test_vllm_backend_e2e.py`.

## Tasks

| # | Task | Output | Estimate |
|---|------|--------|----------|
| 1.1 | `config_align.py` — assert pypto config.py vs vLLM `hf_config` (NUM_HIDDEN_LAYERS=48, HIDDEN=4096, NUM_HEADS_FULL=64, NUM_KV_HEADS=8, INTERMEDIATE=11264, VOCAB=128896, BLOCK_SIZE=128, etc.); raise with diff on mismatch | green import | 1 d |
| 1.2 | `weight_translate.py:vllm_to_pypto_bundle(model)` — walk `model.named_parameters()`, regroup by `weight_loader.expected_shapes()` keys; handle `qkv_proj` Q/K/V split, `gate_up_proj` split, MoE `experts.w13_weight` split, vocab parallel embedding gather, lm_head TP slice | 30-key bundle dict | 5 d |
| 1.3 | `kv_bridge.py:make_kv_views(vllm_kv_cache_layer_i)` — vLLM `[num_blocks, block_size, num_kv_heads, head_dim]` BF16 → pypto `[KV_CACHE_ROWS_DYN, HEAD_DIM]` zero-copy `.view()`; assert layout strides compatible | (k_view, v_view) per layer | 3 d |
| 1.4 | `attn_meta_bridge.py:extract_pypto_meta(attn_metadata)` — pull `seq_lens / block_tables / slot_mapping` from vLLM's `AttentionMetadata`, transpose dtype/shape to pypto contract (INT32, flat block_table) | meta dict | 2 d |
| 1.5 | `compile_cache.py:get_compiled(rank, world_size)` — first call: `ir.compile(Step3p5DecodeFwd, distributed_config=DistributedConfig([rank,...]))`, persist `runtime_dir`; subsequent calls reuse `.so`/`.bin` | callable + `runtime_dir` | 2 d |
| 1.6 | `mixed_moe.py:hybrid_forward_dispatcher(model, positions, hidden_states)` — if every layer is dense → call pypto `decode_fwd`; if any MoE layer present → invoke pypto for the dense block, then re-enter Python and call `Step3p5MLP.forward` / `FusedMoEBlock.forward` for each MoE layer per the mixed-mode plan; reuse vLLM's `Step3p5Model.forward` skeleton with a hot-swappable dispatcher | dispatcher | 3 d |
| 1.7 | `install.py:install()` — backup `Step3p5Model.forward._orig = Step3p5Model.forward`, set new forward that calls `hybrid_forward_dispatcher`; also patch `vllm.entrypoints.api_server` init to call `B.install()` lazily if requested via env `PYPTO_STEP3P5_BACKEND=1` | install/uninstall API + idempotent | 2 d |
| 1.8 | `tests/step3p5/test_vllm_backend_e2e.py` — `B.install(); LLM(model="step3p5-flash-...").generate("Hello", SamplingParams(max_tokens=16, temperature=0.0))`; assert returns 16 token ids without crash | e2e smoke | 2 d |
| 1.9 | Bring-up debug + first green run on 0162 device 0 | green Phase 20 exit | 2 d |

## Exit criteria

```bash
# From a clean 0162 shell:
source /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh
source <workspace>/activate.sh
export PTO_ISA_ROOT=<workspace>/pto-isa
export PYPTO_STEP3P5_BACKEND=1
cd <workspace>/pypto-lib
python -m tests.step3p5.test_vllm_backend_e2e \
    --model-path <jfs>/step3p5_flash_release_hf_mtp3_bf16/ \
    --prompt "Hello, the future of AI is" \
    --max-tokens 16 \
    -p a2a3 -d 0
# Expected: exit 0; prints 16 token ids; no fault; no 507018.
```

Token output need NOT match upstream vLLM precisely; Phase 21 handles
precision. Phase 20 only proves the plumbing works.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| vLLM `qkv_proj` weight layout (concat order Q,K,V) doesn't match pypto's separate `wq/wk/wv` per-rank → silent wrong outputs | High | Task 1.2 must verify dimensional split with a synthetic test before integration; print expected vs got shapes |
| KV cache layout incompatible for zero-copy → must repack each call | Medium | Phase 20 accepts copy; Phase 22 perf round optimises if needed |
| RoPE cos/sin caching inside vLLM `get_rope` is global; pypto kernel expects `[SEQ, ROTARY_DIM]` per-rank | Low | Read vLLM's `RotaryEmbedding.cos_sin_cache` directly, feed in |
| `Step3p5Attention.use_head_wise_attn_gate=True` but pypto head_gate is bypassed (×1) | Numerical (handled in Phase 21) | Document in code; Phase 21 §2.7 calibration |
| First-compile takes ~30s, vLLM may time out warmup | Low | Cache `.so`/`.bin` via `runtime_dir`; pre-compile script |

## Per-layer escape hatch

Even though Phase 20 patches at whole-model level, the install layer
exposes:

```python
B.install(per_layer=True)  # Phase 21 only
```

which switches to patching `Step3p5DecoderLayer.forward` instead. Each
layer call invokes a smaller compiled kernel (`decode_layer_full_dense`
etc., already validated by Phase 19 ST). Slower but allows hook-based
per-layer hidden_states diff for precision validation.

## Status

- 2026-06-22: design landed (this doc).
- Tasks 1.1-1.9 NOT STARTED.
- Critical path. ~3-4 weeks to e2e green.

## References

- [`docs/known-pypto-pitfalls.md`](../../known-pypto-pitfalls.md) — kernel hard limits
- [`docs/dev-workflow-gotchas.md`](../../dev-workflow-gotchas.md) — dev workflow pitfalls
- [`21-precision-validation.md`](21-precision-validation.md) — Phase 21 follow-up
- vLLM reference: `<vllm_repo>/vllm/model_executor/models/step3p5.py` HEAD 0e0901376
- pypto-lib reference: `models/step3p5/decode_fwd.py:198 _build_decode_fwd_program`,
  `models/step3p5/weight_loader.py:197 expected_shapes`
