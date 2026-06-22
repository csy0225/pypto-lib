# Phase 21 — Precision Validation Harness (vs upstream vLLM)

> **Component pin snapshot (at doc creation, 2026-06-22)**
>
> | Repo | Branch | Pin | Notes |
> |------|--------|-----|-------|
> | pypto-lib | `stepfun/develop` | `a6b5faa` (pre-commit) | Will become the commit that adds this file |
> | pypto | `stepfun/develop` | `b00c8b23` | |
> | pto-isa | `stepfun/develop` | `e25732f0` | |
> | PTOAS | `stepfun/develop` | `da011a3d` | binary v0.45 |
> | simpler (submodule) | — | `a6e06406` | |
> | vLLM (reference) | stepcast/develop | `0e0901376` | comparison baseline |

## Goal

Build a precision validation harness that runs the **same** prompt
through (a) upstream vLLM (`Step3p5Model.forward` torch eager) and
(b) pypto-backed vLLM (`B.install()` first), then compares outputs at
three granularities. **Phase 21 PASS is the entry criterion for Phase 22
performance work** — running perf without precision parity reports
meaningless numbers.

## Scope

**In:**
- Dual-engine harness: same prompt, same seed, two vLLM engines —
  one untouched, one with `pypto.step3p5.vllm_backend.install()`.
- Three-tier comparison (L1 / L2 / L3, see below).
- Tolerance specification matching Phase 19 ST conventions.
- Coverage matrix across attention type (full / SWA), batch size,
  prompt category, decode step count.
- CI-runnable: `pytest tests/step3p5/test_vllm_backend_*.py` returns
  green or actionable diagnostics.

**Out:**
- Multi-rank precision tests (Phase 22 + multi-rank gate).
- MoE precision parity beyond mixed-mode (Phase 22 + MoE 507018 fix).
- Numerical-debug tooling beyond what's needed to reach exit criterion.

## Three-tier comparison

| Tier | Object | Tolerance | What it tests |
|------|--------|-----------|---------------|
| **L1 — per-layer** | post-layer `hidden_states` `[B, H]` after each `Step3p5DecoderLayer.forward` (or per-fused-block when whole-model patched) | `ratio_allclose(atol=0.04, rtol=0.04, max_error_ratio=0.10)` (Phase 19 ST convention) | kernel-level numerical correctness; isolates which layer/op drifts |
| **L2 — per-token logits** | sampled-step logits `[B, VOCAB]` before sampling | cosine_similarity ≥ 0.999 **and** top-K (K=5) overlap ≥ 4/5 | numerical stability + lm_head + final RMS gather correctness |
| **L3 — per-token sampling** | sampled `token_id` per decode step (greedy, temperature=0) | top-1 match rate ≥ 95% over N=64 decode steps × 16 prompts | end-user-visible behavioural parity |

L1 is the strongest signal but the noisiest; L3 is what we ship against.
A failing L1 with a passing L3 means precision drifts but stays inside
the argmax basin — acceptable for some workloads, flagged otherwise.

## Tolerance rationale

The L1 numbers are inherited from Phase 19 dense ST device runs which
passed under these bands on bf16 paths with FP32 accumulators. Phase 21
does **not** widen them — if `decode_fwd` end-to-end blows past
`atol=0.04`, that is a real regression vs Phase 19 unit-tested behaviour.

The L2 cosine threshold of 0.999 is calibrated against the head_gate
×1 bypass: bypassing the sigmoid gate scales attention-out by ~2×
relative to upstream, but post-RMS+lm_head the cosine remains high.
Empirically observed bypass gives cos ≈ 0.9995 on Phase 15 single-card
runs.

The L3 95% threshold matches typical greedy-decode top-1 match between
two implementations of the same model when one has a known bf16
quantisation difference vs the other.

## Coverage matrix

| Variable | Values | Why |
|----------|--------|-----|
| Attention type | full (24 layers), SWA (21 layers) | both must work; SWA has narrower KV — different paged path |
| Batch size | 1, 4, 16 | catches batch-dim broadcast bugs; pypto kernels are tile-specialised at 16 |
| Prompt category | en-short / en-long / zh / code / math | exercises different vocab regions and sequence lengths |
| Decode steps | 1, 16, 64 | catches KV-cache append bugs that only surface after several appends |
| MoE mode | `mixed` (Phase 21 default), `dummy0` (sanity) | Phase 21 cannot test full pypto MoE until 507018 is fixed |

The full matrix is `2 × 3 × 5 × 3 × 2 = 180` test cases. Subset for CI
(fast tier) is `1 × 1 × 5 × 16 × 1 = 80` runs.

## Deliverables

```
pypto-lib/tests/step3p5/
├── _vllm_precision_harness.py    # DualRunHarness + hook helpers
├── _vllm_test_prompts.py         # 16 fixed prompts (en/zh/code/math x short/long)
├── test_vllm_backend_per_layer.py    # L1
├── test_vllm_backend_per_token.py    # L2
└── test_vllm_backend_decode_n.py     # L3
```

Plus reports under `build_output/precision_reports/<timestamp>/`:

- `per_layer.json` — per-(layer_idx, prompt_idx, step_idx) max abs/rel diff,
  ratio_allclose pass/fail
- `per_token.json` — per-(prompt_idx, step_idx) cos / top-K overlap
- `per_decode.json` — per-prompt top-1 match rate over 64 steps

## Tasks

| # | Task | Output | Estimate |
|---|------|--------|----------|
| 2.1 | `_vllm_precision_harness.py:DualRunHarness` — same-prompt same-seed two engines, forward hook to capture intermediate states; supports `per_layer=True` mode that uses Phase 20 escape hatch | reusable harness class | 4 d |
| 2.2 | `_vllm_test_prompts.py` — 16 fixed prompts | test inputs | 0.5 d |
| 2.3 | L1 per-layer test — for each prompt+step, hook 45 layer outputs both sides, run `ratio_allclose`; emit failing layer table | `test_vllm_backend_per_layer.py` | 3 d |
| 2.4 | L2 per-token logits test — capture `[B, VOCAB]` pre-sample logits; cos + top-K overlap | `test_vllm_backend_per_token.py` | 2 d |
| 2.5 | L3 per-token sampling test — greedy decode with `temperature=0`; top-1 match rate aggregated | `test_vllm_backend_decode_n.py` | 2 d |
| 2.6 | Coverage matrix parametrize — pytest fixture sweeps `(att_type, batch, prompt_cat, n_steps)` | parametrized tests | 2 d |
| 2.7 | head_gate x1 calibration — patch upstream vLLM's `Step3p5Attention` to also bypass the gate (multiply by 1.0) for parity, OR widen L1 tolerance for the head_gate-affected path; document choice | calibration patch / tolerance note | 2 d |
| 2.8 | CI hookup — local CI script + (optional) GitHub Actions workflow | green CI | 2 d |

## Exit criteria

For the fast-tier subset (80 runs):

```bash
pytest tests/step3p5/test_vllm_backend_per_layer.py -v   # 45 layers x 16 prompts -> 100% layer-pass
pytest tests/step3p5/test_vllm_backend_per_token.py -v   # 16 prompts x 16 steps -> cos>=0.999 + topK>=4/5 100%
pytest tests/step3p5/test_vllm_backend_decode_n.py -v    # 16 prompts x 64 steps -> top-1 match rate >= 95%
```

All three green. JSON reports archived for trend tracking.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| L1 hidden_states cannot pass under head_gate ×1 bypass — semantic divergence is irreducible without TASK-L (cube-matmul block-diag R) | High | Task 2.7 — patch vLLM-side gate to ×1 to align baselines, OR carve head_gate-affected layers into a separate weaker tolerance band |
| RNG state divergence between two engines (sampling, dropout init, weight init order) makes L2/L3 noisy | Medium | Force `temperature=0`, fixed seed, deterministic init order; assert no dropout active in inference |
| vLLM's `FusedMoEBlock` (used for MoE-mode in mixed-mode) has its own quantisation/fusion; not byte-identical to mathematical reference | Medium | MoE layers: Phase 21 trusts vLLM's fused output as the reference (i.e. mixed-mode means MoE layers are baseline, not test target) |
| Hooking 45 layers per step costs significant overhead, slows tests | Low | Run L1 only on subset of steps (1, 8, 64); use Phase 20 whole-model path for L2/L3 |
| KV cache layout differs across runs causing different attention output | Low | Pre-fill KV with identical synthetic data on both engines for L1; for L3 let it diverge naturally and only watch sampled tokens |

## Status

- 2026-06-22: design landed (this doc).
- Tasks 2.1-2.8 NOT STARTED. Gated on Phase 20 completion (need
  `B.install()` working).
- Estimate ~3-4 weeks once Phase 20 lands.

## References

- [`20-vllm-backend-monkey-patch.md`](20-vllm-backend-monkey-patch.md) — Phase 20 prerequisite
- [`22-perf-baseline.md`](22-perf-baseline.md) — Phase 22 follow-up, gated on this Phase
- Phase 19 dense ST tolerances: `tests/step3p5/test_decode_layer_full_dense_st.py`
- vLLM reference Step3p5 model: `<vllm_repo>/vllm/model_executor/models/step3p5.py`
