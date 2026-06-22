# Step3p5 Phase Documents (in-repo)

This directory hosts the design + tracking documents for step3p5
development phases that live **inside the pypto-lib git tree** and are
versioned + pushable.

## Why two homes for phase docs

| Home | What lives there | Versioned |
|------|------------------|-----------|
| `<workspace>/pypto/docs/step3p5/phases/` (outer tracker) | Phases 01-19 (historical bring-up, design + post-mortems) | No — loose files on the dev host |
| `pypto-lib/docs/step3p5/phases/` (this dir) | Phases 20+ (vLLM backend integration, precision validation, perf tuning) | Yes — in git, pushed to `csy0225/pypto-lib` `stepfun/develop` |

Phases 01-19 stayed in the outer tracker because they predate the
decision to version phase docs alongside model code. Starting Phase 20
(2026-06-22) new phase docs live in-repo so they ship with the code
they describe.

## Index

| Phase | Title | Status | Doc |
|------:|-------|--------|-----|
| 20 | vLLM backend monkey-patch — e2e flow | Design landed 2026-06-22; tasks NOT STARTED | [20-vllm-backend-monkey-patch.md](20-vllm-backend-monkey-patch.md) |
| 21 | Precision validation harness (vs upstream vLLM) | Design landed 2026-06-22; gated on Phase 20 | [21-precision-validation.md](21-precision-validation.md) |
| 22 | Perf baseline + tuning | Design landed 2026-06-22; gated on Phase 21 + multi-card gates | [22-perf-baseline.md](22-perf-baseline.md) |

## Update protocol

Per the outer tracker's CLAUDE.md, when a phase node completes or a
key decision lands:

1. Update the phase doc's `## Status` section.
2. Update the row in the table above.
3. Update the outer tracker CLAUDE.md phase timeline table with the
   pin SHA + completion date.
4. Record the new pin snapshot at the top of any new phase doc.

## Cross-references

- [`../../known-pypto-pitfalls.md`](../../known-pypto-pitfalls.md) — pypto / pto-isa / simpler hard limits
- [`../../dev-workflow-gotchas.md`](../../dev-workflow-gotchas.md) — non-pypto dev workflow gotchas (stale pyc, activation, git auth)
- [`../../debugging.md`](../../debugging.md) — pypto kernel debugging playbook
- Outer tracker `<workspace>/pypto/CLAUDE.md` — project-level multi-repo tracker
