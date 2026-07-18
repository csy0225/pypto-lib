# Step3p5 Test Layout

The operator-facing whole-network entry point is documented in
[`../../docs/step3p5/README.md`](../../docs/step3p5/README.md). Start there for
the canonical main → sampler → MTP3 commands; this file describes the
responsibility-based test layout only.

The Step3p5 test tree is grouped by responsibility:

```text
common/      shared topology/configuration patches used by tests
probes/      focused compiler, lowering, and runtime diagnostics
harnesses/   device, real-weight, IPC, and canonical execution programs
unit/        card-free and focused unit/contract tests
system/      layer-level simulator and multi-device system tests
precision/   golden, W8A8, detailed, and end-to-end precision tests
ci/          whole-network CI runner, pytest gate, tests, and documentation
```

The top-level package intentionally contains only this index plus compatibility
entry points for commands that are part of the 0162 canonical workflow:

```text
tests.step3p5.harnesses._stage_whole_faithful_real_ipc
tests.step3p5.harnesses._stage_whole_mtp3_ipc
tests.step3p5.ci.run_whole_network_ci
```

New code should import and invoke the categorized modules directly. The
compatibility modules exist so pinned commands and external automation can be
migrated without changing the validated execution behavior in one step.
