# Step3p5 routed-NZ external kernels

The C++ files in this directory are the checked-in source authority for the
Step3p5 routed-expert external ABI. They were initially imported from PyPTO
compiler output, but the originating DSL program is not part of this
repository. Do not replace them from an unrelated build artifact or assume
that generated SSA names form a stable interface.

The maintained release semantics are:

- one fixed 128-row slab for each of 36 owner-local experts;
- mixed GMM1/SwiGLU/requant execution over the compact active-expert plan;
- a fully initialized 16-row activation tile before fixed-shape quantization;
- no zero-valid or negative-valid GM load/store for an empty 8-row half;
- a fixed 32768-float routed-down pipe workspace for each logical worker.

Manual changes must keep the `PYPTO-LIB-AUTHORITY` regions and their semantic
contract tests synchronized. Validation must use the pinned release image and
include CCEC AIC/AIV compilation, whole-model A2/A3 compilation, BS1 and BS16
device runs, and the Step3p5 precision gate. Compile-only checks do not replace
the device runs.

A future regeneration workflow may replace these files only when it vendors
the exact DSL source, reproduces the external argument ABI, and preserves the
same semantic contracts. Until then, the checked-in C++ remains authoritative.
