"""One-shot transform: add the missing completion wave (Phase 4) to every
hand-rolled `tp_all_reduce` in models/step3p5/decode_layer.py.

Root cause of the A2 collective deadlock (507018 / S1:running-stalled at >=41
pipelined layers): our `tp_all_reduce` is a SINGLE-WAVE barrier
(Phase 2: notify(+1) -> wait(Ge,1); Phase 3: remote_load+accumulate; return),
mirroring a test. The framework's canonical collective (pld.tensor.allreduce,
pypto/.../distributed/op/tensor_ops.py:397) uses TWO barrier waves precisely to
avoid "racing Phase 3 against the previous reduction's Phase 4". Without the
completion wave, nothing guarantees all ranks finished Phase 3 (the remote
reads) before the call returns, so at pipelined depth adjacent per-layer
collectives race -> a collective kernel spins forever (S1) -> 507018.

Fix: append a Phase-4 completion barrier on the SAME signal_window with an
escalated threshold (AtomicAdd(+1) -> wait(Ge,2)), so every cell ends at 2 and
no rank returns until all ranks confirmed they finished Phase 3. This is the
exact two-wave protocol the framework documents. Idempotent.
"""

import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parents[2] / "models" / "step3p5" / "decode_layer.py"

# The completion-wave body. Indent placeholder {i} = the body indent of the
# tp_all_reduce function (i.e. the indent of `return local`). Uses only
# group_size / my_rank / signal_window, so it is identical for full/swa and any
# BATCH/HIDDEN. Same window, threshold escalated 1 -> 2 (framework two-wave).
WAVE = (
    "{i}# Phase 4: completion barrier (framework two-wave protocol). Ensures every\n"
    "{i}# rank finished Phase 3 reads before returning, so the next layer's\n"
    "{i}# collective cannot race this one's reads. Single-wave (Phase 2 only) hung\n"
    "{i}# at >=41 pipelined layers (507018 / S1:running-stalled). Same signal_window,\n"
    "{i}# threshold escalated 1 -> 2.\n"
    "{i}for peer in pl.range(group_size):\n"
    "{i}    if peer != my_rank:\n"
    "{i}        pld.system.notify(\n"
    "{i}            target=signal_window, peer=peer,\n"
    "{i}            offsets=[my_rank, 0], value=1,\n"
    "{i}            op=pld.NotifyOp.AtomicAdd,\n"
    "{i}        )\n"
    "{i}for src in pl.range(group_size):\n"
    "{i}    if src != my_rank:\n"
    "{i}        pld.system.wait(\n"
    "{i}            signal=signal_window, offsets=[src, 0],\n"
    "{i}            expected=2, cmp=pld.WaitCmp.Ge,\n"
    "{i}        )\n"
)

MARKER = "# Phase 4: completion barrier"


def main() -> int:
    text = TARGET.read_text()
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    in_tar = False
    tar_indent = None  # indent of `def tp_all_reduce`
    patched = 0
    already = 0
    i = 0
    while i < len(lines):
        ln = lines[i]
        stripped = ln.lstrip()
        indent = len(ln) - len(stripped)
        if stripped.startswith("def tp_all_reduce("):
            in_tar = True
            tar_indent = indent
            out.append(ln)
            i += 1
            continue
        if in_tar:
            # `return local` at body indent (tar_indent + 4) ends this collective.
            if stripped.rstrip("\n") == "return local" and indent == tar_indent + 4:
                prev_block = "".join(out[-25:])
                if MARKER in prev_block:
                    already += 1
                else:
                    out.append(WAVE.format(i=" " * indent))
                    patched += 1
                out.append(ln)
                in_tar = False
                tar_indent = None
                i += 1
                continue
            # A new def at or below the tar indent means the collective ended
            # without a matched return (defensive) — stop tracking, reprocess.
            if stripped.startswith("def ") and indent <= tar_indent:
                in_tar = False
                tar_indent = None
                continue
        out.append(ln)
        i += 1

    new_text = "".join(out)
    if patched == 0 and already > 0:
        print(f"[wave] idempotent: all {already} tp_all_reduce already have the completion wave.")
        return 0
    if patched == 0:
        print("[wave] WARNING: matched 0 tp_all_reduce return sites — nothing changed.")
        return 1
    TARGET.write_text(new_text)
    print(f"[wave] patched {patched} tp_all_reduce (already had: {already}). wrote {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
