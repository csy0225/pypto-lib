"""One-shot transform: fix the combine zero-vs-push data race in every
`_push_routed_y_to_sources` in models/step3p5/decode_layer.py.

Root cause of the residual NUMERICAL non-determinism (P=42 argmax 303/20/303,
row0|next_hidden| 250-608 across identical runs) after the A2 hang was fixed:
the whole-net combine does
    self._zero_routed_y_buf(routed_y_buf)   # local zero
    self._push_routed_y_to_sources(...)     # peers pld.tensor.put INTO routed_y_buf
with NO cross-rank barrier between the zero and the pushes. A fast peer's
`pld.tensor.put` into rank R's routed_y_buf can land BEFORE R runs its local
zero -> R's zero clobbers the pushed data -> racy moe_out. Low probability per
layer, compounds over 42 layers -> occasional argmax flip.

Reference moe.py `combine_step` HAS this barrier: zero -> _publish_src_route_table
-> pub_route_barrier (route_pub_sig) -> push. This port replaced src_route_table
with recv_r_route and DROPPED the publish + its barrier, removing the zero<->push
sync.

Fix (two-wave on the EXISTING combine_done window, no new window):
  * escalate the existing end barrier wait expected 1 -> 2 (post-push, wave 2);
  * insert a NEW wave-1 barrier at the TOP of _push (notify(+1) -> wait Ge 1).
    Reaching the top of _push == finished zero (zero is called immediately
    before _push in the caller), so wave 1 makes every rank wait until all ranks
    zeroed before any push. Cells go 0 -> 1 (wave1) -> 2 (wave2).
Idempotent (marker: "zero-done barrier (wave 1)"). Escalation runs BEFORE
insertion, so each function has exactly one combine_done wait when escalated.
"""

import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parents[2] / "models" / "step3p5" / "decode_layer.py"
MARKER = "zero-done barrier (wave 1)"

WAVE1 = (
    "{i}# zero-done barrier (wave 1): every rank must finish _zero_routed_y_buf\n"
    "{i}# (called immediately before this push in the combine caller) before any\n"
    "{i}# peer pld.tensor.put lands, else a fast peer's push into a not-yet-zeroed\n"
    "{i}# routed_y_buf is clobbered by the local zero -> racy moe_out. Mirrors\n"
    "{i}# moe.py combine_step's pub_route_barrier (dropped with src_route_table).\n"
    "{i}# Same combine_done window, two-wave: wave 1 =1 here, post-push barrier =2.\n"
    "{i}for peer in pl.range(n_ranks):\n"
    "{i}    if peer != my_rank:\n"
    "{i}        pld.system.notify(\n"
    "{i}            target=combine_done, peer=peer,\n"
    "{i}            offsets=[my_rank, 0], value=1,\n"
    "{i}            op=pld.NotifyOp.AtomicAdd,\n"
    "{i}        )\n"
    "{i}for src in pl.range(n_ranks):\n"
    "{i}    if src != my_rank:\n"
    "{i}        pld.system.wait(\n"
    "{i}            signal=combine_done, offsets=[src, 0],\n"
    "{i}            expected=1, cmp=pld.WaitCmp.Ge,\n"
    "{i}        )\n"
)


def find_pushes(lines):
    """Yield (def_line_idx, sig_close_idx, body_end_idx) for each _push func."""
    n = len(lines)
    i = 0
    while i < n:
        st = lines[i].lstrip()
        ind = len(lines[i]) - len(st)
        if st.startswith("def _push_routed_y_to_sources("):
            def_indent = ind
            j = i + 1
            sig_close = None
            while j < n:
                sj = lines[j].lstrip()
                indj = len(lines[j]) - len(sj)
                if sj.rstrip("\n") == "):" and indj == def_indent:
                    sig_close = j
                    j += 1
                    break
                j += 1
            # body end = next def at <= def_indent
            k = j
            while k < n:
                sk = lines[k].lstrip()
                indk = len(lines[k]) - len(sk)
                if sk.startswith("def ") and indk <= def_indent:
                    break
                k += 1
            yield (i, sig_close, k, def_indent)
            i = k
            continue
        i += 1


def main() -> int:
    lines = TARGET.read_text().splitlines(keepends=True)
    spans = list(find_pushes(lines))
    if not spans:
        print("[combine-race] WARNING: 0 _push_routed_y_to_sources found.")
        return 1
    # Process in REVERSE order so earlier line indices stay valid after edits.
    esc = 0
    ins = 0
    already = 0
    for (def_i, sig_i, end_i, def_indent) in reversed(spans):
        if sig_i is None:
            print(f"[combine-race] WARNING: no sig close for def at {def_i}")
            return 1
        body_indent = def_indent + 4
        block = lines[sig_i + 1 : end_i]
        if any(MARKER in b for b in block):
            already += 1
            continue
        # 1) escalate the single combine_done wait expected=1 -> 2 within [sig_i+1,end_i)
        did_esc = False
        for t in range(sig_i + 1, end_i):
            if lines[t].strip() == "expected=1," and "signal=combine_done" in "".join(
                lines[max(sig_i + 1, t - 4) : t]
            ):
                lines[t] = lines[t].replace("expected=1,", "expected=2,")
                esc += 1
                did_esc = True
                break
        if not did_esc:
            print(f"[combine-race] WARNING: no combine_done wait to escalate in def at {def_i}")
            return 1
        # 2) insert wave-1 right after the signature close line
        lines.insert(sig_i + 1, WAVE1.format(i=" " * body_indent))
        ins += 1

    if ins == 0 and already > 0:
        print(f"[combine-race] idempotent: all {already} _push already patched.")
        return 0
    TARGET.write_text("".join(lines))
    print(f"[combine-race] wave1 inserted={ins} end-barrier escalated={esc} already={already}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
