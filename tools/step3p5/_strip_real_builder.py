"""Strip the generated real builder from decode_layer.py so _gen_faithful_real
can regenerate it (the generator refuses if the real builder already exists).
Removes the inserted block: the leading blank line(s) + def
_build_whole_decode_faithful_real_program ... through the binding line
`whole_decode_faithful_real = ...()`. Preserves the latest base builder and
everything before/after the inserted block. One-shot regen helper.
"""
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "models" / "step3p5" / "decode_layer.py"

t = SRC.read_text()
start_marker = "\n\ndef _build_whole_decode_faithful_real_program("
bind_marker = "whole_decode_faithful_real = _build_whole_decode_faithful_real_program()"
assert t.count("def _build_whole_decode_faithful_real_program(") == 1, "expected exactly one real builder"
a = t.index(start_marker)
b = t.index(bind_marker)
b = t.index("\n", b) + 1  # through the binding line's newline
new = t[:a] + "\n" + t[b:]
assert "_build_whole_decode_faithful_real_program" not in new, "strip incomplete"
SRC.write_text(new)
print(f"[strip] removed real builder span [{a}:{b}] ({b - a} chars); new len {len(new)}")
