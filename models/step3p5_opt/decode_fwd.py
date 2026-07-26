"""Compatibility shim for the canonical Step3p5 loop-form Main.

The loop-form implementation was promoted to ``models.step3p5.decode_fwd``.
Keep this module for older harnesses and sidecars that still import the
historical ``step3p5_opt`` path; all exported program and configuration
symbols resolve to the canonical module, so this shim does not define a
second PyPTO program or duplicate compilation state.
"""
from __future__ import annotations

from models.step3p5 import decode_fwd as _canonical
from models.step3p5.decode_fwd import *  # noqa: F401,F403
from models.step3p5.decode_fwd import (
    WholeDecodeStep3p5,
    whole_decode_step3p5,
)

# Historical class/program names remain import-compatible and point at the
# exact same canonical objects.
WholeDecodeOpt = WholeDecodeStep3p5
whole_decode_opt = whole_decode_step3p5


def __getattr__(name: str):
    """Forward private/module-level constants used by legacy tooling."""
    return getattr(_canonical, name)
