#!/usr/bin/env python3
"""Card-free tests for the canonical fail-closed decode gate.

The only real-request path is the single-chip PyPTO sidecar.  Profile/dummy
calls may retain vLLM's harmless no-op; real requests never fall back to
vanilla or per-layer execution.

Run:
    python3 -m pytest tests/step3p5/unit/test_decode_gate_classifier.py -q
or:
    python3 tests/step3p5/unit/test_decode_gate_classifier.py
"""
from __future__ import annotations

import os
import sys
import unittest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.step3p5.vllm_monkey_patch import (  # noqa: E402
    GATE_FAIL_CLOSED,
    GATE_PROCEED,
    GATE_PROFILE_NOOP,
    classify_decode_gate,
)


class TestDecodeGateClassifier(unittest.TestCase):
    def test_profile_call_is_noop_even_without_sidecar(self):
        # Not a real request => profile/dummy/warmup => safe no-op regardless
        # of sidecar/eligibility.
        for sc in (True, False):
            for el in (True, False):
                self.assertEqual(
                    classify_decode_gate(
                        is_real_request=False,
                        sidecar_available=sc,
                        eligible=el,
                    ),
                    GATE_PROFILE_NOOP,
                )

    def test_real_request_sidecar_up_and_eligible_proceeds(self):
        self.assertEqual(
            classify_decode_gate(
                is_real_request=True,
                sidecar_available=True,
                eligible=True,
            ),
            GATE_PROCEED,
        )

    def test_real_request_sidecar_down_tail_only_fails_closed(self):
        # The critical §5.1 case: tail-only instance, sidecar absent, real
        # request => MUST fail closed (never silent wrong token).
        self.assertEqual(
            classify_decode_gate(
                is_real_request=True,
                sidecar_available=False,
                eligible=False,
            ),
            GATE_FAIL_CLOSED,
        )

    def test_real_request_ineligible_tail_only_fails_closed(self):
        # Sidecar up but request not pure-decode-eligible (e.g. prefill/batch>16)
        # on a tail-only instance => fail closed (no correct fallback).
        self.assertEqual(
            classify_decode_gate(
                is_real_request=True,
                sidecar_available=True,
                eligible=False,
            ),
            GATE_FAIL_CLOSED,
        )

    def test_gate_codes_are_distinct(self):
        self.assertEqual(
            len({GATE_FAIL_CLOSED, GATE_PROCEED, GATE_PROFILE_NOOP}),
            3,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
