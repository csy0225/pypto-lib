#!/usr/bin/env python3
"""Card-free tests for the canonical fail-closed prefill gate.

The only real-request path is the single-chip PyPTO sidecar.  Profile/dummy
calls may retain vLLM's harmless no-op; real prefills never fall back to
vanilla or per-layer execution.

``classify_prefill_gate`` returns one of three string states:
``GATE_PREFILL_PROCEED`` (num_prefills>0 and eligible), ``GATE_DECODE_PROCEED``
(num_prefills==0), or ``GATE_REJECT`` (any eligibility violation / sidecar down).

Run:
    python3 -m pytest tests/step3p5/unit/test_prefill_gate_classifier.py -q
or:
    python3 tests/step3p5/unit/test_prefill_gate_classifier.py
"""
from __future__ import annotations

import os
import sys
import unittest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.step3p5.vllm_monkey_patch import (  # noqa: E402
    GATE_DECODE_PROCEED,
    GATE_PREFILL_PROCEED,
    GATE_PROCEED,
    GATE_REJECT,
    classify_decode_gate,
    classify_prefill_gate,
)

# Mirrors models.step3p5.prefill_qkv_proj_rope.PREFILL_T and the gate contract.
_PREFILL_T = 128


class TestPrefillGateClassifier(unittest.TestCase):
    def test_valid_single_prefill_proceeds(self):
        # Eligible: T within budget, no chunked-prefill, no CP/PCP, PP==1,
        # single KV group, sidecar up.
        self.assertEqual(
            classify_prefill_gate(
                None,
                None,
                num_prefills=1,
                T=_PREFILL_T,
                chunked_prefill=False,
                context_parallel=1,
                pipeline_parallel=1,
                num_kv_groups=1,
                sidecar_available=True,
            ),
            GATE_PREFILL_PROCEED,
        )

    def test_pure_decode_proceeds_on_decode_path(self):
        # num_prefills==0 is pure decode => defer to the decode path.
        self.assertEqual(
            classify_prefill_gate(None, None, num_prefills=0),
            GATE_DECODE_PROCEED,
        )

    def test_reject_chunked_prefill(self):
        self.assertEqual(
            classify_prefill_gate(
                None, None, num_prefills=1, T=_PREFILL_T, chunked_prefill=True
            ),
            GATE_REJECT,
        )

    def test_reject_context_parallel(self):
        # CP/PCP is unsupported on the single-chip path.
        self.assertEqual(
            classify_prefill_gate(
                None, None, num_prefills=1, T=_PREFILL_T, context_parallel=2
            ),
            GATE_REJECT,
        )

    def test_reject_pipeline_parallel_and_t_over_limit(self):
        # PP>1 and T exceeding PREFILL_T are both disqualifying.
        self.assertEqual(
            classify_prefill_gate(
                None, None, num_prefills=1, T=_PREFILL_T, pipeline_parallel=2
            ),
            GATE_REJECT,
        )
        self.assertEqual(
            classify_prefill_gate(None, None, num_prefills=1, T=_PREFILL_T + 1),
            GATE_REJECT,
        )

    def test_consistency_with_decode_gate_on_decode_only(self):
        # A decode-only batch agrees with classify_decode_gate: both reach a
        # "proceed" verdict for a pure-decode, sidecar-up, eligible request.
        self.assertEqual(
            classify_prefill_gate(None, None, num_prefills=0, sidecar_available=True),
            GATE_DECODE_PROCEED,
        )
        self.assertEqual(
            classify_decode_gate(
                is_real_request=True, sidecar_available=True, eligible=True
            ),
            GATE_PROCEED,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
