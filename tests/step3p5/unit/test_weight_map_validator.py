#!/usr/bin/env python3
"""Card-free unit tests for the PyPTO weight pool-map validator.

These are stdlib-only (no torch, no device, no pypto): they exercise the pure
``validate_weight_map`` contract that the model loader uses to fail closed
before declaring a TP rank *ready*.

Run:
    python3 -m pytest tests/step3p5/unit/test_weight_map_validator.py -q
or, with no pytest:
    python3 tests/step3p5/unit/test_weight_map_validator.py
"""
from __future__ import annotations

import os
import sys
import unittest

# Make ``tools`` importable when run directly (repo root is three levels up).
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.step3p5.pypto_weight_ipc import (  # noqa: E402
    WeightMapInvalid,
    validate_weight_map,
)

_ALIGN = 512


def _align_up(n: int, a: int = _ALIGN) -> int:
    return (n + a - 1) // a * a


def _entry(offset: int, shape, dtype: str) -> dict:
    itemsize = {"float32": 4, "bfloat16": 2, "float16": 2, "int8": 1}[dtype]
    nbytes = 1
    for s in shape:
        nbytes *= int(s)
    nbytes *= itemsize
    return {"offset": offset, "shape": list(shape), "dtype": dtype, "nbytes": nbytes}


def _build_map(entries_spec, *, version: int = 1) -> dict:
    """Build a well-formed, contiguous, 512-aligned map from (key,shape,dtype)."""
    offset = 0
    mp: dict[str, dict] = {}
    for key, shape, dtype in entries_spec:
        e = _entry(offset, shape, dtype)
        mp[key] = e
        offset = _align_up(offset + e["nbytes"])
    return {
        "version": version,
        "rank": 0,
        "tp_world_size": 8,
        "pool_bytes": max(offset, _ALIGN),
        "pool_dtype_bytes": 2,
        "map": mp,
    }


# A representative native-W8A8 rank map: routed weights INT8, scales FP32,
# router FP32, a couple of BF16 projection matrices, one FP32 norm.
_W8A8_SPEC = [
    ("wq_full", (1024, 4096), "bfloat16"),
    ("input_rms_weight", (4096,), "float32"),
    ("moe_gate_w", (48, 4096), "float32"),
    ("moe_router_bias", (48,), "float32"),
    ("moe_w_gate_r", (36, 2048, 4096), "int8"),
    ("moe_w_up_r", (36, 2048, 4096), "int8"),
    ("moe_w_down_r", (36, 4096, 2048), "int8"),
    ("moe_w_gate_r_scale", (36, 2048), "float32"),
    ("moe_w_up_r_scale", (36, 2048), "float32"),
    ("moe_w_down_r_scale", (36, 4096), "float32"),
]


class TestWeightMapValidator(unittest.TestCase):
    def test_valid_native_w8a8_map_passes(self):
        mp = _build_map(_W8A8_SPEC)
        validate_weight_map(mp, native_w8a8=True)  # must not raise

    def test_bf16_dequant_routed_weight_rejected(self):
        """Hard-constraint 4: routed weight as BF16 is a forbidden fallback.

        The map is laid out CLEANLY with the routed weight already BF16 (so the
        structural overlap/nbytes checks pass) — the native-W8A8 dtype check is
        the thing that must fail closed.
        """
        spec = [(k, s, ("bfloat16" if k == "moe_w_gate_r" else d)) for k, s, d in _W8A8_SPEC]
        mp = _build_map(spec)
        with self.assertRaises(WeightMapInvalid) as ctx:
            validate_weight_map(mp, native_w8a8=True)
        self.assertIn("moe_w_gate_r", str(ctx.exception))
        self.assertIn("int8", str(ctx.exception))

    def test_routed_scale_must_be_fp32(self):
        spec = [(k, s, ("bfloat16" if k == "moe_w_up_r_scale" else d)) for k, s, d in _W8A8_SPEC]
        mp = _build_map(spec)
        with self.assertRaises(WeightMapInvalid) as ctx:
            validate_weight_map(mp, native_w8a8=True)
        self.assertIn("moe_w_up_r_scale", str(ctx.exception))

    def test_router_weight_must_be_fp32(self):
        spec = [(k, s, ("bfloat16" if k == "moe_gate_w" else d)) for k, s, d in _W8A8_SPEC]
        mp = _build_map(spec)
        with self.assertRaises(WeightMapInvalid) as ctx:
            validate_weight_map(mp, native_w8a8=True)
        self.assertIn("moe_gate_w", str(ctx.exception))

    def test_unaligned_offset_rejected(self):
        mp = _build_map(_W8A8_SPEC)
        # Shift one entry off the 512-byte boundary.
        mp["map"]["wq_full"]["offset"] = 8
        with self.assertRaises(WeightMapInvalid) as ctx:
            validate_weight_map(mp, native_w8a8=False)
        self.assertIn("aligned", str(ctx.exception))

    def test_overlap_rejected(self):
        mp = _build_map([("a", (256,), "bfloat16"), ("b", (256,), "bfloat16")])
        # Force b to overlap a: both at offset 0.
        mp["map"]["b"]["offset"] = 0
        with self.assertRaises(WeightMapInvalid) as ctx:
            validate_weight_map(mp, native_w8a8=False)
        self.assertIn("overlap", str(ctx.exception))

    def test_span_exceeding_pool_rejected(self):
        mp = _build_map([("a", (256,), "bfloat16")])
        mp["pool_bytes"] = 8  # too small for a 512-byte entry
        with self.assertRaises(WeightMapInvalid) as ctx:
            validate_weight_map(mp, native_w8a8=False)
        self.assertIn("exceeds pool_bytes", str(ctx.exception))

    def test_nbytes_shape_dtype_consistency(self):
        mp = _build_map([("a", (256,), "bfloat16")])
        mp["map"]["a"]["nbytes"] = 999  # wrong
        with self.assertRaises(WeightMapInvalid) as ctx:
            validate_weight_map(mp, native_w8a8=False)
        self.assertIn("nbytes", str(ctx.exception))

    def test_unknown_dtype_rejected(self):
        mp = _build_map([("a", (256,), "bfloat16")])
        mp["map"]["a"]["dtype"] = "float64"
        with self.assertRaises(WeightMapInvalid) as ctx:
            validate_weight_map(mp, native_w8a8=False)
        self.assertIn("dtype", str(ctx.exception))

    def test_bad_version_rejected(self):
        mp = _build_map([("a", (256,), "bfloat16")], version=2)
        with self.assertRaises(WeightMapInvalid) as ctx:
            validate_weight_map(mp, native_w8a8=False)
        self.assertIn("version", str(ctx.exception))

    def test_empty_map_rejected(self):
        mp = {"version": 1, "pool_bytes": 512, "map": {}}
        with self.assertRaises(WeightMapInvalid):
            validate_weight_map(mp, native_w8a8=False)

    def test_expected_cross_check_shape_mismatch(self):
        mp = _build_map([("wq_full", (1024, 4096), "bfloat16")])
        expected = {"wq_full": ((999, 4096), "bfloat16")}
        with self.assertRaises(WeightMapInvalid) as ctx:
            validate_weight_map(mp, expected=expected, native_w8a8=False)
        self.assertIn("shape", str(ctx.exception))

    def test_expected_cross_check_missing_key(self):
        mp = _build_map([("wq_full", (1024, 4096), "bfloat16")])
        expected = {
            "wq_full": ((1024, 4096), "bfloat16"),
            "wk_full": ((256, 4096), "bfloat16"),
        }
        with self.assertRaises(WeightMapInvalid) as ctx:
            validate_weight_map(mp, expected=expected, native_w8a8=False)
        self.assertIn("missing expected keys", str(ctx.exception))

    def test_expected_cross_check_extra_key_when_disallowed(self):
        mp = _build_map([
            ("wq_full", (1024, 4096), "bfloat16"),
            ("extra", (16,), "bfloat16"),
        ])
        expected = {"wq_full": ((1024, 4096), "bfloat16")}
        with self.assertRaises(WeightMapInvalid) as ctx:
            validate_weight_map(
                mp, expected=expected, native_w8a8=False, allow_extra_keys=False
            )
        self.assertIn("unexpected keys", str(ctx.exception))

    def test_expected_cross_check_pass(self):
        mp = _build_map([("wq_full", (1024, 4096), "bfloat16")])
        expected = {"wq_full": ((1024, 4096), "bfloat16")}
        validate_weight_map(mp, expected=expected, native_w8a8=False)  # no raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
