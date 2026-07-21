# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility wrapper for the canonical Main hidden-only vLLM backend.

The historical implementation in this module copied an old socket protocol
and silently called the original model forward when the sidecar was absent.
That is unsafe for the tail-only Step3p5 model: the original forward is a
metadata shell, not a correct decoder, so such a fallback would send token
embeddings directly to the final norm/LM head.

Main and MTP now deploy the same ``tools.step3p5`` package and share one
protocol-v2 sidecar connection.  Keep this module only as a compatibility
import for older sitecustomize files; all behavior delegates to
``vllm_monkey_patch`` so fail-closed gating, paged-KV metadata, speculative
target verification and the CPU/Gloo control plane have one implementation.
"""
from __future__ import annotations

import json
import os

from tools.step3p5 import vllm_monkey_patch as _canonical


# Preserve the historical function name for source-level deployment checks.
_full_forward = _canonical._pypto_full_forward


def install():
    """Install the canonical full hidden-only backend."""
    return _canonical.install("full")


def uninstall():
    return _canonical.uninstall()


def status():
    report = dict(_canonical.status())
    report.update(
        {
            "compatibility_wrapper": True,
            "sock": os.environ.get(
                "PYPTO_WHOLE_DECODE_SOCK",
                "/logs/pypto_whole_decode.sock",
            ),
        }
    )
    return report


def maybe_autoload():
    """Compatibility sitecustomize entry; never adds an independent fallback."""
    flag = os.environ.get("PYPTO_WHOLE_DECODE", "")
    if flag and flag != "0":
        return install()
    return {"ok": True, "skipped": True}


if __name__ == "__main__":
    print(json.dumps(status(), indent=2, default=str))
