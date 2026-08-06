# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility wrapper for the canonical Main hidden-only vLLM prefill backend.

This module is the prefill dual of ``pypto_whole_decode_backend``.  The
historical decode wrapper copied an old socket protocol and silently called the
original model forward when the sidecar was absent.  That fallback is unsafe for
the tail-only Step3p5 model: the original forward is a metadata shell, not a
correct decoder, so such a fallback would send token embeddings directly to the
final norm/LM head.

The prefill path is held to the same fail-closed contract: there is never an
independent vanilla or per-layer fallback.  All behavior delegates to
``vllm_monkey_patch`` so the prefill forward (``_pypto_prefill_forward``), the
``classify_prefill_gate`` three-state decision, and the
``_pypto_causal_forward_dispatch`` install point share one implementation with
the decode side.  The canonical prefill dispatch is wired when the environment
variable ``PYPTO_STEP3P5_PREFILL_PATCH=1`` is set; ``install`` activates it and
then delegates the actual patch installation to ``vllm_monkey_patch`` exactly
like the decode wrapper delegates to ``install("full")``.

Main and MTP now deploy the same ``tools.step3p5`` package and share one
protocol-v2 sidecar connection.  Keep this module only as a compatibility
import for older sitecustomize files; the resident single-chip PyPTO program is
the only real-request path, and a prefill PyPTO cannot serve must fail closed.
"""
from __future__ import annotations

import json
import os

from tools.step3p5 import vllm_monkey_patch as _canonical

__all__ = ["install", "uninstall", "status", "maybe_autoload"]

# The canonical prefill forward is added to ``vllm_monkey_patch`` as
# ``_pypto_prefill_forward`` and wired when ``PYPTO_STEP3P5_PREFILL_PATCH=1``.
# Resolve it defensively: importing this wrapper must never silently substitute
# a vanilla forward.  If the symbol is absent (the prefill forward has not been
# wired yet) we fail closed at ``install`` time with a clear error rather than
# letting a raw ``AttributeError`` propagate at import -- a crash here would
# break unrelated sitecustomize imports.  No independent fallback is ever added.
_prefill_forward = getattr(_canonical, "_pypto_prefill_forward", None)


def install():
    """Install the canonical prefill hidden-only backend (fail-closed).

    Activates the canonical prefill dispatch (``PYPTO_STEP3P5_PREFILL_PATCH=1``)
    and delegates the patch installation to ``vllm_monkey_patch``.  Never adds
    an independent fallback; raises ``PyPTOBackendUnavailable`` if the canonical
    prefill forward is absent.
    """
    if _prefill_forward is None:
        raise _canonical.PyPTOBackendUnavailable(
            "canonical _pypto_prefill_forward is not wired: vllm_monkey_patch "
            "must be built with the prefill forward "
            "(set PYPTO_STEP3P5_PREFILL_PATCH=1)"
        )
    # Activate the prefill dispatch in the canonical module, then delegate the
    # actual patch installation exactly like the decode wrapper delegates to
    # ``install("full")``.  This is activation of the canonical path, not a
    # fallback: there is still no vanilla/per-layer execution path.
    os.environ["PYPTO_STEP3P5_PREFILL_PATCH"] = "1"
    return _canonical.install("full")


def uninstall():
    return _canonical.uninstall()


def status():
    report = dict(_canonical.status())
    report.update(
        {
            "compatibility_wrapper": True,
            "prefill_forward_wired": _prefill_forward is not None,
            "sock": os.environ.get(
                "PYPTO_WHOLE_PREFILL_SOCK",
                "/logs/pypto_whole_prefill.sock",
            ),
        }
    )
    return report


def maybe_autoload():
    """Compatibility sitecustomize entry; never adds an independent fallback."""
    flag = os.environ.get("PYPTO_WHOLE_PREFILL", "")
    if flag and flag != "0":
        return install()
    return {"ok": True, "skipped": True}


if __name__ == "__main__":
    print(json.dumps(status(), indent=2, default=str))
