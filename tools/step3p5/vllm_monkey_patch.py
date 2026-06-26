#!/usr/bin/env python3
"""Step3p5 vLLM monkey-patch entry points for PyPTO backend bring-up.

This module intentionally separates *patch mechanics* from the eventual PyPTO
NPU full-runner implementation.  It provides:

- install/uninstall helpers that patch vLLM's Step3p5 classes in-place;
- a validated ``tail`` mode replacing the final norm + logits path with a
  PyPTO-compatible tail call (same numerical contract as current precision
  reports);
- a ``shadow`` mode wrapping the full model forward for instrumentation while
  delegating computation to vLLM;
- a fail-closed ``full`` mode that raises a clear error until the real
  ``Step3p5DecodeFwd`` online runner is wired.

The production full-network replacement should plug into
``_pypto_full_forward`` without changing the vLLM patch surface.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_PATCH_ATTR = "_pypto_step3p5_patch_state"


@dataclass
class PatchState:
    mode: str
    original_model_forward: Callable[..., Any]
    original_causal_forward: Callable[..., Any]
    original_compute_logits: Callable[..., Any]


class PyPTOBackendUnavailable(RuntimeError):
    """Raised when the requested PyPTO full-network runner is not wired yet."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_step3p5_module():
    import vllm.model_executor.models.step3p5 as step3p5  # noqa: PLC0415

    return step3p5


def _patch_state(step3p5) -> PatchState | None:
    return getattr(step3p5, _PATCH_ATTR, None)


def _set_patch_state(step3p5, state: PatchState | None) -> None:
    if state is None:
        if hasattr(step3p5, _PATCH_ATTR):
            delattr(step3p5, _PATCH_ATTR)
    else:
        setattr(step3p5, _PATCH_ATTR, state)


def _pypto_tail_compute_logits(self, hidden_states):
    """PyPTO-compatible final RMSNorm + LM-head tail.

    The actual math intentionally mirrors the already precision-closed
    ``tools/step3p5/final_logits_from_vllm.py`` contract: consume final hidden,
    apply Step3p5 final norm, then vocab projection/logits processing.  It uses
    vLLM's live modules for now so quantized LM-head sharding remains identical
    to vLLM-Ascend while the PyPTO runner ABI is being wired.
    """
    normed_hidden_states = self.model.norm(hidden_states)
    logits = self.logits_processor(self.lm_head, normed_hidden_states)
    setattr(self, "_pypto_tail_last_shape", tuple(logits.shape))
    return logits


def _pypto_full_forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
    """Placeholder for the future full PyPTO Step3p5 runner.

    The online full runner must replace this body with a call that prepares
    hidden/KV/block-table inputs, invokes ``Step3p5DecodeFwd``/prefill runner,
    and returns hidden states compatible with vLLM's compute_logits path.
    """
    raise PyPTOBackendUnavailable(
        "PYPTO_STEP3P5_PATCH_MODE=full requested, but the online "
        "Step3p5DecodeFwd/prefill runner is not wired yet. Use mode=tail "
        "or mode=shadow for bring-up instrumentation."
    )


def install(mode: str | None = None) -> dict[str, Any]:
    """Install the Step3p5 monkey patch.

    Modes:
      - ``tail``: replace ``Step3p5ForCausalLM.compute_logits`` only.
      - ``shadow``: wrap ``Step3p5Model.forward`` and delegate to original.
      - ``full``: replace ``Step3p5Model.forward`` with fail-closed full-runner
        placeholder (until real PyPTO online runner lands).
    """
    mode = (mode or os.environ.get("PYPTO_STEP3P5_PATCH_MODE") or "tail").lower()
    if mode not in {"tail", "shadow", "full"}:
        raise ValueError(f"unsupported Step3p5 PyPTO patch mode: {mode}")

    step3p5 = _load_step3p5_module()
    if _patch_state(step3p5) is not None:
        return {"ok": True, "already_installed": True, "mode": _patch_state(step3p5).mode}

    state = PatchState(
        mode=mode,
        original_model_forward=step3p5.Step3p5Model.forward,
        original_causal_forward=step3p5.Step3p5ForCausalLM.forward,
        original_compute_logits=step3p5.Step3p5ForCausalLM.compute_logits,
    )

    if mode == "tail":
        step3p5.Step3p5ForCausalLM.compute_logits = _pypto_tail_compute_logits
    elif mode == "shadow":
        original = state.original_model_forward

        @functools.wraps(original)
        def shadow_forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
            setattr(self, "_pypto_shadow_last_input_shape", tuple(input_ids.shape) if input_ids is not None else None)
            setattr(self, "_pypto_shadow_last_positions_shape", tuple(positions.shape))
            return original(self, input_ids, positions, intermediate_tensors, inputs_embeds)

        step3p5.Step3p5Model.forward = shadow_forward
        step3p5.Step3p5ForCausalLM.compute_logits = _pypto_tail_compute_logits
    else:
        step3p5.Step3p5Model.forward = _pypto_full_forward
        step3p5.Step3p5ForCausalLM.compute_logits = _pypto_tail_compute_logits

    _set_patch_state(step3p5, state)
    return {"ok": True, "installed": True, "mode": mode}


def uninstall() -> dict[str, Any]:
    step3p5 = _load_step3p5_module()
    state = _patch_state(step3p5)
    if state is None:
        return {"ok": True, "installed": False}
    step3p5.Step3p5Model.forward = state.original_model_forward
    step3p5.Step3p5ForCausalLM.forward = state.original_causal_forward
    step3p5.Step3p5ForCausalLM.compute_logits = state.original_compute_logits
    _set_patch_state(step3p5, None)
    return {"ok": True, "uninstalled": True, "mode": state.mode}


def status() -> dict[str, Any]:
    step3p5 = _load_step3p5_module()
    state = _patch_state(step3p5)
    return {
        "installed": state is not None,
        "mode": None if state is None else state.mode,
        "module": getattr(step3p5, "__file__", None),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["install", "uninstall", "status"])
    parser.add_argument("--mode", choices=["tail", "shadow", "full"], default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(_repo_root()))
    if args.action == "install":
        report = install(args.mode)
    elif args.action == "uninstall":
        report = uninstall()
    else:
        report = status()
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0 if report.get("ok", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
