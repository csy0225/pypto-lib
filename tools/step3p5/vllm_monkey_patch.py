#!/usr/bin/env python3
"""Step3p5 vLLM patch for the canonical PyPTO whole-net hidden-only backend.

The only installable mode is ``full``: Main decoder math runs in the resident
single-chip PyPTO program, while vLLM retains final norm, LM head, sampling,
MTP shared heads and speculative acceptance/rejection.
"""
from __future__ import annotations

import argparse
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
    prefill_dispatch: bool = False


class PyPTOBackendUnavailable(RuntimeError):
    """Raised when the requested PyPTO full-network runner is not wired yet."""


# --- Fail-closed decode-gate decision (pure; unit-tested card-free) ----------
# Broadcast as an INT32 code from rank 0 so all TP ranks take the same branch.
GATE_FAIL_CLOSED = 0   # real request PyPTO cannot serve
GATE_PROCEED = 1       # run the PyPTO whole-net sidecar
GATE_PROFILE_NOOP = 2  # profile/dummy/warmup: original no-op forward is safe


def classify_decode_gate(
    *,
    is_real_request: bool,
    sidecar_available: bool,
    eligible: bool,
) -> int:
    """Decide how a Step3p5 decode forward proceeds (fail-closed).

    Only identifiable profile/dummy/warmup calls may take the metadata-only
    no-op path. Every real request that is not served by the canonical
    single-chip sidecar fails closed; there is no vanilla or per-layer fallback.

    Args:
      is_real_request: a live per-layer ``attn_metadata`` exists (real decode).
        ``False`` marks profile/dummy/warmup, where the no-op fallback is safe.
      sidecar_available: the resident whole-net sidecar socket is present.
      eligible: pure one-token-per-request decode ABI holds (1..16 rows, no
        prefill, no spec, PP==1, hidden ``[T,4096]`` BF16, no token padding).
    Returns one of the ``GATE_*`` codes.
    """
    if not is_real_request:
        return GATE_PROFILE_NOOP
    if sidecar_available and eligible:
        return GATE_PROCEED
    return GATE_FAIL_CLOSED


# --- Fail-closed prefill-gate decision (pure; unit-tested card-free) --------
# Three-way dispatch on a Step3p5 forward's num_prefills, broadcast as an
# INT32 code from rank 0 so all TP ranks take the same branch. Prefill and
# decode share one CPU/Gloo control plane; the codes are deliberately
# disjoint from the decode GATE_* codes so a combined dispatch can route a
# batch to the right resident sidecar without ambiguity.
GATE_PREFILL_PROCEED = 10  # num_prefills>0 and the prefill ABI holds
GATE_DECODE_PROCEED = 11   # num_prefills==0: defer to the decode path
GATE_REJECT = 12           # any eligibility violation / sidecar down


def classify_prefill_gate(
    forward_context,
    vllm_config,
    *,
    num_prefills: int,
    T: int = 128,
    chunked_prefill: bool = False,
    context_parallel: int = 1,
    pipeline_parallel: int = 1,
    num_kv_groups: int = 1,
    sidecar_available: bool = True,
) -> int:
    """Decide how a Step3p5 forward with prefills proceeds (fail-closed).

    A pure decision function: ``forward_context`` / ``vllm_config`` are passed
    through for signature symmetry with the live caller but are not dereferenced
    here (the caller resolves the scalar eligibility inputs first), so the
    card-free unit tests can exercise every branch with ``None``.

    Args:
      forward_context: the vLLM ``ForwardContext`` (unused by the decision).
      vllm_config: the vLLM config (unused by the decision).
      num_prefills: scheduler prefill request count for this step.
      T: active prefill token count (``num_actual_tokens``).
      chunked_prefill: vLLM chunked-prefill scheduling is active.
      context_parallel: ``prefill_context_parallel_size`` (1 => no CP/PCP).
      pipeline_parallel: ``pipeline_parallel_size`` (1 => no PP).
      num_kv_groups: resolved KV cache group count (1 => single group).
      sidecar_available: the resident whole-prefill sidecar socket is present.
    Returns one of ``GATE_PREFILL_PROCEED`` / ``GATE_DECODE_PROCEED`` /
    ``GATE_REJECT``.
    """
    # A pure-decode step is not a prefill request; defer to the decode path so
    # the existing decode sidecar / monkey-patch handles it unchanged.
    if num_prefills == 0:
        return GATE_DECODE_PROCEED
    if (
        not sidecar_available
        or chunked_prefill
        or context_parallel != 1
        or pipeline_parallel != 1
        or num_kv_groups != 1
        or T <= 0
        or T > 128
    ):
        return GATE_REJECT
    return GATE_PREFILL_PROCEED


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
    """vLLM-owned final RMSNorm and LM-head tail."""
    normed_hidden_states = self.model.norm(hidden_states)
    logits = self.logits_processor(self.lm_head, normed_hidden_states)
    return logits


def _wd_sock_path() -> str:
    return os.environ.get("PYPTO_WHOLE_DECODE_SOCK", "/logs/pypto_whole_decode.sock")


# Lazily-created, process-global whole-decode sidecar client.  Rank 0 is the
# only socket peer.  TP coordination around the call is CPU/Gloo-only: using a
# vLLM device-group broadcast while PyPTO owns the same eight NPUs can form a
# cross-runtime collective cycle.
_WD_CLIENT = None


def _wd_client():
    global _WD_CLIENT
    if _WD_CLIENT is None:
        from tools.step3p5.whole_decode_sidecar import WholeDecodeClient  # noqa: PLC0415
        _WD_CLIENT = WholeDecodeClient(_wd_sock_path()).connect()
    return _WD_CLIENT


def _sidecar_result_payload(
    run_sidecar,
    *,
    output_key: str = "next_hidden",
) -> dict[str, Any]:
    """Execute one rank-0 sidecar transaction and turn success/failure into data.

    All ranks consume the returned object through ``tp.broadcast_object``.
    ``run_sidecar`` may issue multiple ordered whole-net rounds for one vLLM
    speculative target forward. A failure in any round is deliberately
    represented as a terminal error, not a fallback request: PyPTO may already
    have mutated paged KV in this or an earlier round.
    """
    try:
        out_meta, output = run_sidecar()
        return {
            "ok": True,
            "out_meta": dict(out_meta),
            output_key: output,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": repr(exc),
        }


def _run_decode_plan(client, hidden_cpu, decode_plan):
    """Execute ordered N=1 whole-net rounds and restore flattened token order.

    vLLM flattens speculative target verification request-major, while PyPTO's
    production program accepts one position per request.  ``decode_plan``
    partitions every flattened token index into ordered speculative-position
    rounds.  Earlier rounds complete (including KV writes) before later rounds
    enter the same resident sidecar/runtime.

    The returned hidden uses vLLM's original flattened token order so the
    unchanged final RMSNorm, LM head, target sampler and rejection sampler
    consume exactly the rows they expect.
    """
    import torch  # noqa: PLC0415

    if (
        hidden_cpu.dtype != torch.bfloat16
        or hidden_cpu.ndim != 2
        or int(hidden_cpu.shape[0]) != int(decode_plan.valid_tokens)
    ):
        raise PyPTOBackendUnavailable(
            "decode-plan hidden ABI mismatch: "
            f"dtype={hidden_cpu.dtype}, shape={tuple(hidden_cpu.shape)}, "
            f"valid_tokens={decode_plan.valid_tokens}"
        )

    next_hidden_cpu = torch.empty_like(hidden_cpu)
    seen = torch.zeros(decode_plan.valid_tokens, dtype=torch.bool)
    round_metas: list[dict[str, Any]] = []
    for round_idx, decode_meta in enumerate(decode_plan.steps):
        token_indices = tuple(int(index) for index in decode_meta.token_indices)
        if len(token_indices) != decode_meta.valid_tokens:
            raise PyPTOBackendUnavailable(
                f"decode round {round_idx}: token-index count "
                f"{len(token_indices)} != valid_tokens "
                f"{decode_meta.valid_tokens}"
            )
        if len(set(token_indices)) != len(token_indices):
            raise PyPTOBackendUnavailable(
                f"decode round {round_idx}: duplicate token indices "
                f"{token_indices}"
            )
        if any(
            index < 0 or index >= decode_plan.valid_tokens
            for index in token_indices
        ):
            raise PyPTOBackendUnavailable(
                f"decode round {round_idx}: token index out of range "
                f"{token_indices}"
            )

        index = torch.tensor(token_indices, dtype=torch.long)
        if torch.any(seen.index_select(0, index)):
            raise PyPTOBackendUnavailable(
                f"decode round {round_idx}: token indices were already "
                f"produced {token_indices}"
            )
        step_hidden = hidden_cpu.index_select(0, index).contiguous()
        tensors = {"hidden": step_hidden}
        tensors.update(decode_meta.protocol_tensors())
        out_meta, out = client.decode(
            tensors,
            decode_meta.protocol_meta(),
        )
        step_next_hidden = out.get("next_hidden")
        if (
            not isinstance(step_next_hidden, torch.Tensor)
            or step_next_hidden.dtype != torch.bfloat16
            or tuple(step_next_hidden.shape) != tuple(step_hidden.shape)
        ):
            raise PyPTOBackendUnavailable(
                f"decode round {round_idx}: sidecar next_hidden ABI "
                f"mismatch; got "
                f"{getattr(step_next_hidden, 'dtype', None)}/"
                f"{getattr(step_next_hidden, 'shape', None)}, expected "
                f"{step_hidden.dtype}/{tuple(step_hidden.shape)}"
            )
        if not torch.isfinite(step_next_hidden.float()).all():
            raise PyPTOBackendUnavailable(
                f"decode round {round_idx}: sidecar returned NaN/Inf "
                "next_hidden"
            )
        next_hidden_cpu.index_copy_(
            0,
            index,
            step_next_hidden.contiguous(),
        )
        seen.index_fill_(0, index, True)
        round_metas.append(
            {
                "round_idx": round_idx,
                "token_indices": list(token_indices),
                "out_meta": dict(out_meta),
            }
        )

    if not torch.all(seen):
        missing = torch.nonzero(~seen, as_tuple=False).flatten().tolist()
        raise PyPTOBackendUnavailable(
            f"decode plan did not produce hidden rows {missing}"
        )
    return (
        {
            "op": "decode_plan",
            "round_count": len(round_metas),
            "query_lengths": list(decode_plan.query_lengths),
            "rounds": round_metas,
        },
        next_hidden_cpu,
    )


def _pypto_full_forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
    """Full PyPTO Step3p5 runner via the resident whole-net sidecar.

    Data flow (live single-handoff, N=1 whole-net; see phases/20 §G):
      1. vLLM embeds locally (embed_tokens) -> hidden [num_tokens, HIDDEN],
         replicated across the 8 TP ranks.
      2. every rank copies its embedding hidden to CPU, completing rank-local
         NPU work before PyPTO takes the shared device partition.
      3. rank-0 extracts the already-built vLLM-Ascend attention metadata.
         A normal decode is one whole-net round. Speculative target
         verification is decomposed into ordered one-position-per-request
         rounds so each earlier round publishes KV before the next consumes it.
         The decision is broadcast on the TP CPU/Gloo group before any rank
         branches.
      4. all ranks rendezvous on the CPU group. Rank-0 drives the sidecar while
         ranks 1..7 wait only on the CPU control plane.
      5. rank-0 broadcasts a CPU result object containing status and replicated
         hidden. Each rank independently copies the hidden back to its NPU.
      6. return next_hidden; vLLM's (patched) compute_logits applies the
         validated final norm + lm_head tail.

    Final live token-exact validation is gated on the held whole-net hang fix +
    a running co-resident vLLM 8001 (SIMPLER_COMM_NO_HCCL sidecar).
    """
    import torch  # noqa: PLC0415
    step3p5 = _load_step3p5_module()
    state = _patch_state(step3p5)
    original_forward = state.original_model_forward if state is not None else None

    # embed locally (mirror vLLM Step3p5Model.forward preamble)
    if inputs_embeds is None:
        hidden = self.embed_tokens(input_ids)
    else:
        hidden = inputs_embeds

    from vllm.distributed import (  # noqa: PLC0415
        get_pp_group,
        get_tensor_model_parallel_rank,
        get_tp_group,
    )
    rank = get_tensor_model_parallel_rank()
    tp = get_tp_group()

    # This copy is required on *every* rank, even though rank 0 alone sends the
    # payload to the sidecar.  Besides producing the socket input, it completes
    # rank-local embedding work before non-rank0 workers enter a CPU-only wait.
    # No vLLM NPU collective is allowed between here and sidecar completion.
    hidden_cpu = hidden.detach().to("cpu", dtype=torch.bfloat16).contiguous()

    decode_plan = None
    local_error = None
    decision = GATE_FAIL_CLOSED
    if rank == 0:
        # metadata-only no-op is allowed ONLY for profile/dummy/warmup, which
        # have no live per-layer attention metadata.  A real decode request that
        # PyPTO cannot serve must fail closed, never silently pass embeddings to
        # the final norm/LM head (design §5.1).
        sidecar_available = os.path.exists(_wd_sock_path())
        is_real_request = False
        eligible = False
        try:
            from vllm.forward_context import get_forward_context  # noqa: PLC0415
            ctx = get_forward_context()
            attn_md = getattr(ctx, "attn_metadata", None)
            if attn_md is not None:
                is_real_request = (
                    len(attn_md) > 0 if hasattr(attn_md, "__len__") else True
                )
        except Exception:  # noqa: BLE001
            # No live forward context => profile/dummy/warmup.
            is_real_request = False
        if is_real_request:
            try:
                if not sidecar_available:
                    raise PyPTOBackendUnavailable("sidecar socket is absent")
                pp = get_pp_group()
                if int(getattr(pp, "world_size", 1)) != 1:
                    raise PyPTOBackendUnavailable(
                        "pipeline parallel live path is unsupported"
                    )
                if hidden.ndim != 2 or hidden.shape[1] != 4096:
                    raise PyPTOBackendUnavailable(
                        f"hidden must be [T,4096], got {tuple(hidden.shape)}"
                    )
                if hidden.dtype != torch.bfloat16:
                    raise PyPTOBackendUnavailable(
                        f"hidden must be BF16, got {hidden.dtype}"
                    )
                from vllm.forward_context import (  # noqa: PLC0415
                    get_forward_context,
                )
                from tools.step3p5.vllm_decode_metadata import (  # noqa: PLC0415
                    extract_pypto_decode_plan,
                )

                decode_plan = extract_pypto_decode_plan(
                    get_forward_context(),
                    vllm_config=self.vllm_config,
                    positions=positions,
                )
                if int(hidden.shape[0]) != decode_plan.valid_tokens:
                    raise PyPTOBackendUnavailable(
                        "graph/token padding is not supported by the first live "
                        f"ABI: hidden rows={hidden.shape[0]}, "
                        f"valid={decode_plan.valid_tokens}"
                    )
                eligible = True
            except Exception as exc:  # noqa: BLE001
                local_error = exc
                eligible = False
        decision = classify_decode_gate(
            is_real_request=is_real_request,
            sidecar_available=sidecar_available,
            eligible=eligible,
        )

    # GroupCoordinator.broadcast_object uses its CPU group (or its host
    # message-queue broadcaster).  Do not replace this with ``tp.broadcast``:
    # the latter uses the NPU device group and can deadlock against PyPTO's
    # all-rank collectives on the same physical cards.
    decision = int(tp.broadcast_object(decision if rank == 0 else None, src=0))

    if decision == GATE_PROFILE_NOOP:
        if original_forward is None:
            raise PyPTOBackendUnavailable(
                f"PyPTO path unavailable and no fallback exists: {local_error!r}"
            )
        setattr(
            self,
            "_pypto_profile_noops",
            int(getattr(self, "_pypto_profile_noops", 0)) + 1,
        )
        return original_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds)

    if decision == GATE_FAIL_CLOSED:
        if rank == 0:
            reason = (
                repr(local_error)
                if local_error is not None
                else (
                    "real decode request is not PyPTO-eligible and no correct "
                    "fallback exists (tail-only instance)"
                )
            )
            setattr(self, "_pypto_full_last_error", reason)
        setattr(
            self,
            "_pypto_full_fail_closed",
            int(getattr(self, "_pypto_full_fail_closed", 0)) + 1,
        )
        raise PyPTOBackendUnavailable(
            "PyPTO real decode request failed closed"
            + (f": {local_error!r}" if rank == 0 and local_error is not None else "")
        )

    # decision == GATE_PROCEED.  This CPU-group rendezvous guarantees every
    # vLLM rank has completed the embedding->CPU copy and no rank is still
    # submitting vLLM NPU work when rank 0 enters the resident PyPTO runtime.
    tp.barrier()

    payload = None
    if rank == 0:
        assert decode_plan is not None

        def _run_sidecar():
            cli = _wd_client()
            return _run_decode_plan(cli, hidden_cpu, decode_plan)

        payload = _sidecar_result_payload(_run_sidecar)

    # Status and output use one CPU control-plane broadcast.  Non-rank0 workers
    # block here without occupying their NPU/HCCL stream while PyPTO runs.
    payload = tp.broadcast_object(payload, src=0)
    if not bool(payload.get("ok")):
        if rank == 0:
            setattr(self, "_pypto_full_last_error", payload.get("error"))
        raise PyPTOBackendUnavailable(
            "PyPTO sidecar decode failed on rank0: "
            f"{payload.get('error_type')}: {payload.get('error')}"
            if rank == 0
            else "PyPTO sidecar decode failed on rank0"
        )

    next_hidden_cpu = payload["next_hidden"]
    if (
        next_hidden_cpu.dtype != torch.bfloat16
        or tuple(next_hidden_cpu.shape) != tuple(hidden_cpu.shape)
    ):
        raise PyPTOBackendUnavailable(
            "CPU control-plane next_hidden ABI mismatch after broadcast"
        )
    next_hidden = next_hidden_cpu.to(
        device=hidden.device,
        dtype=hidden.dtype,
    )
    if rank == 0:
        setattr(self, "_pypto_full_last_meta", dict(payload["out_meta"]))
    setattr(self, "_pypto_full_calls", int(getattr(self, "_pypto_full_calls", 0)) + 1)
    return next_hidden


# ---------------------------------------------------------------------------
# Prefill hidden-only whole-net path (dual of the decode path above).
#
# The prefill sidecar owns the resident single-chip ``whole_prefill_step3p5``
# program.  A forward with ``num_prefills > 0`` is routed here by
# ``_pypto_causal_forward_dispatch`` (installed on ``Step3p5ForCausalLM.forward``
# when ``PYPTO_STEP3P5_PREFILL_PATCH=1``).  A pure-decode forward
# (``num_prefills == 0``) is left on the decode path, which the decode
# monkey-patch already owns on ``Step3p5Model.forward`` — so the two paths are
# disjoint and never both active for one step.
# ---------------------------------------------------------------------------


def _wd_prefill_sock_path() -> str:
    return os.environ.get(
        "PYPTO_WHOLE_PREFILL_SOCK", "/logs/pypto_whole_prefill.sock"
    )


# Lazily-created, process-global whole-prefill sidecar client.  Rank 0 is the
# only socket peer; the CPU/Gloo-only TP coordination mirrors the decode path
# (a vLLM device-group broadcast would form a cross-runtime collective cycle
# against PyPTO on the same eight NPUs).
_WD_PREFILL_CLIENT = None


def _wd_prefill_client():
    global _WD_PREFILL_CLIENT
    if _WD_PREFILL_CLIENT is None:
        from tools.step3p5.whole_prefill_sidecar import (  # noqa: PLC0415
            WholePrefillClient,
        )
        _WD_PREFILL_CLIENT = WholePrefillClient(_wd_prefill_sock_path()).connect()
    return _WD_PREFILL_CLIENT


def _run_prefill_plan(client, hidden_cpu, prefill_plan):
    """Execute one prefill whole-net round and restore the active token rows.

    A normal prefill is a single whole-net round (no speculative target
    decomposition, unlike decode).  ``prefill_plan`` is the validated
    ``PrefillPlan`` from ``vllm_prefill_metadata.extract_pypto_prefill_meta``;
    its ``protocol_tensors`` / ``protocol_meta`` are exactly the sidecar ABI.

    The returned hidden uses vLLM's original flattened token order so the
    unchanged final RMSNorm, LM head and sampler consume exactly the rows they
    expect.  A sidecar failure is terminal, not a fallback request: PyPTO may
    already have written paged KV for this prefill.
    """
    import torch  # noqa: PLC0415

    if (
        hidden_cpu.dtype != torch.bfloat16
        or hidden_cpu.ndim != 2
        or int(hidden_cpu.shape[0]) != int(prefill_plan.valid_tokens)
        or int(hidden_cpu.shape[1]) != 4096
    ):
        raise PyPTOBackendUnavailable(
            "prefill-plan hidden ABI mismatch: "
            f"dtype={hidden_cpu.dtype}, shape={tuple(hidden_cpu.shape)}, "
            f"valid_tokens={prefill_plan.valid_tokens}"
        )

    out_meta, output = client.prefill(
        dict({"hidden": hidden_cpu}, **prefill_plan.protocol_tensors()),
        prefill_plan.protocol_meta(),
    )
    next_hidden = output.get("next_hidden")
    if (
        not isinstance(next_hidden, torch.Tensor)
        or next_hidden.dtype != torch.bfloat16
        or tuple(next_hidden.shape) != tuple(hidden_cpu.shape)
    ):
        raise PyPTOBackendUnavailable(
            "prefill sidecar next_hidden ABI mismatch; got "
            f"{getattr(next_hidden, 'dtype', None)}/"
            f"{getattr(next_hidden, 'shape', None)}, expected "
            f"{hidden_cpu.dtype}/{tuple(hidden_cpu.shape)}"
        )
    if not torch.isfinite(next_hidden.float()).all():
        raise PyPTOBackendUnavailable("prefill sidecar returned NaN/Inf next_hidden")
    return (
        {
            "op": "prefill_plan",
            "round_count": 1,
            "query_lengths": list(prefill_plan.query_lengths),
            "out_meta": dict(out_meta),
        },
        next_hidden,
    )


def _pypto_prefill_forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
    """Prefill hidden-only whole-net runner via the resident sidecar.

    Data flow (live single-handoff, N=1 whole-net; prefill dual of the decode
    ``_pypto_full_forward`` path):
      1. vLLM embeds locally (embed_tokens) -> hidden [num_tokens, HIDDEN],
         replicated across the 8 TP ranks.
      2. every rank copies its embedding hidden to CPU before PyPTO takes the
         shared device partition.
      3. rank-0 extracts the vLLM-Ascend prefill attention metadata
         (``extract_pypto_prefill_meta``) and resolves the scalar eligibility
         inputs (T, chunked-prefill, CP/PCP, PP, KV-group count, sidecar up).
         A single prefill request (``num_prefills == 1``, ``num_decodes == 0``)
         with ``1 <= T <= PREFILL_T`` is the only served shape.  The decision
         is broadcast on the TP CPU/Gloo group before any rank branches.
      4. all ranks rendezvous on the CPU group.  Rank-0 drives the sidecar
         while ranks 1..7 wait only on the CPU control plane.
      5. rank-0 broadcasts a CPU result object containing status and replicated
         hidden.  Each rank independently copies the hidden back to its NPU.
      6. return next_hidden; vLLM's (patched) compute_logits applies the
         validated final norm + lm_head tail.
    """
    import torch  # noqa: PLC0415
    step3p5 = _load_step3p5_module()
    state = _patch_state(step3p5)
    original_forward = state.original_model_forward if state is not None else None

    # embed locally (mirror vLLM Step3p5Model.forward preamble)
    if inputs_embeds is None:
        hidden = self.embed_tokens(input_ids)
    else:
        hidden = inputs_embeds

    from vllm.distributed import (  # noqa: PLC0415
        get_pp_group,
        get_tensor_model_parallel_rank,
        get_tp_group,
    )
    rank = get_tensor_model_parallel_rank()
    tp = get_tp_group()

    # Required on every rank: completes rank-local embedding before non-rank0
    # workers enter a CPU-only wait.  No vLLM NPU collective is allowed between
    # here and sidecar completion.
    hidden_cpu = hidden.detach().to("cpu", dtype=torch.bfloat16).contiguous()

    prefill_plan = None
    local_error = None
    decision = GATE_REJECT
    if rank == 0:
        sidecar_available = os.path.exists(_wd_prefill_sock_path())
        num_prefills = 0
        T = 0
        chunked_prefill = False
        context_parallel = 1
        pipeline_parallel = 1
        num_kv_groups = 1
        try:
            if not sidecar_available:
                raise PyPTOBackendUnavailable("prefill sidecar socket is absent")
            pp = get_pp_group()
            if int(getattr(pp, "world_size", 1)) != 1:
                raise PyPTOBackendUnavailable(
                    "pipeline parallel live path is unsupported"
                )
            if hidden.ndim != 2 or hidden.shape[1] != 4096:
                raise PyPTOBackendUnavailable(
                    f"hidden must be [T,4096], got {tuple(hidden.shape)}"
                )
            if hidden.dtype != torch.bfloat16:
                raise PyPTOBackendUnavailable(
                    f"hidden must be BF16, got {hidden.dtype}"
                )
            from vllm.forward_context import get_forward_context  # noqa: PLC0415
            from tools.step3p5.vllm_prefill_metadata import (  # noqa: PLC0415
                PREFILL_T,
                extract_pypto_prefill_meta,
            )

            forward_context = get_forward_context()
            prefill_plan = extract_pypto_prefill_meta(
                forward_context,
                vllm_config=self.vllm_config,
                positions=positions,
            )
            if int(hidden.shape[0]) != prefill_plan.valid_tokens:
                raise PyPTOBackendUnavailable(
                    "graph/token padding is not supported by the first live "
                    f"prefill ABI: hidden rows={hidden.shape[0]}, "
                    f"valid={prefill_plan.valid_tokens}"
                )
            # Resolve the scalar eligibility inputs the pure gate consumes.
            num_prefills = len(prefill_plan.query_lengths)
            T = prefill_plan.valid_tokens
            parallel = getattr(self.vllm_config, "parallel_config", None)
            pipeline_parallel = int(getattr(parallel, "pipeline_parallel_size", 1))
            context_parallel = int(
                getattr(parallel, "prefill_context_parallel_size", 1)
            )
            scheduler = getattr(self.vllm_config, "scheduler_config", None)
            for flag in ("chunked_prefill_enabled", "enable_chunked_prefill"):
                if bool(getattr(scheduler, flag, False)):
                    chunked_prefill = True
                    break
            groups = getattr(
                getattr(self.vllm_config, "kv_cache_config", None),
                "kv_cache_groups",
                None,
            )
            num_kv_groups = len(groups) if groups else 1
            # extract_pypto_prefill_meta already rejects multi-group, but keep
            # the gate self-contained: a future single-pass metadata path must
            # still fail closed here.
            _ = PREFILL_T  # guard against accidental drift of the constant
        except Exception as exc:  # noqa: BLE001
            local_error = exc
            T = 0
        decision = classify_prefill_gate(
            forward_context,
            self.vllm_config,
            num_prefills=num_prefills,
            T=T,
            chunked_prefill=chunked_prefill,
            context_parallel=context_parallel,
            pipeline_parallel=pipeline_parallel,
            num_kv_groups=num_kv_groups,
            sidecar_available=sidecar_available,
        )

    # CPU/Gloo broadcast (never the NPU device group — see decode path note).
    decision = int(tp.broadcast_object(decision if rank == 0 else None, src=0))

    # Pure decode => defer to the decode path (the decode monkey-patch owns
    # Step3p5Model.forward).  This branch is reached only when the causal
    # dispatcher could not pre-classify, e.g. a metadata-less profile call.
    if decision == GATE_DECODE_PROCEED:
        if original_forward is None:
            raise PyPTOBackendUnavailable(
                f"decode path unavailable and no fallback exists: {local_error!r}"
            )
        return original_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds)

    if decision == GATE_REJECT:
        if rank == 0:
            reason = (
                repr(local_error)
                if local_error is not None
                else (
                    "real prefill request is not PyPTO-eligible and no correct "
                    "fallback exists (tail-only instance)"
                )
            )
            setattr(self, "_pypto_prefill_last_error", reason)
        setattr(
            self,
            "_pypto_prefill_fail_closed",
            int(getattr(self, "_pypto_prefill_fail_closed", 0)) + 1,
        )
        raise PyPTOBackendUnavailable(
            "PyPTO real prefill request failed closed"
            + (f": {local_error!r}" if rank == 0 and local_error is not None else "")
        )

    # decision == GATE_PREFILL_PROCEED.  CPU-group rendezvous: every rank has
    # completed the embedding->CPU copy and no rank is still submitting vLLM
    # NPU work when rank 0 enters the resident PyPTO runtime.
    tp.barrier()

    payload = None
    if rank == 0:
        assert prefill_plan is not None

        def _run_sidecar():
            cli = _wd_prefill_client()
            return _run_prefill_plan(cli, hidden_cpu, prefill_plan)

        payload = _sidecar_result_payload(
            _run_sidecar, output_key="next_hidden"
        )

    payload = tp.broadcast_object(payload, src=0)
    if not bool(payload.get("ok")):
        if rank == 0:
            setattr(self, "_pypto_prefill_last_error", payload.get("error"))
        raise PyPTOBackendUnavailable(
            "PyPTO sidecar prefill failed on rank0: "
            f"{payload.get('error_type')}: {payload.get('error')}"
            if rank == 0
            else "PyPTO sidecar prefill failed on rank0"
        )

    next_hidden_cpu = payload["next_hidden"]
    if (
        next_hidden_cpu.dtype != torch.bfloat16
        or tuple(next_hidden_cpu.shape) != tuple(hidden_cpu.shape)
    ):
        raise PyPTOBackendUnavailable(
            "CPU control-plane next_hidden ABI mismatch after broadcast"
        )
    next_hidden = next_hidden_cpu.to(
        device=hidden.device,
        dtype=hidden.dtype,
    )
    if rank == 0:
        setattr(self, "_pypto_prefill_last_meta", dict(payload["out_meta"]))
    setattr(
        self,
        "_pypto_prefill_calls",
        int(getattr(self, "_pypto_prefill_calls", 0)) + 1,
    )
    return next_hidden


def _pypto_causal_forward_dispatch(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
    """Route a Step3p5ForCausalLM.forward by prefill presence.

    A step with ``num_prefills > 0`` runs the resident whole-prefill sidecar
    via ``_pypto_prefill_forward`` (which itself embeds locally and returns
    pre-final-norm hidden).  A pure-decode step defers to the original
    ``Step3p5ForCausalLM.forward`` so the decode monkey-patch on
    ``Step3p5Model.forward`` handles it unchanged.  Profile/dummy/warmup calls
    with no live attention metadata are treated as decode (deferred), matching
    the decode gate's metadata-only no-op allowance.
    """
    step3p5 = _load_step3p5_module()
    state = _patch_state(step3p5)
    original_causal_forward = state.original_causal_forward if state is not None else None

    num_prefills = 0
    try:
        from vllm.forward_context import get_forward_context  # noqa: PLC0415
        attn_md = getattr(get_forward_context(), "attn_metadata", None)
        if attn_md:
            representative = next(iter(attn_md.values()))
            num_prefills = int(getattr(representative, "num_prefills", 0))
    except Exception:  # noqa: BLE001
        # No live forward context => profile/dummy/warmup => defer (decode path).
        num_prefills = 0

    if num_prefills > 0:
        return _pypto_prefill_forward(
            self, input_ids, positions, intermediate_tensors, inputs_embeds
        )
    if original_causal_forward is None:
        raise PyPTOBackendUnavailable(
            "decode path unavailable: causal forward has no original to defer to"
        )
    return original_causal_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds)


def install(mode: str | None = None) -> dict[str, Any]:
    """Install the only production mode: full hidden-only whole-net.

    The decode path is always installed on ``Step3p5Model.forward``.  When
    ``PYPTO_STEP3P5_PREFILL_PATCH=1`` is set, the prefill causal dispatcher is
    additionally installed on ``Step3p5ForCausalLM.forward`` so a forward with
    ``num_prefills > 0`` routes to the resident whole-prefill sidecar; a
    pure-decode forward still falls through to the decode path.  The two paths
    are disjoint: the dispatcher only intercepts prefill steps.
    """
    mode = (mode or os.environ.get("PYPTO_STEP3P5_PATCH_MODE") or "full").lower()
    if mode != "full":
        raise ValueError(
            "only PYPTO_STEP3P5_PATCH_MODE=full is supported; "
            f"got {mode!r}"
        )
    prefill_dispatch = os.environ.get("PYPTO_STEP3P5_PREFILL_PATCH", "") == "1"

    step3p5 = _load_step3p5_module()
    existing = _patch_state(step3p5)
    if existing is not None:
        # If the caller toggled the prefill flag on an already-installed patch,
        # re-apply just the causal forward so the dispatch matches the env.
        if existing.prefill_dispatch != prefill_dispatch:
            if prefill_dispatch:
                step3p5.Step3p5ForCausalLM.forward = _pypto_causal_forward_dispatch
            else:
                step3p5.Step3p5ForCausalLM.forward = existing.original_causal_forward
            existing.prefill_dispatch = prefill_dispatch
        return {
            "ok": True,
            "already_installed": True,
            "mode": existing.mode,
            "prefill_dispatch": existing.prefill_dispatch,
        }

    state = PatchState(
        mode=mode,
        original_model_forward=step3p5.Step3p5Model.forward,
        original_causal_forward=step3p5.Step3p5ForCausalLM.forward,
        original_compute_logits=step3p5.Step3p5ForCausalLM.compute_logits,
        prefill_dispatch=prefill_dispatch,
    )

    step3p5.Step3p5Model.forward = _pypto_full_forward
    step3p5.Step3p5ForCausalLM.compute_logits = _pypto_tail_compute_logits
    if prefill_dispatch:
        step3p5.Step3p5ForCausalLM.forward = _pypto_causal_forward_dispatch

    _set_patch_state(step3p5, state)
    return {"ok": True, "installed": True, "mode": mode, "prefill_dispatch": prefill_dispatch}


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
        "prefill_dispatch": bool(state.prefill_dispatch) if state else False,
        "module": getattr(step3p5, "__file__", None),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["install", "uninstall", "status"])
    parser.add_argument("--mode", choices=["full"], default=None)
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
