# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""vLLM-side selected-layer MTP bridge.

This module mirrors the Main CPU/Gloo rendezvous:

``vLLM sampled token + previous hidden + draft metadata``
    -> rank-0 AF_UNIX sidecar ``op=mtp_layer``
    -> replicated raw hidden
    -> vLLM shared-head norm/LM-head/argmax

The generic proposer remains unchanged.  In particular, this bridge does not
perform sampling, token feedback, acceptance or rollback.
"""
from __future__ import annotations

import os
from typing import Any

import torch

from tools.step3p5.vllm_monkey_patch import (
    GATE_FAIL_CLOSED,
    GATE_PROCEED,
    _sidecar_result_payload,
)


class PyPTOMtpUnavailable(RuntimeError):
    """Selected MTP layer cannot be served by the PyPTO sidecar."""


def _sock_path() -> str:
    return os.environ.get("PYPTO_WHOLE_DECODE_SOCK", "/logs/pypto_whole_decode.sock")


def _client():
    # Main and MTP must share one persistent connection.  WholeDecodeServer
    # serves one connection until the peer closes it; opening a second
    # process-global client would leave the MTP request queued behind the
    # still-live Main connection.
    from tools.step3p5.vllm_monkey_patch import _wd_client

    return _wd_client()


def _cpu_broadcast(group, value: Any, *, src: int = 0):
    return group.broadcast_object(value, src=src)


def run_pypto_mtp_layer(
    *,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    previous_hidden_states: torch.Tensor,
    spec_step_idx: int,
    vllm_config,
) -> torch.Tensor:
    """Run one selected MTP body and return the raw hidden to vLLM."""
    if input_ids is None:
        raise PyPTOMtpUnavailable(
            "MTP PyPTO ABI requires sampled input_token_ids; inputs_embeds-only "
            "draft calls are not supported"
        )
    if previous_hidden_states.ndim != 2 or previous_hidden_states.shape[-1] != 4096:
        raise PyPTOMtpUnavailable(
            "MTP previous hidden must be [T,4096], got "
            f"{tuple(previous_hidden_states.shape)}"
        )
    valid_tokens = int(previous_hidden_states.shape[0])
    if not 1 <= valid_tokens <= 16:
        raise PyPTOMtpUnavailable(
            f"MTP selected-layer ABI supports 1..16 rows, got {valid_tokens}"
        )
    if int(input_ids.numel()) != valid_tokens:
        raise PyPTOMtpUnavailable(
            "MTP input token count does not match previous hidden rows: "
            f"{input_ids.numel()} != {valid_tokens}"
        )

    from vllm.distributed import get_tensor_model_parallel_rank, get_tp_group
    from vllm.forward_context import get_forward_context
    from tools.step3p5.vllm_mtp_metadata import (
        MtpMetadataError,
        extract_pypto_mtp_layer_meta,
    )

    context = get_forward_context()
    if bool(getattr(context, "in_profile_run", False)):
        # Explicit profile/dummy no-op.  Real requests never fall back after a
        # PyPTO attempt because the selected layer may already have written KV.
        return previous_hidden_states

    tp_group = get_tp_group()
    rank = get_tensor_model_parallel_rank()
    previous_hidden_cpu = (
        previous_hidden_states.detach()
        .to("cpu", dtype=torch.bfloat16)
        .contiguous()
    )
    token_ids_cpu = input_ids.detach().to("cpu", dtype=torch.int32).flatten()
    positions_cpu = positions.detach().to("cpu", dtype=torch.int32).flatten()

    decision = GATE_FAIL_CLOSED
    meta = None
    local_error: Exception | None = None
    if rank == 0:
        try:
            if not os.path.exists(_sock_path()):
                raise PyPTOMtpUnavailable("MTP sidecar socket is absent")
            meta = extract_pypto_mtp_layer_meta(
                context,
                layer_idx=int(spec_step_idx) % 3,
                positions=positions_cpu,
                valid_tokens=valid_tokens,
            )
            decision = GATE_PROCEED
        except (MtpMetadataError, Exception) as exc:  # noqa: PERF203
            local_error = exc
            decision = GATE_FAIL_CLOSED

    decision = int(_cpu_broadcast(tp_group, decision, src=0))
    if decision != GATE_PROCEED:
        raise PyPTOMtpUnavailable(
            "MTP selected-layer request failed closed"
            + (f": {local_error!r}" if rank == 0 and local_error else "")
        )

    # Every TP rank has completed its local hidden/token CPU copy before rank 0
    # enters the PyPTO resident runtime.  This is CPU/Gloo only.
    tp_group.barrier()
    payload = None
    if rank == 0:
        assert meta is not None

        def _run_sidecar():
            active = torch.zeros(16, dtype=torch.int32)
            active[:valid_tokens] = 1
            padded_ids = torch.zeros(16, dtype=torch.int32)
            padded_ids[:valid_tokens] = token_ids_cpu
            tensors = {
                "previous_hidden": previous_hidden_cpu,
                "input_token_ids": padded_ids,
                "active_mask": active,
            }
            tensors.update(meta.protocol_tensors())
            out_meta, out = _client().decode(tensors, meta.protocol_meta())
            mtp_hidden = out.get("mtp_hidden")
            if (
                not isinstance(mtp_hidden, torch.Tensor)
                or mtp_hidden.dtype != torch.bfloat16
                or tuple(mtp_hidden.shape) != tuple(previous_hidden_cpu.shape)
            ):
                raise PyPTOMtpUnavailable(
                    "MTP sidecar returned invalid hidden ABI: "
                    f"{getattr(mtp_hidden, 'dtype', None)} "
                    f"{getattr(mtp_hidden, 'shape', None)}"
                )
            if not torch.isfinite(mtp_hidden.float()).all():
                raise PyPTOMtpUnavailable("MTP sidecar returned NaN/Inf")
            return out_meta, mtp_hidden.contiguous()

        payload = _sidecar_result_payload(
            _run_sidecar,
            output_key="mtp_hidden",
        )

    payload = _cpu_broadcast(tp_group, payload, src=0)
    if not payload or not payload.get("ok"):
        raise PyPTOMtpUnavailable(
            "MTP sidecar failed on rank0"
            + (
                f": {payload.get('error_type')}: {payload.get('error')}"
                if rank == 0 and payload
                else ""
            )
        )
    output_cpu = payload["mtp_hidden"] if "mtp_hidden" in payload else None
    if output_cpu is None:
        raise PyPTOMtpUnavailable("MTP sidecar payload has no hidden output")
    return output_cpu.to(
        device=previous_hidden_states.device,
        dtype=previous_hidden_states.dtype,
    )


__all__ = ["PyPTOMtpUnavailable", "run_pypto_mtp_layer"]
