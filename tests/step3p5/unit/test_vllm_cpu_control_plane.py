#!/usr/bin/env python3
"""Card-free contracts for the vLLM/PyPTO CPU control plane."""
from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import pytest

from tools.step3p5 import pypto_whole_decode_backend
from tools.step3p5 import vllm_monkey_patch


def _function_source(module, name: str) -> tuple[str, ast.FunctionDef]:
    source = inspect.getsource(module)
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment, node
    raise AssertionError(f"{module.__name__}.{name} not found")


def _called_attributes(node: ast.AST) -> list[str]:
    result = []
    for item in ast.walk(node):
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute):
            result.append(item.func.attr)
    return result


def test_live_forward_never_uses_tp_device_broadcast() -> None:
    """Proceed/status/output must not occupy the vLLM NPU device group."""
    source, node = _function_source(
        vllm_monkey_patch,
        "_pypto_full_forward",
    )
    called = _called_attributes(node)
    assert "broadcast" not in called, (
        "vllm_monkey_patch._pypto_full_forward must not call device broadcast"
    )
    assert called.count("broadcast_object") >= 2
    assert "barrier" in called
    assert 'to("cpu"' in source or "to(\n        \"cpu\"" in source

    # The historical container module is now only a compatibility wrapper.
    # It must not own a second live implementation or protocol.
    assert pypto_whole_decode_backend._full_forward is (
        vllm_monkey_patch._pypto_full_forward
    )


def test_legacy_backend_wrapper_has_no_independent_socket_protocol() -> None:
    source = inspect.getsource(pypto_whole_decode_backend)
    assert "_send_frame" not in source
    assert "_recv_frame" not in source
    assert "original(" not in source


def test_rank0_sidecar_error_is_a_terminal_payload() -> None:
    class SidecarFailure(RuntimeError):
        pass

    def fail():
        raise SidecarFailure("injected")

    payload = vllm_monkey_patch._sidecar_result_payload(fail)
    assert payload["ok"] is False
    assert payload["error_type"] == "SidecarFailure"
    assert "injected" in payload["error"]
    assert "next_hidden" not in payload
    assert "fallback" not in payload


def test_rank0_sidecar_success_keeps_cpu_hidden_in_payload() -> None:
    import torch

    hidden = torch.zeros((2, 4096), dtype=torch.bfloat16)
    payload = vllm_monkey_patch._sidecar_result_payload(
        lambda: ({"program": "whole_decode"}, hidden)
    )
    assert payload["ok"] is True
    assert payload["next_hidden"] is hidden
    assert payload["next_hidden"].device.type == "cpu"
    assert payload["out_meta"]["program"] == "whole_decode"


def test_rank0_mtp_sidecar_can_use_mtp_hidden_output_field() -> None:
    import torch

    hidden = torch.zeros((2, 4096), dtype=torch.bfloat16)
    payload = vllm_monkey_patch._sidecar_result_payload(
        lambda: ({"program": "mtp_layer_hidden_0"}, hidden),
        output_key="mtp_hidden",
    )
    assert payload["ok"] is True
    assert payload["mtp_hidden"] is hidden
    assert "next_hidden" not in payload


def test_decode_plan_runs_ordered_rounds_and_restores_flattened_order() -> None:
    import torch

    class Step:
        def __init__(self, token_indices):
            self.token_indices = tuple(token_indices)
            self.valid_tokens = len(self.token_indices)

        def protocol_tensors(self):
            return {
                "meta_seq_lens": torch.ones(16, dtype=torch.int32),
                "meta_positions": torch.zeros(16, dtype=torch.int32),
            }

        def protocol_meta(self):
            return {
                "protocol_version": 2,
                "op": "decode",
                "valid_tokens": self.valid_tokens,
                "valid_requests": self.valid_tokens,
                "storage_batch": 16,
                "kv_group_count": 1,
                "layer_to_group": [0] * 45,
                "query_lengths": [1] * self.valid_tokens,
                "padding_reserve": {
                    "scheduler_num_blocks": 64,
                    "physical_num_blocks": 79,
                    "reserve_start": 64,
                    "padding_block_ids": list(range(64, 79)),
                    "padding_block_count": 15,
                    "block_size": 128,
                },
            }

    class FakePlan:
        valid_tokens = 4
        query_lengths = (4,)
        steps = (
            Step((0,)),
            Step((1,)),
            Step((2,)),
            Step((3,)),
        )

    class FakeClient:
        def __init__(self):
            self.calls = []

        def decode(self, tensors, meta):
            self.calls.append((tensors["hidden"].clone(), dict(meta)))
            # Deliberately return a value that identifies the round. The
            # caller must place it back by token_indices, not append rounds.
            value = len(self.calls)
            return (
                {"round": value},
                {
                    "next_hidden": torch.full_like(
                        tensors["hidden"],
                        value,
                    )
                },
            )

    hidden = torch.arange(4 * 4096, dtype=torch.int32).reshape(4, 4096)
    hidden = hidden.to(torch.bfloat16)
    client = FakeClient()
    meta, output = vllm_monkey_patch._run_decode_plan(
        client,
        hidden,
        FakePlan(),
    )

    assert [call[0][0, 0].item() for call in client.calls] == [
        hidden[0, 0].item(),
        hidden[1, 0].item(),
        hidden[2, 0].item(),
        hidden[3, 0].item(),
    ]
    assert output[:, 0].tolist() == [1, 2, 3, 4]
    assert meta["round_count"] == 4
    assert [item["token_indices"] for item in meta["rounds"]] == [
        [0],
        [1],
        [2],
        [3],
    ]


def test_decode_plan_fails_closed_on_missing_output_row() -> None:
    import torch

    class Step:
        token_indices = (0,)
        valid_tokens = 1

        def protocol_tensors(self):
            return {}

        def protocol_meta(self):
            return {}

    plan = SimpleNamespace(
        valid_tokens=2,
        query_lengths=(2,),
        steps=(Step(),),
    )

    class Client:
        def decode(self, tensors, meta):
            return {}, {
                "next_hidden": torch.zeros(
                    (1, 4096),
                    dtype=torch.bfloat16,
                )
            }

    with pytest.raises(
        vllm_monkey_patch.PyPTOBackendUnavailable,
        match="did not produce hidden rows",
    ):
        vllm_monkey_patch._run_decode_plan(
            Client(),
            torch.zeros((2, 4096), dtype=torch.bfloat16),
            plan,
        )
