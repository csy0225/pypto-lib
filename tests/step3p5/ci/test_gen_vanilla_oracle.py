# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Card-free tests for the native-token-id vanilla oracle protocol."""
from __future__ import annotations

import json
import pytest

from tests.step3p5.ci import gen_vanilla_oracle


def test_native_token_id_accepts_vllm_token_id_marker() -> None:
    assert gen_vanilla_oracle._native_token_id(
        {"tokens": ["token_id:303"]}
    ) == 303


@pytest.mark.parametrize(
    "logprobs",
    [
        {},
        {"tokens": []},
        {"tokens": ["303"]},
        {"tokens": ["token_id:-1"]},
        {"tokens": ["token_id:+1"]},
        {"tokens": ["token_id:01"]},
        {"tokens": ["token_id: 1"]},
        {"tokens": ["token_id:not-an-int"]},
        {"tokens": ["token_id:303", "token_id:304"]},
    ],
)
def test_native_token_id_rejects_text_or_malformed_responses(
    logprobs: dict[str, object],
) -> None:
    with pytest.raises(RuntimeError, match="native token id"):
        gen_vanilla_oracle._native_token_id(logprobs)


def test_next_token_requests_and_returns_native_id(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "logprobs": {
                                "tokens": ["token_id:6127"],
                                "token_logprobs": [-0.25],
                            }
                        }
                    ]
                }
            ).encode()

    def _urlopen(request, timeout):
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(gen_vanilla_oracle.urllib.request, "urlopen", _urlopen)

    assert gen_vanilla_oracle._next_token("http://oracle", [303]) == (
        6127,
        -0.25,
    )
    assert captured["body"]["return_tokens_as_token_ids"] is True
    assert captured["body"]["prompt"] == [303]
    assert captured["timeout"] == 120


def test_next_token_fails_closed_without_logprobs(monkeypatch) -> None:
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def read(self) -> bytes:
            return b'{"choices": [{"text": "303"}]}'

    response = _Response()
    monkeypatch.setattr(
        gen_vanilla_oracle.urllib.request,
        "urlopen",
        lambda request, timeout: response,
    )

    with pytest.raises(RuntimeError, match="no logprobs"):
        gen_vanilla_oracle._next_token("http://oracle", [6127])
