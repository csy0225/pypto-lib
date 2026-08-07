#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Build a vanilla-vLLM greedy oracle token-id sequence for the live A/B gate.

CRITICAL: generate STEP-BY-STEP feeding explicit token ids (NO BOS, single-token
encode per step). Do NOT one-shot generate then ``tokenizer.encode(full_text)`` —
that re-tokenizes at merge boundaries and yields a WRONG id sequence (this is the
exact bug that poisoned the old hardcoded ``DEFAULT_ORACLE_TOKENS`` with token
19384 at position 2, where the true no-BOS greedy is 6127).

Runs where a vanilla vLLM W8A8 oracle (``/v1/completions``) and ``transformers``
are available (on 0162 that is inside the vLLM container; the pypto ``.venv311``
has neither). Prints ``ORACLE_IDS_JSON=[...]`` for the host-side pypto
teacher-forced comparison.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request


def _native_token_id(logprobs: dict[str, object]) -> int:
    """Return the vLLM-native token id and reject text-only responses."""
    tokens = logprobs.get("tokens")
    if not isinstance(tokens, list) or len(tokens) != 1:
        raise RuntimeError(
            "vanilla oracle response must contain exactly one native token id"
        )
    token = tokens[0]
    prefix = "token_id:"
    if not isinstance(token, str) or not token.startswith(prefix):
        raise RuntimeError(
            "vanilla oracle response missing native token id; "
            "set return_tokens_as_token_ids=true"
        )
    raw_id = token[len(prefix):]
    if not re.fullmatch(r"(?:0|[1-9][0-9]*)", raw_id):
        raise RuntimeError(
            f"invalid native token id from vanilla oracle: {token!r}"
        )
    return int(raw_id)


def _next_token(base_url: str, ids: list[int]) -> tuple[int, float]:
    body = json.dumps({
        "model": "step3.5-flash", "prompt": ids, "max_tokens": 1,
        "temperature": 0, "top_p": 1, "logprobs": 1,
        "return_tokens_as_token_ids": True,
    }).encode()
    req = urllib.request.Request(
        base_url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        d = json.loads(response.read())
    try:
        lp = d["choices"][0]["logprobs"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("vanilla oracle response has no logprobs") from exc
    if not isinstance(lp, dict):
        raise RuntimeError("vanilla oracle response logprobs is not an object")
    token_id = _native_token_id(lp)
    token_logprobs = lp.get("token_logprobs")
    logprob = (
        float(token_logprobs[0])
        if isinstance(token_logprobs, list)
        and token_logprobs
        and token_logprobs[0] is not None
        else float("nan")
    )
    return token_id, logprob


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--seed-token", type=int, default=6127)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1/completions")
    args = p.parse_args()
    if args.seed_token < 0:
        p.error("--seed-token must be non-negative")
    if args.n <= 0:
        p.error("--n must be positive")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        args.ckpt,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )

    ctx = [args.seed_token]
    oracle: list[int] = []
    for _ in range(args.n):
        nid, _val = _next_token(args.base_url, ctx)
        oracle.append(nid)
        ctx.append(nid)
    print("SEED=%d SEED_TEXT=%r N=%d" % (
        args.seed_token, tok.decode([args.seed_token]), args.n))
    print("ORACLE_IDS_JSON=" + json.dumps(oracle))
    print("ORACLE_TEXT=" + repr(tok.decode(oracle)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
