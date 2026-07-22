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
import urllib.request


def _next_token(base_url: str, ids: list[int]) -> tuple[str, float]:
    body = json.dumps({
        "model": "step3.5-flash", "prompt": ids, "max_tokens": 1,
        "temperature": 0, "top_p": 1, "logprobs": 1,
    }).encode()
    req = urllib.request.Request(
        base_url, data=body, headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=120).read())
    lp = d["choices"][0]["logprobs"]
    return lp["tokens"][0], lp["token_logprobs"][0]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--seed-token", type=int, default=6127)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1/completions")
    args = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)

    ctx = [args.seed_token]
    oracle: list[int] = []
    multi = 0
    for _ in range(args.n):
        s, _val = _next_token(args.base_url, ctx)
        enc = tok.encode(s, add_special_tokens=False)
        if len(enc) != 1:
            multi += 1
        nid = enc[0] if enc else -1
        oracle.append(nid)
        ctx.append(nid)
    print("SEED=%d SEED_TEXT=%r N=%d" % (
        args.seed_token, tok.decode([args.seed_token]), args.n))
    print("ORACLE_IDS_JSON=" + json.dumps(oracle))
    print("ORACLE_TEXT=" + repr(tok.decode(oracle)))
    print("MULTI_TOKEN_STEPS=%d" % multi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
