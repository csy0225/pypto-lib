# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""L1 ctx=1 A/B helper: query vLLM greedy next-token for a single-token prompt.

Picks the first token of ``--word`` as the single-token prompt id (tid), queries
the local vLLM W8A8 oracle (greedy, max_tokens=1), and prints tid + the generated
token (text + id). Feed the SAME tid to the pypto worker via ``--hidden-token tid``
and compare the pypto argmax token id to the vLLM generated token id.

Run in the pypto .venv311 on the device host (needs transformers + requests)::
    python -m tests.step3p5._l1_ab_vllm --word 北京
"""
from __future__ import annotations

import argparse
import json
import sys

import requests
from transformers import AutoTokenizer

CKPT = "/mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--word", default="北京")
    p.add_argument("--tid", type=int, default=-1)
    p.add_argument("--url", default="http://localhost:8000/v1/completions")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    tid = args.tid if args.tid >= 0 else int(tok.encode(args.word, add_special_tokens=False)[0])

    body = {
        "model": "step3p5",
        "prompt": [tid],
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": 1,
        "return_tokens_as_token_ids": True,
    }
    r = requests.post(args.url, json=body, timeout=60)
    d = r.json()
    ch = d["choices"][0]
    gen_text = ch["text"]
    # try to recover the generated token id from logprobs (return_tokens_as_token_ids
    # makes tokens look like "token_id:123"); else re-encode the text.
    gen_id = None
    lp = ch.get("logprobs") or {}
    toks = lp.get("tokens") or []
    if toks and isinstance(toks[0], str) and toks[0].startswith("token_id:"):
        gen_id = int(toks[0].split(":", 1)[1])
    if gen_id is None:
        enc = tok.encode(gen_text, add_special_tokens=False)
        gen_id = enc[0] if enc else -1

    print(f"L1_INPUT tid={tid} in_text={tok.decode([tid])!r}")
    print(f"L1_VLLM gen_text={gen_text!r} gen_id={gen_id}")
    print(f"L1_VLLM_RAW {json.dumps(ch.get('logprobs'), ensure_ascii=False)[:300]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
