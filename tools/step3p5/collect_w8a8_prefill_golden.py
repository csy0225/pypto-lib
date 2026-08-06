#!/usr/bin/env python3
"""Collect sampled Step3p5 W8A8 prefill vLLM golden dumps.

Prerequisite: the running vLLM Step3p5 model has the debug dump hook enabled
and supports ``VLLM_STEP3P5_DUMP_MAX_TOKENS``/``VLLM_STEP3P5_DUMP_PRUNE``.
The production collection on 0162 used max 128 sampled rows per forward and
kept only tensors required by PyPTO detail/final-logits comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "step3.5-flash-w8a8"
DEFAULT_PORT = 8001
# Log/golden roots are operator-specific; resolve from env with a neutral
# default so no username path is baked into the source (CLAUDE.md: no private
# info). Set STEP3P5_W8A8_PREFILL_LOG_ROOT / STEP3P5_W8A8_PREFILL_LIVE_DUMP in
# the run environment.
DEFAULT_LOG_ROOT = Path(
    os.environ.get(
        "STEP3P5_W8A8_PREFILL_LOG_ROOT",
        "/tmp/step3p5_w8a8_prefill_logs",
    )
)
DEFAULT_GOLDEN_ROOT = DEFAULT_LOG_ROOT / "golden_step3p5_w8a8_prefill_vllm_sampled"
DEFAULT_LIVE_DUMP = Path(
    os.environ.get(
        "STEP3P5_W8A8_PREFILL_LIVE_DUMP",
        "/tmp/step3p5_w8a8_prefill_logs/vllm_tensor_dump_w8a8_prefill_sampled",
    )
)
DEFAULT_SEQ_LENS = (1024, 4096, 8192, 32768, 65536, 131072)

KEEP_SUFFIX = {
    "layer_input",
    "input_norm",
    "qkv_proj",
    "qk_norm",
    "attn_gate_logits",
    "post_attn_residual",
    "post_attn_norm",
    "ffn_out",
    "moe_router",
}
KEEP_EXACT = {"main_logits", "model_input"}


def _dump_name(path: Path) -> str:
    return path.name.removesuffix(".pt").split("_", 2)[2]


def keep_dump_file(path: Path) -> bool:
    name = _dump_name(path)
    return name in KEEP_EXACT or any(name.endswith("_" + suffix) for suffix in KEEP_SUFFIX)


def post_json(port: int, path: str, payload: dict[str, Any], timeout: int = 7200) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_stable(dump_dir: Path, *, min_files: int = 100, timeout: int = 600) -> int:
    last = -1
    same = 0
    start = time.time()
    while time.time() - start < timeout:
        current = len(list(dump_dir.glob("*.pt")))
        if current == last and current >= min_files:
            same += 1
            if same >= 3:
                return current
        else:
            same = 0
            last = current
        time.sleep(2)
    return len(list(dump_dir.glob("*.pt")))


def meta_for_file(path: Path) -> dict[str, Any]:
    import torch  # noqa: PLC0415

    obj = torch.load(path, map_location="cpu")
    return obj.get("__meta__", {"name": path.stem, "rank": -1})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-root", type=Path, default=DEFAULT_GOLDEN_ROOT)
    parser.add_argument("--live-dump", type=Path, default=DEFAULT_LIVE_DUMP)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seq-len", type=int, action="append", default=None)
    parser.add_argument("--prompt-unit", default="的")
    parser.add_argument(
        "--prompt-token-offset",
        type=int,
        default=0,
        help="Subtract this from seq-len when repeating prompt-unit if the tokenizer adds BOS.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seq_lens = tuple(args.seq_len or DEFAULT_SEQ_LENS)

    args.golden_root.parent.mkdir(parents=True, exist_ok=True)
    if args.golden_root.exists():
        shutil.rmtree(args.golden_root)
    args.golden_root.mkdir(parents=True)

    cases: list[dict[str, Any]] = []
    for seq_len in seq_lens:
        case_name = f"prefill_{seq_len // 1024}k"
        print(f"[collect] case={case_name} seq={seq_len}", flush=True)

        args.live_dump.mkdir(parents=True, exist_ok=True)
        for path in args.live_dump.glob("*.pt"):
            path.unlink()
        (args.live_dump / ".enable").touch()

        repeat_count = max(1, seq_len - args.prompt_token_offset)
        payload = {
            "model": args.model,
            "prompt": args.prompt_unit * repeat_count,
            "max_tokens": 1,
            "temperature": 0,
            "stream": False,
        }
        t0 = time.time()
        try:
            response = post_json(args.port, "/v1/completions", payload)
        finally:
            try:
                (args.live_dump / ".enable").unlink()
            except FileNotFoundError:
                pass
        elapsed = time.time() - t0
        num_live_files = wait_stable(args.live_dump, min_files=100)
        print(f"[collect] response in {elapsed:.1f}s files={num_live_files}", flush=True)

        case_root = args.golden_root / case_name
        dump_dst = case_root / "dump"
        dump_dst.mkdir(parents=True)
        dump_files = []
        for src in sorted(args.live_dump.glob("*.pt")):
            if not keep_dump_file(src):
                continue
            dst = dump_dst / src.name
            shutil.copy2(src, dst)
            dump_files.append({"file": str(dst), "meta": meta_for_file(dst)})
        (case_root / "response.json").write_text(
            json.dumps(response, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        cases.append({
            "name": case_name,
            "seq_len": seq_len,
            "request": {
                "model": args.model,
                "prompt_unit": args.prompt_unit,
                "prompt_repeat_count": repeat_count,
                "prompt_tokens_expected": seq_len,
                "max_tokens": 1,
                "temperature": 0,
                "stream": False,
            },
            "response_summary": {
                "id": response.get("id"),
                "usage": response.get("usage"),
                "choices": response.get("choices"),
            },
            "elapsed_sec": elapsed,
            "num_dump_files": len(dump_files),
            "dump_files": dump_files,
            "dump_sampling": {
                "max_tokens_per_forward": 128,
                "unit_prompt_tokenizes_1_to_1": True,
            },
        })
        for path in args.live_dump.glob("*.pt"):
            path.unlink()

    manifest = {
        "root": str(args.golden_root),
        "quantization": "w8a8_dynamic",
        "phase": "prefill",
        "sampling": (
            "per-forward evenly-spaced token sample, max 128 rows; "
            "pruned to tensors required by PyPTO detail/final-logits compare"
        ),
        "cases": cases,
    }
    (args.golden_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps({
        "golden_root": str(args.golden_root),
        "cases": [(case["name"], case["num_dump_files"]) for case in cases],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
