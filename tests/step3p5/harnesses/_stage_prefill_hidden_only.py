#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Standalone Main hidden-only resident prefill gate.

This is the prefill dual of ``tests/step3p5/harnesses/_stage_main_hidden_only.py``
(the decode harness).  It is a production-ABI diagnostic, not a logits-producing
canonical harness:

``checkpoint prompt embedding + vLLM-style single-sequence paged KV metadata``
    -> ``one resident 45-layer PyPTO hidden-only prefill program``
    -> ``vLLM tail reference (final RMSNorm + LM head + greedy sampler on the
        last prompt row)``

The tail reference exists only to close a token diagnostic on the standalone
gate.  It is not passed into PyPTO and PyPTO returns no logits/token.

Prefill semantics differ from decode in three load-bearing ways (design doc
§5.1.3):

* **Single sequence prefill** (``PREFILL_BATCH=1``, ``PREFILL_T=128``).  NOT
  decode's ``BATCH=16`` paged multi-token step.  The token dimension is
  ``PREFILL_T=128``: ``current_hidden [tp, PREFILL_T, HIDDEN]`` and
  ``next_hidden_out [tp, PREFILL_T, HIDDEN]``.
* ``set_live_prompt(hidden, *, seq_lens, positions, block_table, slot_mapping)``
  replaces decode's ``set_live_step``.  ``hidden`` is ``[T, HIDDEN]`` BF16 with
  ``1 <= T <= PREFILL_T`` (one sequence's prompt embedding); ``seq_lens`` is
  ``[1]`` INT32; ``positions`` / ``slot_mapping`` are per-token ``[PREFILL_T]``;
  ``block_table`` is the single sequence's flat ``[1, max_blocks]`` block list.
* **KV write semantics**: prefill WRITES the paged cache (decode reads it).
  The KV-probe assertions therefore check that the active block's slots were
  *written*, not that untouched history was preserved.

The step loop is a single (or multi-length) prefill invocation, NOT decode's
8-step autoregressive chain.  For each requested prompt length ``T`` the
harness loads ``T`` prompt embedding rows, builds single-sequence metadata for
a fresh prompt (``positions == arange(T)``, ``block_table == [0, 1, ...]``),
calls ``holder.set_live_prompt(...)`` + ``holder.run()``, and gates on
``PREFILL_HIDDEN_DEVICE_PASS`` (finite + correct shape + hidden-only ownership).

IS_SCAFFOLD awareness: the prefill program
``models.step3p5.prefill_layer_single_chip_hidden.whole_prefill_step3p5`` is
currently a scaffold (``IS_SCAFFOLD=True``), so ``holder.build()`` raises
``RuntimeError``.  This harness does NOT catch or work around that -- it is
meant to run after P1 completes and ``IS_SCAFFOLD`` flips to ``False``.  The
module still imports cleanly card-free (no device needed at import).

v1 scope (deliberate simplifications, noted as future work):

* ITL/perf-vs-context sweep is dropped (prefill is single-shot, not
  steady-state; a ``--prompt-lens`` latency sweep is the natural analog but is
  left for a later perf gate).
* The KV-probe is kept for structural parity with the decode harness (same
  ``_collect_kv_probe`` / ``_kv_probe_summary`` infrastructure, all 45 layers,
  slots 0/1/2) but will not run until P1 because ``build()`` refuses to compile
  the scaffold.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import torch


TP = 8
BATCH = 16
HIDDEN = 4096
BLOCK_SIZE = 128
PREFILL_T = 128
# No private path: env fallback first, then the sidecar's neutral /mnt default
# (matches tests/step3p5/probes/_l1_ab_vllm.py).  The decode harness's
# username-containing default is intentionally NOT mirrored (repo rule: no
# private paths).
DEFAULT_CKPT = os.environ.get(
    "STEP3P5_CKPT_DIR",
    "/mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="8,9,10,11,12,13,14,15")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-blocks", type=int, default=32)
    parser.add_argument(
        "--prompt-lens",
        default=str(PREFILL_T),
        help=(
            "comma list of fresh-prompt lengths (e.g. 128,64,32); each runs "
            "one single-sequence prefill.  v1 caps each length at "
            f"PREFILL_T={PREFILL_T}"
        ),
    )
    parser.add_argument("--export-rank", type=int, default=-1)
    parser.add_argument("--dev", type=int, default=8)
    parser.add_argument("--reuse-exporters", action="store_true")
    parser.add_argument(
        "--kv-probe",
        action="store_true",
        help=(
            "diagnostic-only: after each holder.run(), ask every allocation "
            "owner to D2H-capture all 45 layers, K/V, and slots 0/1/2 "
            "(prefill writes paged cache; slots 0/1/2 are the first three "
            "rows of block 0)"
        ),
    )
    parser.add_argument("--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    parser.add_argument(
        "--full-attn-online-softmax-blocks-per-task",
        type=int,
        default=0,
        help=(
            "override full-attention SV+segment-recurrence grain before "
            "config import; zero keeps the release/config value"
        ),
    )
    parser.add_argument(
        "--full-attn-online-softmax-partials-per-reduce-task",
        type=int,
        default=0,
        help=(
            "override the number of segment partials merged by each parallel "
            "full-attention online-softmax reduction task before config import; "
            "zero keeps the release/config value"
        ),
    )
    parser.add_argument(
        "--full-attn-out-proj-fuse-cast",
        action="store_true",
        help=(
            "diagnostic-only: fuse the full-attention out-proj FP32-to-BF16 "
            "cast into each out-proj matmul task before config import"
        ),
    )
    parser.add_argument(
        "--swa-out-proj-fuse-cast",
        action="store_true",
        help=(
            "diagnostic-only: fuse the SWA out-proj FP32-to-BF16 cast into "
            "each out-proj matmul task before config import"
        ),
    )
    parser.add_argument(
        "--dfx",
        default="",
        help=(
            "PERF-A1 DFX capture tokens passed to holder via N1_DFX "
            "(e.g. 'swim' for l2_swimlane, 'pmu' for AICore PMU, 'scope'/'dep'). "
            "Artifacts land under {compiled.output_dir}/dfx_outputs/. Run swim and "
            "pmu in SEPARATE invocations (each perturbs timing)."
        ),
    )
    parser.add_argument("--pmu", type=int, default=1, help="AICore PMU event type when --dfx contains 'pmu' (1=CYCLE..4=MEMORY)")
    return parser.parse_args()


def _devices(text: str) -> list[int]:
    devices = [int(item) for item in str(text).split(",") if item.strip()]
    if len(devices) != TP or len(set(devices)) != TP:
        raise ValueError(f"expected {TP} distinct devices, got {devices}")
    return devices


def _export_rank(args: argparse.Namespace) -> int:
    if not 0 <= args.export_rank < TP:
        raise ValueError(f"export rank must be 0..{TP - 1}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from tools.step3p5.main_kv_exporter import MainKvExporter
    from tools.step3p5.pypto_weight_ipc import export_from_checkpoint_resident

    weight_owner, weight_summary, _ = export_from_checkpoint_resident(
        args.ckpt,
        rank=args.export_rank,
        tp_world_size=TP,
        out_dir=str(out),
        dev=args.dev,
        int8_routed=True,
        kv_ipc=False,
        production_hidden_only=True,
    )
    kv_owner = MainKvExporter(args.dev)
    kv_summary = kv_owner.export(
        out_dir=str(out),
        rank=args.export_rank,
        tp_world_size=TP,
        num_blocks=args.num_blocks,
    )
    ready = out / f"ready.rank{args.export_rank}"
    ready.write_text(
        json.dumps(
            {
                "rank": args.export_rank,
                "weight": weight_summary,
                "kv": kv_summary,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        f"[prefill-hidden-export rank={args.export_rank}] ready "
        f"weight_bytes={weight_summary['pool_bytes']} "
        f"kv_bytes={kv_summary['pool_bytes']} dev={args.dev}",
        flush=True,
    )
    try:
        stop = out / "STOP"
        last_probe_id = ""
        while not stop.exists():
            if args.kv_probe:
                request_path = out / (
                    f"kv_probe_request.rank{args.export_rank}.json"
                )
                if request_path.exists():
                    try:
                        request = json.loads(
                            request_path.read_text(encoding="utf-8")
                        )
                        probe_id = str(request["probe_id"])
                        if probe_id != last_probe_id:
                            rows, summary = kv_owner.snapshot_rows(
                                layer_indices=[
                                    int(item)
                                    for item in request["layer_indices"]
                                ],
                                slots=[
                                    int(item) for item in request["slots"]
                                ],
                            )
                            full_pool = None
                            if bool(request.get("full_pool", False)):
                                full_pool = kv_owner.snapshot_full_pool(
                                    out_dir=str(out),
                                    probe_id=probe_id,
                                    chunk_rows=int(
                                        request.get("full_pool_chunk_rows", 8192)
                                    ),
                                )
                            tensor_path = out / (
                                f"kv_probe_{probe_id}."
                                f"rank{args.export_rank}.pt"
                            )
                            tensor_tmp = tensor_path.with_name(
                                tensor_path.name + f".tmp.{os.getpid()}"
                            )
                            torch.save(rows, tensor_tmp)
                            os.replace(tensor_tmp, tensor_path)
                            result = {
                                "ok": True,
                                "probe_id": probe_id,
                                "step": int(request["step"]),
                                "rank": args.export_rank,
                                "device": args.dev,
                                "tensor_path": str(tensor_path),
                                "summary": summary,
                                "full_pool": full_pool,
                            }
                            result_path = out / (
                                f"kv_probe_{probe_id}."
                                f"rank{args.export_rank}.json"
                            )
                            result_tmp = result_path.with_name(
                                result_path.name + f".tmp.{os.getpid()}"
                            )
                            result_tmp.write_text(
                                json.dumps(result, sort_keys=True),
                                encoding="utf-8",
                            )
                            os.replace(result_tmp, result_path)
                            last_probe_id = probe_id
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"[prefill-hidden-export rank={args.export_rank}] "
                            f"KV probe failed: {exc!r}",
                            flush=True,
                        )
            time.sleep(0.2 if args.kv_probe else 2.0)
    finally:
        kv_owner.teardown()
        weight_owner.teardown()
    print(
        f"[prefill-hidden-export rank={args.export_rank}] STOP seen; exit",
        flush=True,
    )
    return 0


def _ready(out: Path, rank: int) -> bool:
    return (
        (out / f"ready.rank{rank}").exists()
        and (out / f"pypto_weight_map.rank{rank}.json.done").exists()
        and (out / f"pypto_kvpool_map.json.rank{rank}.done").exists()
    )


def _stop_exporters(out: Path, procs: list[subprocess.Popen]) -> None:
    try:
        (out / "STOP").write_text("1", encoding="utf-8")
    except OSError:
        pass
    for proc in procs:
        try:
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=30)
        handle = getattr(proc, "_prefill_hidden_log_handle", None)
        if handle is not None:
            handle.close()


def _start_exporters(args: argparse.Namespace, devices: list[int]) -> list[subprocess.Popen]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in out.glob("ready.rank*"):
        path.unlink(missing_ok=True)
    for path in out.glob("*.done"):
        path.unlink(missing_ok=True)
    for pattern in (
        "pypto_weight.*",
        "pypto_kvpool.*",
        "ipc_heartbeat.*",
        "kv_probe_*",
    ):
        for path in out.glob(pattern):
            path.unlink(missing_ok=True)
    (out / "STOP").unlink(missing_ok=True)

    root = Path(__file__).resolve().parents[3]
    procs: list[subprocess.Popen] = []
    for rank, dev in enumerate(devices):
        handle = open(out / f"export_rank{rank}.log", "w", encoding="utf-8")
        command = [
                sys.executable,
                "-m",
                "tests.step3p5.harnesses._stage_prefill_hidden_only",
                "--export-rank",
                str(rank),
                "--dev",
                str(dev),
                "--out",
                str(out),
                "--ckpt",
                args.ckpt,
                "--num-blocks",
                str(args.num_blocks),
            ]
        if args.kv_probe:
            command.append("--kv-probe")
        proc = subprocess.Popen(
            command,
            cwd=str(root),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        setattr(proc, "_prefill_hidden_log_handle", handle)
        procs.append(proc)

    deadline = time.time() + 2400.0
    while time.time() < deadline:
        if all(_ready(out, rank) for rank in range(TP)):
            return procs
        if any(proc.poll() not in (None, 0) for proc in procs):
            _stop_exporters(out, procs)
            raise RuntimeError("Prefill hidden exporter exited before readiness")
        time.sleep(3.0)
    _stop_exporters(out, procs)
    raise TimeoutError("Prefill hidden exporters were not ready within 40 minutes")


def _collect_kv_probe(
    out: Path,
    *,
    step: int,
    timeout_sec: float = 60.0,
    phase: str = "",
    full_pool: bool = False,
    full_pool_chunk_rows: int = 8192,
) -> dict[str, Any]:
    """Collect post-run owner-side KV evidence from all eight ranks.

    Structural mirror of the decode harness's probe.  Prefill writes the paged
    cache (decode reads it), so the slot assertions in the worker check that
    the active block's rows were *written*; this collector itself is
    model-agnostic (it just asks every owner to D2H-snapshot the requested
    layers/slots and durably writes the results).
    """
    probe_id = f"step{int(step)}-{uuid.uuid4().hex}"
    request = {
        "probe_id": probe_id,
        "step": int(step),
        "phase": str(phase),
        "full_pool": bool(full_pool),
        "full_pool_chunk_rows": int(full_pool_chunk_rows),
        # Cover every physical decoder layer, both K/V sections, and the first
        # three rows of block 0 (the slots a fresh prompt of length >= 3
        # writes: slot == block_table[pos // BLOCK_SIZE] * BLOCK_SIZE + pos %
        # BLOCK_SIZE, and block_table[0] == 0 so slots 0/1/2 == positions
        # 0/1/2).  Padding slots (beyond the prompt) point at the
        # allocator-owned reserve and are not written by this prefill.
        "layer_indices": list(range(45)),
        "slots": [0, 1, 2],
    }
    for rank in range(TP):
        path = out / f"kv_probe_request.rank{rank}.json"
        tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

    results: list[dict[str, Any]] = []
    deadline = time.time() + float(timeout_sec)
    while time.time() < deadline:
        paths = [
            out / f"kv_probe_{probe_id}.rank{rank}.json"
            for rank in range(TP)
        ]
        if all(path.exists() for path in paths):
            results = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in paths
            ]
            break
        time.sleep(0.2)
    if len(results) != TP:
        raise TimeoutError(
            f"KV probe {probe_id} did not complete on all {TP} ranks"
        )
    if any(
        not item.get("ok")
        or item.get("probe_id") != probe_id
        or int(item.get("step", -1)) != int(step)
        for item in results
    ):
        raise RuntimeError(f"invalid KV probe results for {probe_id}")

    aggregate = {
        "probe_id": probe_id,
        "step": int(step),
        "layer_indices": request["layer_indices"],
        "slots": request["slots"],
        "ranks": results,
    }
    suffix = f"_{phase}" if phase else ""
    aggregate_path = out / f"kv_probe_step{int(step)}{suffix}.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "kv_probe": probe_id,
                "step": int(step),
                "phase": str(phase),
                "full_pool": bool(full_pool),
                "ranks": TP,
                "path": str(aggregate_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return aggregate


def _kv_probe_summary(aggregate: dict[str, object]) -> dict[str, object]:
    """Summarize all-layer slot state across all owner ranks.

    Structural mirror of the decode harness's summary.  This is diagnostic
    evidence only.  The owner-side probe reads the exact exported rows after
    ``rt.run()``; it does not add a device operation to the production program.
    """
    ranks = aggregate.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != TP:
        raise ValueError("KV probe aggregate must contain all TP ranks")
    summaries: list[dict[str, object]] = []
    for item in ranks:
        if not isinstance(item, dict):
            raise ValueError("invalid KV probe rank result")
        summary = item.get("summary")
        if not isinstance(summary, dict):
            raise ValueError("KV probe rank result has no summary")
        summaries.append(summary)

    if aggregate.get("layer_indices") != list(range(45)):
        raise ValueError("KV probe must cover layers 0..44")
    if aggregate.get("slots") != [0, 1, 2]:
        raise ValueError("KV probe must cover slots 0/1/2")

    def observations(slot: int) -> list[bool]:
        values: list[bool] = []
        for summary in summaries:
            for layer in range(45):
                for which in ("K", "V"):
                    key = f"L{layer}.{which}.slot{slot}"
                    entry = summary.get(key)
                    if not isinstance(entry, dict):
                        raise ValueError(f"KV probe is missing {key}")
                    values.append(int(entry.get("nonzero", -1)) > 0)
        return values

    def hashes(slot: int) -> dict[str, str]:
        values: dict[str, str] = {}
        for rank, summary in enumerate(summaries):
            for layer in range(45):
                for which in ("K", "V"):
                    key = f"L{layer}.{which}.slot{slot}"
                    entry = summary.get(key)
                    if not isinstance(entry, dict):
                        raise ValueError(f"KV probe is missing {key}")
                    values[f"rank{rank}.{key}"] = str(entry.get("sha256", ""))
        return values

    slot0 = observations(0)
    slot1 = observations(1)
    slot2 = observations(2)
    return {
        "slot0_any_nonzero": any(slot0),
        "slot0_all_nonzero": all(slot0),
        "slot1_any_nonzero": any(slot1),
        "slot1_all_nonzero": all(slot1),
        "slot2_any_nonzero": any(slot2),
        "slot2_all_nonzero": all(slot2),
        "slot0_nonzero_count": sum(slot0),
        "slot1_nonzero_count": sum(slot1),
        "slot2_nonzero_count": sum(slot2),
        "observed_values": len(slot0),
        "slot0_hashes": hashes(0),
        "slot1_hashes": hashes(1),
        "slot2_hashes": hashes(2),
    }


def _load_embedding_row(ckpt: str, token: int) -> torch.Tensor:
    """Load a single token embedding row (BF16 ``[HIDDEN]``).

    Mirrored verbatim from the decode harness: the embedding-load primitive is
    model-agnostic.  Prefill's multi-row prompt loader (``_load_prompt_embedding``)
    builds on the same safetensors index/slice path.
    """
    import safetensors.torch as st

    index_path = Path(ckpt) / "quant_model_weights.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard = index["weight_map"]["model.embed_tokens.weight"]
    with st.safe_open(str(Path(ckpt) / shard), framework="pt") as handle:
        row = handle.get_slice("model.embed_tokens.weight")[int(token), :]
    if tuple(row.shape) != (HIDDEN,):
        raise ValueError(f"embedding row shape={tuple(row.shape)}")
    return row.to(torch.bfloat16).contiguous()


def _load_prompt_embedding(ckpt: str, prompt_len: int) -> torch.Tensor:
    """Load ``prompt_len`` contiguous prompt embedding rows (tokens 0..T-1).

    Prefill dual of decode's per-step ``_load_embedding_row``: a prefill step
    feeds ``T`` token embeddings at once (``[T, HIDDEN]`` BF16), not one.  The
    canonical standalone gate uses tokens ``arange(T)`` as a synthetic fresh
    prompt; a real prompt would supply its own token ids.
    """
    import safetensors.torch as st

    if not 1 <= prompt_len <= PREFILL_T:
        raise ValueError(
            f"prompt_len must be in [1,{PREFILL_T}] (PREFILL_T), "
            f"got {prompt_len}"
        )
    index_path = Path(ckpt) / "quant_model_weights.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard = index["weight_map"]["model.embed_tokens.weight"]
    with st.safe_open(str(Path(ckpt) / shard), framework="pt") as handle:
        rows = handle.get_slice("model.embed_tokens.weight")[:prompt_len, :]
    if tuple(rows.shape) != (prompt_len, HIDDEN):
        raise ValueError(f"prompt embedding shape={tuple(rows.shape)}")
    return rows.to(torch.bfloat16).contiguous()


def _cpu_tail_token(hidden: torch.Tensor, *, ckpt: str) -> int:
    """vLLM-tail diagnostic: final RMSNorm + LM head + greedy sampler.

    Mirrored verbatim from the decode harness.  The tail is identical for
    prefill; prefill just produces more hidden rows.  The caller passes the
    last active prompt row (the next-token predictor) and this returns the
    greedy argmax token.
    """
    from models.step3p5.config import EPS, LM_HEAD_K_CHUNK, VOCAB_LOCAL
    from models.step3p5.weight_loader import (
        _ShardCache,
        _read_index,
        _slice_lm_head,
    )
    from tools.step3p5.pypto_mtp3_ctx1_reference import (
        _chunked_mm,
        _zc_rmsnorm,
    )

    hidden = hidden.reshape(1, HIDDEN).to(torch.bfloat16)
    weight_map = _read_index(ckpt)
    with _ShardCache(ckpt, weight_map) as cache:
        norm = _zc_rmsnorm(
            hidden,
            cache.get("model.norm.weight"),
            EPS,
        )
        full_head = cache.get("lm_head.weight")
        shards = []
        for rank in range(TP):
            local = _slice_lm_head(full_head, rank, int(VOCAB_LOCAL))
            shards.append(
                _chunked_mm(
                    norm,
                    local.transpose(0, 1),
                    k_chunk=int(LM_HEAD_K_CHUNK),
                )
            )
        logits = torch.cat(shards, dim=-1)
    return int(logits.argmax(dim=-1).item())


def _prompt_metadata(
    *,
    seq_len: int,
    scheduler_num_blocks: int,
    block_table_flat: int,
    prefill_t: int = PREFILL_T,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build single-sequence prefill metadata for one fresh prompt.

    Prefill dual of decode's ``_step_metadata``.  Builds the SINGLE-SEQUENCE
    prefill ABI (``seq_lens [1]``, ``positions [prefill_t]``, ``block_table
    [1, block_table_flat]``, ``slot_mapping [prefill_t]``) instead of decode's
    ``BATCH=16`` paged multi-token metadata.

    For a fresh prompt of length ``seq_len`` (no prior context):

    * ``seq_lens = [seq_len]``;
    * ``positions[:seq_len] = arange(seq_len)`` (fresh prompt; the holder
      validates ``positions == arange(seq_len - T, seq_len)`` and ``seq_len ==
      T`` for a fresh prompt), padding ``[seq_len:] = 0``;
    * ``block_table[0, :active_blocks] = arange(active_blocks)`` (scheduler-
      owned block ids ``0..active_blocks-1``), trailing columns zero;
    * ``slot_mapping[t] = block_table[t // BLOCK_SIZE] * BLOCK_SIZE + t %
      BLOCK_SIZE`` for ``t < seq_len``; padding slots point at the
      allocator-owned reserve block (one block is enough -- padding KV is never
      read, the tail hidden rows are zeroed by the holder).  Mirrors
      ``vllm_prefill_metadata._extract_prefill_group_metadata``.

    Uses ``make_padding_reserve`` for the allocator-owned reserve (shared with
    decode; ``storage_capacity=STORAGE_BATCH``).  Does NOT call decode's
    ``validate_fixed_batch_metadata`` (it hardcodes ``STORAGE_BATCH=16`` plus 15
    padding rows, wrong for a single-sequence prefill step); the holder's
    ``set_live_prompt`` performs the prefill-specific validation itself.
    """
    from tools.step3p5.kv_padding import make_padding_reserve

    if not 1 <= seq_len <= prefill_t:
        raise ValueError(
            f"seq_len must be in [1,{prefill_t}] (PREFILL_T), got {seq_len}"
        )
    active_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    if active_blocks > scheduler_num_blocks:
        raise ValueError(
            f"seq_len={seq_len} needs {active_blocks} scheduler blocks, "
            f"got {scheduler_num_blocks}"
        )
    if block_table_flat < active_blocks:
        raise ValueError(
            f"block_table_flat={block_table_flat} cannot cover "
            f"{active_blocks} active blocks"
        )

    reserve = make_padding_reserve(
        scheduler_num_blocks,
        scheduler_num_blocks + 15,
    )
    seq_lens = torch.tensor([int(seq_len)], dtype=torch.int32)
    positions = torch.zeros(prefill_t, dtype=torch.int32)
    positions[:seq_len] = torch.arange(seq_len, dtype=torch.int32)

    block_table = torch.zeros(1, int(block_table_flat), dtype=torch.int32)
    block_table[0, :active_blocks] = torch.arange(active_blocks, dtype=torch.int32)

    slot_mapping = torch.zeros(prefill_t, dtype=torch.int32)
    active_positions = torch.arange(seq_len, dtype=torch.long)
    cols = active_positions // BLOCK_SIZE
    slot_mapping[:seq_len] = (
        block_table[0, :active_blocks]
        .to(torch.int32)
        .index_select(0, cols.to(torch.long))
        * BLOCK_SIZE
        + (active_positions % BLOCK_SIZE).to(torch.int32)
    )
    # Padding slots point at the allocator-owned reserve block (one block is
    # enough -- padding KV is never read, the tail hidden rows are zeroed by
    # the holder).  Mirrors vllm_prefill_metadata._extract_prefill_group_metadata.
    if seq_len < prefill_t:
        padding_block = int(reserve.padding_block_ids[0])
        slot_mapping[seq_len:] = padding_block * BLOCK_SIZE
    return seq_lens, positions, block_table, slot_mapping


def _run_worker(args: argparse.Namespace) -> int:
    devices = _devices(args.device)

    prompt_lens = [int(x) for x in str(args.prompt_lens).split(",") if x.strip()]
    if not prompt_lens:
        raise ValueError("--prompt-lens must list at least one positive length")
    for length in prompt_lens:
        if not 1 <= length <= PREFILL_T:
            raise ValueError(
                f"prompt len {length} out of [1,{PREFILL_T}] (PREFILL_T)"
            )
        need = (length + BLOCK_SIZE - 1) // BLOCK_SIZE
        if need > args.num_blocks:
            raise ValueError(
                f"prompt len {length} needs {need} scheduler blocks, "
                f"got --num-blocks {args.num_blocks}"
            )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    procs = [] if args.reuse_exporters else _start_exporters(args, devices)
    if args.reuse_exporters and not all(_ready(out, rank) for rank in range(TP)):
        raise RuntimeError("reuse-exporters requested but Main maps are incomplete")

    # Configure these before importing models.step3p5.config or compiling.
    # The Main KV pool storage capacity is STORAGE_BATCH (=BATCH=16), shared
    # with decode; it is NOT PREFILL_T (the prefill token dimension).  The
    # compiled block_table flat width is BATCH * num_blocks (one
    # per-storage-row scheduler domain), and the single prefill sequence uses
    # only the first ceil(seq_len / BLOCK_SIZE) columns.
    physical_blocks = int(args.num_blocks) + 15
    block_table_flat = BATCH * int(args.num_blocks)
    os.environ["PYPTO_STEP3P5_MAX_SEQ"] = str(int(args.num_blocks) * BLOCK_SIZE)
    os.environ["PYPTO_STEP3P5_BLOCK_TABLE_FLAT"] = str(block_table_flat)
    os.environ["PYPTO_STEP3P5_KV_CACHE_ROWS"] = str(
        45 * physical_blocks * BLOCK_SIZE
    )
    os.environ["PYPTO_STEP3P5_ROPE_SEQ"] = str(int(args.num_blocks) * BLOCK_SIZE)
    if args.full_attn_online_softmax_blocks_per_task < 0:
        raise ValueError(
            "--full-attn-online-softmax-blocks-per-task must be non-negative"
        )
    if args.full_attn_online_softmax_blocks_per_task:
        os.environ[
            "PYPTO_STEP3P5_FULL_ATTN_ONLINE_SOFTMAX_BLOCKS_PER_TASK"
        ] = str(args.full_attn_online_softmax_blocks_per_task)
    if args.full_attn_online_softmax_partials_per_reduce_task < 0:
        raise ValueError(
            "--full-attn-online-softmax-partials-per-reduce-task must be "
            "non-negative"
        )
    if args.full_attn_online_softmax_partials_per_reduce_task:
        os.environ[
            "PYPTO_STEP3P5_FULL_ATTN_ONLINE_SOFTMAX_PARTIALS_PER_REDUCE_TASK"
        ] = str(args.full_attn_online_softmax_partials_per_reduce_task)
    if args.full_attn_out_proj_fuse_cast:
        os.environ["PYPTO_STEP3P5_FULL_ATTN_OUT_PROJ_FUSE_CAST"] = "1"
    if args.swa_out_proj_fuse_cast:
        os.environ["PYPTO_STEP3P5_SWA_OUT_PROJ_FUSE_CAST"] = "1"
    os.environ.setdefault("PYPTO_PROG_BUILD_DIR", str(out / "build_output"))
    if args.dfx:
        # PERF-A1: forward DFX tokens to WholePrefillHolder.run() via env.
        os.environ["N1_DFX"] = args.dfx
        os.environ["N1_PMU"] = str(args.pmu)

    from tools.step3p5.whole_prefill_holder import WholePrefillHolder

    # IS_SCAFFOLD gate: build() raises RuntimeError while the real 45-layer
    # prefill body is not yet ported (P1 pending).  Do NOT catch -- this
    # harness runs only after P1 completes and IS_SCAFFOLD flips to False.
    holder = WholePrefillHolder(
        device_ids=devices,
        out_dir=str(out),
        ckpt=args.ckpt,
        platform=args.platform,
        kv_ipc=True,
    ).build()
    reports: list[dict[str, object]] = []
    try:
        with holder:
            for length in prompt_lens:
                embedding = _load_prompt_embedding(args.ckpt, length)
                seq, pos, table, slot = _prompt_metadata(
                    seq_len=length,
                    scheduler_num_blocks=args.num_blocks,
                    block_table_flat=block_table_flat,
                )
                holder.set_live_prompt(
                    embedding,
                    seq_lens=seq,
                    positions=pos,
                    block_table=table,
                    slot_mapping=slot,
                )
                started = time.time()
                result = holder.run()
                elapsed = time.time() - started
                kv_probe = (
                    _collect_kv_probe(out, step=length)
                    if args.kv_probe
                    else None
                )
                hidden = result["next_hidden"]
                # next_hidden is the resident [tp, PREFILL_T, HIDDEN] output;
                # the active rows are [:, :length, :].
                hidden_active = (
                    hidden[:, :length, :].to(torch.bfloat16).clone()
                )
                torch.save(
                    hidden_active,
                    out / f"prefill_t{length:03d}_active_hidden.pt",
                )
                torch.save(
                    hidden[0, :length, :].to(torch.bfloat16).clone(),
                    out / f"prefill_t{length:03d}_hidden_rank0.pt",
                )
                active_float = hidden_active.float()
                active_finite = bool(torch.isfinite(active_float).all())
                expected_shape = [TP, length, HIDDEN]
                shape_ok = list(hidden_active.shape) == expected_shape
                active_nonzero_rows = int(
                    torch.count_nonzero(
                        active_float.abs().amax(dim=-1) > 0
                    ).item()
                )
                tp_spread = float(
                    (
                        hidden[:, :length, :].float()
                        - hidden[0:1, :length, :].float()
                    )
                    .abs()
                    .max()
                    .item()
                )
                if not active_finite:
                    raise AssertionError(
                        f"Prefill hidden is non-finite at T={length}"
                    )
                if not shape_ok:
                    raise AssertionError(
                        f"Prefill active hidden shape "
                        f"{list(hidden_active.shape)} != {expected_shape}"
                    )
                if active_nonzero_rows != TP * length:
                    raise AssertionError(
                        f"Prefill T={length}: active hidden has "
                        f"{active_nonzero_rows} nonzero rank/rows, "
                        f"expected {TP * length}"
                    )
                # Tail diagnostic on the last prompt row (the next-token
                # predictor).  Same tail as decode (final RMSNorm + LM head +
                # greedy sampler); recorded, not gated (no pinned prefill
                # oracle in v1).
                tail_token = _cpu_tail_token(
                    hidden[0, length - 1], ckpt=args.ckpt
                )
                report = {
                    "prompt_len": length,
                    "run_sec": elapsed,
                    "hidden_shape": list(hidden_active.shape),
                    "hidden_full_shape": list(hidden.shape),
                    "expected_shape": expected_shape,
                    "hidden_finite": active_finite,
                    "hidden_shape_ok": shape_ok,
                    "hidden_nonzero_rank_rows": active_nonzero_rows,
                    "hidden_tp_spread": tp_spread,
                    "hidden_row0_abs_max": float(
                        hidden[0, 0].float().abs().max().item()
                    ),
                    "tail_token": tail_token,
                }
                if kv_probe is not None:
                    report["kv_probe_path"] = str(
                        out / f"kv_probe_step{length}.json"
                    )
                    probe_summary = _kv_probe_summary(kv_probe)
                    report["kv_probe_summary"] = probe_summary
                    # Prefill WRITES paged cache (decode reads).  For a fresh
                    # prompt with block_table[0]=0, slot == pos, so slots
                    # 0/1/2 are written iff the prompt is long enough to reach
                    # positions 0/1/2 (length >= 1/2/3).  This is the write-
                    # semantics dual of decode's read-semantics history checks.
                    if not probe_summary["slot0_all_nonzero"]:
                        raise AssertionError(
                            f"Prefill T={length}: slot0 rows were not all "
                            "written"
                        )
                    if length >= 2 and not probe_summary["slot1_all_nonzero"]:
                        raise AssertionError(
                            f"Prefill T={length}: slot1 rows were not all "
                            "written"
                        )
                    if length >= 3 and not probe_summary["slot2_all_nonzero"]:
                        raise AssertionError(
                            f"Prefill T={length}: slot2 rows were not all "
                            "written"
                        )
                reports.append(report)
                print(json.dumps(report, sort_keys=True), flush=True)
    finally:
        if not args.reuse_exporters:
            _stop_exporters(out, procs)

    report = {
        "ok": True,
        "device_ids": devices,
        "checkpoint": args.ckpt,
        "program_name": holder.program_name,
        "prompts": reports,
        "ownership": {
            "pypto_output": "pre-final-norm BF16 next_hidden only",
            "vllm_tail": "final RMSNorm + LM head + greedy sampler (last row)",
            "kv_metadata": "standalone single-sequence prefill diagnostic",
            "diagnostic_mode": "fresh-prompt prefill sweep",
        },
    }
    (out / "prefill_hidden_only_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    # Canonical liveness marker consumed by the performance release runbook.
    # Print it only after the resident holder has completed every requested
    # prompt and the report has been durably written.
    print("[worker] RUN done", flush=True)
    print("RESULT=PREFILL_HIDDEN_DEVICE_PASS", flush=True)
    return 0


def main() -> int:
    args = _parse_args()
    if args.export_rank >= 0:
        return _export_rank(args)
    return _run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
