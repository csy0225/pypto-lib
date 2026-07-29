#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Standalone Main hidden-only resident decode gate.

This is a production-ABI diagnostic, not the historical logits-producing
canonical harness:

``checkpoint embedding + vLLM-style paged KV metadata``
    -> ``one resident 45-layer PyPTO hidden-only program``
    -> ``vLLM tail reference (final RMSNorm + LM head + greedy sampler)``

The tail reference exists only to close the token comparison on the standalone
gate.  It is not passed into PyPTO and PyPTO returns no logits/token.
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

import torch


TP = 8
BATCH = 16
HIDDEN = 4096
BLOCK_SIZE = 128
DEFAULT_CKPT = (
    "/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp"
)
# Vanilla 8000 greedy oracle for the canonical one-token decode.
DEFAULT_ORACLE_TOKENS = [303, 1207, 19384, 872, 428, 6127, 4231, 2636]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="8,9,10,11,12,13,14,15")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-blocks", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--active-batch",
        type=int,
        default=1,
        help=(
            "number of live rows in the fixed BATCH=16 storage ABI; active "
            "rows use identical ctx=1 metadata and padding rows use the "
            "allocator-owned reserve"
        ),
    )
    parser.add_argument("--teacher-forced", action="store_true", help="feed oracle token each step; log all steps, never raise (per-position top-1 accuracy vs a live/greedy oracle)")
    parser.add_argument("--seed-token", type=int, default=6127, help="first decode input token (default 6127)")
    parser.add_argument("--oracle-token", action="append", type=int)
    parser.add_argument("--export-rank", type=int, default=-1)
    parser.add_argument("--dev", type=int, default=8)
    parser.add_argument("--reuse-exporters", action="store_true")
    parser.add_argument(
        "--kv-probe",
        action="store_true",
        help=(
            "diagnostic-only: after each rt.run(), ask every allocation owner "
            "to D2H-capture all 45 layers, K/V, and slots 0/1/2"
        ),
    )
    parser.add_argument(
        "--repeat-identical",
        action="store_true",
        help=(
            "diagnostic-only: run token=6127 with identical seq_len=1, "
            "position=0 and slot=0 twice in one resident holder"
        ),
    )
    parser.add_argument("--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    parser.add_argument(
        "--itl-context-lens",
        default="",
        help=(
            "perf-only: comma list of decode context lengths (e.g. "
            "1024,8192,32768,65536). When set, skip the token loop; for each L "
            "pin metadata to seq_len=L and time holder.run() over --itl-iters "
            "(attention compute is invariant to KV content, so no prefill is "
            "needed). Emits per-context ITL stats to itl_report.json."
        ),
    )
    parser.add_argument("--itl-iters", type=int, default=20, help="measured decode iters per context")
    parser.add_argument("--itl-warmup", type=int, default=3, help="warmup decode iters per context (not recorded)")
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
        f"[main-hidden-export rank={args.export_rank}] ready "
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
                            f"[main-hidden-export rank={args.export_rank}] "
                            f"KV probe failed: {exc!r}",
                            flush=True,
                        )
            time.sleep(0.2 if args.kv_probe else 2.0)
    finally:
        kv_owner.teardown()
        weight_owner.teardown()
    print(
        f"[main-hidden-export rank={args.export_rank}] STOP seen; exit",
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
        handle = getattr(proc, "_main_hidden_log_handle", None)
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
                "tests.step3p5.harnesses._stage_main_hidden_only",
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
        setattr(proc, "_main_hidden_log_handle", handle)
        procs.append(proc)

    deadline = time.time() + 2400.0
    while time.time() < deadline:
        if all(_ready(out, rank) for rank in range(TP)):
            return procs
        if any(proc.poll() not in (None, 0) for proc in procs):
            _stop_exporters(out, procs)
            raise RuntimeError("Main hidden exporter exited before readiness")
        time.sleep(3.0)
    _stop_exporters(out, procs)
    raise TimeoutError("Main hidden exporters were not ready within 40 minutes")


def _collect_kv_probe(
    out: Path,
    *,
    step: int,
    timeout_sec: float = 60.0,
    phase: str = "",
    full_pool: bool = False,
    full_pool_chunk_rows: int = 8192,
) -> dict[str, Any]:
    """Collect post-run owner-side KV evidence from all eight ranks."""
    probe_id = f"step{int(step)}-{uuid.uuid4().hex}"
    request = {
        "probe_id": probe_id,
        "step": int(step),
        "phase": str(phase),
        "full_pool": bool(full_pool),
        "full_pool_chunk_rows": int(full_pool_chunk_rows),
        # PERF-B3 release evidence: cover every physical decode layer, both
        # K/V sections, the two adjacent active scheduler rows, and one
        # untouched history row. Padding writes use allocator-owned reserve
        # blocks above the scheduler domain and cannot alias slots 0/1/2.
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

    This is diagnostic evidence only.  The owner-side probe reads the exact
    exported rows after ``rt.run()``; it does not add a device operation to
    the production program.
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
        raise ValueError("PERF-B3 KV probe must cover layers 0..44")
    if aggregate.get("slots") != [0, 1, 2]:
        raise ValueError("PERF-B3 KV probe must cover slots 0/1/2")

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
    import safetensors.torch as st

    index_path = Path(ckpt) / "quant_model_weights.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard = index["weight_map"]["model.embed_tokens.weight"]
    with st.safe_open(str(Path(ckpt) / shard), framework="pt") as handle:
        row = handle.get_slice("model.embed_tokens.weight")[int(token), :]
    if tuple(row.shape) != (HIDDEN,):
        raise ValueError(f"embedding row shape={tuple(row.shape)}")
    return row.to(torch.bfloat16).contiguous()


def _cpu_tail_token(hidden: torch.Tensor, *, ckpt: str) -> int:
    """vLLM-tail diagnostic: final RMSNorm + LM head + greedy sampler."""
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


def _step_metadata(
    *,
    step: int,
    scheduler_num_blocks: int,
    valid_rows: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build scheduler-shaped metadata for a direct decode batch.

    This is standalone direct-decode metadata only.  The active row uses the
    scheduler domain; padding rows use the allocator-owned reserve.  Active
    rows share the same context length and use distinct slot rows within the
    scheduler domain.  No active-row mapping is invented in the live vLLM
    bridge.
    """
    from tools.step3p5.kv_padding import (
        make_padding_reserve,
        validate_fixed_batch_metadata,
    )

    reserve = make_padding_reserve(
        scheduler_num_blocks,
        scheduler_num_blocks + 15,
    )
    if not 1 <= int(valid_rows) <= BATCH:
        raise ValueError(f"valid_rows must be in [1,{BATCH}], got {valid_rows}")
    seq = torch.ones(BATCH, dtype=torch.int32)
    pos = torch.zeros(BATCH, dtype=torch.int32)
    table = torch.zeros(BATCH, scheduler_num_blocks, dtype=torch.int32)
    slot = torch.zeros(BATCH, dtype=torch.int32)

    block_index = int(step) // BLOCK_SIZE
    if block_index >= scheduler_num_blocks:
        raise ValueError(
            f"step={step} exceeds scheduler capacity {scheduler_num_blocks}"
        )
    # Give every active row an independent scheduler-owned paged sequence.
    # Interleaving block ids keeps each sequence's history disjoint:
    # row ``r`` owns blocks ``r, r+valid_rows, ...``.
    active_blocks = (
        torch.arange(valid_rows, dtype=torch.int32).unsqueeze(1)
        + valid_rows
        * torch.arange(block_index + 1, dtype=torch.int32).unsqueeze(0)
    )
    table[:valid_rows, : block_index + 1] = active_blocks
    seq[:valid_rows] = int(step) + 1
    pos[:valid_rows] = int(step)
    slot[:valid_rows] = (
        active_blocks[:, block_index] * BLOCK_SIZE
        + (int(step) % BLOCK_SIZE)
    )
    for padding_idx, block_id in enumerate(
        reserve.padding_block_ids[: BATCH - valid_rows], start=valid_rows
    ):
        table[padding_idx, 0] = int(block_id)
        slot[padding_idx] = int(block_id) * BLOCK_SIZE

    validate_fixed_batch_metadata(
        seq_lens=seq,
        positions=pos,
        block_table=table,
        slot_mapping=slot,
        valid_rows=valid_rows,
        reserve=reserve,
        where=f"standalone Main step {step}",
    )
    return seq, pos, table, slot


def _run_itl(holder, args: argparse.Namespace, out: Path) -> int:
    """perf-only: measure decode inter-token latency (ITL) vs context length.

    For each target context length L, pin the active-row metadata to seq_len=L
    and time ``holder.run()`` over ``--itl-iters`` (after ``--itl-warmup``).
    Attention compute is invariant to KV *content*, so we do not prefill — the
    per-step wall time at seq_len=L is the steady-state ITL at that context.
    Writes itl_report.json.
    """
    import statistics

    # ITL honours --active-batch: `valid_rows` active decode rows, each with its
    # own scheduler-owned paged sequence.  Row r owns blocks r, r+R, r+2R, ...
    # (see _step_metadata), so the block table needs R * ceil(L/BLOCK_SIZE)
    # scheduler blocks -- R times more than a single-row run at the same context.
    active = int(getattr(args, "active_batch", 1) or 1)
    if not 1 <= active <= BATCH:
        raise ValueError(f"--active-batch must be in [1,{BATCH}], got {active}")

    ctx_lens = [int(x) for x in str(args.itl_context_lens).split(",") if x.strip()]
    cap = args.num_blocks * BLOCK_SIZE
    for length in ctx_lens:
        if not 1 <= length <= cap:
            raise ValueError(
                f"itl context {length} out of range [1, {cap}] "
                f"(raise --num-blocks; need >= {(length + BLOCK_SIZE - 1) // BLOCK_SIZE})"
            )
        need = active * ((length + BLOCK_SIZE - 1) // BLOCK_SIZE)
        if need > args.num_blocks:
            raise ValueError(
                f"itl context {length} with --active-batch {active} needs "
                f">= {need} scheduler blocks (each of the {active} active rows "
                f"owns its own paged sequence), got --num-blocks "
                f"{args.num_blocks}"
            )

    # Fixed dummy embedding — content is irrelevant to decode-step timing.  One
    # row per active decode slot.
    embedding = (
        _load_embedding_row(args.ckpt, args.seed_token)
        .unsqueeze(0)
        .expand(active, -1)
        .contiguous()
    )
    results: list[dict[str, object]] = []
    for length in ctx_lens:
        seq, pos, table, slot = _step_metadata(
            step=length - 1,
            scheduler_num_blocks=args.num_blocks,
            valid_rows=active,
        )
        set_kwargs = dict(
            seq_lens=seq, positions=pos, block_table=table, slot_mapping=slot
        )
        for _ in range(max(0, args.itl_warmup)):
            holder.set_live_step(embedding, **set_kwargs)
            holder.run()
        samples: list[float] = []
        for _ in range(max(1, args.itl_iters)):
            holder.set_live_step(embedding, **set_kwargs)
            started = time.time()
            holder.run()
            samples.append(time.time() - started)
        ms = sorted(s * 1000.0 for s in samples)
        n = len(ms)
        res = {
            "context_len": length,
            "iters": n,
            "itl_ms_min": round(ms[0], 3),
            "itl_ms_mean": round(statistics.fmean(ms), 3),
            "itl_ms_p50": round(ms[n // 2], 3),
            "itl_ms_p99": round(ms[min(n - 1, int(n * 0.99))], 3),
            "itl_ms_max": round(ms[-1], 3),
        }
        results.append(res)
        print(json.dumps(res, sort_keys=True), flush=True)

    report = {
        "kind": "decode_itl",
        "num_blocks": args.num_blocks,
        "block_size": BLOCK_SIZE,
        # ``batch_capacity`` is the fixed storage upper bound (the BATCH formal);
        # ``active_batch`` is how many decode rows this run actually drove.
        "batch_capacity": BATCH,
        "active_batch": active,
        "warmup": args.itl_warmup,
        "results": results,
    }
    (out / "itl_report.json").write_text(json.dumps(report, indent=2))
    print(f"ITL_REPORT={out / 'itl_report.json'}", flush=True)
    return 0


def _run_worker(args: argparse.Namespace) -> int:
    devices = _devices(args.device)
    if not 1 <= args.active_batch <= BATCH:
        raise ValueError(f"--active-batch must be in [1,{BATCH}]")
    if not 1 <= args.steps <= args.num_blocks * BLOCK_SIZE:
        raise ValueError("--steps must be in [1, num_blocks*128]")
    if args.repeat_identical and args.steps != 2:
        raise ValueError("--repeat-identical requires --steps 2")
    if args.steps > args.num_blocks * BLOCK_SIZE:
        raise ValueError("steps exceed configured scheduler KV capacity")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    procs = [] if args.reuse_exporters else _start_exporters(args, devices)
    if args.reuse_exporters and not all(_ready(out, rank) for rank in range(TP)):
        raise RuntimeError("reuse-exporters requested but Main maps are incomplete")

    # Configure these before importing models.step3p5.config or compiling.
    physical_blocks = int(args.num_blocks) + 15
    os.environ["PYPTO_STEP3P5_MAX_SEQ"] = str(int(args.num_blocks) * BLOCK_SIZE)
    os.environ["PYPTO_STEP3P5_BLOCK_TABLE_FLAT"] = str(
        BATCH * int(args.num_blocks)
    )
    os.environ["PYPTO_STEP3P5_KV_CACHE_ROWS"] = str(
        45 * physical_blocks * BLOCK_SIZE
    )
    os.environ["PYPTO_STEP3P5_ROPE_SEQ"] = str(int(args.num_blocks) * BLOCK_SIZE)
    os.environ.setdefault("PYPTO_PROG_BUILD_DIR", str(out / "build_output"))
    if args.dfx:
        # PERF-A1: forward DFX tokens to WholeDecodeHolder.run() via env.
        os.environ["N1_DFX"] = args.dfx
        os.environ["N1_PMU"] = str(args.pmu)

    from tools.step3p5.whole_decode_holder import WholeDecodeHolder

    expected_tokens = (
        list(args.oracle_token)
        if args.oracle_token
        else list(DEFAULT_ORACLE_TOKENS)
    )
    if not args.repeat_identical and len(expected_tokens) < args.steps:
        raise ValueError(
            f"oracle has {len(expected_tokens)} tokens but steps={args.steps}"
        )
    if (not args.repeat_identical and not args.oracle_token and args.seed_token == 6127
            and expected_tokens[0] != 303):
        raise ValueError("built-in canonical Main oracle must start with token 303")

    holder = WholeDecodeHolder(
        device_ids=devices,
        out_dir=str(out),
        ckpt=args.ckpt,
        platform=args.platform,
        kv_ipc=True,
    ).build()
    reports: list[dict[str, object]] = []
    repeat_hidden: torch.Tensor | None = None
    previous_kv_probe_summary: dict[str, object] | None = None
    try:
        with holder:
            if args.itl_context_lens:
                return _run_itl(holder, args, out)
            token = args.seed_token
            for step in range(args.steps):
                metadata_step = 0 if args.repeat_identical else step
                embedding = _load_embedding_row(args.ckpt, token)
                seq, pos, table, slot = _step_metadata(
                    step=metadata_step,
                    scheduler_num_blocks=args.num_blocks,
                    valid_rows=args.active_batch,
                )
                holder.set_live_step(
                    embedding.unsqueeze(0).expand(args.active_batch, -1).contiguous(),
                    seq_lens=seq,
                    positions=pos,
                    block_table=table,
                    slot_mapping=slot,
                )
                started = time.time()
                result = holder.run()
                elapsed = time.time() - started
                kv_probe = (
                    _collect_kv_probe(out, step=step)
                    if args.kv_probe
                    else None
                )
                hidden = result["next_hidden"]
                hidden_snapshot = hidden.to(torch.bfloat16).clone()
                torch.save(
                    hidden_snapshot[:, : args.active_batch],
                    out / f"main_step{step:02d}_active_hidden.pt",
                )
                row0 = hidden[0, 0].float().clone()
                active_hidden = hidden[:, : args.active_batch].float()
                active_finite = bool(torch.isfinite(active_hidden).all())
                active_nonzero_rows = int(
                    torch.count_nonzero(
                        active_hidden.abs().amax(dim=-1) > 0
                    ).item()
                )
                torch.save(
                    hidden_snapshot[0, 0],
                    out / f"main_step{step:02d}_hidden.pt",
                )
                tp_spread = float(
                    (
                        hidden[:, : args.active_batch].float()
                        - hidden[0:1, : args.active_batch].float()
                    )
                    .abs()
                    .max()
                    .item()
                )
                if not torch.isfinite(row0).all():
                    raise AssertionError(f"Main hidden is non-finite at step {step}")
                if not active_finite:
                    raise AssertionError(
                        f"Main active hidden is non-finite at step {step}"
                    )
                if active_nonzero_rows != args.active_batch * TP:
                    raise AssertionError(
                        f"Main active hidden has {active_nonzero_rows} "
                        f"nonzero rank/rows, expected {args.active_batch * TP}"
                    )
                sampled = _cpu_tail_token(row0, ckpt=args.ckpt)
                expected = (
                    303
                    if args.repeat_identical
                    else int(expected_tokens[step])
                )
                report = {
                    "step": step,
                    "metadata_step": metadata_step,
                    "input_token": token,
                    "output_token": sampled,
                    "expected_token": expected,
                    "token_exact": sampled == expected,
                    "run_sec": elapsed,
                    "hidden_shape": list(hidden.shape),
                    "hidden_finite": True,
                    "active_batch": args.active_batch,
                    "active_hidden_finite": active_finite,
                    "active_hidden_nonzero_rank_rows": active_nonzero_rows,
                    "hidden_tp_spread": tp_spread,
                    "hidden_row0_abs_max": float(row0.abs().max().item()),
                }
                if args.repeat_identical:
                    if repeat_hidden is None:
                        repeat_hidden = hidden_snapshot
                    else:
                        repeat_diff = (
                            hidden_snapshot.float() - repeat_hidden.float()
                        ).abs()
                        repeat_bad = repeat_diff != 0
                        report.update(
                            {
                                "repeat_hidden_exact": bool(
                                    torch.equal(hidden_snapshot, repeat_hidden)
                                ),
                                "repeat_hidden_max_abs": float(
                                    repeat_diff.max().item()
                                ),
                                "repeat_hidden_bad_ratio": float(
                                    repeat_bad.float().mean().item()
                                ),
                            }
                        )
                if kv_probe is not None:
                    report["kv_probe_path"] = str(
                        out / f"kv_probe_step{step}.json"
                    )
                    probe_summary = _kv_probe_summary(kv_probe)
                    report["kv_probe_summary"] = probe_summary
                    if not probe_summary["slot0_all_nonzero"]:
                        raise AssertionError(
                            f"Main step {step}: selected layer/rank slot0 rows "
                            "were not all written"
                        )
                    if (
                        not args.repeat_identical
                        and step == 0
                        and (
                            probe_summary["slot1_any_nonzero"]
                            or probe_summary["slot2_any_nonzero"]
                        )
                    ):
                        raise AssertionError(
                            "Main step 0 unexpectedly wrote slot1/slot2 history"
                        )
                    if (
                        not args.repeat_identical
                        and step >= 1
                        and not probe_summary["slot1_all_nonzero"]
                    ):
                        raise AssertionError(
                            f"Main step {step}: selected layer/rank slot1 rows "
                            "were not all written"
                        )
                    if (
                        not args.repeat_identical
                        and step >= 1
                        and probe_summary["slot2_any_nonzero"]
                    ):
                        raise AssertionError(
                            f"Main step {step}: untouched slot2 history changed"
                        )
                    if (
                        not args.repeat_identical
                        and step == 1
                        and previous_kv_probe_summary is not None
                        and probe_summary["slot0_hashes"]
                        != previous_kv_probe_summary["slot0_hashes"]
                    ):
                        raise AssertionError(
                            "Main step 1 modified slot0 history from step 0"
                        )
                    previous_kv_probe_summary = probe_summary
                reports.append(report)
                print(json.dumps(report, sort_keys=True), flush=True)
                if sampled != expected and not getattr(args, "teacher_forced", False):
                    raise AssertionError(report)
                if (
                    args.repeat_identical
                    and step > 0
                    and not report["repeat_hidden_exact"]
                ):
                    raise AssertionError(report)
                if not args.repeat_identical:
                    token = expected if getattr(args, "teacher_forced", False) else sampled
    finally:
        if not args.reuse_exporters:
            _stop_exporters(out, procs)

    report = {
        "ok": True,
        "device_ids": devices,
        "checkpoint": args.ckpt,
        "steps": reports,
        "ownership": {
            "pypto_output": "pre-final-norm BF16 next_hidden only",
            "vllm_tail": "final RMSNorm + LM head + greedy sampler",
            "kv_metadata": "standalone direct-decode diagnostic",
            "diagnostic_mode": (
                "identical resident invocation A/B"
                if args.repeat_identical
                else "canonical autoregressive chain"
            ),
        },
    }
    (out / "main_hidden_only_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    # Canonical liveness marker consumed by the performance release runbook.
    # Print it only after the resident holder has completed every requested
    # invocation and the report has been durably written.
    print("[worker] RUN done", flush=True)
    print(
        "RESULT="
        + (
            "MAIN_HIDDEN_ONLY_IDENTICAL_REPEAT_EXACT"
            if args.repeat_identical
            else ("MAIN_HIDDEN_ONLY_TEACHER_FORCED_MATCH_%d_of_%d" % (
                sum(1 for r in reports if r.get("token_exact")), len(reports))
                if getattr(args, "teacher_forced", False)
                else "MAIN_HIDDEN_ONLY_8STEP_TOKEN_EXACT")
        ),
        flush=True,
    )
    return 0


def main() -> int:
    args = _parse_args()
    if args.export_rank >= 0:
        return _export_rank(args)
    return _run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
