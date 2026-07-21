#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""0162 standalone selected MTP45/46/47 hidden-only device gate.

This harness deliberately keeps the production ownership boundary visible:

``PyPTO selected MTP body -> raw BF16 hidden -> CPU tail reference -> token``

The CPU tail reference is only a diagnostic stand-in for vLLM's shared-head
norm/LM-head/sampler.  PyPTO never receives or returns logits, draft tokens,
or acceptance state.  The test feeds the resulting token to the next selected
layer and compares only raw hidden output against the pinned offline oracle.

It uses two independent IPC ownership domains per rank:

* ``pypto_weight``: embedding + selected MTP transformer-body weights;
* ``mtp_kv``: K(MTP45..47), aligned gap, V(MTP45..47).

Both owners stay alive until the holder exits.  Strict session mode is enabled
by the launcher so stale keys fail before ACL import.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


TP = 8
BATCH = 16
HIDDEN = 4096
BLOCK_SIZE = 128
DEFAULT_CKPT = (
    "/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp"
)
DEFAULT_ORACLE = (
    "/data/chensiyu/hw_project/pypto/workspace/logs_n1/"
    "live_mtp3_patch_ci4_inline_runtime_20260718_220645"
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="8,9,10,11,12,13,14,15")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--out", required=True)
    parser.add_argument("--previous-hidden", default="")
    parser.add_argument("--oracle-dir", default=DEFAULT_ORACLE)
    parser.add_argument("--export-rank", type=int, default=-1)
    parser.add_argument("--dev", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=32)
    parser.add_argument("--active-batch", type=int, default=16)
    parser.add_argument("--reuse-exporters", action="store_true")
    parser.add_argument("--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    return parser.parse_args()


def _devices(text: str) -> list[int]:
    result = [int(item) for item in str(text).split(",") if item.strip()]
    if len(result) != TP or len(set(result)) != TP:
        raise ValueError(f"expected {TP} distinct devices, got {result}")
    return result


def _ready_paths(out_dir: str, rank: int) -> tuple[Path, Path]:
    root = Path(out_dir)
    return (
        root / f"pypto_weight_map.rank{rank}.json.done",
        root / f"pypto_mtp_kvpool_map.json.rank{rank}.done",
    )


def _export_rank(args: argparse.Namespace) -> int:
    if not 0 <= args.export_rank < TP:
        raise ValueError(f"export rank must be 0..{TP - 1}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    from tools.step3p5.mtp_kv_exporter import MtpKvExporter
    from tools.step3p5.pypto_weight_ipc import (
        export_mtp_hidden_weights_from_checkpoint,
    )

    weight_owner, weight_summary, _ = (
        export_mtp_hidden_weights_from_checkpoint(
            args.ckpt,
            rank=args.export_rank,
            tp_world_size=TP,
            out_dir=str(out_dir),
            dev=args.dev,
        )
    )
    mtp_exporter = MtpKvExporter(args.dev)
    mtp_summary = mtp_exporter.export(
        out_dir=str(out_dir),
        rank=args.export_rank,
        tp_world_size=TP,
        num_blocks=args.num_blocks,
    )
    (out_dir / f"ready.rank{args.export_rank}").write_text(
        json.dumps(
            {
                "rank": args.export_rank,
                "weight": weight_summary,
                "mtp_kv": mtp_summary,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        f"[mtp-hidden-export rank={args.export_rank}] ready "
        f"weight_bytes={weight_summary['pool_bytes']} "
        f"mtp_kv_bytes={mtp_summary['pool_bytes']} dev={args.dev}",
        flush=True,
    )
    try:
        stop = out_dir / "STOP"
        while not stop.exists():
            time.sleep(2.0)
    finally:
        # Graceful close: ACL IPC handles and heartbeat owner are released
        # only after the holder has stopped using the imported pools.
        mtp_exporter.teardown()
        weight_owner.teardown()
    print(
        f"[mtp-hidden-export rank={args.export_rank}] STOP seen; exit",
        flush=True,
    )
    return 0


def _stop_exporters(out_dir: str, procs: list[subprocess.Popen]) -> None:
    try:
        Path(out_dir, "STOP").write_text("1", encoding="utf-8")
    except OSError:
        pass
    for proc in procs:
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=30)
        log_handle = getattr(proc, "_mtp_log_handle", None)
        if log_handle is not None:
            log_handle.close()


def _start_exporters(args: argparse.Namespace, devices: list[int]) -> list[subprocess.Popen]:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.glob("ready.rank*"):
        path.unlink(missing_ok=True)
    for path in out_dir.glob("*.done"):
        path.unlink(missing_ok=True)
    for path in out_dir.glob("pypto_*"):
        if path.is_file():
            path.unlink(missing_ok=True)
    (out_dir / "STOP").unlink(missing_ok=True)
    procs: list[subprocess.Popen] = []
    root = Path(__file__).resolve().parents[3]
    for rank, dev in enumerate(devices):
        log_handle = open(
            out_dir / f"export_rank{rank}.log",
            "w",
            encoding="utf-8",
        )
        proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "tests.step3p5.harnesses._stage_mtp_hidden_selected",
                    "--export-rank",
                    str(rank),
                    "--dev",
                    str(dev),
                    "--out",
                    str(out_dir),
                    "--ckpt",
                    args.ckpt,
                    "--num-blocks",
                    str(args.num_blocks),
                ],
                cwd=str(root),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        setattr(proc, "_mtp_log_handle", log_handle)
        procs.append(proc)
    deadline = time.time() + 2400.0
    while time.time() < deadline:
        if all(
            all(path.exists() for path in _ready_paths(str(out_dir), rank))
            for rank in range(TP)
        ):
            return procs
        if any(proc.poll() not in (None, 0) for proc in procs):
            _stop_exporters(str(out_dir), procs)
            raise RuntimeError("MTP hidden exporter exited before readiness")
        time.sleep(3.0)
    _stop_exporters(str(out_dir), procs)
    raise TimeoutError("MTP hidden exporters were not ready within 40 minutes")


def _load_previous(path: str, oracle_dir: str) -> torch.Tensor:
    if not path:
        path = str(Path(oracle_dir) / "dumps" / "P42_nh_row0.pt")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(value.shape) == (TP, HIDDEN):
        value = value[0]
    if tuple(value.shape) != (HIDDEN,):
        raise ValueError(f"previous hidden must be [{HIDDEN}], got {tuple(value.shape)}")
    return value.to(torch.bfloat16).contiguous()


def _load_oracle(oracle_dir: str) -> torch.Tensor:
    value = torch.load(
        Path(oracle_dir) / "dumps" / "single" / "mtp3_hidden.pt",
        map_location="cpu",
        weights_only=True,
    )
    if tuple(value.shape) != (TP, 3, BATCH, HIDDEN):
        raise ValueError(f"unexpected oracle hidden shape {tuple(value.shape)}")
    return value


def _cpu_tail_token(
    hidden: torch.Tensor,
    *,
    ckpt: str,
    layer_idx: int,
) -> int:
    """Diagnostic vLLM-tail equivalent: shared-head norm + LM head + argmax."""
    from models.step3p5.config import EPS, LM_HEAD_K_CHUNK, VOCAB_LOCAL
    from models.step3p5.weight_loader import (
        _ShardCache,
        _hf_mtp_keys,
        _read_index,
        _slice_lm_head,
    )

    from tools.step3p5.pypto_mtp3_ctx1_reference import (
        _chunked_mm,
        _zc_rmsnorm,
    )

    hidden = hidden.reshape(1, HIDDEN).to(torch.bfloat16)
    weight_map = _read_index(ckpt)
    keys = _hf_mtp_keys(45 + int(layer_idx))
    with _ShardCache(ckpt, weight_map) as cache:
        norm = _zc_rmsnorm(
            hidden,
            cache.get(keys["shared_head_norm"]),
            EPS,
        )
        full = cache.get(keys["shared_head_output"])
        shards = []
        for rank in range(TP):
            local = _slice_lm_head(full, rank, int(VOCAB_LOCAL))
            shards.append(
                _chunked_mm(
                    norm,
                    local.transpose(0, 1),
                    k_chunk=int(LM_HEAD_K_CHUNK),
                )
            )
        logits = torch.cat(shards, dim=-1)
        if tuple(logits.shape) != (1, int(VOCAB_LOCAL) * TP):
            raise RuntimeError(
                f"shared head tail produced unexpected logits {tuple(logits.shape)}"
            )
        return int(logits.argmax(dim=-1).item())


def _run_worker(args: argparse.Namespace) -> int:
    devices = _devices(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_exporters:
        procs: list[subprocess.Popen] = []
        for rank in range(TP):
            if not all(
                path.exists() for path in _ready_paths(str(out_dir), rank)
            ):
                raise RuntimeError(f"missing reused exporter for rank {rank}")
    else:
        procs = _start_exporters(args, devices)

    os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    physical_num_blocks = args.num_blocks + 15
    os.environ.setdefault(
        "PYPTO_STEP3P5_MTP_KV_CACHE_ROWS",
        str(physical_num_blocks * BLOCK_SIZE),
    )
    os.environ.setdefault("PYPTO_STEP3P5_MAX_SEQ", str(args.num_blocks * BLOCK_SIZE))
    os.environ.setdefault(
        "PYPTO_STEP3P5_BLOCK_TABLE_FLAT",
        str(BATCH * args.num_blocks),
    )
    os.environ.setdefault(
        "PYPTO_STEP3P5_ROPE_SEQ",
        str(args.num_blocks * BLOCK_SIZE),
    )
    os.environ.setdefault(
        "PYPTO_PROG_BUILD_DIR",
        str(out_dir / "build_output"),
    )

    from tools.step3p5.mtp_layer_holder import MtpLayerHolder
    from tools.step3p5.kv_padding import (
        make_diagnostic_fixed_batch_metadata,
        make_padding_reserve,
    )

    if not 1 <= args.active_batch <= BATCH:
        raise ValueError(f"active-batch must be 1..{BATCH}")
    if args.active_batch > args.num_blocks:
        raise ValueError(
            f"active-batch={args.active_batch} needs scheduler blocks >= "
            f"{args.active_batch}, got {args.num_blocks}"
        )

    previous = _load_previous(args.previous_hidden, args.oracle_dir)
    oracle = _load_oracle(args.oracle_dir)
    reserve = make_padding_reserve(args.num_blocks, args.num_blocks + 15)
    seq_lens, positions, block_table, slot_mapping = (
        make_diagnostic_fixed_batch_metadata(
            valid_rows=args.active_batch,
            reserve=reserve,
            max_blocks_per_row=args.num_blocks,
        )
    )

    holder = MtpLayerHolder(
        device_ids=devices,
        out_dir=str(out_dir),
        mtp_kv_dir=str(out_dir),
        ckpt=args.ckpt,
        platform=args.platform,
    ).build()
    try:
        with holder:
            current = previous.unsqueeze(0).expand(
                args.active_batch, -1
            ).contiguous()
            token = 303
            reports: list[dict[str, object]] = []
            for layer_idx in range(3):
                token_ids = torch.zeros(BATCH, dtype=torch.int32)
                token_ids[: args.active_batch] = token
                active = torch.zeros(BATCH, dtype=torch.int32)
                active[: args.active_batch] = 1
                holder.set_live_step(
                    current,
                    input_token_ids=token_ids,
                    active_mask=active,
                    seq_lens=seq_lens,
                    positions=positions,
                    block_table=block_table,
                    slot_mapping=slot_mapping,
                )
                started = time.time()
                result = holder.run(layer_idx)
                elapsed = time.time() - started
                hidden = result["mtp_hidden"]
                if tuple(hidden.shape) != (args.active_batch, HIDDEN):
                    raise AssertionError(
                        f"MTP selected layer returned {tuple(hidden.shape)}, "
                        f"expected {(args.active_batch, HIDDEN)}"
                    )
                expected = oracle[0, layer_idx, 0].expand(
                    args.active_batch, -1
                )
                diff = (hidden.float() - expected.float()).abs()
                pass_rate = float(
                    torch.isclose(
                        hidden.float(),
                        expected.float(),
                        rtol=8e-2,
                        atol=2e-1,
                    ).float().mean().item()
                )
                spread = float(
                    (
                        holder.hidden_out.float()
                        - holder.hidden_out[0:1].float()
                    ).abs().max().item()
                )
                next_token = _cpu_tail_token(
                    hidden[0],
                    ckpt=args.ckpt,
                    layer_idx=layer_idx,
                )
                reports.append(
                    {
                        "layer_idx": layer_idx,
                        "absolute_layer": 45 + layer_idx,
                        "input_token": token,
                        "output_token": next_token,
                        "expected_token": [6178, 410, 303][layer_idx],
                        "hidden_pass_rate": pass_rate,
                        "hidden_max_abs_diff": float(diff.max().item()),
                        "hidden_tp_spread": spread,
                        "run_sec": elapsed,
                    }
                )
                if pass_rate < 0.97 or spread != 0.0:
                    raise AssertionError(reports[-1])
                if next_token != [6178, 410, 303][layer_idx]:
                    raise AssertionError(reports[-1])
                current = hidden
                token = next_token
            report = {
                "ok": True,
                "device_ids": devices,
                "active_batch": args.active_batch,
                "checkpoint": args.ckpt,
                "oracle_dir": args.oracle_dir,
                "build_output_dirs": [
                    str(item.output_dir) for item in holder.compiled
                ],
                "reports": reports,
                "ownership": {
                    "pypto_output": "raw BF16 mtp_hidden only",
                    "tail_output": "CPU diagnostic shared-head argmax",
                    "acceptance": "not executed by PyPTO",
                },
            }
            (out_dir / "selected_mtp_hidden_report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
            print("RESULT=MTP_SELECTED_HIDDEN_DEVICE_PASS", flush=True)
    finally:
        if not args.reuse_exporters:
            _stop_exporters(str(out_dir), procs)
    return 0


def main() -> int:
    args = _args()
    if args.export_rank >= 0:
        return _export_rank(args)
    return _run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
