# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Device harness for the standalone Step3p5 ``whole_mtp3`` program.

The target network and MTP drafter are separated by the target sampler.  This
harness consumes a canonical main-network ``next_hidden`` dump plus the sampled
target token, then runs MTP layers 45/46/47 against the same fresh native-W8A8
IPC pool used by ``_stage_whole_faithful_real_ipc``.

Canonical 0162 flow::

    # 1. Launch fresh exporters with the updated IPC exporter (one per card).
    # 2. Run N1-CANONICAL-TEST.md P42 with N1_DUMP_DIR set.
    # 3. Reuse the still-live exporters here:
    python -m tests.step3p5.harnesses._stage_whole_mtp3_ipc \
      --device 8,9,10,11,12,13,14,15 \
      --reuse-exporters \
      --out /tmp/n1_weight_ipc_mtp3 \
      --previous-hidden /tmp/n1_mtp3_chain/P42_nh_row0.pt \
      --first-token 303

All host tensors are allocated in shared memory before ``prepare()``.  MTP
weights and three distinct MTP KV slices are zero-copy ``DeviceTensor`` views
of the per-rank IPC pools.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


CKPT_DEFAULT = (
    "/data/chensiyu/"
    "step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp"
)
TP = 8


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--platform",
        default="a2a3",
        choices=["a2a3", "a2a3sim"],
    )
    parser.add_argument(
        "-d",
        "--device",
        default="8,9,10,11,12,13,14,15",
    )
    parser.add_argument("--ckpt", default=CKPT_DEFAULT)
    parser.add_argument("--out", default="/tmp/n1_weight_ipc_mtp3")
    parser.add_argument("--export-rank", type=int, default=-1)
    parser.add_argument("--dev", type=int, default=8)
    parser.add_argument("--reuse-exporters", action="store_true")
    parser.add_argument(
        "--previous-hidden",
        default="",
        help="P42_nh_row0.pt from the canonical main-network harness",
    )
    parser.add_argument("--first-token", type=int, default=303)
    parser.add_argument(
        "--active-batch",
        type=int,
        default=1,
        help="1..16; identical canonical input is replicated to active rows",
    )
    parser.add_argument(
        "--dump-dir",
        default="",
        help="optional output directory for hidden/logits/tokens",
    )
    return parser.parse_args()


def _assert_pool_contract(out_dir: str, rank: int) -> None:
    """Validate alignment/dtype invariants directly from an exporter map."""
    from models.step3p5 import weight_loader as keys  # noqa: PLC0415

    map_path = os.path.join(
        out_dir, f"pypto_weight_map.rank{rank}.json"
    )
    with open(map_path, encoding="utf-8") as stream:
        pool_map = json.load(stream)
    entries = pool_map["map"]

    for key, entry in entries.items():
        offset = int(entry["offset"])
        if offset % 512 != 0:
            raise AssertionError(
                f"rank{rank} key={key} offset={offset} is not 512B aligned"
            )

    # The main network in this same pool must remain native W8A8.
    for key in (
        keys.KEY_MOE_W_GATE_R,
        keys.KEY_MOE_W_UP_R,
        keys.KEY_MOE_W_DOWN_R,
    ):
        if entries[key]["dtype"] != "int8":
            raise AssertionError(
                f"rank{rank} {key} must be native int8, "
                f"got {entries[key]['dtype']}"
            )
    for key in (
        keys.KEY_MOE_W_GATE_R_SCALE,
        keys.KEY_MOE_W_UP_R_SCALE,
        keys.KEY_MOE_W_DOWN_R_SCALE,
    ):
        if entries[key]["dtype"] != "float32":
            raise AssertionError(
                f"rank{rank} {key} must be float32, "
                f"got {entries[key]['dtype']}"
            )

    mtp_fp32 = (
        keys.KEY_MTP_ENORM,
        keys.KEY_MTP_HNORM,
        keys.KEY_MTP_INPUT_RMS,
        keys.KEY_MTP_POST_ATTN_RMS,
        keys.KEY_MTP_Q_NORM,
        keys.KEY_MTP_K_NORM,
        keys.KEY_MTP_SH_NORM,
    )
    for key in mtp_fp32:
        if entries[key]["dtype"] != "float32":
            raise AssertionError(
                f"rank{rank} {key} ABI dtype must be float32, "
                f"got {entries[key]['dtype']}"
            )

    mtp_bf16 = (
        keys.KEY_EMBED,
        keys.KEY_MTP_EH_PROJ,
        keys.KEY_MTP_WQ,
        keys.KEY_MTP_WK,
        keys.KEY_MTP_WV,
        keys.KEY_MTP_WO,
        keys.KEY_MTP_WG,
        keys.KEY_MTP_DENSE_GATE,
        keys.KEY_MTP_DENSE_UP,
        keys.KEY_MTP_DENSE_DOWN,
        keys.KEY_MTP_SH_OUT,
        "mtp_k_cache",
        "mtp_v_cache",
    )
    for key in mtp_bf16:
        if entries[key]["dtype"] != "bfloat16":
            raise AssertionError(
                f"rank{rank} {key} must be checkpoint-native bfloat16, "
                f"got {entries[key]['dtype']}"
            )


def _do_export(args: argparse.Namespace) -> int:
    """Load and hold one rank's fresh native-W8A8 + MTP IPC pool."""
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from tools.step3p5.pypto_weight_ipc import (  # noqa: PLC0415
        export_from_checkpoint,
    )

    rank = args.export_rank
    os.makedirs(args.out, exist_ok=True)
    export_from_checkpoint(
        args.ckpt,
        rank=rank,
        tp_world_size=TP,
        out_dir=args.out,
        dev=args.dev,
        int8_routed=True,
        kv_ipc=True,
    )
    _assert_pool_contract(args.out, rank)
    Path(args.out, f"ready.rank{rank}").write_text(
        "1", encoding="utf-8"
    )
    print(
        f"[export-rank {rank}] pool contract OK; holding on dev {args.dev}",
        flush=True,
    )
    stop = Path(args.out, "STOP")
    while not stop.exists():
        time.sleep(2)
    print(f"[export-rank {rank}] STOP seen; exit", flush=True)
    return 0


def _stop(out_dir: str, procs: list[subprocess.Popen]) -> None:
    try:
        Path(out_dir, "STOP").write_text("1", encoding="utf-8")
    except OSError:
        pass
    for proc in procs:
        try:
            proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            proc.terminate()


def _wait_for_exporters(
    args: argparse.Namespace,
    repo_root: Path,
    device_ids: list[int],
) -> list[subprocess.Popen]:
    os.makedirs(args.out, exist_ok=True)
    procs: list[subprocess.Popen] = []
    if args.reuse_exporters:
        for rank in range(TP):
            ready = Path(args.out, f"ready.rank{rank}")
            if not ready.exists():
                raise RuntimeError(
                    f"reuse-exporters requires an active exporter: {ready}"
                )
            _assert_pool_contract(args.out, rank)
        print("[worker] reused exporter maps satisfy IPC contract", flush=True)
        return procs

    for entry in os.listdir(args.out):
        if entry.startswith(("ready.rank", "pypto_weight")) or entry == "STOP":
            try:
                os.remove(os.path.join(args.out, entry))
            except OSError:
                pass

    print(f"[worker] launching {TP} fresh exporters", flush=True)
    for rank, dev in enumerate(device_ids):
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "tests.step3p5.harnesses._stage_whole_mtp3_ipc",
                    "--export-rank",
                    str(rank),
                    "--dev",
                    str(dev),
                    "--out",
                    args.out,
                    "--ckpt",
                    args.ckpt,
                ],
                cwd=str(repo_root),
            )
        )

    deadline = time.time() + 2400
    while time.time() < deadline:
        if all(
            Path(args.out, f"ready.rank{rank}").exists()
            for rank in range(TP)
        ):
            print("[worker] all fresh exporters ready", flush=True)
            return procs
        if any(proc.poll() not in (None, 0) for proc in procs):
            _stop(args.out, procs)
            raise RuntimeError("an exporter exited before readiness")
        time.sleep(3)
    _stop(args.out, procs)
    raise RuntimeError("exporters were not ready within 40 minutes")


def _shared_zeros(*shape: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(shape, dtype=dtype).share_memory_()


def _shared_full(
    *shape: int, dtype: torch.dtype, value: float | int
) -> torch.Tensor:
    return torch.full(shape, value, dtype=dtype).share_memory_()


def _load_previous_hidden(
    path: str,
    *,
    tp: int,
    batch: int,
    hidden: int,
    active_batch: int,
) -> torch.Tensor:
    if not path:
        raise ValueError(
            "--previous-hidden is required; use the canonical "
            "P42_nh_row0.pt rather than a random/synthetic hidden"
        )
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(loaded.shape) == (hidden,):
        loaded = loaded.unsqueeze(0).repeat(tp, 1)
    if tuple(loaded.shape) != (tp, hidden):
        raise ValueError(
            f"previous hidden must have shape {(tp, hidden)} or {(hidden,)}, "
            f"got {tuple(loaded.shape)}"
        )
    previous = _shared_zeros(tp, batch, hidden, dtype=torch.bfloat16)
    source = loaded.to(torch.bfloat16)
    for row in range(active_batch):
        previous[:, row, :] = source
    spread = (
        source.float() - source[0:1].float()
    ).abs().max().item()
    if spread != 0.0:
        raise ValueError(
            "previous hidden must be replicated across TP ranks at the "
            f"main-to-MTP boundary; max spread={spread}"
        )
    print(
        f"[worker] previous_hidden={path} "
        f"max|x|={source.float().abs().max().item():.6f} "
        f"tp_spread={spread:.6f}",
        flush=True,
    )
    return previous


def _do_worker(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    device_ids = [int(dev) for dev in args.device.split(",")]
    if len(device_ids) != TP:
        raise ValueError(f"whole_mtp3 requires {TP} devices, got {device_ids}")
    if len(set(device_ids)) != TP:
        raise ValueError(f"device IDs must be distinct: {device_ids}")
    if not 1 <= args.active_batch <= 16:
        raise ValueError("--active-batch must be in [1,16]")
    if not 0 <= args.first_token < 128896:
        raise ValueError("--first-token is outside the Step3p5 vocabulary")

    # Do not inherit a front-8 vLLM visibility mask into the physical-device
    # PyPTO process. The shell isolation script also clears all vLLM knobs.
    os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)

    procs = _wait_for_exporters(args, repo_root, device_ids)
    try:
        from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415

        set_backend_type(BackendType.Ascend910B)
        from pypto import ir  # noqa: PLC0415
        from pypto.ir.distributed_compiled_program import (  # noqa: PLC0415
            DistributedConfig,
        )
        from pypto.runtime.device_tensor import (  # noqa: PLC0415
            DeviceTensor,
            StackedDeviceTensor,
        )

        import models.step3p5.config as cfg  # noqa: PLC0415
        from models.step3p5 import weight_loader as keys  # noqa: PLC0415
        from models.step3p5._ops import (  # noqa: PLC0415
            build_plain_rope_tables,
        )
        from models.step3p5.mtp_fwd import whole_mtp3  # noqa: PLC0415
        from tools.step3p5.pypto_weight_ipc import (  # noqa: PLC0415
            build_stacked_weight,
            import_weights_all,
        )

        os.environ["PYPTO_PROG_BUILD_DIR"] = (
            "/data/chensiyu/hw_project/pypto/workspace/build_output"
        )
        print("[worker] compiling whole_mtp3", flush=True)
        compiled = ir.compile(
            whole_mtp3,
            platform=args.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids,
                num_sub_workers=0,
            ),
            skip_ptoas=False,
            dump_passes=False,
        )
        print(f"[worker] compile OK => {compiled.output_dir}", flush=True)

        batch = int(cfg.BATCH)
        hidden = int(cfg.HIDDEN)
        vocab_local = int(cfg.VOCAB_LOCAL)
        num_mtp = int(cfg.NUM_NEXTN_PREDICT_LAYERS)
        user_batch = int(cfg.USER_BATCH_DYN)
        block_table_flat = int(cfg.BLOCK_TABLE_FLAT_DYN)
        rope_seq = int(cfg.ROPE_SEQ_DYN)
        head_dim = int(cfg.HEAD_DIM)
        cache_rows = int(cfg.KV_CACHE_ROWS_DYN)

        previous_hidden = _load_previous_hidden(
            args.previous_hidden,
            tp=TP,
            batch=batch,
            hidden=hidden,
            active_batch=args.active_batch,
        )
        first_token_ids = _shared_zeros(TP, batch, dtype=torch.int32)
        first_token_ids[:, : args.active_batch] = args.first_token
        active_mask = _shared_zeros(TP, batch, dtype=torch.int32)
        active_mask[:, : args.active_batch] = 1

        # Attention executes the fixed 16 rows, including padding. Every row
        # therefore owns a conflict-free block/slot even if active_mask is 0.
        seq_lens = torch.ones(
            TP, user_batch, dtype=torch.int32
        ).share_memory_()
        max_blocks_per_seq = block_table_flat // user_batch
        block_table = _shared_zeros(
            TP, block_table_flat, dtype=torch.int32
        )
        slot_mapping = _shared_zeros(TP, user_batch, dtype=torch.int32)
        for row in range(user_batch):
            block_table[:, row * max_blocks_per_seq] = row
            slot_mapping[:, row] = row * int(cfg.BLOCK_SIZE)

        rope_cos_one, rope_sin_one = build_plain_rope_tables(
            rope_seq, head_dim, 10_000.0
        )
        rope_cos = (
            rope_cos_one.unsqueeze(0)
            .repeat(TP, 1, 1)
            .contiguous()
            .share_memory_()
        )
        rope_sin = (
            rope_sin_one.unsqueeze(0)
            .repeat(TP, 1, 1)
            .contiguous()
            .share_memory_()
        )

        gate_r = _shared_zeros(
            TP,
            int(cfg.NUM_HEADS_SWA_LOCAL_PAD),
            int(cfg.HIDDEN_Q_SWA_LOCAL),
            dtype=torch.bfloat16,
        )
        for head in range(int(cfg.NUM_HEADS_SWA_LOCAL)):
            start = head * head_dim
            gate_r[:, head, start : start + head_dim] = 1.0

        # Sentinels make incomplete writes visible. The fixed program must
        # overwrite all 16 rows, including padding rows.
        hidden_out = _shared_full(
            TP,
            num_mtp,
            batch,
            hidden,
            dtype=torch.bfloat16,
            value=float("nan"),
        )
        logits_out = _shared_full(
            TP,
            num_mtp,
            user_batch,
            vocab_local,
            dtype=torch.float32,
            value=float("nan"),
        )
        draft_token_ids_out = _shared_full(
            TP,
            num_mtp,
            batch,
            dtype=torch.int32,
            value=-1,
        )

        with compiled.prepare() as runtime:
            weight_maps = import_weights_all(
                runtime,
                args.out,
                tp=TP,
                dev_offset=device_ids[0],
            )

            def weight(key: str):
                return build_stacked_weight(weight_maps, key)

            def flat_weight(key: str, shape: tuple[int, ...]):
                shards = []
                expected_numel = math.prod(shape)
                for rank in range(TP):
                    source = weight_maps[rank].device_tensor(key)
                    if math.prod(source.shape) != expected_numel:
                        raise ValueError(
                            f"{key} rank{rank}: cannot reshape "
                            f"{source.shape} to {shape}"
                        )
                    shards.append(
                        DeviceTensor(
                            source.data_ptr,
                            shape,
                            source.dtype,
                        )
                    )
                return StackedDeviceTensor(
                    shards,
                    (TP, *shape),
                    list(range(TP)),
                )

            mtp_k_cache = weight("mtp_k_cache")
            mtp_v_cache = weight("mtp_v_cache")
            if tuple(mtp_k_cache.full_shape) != (
                TP,
                num_mtp * cache_rows,
                head_dim,
            ):
                raise AssertionError(
                    f"unexpected MTP K cache shape: {mtp_k_cache.full_shape}"
                )

            arguments = [
                previous_hidden,
                first_token_ids,
                active_mask,
                weight(keys.KEY_EMBED),
                weight(keys.KEY_MTP_ENORM),
                weight(keys.KEY_MTP_HNORM),
                flat_weight(
                    keys.KEY_MTP_EH_PROJ,
                    (num_mtp * (hidden // TP), 2 * hidden),
                ),
                weight(keys.KEY_MTP_INPUT_RMS),
                flat_weight(
                    keys.KEY_MTP_WQ,
                    (num_mtp * hidden, int(cfg.HIDDEN_Q_SWA_LOCAL)),
                ),
                flat_weight(
                    keys.KEY_MTP_WK,
                    (num_mtp * hidden, int(cfg.KV_HIDDEN_LOCAL)),
                ),
                flat_weight(
                    keys.KEY_MTP_WV,
                    (num_mtp * hidden, int(cfg.KV_HIDDEN_LOCAL)),
                ),
                weight(keys.KEY_MTP_Q_NORM),
                weight(keys.KEY_MTP_K_NORM),
                flat_weight(
                    keys.KEY_MTP_WO,
                    (
                        num_mtp * int(cfg.HIDDEN_Q_SWA_LOCAL),
                        hidden,
                    ),
                ),
                flat_weight(
                    keys.KEY_MTP_WG,
                    (
                        num_mtp * hidden,
                        int(cfg.NUM_HEADS_SWA_LOCAL_PAD),
                    ),
                ),
                gate_r,
                weight(keys.KEY_MTP_POST_ATTN_RMS),
                flat_weight(
                    keys.KEY_MTP_DENSE_GATE,
                    (
                        num_mtp * hidden,
                        int(cfg.INTERMEDIATE_LOCAL),
                    ),
                ),
                flat_weight(
                    keys.KEY_MTP_DENSE_UP,
                    (
                        num_mtp * hidden,
                        int(cfg.INTERMEDIATE_LOCAL),
                    ),
                ),
                flat_weight(
                    keys.KEY_MTP_DENSE_DOWN,
                    (
                        num_mtp * int(cfg.INTERMEDIATE_LOCAL),
                        hidden,
                    ),
                ),
                weight(keys.KEY_MTP_SH_NORM),
                flat_weight(
                    keys.KEY_MTP_SH_OUT,
                    (num_mtp * vocab_local, hidden),
                ),
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos,
                rope_sin,
                mtp_k_cache,
                mtp_v_cache,
                hidden_out,
                logits_out,
                draft_token_ids_out,
            ]

            print(
                f"[worker] running whole_mtp3 args={len(arguments)} "
                f"active_batch={args.active_batch} first_token={args.first_token}",
                flush=True,
            )
            start = time.time()
            runtime.run(compiled, *arguments)
            elapsed = time.time() - start

        if not torch.isfinite(hidden_out.float()).all():
            raise AssertionError("hidden_out contains NaN/Inf or unwritten rows")
        if not torch.isfinite(logits_out).all():
            raise AssertionError("logits_out contains NaN/Inf or unwritten rows")
        if (draft_token_ids_out < 0).any():
            raise AssertionError("draft_token_ids_out contains unwritten -1")

        active_tokens: list[list[int]] = []
        for layer in range(num_mtp):
            layer_tokens: list[int] = []
            for row in range(args.active_batch):
                rank_tokens = draft_token_ids_out[:, layer, row]
                if not torch.equal(
                    rank_tokens,
                    rank_tokens[0].expand_as(rank_tokens),
                ):
                    raise AssertionError(
                        f"MTP layer {layer} row {row} token differs by TP rank: "
                        f"{rank_tokens.tolist()}"
                    )
                full_logits = torch.cat(
                    [
                        logits_out[rank, layer, row]
                        for rank in range(TP)
                    ],
                    dim=0,
                )
                logits_argmax = int(full_logits.argmax())
                program_token = int(rank_tokens[0])
                if logits_argmax != program_token:
                    raise AssertionError(
                        f"MTP layer {layer} row {row}: "
                        f"logits argmax={logits_argmax} "
                        f"program token={program_token}"
                    )
                layer_tokens.append(program_token)
                if row == 0:
                    top = torch.topk(full_logits, 5)
                    print(
                        f"[worker] MTP{45 + layer} row0 token={program_token} "
                        f"TOP5={top.indices.tolist()} "
                        f"vals={[round(float(v), 4) for v in top.values]}",
                        flush=True,
                    )
            active_tokens.append(layer_tokens)

        if args.active_batch < batch:
            inactive_hidden_max = (
                hidden_out[:, :, args.active_batch :, :]
                .float()
                .abs()
                .max()
                .item()
            )
            inactive_logits_max = (
                logits_out[:, :, args.active_batch :, :]
                .abs()
                .max()
                .item()
            )
            if inactive_hidden_max != 0.0:
                raise AssertionError(
                    f"inactive hidden rows were not zero: {inactive_hidden_max}"
                )
            if inactive_logits_max != 0.0:
                raise AssertionError(
                    f"inactive logits rows were not zero: {inactive_logits_max}"
                )
            print(
                "[worker] padding rows initialized: "
                "max|hidden|=0 max|logits|=0",
                flush=True,
            )

        for layer in range(num_mtp):
            hidden_spread = (
                hidden_out[:, layer].float()
                - hidden_out[0:1, layer].float()
            ).abs().max().item()
            if hidden_spread != 0.0:
                raise AssertionError(
                    f"MTP layer {layer} hidden differs across TP ranks: "
                    f"{hidden_spread}"
                )

        if args.dump_dir:
            dump_dir = Path(args.dump_dir)
            dump_dir.mkdir(parents=True, exist_ok=True)
            torch.save(hidden_out.cpu(), dump_dir / "mtp3_hidden.pt")
            torch.save(logits_out.cpu(), dump_dir / "mtp3_logits_shards.pt")
            torch.save(
                draft_token_ids_out.cpu(),
                dump_dir / "mtp3_draft_token_ids.pt",
            )
            print(f"[worker] dumped outputs to {dump_dir}", flush=True)

        print(
            f"[worker] RUN done {elapsed:.2f}s "
            f"tokens_row0={[tokens[0] for tokens in active_tokens]} "
            f"max|hidden47|="
            f"{hidden_out[:, 2].float().abs().max().item():.6f} "
            f"max|logits|={logits_out.abs().max().item():.6f}",
            flush=True,
        )
        print("[worker] RESULT=MTP3_IPC_RUN_CLEAN", flush=True)
    finally:
        if args.reuse_exporters:
            print(
                "[worker] reuse-exporters: pools remain live; no STOP",
                flush=True,
            )
        else:
            _stop(args.out, procs)
    return 0


def main() -> int:
    args = _parse_args()
    if args.export_rank >= 0:
        return _do_export(args)
    return _do_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
