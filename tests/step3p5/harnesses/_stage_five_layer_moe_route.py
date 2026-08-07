#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Capture exact L3/L4 route metadata from the focused L0-L4 graph."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import torch

from tests.step3p5.harnesses import _stage_five_layer_moe as formal_stage


TP = 8
BATCH = 16
HIDDEN = 4096
TOPK = 8
N_LOCAL_EXPERTS = 36
N_LOCAL_EXPERTS_PAD = 40
CONTEXT_LEN = 65536
GOLDEN_SCHEMA = "step3p5.five-layer-moe-golden.v3"
CHECKPOINT_SCHEMA = "step3p5.checkpoint-identity.v1"
IMAGE_DIGEST_PATTERN = re.compile(r".+@sha256:[0-9a-f]{64}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="8,9,10,11,12,13,14,15")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--checkpoint-manifest",
        default=os.environ.get("PYPTO_CHECKPOINT_MANIFEST", ""),
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--golden-dir", default="")
    parser.add_argument("--num-blocks", type=int, default=512)
    parser.add_argument("--context-len", type=int, default=65536)
    parser.add_argument("--active-batch", type=int, required=True)
    parser.add_argument("--seed-token", type=int, default=6127)
    parser.add_argument("--input-tokens", default="")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    parser.add_argument("--reuse-exporters", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--image-digest",
        default=os.environ.get("PYPTO_IMAGE_DIGEST", ""),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{path}: expected a tensor")
    return value


def _json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _sha256_field(value: object, *, field: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return value


def _image_digest(value: object, *, field: str) -> str:
    if not (
        isinstance(value, str)
        and IMAGE_DIGEST_PATTERN.fullmatch(value)
    ):
        raise ValueError(f"{field} must be an immutable image digest")
    return value


def _load_golden_contract(
    golden_dir: Path,
    *,
    active_batch: int,
    context_len: int,
    image_digest: str,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    """Validate the frozen baseline before starting device exporters."""
    manifest_path = golden_dir / "manifest.json"
    manifest = _json_object(manifest_path)
    if manifest.get("schema") != GOLDEN_SCHEMA:
        raise ValueError(f"{manifest_path}: unsupported golden schema")
    if manifest.get("source_kind") != "baseline":
        raise ValueError(f"{manifest_path}: golden is not from baseline")
    if manifest.get("active_batch") != active_batch:
        raise ValueError(f"{manifest_path}: active_batch mismatch")
    if (
        context_len != CONTEXT_LEN
        or manifest.get("context_len_per_sequence") != context_len
    ):
        raise ValueError(f"{manifest_path}: context mismatch")
    if manifest.get("image_ref") != image_digest:
        raise ValueError(f"{manifest_path}: image digest mismatch")
    source_run = manifest.get("source_run")
    if not isinstance(source_run, str) or not source_run:
        raise ValueError(f"{manifest_path}: source_run is missing")
    source_decode_sha = _sha256_field(
        manifest.get("source_decode_fwd_sha256"),
        field="golden.source_decode_fwd_sha256",
    )
    source_manifest_sha = _sha256_field(
        manifest.get("source_manifest_sha256"),
        field="golden.source_manifest_sha256",
    )
    files = manifest.get("files")
    if (
        not isinstance(files, dict)
        or set(files) != {"hidden_l3.pt", "hidden_l4.pt"}
    ):
        raise ValueError(f"{manifest_path}: files must be a mapping")

    tensors: dict[str, torch.Tensor] = {}
    file_hashes: dict[str, str] = {}
    expected_shape = (TP, active_batch, HIDDEN)
    for stem in ("hidden_l3", "hidden_l4"):
        name = f"{stem}.pt"
        path = golden_dir / name
        expected_sha = _sha256_field(
            files.get(name),
            field=f"golden.files.{name}",
        )
        actual_sha = _sha256(path)
        if actual_sha != expected_sha:
            raise ValueError(f"{manifest_path}: {name} hash mismatch")
        tensor = _load_tensor(path)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{path}: shape={tuple(tensor.shape)}, "
                f"expected={expected_shape}"
            )
        if tensor.dtype != torch.bfloat16:
            raise ValueError(
                f"{path}: dtype={tensor.dtype}, expected=torch.bfloat16"
            )
        tensors[stem] = tensor
        file_hashes[name] = actual_sha

    contract = {
        "schema": GOLDEN_SCHEMA,
        "manifest_sha256": _sha256(manifest_path),
        "source_run": source_run,
        "source_kind": "baseline",
        "source_decode_fwd_sha256": source_decode_sha,
        "source_manifest_sha256": source_manifest_sha,
        "active_batch": active_batch,
        "context_len_per_sequence": context_len,
        "image_ref": image_digest,
        "files": file_hashes,
        "bit_exact": True,
    }
    return contract, tensors


def _checkpoint_identity(
    ckpt: Path,
    manifest_path: Path | None = None,
) -> dict[str, object]:
    index_path = next(
        (
            path
            for path in (
                ckpt / "quant_model_weights.safetensors.index.json",
                ckpt / "model.safetensors.index.json",
            )
            if path.is_file()
        ),
        None,
    )
    if index_path is None:
        raise FileNotFoundError(
            "checkpoint identity requires an inspectable safetensors index"
        )
    index = _json_object(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_path}: missing non-empty weight_map")
    raw_shard_names = list(weight_map.values())
    if any(
        not isinstance(name, str)
        or not name
        or Path(name).is_absolute()
        or ".." in Path(name).parts
        for name in raw_shard_names
    ):
        raise ValueError(f"{index_path}: invalid shard name in weight_map")
    shard_names = sorted(set(raw_shard_names))

    expected_names = {"config.json", index_path.name, *shard_names}
    files: dict[str, dict[str, object]] = {}
    authority_manifest_sha256 = None
    if manifest_path is not None:
        manifest = _json_object(manifest_path)
        if manifest.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError(
                f"{manifest_path}: unsupported checkpoint manifest schema"
            )
        if manifest.get("logical_id") != ckpt.name:
            raise ValueError(
                f"{manifest_path}: checkpoint logical_id mismatch"
            )
        if manifest.get("index_file") != index_path.name:
            raise ValueError(
                f"{manifest_path}: checkpoint index_file mismatch"
            )
        if manifest.get("weight_tensor_count") != len(weight_map):
            raise ValueError(
                f"{manifest_path}: checkpoint tensor count mismatch"
            )
        if manifest.get("weight_shard_count") != len(shard_names):
            raise ValueError(
                f"{manifest_path}: checkpoint shard count mismatch"
            )
        manifest_files = manifest.get("files")
        if not isinstance(manifest_files, dict) or (
            set(manifest_files) != expected_names
        ):
            raise ValueError(
                f"{manifest_path}: checkpoint file set mismatch"
            )
        for name, record in manifest_files.items():
            if not isinstance(record, dict):
                raise ValueError(
                    f"{manifest_path}: invalid checkpoint file record {name}"
                )
            digest = _sha256_field(
                record.get("sha256"),
                field=f"checkpoint.files.{name}.sha256",
            )
            size = record.get("size_bytes")
            if type(size) is not int or size <= 0:
                raise ValueError(
                    f"{manifest_path}: invalid size for checkpoint file {name}"
                )
            path = ckpt / name
            if not path.is_file() or path.stat().st_size != size:
                raise ValueError(
                    f"{manifest_path}: checkpoint file size mismatch: {name}"
                )
            if _sha256(path) != digest:
                raise ValueError(
                    f"{manifest_path}: checkpoint file hash mismatch: {name}"
                )
            files[name] = {
                "size_bytes": size,
                "sha256": digest,
            }
        expected_identity = _json_sha256(files)
        if manifest.get("identity_sha256") != expected_identity:
            raise ValueError(
                f"{manifest_path}: checkpoint identity digest mismatch"
            )
        authority_manifest_sha256 = _sha256(manifest_path)
    else:
        for name in sorted(expected_names):
            path = ckpt / name
            if not path.is_file():
                raise FileNotFoundError(
                    f"checkpoint identity file is missing: {name}"
                )
            files[name] = {
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }

    for name in expected_names:
        path = ckpt / name
        if not path.is_file():
            raise FileNotFoundError(
                f"checkpoint identity file is missing: {name}"
            )
    result = {
        "schema": CHECKPOINT_SCHEMA,
        "logical_id": ckpt.name,
        "index_file": index_path.name,
        "weight_tensor_count": len(weight_map),
        "weight_shard_count": len(shard_names),
        "files": files,
        "identity_sha256": _json_sha256(files),
    }
    if authority_manifest_sha256 is not None:
        result["authority_manifest_sha256"] = authority_manifest_sha256
    return result


def _validate_route_totals(
    recv_meta: torch.Tensor,
    *,
    active_batch: int,
) -> dict[str, object]:
    expected_shape = (TP, 2, TP, N_LOCAL_EXPERTS_PAD)
    if tuple(recv_meta.shape) != expected_shape:
        raise ValueError(
            f"recv_meta shape={tuple(recv_meta.shape)}, "
            f"expected={expected_shape}"
        )
    if recv_meta.dtype != torch.int32:
        raise ValueError(
            f"recv_meta dtype={recv_meta.dtype}, expected=torch.int32"
        )
    if bool(torch.any(recv_meta < 0)):
        raise ValueError("recv_meta contains negative counts")
    if bool(torch.any(recv_meta[:, :, :, N_LOCAL_EXPERTS:] != 0)):
        raise ValueError("recv_meta padded experts 36:40 must be zero")

    routed = recv_meta[:, :, :, :N_LOCAL_EXPERTS].to(torch.int64)
    per_layer_source = routed.sum(dim=(0, 3))
    expected_per_source = int(active_batch) * TOPK
    expected_source_matrix = torch.full(
        (2, TP),
        expected_per_source,
        dtype=torch.int64,
    )
    if not torch.equal(per_layer_source, expected_source_matrix):
        raise ValueError(
            "each source rank/layer must route active_batch * TOPK entries: "
            f"actual={per_layer_source.tolist()}, "
            f"expected={expected_source_matrix.tolist()}"
        )

    global_per_layer = per_layer_source.sum(dim=1)
    expected_global = TP * expected_per_source
    if not torch.equal(
        global_per_layer,
        torch.full((2,), expected_global, dtype=torch.int64),
    ):
        raise ValueError(
            f"global route totals={global_per_layer.tolist()}, "
            f"expected={[expected_global, expected_global]}"
        )
    return {
        "per_layer_per_source": per_layer_source.tolist(),
        "expected_per_source": expected_per_source,
        "global_per_layer": global_per_layer.tolist(),
        "expected_global_per_layer": expected_global,
        "padding_zero": True,
        "nonnegative": True,
    }


def _sidecar_payload(
    *,
    recv_meta_device: torch.Tensor,
    local_expert_count_device: torch.Tensor,
    provenance: dict[str, object],
    window_id_prefix: str,
) -> dict[str, object]:
    if tuple(local_expert_count_device.shape) != (TP, 2, N_LOCAL_EXPERTS):
        raise ValueError(
            "local_expert_count shape="
            f"{tuple(local_expert_count_device.shape)}, "
            f"expected={(TP, 2, N_LOCAL_EXPERTS)}"
        )
    recv_meta = recv_meta_device.permute(1, 0, 2, 3).contiguous()
    local_expert_count = (
        local_expert_count_device.permute(1, 0, 2).contiguous()
    )
    return {
        "schema": "step3p5.five-layer-moe-recv-meta.v1",
        "layers": ["L3", "L4"],
        "axes": [
            "layer",
            "dst_rank",
            "src_rank",
            "local_expert_pad",
        ],
        "recv_meta": recv_meta,
        "local_expert_count": local_expert_count,
        "window_provenance": [
            {
                "layer": "L3",
                "window_id": f"{window_id_prefix}-l3",
                "shape": [TP, N_LOCAL_EXPERTS_PAD],
                "dtype": "int32",
                "byte_size": TP * N_LOCAL_EXPERTS_PAD * 4,
                "source_window": "moe_recv_meta",
                "source_window_reused": True,
                "capture_point": "after_l3_before_l4",
            },
            {
                "layer": "L4",
                "window_id": f"{window_id_prefix}-l4",
                "shape": [TP, N_LOCAL_EXPERTS_PAD],
                "dtype": "int32",
                "byte_size": TP * N_LOCAL_EXPERTS_PAD * 4,
                "source_window": "moe_recv_meta",
                "source_window_reused": True,
                "capture_point": "after_l4",
            },
        ],
        "provenance": provenance,
    }


def _prepare_out(args: argparse.Namespace) -> Path:
    out = Path(args.out)
    if out.exists() and not args.reuse_exporters and any(out.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty route evidence directory {out}"
        )
    out.mkdir(parents=True, exist_ok=True)
    for name in (
        "recv_meta.pt",
        "local_expert_count.pt",
        "recv_meta_sidecar.pt",
        "hidden_l3.pt",
        "hidden_l4.pt",
        "five_layer_moe_route_report.json",
    ):
        if (out / name).exists():
            raise FileExistsError(f"refusing to overwrite {out / name}")
    return out


def main() -> int:
    args = _parse_args()
    args.iters = 1
    args.dfx = False
    args.pmu = False
    args.kv_probe = False
    devices = formal_stage._devices(args.device)
    out = _prepare_out(args)
    layout = formal_stage._configure(args)

    from tools.step3p5.five_layer_moe_route_holder import (
        FiveLayerMoeRouteHolder,
    )

    if args.compile_only:
        holder = FiveLayerMoeRouteHolder(
            devices,
            str(out),
            args.ckpt,
            platform=args.platform,
            kv_ipc=False,
        ).build()
        (out / "compile_report.json").write_text(
            json.dumps(
                {
                    "ok": True,
                    "program": holder.program_name,
                    "output_dir": str(holder.compiled.output_dir),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0

    if not args.golden_dir:
        raise ValueError("--golden-dir is required for route capture")
    if not args.image_digest:
        raise ValueError(
            "--image-digest or PYPTO_IMAGE_DIGEST is required for provenance"
        )
    _image_digest(args.image_digest, field="image_digest")
    golden_dir = Path(args.golden_dir)
    golden_contract, golden_tensors = _load_golden_contract(
        golden_dir,
        active_batch=args.active_batch,
        context_len=args.context_len,
        image_digest=args.image_digest,
    )
    checkpoint = _checkpoint_identity(
        Path(args.ckpt),
        (
            Path(args.checkpoint_manifest)
            if args.checkpoint_manifest
            else None
        ),
    )

    from tests.step3p5.harnesses import _stage_main_hidden_only as main_stage

    export_args = argparse.Namespace(**vars(args))
    export_args.num_blocks = layout["scheduler_num_blocks"]
    export_args.kv_num_layers = 5

    repo_root = Path(__file__).resolve().parents[3]
    procs = []
    try:
        if not args.reuse_exporters:
            procs = main_stage._start_exporters(export_args, devices)
        if args.reuse_exporters and not all(
            main_stage._ready(out, rank) for rank in range(TP)
        ):
            raise RuntimeError(
                "reuse-exporters requested but IPC maps are incomplete"
            )
        holder = FiveLayerMoeRouteHolder(
            devices,
            str(out),
            args.ckpt,
            platform=args.platform,
            kv_ipc=True,
        ).build()
        with holder:
            input_tokens = formal_stage._input_token_ids(args)
            active_hidden = torch.stack(
                [
                    main_stage._load_embedding_row(args.ckpt, token)
                    for token in input_tokens
                ],
                dim=0,
            ).contiguous()
            seq, pos, table, slot = formal_stage._step_metadata(
                context_len=args.context_len,
                active_batch=args.active_batch,
                blocks_per_row_capacity=args.num_blocks,
                scheduler_num_blocks=layout["scheduler_num_blocks"],
            )
            set_kwargs = {
                "seq_lens": seq,
                "positions": pos,
                "block_table": table,
                "slot_mapping": slot,
            }
            for _ in range(args.warmup):
                holder.set_live_step(active_hidden, **set_kwargs)
                holder.run()
            holder.set_live_step(active_hidden, **set_kwargs)
            result = holder.run()

            hidden_l3 = (
                result["hidden_l3"][:, : args.active_batch]
                .to(torch.bfloat16)
                .clone()
                .cpu()
            )
            hidden_l4 = (
                result["hidden_l4"][:, : args.active_batch]
                .to(torch.bfloat16)
                .clone()
                .cpu()
            )
            recv_meta_device = result["recv_meta"].clone().cpu()
            local_expert_count_device = (
                result["local_expert_count"].clone().cpu()
            )
    finally:
        if not args.reuse_exporters:
            main_stage._stop_exporters(out, procs)

    golden_l3 = golden_tensors["hidden_l3"]
    golden_l4 = golden_tensors["hidden_l4"]
    hidden_comparison = {
        "hidden_l3": {
            "exact": bool(torch.equal(hidden_l3, golden_l3)),
            "shape": list(hidden_l3.shape),
            "golden_shape": list(golden_l3.shape),
            "dtype": str(hidden_l3.dtype),
            "golden_dtype": str(golden_l3.dtype),
        },
        "hidden_l4": {
            "exact": bool(torch.equal(hidden_l4, golden_l4)),
            "shape": list(hidden_l4.shape),
            "golden_shape": list(golden_l4.shape),
            "dtype": str(hidden_l4.dtype),
            "golden_dtype": str(golden_l4.dtype),
        },
    }
    if not all(item["exact"] for item in hidden_comparison.values()):
        raise AssertionError(
            "instrumented hidden outputs are not bit-exact to formal golden: "
            f"{hidden_comparison}"
        )

    route_validation = _validate_route_totals(
        recv_meta_device,
        active_batch=args.active_batch,
    )
    derived_counts = recv_meta_device[
        :, :, :, :N_LOCAL_EXPERTS
    ].sum(dim=2, dtype=torch.int64).to(torch.int32)
    if not torch.equal(derived_counts, local_expert_count_device):
        raise AssertionError(
            "holder local_expert_count is not exact sum_src(recv_meta)"
        )

    source = {
        "source_tree_manifest_sha256": formal_stage._source_sha256(
            repo_root,
            "SOURCE_SHA256SUMS",
        ),
        "decode_fwd_sha256": formal_stage._source_sha256(
            repo_root,
            "models/step3p5/decode_fwd.py",
        ),
        "formal_program_sha256": formal_stage._source_sha256(
            repo_root,
            "tests/step3p5/harnesses/_five_layer_moe_program.py",
        ),
        "route_program_sha256": formal_stage._source_sha256(
            repo_root,
            "tests/step3p5/harnesses/_five_layer_moe_route_program.py",
        ),
        "route_holder_sha256": formal_stage._source_sha256(
            repo_root,
            "tools/step3p5/five_layer_moe_route_holder.py",
        ),
        "route_stage_sha256": formal_stage._source_sha256(
            repo_root,
            "tests/step3p5/harnesses/_stage_five_layer_moe_route.py",
        ),
    }
    source_manifest_sha256 = _json_sha256(source)
    input_contract = {
        "workload": {
            "active_batch": args.active_batch,
            "active_batch_semantics": "packed_global_replicated",
            "heterogeneous_owner_counts_supported": False,
            "num_tokens_per_owner": [args.active_batch] * TP,
            "context_len": args.context_len,
            "num_blocks_per_sequence": args.num_blocks,
            "context_semantics": "per_active_sequence",
            **layout,
        },
        "input_tokens": input_tokens,
        "tensor_sha256": {
            "active_hidden": _tensor_sha256(active_hidden),
            "seq_lens": _tensor_sha256(seq),
            "positions": _tensor_sha256(pos),
            "block_table": _tensor_sha256(table),
            "slot_mapping": _tensor_sha256(slot),
        },
    }
    provenance = {
        "image_digest": args.image_digest,
        "checkpoint": checkpoint,
        "source": source,
        "source_manifest_sha256": source_manifest_sha256,
        "input_contract": input_contract,
        "input_contract_sha256": _json_sha256(input_contract),
        "formal_golden": golden_contract,
    }
    sidecar = _sidecar_payload(
        recv_meta_device=recv_meta_device,
        local_expert_count_device=local_expert_count_device,
        provenance=provenance,
        window_id_prefix=(
            f"route-snapshot-{source_manifest_sha256[:16]}-"
            f"bs{args.active_batch}"
        ),
    )

    artifact_tensors = {
        "hidden_l3.pt": hidden_l3,
        "hidden_l4.pt": hidden_l4,
        "recv_meta.pt": recv_meta_device,
        "local_expert_count.pt": local_expert_count_device,
        "recv_meta_sidecar.pt": sidecar,
    }
    for name, value in artifact_tensors.items():
        torch.save(value, out / name)
    artifact_hashes = {
        name: _sha256(out / name) for name in artifact_tensors
    }
    report = {
        "schema": "step3p5.five-layer-moe-route-capture.v1",
        "program": "FiveLayerMoeRoute",
        "devices": devices,
        "hidden_comparison": hidden_comparison,
        "route_validation": route_validation,
        "provenance": provenance,
        "artifacts": artifact_hashes,
        "device_output_semantics": {
            "recv_meta": {
                "shape": [TP, 2, TP, N_LOCAL_EXPERTS_PAD],
                "dtype": "int32",
                "axes": [
                    "dst_rank",
                    "layer",
                    "src_rank",
                    "local_expert_pad",
                ],
            },
            "local_expert_count": {
                "shape": [TP, 2, N_LOCAL_EXPERTS],
                "dtype": "int32",
                "derivation": "sum over src_rank",
            },
        },
        "analyzer_sidecar": "recv_meta_sidecar.pt",
    }
    report_path = out / "five_layer_moe_route_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "report": report_path.name,
                "recv_meta": "recv_meta.pt",
                "sidecar": "recv_meta_sidecar.pt",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    print("[worker] RUN done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
