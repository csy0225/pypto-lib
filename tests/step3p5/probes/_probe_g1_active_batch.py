# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""PERF-G1 dynamic active-token / active-batch acceptance probe.

本探针严格区分两个概念：

* ``storage_batch``：canonical Main 的物理 storage shape，固定为 16；
* ``active_batch``：本次 invocation 的有效 token 数，运行时覆盖
  ``1/2/8/16``。

因此，本探针不会把 fixed-storage ABI 报告成 fixed effective batch。每个
device case 都在同一个 resident holder 内执行，复用同一份 compiled program
和 KV pool，只改变本次 step 的 active rows、metadata reserve rows 以及
``num_tokens_per_owner``。

模式：

* ``contract``：只做 card-free source contract，输出结构化 JSON；
* ``compile``：编译 canonical ``whole_decode_step3p5``，不 prepare、不运行；
* ``device``：启动 standalone Main KV exporters，resident prepare 一次，
  连续执行 active batch ``1/2/8/16``，检查 hidden、inactive rows、KV
  reserve metadata 和 logical-bound contract。

镜像内示例：

    python -m tests.step3p5.probes._probe_g1_active_batch \
        --mode compile --platform a2a3sim --out /tmp/g1-compile

设备示例：

    python -m tests.step3p5.probes._probe_g1_active_batch \
        --mode device --device 8,9,10,11,12,13,14,15 \
        --ckpt /data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp \
        --out /tmp/g1-device
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
CANONICAL = ROOT / "models" / "step3p5" / "decode_fwd.py"
HOLDER = ROOT / "tools" / "step3p5" / "whole_decode_holder.py"
CONFIG = ROOT / "models" / "step3p5" / "config.py"

SCHEMA = "step3p5.perf_g1.active_batch.v1"
TP = 8
STORAGE_BATCH = 16
TOPK = 8
BLOCK_SIZE = 128
ACTIVE_BATCHES = (1, 2, 8, 16)
HETEROGENEOUS_OWNER_COUNTS = (1, 2, 8, 1, 4, 0, 3, 2)
HETEROGENEOUS_OWNER_MAX = 8
DEFAULT_CKPT = (
    "/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp"
)
DEFAULT_DEVICES = "8,9,10,11,12,13,14,15"


def _function_source(source: str, tree: ast.AST, name: str) -> tuple[str, int]:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one function {name}, found {len(matches)}")
    node = matches[0]
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise ValueError(f"cannot recover source for {name}")
    return segment, int(node.lineno)


def _line_for(segment: str, base_line: int, needle: str) -> int | None:
    for offset, line in enumerate(segment.splitlines()):
        if needle in line:
            return base_line + offset
    return None


def _config_constant(name: str, default: int) -> int:
    """读取 config.py 中的静态 BATCH，避免 compile mode 先 import runtime。"""
    try:
        tree = ast.parse(CONFIG.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                if len(node.targets) != 1:
                    continue
                target = node.targets[0]
                if (
                    isinstance(target, ast.Name)
                    and target.id == name
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, int)
                ):
                    return int(node.value.value)
    except (OSError, SyntaxError):
        pass
    return int(default)


def _owner_count_contract(
    owner_counts: tuple[int, ...] = HETEROGENEOUS_OWNER_COUNTS,
) -> dict[str, Any]:
    """验证异构 owner count 归约为全 rank 一致的 max(active_tokens)。

    这是 G1 的 card-free contract，不是设备 telemetry：canonical graph 会
    在 ``whole_chip_orch`` 中遍历 owner vector 并取 max；各 rank 不能各自
    使用不同 active bound，否则 EP dispatch/combine 的 token 数会失配。
    """
    values = [int(item) for item in owner_counts]
    valid_values = all(0 <= item <= STORAGE_BATCH for item in values)
    owner_count_ok = len(values) == TP
    observed_max = max(values) if values else 0
    return {
        "owner_counts": values,
        "owner_count": len(values),
        "expected_owner_count": TP,
        "valid_range": [0, STORAGE_BATCH],
        "values_in_storage_range": valid_values,
        "observed_max": observed_max,
        "expected_max": HETEROGENEOUS_OWNER_MAX,
        "max_reduction_is_correct": observed_max == HETEROGENEOUS_OWNER_MAX,
        "passed": bool(
            owner_count_ok
            and valid_values
            and observed_max == HETEROGENEOUS_OWNER_MAX
        ),
    }


def _synthetic_reserve_metadata(
    *,
    active_batch: int,
    scheduler_num_blocks: int = 32,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """构造 card-free fixed-storage metadata，专门验证 reserve/domain 隔离。"""
    import torch

    active_batch = int(active_batch)
    scheduler_num_blocks = int(scheduler_num_blocks)
    if not 1 <= active_batch <= STORAGE_BATCH:
        raise ValueError(
            f"active_batch must be in [1,{STORAGE_BATCH}], got {active_batch}"
        )
    physical_num_blocks = scheduler_num_blocks + STORAGE_BATCH - 1
    reserve = {
        "scheduler_num_blocks": scheduler_num_blocks,
        "physical_num_blocks": physical_num_blocks,
        "padding_block_ids": list(
            range(scheduler_num_blocks, physical_num_blocks)
        ),
        "block_size": BLOCK_SIZE,
    }
    seq = torch.ones(STORAGE_BATCH, dtype=torch.int32)
    pos = torch.zeros(STORAGE_BATCH, dtype=torch.int32)
    table = torch.zeros(
        STORAGE_BATCH,
        scheduler_num_blocks,
        dtype=torch.int32,
    )
    slots = torch.zeros(STORAGE_BATCH, dtype=torch.int32)

    active_blocks = torch.arange(active_batch, dtype=torch.int32)
    table[:active_batch, 0] = active_blocks
    slots[:active_batch] = active_blocks * BLOCK_SIZE
    for row, block_id in enumerate(
        reserve["padding_block_ids"][: STORAGE_BATCH - active_batch],
        start=active_batch,
    ):
        table[row, 0] = int(block_id)
        slots[row] = int(block_id) * BLOCK_SIZE

    metadata = _metadata_summary(
        seq_lens=seq,
        positions=pos,
        block_table=table,
        slot_mapping=slots,
        active_batch=active_batch,
        reserve=reserve,
    )
    return seq, pos, table, slots, metadata


def _inactive_reserve_domain_contract() -> dict[str, Any]:
    """对 active=1/2/8/16 做 card-free reserve/domain 检查。"""
    cases: list[dict[str, Any]] = []
    for active_batch in ACTIVE_BATCHES:
        _, _, _, _, metadata = _synthetic_reserve_metadata(
            active_batch=active_batch,
        )
        cases.append(
            {
                "active_batch": active_batch,
                "storage_batch": STORAGE_BATCH,
                "inactive_rows": STORAGE_BATCH - active_batch,
                "metadata": metadata,
                "passed": bool(metadata["passed"]),
            }
        )
    return {
        "storage_batch": STORAGE_BATCH,
        "active_batches": list(ACTIVE_BATCHES),
        "cases": cases,
        "passed": bool(cases) and all(item["passed"] for item in cases),
    }


def _source_contract() -> dict[str, Any]:
    """建立 G1 的静态逻辑边界合同，不把它误称为 device telemetry。"""
    canonical_source = CANONICAL.read_text(encoding="utf-8")
    canonical_tree = ast.parse(canonical_source)
    holder_source = HOLDER.read_text(encoding="utf-8")
    holder_tree = ast.parse(holder_source)

    method_needles: dict[str, tuple[str, ...]] = {
        "_gate": (
            "active_tokens = pl.cast(num_tokens, pl.INDEX)",
            "for nb in pl.spmd(",
            "for tt in pl.range(active_tokens):",
        ),
        "_histogram_and_prefix_sum": (
            "active_tokens = pl.cast(num_tokens, pl.INDEX)",
            "for t in pl.range(active_tokens):",
        ),
        "_dispatch_pack_publish": (
            "active_count = pl.read(active_token, [0])",
            "for t in pl.range(active_tokens):",
        ),
        "_dispatch_pull": (
            "active_routes = active_tokens * TOPK",
            "for r in pl.range(active_routes):",
            "for t_inv in pl.range(active_tokens):",
        ),
        "_dispatch_stage": (
            "for row in pl.range(rn):",
            "local_expert_count",
        ),
        "_stage_routed_src": (
            "active_rows = active_rows + pl.read(local_expert_count, [e])",
            "for row in pl.range(0, active_rows, stage_rows):",
        ),
        "_pull_routed_y": (
            "for t in pl.range(active_tokens):",
            "for k in pl.range(TOPK):",
            "for t_inactive in pl.range(active_tokens, BATCH):",
        ),
        "whole_chip_orch": (
            "num_tokens = pl.cast(0, pl.INT32)",
            "for owner_rank in pl.range(n_ranks):",
            "num_tokens = pl.max(",
            "pl.read(num_tokens_per_owner, [owner_rank])",
            "if num_tokens < 0:",
            "if num_tokens > BATCH:",
        ),
    }
    methods: dict[str, Any] = {}
    method_pass = True
    for name, needles in method_needles.items():
        segment, base_line = _function_source(
            canonical_source, canonical_tree, name
        )
        evidence = {
            needle: {
                "present": needle in segment,
                "line": _line_for(segment, base_line, needle),
            }
            for needle in needles
        }
        passed = all(item["present"] for item in evidence.values())
        method_pass = method_pass and passed
        methods[name] = {
            "passed": passed,
            "source_line": base_line,
            "evidence": evidence,
        }

    whole_segment, whole_line = _function_source(
        canonical_source, canonical_tree, "whole_chip_orch"
    )
    holder_live, holder_live_line = _function_source(
        holder_source, holder_tree, "set_live_step"
    )
    holder_enter, holder_enter_line = _function_source(
        holder_source, holder_tree, "__enter__"
    )
    top_level_checks = {
        "storage_batch_is_16": _config_constant("BATCH", STORAGE_BATCH)
        == STORAGE_BATCH,
        "active_batch_cases_are_1_2_8_16": tuple(ACTIVE_BATCHES)
        == (1, 2, 8, 16),
        "active_batch_cases_fit_fixed_storage": all(
            1 <= int(active_batch) <= STORAGE_BATCH
            for active_batch in ACTIVE_BATCHES
        ),
        "canonical_has_runtime_owner_vector": (
            "num_tokens_per_owner" in whole_segment
            and "num_tokens_per_owner" in canonical_source
        ),
        "canonical_owner_vector_storage_is_not_signal_stride": (
            "NUM_TOKENS_STORAGE_I32 = COMM_SIGNAL_STRIDE_I32"
            not in canonical_source
        ),
        "canonical_owner_vector_has_padded_storage": (
            "NUM_TOKENS_STORAGE_I32 = 128" in canonical_source
        ),
        "holder_updates_owner_vector_from_valid_tokens": (
            "self.num_tokens_per_owner[: self.tp].fill_(valid_tokens)"
            in holder_live
        ),
        "holder_keeps_fixed_storage_hidden": (
            "self.current_hidden[:, :valid_tokens, :] = hidden" in holder_live
            and "self.current_hidden.zero_()" in holder_live
        ),
        "holder_prepares_once_persistent": (
            "self.compiled.prepare(persistent=True)" in holder_enter
        ),
        "whole_graph_clamps_owner_max_to_storage": (
            "if num_tokens > BATCH:" in whole_segment
            and "num_tokens = pl.cast(BATCH, pl.INT32)" in whole_segment
        ),
    }
    owner_count_contract = _owner_count_contract()
    inactive_reserve_contract = _inactive_reserve_domain_contract()
    contract_ok = (
        method_pass
        and all(top_level_checks.values())
        and owner_count_contract["passed"]
        and inactive_reserve_contract["passed"]
    )
    return {
        "passed": contract_ok,
        "classification": (
            "source_contract_only; not device telemetry"
        ),
        "storage_batch": STORAGE_BATCH,
        "effective_active_batches_required": list(ACTIVE_BATCHES),
        "heterogeneous_owner_count_contract": owner_count_contract,
        "inactive_reserve_domain_contract": inactive_reserve_contract,
        "logical_bound_semantics": {
            "gate": (
                "top-k and route metadata iterate active rows; gate expert "
                "column fan-out may retain static physical [16,*] tiles"
            ),
            "dispatch": (
                "histogram/pack/pull/inverse-map logical rows are active; "
                "send/recv windows remain fixed-capacity physical storage"
            ),
            "combine": (
                "route gather iterates active rows and active TOPK routes; "
                "inactive output rows are explicitly zeroed"
            ),
        },
        "top_level_checks": top_level_checks,
        "methods": methods,
        "source_locations": {
            "canonical": str(CANONICAL),
            "holder": str(HOLDER),
            "whole_chip_orch": whole_line,
            "holder_set_live_step": holder_live_line,
            "holder_enter": holder_enter_line,
        },
    }


def _expected_bounds(active_batch: int) -> dict[str, Any]:
    active_batch = int(active_batch)
    routes = active_batch * TOPK
    return {
        "storage_batch": STORAGE_BATCH,
        "active_tokens": active_batch,
        "inactive_storage_rows": STORAGE_BATCH - active_batch,
        "gate": {
            "logical_input_rows": active_batch,
            "topk_output_rows": active_batch,
            "route_records": routes,
            "physical_score_rows": STORAGE_BATCH,
        },
        "dispatch": {
            "logical_histogram_rows": active_batch,
            "logical_pack_rows": active_batch,
            "logical_route_records": routes,
            "logical_pull_route_records": routes,
            "logical_inverse_map_rows": active_batch,
            "physical_send_capacity_routes": STORAGE_BATCH * TOPK,
            "physical_recv_capacity_routes": STORAGE_BATCH * TOPK,
        },
        "combine": {
            "logical_output_rows": active_batch,
            "logical_route_reads": routes,
            "logical_staged_routed_rows_upper_bound": routes,
            "physical_output_rows": STORAGE_BATCH,
        },
    }


def _parse_devices(text: str) -> list[int]:
    values = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if len(values) != TP or len(set(values)) != TP:
        raise ValueError(f"expected {TP} distinct devices, got {values}")
    return values


def _reserve_dict(reserve: Any) -> dict[str, Any]:
    if hasattr(reserve, "as_dict"):
        return {
            str(key): value
            for key, value in dict(reserve.as_dict()).items()
        }
    if isinstance(reserve, dict):
        return dict(reserve)
    raise TypeError(f"unsupported reserve object {type(reserve)!r}")


def _metadata_summary(
    *,
    seq_lens: Any,
    positions: Any,
    block_table: Any,
    slot_mapping: Any,
    active_batch: int,
    reserve: Any,
) -> dict[str, Any]:
    """检查固定 storage metadata 中 active/domain 与 reserve/domain 的隔离。"""
    import torch

    active_batch = int(active_batch)
    reserve_obj = _reserve_dict(reserve)
    scheduler = int(reserve_obj["scheduler_num_blocks"])
    physical = int(reserve_obj["physical_num_blocks"])
    reserve_ids = {
        int(item) for item in reserve_obj["padding_block_ids"]
    }
    seq = seq_lens.to(torch.int32).flatten().cpu()
    pos = positions.to(torch.int32).flatten().cpu()
    table = block_table.to(torch.int32).cpu()
    slots = slot_mapping.to(torch.int32).flatten().cpu()
    active_positions = pos[:active_batch].tolist()
    active_seq = seq[:active_batch].tolist()
    active_block_ids: list[int] = []
    for row in range(active_batch):
        col = int(pos[row]) // BLOCK_SIZE
        active_block_ids.append(int(table[row, col]))
    inactive_block_ids = [
        int(item) for item in table[active_batch:, 0].tolist()
    ]
    active_slots = [int(item) for item in slots[:active_batch].tolist()]
    inactive_slots = [int(item) for item in slots[active_batch:].tolist()]
    active_scheduler_domain = all(
        0 <= block < scheduler for block in active_block_ids
    )
    active_slot_domain = all(
        0 <= slot < scheduler * BLOCK_SIZE for slot in active_slots
    )
    inactive_reserve_domain = all(
        block in reserve_ids for block in inactive_block_ids
    )
    inactive_slot_domain = all(
        slot // BLOCK_SIZE in reserve_ids
        and slot % BLOCK_SIZE == 0
        for slot in inactive_slots
    )
    reserve_domain_is_physical_tail = (
        reserve_ids == set(range(scheduler, physical))
    )
    no_active_inactive_alias = not (
        set(active_slots).intersection(inactive_slots)
        or set(active_block_ids).intersection(inactive_block_ids)
    )
    shape_ok = (
        tuple(seq.shape) == (STORAGE_BATCH,)
        and tuple(pos.shape) == (STORAGE_BATCH,)
        and tuple(slots.shape) == (STORAGE_BATCH,)
        and table.ndim == 2
        and table.shape[0] == STORAGE_BATCH
    )
    padding_seq_ok = bool(
        torch.equal(
            seq[active_batch:],
            torch.ones(STORAGE_BATCH - active_batch, dtype=torch.int32),
        )
    )
    padding_pos_ok = bool(torch.count_nonzero(pos[active_batch:]).item() == 0)
    return {
        "storage_shapes": {
            "seq_lens": list(seq.shape),
            "positions": list(pos.shape),
            "block_table": list(table.shape),
            "slot_mapping": list(slots.shape),
        },
        "active_rows": {
            "count": active_batch,
            "seq_lens": active_seq,
            "positions": active_positions,
            "block_ids_at_position": active_block_ids,
            "slot_mapping": active_slots,
            "scheduler_domain": active_scheduler_domain,
            "slot_scheduler_domain": active_slot_domain,
        },
        "inactive_rows": {
            "count": STORAGE_BATCH - active_batch,
            "seq_lens_are_one": padding_seq_ok,
            "positions_are_zero": padding_pos_ok,
            "reserve_block_ids": inactive_block_ids,
            "reserve_domain": inactive_reserve_domain,
            "slot_mapping": inactive_slots,
            "slot_reserve_domain": inactive_slot_domain,
        },
        "reserve": reserve_obj,
        "kv_capacity": {
            "scheduler_num_blocks": scheduler,
            "physical_num_blocks": physical,
            "reserve_block_count": len(reserve_ids),
            "physical_minus_scheduler": physical - scheduler,
            "reserve_is_contiguous_physical_tail": (
                reserve_domain_is_physical_tail
            ),
        },
        "no_active_inactive_alias": no_active_inactive_alias,
        "shape_ok": shape_ok,
        "passed": bool(
            shape_ok
            and active_scheduler_domain
            and active_slot_domain
            and inactive_reserve_domain
            and inactive_slot_domain
            and reserve_domain_is_physical_tail
            and no_active_inactive_alias
            and padding_seq_ok
            and padding_pos_ok
        ),
    }


def _owner_vector_summary(holder: Any, active_batch: int) -> dict[str, Any]:
    import torch

    values = holder.num_tokens_per_owner.detach().cpu().to(torch.int32)
    owners = [int(item) for item in values[:TP].tolist()]
    tail_nonzero = int(torch.count_nonzero(values[TP:]).item())
    return {
        "storage_shape": list(values.shape),
        "owner_count": TP,
        "owner_values": owners,
        "owner_max": max(owners) if owners else 0,
        "expected_owner_value": int(active_batch),
        "tail_nonzero_count": tail_nonzero,
        "passed": (
            len(values) >= TP
            and owners == [int(active_batch)] * TP
            and tail_nonzero == 0
        ),
    }


def _hidden_summary(hidden: Any, active_batch: int) -> dict[str, Any]:
    import torch

    active_batch = int(active_batch)
    tensor = hidden.detach().cpu().to(torch.bfloat16)
    active = tensor[:, :active_batch].float()
    inactive = tensor[:, active_batch:].float()
    active_row_max = active.abs().amax(dim=-1)
    active_nonzero = int(torch.count_nonzero(active_row_max > 0).item())
    active_finite = bool(torch.isfinite(active).all().item())
    if inactive.numel():
        inactive_abs_max = float(inactive.abs().max().item())
        inactive_nonzero = int(
            torch.count_nonzero(inactive.abs().amax(dim=-1) > 0).item()
        )
        inactive_finite = bool(torch.isfinite(inactive).all().item())
    else:
        inactive_abs_max = 0.0
        inactive_nonzero = 0
        inactive_finite = True
    tp_spread = float(
        (
            active - active[0:1]
        ).abs().max().item()
    )
    return {
        "shape": [int(item) for item in tensor.shape],
        "storage_batch": int(tensor.shape[1]) if tensor.ndim >= 2 else None,
        "active_batch": active_batch,
        "active_hidden_finite": active_finite,
        "active_hidden_nonzero_rank_rows": active_nonzero,
        "expected_active_nonzero_rank_rows": TP * active_batch,
        "active_hidden_abs_max": float(active.abs().max().item()),
        "inactive_hidden_finite": inactive_finite,
        "inactive_hidden_nonzero_rank_rows": inactive_nonzero,
        "inactive_hidden_abs_max": inactive_abs_max,
        "inactive_rows_exact_zero": inactive_abs_max == 0.0,
        "hidden_tp_spread": tp_spread,
        "passed": bool(
            list(tensor.shape) == [TP, STORAGE_BATCH, int(tensor.shape[-1])]
            and active_finite
            and inactive_finite
            and active_nonzero == TP * active_batch
            and inactive_abs_max == 0.0
        ),
    }


def _write_report(out: Path, report: dict[str, Any]) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "g1_active_batch_report.json"
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    print(f"G1_REPORT={path}")
    return path


def _base_report(mode: str, contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "mode": mode,
        "product_entry": "models.step3p5.decode_fwd:whole_decode_step3p5",
        "storage_batch": STORAGE_BATCH,
        "effective_active_batches_required": list(ACTIVE_BATCHES),
        "fixed_storage_is_not_fixed_effective_batch": True,
        "logical_bound_contract": contract,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _run_contract(args: argparse.Namespace) -> int:
    contract = _source_contract()
    report = _base_report("contract", contract)
    report.update(
        {
            "ok": bool(contract["passed"]),
            "execution": {
                "card_required": False,
                "device_invocations": 0,
                "compile": False,
            },
        }
    )
    _write_report(Path(args.out), report)
    return 0 if report["ok"] else 1


def _configure_compile_env(out: Path, num_blocks: int) -> None:
    physical_blocks = int(num_blocks) + (STORAGE_BATCH - 1)
    os.environ["PYPTO_STEP3P5_MAX_SEQ"] = str(int(num_blocks) * BLOCK_SIZE)
    os.environ["PYPTO_STEP3P5_BLOCK_TABLE_FLAT"] = str(
        STORAGE_BATCH * int(num_blocks)
    )
    os.environ["PYPTO_STEP3P5_KV_CACHE_ROWS"] = str(
        45 * physical_blocks * BLOCK_SIZE
    )
    os.environ["PYPTO_STEP3P5_ROPE_SEQ"] = str(int(num_blocks) * BLOCK_SIZE)
    os.environ["PYPTO_PROG_BUILD_DIR"] = str(out / "build_output")


def _run_compile(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    contract = _source_contract()
    report = _base_report("compile", contract)
    report["execution"] = {
        "card_required": False,
        "compile": True,
        "platform": args.platform,
        "device_ids": _parse_devices(args.device),
        "build_dir": str(out / "build_output"),
        "active_batch_runtime_cases": list(ACTIVE_BATCHES),
    }
    try:
        _configure_compile_env(out, args.num_blocks)
        from tools.step3p5.whole_decode_holder import WholeDecodeHolder

        holder = WholeDecodeHolder(
            device_ids=_parse_devices(args.device),
            out_dir=str(out),
            ckpt=args.ckpt,
            platform=args.platform,
            kv_ipc=False,
        ).build()
        report["compile_result"] = {
            "passed": True,
            "compiled_output_dir": str(holder.compiled.output_dir),
            "canonical_program": holder.program_name,
            "compiled_storage_batch": int(holder._consts["BATCH"]),
            "compiled_kv_rows": int(holder._consts["KVC"]),
            "expected_kv_rows": (
                45 * (int(args.num_blocks) + STORAGE_BATCH - 1) * BLOCK_SIZE
            ),
        }
        report["ok"] = bool(
            contract["passed"]
            and holder.program_name == "whole_decode_step3p5"
            and int(holder._consts["BATCH"]) == STORAGE_BATCH
        )
    except Exception as exc:  # noqa: BLE001
        report["compile_result"] = {
            "passed": False,
            "error_type": type(exc).__name__,
            "error": repr(exc),
        }
        report["ok"] = False
    _write_report(out, report)
    return 0 if report["ok"] else 1


def _run_device(args: argparse.Namespace) -> int:
    import torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    devices = _parse_devices(args.device)
    contract = _source_contract()
    report = _base_report("device", contract)
    report["execution"] = {
        "card_required": True,
        "compile": True,
        "platform": args.platform,
        "device_ids": devices,
        "resident_holder": True,
        "invocations": list(ACTIVE_BATCHES),
        "same_storage_batch_for_all_cases": STORAGE_BATCH,
        "metadata_step_per_case": {
            str(active): index
            for index, active in enumerate(ACTIVE_BATCHES)
        },
    }
    reports: list[dict[str, Any]] = []
    procs: list[Any] = []
    try:
        _configure_compile_env(out, args.num_blocks)
        # 只复用现有诊断 exporter 的 graceful lifecycle，不在 probe 内
        # 自己 kill device 进程；device launch 前的卡清理仍由 runbook 负责。
        from tests.step3p5.harnesses._stage_main_hidden_only import (
            _start_exporters,
            _stop_exporters,
            _step_metadata,
            _load_embedding_row,
        )

        if not args.reuse_exporters:
            export_args = SimpleNamespace(
                out=str(out),
                ckpt=args.ckpt,
                num_blocks=args.num_blocks,
                kv_probe=False,
            )
            procs = _start_exporters(export_args, devices)
        else:
            ready = all(
                (out / f"ready.rank{rank}").exists()
                for rank in range(TP)
            )
            if not ready:
                raise RuntimeError(
                    "--reuse-exporters requested but exporter readiness "
                    "markers are incomplete"
                )

        from tools.step3p5.whole_decode_holder import WholeDecodeHolder

        holder = WholeDecodeHolder(
            device_ids=devices,
            out_dir=str(out),
            ckpt=args.ckpt,
            platform=args.platform,
            kv_ipc=True,
        ).build()
        embedding = _load_embedding_row(args.ckpt, args.seed_token)
        with holder:
            fixed_storage_ok = int(holder._consts["BATCH"]) == STORAGE_BATCH
            if not fixed_storage_ok:
                raise AssertionError(
                    "canonical compiled storage batch is not 16: "
                    f"{holder._consts['BATCH']}"
                )
            reserve = _reserve_dict(holder.padding_reserve)
            expected_kv_rows = (
                45 * int(reserve["physical_num_blocks"]) * BLOCK_SIZE
            )
            compiled_kv_rows = int(holder._consts["KVC"])
            map_path = out / "pypto_kvpool_map.json.rank0"
            map_summary: dict[str, Any] = {
                "present": map_path.exists(),
            }
            if map_path.exists():
                map_obj = json.loads(map_path.read_text(encoding="utf-8"))
                map_summary.update(
                    {
                        "scheduler_num_blocks": int(
                            map_obj["scheduler_num_blocks"]
                        ),
                        "physical_num_blocks": int(
                            map_obj["physical_num_blocks"]
                        ),
                        "reserve_start": int(map_obj["reserve_start"]),
                        "padding_block_count": int(
                            map_obj["padding_block_count"]
                        ),
                        "consistent_with_holder": (
                            int(map_obj["physical_num_blocks"])
                            == int(reserve["physical_num_blocks"])
                            and int(map_obj["scheduler_num_blocks"])
                            == int(reserve["scheduler_num_blocks"])
                        ),
                    }
                )

            for case_index, active_batch in enumerate(ACTIVE_BATCHES):
                seq, pos, table, slot = _step_metadata(
                    step=case_index,
                    scheduler_num_blocks=args.num_blocks,
                    valid_rows=active_batch,
                )
                holder.set_live_step(
                    embedding.unsqueeze(0).expand(
                        active_batch, -1
                    ).contiguous(),
                    seq_lens=seq,
                    positions=pos,
                    block_table=table,
                    slot_mapping=slot,
                )
                owner = _owner_vector_summary(holder, active_batch)
                metadata = _metadata_summary(
                    seq_lens=seq,
                    positions=pos,
                    block_table=table,
                    slot_mapping=slot,
                    active_batch=active_batch,
                    reserve=holder.padding_reserve,
                )
                started = time.time()
                result = holder.run()
                elapsed = time.time() - started
                hidden = result["next_hidden"]
                hidden_report = _hidden_summary(hidden, active_batch)
                bounds = _expected_bounds(active_batch)
                bounds["observed"] = {
                    "output_storage_shape": hidden_report["shape"],
                    "output_active_rows": active_batch,
                    "output_inactive_rows": STORAGE_BATCH - active_batch,
                    "owner_max": owner["owner_max"],
                    "dispatch_and_combine_route_upper_bound": (
                        active_batch * TOPK
                    ),
                }
                checks = {
                    "fixed_storage_shape": (
                        hidden_report["shape"][1] == STORAGE_BATCH
                    ),
                    "num_tokens_per_owner": bool(owner["passed"]),
                    "active_hidden": bool(
                        hidden_report["active_hidden_finite"]
                        and hidden_report[
                            "active_hidden_nonzero_rank_rows"
                        ]
                        == TP * active_batch
                    ),
                    "inactive_rows": bool(
                        hidden_report["inactive_hidden_finite"]
                        and hidden_report["inactive_rows_exact_zero"]
                    ),
                    "kv_reserve_metadata": bool(
                        metadata["passed"]
                        and compiled_kv_rows == expected_kv_rows
                        and (
                            not map_summary["present"]
                            or map_summary["consistent_with_holder"]
                        )
                    ),
                    "gate_dispatch_combine_logical_bounds": bool(
                        contract["passed"]
                        and bounds["gate"]["logical_input_rows"]
                        == active_batch
                        and bounds["dispatch"]["logical_route_records"]
                        == active_batch * TOPK
                        and bounds["combine"]["logical_route_reads"]
                        == active_batch * TOPK
                    ),
                }
                case_report = {
                    "active_batch": active_batch,
                    "storage_batch": STORAGE_BATCH,
                    "run_sec": elapsed,
                    "num_tokens_per_owner": owner,
                    "metadata": metadata,
                    "kv_map": map_summary,
                    "logical_bounds": bounds,
                    "hidden": hidden_report,
                    "checks": checks,
                    "passed": all(checks.values()),
                }
                reports.append(case_report)
                print(
                    json.dumps(
                        {
                            "schema": SCHEMA,
                            "mode": "device",
                            "active_batch": active_batch,
                            "passed": case_report["passed"],
                            "checks": checks,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        report["cases"] = reports
        report["compile_and_prepare"] = {
            "passed": True,
            "compiled_storage_batch": STORAGE_BATCH,
            "compiled_kv_rows": compiled_kv_rows,
            "expected_kv_rows": expected_kv_rows,
            "kv_reserve": reserve,
            "kv_map": map_summary,
        }
        report["ok"] = bool(
            contract["passed"]
            and len(reports) == len(ACTIVE_BATCHES)
            and all(item["passed"] for item in reports)
        )
    except Exception as exc:  # noqa: BLE001
        report["cases"] = reports
        report["ok"] = False
        report["error"] = {
            "error_type": type(exc).__name__,
            "error": repr(exc),
        }
    finally:
        if procs:
            from tests.step3p5.harnesses._stage_main_hidden_only import (
                _stop_exporters,
            )

            _stop_exporters(out, procs)
    _write_report(out, report)
    return 0 if report["ok"] else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PERF-G1: fixed storage batch=16 with dynamic active batch "
            "acceptance for 1/2/8/16"
        )
    )
    parser.add_argument(
        "--mode",
        choices=("contract", "compile", "device"),
        default="contract",
    )
    parser.add_argument("--out", default="/tmp/step3p5-g1-active-batch")
    parser.add_argument("--device", default=DEFAULT_DEVICES)
    parser.add_argument("--platform", choices=("a2a3", "a2a3sim"), default="a2a3")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--num-blocks", type=int, default=32)
    parser.add_argument("--seed-token", type=int, default=6127)
    parser.add_argument("--reuse-exporters", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.mode == "contract":
        return _run_contract(args)
    if args.mode == "compile":
        return _run_compile(args)
    return _run_device(args)


if __name__ == "__main__":
    raise SystemExit(main())
