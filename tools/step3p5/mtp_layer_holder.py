# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Resident selected-layer MTP hidden-only holder.

One holder prepares the three compile-time selected programs once and chooses
the program by ``layer_idx`` for every proposer call.  The holder never
computes a shared head, logits, draft token or acceptance result.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

from tools.step3p5.device_topology import validate_consecutive_device_ids

_BF16 = torch.bfloat16
_F32 = torch.float32
_I32 = torch.int32


def _shared_zeros(*shape, dtype=_BF16):
    return torch.zeros(shape, dtype=dtype).share_memory_()


def _mtp_build_output_dirs(
    base_dir: str,
    *,
    num_layers: int = 3,
    run_tag: str | None = None,
) -> tuple[str, ...]:
    """为每个 compile-time MTP variant 分配不可冲突的 build 目录。"""
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if run_tag is None:
        run_tag = (
            f"{time.strftime('%Y%m%d_%H%M%S')}_"
            f"{os.getpid()}_{time.time_ns()}"
        )
    if not run_tag or os.sep in run_tag:
        raise ValueError("run_tag must be one non-empty path component")
    root = os.path.abspath(os.path.expanduser(base_dir))
    output_dirs = tuple(
        os.path.join(root, f"MtpLayerHidden{layer_idx}_{run_tag}")
        for layer_idx in range(num_layers)
    )
    if len(set(output_dirs)) != num_layers:
        raise RuntimeError("MTP build output directories must be unique")
    return output_dirs


def _prepare_selected_programs(compiled):
    """在一个 L3 worker 上一次性准备三个 selected-MTP program。

    三个 compile-time layer variant 共享同一组物理设备、IPC 权重和 MTP KV。
    为每个 variant 分别 ``prepare()`` 会创建三套 chip process / communicator，
    第二套 runtime 在第一套已经初始化 HCCL 后无法再次建立同一设备域。正式
    runtime API 已支持 multi-program dispatch，因此这里只允许：

    ``primary.prepare(extra_compiled=(layer1, layer2))``。
    """
    programs = tuple(compiled)
    if len(programs) != 3:
        raise ValueError(
            f"selected MTP requires exactly 3 compiled programs, got {len(programs)}"
        )
    return programs[0].prepare(extra_compiled=programs[1:])


class MtpLayerHolder:
    """Resident MTP45/46/47 selected-layer body."""

    def __init__(
        self,
        device_ids,
        out_dir,
        ckpt,
        *,
        mtp_kv_dir=None,
        platform="a2a3",
    ):
        self.device_ids = validate_consecutive_device_ids(
            device_ids,
            owner="MTP layer",
        )
        self.tp = len(self.device_ids)
        self.dev_offset = self.device_ids[0]
        self.out_dir = out_dir
        self.mtp_kv_dir = mtp_kv_dir or out_dir
        self.ckpt = ckpt
        self.platform = platform
        self.compiled = []
        self.build_output_dirs = ()
        self._prepare_cm = None
        self._prepare_entered = False
        self._runtime = None
        self._cfg = None
        self._K = None
        self._consts = {}
        self._args = []
        self._current_layer = None
        self.previous_hidden = None
        self.input_token_ids = None
        self.active_mask = None
        self.seq_lens = None
        self.block_table = None
        self.slot_mapping = None
        self.rope_cos = None
        self.rope_sin = None
        self.k_cache = None
        self.v_cache = None
        self.hidden_out = None
        self.gate_r = None
        self.padding_reserve = None

    def _infer_rows_from_map(self) -> None:
        path = os.path.join(self.mtp_kv_dir, "pypto_mtp_kvpool_map.json.rank0")
        try:
            with open(path, encoding="utf-8") as file:
                obj = json.load(file)
            rows = int(obj["map"]["MTP0.K"]["num_slots"])
            physical = int(obj["physical_num_blocks"])
            scheduler = int(obj["scheduler_num_blocks"])
            if physical < scheduler + 15:
                raise ValueError(
                    f"MTP KV map physical={physical} lacks 15-block reserve "
                    f"above scheduler={scheduler}"
                )
        except FileNotFoundError:
            return
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid MTP KV padding-reserve map {path}: {exc}"
            ) from exc
        configured = os.environ.get("PYPTO_STEP3P5_MTP_KV_CACHE_ROWS")
        if configured is not None and int(configured) != rows:
            raise ValueError(
                "PYPTO_STEP3P5_MTP_KV_CACHE_ROWS disagrees with MTP IPC map: "
                f"configured={configured}, expected={rows}"
            )
        os.environ["PYPTO_STEP3P5_MTP_KV_CACHE_ROWS"] = str(rows)

    def build(self):
        from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415

        set_backend_type(BackendType.Ascend910B)
        self._infer_rows_from_map()
        import models.step3p5.config as cfg  # noqa: PLC0415
        from models.step3p5 import weight_loader as keys  # noqa: PLC0415
        from models.step3p5.mtp_hidden_fwd import (  # noqa: PLC0415
            MTP_LAYER_HIDDEN_PROGRAMS,
        )
        from pypto import ir  # noqa: PLC0415
        from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415

        self._cfg = cfg
        self._K = keys
        if self.tp != int(cfg.TP_WORLD_SIZE):
            raise ValueError(
                f"MTP holder requires TP={cfg.TP_WORLD_SIZE}, got {self.tp}"
            )
        os.environ.setdefault(
            "PYPTO_PROG_BUILD_DIR",
            "/tmp/pypto_build_output",
        )
        self.build_output_dirs = _mtp_build_output_dirs(
            os.environ["PYPTO_PROG_BUILD_DIR"],
            num_layers=len(MTP_LAYER_HIDDEN_PROGRAMS),
        )
        self.compiled = []
        for layer_idx, (program, output_dir) in enumerate(
            zip(
                MTP_LAYER_HIDDEN_PROGRAMS,
                self.build_output_dirs,
                strict=True,
            )
        ):
            compiled = ir.compile(
                program,
                output_dir=output_dir,
                platform=self.platform,
                distributed_config=DistributedConfig(
                    device_ids=self.device_ids,
                    num_sub_workers=0,
                ),
                skip_ptoas=False,
                dump_passes=False,
            )
            self.compiled.append(compiled)
            print(
                f"[mtp-holder] compile layer={layer_idx} OK => "
                f"{compiled.output_dir}",
                flush=True,
            )
        actual_output_dirs = tuple(
            os.path.abspath(compiled.output_dir)
            for compiled in self.compiled
        )
        if actual_output_dirs != self.build_output_dirs:
            raise RuntimeError(
                "MTP compiler returned unexpected build directories: "
                f"expected={self.build_output_dirs}, "
                f"actual={actual_output_dirs}"
            )
        if len(set(actual_output_dirs)) != len(actual_output_dirs):
            raise RuntimeError("MTP compiled artifacts alias one build directory")

        self._consts = {
            "BATCH": int(cfg.BATCH),
            "HIDDEN": int(cfg.HIDDEN),
            "HEAD_DIM": int(cfg.HEAD_DIM),
            "VOCAB": int(cfg.VOCAB),
            "USER_BATCH": int(cfg.USER_BATCH_DYN),
            "BLOCK_TABLE_FLAT": int(cfg.BLOCK_TABLE_FLAT_DYN),
            "ROPE_SEQ": int(cfg.ROPE_SEQ_DYN),
            "MTP_ROWS": int(cfg.MTP_KV_CACHE_ROWS_DYN),
        }
        return self

    def __enter__(self):
        if not self.compiled:
            raise RuntimeError("call build() before entering MTP holder")
        if self._prepare_cm is not None or self._runtime is not None:
            raise RuntimeError(
                "MTP holder cleanup is incomplete; retry __exit__ before re-entering"
            )
        try:
            return self._enter_impl()
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def _enter_impl(self):
        if not self.compiled:
            raise RuntimeError("call build() before entering MTP holder")
        c = self._consts
        tp, batch, hidden = self.tp, c["BATCH"], c["HIDDEN"]
        self.previous_hidden = _shared_zeros(tp, batch, hidden)
        self.input_token_ids = _shared_zeros(tp, batch, dtype=_I32)
        self.active_mask = _shared_zeros(tp, batch, dtype=_I32)
        self.seq_lens = torch.ones(
            tp, c["USER_BATCH"], dtype=_I32
        ).share_memory_()
        self.block_table = _shared_zeros(
            tp, c["BLOCK_TABLE_FLAT"], dtype=_I32
        )
        self.slot_mapping = _shared_zeros(
            tp, c["USER_BATCH"], dtype=_I32
        )
        self.rope_cos = _shared_zeros(
            tp, c["ROPE_SEQ"], c["HEAD_DIM"], dtype=_F32
        )
        self.rope_sin = _shared_zeros(
            tp, c["ROPE_SEQ"], c["HEAD_DIM"], dtype=_F32
        )
        self.hidden_out = _shared_zeros(tp, batch, hidden)
        heads = int(self._cfg.NUM_HEADS_SWA_LOCAL)
        heads_pad = int(self._cfg.NUM_HEADS_SWA_LOCAL_PAD)
        hidden_q = int(self._cfg.HIDDEN_Q_SWA_LOCAL)
        self.gate_r = _shared_zeros(tp, heads_pad, hidden_q)
        for head in range(heads):
            start = head * c["HEAD_DIM"]
            self.gate_r[:, head, start : start + c["HEAD_DIM"]] = 1.0

        from models.step3p5._ops import build_plain_rope_tables  # noqa: PLC0415

        rope_cos, rope_sin = build_plain_rope_tables(
            c["ROPE_SEQ"], c["HEAD_DIM"], 10_000.0
        )
        self.rope_cos.copy_(rope_cos.unsqueeze(0).expand(tp, -1, -1))
        self.rope_sin.copy_(rope_sin.unsqueeze(0).expand(tp, -1, -1))

        from tools.step3p5.pypto_mtp_kv_ipc import (  # noqa: PLC0415
            build_stacked_mtp_kv_pool,
            import_mtp_kv_all,
        )
        from tools.step3p5.pypto_weight_ipc import (  # noqa: PLC0415
            build_stacked_weight,
            import_weights_all,
        )

        self._args = []
        cm = _prepare_selected_programs(self.compiled)
        self._prepare_cm = cm
        self._prepare_entered = False
        runtime = cm.__enter__()
        self._prepare_entered = True
        self._runtime = runtime
        try:
            weight_maps = import_weights_all(
                runtime,
                self.out_dir,
                tp=tp,
                dev_offset=self.dev_offset,
                native_w8a8=False,
            )
            mtp_maps = import_mtp_kv_all(
                runtime,
                self.mtp_kv_dir,
                tp=tp,
                dev_offset=self.dev_offset,
            )
            from tools.step3p5.kv_padding import (  # noqa: PLC0415
                make_padding_reserve,
            )

            summary = mtp_maps[0].summary
            self.padding_reserve = make_padding_reserve(
                summary.scheduler_num_blocks,
                summary.physical_num_blocks,
                block_size=128,
            )
            self.k_cache, self.v_cache = build_stacked_mtp_kv_pool(mtp_maps)

            def W(key):
                return build_stacked_weight(weight_maps, key)

            from pypto.runtime.device_tensor import (  # noqa: PLC0415
                StackedDeviceTensor,
            )

            def W_reshape(key, shape, dtype):
                shards = []
                expected_numel = 1
                for dim in shape:
                    expected_numel *= int(dim)
                for rank in range(tp):
                    source = weight_maps[rank].device_tensor(key)
                    source_numel = 1
                    for dim in source.shape:
                        source_numel *= int(dim)
                    if source_numel != expected_numel:
                        raise ValueError(
                            f"{key} rank{rank}: cannot reshape "
                            f"{source.shape} to {shape}"
                        )
                    if source.dtype != dtype:
                        raise ValueError(
                            f"{key} rank{rank}: source dtype {source.dtype} "
                            f"does not match requested dtype {dtype}"
                        )
                    shards.append(source.reshape(tuple(shape)))
                return StackedDeviceTensor(
                    shards,
                    (tp, *tuple(shape)),
                    list(range(tp)),
                )

            K = self._K
            num_mtp = int(self._cfg.NUM_NEXTN_PREDICT_LAYERS)
            hidden_local = c["HIDDEN"] // tp
            hidden_q = int(self._cfg.HIDDEN_Q_SWA_LOCAL)
            kv_hidden = int(self._cfg.KV_HIDDEN_LOCAL)
            intermediate = int(self._cfg.INTERMEDIATE_LOCAL)
            # 三个 compile-time variant 使用同一 worker-resident 参数对象；具体
            # MTP45/46/47 weight row 仍由各 program 内的 compile-time index 选择。
            shared_args = [
                self.previous_hidden,
                self.input_token_ids,
                self.active_mask,
                W(K.KEY_EMBED),
                W(K.KEY_MTP_ENORM),
                W(K.KEY_MTP_HNORM),
                W_reshape(
                    K.KEY_MTP_EH_PROJ,
                    (num_mtp * hidden_local, 2 * c["HIDDEN"]),
                    _BF16,
                ),
                W(K.KEY_MTP_INPUT_RMS),
                W_reshape(
                    K.KEY_MTP_WQ,
                    (num_mtp * c["HIDDEN"], hidden_q),
                    _BF16,
                ),
                W_reshape(
                    K.KEY_MTP_WK,
                    (num_mtp * c["HIDDEN"], kv_hidden),
                    _BF16,
                ),
                W_reshape(
                    K.KEY_MTP_WV,
                    (num_mtp * c["HIDDEN"], kv_hidden),
                    _BF16,
                ),
                W(K.KEY_MTP_Q_NORM),
                W(K.KEY_MTP_K_NORM),
                W_reshape(
                    K.KEY_MTP_WO,
                    (num_mtp * hidden_q, c["HIDDEN"]),
                    _BF16,
                ),
                W_reshape(
                    K.KEY_MTP_WG,
                    (
                        num_mtp * c["HIDDEN"],
                        int(self._cfg.NUM_HEADS_SWA_LOCAL_PAD),
                    ),
                    _BF16,
                ),
                self.gate_r,
                W(K.KEY_MTP_POST_ATTN_RMS),
                W_reshape(
                    K.KEY_MTP_DENSE_GATE,
                    (num_mtp * c["HIDDEN"], intermediate),
                    _BF16,
                ),
                W_reshape(
                    K.KEY_MTP_DENSE_UP,
                    (num_mtp * c["HIDDEN"], intermediate),
                    _BF16,
                ),
                W_reshape(
                    K.KEY_MTP_DENSE_DOWN,
                    (num_mtp * intermediate, c["HIDDEN"]),
                    _BF16,
                ),
                self.seq_lens,
                self.block_table,
                self.slot_mapping,
                self.rope_cos,
                self.rope_sin,
                self.k_cache,
                self.v_cache,
                self.hidden_out,
            ]
            self._args = [shared_args for _ in self.compiled]
        except BaseException:
            raise
        print(
            f"[mtp-holder] resident selected programs={len(self.compiled)} "
            "workers=1",
            flush=True,
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        cleanup_error = None
        cm = self._prepare_cm
        if cm is None and self._runtime is not None:
            cleanup_error = RuntimeError(
                "MTP runtime exists without its prepared context manager"
            )
        elif cm is not None:
            try:
                if self._prepare_entered:
                    cm.__exit__(exc_type, exc, tb)
                else:
                    close = getattr(cm, "close", None)
                    if close is None:
                        raise RuntimeError(
                            "MTP prepared context failed before __enter__ "
                            "and exposes no close()"
                        )
                    close()
            except BaseException as err:
                cleanup_error = err
            else:
                self._prepare_cm = None
                self._prepare_entered = False
                self._runtime = None
                self._args = []
                self._current_layer = None
                self.padding_reserve = None

        if exc_type is None and cleanup_error is not None:
            raise cleanup_error
        return False

    def set_live_step(
        self,
        previous_hidden,
        *,
        input_token_ids,
        active_mask,
        seq_lens,
        positions,
        block_table,
        slot_mapping,
    ):
        c = self._consts
        previous_hidden = previous_hidden.to(_BF16).contiguous()
        valid = int(previous_hidden.shape[0]) if previous_hidden.ndim == 2 else -1
        if tuple(previous_hidden.shape) != (valid, c["HIDDEN"]):
            raise ValueError(
                "MTP previous_hidden must be [T,4096], got "
                f"{tuple(previous_hidden.shape)}"
            )
        if not 1 <= valid <= c["BATCH"]:
            raise ValueError("MTP valid rows must be in [1,16]")
        ids = input_token_ids.to(_I32).flatten()
        mask = active_mask.to(_I32).flatten()
        seq = seq_lens.to(_I32).flatten()
        pos = positions.to(_I32).flatten()
        slots = slot_mapping.to(_I32).flatten()
        table = block_table.to(_I32)
        if ids.numel() != c["BATCH"] or mask.numel() != c["BATCH"]:
            raise ValueError("MTP token ids/active mask must have 16 rows")
        if seq.numel() != c["USER_BATCH"] or pos.numel() != c["USER_BATCH"]:
            raise ValueError("MTP seq_lens/positions must have 16 rows")
        if slots.numel() != c["USER_BATCH"] or table.ndim != 2:
            raise ValueError("MTP slot/block metadata has invalid shape")
        if table.shape[0] != c["USER_BATCH"] or table.numel() != c["BLOCK_TABLE_FLAT"]:
            raise ValueError("MTP block table does not match compiled ABI")
        if not torch.equal(mask, torch.tensor(
            [1] * valid + [0] * (c["BATCH"] - valid), dtype=_I32
        )):
            raise ValueError("MTP active mask must be contiguous [1]*T+[0]*pad")
        if not torch.equal(pos[:valid], seq[:valid] - 1):
            raise ValueError("MTP positions must equal seq_lens-1")
        if torch.any(seq[:valid] <= 0):
            raise ValueError("MTP valid seq_lens must be positive")
        if not torch.equal(seq[valid:], torch.ones(c["BATCH"] - valid, dtype=_I32)):
            raise ValueError("MTP padding seq_lens must be one")
        if torch.count_nonzero(pos[valid:]).item():
            raise ValueError("MTP padding positions must be zero")
        if torch.count_nonzero(ids[valid:]).item():
            raise ValueError("MTP padding token ids must be zero")
        if self.padding_reserve is None:
            raise ValueError(
                "allocator-owned MTP padding reserve is unavailable"
            )
        from tools.step3p5.kv_padding import (  # noqa: PLC0415
            PaddingReserveError,
            validate_fixed_batch_metadata,
        )

        try:
            validate_fixed_batch_metadata(
                seq_lens=seq,
                positions=pos,
                block_table=table,
                slot_mapping=slots,
                valid_rows=valid,
                reserve=self.padding_reserve,
                where="MTP holder",
            )
        except PaddingReserveError as exc:
            raise ValueError(str(exc)) from exc
        self.previous_hidden.zero_()
        self.previous_hidden[:, :valid, :] = previous_hidden
        self.input_token_ids.zero_()
        self.input_token_ids[:, :] = ids
        self.active_mask.zero_()
        self.active_mask[:, :] = mask
        self.seq_lens.copy_(seq.unsqueeze(0).expand(self.tp, -1))
        self.block_table.copy_(table.reshape(1, -1).expand(self.tp, -1))
        self.slot_mapping.copy_(slots.unsqueeze(0).expand(self.tp, -1))
        self.hidden_out.zero_()
        self._current_valid = valid

    def run(self, layer_idx: int):
        if not 0 <= int(layer_idx) < 3:
            raise ValueError("MTP layer_idx must be 0..2")
        if self._runtime is None:
            raise RuntimeError("MTP holder is not entered")
        idx = int(layer_idx)
        t0 = time.time()
        self._runtime.run(self.compiled[idx], *self._args[idx])
        dt = time.time() - t0
        valid = int(self._current_valid)
        out = self.hidden_out[0, :valid, :].clone()
        return {
            "dt": dt,
            "mtp_hidden": out,
            "program": f"mtp_layer_hidden_{idx}",
        }
