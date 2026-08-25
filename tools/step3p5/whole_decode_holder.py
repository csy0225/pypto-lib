# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Resident N=1 whole-net hidden-only decode holder.

把 canonical `models.step3p5.decode_fwd:whole_decode_step3p5` 的 compile -> prepare
-> import_weights -> rt.run 生命周期抽成一个常驻 holder：
**build+prepare 一次**，之后每次 `run()` 复用同一个 prepared `rt`
（不重新 prepare），逐步喂 hidden / attn-meta / KV。

设计边界（对齐 SKILL §H：N=1 单 `@pl.program` 是唯一生产形态）：
- **resident（bind 一次，不随 step 变）**：全部权重（import_ipc DeviceTensor）；
  `PYPTO_H4_RESIDENT` 可把四个 RoPE 表和四个 block-diag gate-R 常量也上传一次。
  生产 holder 不绑定 final norm / LM head。
- **per-step handoff（每次 run 前 mutate 这些 share_memory host tensor）**：
  `current_hidden`、attn-meta（seq_lens/block_table/slot_mapping）、
  KV（IPC add_inout，attention 原地读写）。
- **output**：45 层后的 `next_hidden_out`。final RMSNorm、LM head 和 sampling
  全部由 vLLM 完成。

exporter（权重/KV IPC 池）由调用方管理；holder 假设 exporters 已 ready
（`reuse` 语义），只做 attach。offline harness 和 sidecar 共用本 holder，
保证行为一致、不重复维护。

arg 顺序**必须**与编译出的 program 签名逐字一致（见 __enter__ 的 args）。
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
MAIN_PROGRAM = "whole_decode_step3p5"

# canonical Main 把 loader 的 KEY_WQ_FULL[12] / KEY_WQ_SWA[33] 拆成四个
# attention bucket。full_wq/swa_wq 保留完整物理槽位（dead slot 无害，
# L44@full_wq[11] / L43@swa_wq[32] 的 slot 索引依赖 [12]/[33] 尺寸）；moe_full_wq
# 取 canonical FULL slot 1-10（L4,8,…,40）；moe_swa_wq 取 canonical SWA slot 2-31
# （30 个 MoE-swa 层，跳过 dense L1(slot0)/L2(slot1)/L43(slot32)）。
_MOE_FULL_SLOTS = tuple(range(1, 11))            # 10 full-MoE attn layers
_MOE_SWA_SLOTS = tuple(range(2, 32))             # 30 swa-MoE attn layers

# Step-invariant constants that can stay on their consuming workers instead of
# being staged H2D on every decode step.
_RESIDENT_ROPE_ARGS = ("rope_cf", "rope_sf", "rope_cs", "rope_ss")
_RESIDENT_GATE_ARGS = (
    "gate_r_full",
    "gate_r_swa",
    "gate_r_moe_full",
    "gate_r_moe_swa",
)
_RESIDENT_MODES = {
    "none": (),
    "rope": _RESIDENT_ROPE_ARGS,
    "gate": _RESIDENT_GATE_ARGS,
    "all": _RESIDENT_ROPE_ARGS + _RESIDENT_GATE_ARGS,
}


def _resident_const_args() -> tuple[str, ...]:
    mode = os.environ.get("PYPTO_H4_RESIDENT", "none").strip().lower()
    if mode not in _RESIDENT_MODES:
        raise ValueError(
            f"PYPTO_H4_RESIDENT={mode!r} invalid; expected one of "
            f"{sorted(_RESIDENT_MODES)}"
        )
    return _RESIDENT_MODES[mode]


def _zsh(*shape, dtype=_BF16):
    """DistributedWorker 契约：host tensor 必须 share_memory 且 prepare() 前分配。"""
    return torch.zeros(shape, dtype=dtype).share_memory_()


class WholeDecodeHolder:
    """常驻 N=1 whole-net decode：build+prepare 一次，run() 复用 rt。

    典型用法（offline ctx=1 A/B）::

        h = WholeDecodeHolder(device_ids=[0..7], out_dir="/tmp/n1_weight_ipc",
                              ckpt=CKPT, kv_ipc=True)
        h.build()
        with h:
            h.set_ctx1_token(emb_row)          # 每 step 喂输入
            res = h.run()                       # rt.run + 读 hidden
            print(res["next_hidden"].shape)

    sidecar 用法：build()+__enter__ 常驻；每个 live 请求 set_hidden(...) +
    set_meta(...) + run()。
    """

    def __init__(
        self,
        device_ids,
        out_dir,
        ckpt,
        *,
        platform="a2a3",
        kv_ipc=True,
    ):
        self.device_ids = validate_consecutive_device_ids(
            device_ids,
            owner="whole decode",
        )
        self.tp = len(self.device_ids)
        self.dev_offset = self.device_ids[0]
        self.out_dir = out_dir
        self.ckpt = ckpt
        self.program_name = MAIN_PROGRAM
        self.platform = platform
        self.kv_ipc = kv_ipc

        # populated by build()
        self.compiled = None
        self._cfg = None
        self._dl = None
        self._K = None
        self._consts = {}
        self.hidden_only = True

        # populated by __enter__()
        self._prepare_cm = None
        self._prepare_entered = False
        self.rt = None
        self._wmaps = None
        self._kv_maps = None
        self.padding_reserve = None
        self._args_list = None
        # resident host tensors (mutated per-step).  Their leading batch
        # dimension is static storage capacity; active rows are supplied by
        # set_live_step/num_tokens_per_owner for each invocation.
        self.current_hidden = None
        self.num_tokens_per_owner = None
        self.gate_r_full = self.gate_r_swa = None
        self.seq_lens = None
        self.block_table = None
        self.slot_mapping = None
        self.rope_cf = self.rope_sf = self.rope_cs = self.rope_ss = None
        self.k_cache = self.v_cache = None
        self._next_hidden_out = None
        self._last_run_sec = 0.0
        self._rope_ready = False
        self._resident_consts = {}
        self._resident_const_hosts = {}

    def _infer_rows_from_map(self) -> None:
        """从 vLLM-owned Main map 推导 physical flat KV rows。

        ``kv_cache_config.num_blocks`` 是 scheduler domain，不能拿来作为
        PyPTO physical tensor 的容量。map 已经记录了 reserve 扩容后的
        physical capacity，因此必须在导入 cfg/compile 前读取它。
        """
        path = os.path.join(self.out_dir, "pypto_kvpool_map.json.rank0")
        try:
            with open(path, encoding="utf-8") as file:
                obj = json.load(file)
            rows = int(obj["map"]["L0.K"]["num_slots"])
            physical = int(obj["physical_num_blocks"])
            scheduler = int(obj["scheduler_num_blocks"])
            from tools.step3p5.kv_padding import STORAGE_BATCH  # noqa: PLC0415
            required_physical = scheduler + STORAGE_BATCH - 1
            if physical < required_physical:
                raise ValueError(
                    f"Main KV map physical={physical} lacks capacity-derived "
                    f"reserve above scheduler={scheduler}; required>={required_physical}"
                )
        except FileNotFoundError:
            return
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid Main KV padding-reserve map {path}: {exc}"
            ) from exc
        expected = 45 * rows
        configured = os.environ.get("PYPTO_STEP3P5_KV_CACHE_ROWS")
        if configured is not None and int(configured) != expected:
            raise ValueError(
                "PYPTO_STEP3P5_KV_CACHE_ROWS disagrees with Main IPC map: "
                f"configured={configured}, expected={expected}"
            )
        os.environ["PYPTO_STEP3P5_KV_CACHE_ROWS"] = str(expected)

    # ---- build (compile; no device prepare yet) ----------------------------

    def _main_const(self, name: str) -> int:
        """读取 canonical Main 模块的模块级常量。"""
        return int(getattr(self._dl, name))

    def build(self):
        """编译 program + 解析常量/VOCAB。无 device prepare（那在 __enter__）。"""
        from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415
        set_backend_type(BackendType.Ascend910B)
        if self.kv_ipc:
            self._infer_rows_from_map()
        import models.step3p5.config as cfg  # noqa: PLC0415
        from models.step3p5 import weight_loader as K  # noqa: PLC0415
        self._cfg = cfg
        self._K = K
        if self.tp != cfg.TP_WORLD_SIZE:
            raise ValueError(
                f"whole decode holder requires TP={cfg.TP_WORLD_SIZE}, "
                f"got {self.tp}"
            )
        import models.step3p5.decode_fwd as dl  # noqa: PLC0415
        self._dl = dl

        from pypto import ir  # noqa: PLC0415
        from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415
        os.environ.setdefault(
            "PYPTO_PROG_BUILD_DIR",
            "/data/chensiyu/hw_project/pypto/workspace/build_output",
        )
        _mplan = None
        if os.environ.get("PYPTO_MEM_PLANNER", "").lower() == "ptoas":
            from pypto.pypto_core import passes as _passes  # noqa: PLC0415
            _mplan = _passes.MemoryPlanner.PTOAS
        self.compiled = ir.compile(
            dl.whole_decode_step3p5, platform=self.platform,
            distributed_config=DistributedConfig(device_ids=self.device_ids, num_sub_workers=0),
            skip_ptoas=False, dump_passes=False, memory_planner=_mplan,
        )
        print(f"[holder] compile OK => {self.compiled.output_dir}", flush=True)

        def C(name):
            v = getattr(dl, name, None)
            if v is None:
                v = getattr(cfg, name)
            return int(v)
        self._C = C
        self._consts = dict(
            HIDDEN=cfg.HIDDEN, HEAD_DIM=cfg.HEAD_DIM, BATCH=cfg.BATCH,
            UBD=C("USER_BATCH_DYN"), BTF=C("BLOCK_TABLE_FLAT_DYN"),
            RSD=C("ROPE_SEQ_DYN"), KVC=C("KV_CACHE_ROWS_DYN"),
            ROT_FULL=cfg.ROTARY_HALF_FULL * 2, ROT_SWA=cfg.ROTARY_HALF_SWA * 2,
            NHF_PAD=C("NUM_HEADS_FULL_LOCAL_PAD"), NHS_PAD=C("NUM_HEADS_SWA_LOCAL_PAD"),
            HQ_FULL=C("HIDDEN_Q_FULL_LOCAL"), HQ_SWA=C("HIDDEN_Q_SWA_LOCAL"),
        )
        n_full = sum(1 for li in range(cfg.NUM_HIDDEN_LAYERS) if cfg.is_full_attention(li))
        self._consts["N_FULL"] = n_full
        self._consts["N_SWA"] = cfg.NUM_HIDDEN_LAYERS - n_full
        if self.kv_ipc and int(os.environ.get("PYPTO_STEP3P5_KV_GROUP_COUNT", "1")) != 1:
            raise ValueError(
                "the current generated whole-net accepts one KV metadata group; "
                "start the first live bring-up with "
                "--disable-hybrid-kv-cache-manager"
            )

        return self

    # ---- resident prepare + arg wiring -------------------------------------

    def __enter__(self):
        assert self.compiled is not None, "call build() before entering holder"
        if self._prepare_cm is not None or self.rt is not None:
            raise RuntimeError(
                "holder cleanup is incomplete; retry __exit__ before re-entering"
            )
        resident_const_names = _resident_const_args()
        try:
            return self._enter_impl(resident_const_names)
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def _enter_impl(self, resident_const_names):
        c = self._consts
        tp = self.tp
        HIDDEN, HEAD_DIM, BATCH = c["HIDDEN"], c["HEAD_DIM"], c["BATCH"]
        UBD, BTF, RSD, KVC = c["UBD"], c["BTF"], c["RSD"], c["KVC"]
        ROT_FULL, ROT_SWA = c["ROT_FULL"], c["ROT_SWA"]
        NHF_PAD, NHS_PAD = c["NHF_PAD"], c["NHS_PAD"]
        HQ_FULL, HQ_SWA = c["HQ_FULL"], c["HQ_SWA"]
        N_FULL, N_SWA = c["N_FULL"], c["N_SWA"]

        # resident host tensors (allocated BEFORE prepare so forked chips see them)
        self.current_hidden = _zsh(tp, BATCH, HIDDEN)
        # G1: one shared runtime active-token count per owner.  The canonical
        # graph takes max(owner counts) so every rank uses identical dynamic
        # MoE bounds.
        self.num_tokens_per_owner = torch.zeros(
            (128,), dtype=_I32,
        ).share_memory_()
        self.num_tokens_per_owner[:tp].fill_(BATCH)
        self.gate_r_full = _zsh(tp, N_FULL, NHF_PAD, HQ_FULL)
        self.gate_r_swa = _zsh(tp, N_SWA, NHS_PAD, HQ_SWA)
        # canonical loop-form 4-bucket split：full/swa 各对应 dense+post-loop
        # 全量，moe_full/moe_swa 是 MoE loop 专用桶，各自拥有 block-diag R。
        n_moe_full = self._main_const("NUM_FULL_MOE_LAYERS")
        n_moe_swa = self._main_const("NUM_SWA_MOE_LAYERS")
        self.gate_r_moe_full = _zsh(tp, n_moe_full, NHF_PAD, HQ_FULL)
        self.gate_r_moe_swa = _zsh(tp, n_moe_swa, NHS_PAD, HQ_SWA)
        for _h in range(HQ_FULL // HEAD_DIM):
            self.gate_r_moe_full[:, :, _h, _h * HEAD_DIM:(_h + 1) * HEAD_DIM] = 1.0
        for _h in range(HQ_SWA // HEAD_DIM):
            self.gate_r_moe_swa[:, :, _h, _h * HEAD_DIM:(_h + 1) * HEAD_DIM] = 1.0
        # block-diag R constant (layer-independent): R[h, h*HEAD_DIM+d]=1 for real
        # local heads (count = HQ//HEAD_DIM), padded rows stay zero.
        for _h in range(HQ_FULL // HEAD_DIM):
            self.gate_r_full[:, :, _h, _h * HEAD_DIM:(_h + 1) * HEAD_DIM] = 1.0
        for _h in range(HQ_SWA // HEAD_DIM):
            self.gate_r_swa[:, :, _h, _h * HEAD_DIM:(_h + 1) * HEAD_DIM] = 1.0
        self.seq_lens = torch.ones(tp, UBD, dtype=_I32).share_memory_()
        self.block_table = torch.zeros(tp, BTF, dtype=_I32).share_memory_()
        self.slot_mapping = torch.arange(UBD, dtype=_I32).unsqueeze(0).repeat(tp, 1).contiguous().share_memory_()
        self.rope_cf, self.rope_sf = _zsh(tp, RSD, ROT_FULL, dtype=_F32), _zsh(tp, RSD, ROT_FULL, dtype=_F32)
        self.rope_cs, self.rope_ss = _zsh(tp, RSD, ROT_SWA, dtype=_F32), _zsh(tp, RSD, ROT_SWA, dtype=_F32)
        self.k_cache, self.v_cache = _zsh(tp, KVC, HEAD_DIM), _zsh(tp, KVC, HEAD_DIM)
        self._next_hidden_out = _zsh(tp, BATCH, HIDDEN)
        from tools.step3p5.pypto_weight_ipc import (  # noqa: PLC0415
            import_weights_all, build_stacked_weight,
        )
        K = self._K

        # Keep one Worker.run and one CommDomain resident across decode steps.
        # The runtime clears every retained window before each request, so C1
        # layer-local signal epochs restart at 1 without stale Ge thresholds
        # or repeated HCCL domain allocation/release churn.
        self._prepare_runtime()
        self._wmaps = import_weights_all(self.rt, self.out_dir, tp=tp, dev_offset=self.dev_offset)

        if self.kv_ipc:
            from tools.step3p5.pypto_kv_ipc import (  # noqa: PLC0415
                build_stacked_kv_pool,
                import_kv_all,
            )

            self._kv_maps = import_kv_all(
                self.rt, self.out_dir, tp=tp, dev_offset=self.dev_offset,
            )
            from tools.step3p5.kv_padding import (  # noqa: PLC0415
                make_padding_reserve,
            )

            summary = self._kv_maps[0].summary
            self.padding_reserve = make_padding_reserve(
                summary.scheduler_num_blocks,
                summary.physical_num_blocks,
                block_size=summary.block_size,
                storage_capacity=BATCH,
            )
            self.k_cache, self.v_cache = build_stacked_kv_pool(self._kv_maps)
            imported_rows = int(self._kv_maps[0].section_spec("K")[1][0])
            if imported_rows != KVC:
                raise ValueError(
                    "compiled KV_CACHE_ROWS_DYN does not match imported vLLM KV: "
                    f"compiled={KVC}, imported={imported_rows}; set "
                    "PYPTO_STEP3P5_KV_CACHE_ROWS=45*num_blocks*128 before "
                    "compiling the live variant"
                )
            print(
                f"[holder] KV cache via IPC: flat K/V rows={imported_rows}",
                flush=True,
            )

        self._initialize_rope_tables()
        self._resident_consts = self._make_resident_consts(resident_const_names)

        # arg order MUST match the compiled program signature (see harness _do_worker)
        def W(key):
            return build_stacked_weight(self._wmaps, key)

        def Wsub(key, slots):
            """canonical 4-bucket split：从 loader 整桶 [N,...] 里按 leading-dim
            slots 取零拷贝连续子视图（每 rank 独立切片，再叠成 StackedDeviceTensor）。

            KEY_WQ_FULL=[12] 全 12 个 FULL 层紧凑叠放；moe_full_wq[10] 取
            slot 1-10（L4,8,…,40；slot0=L0 dense、slot11=L44 post-loop 留在
            full_wq 桶里）。KEY_WQ_SWA=[33] 同理，moe_swa_wq[30] 取 slot 2-31。
            DeviceTensor.__getitem__ 只允许 outermost partial slice（连续），满足。
            """
            from pypto.runtime.device_tensor import StackedDeviceTensor  # noqa: PLC0415
            start, stop = slots[0], slots[-1] + 1
            shards = [
                self._wmaps[r].device_tensor_slice(key, start, stop) for r in range(tp)
            ]
            full = (tp, *tuple(shards[0].shape))
            return StackedDeviceTensor(shards, full, list(range(tp)))

        args = self._build_loop_form_args(W, Wsub)
        self._args_list = args
        print(
            f"[holder] resident: built {len(args)} args; "
            f"program={self.program_name}",
            flush=True,
        )
        return self

    def _prepare_runtime(self):
        """Enter the prepared worker under its current context-manager contract.

        ``DistributedCompiledProgram.prepare()`` currently returns a
        ``DistributedWorker`` whose ``__enter__`` is a non-throwing identity
        return; construction has already completed setup and owns any failure
        cleanup. We still journal the CM before entering it so a future
        partial-enter implementation can be closed/retried explicitly.
        """
        prepare_cm = self.compiled.prepare(persistent=True)
        self._prepare_cm = prepare_cm
        self._prepare_entered = False
        runtime = prepare_cm.__enter__()
        self._prepare_entered = True
        self.rt = runtime
        return runtime

    def _build_loop_form_args(self, W, Wsub):
        """Canonical loop-form 53-arg host_orch arg-list
        （严格按 models/step3p5/decode_fwd.py 签名）。

        canonical Main 只有一个 Out（next_hidden_out）。
        attention 采用 full/swa/moe_full/moe_swa 四个 bucket；MoE bucket
        从 resident weight pool 取连续 zero-copy 子视图。
        常量名读取 canonical Main 模块（N_FULL_ATTN_LAYERS 等）。
        """
        K = self._K
        args = [self.current_hidden]
        # 5 activations / RMS
        args += [W(K.KEY_INPUT_RMS), W(K.KEY_POST_ATTN_RMS), W(K.KEY_Q_NORM), W(K.KEY_K_NORM)]
        # dense full-attention weights + block-diag gate R
        args += [W(K.KEY_WQ_FULL), W(K.KEY_WK_FULL), W(K.KEY_WV_FULL), W(K.KEY_WO_FULL), W(K.KEY_WG_FULL),
                 self.gate_r_full]
        # dense SWA weights + block-diag gate R
        args += [W(K.KEY_WQ_SWA), W(K.KEY_WK_SWA), W(K.KEY_WV_SWA), W(K.KEY_WO_SWA), W(K.KEY_WG_SWA),
                 self.gate_r_swa]
        # dense MLP
        args += [W(K.KEY_DENSE_GATE), W(K.KEY_DENSE_UP), W(K.KEY_DENSE_DOWN)]
        # MoE full-attention bucket
        args += [Wsub(K.KEY_WQ_FULL, _MOE_FULL_SLOTS),
                 Wsub(K.KEY_WK_FULL, _MOE_FULL_SLOTS),
                 Wsub(K.KEY_WV_FULL, _MOE_FULL_SLOTS),
                 Wsub(K.KEY_WO_FULL, _MOE_FULL_SLOTS),
                 Wsub(K.KEY_WG_FULL, _MOE_FULL_SLOTS),
                 self.gate_r_moe_full]
        # MoE SWA bucket
        args += [Wsub(K.KEY_WQ_SWA, _MOE_SWA_SLOTS),
                 Wsub(K.KEY_WK_SWA, _MOE_SWA_SLOTS),
                 Wsub(K.KEY_WV_SWA, _MOE_SWA_SLOTS),
                 Wsub(K.KEY_WO_SWA, _MOE_SWA_SLOTS),
                 Wsub(K.KEY_WG_SWA, _MOE_SWA_SLOTS),
                 self.gate_r_moe_swa]
        # MoE expert/router weights
        args += [W(K.KEY_MOE_GATE_W_NK), W(K.KEY_MOE_ROUTER_BIAS),
                 W(K.KEY_MOE_W13_R), W(K.KEY_MOE_W13_R_SCALE),
                 W(K.KEY_MOE_W_DOWN_R), W(K.KEY_MOE_W_DOWN_R_SCALE),
                 W(K.KEY_MOE_W_GATE_S_NK),
                 W(K.KEY_MOE_W_UP_S_NK), W(K.KEY_MOE_W_DOWN_S)]
        # KV/RoPE metadata
        args += [self.seq_lens, self.block_table, self.slot_mapping,
                 self.rope_cf, self.rope_sf, self.rope_cs, self.rope_ss]
        # InOut cache
        args += [self.k_cache, self.v_cache]
        # canonical output
        args += [self._next_hidden_out]
        # G1 runtime active-token counts.  This remains a tensor argument so
        # the long-lived holder can update it before every rt.run().
        args += [self.num_tokens_per_owner]
        return args

    def __exit__(self, exc_type, exc, tb):
        cleanup_error = None
        runtime = self.rt
        residents_clean = True
        if self._resident_consts and runtime is not None:
            remaining = {}
            for name, stacked in reversed(tuple(self._resident_consts.items())):
                try:
                    runtime.free_stacked_tensor(stacked)
                except BaseException as err:  # noqa: BLE001
                    print(
                        f"[holder] free resident constant {name} failed: {err}",
                        flush=True,
                    )
                    if cleanup_error is None:
                        cleanup_error = err
                    remaining[name] = stacked
                else:
                    self._restore_resident_host(name)
            self._resident_consts = remaining
            residents_clean = not remaining
        elif self._resident_consts:
            cleanup_error = RuntimeError(
                "resident constants remain owned without a prepared runtime"
            )
            print(f"[holder] {cleanup_error}", flush=True)
            residents_clean = False
        prepare_cm = self._prepare_cm
        if prepare_cm is None and self.rt is not None:
            cleanup_error = cleanup_error or RuntimeError(
                "prepared runtime exists without its context manager"
            )
            print(f"[holder] {cleanup_error}", flush=True)
            residents_clean = False
        # Keep the prepared worker alive when any resident free failed. The
        # backend may admit the same free on a later attempt; closing it here
        # would make that retry impossible. All resident frees were attempted
        # above, so this remains best-effort without sacrificing retryability.
        if prepare_cm is not None and residents_clean:
            try:
                if self._prepare_entered:
                    prepare_cm.__exit__(exc_type, exc, tb)
                else:
                    close = getattr(prepare_cm, "close", None)
                    if close is None:
                        raise RuntimeError(
                            "prepared context manager failed before __enter__ "
                            "and exposes no close()"
                        )
                    close()
            except BaseException as err:  # noqa: BLE001
                print(f"[holder] prepared runtime cleanup failed: {err}", flush=True)
                if cleanup_error is None:
                    cleanup_error = err
            else:
                self._finalize_closed_state()
        elif prepare_cm is None and self.rt is None and residents_clean:
            self._finalize_closed_state()

        if exc_type is None and cleanup_error is not None:
            raise cleanup_error
        return False

    def _restore_resident_host(self, name):
        host_tensor = self._resident_const_hosts.pop(name, None)
        if host_tensor is not None:
            setattr(self, name, host_tensor)
        elif name in _RESIDENT_ROPE_ARGS + _RESIDENT_GATE_ARGS:
            # A defensive fallback for manually injected/legacy ownership
            # journals that predate _resident_const_hosts.
            setattr(self, name, None)

    def _finalize_closed_state(self):
        """Drop runtime-owned state after the prepared worker is terminally closed."""
        for name in tuple(self._resident_const_hosts):
            self._restore_resident_host(name)
        self._resident_consts = {}
        self._resident_const_hosts = {}
        self._prepare_cm = None
        self._prepare_entered = False
        self.rt = None
        self._wmaps = None
        self._kv_maps = None
        self.padding_reserve = None
        self._args_list = None
        # __enter__ allocates fresh host tables on every lifecycle. A stale
        # ready bit would skip their rebuild and leave zero-filled RoPE buffers.
        self._rope_ready = False

    # ---- per-step input setters --------------------------------------------

    def set_ctx1_token(self, emb_row, *, fill_batch=False):
        """offline ctx=1 A/B：把 embed(token) 灌进 row0（每 rank 复制）+ 位置0 identity rope。"""
        self.current_hidden.zero_()
        self.current_hidden[:, 0, :] = emb_row
        if fill_batch:
            self.current_hidden[:, :, :] = emb_row
        if self.num_tokens_per_owner is not None:
            self.num_tokens_per_owner[: self.tp].fill_(
                self._consts["BATCH"] if fill_batch else 1
            )

    def set_hidden(self, hidden):
        """Set a full-capacity hidden buffer for offline/sidecar callers.

        Live variable-batch callers should use :meth:`set_live_step`, which
        accepts active rows and updates the runtime token metadata separately.
        Rank-specific input must be TP-replicated.
        """
        self.current_hidden.zero_()
        if hidden.dim() == 2:
            expected = (self._consts["BATCH"], self._consts["HIDDEN"])
            if tuple(hidden.shape) != expected:
                raise ValueError(
                    f"set_hidden expects full storage shape {expected}, "
                    f"got {tuple(hidden.shape)}"
                )
            self.current_hidden[:, :, :] = hidden.to(_BF16)
        else:
            expected = (
                self.tp,
                self._consts["BATCH"],
                self._consts["HIDDEN"],
            )
            if tuple(hidden.shape) != expected:
                raise ValueError(
                    f"set_hidden expects TP storage shape {expected}, "
                    f"got {tuple(hidden.shape)}"
                )
            hidden_bf16 = hidden.to(_BF16)
            if any(
                not torch.equal(hidden_bf16[rank], hidden_bf16[0])
                for rank in range(1, self.tp)
            ):
                raise ValueError(
                    "replicated-input local-owner MoE requires identical "
                    "hidden values on every TP rank"
                )
            self.current_hidden.copy_(hidden_bf16)
        if self.num_tokens_per_owner is not None:
            self.num_tokens_per_owner[: self.tp].fill_(self._consts["BATCH"])

    def _make_resident_consts(self, names=None):
        """Upload selected step-invariant constants once and rebind their args."""
        if names is None:
            names = _resident_const_args()
        if not names:
            return {}
        if self.rt is None:
            raise ValueError("_make_resident_consts requires a prepared rt")
        if not self._rope_ready:
            raise ValueError("_make_resident_consts requires filled RoPE tables")

        owned = {}
        host_tensors = {}
        bytes_per_rank = 0
        try:
            for name in names:
                host_tensor = getattr(self, name)
                if not isinstance(host_tensor, torch.Tensor):
                    raise ValueError(
                        f"resident constant {name} must be a host tensor, "
                        f"got {type(host_tensor).__name__}"
                    )
                if host_tensor.shape[0] != self.tp:
                    raise ValueError(
                        f"resident constant {name} leading dim "
                        f"{host_tensor.shape[0]} != tp {self.tp}"
                    )
                stacked = self.rt.alloc_stacked_tensor(host_tensor)
                host_tensors[name] = host_tensor
                owned[name] = stacked
                self._resident_const_hosts[name] = host_tensor
                setattr(self, name, stacked)
                bytes_per_rank += host_tensor[0].numel() * host_tensor.element_size()
        except BaseException:
            # Keep failed frees journaled so WholeDecodeHolder.__enter__ can
            # retry them before tearing down the prepared runtime. The upload
            # exception remains the one re-raised below; cleanup errors are
            # deliberately secondary.
            remaining = {}
            self._resident_consts = remaining
            for name, stacked in reversed(tuple(owned.items())):
                try:
                    self.rt.free_stacked_tensor(stacked)
                except BaseException as err:  # noqa: BLE001
                    print(
                        f"[holder] rollback resident constant {name} failed: {err}",
                        flush=True,
                    )
                    remaining[name] = stacked
                finally:
                    if name not in remaining:
                        self._resident_const_hosts.pop(name, None)
                        setattr(self, name, host_tensors[name])
            raise

        print(
            f"[holder] resident constants "
            f"({os.environ.get('PYPTO_H4_RESIDENT', 'none')}): "
            f"{len(owned)} args, {bytes_per_rank / (1024 * 1024):.2f} MiB/rank",
            flush=True,
        )
        return owned

    def set_meta(self, *, seq_lens=None, block_table=None, slot_mapping=None,
                 rope_cf=None, rope_sf=None, rope_cs=None, rope_ss=None):
        """只允许更新诊断 RoPE；paged-KV metadata 必须走 set_live_step。

        seq/block/slot 缺少 positions、valid_rows 和 allocator reserve 上下文，
        不能满足生产 fail-closed 契约，因此禁止通过这个旧入口覆盖。
        """
        if any(
            value is not None
            for value in (seq_lens, block_table, slot_mapping)
        ):
            raise ValueError(
                "paged-KV metadata must be updated through set_live_step so "
                "the allocator-owned padding reserve can be validated"
            )
        if any(
            value is not None
            for value in (rope_cf, rope_sf, rope_cs, rope_ss)
        ) and any(name in self._resident_consts for name in _RESIDENT_ROPE_ARGS):
            raise ValueError(
                "RoPE tables are worker-resident constants; rebuild the holder "
                "instead of mutating an unbound host copy"
            )
        for dst, src in ((self.rope_cf, rope_cf), (self.rope_sf, rope_sf),
                         (self.rope_cs, rope_cs), (self.rope_ss, rope_ss)):
            if src is not None:
                dst.copy_(src.to(_F32))

    def _initialize_rope_tables(self):
        """Build resident full/SWA RoPE tables exactly once."""
        if self._rope_ready:
            return
        from models.step3p5._ops import (  # noqa: PLC0415
            build_llama3_yarn_rope_tables,
            build_plain_rope_tables,
        )

        cfg = self._cfg
        rsd = self._consts["RSD"]
        full_cos, full_sin = build_llama3_yarn_rope_tables(
            rsd,
            cfg.ROTARY_HALF_FULL * 2,
            cfg.LAYER_ROPE_THETA[0],
            factor=cfg.ROPE_SCALING["factor"],
            low=cfg.ROPE_SCALING["low_freq_factor"],
            high=cfg.ROPE_SCALING["high_freq_factor"],
            orig_max=cfg.ROPE_SCALING["original_max_position_embeddings"],
        )
        swa_cos, swa_sin = build_plain_rope_tables(
            rsd,
            cfg.ROTARY_HALF_SWA * 2,
            cfg.LAYER_ROPE_THETA[1],
        )
        for rank in range(self.tp):
            self.rope_cf[rank].copy_(full_cos)
            self.rope_sf[rank].copy_(full_sin)
            self.rope_cs[rank].copy_(swa_cos)
            self.rope_ss[rank].copy_(swa_sin)
        self._rope_ready = True

    def set_live_step(
        self,
        hidden,
        *,
        seq_lens,
        positions,
        block_table,
        slot_mapping,
    ):
        """Set one pure-decode step and initialize every padded row."""
        self._initialize_rope_tables()
        if hidden.dtype != _BF16 or hidden.ndim != 2:
            raise ValueError(
                f"live hidden must be BF16 [T,H], got "
                f"{hidden.dtype} {tuple(hidden.shape)}"
            )
        valid_tokens = int(hidden.shape[0])
        batch = self._consts["BATCH"]
        if valid_tokens > batch:
            raise ValueError(
                f"active batch/tokens={valid_tokens} exceeds storage capacity {batch}"
            )
        hidden_size = self._consts["HIDDEN"]
        user_batch = self._consts["UBD"]
        if not (1 <= valid_tokens <= batch) or hidden.shape[1] != hidden_size:
            raise ValueError(
                f"live hidden must be [1..{batch},{hidden_size}], "
                f"got {tuple(hidden.shape)}"
            )

        seq = seq_lens.to(_I32).flatten()
        pos = positions.to(_I32).flatten()
        slots = slot_mapping.to(_I32).flatten()
        table = block_table.to(_I32)
        if tuple(seq.shape) != (user_batch,) or tuple(pos.shape) != (user_batch,):
            raise ValueError(f"seq_lens/positions must be [{user_batch}]")
        if tuple(slots.shape) != (user_batch,):
            raise ValueError(f"slot_mapping must be [{user_batch}]")
        if table.ndim != 2 or table.shape[0] != user_batch:
            raise ValueError("block_table must be [storage_batch,max_blocks]")
        if table.numel() != self._consts["BTF"]:
            raise ValueError(
                f"block_table has {table.numel()} elements, compiled ABI needs "
                f"{self._consts['BTF']}"
            )
        if torch.any(seq[:valid_tokens] <= 0):
            raise ValueError("valid seq_lens must be positive")
        if not torch.equal(pos[:valid_tokens], seq[:valid_tokens] - 1):
            raise ValueError("valid positions must equal seq_lens-1")
        if not torch.equal(
            seq[valid_tokens:],
            torch.ones(user_batch - valid_tokens, dtype=_I32),
        ):
            raise ValueError("seq_lens padding must be one")
        if torch.count_nonzero(pos[valid_tokens:]).item():
            raise ValueError("positions padding must be zero")
        if self.padding_reserve is None:
            raise ValueError(
                "allocator-owned Main padding reserve is unavailable"
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
                valid_rows=valid_tokens,
                reserve=self.padding_reserve,
                where="Main holder",
            )
        except PaddingReserveError as exc:
            raise ValueError(str(exc)) from exc
        if int(seq[:valid_tokens].max()) > self._consts["RSD"]:
            raise ValueError("requested position exceeds resident RoPE table")

        self.current_hidden.zero_()
        self.current_hidden[:, :valid_tokens, :] = hidden
        self.num_tokens_per_owner[: self.tp].fill_(valid_tokens)
        self.seq_lens.copy_(seq.unsqueeze(0).expand(self.tp, -1))
        self.block_table.copy_(table.reshape(1, -1).expand(self.tp, -1))
        self.slot_mapping.copy_(slots.unsqueeze(0).expand(self.tp, -1))

    # ---- run ----------------------------------------------------------------

    def run(self):
        """复用常驻 rt 跑一次 whole-net forward，只返回 raw hidden。"""
        if self.rt is None:
            raise RuntimeError("enter holder (with h:) before run()")
        self._validate_replicated_owner_counts()
        t0 = time.time()
        # DFX capture (PERF-A1 baseline). N1_DFX = tokens; N1_PMU = int event type.
        #   "swim"/"l2" -> enable_l2_swimlane (dfx_outputs/l2_swimlane_records.json
        #                  + merged_swimlane_*.json = per-task/per-layer timing)
        #   "pmu"       -> enable_pmu = int(N1_PMU or 1)  (dfx_outputs/pmu.csv)
        #   "dep"       -> enable_dep_gen ; "scope" -> enable_scope_stats
        # Artifacts under {self.compiled.output_dir}/dfx_outputs/. Swimlane and PMU
        # perturb timing -> collect in SEPARATE run() calls from clean wallclock.
        _dfx = os.environ.get("N1_DFX", "")
        if _dfx:
            from pypto.runtime.runner import RunConfig  # noqa: PLC0415
            _pmu = int(os.environ.get("N1_PMU", "1")) if "pmu" in _dfx else 0
            _rc = RunConfig(platform=self.platform,
                            enable_dep_gen=("dep" in _dfx),
                            enable_scope_stats=("scope" in _dfx),
                            enable_l2_swimlane=("swim" in _dfx or "l2" in _dfx),
                            enable_pmu=_pmu)
            self.rt.run(self.compiled, *self._args_list, config=_rc)
        else:
            self.rt.run(self.compiled, *self._args_list)
        self._last_run_sec = time.time() - t0
        result = {"next_hidden": self._next_hidden_out}
        return result

    def _validate_replicated_owner_counts(self) -> int:
        """Reject host ABI inputs that violate replicated-input semantics."""
        if self.num_tokens_per_owner is None:
            raise RuntimeError("owner-count storage is unavailable")
        counts = self.num_tokens_per_owner[: self.tp].to(torch.int64)
        active_batch = int(counts[0].item())
        storage_batch = int(self._consts["BATCH"])
        if not 1 <= active_batch <= storage_batch:
            raise ValueError(
                "active batch must be within "
                f"[1, {storage_batch}], got {active_batch}"
            )
        if not bool(torch.all(counts == active_batch)):
            raise ValueError(
                "replicated-input local-owner MoE requires identical "
                "num_tokens_per_owner on every TP rank, got "
                f"{counts.tolist()}"
            )
        return active_batch
