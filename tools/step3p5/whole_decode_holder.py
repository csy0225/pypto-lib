# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Resident N=1 whole-net decode holder.

把 `whole_decode_faithful_real` 的 compile -> prepare -> import_weights ->
rt.run 生命周期抽成一个常驻 holder：**build+prepare 一次**，之后每次
`run()` 复用同一个 prepared `rt`（不重新 prepare），逐步喂 hidden / attn-meta /
KV。这是 N=1 整网接进 vLLM serving（live single-handoff sidecar）的地基。

设计边界（对齐 SKILL §H：N=1 单 `@pl.program` 是唯一生产形态）：
- **resident（bind 一次，不随 step 变）**：全部权重（import_ipc DeviceTensor）、
  `gate_r_full/swa`（block-diag R 常量）、`final_norm`、`lm_head`。
- **per-step handoff（每次 run 前 mutate 这些 share_memory host tensor）**：
  `current_hidden`、attn-meta（seq_lens/block_table/slot_mapping/rope_*）、
  KV（IPC add_inout，attention 原地读写）。
- **output**：`logits_shard_out` -> 跨 rank concat -> argmax = next token。

exporter（权重/KV IPC 池）由调用方管理；holder 假设 exporters 已 ready
（`reuse` 语义），只做 attach。offline harness 和 sidecar 共用本 holder，
保证行为一致、不重复维护。

arg 顺序**必须**与编译出的 program 签名逐字一致（见 __enter__ 的 args）。
"""
from __future__ import annotations
import ctypes

import json
import os
import time

import torch

_BF16 = torch.bfloat16
_F32 = torch.float32
_I32 = torch.int32


def _zsh(*shape, dtype=_BF16):
    """DistributedWorker 契约：host tensor 必须 share_memory 且 prepare() 前分配。"""
    return torch.zeros(shape, dtype=dtype).share_memory_()


def _maybe_seed_live_kv_env(out_dir: str) -> None:
    """Seed compile-time KV row envs from a vLLM-exported KV map if present."""
    map_path = os.path.join(out_dir, "pypto_kvpool_map.json.rank0")
    if not os.path.exists(map_path):
        return
    with open(map_path, encoding="utf-8") as file:
        pool_map = json.load(file)
    entry = pool_map.get("map", {}).get("L0.K")
    if not isinstance(entry, dict):
        return
    num_slots = int(entry["num_slots"])
    num_layers = int(pool_map.get("num_layers", 45))
    os.environ.setdefault("PYPTO_STEP3P5_KV_CACHE_ROWS", str(num_layers * num_slots))
    os.environ.setdefault("PYPTO_STEP3P5_MTP_KV_CACHE_ROWS", str(num_slots))
    # Do NOT infer PYPTO_STEP3P5_MAX_SEQ from num_slots: KV capacity can be much
    # larger than model max context and would explode RoPE/block-table buffers.
    # Launchers must set MAX_SEQ/ROPE_SEQ explicitly when they need >4096.


class WholeDecodeHolder:
    """常驻 N=1 whole-net decode：build+prepare 一次，run() 复用 rt。

    典型用法（offline ctx=1 A/B）::

        h = WholeDecodeHolder(device_ids=[0..7], out_dir="/tmp/n1_weight_ipc",
                              ckpt=CKPT, kv_ipc=True)
        h.build()
        with h:
            h.set_ctx1_token(emb_row)          # 每 step 喂输入
            res = h.run()                       # rt.run + 读 logits
            print(res["argmax"])                # 期望 303

    sidecar 用法：build()+__enter__ 常驻；每个 live 请求 set_hidden(...) +
    set_meta(...) + run()。
    """

    def __init__(self, device_ids, out_dir, ckpt, *,
                 layer_name="whole_decode_faithful_real_single_chip",
                 platform="a2a3", kv_ipc=True):
        self.device_ids = list(device_ids)
        self.tp = len(self.device_ids)
        self.dev_offset = self.device_ids[0]
        self.out_dir = out_dir
        self.ckpt = ckpt
        self.layer_name = layer_name
        self.platform = platform
        self.kv_ipc = kv_ipc

        # populated by build()
        self.compiled = None
        self.mtp_compiled = None
        self._cfg = None
        self._dl = None
        self._K = None
        self._consts = {}
        self.VOCAB_LOCAL = None

        # populated by __enter__()
        self._prepare_cm = None
        self.rt = None
        self._wmaps = None
        self._kv_maps = None
        self._args_list = None
        self._run_gen = 0  # cumulative run() count for barrier generation-scaling (E2)
        self._mtp_args_list = None
        # resident host tensors (mutated per-step)
        self.current_hidden = None
        self.gate_r_full = self.gate_r_swa = None
        self.seq_lens = None
        self.block_table = None
        self.slot_mapping = None
        self.rope_cf = self.rope_sf = self.rope_cs = self.rope_ss = None
        self.k_cache = self.v_cache = None
        self.h_mid_out = self.next_hidden_out = None
        self.dbg_out = None
        self.logits_shard_out = None
        self._rope_ready = False
        self.mtp_previous_hidden = None
        self.mtp_first_token_ids = None
        self.mtp_active_mask = None
        self.mtp_seq_lens = None
        self.mtp_block_table = None
        self.mtp_slot_mapping = None
        self.mtp_gate_r = None
        self.mtp_k_cache = None
        self.mtp_v_cache = None
        self.mtp_hidden_out = None
        self.mtp_logits_out = None
        self.mtp_draft_token_ids_out = None

    # ---- build (compile; no device prepare yet) ----------------------------

    def build(self):
        """编译 program + 解析常量/VOCAB。无 device prepare（那在 __enter__）。"""
        if self.kv_ipc:
            _maybe_seed_live_kv_env(self.out_dir)
        from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415
        set_backend_type(BackendType.Ascend910B)
        import models.step3p5.config as cfg  # noqa: PLC0415
        import importlib  # noqa: PLC0415
        dl = importlib.import_module(os.environ.get("PYPTO_STEP3P5_LAYER_MODULE", "models.step3p5.decode_layer_single_chip"))
        from models.step3p5 import weight_loader as K  # noqa: PLC0415
        self._cfg = cfg
        self._dl = dl
        self._K = K
        assert self.tp == cfg.TP_WORLD_SIZE, f"need {cfg.TP_WORLD_SIZE} cards; got {self.tp}"

        from pypto import ir  # noqa: PLC0415
        from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415
        os.environ.setdefault(
            "PYPTO_PROG_BUILD_DIR",
            "/data/chensiyu/hw_project/pypto/workspace/build_output",
        )
        program = getattr(dl, self.layer_name)
        _mplan = None
        if os.environ.get("PYPTO_MEM_PLANNER", "").lower() == "ptoas":
            from pypto.pypto_core import passes as _passes  # noqa: PLC0415
            _mplan = _passes.MemoryPlanner.PTOAS
        self.compiled = ir.compile(
            program, platform=self.platform,
            distributed_config=DistributedConfig(device_ids=self.device_ids, num_sub_workers=0),
            skip_ptoas=False, dump_passes=False, memory_planner=_mplan,
        )
        print(f"[holder] main compile OK => {self.compiled.output_dir}", flush=True)
        from models.step3p5.mtp_fwd import whole_mtp3  # noqa: PLC0415
        self.mtp_compiled = ir.compile(
            whole_mtp3, platform=self.platform,
            distributed_config=DistributedConfig(device_ids=self.device_ids, num_sub_workers=0),
            skip_ptoas=False, dump_passes=False, memory_planner=_mplan,
        )
        print(f"[holder] mtp3 compile OK => {self.mtp_compiled.output_dir}", flush=True)

        def C(name):
            v = getattr(dl, name, None)
            if v is None:
                v = getattr(cfg, name)
            return int(v)
        self._C = C
        self._consts = dict(
            HIDDEN=cfg.HIDDEN, HEAD_DIM=cfg.HEAD_DIM, BATCH=cfg.BATCH,
            UBD=C("USER_BATCH_DYN"), BTF=C("BLOCK_TABLE_FLAT_DYN"),
            RSD=C("ROPE_SEQ_DYN"), KVC=C("KV_CACHE_ROWS_DYN"), MTP_KVC=int(getattr(cfg, "MTP_KV_CACHE_ROWS_DYN", C("KV_CACHE_ROWS_DYN"))),
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

        with open(os.path.join(self.out_dir, "pypto_weight_map.rank0.json")) as mf:
            self.VOCAB_LOCAL = int(json.load(mf)["map"][K.KEY_LM_HEAD]["shape"][0])
        return self

    # ---- resident prepare + arg wiring -------------------------------------

    def __enter__(self):
        assert self.compiled is not None, "call build() before entering holder"
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
        self.gate_r_full = _zsh(tp, N_FULL, NHF_PAD, HQ_FULL)
        self.gate_r_swa = _zsh(tp, N_SWA, NHS_PAD, HQ_SWA)
        # block-diag R constant (layer-independent): R[h, h*HEAD_DIM+d]=1 for real
        # local heads (count = HQ//HEAD_DIM), padded rows stay zero.
        for _h in range(HQ_FULL // HEAD_DIM):
            self.gate_r_full[:, :, _h, _h * HEAD_DIM:(_h + 1) * HEAD_DIM] = 1.0
        for _h in range(HQ_SWA // HEAD_DIM):
            self.gate_r_swa[:, :, _h, _h * HEAD_DIM:(_h + 1) * HEAD_DIM] = 1.0
        self.seq_lens = torch.ones(tp, UBD, dtype=_I32).share_memory_()
        self.run_gen_t = torch.zeros(tp, 1, dtype=_I32).share_memory_()  # E2 runtime run_gen tensor
        self.block_table = torch.zeros(tp, BTF, dtype=_I32).share_memory_()
        self.slot_mapping = torch.arange(UBD, dtype=_I32).unsqueeze(0).repeat(tp, 1).contiguous().share_memory_()
        self.rope_cf, self.rope_sf = _zsh(tp, RSD, ROT_FULL, dtype=_F32), _zsh(tp, RSD, ROT_FULL, dtype=_F32)
        self.rope_cs, self.rope_ss = _zsh(tp, RSD, ROT_SWA, dtype=_F32), _zsh(tp, RSD, ROT_SWA, dtype=_F32)
        self.k_cache, self.v_cache = _zsh(tp, KVC, HEAD_DIM), _zsh(tp, KVC, HEAD_DIM)
        self.h_mid_out, self.next_hidden_out = _zsh(tp, BATCH, HIDDEN), _zsh(tp, BATCH, HIDDEN)
        self.dbg_out = _zsh(tp, BATCH, HIDDEN)
        self.logits_shard_out = torch.zeros(tp, UBD, self.VOCAB_LOCAL, dtype=_F32).share_memory_()
        self.mtp_previous_hidden = _zsh(tp, BATCH, HIDDEN)
        self.mtp_first_token_ids = torch.zeros(tp, BATCH, dtype=_I32).share_memory_()
        self.mtp_active_mask = torch.zeros(tp, BATCH, dtype=_I32).share_memory_()
        self.mtp_seq_lens = torch.ones(tp, UBD, dtype=_I32).share_memory_()
        self.mtp_block_table = torch.zeros(tp, BTF, dtype=_I32).share_memory_()
        self.mtp_slot_mapping = torch.zeros(tp, UBD, dtype=_I32).share_memory_()
        self.mtp_gate_r = _zsh(tp, NHS_PAD, HQ_SWA)
        for _h in range(HQ_SWA // HEAD_DIM):
            self.mtp_gate_r[:, _h, _h * HEAD_DIM:(_h + 1) * HEAD_DIM] = 1.0
        self.mtp_k_cache = _zsh(tp, 3 * c["MTP_KVC"], HEAD_DIM)
        self.mtp_v_cache = _zsh(tp, 3 * c["MTP_KVC"], HEAD_DIM)
        self.mtp_hidden_out = _zsh(tp, 3, BATCH, HIDDEN)
        self.mtp_logits_out = torch.zeros(tp, 3, UBD, self.VOCAB_LOCAL, dtype=_F32).share_memory_()
        self.mtp_draft_token_ids_out = torch.full((tp, 3, BATCH), -1, dtype=_I32).share_memory_()

        from tools.step3p5.pypto_weight_ipc import (  # noqa: PLC0415
            import_weights_all, build_stacked_weight,
        )
        from pypto.runtime.device_tensor import DeviceTensor, StackedDeviceTensor  # noqa: PLC0415
        K = self._K

        # open the prepare() context manager and keep it resident
        if os.environ.get("PYPTO_HOLDER_PREPARE_MTP", "1").lower() in {"0", "false", "no", "off"}:
            self._prepare_cm = self.compiled.prepare()
            print("[holder] prepare main only (PYPTO_HOLDER_PREPARE_MTP=0)", flush=True)
        else:
            self._prepare_cm = self.compiled.prepare(extra_compiled=[self.mtp_compiled])
        self.rt = self._prepare_cm.__enter__()
        self._wmaps = import_weights_all(self.rt, self.out_dir, tp=tp, dev_offset=self.dev_offset)

        def W(key):
            return build_stacked_weight(self._wmaps, key)

        def W_reshape(key, per_rank_shape, dtype):
            shards = [DeviceTensor(self._wmaps[r].peer_base + self._wmaps[r].offset(key),
                                   tuple(per_rank_shape), dtype) for r in range(tp)]
            return StackedDeviceTensor(shards, (tp, *per_rank_shape), list(range(tp)))

        if self.kv_ipc:
            # Main-network KV may come from a vLLM-exported K-major/V-major pool.
            # MTP3 KV remains a distinct resident IPC slice from the weight pool in
            # the first live integration; the MTP program updates it across draft
            # steps and it never aliases the main 45-layer cache.
            try:
                from tools.step3p5.pypto_kv_ipc import (  # noqa: PLC0415
                    build_stacked_kv_pool,
                    import_kv_all,
                )
                self._kv_maps = import_kv_all(
                    self.rt, self.out_dir, tp=tp, dev_offset=self.dev_offset,
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
                    f"[holder] main KV via vLLM IPC: flat K/V rows={imported_rows}",
                    flush=True,
                )
            except FileNotFoundError:
                # Standalone/canonical path: k_cache/v_cache are carved in the
                # same weight IPC pool by export_from_checkpoint(..., kv_ipc=True).
                self.k_cache, self.v_cache = W("k_cache"), W("v_cache")
                print("[holder] main KV via weight IPC fallback", flush=True)
            def W_prefix_view(key, per_rank_shape, dtype):
                """Bind a prefix DeviceTensor view over a possibly larger IPC slice.

                Live vLLM tail-only exports its KV pool after the weight
                exporters have already carved the standalone canonical MTP KV
                slice (3*4096 rows).  The compiled live MTP program, however,
                uses the vLLM slot count (3*num_slots rows).  The prefix view
                keeps the program ABI exact without copying or reallocating: it
                only narrows metadata over the same 512B-aligned IPC base.
                """
                expected = 1
                for dim in per_rank_shape:
                    expected *= int(dim)
                shards = []
                for r in range(tp):
                    source = self._wmaps[r].device_tensor(key)
                    available = 1
                    for dim in source.shape:
                        available *= int(dim)
                    if source.dtype != dtype:
                        raise ValueError(
                            f"{key} rank{r}: dtype {source.dtype} != expected {dtype}"
                        )
                    if available < expected:
                        raise ValueError(
                            f"{key} rank{r}: IPC slice {source.shape} has {available} "
                            f"elements, cannot bind required prefix {tuple(per_rank_shape)} "
                            f"({expected} elements)"
                        )
                    shards.append(DeviceTensor(source.data_ptr, tuple(per_rank_shape), dtype))
                return StackedDeviceTensor(shards, (tp, *per_rank_shape), list(range(tp)))

            self.mtp_k_cache = W_prefix_view(
                "mtp_k_cache", (3 * c["MTP_KVC"], HEAD_DIM), _BF16
            )
            self.mtp_v_cache = W_prefix_view(
                "mtp_v_cache", (3 * c["MTP_KVC"], HEAD_DIM), _BF16
            )
            print(
                f"[holder] MTP3 KV via distinct IPC prefix rows={3 * c['MTP_KVC']}",
                flush=True,
            )

        self._initialize_rope_tables()

        # arg order MUST match the compiled program signature (see harness _do_worker)
        args = [self.current_hidden]
        args += [W(K.KEY_INPUT_RMS), W(K.KEY_POST_ATTN_RMS), W(K.KEY_Q_NORM), W(K.KEY_K_NORM)]
        args += [W(K.KEY_WQ_FULL), W(K.KEY_WK_FULL), W(K.KEY_WV_FULL), W(K.KEY_WO_FULL), W(K.KEY_WG_FULL),
                 self.gate_r_full]
        args += [W(K.KEY_WQ_SWA), W(K.KEY_WK_SWA), W(K.KEY_WV_SWA), W(K.KEY_WO_SWA), W(K.KEY_WG_SWA),
                 self.gate_r_swa]
        args += [W(K.KEY_DENSE_GATE), W(K.KEY_DENSE_UP), W(K.KEY_DENSE_DOWN)]
        args += [W(K.KEY_MOE_GATE_W), W(K.KEY_MOE_ROUTER_BIAS),
                 W(K.KEY_MOE_W_GATE_R), W(K.KEY_MOE_W_GATE_R_SCALE),
                 W(K.KEY_MOE_W_UP_R), W(K.KEY_MOE_W_UP_R_SCALE),
                 W(K.KEY_MOE_W_DOWN_R), W(K.KEY_MOE_W_DOWN_R_SCALE),
                 W(K.KEY_MOE_W_GATE_S),
                 W(K.KEY_MOE_W_UP_S), W(K.KEY_MOE_W_DOWN_S)]
        args += [self.seq_lens, self.block_table, self.slot_mapping,
                 self.rope_cf, self.rope_sf, self.rope_cs, self.rope_ss,
                 self.k_cache, self.v_cache]
        args += [self.h_mid_out, self.next_hidden_out]
        args += [self.dbg_out]
        args += [W_reshape(K.KEY_FINAL_NORM, (1, HIDDEN), _F32), W(K.KEY_LM_HEAD)]
        args += [self.logits_shard_out]
        # E2-baseline: run_gen_t removed (pre-rungen host_orch takes 46 args, no run_gen)
        self._args_list = args

        def W_flat(key, shape, dtype=_BF16):
            shards = []
            for r in range(tp):
                source = self._wmaps[r].device_tensor(key)
                n_src = 1
                for dim in source.shape:
                    n_src *= int(dim)
                n_dst = 1
                for dim in shape:
                    n_dst *= int(dim)
                if n_src != n_dst:
                    raise ValueError(f"{key} rank{r}: cannot reshape {source.shape} to {shape}")
                shards.append(DeviceTensor(source.data_ptr, tuple(shape), dtype))
            return StackedDeviceTensor(shards, (tp, *shape), list(range(tp)))

        num_mtp = int(getattr(self._cfg, "NUM_NEXTN_PREDICT_LAYERS", 3))
        self._mtp_args_list = [
            self.mtp_previous_hidden,
            self.mtp_first_token_ids,
            self.mtp_active_mask,
            W(K.KEY_EMBED),
            W(K.KEY_MTP_ENORM),
            W(K.KEY_MTP_HNORM),
            W_flat(K.KEY_MTP_EH_PROJ, (num_mtp * (HIDDEN // tp), 2 * HIDDEN)),
            W(K.KEY_MTP_INPUT_RMS),
            W_flat(K.KEY_MTP_WQ, (num_mtp * HIDDEN, HQ_SWA)),
            W_flat(K.KEY_MTP_WK, (num_mtp * HIDDEN, int(self._cfg.KV_HIDDEN_LOCAL))),
            W_flat(K.KEY_MTP_WV, (num_mtp * HIDDEN, int(self._cfg.KV_HIDDEN_LOCAL))),
            W(K.KEY_MTP_Q_NORM),
            W(K.KEY_MTP_K_NORM),
            W_flat(K.KEY_MTP_WO, (num_mtp * HQ_SWA, HIDDEN)),
            W_flat(K.KEY_MTP_WG, (num_mtp * HIDDEN, NHS_PAD)),
            self.mtp_gate_r,
            W(K.KEY_MTP_POST_ATTN_RMS),
            W_flat(K.KEY_MTP_DENSE_GATE, (num_mtp * HIDDEN, int(self._cfg.INTERMEDIATE_LOCAL))),
            W_flat(K.KEY_MTP_DENSE_UP, (num_mtp * HIDDEN, int(self._cfg.INTERMEDIATE_LOCAL))),
            W_flat(K.KEY_MTP_DENSE_DOWN, (num_mtp * int(self._cfg.INTERMEDIATE_LOCAL), HIDDEN)),
            W(K.KEY_MTP_SH_NORM),
            W_flat(K.KEY_MTP_SH_OUT, (num_mtp * self.VOCAB_LOCAL, HIDDEN)),
            self.mtp_seq_lens,
            self.mtp_block_table,
            self.mtp_slot_mapping,
            self.rope_cs,
            self.rope_ss,
            self.mtp_k_cache,
            self.mtp_v_cache,
            self.mtp_hidden_out,
            self.mtp_logits_out,
            self.mtp_draft_token_ids_out,
        ]
        print(
            f"[holder] resident: main_args={len(args)} mtp_args={len(self._mtp_args_list)} "
            f"VOCAB_LOCAL={self.VOCAB_LOCAL}",
            flush=True,
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._prepare_cm is not None:
            self._prepare_cm.__exit__(exc_type, exc, tb)
            self._prepare_cm = None
            self.rt = None

    # ---- per-step input setters --------------------------------------------

    def set_ctx1_token(self, emb_row, *, fill_batch=False):
        """offline ctx=1 A/B：把 embed(token) 灌进 row0（每 rank 复制）+ 位置0 identity rope。"""
        self.current_hidden.zero_()
        self.current_hidden[:, 0, :] = emb_row
        if fill_batch:
            self.current_hidden[:, :, :] = emb_row

    def set_hidden(self, hidden):
        """通用 per-step hidden 设置（sidecar/live）：hidden 可为 [BATCH,HIDDEN]（replicated 到每 rank）或 [tp,BATCH,HIDDEN]。"""
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
            self.current_hidden.copy_(hidden.to(_BF16))

    def set_meta(self, *, seq_lens=None, block_table=None, slot_mapping=None,
                 rope_cf=None, rope_sf=None, rope_cs=None, rope_ss=None):
        """per-step attn-meta（sidecar/live）：只覆盖传入的项，其余保持不变。"""
        if seq_lens is not None:
            self.seq_lens.copy_(seq_lens.to(_I32))
        if block_table is not None:
            self.block_table.copy_(block_table.to(_I32))
        if slot_mapping is not None:
            self.slot_mapping.copy_(slot_mapping.to(_I32))
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
        if torch.count_nonzero(table[valid_tokens:]).item():
            raise ValueError("block-table padding must be zero")
        slot_values = [int(x) for x in slots.tolist()]
        if len(set(slot_values)) != len(slot_values):
            raise ValueError(f"slot-mapping rows must be conflict-free: {slot_values}")
        if int(seq[:valid_tokens].max()) > self._consts["RSD"]:
            raise ValueError("requested position exceeds resident RoPE table")
        # Canonical ctx=1 validation uses an all-identity RoPE table, not just
        # row0 identity.  Preserve that boundary when every live token is at
        # position 0; later multi-step decode with position>0 keeps real tables.
        if torch.count_nonzero(pos[:valid_tokens]).item() == 0:
            self.rope_cf.fill_(1.0)
            self.rope_sf.zero_()
            self.rope_cs.fill_(1.0)
            self.rope_ss.zero_()
        else:
            self._rope_ready = False
            self._initialize_rope_tables()

        self.current_hidden.zero_()
        self.current_hidden[:, :valid_tokens, :] = hidden
        self.seq_lens.copy_(seq.unsqueeze(0).expand(self.tp, -1))
        self.block_table.copy_(table.reshape(1, -1).expand(self.tp, -1))
        self.slot_mapping.copy_(slots.unsqueeze(0).expand(self.tp, -1))

    def set_mtp3_step(
        self,
        previous_hidden,
        first_token_ids,
        *,
        seq_lens,
        positions,
        block_table,
        slot_mapping,
    ):
        """Set one MTP3 draft step.

        ``previous_hidden`` and ``first_token_ids`` contain only the valid rows
        from the target sampler boundary.  Padding rows are explicitly zeroed;
        ``active_mask`` is the only thing allowed to distinguish live rows from
        padding inside the MTP program.
        """
        self._initialize_rope_tables()
        if previous_hidden.dtype != _BF16 or previous_hidden.ndim != 2:
            raise ValueError(
                f"MTP previous_hidden must be BF16 [T,H], got "
                f"{previous_hidden.dtype} {tuple(previous_hidden.shape)}"
            )
        valid_tokens = int(previous_hidden.shape[0])
        batch = self._consts["BATCH"]
        hidden_size = self._consts["HIDDEN"]
        user_batch = self._consts["UBD"]
        if not (1 <= valid_tokens <= batch) or previous_hidden.shape[1] != hidden_size:
            raise ValueError(
                f"MTP previous_hidden must be [1..{batch},{hidden_size}], "
                f"got {tuple(previous_hidden.shape)}"
            )
        tokens = first_token_ids.to(_I32).flatten()
        if tuple(tokens.shape) != (valid_tokens,):
            raise ValueError(f"first_token_ids must be [{valid_tokens}], got {tuple(tokens.shape)}")
        seq = seq_lens.to(_I32).flatten()
        pos = positions.to(_I32).flatten()
        slots = slot_mapping.to(_I32).flatten()
        table = block_table.to(_I32)
        if tuple(seq.shape) != (user_batch,) or tuple(pos.shape) != (user_batch,):
            raise ValueError(f"seq_lens/positions must be [{user_batch}]")
        if tuple(slots.shape) != (user_batch,):
            raise ValueError(f"slot_mapping must be [{user_batch}]")
        if table.ndim != 2 or table.shape[0] != user_batch or table.numel() != self._consts["BTF"]:
            raise ValueError("block_table must match [storage_batch,max_blocks] compiled ABI")
        if torch.any(seq[:valid_tokens] <= 0):
            raise ValueError("valid MTP seq_lens must be positive")
        if not torch.equal(pos[:valid_tokens], seq[:valid_tokens] - 1):
            raise ValueError("valid MTP positions must equal seq_lens-1")
        if not torch.equal(seq[valid_tokens:], torch.ones(user_batch - valid_tokens, dtype=_I32)):
            raise ValueError("MTP seq_lens padding must be one")
        if torch.count_nonzero(pos[valid_tokens:]).item():
            raise ValueError("MTP positions padding must be zero")
        self.mtp_previous_hidden.zero_()
        self.mtp_first_token_ids.zero_()
        self.mtp_active_mask.zero_()
        self.mtp_previous_hidden[:, :valid_tokens, :] = previous_hidden
        self.mtp_first_token_ids[:, :valid_tokens] = tokens
        self.mtp_active_mask[:, :valid_tokens] = 1
        self.mtp_seq_lens.copy_(seq.unsqueeze(0).expand(self.tp, -1))
        self.mtp_block_table.copy_(table.reshape(1, -1).expand(self.tp, -1))
        self.mtp_slot_mapping.copy_(slots.unsqueeze(0).expand(self.tp, -1))

    def run_mtp3(self):
        """Run the resident three-layer MTP program once."""
        assert self.rt is not None, "enter holder (with h:) before run_mtp3()"
        self.mtp_hidden_out.zero_()
        self.mtp_logits_out.zero_()
        self.mtp_draft_token_ids_out.fill_(-1)
        t0 = time.time()
        self.rt.run(self.mtp_compiled, *self._mtp_args_list)
        dt = time.time() - t0
        if (self.mtp_draft_token_ids_out < 0).any():
            raise RuntimeError("whole_mtp3 left draft_token_ids_out unwritten")
        return dict(
            dt=dt,
            hidden=self.mtp_hidden_out,
            logits=self.mtp_logits_out,
            draft_token_ids=self.mtp_draft_token_ids_out,
        )

    # ---- run ----------------------------------------------------------------

    def run(self):
        """复用常驻 rt 跑一次 whole-net forward，返回 logits/诊断。"""
        assert self.rt is not None, "enter holder (with h:) before run()"
        t0 = time.time()
        _dfx = os.environ.get("N1_DFX", "")
        if _dfx:
            from pypto.runtime.runner import RunConfig  # noqa: PLC0415
            _rc = RunConfig(platform=self.platform,
                            enable_dep_gen=("dep" in _dfx),
                            enable_scope_stats=("scope" in _dfx))
            self.rt.run(self.compiled, *self._args_list, config=_rc)
        else:
            self.run_gen_t.fill_(int(self._run_gen))
            self.rt.run(self.compiled, *self._args_list)
            self._run_gen += 1
        dt = time.time() - t0
        tp = self.tp
        full_logits = torch.cat([self.logits_shard_out[r, 0] for r in range(tp)], dim=0)
        return dict(
            dt=dt,
            full_logits=full_logits,
            argmax=int(full_logits.argmax()),
            next_hidden=self.next_hidden_out,
            h_mid=self.h_mid_out,
            dbg=self.dbg_out,
            nh_row0_max=float(self.next_hidden_out[:, 0, :].float().abs().max()),
        )
