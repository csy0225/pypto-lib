# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Resident N=1 whole-net hidden-only prefill holder.

Prefill dual of ``tools.step3p5/whole_decode_holder.py``.  It lifts the
canonical ``models.step3p5.prefill_layer_single_chip_hidden:whole_prefill_step3p5``
compile -> prepare -> import_weights -> rt.run lifecycle into a resident
holder: **build+prepare once**, then every ``run()`` reuses the same prepared
``rt`` (no re-prepare), feeding hidden / attn-meta / KV / positions per step.

Design boundary (aligns with SKILL section H: N=1 single ``@pl.program`` is
the only production form):

* **resident (bind once, invariant across steps)**: every weight
  (import_ipc DeviceTensor), ``gate_r_full/swa/moe_full/moe_swa``
  (block-diag R constants).  The production holder does not bind final norm
  or the LM head.
* **per-step handoff (mutate these share_memory host tensors before each
  run)**: ``current_hidden``, ``position_ids`` (prefill-only), attn-meta
  (seq_lens/block_table/slot_mapping/rope_*), KV (IPC add_inout, attention
  reads/writes in place).
* **output**: the post-45-layer ``next_hidden_out``.  Final RMSNorm, LM head
  and sampling all stay in vLLM (the hidden-only boundary).

Prefill-specific deltas vs the decode holder (everything else is reused
verbatim):

* token dimension is ``PREFILL_T=128`` (decode ``BATCH=16``); the resident
  hidden/slot_mapping/next_hidden_out carry ``PREFILL_T``;
* a new resident ``position_ids [tp, PREFILL_T] INT32`` (decode has none),
  inserted into the arg list after ``slot_mapping`` and before the RoPE
  tables to match the scaffold ``host_orch`` signature exactly;
* ``seq_lens`` is ``[tp, 1]`` (one sequence; decode was ``[tp, USER_BATCH_DYN]``)
  and ``block_table`` is the single sequence's flat ``[tp, BLOCK_TABLE_FLAT_DYN]``
  block list;
* the live handoff entry is ``set_live_prompt`` (single-sequence prefill
  metadata); it does NOT reuse decode's ``validate_fixed_batch_metadata``
  (that helper hardcodes ``STORAGE_BATCH=16`` plus 15 padding rows, which is
  wrong for a single-sequence prefill step).

The exporter (weight/KV IPC pool) is managed by the caller; the holder
assumes exporters are already ready (``reuse`` semantics) and only attaches.
The offline harness and the sidecar share this holder so behaviour is
consistent and not duplicated.

The arg order **must** match the compiled program signature verbatim (see
``_build_loop_form_args``).

Scaffold gate: ``models.step3p5.prefill_layer_single_chip_hidden.IS_SCAFFOLD``
is currently ``True`` (the ``whole_chip_orch`` body is a token-tiled
passthrough placeholder, not the real 45-layer dispatch).  ``build()``
refuses to compile while that flag is set so the placeholder is never
accidentally run as production.  See RECOVERY_PROGRESS.md section 5.1.
"""
from __future__ import annotations

import json
import os
import time

import torch

_BF16 = torch.bfloat16
_F32 = torch.float32
_I32 = torch.int32
MAIN_PROGRAM = "whole_prefill_step3p5"

# canonical Main splits the loader's KEY_WQ_FULL[12] / KEY_WQ_SWA[33] into
# four attention buckets.  full_wq/swa_wq keep the full physical slots (dead
# slots are harmless: L44@full_wq[11] / L43@swa_wq[32] slot indexing depends
# on the [12]/[33] sizes); moe_full_wq takes canonical FULL slots 1-10
# (L4,8,...,40); moe_swa_wq takes canonical SWA slots 2-31 (30 MoE-swa layers,
# skipping dense L1(slot0)/L2(slot1)/L43(slot32)).
_MOE_FULL_SLOTS = tuple(range(1, 11))            # 10 full-MoE attn layers
_MOE_SWA_SLOTS = tuple(range(2, 32))             # 30 swa-MoE attn layers


def _zsh(*shape, dtype=_BF16):
    """DistributedWorker contract: host tensor must share_memory and be
    allocated before prepare()."""
    return torch.zeros(shape, dtype=dtype).share_memory_()


class WholePrefillHolder:
    """Resident N=1 whole-net prefill: build+prepare once, run() reuses rt.

    Typical usage (offline single-prompt A/B)::

        h = WholePrefillHolder(device_ids=[0..7], out_dir="/tmp/n1_weight_ipc",
                               ckpt=CKPT, kv_ipc=True)
        h.build()
        with h:
            h.set_live_prompt(hidden, seq_lens=seq, positions=pos,
                              block_table=table, slot_mapping=slots)
            res = h.run()
            print(res["next_hidden"].shape)   # [tp, PREFILL_T, HIDDEN]

    Sidecar usage: build()+__enter__ resident; each live request calls
    set_live_prompt(...) + run().
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
        self.device_ids = list(device_ids)
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
        self._K = None
        self._dl = None
        self._consts = {}
        self.hidden_only = True

        # populated by __enter__()
        self._prepare_cm = None
        self.rt = None
        self._wmaps = None
        self._kv_maps = None
        self.padding_reserve = None
        self._args_list = None
        # resident host tensors (mutated per-step).  Their leading token
        # dimension is static storage capacity; active rows are supplied by
        # set_live_prompt / num_tokens_per_owner for each invocation.
        self.current_hidden = None
        self.num_tokens_per_owner = None
        self.gate_r_full = self.gate_r_swa = None
        self.gate_r_moe_full = self.gate_r_moe_swa = None
        self.seq_lens = None
        self.block_table = None
        self.slot_mapping = None
        # prefill-only resident: per-token positions (decode has none).
        self.position_ids = None
        self.rope_cf = self.rope_sf = self.rope_cs = self.rope_ss = None
        self.k_cache = self.v_cache = None
        self._next_hidden_out = None
        self._last_run_sec = 0.0
        self._rope_ready = False

    def _infer_rows_from_map(self) -> None:
        """Derive physical flat KV rows from the vLLM-owned Main map.

        ``kv_cache_config.num_blocks`` is a scheduler-domain quantity and
        must not be used directly as the PyPTO physical tensor capacity.
        The map records the reserve-extended physical capacity, so it must
        be read before importing cfg / compiling.
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
        """Read a module-level constant from the canonical Main module."""
        return int(getattr(self._dl, name))

    def build(self):
        """Compile the program + resolve constants/VOCAB.  No device prepare
        (that lives in __enter__).

        Scaffold gate: refuses to compile while
        ``prefill_layer_single_chip_hidden.IS_SCAFFOLD`` is True.
        """
        from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415
        set_backend_type(BackendType.Ascend910B)
        if self.kv_ipc:
            self._infer_rows_from_map()
        import models.step3p5.config as cfg  # noqa: PLC0415
        from models.step3p5 import weight_loader as K  # noqa: PLC0415
        self._cfg = cfg
        self._K = K
        assert self.tp == cfg.TP_WORLD_SIZE, f"need {cfg.TP_WORLD_SIZE} cards; got {self.tp}"
        import models.step3p5.prefill_layer_single_chip_hidden as dl  # noqa: PLC0415
        self._dl = dl

        # Scaffold gate: whole_chip_orch is a token-tiled passthrough
        # placeholder, NOT the real 45-layer prefill dispatch.  Refuse to
        # compile so the placeholder is never accidentally run as production.
        # Flip IS_SCAFFOLD to False once the P1 kernel bodies (token-tiling,
        # W8A8, attention fixes, dual-index) are ported.  See
        # RECOVERY_PROGRESS.md section 5.1.
        if bool(getattr(dl, "IS_SCAFFOLD", False)):
            raise RuntimeError(
                "whole_prefill_step3p5 is a scaffold placeholder "
                "(IS_SCAFFOLD=True): the real 45-layer prefill body is not "
                "yet ported (P1 pending). Refusing to compile so the "
                "placeholder is never accidentally run as production. See "
                "models/step3p5/prefill_layer_single_chip_hidden.py and "
                "RECOVERY_PROGRESS.md section 5.1."
            )

        from pypto import ir  # noqa: PLC0415
        from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415
        os.environ.setdefault(
            "PYPTO_PROG_BUILD_DIR",
            "/tmp/pypto_build_output",
        )
        _mplan = None
        if os.environ.get("PYPTO_MEM_PLANNER", "").lower() == "ptoas":
            from pypto.pypto_core import passes as _passes  # noqa: PLC0415
            _mplan = _passes.MemoryPlanner.PTOAS
        self.compiled = ir.compile(
            dl.whole_prefill_step3p5, platform=self.platform,
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
        # PREFILL_T is the prefill token dimension (decode uses BATCH=16).
        # Source of truth: models.step3p5.prefill_qkv_proj_rope.PREFILL_T (=128),
        # re-exported by the scaffold module.
        from models.step3p5.prefill_qkv_proj_rope import PREFILL_T  # noqa: PLC0415
        prefill_t = int(PREFILL_T)
        self._consts = dict(
            HIDDEN=cfg.HIDDEN, HEAD_DIM=cfg.HEAD_DIM, PREFILL_T=prefill_t,
            BTF=C("BLOCK_TABLE_FLAT_DYN"),
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
        c = self._consts
        tp = self.tp
        HIDDEN, HEAD_DIM = c["HIDDEN"], c["HEAD_DIM"]
        PREFILL_T = c["PREFILL_T"]
        BTF, RSD, KVC = c["BTF"], c["RSD"], c["KVC"]
        ROT_FULL, ROT_SWA = c["ROT_FULL"], c["ROT_SWA"]
        NHF_PAD, NHS_PAD = c["NHF_PAD"], c["NHS_PAD"]
        HQ_FULL, HQ_SWA = c["HQ_FULL"], c["HQ_SWA"]
        N_FULL, N_SWA = c["N_FULL"], c["N_SWA"]

        # resident host tensors (allocated BEFORE prepare so forked chips see them)
        self.current_hidden = _zsh(tp, PREFILL_T, HIDDEN)
        # G1: one shared runtime active-token count per owner.  The canonical
        # graph takes max(owner counts) so every rank uses identical dynamic
        # MoE bounds.
        self.num_tokens_per_owner = torch.zeros(
            (128,), dtype=_I32,
        ).share_memory_()
        self.num_tokens_per_owner[:tp].fill_(PREFILL_T)
        self.gate_r_full = _zsh(tp, N_FULL, NHF_PAD, HQ_FULL)
        self.gate_r_swa = _zsh(tp, N_SWA, NHS_PAD, HQ_SWA)
        # canonical loop-form 4-bucket split: full/swa each cover dense +
        # post-loop full; moe_full/moe_swa are the MoE-loop-only buckets,
        # each with its own block-diag R.
        n_moe_full = self._main_const("NUM_FULL_MOE_LAYERS")
        n_moe_swa = self._main_const("NUM_SWA_MOE_LAYERS")
        self.gate_r_moe_full = _zsh(tp, n_moe_full, NHF_PAD, HQ_FULL)
        self.gate_r_moe_swa = _zsh(tp, n_moe_swa, NHS_PAD, HQ_SWA)
        for _h in range(HQ_FULL // HEAD_DIM):
            self.gate_r_moe_full[:, :, _h, _h * HEAD_DIM:(_h + 1) * HEAD_DIM] = 1.0
        for _h in range(HQ_SWA // HEAD_DIM):
            self.gate_r_moe_swa[:, :, _h, _h * HEAD_DIM:(_h + 1) * HEAD_DIM] = 1.0
        # block-diag R constant (layer-independent): R[h, h*HEAD_DIM+d]=1 for
        # real local heads (count = HQ//HEAD_DIM); padded rows stay zero.
        for _h in range(HQ_FULL // HEAD_DIM):
            self.gate_r_full[:, :, _h, _h * HEAD_DIM:(_h + 1) * HEAD_DIM] = 1.0
        for _h in range(HQ_SWA // HEAD_DIM):
            self.gate_r_swa[:, :, _h, _h * HEAD_DIM:(_h + 1) * HEAD_DIM] = 1.0
        # Prefill single-sequence metadata: seq_lens is [tp, 1] (one sequence
        # per step), block_table is the flat single-sequence block list
        # [tp, BLOCK_TABLE_FLAT_DYN], slot_mapping / position_ids are per-token
        # [tp, PREFILL_T].
        self.seq_lens = torch.ones(tp, 1, dtype=_I32).share_memory_()
        self.block_table = torch.zeros(tp, BTF, dtype=_I32).share_memory_()
        self.slot_mapping = (
            torch.arange(PREFILL_T, dtype=_I32).unsqueeze(0)
            .repeat(tp, 1).contiguous().share_memory_()
        )
        self.position_ids = torch.zeros(tp, PREFILL_T, dtype=_I32).share_memory_()
        self.rope_cf, self.rope_sf = _zsh(tp, RSD, ROT_FULL, dtype=_F32), _zsh(tp, RSD, ROT_FULL, dtype=_F32)
        self.rope_cs, self.rope_ss = _zsh(tp, RSD, ROT_SWA, dtype=_F32), _zsh(tp, RSD, ROT_SWA, dtype=_F32)
        self.k_cache, self.v_cache = _zsh(tp, KVC, HEAD_DIM), _zsh(tp, KVC, HEAD_DIM)
        self._next_hidden_out = _zsh(tp, PREFILL_T, HIDDEN)
        from tools.step3p5.pypto_weight_ipc import (  # noqa: PLC0415
            import_weights_all, build_stacked_weight,
        )
        K = self._K

        # Keep one Worker.run and one CommDomain resident across prefill
        # steps.  The runtime clears every retained window before each
        # request, so signal epochs restart at 1 without stale Ge thresholds
        # or repeated HCCL domain allocation/release churn.
        self._prepare_cm = self.compiled.prepare(persistent=True)
        self.rt = self._prepare_cm.__enter__()
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
                STORAGE_BATCH,
                make_padding_reserve,
            )

            summary = self._kv_maps[0].summary
            # The Main KV pool storage capacity is STORAGE_BATCH (shared with
            # decode); it is NOT PREFILL_T (the prefill token dimension).
            self.padding_reserve = make_padding_reserve(
                summary.scheduler_num_blocks,
                summary.physical_num_blocks,
                block_size=summary.block_size,
                storage_capacity=STORAGE_BATCH,
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

        # arg order MUST match the compiled program signature (see harness _do_worker)
        def W(key):
            return build_stacked_weight(self._wmaps, key)

        def Wsub(key, slots):
            """canonical 4-bucket split: take a zero-copy contiguous sub-view
            by leading-dim slots from the loader's full [N,...] bucket (each
            rank slices independently, then stacks into StackedDeviceTensor).

            KEY_WQ_FULL=[12] stacks all 12 FULL layers; moe_full_wq[10] takes
            slots 1-10 (L4,8,...,40; slot0=L0 dense, slot11=L44 post-loop stay
            in the full_wq bucket).  KEY_WQ_SWA=[33] likewise; moe_swa_wq[30]
            takes slots 2-31.  DeviceTensor.__getitem__ only permits an
            outermost partial (contiguous) slice, which this satisfies.
            """
            from pypto.runtime.device_tensor import StackedDeviceTensor  # noqa: PLC0415
            start, stop = slots[0], slots[-1] + 1
            shards = [self._wmaps[r].device_tensor(key)[start:stop] for r in range(tp)]
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

    def _build_loop_form_args(self, W, Wsub):
        """Canonical loop-form 55-arg host_orch arg-list
        (strictly follows models/step3p5/prefill_layer_single_chip_hidden.py
        host_orch signature).

        Prefill = decode's 54-arg host_orch + ``position_ids`` inserted after
        ``slot_mapping`` and before the RoPE tables.  The canonical Main has a
        single Out (next_hidden_out).  Attention uses four buckets
        (full/swa/moe_full/moe_swa); MoE buckets take contiguous zero-copy
        sub-views from the resident weight pool.  Constant names are read
        from the canonical Main module (N_FULL_ATTN_LAYERS etc.).
        """
        K = self._K
        args = [self.current_hidden]
        # 4 activations / RMS norms
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
        # MoE expert/router weights.  Routed expert weights are INT8 + FP32
        # per-channel scale (W8A8, design §2.2 invariant 2; int8_routed=True
        # exporter populates KEY_MOE_W_*_R_SCALE).  Order matches host_orch
        # signature exactly: each routed weight immediately followed by its
        # scale, before the shared weights (R7 arg-order contract).
        args += [W(K.KEY_MOE_GATE_W), W(K.KEY_MOE_ROUTER_BIAS),
                 W(K.KEY_MOE_W_GATE_R), W(K.KEY_MOE_W_GATE_R_SCALE),
                 W(K.KEY_MOE_W_UP_R), W(K.KEY_MOE_W_UP_R_SCALE),
                 W(K.KEY_MOE_W_DOWN_R), W(K.KEY_MOE_W_DOWN_R_SCALE),
                 W(K.KEY_MOE_W_GATE_S),
                 W(K.KEY_MOE_W_UP_S), W(K.KEY_MOE_W_DOWN_S)]
        # KV/RoPE metadata + prefill-only position_ids (after slot_mapping,
        # before rope_* -- matches host_orch signature exactly)
        args += [self.seq_lens, self.block_table, self.slot_mapping,
                 self.position_ids,
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
        if self._prepare_cm is not None:
            self._prepare_cm.__exit__(exc_type, exc, tb)
            self._prepare_cm = None
            self.rt = None

    # ---- per-step input setters --------------------------------------------

    def set_ctx1_token(self, emb_row, *, fill_batch=False):
        """Offline ctx=1 A/B: load embed(token) into row 0 (copied per rank)
        + position 0 identity rope.  Prefill dual of decode's set_ctx1_token.

        ``fill_batch`` fills every PREFILL_T row with the same embedding and
        sets positions to arange(PREFILL_T) (a synthetic full prompt).  KV
        metadata (seq_lens/block_table/slot_mapping) must still be supplied
        via set_live_prompt or set_meta before run().
        """
        self.current_hidden.zero_()
        self.current_hidden[:, 0, :] = emb_row
        if fill_batch:
            self.current_hidden[:, :, :] = emb_row
        if self.position_ids is not None:
            self.position_ids.zero_()
            if fill_batch:
                prefill_t = self._consts["PREFILL_T"]
                self.position_ids[:, :] = (
                    torch.arange(prefill_t, dtype=_I32).unsqueeze(0)
                )
        if self.num_tokens_per_owner is not None:
            self.num_tokens_per_owner[: self.tp].fill_(
                self._consts["PREFILL_T"] if fill_batch else 1
            )

    def set_hidden(self, hidden):
        """Set a full-capacity hidden buffer for offline/sidecar callers.

        Live variable-token callers should use :meth:`set_live_prompt`, which
        accepts active rows and updates positions / runtime token metadata
        together.  This setter does NOT touch ``position_ids``; the caller
        must populate positions separately before run().
        """
        self.current_hidden.zero_()
        if hidden.dim() == 2:
            expected = (self._consts["PREFILL_T"], self._consts["HIDDEN"])
            if tuple(hidden.shape) != expected:
                raise ValueError(
                    f"set_hidden expects full storage shape {expected}, "
                    f"got {tuple(hidden.shape)}"
                )
            self.current_hidden[:, :, :] = hidden.to(_BF16)
        else:
            expected = (
                self.tp,
                self._consts["PREFILL_T"],
                self._consts["HIDDEN"],
            )
            if tuple(hidden.shape) != expected:
                raise ValueError(
                    f"set_hidden expects TP storage shape {expected}, "
                    f"got {tuple(hidden.shape)}"
                )
            self.current_hidden.copy_(hidden.to(_BF16))
        if self.num_tokens_per_owner is not None:
            self.num_tokens_per_owner[: self.tp].fill_(self._consts["PREFILL_T"])

    def set_meta(self, *, seq_lens=None, block_table=None, slot_mapping=None,
                 rope_cf=None, rope_sf=None, rope_cs=None, rope_ss=None):
        """Only diagnostic RoPE may be updated here; paged-KV metadata and
        positions must go through set_live_prompt.

        seq/block/slot lack positions, valid_rows and the allocator reserve
        context, so they cannot satisfy the production fail-closed contract
        and must not be overwritten through this legacy entry point.
        """
        if any(
            value is not None
            for value in (seq_lens, block_table, slot_mapping)
        ):
            raise ValueError(
                "paged-KV metadata must be updated through set_live_prompt so "
                "the allocator-owned padding reserve can be validated"
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

    def set_live_prompt(
        self,
        hidden,
        *,
        seq_lens,
        positions,
        block_table,
        slot_mapping,
    ):
        """Set one single-sequence prefill step and initialize padding rows.

        Prefill dual of decode's ``set_live_step``.  This does NOT reuse
        decode's ``validate_fixed_batch_metadata`` (it hardcodes
        ``STORAGE_BATCH=16`` plus 15 padding rows, wrong for a single-sequence
        prefill step); instead it validates the prefill-specific ABI directly:

        * ``hidden`` is BF16 ``[T, HIDDEN]`` with ``1 <= T <= PREFILL_T``;
        * ``seq_lens`` is INT32 ``[1]`` (one sequence; value = seq_len);
        * ``positions`` is INT32 ``[PREFILL_T]``; the active prefix
          ``[0:T]`` must be contiguous ending at ``seq_len - 1`` (a fresh
          prompt has ``positions == arange(T)``), and padding ``[T:]`` must be
          zero;
        * ``block_table`` is INT32 2-D ``[1, max_blocks]`` (the single
          sequence's flat block list, width == compiled BLOCK_TABLE_FLAT_DYN);
        * ``slot_mapping`` is INT32 ``[PREFILL_T]`` and consistent with
          ``block_table`` for the active tokens
          (``slot == block_table[pos // BLOCK_SIZE] * BLOCK_SIZE +
          pos % BLOCK_SIZE``).

        The allocator-owned Main padding reserve must be available
        (fail-closed).
        """
        self._initialize_rope_tables()
        prefill_t = self._consts["PREFILL_T"]
        hidden_size = self._consts["HIDDEN"]
        if hidden.dtype != _BF16 or hidden.ndim != 2:
            raise ValueError(
                f"live hidden must be BF16 [T,H], got "
                f"{hidden.dtype} {tuple(hidden.shape)}"
            )
        query_t = int(hidden.shape[0])
        if not (1 <= query_t <= prefill_t) or hidden.shape[1] != hidden_size:
            raise ValueError(
                f"live hidden must be [1..{prefill_t},{hidden_size}], "
                f"got {tuple(hidden.shape)}"
            )

        seq = seq_lens.to(_I32).flatten()
        pos = positions.to(_I32).flatten()
        slots = slot_mapping.to(_I32).flatten()
        table = block_table.to(_I32)
        if tuple(seq.shape) != (1,):
            raise ValueError(
                f"seq_lens must be INT32 [1] (single sequence), got "
                f"{tuple(seq.shape)}"
            )
        if tuple(pos.shape) != (prefill_t,):
            raise ValueError(
                f"positions must be INT32 [{prefill_t}], got "
                f"{tuple(pos.shape)}"
            )
        if tuple(slots.shape) != (prefill_t,):
            raise ValueError(
                f"slot_mapping must be INT32 [{prefill_t}], got "
                f"{tuple(slots.shape)}"
            )
        btf = self._consts["BTF"]
        if table.ndim != 2 or table.shape[0] != 1:
            raise ValueError(
                f"block_table must be INT32 [1,max_blocks] (single sequence "
                f"flat block list), got {tuple(table.shape)}"
            )
        if table.shape[1] != btf:
            raise ValueError(
                f"block_table width {table.shape[1]} != compiled "
                f"BLOCK_TABLE_FLAT_DYN {btf}"
            )

        seq_len = int(seq[0])
        if seq_len <= 0:
            raise ValueError("seq_lens must be positive")
        if seq_len < query_t:
            raise ValueError(f"seq_len={seq_len} < query T={query_t}")
        # Active positions must be contiguous [seq_len-T, seq_len).
        expected_pos = torch.arange(seq_len - query_t, seq_len, dtype=_I32)
        if not torch.equal(pos[:query_t], expected_pos):
            raise ValueError(
                f"active positions must be contiguous [seq_len-T, seq_len) = "
                f"[{seq_len - query_t}, {seq_len}); got "
                f"{pos[:query_t].tolist()}"
            )
        if torch.count_nonzero(pos[query_t:]).item():
            raise ValueError("padding positions must be zero")
        if int(pos[:query_t].max()) >= self._consts["RSD"]:
            raise ValueError("requested position exceeds resident RoPE table")

        if self.padding_reserve is None:
            raise ValueError(
                "allocator-owned Main padding reserve is unavailable"
            )

        # Validate slot_mapping consistency with block_table for active
        # tokens (mirror vllm_prefill_metadata._extract_prefill_group_metadata).
        from tools.step3p5.kv_padding import BLOCK_SIZE  # noqa: PLC0415
        block_size = int(BLOCK_SIZE)
        active_blocks_needed = (seq_len + block_size - 1) // block_size
        if table.shape[1] < active_blocks_needed:
            raise ValueError(
                f"block_table width {table.shape[1]} cannot cover "
                f"seq_len={seq_len} (needs {active_blocks_needed} blocks of "
                f"size {block_size})"
            )
        reserve = self.padding_reserve
        active_table = table[0, :active_blocks_needed].to(torch.int32)
        if torch.any(active_table < 0) or torch.any(
            active_table >= reserve.scheduler_num_blocks
        ):
            raise ValueError(
                f"active block_table ids must lie in scheduler domain "
                f"[0,{reserve.scheduler_num_blocks}), got {active_table.tolist()}"
            )
        pos_active = pos[:query_t].to(torch.long)
        cols = (pos_active // block_size).to(torch.long)
        if int(cols.max().item()) >= active_blocks_needed:
            raise ValueError(
                "an active position maps to a block beyond the active block "
                "table"
            )
        expected_slot = (
            active_table.index_select(0, cols) * block_size
            + (pos_active % block_size).to(torch.int32)
        )
        if not torch.equal(slots[:query_t], expected_slot):
            raise ValueError(
                "slot_mapping does not match "
                "block_table*BLOCK_SIZE+pos%BLOCK_SIZE for active tokens: "
                f"slot={slots[:query_t].tolist()}, "
                f"expected={expected_slot.tolist()}"
            )

        # Write hidden into current_hidden[:, :T, :]; zero padding rows.
        self.current_hidden.zero_()
        self.current_hidden[:, :query_t, :] = hidden
        # Fill position_ids[:, :T] from positions; padding [:, T:] stays zero.
        self.position_ids.zero_()
        self.position_ids[:, :query_t] = pos[:query_t]
        self.num_tokens_per_owner[: self.tp].fill_(query_t)
        # Single-sequence metadata is identical across tp ranks.
        self.seq_lens.copy_(seq.unsqueeze(0).expand(self.tp, -1))
        self.block_table.copy_(table.expand(self.tp, -1))
        self.slot_mapping.copy_(slots.unsqueeze(0).expand(self.tp, -1))

    # ---- run ----------------------------------------------------------------

    def run(self):
        """Reuse the resident rt for one whole-net prefill forward; return
        the raw hidden output."""
        assert self.rt is not None, "enter holder (with h:) before run()"
        t0 = time.time()
        # DFX capture (PERF-A1 baseline). N1_DFX = tokens; N1_PMU = int event.
        #   "swim"/"l2" -> enable_l2_swimlane (dfx_outputs/l2_swimlane_records.json
        #                  + merged_swimlane_*.json = per-task/per-layer timing)
        #   "pmu"       -> enable_pmu = int(N1_PMU or 1)  (dfx_outputs/pmu.csv)
        #   "dep"       -> enable_dep_gen ; "scope" -> enable_scope_stats
        # Artifacts under {self.compiled.output_dir}/dfx_outputs/. Swimlane and
        # PMU perturb timing -> collect in SEPARATE run() calls from clean
        # wallclock.
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
