# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Resident real-checkpoint holder for the focused Step3p5 L0-L4 graph."""
from __future__ import annotations

import os
import time

import torch

from tools.step3p5.whole_decode_holder import WholeDecodeHolder, _zsh


_BF16 = torch.bfloat16
_F32 = torch.float32
_I32 = torch.int32

# Loader slot mapping for the first five physical layers.
_FULL_SLOTS = (0, 1)  # L0, L4
_SWA_SLOTS = (0, 1, 2)  # L1, L2, L3
_MOE_SLOTS = (0, 1)  # L3, L4


class FiveLayerMoeHolder(WholeDecodeHolder):
    """Build once and repeatedly run the canonical L0-L4 focused program."""

    def __init__(
        self,
        device_ids,
        out_dir,
        ckpt,
        *,
        platform="a2a3",
        kv_ipc=True,
    ):
        super().__init__(
            device_ids=device_ids,
            out_dir=out_dir,
            ckpt=ckpt,
            platform=platform,
            kv_ipc=kv_ipc,
        )
        self.program_name = "five_layer_moe"
        self._hidden_l3_out = None
        self._hidden_l4_out = None

    def build(self):
        """Compile the focused graph without preparing device state."""
        from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415

        set_backend_type(BackendType.Ascend910B)
        if self.kv_ipc:
            self._infer_rows_from_map()

        import models.step3p5.config as cfg  # noqa: PLC0415
        from models.step3p5 import weight_loader as keys  # noqa: PLC0415
        from tests.step3p5.harnesses import (  # noqa: PLC0415
            _five_layer_moe_program as focused,
        )

        self._cfg = cfg
        self._K = keys
        self._focused = focused
        if self.tp != cfg.TP_WORLD_SIZE:
            raise ValueError(
                f"focused holder requires TP={cfg.TP_WORLD_SIZE}, got {self.tp}"
            )

        from pypto import ir  # noqa: PLC0415
        from pypto.ir.distributed_compiled_program import (  # noqa: PLC0415
            DistributedConfig,
        )

        os.environ.setdefault(
            "PYPTO_PROG_BUILD_DIR",
            os.path.join(self.out_dir, "build_output"),
        )
        memory_planner = None
        if os.environ.get("PYPTO_MEM_PLANNER", "").lower() == "ptoas":
            from pypto.pypto_core import passes  # noqa: PLC0415

            memory_planner = passes.MemoryPlanner.PTOAS
        self.compiled = ir.compile(
            focused.five_layer_moe,
            platform=self.platform,
            distributed_config=DistributedConfig(
                device_ids=self.device_ids,
                num_sub_workers=0,
            ),
            skip_ptoas=False,
            dump_passes=False,
            memory_planner=memory_planner,
        )
        print(
            f"[five-layer-holder] compile OK => {self.compiled.output_dir}",
            flush=True,
        )

        self._consts = {
            "HIDDEN": cfg.HIDDEN,
            "HEAD_DIM": cfg.HEAD_DIM,
            "BATCH": cfg.BATCH,
            "UBD": focused.USER_BATCH_DYN,
            "BTF": focused.BLOCK_TABLE_FLAT_DYN,
            "RSD": focused.ROPE_SEQ_DYN,
            "KVC": focused.KV_CACHE_ROWS_DYN,
            "ROT_FULL": cfg.ROTARY_HALF_FULL * 2,
            "ROT_SWA": cfg.ROTARY_HALF_SWA * 2,
            "NHF_PAD": focused.nh_full_pad,
            "NHS_PAD": focused.nh_swa_pad,
            "HQ_FULL": focused.hidden_q_full,
            "HQ_SWA": focused.hidden_q_swa,
            "N_FULL": focused.N_FULL_FIVE,
            "N_SWA": focused.N_SWA_FIVE,
        }
        if self.kv_ipc and int(
            os.environ.get("PYPTO_STEP3P5_KV_GROUP_COUNT", "1")
        ) != 1:
            raise ValueError(
                "the focused program accepts one KV metadata group; use "
                "--disable-hybrid-kv-cache-manager for this diagnostic"
            )
        return self

    def __enter__(self):
        if self.compiled is None:
            raise RuntimeError("call build() before entering the holder")

        c = self._consts
        tp = self.tp
        hidden = c["HIDDEN"]
        head_dim = c["HEAD_DIM"]
        batch = c["BATCH"]
        ubd = c["UBD"]
        btf = c["BTF"]
        rsd = c["RSD"]
        kvc = c["KVC"]
        rot_full = c["ROT_FULL"]
        rot_swa = c["ROT_SWA"]
        nhf_pad = c["NHF_PAD"]
        nhs_pad = c["NHS_PAD"]
        hq_full = c["HQ_FULL"]
        hq_swa = c["HQ_SWA"]

        # All host tensors must exist before prepare() forks chip workers.
        self.current_hidden = _zsh(tp, batch, hidden)
        self.num_tokens_per_owner = torch.zeros(
            (128,),
            dtype=_I32,
        ).share_memory_()
        self.num_tokens_per_owner[:tp].fill_(batch)

        self.gate_r_full = _zsh(
            tp,
            c["N_FULL"],
            nhf_pad,
            hq_full,
        )
        self.gate_r_swa = _zsh(
            tp,
            c["N_SWA"],
            nhs_pad,
            hq_swa,
        )
        for head in range(hq_full // head_dim):
            self.gate_r_full[
                :,
                :,
                head,
                head * head_dim : (head + 1) * head_dim,
            ] = 1.0
        for head in range(hq_swa // head_dim):
            self.gate_r_swa[
                :,
                :,
                head,
                head * head_dim : (head + 1) * head_dim,
            ] = 1.0

        self.seq_lens = torch.ones(tp, ubd, dtype=_I32).share_memory_()
        self.block_table = torch.zeros(tp, btf, dtype=_I32).share_memory_()
        self.slot_mapping = (
            torch.arange(ubd, dtype=_I32)
            .unsqueeze(0)
            .repeat(tp, 1)
            .contiguous()
            .share_memory_()
        )
        self.rope_cf = _zsh(tp, rsd, rot_full, dtype=_F32)
        self.rope_sf = _zsh(tp, rsd, rot_full, dtype=_F32)
        self.rope_cs = _zsh(tp, rsd, rot_swa, dtype=_F32)
        self.rope_ss = _zsh(tp, rsd, rot_swa, dtype=_F32)
        self.k_cache = _zsh(tp, kvc, head_dim)
        self.v_cache = _zsh(tp, kvc, head_dim)
        self._hidden_l3_out = _zsh(tp, batch, hidden)
        self._hidden_l4_out = _zsh(tp, batch, hidden)

        from tools.step3p5.pypto_weight_ipc import (  # noqa: PLC0415
            build_stacked_weight,
            import_weights_all,
        )

        self._prepare_cm = self.compiled.prepare(persistent=True)
        self.rt = self._prepare_cm.__enter__()
        self._wmaps = import_weights_all(
            self.rt,
            self.out_dir,
            tp=tp,
            dev_offset=self.dev_offset,
        )

        if self.kv_ipc:
            from tools.step3p5.pypto_kv_ipc import (  # noqa: PLC0415
                build_stacked_kv_pool,
                import_kv_all,
            )
            from tools.step3p5.kv_padding import (  # noqa: PLC0415
                make_padding_reserve,
            )

            self._kv_maps = import_kv_all(
                self.rt,
                self.out_dir,
                tp=tp,
                dev_offset=self.dev_offset,
            )
            summary = self._kv_maps[0].summary
            self.padding_reserve = make_padding_reserve(
                summary.scheduler_num_blocks,
                summary.physical_num_blocks,
                block_size=summary.block_size,
                storage_capacity=batch,
            )
            self.k_cache, self.v_cache = build_stacked_kv_pool(self._kv_maps)
            imported_rows = int(
                self._kv_maps[0].section_spec("K")[1][0]
            )
            if imported_rows != kvc:
                raise ValueError(
                    "compiled KV rows do not match the imported full 45-layer "
                    f"pool: compiled={kvc}, imported={imported_rows}"
                )

        self._initialize_rope_tables()

        def weight(key):
            return build_stacked_weight(self._wmaps, key)

        def weight_slots(key, slots):
            from pypto.runtime.device_tensor import (  # noqa: PLC0415
                StackedDeviceTensor,
            )

            start = slots[0]
            stop = slots[-1] + 1
            if tuple(slots) != tuple(range(start, stop)):
                raise ValueError(
                    f"focused weight slots must be contiguous, got {slots}"
                )
            shards = [
                self._wmaps[rank].device_tensor(key)[start:stop]
                for rank in range(tp)
            ]
            full_shape = (tp, *tuple(shards[0].shape))
            return StackedDeviceTensor(
                shards,
                full_shape,
                list(range(tp)),
            )

        keys = self._K
        args = [self.current_hidden]
        args += [
            weight(keys.KEY_INPUT_RMS),
            weight(keys.KEY_POST_ATTN_RMS),
            weight(keys.KEY_Q_NORM),
            weight(keys.KEY_K_NORM),
        ]
        args += [
            weight_slots(keys.KEY_WQ_FULL, _FULL_SLOTS),
            weight_slots(keys.KEY_WK_FULL, _FULL_SLOTS),
            weight_slots(keys.KEY_WV_FULL, _FULL_SLOTS),
            weight_slots(keys.KEY_WO_FULL, _FULL_SLOTS),
            weight_slots(keys.KEY_WG_FULL, _FULL_SLOTS),
            self.gate_r_full,
        ]
        args += [
            weight_slots(keys.KEY_WQ_SWA, _SWA_SLOTS),
            weight_slots(keys.KEY_WK_SWA, _SWA_SLOTS),
            weight_slots(keys.KEY_WV_SWA, _SWA_SLOTS),
            weight_slots(keys.KEY_WO_SWA, _SWA_SLOTS),
            weight_slots(keys.KEY_WG_SWA, _SWA_SLOTS),
            self.gate_r_swa,
        ]
        args += [
            weight(keys.KEY_DENSE_GATE),
            weight(keys.KEY_DENSE_UP),
            weight(keys.KEY_DENSE_DOWN),
        ]
        args += [
            weight_slots(keys.KEY_MOE_GATE_W, _MOE_SLOTS),
            weight_slots(keys.KEY_MOE_ROUTER_BIAS, _MOE_SLOTS),
            weight_slots(keys.KEY_MOE_W_GATE_R, _MOE_SLOTS),
            weight_slots(keys.KEY_MOE_W_GATE_R_SCALE, _MOE_SLOTS),
            weight_slots(keys.KEY_MOE_W_UP_R, _MOE_SLOTS),
            weight_slots(keys.KEY_MOE_W_UP_R_SCALE, _MOE_SLOTS),
            weight_slots(keys.KEY_MOE_W_DOWN_R, _MOE_SLOTS),
            weight_slots(keys.KEY_MOE_W_DOWN_R_SCALE, _MOE_SLOTS),
            weight_slots(keys.KEY_MOE_W_GATE_S, _MOE_SLOTS),
            weight_slots(keys.KEY_MOE_W_UP_S, _MOE_SLOTS),
            weight_slots(keys.KEY_MOE_W_DOWN_S, _MOE_SLOTS),
        ]
        args += [
            self.seq_lens,
            self.block_table,
            self.slot_mapping,
            self.rope_cf,
            self.rope_sf,
            self.rope_cs,
            self.rope_ss,
            self.k_cache,
            self.v_cache,
            self._hidden_l3_out,
            self._hidden_l4_out,
            self.num_tokens_per_owner,
        ]
        self._args_list = args
        print(
            f"[five-layer-holder] resident args={len(args)} "
            f"program={self.program_name}",
            flush=True,
        )
        return self

    def run(self, *, dfx: str = ""):
        """Run one focused forward and expose both post-MoE hidden states."""
        if self.rt is None:
            raise RuntimeError("enter the holder before run()")
        started = time.time()
        if dfx:
            from pypto.runtime.runner import RunConfig  # noqa: PLC0415

            config = RunConfig(
                platform=self.platform,
                enable_dep_gen=(dfx == "dep"),
                enable_l2_swimlane=(dfx in {"swim", "l2"}),
                enable_scope_stats=(dfx == "scope"),
                enable_pmu=(
                    int(os.environ.get("FIVE_LAYER_MOE_PMU", "1"))
                    if dfx == "pmu"
                    else 0
                ),
            )
            self.rt.run(
                self.compiled,
                *self._args_list,
                config=config,
            )
        else:
            self.rt.run(self.compiled, *self._args_list)
        self._last_run_sec = time.time() - started
        return {
            "hidden_l3": self._hidden_l3_out,
            "hidden_l4": self._hidden_l4_out,
        }


__all__ = ["FiveLayerMoeHolder"]
