# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Resident holder for the instrumented Step3p5 L0-L4 route program."""
from __future__ import annotations

import os
import sys
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
_NORM_SLOTS = (0, 1, 2, 3, 4)
_KV_NUM_LAYERS = 5


def assemble_route_outputs(
    recv_meta_l3: torch.Tensor,
    recv_meta_l4: torch.Tensor,
    *,
    tp: int = 8,
    n_local_experts: int = 36,
    n_local_experts_pad: int = 40,
    local_expert_count_l3: torch.Tensor | None = None,
    local_expert_count_l4: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate diagonal local-owner snapshots and explicit counts."""
    expected = (tp, tp, n_local_experts_pad)
    for layer, tensor in (("L3", recv_meta_l3), ("L4", recv_meta_l4)):
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"{layer} recv_meta shape={tuple(tensor.shape)}, "
                f"expected={expected}"
            )
        if tensor.dtype != _I32:
            raise ValueError(
                f"{layer} recv_meta dtype={tensor.dtype}, expected={_I32}"
            )
        if bool(torch.any(tensor < 0)):
            raise ValueError(f"{layer} recv_meta contains negative counts")
        if bool(torch.any(tensor[:, :, n_local_experts:] != 0)):
            raise ValueError(
                f"{layer} recv_meta padding "
                f"{n_local_experts}:{n_local_experts_pad} is non-zero"
            )

    recv_meta = torch.stack((recv_meta_l3, recv_meta_l4), dim=1)
    routed = recv_meta[:, :, :, :n_local_experts]
    for owner_rank in range(tp):
        for route_owner_rank in range(tp):
            if owner_rank == route_owner_rank:
                continue
            if bool(torch.any(routed[owner_rank, :, route_owner_rank])):
                raise ValueError(
                    "local-owner route snapshot contains a non-zero "
                    "off-owner row at "
                    f"owner_rank={owner_rank}, "
                    f"route_owner_rank={route_owner_rank}"
                )
    diagonal_padded = torch.stack(
        [
            recv_meta[owner_rank, :, owner_rank]
            for owner_rank in range(tp)
        ],
        dim=0,
    )
    diagonal_padded_i64 = diagonal_padded.to(torch.int64)
    if bool(torch.any(diagonal_padded_i64 > torch.iinfo(_I32).max)):
        raise OverflowError("local expert count exceeds INT32")
    diagonal_counts = diagonal_padded[:, :, :n_local_experts]

    if (local_expert_count_l3 is None) != (local_expert_count_l4 is None):
        raise ValueError(
            "L3/L4 explicit local expert counts must be supplied together"
        )
    if local_expert_count_l3 is None:
        return recv_meta, diagonal_counts.contiguous()

    assert local_expert_count_l4 is not None
    explicit = (local_expert_count_l3, local_expert_count_l4)
    expected_count_shape = (tp, n_local_experts_pad)
    for layer, tensor in zip(("L3", "L4"), explicit, strict=True):
        if tuple(tensor.shape) != expected_count_shape:
            raise ValueError(
                f"{layer} local_expert_count shape={tuple(tensor.shape)}, "
                f"expected={expected_count_shape}"
            )
        if tensor.dtype != _I32:
            raise ValueError(
                f"{layer} local_expert_count dtype={tensor.dtype}, "
                f"expected={_I32}"
            )
        if bool(torch.any(tensor < 0)):
            raise ValueError(
                f"{layer} local_expert_count contains negative counts"
            )
        if bool(torch.any(tensor[:, n_local_experts:] != 0)):
            raise ValueError(
                f"{layer} local_expert_count padding "
                f"{n_local_experts}:{n_local_experts_pad} is non-zero"
            )

    explicit_padded = torch.stack(explicit, dim=1)
    explicit_padded_i64 = explicit_padded.to(torch.int64)
    if not torch.equal(diagonal_padded_i64, explicit_padded_i64):
        mismatch = torch.nonzero(
            diagonal_padded_i64 != explicit_padded_i64,
            as_tuple=False,
        )[0].tolist()
        raise ValueError(
            "local_expert_count disagrees with diagonal owner route row "
            f"at index={mismatch}"
        )
    return (
        recv_meta,
        explicit_padded[:, :, :n_local_experts].contiguous(),
    )


class FiveLayerMoeRouteHolder(WholeDecodeHolder):
    """Run L0-L4 while exporting exact L3/L4 local-owner route counts."""

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
        self.program_name = "five_layer_moe_route"
        self._hidden_l3_out = None
        self._hidden_l4_out = None
        self._local_expert_count_l3_out = None
        self._local_expert_count_l4_out = None
        self._recv_meta_l3_out = None
        self._recv_meta_l4_out = None

    def _infer_rows_from_map(self) -> None:
        """Infer the focused five-layer KV shape before config import."""
        import json  # noqa: PLC0415

        path = os.path.join(
            self.out_dir,
            "pypto_kvpool_map.json.rank0",
        )
        try:
            with open(path, encoding="utf-8") as file:
                obj = json.load(file)
            num_layers = int(obj["num_layers"])
            rows = int(obj["map"]["L0.K"]["num_slots"])
            physical = int(obj["physical_num_blocks"])
            scheduler = int(obj["scheduler_num_blocks"])
            from tools.step3p5.kv_padding import (  # noqa: PLC0415
                STORAGE_BATCH,
            )

            required_physical = scheduler + STORAGE_BATCH - 1
            if physical < required_physical:
                raise ValueError(
                    f"focused KV map physical={physical} lacks reserve above "
                    f"scheduler={scheduler}; required>={required_physical}"
                )
            if num_layers != _KV_NUM_LAYERS:
                raise ValueError(
                    f"focused KV map has {num_layers} layers, "
                    f"expected {_KV_NUM_LAYERS}"
                )
        except FileNotFoundError:
            return
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid focused KV padding-reserve map {path}: {exc}"
            ) from exc
        expected = _KV_NUM_LAYERS * rows
        configured = os.environ.get("PYPTO_STEP3P5_KV_CACHE_ROWS")
        if configured is not None and int(configured) != expected:
            raise ValueError(
                "PYPTO_STEP3P5_KV_CACHE_ROWS disagrees with focused KV map: "
                f"configured={configured}, expected={expected}"
            )
        os.environ["PYPTO_STEP3P5_KV_NUM_LAYERS"] = str(_KV_NUM_LAYERS)
        os.environ["PYPTO_STEP3P5_KV_CACHE_ROWS"] = str(expected)

    def build(self):
        """Compile the focused graph without preparing device state."""
        from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415

        set_backend_type(BackendType.Ascend910B)
        if self.kv_ipc:
            self._infer_rows_from_map()

        import models.step3p5.config as cfg  # noqa: PLC0415
        from models.step3p5 import weight_loader as keys  # noqa: PLC0415
        from tests.step3p5.harnesses import (  # noqa: PLC0415
            _five_layer_moe_route_program as focused,
        )

        self._cfg = cfg
        self._K = keys
        self._focused = focused
        layer_contract = {
            "config.KV_NUM_LAYERS": int(cfg.KV_NUM_LAYERS),
            "config.LAYER_DYN": int(cfg.LAYER_DYN),
            "canonical.LAYER_DYN": int(focused._canonical.LAYER_DYN),
            "focused.LAYER_DYN": int(focused.LAYER_DYN),
        }
        if set(layer_contract.values()) != {_KV_NUM_LAYERS}:
            raise ValueError(
                "focused layer ABI was imported before the five-layer "
                f"environment was configured: {layer_contract}"
            )
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
            focused.five_layer_moe_route,
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
            f"[five-layer-route-holder] compile OK => {self.compiled.output_dir}",
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
        if self._prepare_cm is not None or self.rt is not None:
            raise RuntimeError(
                "holder cleanup is incomplete; retry __exit__ before re-entering"
            )
        try:
            return self._enter_impl()
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def _enter_impl(self):
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
        self._local_expert_count_l3_out = _zsh(
            tp,
            self._focused.n_local_experts_pad,
            dtype=_I32,
        )
        self._local_expert_count_l4_out = _zsh(
            tp,
            self._focused.n_local_experts_pad,
            dtype=_I32,
        )
        self._recv_meta_l3_out = _zsh(
            tp,
            tp,
            self._focused.n_local_experts_pad,
            dtype=_I32,
        )
        self._recv_meta_l4_out = _zsh(
            tp,
            tp,
            self._focused.n_local_experts_pad,
            dtype=_I32,
        )

        from tools.step3p5.pypto_weight_ipc import (  # noqa: PLC0415
            build_stacked_weight,
            import_weights_all,
        )

        self._prepare_runtime()
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
                expected_num_layers=_KV_NUM_LAYERS,
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
                    "compiled KV rows do not match the imported five-layer "
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
                self._wmaps[rank].device_tensor_slice(key, start, stop)
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
            weight_slots(keys.KEY_INPUT_RMS, _NORM_SLOTS),
            weight_slots(keys.KEY_POST_ATTN_RMS, _NORM_SLOTS),
            weight_slots(keys.KEY_Q_NORM, _NORM_SLOTS),
            weight_slots(keys.KEY_K_NORM, _NORM_SLOTS),
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
            weight_slots(keys.KEY_MOE_GATE_W_NK, _MOE_SLOTS),
            weight_slots(keys.KEY_MOE_ROUTER_BIAS, _MOE_SLOTS),
            weight_slots(keys.KEY_MOE_W13_R, _MOE_SLOTS),
            weight_slots(keys.KEY_MOE_W13_R_SCALE, _MOE_SLOTS),
            weight_slots(keys.KEY_MOE_W_DOWN_R, _MOE_SLOTS),
            weight_slots(keys.KEY_MOE_W_DOWN_R_SCALE, _MOE_SLOTS),
            weight_slots(keys.KEY_MOE_W_GATE_S_NK, _MOE_SLOTS),
            weight_slots(keys.KEY_MOE_W_UP_S_NK, _MOE_SLOTS),
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
            self._local_expert_count_l3_out,
            self._local_expert_count_l4_out,
            self._recv_meta_l3_out,
            self._recv_meta_l4_out,
            self.num_tokens_per_owner,
        ]
        self._args_list = args
        print(
            f"[five-layer-route-holder] resident args={len(args)} "
            f"program={self.program_name}",
            flush=True,
        )
        return self

    def run(self, *, dfx: str = ""):
        """Run once and expose hidden states plus exact owner-count metadata."""
        if self.rt is None:
            raise RuntimeError("enter the holder before run()")
        self._validate_replicated_owner_counts()
        started = time.time()
        if dfx:
            from pypto.runtime.runner import RunConfig  # noqa: PLC0415

            config = RunConfig(
                platform=self.platform,
                enable_dep_gen=(dfx == "dep"),
                enable_l2_swimlane=(dfx in {"swim", "l2"}),
                l2_swimlane_reuse_dep_gen=(dfx in {"swim", "l2"}),
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
        recv_meta, local_expert_count = assemble_route_outputs(
            self._recv_meta_l3_out,
            self._recv_meta_l4_out,
            local_expert_count_l3=self._local_expert_count_l3_out,
            local_expert_count_l4=self._local_expert_count_l4_out,
            tp=self.tp,
            n_local_experts=self._focused.n_local_experts,
            n_local_experts_pad=self._focused.n_local_experts_pad,
        )
        return {
            "hidden_l3": self._hidden_l3_out,
            "hidden_l4": self._hidden_l4_out,
            "recv_meta": recv_meta,
            "local_expert_count": local_expert_count,
        }

__all__ = ["FiveLayerMoeRouteHolder", "assemble_route_outputs"]
