"""Whole-decode via PER-LAYER orchestration (option a) — #10 driver.

Runs each ALREADY-VALIDATED per-layer @pl.program (``select_decode_layer``)
SEQUENTIALLY from ONE driver, threading the hidden state layer->layer, then the
tail. This is the CORRECTNESS route (vLLM idle during pypto compute -> dissolves
the routed-MoE 507018 co-tenancy deadlock); it is NOT the fused DenseChainN perf
route (which is pypto-framework-blocked).

Scope of THIS revision (card-free milestone):
  * host-compile (``--smoke``) the dense prefix per-layer loop: layer 0 =
    full_dense, layers 1,2 = swa_dense. Both kinds share host_orch input NAMES
    (dims differ); the program selects the layer via the ``layer_idx`` scalar,
    so ONE layer-stacked weight bundle serves every layer of a kind.
  * MoE layers (3..44), DEVICE chaining (read layer i out -> feed i+1) and the
    vLLM-dump precision compare (whole_decode_compare.py) are the next
    increments (#11); left as clearly-marked TODOs.

  python _stage_whole_decode_run.py -p a2a3sim --smoke --layers 0,1,2   # host compile (no card)
  python _stage_whole_decode_run.py -p a2a3 -d 8 --layers 0,1,2         # device run (dense prefix)

env: source /usr/local/Ascend/cann/set_env.sh && source $WS/activate.sh &&
     export PTO_ISA_ROOT=$WS/pto-isa && export PYTHONPATH=$WS/pypto/python:$WS/pypto-lib
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-p", "--platform", default="a2a3sim",
                   choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    p.add_argument("-d", "--device", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layers", default="0,1,2",
                   help="comma layer indices, e.g. 0,1,2")
    p.add_argument("--tp", type=int, default=1,
                   help="1 = TP=1 single-card bring-up (dense only); "
                        "8 = canonical TP=8 (dense+MoE, DistributedConfig)")
    p.add_argument("--dev-offset", type=int, default=8,
                   help="first card for TP=8 DistributedConfig (cards off..off+7)")
    p.add_argument("--smoke", action="store_true",
                   help="compile-only per layer (host, no card)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root))

    # TP=1 single-card bring-up path (reload attn/decode_layer under TP=1 so the
    # module-level programs returned by select_decode_layer are TP=1). MoE layers
    # REQUIRE canonical TP=8 (--tp 8): the TP=1 unslice hits the SWA-MoE tile.full
    # dyn-shape (CLAUDE.md single-card iron rule), so tp1 is dense-only bring-up.
    if args.tp == 1:
        from tests.step3p5.common._tp1_setup import apply_tp1_patch  # noqa: PLC0415
        summary = apply_tp1_patch(reload_modules=[
            "models.step3p5.attention_full",
            "models.step3p5.attention_swa",
            "models.step3p5.decode_layer",
        ])
        print(f"[wd] TP=1 patch: {summary}", flush=True)
    else:
        print(f"[wd] canonical TP={args.tp} (no unslice patch)", flush=True)

    from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415
    set_backend_type({
        "a2a3": BackendType.Ascend910B, "a2a3sim": BackendType.Ascend910B,
        "a5": BackendType.Ascend950, "a5sim": BackendType.Ascend950,
    }[args.platform])

    import models.step3p5.config as cfg_mod  # noqa: PLC0415
    from models.step3p5.config import (  # noqa: PLC0415
        BATCH, HEAD_DIM, HIDDEN, LAYER_TYPE_FULL, LAYER_TYPE_SWA, LAYER_TYPES,
        MAX_BLOCKS_PER_SEQ, MAX_SEQ_DEFAULT, ROTARY_HALF_FULL, ROTARY_HALF_SWA,
        NUM_HIDDEN_LAYERS, DENSE_LAYER_INDICES, MOE_INTERMEDIATE, MOE_NUM_EXPERTS,
    )
    from models.step3p5.decode_layer import select_decode_layer  # noqa: PLC0415
    from golden import ScalarSpec, TensorSpec, run  # noqa: PLC0415

    bf16 = torch.bfloat16
    g = torch.Generator().manual_seed(args.seed)

    def _randn(shape, dtype=bf16, std=0.02):
        return torch.empty(shape, dtype=torch.float32).normal_(
            0.0, std, generator=g).to(dtype)

    def flat3(t):
        L, M, N = t.shape
        return t.reshape(1, L * M, N)

    n_full = sum(1 for t in LAYER_TYPES if t == LAYER_TYPE_FULL)
    n_swa = sum(1 for t in LAYER_TYPES if t == LAYER_TYPE_SWA)
    n_dense = len(DENSE_LAYER_INDICES)
    KV_H = cfg_mod.KV_HEADS_LOCAL * HEAD_DIM
    INT_LOC = cfg_mod.INTERMEDIATE_LOCAL

    # Shared (kind-agnostic) small weights + runtime inputs.
    w_input_rms = torch.empty(NUM_HIDDEN_LAYERS, HIDDEN).normal_(0.0, 0.05, generator=g)
    w_post_rms = torch.empty(NUM_HIDDEN_LAYERS, HIDDEN).normal_(0.0, 0.05, generator=g)
    w_q_norm = torch.empty(NUM_HIDDEN_LAYERS, HEAD_DIM).normal_(0.0, 0.05, generator=g)
    w_k_norm = torch.empty(NUM_HIDDEN_LAYERS, HEAD_DIM).normal_(0.0, 0.05, generator=g)
    w_gate = _randn([n_dense, HIDDEN, INT_LOC])
    w_up = _randn([n_dense, HIDDEN, INT_LOC])
    w_down = _randn([n_dense, INT_LOC, HIDDEN])
    seq_lens = torch.ones(1, BATCH, dtype=torch.int32)
    block_table = torch.zeros(1, MAX_BLOCKS_PER_SEQ * BATCH, dtype=torch.int32)
    slot_mapping = torch.arange(BATCH, dtype=torch.int32).unsqueeze(0)
    k_cache = torch.zeros(1, MAX_SEQ_DEFAULT, HEAD_DIM, dtype=bf16)
    v_cache = torch.zeros(1, MAX_SEQ_DEFAULT, HEAD_DIM, dtype=bf16)

    def _expand_tp(inputs, tp):
        if tp == 1:
            return inputs
        out = {}
        for k, v in inputs.items():
            # all runtime tensors here have leading dim 1 -> replicate to tp ranks
            out[k] = v.repeat(tp, *([1] * (v.dim() - 1)))
        return out

    def build_dense_inputs(full: bool, layer_idx: int, current_hidden, tp: int):
        if full:
            n_attn = n_full
            h_q = cfg_mod.NUM_HEADS_FULL_LOCAL * HEAD_DIM
            pad = cfg_mod.NUM_HEADS_FULL_LOCAL_PAD
            rotary_dim = ROTARY_HALF_FULL * 2
        else:
            n_attn = n_swa
            h_q = cfg_mod.NUM_HEADS_SWA_LOCAL * HEAD_DIM
            pad = cfg_mod.NUM_HEADS_SWA_LOCAL_PAD
            rotary_dim = ROTARY_HALF_SWA * 2
        wq = _randn([n_attn, HIDDEN, h_q]); wk = _randn([n_attn, HIDDEN, KV_H])
        wv = _randn([n_attn, HIDDEN, KV_H]); wo = _randn([n_attn, h_q, HIDDEN])
        w_g = _randn([n_attn, HIDDEN, pad])
        gate_r = torch.ones(pad, h_q, dtype=bf16)
        rope_cos = torch.empty(1, MAX_SEQ_DEFAULT, rotary_dim).normal_(0.0, 0.5, generator=g)
        rope_sin = torch.empty(1, MAX_SEQ_DEFAULT, rotary_dim).normal_(0.0, 0.5, generator=g)
        next_hidden_out = torch.zeros(1, BATCH, HIDDEN, dtype=bf16)
        inputs = {
            "current_hidden": current_hidden,
            "input_rms_weight": w_input_rms.float().unsqueeze(0),
            "wq": flat3(wq), "wk": flat3(wk), "wv": flat3(wv),
            "q_norm_weight": w_q_norm.float().unsqueeze(0),
            "k_norm_weight": w_k_norm.float().unsqueeze(0),
            "seq_lens": seq_lens, "block_table": block_table, "slot_mapping": slot_mapping,
            "rope_cos": rope_cos, "rope_sin": rope_sin, "k_cache": k_cache, "v_cache": v_cache,
            "wo": flat3(wo), "w_g": flat3(w_g), "gate_r": gate_r.unsqueeze(0),
            "post_rms_weight": w_post_rms.float().unsqueeze(0),
            "w_gate": flat3(w_gate), "w_up": flat3(w_up), "w_down": flat3(w_down),
            "next_hidden_out": next_hidden_out,
        }
        order = [
            ("current_hidden", bf16, False), ("input_rms_weight", torch.float32, False),
            ("wq", bf16, False), ("wk", bf16, False), ("wv", bf16, False),
            ("q_norm_weight", torch.float32, False), ("k_norm_weight", torch.float32, False),
            ("seq_lens", torch.int32, False), ("block_table", torch.int32, False),
            ("slot_mapping", torch.int32, False), ("rope_cos", torch.float32, False),
            ("rope_sin", torch.float32, False), ("k_cache", bf16, False), ("v_cache", bf16, False),
            ("wo", bf16, False), ("w_g", bf16, False), ("gate_r", bf16, False),
            ("post_rms_weight", torch.float32, False), ("w_gate", bf16, False),
            ("w_up", bf16, False), ("w_down", bf16, False),
            ("next_hidden_out", bf16, True),
        ]
        inputs = _expand_tp(inputs, tp)
        specs = [
            TensorSpec(nm, list(inputs[nm].shape), dt,
                       init_value=None if is_out else inputs[nm], is_output=is_out)
            for (nm, dt, is_out) in order
        ]
        specs.append(ScalarSpec("layer_idx", torch.int32,
                                value=torch.tensor(layer_idx, dtype=torch.int32)))
        return inputs, specs

    def build_moe_inputs(full: bool, layer_idx: int, current_hidden, tp: int):
        if full:
            h_q = cfg_mod.NUM_HEADS_FULL_LOCAL * HEAD_DIM
            pad = cfg_mod.NUM_HEADS_FULL_LOCAL_PAD
            rotary_dim = ROTARY_HALF_FULL * 2
            n_attn = n_full
        else:
            h_q = cfg_mod.NUM_HEADS_SWA_LOCAL * HEAD_DIM
            pad = cfg_mod.NUM_HEADS_SWA_LOCAL_PAD
            rotary_dim = ROTARY_HALF_SWA * 2
            n_attn = n_swa
        int_s = cfg_mod.SHARE_EXPERT_DIM_LOCAL
        n_loc_e = cfg_mod.MOE_NUM_EXPERTS_LOCAL
        int_r = MOE_INTERMEDIATE
        wq = _randn([n_attn, HIDDEN, h_q]); wk = _randn([n_attn, HIDDEN, KV_H])
        wv = _randn([n_attn, HIDDEN, KV_H]); wo = _randn([n_attn, h_q, HIDDEN])
        w_g = _randn([n_attn, HIDDEN, pad])
        gate_r = torch.ones(pad, h_q, dtype=bf16)
        rope_cos = torch.empty(1, MAX_SEQ_DEFAULT, rotary_dim).normal_(0.0, 0.5, generator=g)
        rope_sin = torch.empty(1, MAX_SEQ_DEFAULT, rotary_dim).normal_(0.0, 0.5, generator=g)
        gate_w = _randn([HIDDEN, MOE_NUM_EXPERTS], torch.float32, std=0.02).float()
        router_bias = torch.zeros([MOE_NUM_EXPERTS], dtype=torch.float32)
        w_gate_r = torch.zeros(n_loc_e, HIDDEN, int_r, dtype=bf16)
        w_up_r = torch.zeros(n_loc_e, HIDDEN, int_r, dtype=bf16)
        w_down_r = torch.zeros(n_loc_e, int_r, HIDDEN, dtype=bf16)
        w_gate_s = torch.zeros(HIDDEN, int_s, dtype=bf16)
        w_up_s = torch.zeros(HIDDEN, int_s, dtype=bf16)
        w_down_s = torch.zeros(int_s, HIDDEN, dtype=bf16)
        next_hidden_out = torch.zeros(1, BATCH, HIDDEN, dtype=bf16)
        inputs = {
            "current_hidden": current_hidden,
            "input_rms_weight": w_input_rms.float().unsqueeze(0),
            "wq": flat3(wq), "wk": flat3(wk), "wv": flat3(wv),
            "q_norm_weight": w_q_norm.float().unsqueeze(0),
            "k_norm_weight": w_k_norm.float().unsqueeze(0),
            "seq_lens": seq_lens, "block_table": block_table, "slot_mapping": slot_mapping,
            "rope_cos": rope_cos, "rope_sin": rope_sin, "k_cache": k_cache, "v_cache": v_cache,
            "wo": flat3(wo), "w_g": flat3(w_g), "gate_r": gate_r.unsqueeze(0),
            "post_rms_weight": w_post_rms.float().unsqueeze(0),
            "gate_w": gate_w.unsqueeze(0), "router_bias": router_bias.unsqueeze(0),
            "w_gate_r": w_gate_r.unsqueeze(0), "w_up_r": w_up_r.unsqueeze(0),
            "w_down_r": w_down_r.unsqueeze(0),
            "w_gate_s": w_gate_s.unsqueeze(0), "w_up_s": w_up_s.unsqueeze(0),
            "w_down_s": w_down_s.unsqueeze(0),
            "next_hidden_out": next_hidden_out,
        }
        order = [
            ("current_hidden", bf16, False), ("input_rms_weight", torch.float32, False),
            ("wq", bf16, False), ("wk", bf16, False), ("wv", bf16, False),
            ("q_norm_weight", torch.float32, False), ("k_norm_weight", torch.float32, False),
            ("seq_lens", torch.int32, False), ("block_table", torch.int32, False),
            ("slot_mapping", torch.int32, False), ("rope_cos", torch.float32, False),
            ("rope_sin", torch.float32, False), ("k_cache", bf16, False), ("v_cache", bf16, False),
            ("wo", bf16, False), ("w_g", bf16, False), ("gate_r", bf16, False),
            ("post_rms_weight", torch.float32, False),
            ("gate_w", torch.float32, False), ("router_bias", torch.float32, False),
            ("w_gate_r", bf16, False), ("w_up_r", bf16, False), ("w_down_r", bf16, False),
            ("w_gate_s", bf16, False), ("w_up_s", bf16, False), ("w_down_s", bf16, False),
            ("next_hidden_out", bf16, True),
        ]
        inputs = _expand_tp(inputs, tp)
        specs = [
            TensorSpec(nm, list(inputs[nm].shape), dt,
                       init_value=None if is_out else inputs[nm], is_output=is_out)
            for (nm, dt, is_out) in order
        ]
        specs.append(ScalarSpec("layer_idx", torch.int32,
                                value=torch.tensor(layer_idx, dtype=torch.int32)))
        return inputs, specs

    layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    print(f"[wd] layers={layers} n_full={n_full} n_swa={n_swa} n_dense={n_dense}", flush=True)

    # Single-rank allocate_domain stub (only needed for device execute at TP=1).
    from simpler.orchestrator import Orchestrator  # noqa: PLC0415
    from simpler.task_interface import ChipDomainContext, CommDomainHandle  # noqa: PLC0415
    _orig = Orchestrator.allocate_domain

    def _stub(self, *, name, workers, window_size, buffers):
        workers = tuple(int(w) for w in workers)
        if len(workers) > 1:
            return _orig(self, name=name, workers=workers, window_size=window_size, buffers=buffers)
        chip_idx = workers[0]; base = int(self.malloc(chip_idx, int(window_size)))
        off = 0; ptrs = {}
        for spec in buffers:
            ptrs[spec.name] = base + off; off += int(spec.nbytes)
        ctx = ChipDomainContext(name=str(name), domain_rank=0, domain_size=1, device_ctx=0,
                                local_window_base=base, actual_window_size=int(window_size),
                                buffer_ptrs=ptrs)

        def _rel(_h):
            try:
                self.free(chip_idx, base)
            except Exception:
                pass
        return CommDomainHandle(name=str(name), workers=workers, contexts={chip_idx: ctx},
                                allocation_id=-1, _release_fn=_rel)

    # compile_cfg: canonical TP=8 needs DistributedConfig over cards off..off+7.
    compile_cfg = {}
    if args.tp > 1:
        from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415
        compile_cfg["distributed_config"] = DistributedConfig(
            device_ids=[args.dev_offset + d for d in range(args.tp)],
            num_sub_workers=0,
        )

    if args.tp == 1:
        Orchestrator.allocate_domain = _stub
    rc = 0
    try:
        current_hidden = _randn([1, BATCH, HIDDEN], std=0.02)
        for li in layers:
            program, kind = select_decode_layer(li)
            if kind in ("full_dense", "swa_dense"):
                full = kind == "full_dense"
                inputs, specs = build_dense_inputs(full, li, current_hidden, args.tp)
            else:
                # MoE layer (full/swa x silu/swiglu variants). Zeroed expert
                # weights here = attention-only compile/run check; real W8A8 +
                # precision is the #11 device path.
                full = kind.startswith("full")
                inputs, specs = build_moe_inputs(full, li, current_hidden, args.tp)
            runtime_cfg = dict(platform=args.platform, device_id=args.device)
            if args.smoke or args.platform.endswith("sim"):
                res = run(program=program, specs=specs, runtime_cfg=runtime_cfg,
                          compile_only=True, compile_cfg=compile_cfg)
                print(f"[wd] layer {li} kind={kind} COMPILE: {res}", flush=True)
                if not res.passed:
                    rc = 1
                    break
            else:
                # DEVICE run (validation skipped: no golden here). Device-output
                # chaining (read next_hidden_out to feed li+1) is the #11 TODO;
                # for now re-seed current_hidden per layer just to exercise run.
                res = run(program=program, specs=specs, runtime_cfg=runtime_cfg,
                          compile_cfg=compile_cfg)
                print(f"[wd] layer {li} kind={kind} DEVICE: {res}", flush=True)
                if not res.passed:
                    rc = 1
                    break
    finally:
        Orchestrator.allocate_domain = _orig
    print(f"[wd] done rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
