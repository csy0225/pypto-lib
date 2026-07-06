"""MoE-block real-W8A8 DEVICE precision (Option-C's MoE half).

Feed the vLLM-dumped ``layer_XX_post_attn_residual`` into the STANDALONE
``EpTpMoE`` block (per-rank REAL W8A8 experts, canonical TP=8/EP=8 on 8 cards)
and compare the block output to the vLLM-dumped ``ffn_out`` /
``moe_after_allreduce``. This is the real-weight MoE-block device precision the
prior MoE ST never covered (it used ZEROED experts).

Integration constraints handled here (see memory whole-model design):
  * PER-RANK EP weights: slot r holds rank r's OWN 36 experts (w_*_r/w_*_s
    EP/TP-sharded); x/gate_w/router_bias replicated. We load all 8 ranks'
    layer-L slices sequentially (free each ~47GB bundle after slicing).
  * T mismatch: dump is prefill T=18, EpTpMoE is decode T=BATCH=16; MoE is
    token-independent -> feed post_attn_residual[0:16], compare ffn_out[0:16].

  python _stage_moe_block_precision.py --layer 3 --dev-offset 8 \
      --dump <dump-dir> --ckpt <w8a8-ckpt> [--target ffn_out|moe_after_allreduce] [--run-only]

env: source /usr/local/Ascend/cann/set_env.sh && source $WS/activate.sh &&
     export PTO_ISA_ROOT=$WS/pto-isa && export PYTHONPATH=$WS/pypto/python:$WS/pypto-lib
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import torch


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    p.add_argument("--layer", type=int, default=3, help="MoE layer index (3..44)")
    p.add_argument("--dev-offset", type=int, default=8, help="first card (uses off..off+7)")
    p.add_argument("--dump", required=True, help="vLLM golden dump dir (beijing_1tok/dump)")
    p.add_argument("--ckpt", required=True, help="W8A8 checkpoint dir")
    p.add_argument("--target", default="ffn_out",
                   choices=["ffn_out", "moe_after_allreduce", "moe_parts_shared", "moe_parts_routed"],
                   help="golden tensor to compare EpTpMoE moe_out against")
    p.add_argument("--run-only", action="store_true",
                   help="skip golden compare; just confirm the block RUNS on device")
    p.add_argument("--zero-routed", action="store_true",
                   help="zero routed expert weights -> moe_out = sh_y (shared) only; isolates sh_y vs routed")
    p.add_argument("--zero-shared", action="store_true",
                   help="zero shared expert weights -> moe_out = weighted routed only")
    p.add_argument("--bypass-gate", action="store_true",
                   help="feed vLLM moe_router topk via gate_w; skip on-device gate top-k (TSORT hang)")
    p.add_argument("--torch-golden", action="store_true",
                   help="compare device vs torch reference (dequant weights) not the dump")
    return p.parse_args()


def _load_dump_tensor(dump_dir: str, layer: int, name: str, rank: int = 0,
                      inner_key: str | None = None):
    """Load NNN_rank{rank}_layer_{layer:02d}_{name}.pt and return its [.,HIDDEN] tensor.

    ``inner_key`` selects the tensor inside the dict when ambiguous (e.g.
    post_attn_residual has both ``hidden_states`` and ``attn_delta``; ffn_out
    has both ``ffn_output`` and ``hidden_states``). EpTpMoE moe_out matches
    ffn_out's ``ffn_output`` (== moe_after_allreduce; golden-ref #16).
    """
    pat = os.path.join(dump_dir, f"*_rank{rank}_layer_{layer:02d}_{name}.pt")
    matches = sorted(glob.glob(pat))
    if not matches:
        raise FileNotFoundError(f"no dump match: {pat}")
    obj = torch.load(matches[-1], map_location="cpu")
    if isinstance(obj, dict):
        if inner_key is not None:
            if inner_key not in obj or not torch.is_tensor(obj[inner_key]):
                raise KeyError(f"{inner_key!r} not a tensor in {matches[-1]} keys={list(obj)}")
            return obj[inner_key]
        for k in ("ffn_output", "hidden_states", name, "tensor", "out", "value"):
            if k in obj and torch.is_tensor(obj[k]):
                return obj[k]
        for v in obj.values():
            if torch.is_tensor(v) and v.dim() >= 2:
                return v
        raise TypeError(f"no tensor in dump dict {matches[-1]} keys={list(obj)}")
    if torch.is_tensor(obj):
        return obj
    raise TypeError(f"unexpected dump obj type {type(obj)}: {matches[-1]}")


def main() -> int:
    args = _parse_args()
    if args.bypass_gate:
        os.environ["EPMOE_BYPASS_GATE"] = "1"  # must precede model imports
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root))

    from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415
    set_backend_type(BackendType.Ascend910B)

    import models.step3p5.config as cfg_mod  # noqa: PLC0415, F401
    from models.step3p5.config import (  # noqa: PLC0415
        BATCH, HIDDEN, MOE_INTERMEDIATE, MOE_NUM_EXPERTS, MOE_NUM_EXPERTS_LOCAL,
        SHARE_EXPERT_DIM_LOCAL, TP_WORLD_SIZE,
    )
    from models.step3p5 import weight_loader as wl  # noqa: PLC0415
    from models.step3p5.decode_layer import select_moe_block  # noqa: PLC0415
    from golden import TensorSpec, ratio_allclose, run  # noqa: PLC0415

    bf16 = torch.bfloat16
    TP = TP_WORLD_SIZE
    T = BATCH
    INTER = MOE_INTERMEDIATE
    N_LOC = MOE_NUM_EXPERTS_LOCAL
    SH = SHARE_EXPERT_DIM_LOCAL
    pos = args.layer - 3
    if pos < 0:
        raise ValueError(f"layer {args.layer} is not MoE (>=3)")

    program = select_moe_block(args.layer)
    prog_name = getattr(program, "name", None) or type(program).__name__
    print(f"[moe-prec] layer={args.layer} program={prog_name} TP={TP} "
          f"T={T} HIDDEN={HIDDEN} INTER={INTER} N_LOC={N_LOC} SH={SH}", flush=True)

    # ---- input: dumped post_attn_residual[0:T], broadcast to [TP,T,HIDDEN] ----
    resid = _load_dump_tensor(args.dump, args.layer, "post_attn_norm", rank=0,
                              inner_key="hidden_states")
    resid = resid.reshape(-1, HIDDEN)[:T].to(bf16)
    if resid.shape[0] < T:
        resid = torch.cat([resid, torch.zeros(T - resid.shape[0], HIDDEN, dtype=bf16)], 0)
    x = resid.unsqueeze(0).expand(TP, T, HIDDEN).contiguous()
    print(f"[moe-prec] input post_attn_residual -> x{tuple(x.shape)}", flush=True)

    # ---- per-rank weights: load each rank's bundle, slice layer-L, free ----
    wg_r = torch.empty(TP, N_LOC, HIDDEN, INTER, dtype=bf16)
    wu_r = torch.empty(TP, N_LOC, HIDDEN, INTER, dtype=bf16)
    wd_r = torch.empty(TP, N_LOC, INTER, HIDDEN, dtype=bf16)
    wg_s = torch.empty(TP, HIDDEN, SH, dtype=bf16)
    wu_s = torch.empty(TP, HIDDEN, SH, dtype=bf16)
    wd_s = torch.empty(TP, SH, HIDDEN, dtype=bf16)
    gate_w0 = None
    router_bias0 = None
    for r in range(TP):
        print(f"[moe-prec] loading rank {r} bundle ...", flush=True)
        b = wl.load_step3p5_weights_for_rank(args.ckpt, r, TP)
        wg_r[r] = b[wl.KEY_MOE_W_GATE_R][pos].to(bf16)
        wu_r[r] = b[wl.KEY_MOE_W_UP_R][pos].to(bf16)
        wd_r[r] = b[wl.KEY_MOE_W_DOWN_R][pos].to(bf16)
        wg_s[r] = b[wl.KEY_MOE_W_GATE_S][pos].to(bf16)
        wu_s[r] = b[wl.KEY_MOE_W_UP_S][pos].to(bf16)
        wd_s[r] = b[wl.KEY_MOE_W_DOWN_S][pos].to(bf16)
        if r == 0:
            gate_w0 = b[wl.KEY_MOE_GATE_W][pos].to(torch.float32)
            router_bias0 = b[wl.KEY_MOE_ROUTER_BIAS][pos].to(torch.float32)
        del b
    gate_w = gate_w0.unsqueeze(0).expand(TP, HIDDEN, MOE_NUM_EXPERTS).contiguous()
    router_bias = router_bias0.unsqueeze(0).expand(TP, MOE_NUM_EXPERTS).contiguous()
    if args.bypass_gate:
        from models.step3p5.config import MOE_TOP_K as _TOPK  # noqa: PLC0415
        ROUTE_SCALE = 1.0  # vLLM topk_weights already sum to ROUTER_SCALE=3.0
        tk_ids = _load_dump_tensor(args.dump, args.layer, "moe_router", rank=0,
                                   inner_key="topk_ids").reshape(-1, _TOPK)[:T].to(torch.int64)
        tk_w = _load_dump_tensor(args.dump, args.layer, "moe_router", rank=0,
                                 inner_key="topk_weights").reshape(-1, _TOPK)[:T].to(torch.float32)
        if tk_ids.shape[0] < T:
            tk_ids = torch.cat([tk_ids, torch.zeros(T - tk_ids.shape[0], _TOPK, dtype=torch.int64)], 0)
            tk_w = torch.cat([tk_w, torch.zeros(T - tk_w.shape[0], _TOPK)], 0)
        gate_w = torch.zeros(TP, HIDDEN, MOE_NUM_EXPERTS, dtype=torch.float32)
        for _tt in range(T):
            for _k in range(_TOPK):
                gate_w[:, _tt, _k] = float(int(tk_ids[_tt, _k]))
                gate_w[:, _tt, _TOPK + _k] = float(tk_w[_tt, _k]) * ROUTE_SCALE
        print(f"[moe-prec] BYPASS-GATE: injected vLLM topk via gate_w; "
              f"ids[0]={tk_ids[0].tolist()} w[0]={(tk_w[0]*ROUTE_SCALE).tolist()}", flush=True)
    if args.zero_routed:
        wg_r.zero_(); wu_r.zero_(); wd_r.zero_()
        print("[moe-prec] ZEROED routed experts -> moe_out = sh_y (shared) only", flush=True)
    if args.zero_shared:
        wg_s.zero_(); wu_s.zero_(); wd_s.zero_()
        print("[moe-prec] ZEROED shared expert -> moe_out = weighted routed only", flush=True)
    moe_out = torch.zeros(TP, T, HIDDEN, dtype=bf16)

    inputs = {
        "x": x, "gate_w": gate_w, "router_bias": router_bias,
        "w_gate_r": wg_r, "w_up_r": wu_r, "w_down_r": wd_r,
        "w_gate_s": wg_s, "w_up_s": wu_s, "w_down_s": wd_s,
        "moe_out": moe_out,
    }
    order = [
        ("x", bf16, False), ("gate_w", torch.float32, False),
        ("router_bias", torch.float32, False),
        ("w_gate_r", bf16, False), ("w_up_r", bf16, False), ("w_down_r", bf16, False),
        ("w_gate_s", bf16, False), ("w_up_s", bf16, False), ("w_down_s", bf16, False),
        ("moe_out", bf16, True),
    ]
    specs = [TensorSpec(nm, list(inputs[nm].shape), dt,
                        init_value=None if o else inputs[nm], is_output=o)
             for (nm, dt, o) in order]

    from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415
    compile_cfg = {"distributed_config": DistributedConfig(
        device_ids=[args.dev_offset + d for d in range(TP)], num_sub_workers=0)}
    runtime_cfg = dict(platform=args.platform, device_id=args.dev_offset)

    if args.run_only:
        res = run(program=program, specs=specs, runtime_cfg=runtime_cfg,
                  compile_cfg=compile_cfg)
        print(f"[moe-prec] RUN-ONLY: {res}", flush=True)
        return 0 if res.passed else 1

    if args.target in ("moe_parts_shared", "moe_parts_routed"):
        _pk = "shared_output" if args.target == "moe_parts_shared" else "routed_output"
        tgt = _load_dump_tensor(args.dump, args.layer, "moe_parts", rank=0, inner_key=_pk)
    else:
        _tgt_inner = "ffn_output" if args.target == "ffn_out" else None
        tgt = _load_dump_tensor(args.dump, args.layer, args.target, rank=0, inner_key=_tgt_inner)
    tgt = tgt.reshape(-1, HIDDEN)[:T].to(torch.float32)
    if tgt.shape[0] < T:
        tgt = torch.cat([tgt, torch.zeros(T - tgt.shape[0], HIDDEN)], 0)

    if args.torch_golden:
        import torch.nn.functional as _F  # noqa: PLC0415
        xf = x[0].float()  # [T,HIDDEN] replicated
        sh_ref = torch.zeros(T, HIDDEN, dtype=torch.float32)
        if not args.zero_shared:
            for r in range(TP):
                g = xf @ wg_s[r].float(); u = xf @ wu_s[r].float()
                sh_ref += (_F.silu(g) * u) @ wd_s[r].float()
        ro_ref = torch.zeros(T, HIDDEN, dtype=torch.float32)
        if not args.zero_routed:
            tk_ids = _load_dump_tensor(args.dump, args.layer, 'moe_router', rank=0, inner_key='topk_ids').reshape(-1, 8)[:T].long()
            tk_w = _load_dump_tensor(args.dump, args.layer, 'moe_router', rank=0, inner_key='topk_weights').reshape(-1, 8)[:T].float()
            for t in range(T):
                for kk in range(8):
                    eid = int(tk_ids[t, kk]); w = float(tk_w[t, kk])
                    dst = eid // N_LOC; le = eid % N_LOC
                    g = xf[t:t+1] @ wg_r[dst, le].float(); u = xf[t:t+1] @ wu_r[dst, le].float()
                    ro_ref[t:t+1] += w * ((_F.silu(g) * u) @ wd_r[dst, le].float())
        tgt = (sh_ref + ro_ref)
        print(f'[moe-prec] TORCH-GOLDEN: sh_ref|max|={float(sh_ref.abs().max()):.3f} ro_ref|max|={float(ro_ref.abs().max()):.3f}', flush=True)
    def golden_fn(values):
        for r in range(TP):
            values["moe_out"][r] = tgt.to(bf16)

    res = run(program=program, specs=specs, golden_fn=golden_fn,
              runtime_cfg=runtime_cfg, compile_cfg=compile_cfg,
              rtol=4e-2, atol=4e-2,
              compare_fn={"moe_out": ratio_allclose(atol=4e-2, rtol=4e-2, max_error_ratio=0.10)})
    print(f"[moe-prec] layer={args.layer} target={args.target} DEVICE: {res}", flush=True)
    return 0 if res.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
