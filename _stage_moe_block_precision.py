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
                   choices=["ffn_out", "moe_after_allreduce", "moe_parts_shared", "moe_parts_routed", "out"],
                   help="golden tensor to compare against: ffn_out/moe_after_allreduce/"
                        "moe_parts_* (moe_out, pre-residual), or 'out' "
                        "(next_hidden_out = resid1+moe_out, post-norm+residual glue path)")
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
    p.add_argument("--dump-stages", action="store_true",
                   help="dump local_routed_x/y (post-dispatch/post-expert) + compare valid rows to torch ref (requires --bypass-gate)")
    p.add_argument("--w8a8-native", action="store_true",
                   help="gap-5: build INT8-native routed program (select_moe_block w8a8_native=True) + feed INT8 routed weights + per-channel scales")
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
    from golden import ScalarSpec, TensorSpec, ratio_allclose, run  # noqa: PLC0415

    bf16 = torch.bfloat16
    TP = TP_WORLD_SIZE
    T = BATCH
    INTER = MOE_INTERMEDIATE
    N_LOC = MOE_NUM_EXPERTS_LOCAL
    SH = SHARE_EXPERT_DIM_LOCAL
    pos = args.layer - 3
    if pos < 0:
        raise ValueError(f"layer {args.layer} is not MoE (>=3)")

    program = select_moe_block(args.layer, w8a8_native=args.w8a8_native)
    prog_name = getattr(program, "name", None) or type(program).__name__
    print(f"[moe-prec] layer={args.layer} program={prog_name} TP={TP} "
          f"T={T} HIDDEN={HIDDEN} INTER={INTER} N_LOC={N_LOC} SH={SH}", flush=True)

    # ---- input: broadcast to [TP,T,HIDDEN] ----
    # --target out: feed UN-NORMED resid1 (the post-attn residual stream);
    #   the extended chip_orch norms it internally. resid1 source = the live
    #   residual stream = post_attn_residual["hidden_states"] (already summed
    #   prev_residual+attn_delta in vLLM; attn_delta is stored separately for
    #   reference only — do NOT add it). Verified: RMSNorm(resid1)*(gamma+1) ==
    #   post_attn_norm (BF16 rounding); out == resid1 + ffn_output.
    # other targets: feed the NORMED post_attn_norm (old ffn_out path; chip_orch
    #   then runs without its norm prologue — only valid for the PRE-extension
    #   moe_block. Kept for backward compatibility.)
    if args.target == "out":
        resid = _load_dump_tensor(args.dump, args.layer, "post_attn_residual",
                                  rank=0, inner_key="hidden_states")
        in_label = "post_attn_residual.hidden_states (UN-NORMED resid1)"
    else:
        resid = _load_dump_tensor(args.dump, args.layer, "post_attn_norm", rank=0,
                                  inner_key="hidden_states")
        in_label = "post_attn_norm.hidden_states (NORMED x)"
    resid = resid.reshape(-1, HIDDEN)[:T].to(bf16)
    if resid.shape[0] < T:
        resid = torch.cat([resid, torch.zeros(T - resid.shape[0], HIDDEN, dtype=bf16)], 0)
    x = resid.unsqueeze(0).expand(TP, T, HIDDEN).contiguous()
    print(f"[moe-prec] input {in_label} -> x{tuple(x.shape)}", flush=True)

    # ---- per-rank weights: load each rank's bundle, slice layer-L, free ----
    wg_r = torch.empty(TP, N_LOC, HIDDEN, INTER, dtype=bf16)
    wu_r = torch.empty(TP, N_LOC, HIDDEN, INTER, dtype=bf16)
    wd_r = torch.empty(TP, N_LOC, INTER, HIDDEN, dtype=bf16)
    wg_s = torch.empty(TP, HIDDEN, SH, dtype=bf16)
    wu_s = torch.empty(TP, HIDDEN, SH, dtype=bf16)
    wd_s = torch.empty(TP, SH, HIDDEN, dtype=bf16)
    wg_r_i8 = wu_r_i8 = wd_r_i8 = None
    wg_r_sc = wu_r_sc = wd_r_sc = None
    if args.w8a8_native:
        wg_r_i8 = torch.empty(TP, N_LOC, HIDDEN, INTER, dtype=torch.int8)
        wu_r_i8 = torch.empty(TP, N_LOC, HIDDEN, INTER, dtype=torch.int8)
        wd_r_i8 = torch.empty(TP, N_LOC, INTER, HIDDEN, dtype=torch.int8)
        wg_r_sc = torch.empty(TP, N_LOC, INTER, dtype=torch.float32)
        wu_r_sc = torch.empty(TP, N_LOC, INTER, dtype=torch.float32)
        wd_r_sc = torch.empty(TP, N_LOC, HIDDEN, dtype=torch.float32)
    gate_w0 = None
    router_bias0 = None
    post_rms0 = None  # post_attention_layernorm weight (replicated across TP)
    for r in range(TP):
        print(f"[moe-prec] loading rank {r} bundle ...", flush=True)
        b = wl.load_step3p5_weights_for_rank(args.ckpt, r, TP, w8a8_native=args.w8a8_native)
        wg_r[r] = b[wl.KEY_MOE_W_GATE_R][pos].to(bf16)
        wu_r[r] = b[wl.KEY_MOE_W_UP_R][pos].to(bf16)
        wd_r[r] = b[wl.KEY_MOE_W_DOWN_R][pos].to(bf16)
        if args.w8a8_native:
            wg_r_i8[r] = b[wl.KEY_MOE_W_GATE_R_I8][pos]
            wu_r_i8[r] = b[wl.KEY_MOE_W_UP_R_I8][pos]
            wd_r_i8[r] = b[wl.KEY_MOE_W_DOWN_R_I8][pos]
            wg_r_sc[r] = b[wl.KEY_MOE_W_GATE_R_SCALE][pos].to(torch.float32)
            wu_r_sc[r] = b[wl.KEY_MOE_W_UP_R_SCALE][pos].to(torch.float32)
            wd_r_sc[r] = b[wl.KEY_MOE_W_DOWN_R_SCALE][pos].to(torch.float32)
        wg_s[r] = b[wl.KEY_MOE_W_GATE_S][pos].to(bf16)
        wu_s[r] = b[wl.KEY_MOE_W_UP_S][pos].to(bf16)
        wd_s[r] = b[wl.KEY_MOE_W_DOWN_S][pos].to(bf16)
        if r == 0:
            gate_w0 = b[wl.KEY_MOE_GATE_W][pos].to(torch.float32)
            router_bias0 = b[wl.KEY_MOE_ROUTER_BIAS][pos].to(torch.float32)
            # KEY_POST_ATTN_RMS shape [NUM_HIDDEN_LAYERS=45, HIDDEN] (ALL layers,
            # NOT MoE-relative) -> keep the FULL stack; moe_block indexes it by
            # layer_idx=args.layer internally. Cast BF16->FP32 to match the
            # host_orch post_rms_weight FP32 param. (Do NOT slice [pos]=layer-3,
            # which is the WRONG layer for a 45-row stack.)
            post_rms0 = b[wl.KEY_POST_ATTN_RMS].to(torch.float32)
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

    # ---- NEW extended moe_block (norm+moe+residual) validation: feed un-normed
    # resid1 + post_rms_weight + layer_idx, compare next_hidden_out vs out.pt.
    # Self-contained (matches the post-task-#4 host_orch signature); returns early.
    if args.target == "out":
        post_rms = post_rms0.unsqueeze(0).expand(
            TP, post_rms0.shape[0], HIDDEN).contiguous()  # [TP, 45, HIDDEN] FP32
        nh_out = torch.zeros(TP, T, HIDDEN, dtype=bf16)
        _inp = {
            "resid1": x, "post_rms_weight": post_rms, "gate_w": gate_w,
            "router_bias": router_bias,
            "w_gate_s": wg_s, "w_up_s": wu_s, "w_down_s": wd_s,
            "next_hidden_out": nh_out,
        }
        _ord = [
            ("resid1", bf16, False), ("post_rms_weight", torch.float32, False),
            ("gate_w", torch.float32, False), ("router_bias", torch.float32, False),
        ]
        if args.w8a8_native:
            _inp.update({
                "w_gate_r_i8": wg_r_i8, "w_up_r_i8": wu_r_i8, "w_down_r_i8": wd_r_i8,
                "w_gate_r_scale": wg_r_sc, "w_up_r_scale": wu_r_sc, "w_down_r_scale": wd_r_sc,
            })
            _ord += [
                ("w_gate_r_i8", torch.int8, False), ("w_up_r_i8", torch.int8, False),
                ("w_down_r_i8", torch.int8, False),
                ("w_gate_r_scale", torch.float32, False), ("w_up_r_scale", torch.float32, False),
                ("w_down_r_scale", torch.float32, False),
            ]
        else:
            _inp.update({"w_gate_r": wg_r, "w_up_r": wu_r, "w_down_r": wd_r})
            _ord += [
                ("w_gate_r", bf16, False), ("w_up_r", bf16, False), ("w_down_r", bf16, False),
            ]
        _ord += [
            ("w_gate_s", bf16, False), ("w_up_s", bf16, False), ("w_down_s", bf16, False),
            ("next_hidden_out", bf16, True),
        ]
        _specs = [TensorSpec(nm, list(_inp[nm].shape), dt,
                             init_value=None if o else _inp[nm], is_output=o)
                  for (nm, dt, o) in _ord]
        _specs.append(ScalarSpec("layer_idx", torch.int32,
                                 value=torch.tensor(args.layer, dtype=torch.int32)))
        from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415
        _ccfg = {"distributed_config": DistributedConfig(
            device_ids=[args.dev_offset + d for d in range(TP)], num_sub_workers=0)}
        _rcfg = dict(platform=args.platform, device_id=args.dev_offset)
        tgt_out = _load_dump_tensor(args.dump, args.layer, "out", rank=0,
                                    inner_key="hidden_states")
        tgt_out = tgt_out.reshape(-1, HIDDEN)[:T].to(torch.float32)
        if tgt_out.shape[0] < T:
            tgt_out = torch.cat([tgt_out, torch.zeros(T - tgt_out.shape[0], HIDDEN)], 0)
        _cmp_out = {"next_hidden_out": ratio_allclose(
            atol=4e-2, rtol=4e-2, max_error_ratio=0.10)}

        def _golden_out(values):
            for r in range(TP):
                values["next_hidden_out"][r] = tgt_out.to(bf16)

        res = run(program=program, specs=_specs, golden_fn=_golden_out,
                  runtime_cfg=_rcfg, compile_cfg=_ccfg,
                  rtol=4e-2, atol=4e-2, compare_fn=_cmp_out)
        print(f"[moe-prec] layer={args.layer} target=out next_hidden_out vs "
              f"out.pt DEVICE: {res}", flush=True)
        return 0 if res.passed else 1

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
        # Ground-truth swiglu limits per layer (moe.py _SWIGLU_LIMITS): L43 routed=7.0
        # shared=0.0; L44 routed=7.0 shared=16.0; all other MoE layers silu (0,0).
        _RL = 7.0 if args.layer in (43, 44) else 0.0
        _SL = 16.0 if args.layer == 44 else 0.0
        def _swiglu(g, u, rl, do_quant=False):
            s = _F.silu(g)
            if rl > 0.0:
                s = s.clamp(max=rl)
                u = u.clamp(min=-rl, max=rl)
            h = (s * u).to(torch.bfloat16).float()
            if do_quant:
                # match DEVICE per-token INT8 interm-quant (ROUTED expert only;
                # shared expert is UNQUANTIZED BF16).
                amax = h.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
                sc = amax / 127.0
                h = (torch.round(h / sc).clamp(-127, 127) * sc)
            return h.to(torch.bfloat16).float()
        xf = x[0].float()  # [T,HIDDEN] replicated
        sh_ref = torch.zeros(T, HIDDEN, dtype=torch.float32)
        if not args.zero_shared:
            for r in range(TP):
                g = xf @ wg_s[r].float(); u = xf @ wu_s[r].float()
                sh_ref += _swiglu(g, u, _SL) @ wd_s[r].float()
        ro_ref = torch.zeros(T, HIDDEN, dtype=torch.float32)
        if not args.zero_routed:
            tk_ids = _load_dump_tensor(args.dump, args.layer, 'moe_router', rank=0, inner_key='topk_ids').reshape(-1, 8)[:T].long()
            tk_w = _load_dump_tensor(args.dump, args.layer, 'moe_router', rank=0, inner_key='topk_weights').reshape(-1, 8)[:T].float()
            def _qin(xr):
                if _RL <= 0.0:
                    return xr
                _a = xr.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
                _s = _a / 127.0
                return torch.round(xr / _s).clamp(-127, 127) * _s
            for t in range(T):
                for kk in range(8):
                    eid = int(tk_ids[t, kk]); w = float(tk_w[t, kk])
                    dst = eid // N_LOC; le = eid % N_LOC
                    xin = _qin(xf[t:t+1])
                    g = xin @ wg_r[dst, le].float(); u = xin @ wu_r[dst, le].float()
                    ro_ref[t:t+1] += w * (_swiglu(g, u, _RL, do_quant=True) @ wd_r[dst, le].float())
        tgt = (sh_ref + ro_ref)
        print(f'[moe-prec] TORCH-GOLDEN(RL={_RL},SL={_SL}): sh_ref|max|={float(sh_ref.abs().max()):.3f} ro_ref|max|={float(ro_ref.abs().max()):.3f}', flush=True)
    _cmp = {"moe_out": ratio_allclose(atol=4e-2, rtol=4e-2, max_error_ratio=0.10)}
    _golden_extra = {}
    if args.dump_stages:
        assert args.bypass_gate, "--dump-stages requires --bypass-gate (known routing)"
        import torch.nn.functional as _F  # noqa: PLC0415
        from models.step3p5.moe import LOCAL_RECV_MAX as _LRM  # noqa: PLC0415
        _tkids = _load_dump_tensor(args.dump, args.layer, "moe_router", rank=0, inner_key="topk_ids").reshape(-1, 8)[:T].long()
        _xr = x[0].float()  # [T,HIDDEN] (broadcast across ranks)
        # rank-0 expected dispatch layout: e-major, src-secondary, token-ascending
        _xref = torch.zeros(_LRM, HIDDEN, dtype=torch.float32)
        _yref = torch.zeros(_LRM, HIDDEN, dtype=torch.float32)
        _row = 0
        for _e in range(N_LOC):
            _eid = 0 * N_LOC + _e  # rank 0 owns experts [0:N_LOC)
            _tok = [tt for tt in range(T) if _eid in _tkids[tt].tolist()]
            for _src in range(TP):
                for _t in _tok:
                    _xref[_row] = _xr[_t]
                    _g = _xr[_t:_t+1] @ wg_r[0, _e].float()
                    _u = _xr[_t:_t+1] @ wu_r[0, _e].float()
                    _yref[_row] = ((_F.silu(_g) * _u) @ wd_r[0, _e].float())[0]
                    _row += 1
        _total0 = _row
        print(f"[dump] rank0 total valid routed rows = {_total0} (LOCAL_RECV_MAX={_LRM})", flush=True)
        _dbgx = torch.zeros(TP, _LRM, HIDDEN, dtype=bf16); _dbgx[0] = _xref.to(bf16)
        _dbgy = torch.zeros(TP, _LRM, HIDDEN, dtype=bf16); _dbgy[0] = _yref.to(bf16)
        _yref1 = torch.zeros(_LRM, HIDDEN, dtype=torch.float32)
        _xref1 = torch.zeros(_LRM, HIDDEN, dtype=torch.float32); _row1 = 0
        for _e in range(N_LOC):
            _eid = 1 * N_LOC + _e  # rank 1 experts
            _tok = [tt for tt in range(T) if _eid in _tkids[tt].tolist()]
            for _src in range(TP):
                for _t in _tok:
                    _xref1[_row1] = _xr[_t]
                    _g = _xr[_t:_t+1] @ wg_r[1, _e].float()
                    _u = _xr[_t:_t+1] @ wu_r[1, _e].float()
                    _yref1[_row1] = ((_F.silu(_g) * _u) @ wd_r[1, _e].float())[0]
                    _row1 += 1
        _total1 = _row1
        _dbgy[1] = _yref1.to(bf16); _dbgx[1] = _xref1.to(bf16)
        print(f"[dump] rank1 total valid routed rows = {_total1}", flush=True)
        inputs["dbg_routed_x"] = _dbgx; inputs["dbg_routed_y"] = _dbgy
        specs.append(TensorSpec("dbg_routed_x", [TP, _LRM, HIDDEN], bf16, init_value=None, is_output=True))
        specs.append(TensorSpec("dbg_routed_y", [TP, _LRM, HIDDEN], bf16, init_value=None, is_output=True))
        _golden_extra = {"dbg_routed_x": _dbgx, "dbg_routed_y": _dbgy}
        def _mk_cmp(total, label, dstcls=None, rank=0, permcheck=False):
            def _f(actual, expected, *, actual_outputs=None, expected_outputs=None, inputs=None, rtol=None, atol=None):
                a = actual[rank, :total].float(); e = expected[rank, :total].float()
                d = (a - e).abs(); tol = 4e-2 + 4e-2 * e.abs(); bad = d > tol
                nb = int(bad.sum()); nt = int(bad.numel()); ratio = 100.0 * nb / max(nt, 1)
                print(f"[dump] {label}: valid_rows={total} bad={nb}/{nt} ({ratio:.2f}%) max|diff|={float(d.max()):.4f}", flush=True)
                _rb = bad.any(dim=1).nonzero().flatten()[:6].tolist()
                for _rr in _rb:
                    print(f"[dump]   row {_rr}: a[:4]={[round(v,4) for v in a[_rr,:4].tolist()]} e[:4]={[round(v,4) for v in e[_rr,:4].tolist()]}", flush=True)
                if not _rb:
                    print(f"[dump]   {label} rank0 valid rows MATCH torch ref", flush=True)
                if permcheck:
                    _er = expected[rank, :total].float()
                    _bad = bad.any(dim=1).nonzero().flatten().tolist()[:6]
                    for _rr in _bad:
                        _dd = (_er - a[_rr].unsqueeze(0)).abs().mean(dim=1)
                        _bm = int(_dd.argmin())
                        print(f"[dump]   {label} bad row {_rr}: best-match ref row={_bm} (meandiff={float(_dd[_bm]):.4f}) self-ref meandiff={float((_er[_rr]-a[_rr]).abs().mean()):.4f}", flush=True)
                if dstcls is not None:
                    _all = bad.any(dim=1).nonzero().flatten().tolist()
                    _bs = sum(1 for _r in _all if _r < len(dstcls) and dstcls[_r] == 0)
                    _bc = sum(1 for _r in _all if _r < len(dstcls) and dstcls[_r] != 0)
                    _ns = sum(1 for _d in dstcls if _d == 0); _nc = len(dstcls) - _ns
                    print(f"[dump]   {label} BAD-ROW SPLIT: self(dst==0)={_bs}/{_ns} cross(dst!=0)={_bc}/{_nc}", flush=True)
                    _eall = expected[0, :len(dstcls)].float()
                    _crossbad = [_r for _r in _all if _r < len(dstcls) and dstcls[_r] != 0][:5]
                    for _r in _crossbad:
                        _dev = a[_r]
                        _dif = (_eall - _dev.unsqueeze(0)).abs().mean(dim=1)
                        _best = int(_dif.argmin()); _selfd = float((_eall[_r] - _dev).abs().mean())
                        print(f"[dump]     cross-bad row {_r}: best-match ref row={_best} (meandiff={float(_dif[_best]):.4f}) vs self-ref meandiff={_selfd:.4f}", flush=True)
                return (ratio <= 10.0, "dump-cmp")
            return _f
        _LRB = T * _TOPK  # N_ROUTES_PER_RANK = 128; routed_y_buf rows
        _ybufref = torch.zeros(_LRB, HIDDEN, dtype=torch.float32)
        for _b in range(T):
            for _k in range(_TOPK):
                _eid = int(_tkids[_b, _k]); _dst = _eid // N_LOC; _le = _eid % N_LOC
                _g = _xr[_b:_b+1] @ wg_r[_dst, _le].float()
                _u = _xr[_b:_b+1] @ wu_r[_dst, _le].float()
                _ybufref[_b*_TOPK+_k] = ((_F.silu(_g) * _u) @ wd_r[_dst, _le].float())[0]
        _dbgyb = torch.zeros(TP, _LRB, HIDDEN, dtype=bf16); _dbgyb[0] = _ybufref.to(bf16)
        inputs["dbg_routed_ybuf"] = _dbgyb
        specs.append(TensorSpec("dbg_routed_ybuf", [TP, _LRB, HIDDEN], bf16, init_value=None, is_output=True))
        _golden_extra["dbg_routed_ybuf"] = _dbgyb
        _dstcls = [int(_tkids[_b, _k]) // N_LOC for _b in range(T) for _k in range(_TOPK)]
        _recvxref1 = torch.zeros(_LRM, HIDDEN, dtype=torch.float32); _rr = 0
        for _src in range(TP):
            for _e in range(N_LOC):
                _eid = 0 * N_LOC + _e  # VALIDATE recv_x_ref layout on RANK0 (should be 0%)
                _tok = [tt for tt in range(T) if _eid in _tkids[tt].tolist()]
                for _t in _tok:
                    _recvxref1[_rr] = _xr[_t]; _rr += 1
        _dbgrx = torch.zeros(TP, _LRM, HIDDEN, dtype=bf16); _dbgrx[0] = _recvxref1.to(bf16)
        inputs["dbg_recv_x"] = _dbgrx
        specs.append(TensorSpec("dbg_recv_x", [TP, _LRM, HIDDEN], bf16, init_value=None, is_output=True))
        _golden_extra["dbg_recv_x"] = _dbgrx
        print(f"[dump] recv_x[RANK1] ref rows = {_rr}", flush=True)
        _cmp["dbg_routed_ybuf"] = _mk_cmp(_LRB, "routed_y_buf(post-push)", dstcls=_dstcls)
        _cmp["dbg_recv_x"] = _mk_cmp(_rr, "recv_x[RANK0-VALIDATE](post-a2a)", rank=0, permcheck=True)
        _cmp["dbg_routed_x"] = _mk_cmp(_total1, "local_routed_x[RANK1](post-dispatch)", rank=1, permcheck=True)
        _cmp["dbg_routed_y"] = _mk_cmp(_total1, "local_routed_y[RANK1](post-expert)", rank=1)
        # INT32 offset dump: recv_counts/recv_offsets/read_offsets/send_offsets_rank
        _C = [0]*TP
        for _b in range(T):
            for _k in range(_TOPK):
                _C[int(_tkids[_b,_k])//N_LOC] += 1
        _pref = [0]*TP
        for _d in range(1,TP):
            _pref[_d] = _pref[_d-1] + _C[_d-1]
        print(f"[offdump] per-rank token counts C={_C} prefix={_pref}", flush=True)
        for _M in (0,1,2):
            _rc=[_C[_M]]*TP; _ro=[r*_C[_M] for r in range(TP)]; _rdo=[_pref[_M]]*TP; _so=list(_pref)
            print(f"[offdump] EXPECT rank{_M}: recv_counts={_rc} recv_offsets={_ro} read_offsets={_rdo} send_offsets={_so}", flush=True)
        inputs["dbg_offsets"] = torch.zeros(TP, TP, 4, dtype=torch.int32)
        specs.append(TensorSpec("dbg_offsets", [TP, TP, 4], torch.int32, init_value=None, is_output=True))
        def _cmp_off(actual, expected, *, actual_outputs=None, expected_outputs=None, inputs=None, rtol=None, atol=None):
            aa = actual.to(torch.int64)
            for _M in range(TP):
                print(f"[offdump] DEVICE rank{_M}: recv_counts={aa[_M,:,0].tolist()} recv_offsets={aa[_M,:,1].tolist()} read_offsets={aa[_M,:,2].tolist()} send_offsets={aa[_M,:,3].tolist()}", flush=True)
            return (True, "offdump")
        _cmp["dbg_offsets"] = _cmp_off
        # rebuild specs order: outputs must appear; recompute specs list from inputs+order
    def golden_fn(values):
        for r in range(TP):
            values["moe_out"][r] = tgt.to(bf16)
        for _k, _v in _golden_extra.items():
            for r in range(TP):
                values[_k][r] = _v[r]

    res = run(program=program, specs=specs, golden_fn=golden_fn,
              runtime_cfg=runtime_cfg, compile_cfg=compile_cfg,
              rtol=4e-2, atol=4e-2, compare_fn=_cmp)
    print(f"[moe-prec] layer={args.layer} target={args.target} DEVICE: {res}", flush=True)
    return 0 if res.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
