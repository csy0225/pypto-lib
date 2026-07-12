# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""N=1 whole_decode_faithful_real with REAL W8A8 weights via import_ipc (Task2).

Self-load stacks all 8 rank bundles + [tp,...] copies on ONE host -> ~752GB OOM
(exit137). The N=1 program is a single dispatch feeding all 45 layers' weights at
once, so per-layer host streaming (G-series) cannot apply. The ONLY non-OOM path
is zero-copy import_ipc: each forked chip imports ITS rank's 47GB pool.

Borrows ONLY the import_ipc feature from the G-series (pure-Python
DistributedWorker.import_ipc_all, no C++ rebuild) + pypto_weight_ipc's
import_weights_all/build_stacked_weight; stays on the N=1 program (NOT Option-C).

Two modes:
  --export-rank R --dev D --out DIR --ckpt C : load rank R bundle, export ONE pool
        + key, then HOLD (wait for DIR/STOP) so the pool stays mapped.
  (default worker): launch 8 --export-rank children, wait for keys, compile +
        prepare() + import_weights_all + build_stacked_weight + rt.run (weights =
        StackedDeviceTensor; KV/hidden/gate_r = dummy host) -> finite logits.

    python -m tests.step3p5._stage_whole_faithful_real_ipc -d 0,1,2,3,4,5,6,7 \
        --ckpt /mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp

env: source cann/set_env.sh && source WS/activate.sh && export PTO_ISA_ROOT=WS/pto-isa
     PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

CKPT_DEFAULT = "/mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    p.add_argument("-d", "--device", default="0,1,2,3,4,5,6,7")
    p.add_argument("--layer-name", default="whole_decode_faithful_real")
    p.add_argument("--ckpt", default=CKPT_DEFAULT)
    p.add_argument("--out", default="/tmp/n1_weight_ipc")
    p.add_argument("--export-rank", type=int, default=-1)
    p.add_argument("--dev", type=int, default=0)
    # Reuse externally-launched, still-holding exporters: skip cleanup/launch of
    # exporter children and do NOT write STOP at exit, so the IPC pools stay
    # mapped for a subsequent worker run (fast bisect: compile+run only).
    p.add_argument("--reuse-exporters", action="store_true")
    # KV cache also via IPC (user hard constraint): exporter carves dummy
    # k/v_cache into the pool; worker binds them as add_inout DeviceTensors.
    p.add_argument("--kv-ipc", action="store_true")
    # ctx=1 token-exact A/B: feed current_hidden = embed(token) so the whole-net
    # decodes position-0 (self-attention over 1 token, rope identity) and its
    # argmax = the next token to compare vs vLLM's completion for prompt=[token].
    p.add_argument("--hidden-token", type=int, default=-1)
    return p.parse_args()


def _do_export(args) -> int:
    """Single-rank exporter: load rank bundle -> one pool + key -> HOLD."""
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from tools.step3p5.pypto_weight_ipc import export_from_checkpoint  # noqa: PLC0415
    r = args.export_rank
    os.makedirs(args.out, exist_ok=True)
    export_from_checkpoint(args.ckpt, rank=r, tp_world_size=8, out_dir=args.out, dev=args.dev, int8_routed=True, kv_ipc=args.kv_ipc)
    # Signal readiness; hold the pool mapped until the worker writes STOP.
    Path(os.path.join(args.out, f"ready.rank{r}")).write_text("1")
    print(f"[export-rank {r}] holding pool on dev {args.dev}; waiting for STOP", flush=True)
    stop = os.path.join(args.out, "STOP")
    while not os.path.exists(stop):
        time.sleep(2)
    print(f"[export-rank {r}] STOP seen; exit", flush=True)
    return 0


def _stop(out_dir, procs):
    try:
        Path(os.path.join(out_dir, "STOP")).write_text("1")
    except OSError:
        pass
    for pp in procs:
        try:
            pp.wait(timeout=30)
        except Exception:  # noqa: BLE001
            pp.terminate()


def _do_worker(args) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    # Surface the runtime's LOG_INFO_V0 VA-layout diagnostics (comm domain
    # window base/size vs IPC pool). simpler maps python log level <=15 -> info_v=0.
    import logging  # noqa: PLC0415
    logging.getLogger("simpler").setLevel(15)
    device_ids = [int(d) for d in str(args.device).split(",")]
    tp = len(device_ids)
    dev_offset = device_ids[0]
    os.makedirs(args.out, exist_ok=True)
    if not args.reuse_exporters:
        for f in os.listdir(args.out):
            if f.startswith(("ready.rank", "pypto_weight.")) or f == "STOP":
                try:
                    os.remove(os.path.join(args.out, f))
                except OSError:
                    pass

    from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415
    set_backend_type(BackendType.Ascend910B)
    import models.step3p5.config as cfg  # noqa: PLC0415
    import models.step3p5.decode_layer as dl  # noqa: PLC0415
    from models.step3p5 import weight_loader as K  # noqa: PLC0415
    from tools.step3p5.pypto_weight_ipc import (  # noqa: PLC0415
        import_weights_all, build_stacked_weight,
    )
    assert tp == cfg.TP_WORLD_SIZE, f"need {cfg.TP_WORLD_SIZE} cards; got {tp}"

    print(f"[worker] launching {tp} exporters dev_offset={dev_offset} ...", flush=True)
    procs = []
    if args.reuse_exporters:
        # Attach to externally-launched exporters that are already holding.
        print("[worker] reuse-exporters: expecting existing ready.rank* keys", flush=True)
        for r in range(tp):
            rp = os.path.join(args.out, f"ready.rank{r}")
            if not os.path.exists(rp):
                raise RuntimeError(f"reuse-exporters: missing {rp}; launch exporters first")
    else:
        for r in range(tp):
            procs.append(subprocess.Popen(
                [sys.executable, "-m", "tests.step3p5._stage_whole_faithful_real_ipc",
                 "--export-rank", str(r), "--dev", str(dev_offset + r),
                 "--out", args.out, "--ckpt", args.ckpt]
                + (["--kv-ipc"] if args.kv_ipc else []),
                cwd=str(repo_root),
            ))
        deadline = time.time() + 2400  # 40 min for cold jfs loads
        while time.time() < deadline:
            if all(os.path.exists(os.path.join(args.out, f"ready.rank{r}")) for r in range(tp)):
                break
            if any(pp.poll() not in (None, 0) for pp in procs):
                _stop(args.out, procs)
                raise RuntimeError("an exporter child died before readiness")
            time.sleep(3)
        else:
            _stop(args.out, procs)
            raise RuntimeError("exporters not ready within deadline")
    print("[worker] all exporters ready; compiling program ...", flush=True)

    try:
        from pypto import ir  # noqa: PLC0415
        from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415
        from pypto.runtime.device_tensor import DeviceTensor, StackedDeviceTensor  # noqa: PLC0415
        os.environ["PYPTO_PROG_BUILD_DIR"] = "/data/chensiyu/hw_project/pypto/workspace/build_output"
        program = getattr(dl, args.layer_name)
        # Diagnostic knob: PYPTO_MEM_PLANNER=ptoas skips PyPTO MemoryReuse +
        # AllocateMemoryAddr (ptoas owns reuse) — used to test whether the
        # 42-layer gate_topk stall is a cross-chip buffer-aliasing artifact.
        _mplan = None
        if os.environ.get("PYPTO_MEM_PLANNER", "").lower() == "ptoas":
            from pypto.pypto_core import passes as _passes  # noqa: PLC0415
            _mplan = _passes.MemoryPlanner.PTOAS
            print("[worker] memory_planner=PTOAS (skip PyPTO MemoryReuse)", flush=True)
        compiled = ir.compile(
            program, platform=args.platform,
            distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
            skip_ptoas=False, dump_passes=False, memory_planner=_mplan,
        )
        print(f"[worker] compile OK => {compiled.output_dir}", flush=True)

        def C(name):
            v = getattr(dl, name, None)
            if v is None:
                v = getattr(cfg, name)
            return int(v)
        HIDDEN, HEAD_DIM, BATCH = cfg.HIDDEN, cfg.HEAD_DIM, cfg.BATCH
        UBD, BTF, RSD, KVC = C("USER_BATCH_DYN"), C("BLOCK_TABLE_FLAT_DYN"), C("ROPE_SEQ_DYN"), C("KV_CACHE_ROWS_DYN")
        ROT_FULL, ROT_SWA = cfg.ROTARY_HALF_FULL * 2, cfg.ROTARY_HALF_SWA * 2
        NHF_PAD, NHS_PAD = C("NUM_HEADS_FULL_LOCAL_PAD"), C("NUM_HEADS_SWA_LOCAL_PAD")
        HQ_FULL, HQ_SWA = C("HIDDEN_Q_FULL_LOCAL"), C("HIDDEN_Q_SWA_LOCAL")
        N_FULL = sum(1 for li in range(cfg.NUM_HIDDEN_LAYERS) if cfg.is_full_attention(li))
        N_SWA = cfg.NUM_HIDDEN_LAYERS - N_FULL
        bf16, f32, i32 = torch.bfloat16, torch.float32, torch.int32

        # VOCAB from the exported map (available now; no rt needed).
        import json as _json  # noqa: PLC0415
        with open(os.path.join(args.out, "pypto_weight_map.rank0.json")) as _mf:
            VOCAB_LOCAL = int(_json.load(_mf)["map"][K.KEY_LM_HEAD]["shape"][0])

        # DistributedWorker contract: host tensors must be .share_memory_() AND
        # allocated BEFORE prepare() (so forked chips can see them). Weights come
        # in as DeviceTensor (import_ipc, after prepare) so they are exempt.
        def zsh(*shape, dtype=bf16):
            return torch.zeros(shape, dtype=dtype).share_memory_()
        current_hidden = zsh(tp, BATCH, HIDDEN)
        if args.hidden_token >= 0:
            # embed(token) into row 0 of every rank (replicated); ctx=1 (seq_lens=ones)
            # -> whole-net decodes position-0, argmax(logits) = next token vs vLLM.
            import safetensors.torch as _st  # noqa: PLC0415
            import json as _json2  # noqa: PLC0415
            _idx = _json2.load(open(os.path.join(args.ckpt, "quant_model_weights.safetensors.index.json")))
            _shard = _idx["weight_map"]["model.embed_tokens.weight"]
            with _st.safe_open(os.path.join(args.ckpt, _shard), framework="pt") as _f:
                _emb_row = _f.get_slice("model.embed_tokens.weight")[args.hidden_token, :].to(torch.bfloat16)
            current_hidden[:, 0, :] = _emb_row
            print(f"[worker] ctx=1 A/B: current_hidden row0 = embed(token={args.hidden_token}) "
                  f"|emb|max={_emb_row.float().abs().max():.4f}", flush=True)
        gate_r_full = zsh(tp, N_FULL, NHF_PAD, HQ_FULL)
        gate_r_swa = zsh(tp, N_SWA, NHS_PAD, HQ_SWA)
        # Fill gate_r with the block-diag R constant (layer-independent). The
        # on-device head-gate (attention_full/swa Scope 1.f) computes
        # gate_exp = sigmoid(normed_all @ w_g) @ R, so R[h, h*HEAD_DIM+d] = 1
        # for real local heads (count = HQ//HEAD_DIM: full=8, swa=12); padded
        # rows [real_heads : NHF_PAD] stay zero.
        for _h in range(HQ_FULL // HEAD_DIM):
            gate_r_full[:, :, _h, _h * HEAD_DIM:(_h + 1) * HEAD_DIM] = 1.0
        for _h in range(HQ_SWA // HEAD_DIM):
            gate_r_swa[:, :, _h, _h * HEAD_DIM:(_h + 1) * HEAD_DIM] = 1.0
        seq_lens = torch.ones(tp, UBD, dtype=i32).share_memory_()
        block_table = torch.zeros(tp, BTF, dtype=i32).share_memory_()
        slot_mapping = torch.arange(UBD, dtype=i32).unsqueeze(0).repeat(tp, 1).contiguous().share_memory_()
        rope_cf, rope_sf = zsh(tp, RSD, ROT_FULL, dtype=f32), zsh(tp, RSD, ROT_FULL, dtype=f32)
        rope_cs, rope_ss = zsh(tp, RSD, ROT_SWA, dtype=f32), zsh(tp, RSD, ROT_SWA, dtype=f32)
        if args.hidden_token >= 0:
            # position-0 rope = identity (angle 0 -> cos=1, sin=0).
            rope_cf.fill_(1.0); rope_cs.fill_(1.0)  # sin stays 0
        k_cache, v_cache = zsh(tp, KVC, HEAD_DIM), zsh(tp, KVC, HEAD_DIM)
        h_mid_out, next_hidden_out = zsh(tp, BATCH, HIDDEN), zsh(tp, BATCH, HIDDEN)
        logits_shard_out = torch.zeros(tp, UBD, VOCAB_LOCAL, dtype=f32).share_memory_()

        with compiled.prepare() as rt:
            wmaps = import_weights_all(rt, args.out, tp=tp, dev_offset=dev_offset)

            def W(key):
                return build_stacked_weight(wmaps, key)

            def W_reshape(key, per_rank_shape, dtype):
                shards = [DeviceTensor(wmaps[r].peer_base + wmaps[r].offset(key),
                                       tuple(per_rank_shape), dtype) for r in range(tp)]
                return StackedDeviceTensor(shards, (tp, *per_rank_shape), list(range(tp)))

            if args.kv_ipc:
                # KV via IPC (user hard constraint): rebind k/v_cache from the
                # pool as add_inout DeviceTensors (attention reads context + writes
                # new K/V into this shared peer memory), replacing the dummy host
                # tensors allocated pre-prepare above.
                k_cache, v_cache = W("k_cache"), W("v_cache")
                print("[worker] KV cache via IPC (k_cache/v_cache bound from pool)", flush=True)

            args_list = [current_hidden]  # dummy host, shared
            args_list += [W(K.KEY_INPUT_RMS), W(K.KEY_POST_ATTN_RMS), W(K.KEY_Q_NORM), W(K.KEY_K_NORM)]
            args_list += [W(K.KEY_WQ_FULL), W(K.KEY_WK_FULL), W(K.KEY_WV_FULL), W(K.KEY_WO_FULL), W(K.KEY_WG_FULL),
                          gate_r_full]
            args_list += [W(K.KEY_WQ_SWA), W(K.KEY_WK_SWA), W(K.KEY_WV_SWA), W(K.KEY_WO_SWA), W(K.KEY_WG_SWA),
                          gate_r_swa]
            args_list += [W(K.KEY_DENSE_GATE), W(K.KEY_DENSE_UP), W(K.KEY_DENSE_DOWN)]
            args_list += [W(K.KEY_MOE_GATE_W), W(K.KEY_MOE_ROUTER_BIAS),
                          W(K.KEY_MOE_W_GATE_R), W(K.KEY_MOE_W_GATE_R_SCALE),
                          W(K.KEY_MOE_W_UP_R), W(K.KEY_MOE_W_UP_R_SCALE),
                          W(K.KEY_MOE_W_DOWN_R), W(K.KEY_MOE_W_DOWN_R_SCALE),
                          W(K.KEY_MOE_W_GATE_S),
                          W(K.KEY_MOE_W_UP_S), W(K.KEY_MOE_W_DOWN_S)]
            args_list += [seq_lens, block_table, slot_mapping, rope_cf, rope_sf, rope_cs, rope_ss,
                          k_cache, v_cache]
            args_list += [h_mid_out, next_hidden_out]
            args_list += [W_reshape(K.KEY_FINAL_NORM, (1, HIDDEN), f32), W(K.KEY_LM_HEAD)]
            args_list += [logits_shard_out]

            print(f"[worker] built {len(args_list)} args (weights via IPC); VOCAB_LOCAL={VOCAB_LOCAL}; running ...", flush=True)
            t0 = time.time()
            rt.run(compiled, *args_list)
            dt = time.time() - t0
            full_logits = torch.cat([logits_shard_out[r, 0] for r in range(tp)], dim=0)
            print(f"[worker] RUN done {dt:.2f}s max|next_hidden|={next_hidden_out.float().abs().max():.4f} "
                  f"max|logits|={logits_shard_out.abs().max():.4f} argmax={int(full_logits.argmax())}", flush=True)
            print("[worker] RESULT=REAL_WEIGHT_IPC_RUN_CLEAN", flush=True)
    finally:
        if args.reuse_exporters:
            print("[worker] reuse-exporters: leaving pools mapped (no STOP)", flush=True)
        else:
            _stop(args.out, procs)
    return 0


def main() -> int:
    args = _parse_args()
    if args.export_rank >= 0:
        return _do_export(args)
    return _do_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
