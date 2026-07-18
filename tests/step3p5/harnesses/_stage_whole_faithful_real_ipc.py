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

    python -m tests.step3p5.harnesses._stage_whole_faithful_real_ipc -d 0,1,2,3,4,5,6,7 \
        --ckpt /mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp

env: source cann/set_env.sh && source WS/activate.sh && export PTO_ISA_ROOT=WS/pto-isa
     PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072
"""
from __future__ import annotations

import argparse
import importlib
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
    p.add_argument(
        "--layer-module",
        default="models.step3p5.decode_layer_single_chip",
        help=(
            "Module containing --layer-name. Default is the canonical N1 "
            "single HOST→CHIP submission implementation."
        ),
    )
    p.add_argument("--layer-name", default="whole_decode_faithful_real_single_chip")
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
    p.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "Run the same prepared device program this many consecutive times. "
            "Each run clears host-visible output/padding buffers and must produce "
            "the canonical argmax 303."
        ),
    )
    p.add_argument(
        "--dmesg-dir",
        default="",
        help=(
            "Optional directory for per-run `sudo -n dmesg -T` before/after "
            "snapshots. Capture occurs outside the measured rt.run interval."
        ),
    )
    p.add_argument(
        "--expected-argmax",
        type=int,
        default=303,
        help=(
            "Canonical token to enforce. Use -1 only for an explicitly named "
            "layer-boundary diagnostic with P_FAITHFUL_MOE_LAYERS < 42; "
            "the complete P42 canonical test must keep the default 303."
        ),
    )
    args = p.parse_args()
    if args.repeat < 1:
        p.error("--repeat must be >= 1")
    return args


def _do_export(args) -> int:
    """Single-rank exporter: load rank bundle -> one pool + key -> HOLD."""
    repo_root = Path(__file__).resolve().parents[3]
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
    repo_root = Path(__file__).resolve().parents[3]
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
    import pypto as _pypto_pkg  # noqa: PLC0415
    import pypto.pypto_core as _pypto_core  # noqa: PLC0415
    import simpler as _simpler_pkg  # noqa: PLC0415
    import models.step3p5.config as cfg  # noqa: PLC0415
    dl = importlib.import_module(args.layer_module)
    from models.step3p5 import weight_loader as K  # noqa: PLC0415
    from tools.step3p5.pypto_weight_ipc import (  # noqa: PLC0415
        import_weights_all, build_stacked_weight,
    )
    assert tp == cfg.TP_WORLD_SIZE, f"need {cfg.TP_WORLD_SIZE} cards; got {tp}"
    print(
        "[worker] import provenance "
        f"python={Path(sys.executable).resolve()} "
        f"pypto={Path(_pypto_pkg.__file__).resolve()} "
        f"pypto_core={Path(_pypto_core.__file__).resolve()} "
        f"simpler={Path(_simpler_pkg.__file__).resolve()}",
        flush=True,
    )

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
                [sys.executable, "-m", "tests.step3p5.harnesses._stage_whole_faithful_real_ipc",
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
        print(
            f"[worker] program module={dl.__name__} file={Path(dl.__file__).resolve()} "
            f"name={program.name}",
            flush=True,
        )
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
            if int(os.environ.get("P_FILL_BATCH", "0")) > 0:
                # E1 diagnostic: fill ALL BATCH rows with embed(token) so there
                # are no zero-embedding rows. Isolates whether the MoE ~1e11 is
                # driven by degenerate zero rows (INT8 amax quant divide-by-tiny
                # / combine reading unwritten routed cells) vs an input-agnostic
                # bug (buffer aliasing / collective).
                current_hidden[:, :, :] = _emb_row
                print("[worker] E1: P_FILL_BATCH — all BATCH rows = embed(token)", flush=True)
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
        dbg_out = zsh(tp, BATCH, HIDDEN)  # E2 op-level stage dump (P_DBG_STAGE)
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
            args_list += [dbg_out]
            args_list += [W_reshape(K.KEY_FINAL_NORM, (1, HIDDEN), f32), W(K.KEY_LM_HEAD)]
            args_list += [logits_shard_out]

            print(
                f"[worker] built {len(args_list)} args (weights via IPC); "
                f"VOCAB_LOCAL={VOCAB_LOCAL}; repeat={args.repeat}; running ...",
                flush=True,
            )
            run_times = []
            run_fingerprints = []
            dmesg_dir = Path(args.dmesg_dir) if args.dmesg_dir else None
            if dmesg_dir is not None:
                dmesg_dir.mkdir(parents=True, exist_ok=True)

            def capture_dmesg(path: Path) -> None:
                completed = subprocess.run(
                    ["sudo", "-n", "dmesg", "-T"],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"dmesg capture failed rc={completed.returncode}: "
                        f"{completed.stdout.strip()}"
                    )
                path.write_text(
                    f"# captured_at_ns={time.time_ns()}\n{completed.stdout}"
                )

            for run_idx in range(1, args.repeat + 1):
                # Never let a previous invocation's host-visible output or
                # padding rows make a later run look successful. Communication
                # windows remain runtime-owned and are allocated/initialized by
                # the generated host orchestration on every rt.run.
                h_mid_out.zero_()
                next_hidden_out.zero_()
                dbg_out.zero_()
                logits_shard_out.zero_()
                if dmesg_dir is not None:
                    capture_dmesg(dmesg_dir / f"dmesg.run{run_idx:02d}.before.txt")

                t0 = time.time()
                try:
                    rt.run(compiled, *args_list)
                finally:
                    t1 = time.time()
                    if dmesg_dir is not None:
                        capture_dmesg(dmesg_dir / f"dmesg.run{run_idx:02d}.after.txt")
                dt = t1 - t0
                run_times.append(dt)
                full_logits = torch.cat([logits_shard_out[r, 0] for r in range(tp)], dim=0)
                # Row-0 = the single valid ctx=1 token; rows 1..15 are batch
                # padding. Report both whole-buffer and valid-row magnitudes,
                # but use only row0 logits for the canonical golden.
                nh_max = float(next_hidden_out.float().abs().max())
                nh_row0 = float(next_hidden_out[:, 0, :].float().abs().max())
                hmid_max = float(h_mid_out.float().abs().max())
                dbg_max = float(dbg_out.float().abs().max())
                dbg_row0 = float(dbg_out[:, 0, :].float().abs().max())
                logits_max = float(logits_shard_out.abs().max())
                argmax = int(full_logits.argmax())
                run_fingerprints.append(
                    (nh_max, nh_row0, hmid_max, dbg_max, dbg_row0, logits_max, argmax)
                )
                print(
                    f"[worker] RUN done {dt:.2f}s run={run_idx}/{args.repeat} "
                    f"max|next_hidden|={nh_max:.4f} row0|next_hidden|={nh_row0:.4f} "
                    f"max|h_mid|={hmid_max:.4f} max|dbg|={dbg_max:.4f} "
                    f"row0|dbg|={dbg_row0:.4f} max|logits|={logits_max:.4f} "
                    f"argmax={argmax}",
                    flush=True,
                )
                _t5 = torch.topk(full_logits.float(), 5)
                print(
                    f"[worker] TOP5 run={run_idx}/{args.repeat} ids={_t5.indices.tolist()} "
                    f"vals={[round(float(v), 3) for v in _t5.values.tolist()]} "
                    f"(expected argmax={args.expected_argmax})",
                    flush=True,
                )
                if args.expected_argmax >= 0 and argmax != args.expected_argmax:
                    raise RuntimeError(
                        f"canonical accuracy failure at run {run_idx}/{args.repeat}: "
                        f"argmax={argmax}, expected {args.expected_argmax}"
                    )

            print(
                f"[worker] REPEAT summary pass={len(run_times)}/{args.repeat} "
                f"min={min(run_times):.4f}s mean={sum(run_times) / len(run_times):.4f}s "
                f"max={max(run_times):.4f}s fingerprints_unique={len(set(run_fingerprints))}",
                flush=True,
            )
            _pdir = os.environ.get("N1_DUMP_DIR", "")
            if _pdir:
                os.makedirs(_pdir, exist_ok=True)
                _P = os.environ.get("P_FAITHFUL_MOE_LAYERS", "42")
                _S = os.environ.get("P_DBG_STAGE", "0")
                _logits_shards_row0 = logits_shard_out[:, 0, :].float().cpu()
                _full_logits_row0 = torch.cat(
                    [_logits_shards_row0[r] for r in range(tp)],
                    dim=0,
                )
                # Preserve the complete physical output for layer-boundary
                # diagnostics.  The canonical valid token is row 0, but rows
                # 1..15 are still materialized tensors and must not be
                # silently discarded: they are part of the padding,
                # initialization, dynamic-quant, route, and lifetime audit.
                _next_hidden_full = next_hidden_out.float().cpu().contiguous()
                _h_mid_full = h_mid_out.float().cpu().contiguous()
                _dbg_full = dbg_out.float().cpu().contiguous()
                _logits_shards_full = logits_shard_out.float().cpu().contiguous()
                _full_logits_all_rows = torch.cat(
                    [_logits_shards_full[r] for r in range(tp)],
                    dim=-1,
                )
                torch.save(
                    _next_hidden_full[:, 0, :],
                    os.path.join(_pdir, f"P{_P}_nh_row0.pt"),
                )
                torch.save(
                    _h_mid_full[:, 0, :],
                    os.path.join(_pdir, f"P{_P}_hmid_row0.pt"),
                )
                torch.save(
                    _dbg_full[:, 0, :],
                    os.path.join(_pdir, f"P{_P}_S{_S}_dbg_row0.pt"),
                )
                torch.save(
                    _logits_shards_row0,
                    os.path.join(_pdir, f"P{_P}_logits_shards_row0.pt"),
                )
                torch.save(
                    _full_logits_row0,
                    os.path.join(_pdir, f"P{_P}_full_logits_row0.pt"),
                )
                torch.save(
                    _next_hidden_full,
                    os.path.join(_pdir, f"P{_P}_nh_full.pt"),
                )
                torch.save(
                    _h_mid_full,
                    os.path.join(_pdir, f"P{_P}_hmid_full.pt"),
                )
                torch.save(
                    _dbg_full,
                    os.path.join(_pdir, f"P{_P}_S{_S}_dbg_full.pt"),
                )
                torch.save(
                    _logits_shards_full,
                    os.path.join(_pdir, f"P{_P}_logits_shards_full.pt"),
                )
                torch.save(
                    _full_logits_all_rows,
                    os.path.join(_pdir, f"P{_P}_full_logits_all_rows.pt"),
                )
                for _name, _tensor in (
                    ("next_hidden", _next_hidden_full),
                    ("h_mid", _h_mid_full),
                    ("dbg", _dbg_full),
                ):
                    _finite = bool(torch.isfinite(_tensor).all())
                    _row_max = _tensor.abs().amax(dim=-1)
                    print(
                        f"[worker] FULL {_name} shape={tuple(_tensor.shape)} "
                        f"finite={_finite} "
                        f"row_max={[_row_max[r].tolist() for r in range(tp)]}",
                        flush=True,
                    )
                print(
                    f"[worker] FULL logits shape={tuple(_full_logits_all_rows.shape)} "
                    f"finite={bool(torch.isfinite(_full_logits_all_rows).all())}",
                    flush=True,
                )
                print(
                    f"[worker] DUMPED vectors+logits P={_P} S={_S} -> {_pdir}",
                    flush=True,
                )
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
