"""Multi-rank (TP=8) numerical golden for the full_dense decode layer.

Validates that the barrier-mesh ``tp_all_reduce`` sums per-rank partials
*correctly* across 8 cards - not merely that the 8-card run is fault-free.

Strategy (relies on TP linearity):
  * The full attention out-proj  o = attn_out[B, NH*HD] @ wo[NH*HD, H]  equals
    the sum over ranks of each rank's partial  (attn_out_r @ wo_r), because
    partitioning a matmul's contraction dim is just a sum of partial matmuls.
    The kernel computes those partials per rank and ``tp_all_reduce`` sums them.
    Same for the dense MLP down-proj.
  * So a single full-head torch reference (NUM_KV_HEADS KV heads, NUM_HEADS_FULL
    Q heads, full INTERMEDIATE) == the TP=8 device output on every rank.
  * Head/intermediate -> rank assignment order is irrelevant to the sum, so a
    contiguous slice (rank r = block r) is sufficient and unambiguous.

Run (8 cards):
    python -m tests.step3p5.test_decode_layer_full_dense_multirank_st \
        -p a2a3 -d 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-p", "--platform", default="a2a3",
                   choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    p.add_argument("-d", "--device", type=str, default="0,1,2,3,4,5,6,7")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    device_ids = [int(d) for d in str(args.device).split(",")]
    N_RANKS = len(device_ids)

    from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415
    set_backend_type({
        "a2a3": BackendType.Ascend910B, "a2a3sim": BackendType.Ascend910B,
        "a5": BackendType.Ascend950, "a5sim": BackendType.Ascend950,
    }[args.platform])

    # Canonical TP=8 config - NO apply_tp1_patch.
    import models.step3p5.config as cfg  # noqa: PLC0415
    from models.step3p5.decode_layer import select_decode_layer  # noqa: PLC0415
    from tests.step3p5.test_decode_layer_full_dense_st import (  # noqa: PLC0415
        _torch_attn_no_gate,
        _torch_dense_mlp,
    )

    assert N_RANKS == cfg.TP_WORLD_SIZE, (
        f"this golden needs {cfg.TP_WORLD_SIZE} cards; got {N_RANKS}"
    )

    H = cfg.HIDDEN
    HD = cfg.HEAD_DIM
    NH = cfg.NUM_HEADS_FULL          # 64
    NKV = cfg.NUM_KV_HEADS           # 8
    QPK = cfg.Q_PER_KV_FULL          # 8
    INTER = cfg.INTERMEDIATE         # 11264
    BATCH = cfg.BATCH                # 16
    EPS = cfg.EPS
    BLOCK = cfg.BLOCK_SIZE
    MAXB = cfg.MAX_BLOCKS_PER_SEQ
    SEQ = cfg.MAX_SEQ_DEFAULT
    NLAYERS = cfg.NUM_HIDDEN_LAYERS
    ROT_HALF = cfg.ROTARY_HALF_FULL  # 32
    ROT_DIM = ROT_HALF * 2           # 64
    ROT_PASS = HD - ROT_DIM          # 64
    PAD = cfg.NUM_HEADS_FULL_LOCAL_PAD  # 16

    HQ_LOC = cfg.NUM_HEADS_FULL_LOCAL * HD   # 8*128 = 1024
    KV_LOC = cfg.KV_HEADS_LOCAL * HD         # 1*128 = 128
    INT_LOC = cfg.INTERMEDIATE_LOCAL         # 1408
    n_full = sum(1 for t in cfg.LAYER_TYPES if t == cfg.LAYER_TYPE_FULL)
    n_dense = len(cfg.DENSE_LAYER_INDICES)

    bf16 = torch.bfloat16
    g = torch.Generator().manual_seed(args.seed)

    def rnd(shape, std):
        return torch.empty(shape, dtype=torch.float32).normal_(0.0, std, generator=g)

    # -- Full layer-0 weights (the reference's truth). --
    wq_f = rnd([H, NH * HD], 0.02).bfloat16()
    wk_f = rnd([H, NKV * HD], 0.02).bfloat16()
    wv_f = rnd([H, NKV * HD], 0.02).bfloat16()
    wo_f = rnd([NH * HD, H], 0.02).bfloat16()
    wg_f = rnd([H, INTER], 0.02).bfloat16()
    wu_f = rnd([H, INTER], 0.02).bfloat16()
    wd_f = rnd([INTER, H], 0.02).bfloat16()
    irms = rnd([H], 0.05)
    prms = rnd([H], 0.05)
    qn = rnd([HD], 0.05)
    kn = rnd([HD], 0.05)
    cur = rnd([BATCH, H], 0.02).bfloat16()
    rcos = rnd([SEQ, ROT_DIM], 0.5)
    rsin = rnd([SEQ, ROT_DIM], 0.5)
    seq_lens = torch.ones(BATCH, dtype=torch.int32)
    block_table = torch.zeros(MAXB * BATCH, dtype=torch.int32)
    slot_mapping = torch.arange(BATCH, dtype=torch.int32)

    # -- Full-head torch reference (== TP=8 device output by linearity). --
    resid1 = _torch_attn_no_gate(
        hidden_states=cur, input_rms_weight=irms,
        wq_full=wq_f, wk_full=wk_f, wv_full=wv_f,
        q_norm_weight=qn, k_norm_weight=kn, wo_full=wo_f,
        seq_lens=seq_lens, block_table=block_table, slot_mapping=slot_mapping,
        rope_cos=rcos, rope_sin=rsin,
        k_cache_full=torch.zeros(SEQ, HD, dtype=bf16),
        v_cache_full=torch.zeros(SEQ, HD, dtype=bf16),
        num_heads_full=NH, num_kv_heads_full=NKV,
        head_dim=HD, rotary_dim=ROT_DIM, rotary_half=ROT_HALF,
        rotary_pass=ROT_PASS, q_per_kv=QPK, eps=EPS,
        block_size=BLOCK, max_blocks_per_seq=MAXB,
    )
    ref = _torch_dense_mlp(resid1, prms, wg_f, wu_f, wd_f, EPS).float()  # [B, H]

    # -- Per-rank contiguous slices into device inputs ([N_RANKS, ...]). --
    # Weights stacked over layers; only layer 0 (first block) is read.
    def zeros(*shape):
        return torch.zeros(*shape, dtype=bf16)

    wq_d = zeros(N_RANKS, n_full * H, HQ_LOC)
    wk_d = zeros(N_RANKS, n_full * H, KV_LOC)
    wv_d = zeros(N_RANKS, n_full * H, KV_LOC)
    wo_d = zeros(N_RANKS, n_full * HQ_LOC, H)
    wg_d = zeros(N_RANKS, n_full * H, PAD)          # head_gate weight (bypassed)
    wgate_d = zeros(N_RANKS, n_dense * H, INT_LOC)
    wup_d = zeros(N_RANKS, n_dense * H, INT_LOC)
    wdown_d = zeros(N_RANKS, n_dense * INT_LOC, H)
    for r in range(N_RANKS):
        wq_d[r, :H, :] = wq_f[:, r * HQ_LOC:(r + 1) * HQ_LOC]
        wk_d[r, :H, :] = wk_f[:, r * KV_LOC:(r + 1) * KV_LOC]
        wv_d[r, :H, :] = wv_f[:, r * KV_LOC:(r + 1) * KV_LOC]
        wo_d[r, :HQ_LOC, :] = wo_f[r * HQ_LOC:(r + 1) * HQ_LOC, :]
        wgate_d[r, :H, :] = wg_f[:, r * INT_LOC:(r + 1) * INT_LOC]
        wup_d[r, :H, :] = wu_f[:, r * INT_LOC:(r + 1) * INT_LOC]
        wdown_d[r, :INT_LOC, :] = wd_f[r * INT_LOC:(r + 1) * INT_LOC, :]

    def rep(t):
        """Replicate an unbatched tensor across ranks: [...] -> [N_RANKS, ...]."""
        return t.unsqueeze(0).expand(N_RANKS, *t.shape).contiguous()

    irms_d = rep(torch.zeros(NLAYERS, H)); irms_d[:, 0, :] = irms
    prms_d = rep(torch.zeros(NLAYERS, H)); prms_d[:, 0, :] = prms
    qn_d = rep(torch.zeros(NLAYERS, HD)); qn_d[:, 0, :] = qn
    kn_d = rep(torch.zeros(NLAYERS, HD)); kn_d[:, 0, :] = kn

    cur_d = rep(cur)
    rcos_d = rep(rcos)
    rsin_d = rep(rsin)
    seq_d = rep(seq_lens)
    bt_d = rep(block_table)
    sm_d = rep(slot_mapping)
    kc_d = zeros(N_RANKS, SEQ, HD)
    vc_d = zeros(N_RANKS, SEQ, HD)
    next_out = torch.zeros(N_RANKS, BATCH, H, dtype=bf16)

    inputs = [
        cur_d, irms_d.float(), wq_d, wk_d, wv_d, qn_d.float(), kn_d.float(),
        seq_d, bt_d, sm_d, rcos_d.float(), rsin_d.float(), kc_d, vc_d,
        wo_d, wg_d, prms_d.float(), wgate_d, wup_d, wdown_d, next_out,
        torch.tensor(0, dtype=torch.int32),  # norm_layer_idx (L0)
        torch.tensor(0, dtype=torch.int32),  # attn_layer_idx (L0)
        torch.tensor(0, dtype=torch.int32),  # mlp_layer_idx (L0)
    ]

    # -- Compile canonical TP=8 + run on N_RANKS cards. --
    import os  # noqa: PLC0415
    import time  # noqa: PLC0415

    from pypto import ir  # noqa: PLC0415
    from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415

    build_dir = f"/tmp/p_multirank_golden_d{args.device.replace(',', '_')}"
    os.environ["PYPTO_PROG_BUILD_DIR"] = build_dir
    os.makedirs(build_dir, exist_ok=True)

    prog, kind = select_decode_layer(0)
    assert kind == "full_dense", f"layer 0 should be full_dense, got {kind}"
    print(f"[MR-golden] compiling canonical TP={cfg.TP_WORLD_SIZE} layer 0 "
          f"on {args.platform} device_ids={device_ids} ...", flush=True)
    compiled = ir.compile(
        prog, platform=args.platform,
        distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
        skip_ptoas=False, dump_passes=False,
    )
    print(f"[MR-golden] compile OK => {compiled.output_dir}", flush=True)

    t0 = time.time()
    compiled(*inputs)
    print(f"[MR-golden] run done in {time.time() - t0:.2f}s; "
          f"out shape={list(next_out.shape)} max|out|={next_out.float().abs().max():.4f} "
          f"max|ref|={ref.abs().max():.4f}", flush=True)

    # -- Validate each rank's output against the full reference (ratio_allclose). --
    atol = rtol = 4e-2
    max_bad_ratio = 0.10
    all_pass = True
    for r in range(N_RANKS):
        out_r = next_out[r].float()
        diff = (out_r - ref).abs()
        tol = atol + rtol * ref.abs()
        bad = (diff > tol).float().mean().item()
        ok = bad <= max_bad_ratio
        all_pass = all_pass and ok
        print(f"  rank {r}: bad_ratio={bad:.4f} max|out|={out_r.abs().max():.4f} "
              f"{'PASS' if ok else 'FAIL'}", flush=True)

    print(f"[MR-golden] {'PASS' if all_pass else 'FAIL'}", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
