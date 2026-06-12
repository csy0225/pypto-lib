"""Phase 15 Feasibility Probe — throwaway script.

Tests whether:
  A) The TP=8 compiled binary has HARDCODED collective loop bounds
  B) Re-compiling with TP=1 (no collective loops) succeeds
  C) TP=1 compiled program runs on real NPU with synthetic inputs
"""
import os, sys, time, traceback, pathlib

os.chdir("/data/chensiyu/hw_project/pypto/workspace/pypto-lib")

print("=" * 68)
print("PROBE A — Inspect TP=8 compiled binary host_orch.py")
print("=" * 68)

COMPILED_8R = (
    "/data/chensiyu/hw_project/pypto/workspace/build_output/p14/"
    "DecodeLayerDense_20260606_080006"
)
orch_py = pathlib.Path(COMPILED_8R) / "orchestration" / "host_orch.py"
if orch_py.exists():
    content = orch_py.read_text()
    print("Relevant lines in host_orch.py (workers/world_size):")
    for line in content.split("\n"):
        if "workers" in line or "world_size" in line:
            print(f"  >> {line.rstrip()}")
else:
    print(f"  host_orch.py not found at {orch_py}")

ar_pto = pathlib.Path(COMPILED_8R) / "next_levels" / "chip_orch" / "ptoas" / "tp_all_reduce.pto"
if ar_pto.exists():
    pto_txt = ar_pto.read_text()
    loop_lines = [l.strip() for l in pto_txt.split("\n")
                  if "scf.for" in l or "c7_index" in l or "twait" in l or "tnotify" in l]
    print(f"\ntp_all_reduce.pto loop/comm lines ({len(loop_lines)} found):")
    for l in loop_lines[:15]:
        print(f"  >> {l}")
    print("=> TP=8 collective loop HARDCODED group_size-1=7 steps => DEADLOCK if 1 rank")
else:
    print(f"  tp_all_reduce.pto not found at {ar_pto}")

print()
print("=" * 68)
print("PROBE B — Patch config TP=1 + recompile for a2a3")
print("=" * 68)

import importlib
import models.step3p5.config as cfg_mod

# Save originals
orig_keys = [
    "TP_WORLD_SIZE", "EP_WORLD_SIZE",
    "NUM_HEADS_FULL_LOCAL", "NUM_HEADS_SWA_LOCAL", "KV_HEADS_LOCAL",
    "NUM_HEADS_FULL_LOCAL_PAD", "NUM_HEADS_SWA_LOCAL_PAD",
    "HIDDEN_Q_FULL_LOCAL", "HIDDEN_Q_SWA_LOCAL", "KV_HIDDEN_LOCAL",
    "INTERMEDIATE_LOCAL", "SHARE_EXPERT_DIM_LOCAL", "VOCAB_LOCAL",
    "MOE_NUM_EXPERTS_LOCAL", "KV_PROJ_K_CHUNK_LOCAL",
]
_orig = {k: getattr(cfg_mod, k) for k in orig_keys}

# Apply TP=1 / EP=1 patch
# --- world-size scalars ---
cfg_mod.TP_WORLD_SIZE = 1
cfg_mod.EP_WORLD_SIZE = 1
# --- per-rank head/dim counts ---
cfg_mod.NUM_HEADS_FULL_LOCAL   = cfg_mod.NUM_HEADS_FULL      # 64
cfg_mod.NUM_HEADS_SWA_LOCAL    = cfg_mod.NUM_HEADS_SWA       # 96
cfg_mod.KV_HEADS_LOCAL         = cfg_mod.NUM_KV_HEADS        # 8
cfg_mod.HIDDEN_Q_FULL_LOCAL    = cfg_mod.HIDDEN_Q_FULL       # 8192
cfg_mod.HIDDEN_Q_SWA_LOCAL     = cfg_mod.HIDDEN_Q_SWA        # 12288
cfg_mod.KV_HIDDEN_LOCAL        = cfg_mod.KV_HIDDEN           # 1024
cfg_mod.INTERMEDIATE_LOCAL     = cfg_mod.INTERMEDIATE        # 11264
cfg_mod.SHARE_EXPERT_DIM_LOCAL = cfg_mod.SHARE_EXPERT_DIM    # 1280
cfg_mod.VOCAB_LOCAL            = cfg_mod.VOCAB               # 128896
cfg_mod.MOE_NUM_EXPERTS_LOCAL  = cfg_mod.MOE_NUM_EXPERTS     # 288
# KV_PROJ_K_CHUNK_LOCAL: config.py sets this at module-import time based on
# KV_HIDDEN_LOCAL. After patching KV_HIDDEN_LOCAL = 1024 above, we must also
# explicitly set the adaptive chunk to 128 (TP=1 path) or it stays at 256
# (TP=8 default), causing L0B overflow in full_k_proj / full_v_proj.
cfg_mod.KV_PROJ_K_CHUNK_LOCAL  = cfg_mod.KV_PROJ_K_CHUNK     # 128 (TP=1: KV_HIDDEN_LOCAL=1024 > 256)
# --- PAD constants: ceil(N/16)*16 ---
import math
cfg_mod.NUM_HEADS_FULL_LOCAL_PAD = math.ceil(cfg_mod.NUM_HEADS_FULL / 16) * 16  # 64
cfg_mod.NUM_HEADS_SWA_LOCAL_PAD  = math.ceil(cfg_mod.NUM_HEADS_SWA  / 16) * 16  # 96
print(f"  PAD: FULL={cfg_mod.NUM_HEADS_FULL_LOCAL_PAD} SWA={cfg_mod.NUM_HEADS_SWA_LOCAL_PAD}")

# Reload all modules that capture config constants at module level, in dep order
import sys
import models.step3p5.attention_full as attn_full_mod
import models.step3p5.attention_swa  as attn_swa_mod
import models.step3p5.decode_layer   as dl_mod
attn_full_mod = importlib.reload(attn_full_mod)
attn_swa_mod  = importlib.reload(attn_swa_mod)
dl_mod        = importlib.reload(dl_mod)
print(f"  After TP=1 patch: TP_CHUNK={dl_mod.TP_CHUNK} (expect {cfg_mod.HIDDEN}=4096)")

from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

program, kind = dl_mod.select_decode_layer(0)
print(f"  Layer 0: kind={kind} type={type(program).__name__}")

build_dir = "/tmp/p15_probe_tp1"
os.makedirs(build_dir, exist_ok=True)
os.environ["PYPTO_PROG_BUILD_DIR"] = build_dir

dist_cfg = DistributedConfig(device_ids=[0], num_sub_workers=0)
print(f"  Compiling with platform='a2a3' device_ids=[0] TP=1 ...")
sys.stdout.flush()
t0 = time.time()
try:
    compiled = ir.compile(
        program,
        platform="a2a3",
        distributed_config=dist_cfg,
        skip_ptoas=False,
        dump_passes=False,
    )
    print(f"  Compile OK in {time.time()-t0:.1f}s  => {compiled.output_dir}")
except Exception as e:
    print(f"  Compile FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

# Inspect generated .pto for tp_all_reduce loop bound
pto_dir = pathlib.Path(compiled.output_dir) / "next_levels" / "chip_orch" / "ptoas"
ar2 = list(pto_dir.glob("*all_reduce*")) if pto_dir.exists() else []
if ar2:
    pto2_txt = ar2[0].read_text()
    loop2 = [l.strip() for l in pto2_txt.split("\n")
              if "scf.for" in l or "twait" in l or "tnotify" in l or "c0_index" in l]
    print(f"  tp_all_reduce.pto loop/comm lines ({len(loop2)}):")
    for l in loop2[:8]:
        print(f"    >> {l}")
    if not loop2:
        print("    (no scf.for / twait / tnotify => collective is a no-op!)")
else:
    all_ptos = list(pto_dir.glob("*.pto")) if pto_dir.exists() else []
    print(f"  No tp_all_reduce.pto found — TP=1 emits no collective kernel.")
    print(f"  All .pto files: {[p.name for p in all_ptos]}")

# Check host_orch.py world_size usage
orch2 = pathlib.Path(compiled.output_dir) / "orchestration" / "host_orch.py"
if orch2.exists():
    c2 = orch2.read_text()
    print("\n  host_orch.py relevant lines (workers/world_size):")
    for line in c2.split("\n"):
        if "workers" in line or "world_size" in line:
            print(f"    >> {line.rstrip()}")

print()
print("=" * 68)
print("PROBE C — Execute TP=1 compiled program on real NPU")
print("=" * 68)
try:
    import torch

    WS = 1
    B  = cfg_mod.BATCH
    H  = cfg_mod.HIDDEN
    H_INT = cfg_mod.INTERMEDIATE
    H_KV  = cfg_mod.KV_HIDDEN
    H_Q   = cfg_mod.HIDDEN_Q_FULL
    HDim  = cfg_mod.HEAD_DIM
    NKV   = cfg_mod.NUM_KV_HEADS
    NQ    = cfg_mod.NUM_HEADS_FULL
    NQ_PAD = cfg_mod.NUM_HEADS_FULL_LOCAL_PAD
    LROWS = cfg_mod.NUM_TOTAL_LAYERS
    SEQ   = cfg_mod.MAX_SEQ_DEFAULT
    MBS   = cfg_mod.MAX_BLOCKS_PER_SEQ

    print(f"  Shapes: WS={WS} B={B} H={H} INTERMEDIATE={H_INT} NQ={NQ} NKV={NKV}")

    def t(*shape, dt=torch.bfloat16): return torch.randn(*shape, dtype=dt)
    def z(*shape, dt=torch.bfloat16): return torch.zeros(*shape, dtype=dt)
    def i(*shape): return torch.zeros(*shape, dtype=torch.int32)

    layer_idx_t = torch.tensor(0, dtype=torch.int32)
    inputs = [
        layer_idx_t,
        t(WS, B, H),                           # current_hidden
        t(WS, LROWS, H, dt=torch.float32),     # input_rms_weight
        t(WS, LROWS*H, H_Q),                   # wq
        t(WS, LROWS*HDim, HDim),               # wk
        t(WS, LROWS*HDim, HDim),               # wv
        t(WS, LROWS, HDim, dt=torch.float32),  # q_norm
        t(WS, LROWS, HDim, dt=torch.float32),  # k_norm
        i(WS, B),                               # seq_lens
        i(WS, MBS*B),                           # block_table
        i(WS, B),                               # slot_mapping
        t(WS, SEQ, HDim//4, dt=torch.float32), # rope_cos
        t(WS, SEQ, HDim//4, dt=torch.float32), # rope_sin
        t(WS, SEQ, HDim),                      # k_cache
        t(WS, SEQ, HDim),                      # v_cache
        t(WS, LROWS*H_Q, H),                   # wo
        t(WS, LROWS*H, NQ_PAD),               # w_g
        t(WS, LROWS, H, dt=torch.float32),     # post_rms_weight
        t(WS, LROWS*H, H_INT),                # w_gate
        t(WS, LROWS*H, H_INT),                # w_up
        t(WS, LROWS*H_INT, H),               # w_down
        z(WS, B, H),                            # next_hidden_out (output)
    ]

    print(f"  All {len(inputs)} inputs allocated. Calling compiled() on NPU ...")
    sys.stdout.flush()
    t0 = time.time()
    compiled(*inputs)
    elapsed = time.time() - t0
    out = inputs[-1]
    print(f"  EXECUTE OK in {elapsed:.2f}s")
    print(f"  next_hidden_out: shape={list(out.shape)} max={out.float().abs().max().item():.4f}")
    print("PROBE_RESULT=SUCCESS_TP1_SINGLE_RANK")

except Exception as e:
    print(f"  Execute FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    print("PROBE_RESULT=EXECUTE_ERROR")

print()
print("PROBE DONE")
