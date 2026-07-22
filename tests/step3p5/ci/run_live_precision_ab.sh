#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Live token-alignment A/B gate: pypto whole-net decode vs a LIVE vanilla vLLM
# W8A8 oracle over N teacher-forced greedy tokens. Replaces the old hardcoded
# DEFAULT_ORACLE_TOKENS (which was stale — see LIVE_PRECISION_AB.md).
#
# Two stages because the pypto .venv311 has no transformers and the oracle runs
# in a separate (container) env:
#   Stage 1 (oracle env): gen_vanilla_oracle.py -> ORACLE_IDS_JSON
#   Stage 2 (pypto host):  _stage_main_hidden_only --teacher-forced --oracle-token ...
#
# Env knobs:
#   ORACLE_EXEC : command prefix that runs a python (with transformers) able to
#                 reach the oracle, e.g. on 0162:
#                 "sudo -n nsenter -t <sleep-inf-pid> -m -p -- \
#                    /usr/local/python3.11.14/bin/python3"
#   PYPTO_PY    : pypto host python (default .venv311)
#   CKPT_ORACLE : ckpt path visible to the oracle env (e.g. /mnt/hw910test/...)
#   CKPT_PYPTO  : ckpt path visible to the pypto host (e.g. /data/chensiyu/...)
set -uo pipefail
WS=${WS:-/data/chensiyu/hw_project/pypto/workspace}
LIB=${LIB:-$WS/pypto-lib-live}
SEED=${SEED:-6127}
N=${N:-128}
DEVICES=${DEVICES:-8,9,10,11,12,13,14,15}
CKPT_ORACLE=${CKPT_ORACLE:-/mnt/hw910test/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp}
CKPT_PYPTO=${CKPT_PYPTO:-/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp}
ORACLE_EXEC=${ORACLE_EXEC:?set ORACLE_EXEC (python-with-transformers that can reach the oracle)}
PYPTO_PY=${PYPTO_PY:-$WS/.venv311/bin/python}
OUT=${OUT:-$WS/logs_n1_0162/live_ab_$(date +%Y%m%d_%H%M%S)}; mkdir -p "$OUT"

echo "[stage1] generating vanilla oracle ($N tokens, seed=$SEED)"
$ORACLE_EXEC "$LIB/tests/step3p5/ci/gen_vanilla_oracle.py" \
    --ckpt "$CKPT_ORACLE" --seed-token "$SEED" --n "$N" > "$OUT/oracle.txt" 2>&1
grep -E "SEED=|ORACLE_TEXT=|MULTI_TOKEN_STEPS=" "$OUT/oracle.txt"
IDS=$(grep -oE 'ORACLE_IDS_JSON=.*' "$OUT/oracle.txt" | sed 's/ORACLE_IDS_JSON=//')
[ -z "$IDS" ] && { echo "FAIL: no oracle ids"; cat "$OUT/oracle.txt"; exit 1; }
echo "$IDS" > "$OUT/oracle_ids.json"

echo "[stage2] pypto teacher-forced over the oracle"
set +u; source /usr/local/Ascend/cann/set_env.sh >/dev/null 2>&1 || true
source "$WS/activate.sh" >/dev/null; set -u
export PTO_ISA_ROOT="$WS/pto-isa"
export PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072
export PYTHONPATH="$LIB:${PYTHONPATH:-}"
cd "$LIB"
ORACLE_ARGS=$($PYPTO_PY -c "import json;print(' '.join('--oracle-token %d'%t for t in json.load(open('$OUT/oracle_ids.json'))[:$N]))")
$PYPTO_PY -m tests.step3p5.harnesses._stage_main_hidden_only \
    --device "$DEVICES" --out "$OUT" --ckpt "$CKPT_PYPTO" \
    --seed-token "$SEED" --teacher-forced --steps "$N" --num-blocks 32 \
    $ORACLE_ARGS > "$OUT/pypto.log" 2>&1
grep -E "RESULT=|TEACHER_FORCED_MATCH" "$OUT/pypto.log" | tail -2
$PYPTO_PY - "$OUT/pypto.log" <<'PY'
import sys,json
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{") and "output_token" in l and "step" in l]
rows=[r for r in rows if isinstance(r.get("step"),int)]
m=sum(1 for r in rows if r.get("token_exact"))
print("LIVE_AB_ALIGNED=%d/%d (%.1f%%)"%(m,len(rows),100.0*m/max(1,len(rows))))
PY
