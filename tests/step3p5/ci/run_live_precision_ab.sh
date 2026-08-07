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
#   ORACLE_PYTHON: executable that runs Python with transformers and can reach
#                  the oracle. Use a fixed wrapper executable for namespaces.
#   PYPTO_PY    : pypto host python (default .venv311)
#   CKPT_ORACLE : ckpt path visible to the oracle env (e.g. /mnt/hw910test/...)
#   CKPT_PYPTO  : ckpt path visible to the pypto host (e.g. /data/chensiyu/...)
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LIB=${LIB:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
WS=${WS:-$(cd "$LIB/.." && pwd)}
SEED=${SEED:-6127}
N=${N:-128}
MIN_ALIGNMENT_PCT=${MIN_ALIGNMENT_PCT:-95}
LIVE_PRECISION_RELEASE_GATE=${LIVE_PRECISION_RELEASE_GATE:-1}
DEVICES=${DEVICES:-8,9,10,11,12,13,14,15}
CKPT_ORACLE=${CKPT_ORACLE:-/mnt/hw910test/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp}
CKPT_PYPTO=${CKPT_PYPTO:-/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp}
CHECKPOINT_MANIFEST=${CHECKPOINT_MANIFEST:?set CHECKPOINT_MANIFEST to a trusted full-shard identity manifest}
ORACLE_PYTHON=${ORACLE_PYTHON:?set ORACLE_PYTHON to the oracle Python executable}
PYPTO_PY=${PYPTO_PY:-$WS/.venv311/bin/python}
OUT=${OUT:-$WS/logs_n1_0162/live_ab_$(date +%Y%m%d_%H%M%S)_$$}; mkdir -p "$OUT"
GATE="$LIB/tests/step3p5/ci/live_precision_gate.py"
case "$LIVE_PRECISION_RELEASE_GATE" in
    0) GATE_MODE_ARGS=() ;;
    1) GATE_MODE_ARGS=(--release) ;;
    *)
        echo "FAIL: LIVE_PRECISION_RELEASE_GATE must be 0 or 1" >&2
        exit 2
        ;;
esac

"$PYPTO_PY" "$GATE" validate-config \
    --expected "$N" --threshold "$MIN_ALIGNMENT_PCT" --seed "$SEED" \
    "${GATE_MODE_ARGS[@]}"
test -s "$CHECKPOINT_MANIFEST"

SOURCE_COMMIT=$(git -C "$LIB" rev-parse HEAD)
SOURCE_STATUS=$(git -C "$LIB" status --porcelain --untracked-files=all)
printf '%s\n' "$SOURCE_COMMIT" > "$OUT/source_commit.txt"
printf '%s' "$SOURCE_STATUS" > "$OUT/source_status.txt"
if [ "$LIVE_PRECISION_RELEASE_GATE" = 1 ] && [ -n "$SOURCE_STATUS" ]; then
    echo "FAIL: release precision gate requires a clean source tree" >&2
    cat "$OUT/source_status.txt" >&2
    exit 2
fi
echo "[source] repo=$LIB commit=$SOURCE_COMMIT release_gate=$LIVE_PRECISION_RELEASE_GATE"

echo "[stage1] generating vanilla oracle ($N tokens, seed=$SEED)"
"$ORACLE_PYTHON" "$GATE" verify-checkpoint \
    --checkpoint "$CKPT_ORACLE" \
    --manifest "$CHECKPOINT_MANIFEST" \
    --out "$OUT/oracle_checkpoint_identity.json"
"$ORACLE_PYTHON" "$LIB/tests/step3p5/ci/gen_vanilla_oracle.py" \
    --ckpt "$CKPT_ORACLE" --seed-token "$SEED" --n "$N" > "$OUT/oracle.txt" 2>&1
grep -E "SEED=|ORACLE_TEXT=" "$OUT/oracle.txt"
"$PYPTO_PY" "$GATE" extract-oracle \
    --log "$OUT/oracle.txt" --expected "$N" --out "$OUT/oracle_ids.json"

echo "[stage2] pypto teacher-forced over the oracle"
set +u; source /usr/local/Ascend/cann/set_env.sh >/dev/null 2>&1 || true
source "$WS/activate.sh" >/dev/null; set -u
export PTO_ISA_ROOT="$WS/pto-isa"
export PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072
export PYTHONPATH="$LIB:${PYTHONPATH:-}"
export PYPTO_PROG_BUILD_DIR="$OUT/build_output"
cd "$LIB"
"$PYPTO_PY" "$GATE" verify-checkpoint \
    --checkpoint "$CKPT_PYPTO" \
    --manifest "$CHECKPOINT_MANIFEST" \
    --out "$OUT/pypto_checkpoint_identity.json"
"$PYPTO_PY" "$GATE" compare-checkpoint \
    --oracle "$OUT/oracle_checkpoint_identity.json" \
    --pypto "$OUT/pypto_checkpoint_identity.json"
read -r -a ORACLE_ARGS <<< "$("$PYPTO_PY" "$GATE" render-args \
    --oracle-json "$OUT/oracle_ids.json" --expected "$N")"
"$PYPTO_PY" -m tests.step3p5.harnesses._stage_main_hidden_only \
    --device "$DEVICES" --out "$OUT" --ckpt "$CKPT_PYPTO" \
    --seed-token "$SEED" --teacher-forced --steps "$N" --num-blocks 32 \
    "${ORACLE_ARGS[@]}" > "$OUT/pypto.log" 2>&1
grep -E "RESULT=|TEACHER_FORCED_MATCH" "$OUT/pypto.log" | tail -2
"$PYPTO_PY" "$GATE" validate-result \
    --log "$OUT/pypto.log" \
    --oracle-json "$OUT/oracle_ids.json" \
    --expected "$N" \
    --threshold "$MIN_ALIGNMENT_PCT" \
    --seed "$SEED" \
    "${GATE_MODE_ARGS[@]}"
