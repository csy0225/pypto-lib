#!/usr/bin/env bash
# Capture and analyze the Step3p5 BS1 first-five-layer (L0-L4) chip swimlane.
#
# The harness submits each kernel twice: dependency generation first, then a
# timing-only pass that reuses the prepared graph. These instrumented makespans
# explain critical-path composition; they are not clean absolute latency
# claims. Use the independent clean ITL gate for release latency.
#
# Usage:
#   PYPTO_UPGRADE_WORKSPACE=/path/to/run-root CKPT=/path/to/checkpoint \
#     bash deployment/docker/run_swimlane_gate.sh <image-id-or-digest>
set -Eeuo pipefail

cd /tmp

IMGID=${1:?need image id or digest}
WORKSPACE_ROOT=${PYPTO_UPGRADE_WORKSPACE:?set PYPTO_UPGRADE_WORKSPACE}
CKPT=${CKPT:?set CKPT}
OUT=${PYPTO_SWIMLANE_OUT:-"$WORKSPACE_ROOT/swimlane-$(date +%Y%m%d-%H%M%S)"}
DEVCSV=${PYPTO_DEVICES:-0,1,2,3,4,5,6,7}
PROFILE=${PYPTO_MOE_DFX_PROFILE:-candidate}
RECV_META_SIDECAR=${PYPTO_RECV_META_SIDECAR:-}
SWIMLANE_RECORDS_NAME=chip_swimlane_records.json
NERDCTL=${NERDCTL:-/mnt/persist/k8s-install/containerd/bin/nerdctl}
NONCE=$(printf 'swim:%s:%s' "$$" "$(date +%s%N)" | sha256sum | awk '{print $1}')

IFS=, read -r -a DEVICE_IDS <<< "$DEVCSV"
if [[ ${#DEVICE_IDS[@]} -ne 8 ]]; then
  echo "expected exactly 8 comma-separated devices, got: $DEVCSV" >&2
  exit 2
fi

DEVICE_ARGS=()
for device_id in "${DEVICE_IDS[@]}"; do
  DEVICE_ARGS+=(--device "/dev/davinci${device_id}")
done

RECV_META_ARGS=()
RECV_META_SUPPLIED=false
if [[ -n $RECV_META_SIDECAR ]]; then
  if [[ ! -f $RECV_META_SIDECAR ]]; then
    echo "recv_meta sidecar does not exist: $RECV_META_SIDECAR" >&2
    exit 2
  fi
  RECV_META_ARGS=(
    -v "$RECV_META_SIDECAR":/input/recv_meta_sidecar:ro
    --env PYPTO_RECV_META_SIDECAR=/input/recv_meta_sidecar
  )
  RECV_META_SUPPLIED=true
fi

mkdir -p "$OUT/runtime"
cat > "$OUT/run_contract.json" <<JSON
{"kind":"five_layer_chip_swimlane","image":"$IMGID","devices":"$DEVCSV",
 "active_batch":1,"context_len":65536,"num_blocks":512,"warmup":3,"iters":20,
 "seed_token":6127,"instrumented":true,
 "absolute_latency_is_not_a_clean_claim":true,
 "digest_only":true,"source_overlay":false,"runtime_overlay":false,
 "analyzer_profile":"$PROFILE",
 "recv_meta_sidecar_supplied":$RECV_META_SUPPLIED,
 "ipc_session_nonce":"$NONCE","codegen_max_workers":1}
JSON
cat "$OUT/run_contract.json"
date -Is > "$OUT/started_at.txt"

set +e
sudo -n "$NERDCTL" run --rm --net host --ipc host --privileged \
  --security-opt apparmor=unconfined \
  --env PYPTO_LIVE_IPC_STRICT=1 \
  --env PYPTO_CODEGEN_MAX_WORKERS=1 \
  --env PYPTO_RELEASE_CONTEXT_LEN=65536 \
  --env PYPTO_STEP3P5_MAX_SEQ=65536 \
  --env PYPTO_STEP3P5_ROPE_SEQ=65536 \
  --env SIMPLER_A2A3_FORCE_VMM_IPC=1 \
  --env PYPTO_IPC_SESSION_NONCE="$NONCE" \
  --env PYPTO_MOE_DFX_PROFILE="$PROFILE" \
  --env PYPTO_REQUIRE_L2_SWIMLANE_REUSE_DEP_GEN=1 \
  --env PYPTO_PROG_BUILD_DIR=/out/runtime/build_output \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTHONNOUSERSITE=1 \
  --env PTO2_RING_HEAP=4294967296 \
  --env PTO2_RING_TASK_WINDOW=131072 \
  --env PTO2_RING_DEP_POOL=131072 \
  --env PYPTO_STEP3P5_STORAGE_BATCH_CAPACITY=16 \
  --env PYPTO_DEVICES="$DEVCSV" \
  --env CKPT="$CKPT" \
  "${DEVICE_ARGS[@]}" \
  "${RECV_META_ARGS[@]}" \
  --device /dev/davinci_manager \
  --device /dev/hisi_hdc \
  --device /dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$CKPT":"$CKPT":ro \
  -v "$OUT":/out \
  --shm-size 32g \
  "$IMGID" bash -lc '
    set -Eeuo pipefail
    cd /workspace/pypto-lib
    PYPTO_IPC_LAUNCH_EPOCH=$(python -c "import time; print(f\"{time.time():.9f}\")")
    export PYPTO_IPC_LAUNCH_EPOCH
    sha256sum models/step3p5/decode_fwd.py > /out/decode_fwd.container.sha256
    sha256sum /workspace/pypto/python/pypto/runtime/distributed_runner.py \
      > /out/distributed_runner.container.sha256
    git -C /workspace/pypto-lib rev-parse HEAD > /out/pypto_lib_head.txt 2>/dev/null || true
    git -C /workspace/pypto rev-parse HEAD > /out/pypto_head.txt 2>/dev/null || true
    python -m tests.step3p5.harnesses._stage_five_layer_moe \
      --device "$PYPTO_DEVICES" \
      --ckpt "$CKPT" \
      --out /out/runtime \
      --num-blocks 512 \
      --context-len 65536 \
      --active-batch 1 \
      --seed-token 6127 \
      --warmup 3 \
      --iters 20 \
      --dfx
    echo SWIMLANE_RUN_OK
  ' 2>&1 | tee "$OUT/container.log"
container_rc=${PIPESTATUS[0]}
set -e

printf '%s\n' "$container_rc" > "$OUT/container.rc"
date -Is > "$OUT/finished_at.txt"

find "$OUT" -name "$SWIMLANE_RECORDS_NAME" -print \
  | sort \
  | sed "s|^$OUT/||" \
  > "$OUT/swimlane_records.list"
record_count=$(wc -l < "$OUT/swimlane_records.list")

echo "===== container rc=$container_rc ====="
echo "===== $SWIMLANE_RECORDS_NAME ($record_count/8) ====="
cat "$OUT/swimlane_records.list"
echo "===== critical-path makespan per rank ====="
while IFS= read -r report; do
  rank=$(sed -n 's|.*\(rank[0-7]\)/.*|\1|p' <<< "$report")
  makespan=$(grep -m1 -iE "makespan" "$report" | tr -s ' ' || true)
  printf '%s  %s\n' "$rank" "${makespan:-<missing makespan>}"
done < <(find "$OUT" -name "critical_path_report.md" -print | sort)

gate_rc=$container_rc
if [[ $record_count -ne 8 && $gate_rc -eq 0 ]]; then
  gate_rc=1
fi
printf '%s\n' "$gate_rc" > "$OUT/gate.rc"

echo "[gate] OUT=$OUT container_rc=$container_rc records=$record_count gate_rc=$gate_rc"
if [[ $gate_rc -eq 0 ]]; then
  echo SWIMLANE_GATE_RC0
else
  echo SWIMLANE_GATE_RC_NONZERO
fi
exit "$gate_rc"
