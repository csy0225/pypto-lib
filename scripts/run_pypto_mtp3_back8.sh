#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Isolated cards-8..15 PyPTO main -> sampler -> MTP3 validation.
#
# This script never starts/stops the cards-0..7 vLLM service. It launches and
# cleans up a fresh cards-8..15 MTP-capable IPC exporter pool itself.
set -euo pipefail

(
  # PyPTO takes explicit physical device IDs. Do not inherit a front-8 logical
  # remapping or any vLLM process-global behavior knob.
  unset ASCEND_RT_VISIBLE_DEVICES
  unset VLLM_VERSION
  unset VLLM_USE_V1
  unset VLLM_ASCEND_ENABLE_FLASHCOMM1
  unset VLLM_ASCEND_ENABLE_PREFETCH_MLP
  unset VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE
  unset VLLM_ENABLE_V1_MULTIPROCESSING
  unset HCCL_OP_EXPANSION_MODE
  unset HCCL_BUFFSIZE
  unset TASK_QUEUE_ENABLE
  unset SHM_BARRIER
  unset CPU_AFFINITY_CONF
  unset LD_PRELOAD
  export PYTORCH_NPU_ALLOC_CONF=""

  # Some vendor environment scripts reference unset variables.
  set +u
  source /usr/local/Ascend/cann/set_env.sh
  source /data/chensiyu/hw_project/pypto/workspace/activate.sh
  set -u
  export PTO_ISA_ROOT=/data/chensiyu/hw_project/pypto/workspace/pto-isa
  export PTO2_RING_HEAP=4294967296
  export PTO2_RING_TASK_WINDOW=131072
  export PTO2_RING_DEP_POOL=131072
  export P_FAITHFUL_MOE_LAYERS=42

  cd /data/chensiyu/hw_project/pypto/workspace/pypto-lib-mtp3

  CKPT="${CKPT:-/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp}"
  OUT="${OUT:-/tmp/n1_weight_ipc_mtp3}"
  DUMP="${DUMP:-/tmp/n1_mtp3_chain}"
  LOG_DIR="${LOG_DIR:-/data/chensiyu/hw_project/pypto/workspace/logs_n1/mtp3}"
  FIRST_TOKEN="${FIRST_TOKEN:-303}"
  mkdir -p "${DUMP}" "${OUT}" "${LOG_DIR}"

  exporter_pids=()
  stop_exporters() {
    touch "${OUT}/STOP" 2>/dev/null || true
    for pid in "${exporter_pids[@]}"; do
      wait "${pid}" 2>/dev/null || true
    done
    rm -f "${OUT}"/ready.rank* "${OUT}/STOP"
  }
  trap stop_exporters EXIT

  rm -f \
    "${OUT}"/ready.rank* \
    "${OUT}/STOP" \
    "${OUT}"/pypto_weight.key.rank* \
    "${OUT}"/pypto_weight_map.rank*.json \
    "${OUT}"/pypto_weight_map.rank*.json.done

  for rank in $(seq 0 7); do
    dev=$((8 + rank))
    python -m tests.step3p5._stage_whole_mtp3_ipc \
      --export-rank "${rank}" \
      --dev "${dev}" \
      --out "${OUT}" \
      --ckpt "${CKPT}" \
      > "${LOG_DIR}/export_rank${rank}.log" 2>&1 &
    exporter_pids+=("$!")
  done

  deadline=$((SECONDS + 2400))
  while [[ "$(find "${OUT}" -maxdepth 1 -name 'ready.rank*' | wc -l)" -lt 8 ]]; do
    if (( SECONDS >= deadline )); then
      echo "MTP3 exporters were not ready within 40 minutes" >&2
      exit 1
    fi
    for pid in "${exporter_pids[@]}"; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        echo "MTP3 exporter ${pid} exited before readiness" >&2
        exit 1
      fi
    done
    sleep 15
  done

  # Canonical main remains the sole P42 release gate.
  N1_DUMP_DIR="${DUMP}" \
    python -m tests.step3p5._stage_whole_faithful_real_ipc \
      --device 8,9,10,11,12,13,14,15 \
      --reuse-exporters \
      --kv-ipc \
      --hidden-token 6127 \
      --out "${OUT}" \
      --ckpt "${CKPT}"

  python -m tests.step3p5._stage_whole_mtp3_ipc \
    --device 8,9,10,11,12,13,14,15 \
    --reuse-exporters \
    --out "${OUT}" \
    --ckpt "${CKPT}" \
    --previous-hidden "${DUMP}/P42_nh_row0.pt" \
    --first-token "${FIRST_TOKEN}" \
    --dump-dir "${DUMP}"
)
