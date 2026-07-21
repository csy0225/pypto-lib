#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Operator entry point for the isolated cards-8..15 single-chip hidden-only gate.
#
# The authoritative orchestration now lives in
# tests.step3p5.ci.run_whole_network_ci.  Keep this shell file only for workspace
# activation and backwards-compatible operator usage; do not add a second
# exporter/stage/cleanup implementation here.
set -euo pipefail

(
  # Some vendor environment scripts reference unset variables.
  set +u
  source "${ASCEND_SET_ENV:-/usr/local/Ascend/cann/set_env.sh}"
  source "${PYPTO_ACTIVATE:-/data/chensiyu/hw_project/pypto/workspace/activate.sh}"
  set -u

  export PTO_ISA_ROOT="${PTO_ISA_ROOT:-/data/chensiyu/hw_project/pypto/workspace/pto-isa}"
  export PTO2_RING_HEAP="${PTO2_RING_HEAP:-4294967296}"
  export PTO2_RING_TASK_WINDOW="${PTO2_RING_TASK_WINDOW:-131072}"
  export PTO2_RING_DEP_POOL="${PTO2_RING_DEP_POOL:-131072}"

  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  cd "${repo_root}"

  ckpt="${CKPT:-/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp}"
  devices="${DEVICES:-8,9,10,11,12,13,14,15}"
  out="${OUT:-/tmp/n1_single_chip_hidden_ci}"
  artifact_dir="${ARTIFACT_DIR:-/data/chensiyu/hw_project/pypto/workspace/logs_n1_0162/single_chip_hidden_ci}"

  args=(
    --ckpt "${ckpt}"
    --devices "${devices}"
    --out "${out}"
    --artifact-dir "${artifact_dir}"
  )
  if [[ "${RUN_BATCH16:-1}" == "0" ]]; then
    args+=(--no-run-batch16)
  fi

  exec python -m tests.step3p5.ci.run_whole_network_ci "${args[@]}" "$@"
)
