#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Reference front-8 MTP3 vLLM launch, isolated in a subshell.
# IMPORTANT: cards 0..7 on 0162 already host a root-owned vLLM service. This
# script is a reproducible launch artifact only; do not run it while port 8000
# or cards 0..7 are occupied.
set -euo pipefail

(
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export VLLM_VERSION=0.17.0
  export VLLM_USE_V1=1
  export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
  export HCCL_OP_EXPANSION_MODE=AIV
  export HCCL_BUFFSIZE=512
  export TASK_QUEUE_ENABLE=0
  export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
  export SHM_BARRIER=true
  export VLLM_ASCEND_ENABLE_PREFETCH_MLP=0
  unset CPU_AFFINITY_CONF
  unset VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE

  CKPT="${CKPT:-/mnt/hw910test/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp}"
  PORT="${PORT:-8000}"
  LOG="${LOG:-/data/LY/logs/step3p5_910b_v017/precision/mtp3.log}"
  mkdir -p "$(dirname "${LOG}")"
  capture_size="$(seq -s, 1 64)"

  LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2 \
    vllm serve "${CKPT}" \
      --trust-remote-code \
      --tensor-parallel-size 8 \
      --pipeline-parallel-size 1 \
      --port "${PORT}" \
      --compilation-config \
        "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[${capture_size}]}" \
      --reasoning-parser step3p5 \
      --enable-auto-tool-choice \
      --tool-call-parser step3p5 \
      --gpu-memory-utilization 0.96 \
      --no-enable-prefix-caching \
      --served-model-name step3.5-flash \
      --quantization=ascend \
      --max-num-seqs 16 \
      --async-scheduling \
      --max-num-batched-tokens 16384 \
      --enable-expert-parallel \
      --additional-config \
        '{"weight_prefetch_config":{"enable":true}}' \
      --speculative_config \
        '{"method":"step3p5_mtp","num_speculative_tokens":3,"enable_multi_layers_mtp":true}' \
      2>&1 | tee "${LOG}"
)
