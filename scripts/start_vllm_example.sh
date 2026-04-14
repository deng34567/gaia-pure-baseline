#!/usr/bin/env bash

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/ma-user/work/download/Qwen/Qwen3.5-9B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35-9b}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-sk-12345}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

HCCL_OP_EXPANSION_MODE=AIV python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --api-key "${API_KEY}"
