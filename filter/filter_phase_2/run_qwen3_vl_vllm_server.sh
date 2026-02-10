#!/usr/bin/env bash
set -euo pipefail

# Qwen3-VL-235B-A22B-Thinking (H200/B200 recommended)
# Text-only mode: disable image/video to free memory.
# See: https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html

MODEL="Qwen/Qwen3-VL-235B-A22B-Thinking"
API_KEY="${API_KEY:-token-abc123}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"
REASONING_PARSER="${REASONING_PARSER:-deepseek_r1}"

vllm serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --api-key "${API_KEY}" \
  --served-model-name "${MODEL}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --reasoning-parser "${REASONING_PARSER}" \
  --mm-encoder-tp-mode data \
  --async-scheduling \
  --limit-mm-per-prompt.image 4 \
  --limit-mm-per-prompt.video 0

  # --limit-mm-per-prompt.image 는 입력 이미지 존재 여부에 따라 동적으로 변화. 
