#!/usr/bin/env bash
set -euo pipefail

# Activate env outside if needed: conda activate deepseekocr
# vLLM DeepSeek-OCR online serving

MODEL="deepseek-ai/DeepSeek-OCR"

exec vllm serve "$MODEL" \
  --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0
