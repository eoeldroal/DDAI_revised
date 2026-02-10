#!/usr/bin/env bash
set -euo pipefail

ROOT="/opt/dlami/nvme/isdslab/HyunBin/DDAI_Revised/verl"
SCRIPT="$ROOT/filter/filter_phase_2/build_decks_mix_full.py"
LOG_DIR="$ROOT/data/VDR_processed_filtered_2/Deck_processing/exp_setting2_mix_full/runner_logs"

# Unified in-script configuration (no external export required)
ENV_NAME="deckmix310"
DATASETS="OpenDocVQA,VDR_ibm"
WORLD_SIZE=8
RERANK_BATCH_SIZE=64
MAX_CACHE_SHARDS=24
INDEX_TYPE="ivf"
NLIST=4096
NPROBE=256
LOG_EVERY=200
LOG_SHARD_EVERY=16

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH

mkdir -p "$LOG_DIR"
cd "$ROOT"

echo "[runner] start $(date '+%F %T') datasets=$DATASETS world_size=$WORLD_SIZE env=$ENV_NAME cache_shards=$MAX_CACHE_SHARDS rerank_batch=$RERANK_BATCH_SIZE"

PREP_LOG="$LOG_DIR/prepare_cache.log"
echo "[runner] prepare cache/index once -> $PREP_LOG"
conda run --no-capture-output -n "$ENV_NAME" \
  python -u "$SCRIPT" \
    --datasets "$DATASETS" \
    --prepare-cache-only \
    --index "$INDEX_TYPE" \
    --nlist "$NLIST" \
    --nprobe "$NPROBE" \
    --max-cache-shards "$MAX_CACHE_SHARDS" \
    --rerank-batch-size "$RERANK_BATCH_SIZE" \
    --log-shard-every "$LOG_SHARD_EVERY" \
  > "$PREP_LOG" 2>&1
echo "[runner] prepare done $(date '+%F %T')"

pids=()
for i in $(seq 0 $((WORLD_SIZE-1))); do
  log_file="$LOG_DIR/rank${i}.log"
  echo "[runner] launch rank=$i log=$log_file"
  CUDA_VISIBLE_DEVICES=$i conda run --no-capture-output -n "$ENV_NAME" \
    python -u "$SCRIPT" \
      --datasets "$DATASETS" \
      --rank "$i" --world-size "$WORLD_SIZE" \
      --device cuda:0 \
      --index "$INDEX_TYPE" --faiss-gpu \
      --nlist "$NLIST" --nprobe "$NPROBE" \
      --max-cache-shards "$MAX_CACHE_SHARDS" \
      --rerank-batch-size "$RERANK_BATCH_SIZE" \
      --resume \
      --log-every "$LOG_EVERY" --log-shard-every "$LOG_SHARD_EVERY" \
    > "$log_file" 2>&1 &
  pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

if [[ $fail -ne 0 ]]; then
  echo "[runner] finished with failures $(date '+%F %T')"
  exit 1
fi

echo "[runner] finished successfully $(date '+%F %T')"
