#!/usr/bin/env bash

set -euo pipefail
set -x

[ -f .env ] && echo ">>> loading .env" && export $(grep -v '^#' .env | xargs)

export PYTHONNOUSERSITE=1
export VERL_PRETTY_ROLLOUT_LOG=1
export VLLM_USE_V1=1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

ray_tmp="/tmp/ray_${USER}"
mkdir -p "${ray_tmp}"
export TMPDIR="${ray_tmp}"
export RAY_TMPDIR="${ray_tmp}"

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG_PATH="${PROJECT_DIR}/LNS/ToDo/fully_async_vllm"
CONFIG_NAME="search_multiturn_grpo_async_vllm"

RUN_TS="$(date +%m%d_%H%M)"
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_DIR}/data/rag/slidevqa_train_6667.parquet}"
VAL_DATA="${VAL_DATA:-${PROJECT_DIR}/data/rag/overall_test_crop.parquet}"
TOOL_CONFIG="${TOOL_CONFIG:-${PROJECT_DIR}/LNS/LNS_tool_config.yaml}"
REWARD_FN="${REWARD_FN:-${PROJECT_DIR}/verl/utils/reward_score/format_ndcg_reward.py}"

TRAINER_NNODES="${TRAINER_NNODES:-1}"
TRAINER_GPUS_PER_NODE="${TRAINER_GPUS_PER_NODE:-4}"
ROLLOUT_NNODES="${ROLLOUT_NNODES:-1}"
ROLLOUT_GPUS_PER_NODE="${ROLLOUT_GPUS_PER_NODE:-4}"

ROLLOUT_N="${ROLLOUT_N:-8}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-8192}"
ROLLOUT_TEST_FREQ="${ROLLOUT_TEST_FREQ:-20}"
ROLLOUT_TOTAL_EPOCHS="${ROLLOUT_TOTAL_EPOCHS:-1}"

STALENESS_THRESHOLD="${STALENESS_THRESHOLD:-0.3}"
TRIGGER_SYNC_STEP="${TRIGGER_SYNC_STEP:-2}"
REQUIRE_BATCHES="${REQUIRE_BATCHES:-1}"
PARTIAL_ROLLOUT="${PARTIAL_ROLLOUT:-True}"
USE_ROLLOUT_LOG_PROBS="${USE_ROLLOUT_LOG_PROBS:-True}"

GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-1}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.6}"

if [[ ! -f "${TRAIN_DATA}" ]]; then
  echo "missing TRAIN_DATA: ${TRAIN_DATA}" >&2
  exit 1
fi
if [[ ! -f "${VAL_DATA}" ]]; then
  echo "missing VAL_DATA: ${VAL_DATA}" >&2
  exit 1
fi
if [[ ! -f "${TOOL_CONFIG}" ]]; then
  echo "missing TOOL_CONFIG: ${TOOL_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${REWARD_FN}" ]]; then
  echo "missing REWARD_FN: ${REWARD_FN}" >&2
  exit 1
fi

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

EXTRA_ARGS=()
if [[ "${CFG_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--cfg job)
fi

python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path="${CONFIG_PATH}" \
    --config-name="${CONFIG_NAME}" \
    custom_reward_function.path="${REWARD_FN}" \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.clip_ratio_low=0.0003 \
    actor_rollout_ref.actor.clip_ratio_high=0.0004 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${VLLM_TP_SIZE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=left \
    actor_rollout_ref.rollout.multi_turn.format=qwen \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.hybrid_engine=False \
    trainer.val_before_train=False \
    trainer.rollout_data_dir="${PROJECT_DIR}/logs/rollout_data_async_${RUN_TS}" \
    trainer.validation_data_dir="${PROJECT_DIR}/logs/val_data_async_${RUN_TS}" \
    trainer.project_name=gspo_phase1_revised_async_vllm \
    trainer.experiment_name=gspo_phase1_revised_async_vllm \
    trainer.logger='["console","wandb"]' \
    trainer.n_gpus_per_node="${TRAINER_GPUS_PER_NODE}" \
    trainer.nnodes="${TRAINER_NNODES}" \
    trainer.save_freq=100 \
    trainer.test_freq=50000000000000 \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.gen_batch_size="${GEN_BATCH_SIZE}" \
    data.train_batch_size=0 \
    trainer.total_epochs=1 \
    rollout.nnodes="${ROLLOUT_NNODES}" \
    rollout.n_gpus_per_node="${ROLLOUT_GPUS_PER_NODE}" \
    rollout.total_rollout_steps="${ROLLOUT_STEPS}" \
    rollout.total_epochs="${ROLLOUT_TOTAL_EPOCHS}" \
    rollout.test_freq="${ROLLOUT_TEST_FREQ}" \
    async_training.staleness_threshold="${STALENESS_THRESHOLD}" \
    async_training.trigger_parameter_sync_step="${TRIGGER_SYNC_STEP}" \
    async_training.require_batches="${REQUIRE_BATCHES}" \
    async_training.partial_rollout="${PARTIAL_ROLLOUT}" \
    async_training.use_rollout_log_probs="${USE_ROLLOUT_LOG_PROBS}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
