set -x

# =============================================================================
# 1. 환경 설정
# =============================================================================
# .env 파일 로드 (루트 폴더)
if [ -f .env ]; then
    echo ">>> .env 파일 로드 중..."
    export $(grep -v '^#' .env | xargs)
fi

export PYTHONNOUSERSITE=1
export PYTHONASYNCIODEBUG=1

export SEARCH_DEBUG=1
export SEARCH_DEBUG_LOG_ALL=1
export SEARCH_DEBUG_MAX_LINES=1000000000
# =============================================================================
# Unified trajectory logging (single JSONL; append)
# =============================================================================
export UNIFIED_LOG_ENABLE=1
export UNIFIED_LOG_PATH=./logs/unified_trajectory.jsonl
export UNIFIED_LOG_CLIENT_BATCH_SIZE=200
export UNIFIED_LOG_CLIENT_FLUSH_INTERVAL_S=1.0
export UNIFIED_LOG_WRITER_FLUSH_EVERY_N=2000
export UNIFIED_LOG_WRITER_FLUSH_INTERVAL_S=2.0

export UVLOOP_AUTO=0

n_gpus=4
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Ray 임시 디렉토리 설정 (AF_UNIX 경로 길이 제한 107바이트 대응)
mkdir -p /tmp/ray_$USER
export TMPDIR=/tmp/ray_$USER
export RAY_TMPDIR=/tmp/ray_$USER

# WandB 설정
export WANDB_API_KEY=$WANDB_API_KEY
export WANDB_PROJECT='gspo_phase1_revised'

export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True

#phsae 설정
experiment_name='gspo_phase1_revised'
project_name='gspo_phase1_revised'
# 모델 경로 (필요 시 환경변수/override로 변경)
model_path=Qwen/Qwen2.5-VL-7B-Instruct



# 배치 크기 설정
# - train_batch_size: 원본 프롬프트 수
# - n_agent: 프롬프트당 생성할 응답 수
train_batch_size=16
ppo_mini_batch_size=4
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1
n_agent=2

# 기타 설정
tensor_model_parallel_size=1
max_turns=3

# Retriever 설정
search_url="http://163.239.28.21:5002/search"

# 이미지 경로 설정 (프로젝트 루트 기준 상대 경로)
local_image_root="./search_engine/corpus/img"

#Total max response length
single_turn_max_response_length=2048
total_max_response_length=$((single_turn_max_response_length * max_turns))

# =============================================================================
# 3. 로그 디렉토리 생성
# =============================================================================
mkdir -p ./logs

# =============================================================================
# 4. Ray 메모리 설정
# =============================================================================
export RAY_memory_usage_threshold=0.995

# =============================================================================
# 5. 훈련 실행 (Phase 1: gate reward = 0.1*format + 0.9*ndcg)
# =============================================================================
echo "=========================================="
echo "GSPO Phase 1 Training - Format + NDCG (Gate)"
echo "=========================================="
echo "모델: $model_path"
echo "배치 크기: $train_batch_size × $n_agent = $((train_batch_size * n_agent))"
echo "----------------------------------------"
echo "Reward: if format pass -> 0.1 + 0.9*NDCG else 0"
echo "Generation: phase1 (no frozen generator)"
echo "Unified log: $UNIFIED_LOG_PATH"
echo "=========================================="


# ===========================================================================
ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"


TRAIN_DATA="$HOME/data/rag/slidevqa_train_6667.parquet"
VAL_DATA="$HOME/data/rag/overall_test_crop.parquet"

#TOOL_CONFIG="$CONFIG_PATH/tool_config/search_tool_config.yaml"
#actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \



python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='search_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=32 \
    data.max_prompt_length=256 \
    data.max_response_length=$total_max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$n_agent \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA"  \
    trainer.total_epochs=1 \
    \
    data.image_key=images \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.rollout.max_model_len=16384 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    retriever.url=$search_url \
    actor_rollout_ref.rollout.multi_turn.tool_settings.local_image_root=$local_image_root \
    "$@"

