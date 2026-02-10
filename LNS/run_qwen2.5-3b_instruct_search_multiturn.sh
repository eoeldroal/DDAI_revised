set -x

[ -f .env ] && echo ">>> .env 파일 로드 중..." && export $(grep -v '^#' .env | xargs)

export PYTHONNOUSERSITE=1
export VERL_PRETTY_ROLLOUT_LOG=1
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ray_tmp="/tmp/ray_$USER"
mkdir -p "$ray_tmp"
export TMPDIR="$ray_tmp"
export RAY_TMPDIR="$ray_tmp"
export WANDB_PROJECT="${WANDB_PROJECT:-gspo_phase1_revised}"
export WEAVE_LOG_LEVEL=ERROR

mkdir -p ./logs
ulimit -n 65535

PROJECT_DIR="$(pwd)"
RUN_TS=$(date +%m%d_%H%M)

TRAIN_DATA="$PROJECT_DIR/data/rag/slidevqa_train_6667.parquet"
VAL_DATA="$PROJECT_DIR/data/rag/overall_test_crop.parquet"
python3 -m verl.trainer.main_ppo \
    --config-path="$PROJECT_DIR/LNS" \
    --config-name='search_multiturn_grpo' \
    custom_reward_function.path="$PROJECT_DIR/verl/utils/reward_score/format_ndcg_reward.py" \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.val_before_train=False \
    trainer.rollout_data_dir=./logs/rollout_data_${RUN_TS} \
    trainer.validation_data_dir=./logs/val_data_${RUN_TS} \
    trainer.project_name=gspo_phase1_revised \
    trainer.experiment_name=gspo_phase1_revised \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50000000000000 \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA"  \
    trainer.total_epochs=1 \
    "$@"
