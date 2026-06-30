#!/usr/bin/bash
#SBATCH --job-name=grpo_math
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --mem=256g
#SBATCH -e grpo_math_qwen3_4b.err
#SBATCH -o grpo_math_qwen3_4b.out
#SBATCH --partition=athena-genai
#SBATCH --account=yw784
#
# GRPO RLVR | math only (GSM8K) | Qwen3-4B | FSDP + vLLM
# Verifiable reward: exact-match on the GSM8K "#### answer" (data_source=openai/gsm8k).

set -xeuo pipefail

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export WANDB_API_KEY=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
export HYDRA_FULL_ERROR=1
# This SLURM cluster sets ROCR_VISIBLE_DEVICES (an AMD/ROCm var) alongside
# CUDA_VISIBLE_DEVICES; verl refuses to run when both are set. We use NVIDIA GPUs,
# so the ROCR var is spurious -- drop it before launching.
unset ROCR_VISIBLE_DEVICES

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
DATA_ROOT=${DATA_ROOT:-$HOME/data}
TRAIN_FILES=${TRAIN_FILES:-"${DATA_ROOT}/gsm8k/train.parquet"}
VAL_FILES=${VAL_FILES:-"${DATA_ROOT}/gsm8k/test.parquet"}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}      
ROLLOUT_TP=${ROLLOUT_TP:-1}         

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}

ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ROLLOUT_N=${ROLLOUT_N:-8}                  # GRPO group size (responses per prompt)
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.75}

PARAM_OFFLOAD=${PARAM_OFFLOAD:-False}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-False}

PROJECT_NAME=${PROJECT_NAME:-grpo_rlvr_math}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_gsm8k}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-10}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
LOGGER=${LOGGER:-"['console','wandb']"}
# ---- end user-adjustable ----

# Default 'naive' reward manager routes openai/gsm8k -> math exact-match reward.
# No code execution / sandbox needed for math-only.
RAY_TEMP_DIR=${RAY_TEMP_DIR:-$HOME/.cache/ray_tmp}
python3 -m verl.trainer.main_ppo \
    +ray_kwargs.ray_init._temp_dir="${RAY_TEMP_DIR}" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger="${LOGGER}" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    "$@"
