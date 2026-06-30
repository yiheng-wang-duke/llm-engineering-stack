#!/usr/bin/bash
#SBATCH --job-name=eval_math
#SBATCH --cpus-per-task=16
#SBATCH --gpus=2
#SBATCH --mem=128g
#SBATCH -e eval_math.err
#SBATCH -o eval_math.out
#SBATCH --partition=athena-genai
#SBATCH --account=yw784

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NUM_GPUS=2
MAX_NEW_TOKENS=2048

python eval_gsm8k_math500.py \
    --model_path /home/yw784/Projects/RL/verl/qwen3_4b_gsm8k_grpo_hf \
    --benchmark gsm8k \
    --max_new_tokens $MAX_NEW_TOKENS \
    --num_gpus $NUM_GPUS \
    --output ../outputs/eval/gsm8k/Qwen3-4B-rl-gsmk/eval_results.json
