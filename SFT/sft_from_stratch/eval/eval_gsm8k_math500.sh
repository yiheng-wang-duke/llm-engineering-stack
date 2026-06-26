#!/usr/bin/bash
#SBATCH --job-name=eval_math
#SBATCH --cpus-per-task=16
#SBATCH --gpus=4
#SBATCH --mem=128g
#SBATCH -e eval_math.err
#SBATCH -o eval_math.out
#SBATCH --partition=athena-genai
#SBATCH --account=yw784


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NUM_GPUS=4
MAX_NEW_TOKENS=2048

# ── GSM8K ──────────────────────────────────────────────────────────

python eval_gsm8k_math500.py \
    --model_path ../outputs/checkpoint/math/checkpoints_openr1math/epoch_1 \
    --benchmark gsm8k \
    --max_new_tokens $MAX_NEW_TOKENS \
    --num_gpus $NUM_GPUS \
    --output ../outputs/eval/gsm8k/Qwen3-4B-sft-openr1math-5k-subset/eval_results.json

python eval_gsm8k_math500.py \
    --model_path Qwen/Qwen3-4B \
    --benchmark gsm8k \
    --max_new_tokens $MAX_NEW_TOKENS \
    --num_gpus $NUM_GPUS \
    --output ../outputs/eval/gsm8k/Qwen3-4B/eval_results.json

# ── MATH-500 ───────────────────────────────────────────────────────
python eval_gsm8k_math500.py \
    --model_path ../outputs/checkpoint/math/checkpoints_openr1math/epoch_1 \
    --benchmark math500 \
    --max_new_tokens $MAX_NEW_TOKENS \
    --num_gpus $NUM_GPUS \
    --output ../outputs/eval/math500/Qwen3-4B-openr1math-5k-subset/eval_results.json

python eval_gsm8k_math500.py \
    --model_path Qwen/Qwen3-4B \
    --benchmark math500 \
    --max_new_tokens $MAX_NEW_TOKENS \
    --num_gpus $NUM_GPUS \
    --output ../outputs/eval/math500/Qwen3-4B/eval_results.json