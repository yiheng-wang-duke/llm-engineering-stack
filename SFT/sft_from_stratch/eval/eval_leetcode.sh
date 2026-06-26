#!/usr/bin/bash
#SBATCH --job-name=eval_leetcode
#SBATCH --cpus-per-task=16
#SBATCH --gpus=2
#SBATCH --mem=128g
#SBATCH -e eval_leetcode.err
#SBATCH -o eval_leetcode.out
#SBATCH --partition=athena-genai
#SBATCH --account=yw784


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NUM_GPUS=2
MAX_NEW_TOKENS=9192

python eval_leetcode.py \
    --model_path ../outputs/checkpoint/coding/checkpoints_multigpu/epoch_1 \
    --data_path ../../data/coding/LeetCodeDataset \
    --n_samples 5 \
    --k 1 5 \
    --max_new_tokens $MAX_NEW_TOKENS \
    --num_gpus $NUM_GPUS \
    --output ../outputs/eval/LeetCodeDataset/Qwen3-4B-sft-multigpu/eval_results.json

# python eval_leetcode.py \
#     --model_path ../outputs/checkpoint/coding/checkpoints_multigpu_CoT_coding/epoch_1 \
#     --data_path ../../data/coding/LeetCodeDataset \
#     --n_samples 5 \
#     --k 1 5 \
#     --max_new_tokens $MAX_NEW_TOKENS \
#     --num_gpus $NUM_GPUS \
#     --output ../outputs/eval/LeetCodeDataset/Qwen3-4B-sft-multigpu-CoT-coding/eval_results.json

# python eval_leetcode.py \
#     --model_path Qwen/Qwen3-4B \
#     --data_path ../../data/coding/LeetCodeDataset \
#     --n_samples 5 \
#     --k 1 5 \
#     --max_new_tokens $MAX_NEW_TOKENS \
#     --num_gpus $NUM_GPUS \
#     --output ../outputs/eval/LeetCodeDataset/Qwen3-4B/eval_results.json
