#!/usr/bin/bash
#SBATCH --job-name=sft_qwen3_singlegpu
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=256g
#SBATCH -e sft_qwen3_singlegpu.err
#SBATCH -o sft_qwen3_singlegpu.out
#SBATCH --partition=athena-genai
#SBATCH --account=yw784

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py \
    --model_path Qwen/Qwen3-4B \
    --data_path ../data/coding/LeetCodeDataset \
    --epochs 1 \
    --lr 1e-6 \
    --batch_size 1 \
    --max_seq_len 4096 \
    --gradient_accumulation_steps 32 \
    --save_dir checkpoints_singlegpu_CoT_coding
