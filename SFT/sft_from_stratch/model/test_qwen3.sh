#!/usr/bin/bash
#SBATCH --job-name=test_qwen3
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=256g
#SBATCH -e test_qwen3.err
#SBATCH -o test_qwen3.out
#SBATCH --partition=athena-genai
#SBATCH --account=yw784

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
python qwen3.py