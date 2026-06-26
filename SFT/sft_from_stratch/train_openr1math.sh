#!/usr/bin/bash
#SBATCH --job-name=sft_openr1math
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --mem=256g
#SBATCH -e sft_openr1math.err
#SBATCH -o sft_openr1math.out
#SBATCH --partition=athena-genai
#SBATCH --account=yw784

NUM_GPUS=4
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch \
    --num_processes $NUM_GPUS \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file ds_config.json \
    main.py \
    --model_path Qwen/Qwen3-4B \
    --dataset openr1math \
    --data_path ../data/math/OpenR1-Math-220k \
    --openr1_subset default \
    --total_size 50000 \
    --epochs 1 \
    --lr 1e-6 \
    --batch_size 1 \
    --max_seq_len 4096 \
    --gradient_accumulation_steps 32 \
    --save_dir outputs/checkpoint/math/checkpoints_openr1math
