#!/usr/bin/bash
#SBATCH --job-name=sft_qwen3_multigpu
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --mem=256g
#SBATCH -e sft_qwen3_multigpu.err
#SBATCH -o sft_qwen3_multigpu.out
#SBATCH --partition=athena-genai
#SBATCH --account=yw784

NUM_GPUS=4  # 改成你的卡数
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /home/yw784/miniconda3/etc/profile.d/conda.sh
conda activate llm_sft
export PATH=/home/yw784/miniconda3/envs/llm_sft/bin:$PATH
echo "Python: $(which python)"
echo "Accelerate: $(which accelerate)"
echo "DeepSpeed: $(python -c 'import deepspeed; print(deepspeed.__version__)' 2>&1)"

accelerate launch \
    --num_processes $NUM_GPUS \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file ds_config.json \
    main.py \
    --model_path Qwen/Qwen3-4B \
    --data_path ../data/coding/LeetCodeDataset \
    --epochs 1 \
    --total_size 2641 \
    --lr 1e-6 \
    --batch_size 1 \
    --max_seq_len 4096 \
    --gradient_accumulation_steps 32 \
    --save_dir checkpoints_multigpu
