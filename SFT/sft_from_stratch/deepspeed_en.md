# DeepSpeed: Principles & Usage Guide

## 1. What is DeepSpeed

DeepSpeed is an open-source deep learning optimization library from Microsoft. Its core contribution is the **ZeRO (Zero Redundancy Optimizer)** technology, which partitions model states across GPUs to dramatically reduce per-GPU memory usage, enabling training of larger models on limited hardware.

## 2. Core Principle: ZeRO's Three Stages

In standard Data Parallel (DDP), every GPU holds a full copy of model parameters, gradients, and optimizer states — massive memory redundancy. ZeRO eliminates this through **partitioning**:

### What's Stored on Each GPU During Training

For a model with Φ parameters (FP16 training + Adam optimizer):

| Component | Size | Description |
|-----------|------|-------------|
| Parameters (FP16) | 2Φ bytes | Parameters used in forward/backward |
| Gradients (FP16) | 2Φ bytes | Gradients from backpropagation |
| Optimizer States | 12Φ bytes | Adam requires: FP32 param copy (4Φ) + FP32 momentum (4Φ) + FP32 variance (4Φ) |
| **Total** | **16Φ bytes** | **In DDP, every GPU stores a full 16Φ** |

### ZeRO Stage 1: Partition Optimizer States

- Each GPU only stores 1/N of optimizer states
- Parameters and gradients are still fully replicated
- Memory saving: 12Φ → 12Φ/N (N = number of GPUs)

### ZeRO Stage 2: Partition Optimizer States + Gradients (Used in This Project)

- Each GPU only stores 1/N of **optimizer states + gradients**
- Parameters are still fully replicated
- Memory saving: (12Φ + 2Φ) → (12Φ + 2Φ)/N
- Gradients are released after reduce-scatter during backprop, no need to keep full copies

### ZeRO Stage 3: Full Partitioning

- Parameters, gradients, and optimizer states are all partitioned
- Parameters are gathered on-demand via all-gather during forward/backward
- Maximum memory saving, but highest communication overhead

```
Memory comparison (4B parameter model, 2 GPUs):

DDP:      ~64 GB per GPU
ZeRO-1:   ~40 GB per GPU  (optimizer states halved)
ZeRO-2:   ~33 GB per GPU  (optimizer + gradients halved)  ← This project
ZeRO-3:   ~29 GB per GPU  (everything halved)
```

## 3. Offload Technique

ZeRO-Offload can offload optimizer states and/or parameters to CPU memory, further reducing GPU memory usage:

```json
"offload_optimizer": {
    "device": "cpu",
    "pin_memory": true
}
```

- `device: "cpu"` — Optimizer states are stored in CPU memory; parameter updates run on CPU
- `pin_memory: true` — Uses pinned memory to accelerate CPU↔GPU data transfer

The trade-off is increased CPU computation and PCIe transfer overhead. Best suited for scenarios with tight GPU memory but ample CPU memory.

## 4. DeepSpeed Configuration in This Project

The config file `ds_config.json`:

```json
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  },
  "gradient_accumulation_steps": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_clipping": 1.0
}
```

Field descriptions:

| Field | Value | Description |
|-------|-------|-------------|
| `bf16.enabled` | `true` | Use BF16 mixed precision |
| `zero_optimization.stage` | `2` | Use ZeRO Stage 2 |
| `overlap_comm` | `true` | Overlap gradient communication with backward computation to hide latency |
| `contiguous_gradients` | `true` | Store gradients in contiguous memory to reduce fragmentation |
| `offload_optimizer` | `cpu` | Offload optimizer states to CPU |
| `gradient_accumulation_steps` | `"auto"` | Auto-filled by Accelerate |
| `train_micro_batch_size_per_gpu` | `"auto"` | Auto-filled by Accelerate |
| `gradient_clipping` | `1.0` | Gradient clipping to prevent gradient explosion |

## 5. Integration with Accelerate

In this project, DeepSpeed is integrated via Accelerate's `accelerate launch` command — the training code itself requires no modifications:

```bash
accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file ds_config.json \
    main.py \
    --model_path Qwen/Qwen3-4B \
    --data_path ../data/coding/LeetCodeDataset \
    --batch_size 1 \
    --gradient_accumulation_steps 32
```

When Accelerate detects the DeepSpeed configuration during `prepare()`, it automatically:
1. Replaces DDP initialization with `deepspeed.initialize()`
2. Substitutes standard AdamW with DeepSpeed's FusedAdam
3. Partitions model states according to the ZeRO Stage
4. Fills in `"auto"` fields with actual values

## 6. How to Choose a ZeRO Stage

| Scenario | Recommendation |
|----------|---------------|
| Model fits on a single GPU | No DeepSpeed needed — use DDP or single-GPU training |
| Multi-GPU, slightly tight memory (this project) | **ZeRO Stage 2** — best cost-efficiency, low communication overhead |
| Large models (>10B) on few GPUs | ZeRO Stage 3 + Offload |
| Very large models (>70B) | ZeRO Stage 3 + CPU/NVMe Offload |

## 7. Summary

- **DeepSpeed ZeRO** core idea: trade communication for memory — distribute redundant model states across GPUs
- **Stage 2** is the most commonly used choice, achieving a good balance between communication overhead and memory savings
- **Offload** techniques further trade CPU memory for GPU memory
- With Accelerate integration, no training code changes are needed — just specify the DeepSpeed config in the launch command
