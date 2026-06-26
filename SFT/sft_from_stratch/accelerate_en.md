# Hugging Face Accelerate: Principles & Usage Guide

## 1. What is Accelerate

Accelerate is a lightweight distributed training library from Hugging Face. Its core idea: **with minimal code changes, run the same PyTorch training script seamlessly on single GPU, multi-GPU, multi-node, or even TPU.**

You don't need to manually write `torch.distributed` initialization, data sharding, or gradient synchronization — Accelerate handles it all.

## 2. Core Principles

### 2.1 The Accelerator Object

The core of Accelerate is the `Accelerator` class. On initialization, it auto-detects the runtime environment (single/multi-GPU/multi-node) and configures the distributed backend accordingly.

```python
from accelerate import Accelerator

accelerator = Accelerator(
    gradient_accumulation_steps=4,  # gradient accumulation steps
    mixed_precision="bf16",         # mixed precision training
)
```

### 2.2 `prepare()` — Distributed in One Line

`prepare()` is the most critical API in Accelerate. It takes model, optimizer, dataloader, and scheduler, and automatically performs:

| Object | What `prepare()` does |
|--------|----------------------|
| **Model** | Wraps with `DistributedDataParallel` (DDP), or integrates with DeepSpeed / FSDP |
| **Optimizer** | Replaces with DeepSpeed optimizer if using DeepSpeed |
| **DataLoader** | Auto-shards data so each GPU sees only 1/N of the dataset |
| **Scheduler** | Keeps in sync with the optimizer |

```python
model, optimizer, dataloader, scheduler = accelerator.prepare(
    model, optimizer, dataloader, scheduler
)
```

### 2.3 Gradient Accumulation

Using the `accelerator.accumulate()` context manager, Accelerate automatically:
- **Skips** gradient synchronization during accumulation steps (saves communication overhead)
- **Performs** gradient sync and parameter update at the accumulation boundary

```python
for batch in dataloader:
    with accelerator.accumulate(model):
        loss = model(**batch).loss
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

### 2.4 Mixed Precision Training

With `mixed_precision="bf16"`, Accelerate automatically uses `torch.autocast` to run forward passes in BF16 while keeping master parameter copies in FP32, balancing speed and accuracy.

## 3. Usage in This Project

The `SFTTrainer` in this project (see `trainer/sft_trainer.py`) has Accelerate fully integrated. Key code:

```python
# Initialization
self.accelerator = Accelerator(
    gradient_accumulation_steps=...,
    mixed_precision="bf16",
)

# Prepare
self.model, self.optimizer, self.dataloader, self.scheduler = (
    self.accelerator.prepare(model, self.optimizer, dataloader, self.scheduler)
)

# Training loop
with self.accelerator.accumulate(self.model):
    outputs = self.model(**inputs)
    loss = outputs.loss
    self.accelerator.backward(loss)     # replaces loss.backward()
    self.optimizer.step()
    self.scheduler.step()
    self.optimizer.zero_grad()

# Save model
unwrapped = self.accelerator.unwrap_model(self.model)  # unwrap DDP wrapper
unwrapped.save_pretrained(save_dir)
```

## 4. Launch Methods

### Single GPU

Run directly with `python` — Accelerate auto-detects single-GPU mode:

```bash
python main.py \
    --model_path Qwen/Qwen3-4B \
    --data_path ../data/coding/LeetCodeDataset \
    --batch_size 1 \
    --gradient_accumulation_steps 32
```

### Multi-GPU (Single Node)

Use `accelerate launch`, which automatically sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, and other distributed environment variables:

```bash
accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    main.py \
    --model_path Qwen/Qwen3-4B \
    --data_path ../data/coding/LeetCodeDataset \
    --batch_size 1 \
    --gradient_accumulation_steps 32
```

### Multi-GPU + DeepSpeed

Add `--use_deepspeed` and a DeepSpeed config file:

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

## 5. API Quick Reference

| API | Purpose |
|-----|---------|
| `Accelerator()` | Create accelerator, configure distributed strategy |
| `accelerator.prepare(...)` | Wrap model / optimizer / dataloader / scheduler |
| `accelerator.accumulate(model)` | Gradient accumulation context manager |
| `accelerator.backward(loss)` | Replaces `loss.backward()`, handles mixed-precision scaling |
| `accelerator.unwrap_model(model)` | Unwrap DDP / DeepSpeed wrapper to get the raw model |
| `accelerator.is_main_process` | Check if current process is main (for logging, saving) |
| `accelerator.wait_for_everyone()` | Synchronize all processes |

## 6. Summary

Accelerate's design philosophy is **minimal intrusion**:

1. Replace manual distributed init with `Accelerator()`
2. Wrap all training components with a single `prepare()` call
3. Use `accumulate()` + `backward()` instead of native gradient computation
4. The same code runs on single GPU / multi-GPU / DeepSpeed by simply changing the launch command
