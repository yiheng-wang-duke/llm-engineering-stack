# Hugging Face Accelerate 原理与使用说明

## 1. Accelerate 是什么

Accelerate 是 Hugging Face 提供的一个轻量级分布式训练库。它的核心理念是：**用最少的代码改动，让同一份 PyTorch 训练脚本无缝运行在单 GPU、多 GPU、多节点、甚至 TPU 上。**

你不需要手写 `torch.distributed` 的初始化、数据分发、梯度同步等繁琐逻辑，Accelerate 帮你全部封装好了。

## 2. 核心原理

### 2.1 Accelerator 对象

Accelerate 的核心是 `Accelerator` 类，它在初始化时自动检测运行环境（单卡/多卡/多节点），并据此配置分布式后端。

```python
from accelerate import Accelerator

accelerator = Accelerator(
    gradient_accumulation_steps=4,  # 梯度累积步数
    mixed_precision="bf16",         # 混合精度训练
)
```

### 2.2 `prepare()` —— 一行代码实现分布式

`prepare()` 是 Accelerate 最关键的 API。它接收 model、optimizer、dataloader、scheduler，自动完成以下操作：

| 对象 | `prepare()` 做了什么 |
|------|---------------------|
| **Model** | 用 `DistributedDataParallel`（DDP）包裹模型，或接入 DeepSpeed / FSDP |
| **Optimizer** | 如果使用 DeepSpeed，替换为 DeepSpeed 优化器 |
| **DataLoader** | 自动对数据做分片（shard），每张卡只看到数据的 1/N |
| **Scheduler** | 与优化器保持同步 |

```python
model, optimizer, dataloader, scheduler = accelerator.prepare(
    model, optimizer, dataloader, scheduler
)
```

### 2.3 梯度累积

使用 `accelerator.accumulate()` 上下文管理器，Accelerate 自动处理：
- 在累积步内**跳过**梯度同步（节省通信开销）
- 在累积完毕时**执行**梯度同步与参数更新

```python
for batch in dataloader:
    with accelerator.accumulate(model):
        loss = model(**batch).loss
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

### 2.4 混合精度训练

设置 `mixed_precision="bf16"` 后，Accelerate 自动用 `torch.autocast` 以 BF16 执行前向计算，同时保持参数主拷贝为 FP32，兼顾速度和精度。

## 3. 在本项目中的使用

本项目的 `SFTTrainer`（见 `trainer/sft_trainer.py`）已经集成了 Accelerate。关键代码如下：

```python
# 初始化
self.accelerator = Accelerator(
    gradient_accumulation_steps=...,
    mixed_precision="bf16",
)

# prepare
self.model, self.optimizer, self.dataloader, self.scheduler = (
    self.accelerator.prepare(model, self.optimizer, dataloader, self.scheduler)
)

# 训练循环
with self.accelerator.accumulate(self.model):
    outputs = self.model(**inputs)
    loss = outputs.loss
    self.accelerator.backward(loss)     # 替代 loss.backward()
    self.optimizer.step()
    self.scheduler.step()
    self.optimizer.zero_grad()

# 保存模型
unwrapped = self.accelerator.unwrap_model(self.model)  # 解包 DDP wrapper
unwrapped.save_pretrained(save_dir)
```

## 4. 启动方式

### 单 GPU

直接用 `python` 运行即可，Accelerate 自动识别为单卡环境：

```bash
python main.py \
    --model_path Qwen/Qwen3-4B \
    --data_path ../data/coding/LeetCodeDataset \
    --batch_size 1 \
    --gradient_accumulation_steps 32
```

### 多 GPU（单节点）

使用 `accelerate launch` 启动，它自动设置 `RANK`、`LOCAL_RANK`、`WORLD_SIZE` 等分布式环境变量：

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

### 多 GPU + DeepSpeed

添加 `--use_deepspeed` 和 DeepSpeed 配置文件即可：

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

## 5. 核心 API 速查

| API | 作用 |
|-----|------|
| `Accelerator()` | 创建加速器，配置分布式策略 |
| `accelerator.prepare(...)` | 包装 model / optimizer / dataloader / scheduler |
| `accelerator.accumulate(model)` | 梯度累积上下文管理器 |
| `accelerator.backward(loss)` | 替代 `loss.backward()`，处理混合精度缩放 |
| `accelerator.unwrap_model(model)` | 解包 DDP / DeepSpeed wrapper，获取原始模型 |
| `accelerator.is_main_process` | 判断是否为主进程（用于日志、保存等） |
| `accelerator.wait_for_everyone()` | 同步所有进程 |

## 6. 总结

Accelerate 的设计哲学是**最小侵入**：

1. 用 `Accelerator()` 替代手动分布式初始化
2. 用 `prepare()` 一行代码包装所有训练组件
3. 用 `accumulate()` + `backward()` 替代原生梯度计算
4. 同一份代码，通过不同的启动命令，即可运行在单卡 / 多卡 / DeepSpeed 等不同环境
