# DeepSpeed 原理与使用说明

## 1. DeepSpeed 是什么

DeepSpeed 是微软开源的深度学习训练优化库。它的核心贡献是 **ZeRO（Zero Redundancy Optimizer）** 技术，通过在多张 GPU 之间分片存储模型状态，大幅降低单卡显存占用，使得在有限硬件上训练更大模型成为可能。

## 2. 核心原理：ZeRO 三阶段

在标准数据并行（DDP）中，每张 GPU 都保存完整的模型参数、梯度和优化器状态，显存冗余极大。ZeRO 通过**分片**来消除冗余：

### 训练时 GPU 上存了什么

以一个参数量为 Φ 的模型（FP16 训练 + Adam 优化器）为例：

| 组件 | 大小 | 说明 |
|------|------|------|
| 参数（FP16） | 2Φ bytes | 前向/反向用的参数 |
| 梯度（FP16） | 2Φ bytes | 反向传播产生的梯度 |
| 优化器状态 | 12Φ bytes | Adam 需要：FP32 参数拷贝 (4Φ) + FP32 动量 (4Φ) + FP32 方差 (4Φ) |
| **总计** | **16Φ bytes** | **DDP 中每张卡都存一份完整的 16Φ** |

### ZeRO Stage 1：分片优化器状态

- 每张 GPU 只保存 1/N 的优化器状态
- 参数和梯度仍然每卡完整保留
- 显存节省：12Φ → 12Φ/N（N 为 GPU 数量）

### ZeRO Stage 2：分片优化器状态 + 梯度（本项目使用）

- 每张 GPU 只保存 1/N 的**优化器状态 + 梯度**
- 参数仍然每卡完整保留
- 显存节省：(12Φ + 2Φ) → (12Φ + 2Φ)/N
- 反向传播时梯度 reduce-scatter 后即可释放，不需要全量保存

### ZeRO Stage 3：全量分片

- 参数、梯度、优化器状态全部分片
- 前向/反向时按需通过 all-gather 拉取所需参数
- 显存节省最大，但通信开销也最大

```
显存占用对比（4B 参数模型，2 张 GPU）：

DDP:      每卡 ~64 GB
ZeRO-1:   每卡 ~40 GB  (优化器状态减半)
ZeRO-2:   每卡 ~33 GB  (优化器+梯度减半)  ← 本项目
ZeRO-3:   每卡 ~29 GB  (全部减半)
```

## 3. Offload 技术

ZeRO-Offload 可以将优化器状态和/或参数卸载到 CPU 内存，进一步降低 GPU 显存占用：

```json
"offload_optimizer": {
    "device": "cpu",
    "pin_memory": true
}
```

- `device: "cpu"` — 优化器状态存在 CPU 内存中，参数更新在 CPU 上执行
- `pin_memory: true` — 使用锁页内存加速 CPU↔GPU 数据传输

代价是增加了 CPU 计算和 PCIe 传输开销，适合 GPU 显存紧张但 CPU 内存充裕的场景。

## 4. 本项目的 DeepSpeed 配置

本项目的配置文件 `ds_config.json`：

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

各字段说明：

| 字段 | 值 | 说明 |
|------|----|------|
| `bf16.enabled` | `true` | 使用 BF16 混合精度 |
| `zero_optimization.stage` | `2` | 使用 ZeRO Stage 2 |
| `overlap_comm` | `true` | 梯度通信与反向计算重叠，隐藏通信延迟 |
| `contiguous_gradients` | `true` | 梯度在连续内存中存储，减少内存碎片 |
| `offload_optimizer` | `cpu` | 优化器状态卸载到 CPU |
| `gradient_accumulation_steps` | `"auto"` | 由 Accelerate 自动填入 |
| `train_micro_batch_size_per_gpu` | `"auto"` | 由 Accelerate 自动填入 |
| `gradient_clipping` | `1.0` | 梯度裁剪，防止梯度爆炸 |

## 5. 与 Accelerate 的集成

在本项目中，DeepSpeed 通过 Accelerate 的 `accelerate launch` 命令集成，训练代码本身不需要任何修改：

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

Accelerate 在 `prepare()` 阶段检测到 DeepSpeed 配置后，会自动：
1. 用 `deepspeed.initialize()` 替代 DDP 初始化
2. 用 DeepSpeed 的 FusedAdam 替代标准 AdamW
3. 按 ZeRO Stage 分片模型状态
4. 处理 `"auto"` 字段的自动填充

## 6. 如何选择 ZeRO Stage

| 场景 | 推荐 |
|------|------|
| 单卡能放下完整模型 | 不需要 DeepSpeed，直接用 DDP 或单卡训练 |
| 多卡但显存稍紧（本项目） | **ZeRO Stage 2** — 性价比最高，通信开销小 |
| 单卡/少卡训练大模型（>10B） | ZeRO Stage 3 + Offload |
| 极大模型（>70B） | ZeRO Stage 3 + CPU/NVMe Offload |

## 7. 总结

- **DeepSpeed ZeRO** 的核心思路：用通信换显存，把冗余的模型状态分摊到多张卡上
- **Stage 2** 是最常用的选择，在通信开销和显存节省之间取得良好平衡
- **Offload** 技术可以进一步用 CPU 内存换 GPU 显存
- 通过 Accelerate 集成后，训练代码无需修改，只需在启动命令中指定 DeepSpeed 配置
