# 两阶段渐进式微调指南

## 策略概述

两阶段渐进式微调是一种更稳定、更有效的微调方法：

1. **第一阶段**：加载模型A权重 → 冻结backbone → 只训练分类头（head + margin_head）
2. **第二阶段**：如果第一阶段有效（F1从0.776→0.80+），解冻部分backbone，用小学习率继续训练

## 为什么有效？

1. **稳定性**：先让分类头适应新数据，避免破坏预训练的特征提取能力
2. **效率**：第一阶段通常只需要5-10个epoch就能看到效果
3. **可控性**：如果第一阶段效果不好，可以调整策略，避免浪费计算资源

## 使用方法

### 快速开始

使用提供的脚本：

```bash
bash train_finetune_progressive.sh
```

### 手动配置

```bash
python -m ocec train \
    --data_root /path/to/dataset_B \
    --output_dir runs/finetune_progressive \
    --pretrain runs/ocec_6Whq+24WPublic/v8/ocec_best_epoch*.pt \
    --epochs 110 \
    --batch_size 1024 \
    --lr 1e-4 \
    --freeze_backbone \
    --unfreeze_backbone_epoch 11 \
    --margin_method cosface \
    --use_amp
```

### 参数说明

- `--freeze_backbone`: 冻结backbone，只训练head和margin_head
- `--unfreeze_backbone_epoch 11`: 在第11个epoch解冻部分backbone（第一阶段10个epoch后）
- `--lr 1e-4`: 第一阶段学习率（head可以稍大）
- 第二阶段学习率会自动调整为 `lr * 0.05`（更小，适合backbone）

## 训练流程

### 第一阶段（Epoch 1-10）

- **状态**：Backbone冻结，只训练head和margin_head
- **学习率**：1e-4（可以稍大，因为只训练分类层）
- **目标**：验证集F1从0.776提升到0.80+

### 第二阶段（Epoch 11+）

- **触发条件**：第一阶段F1 >= 0.80
- **状态**：解冻50%的backbone参数（从后往前）
- **学习率**：自动调整为 `lr * 0.05`（5e-6，更小）
- **目标**：进一步提升F1到0.82-0.85+

## 监控指标

### 第一阶段检查点

在Epoch 10结束后，检查：
- **验证集F1**：是否达到0.80+
- **训练Loss**：是否在下降
- **验证Loss**：是否低于训练Loss（避免过拟合）

### 第二阶段监控

- **验证集F1**：应该继续提升
- **Fisher Ratio**：类间分离度应该改善
- **学习率**：观察是否合理调整

## 预期效果

| 阶段 | 验证集F1 | 说明 |
|------|----------|------|
| 初始（从头训练） | 0.776 | 基线 |
| 第一阶段后 | 0.80-0.82 | 只训练head |
| 第二阶段后 | 0.82-0.85+ | 解冻部分backbone |

## 如果第一阶段效果不理想

### 情况1：F1 < 0.80

**可能原因**：
- 学习率太大或太小
- 数据集B分布差异太大
- 预训练模型不匹配

**解决方案**：
1. 调整学习率：尝试 `5e-5` 或 `2e-4`
2. 增加第一阶段epoch：从10增加到15-20
3. 检查预训练模型是否是最佳的

### 情况2：F1达到0.80但第二阶段不触发

**检查**：
- 日志中是否有 "Stage 1 F1=... >= 0.80" 的消息
- 验证集是否正常评估

**解决方案**：
- 手动检查验证集F1
- 如果达到0.80，可以手动继续训练（不使用 `--unfreeze_backbone_epoch`）

## 高级配置

### 调整解冻比例

如果需要修改解冻的backbone比例，需要修改代码中的 `unfreeze_ratio` 参数：

```python
# 在 ocec/pipeline.py 的 _unfreeze_backbone_partially 调用处
_unfreeze_backbone_partially(model, unfreeze_ratio=0.3)  # 只解冻30%
```

### 完全解冻backbone

如果第二阶段效果很好，可以完全解冻：

```python
_unfreeze_backbone_fully(model)
```

### 分层学习率

如果需要更精细的控制，可以实现分层学习率（需要修改代码）：

```python
backbone_params = []
head_params = []
for name, param in model.named_parameters():
    if 'head' in name or 'margin_head' in name:
        head_params.append(param)
    else:
        backbone_params.append(param)

optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': 5e-6},  # Backbone: 很小
    {'params': head_params, 'lr': 1e-4},      # Head: 较大
], weight_decay=config.weight_decay)
```

## 与全量微调对比

| 方法 | 第一阶段F1 | 最终F1 | 稳定性 | 训练时间 |
|------|------------|--------|--------|---------|
| 全量微调 | - | 0.80-0.85 | 中等 | 200 epochs |
| 渐进式微调 | 0.80-0.82 | 0.82-0.87 | 高 | 110 epochs |

## 总结

✅ **推荐使用渐进式微调**：
- 更稳定
- 更快看到效果
- 更好的最终性能

✅ **关键参数**：
- `--freeze_backbone`: 冻结backbone
- `--unfreeze_backbone_epoch 11`: 在epoch 11解冻
- `--lr 1e-4`: 第一阶段学习率

✅ **预期效果**：
- 第一阶段：F1 0.80-0.82
- 第二阶段：F1 0.82-0.85+

