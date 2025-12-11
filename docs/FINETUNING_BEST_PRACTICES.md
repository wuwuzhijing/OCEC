# 微调最佳实践指南

## 场景分析

### 当前情况
- **模型A**：数据集A（25W public + 5W private）训练，F1=0.985
- **数据集B**：40W private，分布差异大，直接训练F1=0.776
- **目标**：在模型A基础上微调，提升数据集B上的效果

### 为什么微调有效？

1. **迁移学习优势**：
   - 模型A已经学习到通用的特征表示
   - 这些特征对数据集B仍然有用
   - 只需要适应数据集B的特定分布

2. **数据分布差异**：
   - 数据集B分布差异大，从头训练需要更多数据和时间
   - 微调可以快速适应新分布

## 微调策略

### 策略1：全量微调（推荐起点）

**适用场景**：数据集B与数据集A有一定相似性

**配置**：
```bash
--pretrain <模型A的checkpoint>
--lr 5e-5          # 原始学习率的 1/2
--warmup_epochs 10  # 较短的warmup
--epochs 200        # 足够的训练轮数
```

**优点**：
- 简单直接
- 所有层都能适应新数据
- 通常效果最好

**缺点**：
- 如果分布差异太大，可能过拟合

### 策略2：分层学习率（如果效果不理想）

**适用场景**：数据集B分布差异很大

**实现方式**（需要修改代码）：
```python
# 在 train_pipeline 中，创建优化器时
backbone_params = []
head_params = []
margin_params = []

for name, param in model.named_parameters():
    if 'margin_head' in name:
        margin_params.append(param)
    elif 'head' in name:
        head_params.append(param)
    else:
        backbone_params.append(param)

optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': 1e-5},      # Backbone: 更小的学习率
    {'params': head_params, 'lr': 5e-5},          # Head: 中等学习率
    {'params': margin_params, 'lr': 1e-4},        # Margin head: 较大的学习率
], weight_decay=config.weight_decay)
```

**优点**：
- Backbone 保持稳定，只做小幅调整
- Head 和 Margin head 快速适应新数据

### 策略3：冻结部分层（极端情况）

**适用场景**：数据集B分布差异极大，或数据量很小

**实现方式**（需要修改代码）：
```python
# 冻结 backbone，只训练 head
for name, param in model.named_parameters():
    if 'head' not in name and 'margin_head' not in name:
        param.requires_grad = False
```

**优点**：
- 防止过拟合
- 快速适应新数据

**缺点**：
- 可能无法充分利用预训练权重
- 效果可能不如全量微调

## 推荐配置

### 基础配置（推荐起点）

```bash
python -m ocec train \
    --data_root /path/to/dataset_B \
    --output_dir runs/finetune_B \
    --pretrain runs/ocec_hq/v8/ocec_best_epoch*.pt \
    --epochs 200 \
    --batch_size 1024 \
    --lr 5e-5 \
    --weight_decay 1e-4 \
    --warmup_epochs 10 \
    --use_amp \
    --margin_method cosface
```

### 如果效果不理想，尝试以下调整

#### 调整1：降低学习率
```bash
--lr 2e-5  # 更保守的学习率
```

#### 调整2：增加warmup
```bash
--warmup_epochs 20  # 更长的warmup
```

#### 调整3：增加正则化
```bash
--dropout 0.4       # 增加dropout
--weight_decay 2e-4 # 增加weight decay
```

#### 调整4：减少训练轮数
```bash
--epochs 100  # 如果很快收敛，可以减少epoch
```

## 微调流程

### 步骤1：准备预训练模型

找到模型A的最佳checkpoint：
```bash
# 通常是最新的 best checkpoint
ls -t runs/ocec_hq/v8/ocec_best_epoch*.pt | head -1
```

### 步骤2：检查模型配置一致性

确保微调时的配置与预训练模型一致：
- `base_channels`: 32
- `num_blocks`: 4
- `arch_variant`: baseline
- `margin_method`: cosface
- `image_size`: 32x64

### 步骤3：运行微调

```bash
bash train_finetune.sh
```

或手动运行：
```bash
python -m ocec train \
    --data_root /path/to/dataset_B \
    --output_dir runs/finetune_B \
    --pretrain runs/ocec_hq/v8/ocec_best_epoch0055_f1_0.7759.pt \
    --epochs 200 \
    --batch_size 1024 \
    --lr 5e-5 \
    --warmup_epochs 10 \
    --use_amp \
    --margin_method cosface
```

### 步骤4：监控训练

1. **观察验证集 F1**：
   - 应该比从头训练（0.776）更高
   - 目标：0.80-0.85+

2. **观察 Loss 曲线**：
   - 应该从较低的值开始
   - 下降应该更平滑

3. **观察 Fisher Ratio**：
   - 应该比从头训练更高
   - 说明类间分离更好

## 预期效果对比

| 方法 | 验证集 F1 | 收敛速度 | 稳定性 |
|------|-----------|---------|--------|
| 从头训练 | 0.776 | 慢（需要更多epoch） | 中等 |
| 微调（lr=5e-5） | 0.80-0.85+ | 快（50-100 epochs） | 高 |
| 微调（lr=2e-5） | 0.82-0.87+ | 中等（100-150 epochs） | 很高 |

## 常见问题

### Q1: 微调后效果不如从头训练？

**可能原因**：
1. 学习率太大，破坏了预训练权重
2. 数据集B分布差异太大
3. 预训练模型与数据集B不匹配

**解决方案**：
1. 降低学习率到 `2e-5` 或 `1e-5`
2. 增加warmup到 20 epochs
3. 检查预训练模型是否是最佳的

### Q2: 微调后过拟合？

**解决方案**：
1. 增加dropout：`--dropout 0.4`
2. 增加weight decay：`--weight_decay 2e-4`
3. 减少训练轮数：`--epochs 100`

### Q3: 微调收敛太慢？

**解决方案**：
1. 适当提高学习率：`--lr 1e-4`
2. 减少warmup：`--warmup_epochs 5`
3. 检查数据加载是否正常

### Q4: 如何选择最佳checkpoint？

**推荐**：
- 使用验证集F1最高的checkpoint
- 通常文件名包含 `ocec_best_epoch*_f1_*.pt`
- 选择F1最高的那个

## 总结

✅ **代码已支持微调**：使用 `--pretrain` 参数

✅ **推荐配置**：
- 学习率：5e-5（原始学习率的 1/2）
- Warmup：10 epochs
- Epochs：200
- 其他参数保持与原始训练一致

✅ **预期效果**：F1 从 0.776 提升到 0.80-0.85+

✅ **如果效果不理想**：
- 降低学习率到 2e-5
- 增加warmup到 20 epochs
- 增加正则化（dropout, weight_decay）

