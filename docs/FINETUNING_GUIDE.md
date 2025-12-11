# 微调指南：在模型A基础上微调数据集B

## 场景描述

- **模型A**：在数据集A（25W public + 5W private）上训练，F1=0.985
- **数据集B**：40W private，分布差异大，直接训练F1=0.776
- **目标**：在模型A基础上微调，提升数据集B上的效果

## 微调策略

### 1. 使用预训练权重（推荐）✅

代码已支持 `--pretrain` 参数，可以加载模型A的权重进行微调：

```bash
python -m ocec train \
    --data_root /path/to/dataset_B \
    --output_dir runs/finetune_B \
    --pretrain runs/ocec_hq/v8/ocec_best_epoch*.pt \
    --epochs 200 \
    --lr 5e-5 \
    --warmup_epochs 10 \
    --use_amp \
    --margin_method cosface
```

**关键参数**：
- `--pretrain`：模型A的最佳checkpoint路径
- `--lr 5e-5`：使用较小的学习率（通常是初始训练的 1/2 到 1/10）
- `--warmup_epochs 10`：较短的warmup（因为模型已经训练过）
- `--epochs 200`：较少的epoch（微调通常收敛更快）

### 2. 微调最佳实践

#### 2.1 学习率设置

**推荐策略**：
- **初始学习率**：`5e-5` 到 `1e-4`（原始训练学习率的 1/2 到 1/10）
- **分层学习率**（如果支持）：
  - Backbone（特征提取层）：`1e-5` 到 `5e-5`
  - Head（分类层）：`5e-5` 到 `1e-4`
  - Margin head（如果使用）：`1e-4` 到 `2e-4`

**当前代码限制**：所有层使用相同学习率。如果需要分层学习率，可以修改优化器配置。

#### 2.2 Warmup 设置

- **推荐**：5-10 epochs
- **原因**：模型已经训练过，不需要太长的warmup

#### 2.3 训练轮数

- **推荐**：100-200 epochs
- **原因**：微调通常收敛更快，但需要足够的时间适应新数据分布

#### 2.4 数据增强

- **保持**：使用与原始训练相同的数据增强策略
- **原因**：帮助模型适应数据集B的分布差异

#### 2.5 正则化

- **Dropout**：保持 0.3（或略微增加）
- **Weight Decay**：保持 1e-4（或略微增加）
- **原因**：防止过拟合到数据集B

### 3. 微调脚本

已创建 `train_finetune.sh`，包含推荐的微调配置。

**使用方法**：
1. 修改脚本中的路径：
   - `PRETRAIN_MODEL`：模型A的最佳checkpoint路径
   - `DATA_ROOT_B`：数据集B的路径
   - `OUTPUT_DIR`：微调输出目录

2. 运行脚本：
```bash
bash train_finetune.sh
```

### 4. 预期效果

#### 4.1 微调 vs 从头训练

| 方法 | 预期 F1 | 收敛速度 | 稳定性 |
|------|---------|---------|--------|
| 从头训练 | 0.776 | 慢 | 中等 |
| 微调（模型A） | 0.80-0.85+ | 快 | 高 |

#### 4.2 微调的优势

1. **更快的收敛**：模型已经学习到通用特征
2. **更好的性能**：通常比从头训练高 3-5 个百分点
3. **更稳定**：预训练权重提供了良好的初始化

### 5. 监控指标

微调过程中重点关注：

1. **验证集 F1**：应该比从头训练（0.776）更高
2. **训练/验证 Loss 差距**：不应该过大（避免过拟合）
3. **Fisher Ratio**：类间分离度应该持续改善
4. **学习率变化**：观察学习率调度是否合理

### 6. 如果效果不理想

#### 6.1 学习率调整

如果微调效果不佳，尝试：
- **降低学习率**：`2e-5` 或 `1e-5`
- **增加warmup**：15-20 epochs
- **使用更小的学习率衰减**：`factor=0.3`（需要修改代码）

#### 6.2 冻结部分层

如果数据集B分布差异很大，可以考虑：
- **冻结 Backbone**：只训练 Head 和 Margin head
- **部分解冻**：先冻结，训练几个epoch后解冻

**实现方式**（需要修改代码）：
```python
# 冻结 backbone
for name, param in model.named_parameters():
    if 'head' not in name and 'margin_head' not in name:
        param.requires_grad = False
```

#### 6.3 数据混合

如果数据集A和B可以混合：
- **混合训练**：使用数据集A+B混合训练
- **两阶段训练**：先在A上训练，再在B上微调

### 7. 完整微调命令示例

```bash
python -m ocec train \
    --data_root /ssddisk/guochuang/ocec/parquet_hq \
    --output_dir runs/ocec_hq_finetune \
    --pretrain runs/ocec_hq/v8/ocec_best_epoch0055_f1_0.7759.pt \
    --epochs 200 \
    --batch_size 1024 \
    --num_workers 16 \
    --train_ratio 0.8 \
    --val_ratio 0.2 \
    --image_size 32x64 \
    --base_channels 32 \
    --num_blocks 4 \
    --arch_variant baseline \
    --margin_method cosface \
    --lr 5e-5 \
    --weight_decay 1e-4 \
    --seed 42 \
    --device auto \
    --warmup_epochs 10 \
    --use_amp
```

### 8. 注意事项

1. **模型架构一致性**：
   - 确保微调时的模型架构与预训练模型一致
   - `base_channels`、`num_blocks`、`arch_variant` 等参数应该相同

2. **Margin Method 一致性**：
   - 如果模型A使用 `cosface`，微调时也应该使用 `cosface`
   - 如果不同，需要特殊处理（可能需要重新初始化 margin_head）

3. **数据分布差异**：
   - 如果数据集B分布差异很大，可能需要更小的学习率和更多的epoch
   - 考虑使用数据混合或两阶段训练

4. **Checkpoint 选择**：
   - 使用模型A的**最佳checkpoint**（通常是 `ocec_best_epoch*.pt`）
   - 不要使用中间checkpoint，可能效果不佳

### 9. 验证微调效果

微调完成后，对比：

1. **验证集 F1**：
   - 从头训练：0.776
   - 微调后：应该 > 0.80（目标 0.82-0.85）

2. **训练曲线**：
   - 微调应该更快收敛
   - Loss 应该从较低的值开始下降

3. **类间分离度**：
   - Fisher Ratio 应该比从头训练更高
   - 说明模型更好地学习了数据集B的特征

## 总结

✅ **代码已支持微调**：使用 `--pretrain` 参数即可

✅ **推荐配置**：
- 学习率：5e-5（原始学习率的 1/2）
- Warmup：10 epochs
- Epochs：200
- 其他参数保持与原始训练一致

✅ **预期效果**：F1 从 0.776 提升到 0.80-0.85+

