# ConvNext 架构训练指南

## 概述

ConvNext 是一个现代化的卷积神经网络架构，相比 baseline 架构有以下特点：

1. **LayerNorm 替代 BatchNorm**：对 batch size 不敏感，训练更稳定
2. **7x7 Depthwise Convolution**：更大的感受野，更好的特征提取
3. **Channel MLP**：使用 4x expansion 的 MLP 进行通道混合
4. **GELU 激活函数**：更平滑的激活函数
5. **Transformer Head**：自动使用 transformer 风格的分类头

## 推荐的参数调整

### 1. **输出目录** ⭐ 必须更改
```bash
--output_dir runs/ocec_hq_convnext
```
**原因**：需要与 baseline 结果分开保存，便于对比

### 2. **Base Channels** ⭐ 建议调整
```bash
--base_channels 48  # 从 32 增加到 48
```
**原因**：
- ConvNext 的 channel MLP 需要更多参数来发挥优势
- 48 通道在保持模型大小的同时提供更好的表达能力
- 如果显存充足，可以尝试 64

### 3. **学习率** ⚠️ 可选调整
```bash
--lr 1.2e-4  # 从 1e-4 略微增加到 1.2e-4
```
**原因**：
- LayerNorm 比 BatchNorm 更稳定，可以承受稍高的学习率
- 1.2e-4 是一个保守的增加，如果训练不稳定可以回到 1e-4
- 也可以尝试 1.5e-4，但需要密切观察训练曲线

### 4. **Warmup Epochs** ⚠️ 建议调整
```bash
--warmup_epochs 20  # 从 15 增加到 20
```
**原因**：
- ConvNext 的 LayerNorm 和 channel MLP 在训练初期需要更长的适应期
- 更长的 warmup 有助于稳定训练

### 5. **Batch Size** ✅ 可以保持或增大
```bash
--batch_size 1024  # 保持或尝试 1280/1536
```
**原因**：
- LayerNorm 对 batch size 不敏感，可以使用更大的 batch size
- 更大的 batch size 可以：
  - 提高训练速度
  - 提供更稳定的梯度估计
  - 如果显存允许，可以尝试增大

### 6. **CosFace 参数** ✅ 可以保持
```bash
--margin_method cosface
```
**原因**：
- CosFace 与 ConvNext 配合良好
- 如果需要，可以在训练后根据结果调整 `m` 和 `s` 参数（需要在代码中修改）

### 7. **其他参数** ✅ 保持默认
- `num_blocks: 4` - 保持 4 层，ConvNext 的 block 更强大
- `dropout: 0.3` - 保持默认
- `weight_decay: 1e-4` - 保持默认

## 训练监控要点

### 1. 关注指标
- **Fisher Ratio**：应该比 baseline 提升更快
- **类内距离**：应该更快地减小
- **验证集 F1**：应该达到或超过 baseline

### 2. 训练曲线对比
使用 TensorBoard 对比两个实验：
```bash
tensorboard --logdir runs/
```
然后对比：
- `runs/ocec_hq` (baseline)
- `runs/ocec_hq_convnext` (ConvNext)

### 3. 异常情况处理

**如果训练不稳定**：
- 降低学习率到 `1e-4` 或 `8e-5`
- 增加 warmup 到 25-30 epochs
- 检查梯度是否爆炸（查看 TensorBoard 的梯度直方图）

**如果过拟合**：
- 增加 dropout 到 0.4 或 0.5
- 增加 weight_decay 到 2e-4
- 使用数据增强（如果还没有使用）

**如果收敛慢**：
- 检查学习率是否太低
- 检查 warmup 是否太长
- 考虑使用学习率调度器的不同策略

## 预期效果

### ConvNext 的优势
1. **更好的特征提取**：7x7 卷积核提供更大的感受野
2. **更稳定的训练**：LayerNorm 对 batch size 不敏感
3. **更强的表达能力**：Channel MLP 提供更好的特征混合

### 可能的改进
- **Fisher Ratio**：应该比 baseline 提升 10-20%
- **验证集 F1**：可能提升 1-3 个百分点
- **训练稳定性**：训练曲线应该更平滑

## 完整训练命令

```bash
bash train_convnext.sh
```

或者手动运行：
```bash
python -m ocec train \
    --data_root /ssddisk/guochuang/ocec/parquet_hq \
    --output_dir runs/ocec_hq_convnext \
    --epochs 1000 \
    --batch_size 1024 \
    --num_workers 16 \
    --train_ratio 0.8 \
    --val_ratio 0.2 \
    --image_size 32x64 \
    --base_channels 48 \
    --num_blocks 4 \
    --arch_variant convnext \
    --margin_method cosface \
    --lr 1.2e-4 \
    --weight_decay 1e-4 \
    --seed 42 \
    --device auto \
    --warmup_epochs 20
```

## 对比实验建议

为了公平对比，建议同时运行：
1. **Baseline + CosFace**：`runs/ocec_hq`（当前运行）
2. **ConvNext + CosFace**：`runs/ocec_hq_convnext`（新实验）

对比维度：
- 训练速度（每个 epoch 的时间）
- 收敛速度（达到相同 F1 的 epoch 数）
- 最终性能（最佳 F1、Fisher Ratio）
- 训练稳定性（loss 曲线的平滑度）

