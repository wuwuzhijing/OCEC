# 训练速度优化指南

## 问题诊断

训练速度慢（一个 epoch 30 分钟）可能的原因：
1. **混合精度训练未启用** - 这是最大的优化点
2. **ConvNext 架构计算量较大** - 相比 baseline 更复杂
3. **数据加载瓶颈** - 虽然已优化，但可能还有空间
4. **不必要的计算** - 某些指标计算可能过于频繁

## 优化方案

### 1. 启用混合精度训练（AMP）⭐ 最重要

**效果**：通常可以提升 **1.5-2x** 的训练速度，显存占用减少约 50%

**已更新**：`train_convnext.sh` 已添加 `--use_amp` 参数

```bash
--use_amp
```

**原理**：
- 使用 FP16 进行前向传播和梯度计算
- 使用 FP32 保存模型参数（master weights）
- 自动处理梯度缩放，避免下溢

**注意事项**：
- 仅适用于 CUDA 设备
- 某些操作会自动转换为 FP32（如 loss 计算）
- 训练稳定性通常不受影响

### 2. 数据加载优化（已优化）

当前配置已经很好：
- `pin_memory=True` - 加速 CPU 到 GPU 的数据传输
- `persistent_workers=True` - 保持 worker 进程存活，避免重复创建
- `prefetch_factor=8` - 预取 8 个 batch
- `num_workers=16` - 16 个数据加载进程

**进一步优化**（如果 CPU 充足）：
```bash
--num_workers 24  # 可以尝试增加到 24
```

### 3. 减少不必要的计算

#### 3.1 减少指标计算频率

当前每 10 个 epoch 计算一次分离指标，这是合理的。但如果还是太慢，可以：

```python
# 在 train_pipeline 中修改
if epoch % 20 == 0:  # 从 10 改为 20
    log_all_separation_metrics("train", train_outputs, epoch, tb_writer, config)
    log_all_separation_metrics("val", val_outputs, epoch, tb_writer, config)
```

#### 3.2 减少可视化频率

当前每 50 个 epoch 保存一次 embedding 和可视化，可以改为：

```python
# 在 train_pipeline 中修改
if epoch % 100 == 0:  # 从 50 改为 100
    # 保存 embedding 和可视化
```

### 4. Batch Size 优化

当前 `batch_size=1024`，使用 2 个 GPU，有效 batch size 是 2048。

**如果显存充足**：
- 可以尝试增大到 `batch_size=1280` 或 `1536`
- 更大的 batch size 可以提高 GPU 利用率
- 但需要相应调整学习率（线性缩放）

**如果显存不足**：
- 保持当前 batch size
- 启用 AMP 后显存占用会减少，可能可以增大 batch size

### 5. 编译优化（PyTorch 2.0+）

如果使用 PyTorch 2.0+，可以启用 `torch.compile`：

```python
# 在 train_pipeline 中，模型创建后添加
if hasattr(torch, 'compile'):
    model = torch.compile(model, mode='reduce-overhead')
```

**注意**：首次运行会较慢（编译时间），后续会加速。

### 6. 使用 DataParallel 优化

当前使用 `DataParallel`，如果有多 GPU，可以考虑：

- **DistributedDataParallel (DDP)**：比 DataParallel 更高效
- 但需要修改代码，实现较复杂

## 预期效果

启用 AMP 后：
- **训练速度**：提升 1.5-2x（30 分钟 → 15-20 分钟/epoch）
- **显存占用**：减少约 50%
- **训练稳定性**：通常不受影响（PyTorch 的 AMP 很稳定）

## 完整优化后的训练脚本

已更新 `train_convnext.sh`，包含：
- ✅ `--use_amp` - 混合精度训练
- ✅ 其他参数保持不变

## 监控训练速度

训练开始后，观察：
1. **每个 batch 的时间**：应该明显减少
2. **GPU 利用率**：应该保持在 90%+（使用 `nvidia-smi`）
3. **显存占用**：应该减少约 50%

## 如果还是太慢

1. **检查 GPU 利用率**：
   ```bash
   watch -n 1 nvidia-smi
   ```
   - 如果 GPU 利用率 < 80%，可能是数据加载瓶颈
   - 如果 GPU 利用率 > 95%，可能是模型计算瓶颈

2. **检查数据加载时间**：
   - 在日志中查看每个 batch 的时间
   - 如果数据加载时间占比高，增加 `num_workers`

3. **简化模型**：
   - 减少 `base_channels`（从 64 到 48）
   - 减少 `num_blocks`（从 4 到 3）

4. **使用更快的架构**：
   - 回到 `baseline` 架构（比 ConvNext 快）

## 性能对比

| 配置 | 每个 epoch 时间 | 显存占用 | 备注 |
|------|---------------|---------|------|
| ConvNext + FP32 | ~30 分钟 | 高 | 当前配置 |
| ConvNext + AMP | ~15-20 分钟 | 中 | 推荐 |
| Baseline + FP32 | ~10-15 分钟 | 低 | 更快但性能可能略低 |
| Baseline + AMP | ~5-10 分钟 | 低 | 最快 |

