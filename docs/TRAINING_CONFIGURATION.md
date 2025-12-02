# OCEC 训练配置分析文档

本文档详细分析当前训练配置的参数设置、优化策略，以及对二分类任务的适用性评估。

## 1. 训练参数一览

### 1.1 基础训练参数

| 参数 | 当前值 | 说明 |
|------|--------|------|
| **Epochs** | 1000 | 最大训练轮数 |
| **Batch Size** | 4096 | 单GPU批次大小 |
| **Effective Batch Size** | 8192 | 多GPU时：4096 × 2 GPUs |
| **Learning Rate (lr)** | 0.0001 (1e-4) | 初始学习率 |
| **Weight Decay** | 0.0001 (1e-4) | L2正则化系数 |
| **Random Seed** | 42 | 随机种子（用于数据划分和模型初始化） |
| **Num Workers** | 16 | 数据加载线程数 |
| **Use AMP** | False | 混合精度训练（当前未启用） |

### 1.2 数据相关参数

| 参数 | 当前值 | 说明 |
|------|--------|------|
| **Image Size** | 32×64 | 输入图像尺寸（高度×宽度） |
| **Train Ratio** | 0.8 | 训练集比例 |
| **Val Ratio** | 0.2 | 验证集比例 |
| **Test Ratio** | 0.0 | 测试集比例（当前未使用） |
| **Dataset Size** | ~470,811 | 总样本数（train: 376,648, val: 94,163） |
| **Class Distribution** | 53.2% open | 正类（睁眼）占比 |

### 1.3 模型架构参数

| 参数 | 当前值 | 说明 |
|------|--------|------|
| **Arch Variant** | baseline | 骨干网络架构 |
| **Head Variant** | auto | 分类头类型（自动选择） |
| **Base Channels** | 64 | 基础通道数 |
| **Num Blocks** | 8 | 网络块数量 |
| **Dropout** | 0.3 | Dropout比率 |
| **Token Mixer Grid** | [2, 3] | Transformer/MLP-Mixer的token网格 |
| **Token Mixer Layers** | 2 | Transformer/MLP-Mixer层数 |

### 1.4 优化器与学习率调度

| 组件 | 类型 | 参数 | 说明 |
|------|------|------|------|
| **Optimizer** | AdamW | lr=1e-4, weight_decay=1e-4 | Adam的权重衰减版本 |
| **LR Scheduler** | ReduceLROnPlateau | mode="min", factor=0.5, patience=2 | 基于验证损失的动态学习率调整 |
| **EMA** | Exponential Moving Average | decay=0.999 | 指数移动平均，用于模型平滑 |

### 1.5 损失函数

| 损失函数 | 当前状态 | 参数 | 说明 |
|---------|---------|------|------|
| **BCEWithLogitsLoss** | ✅ 使用中 | pos_weight=自动计算 | 二分类交叉熵损失（带类别权重） |
| **FocalLabelSmoothCE** | ⚪ 可选 | smoothing=0.05, gamma=2.0 | Focal Loss + Label Smoothing（当前未使用） |

### 1.6 正则化与数据增强

| 技术 | 参数 | 说明 |
|------|------|------|
| **Weight Decay** | 0.0001 | L2正则化 |
| **Dropout** | 0.3 | 随机失活 |
| **Random Horizontal Flip** | p=0.5 | 随机水平翻转 |
| **Random Photometric Distort** | p=0.5 | 随机光度畸变 |
| **Random CLAHE** | p=0.01 | 随机对比度限制自适应直方图均衡化 |

---

## 2. 优化器与学习率调度详细分析

### 2.1 AdamW 优化器

**原理：**
- **AdamW** 是 Adam 优化器的改进版本，将权重衰减从梯度更新中分离出来
- 公式：`θ_t = θ_{t-1} - lr × (m_t / (√v_t + ε) + λ × θ_{t-1})`
  - `m_t`: 梯度的一阶矩估计（动量）
  - `v_t`: 梯度的二阶矩估计（自适应学习率）
  - `λ`: 权重衰减系数（weight_decay）

**当前配置：**
- `lr = 1e-4`: 初始学习率
- `weight_decay = 1e-4`: 权重衰减系数

**优点：**
- ✅ 自适应学习率，对超参数不敏感
- ✅ 适合大规模数据和复杂模型
- ✅ 权重衰减独立于梯度，正则化效果更好
- ✅ 适合当前的大batch size (4096)

**缺点：**
- ⚠️ 内存占用较大（需要保存一阶和二阶矩）
- ⚠️ 对于简单任务可能收敛较慢

### 2.2 ReduceLROnPlateau 学习率调度器

**原理：**
- 监控验证集损失（或训练集损失，如果验证集不可用）
- 当损失停止下降时，按比例降低学习率
- 公式：`new_lr = old_lr × factor`（当patience个epoch内损失未改善时）

**当前配置：**
- `mode = "min"`: 监控指标越小越好（损失）
- `factor = 0.5`: 学习率衰减因子（每次减半）
- `patience = 2`: 容忍2个epoch不改善后降低学习率

**工作流程：**
```
Epoch 1: lr = 1e-4, val_loss = 0.5
Epoch 2: lr = 1e-4, val_loss = 0.4  (改善)
Epoch 3: lr = 1e-4, val_loss = 0.41 (未改善，patience=1)
Epoch 4: lr = 1e-4, val_loss = 0.42 (未改善，patience=2) → 触发衰减
Epoch 5: lr = 5e-5, val_loss = 0.38 (改善，重置patience)
```

**优点：**
- ✅ 自适应调整，无需手动设置学习率衰减时机
- ✅ 适合长期训练（1000 epochs）
- ✅ 防止学习率过早衰减

**缺点：**
- ⚠️ patience=2 可能过于敏感，容易过早降低学习率
- ⚠️ 对于波动较大的损失可能不够稳定

### 2.3 EMA (Exponential Moving Average)

**原理：**
- 维护模型参数的指数移动平均
- 公式：`shadow_t = decay × shadow_{t-1} + (1 - decay) × param_t`
- 验证时使用EMA参数，训练时使用原始参数

**当前配置：**
- `decay = 0.999`: 非常高的衰减率，EMA参数变化缓慢

**优点：**
- ✅ 提供更稳定的模型参数
- ✅ 通常能提高验证集性能
- ✅ 减少训练过程中的波动

**缺点：**
- ⚠️ decay=0.999 可能过于保守，EMA参数更新很慢
- ⚠️ 需要额外的内存存储shadow参数

---

## 3. 对于二分类任务的适用性分析

### 3.1 优化器适用性

#### ✅ **AdamW 适合当前任务的原因：**

1. **大规模数据（~47万样本）**
   - AdamW的自适应特性适合处理大量数据
   - 能够自动调整不同参数的学习率

2. **大Batch Size (4096)**
   - 大batch size需要较小的学习率
   - AdamW的自适应机制能更好地处理大batch训练

3. **复杂模型架构**
   - 8个blocks，64个基础通道
   - 不同层可能需要不同的学习率，AdamW能自动适应

4. **长期训练（1000 epochs）**
   - AdamW在长期训练中表现稳定
   - 配合学习率调度器能持续优化

#### ⚠️ **潜在问题：**

1. **学习率可能偏小**
   - `lr=1e-4` 对于大batch size可能偏保守
   - 建议：可以尝试 `lr=2e-4` 或 `lr=3e-4`

2. **Weight Decay可能偏小**
   - `weight_decay=1e-4` 对于当前模型可能不够
   - 建议：可以尝试 `weight_decay=1e-3` 或 `weight_decay=5e-4`

### 3.2 学习率调度器适用性

#### ✅ **ReduceLROnPlateau 适合当前任务的原因：**

1. **基于验证损失调整**
   - 直接监控模型性能，而非训练轮数
   - 更符合实际需求

2. **自适应衰减**
   - 不需要预设衰减时机
   - 适合长期训练（1000 epochs）

#### ⚠️ **潜在问题：**

1. **Patience=2 可能过于敏感**
   - 验证损失可能有正常波动
   - 建议：增加到 `patience=5` 或 `patience=10`

2. **Factor=0.5 可能过于激进**
   - 每次减半可能导致学习率下降过快
   - 建议：使用 `factor=0.7` 或 `factor=0.8` 更温和

3. **缺少warmup阶段**
   - 训练初期可能需要warmup
   - 建议：添加warmup scheduler（如LinearWarmup）

### 3.3 损失函数适用性

#### ✅ **BCEWithLogitsLoss 适合当前任务的原因：**

1. **标准二分类损失**
   - 专门为二分类设计
   - 数值稳定（使用logits而非概率）

2. **类别权重平衡**
   - 自动计算 `pos_weight = negatives / positives`
   - 处理类别不平衡（53.2% vs 46.8%）

3. **与sigmoid激活匹配**
   - 模型输出单个分数，使用sigmoid转换为概率
   - 训练和评估一致

#### ⚠️ **FocalLabelSmoothCE 的潜在优势：**

1. **Focal Loss**
   - 关注难例，自动调整样本权重
   - 对于类别不平衡更有效

2. **Label Smoothing**
   - 防止过拟合
   - 提高模型泛化能力

3. **当前未使用的原因**
   - 需要确保训练和评估的概率计算一致（已修复）
   - 可以尝试切换使用

### 3.4 Batch Size 适用性

#### ✅ **大Batch Size (4096) 的优势：**

1. **训练速度快**
   - 每个epoch的迭代次数少
   - GPU利用率高

2. **梯度估计更稳定**
   - 大批次提供更准确的梯度估计
   - 减少训练波动

#### ⚠️ **潜在问题：**

1. **泛化能力可能下降**
   - 大batch size可能导致模型泛化能力下降
   - 建议：可以尝试较小的batch size（如2048或1024）

2. **学习率需要调整**
   - 大batch size通常需要更大的学习率
   - 线性缩放规则：`lr = base_lr × (batch_size / base_batch_size)`
   - 当前：`lr = 1e-4` 可能偏小

### 3.5 数据增强适用性

#### ✅ **当前数据增强策略：**

1. **Random Horizontal Flip (p=0.5)**
   - 适合眼睛图像（左右对称）
   - ✅ 合理

2. **Random Photometric Distort (p=0.5)**
   - 模拟不同光照条件
   - ✅ 适合真实场景

3. **Random CLAHE (p=0.01)**
   - 概率很低，偶尔增强对比度
   - ✅ 合理，避免过度增强

#### ⚠️ **可能缺失的增强：**

1. **没有旋转增强**
   - 代码中注释掉了 `RandomRotation`
   - 眼睛图像可能不需要旋转

2. **没有随机裁剪**
   - 当前只有Resize
   - 可以考虑添加RandomCrop增加多样性

### 3.6 正则化适用性

#### ✅ **当前正则化策略：**

1. **Weight Decay (1e-4)**
   - L2正则化，防止过拟合
   - ✅ 合理

2. **Dropout (0.3)**
   - 随机失活30%的神经元
   - ✅ 适合当前模型复杂度

#### ⚠️ **潜在改进：**

1. **Dropout可能偏高**
   - 0.3对于当前模型可能过于激进
   - 建议：可以尝试0.2或0.25

2. **可以考虑其他正则化**
   - Mixup/CutMix（数据增强）
   - Stochastic Depth（随机深度）

---

## 4. 综合评估

### 4.1 当前配置的优势

1. ✅ **优化器选择合理**
   - AdamW适合大规模数据和复杂模型
   - 自适应学习率减少调参负担

2. ✅ **学习率调度策略合理**
   - ReduceLROnPlateau自适应调整
   - 适合长期训练

3. ✅ **正则化充分**
   - Weight Decay + Dropout
   - 防止过拟合

4. ✅ **数据增强适度**
   - 不过度增强，保持数据真实性

5. ✅ **EMA提升稳定性**
   - 使用EMA参数进行验证
   - 提高模型稳定性

### 4.2 当前配置的潜在问题

1. ⚠️ **学习率可能偏小**
   - 对于batch size 4096，1e-4可能偏保守
   - 建议尝试2e-4或3e-4

2. ⚠️ **学习率调度器patience过短**
   - patience=2可能过于敏感
   - 建议增加到5-10

3. ⚠️ **缺少warmup阶段**
   - 训练初期可能需要warmup
   - 有助于稳定训练

4. ⚠️ **Weight Decay可能偏小**
   - 1e-4对于当前模型可能不够
   - 建议尝试1e-3

5. ⚠️ **Batch Size过大可能影响泛化**
   - 4096可能过大
   - 建议尝试2048或1024

---

## 5. 改进建议

### 5.1 短期优化（立即可尝试）

#### 5.1.1 学习率相关

1. **增加初始学习率**
   ```python
   lr = 2e-4  # 或 3e-4
   ```
   - 理由：大batch size需要更大的学习率
   - 风险：可能训练不稳定，需要监控

2. **调整学习率调度器参数**
   ```python
   scheduler = ReduceLROnPlateau(
       optimizer, 
       mode="min", 
       factor=0.7,      # 从0.5改为0.7，更温和
       patience=5       # 从2改为5，更稳定
   )
   ```
   - 理由：减少过早衰减，更稳定的训练

3. **添加Warmup阶段**
   ```python
   from torch.optim.lr_scheduler import LambdaLR
   
   def warmup_lambda(epoch):
       if epoch < 5:
           return (epoch + 1) / 5
       return 1.0
   
   warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
   # 在训练循环中，前5个epoch使用warmup_scheduler，之后使用ReduceLROnPlateau
   ```
   - 理由：训练初期稳定，避免早期震荡

#### 5.1.2 正则化相关

1. **增加Weight Decay**
   ```python
   weight_decay = 1e-3  # 或 5e-4
   ```
   - 理由：更强的正则化，防止过拟合

2. **调整Dropout**
   ```python
   dropout = 0.25  # 或 0.2
   ```
   - 理由：0.3可能过于激进，降低可能提升性能

#### 5.1.3 损失函数

1. **尝试FocalLabelSmoothCE**
   ```python
   use_focal_loss = True
   criterion = FocalLabelSmoothCE(smoothing=0.05, gamma=2.0)
   ```
   - 理由：关注难例，可能提升性能
   - 注意：已修复训练/评估一致性，可以安全使用

### 5.2 中期优化（需要实验验证）

#### 5.2.1 Batch Size调整

1. **尝试较小的Batch Size**
   ```python
   batch_size = 2048  # 或 1024
   lr = 2e-4  # 相应调整学习率
   ```
   - 理由：可能提升泛化能力
   - 权衡：训练速度会变慢

2. **使用梯度累积模拟大batch**
   ```python
   accumulation_steps = 2  # 模拟batch_size=8192
   # 每accumulation_steps个batch才更新一次
   ```
   - 理由：保持大batch的优势，同时节省内存

#### 5.2.2 学习率调度策略

1. **使用Cosine Annealing**
   ```python
   from torch.optim.lr_scheduler import CosineAnnealingLR
   scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
   ```
   - 理由：更平滑的学习率衰减
   - 适合：长期训练（1000 epochs）

2. **组合调度器（Warmup + Cosine）**
   ```python
   # Warmup 5 epochs, 然后 Cosine Annealing
   ```
   - 理由：结合两者优势

#### 5.2.3 数据增强增强

1. **添加Mixup/CutMix**
   ```python
   # 在数据增强中添加Mixup或CutMix
   ```
   - 理由：提升模型泛化能力
   - 注意：需要调整损失函数计算方式

2. **添加随机裁剪**
   ```python
   transforms.RandomCrop(size, padding=4)
   ```
   - 理由：增加数据多样性

### 5.3 长期优化（架构级别）

#### 5.3.1 模型架构

1. **尝试不同的Arch Variant**
   - `inverted_se`: 倒置残差 + SE注意力
   - `convnext`: 现代卷积架构
   - 可能提升性能

2. **调整模型容量**
   - 增加 `base_channels` 或 `num_blocks`
   - 权衡：计算成本和性能

#### 5.3.2 训练策略

1. **使用混合精度训练**
   ```python
   use_amp = True
   ```
   - 理由：加速训练，节省显存
   - 注意：需要验证数值稳定性

2. **使用学习率查找（LR Finder）**
   - 找到最优的初始学习率
   - 工具：`torch-lr-finder` 或手动实现

3. **使用Cyclic Learning Rate**
   ```python
   from torch.optim.lr_scheduler import CyclicLR
   scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=2000)
   ```
   - 理由：可能跳出局部最优

---

## 6. 推荐配置方案

### 6.1 保守方案（稳定优先）

```python
# 优化器
optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=5e-4)

# 学习率调度器
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=5)

# 其他参数保持不变
batch_size = 4096
dropout = 0.25
use_focal_loss = False  # 继续使用BCEWithLogitsLoss
```

**预期效果：**
- 训练更稳定
- 学习率衰减更合理
- 性能可能略有提升

### 6.2 激进方案（性能优先）

```python
# 优化器
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)

# 学习率调度器（带warmup）
# 前5个epoch: warmup
# 之后: ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=10)

# 损失函数
use_focal_loss = True
criterion = FocalLabelSmoothCE(smoothing=0.05, gamma=2.0)

# 其他参数
batch_size = 2048  # 减小batch size
dropout = 0.2
```

**预期效果：**
- 可能获得更好的性能
- 需要更多实验验证
- 训练时间可能增加

### 6.3 平衡方案（推荐）

```python
# 优化器
optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=5e-4)

# 学习率调度器
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=5)

# 损失函数（可选）
use_focal_loss = True  # 尝试FocalLabelSmoothCE
criterion = FocalLabelSmoothCE(smoothing=0.05, gamma=2.0)

# 其他参数
batch_size = 4096  # 保持大batch size
dropout = 0.25
```

**预期效果：**
- 平衡稳定性和性能
- 风险较低
- 适合生产环境

---

## 7. 监控指标建议

### 7.1 关键指标

1. **训练损失 vs 验证损失**
   - 监控过拟合迹象
   - 如果差距过大，增加正则化

2. **学习率变化**
   - 记录每次学习率调整
   - 如果频繁调整，增加patience

3. **F1 Score趋势**
   - 主要优化目标
   - 如果停滞，考虑调整学习率或损失函数

4. **梯度范数**
   - 监控梯度爆炸/消失
   - 如果异常，调整学习率

### 7.2 调参优先级

1. **高优先级**
   - 学习率（lr）
   - 学习率调度器参数（patience, factor）
   - 损失函数选择

2. **中优先级**
   - Weight Decay
   - Dropout
   - Batch Size

3. **低优先级**
   - 数据增强细节
   - EMA decay
   - 模型架构参数

---

## 8. 总结

### 8.1 当前配置评估

**总体评价：⭐⭐⭐⭐ (4/5)**

- ✅ 优化器选择合理（AdamW）
- ✅ 学习率调度策略合理（ReduceLROnPlateau）
- ✅ 正则化充分（Weight Decay + Dropout）
- ⚠️ 学习率可能偏小
- ⚠️ 学习率调度器patience过短
- ⚠️ 缺少warmup阶段

### 8.2 适用性结论

当前配置**基本适合**二分类任务，但有以下改进空间：

1. **学习率相关**：可以适当增大初始学习率，调整调度器参数
2. **正则化**：可以适当增加weight decay，降低dropout
3. **损失函数**：可以尝试FocalLabelSmoothCE，关注难例
4. **训练策略**：可以添加warmup，提升训练稳定性

### 8.3 下一步行动

1. **立即尝试**：
   - 调整学习率调度器参数（patience=5, factor=0.7）
   - 尝试FocalLabelSmoothCE损失函数

2. **实验验证**：
   - 尝试不同的学习率（2e-4, 3e-4）
   - 尝试不同的weight decay（5e-4, 1e-3）

3. **长期优化**：
   - 添加warmup阶段
   - 尝试不同的batch size
   - 使用学习率查找工具

---

**文档版本**: 1.0  
**最后更新**: 2025-12-01  
**维护者**: OCEC项目组

