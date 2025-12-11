# 训练指标修复说明

## 修复的问题

### 1. ValueError: too many values to unpack

**错误位置**: `ocec/pipeline.py:1008`

**问题**: 
```python
logits_no_margin, _ = model(images, labels=None, return_embedding=False, training=False)
```

当 `return_embedding=False` 时，模型只返回一个值（logits），不是两个值。

**修复**:
```python
logits_no_margin = model(images, labels=None, return_embedding=False, training=False)
```

### 2. 训练时 f1 不正常

**问题原因**:
- 训练时使用带 margin 的 logits 计算概率，导致分布异常
- 验证时使用不带 margin 的 logits 计算概率，分布正常
- 训练和验证的指标计算方式不一致

**修复方案**:
- **损失函数**: 仍使用带 margin 的 logits（训练需要）
- **指标计算**: 训练时也使用不带 margin 的 logits（与验证时一致）

**实现**:
```python
if use_margin:
    # 损失函数使用带 margin 的 logits
    loss = criterion(logits, labels.long())
    
    if train_mode:
        # 训练时：重新计算不带 margin 的 logits 用于指标计算
        with torch.no_grad():
            was_training = model.training
            model.eval()
            logits_no_margin = model(images, labels=None, return_embedding=False, training=False)
            model.train(was_training)
            # 使用不带 margin 的 logits 计算概率
            if logits_no_margin.ndim == 2 and logits_no_margin.size(1) == 2:
                probs = torch.softmax(logits_no_margin, dim=1)[:, 1]
            else:
                LOGGER.warning(f"Unexpected logits_no_margin shape: {logits_no_margin.shape}")
                probs = torch.softmax(logits, dim=1)[:, 1]
    else:
        # 验证时：直接使用不带 margin 的 logits
        probs = torch.softmax(logits, dim=1)[:, 1]
```

### 3. 输出维度匹配检查

**检查点**:

1. **Margin-based loss (ArcFace/CosFace)**:
   - 输出: `[B, 2]` logits
   - 概率计算: `softmax(logits, dim=1)[:, 1]` (取第二列，睁眼概率)

2. **标准 BCE loss**:
   - 输出: `[B]` 或 `[B, 1]` logits
   - 概率计算: `sigmoid(logits)` 或 `sigmoid(logits.squeeze(1))`

3. **维度验证**:
   - 所有概率计算前都检查 logits 维度
   - 添加了警告日志，如果维度不匹配

**代码位置**:
- `_run_epoch`: 训练和验证时的概率计算
- `_evaluate_predictions`: 评估时的概率计算
- `predict_images`: 推理时的概率计算

## 修复后的行为

### 训练时
1. 使用带 margin 的 logits 计算损失（增强类间分离）
2. 使用不带 margin 的 logits 计算指标（与验证一致）
3. 调试信息显示实际用于计算概率的 logits

### 验证时
1. 使用不带 margin 的 logits 计算概率和指标
2. 确保评估准确性

### 指标计算一致性
- 训练和验证使用相同的概率计算方式
- 都使用不带 margin 的 logits
- 确保指标的可比性

## 维度检查清单

### Margin-based loss (ArcFace/CosFace)
- ✅ 模型输出: `[B, 2]`
- ✅ 概率计算: `softmax(logits, dim=1)[:, 1]` → `[B]`
- ✅ 预测: `(probs >= 0.5).long()` → `[B]`

### 标准 BCE loss
- ✅ 模型输出: `[B]` 或 `[B, 1]`
- ✅ 概率计算: `sigmoid(logits)` → `[B]`
- ✅ 预测: `(probs >= 0.5).long()` → `[B]`

## 预期效果

修复后：
1. ✅ 不再出现 "too many values to unpack" 错误
2. ✅ 训练集指标恢复正常，与验证集指标趋势一致
3. ✅ 训练和验证的指标计算方式一致
4. ✅ 所有维度匹配，概率计算正确

## 验证方法

运行训练后，检查：
1. 训练集和验证集的 f1 应该逐渐提升
2. 训练集和验证集的指标趋势应该一致
3. 调试日志中的 logits 和 probs 分布应该合理
4. 不应该出现维度不匹配的警告

