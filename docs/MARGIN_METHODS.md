# Margin-based Loss Methods (ArcFace & CosFace)

## 概述

本项目现在支持三种损失函数模式：
- **none**: 使用标准的 BCEWithLogitsLoss（二元交叉熵）
- **arcface**: 使用 ArcFace (Additive Angular Margin Loss)
- **cosface**: 使用 CosFace/AM-Softmax (Additive Margin Softmax Loss) - **默认**

## 实现细节

### 1. ArcFace (Additive Angular Margin Loss)

**公式**: `cos(θ + m)`

**特点**:
- 在角度空间添加 margin
- 适用于需要强类间分离的场景
- 参数：m=0.2 (角度间隔), s=8 (scale)

**实现**: `ArcFaceHead` 类

### 2. CosFace (AM-Softmax)

**公式**: `cos(θ) - m`

**特点**:
- 在余弦空间添加 margin
- 计算更简单，训练更稳定
- 参数：m=0.35 (余弦间隔), s=8 (scale)
- **默认方法**

**实现**: `CosFaceHead` 类

### 3. 标准 BCE Loss

**特点**:
- 简单的二元交叉熵损失
- 适用于简单的二分类任务
- 不需要 margin 机制

## 使用方法

### 命令行参数

```bash
--margin_method {none,arcface,cosface}
```

**默认值**: `cosface`

**示例**:
```bash
# 使用 CosFace (默认)
python -m ocec train --margin_method cosface ...

# 使用 ArcFace
python -m ocec train --margin_method arcface ...

# 不使用 margin (标准 BCE)
python -m ocec train --margin_method none ...
```

### 代码配置

在 `TrainConfig` 中：
```python
margin_method: str = "cosface"  # "none", "arcface", "cosface"
```

在 `ModelConfig` 中：
```python
margin_method: str = "cosface"  # "none", "arcface", "cosface"
```

## 参数说明

### ArcFace 参数
- **m (margin)**: 0.2 - 角度间隔，值越大类间分离越强
- **s (scale)**: 8 - 缩放因子，用于放大 logits

### CosFace 参数
- **m (margin)**: 0.35 - 余弦间隔，值越大类间分离越强
- **s (scale)**: 8 - 缩放因子，用于放大 logits

## 训练和验证行为

### 训练时
- 使用带 margin 的 logits 计算损失（增强类间分离）
- 使用不带 margin 的 logits 计算指标（与验证时一致）

### 验证/推理时
- 不使用 margin，直接使用标准 logits
- 确保评估准确性和推理一致性

## 适用场景

### CosFace (推荐)
- **默认选择**
- 训练稳定，计算简单
- 适合大多数二分类任务
- 解决灰区问题（睁眼/闭眼边界模糊）

### ArcFace
- 需要更强的类间分离
- 类别边界非常模糊
- CosFace 效果不理想时尝试

### None (BCE)
- 简单的二分类任务
- 类别差异明显
- 不需要 margin 机制

## 代码结构

### 模型类
- `ArcFaceHead`: ArcFace 实现
- `CosFaceHead`: CosFace 实现
- `OCEC`: 主模型，根据 `margin_method` 选择使用哪个 head

### 训练流程
- `_run_epoch`: 根据 `margin_method` 选择损失函数和概率计算方式
- `train_pipeline`: 配置损失函数（CrossEntropyLoss 或 BCEWithLogitsLoss）

## 迁移说明

### 从 `use_arcface` 迁移

**旧代码**:
```python
--use_arcface  # 布尔值
```

**新代码**:
```python
--margin_method arcface  # 字符串选择
```

**等价关系**:
- `--use_arcface` → `--margin_method arcface`
- 不使用 `--use_arcface` → `--margin_method none` (或默认 `cosface`)

## 性能对比

| 方法 | 训练稳定性 | 计算复杂度 | 类间分离 | 推荐场景 |
|------|-----------|-----------|---------|---------|
| None (BCE) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | 简单任务 |
| CosFace | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | **默认推荐** |
| ArcFace | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 强分离需求 |

## 注意事项

1. **训练和验证指标一致性**: 训练时使用不带 margin 的 logits 计算指标，确保与验证时一致
2. **参数调整**: 如果效果不理想，可以尝试调整 margin (m) 和 scale (s) 参数
3. **默认选择**: CosFace 是默认方法，通常效果最好且训练最稳定

