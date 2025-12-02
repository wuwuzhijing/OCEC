# OCEC 训练指标说明文档

本文档详细说明训练过程中保存的所有评估指标，包括其原理、作用、实现方式和改进建议。

## 1. 指标概览

### 1.1 基础分类指标

| 指标名称 | 类型 | 范围 | 保存位置 | 更新频率 |
|---------|------|------|---------|---------|
| Loss | 损失函数 | [0, +∞) | TensorBoard: `loss/train`, `loss/val` | 每个epoch |
| Accuracy | 准确率 | [0, 1] | TensorBoard: `metrics/train_accuracy`, `metrics/val_accuracy` | 每个epoch |
| Precision | 精确率 | [0, 1] | 日志输出 | 每个epoch |
| Recall | 召回率 | [0, 1] | 日志输出 | 每个epoch |
| F1 Score | F1分数 | [0, 1] | TensorBoard: `metrics/train_f1`, `metrics/val_f1` | 每个epoch |
| AUC | ROC曲线下面积 | [0, 1] | ROC图像 | 每个epoch |

### 1.2 概率分布分离指标

| 指标名称 | 类型 | 范围 | 保存位置 | 更新频率 |
|---------|------|------|---------|---------|
| KS Distance | 分布距离 | [0, 1] | TensorBoard: `{split}/ks_distance` | 每10个epoch |
| Hellinger Distance | 分布距离 | [0, 1] | TensorBoard: `{split}/hellinger` | 每10个epoch |
| JS Divergence | 分布散度 | [0, +∞) | TensorBoard: `{split}/js_divergence` | 每10个epoch |
| ECE | 校准误差 | [0, 1] | TensorBoard: `{split}/ece` | 每10个epoch |
| Margin Score | 置信度 | [0, 0.5] | TensorBoard: `{split}/margin` | 每10个epoch |
| Bayes Error | 理论误差下界 | [0, 0.5] | TensorBoard: `{split}/bayes_error` | 每10个epoch |

### 1.3 嵌入空间可分性指标

| 指标名称 | 类型 | 范围 | 保存位置 | 更新频率 |
|---------|------|------|---------|---------|
| Intra-class Distance | 类内距离 | [0, +∞) | TensorBoard: `{split}/intra` | 每10个epoch |
| Inter-class Distance | 类间距离 | [0, +∞) | TensorBoard: `{split}/inter` | 每10个epoch |
| Fisher Ratio | 可分性比率 | [0, +∞) | TensorBoard: `{split}/fisher` | 每10个epoch |
| Bhattacharyya Distance | 分布距离 | [0, +∞) | TensorBoard: `{split}/bhattacharyya` | 每10个epoch |
| PCA Energy | 主成分能量 | [0, 1] | TensorBoard: `{split}/pca_energy_{i}` | 每10个epoch |

### 1.4 可视化与异常检测

| 类型 | 内容 | 保存位置 | 更新频率 |
|------|------|---------|---------|
| Confusion Matrix | 混淆矩阵 | `diagnostics/{split}/confusion_{split}_epoch{epoch}.png` | 每个epoch |
| ROC Curve | ROC曲线 | `diagnostics/{split}/roc_{split}_epoch{epoch}.png` | 每个epoch |
| t-SNE | 2D嵌入可视化 | `tsne/epoch_{epoch}.png` | 每10个epoch |
| PCA 3D | 3D嵌入可视化 | `pca3d/epoch_{epoch}.png` | 每10个epoch |
| Outliers | 离群点检测 | `outliers.csv` | 每10个epoch |
| Hard Samples | 难例检测 | `hard_samples.csv` | 每10个epoch |
| Mislabeled | 错标检测 | `mislabeled.csv` | 每10个epoch |

---

## 2. 指标详细说明

### 2.1 基础分类指标

#### 2.1.1 Loss (损失函数)

**原理：**
- 训练时使用 `FocalLabelSmoothCE` 或 `BCEWithLogitsLoss`
- `FocalLabelSmoothCE`: 结合了Focal Loss（关注难例）和Label Smoothing（防止过拟合）
- `BCEWithLogitsLoss`: 标准二分类交叉熵损失，支持类别权重平衡

**作用：**
- 衡量模型预测与真实标签的差异
- 训练过程中的主要优化目标
- 用于学习率调度（ReduceLROnPlateau）

**实现方式：**
```python
# FocalLabelSmoothCE: 对单个分数 [B] 自动转换为 [B, 2]
# BCEWithLogitsLoss: 直接处理单个分数或两个类别输出
loss = criterion(logits, labels)
```

**改进方向：**
- Loss下降缓慢：检查学习率、数据质量、模型容量
- Loss震荡：减小学习率、增加batch size、使用梯度裁剪
- Loss不下降：检查数据标注、模型初始化、损失函数选择

---

#### 2.1.2 Accuracy (准确率)

**原理：**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
其中 TP=真阳性, TN=真阴性, FP=假阳性, FN=假阴性

**作用：**
- 整体分类正确率
- 适用于类别平衡的数据集

**实现方式：**
```python
preds = (probs >= 0.5).long()
accuracy = (stats["tp"] + stats["tn"]) / stats["samples"]
```

**改进方向：**
- 准确率低：检查数据质量、模型容量、训练策略
- 准确率高但F1低：可能存在类别不平衡，关注Precision/Recall

---

#### 2.1.3 Precision (精确率)

**原理：**
```
Precision = TP / (TP + FP)
```
预测为正类的样本中，真正为正类的比例

**作用：**
- 衡量模型预测正类的可靠性
- 高Precision意味着假阳性少

**实现方式：**
```python
precision = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0.0
```

**改进方向：**
- Precision低：提高分类阈值、增加正类样本、使用Focal Loss关注难例

---

#### 2.1.4 Recall (召回率)

**原理：**
```
Recall = TP / (TP + FN)
```
真实正类样本中，被正确预测为正类的比例

**作用：**
- 衡量模型对正类的覆盖能力
- 高Recall意味着漏检少

**实现方式：**
```python
recall = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0.0
```

**改进方向：**
- Recall低：降低分类阈值、增加正类样本、使用数据增强

---

#### 2.1.5 F1 Score

**原理：**
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
Precision和Recall的调和平均数

**作用：**
- 平衡精确率和召回率的综合指标
- 用于选择最佳模型（best checkpoint基于F1）

**实现方式：**
```python
f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
```

**改进方向：**
- F1低：分析Precision和Recall哪个更低，针对性改进
- F1不提升：检查过拟合、数据质量、模型架构

---

#### 2.1.6 AUC (ROC曲线下面积)

**原理：**
- ROC曲线：以假阳性率(FPR)为横轴，真阳性率(TPR)为纵轴
- AUC：ROC曲线下的面积，衡量模型区分能力

**作用：**
- 不受分类阈值影响
- 衡量模型的整体分类能力

**实现方式：**
```python
# 计算TPR和FPR，然后使用梯形积分
auc = float(np.trapz(tpr, fpr))
```

**改进方向：**
- AUC < 0.7：模型区分能力差，检查特征提取、数据质量
- AUC > 0.9但准确率低：可能存在类别不平衡，调整阈值

---

### 2.2 概率分布分离指标

#### 2.2.1 KS Distance (Kolmogorov-Smirnov距离)

**原理：**
```
KS = max |CDF₀(x) - CDF₁(x)|
```
两个类别概率分布的累积分布函数(CDF)之间的最大差异

**作用：**
- 衡量两个类别概率分布的分离程度
- 值越大（接近1），分离越好

**实现方式：**
```python
# 计算两个类别的CDF，找到最大差异
cdf0 = np.arange(1, len(p0)+1) / len(p0)
cdf1 = np.arange(1, len(p1)+1) / len(p1)
ks = np.max(np.abs(cdf0_i - cdf1_i))
```

**改进方向：**
- KS < 0.3：分布重叠严重，需要更强的特征学习
- KS > 0.7：分布分离良好，模型性能应该较好

---

#### 2.2.2 Hellinger Distance

**原理：**
```
H = √(1 - Σ√(P₀(i) × P₁(i)))
```
两个概率分布之间的Hellinger距离，范围[0, 1]

**作用：**
- 衡量概率分布的相似性
- 值越大，分布差异越大，分类越容易

**实现方式：**
```python
hellinger = np.sqrt(1 - np.sum(np.sqrt(hist0 * hist1)))
```

**改进方向：**
- Hellinger < 0.3：分布相似，难以区分
- Hellinger > 0.7：分布差异大，分类应该容易

---

#### 2.2.3 JS Divergence (Jensen-Shannon散度)

**原理：**
```
JS(P||Q) = 0.5 × KL(P||M) + 0.5 × KL(Q||M)
其中 M = 0.5 × (P + Q)
```
基于KL散度的对称度量

**作用：**
- 衡量两个概率分布的差异
- 值越大，分布差异越大

**实现方式：**
```python
m = 0.5 * (hist0 + hist1)
js = 0.5 * kl_divergence(hist0, m) + 0.5 * kl_divergence(hist1, m)
```

**改进方向：**
- JS < 0.1：分布几乎相同，需要改进特征学习
- JS > 0.5：分布差异明显，模型应该表现良好

---

#### 2.2.4 ECE (Expected Calibration Error，期望校准误差)

**原理：**
```
ECE = Σ (|Bₘ|/n) × |acc(Bₘ) - conf(Bₘ)|
```
将概率分成bins，计算每个bin内准确率与置信度的差异

**作用：**
- 衡量模型预测概率的校准程度
- ECE=0表示完美校准（预测概率=真实概率）

**实现方式：**
```python
# 将[0,1]分成10个bins，计算每个bin的准确率和置信度
for i in range(bins):
    mask = (probs >= i/bins) & (probs < (i+1)/bins)
    conf = probs[mask].mean()  # 置信度
    acc = (probs[mask] > 0.5).astype(int).mean()  # 准确率
    ece += len(probs[mask]) / len(probs) * abs(acc - conf)
```

**改进方向：**
- ECE > 0.1：模型过度自信或欠自信，使用温度缩放、Platt缩放
- ECE < 0.05：模型校准良好

---

#### 2.2.5 Margin Score

**原理：**
```
Margin = mean(|p - 0.5|)
```
所有样本预测概率与决策边界(0.5)的平均距离

**作用：**
- 衡量模型预测的置信度
- 值越大，预测越确定

**实现方式：**
```python
margin = np.mean(np.abs(probs - 0.5))
```

**改进方向：**
- Margin < 0.1：预测不确定，模型需要更多训练或数据
- Margin > 0.3：预测很确定，但需检查是否过拟合

---

#### 2.2.6 Bayes Error (贝叶斯误差)

**原理：**
```
Bayes Error = 0.5 × overlap(P₀, P₁)
```
两个类别概率分布重叠区域的一半

**作用：**
- 理论上的最小分类误差
- 表示数据本身的难度

**实现方式：**
```python
hist0, edges = np.histogram(p0, bins=200, range=(0,1), density=True)
hist1, _ = np.histogram(p1, bins=200, range=(0,1), density=True)
overlap = np.minimum(hist0, hist1).sum() * (edges[1] - edges[0])
bayes_error = 0.5 * overlap
```

**改进方向：**
- Bayes Error > 0.2：数据本身难以区分，需要更好的特征或更多数据
- Bayes Error < 0.05：数据容易区分，模型应该能达到很高准确率

---

### 2.3 嵌入空间可分性指标

#### 2.3.1 Intra-class Distance (类内距离)

**原理：**
```
Intra = 0.5 × (mean(||x₀ - μ₀||²) + mean(||x₁ - μ₁||²))
```
每个类别内部样本到类别中心的平均距离

**作用：**
- 衡量类别内部的紧密度
- 值越小，同类样本越聚集

**实现方式：**
```python
mu0 = cls0.mean(dim=0)
mu1 = cls1.mean(dim=0)
intra0 = ((cls0 - mu0) ** 2).sum(dim=1).mean()
intra1 = ((cls1 - mu1) ** 2).sum(dim=1).mean()
intra = (intra0 + intra1) / 2
```

**改进方向：**
- Intra大：同类样本分散，使用对比学习、中心损失、Triplet Loss
- Intra小：同类样本聚集，模型学习良好

---

#### 2.3.2 Inter-class Distance (类间距离)

**原理：**
```
Inter = ||μ₀ - μ₁||
```
两个类别中心之间的欧氏距离

**作用：**
- 衡量不同类别之间的分离程度
- 值越大，类别分离越好

**实现方式：**
```python
inter = torch.dist(mu0, mu1)
```

**改进方向：**
- Inter小：类别中心接近，需要增强特征区分能力
- Inter大：类别分离良好，但需结合Intra判断

---

#### 2.3.3 Fisher Ratio

**原理：**
```
Fisher = Inter / (Intra + ε)
```
类间距离与类内距离的比值

**作用：**
- 综合衡量类别可分性
- 值越大，分类越容易（类间距离大，类内距离小）

**实现方式：**
```python
fisher = inter / (intra + 1e-8)
```

**改进方向：**
- Fisher < 1：类内距离大于类间距离，分类困难
- Fisher > 5：可分性良好
- Fisher在1-3之间：需要改进，使用ArcFace、CosFace等损失函数

---

#### 2.3.4 Bhattacharyya Distance

**原理：**
```
BC = 0.25 × Σ log(0.25 × (σ₀/σ₁ + σ₁/σ₀ + 2)) + 0.25 × Σ (μ₀-μ₁)²/(σ₀+σ₁)
```
考虑均值和方差的分布距离度量

**作用：**
- 更全面地衡量两个类别分布的差异
- 考虑了一阶矩（均值）和二阶矩（方差）

**实现方式：**
```python
sigma0 = cls0.var(dim=0) + 1e-8
sigma1 = cls1.var(dim=0) + 1e-8
bc = 0.25 * torch.sum(torch.log(0.25 * (sigma0/sigma1 + sigma1/sigma0 + 2)))
bc += 0.25 * torch.sum((mu0 - mu1)**2 / (sigma0 + sigma1))
```

**改进方向：**
- BC小：分布重叠严重
- BC大：分布分离良好

---

#### 2.3.5 PCA Energy (主成分能量)

**原理：**
```
Energy_i = λᵢ / Σλⱼ
```
前k个主成分的特征值占总特征值的比例

**作用：**
- 衡量嵌入空间的维度利用效率
- 值越大，信息越集中在少数维度

**实现方式：**
```python
cov = np.cov(emb.T)
eigvals = np.linalg.eigvalsh(cov)[::-1]
energy = eigvals[:k] / eigvals.sum()
```

**改进方向：**
- 前5个主成分能量 < 0.8：信息分散，可能需要降维或特征选择
- 前5个主成分能量 > 0.9：信息集中，但需检查是否过拟合

---

### 2.4 可视化与异常检测

#### 2.4.1 Confusion Matrix (混淆矩阵)

**作用：**
- 直观展示分类结果
- 识别主要错误类型（FP vs FN）

**改进方向：**
- FP多：提高分类阈值、增加负类样本
- FN多：降低分类阈值、增加正类样本

---

#### 2.4.2 ROC Curve

**作用：**
- 展示不同阈值下的TPR和FPR
- AUC值衡量整体性能

---

#### 2.4.3 t-SNE / PCA 3D

**作用：**
- 可视化嵌入空间的分布
- 检查类别是否分离、是否存在异常聚类

**改进方向：**
- 类别重叠：需要更强的特征学习
- 异常聚类：检查数据标注、离群点

---

#### 2.4.4 Outliers (离群点)

**原理：**
使用马氏距离检测每个类别内的离群点

**作用：**
- 识别可能标注错误的样本
- 识别异常样本

**改进方向：**
- 检查离群点的标注是否正确
- 考虑移除或重新标注离群点

---

#### 2.4.5 Hard Samples (难例)

**原理：**
```
Hard = {i | |pᵢ - 0.5| < threshold}
```
预测概率接近0.5的样本

**作用：**
- 识别模型难以分类的样本
- 用于困难样本挖掘

**改进方向：**
- 增加难例的训练权重
- 检查难例是否有共同特征，针对性改进

---

#### 2.4.6 Mislabeled (错标检测)

**原理：**
高置信度预测与标签冲突的样本

**作用：**
- 识别可能标注错误的样本
- 提高数据质量

**改进方向：**
- 人工检查错标样本，修正标注
- 使用半监督学习利用高置信度预测

---

## 3. 综合分析方法

### 3.1 指标组合分析

#### 3.1.1 性能评估组合
- **Accuracy + F1 + AUC**: 全面评估分类性能
- **Precision + Recall**: 了解错误类型（FP vs FN）
- **Loss趋势**: 判断训练是否收敛

#### 3.1.2 分布分析组合
- **KS + Hellinger + JS**: 多角度评估概率分布分离
- **Bayes Error**: 了解数据本身难度
- **Margin**: 评估预测置信度

#### 3.1.3 嵌入空间分析组合
- **Fisher Ratio**: 综合可分性（Inter/Intra）
- **Bhattacharyya**: 考虑方差的分布距离
- **PCA Energy**: 维度利用效率

### 3.2 训练阶段分析

#### 3.2.1 早期训练（Epoch 1-10）
- **关注**: Loss下降速度、Accuracy提升
- **检查**: 数据加载、模型初始化
- **指标**: Loss, Accuracy, F1

#### 3.2.2 中期训练（Epoch 10-50）
- **关注**: 过拟合迹象、类别平衡
- **检查**: Train/Val指标差异
- **指标**: ECE, Margin, Fisher Ratio

#### 3.2.3 后期训练（Epoch 50+）
- **关注**: 模型稳定性、最佳checkpoint
- **检查**: 所有指标的综合表现
- **指标**: 全部指标，重点关注Bayes Error和Fisher Ratio

### 3.3 问题诊断流程

```
1. 检查基础指标
   ├─ Loss不下降 → 检查学习率、数据、模型
   ├─ Accuracy低 → 检查数据质量、模型容量
   └─ F1低 → 分析Precision/Recall

2. 检查分布指标
   ├─ KS/Hellinger低 → 分布重叠，需要更强特征
   ├─ ECE高 → 校准问题，使用温度缩放
   └─ Bayes Error高 → 数据本身困难

3. 检查嵌入空间
   ├─ Fisher Ratio低 → 类内距离大或类间距离小
   ├─ Intra大 → 同类样本分散
   └─ Inter小 → 类别中心接近

4. 检查异常
   ├─ Outliers多 → 检查标注
   ├─ Hard Samples多 → 增加困难样本训练
   └─ Mislabeled多 → 修正标注
```

---

## 4. 针对分类任务的指标表现与改进

### 4.1 理想状态

| 指标 | 理想值 | 说明 |
|------|--------|------|
| Accuracy | > 0.95 | 整体准确率高 |
| F1 | > 0.95 | 精确率和召回率平衡 |
| AUC | > 0.98 | 区分能力强 |
| KS Distance | > 0.7 | 分布分离良好 |
| Fisher Ratio | > 5 | 类间距离大，类内距离小 |
| ECE | < 0.05 | 校准良好 |
| Bayes Error | < 0.05 | 数据容易区分 |

### 4.2 常见问题与改进

#### 4.2.1 准确率高但F1低

**表现：**
- Accuracy > 0.9, F1 < 0.7
- Precision和Recall不平衡

**原因：**
- 类别不平衡
- 分类阈值不合适

**改进：**
1. 调整分类阈值（使用验证集优化）
2. 使用类别权重（pos_weight）
3. 使用Focal Loss关注难例
4. 数据增强平衡类别

---

#### 4.2.2 训练集好但验证集差（过拟合）

**表现：**
- Train Accuracy > 0.95, Val Accuracy < 0.85
- Train Loss << Val Loss

**原因：**
- 模型容量过大
- 训练数据不足
- 正则化不足

**改进：**
1. 增加Dropout
2. 增加Weight Decay
3. 使用Label Smoothing（已在FocalLabelSmoothCE中）
4. 数据增强
5. Early Stopping
6. 减少模型容量

---

#### 4.2.3 分布重叠严重

**表现：**
- KS Distance < 0.3
- Hellinger < 0.3
- Bayes Error > 0.2

**原因：**
- 特征提取能力不足
- 数据质量差
- 类别本身相似

**改进：**
1. 使用更强的backbone（如ConvNeXt）
2. 使用ArcFace/CosFace损失函数
3. 增加模型容量
4. 数据清洗和标注检查
5. 使用对比学习

---

#### 4.2.4 类内距离大（同类样本分散）

**表现：**
- Intra > Inter
- Fisher Ratio < 1

**原因：**
- 同类样本变化大
- 特征学习不充分

**改进：**
1. 使用中心损失（Center Loss）
2. 使用Triplet Loss
3. 使用ArcFace（已在代码中支持）
4. 增加同类样本的相似性约束

---

#### 4.2.5 类间距离小（不同类别接近）

**表现：**
- Inter < Intra
- Fisher Ratio < 1

**原因：**
- 特征区分能力不足
- 类别边界模糊

**改进：**
1. 使用Margin-based损失（ArcFace）
2. 增加特征维度
3. 使用更强的特征提取器
4. 数据增强增加类别差异

---

#### 4.2.6 校准误差大（ECE高）

**表现：**
- ECE > 0.1
- 预测概率不准确

**原因：**
- 模型过度自信或欠自信
- 损失函数特性

**改进：**
1. 温度缩放（Temperature Scaling）
2. Platt缩放
3. 使用Label Smoothing（已在FocalLabelSmoothCE中）
4. 校准后处理

---

#### 4.2.7 预测不确定（Margin小）

**表现：**
- Margin < 0.1
- 很多样本接近0.5

**原因：**
- 模型训练不充分
- 数据质量差
- 类别边界模糊

**改进：**
1. 增加训练轮数
2. 使用Focal Loss关注难例
3. 数据清洗
4. 增加模型容量

---

#### 4.2.8 信息分散（PCA Energy低）

**表现：**
- 前5个主成分能量 < 0.8

**原因：**
- 嵌入维度冗余
- 特征提取效率低

**改进：**
1. 降维（但需谨慎，可能丢失信息）
2. 使用更有效的特征提取器
3. 特征选择
4. 使用注意力机制

---

### 4.3 改进策略优先级

1. **基础性能（Loss, Accuracy, F1）**
   - 首先确保基础指标正常
   - 如果基础指标差，其他指标意义不大

2. **分布分离（KS, Hellinger, Bayes Error）**
   - 如果分布重叠严重，优先改进特征学习
   - 使用更强的损失函数（ArcFace）

3. **嵌入空间（Fisher Ratio, Intra, Inter）**
   - 如果基础性能好但嵌入空间差，使用对比学习
   - 优化损失函数

4. **校准和置信度（ECE, Margin）**
   - 在性能稳定后优化
   - 使用后处理技术

---

## 5. 使用建议

### 5.1 日常监控

- **每个epoch**: 关注Loss, Accuracy, F1
- **每10个epoch**: 查看所有分布和嵌入指标
- **每50个epoch**: 检查可视化（t-SNE, PCA）和异常检测

### 5.2 问题定位

1. 先看基础指标（Loss, Accuracy, F1）
2. 再看分布指标（KS, Hellinger, Bayes Error）
3. 最后看嵌入空间（Fisher Ratio, Intra, Inter）
4. 结合可视化（t-SNE, PCA）和异常检测

### 5.3 模型选择

- **Best Checkpoint**: 基于F1 Score选择
- **综合评估**: 结合Accuracy, F1, AUC, Fisher Ratio
- **稳定性**: 检查多个epoch的指标稳定性

---

## 6. 附录：指标计算公式汇总

### 6.1 基础分类指标

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### 6.2 分布指标

```
KS = max |CDF₀(x) - CDF₁(x)|
Hellinger = √(1 - Σ√(P₀(i) × P₁(i)))
JS = 0.5 × KL(P||M) + 0.5 × KL(Q||M), M = 0.5(P + Q)
ECE = Σ (|Bₘ|/n) × |acc(Bₘ) - conf(Bₘ)|
Margin = mean(|p - 0.5|)
Bayes Error = 0.5 × overlap(P₀, P₁)
```

### 6.3 嵌入空间指标

```
Intra = 0.5 × (mean(||x₀ - μ₀||²) + mean(||x₁ - μ₁||²))
Inter = ||μ₀ - μ₁||
Fisher = Inter / (Intra + ε)
BC = 0.25 × Σ log(0.25 × (σ₀/σ₁ + σ₁/σ₀ + 2)) + 0.25 × Σ (μ₀-μ₁)²/(σ₀+σ₁)
PCA Energy_i = λᵢ / Σλⱼ
```

---

**文档版本**: 1.0  
**最后更新**: 2025-12-01  
**维护者**: OCEC项目组

