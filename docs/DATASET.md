# 数据集说明文档

## 数据集概览

本文档记录了OCEC数据集的详细统计信息。

## CSV标注文件统计

### 文件分布

| 文件范围 | 总行数 | 标签0 (闭眼) | 标签1 (睁眼) |
|---------|--------|-------------|-------------|
| annotation_0001-0023 | 259,869 | 148,485 | 111,384 |
| annotation_0024 | 470,811 | 220,549 | 250,262 |
| **总计** | **730,680** | **369,034** | **361,646** |

### 标签分布说明

- **标签0**: 闭眼（closed eyes）
- **标签1**: 睁眼（open eyes）

## 图片数据统计

### /ssddisk/guochuang/ocec/public 目录

该目录包含从annotation_0001.csv到annotation_0023.csv对应的图片数据。

| 维度 | 平均值 | 中位值 | 最小值 | 最大值 | 标准差 |
|-----|--------|--------|--------|--------|--------|
| 宽度 | 43.50 | 42.00 | 10 | 80 | 13.44 |
| 高度 | 18.66 | 17.00 | 5 | 50 | 6.99 |

**总图片数**: 520,154

**异常值说明**:
- 宽度最小值 (10px) 对应 265 张图片
- 宽度最大值 (80px) 对应 865 张图片
- 高度最小值 (5px) 对应 433 张图片
- 高度最大值 (50px) 对应 103 张图片

详细路径请查看 `extreme_image_paths.txt` 文件。

### /ssddisk/guochuang/ocec/hq 目录

该目录包含高质量图片数据，对应annotation_0024.csv。

| 维度 | 平均值 | 中位值 | 最小值 | 最大值 | 标准差 |
|-----|--------|--------|--------|--------|--------|
| 宽度 | 58.98 | 59.00 | 9 | 180 | 12.21 |
| 高度 | 34.87 | 34.00 | 9 | 141 | 8.77 |

**总图片数**: 471,553

**异常值说明**:
- 宽度最小值 (9px) 对应 1 张图片
- 宽度最大值 (180px) 对应 1 张图片
- 高度最小值 (9px) 对应 1 张图片
- 高度最大值 (141px) 对应 1 张图片

详细路径请查看 `extreme_image_paths.txt` 文件。

## 数据集特点

1. **public目录**: 图片尺寸相对较小，宽度中位值42像素，高度中位值17像素，适合快速处理。
2. **hq目录**: 图片尺寸较大，宽度中位值59像素，高度中位值34像素，提供更高质量的图像数据。
3. **标签平衡**: 整体数据集标签分布相对均衡，闭眼和睁眼样本数量接近。

## 数据路径说明

- CSV标注文件位置: `/ssddisk/guochuang/ocec/list/`
- public图片路径: `/ssddisk/guochuang/ocec/public/cropped/`
- hq图片路径: `/ssddisk/guochuang/ocec/hq/`

## 相关文件

- 详细统计结果: `dataset_statistics.txt`
- 异常值图片路径: `extreme_image_paths.txt`

## v4.3 训练所用数据集（5个）

### 汇总

| 数据集 | 数量 | 尺寸 | 模式 | 类别 | 路径 |
|--------|------|------|------|------|------|
| MRL Eye | 84,898 | 58-219×58-219 | 灰度(L) | 2类(open/closed) | `/10/cvz/guochuang/dataset/MRL-Eye-Dataset/mrlEyes_2018_01/` |
| OCEC existing | 11,556 | 9-180×9-141 | RGB | 2类(open/closed) | `data/cropped/100000001/` (嵌入 parquet) |
| fatigue src | 169,630 | **24×24** | RGB | 2类(0_normal/1_squint) | `/103/gupengli/dataset/fatigue_dataset/src_dataset/` |
| fatigue L2+ | 2,535 | 18-74×18-74 | L+RGB混合 | 3类(+2_unknown) | `/103/gupengli/dataset/fatigue_dataset/L2+_dataset/` |
| fatigue futian | 6,571 | **24×24** | RGB | 4类(含3_wrong/2_half) | `/103/gupengli/dataset/fatigue_dataset/futian_dataset/` |

### 详细

#### 1. MRL Eye Dataset

- **路径**: `/10/cvz/guochuang/dataset/MRL-Eye-Dataset/mrlEyes_2018_01/`
- **数量**: 84,898 张
- **尺寸**: 58-219×58-219 像素（正方形），以 78×78 为中心分布
- **模式**: 灰度 (L) → 训练时转 RGB
- **标注**: 文件名第5字段 (sXXXX_NNNNN_GENDER_GLASSES_**EYESTATE**_REFLECTIONS_LIGHTING_SENSOR.png)
  - `0` = closed (41,946) / `1` = open (42,952)
- **划分**: 按 subject (37人) 分层，train 29人 / val 8人
- **特点**: 完整眼眶区域，含眼镜、眉毛等上下文；ROI 较大

#### 2. OCEC existing

- **路径**: 嵌入 `data/dataset.parquet` (image_bytes)
- **数量**: 11,556 张
- **尺寸**: 9-180×9-141 像素，宽中位 ~43px，高中位 ~19px
- **模式**: RGB (PNG bytes)
- **标注**: closed (11,332) / open (224)，极不平衡 (98:2)
- **划分**: 按时间顺序 (连续 frame 不跨 train/val)，train 9,246 / val 2,310
- **特点**: 单段录像的眼部 crop，ROI 较小

#### 3. fatigue src_dataset

- **路径**: `/103/gupengli/dataset/fatigue_dataset/src_dataset/`
- **数量**: 169,630 张
- **尺寸**: 固定 24×24，RGB
- **类别**: `0_normal` (109,588) / `1_squint` (60,042) ≈ 1.8:1
- **映射**: `0_normal` → open / `1_squint` → closed
- **划分**: 随机 90/10 (train/val)
- **特点**: 仅眼球区域，极小 ROI，无上下文

#### 4. fatigue L2+_dataset

- **路径**: `/103/gupengli/dataset/fatigue_dataset/L2+_dataset/`
- **数量**: 2,535 张
- **尺寸**: 18-74×18-74 像素，混合尺寸
- **模式**: 灰度+RGB 混合 (train 时统一 `.convert("RGB")`)
- **类别**: `0_normal` (1,046) / `1_squint` (180) / `2_unknown` (1,309)
  - `2_unknown` → **跳过**（不确定类，样本少且模糊）
- **特点**: 混合尺寸和质量，需 resize 到统一尺寸

#### 5. fatigue futian_dataset

- **路径**: `/103/gupengli/dataset/fatigue_dataset/futian_dataset/`
- **数量**: 6,571 张
- **尺寸**: 固定 24×24，RGB
- **类别**: `0_normal` (5,913) / `1_squint` (373) / `2_half` (280) / `3_wrong` (5)
  - 仅使用 `0_normal` 和 `1_squint`，其余跳过

### 训练集合并

v4.3_fatigue 训练使用上述 5 个数据集的并集：

```
总量: ~275K (MRL 85K + OCEC 12K + fatigue_src 170K + L2+ 2K + futian 7K)
open (睁眼): ~154K
closed (闭眼): ~121K
比例: open:closed ≈ 1.27:1
```

转换脚本: `scripts/convert_fatigue_to_parquet.py`
合并训练脚本: `train_mrl_v4.3_fatigue.sh`

## 更新记录

- 自动更新: 统计完成后自动更新

## 数据集筛选记录

- /ssddisk/guochuang/ocec/hq
1. 去掉有文字遮挡的图片；
2. 去掉亮度、清晰度异常的图片；