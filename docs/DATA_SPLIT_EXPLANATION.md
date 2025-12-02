# 数据划分方式说明：按样本划分 vs 按视频划分

## 1. 当前划分方式：按样本随机划分

### 工作原理

当前代码（`04_dataset_convert_to_parquet.py` 中的 `stratified_split` 函数）是这样划分的：

```python
for label, group in df.groupby("label"):  # 按标签分组（睁眼/闭眼）
    records = list(group.to_dict(orient="records"))
    rng.shuffle(records)  # 随机打乱所有样本
    n_train = int(n_total * train_ratio)
    train_subset = records[:n_train]      # 前80% → 训练集
    val_subset = records[n_train:]        # 后20% → 验证集
```

### 示例

假设有一个视频 `video_001.mp4`，从中提取了1000帧：
- 帧1-1000 都来自 `video_001.mp4`

**按样本随机划分的结果：**
- 训练集：可能包含 `video_001` 的帧 1, 5, 23, 100, 200, ...（随机选择）
- 验证集：可能包含 `video_001` 的帧 2, 10, 50, 150, 300, ...（随机选择）

**问题：**
- ❌ 同一个视频的帧被分到了训练集和验证集
- ❌ 训练时模型可能"记住"了验证集中视频的特征
- ❌ 验证集性能可能被高估（数据泄漏）
- ❌ 训练集和验证集的数据分布可能不一致

---

## 2. 按视频划分（推荐方式）

### 工作原理

按视频划分是指：**整个视频的所有帧都分到同一个集合（训练集或验证集）**。

```python
# 伪代码
for video_name, video_group in df.groupby("video_name"):  # 按视频分组
    # 整个视频 → 要么全部训练集，要么全部验证集
    if random() < train_ratio:
        train_rows.extend(video_group)  # 整个视频 → 训练集
    else:
        val_rows.extend(video_group)    # 整个视频 → 验证集
```

### 示例

假设有3个视频：
- `video_001.mp4`：1000帧
- `video_002.mp4`：800帧
- `video_003.mp4`：1200帧

**按视频划分的结果（train_ratio=0.8）：**
- 训练集：`video_001` 的所有1000帧 + `video_002` 的所有800帧 = 1800帧
- 验证集：`video_003` 的所有1200帧 = 1200帧

**优势：**
- ✅ 同一个视频的所有帧都在同一个集合
- ✅ 避免数据泄漏
- ✅ 更真实的验证性能
- ✅ 训练集和验证集的数据分布更一致

---

## 3. 为什么会出现训练集和验证集recall差异大？

### 问题现象

从你的日志看：
- **训练集**：pred_pos_ratio=0.0602（只有6%预测为正类）→ recall=0.0654
- **验证集**：pred_pos_ratio=0.9219（92%预测为正类）→ recall=0.9165

### 可能的原因

1. **数据分布不一致**
   - 如果按样本随机划分，某些"容易"的视频可能被分到验证集
   - 某些"困难"的视频可能被分到训练集
   - 导致训练集和验证集的难度不同

2. **数据泄漏**
   - 同一个视频的帧在训练集和验证集都有
   - 模型在训练时"见过"验证集的视频特征
   - 导致验证集性能被高估

3. **视频特性差异**
   - 不同视频可能有不同的：
     - 光照条件
     - 拍摄角度
     - 人物特征
     - 背景环境
   - 如果这些视频被随机划分，训练集和验证集的分布可能不同

---

## 4. 如何实现按视频划分？

### 方案1：修改数据划分脚本（推荐）

修改 `04_dataset_convert_to_parquet.py` 中的 `stratified_split` 函数：

```python
def stratified_split_by_video(
    df: pd.DataFrame,
    train_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按视频划分，确保同一个视频的所有帧都在同一个集合"""
    import random
    rng = random.Random(seed)
    
    # 按视频分组
    video_groups = {}
    for video_name, group in df.groupby("source"):  # 或 "video_name"
        video_groups[video_name] = group
    
    # 随机打乱视频列表
    video_list = list(video_groups.keys())
    rng.shuffle(video_list)
    
    # 按比例划分视频
    n_videos = len(video_list)
    n_train_videos = int(n_videos * train_ratio)
    
    train_videos = set(video_list[:n_train_videos])
    val_videos = set(video_list[n_train_videos:])
    
    # 收集训练集和验证集的样本
    train_rows = []
    val_rows = []
    
    for video_name, group in video_groups.items():
        if video_name in train_videos:
            train_rows.extend(group.to_dict(orient="records"))
        else:
            val_rows.extend(group.to_dict(orient="records"))
    
    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    train_df["split"] = "train"
    val_df["split"] = "val"
    
    return train_df, val_df
```

### 方案2：检查现有数据

如果数据已经划分好了，可以检查一下：

```python
# 检查训练集和验证集中是否有相同的视频
train_videos = set(train_df["source"].unique())
val_videos = set(val_df["source"].unique())
overlap = train_videos & val_videos
print(f"重叠的视频数量: {len(overlap)}")
if overlap:
    print("警告：训练集和验证集包含相同的视频！")
```

---

## 5. 总结

| 特性 | 按样本随机划分 | 按视频划分 |
|------|---------------|-----------|
| **划分单位** | 单个样本（帧） | 整个视频 |
| **数据泄漏风险** | ⚠️ 高（同一视频可能分到两个集合） | ✅ 低（同一视频只在一个集合） |
| **验证性能真实性** | ⚠️ 可能被高估 | ✅ 更真实 |
| **实现复杂度** | ✅ 简单 | ⚠️ 稍复杂 |
| **适用场景** | 独立样本（如照片） | 视频帧序列 |

### 建议

对于你的任务（眼睛状态分类，数据来自视频）：
- ✅ **推荐使用按视频划分**
- ✅ 可以避免数据泄漏
- ✅ 验证集性能更真实
- ✅ 训练集和验证集分布更一致

---

## 6. 如何检查当前数据是否按视频划分？

运行以下代码检查：

```python
import pandas as pd
from pathlib import Path

# 加载parquet文件
data_root = Path("/ssddisk/guochuang/ocec/parquet_hq")
parquet_files = list(data_root.glob("*.parquet"))

all_dfs = []
for f in parquet_files:
    df = pd.read_parquet(f)
    all_dfs.append(df)

df = pd.concat(all_dfs, ignore_index=True)

# 检查训练集和验证集的视频重叠
train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "val"]

train_videos = set(train_df["source"].unique())
val_videos = set(val_df["source"].unique())
overlap = train_videos & val_videos

print(f"训练集视频数: {len(train_videos)}")
print(f"验证集视频数: {len(val_videos)}")
print(f"重叠的视频数: {len(overlap)}")

if overlap:
    print(f"\n⚠️ 警告：以下视频同时出现在训练集和验证集：")
    for video in list(overlap)[:10]:  # 只显示前10个
        print(f"  - {video}")
    if len(overlap) > 10:
        print(f"  ... 还有 {len(overlap) - 10} 个")
else:
    print("\n✅ 训练集和验证集没有重叠的视频（已按视频划分）")
```

