# 单独训练MRL数据集
## 数据集
- 数据来源: /10/cvz/guochuang/dataset/MRL-Eye-Dataset/mrlEyes_2018_01/
- 输出位置: /10/cvz/guochuang/dataset/mrl_eyes_2018/dataset.parquet (465.5MB)
- 转换脚本: scripts/convert_mrl_to_parquet.py
- 84,898 张图片，从文件名解析标签（第5字段=eye state: 0=闭眼, 1=睁眼）
- 按 subject 分层划分：29人训练 (73,313张) + 8人验证 (11,585张)
- 标签分布：closed 41,946 (49.4%) / open 42,952 (50.6%)
## 训练
1. 训练脚本

- 脚本: train_mrl.sh
- 输出目录: runs/ocec_mrl/
- 200 epochs, batch_size 1024, cosface margin, 32x64 输入
- 使用 mmpose conda 环境

2. 训练状态

- 当前日志: logs/train/mrl/train_mrl_20260613_152657.log
- TensorBoard: tensorboard --logdir runs/ocec_mrl (http://localhost:6006)
- Epoch 1 已完成，模型正在训练中

### 查看进度
tail -f logs/train/mrl/train_mrl_20260613_152657.log

### 只看 epoch 摘要
grep "Epoch " logs/train/mrl/train_mrl_20260613_152657.log

# 混合训练
1. 数据集合并 — scripts/merge_datasets_to_parquet.py

┌───────────────────────────┬────────┬───────────────────────┐
│           来源            │ 样本数 │       标签分布        │
├───────────────────────────┼────────┼───────────────────────┤
│ Existing OCEC (real_data) │ 11,556 │ 98% closed            │
├───────────────────────────┼────────┼───────────────────────┤
│ MRL Eye Dataset           │ 84,898 │ 51% open / 49% closed │
├───────────────────────────┼────────┼───────────────────────┤
│ 合并总计                  │ 96,454 │ 55% closed / 45% open │
└───────────────────────────┴────────┴───────────────────────┘

合并后数据：/10/cvz/guochuang/dataset/ocec_combined/dataset.parquet (476MB)

2. 训练对比

┌──────┬────────────────────────┬────────────────────────┐
│      │        MRL 独立        │ 联合 (Existing + MRL)  │
├──────┼────────────────────────┼────────────────────────┤
│ 脚本 │ train_mrl.sh           │ train_combined.sh      │
├──────┼────────────────────────┼────────────────────────┤
│ 数据 │ /10/.../mrl_eyes_2018/ │ /10/.../ocec_combined/ │
├──────┼────────────────────────┼────────────────────────┤
│ 输出 │ runs/ocec_mrl/         │ runs/ocec_combined/    │
├──────┼────────────────────────┼────────────────────────┤
│ 日志 │ logs/train/mrl/        │ logs/train/combined/   │
├──────┼────────────────────────┼────────────────────────┤
│ TB   │ localhost:6006         │ localhost:6007         │
└──────┴────────────────────────┴────────────────────────┘

3. 文件结构

scripts/
├── convert_mrl_to_parquet.py     # MRL → parquet 转换
├── merge_datasets_to_parquet.py  # 两数据集合并
train_mrl.sh                      # MRL 独立训练
train_combined.sh                 # 联合训练（自动检测合并数据是否存在）