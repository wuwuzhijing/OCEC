#!/usr/bin/env python
"""检查数据划分方式：是否按视频划分"""

import argparse
from pathlib import Path
from collections import Counter

import pandas as pd


def check_split_by_video(data_root: Path):
    """检查训练集和验证集是否有视频重叠"""
    
    # 加载所有parquet文件
    parquet_files = sorted(data_root.glob("*.parquet"))
    if not parquet_files:
        print(f"❌ 在 {data_root} 中未找到parquet文件")
        return
    
    print(f"📁 找到 {len(parquet_files)} 个parquet文件")
    
    all_dfs = []
    for f in parquet_files:
        df = pd.read_parquet(f)
        all_dfs.append(df)
        print(f"  - {f.name}: {len(df)} 行")
    
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n📊 总样本数: {len(df)}")
    
    # 检查split列
    if "split" not in df.columns:
        print("❌ 数据中没有'split'列")
        return
    
    # 检查source或video_name列
    video_col = None
    if "source" in df.columns:
        video_col = "source"
    elif "video_name" in df.columns:
        video_col = "video_name"
    else:
        print("⚠️  数据中没有'source'或'video_name'列，无法检查视频重叠")
        print("   可用列:", df.columns.tolist())
        return
    
    # 按split分组
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"] if "test" in df["split"].values else None
    
    print(f"\n📈 数据划分统计:")
    print(f"  训练集: {len(train_df)} 样本")
    print(f"  验证集: {len(val_df)} 样本")
    if test_df is not None and len(test_df) > 0:
        print(f"  测试集: {len(test_df)} 样本")
    
    # 检查视频重叠
    train_videos = set(train_df[video_col].unique())
    val_videos = set(val_df[video_col].unique())
    overlap = train_videos & val_videos
    
    print(f"\n🎬 视频统计:")
    print(f"  训练集视频数: {len(train_videos)}")
    print(f"  验证集视频数: {len(val_videos)}")
    print(f"  重叠的视频数: {len(overlap)}")
    
    if overlap:
        print(f"\n⚠️  警告：训练集和验证集包含 {len(overlap)} 个相同的视频！")
        print(f"   这表明数据是按样本随机划分的，而不是按视频划分的。")
        print(f"\n   重叠的视频示例（前10个）:")
        for i, video in enumerate(list(overlap)[:10], 1):
            train_count = len(train_df[train_df[video_col] == video])
            val_count = len(val_df[val_df[video_col] == video])
            print(f"     {i}. {video}: 训练集{train_count}帧, 验证集{val_count}帧")
        if len(overlap) > 10:
            print(f"     ... 还有 {len(overlap) - 10} 个重叠的视频")
        
        print(f"\n💡 建议：")
        print(f"   1. 重新划分数据，使用按视频划分的方式")
        print(f"   2. 这可以避免数据泄漏，提高验证集性能的真实性")
    else:
        print(f"\n✅ 训练集和验证集没有重叠的视频")
        print(f"   数据已按视频划分，这是推荐的方式！")
    
    # 检查标签分布
    print(f"\n📊 标签分布:")
    for split_name, split_df in [("训练集", train_df), ("验证集", val_df)]:
        if len(split_df) > 0:
            label_counts = split_df["label"].value_counts().sort_index()
            total = len(split_df)
            pos_count = label_counts.get(1, 0)
            neg_count = label_counts.get(0, 0)
            pos_ratio = (pos_count / total * 100) if total > 0 else 0
            print(f"  {split_name}: 正类={pos_count} ({pos_ratio:.1f}%), 负类={neg_count} ({100-pos_ratio:.1f}%)")
    
    # 检查每个视频的样本数分布
    if video_col:
        print(f"\n📹 每个视频的样本数统计:")
        video_counts = df.groupby(video_col).size()
        print(f"  视频总数: {len(video_counts)}")
        print(f"  平均每视频样本数: {video_counts.mean():.1f}")
        print(f"  最小样本数: {video_counts.min()}")
        print(f"  最大样本数: {video_counts.max()}")
        print(f"  中位数: {video_counts.median():.1f}")


def main():
    parser = argparse.ArgumentParser(description="检查数据划分方式")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/ssddisk/guochuang/ocec/parquet_hq"),
        help="Parquet数据目录路径",
    )
    args = parser.parse_args()
    
    if not args.data_root.exists():
        print(f"❌ 目录不存在: {args.data_root}")
        return
    
    check_split_by_video(args.data_root)


if __name__ == "__main__":
    main()

