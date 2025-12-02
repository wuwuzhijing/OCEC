#!/usr/bin/env python3
"""
统计数据集信息：
1. CSV文件行数和标签分布
2. 图片尺寸统计
"""
import os
import csv
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from tqdm import tqdm

def count_csv_rows_and_labels(csv_dir, start_idx=1, end_idx=23):
    """统计CSV文件的行数和标签分布"""
    results = {}
    
    # 统计 annotation_0001.csv 到 annotation_0023.csv
    total_rows_1_23 = 0
    label_0_count_1_23 = 0
    label_1_count_1_23 = 0
    
    for i in range(start_idx, end_idx + 1):
        csv_file = csv_dir / f"annotation_{i:04d}.csv"
        if not csv_file.exists():
            print(f"警告: {csv_file} 不存在")
            continue
            
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            total_rows = len(rows)
            total_rows_1_23 += total_rows
            
            label_0 = sum(1 for row in rows if len(row) > 1 and row[1] == '0')
            label_1 = sum(1 for row in rows if len(row) > 1 and row[1] == '1')
            label_0_count_1_23 += label_0
            label_1_count_1_23 += label_1
    
    results['annotation_0001_0023'] = {
        'total_rows': total_rows_1_23,
        'label_0': label_0_count_1_23,
        'label_1': label_1_count_1_23
    }
    
    # 统计 annotation_0024.csv
    csv_file_24 = csv_dir / "annotation_0024.csv"
    if csv_file_24.exists():
        with open(csv_file_24, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            total_rows_24 = len(rows)
            label_0_24 = sum(1 for row in rows if len(row) > 1 and row[1] == '0')
            label_1_24 = sum(1 for row in rows if len(row) > 1 and row[1] == '1')
        
        results['annotation_0024'] = {
            'total_rows': total_rows_24,
            'label_0': label_0_24,
            'label_1': label_1_24
        }
    else:
        print(f"警告: {csv_file_24} 不存在")
        results['annotation_0024'] = {
            'total_rows': 0,
            'label_0': 0,
            'label_1': 0
        }
    
    return results

def get_image_size_stats(image_dir):
    """统计图片尺寸"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    sizes = []
    widths = []
    heights = []
    width_paths = []  # 存储每个宽度对应的路径
    height_paths = []  # 存储每个高度对应的路径
    
    print(f"正在扫描 {image_dir} 目录...")
    image_files = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    print(f"找到 {len(image_files)} 张图片，正在统计尺寸...")
    
    failed_count = 0
    for img_path in tqdm(image_files, desc="处理图片"):
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                sizes.append((width, height))
                widths.append(width)
                heights.append(height)
                width_paths.append((width, img_path))
                height_paths.append((height, img_path))
        except Exception as e:
            failed_count += 1
            continue
    
    if failed_count > 0:
        print(f"警告: {failed_count} 张图片无法读取")
    
    if not sizes:
        return {
            'total_images': 0,
            'width': {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0, 'min_paths': [], 'max_paths': []},
            'height': {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0, 'min_paths': [], 'max_paths': []}
        }
    
    widths = np.array(widths)
    heights = np.array(heights)
    
    # 找出最小值和最大值对应的所有图片路径
    min_width = int(np.min(widths))
    max_width = int(np.max(widths))
    min_height = int(np.min(heights))
    max_height = int(np.max(heights))
    
    min_width_paths = [path for w, path in width_paths if w == min_width]
    max_width_paths = [path for w, path in width_paths if w == max_width]
    min_height_paths = [path for h, path in height_paths if h == min_height]
    max_height_paths = [path for h, path in height_paths if h == max_height]
    
    return {
        'total_images': len(sizes),
        'width': {
            'mean': float(np.mean(widths)),
            'median': float(np.median(widths)),
            'min': min_width,
            'max': max_width,
            'std': float(np.std(widths)),
            'min_paths': min_width_paths,
            'max_paths': max_width_paths
        },
        'height': {
            'mean': float(np.mean(heights)),
            'median': float(np.median(heights)),
            'min': min_height,
            'max': max_height,
            'std': float(np.std(heights)),
            'min_paths': min_height_paths,
            'max_paths': max_height_paths
        }
    }

def main():
    csv_dir = Path("/ssddisk/guochuang/ocec/list")
    public_dir = Path("/ssddisk/guochuang/ocec/public")
    hq_dir = Path("/ssddisk/guochuang/ocec/hq")
    
    print("=" * 60)
    print("开始统计数据集信息...")
    print("=" * 60)
    
    # 1. 统计CSV文件
    print("\n1. 统计CSV文件行数和标签分布...")
    csv_stats = count_csv_rows_and_labels(csv_dir, 1, 23)
    
    print(f"\nannotation_0001.csv - annotation_0023.csv:")
    print(f"  总行数: {csv_stats['annotation_0001_0023']['total_rows']:,}")
    print(f"  标签0 (闭眼): {csv_stats['annotation_0001_0023']['label_0']:,}")
    print(f"  标签1 (睁眼): {csv_stats['annotation_0001_0023']['label_1']:,}")
    
    print(f"\nannotation_0024.csv:")
    print(f"  总行数: {csv_stats['annotation_0024']['total_rows']:,}")
    print(f"  标签0 (闭眼): {csv_stats['annotation_0024']['label_0']:,}")
    print(f"  标签1 (睁眼): {csv_stats['annotation_0024']['label_1']:,}")
    
    # 2. 统计图片尺寸
    print("\n2. 统计图片尺寸...")
    print("\n统计 /ssddisk/guochuang/ocec/public 目录...")
    public_stats = get_image_size_stats(public_dir)
    
    print("\n统计 /ssddisk/guochuang/ocec/hq 目录...")
    hq_stats = get_image_size_stats(hq_dir)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("统计结果汇总")
    print("=" * 60)
    
    print("\n### CSV文件统计")
    print(f"| 文件范围 | 总行数 | 标签0 (闭眼) | 标签1 (睁眼) |")
    print(f"|---------|--------|-------------|-------------|")
    print(f"| annotation_0001-0023 | {csv_stats['annotation_0001_0023']['total_rows']:,} | {csv_stats['annotation_0001_0023']['label_0']:,} | {csv_stats['annotation_0001_0023']['label_1']:,} |")
    print(f"| annotation_0024 | {csv_stats['annotation_0024']['total_rows']:,} | {csv_stats['annotation_0024']['label_0']:,} | {csv_stats['annotation_0024']['label_1']:,} |")
    
    print("\n### 图片尺寸统计 - /ssddisk/guochuang/ocec/public")
    print(f"| 维度 | 平均值 | 中位值 | 最小值 | 最大值 | 标准差 |")
    print(f"|-----|--------|--------|--------|--------|--------|")
    print(f"| 宽度 | {public_stats['width']['mean']:.2f} | {public_stats['width']['median']:.2f} | {public_stats['width']['min']} | {public_stats['width']['max']} | {public_stats['width']['std']:.2f} |")
    print(f"| 高度 | {public_stats['height']['mean']:.2f} | {public_stats['height']['median']:.2f} | {public_stats['height']['min']} | {public_stats['height']['max']} | {public_stats['height']['std']:.2f} |")
    print(f"总图片数: {public_stats['total_images']:,}")
    
    print("\n### 图片尺寸统计 - /ssddisk/guochuang/ocec/hq")
    print(f"| 维度 | 平均值 | 中位值 | 最小值 | 最大值 | 标准差 |")
    print(f"|-----|--------|--------|--------|--------|--------|")
    print(f"| 宽度 | {hq_stats['width']['mean']:.2f} | {hq_stats['width']['median']:.2f} | {hq_stats['width']['min']} | {hq_stats['width']['max']} | {hq_stats['width']['std']:.2f} |")
    print(f"| 高度 | {hq_stats['height']['mean']:.2f} | {hq_stats['height']['median']:.2f} | {hq_stats['height']['min']} | {hq_stats['height']['max']} | {hq_stats['height']['std']:.2f} |")
    print(f"总图片数: {hq_stats['total_images']:,}")
    
    # 保存结果到文件
    result_file = Path("/103/guochuang/Code/myOCEC/dataset_statistics.txt")
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("数据集统计结果\n")
        f.write("=" * 60 + "\n\n")
        f.write("### CSV文件统计\n")
        f.write(f"| 文件范围 | 总行数 | 标签0 (闭眼) | 标签1 (睁眼) |\n")
        f.write(f"|---------|--------|-------------|-------------|\n")
        f.write(f"| annotation_0001-0023 | {csv_stats['annotation_0001_0023']['total_rows']:,} | {csv_stats['annotation_0001_0023']['label_0']:,} | {csv_stats['annotation_0001_0023']['label_1']:,} |\n")
        f.write(f"| annotation_0024 | {csv_stats['annotation_0024']['total_rows']:,} | {csv_stats['annotation_0024']['label_0']:,} | {csv_stats['annotation_0024']['label_1']:,} |\n\n")
        
        f.write("### 图片尺寸统计 - /ssddisk/guochuang/ocec/public\n")
        f.write(f"| 维度 | 平均值 | 中位值 | 最小值 | 最大值 | 标准差 |\n")
        f.write(f"|-----|--------|--------|--------|--------|--------|\n")
        f.write(f"| 宽度 | {public_stats['width']['mean']:.2f} | {public_stats['width']['median']:.2f} | {public_stats['width']['min']} | {public_stats['width']['max']} | {public_stats['width']['std']:.2f} |\n")
        f.write(f"| 高度 | {public_stats['height']['mean']:.2f} | {public_stats['height']['median']:.2f} | {public_stats['height']['min']} | {public_stats['height']['max']} | {public_stats['height']['std']:.2f} |\n")
        f.write(f"总图片数: {public_stats['total_images']:,}\n\n")
        
        f.write("### 图片尺寸统计 - /ssddisk/guochuang/ocec/hq\n")
        f.write(f"| 维度 | 平均值 | 中位值 | 最小值 | 最大值 | 标准差 |\n")
        f.write(f"|-----|--------|--------|--------|--------|--------|\n")
        f.write(f"| 宽度 | {hq_stats['width']['mean']:.2f} | {hq_stats['width']['median']:.2f} | {hq_stats['width']['min']} | {hq_stats['width']['max']} | {hq_stats['width']['std']:.2f} |\n")
        f.write(f"| 高度 | {hq_stats['height']['mean']:.2f} | {hq_stats['height']['median']:.2f} | {hq_stats['height']['min']} | {hq_stats['height']['max']} | {hq_stats['height']['std']:.2f} |\n")
        f.write(f"总图片数: {hq_stats['total_images']:,}\n")
    
    print(f"\n统计结果已保存到: {result_file}")
    
    # 保存最小值和最大值对应的图片路径
    extreme_paths_file = Path("/103/guochuang/Code/myOCEC/extreme_image_paths.txt")
    with open(extreme_paths_file, 'w', encoding='utf-8') as f:
        f.write("最小值和最大值对应的图片路径\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("## /ssddisk/guochuang/ocec/public 目录\n\n")
        f.write(f"### 宽度最小值 ({public_stats['width']['min']}px) 对应的图片:\n")
        for path in public_stats['width']['min_paths']:
            f.write(f"{path}\n")
        f.write(f"\n共 {len(public_stats['width']['min_paths'])} 张图片\n\n")
        
        f.write(f"### 宽度最大值 ({public_stats['width']['max']}px) 对应的图片:\n")
        for path in public_stats['width']['max_paths']:
            f.write(f"{path}\n")
        f.write(f"\n共 {len(public_stats['width']['max_paths'])} 张图片\n\n")
        
        f.write(f"### 高度最小值 ({public_stats['height']['min']}px) 对应的图片:\n")
        for path in public_stats['height']['min_paths']:
            f.write(f"{path}\n")
        f.write(f"\n共 {len(public_stats['height']['min_paths'])} 张图片\n\n")
        
        f.write(f"### 高度最大值 ({public_stats['height']['max']}px) 对应的图片:\n")
        for path in public_stats['height']['max_paths']:
            f.write(f"{path}\n")
        f.write(f"\n共 {len(public_stats['height']['max_paths'])} 张图片\n\n")
        
        f.write("\n" + "=" * 60 + "\n\n")
        f.write("## /ssddisk/guochuang/ocec/hq 目录\n\n")
        f.write(f"### 宽度最小值 ({hq_stats['width']['min']}px) 对应的图片:\n")
        for path in hq_stats['width']['min_paths']:
            f.write(f"{path}\n")
        f.write(f"\n共 {len(hq_stats['width']['min_paths'])} 张图片\n\n")
        
        f.write(f"### 宽度最大值 ({hq_stats['width']['max']}px) 对应的图片:\n")
        for path in hq_stats['width']['max_paths']:
            f.write(f"{path}\n")
        f.write(f"\n共 {len(hq_stats['width']['max_paths'])} 张图片\n\n")
        
        f.write(f"### 高度最小值 ({hq_stats['height']['min']}px) 对应的图片:\n")
        for path in hq_stats['height']['min_paths']:
            f.write(f"{path}\n")
        f.write(f"\n共 {len(hq_stats['height']['min_paths'])} 张图片\n\n")
        
        f.write(f"### 高度最大值 ({hq_stats['height']['max']}px) 对应的图片:\n")
        for path in hq_stats['height']['max_paths']:
            f.write(f"{path}\n")
        f.write(f"\n共 {len(hq_stats['height']['max_paths'])} 张图片\n")
    
    print(f"最小值和最大值对应的图片路径已保存到: {extreme_paths_file}")
    
    # 自动更新DATASET.md文件
    dataset_md_file = Path("/103/guochuang/Code/myOCEC/DATASET.md")
    dataset_md_content = f"""# 数据集说明文档

## 数据集概览

本文档记录了OCEC数据集的详细统计信息。

## CSV标注文件统计

### 文件分布

| 文件范围 | 总行数 | 标签0 (闭眼) | 标签1 (睁眼) |
|---------|--------|-------------|-------------|
| annotation_0001-0023 | {csv_stats['annotation_0001_0023']['total_rows']:,} | {csv_stats['annotation_0001_0023']['label_0']:,} | {csv_stats['annotation_0001_0023']['label_1']:,} |
| annotation_0024 | {csv_stats['annotation_0024']['total_rows']:,} | {csv_stats['annotation_0024']['label_0']:,} | {csv_stats['annotation_0024']['label_1']:,} |
| **总计** | **{csv_stats['annotation_0001_0023']['total_rows'] + csv_stats['annotation_0024']['total_rows']:,}** | **{csv_stats['annotation_0001_0023']['label_0'] + csv_stats['annotation_0024']['label_0']:,}** | **{csv_stats['annotation_0001_0023']['label_1'] + csv_stats['annotation_0024']['label_1']:,}** |

### 标签分布说明

- **标签0**: 闭眼（closed eyes）
- **标签1**: 睁眼（open eyes）

## 图片数据统计

### /ssddisk/guochuang/ocec/public 目录

该目录包含从annotation_0001.csv到annotation_0023.csv对应的图片数据。

| 维度 | 平均值 | 中位值 | 最小值 | 最大值 | 标准差 |
|-----|--------|--------|--------|--------|--------|
| 宽度 | {public_stats['width']['mean']:.2f} | {public_stats['width']['median']:.2f} | {public_stats['width']['min']} | {public_stats['width']['max']} | {public_stats['width']['std']:.2f} |
| 高度 | {public_stats['height']['mean']:.2f} | {public_stats['height']['median']:.2f} | {public_stats['height']['min']} | {public_stats['height']['max']} | {public_stats['height']['std']:.2f} |

**总图片数**: {public_stats['total_images']:,}

**异常值说明**:
- 宽度最小值 ({public_stats['width']['min']}px) 对应 {len(public_stats['width']['min_paths'])} 张图片
- 宽度最大值 ({public_stats['width']['max']}px) 对应 {len(public_stats['width']['max_paths'])} 张图片
- 高度最小值 ({public_stats['height']['min']}px) 对应 {len(public_stats['height']['min_paths'])} 张图片
- 高度最大值 ({public_stats['height']['max']}px) 对应 {len(public_stats['height']['max_paths'])} 张图片

详细路径请查看 `extreme_image_paths.txt` 文件。

### /ssddisk/guochuang/ocec/hq 目录

该目录包含高质量图片数据，对应annotation_0024.csv。

| 维度 | 平均值 | 中位值 | 最小值 | 最大值 | 标准差 |
|-----|--------|--------|--------|--------|--------|
| 宽度 | {hq_stats['width']['mean']:.2f} | {hq_stats['width']['median']:.2f} | {hq_stats['width']['min']} | {hq_stats['width']['max']} | {hq_stats['width']['std']:.2f} |
| 高度 | {hq_stats['height']['mean']:.2f} | {hq_stats['height']['median']:.2f} | {hq_stats['height']['min']} | {hq_stats['height']['max']} | {hq_stats['height']['std']:.2f} |

**总图片数**: {hq_stats['total_images']:,}

**异常值说明**:
- 宽度最小值 ({hq_stats['width']['min']}px) 对应 {len(hq_stats['width']['min_paths'])} 张图片
- 宽度最大值 ({hq_stats['width']['max']}px) 对应 {len(hq_stats['width']['max_paths'])} 张图片
- 高度最小值 ({hq_stats['height']['min']}px) 对应 {len(hq_stats['height']['min_paths'])} 张图片
- 高度最大值 ({hq_stats['height']['max']}px) 对应 {len(hq_stats['height']['max_paths'])} 张图片

详细路径请查看 `extreme_image_paths.txt` 文件。

## 数据集特点

1. **public目录**: 图片尺寸相对较小，宽度中位值{public_stats['width']['median']:.0f}像素，高度中位值{public_stats['height']['median']:.0f}像素，适合快速处理。
2. **hq目录**: 图片尺寸较大，宽度中位值{hq_stats['width']['median']:.0f}像素，高度中位值{hq_stats['height']['median']:.0f}像素，提供更高质量的图像数据。
3. **标签平衡**: 整体数据集标签分布相对均衡，闭眼和睁眼样本数量接近。

## 数据路径说明

- CSV标注文件位置: `/ssddisk/guochuang/ocec/list/`
- public图片路径: `/ssddisk/guochuang/ocec/public/cropped/`
- hq图片路径: `/ssddisk/guochuang/ocec/hq/`

## 相关文件

- 详细统计结果: `dataset_statistics.txt`
- 异常值图片路径: `extreme_image_paths.txt`

## 更新记录

- 自动更新: 统计完成后自动更新
"""
    
    with open(dataset_md_file, 'w', encoding='utf-8') as f:
        f.write(dataset_md_content)
    
    print(f"DATASET.md 文件已自动更新: {dataset_md_file}")
    
    return {
        'csv_stats': csv_stats,
        'public_stats': public_stats,
        'hq_stats': hq_stats
    }

if __name__ == "__main__":
    main()

