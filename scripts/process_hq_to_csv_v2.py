#!/usr/bin/env python3
"""
处理 /ssddisk/guochuang/ocec/hq/eyes_compact 下的jpg文件，写入CSV
目录结构：
  - 0/ 文件夹：闭眼图片（标签0）
  - 1/ 文件夹：睁眼图片（标签1）
为了保证样本均衡性，0和1的比例约1:1.2
使用绝对路径
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple
import random

# Add parent directory to path to import Color
sys.path.insert(0, str(Path(__file__).parent))

try:
    from myOCEC.utils.color import Color
except ImportError:
    # Fallback if Color not available
    class Color:
        @staticmethod
        def GREEN(s): return s
        @staticmethod
        def YELLOW(s): return s
        @staticmethod
        def RED(s): return s
        @staticmethod
        def CYAN(s): return s


def get_image_files(folder: Path) -> List[Path]:
    """获取文件夹下的所有图片文件（支持jpg, jpeg, png等）- 优化版本"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = []
    
    # 使用 os.scandir 代替 glob，更快
    folder_abs = folder.resolve()  # 确保是绝对路径
    folder_str = str(folder_abs)
    try:
        with os.scandir(folder_str) as entries:
            for entry in entries:
                if entry.is_file():
                    # entry.path 已经是绝对路径
                    file_path = Path(entry.path)
                    if file_path.suffix in image_extensions:
                        image_files.append(file_path)
    except PermissionError:
        print(Color.YELLOW(f'Permission denied: {folder_str}'))
        return []
    
    return sorted(image_files)


def process_eyes_compact_folder(base_dir: Path, ratio: float = 1.2, seed: int = 42) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    处理eyes_compact文件夹
    返回: (主CSV条目列表, 剩余1类图片条目列表)
    """
    folder_0 = base_dir / '0'  # 闭眼
    folder_1 = base_dir / '1'  # 睁眼
    
    if not folder_0.exists():
        print(Color.RED(f'Directory not found: {folder_0}'))
        return [], []
    
    if not folder_1.exists():
        print(Color.RED(f'Directory not found: {folder_1}'))
        return [], []
    
    # 获取所有图片文件
    print(Color.CYAN('Scanning folder 0 (closed eyes)...'))
    files_0 = get_image_files(folder_0)
    print(Color.CYAN(f'Found {len(files_0)} files in folder 0'))
    
    print(Color.CYAN('Scanning folder 1 (open eyes)...'))
    files_1 = get_image_files(folder_1)
    print(Color.CYAN(f'Found {len(files_1)} files in folder 1'))
    
    # 计算需要从1中选择的数量（约1.2倍）
    num_0 = len(files_0)
    num_1_to_select = int(num_0 * ratio)
    
    print(Color.CYAN(f'Target ratio: 1:{ratio}'))
    print(Color.CYAN(f'Will select {num_1_to_select} files from folder 1 (out of {len(files_1)})'))
    
    # 优化：使用索引随机选择，而不是 random.sample（避免创建大列表副本）
    random.seed(seed)
    num_1_total = len(files_1)
    num_1_to_select = min(num_1_to_select, num_1_total)
    
    # 生成随机索引
    selected_indices = set(random.sample(range(num_1_total), num_1_to_select))
    
    print(Color.CYAN(f'Selecting {num_1_to_select} files from folder 1...'))
    
    # 构建主CSV条目（0类全部 + 1类选中的）
    print(Color.CYAN('Building main CSV entries...'))
    main_entries = []
    
    # 处理0类文件（全部）- 直接使用路径，已经是绝对路径
    for file_path in files_0:
        main_entries.append((str(file_path), 0))  # 0表示闭眼
    
    # 处理1类选中的文件
    for idx, file_path in enumerate(files_1):
        if idx in selected_indices:
            main_entries.append((str(file_path), 1))  # 1表示睁眼
    
    # 构建剩余1类图片条目
    print(Color.CYAN('Building remaining CSV entries...'))
    remaining_entries = []
    for idx, file_path in enumerate(files_1):
        if idx not in selected_indices:
            remaining_entries.append((str(file_path), 1))  # 1表示睁眼
    
    print(Color.GREEN(f'Main CSV will contain {len(main_entries)} entries (0: {num_0}, 1: {len(selected_indices)})'))
    print(Color.GREEN(f'Remaining CSV will contain {len(remaining_entries)} entries (all from folder 1)'))
    
    return main_entries, remaining_entries


def find_next_csv_index(csv_dir: Path) -> int:
    """找到下一个可用的CSV文件索引"""
    csv_files = sorted(csv_dir.glob('annotation_*.csv'))
    if not csv_files:
        return 1
    
    # 从最后一个文件提取索引
    last_csv = csv_files[-1]
    try:
        stem = last_csv.stem  # annotation_0023
        index_str = stem.split('_')[-1]
        last_index = int(index_str)
        return last_index + 1
    except (ValueError, IndexError):
        return 1


def write_csv_file(entries: List[Tuple[str, int]], output_path: Path):
    """写入CSV文件"""
    # 按路径排序
    sorted_entries = sorted(entries, key=lambda x: x[0])
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for rel_path, classid in sorted_entries:
                f.write(f'{rel_path},{classid}\n')
        print(Color.GREEN(f'✓ Wrote {len(sorted_entries)} entries to {output_path.name}'))
    except Exception as e:
        print(Color.RED(f'Error writing {output_path}: {e}'))
        raise


def main():
    # 路径设置
    eyes_compact_dir = Path('/ssddisk/guochuang/ocec/hq/eyes_compact')
    csv_dir = Path('/ssddisk/guochuang/ocec/list_hq_v3')
    
    if not eyes_compact_dir.exists():
        print(Color.RED(f'Directory not found: {eyes_compact_dir}'))
        return
    
    if not csv_dir.exists():
        csv_dir.mkdir(parents=True, exist_ok=True)
        print(Color.CYAN(f'Created CSV directory: {csv_dir}'))
    
    print(Color.CYAN('=' * 60))
    print(Color.CYAN('Processing eyes_compact folder to CSV'))
    print(Color.CYAN('=' * 60))
    
    # 处理eyes_compact文件夹
    print(Color.CYAN('\nProcessing eyes_compact folder...'))
    main_entries, remaining_entries = process_eyes_compact_folder(eyes_compact_dir, ratio=1.2)
    
    if not main_entries:
        print(Color.YELLOW('No valid entries found'))
        return
    
    # 写入主CSV文件
    print(Color.CYAN('\nWriting main CSV file...'))
    next_index = find_next_csv_index(csv_dir)
    main_csv_path = csv_dir / f'annotation_{next_index:04d}.csv'
    write_csv_file(main_entries, main_csv_path)
    
    # 写入剩余1类图片的CSV文件
    if remaining_entries:
        print(Color.CYAN('\nWriting remaining CSV file...'))
        remaining_csv_path = csv_dir / f'annotation_{next_index:04d}_remaining.csv'
        write_csv_file(remaining_entries, remaining_csv_path)
    
    print(Color.CYAN('\n' + '=' * 60))
    print(Color.GREEN('✓ Processing completed successfully!'))
    print(Color.GREEN(f'  Created main file: {main_csv_path.name}'))
    if remaining_entries:
        print(Color.GREEN(f'  Created remaining file: {remaining_csv_path.name}'))
    print(Color.CYAN('=' * 60))


if __name__ == '__main__':
    main()

