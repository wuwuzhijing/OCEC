#!/usr/bin/env python3
"""
处理 /ssddisk/guochuang/ocec/hq 下的jpg文件，写入CSV
文件名最后一个数字：1表示闭眼，0表示睁眼
但CSV中要反转：文件名是1，CSV写0；文件名是0，CSV写1
使用绝对路径
"""

import sys
from pathlib import Path
from typing import List, Tuple
import re

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


def extract_label_from_filename(filename: str) -> Tuple[int, bool]:
    """
    从文件名提取标签
    返回: (classid, is_valid)
    文件名最后一个数字：1表示闭眼，0表示睁眼
    CSV中要反转：文件名是1，CSV写0；文件名是0，CSV写1
    
    匹配模式：
    - 文件名以 _0.jpg 或 _1.jpg 结尾
    - 或者文件名以 _0 或 _1 结尾（无扩展名）
    """
    filename_lower = filename.lower()
    
    # 优先匹配：_0.jpg 或 _1.jpg
    if filename_lower.endswith('_0.jpg'):
        # 文件名是0（睁眼），CSV写1
        return 1, True
    elif filename_lower.endswith('_1.jpg'):
        # 文件名是1（闭眼），CSV写0
        return 0, True
    elif filename_lower.endswith('_0'):
        # 文件名是0（睁眼），CSV写1
        return 1, True
    elif filename_lower.endswith('_1'):
        # 文件名是1（闭眼），CSV写0
        return 0, True
    
    return None, False


def process_hq_folder(hq_dir: Path) -> List[Tuple[str, int]]:
    """处理hq文件夹下的jpg文件（递归查找）"""
    entries = []
    
    # 递归查找所有jpg文件
    jpg_files = sorted(hq_dir.rglob('*.jpg'))
    print(Color.CYAN(f'Found {len(jpg_files)} JPG files in {hq_dir} (recursive)'))
    
    skipped_count = 0
    skipped_files = []
    for jpg_file in jpg_files:
        classid, is_valid = extract_label_from_filename(jpg_file.name)
        
        if not is_valid:
            skipped_count += 1
            if len(skipped_files) < 10:  # 只保存前10个作为示例
                skipped_files.append(jpg_file.name)
            continue
        
        # 使用绝对路径
        abs_path = jpg_file.resolve()
        # 格式：data/cropped/000001000/00001522_2_cd0c4e4750fe.png,0
        # 但用户要求写绝对路径，所以直接写绝对路径
        rel_path = str(abs_path)
        
        entries.append((rel_path, classid))
    
    if skipped_count > 0:
        print(Color.YELLOW(f'Skipped {skipped_count} files due to invalid filename format (not ending with _0.jpg or _1.jpg)'))
        if skipped_files:
            print(Color.YELLOW(f'  Examples: {", ".join(skipped_files[:5])}'))
    
    print(Color.GREEN(f'Processed {len(entries)} files from hq folder'))
    return entries


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
    hq_dir = Path('/ssddisk/guochuang/ocec/hq')
    csv_dir = Path('/ssddisk/guochuang/ocec/list_hq_v2/')
    
    if not hq_dir.exists():
        print(Color.RED(f'Directory not found: {hq_dir}'))
        return
    
    if not csv_dir.exists():
        csv_dir.mkdir(parents=True, exist_ok=True)
        print(Color.CYAN(f'Created CSV directory: {csv_dir}'))
    
    print(Color.CYAN('=' * 60))
    print(Color.CYAN('Processing hq folder to CSV'))
    print(Color.CYAN('=' * 60))
    
    # 处理hq文件夹
    print(Color.CYAN('\nProcessing hq folder...'))
    entries = process_hq_folder(hq_dir)
    
    if not entries:
        print(Color.YELLOW('No valid entries found'))
        return
    
    # 写入新的CSV文件
    print(Color.CYAN('\nWriting CSV file...'))
    next_index = find_next_csv_index(csv_dir)
    new_csv_path = csv_dir / f'annotation_{next_index:04d}.csv'
    write_csv_file(entries, new_csv_path)
    
    print(Color.CYAN('\n' + '=' * 60))
    print(Color.GREEN('✓ Processing completed successfully!'))
    print(Color.GREEN(f'  Created file: {new_csv_path.name}'))
    print(Color.CYAN('=' * 60))


if __name__ == '__main__':
    main()

