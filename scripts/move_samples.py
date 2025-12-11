import pandas as pd
import argparse
from pathlib import Path
import shutil
import os

def format_value(value):
    """格式化数值，用于文件名"""
    if value is None or pd.isna(value) or value == '' or str(value).strip() == '':
        return None  # 返回None表示跳过该字段
    
    # 尝试转换为数值
    try:
        num_value = float(value)
        # 如果是整数，显示为整数
        if num_value.is_integer():
            return str(int(num_value))
        # 否则保留3位小数，去掉末尾的0
        return f"{num_value:.3f}".rstrip('0').rstrip('.')
    except (ValueError, TypeError):
        # 不是数值，直接返回字符串
        return str(value).strip()

def move_and_rename_images(csv_path, output_dir, fields_to_add):
    """
    从CSV读取图片路径，剪切到目标文件夹，并在文件名中加入指定字段
    
    Args:
        csv_path: CSV文件路径
        output_dir: 输出目录
        fields_to_add: 要添加到文件名的字段列表
    """
    print(f"\n处理文件: {csv_path}")
    
    # 读取CSV
    df = pd.read_csv(csv_path)
    print(f"找到 {len(df)} 条记录")
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    moved_count = 0
    skipped_count = 0
    error_count = 0
    
    for idx, row in df.iterrows():
        try:
            source_path = Path(row["path"])
            
            # 检查源文件是否存在
            if not source_path.exists():
                print(f"警告: 文件不存在，跳过: {source_path}")
                skipped_count += 1
                continue
            
            # 获取原文件名和扩展名
            original_name = source_path.stem
            extension = source_path.suffix
            
            # 构建新文件名：原文件名_字段1_字段2_...
            field_values = []
            for field in fields_to_add:
                if field in row:
                    value = format_value(row[field])
                    if value is not None:  # 只添加非空值
                        field_values.append(f"{field}_{value}")
                else:
                    print(f"警告: 字段 '{field}' 不存在于CSV中，跳过")
            
            # 组合新文件名
            if field_values:
                new_name = f"{original_name}_{'_'.join(field_values)}{extension}"
            else:
                new_name = f"{original_name}{extension}"
            
            # 确保文件名长度不超过255（文件系统限制）
            if len(new_name) > 255:
                # 截断文件名，保留扩展名
                max_stem_length = 255 - len(extension) - len('_'.join(field_values)) - 1
                truncated_stem = original_name[:max_stem_length]
                new_name = f"{truncated_stem}_{'_'.join(field_values)}{extension}"
            
            # 目标路径
            dest_path = output_dir / new_name
            
            # 如果目标文件已存在，添加序号
            if dest_path.exists():
                counter = 1
                while dest_path.exists():
                    stem_without_ext = dest_path.stem
                    dest_path = output_dir / f"{stem_without_ext}_{counter}{extension}"
                    counter += 1
            
            # 剪切文件
            shutil.move(str(source_path), str(dest_path))
            moved_count += 1
            
            if (moved_count + skipped_count + error_count) % 1000 == 0:
                print(f"  已处理: {moved_count} 移动, {skipped_count} 跳过, {error_count} 错误")
                
        except Exception as e:
            print(f"错误: 处理 {row.get('path', 'unknown')} 时出错: {e}")
            error_count += 1
    
    print(f"\n完成! 移动: {moved_count}, 跳过: {skipped_count}, 错误: {error_count}")
    return moved_count, skipped_count, error_count

def main():
    parser = argparse.ArgumentParser(description="将CSV中的图片剪切到对应文件夹并重命名")
    parser.add_argument("--csv_dir", type=str, required=True,
                        help="包含三个CSV文件的目录")
    parser.add_argument("--output_base", type=str, required=True,
                        help="输出基础目录，会在其下创建三个子文件夹")
    
    args = parser.parse_args()
    
    csv_dir = Path(args.csv_dir)
    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # 定义三个CSV文件和对应的配置
    configs = [
        {
            "csv_name": "outliers_with_path.csv",
            "output_dir": output_base / "outliers",
            "fields": ["pred_prob", "pred_label", "distance"]
        },
        {
            "csv_name": "mislabeled_with_path.csv",
            "output_dir": output_base / "mislabeled",
            "fields": ["pred_prob", "pred_label"]
        },
        {
            "csv_name": "hard_samples_with_path.csv",
            "output_dir": output_base / "hard_samples",
            "fields": ["pred_prob", "pred_label"]
        }
    ]
    
    total_moved = 0
    total_skipped = 0
    total_errors = 0
    
    for config in configs:
        csv_path = csv_dir / config["csv_name"]
        
        if not csv_path.exists():
            print(f"警告: 文件不存在，跳过: {csv_path}")
            continue
        
        moved, skipped, errors = move_and_rename_images(
            csv_path,
            config["output_dir"],
            config["fields"]
        )
        
        total_moved += moved
        total_skipped += skipped
        total_errors += errors
    
    print(f"\n{'='*60}")
    print(f"总计: 移动 {total_moved}, 跳过 {total_skipped}, 错误 {total_errors}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

