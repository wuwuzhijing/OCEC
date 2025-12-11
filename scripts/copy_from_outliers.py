import pandas as pd
import argparse
from pathlib import Path
import shutil

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

def copy_and_rename_from_outliers(csv_path, outliers_dir, output_dir, fields_to_add):
    """
    从outliers目录中找到CSV中存在的文件，复制到目标目录并重命名
    
    Args:
        csv_path: CSV文件路径
        outliers_dir: outliers目录路径
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
    
    outliers_dir = Path(outliers_dir)
    if not outliers_dir.exists():
        print(f"错误: outliers目录不存在: {outliers_dir}")
        return 0, 0, 0
    
    copied_count = 0
    skipped_count = 0
    error_count = 0
    
    # 创建文件名到CSV行的映射（只使用文件名，不包含路径）
    file_to_row = {}
    for idx, row in df.iterrows():
        source_path = Path(row["path"])
        filename = source_path.name  # 只取文件名
        file_to_row[filename] = row
    
    print(f"在outliers目录中查找 {len(file_to_row)} 个文件...")
    
    # 遍历outliers目录中的所有文件
    for file_path in outliers_dir.iterdir():
        if not file_path.is_file():
            continue
        
        filename = file_path.name
        
        # 检查文件是否在CSV中
        if filename not in file_to_row:
            continue
        
        try:
            row = file_to_row[filename]
            
            # 获取原文件名和扩展名
            original_name = file_path.stem
            extension = file_path.suffix
            
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
            
            # 复制文件
            shutil.copy2(str(file_path), str(dest_path))
            copied_count += 1
            
            if copied_count % 1000 == 0:
                print(f"  已复制: {copied_count} 个文件")
                
        except Exception as e:
            print(f"错误: 处理 {filename} 时出错: {e}")
            error_count += 1
    
    # 检查是否有CSV中的文件在outliers目录中找不到
    found_files = set()
    for file_path in outliers_dir.iterdir():
        if file_path.is_file():
            found_files.add(file_path.name)
    
    missing_files = set(file_to_row.keys()) - found_files
    if missing_files:
        print(f"\n警告: 有 {len(missing_files)} 个CSV中的文件在outliers目录中未找到")
        if len(missing_files) <= 10:
            for f in missing_files:
                print(f"  - {f}")
        else:
            print(f"  前10个: {list(missing_files)[:10]}")
        skipped_count += len(missing_files)
    
    print(f"\n完成! 复制: {copied_count}, 跳过: {skipped_count}, 错误: {error_count}")
    return copied_count, skipped_count, error_count

def main():
    parser = argparse.ArgumentParser(description="从outliers目录复制文件到mislabeled和hard_samples目录并重命名")
    parser.add_argument("--csv_dir", type=str, required=True,
                        help="包含CSV文件的目录")
    parser.add_argument("--outliers_dir", type=str, 
                        default="/103/guochuang/Code/myOCEC/data/hq/eyes_compact_v2/outliers",
                        help="outliers目录路径")
    parser.add_argument("--base_dir", type=str,
                        default="/103/guochuang/Code/myOCEC/data/hq/eyes_compact_v2",
                        help="基础目录，会在其下创建mislabeled和hard_samples子目录")
    
    args = parser.parse_args()
    
    csv_dir = Path(args.csv_dir)
    outliers_dir = Path(args.outliers_dir)
    base_dir = Path(args.base_dir)
    
    # 定义两个CSV文件和对应的配置
    configs = [
        {
            "csv_name": "mislabeled_with_path.csv",
            "output_dir": base_dir / "mislabeled",
            "fields": ["pred_prob", "pred_label"]
        },
        {
            "csv_name": "hard_samples_with_path.csv",
            "output_dir": base_dir / "hard_samples",
            "fields": ["pred_prob", "pred_label"]
        }
    ]
    
    total_copied = 0
    total_skipped = 0
    total_errors = 0
    
    for config in configs:
        csv_path = csv_dir / config["csv_name"]
        
        if not csv_path.exists():
            print(f"警告: 文件不存在，跳过: {csv_path}")
            continue
        
        copied, skipped, errors = copy_and_rename_from_outliers(
            csv_path,
            outliers_dir,
            config["output_dir"],
            config["fields"]
        )
        
        total_copied += copied
        total_skipped += skipped
        total_errors += errors
    
    print(f"\n{'='*60}")
    print(f"总计: 复制 {total_copied}, 跳过 {total_skipped}, 错误 {total_errors}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

