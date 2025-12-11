import pandas as pd
import argparse
from pathlib import Path
import shutil

def copy_files_from_csv(csv_path, output_dir):
    """
    从CSV文件中读取路径，将文件拷贝到目标目录
    
    Args:
        csv_path: CSV文件路径
        output_dir: 输出目录
    """
    print(f"\n处理文件: {csv_path}")
    
    # 读取CSV
    df = pd.read_csv(csv_path)
    print(f"找到 {len(df)} 条记录")
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    copied_count = 0
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
            
            # 获取文件名
            filename = source_path.name
            
            # 目标路径
            dest_path = output_dir / filename
            
            # 如果目标文件已存在，添加序号
            if dest_path.exists():
                counter = 1
                stem = source_path.stem
                extension = source_path.suffix
                while dest_path.exists():
                    dest_path = output_dir / f"{stem}_{counter}{extension}"
                    counter += 1
            
            # 拷贝文件
            shutil.copy2(str(source_path), str(dest_path))
            copied_count += 1
            
            if copied_count % 1000 == 0:
                print(f"  已拷贝: {copied_count} 个文件")
                
        except Exception as e:
            print(f"错误: 处理 {row.get('path', 'unknown')} 时出错: {e}")
            error_count += 1
    
    print(f"\n完成! 拷贝: {copied_count}, 跳过: {skipped_count}, 错误: {error_count}")
    return copied_count, skipped_count, error_count

def main():
    parser = argparse.ArgumentParser(description="从CSV文件中读取路径，将文件拷贝到对应文件夹")
    parser.add_argument("--hard_samples_csv", type=str,
                        default="/103/guochuang/Code/myOCEC/runs/ocec_hq_finetune_progressive_v2.2.3/v4/hard_samples_with_path.csv",
                        help="hard_samples CSV文件路径")
    parser.add_argument("--mislabeled_csv", type=str,
                        default="/103/guochuang/Code/myOCEC/runs/ocec_hq_finetune_progressive_v2.2.3/v4/mislabeled_with_path.csv",
                        help="mislabeled CSV文件路径")
    parser.add_argument("--output_base", type=str,
                        default="/103/guochuang/Code/myOCEC/data/hq/eyes_compact_2.2.3",
                        help="输出基础目录")
    
    args = parser.parse_args()
    
    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # 定义两个CSV文件和对应的输出目录
    configs = [
        {
            "csv_path": args.hard_samples_csv,
            "output_dir": output_base / "hard_samples",
            "name": "hard_samples"
        },
        {
            "csv_path": args.mislabeled_csv,
            "output_dir": output_base / "mislabeled",
            "name": "mislabeled"
        }
    ]
    
    total_copied = 0
    total_skipped = 0
    total_errors = 0
    
    for config in configs:
        csv_path = Path(config["csv_path"])
        
        if not csv_path.exists():
            print(f"警告: 文件不存在，跳过: {csv_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"处理 {config['name']}")
        print(f"{'='*60}")
        
        copied, skipped, errors = copy_files_from_csv(
            csv_path,
            config["output_dir"]
        )
        
        total_copied += copied
        total_skipped += skipped
        total_errors += errors
    
    print(f"\n{'='*60}")
    print(f"总计: 拷贝 {total_copied}, 跳过 {total_skipped}, 错误 {total_errors}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

