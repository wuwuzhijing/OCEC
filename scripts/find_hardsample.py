import pandas as pd
import argparse
from pathlib import Path

def load_dataset(dataset_path):
    """加载数据集，支持单个parquet文件或包含多个parquet文件的目录"""
    path = Path(dataset_path)
    
    if path.is_file():
        # 如果是单个文件，直接读取
        print(f"读取单个parquet文件: {dataset_path}")
        return pd.read_parquet(dataset_path)
    elif path.is_dir():
        # 如果是目录，读取所有.parquet文件并合并
        parquet_files = sorted(path.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"目录 {dataset_path} 中没有找到.parquet文件")
        
        print(f"找到 {len(parquet_files)} 个parquet文件，开始合并...")
        dfs = []
        for i, pfile in enumerate(parquet_files, 1):
            print(f"  读取 ({i}/{len(parquet_files)}): {pfile.name}")
            dfs.append(pd.read_parquet(pfile))
        
        meta = pd.concat(dfs, ignore_index=True)
        print(f"合并完成，总共 {len(meta)} 条记录")
        return meta
    else:
        raise ValueError(f"路径不存在: {dataset_path}")

def process_csv(input_file, meta, output_file=None):
    """处理单个CSV文件，将index映射为路径"""
    # 读取文件
    print(f"\n读取输入CSV: {input_file}")
    input_df = pd.read_csv(input_file)
    print(f"找到 {len(input_df)} 条记录")
    print(f"列名: {list(input_df.columns)}")
    
    # 自动检测列名格式
    has_true = "true" in input_df.columns
    has_pred = "pred" in input_df.columns
    has_label = "label" in input_df.columns
    has_prob = "prob" in input_df.columns
    has_distance = "distance" in input_df.columns
    
    def resolve(row):
        idx = int(row["index"])
        meta_row = meta.iloc[idx]
        
        # 根据不同的输入格式提取信息
        if has_true:
            # mislabeled.csv 格式: index, true, pred, prob
            true_label = row.get("true", None)
            pred_label = row.get("pred", None)
            pred_prob = row.get("prob", None)
            distance = row.get("distance", None)
        elif has_distance:
            # outliers.csv 格式: index, label, distance
            true_label = meta_row["label"]  # 从meta中获取真实标签
            pred_label = row.get("label", None)
            pred_prob = None  # outliers.csv没有prob
            distance = row.get("distance", None)
        else:
            # hard_samples.csv 格式: index, prob, label
            true_label = meta_row["label"]  # 从meta中获取真实标签
            pred_label = row.get("label", None)
            pred_prob = row.get("prob", None)
            distance = None
        
        # 根据是否有distance列决定返回的列数
        if has_distance:
            return (
                idx,
                meta_row["image_path"],
                true_label,
                pred_prob,
                pred_label,
                distance
            )
        else:
            return (
                idx,
                meta_row["image_path"],
                true_label,
                pred_prob,
                pred_label
            )
    
    resolved = [resolve(row) for _, row in input_df.iterrows()]
    
    # 根据是否有distance列决定输出列
    output_columns = ["idx", "path", "true_label", "pred_prob", "pred_label"]
    if has_distance:
        output_columns.append("distance")
    
    # 如果没有指定输出文件，自动生成
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_with_path{input_path.suffix}"
    
    pd.DataFrame(
        resolved,
        columns=output_columns
    ).to_csv(output_file, index=False)
    
    print(f"Saved → {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description="将CSV中的index映射为图片路径")
    parser.add_argument("--input", type=str, nargs="*",
                        help="输入CSV文件路径（可指定多个文件，支持hard_samples.csv、mislabeled.csv或outliers.csv格式）")
    parser.add_argument("--input_dir", type=str,
                        help="输入目录，自动查找hard_samples.csv、mislabeled.csv和outliers.csv")
    parser.add_argument("--hard_samples", type=str, 
                        help="[已废弃] 使用--input代替。hard samples CSV文件路径")
    parser.add_argument("--dataset", type=str, default="/ssddisk/guochuang/ocec/parquet_hq_v2/",
                        help="数据集parquet文件路径或包含多个parquet文件的目录")
    parser.add_argument("--output", type=str, nargs="*",
                        help="输出CSV文件路径（可选，如果不指定则自动生成。如果指定多个，数量需与输入文件数量一致）")
    parser.add_argument("--output_dir", type=str,
                        help="输出目录（如果指定，所有输出文件将保存到此目录）")
    
    args = parser.parse_args()
    
    # 确定输入文件列表
    input_files = []
    
    if args.input_dir:
        # 从目录中查找三个文件
        input_dir = Path(args.input_dir)
        target_files = ["hard_samples.csv", "mislabeled.csv", "outliers.csv"]
        for filename in target_files:
            file_path = input_dir / filename
            if file_path.exists():
                input_files.append(str(file_path))
            else:
                print(f"警告: 未找到 {file_path}")
    elif args.input:
        # 使用指定的输入文件
        input_files = args.input
    elif args.hard_samples:
        # 向后兼容
        input_files = [args.hard_samples]
    else:
        parser.error("必须指定--input、--input_dir或--hard_samples参数")
    
    if not input_files:
        parser.error("没有找到任何输入文件")
    
    print(f"找到 {len(input_files)} 个输入文件:")
    for f in input_files:
        print(f"  - {f}")
    
    # 加载数据集（只加载一次，所有文件共享）
    print(f"\n加载数据集...")
    meta = load_dataset(args.dataset)
    
    # 处理每个输入文件
    output_files = args.output if args.output is not None else [None] * len(input_files)
    
    if len(output_files) != len(input_files) and len(output_files) > 1:
        parser.error(f"输出文件数量({len(output_files)})与输入文件数量({len(input_files)})不匹配")
    
    results = []
    for i, input_file in enumerate(input_files):
        output_file = output_files[i] if i < len(output_files) else None
        
        # 如果指定了输出目录，将输出文件放到该目录
        if args.output_dir:
            input_path = Path(input_file)
            if output_file is None:
                output_file = Path(args.output_dir) / f"{input_path.stem}_with_path{input_path.suffix}"
            else:
                output_file = Path(args.output_dir) / Path(output_file).name
        
        result = process_csv(input_file, meta, output_file)
        results.append(result)
    
    print(f"\n处理完成！共处理 {len(results)} 个文件:")
    for r in results:
        print(f"  ✓ {r}")

if __name__ == "__main__":
    main()