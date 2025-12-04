#!/usr/bin/env python

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert cropped eye dataset into a single parquet file with "
            "balanced train/validation splits."
        )
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        nargs='+',
        default=None,
        help=(
            "Path(s) to annotation CSV file(s) or a directory containing CSV files. "
            "If a directory is provided, all annotation_*.csv files will be loaded. "
            "If multiple files are provided, they will be merged. "
            "Default: data/cropped/list/ (if exists) or data/cropped/annotation.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/cropped/eye_dataset.parquet"),
        help="Destination parquet file (default: data/cropped/eye_dataset.parquet).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio (between 0 and 1). Default: 0.8.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling prior to splitting. Default: 42.",
    )
    parser.add_argument(
        "--embed-images",
        action="store_true",
        help="Embed raw image bytes into the parquet output (column: image_bytes).",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=None,
        help=(
            "Maximum number of rows per parquet file. "
            "If specified, the dataset will be split into multiple files. "
            "Files will be named like: output_0001.parquet, output_0002.parquet, etc. "
            "If not specified, all data will be written to a single file."
        ),
    )
    args = parser.parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        parser.error("--train-ratio must be in the (0, 1) interval.")
    return args


def find_csv_files(paths: List[Path]) -> List[Path]:
    """Find all CSV files from given paths (files or directories)."""
    csv_files = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        
        if path.is_file():
            if path.suffix.lower() == '.csv':
                csv_files.append(path)
            else:
                raise ValueError(f"Not a CSV file: {path}")
        elif path.is_dir():
            # Find all CSV files in directory (any prefix)
            found = sorted(path.glob('*.csv'))
            if not found:
                raise ValueError(f"No CSV files found in directory: {path}")
            csv_files.extend(found)
        else:
            raise ValueError(f"Invalid path (not a file or directory): {path}")
    
    return sorted(set(csv_files))  # Remove duplicates and sort


def load_annotations(annotation_paths: List[Path]) -> pd.DataFrame:
    """Load annotations from one or more CSV files."""
    csv_files = find_csv_files(annotation_paths)
    
    if not csv_files:
        raise ValueError("No CSV files found to load.")
    
    print(f"Loading annotations from {len(csv_files)} CSV file(s):")
    for csv_file in csv_files:
        print(f"  - {csv_file}")
    
    dfs = []
    for csv_file in csv_files:
        if not csv_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {csv_file}")
        
        # Try to detect if CSV has header
        # Read first line to check if it's a header
        with open(csv_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            first_values = [v.strip() for v in first_line.split(',')]
        
        # Check if first line looks like a header
        # Headers typically contain keywords like "File", "Path", "Label", "Class"
        # and don't look like file paths (don't contain "/" or start with absolute paths)
        has_header = False
        if len(first_values) == 2:
            col0_lower = first_values[0].lower()
            col1_lower = first_values[1].lower()
            header_keywords = ['file', 'path', 'image', 'label', 'class', 'id']
            # Check if first line contains header keywords and doesn't look like a file path
            is_likely_header = (
                any(kw in col0_lower for kw in header_keywords) or 
                any(kw in col1_lower for kw in header_keywords)
            ) and not ('/' in first_values[0] or '\\' in first_values[0])
            
            if is_likely_header:
                has_header = True
        
        # Read the full CSV file
        if has_header:
            # CSV has header, use it and map columns
            df = pd.read_csv(csv_file, header=0)
            
            # Map column names to standard names
            # Handle various possible column name formats
            # First column should be image path, second should be class_id
            if len(df.columns) != 2:
                raise ValueError(
                    f"Expected 2 columns in {csv_file}, got {len(df.columns)}: {list(df.columns)}"
                )
            
            col_mapping = {}
            # Map first column to image_path (if it contains path/image/file keywords)
            col0_lower = str(df.columns[0]).lower()
            col1_lower = str(df.columns[1]).lower()
            
            # Determine which column is image_path and which is class_id
            if any(kw in col0_lower for kw in ['file', 'path', 'image']):
                col_mapping[df.columns[0]] = "image_path"
                col_mapping[df.columns[1]] = "class_id"
            elif any(kw in col1_lower for kw in ['file', 'path', 'image']):
                col_mapping[df.columns[1]] = "image_path"
                col_mapping[df.columns[0]] = "class_id"
            else:
                # Default: first column is image_path, second is class_id
                col_mapping[df.columns[0]] = "image_path"
                col_mapping[df.columns[1]] = "class_id"
            
            df = df.rename(columns=col_mapping)
            # Ensure correct data types
            df["image_path"] = df["image_path"].astype(str)
            df["class_id"] = df["class_id"].astype(int)
        else:
            # CSV has no header, read as data
            df = pd.read_csv(
                csv_file,
                header=None,
                names=("image_path", "class_id"),
                dtype={"image_path": str, "class_id": int},
            )
        
        if not df.empty:
            dfs.append(df)
    
    if not dfs:
        raise ValueError("No entries loaded from any CSV files.")
    
    # Merge all dataframes
    df = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates (same image_path)
    original_count = len(df)
    df = df.drop_duplicates(subset=["image_path"], keep="first")
    if len(df) < original_count:
        print(f"Removed {original_count - len(df)} duplicate entries.")
    
    df["image_path"] = df["image_path"].apply(lambda p: str(Path(p)))
    df["class_id"] = df["class_id"].astype(int)
    df["label"] = df.apply(lambda row: infer_label(row["image_path"], row["class_id"]), axis=1)
    df["source"] = df["image_path"].apply(determine_source)
    df["exists"] = df["image_path"].apply(lambda p: Path(p).exists())
    missing = df.loc[~df["exists"], "image_path"].tolist()
    if missing:
        raise FileNotFoundError(
            "Some image files are missing. "
            "Examples:\n" + "\n".join(missing[:10])
        )
    df.drop(columns=["exists"], inplace=True)
    return df


def embed_image_bytes(df: pd.DataFrame, column_name: str = "image_bytes") -> pd.DataFrame:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    def read_bytes(path_str: str) -> bytes:
        return Path(path_str).read_bytes()

    iterator: Iterable[str] = df["image_path"]
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Embedding images", unit="img")

    df = df.copy()
    df[column_name] = [read_bytes(path) for path in iterator]
    return df


def infer_label(path_str: str, class_id: int) -> str:
    stem = Path(path_str).stem.lower()
    if "open" in stem:
        return "open"
    if "closed" in stem:
        return "closed"
    return "open" if class_id == 1 else "closed"


def determine_source(path_str: str) -> str:
    path = Path(path_str)
    try:
        folder_name = path.parts[path.parts.index("cropped") + 1]
    except (ValueError, IndexError):
        return "unknown"
    if folder_name.isdigit() and int(folder_name) >= 100000000:
        return "real_data"
    if folder_name.isdigit():
        return "train_dataset"
    return "unknown"


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    train_rows: List[pd.Series] = []
    val_rows: List[pd.Series] = []

    for label, group in df.groupby("label"):
        records = list(group.to_dict(orient="records"))
        rng.shuffle(records)
        n_total = len(records)
        if n_total == 0:
            continue
        n_train = int(n_total * train_ratio)
        if n_train == 0 and n_total > 1:
            n_train = 1
        if n_total - n_train == 0 and n_total > 1:
            n_train -= 1
        train_subset = records[:n_train] if n_train > 0 else []
        val_subset = records[n_train:]
        train_rows.extend(train_subset)
        val_rows.extend(val_subset)

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    train_df["split"] = "train"
    val_df["split"] = "val"
    return train_df, val_df


def validate_balances(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    def describe(df: pd.DataFrame, split: str) -> Dict[str, int]:
        counter = Counter(df["label"])
        total = int(counter.total())
        description = {f"{split}_total": total}
        for label, count in counter.items():
            description[f"{split}_{label}"] = count
        return description

    train_counts = describe(train_df, "train")
    val_counts = describe(val_df, "val")
    summary = train_counts | val_counts
    print("Split summary:", summary)


def save_parquet_files(
    df: pd.DataFrame,
    output_path: Path,
    max_rows_per_file: Optional[int] = None,
) -> None:
    """Save DataFrame to one or more parquet files."""
    if max_rows_per_file is None or len(df) <= max_rows_per_file:
        # Single file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"Saved dataset to {output_path} ({len(df)} rows).")
        return
    
    # Multiple files
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate base name and extension
    base_name = output_path.stem
    extension = output_path.suffix
    output_dir = output_path.parent
    
    num_files = (len(df) + max_rows_per_file - 1) // max_rows_per_file
    print(f"Splitting dataset into {num_files} files (max {max_rows_per_file} rows per file)...")
    
    for i in range(num_files):
        start_idx = i * max_rows_per_file
        end_idx = min((i + 1) * max_rows_per_file, len(df))
        chunk_df = df.iloc[start_idx:end_idx]
        
        file_name = f"{base_name}_{i+1:04d}{extension}"
        file_path = output_dir / file_name
        chunk_df.to_parquet(file_path, index=False)
        print(f"  Saved {file_path.name} ({len(chunk_df)} rows)")
    
    print(f"Saved dataset to {num_files} files in {output_dir}")


def main() -> None:
    args = parse_args()
    
    # Handle default annotation paths
    if args.annotation is None:
        # Try default locations
        default_list_dir = Path("data/cropped/list")
        default_csv = Path("data/cropped/annotation.csv")
        
        if default_list_dir.exists() and default_list_dir.is_dir():
            args.annotation = [default_list_dir]
            print(f"Using default directory: {default_list_dir}")
        elif default_csv.exists():
            args.annotation = [default_csv]
            print(f"Using default file: {default_csv}")
        else:
            raise FileNotFoundError(
                f"Default annotation paths not found. "
                f"Please specify --annotation. "
                f"Tried: {default_list_dir}, {default_csv}"
            )
    
    df = load_annotations(args.annotation)
    if args.embed_images:
        df = embed_image_bytes(df)
    train_df, val_df = stratified_split(df, args.train_ratio, args.seed)

    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    column_order = ["split", "label", "class_id", "image_path", "source"]
    if args.embed_images:
        column_order.append("image_bytes")
    combined_df = combined_df[column_order].sort_values(["split", "label", "image_path"])

    validate_balances(train_df, val_df)
    
    # Save to one or more parquet files
    save_parquet_files(combined_df, args.output, args.max_rows_per_file)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
