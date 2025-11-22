#!/usr/bin/env python3
"""
Hugging Face Dataset Viewer (OpenCV version)
- データセット: MichalMlodawski/closed-open-eyes
- Parquet形式で保存 (data/{split}/train.parquet)
- 既存ファイルがあれば再利用
- OpenCVでランダムサンプルを可視化
"""

import argparse
import io
import json
import os
import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from datasets import load_dataset, Dataset
import requests


def resolve_image(image_data):
    """Image-likeオブジェクトをRGBのPIL.Imageに変換して返す"""
    img = None

    if isinstance(image_data, dict):
        file_info = image_data.get("file")

        if isinstance(file_info, Image.Image):
            img = file_info

        elif isinstance(file_info, bytes):
            # 二进制数据（从 parquet 文件中读取）
            try:
                img = Image.open(io.BytesIO(file_info))
            except Exception as e:
                print(f"[WARN] Could not decode image from bytes: {e}")

        elif isinstance(file_info, str) and os.path.exists(file_info):
            try:
                img = Image.open(file_info)
            except Exception as e:
                print(f"[WARN] Could not open local image '{file_info}': {e}")

        elif isinstance(file_info, dict) and "src" in file_info:
            url = file_info["src"]
            try:
                res = requests.get(url, timeout=10)
                res.raise_for_status()
                img = Image.open(io.BytesIO(res.content))
            except Exception as e:
                print(f"[WARN] Could not load from URL '{url}': {e}")

    elif isinstance(image_data, Image.Image):
        img = image_data

    elif isinstance(image_data, bytes):
        # 直接传入二进制数据
        try:
            img = Image.open(io.BytesIO(image_data))
        except Exception as e:
            print(f"[WARN] Could not decode image from bytes: {e}")

    if img is None:
        return None

    if img.mode != "RGB":
        return img.convert("RGB")

    return img.copy()


def visualize_with_opencv(dataset, sample_count: int = 6, output_dir: str = "visualization_output"):
    """OpenCVを使ってランダムサンプルを可視化して保存"""
    indices = random.sample(range(len(dataset)), min(sample_count, len(dataset)))
    print(f"👁 Saving {len(indices)} random samples to {output_dir}...")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for i, idx in enumerate(indices):
        record = dataset[idx]
        label = record.get("Label", "unknown")
        image_data = record.get("Image_data")
        img = resolve_image(image_data)

        if img is None:
            print(f"[WARN] Skipping index {idx}, no image data found.")
            continue

        # PIL → OpenCV形式（numpy BGR）
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # ラベルテキスト描画
        cv2.putText(img_np, f"Label: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # 目の反応（座標）からバウンディングボックスを描画
        height, width = img_np.shape[:2]

        def draw_react_box(box, color, title):
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                return
            x, y, w, h = box
            if w is None or h is None:
                return
            if w <= 0 or h <= 0:
                return
            x1 = int(round(x))
            y1 = int(round(y))
            x2 = int(round(x + w))
            y2 = int(round(y + h))
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))
            if x2 <= x1 or y2 <= y1:
                return
            cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
            text_pos = (x1, max(0, y1 - 10))
            cv2.putText(img_np, title, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        draw_react_box(record.get("Left_eye_react"), (0, 255, 255), "Left eye")
        draw_react_box(record.get("Right_eye_react"), (255, 0, 0), "Right eye")

        # 保存图像到本地
        image_id = record.get("Image_id", idx)
        filename = f"sample_{i+1:03d}_idx_{idx}_id_{image_id}_label_{label}.jpg"
        filepath = output_path / filename
        
        try:
            cv2.imwrite(str(filepath), img_np)
            saved_count += 1
            print(f"  ✅ Saved: {filename}")
        except Exception as e:
            print(f"  ⚠️  Failed to save {filename}: {e}")

    print(f"✅ Saved {saved_count}/{len(indices)} samples to {output_path}")


def extract_dataset(dataset, base_outdir: str, split: str):
    """Parquetに含まれる画像とアノテーションをディスクへ展開"""
    extract_root = Path(base_outdir) / "extracted" / split
    total = len(dataset)
    print(f"📤 Extracting {total} samples to {extract_root} ...")

    extracted_count = 0
    created_chunks = set()

    for idx, record in enumerate(dataset):
        img = resolve_image(record.get("Image_data"))
        if img is None:
            print(f"[WARN] Skipping extraction for index {idx}, no image data found.")
            continue

        extracted_count += 1
        base_name = f"{extracted_count:08d}"
        chunk_index = (extracted_count - 1) // 1000 + 1
        chunk_name = f"{chunk_index:08d}"
        chunk_dir = extract_root / chunk_name

        if chunk_name not in created_chunks:
            chunk_dir.mkdir(parents=True, exist_ok=True)
            created_chunks.add(chunk_name)

        image_data = record.get("Image_data")
        ext = ".png"
        if isinstance(image_data, dict):
            filename = image_data.get("filename")
            if filename:
                _, orig_ext = os.path.splitext(filename)
                if orig_ext:
                    ext = orig_ext.lower()

        if ext == ".jpeg":
            ext = ".jpg"
        if ext not in (".jpg", ".png"):
            ext = ".png"

        candidate = chunk_dir / f"{base_name}{ext}"

        save_format = "PNG"
        if ext == ".jpg":
            save_format = "JPEG"

        try:
            img.save(candidate, format=save_format)
        except Exception as e:
            print(f"[WARN] Failed to save image for index {idx}: {e}")
            continue

        annotation = {
            "image_filename": candidate.name,
            "image_id": record.get("Image_id"),
            "label": record.get("Label"),
            "left_eye_react": record.get("Left_eye_react"),
            "right_eye_react": record.get("Right_eye_react"),
            "split": split,
        }
        ann_path = chunk_dir / f"{base_name}.json"
        try:
            with ann_path.open("w", encoding="utf-8") as f:
                json.dump(annotation, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to write annotation for index {idx}: {e}")

        if (idx + 1) % 1000 == 0 or (idx + 1) == total:
            print(f"  - Processed {idx + 1}/{total}, saved {extracted_count}")


def main():
    parser = argparse.ArgumentParser(description="Download and visualize MichalMlodawski/closed-open-eyes dataset with OpenCV.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: train)")
    parser.add_argument("--visualize", action="store_true", help="Visualize random samples and save to local files")
    parser.add_argument("--sample-count", type=int, default=6, help="Number of samples to visualize")
    parser.add_argument("--visualize-output", type=str, default="visualization_output", help="Output directory for visualized images (default: ./visualization_output)")
    parser.add_argument("--outdir", type=str, default="data", help="Output directory for extracted images (default: ./data)")
    parser.add_argument("--force", action="store_true", help="Force re-download even if parquet exists")
    parser.add_argument("--extract", action="store_true", help="Extract images and annotations to --outdir/extracted/{split}")
    parser.add_argument("--dataset-path", type=str, default="/ssddisk/guochuang/ocec/data", 
                        help="Path to local dataset directory containing parquet files (default: /ssddisk/guochuang/ocec/data)")
    args = parser.parse_args()

    split = args.split
    ds = None
    dataset_loaded = False

    # --- 优先检查本地数据集路径 ---
    dataset_path = Path(args.dataset_path)
    if dataset_path.exists() and dataset_path.is_dir():
        # 查找 dataset_*.parquet 文件（Hugging Face 下载的格式）
        parquet_files = sorted(dataset_path.glob("dataset_*.parquet"))
        if parquet_files:
            print(f"✅ Found local dataset directory: {dataset_path}")
            print(f"📖 Found {len(parquet_files)} parquet files (dataset_*.parquet)")
            print(f"📖 Loading dataset from multiple parquet files...")
            try:
                # 使用通配符模式加载所有 parquet 文件
                parquet_pattern = str(dataset_path / "dataset_*.parquet")
                ds = Dataset.from_parquet(parquet_pattern)
                print(f"✅ Loaded {len(ds)} samples from {len(parquet_files)} parquet files")
                dataset_loaded = True
            except Exception as e:
                print(f"⚠️  Failed to load dataset from parquet files: {e}")
                print(f"📦 Falling back to download...")
        else:
            # 如果没有找到 dataset_*.parquet，尝试查找单个 {split}.parquet
            parquet_file = dataset_path / f"{split}.parquet"
            if parquet_file.exists():
                print(f"✅ Found local dataset file: {parquet_file}")
                print(f"📖 Loading dataset from {parquet_file}...")
                try:
                    ds = Dataset.from_parquet(str(parquet_file))
                    print(f"✅ Loaded {len(ds)} samples from local file")
                    dataset_loaded = True
                except Exception as e:
                    print(f"⚠️  Failed to load dataset from {parquet_file}: {e}")
                    print(f"📦 Falling back to download...")
            else:
                print(f"⚠️  Directory {dataset_path} exists but no parquet files found.")
                print(f"   Looking for: dataset_*.parquet or {split}.parquet")
                print(f"📦 Falling back to download...")
    elif dataset_path.exists() and dataset_path.is_file() and dataset_path.suffix == '.parquet':
        # 直接指定了单个 parquet 文件
        print(f"✅ Found local dataset file: {dataset_path}")
        print("📖 Loading dataset from specified parquet file...")
        try:
            ds = Dataset.from_parquet(str(dataset_path))
            print(f"✅ Loaded {len(ds)} samples from local file")
            dataset_loaded = True
        except Exception as e:
            print(f"⚠️  Failed to load dataset from {dataset_path}: {e}")
            print(f"📦 Falling back to download...")

    # --- 如果本地加载失败，尝试从网上下载 ---
    if not dataset_loaded:
        if args.force:
            print("⚠️  Force mode enabled. Re-downloading dataset...")
        print(f"📦 Downloading dataset split='{split}' from Hugging Face...")
        try:
            ds = load_dataset("MichalMlodawski/closed-open-eyes", split=split)
            print(f"✅ Loaded {len(ds)} samples")
            
            # 保存到本地（可选）
            outdir = os.path.join(args.outdir, split)
            os.makedirs(outdir, exist_ok=True)
            parquet_path = os.path.join(outdir, f"{split}.parquet")
            print(f"💾 Saving dataset to {parquet_path} ...")
            ds.to_parquet(parquet_path)
            print(f"✅ Saved parquet: {parquet_path}")
        except Exception as e:
            print(f"❌ Failed to download dataset: {e}")
            return

    if args.extract:
        extract_dataset(ds, args.outdir, split)

    if args.visualize:
        visualize_with_opencv(ds, args.sample_count, args.visualize_output)
    elif not args.extract:
        print("👁 Visualization disabled. Use --visualize to enable.")


if __name__ == "__main__":
    main()
