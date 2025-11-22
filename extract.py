#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import pyarrow.parquet as pq


def auto_find_image_key(record):
    """自动寻找图片字段名，例如 image / Image_data / img / image_bytes 等."""
    for key in record:
        lk = key.lower()
        if "image" in lk or "img" in lk:
            return key
    return None


def decode_image(raw):
    """把 raw 数据解码成 PIL.Image"""
    if raw is None:
        return None

    # case 1: HuggingFace Image dict: {"bytes": ..., "path": ...}
    if isinstance(raw, dict):
        # raw bytes
        if "bytes" in raw and raw["bytes"] is not None:
            try:
                return Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
            except Exception:
                pass

        # PNG/JPG file path
        if "path" in raw and raw["path"]:
            try:
                return Image.open(raw["path"]).convert("RGB")
            except Exception:
                pass

        # filename
        if "filename" in raw and raw["filename"]:
            try:
                return Image.open(raw["filename"]).convert("RGB")
            except Exception:
                pass

    # case 2: directly bytes
    if isinstance(raw, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return None

    # case 3: already PIL
    if isinstance(raw, Image.Image):
        return raw

    # case 4: string path
    if isinstance(raw, str) and os.path.exists(raw):
        try:
            return Image.open(raw).convert("RGB")
        except Exception:
            return None

    return None


def extract_parquet(parquet_path, outdir, split):
    parquet_path = Path(parquet_path)
    outdir = Path(outdir)
    extract_root = outdir / "extracted" / split

    print(f"📦 Loading parquet: {parquet_path}")
    table = pq.read_table(parquet_path)
    data = table.to_pylist()
    total = len(data)
    print(f"📑 Total records: {total}")

    # Auto-detect image field
    image_key = auto_find_image_key(data[0])
    if image_key is None:
        raise ValueError("❌ 找不到 image 字段！请检查 parquet 的结构。")

    print(f"🖼  Detected image field: '{image_key}'")

    extracted_count = 0
    created_chunks = set()

    for idx, record in enumerate(tqdm(data, desc="Extracting")):
        raw_img = record.get(image_key)
        img = decode_image(raw_img)

        if img is None:
            print(f"[WARN] Cannot decode image at index {idx}, skip.")
            continue

        extracted_count += 1

        # chunk directory
        base_name = f"{extracted_count:08d}"
        chunk_idx = (extracted_count - 1) // 1000 + 1
        chunk_dir = extract_root / f"{chunk_idx:08d}"

        if chunk_dir not in created_chunks:
            chunk_dir.mkdir(parents=True, exist_ok=True)
            created_chunks.add(chunk_dir)

        # determine file extension
        ext = ".png"
        if isinstance(raw_img, dict):
            fn = raw_img.get("filename") or raw_img.get("path")
            if fn:
                _, e = os.path.splitext(fn)
                if e.lower() in [".jpg", ".jpeg", ".png"]:
                    ext = ".jpg" if e.lower() == ".jpeg" else e.lower()

        if ext not in (".jpg", ".png"):
            ext = ".png"

        save_format = "JPEG" if ext == ".jpg" else "PNG"
        img_path = chunk_dir / f"{base_name}{ext}"

        # save image
        try:
            img.save(img_path, format=save_format)
        except Exception as e:
            print(f"[WARN] Failed saving {img_path}: {e}")
            continue

        # save JSON annotation
        ann = {
            "image_filename": img_path.name,
            "split": split,
        }

        # copy all non-image fields to annotation
        for k, v in record.items():
            if k != image_key:
                ann[k] = v

        ann_path = chunk_dir / f"{base_name}.json"
        with ann_path.open("w", encoding="utf-8") as f:
            json.dump(ann, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Extraction complete! Saved {extracted_count} images to:\n   {extract_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, help="Path to parquet file")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--split", default="train", help="Dataset split name")
    args = parser.parse_args()

    extract_parquet(args.parquet, args.outdir, args.split)


if __name__ == "__main__":
    main()
