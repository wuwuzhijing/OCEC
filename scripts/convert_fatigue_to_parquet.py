#!/usr/bin/env python3
"""
Convert fatigue dataset (gupengli's DSM project) to OCEC parquet format.

Fatigue classes → OCEC mapping:
  0_normal  → open   (class_id=1)
  1_squint  → closed (class_id=0)
  2_unknown → skip   (too few samples, ambiguous)

Output: parquet compatible with OCEC training pipeline.
"""

from __future__ import annotations

import argparse
import io
import random
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


CLASS_MAP = {
    "0_normal":  "open",
    "1_squint":  "closed",
    # "2_unknown": skipped
}

TRAIN_RATIO = 0.9  # per class


def collect_fatigue_samples(roots: list[Path]) -> list[dict]:
    samples = []
    for root in roots:
        if not root.is_dir():
            print(f"  [warn] {root} not found, skipping")
            continue
        for cls_dir in sorted(root.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls_name = cls_dir.name
            if cls_name not in CLASS_MAP:
                print(f"  [skip] {cls_name} (not in CLASS_MAP)")
                continue
            ocec_label = CLASS_MAP[cls_name]
            print(f"  {cls_dir} → {ocec_label}")
            for img_path in sorted(cls_dir.glob("*")):
                if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                    samples.append({
                        "path": str(img_path),
                        "class_name": cls_name,
                        "label": ocec_label,
                        "class_id": 1 if ocec_label == "open" else 0,
                        "source": f"fatigue_{cls_name}",
                    })
    return samples


def split_samples(samples: list[dict], train_ratio: float, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for label_name in ["open", "closed"]:
        label_samples = [s for s in samples if s["label"] == label_name]
        rng.shuffle(label_samples)
        n_train = max(1, int(len(label_samples) * train_ratio))
        for i, s in enumerate(label_samples):
            split = "train" if i < n_train else "val"
            row = {**s, "split": split}
            rows.append(row)
    return pd.DataFrame(rows)


def embed_image_bytes(df: pd.DataFrame) -> pd.DataFrame:
    image_bytes_list = []
    for path in tqdm(df["path"], desc="Embedding images"):
        try:
            img = Image.open(path).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_bytes_list.append(buf.getvalue())
        except Exception as e:
            print(f"  [warn] {path}: {e}")
            image_bytes_list.append(None)
    df["image_bytes"] = image_bytes_list
    return df[df["image_bytes"].notna()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dataset", type=Path,
                        default=Path("/home/shares/gupengli/dataset/fatigue_dataset/src_dataset"),
                        help="src_dataset root (2-class)")
    parser.add_argument("--L2-dataset", type=Path,
                        default=Path("/home/shares/gupengli/dataset/fatigue_dataset/L2+_dataset"),
                        help="L2+ dataset root (3-class)")
    parser.add_argument("--futian-dataset", type=Path, default=None,
                        help="futian_dataset root (optional)")
    parser.add_argument("--output", type=Path,
                        default=Path("/ssddisk/guochuang/ocec/fatigue_dataset"),
                        help="Output directory for parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    roots = [args.src_dataset, args.L2_dataset]
    if args.futian_dataset:
        roots.append(args.futian_dataset)

    print("Collecting fatigue samples...")
    samples = collect_fatigue_samples(roots)
    print(f"Total: {len(samples)} samples")

    for label in ["open", "closed"]:
        cnt = sum(1 for s in samples if s["label"] == label)
        print(f"  {label}: {cnt}")

    print("\nSplitting...")
    df = split_samples(samples, TRAIN_RATIO, args.seed)
    for s in ["train", "val"]:
        sub = df[df["split"] == s]
        lb = sub["label"].value_counts()
        print(f"  {s}: {len(sub)} total, open={lb.get('open',0)}, closed={lb.get('closed',0)}")

    print("\nEmbedding images...")
    df = embed_image_bytes(df)

    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / "dataset.parquet"
    df.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / 1024**2
    print(f"\nSaved: {out_path}  ({len(df)} rows, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
