#!/usr/bin/env python3
"""Convert MRL Eye Dataset to OCEC-compatible parquet format.

MRL filename format:
  sXXXX_NNNNN_GENDER_GLASSES_EYESTATE_REFLECTION_LIGHTING_SENSOR.png

Fields (0-indexed after split by _):
  [0] subject ID (sXXXX)
  [1] image number
  [2] gender (0=male, 1=female)
  [3] glasses (0=no, 1=yes)
  [4] eye state (0=close, 1=open)  ← target label
  [5] reflections (0=none, 1=low, 2=high)
  [6] lighting conditions (0=bad, 1=good)
  [7] sensor type (01/02/03)

Output: parquet with columns [split, label, class_id, image_path, source, image_bytes]
"""

from __future__ import annotations

import argparse
import io
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image
from tqdm import tqdm


def parse_mrl_filename(filename: str) -> Optional[dict]:
    """Parse MRL filename and return metadata dict, or None if invalid."""
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 8:
        return None
    try:
        return {
            "subject": parts[0],
            "image_num": int(parts[1]),
            "gender": int(parts[2]),
            "glasses": int(parts[3]),
            "eye_state": int(parts[4]),  # 0=close, 1=open
            "reflections": int(parts[5]),
            "lighting": int(parts[6]),
            "sensor": parts[7],
        }
    except (ValueError, IndexError):
        return None


def collect_mrl_samples(mrl_root: Path) -> List[dict]:
    """Walk MRL dataset and collect all valid samples."""
    samples = []
    subject_dirs = sorted(
        d for d in mrl_root.iterdir() if d.is_dir() and d.name.startswith("s")
    )
    print(f"Found {len(subject_dirs)} subject directories")

    for subj_dir in tqdm(subject_dirs, desc="Scanning subjects"):
        png_files = sorted(subj_dir.glob("*.png"))
        for png_path in png_files:
            meta = parse_mrl_filename(png_path.name)
            if meta is None:
                continue
            samples.append(
                {
                    "path": str(png_path),
                    "subject": meta["subject"],
                    "eye_state": meta["eye_state"],
                    "gender": meta["gender"],
                    "glasses": meta["glasses"],
                    "lighting": meta["lighting"],
                    "sensor": meta["sensor"],
                }
            )
    return samples


def split_by_subject(
    samples: List[dict], train_ratio: float, seed: int
) -> Dict[str, List[dict]]:
    """Split samples into train/val by subject with stratified open-ratio sampling.

    Ensures val set has a balanced open/closed ratio representative of the full
    dataset, rather than being skewed by unlucky subject selection.
    """
    subjects = sorted(set(s["subject"] for s in samples))

    # Calculate per-subject open ratio and sample count
    subj_info = {}
    for subj in subjects:
        subj_samples = [s for s in samples if s["subject"] == subj]
        open_count = sum(1 for s in subj_samples if s["eye_state"] == 1)
        subj_info[subj] = {
            "count": len(subj_samples),
            "open_ratio": open_count / len(subj_samples) if subj_samples else 0.0,
        }

    # Sort subjects by open ratio for stratified selection
    sorted_subjects = sorted(subjects, key=lambda s: subj_info[s]["open_ratio"])

    # Take every Nth subject for val to maintain ratio distribution
    val_ratio = 1.0 - train_ratio
    step = max(2, int(1.0 / val_ratio))
    rng = random.Random(seed)

    # Use random offset within step for variability
    offset = rng.randint(0, step - 1)
    val_subjects = set()
    for i, subj in enumerate(sorted_subjects):
        if (i + offset) % step == 0:
            val_subjects.add(subj)
    train_subjects = set(subjects) - val_subjects

    print(f"Train subjects ({len(train_subjects)}): {sorted(train_subjects)}")
    print(f"Val subjects ({len(val_subjects)}): {sorted(val_subjects)}")

    train_samples = [s for s in samples if s["subject"] in train_subjects]
    val_samples = [s for s in samples if s["subject"] in val_subjects]

    return {"train": train_samples, "val": val_samples}


def load_image_as_rgb_bytes(path: str) -> bytes:
    """Load image, convert to RGB, return PNG bytes."""
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_parquet(
    samples: List[dict],
    split: str,
    embed_images: bool,
) -> pd.DataFrame:
    """Build a DataFrame from samples for a given split."""
    rows = []
    for s in tqdm(samples, desc=f"Building {split} split"):
        label = "open" if s["eye_state"] == 1 else "closed"
        class_id = s["eye_state"]
        image_path = s["path"]
        source = f"mrl_{s['subject']}"
        row = {
            "split": split,
            "label": label,
            "class_id": class_id,
            "image_path": image_path,
            "source": source,
        }
        if embed_images:
            row["image_bytes"] = load_image_as_rgb_bytes(image_path)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Convert MRL Eye Dataset to parquet")
    parser.add_argument(
        "--mrl-root",
        type=Path,
        default=Path("/10/cvz/guochuang/dataset/MRL-Eye-Dataset/mrlEyes_2018_01"),
        help="Root directory of MRL dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/10/cvz/guochuang/dataset/mrl_eyes_2018"),
        help="Output directory for parquet file(s)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of subjects for training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Do not embed image bytes (store only paths)",
    )
    args = parser.parse_args()

    # Step 1: Collect all samples
    print(f"Scanning MRL dataset: {args.mrl_root}")
    samples = collect_mrl_samples(args.mrl_root)
    print(f"Total samples: {len(samples)}")

    # Quick stats
    labels = [s["eye_state"] for s in samples]
    label_counts = Counter(labels)
    print(f"Label distribution: closed={label_counts[0]}, open={label_counts[1]} "
          f"({label_counts[1]/len(samples)*100:.1f}% open)")

    # Step 2: Split by subject
    split_samples = split_by_subject(samples, args.train_ratio, args.seed)
    print(f"Split: train={len(split_samples['train'])}, val={len(split_samples['val'])}")

    for sname, slist in split_samples.items():
        sl = [x["eye_state"] for x in slist]
        sc = Counter(sl)
        print(f"  {sname}: closed={sc[0]}, open={sc[1]} "
              f"({sc[1]/len(slist)*100:.1f}% open)")

    # Step 3: Build parquet
    embed = not args.no_embed
    dfs = []
    for split_name in ["train", "val"]:
        df = build_parquet(split_samples[split_name], split_name, embed_images=embed)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Step 4: Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "dataset.parquet"
    combined.to_parquet(output_path, index=False)

    total_bytes = 0
    if embed and "image_bytes" in combined.columns:
        total_bytes = combined["image_bytes"].apply(len).sum()
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved: {output_path}")
    print(f"  Rows: {len(combined)}")
    print(f"  File size: {file_size_mb:.1f} MB")
    if embed:
        print(f"  Image data: {total_bytes / (1024*1024):.1f} MB")

    # Print label translation for reference
    print(f"\nLabel mapping:")
    print(f"  class_id=1 -> 'open'  -> 睁眼")
    print(f"  class_id=0 -> 'closed' -> 闭眼")


if __name__ == "__main__":
    main()
