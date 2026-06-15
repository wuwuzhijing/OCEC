#!/usr/bin/env python3
"""Merge existing OCEC dataset with MRL Eye dataset into a combined parquet.

Key fix: the existing OCEC data comes from a single recording session.
The original random 80/20 split causes severe frame leakage (consecutive
frames in train and val).  This script now re-splits the existing data by
temporal order (last 20% of frames per label group → val), eliminating
the leakage.

Output: /10/cvz/guochuang/dataset/ocec_combined/dataset.parquet

Dataset composition:
  - Existing OCEC (real_data): 11,556 samples  (98% closed, 2% open)
  - MRL Eye Dataset:           84,898 samples  (49% closed, 51% open)
  ─────────────────────────────────────────────────────────────────
  - Combined total:            96,454 samples  (55% closed, 45% open)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def _extract_frame_index(path_str: str) -> int:
    """Extract frame index from filename for temporal ordering.

    Examples:
        closed_0000000.png       → 0
        closed_0000000111.png    → 111
        open_00000131_2.png      → 131
        open_00000137_1.png      → 137
    """
    stem = Path(path_str).stem
    # Match digits after open_ or closed_ prefix
    m = re.search(r'(?:open|closed)_(\d+)', stem)
    if m:
        return int(m.group(1))
    return 0


def re_split_existing_by_time(df: pd.DataFrame, val_ratio: float = 0.2) -> pd.DataFrame:
    """Re-split existing OCEC data by temporal order to prevent frame leakage.

    Within each label group (open/closed), sorts by frame index and assigns
    the last `val_ratio` portion to val, earlier frames to train.

    This ensures val contains temporally distinct frames, giving honest metrics.
    """
    df = df.copy()

    # Extract frame index
    df["_frame_idx"] = df["image_path"].apply(_extract_frame_index)

    # Sort by label then frame index, then assign split
    train_parts = []
    val_parts = []

    for label_name, group in df.groupby("label"):
        group = group.sort_values("_frame_idx")
        n_val = max(1, int(len(group) * val_ratio))
        if n_val >= len(group):
            n_val = max(1, len(group) // 5)  # at least 1 val sample, at most 20%

        group_train = group.iloc[:-n_val].copy()
        group_val = group.iloc[-n_val:].copy()

        group_train["split"] = "train"
        group_val["split"] = "val"

        train_parts.append(group_train)
        val_parts.append(group_val)
        print(f"  {label_name}: {len(group_train)} train (frames {group_train['_frame_idx'].min()}-{group_train['_frame_idx'].max()}), "
              f"{len(group_val)} val (frames {group_val['_frame_idx'].min()}-{group_val['_frame_idx'].max()})")

    result = pd.concat(train_parts + val_parts, ignore_index=True)
    result.drop(columns=["_frame_idx"], inplace=True)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Merge existing OCEC + MRL Eye datasets into one combined parquet"
    )
    parser.add_argument(
        "--existing",
        type=Path,
        default=Path("/103/guochuang/Code/myOCEC/data/dataset.parquet"),
        help="Path to existing OCEC parquet",
    )
    parser.add_argument(
        "--mrl",
        type=Path,
        default=Path("/10/cvz/guochuang/dataset/mrl_eyes_2018/dataset.parquet"),
        help="Path to MRL Eye dataset parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/10/cvz/guochuang/dataset/ocec_combined"),
        help="Output directory for merged parquet",
    )
    parser.add_argument(
        "--existing-val-ratio",
        type=float,
        default=0.2,
        help="Val ratio for existing data temporal re-split (default: 0.2)",
    )
    args = parser.parse_args()

    # Load both datasets
    print(f"Loading existing:  {args.existing}")
    df_existing = pd.read_parquet(args.existing)
    print(f"  {len(df_existing)} rows, original split: {df_existing['split'].value_counts().to_dict()}")
    print(f"  Label: {df_existing['label'].value_counts().to_dict()}")

    # ── Re-split existing by temporal order ──
    print("\nRe-splitting existing data by temporal order (fixing frame leakage)...")
    df_existing = re_split_existing_by_time(df_existing, args.existing_val_ratio)
    print(f"  New split: {df_existing['split'].value_counts().to_dict()}")

    print(f"\nLoading MRL:       {args.mrl}")
    df_mrl = pd.read_parquet(args.mrl)
    print(f"  {len(df_mrl)} rows, split: {df_mrl['split'].value_counts().to_dict()}")

    # Verify column match
    assert df_existing.columns.tolist() == df_mrl.columns.tolist(), (
        f"Column mismatch!\n  Existing: {df_existing.columns.tolist()}\n  MRL: {df_mrl.columns.tolist()}"
    )

    # Merge
    df_combined = pd.concat([df_existing, df_mrl], ignore_index=True)

    print(f"\nCombined: {len(df_combined)} rows")
    print(f"  Split:  {df_combined['split'].value_counts().to_dict()}")
    print(f"  Label:  {df_combined['label'].value_counts().to_dict()}")
    for s in ["train", "val"]:
        sub = df_combined[df_combined["split"] == s]
        lb = sub["label"].value_counts()
        print(f"    {s}: closed={lb.get('closed', 0)}, open={lb.get('open', 0)} "
              f"({lb.get('open', 0) / len(sub) * 100:.1f}% open)")

    # Show source composition
    sources = df_combined["source"].value_counts()
    print(f"\nSources ({len(sources)} unique):")
    existing_src = [s for s in sources.index if s == "real_data"]
    mrl_src = [s for s in sources.index if s.startswith("mrl_")]
    print(f"  OCEC existing: {sum(sources[s] for s in existing_src)} rows "
          f"({len(existing_src)} source(s)) - temporally split")
    print(f"  MRL subjects:  {sum(sources[s] for s in mrl_src)} rows "
          f"({len(mrl_src)} source(s)) - subject split")

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "dataset.parquet"
    df_combined.to_parquet(output_path, index=False)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved: {output_path}")
    print(f"  Rows: {len(df_combined)}")
    print(f"  Size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
