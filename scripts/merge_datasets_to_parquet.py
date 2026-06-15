#!/usr/bin/env python3
"""Merge existing OCEC dataset with MRL Eye dataset into a combined parquet.

This produces a unified dataset that preserves the original train/val splits
from both source datasets, with source tags to track origin.

Output: /10/cvz/guochuang/dataset/ocec_combined/dataset.parquet

Dataset composition:
  - Existing OCEC (real_data): 11,556 samples  (98% closed, 2% open)
  - MRL Eye Dataset:           84,898 samples  (49% closed, 51% open)
  ─────────────────────────────────────────────────────────────────
  - Combined total:            96,454 samples  (55% closed, 45% open)

Splits:
  - Train: 9,244 (existing) + 73,313 (MRL) = 82,557
  - Val:   2,312 (existing) + 11,585 (MRL) = 13,897
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


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
    args = parser.parse_args()

    # Load both datasets
    print(f"Loading existing:  {args.existing}")
    df_existing = pd.read_parquet(args.existing)
    print(f"  {len(df_existing)} rows, split: {df_existing['split'].value_counts().to_dict()}")

    print(f"Loading MRL:       {args.mrl}")
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
    # Group by source prefix
    existing_src = [s for s in sources.index if s == "real_data"]
    mrl_src = [s for s in sources.index if s.startswith("mrl_")]
    print(f"  OCEC existing: {sum(sources[s] for s in existing_src)} rows "
          f"({len(existing_src)} source(s))")
    print(f"  MRL subjects:  {sum(sources[s] for s in mrl_src)} rows "
          f"({len(mrl_src)} source(s))")

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
