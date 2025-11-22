#!/usr/bin/env python3
"""
Utility script to rebuild `data/cropped/annotation.csv` after manual edits.

Typical usage:

```bash
python rebuild_cropped_annotation.py \
    --root data/cropped \
    --output data/cropped/annotation.csv
```

The script walks through all cropped image files, infers their class IDs from
the file names (e.g. containing "closed" or "open"), and writes a fresh CSV.
You can add extra keyword→class mappings with `--label-map` if your naming
scheme differs.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


DEFAULT_LABEL_RULES: List[Tuple[str, int]] = [
    ("closeed", 0),  # common typo kept for backward compatibility
    ("closed", 0),
    ("open", 1),
]


def parse_label_rules(extra_rules: Sequence[str], include_defaults: bool) -> List[Tuple[str, int]]:
    rules: List[Tuple[str, int]] = []
    if include_defaults:
        rules.extend(DEFAULT_LABEL_RULES)
    for raw_rule in extra_rules:
        if "=" not in raw_rule:
            raise ValueError(f"Invalid --label-map entry '{raw_rule}'. Use the form keyword=class_id.")
        keyword, class_str = raw_rule.split("=", 1)
        keyword = keyword.strip().lower()
        if not keyword:
            raise ValueError(f"Invalid --label-map entry '{raw_rule}': empty keyword.")
        try:
            class_id = int(class_str.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid class id in --label-map entry '{raw_rule}'.") from exc
        rules.append((keyword, class_id))
    return rules


def discover_images(root: Path, extensions: Iterable[str]) -> List[Path]:
    normalized_exts = {ext.lower() for ext in extensions}
    results: List[Path] = []
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() in normalized_exts:
            results.append(file_path)
    return sorted(results)


def infer_class_id(relative_path: Path, label_rules: Sequence[Tuple[str, int]]) -> Tuple[int, str]:
    rel_lower = str(relative_path).lower()
    for keyword, class_id in label_rules:
        if keyword in rel_lower:
            return class_id, keyword
    raise ValueError(f"Could not infer class id for '{relative_path}'. "
                     f"Add a --label-map entry to handle it.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild data/cropped/annotation.csv after manual edits.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data") / "cropped",
        help="Root directory containing cropped images (default: data/cropped)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "cropped" / "annotation.csv",
        help="Destination CSV path (default: data/cropped/annotation.csv)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Path prefix to store in the CSV. "
             "Defaults to the root path expressed relative to the current working directory.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="*",
        default=[".png", ".jpg", ".jpeg"],
        help="Image file extensions to include (default: .png .jpg .jpeg)",
    )
    parser.add_argument(
        "--label-map",
        type=str,
        nargs="*",
        default=[],
        metavar="KEY=ID",
        help="Additional keyword=class_id pairs used to infer labels from file names. "
             "Match is case-insensitive and checks if the keyword appears anywhere in the path.",
    )
    parser.add_argument(
        "--no-default-labels",
        action="store_true",
        help="Disable built-in keyword rules (closed/open). Only use mappings provided via --label-map.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write the CSV; just report how many files would be included.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information for each processed file.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        print(f"[ERROR] Root directory '{root}' does not exist.", file=sys.stderr)
        sys.exit(1)
    if not root.is_dir():
        print(f"[ERROR] Root path '{root}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    try:
        label_rules = parse_label_rules(args.label_map, include_defaults=not args.no_default_labels)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    if not label_rules:
        print("[ERROR] No label rules defined. Provide --label-map entries or omit --no-default-labels.", file=sys.stderr)
        sys.exit(1)

    images = discover_images(root, args.extensions)
    if not images:
        print(f"[WARN] No image files found under {root} matching extensions {args.extensions}.")

    if args.prefix:
        csv_prefix = args.prefix.rstrip("/\\")
    else:
        try:
            csv_prefix = str(root.relative_to(Path.cwd()))
        except ValueError:
            csv_prefix = str(root)
        csv_prefix = csv_prefix.replace("\\", "/")

    rows: List[Tuple[str, int]] = []
    unmatched_paths: List[Path] = []

    for img_path in images:
        rel_path = img_path.relative_to(root)
        try:
            class_id, keyword = infer_class_id(rel_path, label_rules)
        except ValueError:
            unmatched_paths.append(rel_path)
            continue
        csv_path = f"{csv_prefix}/{rel_path.as_posix()}"
        rows.append((csv_path, class_id))
        if args.verbose:
            print(f"[OK] {csv_path} -> class {class_id} (matched '{keyword}')")

    if unmatched_paths:
        print(f"[WARN] {len(unmatched_paths)} files have unknown labels. "
              "Add new --label-map entries to cover them, or rename the files.")
        for path in unmatched_paths[:20]:
            print(f"  - {path}")
        if len(unmatched_paths) > 20:
            print("  ... (list truncated)")

    rows.sort(key=lambda item: item[0])

    if args.dry_run:
        print(f"[DRY-RUN] Would write {len(rows)} labeled entries to {args.output}")
        if unmatched_paths:
            sys.exit(2)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        for csv_path, class_id in rows:
            writer.writerow([csv_path, class_id])

    print(f"[DONE] Wrote {len(rows)} entries to {args.output}")
    if unmatched_paths:
        print("[NOTE] Some files were skipped due to missing label rules.")
        sys.exit(2)


if __name__ == "__main__":
    main()

