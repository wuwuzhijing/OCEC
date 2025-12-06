from __future__ import annotations

import io
import logging
import time
import threading
import multiprocessing
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x # 如果没安装，则不显示进度条

DEFAULT_MEAN = [0.42740763060926645, 0.39452373400411184, 0.38825020228993445]
DEFAULT_STD  = [0.194970523692498, 0.20245007710754218, 0.20778578022977737]

_SPLIT_ALIASES: Dict[str, str] = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
    "testing": "test",
}


@dataclass(frozen=True)
class Sample:
    """Lightweight container describing a single cropped eye frame."""

    index: int
    path: str
    resolved_path: Optional[Path]
    label: int
    label_name: str
    split: str
    source: str
    base_frame: str
    video_name: str
    image_bytes: Optional[bytes]


def _resolve_dataset_paths(data_root: Path) -> List[Path]:
    """
    Resolve dataset parquet file(s) from data_root.
    Returns a list of parquet file paths.
    """
    parquet_files = []
    
    if data_root.is_file() and data_root.suffix == ".parquet":
        # Single file
        return [data_root]
    
    if data_root.is_dir():
        # Look for parquet files in directory
        # First try: dataset_*.parquet (split files)
        parquet_files = sorted(data_root.glob("dataset_*.parquet"))
        if parquet_files:
            return parquet_files
        
        # Second try: dataset.parquet (single file)
        candidate = data_root / "dataset.parquet"
        if candidate.exists():
            return [candidate]
        
        # Third try: any *.parquet files
        parquet_files = sorted(data_root.glob("*.parquet"))
        if parquet_files:
            return parquet_files
    
    raise FileNotFoundError(
        f"Could not locate dataset parquet file(s) under {data_root}. "
        f"Expected: a .parquet file, or a directory containing dataset_*.parquet or dataset.parquet"
    )


def _normalize_split(value: str) -> Optional[str]:
    if value is None:
        return None
    key = str(value).strip().lower()
    return _SPLIT_ALIASES.get(key)


def _to_label(row: pd.Series) -> int:
    if "class_id" in row and pd.notna(row["class_id"]):
        return int(row["class_id"])
    label_text = str(row.get("label", "")).strip().lower()
    if label_text not in {"open", "closed"}:
        raise ValueError(f"Unsupported label value: {label_text!r}")
    return 1 if label_text == "open" else 0


def _prepare_image_bytes(value: object) -> Optional[bytes]:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    return None


def collect_samples(data_root: Path, logger: Optional[logging.Logger] = None) -> List[Sample]:
    """Load samples from one or more parquet dataset files."""

    logger = logger or logging.getLogger(__name__)
    dataset_paths = _resolve_dataset_paths(data_root)
    
    if len(dataset_paths) > 1:
        logger.info(f"Loading samples from {len(dataset_paths)} parquet files...")
        for path in dataset_paths:
            logger.info(f"  - {path.name}")
    
    # Load and merge all parquet files
    dfs = []
    for dataset_path in dataset_paths:
        df_chunk = pd.read_parquet(dataset_path)
        if "split" not in df_chunk.columns:
            raise ValueError(f"Dataset at {dataset_path} must contain a 'split' column.")
        dfs.append(df_chunk)
    
    # Merge all dataframes
    df = pd.concat(dfs, ignore_index=True)
    
    if len(dataset_paths) > 1:
        logger.info(f"Merged {len(dataset_paths)} parquet files into {len(df)} total rows.")

    samples: List[Sample] = []
    # Use the first parquet file's directory as base for relative paths
    dataset_dir = dataset_paths[0].parent
    for idx, row in df.iterrows():
        split = _normalize_split(row.get("split"))
        if split is None:
            logger.debug("Skipping row %d with unsupported split value %r.", idx, row.get("split"))
            continue

        raw_path = str(row.get("image_path", "")).strip()
        if not raw_path:
            logger.debug("Skipping row %d without image_path.", idx)
            continue

        display_path = raw_path.replace("\\", "/")
        path_candidate = Path(display_path)
        resolved_path: Optional[Path] = None
        if path_candidate.is_absolute() and path_candidate.exists():
            resolved_path = path_candidate
        else:
            candidate = (dataset_dir / path_candidate).resolve()
            if candidate.exists():
                resolved_path = candidate

        label = _to_label(row)
        label_name = str(row.get("label", "open" if label == 1 else "closed"))
        source = str(row.get("source", "unknown") or "unknown")
        base_frame = Path(display_path).name
        video_group = source if source else Path(display_path).parent.name

        samples.append(
            Sample(
                index=int(idx),
                path=display_path,
                resolved_path=resolved_path,
                label=label,
                label_name=label_name,
                split=split,
                source=source,
                base_frame=base_frame,
                video_name=video_group,
                image_bytes=_prepare_image_bytes(row.get("image_bytes")),
            )
        )

    if not samples:
        dataset_info = f"{len(dataset_paths)} file(s)" if len(dataset_paths) > 1 else str(dataset_paths[0])
        raise RuntimeError(f"No usable samples found in {dataset_info}.")

    total = len(samples)
    counts = Counter(sample.split for sample in samples)
    dataset_info = f"{len(dataset_paths)} file(s)" if len(dataset_paths) > 1 else dataset_paths[0].name
    logger.info(
        "Loaded %d samples from %s (splits: %s).",
        total,
        dataset_info,
        ", ".join(f"{name}={counts.get(name, 0)}" for name in ("train", "val", "test")),
    )
    return samples


def split_samples(
    samples: Sequence[Sample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,  # pragma: no cover - retained for interface compatibility
    logger: Optional[logging.Logger] = None,
) -> Dict[str, List[Sample]]:
    """Group samples by pre-defined split labels."""

    logger = logger or logging.getLogger(__name__)
    logger.info(
        "Using pre-defined dataset splits; ratio arguments (train=%.3f, val=%.3f, test=%.3f) are informational only.",
        train_ratio,
        val_ratio,
        test_ratio,
    )
    assignments: Dict[str, List[Sample]] = {split: [] for split in ("train", "val", "test")}
    for sample in samples:
        if sample.split in assignments:
            assignments[sample.split].append(sample)
        else:
            logger.debug("Ignoring sample %s with unsupported split '%s'.", sample.path, sample.split)

    for split, subset in assignments.items():
        if not subset:
            continue
        positives = sum(sample.label for sample in subset)
        total = len(subset)
        open_ratio = (positives / total) * 100.0 if total else 0.0
        logger.info(
            "Split '%s': %d samples from %d sources (%.1f%% labelled 'open').",
            split,
            total,
            len({sample.video_name for sample in subset}),
            open_ratio,
        )

    if not assignments["train"]:
        raise ValueError("Training split is empty. Ensure the dataset includes 'train' rows.")

    return assignments


class OCECDataset(Dataset):
    """Torch dataset that reads eyes crops from disk or embedded bytes."""

    def __init__(self, samples: Sequence[Sample], transform=None, confidence_dict=None) -> None:
        self.samples = list(samples)
        self.transform = transform

        # ---- Confidence support ----
        self._has_confidence = confidence_dict is not None
        if self._has_confidence:
            self.confidence = [confidence_dict.get(s.path, 1.0) for s in samples]
        else:
            self.confidence = [1.0] * len(samples)

        logging.info(f"Pre-loading {len(samples)} image data into RAM...")
        self._image_cache = []
        # 遍历所有样本，执行耗时的解码操作
        for sample in tqdm(self.samples, desc="Loading images to RAM"):
            # 仅在初始化时执行一次昂贵的解码操作
            # Image.open(io.BytesIO(...)) 和 .convert("RGB") 是耗时操作
            try:
                image = Image.open(io.BytesIO(sample.image_bytes)).convert("RGB")
                self._image_cache.append(image)
            except Exception as e:
                logging.warning(f"Failed to load image at index {sample.index}. Error: {e}")
                # 遇到错误时，存储一个 None 或一个占位符，避免崩溃
                self._image_cache.append(None) 
                
        logging.info(f"Finished loading {len(self._image_cache)} images to RAM.")            
        # Pre-detect if transform is from albumentations
        self._is_albumentations = False
        if transform is not None:
            # Method 1: Try to import and check isinstance
            try:
                import albumentations
                if isinstance(transform, albumentations.core.composition.Compose):
                    self._is_albumentations = True
            except (ImportError, AttributeError):
                pass
            
            # Method 2: Check module name
            if not self._is_albumentations:
                try:
                    module_name = transform.__class__.__module__
                    if 'albumentations' in module_name:
                        self._is_albumentations = True
                except (AttributeError, TypeError):
                    pass
            
            # Method 3: Check type string
            if not self._is_albumentations:
                try:
                    type_str = str(type(transform))
                    if 'albumentations' in type_str.lower():
                        self._is_albumentations = True
                except (AttributeError, TypeError):
                    pass

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, sample: Sample) -> Image.Image:
        if sample.image_bytes is not None:
            return Image.open(io.BytesIO(sample.image_bytes)).convert("RGB")
        if sample.resolved_path is None:
            raise FileNotFoundError(f"Image file not found for sample {sample.path!r}.")
        return Image.open(sample.resolved_path).convert("RGB")

    def __getitem__(self, index: int):
        sample = self.samples[index]
        # image = self._load_image(sample)
        image = self._image_cache[index]
        if image is None:
            # 如果缓存中是 None，说明加载失败了，跳过或报错
            raise RuntimeError(f"Image at index {index} was not loaded correctly during initialization.")
        if self.transform is not None:
            if self._is_albumentations:
                # It's albumentations: convert PIL to numpy and use named args
                import numpy as np
                image_np = np.array(image)
                result = self.transform(image=image_np)
                image = result['image']
            else:
                # It's torchvision: use positional args with PIL Image
                image = self.transform(image)
        label = torch.tensor(float(sample.label), dtype=torch.float32)
        return {
            "image": image,
            "label": label,
            "confidence": torch.tensor(self.confidence[index], dtype=torch.float32),
            "video_name": sample.video_name,
            "path": sample.path,
            "base_frame": sample.base_frame,
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 8,
    sampler: Optional[WeightedRandomSampler] = None,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs for faster data loading
        prefetch_factor=8 if num_workers > 0 else None,  # Increased prefetch for better GPU utilization
        drop_last=False,  # Don't drop last incomplete batch
    )


def build_weighted_sampler(samples: Sequence[Sample]) -> WeightedRandomSampler:
    labels = [sample.label for sample in samples]
    counts = Counter(labels)
    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
