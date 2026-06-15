#!/usr/bin/env python3
"""Error analysis: score distribution, FP/FN sample export.

Loads a trained model, runs inference on a dataset, and produces:
  1. Per-class score (probability) histograms
  2. FP/FN sample montages for visual inspection
  3. Top-k worst error samples with metadata

Usage:
  python3 scripts/analyze_errors.py \
      --checkpoint runs/ocec_mrl_v4/v1/ocec_best_epoch0045_f1_0.9350.pt \
      --data /ssddisk/guochuang/ocec/MRL/mrl_eyes_2018/ \
      --output output/error_analysis \
      --top-k 50
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Import OCEC
import torch
from torch.utils.data import DataLoader

from ocec.pipeline import _build_transforms, _resolve_device
from ocec.data import DEFAULT_MEAN, DEFAULT_STD, OCECDataset, collect_samples, split_samples
from ocec.model import OCEC, ModelConfig


def load_model(checkpoint_path: Path, device: torch.device) -> OCEC:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config_dict = payload.get("model_config")
    if model_config_dict is None:
        raise ValueError("Checkpoint missing model_config")

    model_config = ModelConfig(**model_config_dict)
    model = OCEC(model_config).to(device)

    state = payload.get("model_state") or payload.get("model")
    if state is None:
        raise ValueError("Checkpoint missing model state")

    # Strip 'module.' prefix if present
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    return model


def run_inference(
    model: OCEC,
    dataloader: DataLoader,
    device: torch.device,
) -> List[dict]:
    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device).long()
            paths = batch["path"]

            logits = model(images)
            if logits.ndim == 2 and logits.size(1) == 2:
                probs = torch.softmax(logits, dim=1)[:, 1]
            else:
                probs = torch.sigmoid(logits.squeeze())

            for i in range(len(paths)):
                results.append({
                    "path": paths[i],
                    "label": int(labels[i].item()),
                    "prob_open": float(probs[i].item()),
                })
    return results


def plot_score_histograms(results: List[dict], output_dir: Path) -> None:
    """Generate per-class score (open-eye probability) histograms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probs = np.array([r["prob_open"] for r in results])
    labels = np.array([r["label"] for r in results])

    pos_probs = probs[labels == 1]  # open eyes
    neg_probs = probs[labels == 0]  # closed eyes

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Full histogram
    ax = axes[0, 0]
    ax.hist(neg_probs, bins=50, alpha=0.6, label=f"Closed (n={len(neg_probs)})", color="red")
    ax.hist(pos_probs, bins=50, alpha=0.6, label=f"Open (n={len(pos_probs)})", color="green")
    ax.axvline(0.5, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("P(open)")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution (all)")
    ax.legend()

    # FP histogram: prob >= 0.5, label = 0
    fp_mask = (probs >= 0.5) & (labels == 0)
    fp_probs = probs[fp_mask]
    ax = axes[0, 1]
    ax.hist(fp_probs, bins=30, alpha=0.8, color="orange")
    ax.set_xlabel("P(open)")
    ax.set_title(f"False Positives (n={len(fp_probs)})")
    if len(fp_probs) > 0:
        ax.axvline(0.7, color="red", linestyle="--", label="high-conf FP >0.7")
        ax.legend()

    # FN histogram: prob < 0.5, label = 1
    fn_mask = (probs < 0.5) & (labels == 1)
    fn_probs = probs[fn_mask]
    ax = axes[1, 0]
    ax.hist(fn_probs, bins=30, alpha=0.8, color="blue")
    ax.set_xlabel("P(open)")
    ax.set_title(f"False Negatives (n={len(fn_probs)})")

    # Stats text
    ax = axes[1, 1]
    ax.axis("off")
    preds = (probs >= 0.5).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum(fp_mask))
    fn = int(np.sum(fn_mask))
    acc = (tp + tn) / len(labels)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    stats = (
        f"Total: {len(labels)}\n"
        f"TP={tp}, TN={tn}, FP={fp}, FN={fn}\n"
        f"Open ratio: {labels.mean():.1%}\n\n"
        f"Accuracy:  {acc:.4f}\n"
        f"Precision: {prec:.4f}\n"
        f"Recall:    {rec:.4f}\n"
        f"F1:        {f1:.4f}\n\n"
        f"Mean P(open) | closed: {neg_probs.mean():.3f}\n"
        f"Mean P(open) | open:   {pos_probs.mean():.3f}\n"
        f"KS distance: {ks_distance(probs, labels):.3f}"
    )
    ax.text(0.1, 0.5, stats, transform=ax.transAxes, fontfamily="monospace",
            verticalalignment="center", fontsize=11)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "score_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Score histograms saved: {path}")


def ks_distance(probs: np.ndarray, labels: np.ndarray) -> float:
    p0 = np.sort(probs[labels == 0])
    p1 = np.sort(probs[labels == 1])
    cdf0 = np.arange(1, len(p0) + 1) / len(p0) if len(p0) > 0 else np.array([0])
    cdf1 = np.arange(1, len(p1) + 1) / len(p1) if len(p1) > 0 else np.array([0])
    m = max(len(p0), len(p1))
    cdf0_i = np.interp(np.linspace(0, 1, m), np.linspace(0, 1, len(p0)), cdf0) if len(p0) > 0 else np.zeros(m)
    cdf1_i = np.interp(np.linspace(0, 1, m), np.linspace(0, 1, len(p1)), cdf1) if len(p1) > 0 else np.zeros(m)
    return float(np.max(np.abs(cdf0_i - cdf1_i)))


def export_error_montages(results: List[dict], output_dir: Path, top_k: int = 50) -> None:
    """Export FP and FN sample montages for visual inspection."""
    probs = np.array([r["prob_open"] for r in results])
    labels = np.array([r["label"] for r in results])
    indices = np.arange(len(results))

    output_dir.mkdir(parents=True, exist_ok=True)

    # High-confidence FP: prob >= 0.7, label = 0 (most dangerous)
    fp_high_idx = indices[(probs >= 0.7) & (labels == 0)]
    fp_high_idx = fp_high_idx[np.argsort(-probs[fp_high_idx])][:top_k]
    _save_montage([results[i] for i in fp_high_idx],
                  output_dir / "fp_high_conf.png", title="High-Confidence FP (prob>=0.7)")

    # All FP
    fp_idx = indices[(probs >= 0.5) & (labels == 0)]
    fp_idx = fp_idx[np.argsort(-probs[fp_idx])][:top_k]
    _save_montage([results[i] for i in fp_idx],
                  output_dir / "fp_all.png", title="All FP (prob>=0.5)")

    # FN: prob < 0.5, label = 1
    fn_idx = indices[(probs < 0.5) & (labels == 1)]
    fn_idx = fn_idx[np.argsort(probs[fn_idx])][:top_k]
    _save_montage([results[i] for i in fn_idx],
                  output_dir / "fn_all.png", title="False Negatives (prob<0.5)")

    print(f"Error montages saved: {output_dir}")


def _save_montage(samples: List[dict], path: Path, title: str, cols: int = 10) -> None:
    if not samples:
        print(f"  [skip] No samples for {path.name}")
        return

    thumb_w, thumb_h = 96, 96
    rows = math.ceil(len(samples) / cols)

    canvas = Image.new("RGB", (cols * thumb_w, rows * thumb_h + 30), "white")

    for i, s in enumerate(samples):
        try:
            img = Image.open(s["path"]).convert("RGB").resize((thumb_w, thumb_h))
            x = (i % cols) * thumb_w
            y = 30 + (i // cols) * thumb_h
            canvas.paste(img, (x, y))
            # Draw prob text below each image
        except Exception:
            continue

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    n = len(samples)
    print(f"  {path.name}: {n} samples, cols={cols}, rows={rows}")


def main():
    parser = argparse.ArgumentParser(description="Error analysis for OCEC model")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint .pt file")
    parser.add_argument("--data", type=Path, required=True, help="Parquet dataset directory or file")
    parser.add_argument("--output", type=Path, default=Path("output/error_analysis"), help="Output directory")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=50, help="Max FP/FN samples to export")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to analyze")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    print(f"Device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Load data
    samples = collect_samples(args.data)
    splits = split_samples(samples, train_ratio=0.8, val_ratio=0.2, test_ratio=0, seed=42)
    target_samples = splits.get(args.split, [])
    if not target_samples:
        raise ValueError(f"No samples for split '{args.split}'. Available: {list(splits.keys())}")

    print(f"Analyzing {len(target_samples)} {args.split} samples")

    transform, _ = _build_transforms((64, 64), DEFAULT_MEAN, DEFAULT_STD)
    dataset = OCECDataset(target_samples, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    # Run inference
    results = run_inference(model, loader, device)

    # Plot score histograms
    plot_score_histograms(results, args.output)

    # Export error montages
    export_error_montages(results, args.output, top_k=args.top_k)


if __name__ == "__main__":
    main()
