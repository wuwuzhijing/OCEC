from __future__ import annotations

import argparse
import copy
import importlib.util
import inspect
import json
import logging
import math
import re
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import v2 as transforms_v2
from PIL import Image
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import torch.nn.functional as F
import os
import subprocess
import atexit

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    raise ImportError(
        "albumentations is required for data augmentation. "
        "Install it with: pip install albumentations"
    )

CONF_THRESHOLD = 0.5

def plot_dual_pca(emb_raw, emb_arc, labels, save_path):
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        raise ImportError("sklearn is required for PCA visualization. Install it with: pip install scikit-learn")
    
    import matplotlib.pyplot as plt
    import numpy as np

    pca = PCA(n_components=2)
    raw2 = pca.fit_transform(emb_raw)
    arc2 = pca.fit_transform(emb_arc)

    plt.figure(figsize=(12,5))

    # Raw
    plt.subplot(1,2,1)
    for c in [0,1]:
        idx = np.where(labels == c)
        plt.scatter(raw2[idx,0], raw2[idx,1], s=6, label=str(c))
    plt.title("Raw Embedding PCA")

    # ArcFace
    plt.subplot(1,2,2)
    for c in [0,1]:
        idx = np.where(labels == c)
        plt.scatter(arc2[idx,0], arc2[idx,1], s=6, label=str(c))
    plt.title("ArcFace Embedding PCA")
    
    plt.savefig(save_path, dpi=160)
    plt.close()
    
def cluster_small_eye(emb, labels, image_paths, save_root, n_clusters=5):
    try:
        from sklearn.manifold import TSNE
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError("sklearn is required for clustering. Install it with: pip install scikit-learn")
    
    import os, shutil
    import numpy as np

    # 只聚类：label=0（闭眼）
    idx = np.where(labels == 0)[0]
    X = emb[idx]
    P = np.array(image_paths)[idx]

    # t-SNE 降维到 2D 再聚
    X_2d = TSNE(n_components=2, init="pca").fit_transform(X)

    km = KMeans(n_clusters=n_clusters, n_init=10).fit(X_2d)
    cluster_ids = km.labels_

    for c in range(n_clusters):
        cluster_dir = os.path.join(save_root, f"cluster_{c}")
        os.makedirs(cluster_dir, exist_ok=True)

        members = np.where(cluster_ids == c)[0]
        for m in members:
            shutil.copy(P[m], cluster_dir)

    return True

def create_montage(image_paths, save_path, cols=10, thumb_size=(64, 64)):
    import math
    from PIL import Image

    if len(image_paths) == 0:
        return

    rows = math.ceil(len(image_paths) / cols)
    mw = cols * thumb_size[0]
    mh = rows * thumb_size[1]

    canvas = Image.new("RGB", (mw, mh), "white")

    for idx, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize(thumb_size)
            x = (idx % cols) * thumb_size[0]
            y = (idx // cols) * thumb_size[1]
            canvas.paste(img, (x, y))
        except:
            continue

    canvas.save(save_path)


def plot_tsne_2d(emb, labels, save_path):
    try:
        from sklearn.manifold import TSNE
    except ImportError as e:
        raise ImportError(
            "sklearn is required for t-SNE visualization. "
            "Install it with: pip install scikit-learn. "
            f"Original error: {e}"
        )
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    try:
        tsne = TSNE(n_components=2, learning_rate='auto', init='pca')
        emb_2d = tsne.fit_transform(emb)
    
        plt.figure(figsize=(6, 6))
        for c in [0, 1]:
            idx = np.where(labels == c)
            plt.scatter(emb_2d[idx, 0], emb_2d[idx, 1], s=6, alpha=0.6, label=str(c))
        plt.legend()
        plt.title("t-SNE 2D Embedding")
        plt.savefig(save_path, dpi=160)
        plt.close()
    except Exception:
        plt.close()  # 确保关闭图形
        raise

def make_pca_video(pca_folder, output_video="pca3d.mp4", fps=5):
    cmd = [
        "ffmpeg",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", f"{pca_folder}/*.png",
        "-vf", "scale=800:-1",
        "-pix_fmt", "yuv420p",
        output_video
    ]
    subprocess.run(cmd)

def plot_pca_3d(emb, labels, save_path):
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        raise ImportError("sklearn is required for PCA visualization. Install it with: pip install scikit-learn")
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import numpy as np

    pca = PCA(n_components=3)
    emb3 = pca.fit_transform(emb)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    for c in [0,1]:
        idx = np.where(labels == c)
        ax.scatter(emb3[idx,0], emb3[idx,1], emb3[idx,2], s=6, alpha=0.6)

    ax.set_title("PCA 3D Embedding")
    plt.savefig(save_path, dpi=160)
    plt.close()


def detect_outliers_mahalanobis(emb, labels, threshold=3.0):
    import numpy as np

    outliers = []
    for c in [0,1]:
        idx = np.where(labels == c)[0]
        X = emb[idx]
        mu = X.mean(axis=0)
        cov = np.cov(X.T) + np.eye(X.shape[1]) * 1e-6
        inv_cov = np.linalg.inv(cov)

        for i in idx:
            diff = emb[i] - mu
            dist = np.sqrt(diff @ inv_cov @ diff)
            if dist > threshold:
                outliers.append((i, labels[i], dist))

    return sorted(outliers, key=lambda x: -x[2])

#
def detect_hard_samples(probs, labels, margin_th=0.15):
    probs = np.array(probs)
    margin = np.abs(probs - CONF_THRESHOLD)
    hard_idx = np.where(margin < margin_th)[0]
    return [(int(i), float(probs[i]), int(labels[i])) for i in hard_idx]



def detect_mislabeled(probs, labels, emb, threshold=0.15):
    # 思路：高置信度预测与标签冲突 + embedding 位置偏离
    import numpy as np

    wrong = []
    for i in range(len(probs)):
        p = probs[i]
        pred = int(p >= CONF_THRESHOLD)
        if pred != labels[i] and abs(p - CONF_THRESHOLD) > threshold:
            wrong.append((i, int(labels[i]), pred, float(p)))
    return wrong

class FocalLabelSmoothCE(nn.Module):
    def __init__(self, smoothing=0.05, gamma=2.0):
        super().__init__()
        self.smoothing = smoothing
        self.gamma = gamma

    def forward(self, logits, labels):
        """
        logits: [B, 2] 或 [B] 或 [B, 1]
        labels: [B]
        """
        # 处理单个分数的情况：将 [B] 或 [B, 1] 转换为 [B, 2]
        if logits.ndim == 1:
            # [B] -> [B, 2]: 第一列是闭眼分数(-logit)，第二列是睁眼分数(logit)
            logits = torch.stack([-logits, logits], dim=1)
        elif logits.ndim == 2 and logits.size(1) == 1:
            # [B, 1] -> [B, 2]
            logits = logits.squeeze(1)
            logits = torch.stack([-logits, logits], dim=1)
        
        num_classes = logits.size(1)

        # ===== Label smoothing =====
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, labels.unsqueeze(1), 1 - self.smoothing)

        # ===== Softmax =====
        probs = torch.softmax(logits, dim=1)

        # ===== Cross entropy =====
        ce_loss = -(true_dist * torch.log(probs + 1e-7)).sum(dim=1)

        # ===== Focal term =====
        pt = probs.gather(1, labels.unsqueeze(1)).squeeze()  # p_t
        focal_weight = (1 - pt) ** self.gamma

        loss = focal_weight * ce_loss
        return loss.mean()
    
class FocalLabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.05, gamma=2.0):
        super().__init__()
        self.smoothing = smoothing
        self.gamma = gamma

    def forward(self, logits, labels):
        labels = labels.float()
        with torch.no_grad():
            smooth = labels * (1 - self.smoothing) + 0.5 * self.smoothing

        prob = torch.sigmoid(logits)
        bce = F.binary_cross_entropy(prob, smooth, reduction='none')

        pt = prob * labels + (1 - prob) * (1 - labels)
        focal = (1 - pt) ** self.gamma

        return (focal * bce).mean()
    
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # initialize shadow parameters
        for name, p in model.named_parameters():
            if p.requires_grad:
                key = name
                # strip DP prefix from the name for shadow keys
                if key.startswith("module."):
                    key = key[7:]
                self.shadow[key] = p.data.clone()

    @torch.no_grad()
    def update(self, model):
        # EMA update
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            key = name
            if key.startswith("module."):
                key = key[7:]

            assert key in self.shadow, f"[EMA] Missing key: {key}"
            new_average = (1.0 - self.decay) * p.data + self.decay * self.shadow[key]
            self.shadow[key] = new_average.clone()

    @torch.no_grad()
    def apply(self, model):
        # Backup current params then load EMA shadow
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            key = name
            if key.startswith("module."):
                key = key[7:]

            if key not in self.shadow:
                print(f"[EMA] Warning: shadow missing for key {key}")
                continue

            self.backup[name] = p.data.clone()
            p.data = self.shadow[key].clone()

    @torch.no_grad()
    def restore(self, model):
        # Restore original params
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name in self.backup:
                p.data = self.backup[name]
        self.backup = {}
        
def compute_bayes_error(probs, labels, bins=200):
    p0 = probs[labels == 0]
    p1 = probs[labels == 1]
    hist0, edges = np.histogram(p0, bins=bins, range=(0, 1), density=True)
    hist1, _ = np.histogram(p1, bins=bins, range=(0, 1), density=True)
    overlap = np.minimum(hist0, hist1).sum() * (edges[1] - edges[0])
    return 0.5 * overlap

def compute_class_separation(embeddings, labels):
    emb = torch.tensor(embeddings)
    lab = torch.tensor(labels)

    cls0 = emb[lab == 0]
    cls1 = emb[lab == 1]

    mu0 = cls0.mean(dim=0)
    mu1 = cls1.mean(dim=0)

    intra0 = ((cls0 - mu0) ** 2).sum(dim=1).mean()
    intra1 = ((cls1 - mu1) ** 2).sum(dim=1).mean()
    intra = (intra0 + intra1) / 2

    inter = torch.dist(mu0, mu1)

    fisher = inter / (intra + 1e-8)

    sigma0 = cls0.var(dim=0) + 1e-8
    sigma1 = cls1.var(dim=0) + 1e-8
    bc = 0.25 * torch.sum(torch.log(0.25 * (sigma0/sigma1 + sigma1/sigma0 + 2)))
    bc += 0.25 * torch.sum((mu0 - mu1)**2 / (sigma0 + sigma1))

    return dict(intra=float(intra), inter=float(inter), fisher=float(fisher), bhatta=float(bc))

###############################################
# 可分性 / 概率分布 / 特征结构监控指标（新增）
###############################################

def ks_distance(probs, labels):
    probs = np.array(probs)
    labels = np.array(labels)
    p0 = np.sort(probs[labels == 0])
    p1 = np.sort(probs[labels == 1])

    cdf0 = np.arange(1, len(p0)+1) / len(p0)
    cdf1 = np.arange(1, len(p1)+1) / len(p1)

    m = max(len(p0), len(p1))
    cdf0_i = np.interp(np.linspace(0,1,m), np.linspace(0,1,len(p0)), cdf0)
    cdf1_i = np.interp(np.linspace(0,1,m), np.linspace(0,1,len(p1)), cdf1)

    return float(np.max(np.abs(cdf0_i - cdf1_i)))


def hellinger_distance(hist0, hist1):
    # 确保直方图值非负，避免 sqrt 计算错误
    hist0 = np.clip(hist0, 0, None)
    hist1 = np.clip(hist1, 0, None)
    # 归一化直方图
    sum0 = np.sum(hist0)
    sum1 = np.sum(hist1)
    if sum0 > 0:
        hist0 = hist0 / sum0
    if sum1 > 0:
        hist1 = hist1 / sum1
    # 计算 Hellinger 距离
    sqrt_product = np.sqrt(hist0 * hist1)
    bc = np.sum(sqrt_product)  # Bhattacharyya coefficient
    bc = np.clip(bc, 0, 1)  # 确保在 [0, 1] 范围内
    return float(np.sqrt(1 - bc))


def kl_divergence(p, q):
    eps = 1e-8
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return float(np.sum(p * np.log(p / q)))


def js_divergence(hist0, hist1):
    m = 0.5 * (hist0 + hist1)
    return 0.5 * kl_divergence(hist0, m) + 0.5 * kl_divergence(hist1, m)


def expected_calibration_error(probs, labels, bins=10):
    probs = np.array(probs)
    labels = np.array(labels)
    ece = 0.0
    for i in range(bins):
        s = i/bins
        e = (i+1)/bins
        mask = (probs >= s) & (probs < e)
        if mask.sum() == 0: 
            continue
        conf = probs[mask].mean()
        acc  = (probs[mask] > 0.5).astype(int).mean()
        ece += len(probs[mask]) / len(probs) * abs(acc - conf)
    return float(ece)


def margin_score(probs):
    probs = np.array(probs)
    return float(np.mean(np.abs(probs - 0.5)))


def embedding_pca_energy(embeddings, k=5):
    emb = np.array(embeddings)
    cov = np.cov(emb.T)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    energy = eigvals[:k] / eigvals.sum()
    return energy.tolist()


def sample_margin(probs, labels):
    return np.mean(np.abs(probs - 0.5))

def confidence_entropy(probs):
    eps = 1e-8
    return -(probs*np.log(probs+eps) + (1-probs)*np.log(1-probs+eps)).mean()
from .data import (
    DEFAULT_MEAN,
    DEFAULT_STD,
    OCECDataset,
    build_weighted_sampler,
    collect_samples,
    create_dataloader,
    split_samples,
)
from .model import OCEC, ModelConfig

LOGGER = logging.getLogger("ocec")

LABEL_MAP = {0: "closed", 1: "open"}

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEIM_MODULE_PATH = _REPO_ROOT / "03_wholebody34_data_extractor.py"
_DEFAULT_DETECTOR_MODEL = _REPO_ROOT / "deimv2_dinov3_x_wholebody34_680query_n_batch_640x640.onnx"
_DEIM_MODULE: Optional[ModuleType] = None


def _load_deim_module() -> ModuleType:
    global _DEIM_MODULE
    if _DEIM_MODULE is not None:
        return _DEIM_MODULE
    if not _DEIM_MODULE_PATH.exists():
        raise FileNotFoundError(
            f"DEIMv2 demo module not found at {_DEIM_MODULE_PATH}. Please ensure the script is available."
        )
    spec = importlib.util.spec_from_file_location("ocec._deimv2_detector", _DEIM_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load DEIMv2 demo module from {_DEIM_MODULE_PATH}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _DEIM_MODULE = module
    return module


def _create_mouth_detector(model_path: Path, providers: Sequence[Any]):
    module = _load_deim_module()
    if not model_path.exists():
        raise FileNotFoundError(f"Detector model not found: {model_path}")
    provider_list = list(providers) if providers else ["CPUExecutionProvider"]
    try:
        detector = module.DEIMv2(
            runtime="onnx",
            model_path=str(model_path),
            providers=provider_list,
        )
    except Exception as exc:  # pragma: no cover - defensive for runtime issues
        raise RuntimeError(f"Failed to initialize DEIMv2 mouth detector: {exc}") from exc
    return detector


def _resolve_onnx_providers(provider: str) -> List[Any]:
    key = (provider or "cpu").lower()
    if key == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if key == "tensorrt":
        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    'trt_engine_cache_enable': True, # .engine, .profile export
                    'trt_engine_cache_path': '.',
                    # 'trt_max_workspace_size': 4e9, # Maximum workspace size for TensorRT engine (1e9 ≈ 1GB)
                    # onnxruntime>=1.21.0 breaking changes
                    # https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#data-dependant-shape-dds-ops
                    # https://github.com/microsoft/onnxruntime/pull/22681/files
                    # https://github.com/microsoft/onnxruntime/pull/23893/files
                    'trt_op_types_to_exclude': 'NonMaxSuppression,NonZero,RoiAlign',
                    "trt_fp16_enable": True,
                }
            ),
            "CUDAExecutionProvider",
            'CPUExecutionProvider',
        ]
        return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _extract_mouth_crop(
    frame: np.ndarray,
    box: Any,
    *,
    margin_top: int,
    margin_bottom: int,
    margin_left: int,
    margin_right: int,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    if frame is None or frame.size == 0 or box is None:
        return None, None
    h, w = frame.shape[:2]
    x1 = max(int(box.x1) - margin_left, 0)
    y1 = max(int(box.y1) - margin_top, 0)
    x2 = min(int(box.x2) + margin_right, w - 1)
    y2 = min(int(box.y2) + margin_bottom, h - 1)
    if x2 <= x1 or y2 <= y1:
        return None, None
    crop = frame[y1 : y2 + 1, x1 : x2 + 1]
    if crop.size == 0:
        return None, None
    return crop.copy(), (x1, y1, x2, y2)


if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):

    _grad_scaler_params = inspect.signature(torch.amp.GradScaler).parameters

    if "device" in _grad_scaler_params:

        def _create_grad_scaler(enabled: bool):
            return torch.amp.GradScaler("cuda", enabled=enabled)

    elif "device_type" in _grad_scaler_params:

        def _create_grad_scaler(enabled: bool):
            return torch.amp.GradScaler(device_type="cuda", enabled=enabled)

    else:

        def _create_grad_scaler(enabled: bool):
            return torch.amp.GradScaler(enabled=enabled)

    _autocast_params = inspect.signature(torch.amp.autocast).parameters

    if "device_type" in _autocast_params:

        def _autocast(enabled: bool):
            if not enabled:
                return nullcontext()
            return torch.amp.autocast(device_type="cuda", enabled=True)

    else:

        def _autocast(enabled: bool):
            if not enabled:
                return nullcontext()
            return torch.amp.autocast("cuda", enabled=True)


else:
    from torch.cuda.amp import GradScaler as _CudaGradScaler
    from torch.cuda.amp import autocast as _cuda_autocast

    def _create_grad_scaler(enabled: bool):
        return _CudaGradScaler(enabled=enabled)

    def _autocast(enabled: bool):
        if not enabled:
            return nullcontext()
        return _cuda_autocast(enabled=True)


class RandomCLAHE:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8), p: float = 0.01) -> None:
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = tile_grid_size
        self.p = float(p)
        # Pre-create CLAHE object to avoid repeated initialization overhead
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def __call__(self, img: Image.Image) -> Image.Image:
        if torch.rand(1).item() >= self.p:
            return img
        np_img = np.array(img)
        lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_channel = self.clahe.apply(l_channel)
        lab = cv2.merge((l_channel, a_channel, b_channel))
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb)


class _BatchNormAffine(nn.Module):
    def __init__(self, scale: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = [1] * x.dim()
        if len(shape) >= 2:
            shape[1] = -1
        return x * self.scale.view(*shape) + self.bias.view(*shape)


def _decompose_batchnorms(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            running_mean = child.running_mean.detach()
            running_var = child.running_var.detach()
            if child.affine:
                weight = child.weight.detach()
                bias = child.bias.detach()
            else:
                weight = torch.ones_like(running_mean)
                bias = torch.zeros_like(running_mean)
            scale = weight / torch.sqrt(running_var + child.eps)
            bias_term = bias - running_mean * scale
            affine = _BatchNormAffine(scale, bias_term)
            setattr(module, name, affine)
        else:
            _decompose_batchnorms(child)


def _remove_batchnorm_from_onnx(model):
    from onnx import helper, numpy_helper

    graph = model.graph
    initializer_map = {init.name: init for init in graph.initializer}
    value_map = {name: numpy_helper.to_array(init) for name, init in initializer_map.items()}

    additional_initializers = []
    removed_initializers = set()
    new_nodes = []

    for node in graph.node:
        if node.op_type != "BatchNormalization":
            new_nodes.append(node)
            continue

        if len(node.input) < 5:
            new_nodes.append(node)
            continue

        inputs = node.input
        if any(name not in value_map for name in inputs[1:5]):
            new_nodes.append(node)
            continue

        eps = 1e-5
        for attr in node.attribute:
            if attr.name == "epsilon":
                eps = attr.f
                break

        scale = value_map[inputs[1]].astype(np.float32)
        bias = value_map[inputs[2]].astype(np.float32)
        mean = value_map[inputs[3]].astype(np.float32)
        var = value_map[inputs[4]].astype(np.float32)

        denom = np.sqrt(var + eps).astype(np.float32)
        alpha = (scale / denom).astype(np.float32)
        beta = (bias - mean * alpha).astype(np.float32)

        alpha_name = f"{node.output[0]}_bn_alpha"
        beta_name = f"{node.output[0]}_bn_beta"
        alpha_init = numpy_helper.from_array(alpha, name=alpha_name)
        beta_init = numpy_helper.from_array(beta, name=beta_name)
        additional_initializers.append(alpha_init)
        additional_initializers.append(beta_init)

        mul_out = f"{node.output[0]}_mul"
        mul_node = helper.make_node(
            "Mul",
            [inputs[0], alpha_name],
            [mul_out],
            name=f"{node.name}_Mul" if node.name else "",
        )
        add_node = helper.make_node(
            "Add",
            [mul_out, beta_name],
            [node.output[0]],
            name=f"{node.name}_Add" if node.name else "",
        )
        new_nodes.extend([mul_node, add_node])
        removed_initializers.update(inputs[1:5])

    graph.ClearField("node")
    graph.node.extend(new_nodes)

    remaining_initializers = [init for init in graph.initializer if init.name not in removed_initializers]
    existing_names = {init.name for init in remaining_initializers}
    for init in additional_initializers:
        if init.name not in existing_names:
            remaining_initializers.append(init)
            existing_names.add(init.name)
    graph.ClearField("initializer")
    graph.initializer.extend(remaining_initializers)

    return model


@dataclass
class TrainConfig:
    data_root: Path
    output_dir: Path
    margin_method: str = "cosface"  # "none", "arcface", "cosface"
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 8
    image_size: tuple[int, int] = (112, 112)
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    test_ratio: float = 0.0
    seed: int = 42
    base_channels: int = 32
    num_blocks: int = 4
    dropout: float = 0.3
    arch_variant: str = "baseline"
    head_variant: str = "auto"
    token_mixer_grid: tuple[int, int] = (2, 3)
    token_mixer_layers: int = 2
    device: str = "auto"
    resume_from: Optional[Path] = None
    pretrain_from: Optional[Path] = None  # 预训练权重路径（只加载模型权重，不恢复训练状态）
    use_amp: bool = False
    warmup_epochs: int = 5  # Warmup阶段的epoch数，0表示不使用warmup
    freeze_backbone: bool = False  # 是否冻结backbone（只训练head）
    unfreeze_backbone_epoch: Optional[int] = None  # 在指定epoch解冻backbone（用于渐进式微调）
    stage2_lr: Optional[float] = None  # 第二阶段学习率（如果为None，则使用lr * 0.05）

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["data_root"] = str(self.data_root)
        data["output_dir"] = str(self.output_dir)
        data["image_size"] = list(self.image_size)
        data["token_mixer_grid"] = list(self.token_mixer_grid)
        if self.resume_from is not None:
            data["resume_from"] = str(self.resume_from)
        if self.pretrain_from is not None:
            data["pretrain_from"] = str(self.pretrain_from)
        return data


def _ensure_image_size_tuple(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("Image dimensions must be positive integers.")
        return value, value
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"Expected tuple of length 2 for image size, got {value!r}.")
        height, width = value
    elif isinstance(value, list):
        if len(value) != 2:
            raise ValueError(f"Expected list of length 2 for image size, got {value!r}.")
        height, width = value
    else:
        raise ValueError(f"Unsupported image size specification: {value!r}.")

    height = int(height)
    width = int(width)
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive integers.")
    return height, width


def _parse_image_size_arg(raw: Any) -> tuple[int, int]:
    if isinstance(raw, (tuple, list, int)):
        try:
            return _ensure_image_size_tuple(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc

    if not isinstance(raw, str):
        raise argparse.ArgumentTypeError(f"Unsupported image size value: {raw!r}")

    text = raw.strip().lower().replace("×", "x").replace(",", "x")
    parts = [part for part in text.split("x") if part]
    try:
        if len(parts) == 1:
            size = int(parts[0])
            return _ensure_image_size_tuple(size)
        if len(parts) == 2:
            height, width = (int(part) for part in parts)
            return _ensure_image_size_tuple((height, width))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc

    raise argparse.ArgumentTypeError(
        "Image size must be specified as a single integer (e.g. '48') or as 'HEIGHTxWIDTH' (e.g. '64x48')."
    )


def _parse_token_mixer_grid_arg(raw: Any) -> tuple[int, int]:
    if isinstance(raw, (tuple, list)):
        values = list(raw)
    elif isinstance(raw, str):
        text = raw.strip().lower().replace("×", "x")
        normalized = re.sub(r"[,\s]+", "x", text)
        parts = [part for part in normalized.split("x") if part]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                "Token mixer grid must be provided as 'HEIGHTxWIDTH', e.g. '2x3'."
            )
        values = parts
    else:
        raise argparse.ArgumentTypeError(f"Unsupported token mixer grid value: {raw!r}")

    if len(values) != 2:
        raise argparse.ArgumentTypeError(
            "Token mixer grid must contain exactly two positive integers, e.g. '2x3'."
        )

    try:
        height, width = (int(values[0]), int(values[1]))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Token mixer grid values must be integers.") from exc

    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("Token mixer grid values must be positive integers.")
    return height, width


def _resolve_device(device_spec: str) -> torch.device:
    if device_spec and device_spec.lower() not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"Unsupported device specifier: {device_spec}")
    if device_spec is None or device_spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_spec == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_spec)


def _setup_logging(output_dir: Path, verbose: bool) -> None:
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")

    if LOGGER.handlers:
        for handler in list(LOGGER.handlers):
            LOGGER.removeHandler(handler)
            handler.close()

    handlers: List[logging.Handler] = []

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    handlers.append(console)

    output_dir.mkdir(parents=True, exist_ok=True)
    logfile = output_dir / "train.log"
    file_handler = logging.FileHandler(logfile, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    handlers.append(file_handler)

    for handler in handlers:
        LOGGER.addHandler(handler)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RandomNoiseWrapper(torch.nn.Module):
    def __init__(self, min_std=0.01, max_std=0.03, p=0.1):
        super().__init__()
        self.min_std, self.max_std, self.p = min_std, max_std, p

    def forward(self, x):
        if torch.rand(1) > self.p:
            return x
        std = torch.empty(1).uniform_(self.min_std, self.max_std).item()
        return x + torch.randn_like(x) * std

def _build_transforms(image_size: Any, mean: Sequence[float], std: Sequence[float]):
    """Hybrid CPU + GPU transform: IR-friendly"""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    height, width = _ensure_image_size_tuple(image_size)

    # CPU 部分（轻量 deterministic）：只负责 resize + normalize
    base_transform = A.Compose([
        A.Resize(height, width),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),   # torch tensor 输出
    ])

    # GPU 部分（随机transform，用于训练）
    import kornia.augmentation as K
    # gpu_aug = torch.nn.Sequential(
    #     K.RandomAffine(
    #         degrees=3, translate=(0.02, 0.02), scale=(0.97, 1.03),
    #         p=0.30, padding_mode="zeros"
    #     ),
    #     K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0), p=0.10),
    #     K.RandomGaussianNoise(std=(0.01, 0.03), p=0.10),
    #     K.RandomBrightness(0.25, p=0.35),
    #     K.RandomContrast(0.25, p=0.35),

    #     # ↓↓↓ Older Kornia API
    #     K.RandomJPEG(jpeg_quality=(50, 90), p=0.20)
    # )

    gpu_aug = torch.nn.Sequential(
        K.RandomAffine(
            degrees=3, translate=(0.02, 0.02), scale=(0.97, 1.03),
            p=0.30, padding_mode="zeros"
        ),

        # GaussianBlur: tuple sigma sometimes works, but we lock range manually
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.5, 1.2), p=0.10),

        # ❗ GaussianNoise must use scalar std for Kornia 0.8.2
        RandomNoiseWrapper(),

        K.RandomBrightness(brightness=0.25, p=0.35),
        K.RandomContrast(contrast=0.25, p=0.35),

        # Kornia 0.8.2 JPEG API
        K.RandomJPEG(jpeg_quality=(50, 90), p=0.20),
    )

    # ⚠️ 绑定属性（外部 pipeline 不改）
    base_transform.gpu_aug = gpu_aug
    return base_transform, base_transform   # train / val 由调用处决定是否应用 gpu_aug


def _freeze_backbone(model: nn.Module) -> None:
    """冻结backbone（stem和features），只训练head和margin_head"""
    # 处理DataParallel包装的模型
    base_model = model.module if hasattr(model, 'module') else model
    
    frozen_count = 0
    trainable_count = 0
    
    for name, param in base_model.named_parameters():
        # 冻结stem和features（backbone）
        if 'stem' in name or 'features' in name:
            param.requires_grad = False
            frozen_count += 1
        else:
            # head和margin_head保持可训练
            param.requires_grad = True
            trainable_count += 1
    
    LOGGER.info(f"Frozen {frozen_count} backbone parameters, {trainable_count} head parameters remain trainable")


def _unfreeze_backbone_last_layers(model: nn.Module, num_layers: int = 2) -> None:
    """
    解冻backbone的最后N层（features的最后N个block）
    num_layers: 解冻的层数（默认2层）
    """
    # 处理DataParallel包装的模型
    base_model = model.module if hasattr(model, 'module') else model
    
    # 获取features模块（Sequential）
    if not hasattr(base_model, 'features'):
        LOGGER.warning("Model does not have 'features' attribute")
        return
    
    features_module = base_model.features
    if not isinstance(features_module, nn.Sequential):
        LOGGER.warning("Features module is not Sequential, cannot unfreeze by layer")
        return
    
    # 获取features中的block数量
    num_blocks = len(features_module)
    if num_layers > num_blocks:
        LOGGER.warning(f"Requested {num_layers} layers but only {num_blocks} blocks available, unfreezing all blocks")
        num_layers = num_blocks
    
    # 解冻最后N个block
    unfrozen_count = 0
    for i in range(num_blocks - num_layers, num_blocks):
        block = features_module[i]
        for param in block.parameters():
            param.requires_grad = True
            unfrozen_count += 1
    
    LOGGER.info(f"Unfroze last {num_layers} blocks of features ({unfrozen_count} parameters)")
    
    # 同时解冻stem（如果需要的话，通常stem参数较少，解冻影响不大）
    # 这里我们只解冻features，stem保持冻结


def _unfreeze_backbone_fully(model: nn.Module) -> None:
    """完全解冻backbone"""
    # 处理DataParallel包装的模型
    base_model = model.module if hasattr(model, 'module') else model
    
    unfrozen_count = 0
    for name, param in base_model.named_parameters():
        if 'stem' in name or 'features' in name:
            param.requires_grad = True
            unfrozen_count += 1
    
    LOGGER.info(f"Fully unfroze {unfrozen_count} backbone parameters")


def _compute_pos_weight(samples: Sequence) -> torch.Tensor:
    positives = sum(sample.label for sample in samples)
    negatives = len(samples) - positives
    if positives == 0 or negatives == 0:
        LOGGER.warning("Cannot compute class-balanced pos_weight (pos=%d, neg=%d). Using 1.0.", positives, negatives)
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(negatives / positives, dtype=torch.float32)


def _infer_accuracy(train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]]) -> float:
    if val_metrics is not None:
        acc = val_metrics.get("accuracy")
        if acc is not None and not math.isnan(acc):
            return float(acc)
    return float(train_metrics.get("accuracy", 0.0))


def _prune_checkpoints(directory: Path, prefix: str, max_keep: int) -> None:
    checkpoints = sorted(directory.glob(f"{prefix}*.pt"))
    if len(checkpoints) <= max_keep:
        return
    for path in checkpoints[:-max_keep]:
        try:
            path.unlink()
        except FileNotFoundError:
            continue


def _run_epoch(
    model: nn.Module,
    dataloader: Optional[DataLoader],
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[Any] = None,
    autocast_enabled: bool = False,
    progress_desc: Optional[str] = None,
    collect_outputs: bool = False,
    margin_method: str = "cosface",  # "none", "arcface", "cosface"
    ema: Optional["EMA"] = None,     # 可选的 EMA 对象，用于在训练阶段更新滑动平均权重
) -> Tuple[Dict[str, float], Optional[Dict[str, np.ndarray]]]:
    all_embeddings = []
    if dataloader is None or len(dataloader.dataset) == 0:
        empty_metrics = {
            "loss": float("nan"),
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }
        return empty_metrics, None

    train_mode = optimizer is not None
    model.train(mode=train_mode)

    stats = {"loss": 0.0, "samples": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}
    collected_probs: List[torch.Tensor] = []
    collected_labels: List[torch.Tensor] = []

    iterator = dataloader
    if progress_desc:
        iterator = tqdm(iterator, desc=progress_desc, leave=False, dynamic_ncols=True)

    for batch in iterator:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).long()
        # === Apply GPU augment only during training ===
        if train_mode:
            if images.dtype != torch.float32:
                images = images.float()
            if hasattr(dataloader.dataset.transform, "gpu_aug"):
                gpu_aug = dataloader.dataset.transform.gpu_aug
                if gpu_aug is not None:
                    # Kornia expects (B,C,H,W) float and normalized input
                    images = gpu_aug(images)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        
        with _autocast(autocast_enabled):
            logits, embedding = model(images, labels=labels, return_embedding=True)
            use_margin = margin_method in ["arcface", "cosface"]
            if use_margin:
                # 损失函数使用带 margin 的 logits（训练时）
                # 验证时，如果 logits 不是 [B, 2] 形状，说明使用的是原始 head，需要特殊处理
                if train_mode:
                    # 训练时：使用 CrossEntropyLoss，logits 应该是 [B, 2] 形状
                    loss = criterion(logits, labels.long())
                else:
                    # 验证时：根据 logits 形状选择合适的损失函数
                    if logits.ndim == 2 and logits.size(1) == 2:
                        # [B, 2] 形状：使用 CrossEntropyLoss
                        loss = criterion(logits, labels.long())
                    else:
                        # [B] 或 [B, 1] 形状：使用 BCEWithLogitsLoss
                        logits_bce = logits.squeeze(1) if logits.ndim == 2 and logits.size(1) == 1 else logits
                        # 创建一个临时的 BCE 损失函数（不使用 pos_weight，因为验证时不需要）
                        bce_criterion = nn.BCEWithLogitsLoss()
                        loss = bce_criterion(logits_bce, labels.float())
                
                # 计算指标时，训练和验证都使用 margin_head（labels=None，不应用 margin）
                # 关键修复：确保训练和验证使用同一个分类器，指标计算方式一致
                if train_mode:
                    # 训练时：使用 margin_head 的 logits（labels=None，不应用 margin）用于指标计算
                    # 这样训练和验证的指标计算方式一致，更准确反映模型性能
                    with torch.no_grad():
                        # 获取 margin_head 的 logits（labels=None 时不应用 margin）
                        # 注意：需要从模型内部获取 margin_head，支持 DataParallel
                        if hasattr(model, 'module'):  # DataParallel 包装
                            margin_head = model.module.margin_head
                        else:
                            margin_head = model.margin_head
                        
                        if margin_head is not None:
                            # 使用 margin_head 获取 logits（labels=None 时不应用 margin）
                            logits_no_margin = margin_head(embedding, labels=None)#已经在模型中定义好了
                            # 应用 scale 以匹配训练时的 logits 范围
                            logits_no_margin_scaled = logits_no_margin
                            # 使用 softmax 计算概率
                            probs = torch.softmax(logits_no_margin_scaled, dim=1)[:, 1]
                            logits_for_debug = logits_no_margin_scaled
                        else:
                            # 如果没有 margin_head（不应该发生），回退到原始 logits
                            LOGGER.warning("Training: margin_method is set but margin_head is None, using original logits")
                            if logits.ndim == 2 and logits.size(1) == 2:
                                probs = torch.softmax(logits, dim=1)[:, 1]
                            else:
                                logits_bce = logits.squeeze(1) if logits.ndim == 2 and logits.size(1) == 1 else logits
                                probs = torch.sigmoid(logits_bce)
                            logits_for_debug = logits
                else:
                    # 验证时：使用 margin_head 的 logits（labels=None，不应用 margin，但使用同一个分类器）
                    # 关键修复：训练和验证必须使用同一个分类器（margin_head），否则会导致指标不一致
                    # 训练时：margin_head(embedding, labels) -> 带 margin 和 scale 的 logits
                    # 验证时：margin_head(embedding, labels=None) -> 不带 margin 但带 scale 的 logits
                    with torch.no_grad():
                        # 获取 margin_head 的 logits（labels=None 时不应用 margin）
                        # 注意：需要从模型内部获取 margin_head，支持 DataParallel
                        if hasattr(model, 'module'):  # DataParallel 包装
                            margin_head = model.module.margin_head
                        else:
                            margin_head = model.margin_head
                        
                        if margin_head is not None:
                            # 使用 margin_head 获取 logits（labels=None 时不应用 margin）
                            logits_margin = margin_head(embedding, labels=None)  # (B, 2)，范围 [-1, 1]
                            # 应用 scale 以匹配训练时的 logits 范围
                            logits_margin_scaled = logits_margin * margin_head.s_val  # (B, 2)，范围约 [-s, s]
                            # 使用 softmax 计算概率
                            probs = torch.softmax(logits_margin_scaled, dim=1)[:, 1]
                            logits_for_debug = logits_margin_scaled
                        else:
                            # 如果没有 margin_head（不应该发生，因为 use_margin=True），回退到原始逻辑
                            LOGGER.warning("Validation: margin_method is set but margin_head is None, using original logits")
                            if logits.ndim == 2 and logits.size(1) == 2:
                                probs = torch.softmax(logits, dim=1)[:, 1]
                            elif logits.ndim == 2 and logits.size(1) == 1:
                                probs = torch.sigmoid(logits.squeeze(1))
                            elif logits.ndim == 1:
                                probs = torch.sigmoid(logits)
                            else:
                                probs = torch.sigmoid(logits.squeeze() if logits.ndim > 1 else logits)
                            logits_for_debug = logits
            else:
                if logits.ndim == 2 and logits.shape[1] == 2:
                    logits_bce = logits[:, 1]
                else:
                    logits_bce = logits
                loss = criterion(logits_bce, labels.float())
                probs = torch.sigmoid(logits_bce)
                logits_for_debug = logits

        if train_mode:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            # 仅在训练阶段且提供了 EMA 对象时更新 EMA 权重
            if ema is not None:
                ema.update(model)

        batch_size = labels.size(0)
        stats["loss"] += loss.detach().item() * batch_size
        stats["samples"] += batch_size

        probs = torch.clamp(probs, min=0.0, max=1.0)
        preds = (probs >= CONF_THRESHOLD).long()
        
        labels_int = labels.long()
        stats["tp"] += ((preds == 1) & (labels_int == 1)).sum().item()
        stats["tn"] += ((preds == 0) & (labels_int == 0)).sum().item()
        stats["fp"] += ((preds == 1) & (labels_int == 0)).sum().item()
        stats["fn"] += ((preds == 0) & (labels_int == 1)).sum().item()
        
        # 调试信息：检查logits和probs的分布（训练和验证时都输出第一个batch）
        if stats["samples"] == batch_size:
            # 获取用于计算概率的 logits 用于调试（训练时是 logits_no_margin，验证时是 logits）
            # 如果 logits_for_debug 未定义（不应该发生），使用原始 logits
            if 'logits_for_debug' not in locals():
                logits_for_debug = logits
            debug_logits = logits_for_debug
            if debug_logits.ndim == 2 and debug_logits.size(1) == 2:
                logits_col0 = debug_logits[:, 0].detach()  # 闭眼类别的logit
                logits_col1 = debug_logits[:, 1].detach()  # 睁眼类别的logit
                logits_debug = logits_col1  # 用于显示睁眼类别的logit
                logits_col0_mean = logits_col0.mean().item()
                logits_col0_std = logits_col0.std().item()
                logits_col0_min = logits_col0.min().item()
                logits_col0_max = logits_col0.max().item()
                logits_col1_mean = logits_col1.mean().item()
                logits_col1_std = logits_col1.std().item()
                logits_col1_min = logits_col1.min().item()
                logits_col1_max = logits_col1.max().item()
                logits_info = (
                    f"col0: mean={logits_col0_mean:.3f}, std={logits_col0_std:.3f}, "
                    f"range=[{logits_col0_min:.3f}, {logits_col0_max:.3f}]; "
                    f"col1: mean={logits_col1_mean:.3f}, std={logits_col1_std:.3f}, "
                    f"range=[{logits_col1_min:.3f}, {logits_col1_max:.3f}]"
                )
            elif debug_logits.ndim == 2 and debug_logits.size(1) == 1:
                logits_debug = debug_logits.squeeze(1).detach()
                logits_info = "single column [B, 1]"
            else:
                logits_debug = debug_logits.detach()
                logits_info = "single dimension [B]"
            
            logits_mean = logits_debug.mean().item()
            logits_std = logits_debug.std().item()
            logits_min = logits_debug.min().item()
            logits_max = logits_debug.max().item()
            probs_mean = probs.detach().mean().item()
            probs_std = probs.detach().std().item()
            probs_min = probs.detach().min().item()
            probs_max = probs.detach().max().item()
            pred_pos_ratio = (preds == 1).float().mean().item()
            label_pos_ratio = (labels_int == 1).float().mean().item()
            batch_tp = ((preds == 1) & (labels_int == 1)).sum().item()
            batch_fp = ((preds == 1) & (labels_int == 0)).sum().item()
            batch_fn = ((preds == 0) & (labels_int == 1)).sum().item()
            batch_tn = ((preds == 0) & (labels_int == 0)).sum().item()
            split_name = "Train" if train_mode else "Val"
            LOGGER.info(
                f"{split_name} batch debug - logits shape: {debug_logits.shape}, "
                f"logits ({logits_info}), "
                f"probs: mean={probs_mean:.3f}, std={probs_std:.3f}, "
                f"range=[{probs_min:.3f}, {probs_max:.3f}], "
                f"pred_pos_ratio={pred_pos_ratio:.3f}, label_pos_ratio={label_pos_ratio:.3f}, "
                f"TP={batch_tp}, FP={batch_fp}, FN={batch_fn}, TN={batch_tn}"
            )

        if collect_outputs:
            collected_probs.append(probs.detach().cpu())
            collected_labels.append(labels.detach().cpu())
            all_embeddings.append(embedding.detach().cpu())

    assert stats["samples"] > 0, "No samples processed during epoch."

    avg_loss = stats["loss"] / stats["samples"]
    accuracy = (stats["tp"] + stats["tn"]) / stats["samples"]
    precision = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0.0
    recall = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    # 添加详细的统计信息
    total_pos_labels = stats["tp"] + stats["fn"]  # 实际正类数量
    total_neg_labels = stats["tn"] + stats["fp"]  # 实际负类数量
    total_pos_preds = stats["tp"] + stats["fp"]    # 预测为正类的数量
    total_neg_preds = stats["tn"] + stats["fn"]   # 预测为负类的数量
    label_pos_ratio = total_pos_labels / stats["samples"] if stats["samples"] > 0 else 0.0
    pred_pos_ratio = total_pos_preds / stats["samples"] if stats["samples"] > 0 else 0.0
    
    # 计算概率统计（如果收集了输出）
    prob_stats = ""
    if collect_outputs and collected_probs:
        all_probs = torch.cat(collected_probs).numpy()
        prob_mean = float(all_probs.mean())
        prob_std = float(all_probs.std())
        prob_median = float(np.median(all_probs))
        prob_q25 = float(np.percentile(all_probs, 25))
        prob_q75 = float(np.percentile(all_probs, 75))
        prob_stats = f", probs: mean={prob_mean:.3f}, std={prob_std:.3f}, median={prob_median:.3f}, q25={prob_q25:.3f}, q75={prob_q75:.3f}"
    
    split_name = "Val" if not train_mode else "Train"
    LOGGER.info(
        f"{split_name} epoch summary - samples: {stats['samples']}, "
        f"TP: {stats['tp']}, TN: {stats['tn']}, FP: {stats['fp']}, FN: {stats['fn']}, "
        f"label_pos_ratio: {label_pos_ratio:.4f}, pred_pos_ratio: {pred_pos_ratio:.4f}, "
        f"recall: {recall:.4f}, precision: {precision:.4f}{prob_stats}"
    )

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    extras = None
    if collect_outputs:
        all_probs = torch.cat(collected_probs).squeeze().numpy() if collected_probs else np.array([], dtype=float)
        all_labels = torch.cat(collected_labels).squeeze().numpy() if collected_labels else np.array([], dtype=float)
        extras = {
            "probs": all_probs.astype(float, copy=False),
            "labels": all_labels.astype(int, copy=False),
            "embeddings": torch.cat(all_embeddings).numpy() if all_embeddings else None,
        }
        
    return metrics, extras


def _compute_binary_roc_curve(labels: np.ndarray, scores: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    if labels.size == 0 or scores.size == 0:
        return None
    positives = np.sum(labels == 1)
    negatives = np.sum(labels == 0)
    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    true_positive_cumsum = np.cumsum(sorted_labels == 1, dtype=float)
    false_positive_cumsum = np.cumsum(sorted_labels == 0, dtype=float)

    tpr = np.concatenate(([0.0], true_positive_cumsum / positives, [1.0]))
    fpr = np.concatenate(([0.0], false_positive_cumsum / negatives, [1.0]))
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def _save_epoch_diagnostics(
    labels: np.ndarray,
    scores: np.ndarray,
    split_name: str,
    epoch: int,
    output_dir: Path,
) -> None:
    if labels.size == 0 or scores.size == 0:
        return

    split_dir = output_dir / "diagnostics" / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    preds = (scores >= CONF_THRESHOLD).astype(int)
    tn = int(np.sum((labels == 0) & (preds == 0)))
    fp = int(np.sum((labels == 0) & (preds == 1)))
    fn = int(np.sum((labels == 1) & (preds == 0)))
    tp = int(np.sum((labels == 1) & (preds == 1)))
    confusion = np.array([[tn, fp], [fn, tp]], dtype=int)

    cm_fig, cm_ax = plt.subplots(figsize=(4, 4))
    cm_im = cm_ax.imshow(confusion, interpolation="nearest", cmap="Blues")
    cm_ax.figure.colorbar(cm_im, ax=cm_ax, fraction=0.046, pad=0.04)
    cm_ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=[LABEL_MAP[0], LABEL_MAP[1]],
        yticklabels=[LABEL_MAP[0], LABEL_MAP[1]],
        xlabel="Predicted label",
        ylabel="True label",
        title=f"{split_name.capitalize()} Confusion Matrix (epoch {epoch})",
    )
    thresh = confusion.max() / 2 if confusion.max() > 0 else 0.5
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            cm_ax.text(
                j,
                i,
                f"{confusion[i, j]}",
                ha="center",
                va="center",
                color="white" if confusion[i, j] > thresh else "black",
            )
    cm_fig.tight_layout()
    cm_path = split_dir / f"confusion_{split_name}_epoch{epoch:04d}.png"
    cm_fig.savefig(cm_path, dpi=150)
    plt.close(cm_fig)

    roc_payload = _compute_binary_roc_curve(labels, scores)
    roc_fig, roc_ax = plt.subplots(figsize=(5, 4))
    roc_ax.set_xlim(0, 1)
    roc_ax.set_ylim(0, 1)
    roc_ax.set_xlabel("False Positive Rate")
    roc_ax.set_ylabel("True Positive Rate")
    if roc_payload is None:
        roc_ax.set_title(f"{split_name.capitalize()} ROC (epoch {epoch})")
        roc_ax.text(0.5, 0.5, "ROC unavailable (single-class data)", ha="center", va="center")
    else:
        fpr, tpr, auc = roc_payload
        roc_ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        roc_ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        roc_ax.set_title(f"{split_name.capitalize()} ROC (epoch {epoch})")
        roc_ax.legend(loc="lower right")
    roc_fig.tight_layout()
    roc_path = split_dir / f"roc_{split_name}_epoch{epoch:04d}.png"
    roc_fig.savefig(roc_path, dpi=150)
    plt.close(roc_fig)


def _evaluate_predictions(model: nn.Module, dataloader: DataLoader, device: torch.device) -> List[Dict[str, Any]]:
    model.eval()
    results: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images)
            # 处理不同形状的 logits
            if logits.ndim == 2 and logits.size(1) == 2:
                # [B, 2] -> 使用 softmax 取第二列（睁眼概率）
                probs = torch.softmax(logits, dim=1)[:, 1]
                logit_values = logits[:, 1]  # 使用睁眼类别的 logit
            elif logits.ndim == 2 and logits.size(1) == 1:
                # [B, 1] -> 使用 sigmoid
                probs = torch.sigmoid(logits.squeeze(1))
                logit_values = logits.squeeze(1)
            else:
                # [B] -> 使用 sigmoid
                probs = torch.sigmoid(logits)
                logit_values = logits
            for idx in range(images.size(0)):
                results.append(
                    {
                        "path": batch["path"][idx],
                        "video_name": batch["video_name"][idx],
                        "base_frame": batch["base_frame"][idx],
                        "label": int(batch["label"][idx].item()),
                        "logit": float(logit_values[idx].detach().cpu().item()),
                        "prob_open": float(probs[idx].detach().cpu().item()),
                    }
                )
    return results

###############################################
# 统一的可分性指标写入函数
###############################################
def log_all_separation_metrics(split, outputs, epoch, tb_writer, config):
    if outputs is None or outputs["embeddings"] is None:
        return

    probs = outputs["probs"]
    labels = outputs["labels"]
    emb = outputs["embeddings"]

    # 限制样本数量，避免计算时间过长
    MAX_SAMPLES_FOR_VIS = 10000  # 可视化最多使用10000个样本
    MAX_SAMPLES_FOR_METRICS = 50000  # 指标计算最多使用50000个样本
    
    import numpy as np
    n_samples = len(probs)
    
    # 对数据进行采样（如果需要）
    if n_samples > MAX_SAMPLES_FOR_VIS:
        # 随机采样，保持类别比例
        indices = np.arange(n_samples)
        np.random.seed(42)  # 固定随机种子，保证可复现
        np.random.shuffle(indices)
        vis_indices = indices[:MAX_SAMPLES_FOR_VIS]
        vis_probs = probs[vis_indices]
        vis_labels = labels[vis_indices]
        vis_emb = emb[vis_indices]
        LOGGER.info(f"[{split}] Sampling {MAX_SAMPLES_FOR_VIS} samples from {n_samples} for visualization")
    else:
        vis_probs = probs
        vis_labels = labels
        vis_emb = emb
        vis_indices = np.arange(n_samples)
    
    if n_samples > MAX_SAMPLES_FOR_METRICS:
        # 对指标计算也进行采样
        indices = np.arange(n_samples)
        np.random.seed(42)
        np.random.shuffle(indices)
        metrics_indices = indices[:MAX_SAMPLES_FOR_METRICS]
        metrics_probs = probs[metrics_indices]
        metrics_labels = labels[metrics_indices]
        metrics_emb = emb[metrics_indices]
        LOGGER.info(f"[{split}] Sampling {MAX_SAMPLES_FOR_METRICS} samples from {n_samples} for metrics calculation")
    else:
        metrics_probs = probs
        metrics_labels = labels
        metrics_emb = emb

    # 检查sklearn是否可用
    try:
        import sklearn
        sklearn_available = True
    except ImportError:
        sklearn_available = False
        LOGGER.warning("sklearn not available, skipping t-SNE and PCA visualizations")

    import os
    
    # 1) t-SNE (需要sklearn，使用采样后的数据)
    if sklearn_available:
        try:
            LOGGER.info(f"[{split}] Computing t-SNE for {len(vis_emb)} samples...")
            tsne_path = os.path.join(config.output_dir, f"tsne/epoch_{epoch}.png")
            os.makedirs(os.path.dirname(tsne_path), exist_ok=True)
            plot_tsne_2d(vis_emb, vis_labels, tsne_path)
            LOGGER.info(f"[{split}] t-SNE completed")
        except Exception as e:
            LOGGER.warning(f"Failed to plot t-SNE: {e}")
    else:
        LOGGER.debug("Skipping t-SNE visualization (sklearn not available)")

    # 2) PCA 3D (需要sklearn，使用采样后的数据)
    if sklearn_available:
        try:
            LOGGER.info(f"[{split}] Computing PCA 3D for {len(vis_emb)} samples...")
            pca3d_path = os.path.join(config.output_dir, f"pca3d/epoch_{epoch}.png")
            os.makedirs(os.path.dirname(pca3d_path), exist_ok=True)
            plot_pca_3d(vis_emb, vis_labels, pca3d_path)
            LOGGER.info(f"[{split}] PCA 3D completed")
        except Exception as e:
            LOGGER.warning(f"Failed to plot PCA 3D: {e}")
    else:
        LOGGER.debug("Skipping PCA 3D visualization (sklearn not available)")

    # 3) 离群点（使用采样后的数据）
    LOGGER.info(f"[{split}] Computing outliers for {len(metrics_emb)} samples...")
    outliers = detect_outliers_mahalanobis(metrics_emb, metrics_labels)
    csv_path = os.path.join(config.output_dir, "outliers.csv")
    with open(csv_path, "w") as f:
        f.write("index,label,distance\n")
        for i,l,d in outliers:
            f.write(f"{i},{l},{d}\n")
    LOGGER.info(f"[{split}] Outliers computation completed")

    # 4) 难例（使用采样后的数据）
    LOGGER.info(f"[{split}] Computing hard samples...")
    hard = detect_hard_samples(metrics_probs, metrics_labels)
    csv_path = os.path.join(config.output_dir, "hard_samples.csv")
    with open(csv_path, "w") as f:
        f.write("index,prob,label\n")
        for i,p,l in hard:
            f.write(f"{i},{p},{l}\n")
    LOGGER.info(f"[{split}] Hard samples computation completed")

    # 5) 错标检测（使用采样后的数据）
    LOGGER.info(f"[{split}] Computing mislabeled samples...")
    wrong = detect_mislabeled(metrics_probs, metrics_labels, metrics_emb)
    csv_path = os.path.join(config.output_dir, "mislabeled.csv")
    with open(csv_path, "w") as f:
        f.write("index,true,pred,prob\n")
        for i,t,p,pr in wrong:
            f.write(f"{i},{t},{p},{pr}\n")
    LOGGER.info(f"[{split}] Mislabeled samples computation completed")

    #-------------------------------------
    # 基本统计 + 概率直方图（使用采样后的数据）
    #-------------------------------------
    LOGGER.info(f"[{split}] Computing histograms...")
    hist0, _ = np.histogram(metrics_probs[metrics_labels == 0], bins=200, range=(0,1), density=True)
    hist1, _ = np.histogram(metrics_probs[metrics_labels == 1], bins=200, range=(0,1), density=True)

    #-------------------------------------
    # 基础分布指标（KS / Hellinger / JS / ECE / Margin）
    #-------------------------------------
    LOGGER.info(f"[{split}] Computing distribution metrics...")
    metrics = {}
    metrics["ks_distance"] = ks_distance(metrics_probs, metrics_labels)
    metrics["hellinger"] = hellinger_distance(hist0, hist1)
    metrics["js_divergence"] = js_divergence(hist0, hist1)
    metrics["ece"] = expected_calibration_error(metrics_probs, metrics_labels)
    metrics["margin"] = margin_score(metrics_probs)

    #-------------------------------------
    # 贝叶斯误差 + 类间可分性（intra/inter/fisher/bhatt）
    #-------------------------------------
    LOGGER.info(f"[{split}] Computing class separation metrics...")
    metrics["bayes_error"] = compute_bayes_error(metrics_probs, metrics_labels)
    sep = compute_class_separation(metrics_emb, metrics_labels)
    metrics["intra"] = sep["intra"]
    metrics["inter"] = sep["inter"]
    metrics["fisher"] = sep["fisher"]
    metrics["bhattacharyya"] = sep["bhatta"]

    #-------------------------------------
    # PCA embedding 能量
    #-------------------------------------
    LOGGER.info(f"[{split}] Computing PCA energy...")
    pca_energy = embedding_pca_energy(metrics_emb, k=5)
    LOGGER.info(f"[{split}] All metrics computation completed")

    #-------------------------------------
    # 写入 TensorBoard
    #-------------------------------------
    for key, val in metrics.items():
        tb_writer.add_scalar(f"{split}/{key}", val, epoch)

    for i, e in enumerate(pca_energy):
        tb_writer.add_scalar(f"{split}/pca_energy_{i}", e, epoch)
    
    #-------------------------------------
    # 保存类间分离效果可视化图
    #-------------------------------------
    try:
        LOGGER.info(f"[{split}] Saving class separation visualization...")
        sep_vis_path = os.path.join(config.output_dir, f"class_separation/{split}_epoch_{epoch}.png")
        os.makedirs(os.path.dirname(sep_vis_path), exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"{split.capitalize()} Class Separation Metrics (Epoch {epoch})", fontsize=14, fontweight='bold')
        
        # 1. Intra-class vs Inter-class Distance
        ax1 = axes[0, 0]
        ax1.bar(['Intra-class\nDistance', 'Inter-class\nDistance'], 
                [sep["intra"], sep["inter"]], 
                color=['#ff7f7f', '#7f7fff'], alpha=0.7)
        ax1.set_ylabel('Distance', fontsize=11)
        ax1.set_title('Intra vs Inter-class Distance', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        # 添加数值标签
        ax1.text(0, sep["intra"], f'{sep["intra"]:.3f}', ha='center', va='bottom', fontsize=10)
        ax1.text(1, sep["inter"], f'{sep["inter"]:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Fisher Ratio (类间分离度)
        ax2 = axes[0, 1]
        fisher_val = sep["fisher"]
        ax2.barh(['Fisher Ratio'], [fisher_val], color='#7fbf7f', alpha=0.7)
        ax2.set_xlabel('Fisher Ratio (Inter / Intra)', fontsize=11)
        ax2.set_title(f'Class Separability\n(Higher = Better Separation)', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        # 添加数值标签
        ax2.text(fisher_val, 0, f'{fisher_val:.3f}', ha='left' if fisher_val < ax2.get_xlim()[1] * 0.5 else 'right', 
                va='center', fontsize=12, fontweight='bold')
        
        # 3. Bhattacharyya Distance
        ax3 = axes[1, 0]
        bhatt_val = sep["bhatta"]
        ax3.barh(['Bhattacharyya\nDistance'], [bhatt_val], color='#bf7fbf', alpha=0.7)
        ax3.set_xlabel('Distance', fontsize=11)
        ax3.set_title('Distribution Distance\n(Higher = More Separated)', fontsize=12, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        # 添加数值标签
        ax3.text(bhatt_val, 0, f'{bhatt_val:.3f}', ha='left' if bhatt_val < ax3.get_xlim()[1] * 0.5 else 'right', 
                va='center', fontsize=12, fontweight='bold')
        
        # 4. 综合指标对比（归一化显示）
        ax4 = axes[1, 1]
        # 归一化到 [0, 1] 范围用于可视化（使用相对值）
        max_inter = max(sep["inter"], 1.0)  # 避免除零
        max_intra = max(sep["intra"], 1.0)
        max_fisher = max(fisher_val, 1.0)
        max_bhatt = max(bhatt_val, 1.0)
        
        normalized_values = {
            'Inter/Intra\nRatio': sep["inter"] / max_inter if max_intra > 0 else 0,
            'Fisher Ratio': fisher_val / max_fisher,
            'Bhattacharyya': bhatt_val / max_bhatt,
        }
        
        bars = ax4.barh(list(normalized_values.keys()), list(normalized_values.values()), 
                       color=['#7f7fff', '#7fbf7f', '#bf7fbf'], alpha=0.7)
        ax4.set_xlabel('Normalized Value', fontsize=11)
        ax4.set_title('Normalized Metrics Comparison', fontsize=12, fontweight='bold')
        ax4.set_xlim(0, 1.1)
        ax4.grid(axis='x', alpha=0.3)
        # 添加原始值标签
        for i, (key, val) in enumerate(normalized_values.items()):
            if key == 'Inter/Intra\nRatio':
                orig_val = sep["inter"] / max_intra if max_intra > 0 else 0
                ax4.text(val, i, f'{orig_val:.3f}', ha='left', va='center', fontsize=9)
            elif key == 'Fisher Ratio':
                ax4.text(val, i, f'{fisher_val:.3f}', ha='left', va='center', fontsize=9)
            else:
                ax4.text(val, i, f'{bhatt_val:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(sep_vis_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        LOGGER.info(f"[{split}] Class separation visualization saved to {sep_vis_path}")
    except Exception as e:
        LOGGER.warning(f"Failed to save class separation visualization: {e}")

def save_embeddings_for_projector(writer, emb, labels, epoch, projector_dir, split="val", tag="embedding", max_samples=50000):
    """
    Save embedding + metadata for TensorBoard Projector.
    emb: numpy (N, D)
    labels: numpy (N,)
    max_samples: 限制保存的样本数量，避免内存问题
    """
    import os
    
    # 限制样本数量，避免内存问题
    total_samples = min(emb.shape[0], len(labels))
    if total_samples > max_samples:
        # 随机采样
        indices = np.random.choice(total_samples, max_samples, replace=False)
        emb = emb[indices]
        labels = labels[indices]
        N = max_samples
        print(f"[Projector] Sampling {N} samples from {total_samples} total samples for {split} embeddings (epoch {epoch})")
    else:
        N = total_samples
        if emb.shape[0] != len(labels):
            print(f"[Warning] embedding count {emb.shape[0]} != labels {len(labels)}, trimming to {N}")
            emb = emb[:N]
            labels = labels[:N]

    # 检查并过滤无效值（NaN, Inf）
    valid_mask = np.isfinite(emb).all(axis=1)
    if not valid_mask.all():
        invalid_count = (~valid_mask).sum()
        print(f"[Projector] Warning: {invalid_count} invalid embeddings (NaN/Inf) detected, filtering them out")
        emb = emb[valid_mask]
        labels = labels[valid_mask]
        N = len(emb)
        print(f"[Projector] After filtering: {N} valid embeddings")
    
    if N == 0:
        print(f"[Projector] Error: No valid embeddings to save for {split} (epoch {epoch})")
        return

    emb_tensor = torch.tensor(emb, dtype=torch.float32)
    
    # 确保tensor和labels数量一致
    assert emb_tensor.shape[0] == len(labels), f"Tensor shape {emb_tensor.shape[0]} != labels length {len(labels)}"
    
    # metadata file
    metadata_path = os.path.join(projector_dir, f"{split}_metadata_epoch_{epoch}.tsv")
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w") as f:
        f.write("label\n")
        for l in labels:
            f.write(f"{int(l)}\n")

    # embedding variable name
    var_name = f"{split}_emb_epoch_{epoch}"
    
    # write embedding
    print(f"[Projector] Writing {N} {split} embeddings for epoch {epoch}...")
    print(f"[Projector] Tensor shape: {emb_tensor.shape}, Labels length: {len(labels)}")
    
    # 确保metadata是列表且长度匹配
    metadata_list = labels.tolist() if hasattr(labels, 'tolist') else [int(l) for l in labels]
    
    # 最终验证
    if len(metadata_list) != emb_tensor.shape[0]:
        print(f"[Projector] ERROR: Metadata length {len(metadata_list)} != tensor shape {emb_tensor.shape[0]}")
        print(f"[Projector] Trimming to match tensor shape...")
        min_len = min(len(metadata_list), emb_tensor.shape[0])
        metadata_list = metadata_list[:min_len]
        emb_tensor = emb_tensor[:min_len]
        labels = labels[:min_len]
        N = min_len
        print(f"[Projector] After trimming: {N} embeddings")
    
    # 重新写入metadata文件，确保数量一致
    with open(metadata_path, "w") as f:
        f.write("label\n")
        for l in labels:
            f.write(f"{int(l)}\n")
    
    # 最终验证
    assert emb_tensor.shape[0] == len(metadata_list) == len(labels), \
        f"Final mismatch: tensor={emb_tensor.shape[0]}, metadata={len(metadata_list)}, labels={len(labels)}"
    
    writer.add_embedding(
        emb_tensor,
        metadata=metadata_list,
        tag=f"{split}_epoch_{epoch}",
        global_step=epoch
    )

    print(f"[Projector] Saved: {N} {split} embeddings for epoch {epoch} (tensor shape: {emb_tensor.shape})")
    
def train_pipeline(config: TrainConfig, verbose: bool = False) -> Dict[str, Any]:
    # 创建带版本号的输出目录
    base_output_dir = config.output_dir
    version = 1
    while (base_output_dir / f"v{version}").exists():
        version += 1
    config.output_dir = base_output_dir / f"v{version}"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Output directory: {config.output_dir} (version {version})")
    
    _setup_logging(config.output_dir, verbose=verbose)
    config_dict = config.to_dict()
    train_config_serialized = copy.deepcopy(config_dict)
    LOGGER.info("Starting training with config: %s", json.dumps(config_dict, indent=2))

    device = _resolve_device(config.device)
    LOGGER.info("Using device: %s", device)
    _set_seed(config.seed)
    amp_enabled = bool(config.use_amp and device.type == "cuda")
    if config.use_amp and not amp_enabled:
        LOGGER.warning("--use_amp requested but CUDA device is not available; proceeding without AMP.")
    tb_dir = config.output_dir
    tb_writer = SummaryWriter(log_dir=str(tb_dir))
    
    # 自动启动TensorBoard服务器
    tb_process = None  # 在函数作用域内定义，确保后续可以访问
    try:
        import shutil
        if shutil.which("tensorboard"):
            # 查找可用端口
            import socket
            tb_port = 6006
            for port in range(6006, 6010):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('127.0.0.1', port))
                sock.close()
                if result != 0:  # 端口未被占用
                    tb_port = port
                    break
            
            # 启动TensorBoard
            tb_cmd = [
                "tensorboard",
                "--logdir", str(tb_dir),
                "--port", str(tb_port),
                "--host", "0.0.0.0",  # 允许外部访问
            ]
            tb_process = subprocess.Popen(
                tb_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True  # 创建新的进程组，避免训练结束时被终止
            )
            LOGGER.info(f"TensorBoard started automatically at http://localhost:{tb_port}")
            LOGGER.info(f"TensorBoard log directory: {tb_dir}")
            
            # 注册清理函数
            def cleanup_tensorboard():
                if tb_process and tb_process.poll() is None:
                    tb_process.terminate()
                    try:
                        tb_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        tb_process.kill()
            
            atexit.register(cleanup_tensorboard)
        else:
            LOGGER.warning("TensorBoard not found in PATH. Install it with: pip install tensorboard")
            LOGGER.info(f"To start TensorBoard manually, run: tensorboard --logdir {tb_dir}")
    except Exception as e:
        LOGGER.warning(f"Failed to start TensorBoard automatically: {e}")
        LOGGER.info(f"To start TensorBoard manually, run: tensorboard --logdir {tb_dir}")
    
    history_path = config.output_dir / "history.json"
    history: List[Dict[str, Any]] = []
    if history_path.exists():
        try:
            with open(history_path, "r", encoding="utf-8") as fp:
                loaded_history = json.load(fp)
                if isinstance(loaded_history, list):
                    history = loaded_history
        except Exception as exc:
            LOGGER.warning("Failed to load existing history from %s: %s", history_path, exc)
            history = []

    samples = collect_samples(config.data_root, logger=LOGGER)
    splits = split_samples(
        samples,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
        logger=LOGGER,
    )

    mean, std = DEFAULT_MEAN, DEFAULT_STD
    train_transform, eval_transform = _build_transforms(config.image_size, mean, std)
    normalization = {"mean": list(mean), "std": list(std), "image_size": list(config.image_size)}

    train_dataset = OCECDataset(splits["train"], transform=train_transform)
    val_dataset = OCECDataset(splits["val"], transform=eval_transform) if splits["val"] else None
    test_dataset = OCECDataset(splits["test"], transform=eval_transform) if splits["test"] else None

    # train_sampler = build_weighted_sampler(splits["train"])
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        sampler=None,
        num_workers=config.num_workers,
    )
    val_loader = (
        create_dataloader(
            val_dataset, batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )
    test_loader = (
        create_dataloader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers)
        if test_dataset is not None and len(test_dataset) > 0
        else None
    )

    model_config = ModelConfig(
        base_channels=config.base_channels,
        num_blocks=config.num_blocks,
        dropout=config.dropout,
        arch_variant=config.arch_variant,
        head_variant=config.head_variant,
        token_mixer_grid=config.token_mixer_grid,
        token_mixer_layers=config.token_mixer_layers,
        margin_method=config.margin_method,
    )
    model = OCEC(model_config).to(device)
    
    # 加载预训练权重（如果指定）
    if config.pretrain_from:
        pretrain_path = config.pretrain_from
        if not pretrain_path.exists():
            raise FileNotFoundError(f"Pretrain checkpoint not found: {pretrain_path}")
        LOGGER.info(f"Loading pretrained weights from: {pretrain_path}")
        pretrain_payload = torch.load(pretrain_path, map_location=device)
        
        # 只加载模型权重，不恢复训练状态
        pretrain_model_state = pretrain_payload.get("model_state") or pretrain_payload.get("model")
        if pretrain_model_state is None:
            raise ValueError(f"No model state found in pretrain checkpoint: {pretrain_path}")
        
        # 处理 DataParallel 格式的权重（移除 'module.' 前缀）
        if any(k.startswith("module.") for k in pretrain_model_state.keys()):
            pretrain_model_state = {k.replace("module.", ""): v for k, v in pretrain_model_state.items() if k.startswith("module.")}
        
        # 尝试加载权重，允许部分匹配（忽略不匹配的层）
        model_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_model_state.items() if k in model_dict and model_dict[k].shape == v.shape}
        missing_keys = set(model_dict.keys()) - set(pretrain_dict.keys())
        unexpected_keys = set(pretrain_dict.keys()) - set(model_dict.keys())
        
        if missing_keys:
            LOGGER.warning(f"Missing keys in pretrain checkpoint: {missing_keys}")
        if unexpected_keys:
            LOGGER.warning(f"Unexpected keys in pretrain checkpoint: {unexpected_keys}")
        
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict, strict=False)
        LOGGER.info(f"Loaded {len(pretrain_dict)}/{len(model_dict)} layers from pretrain checkpoint")
    
    # 冻结backbone（如果指定）
    if config.freeze_backbone:
        _freeze_backbone(model)
        LOGGER.info("Backbone frozen: only head and margin_head will be trained")
    
    ema = EMA(model, decay=0.999)
    
    # Multi-GPU support using DataParallel
    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    use_multi_gpu = num_gpus > 1
    if use_multi_gpu:
        LOGGER.info(f"Using {num_gpus} GPUs for training with DataParallel")
        model = nn.DataParallel(model)
        # Note: Effective batch size will be batch_size * num_gpus
        LOGGER.info(f"Effective batch size: {config.batch_size * num_gpus} (batch_size={config.batch_size} × {num_gpus} GPUs)")
    else:
        LOGGER.info("Using single GPU or CPU for training")
    
    base_metadata = {
        "model_config": asdict(model_config),
        "train_config": train_config_serialized,
        "normalization": normalization,
        "label_map": LABEL_MAP,
        "amp_enabled": amp_enabled,
        "tensorboard_logdir": str(tb_dir),
        "num_gpus": num_gpus,
        "use_multi_gpu": use_multi_gpu,
    }

    pos_weight = _compute_pos_weight(splits["train"]).to(device)

    if config.margin_method in ["arcface", "cosface"]:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    
    # 创建主调度器（ReduceLROnPlateau）
    main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    
    # 如果启用warmup，创建组合调度器
    warmup_epochs = config.warmup_epochs
    if warmup_epochs > 0:
        # Warmup阶段：线性增加学习率从0到目标学习率
        def warmup_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 1.0
        
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
        scheduler = {"type": "warmup", "warmup": warmup_scheduler, "main": main_scheduler, "warmup_epochs": warmup_epochs}
        LOGGER.info(f"Using warmup scheduler: {warmup_epochs} epochs of linear warmup, then ReduceLROnPlateau")
    else:
        scheduler = {"type": "main", "main": main_scheduler, "warmup_epochs": 0}
        LOGGER.info("Using ReduceLROnPlateau scheduler (no warmup)")
    
    scaler = _create_grad_scaler(amp_enabled)

    start_epoch = 1
    best_state: Optional[Dict[str, Any]] = None
    best_val_loss = math.inf
    best_f1 = float("-inf")
    best_checkpoint_path: Optional[Path] = None

    if config.resume_from:
        resume_path = config.resume_from
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_payload = torch.load(resume_path, map_location=device)
        resume_epoch = int(resume_payload.get("epoch", 0))
        LOGGER.info("Resuming from checkpoint %s (epoch %d).", resume_path, resume_epoch)

        checkpoint_norm = resume_payload.get("normalization")
        if checkpoint_norm:
            current_size = _ensure_image_size_tuple(normalization["image_size"])
            try:
                checkpoint_size = _ensure_image_size_tuple(checkpoint_norm.get("image_size", current_size))
            except ValueError:
                checkpoint_size = current_size
            checkpoint_mean = list(checkpoint_norm.get("mean", normalization["mean"]))
            checkpoint_std = list(checkpoint_norm.get("std", normalization["std"]))
            norm_mismatch = (
                checkpoint_mean != normalization["mean"]
                or checkpoint_std != normalization["std"]
                or checkpoint_size != current_size
            )
            if norm_mismatch:
                LOGGER.warning(
                    "Checkpoint normalization %s differs from current settings %s.",
                    checkpoint_norm,
                    normalization,
                )

        # Handle DataParallel model state dict
        model_state = resume_payload["model_state"]
        # If resuming to multi-GPU but checkpoint was single-GPU, or vice versa
        if use_multi_gpu and not any(k.startswith("module.") for k in model_state.keys()):
            # Checkpoint was single-GPU, but we're using multi-GPU now
            LOGGER.info("Converting single-GPU checkpoint to multi-GPU format")
            model_state = {f"module.{k}": v for k, v in model_state.items()}
        elif not use_multi_gpu and any(k.startswith("module.") for k in model_state.keys()):
            # Checkpoint was multi-GPU, but we're using single-GPU now
            LOGGER.info("Converting multi-GPU checkpoint to single-GPU format")
            model_state = {k.replace("module.", ""): v for k, v in model_state.items() if k.startswith("module.")}
        
        model.load_state_dict(model_state)
        if resume_payload.get("optimizer_state"):
            optimizer.load_state_dict(resume_payload["optimizer_state"])
        if resume_payload.get("scheduler_state"):
            # 恢复调度器状态
            scheduler_state = resume_payload["scheduler_state"]
            try:
                # 检查scheduler_state的格式，判断是哪种调度器
                # LambdaLR有'lr_lambdas'键，ReduceLROnPlateau没有
                has_lr_lambdas = "lr_lambdas" in scheduler_state
                
                if scheduler["type"] == "warmup":
                    # 如果恢复时还在warmup阶段，且scheduler_state是LambdaLR格式
                    if start_epoch <= scheduler["warmup_epochs"] and has_lr_lambdas:
                        scheduler["warmup"].load_state_dict(scheduler_state)
                    else:
                        # 如果已经过了warmup阶段，或者scheduler_state是ReduceLROnPlateau格式
                        # 尝试加载到主调度器
                        try:
                            scheduler["main"].load_state_dict(scheduler_state)
                        except Exception as e:
                            LOGGER.warning(f"Failed to load scheduler state to main scheduler: {e}. Skipping scheduler state restoration.")
                else:
                    # 没有warmup，直接加载到主调度器
                    try:
                        scheduler["main"].load_state_dict(scheduler_state)
                    except Exception as e:
                        LOGGER.warning(f"Failed to load scheduler state: {e}. Skipping scheduler state restoration.")
            except Exception as e:
                LOGGER.warning(f"Failed to restore scheduler state: {e}. Continuing without scheduler state restoration.")
        if resume_payload.get("scaler_state") and amp_enabled:
            scaler.load_state_dict(resume_payload["scaler_state"])

        start_epoch = resume_epoch + 1
        resume_train_metrics = copy.deepcopy(resume_payload.get("train_metrics") or {})
        resume_val_metrics = copy.deepcopy(resume_payload.get("val_metrics") or None)
        best_val_loss = resume_payload.get("best_val_loss")
        if best_val_loss is None:
            best_val_loss = (
                resume_val_metrics["loss"]
                if isinstance(resume_val_metrics, dict) and "loss" in resume_val_metrics
                else math.inf
            )
        best_accuracy = resume_payload.get("best_accuracy")
        if best_accuracy is None:
            best_accuracy = _infer_accuracy(
                resume_train_metrics,
                resume_val_metrics if isinstance(resume_val_metrics, dict) else None,
            )
        best_f1_candidate = resume_payload.get("best_f1")
        if best_f1_candidate is not None:
            best_f1 = float(best_f1_candidate)
        else:
            candidate_sources = [
                resume_val_metrics if isinstance(resume_val_metrics, dict) else None,
                resume_train_metrics if isinstance(resume_train_metrics, dict) else None,
            ]
            extracted_f1 = None
            for source in candidate_sources:
                if source and source.get("f1") is not None and not math.isnan(source.get("f1")):
                    extracted_f1 = float(source["f1"])
                    break
            if extracted_f1 is not None:
                best_f1 = extracted_f1
            else:
                best_f1 = float("-inf")

        # Handle model state dict format for resume
        resume_model_state = copy.deepcopy(resume_payload["model_state"])
        if use_multi_gpu and not any(k.startswith("module.") for k in resume_model_state.keys()):
            # Checkpoint was single-GPU, but we're using multi-GPU now
            resume_model_state = {f"module.{k}": v for k, v in resume_model_state.items()}
        elif not use_multi_gpu and any(k.startswith("module.") for k in resume_model_state.keys()):
            # Checkpoint was multi-GPU, but we're using single-GPU now
            resume_model_state = {k.replace("module.", ""): v for k, v in resume_model_state.items() if k.startswith("module.")}
        
        best_state = {
            "epoch": resume_payload.get("best_epoch", resume_epoch),
            "model_state": resume_model_state,
            "optimizer_state": copy.deepcopy(resume_payload.get("optimizer_state")),
            "scheduler_state": copy.deepcopy(resume_payload.get("scheduler_state")),
            "scaler_state": copy.deepcopy(resume_payload.get("scaler_state")),
            "train_metrics": resume_train_metrics,
            "val_metrics": resume_val_metrics,
            "best_val_loss": best_val_loss,
            "best_accuracy": best_accuracy,
            "best_f1": best_f1,
            "checkpoint_path": str(resume_path),
        }
        best_checkpoint_path = resume_path

        if history:
            history = [entry for entry in history if entry.get("epoch", 0) < start_epoch]

        if start_epoch > config.epochs:
            LOGGER.info(
                "Checkpoint epoch %d exceeds requested total epochs %d; no additional training will be performed.",
                start_epoch - 1,
                config.epochs,
            )

    for epoch in range(start_epoch, config.epochs + 1):
        # 检查是否需要解冻backbone（渐进式微调）
        if config.unfreeze_backbone_epoch is not None and epoch == config.unfreeze_backbone_epoch:
            # 检查第一阶段效果（如果验证集F1达到目标，才解冻）
            if val_metrics is not None and val_metrics.get("f1", 0) >= 0.80:
                LOGGER.info(f"Epoch {epoch}: Stage 1 F1={val_metrics['f1']:.4f} >= 0.80, proceeding to Stage 2")
                LOGGER.info(f"Epoch {epoch}: Unfreezing last 2 layers of backbone for progressive fine-tuning")
                _unfreeze_backbone_last_layers(model, num_layers=2)
                # 重新创建优化器，包含新解冻的参数，使用更小的学习率
                if config.stage2_lr is not None:
                    stage2_lr = config.stage2_lr
                else:
                    stage2_lr = config.lr * 0.05  # 默认使用原始学习率的5%
                optimizer = torch.optim.AdamW(
                    (param for param in model.parameters() if param.requires_grad),
                    lr=stage2_lr,
                    weight_decay=config.weight_decay,
                )
                LOGGER.info(f"Recreated optimizer with Stage 2 LR: {stage2_lr:.6f}")
                # 重新创建学习率调度器
                main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=2
                )
                if warmup_epochs > 0:
                    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
                    scheduler = {"type": "warmup", "warmup": warmup_scheduler, "main": main_scheduler, "warmup_epochs": warmup_epochs}
                else:
                    scheduler = {"type": "main", "main": main_scheduler, "warmup_epochs": 0}
            else:
                if val_metrics is not None:
                    LOGGER.warning(f"Epoch {epoch}: Stage 1 F1={val_metrics['f1']:.4f} < 0.80, skipping Stage 2 unfreezing")
                else:
                    LOGGER.warning(f"Epoch {epoch}: No validation metrics available, skipping Stage 2 unfreezing")
        
        train_metrics, train_outputs = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler=scaler,
            autocast_enabled=amp_enabled,
            progress_desc=f"Train {epoch}/{config.epochs}",
            collect_outputs=True,
            margin_method=config.margin_method,
            ema=ema,  # 训练阶段更新 EMA
        )
        # 只在特定epoch保存训练集embedding（每50个epoch或第一个epoch）
        if (epoch == 1 or epoch % 50 == 0) and train_outputs is not None and train_outputs["embeddings"] is not None:
            LOGGER.info("Saving training embeddings for TensorBoard Projector...")
            save_embeddings_for_projector(
                tb_writer,
                emb=train_outputs["embeddings"],
                labels=train_outputs["labels"],
                epoch=epoch,
                projector_dir=config.output_dir,
                split="train"
            )
            LOGGER.info("Training embeddings saved.")
        if val_loader:
            LOGGER.info("Starting validation...")
            # 应用 EMA 权重进行验证
            ema.apply(model)
            # 确保模型处于 eval 模式（虽然 _run_epoch 会设置，但这里明确设置更安全）
            model.eval()
            val_metrics, val_outputs = _run_epoch(
                model,
                val_loader,
                criterion,
                device,
                optimizer=None,
                scaler=None,
                autocast_enabled=amp_enabled,
                progress_desc=f"Val   {epoch}/{config.epochs}",
                collect_outputs=True,
                margin_method=config.margin_method,
                ema=None,  # 验证阶段不再更新 EMA，只使用 apply() 后的权重
            )
            
            if epoch % 50 == 0:
                if val_outputs is not None and val_outputs["embeddings"] is not None:
                    # ========= 错误样本蒙太奇 =========
                    wrong = detect_mislabeled(val_outputs["probs"], val_outputs["labels"], val_outputs["embeddings"])

                    wrong_paths = [ val_dataset.samples[idx].path for (idx,_,_,_) in wrong ]

                    montage_path = os.path.join(config.output_dir, f"montage/mislabeled_epoch_{epoch}.png")
                    os.makedirs(os.path.dirname(montage_path), exist_ok=True)
                    create_montage(wrong_paths, montage_path, cols=12)

                    # ========= 小眼睛聚类（子簇提取） =========
                    cluster_dir = os.path.join(config.output_dir, f"clusters/epoch_{epoch}")
                    os.makedirs(cluster_dir, exist_ok=True)

                    val_image_paths = [
                        s.path for s in val_dataset.samples 
                        if s.label == 0    # 假设 0 = 闭眼，小眼睛专用
                    ]

                    # 聚类小眼睛（需要sklearn）
                    try:
                        cluster_small_eye(
                            emb=val_outputs["embeddings"],
                            labels=val_outputs["labels"],
                            image_paths=val_image_paths,
                            save_root=str(config.output_dir / f"clusters_epoch{epoch:04d}"),
                            n_clusters=5,
                        )
                    except ImportError as e:
                        LOGGER.warning(f"Skipping clustering (sklearn not available): {e}")
                    except Exception as e:
                        LOGGER.warning(f"Failed to cluster small eyes: {e}")
                    
                    # 双PCA可视化（需要sklearn）
                    try:
                        emb_raw = val_outputs["embeddings"]
                        emb_arc = emb_raw / np.linalg.norm(emb_raw, axis=1, keepdims=True)

                        dual_pca_path = os.path.join(config.output_dir, f"dual_pca/epoch_{epoch}.png")
                        os.makedirs(os.path.dirname(dual_pca_path), exist_ok=True)
                        plot_dual_pca(emb_raw, emb_arc, val_outputs["labels"], dual_pca_path)
                    except ImportError as e:
                        LOGGER.warning(f"Skipping dual PCA visualization (sklearn not available): {e}")
                    except Exception as e:
                        LOGGER.warning(f"Failed to plot dual PCA: {e}")
                    
                    save_embeddings_for_projector(
                        tb_writer,
                        emb=val_outputs["embeddings"],
                        labels=val_outputs["labels"],
                        epoch=epoch,
                        projector_dir=config.output_dir,
                        split="val"
                    )
    
            # 验证后恢复原始模型权重（继续训练时使用原始权重，不是 EMA 权重）
            ema.restore(model)
            # 恢复训练模式
            model.train()
        else:
            val_metrics, val_outputs = None, None
        if epoch % 10 == 0:
            log_all_separation_metrics("train", train_outputs, epoch, tb_writer, config)
            log_all_separation_metrics("val",   val_outputs,   epoch, tb_writer, config)     
               
        # 更新学习率调度器
        if scheduler["type"] == "warmup" and epoch <= scheduler["warmup_epochs"]:
            # Warmup阶段：使用LambdaLR线性增加学习率
            scheduler["warmup"].step()
            current_lr = optimizer.param_groups[0]["lr"]
            if epoch == scheduler["warmup_epochs"]:
                LOGGER.info(f"Warmup completed. Learning rate: {current_lr:.6f}")
        else:
            # Warmup结束后，使用ReduceLROnPlateau
            if val_metrics:
                scheduler["main"].step(val_metrics["loss"])
            else:
                scheduler["main"].step(train_metrics["loss"])

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics or {"loss": float("nan"), "accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")},
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        train_summary = (
            f"loss {train_metrics['loss']:.4f} "
            f"acc {train_metrics['accuracy']:.4f} "
            f"prec {train_metrics['precision']:.4f} "
            f"rec {train_metrics['recall']:.4f} "
            f"f1 {train_metrics['f1']:.4f}"
        )
        if val_metrics:
            val_summary = (
                f"loss {val_metrics['loss']:.4f} "
                f"acc {val_metrics['accuracy']:.4f} "
                f"prec {val_metrics['precision']:.4f} "
                f"rec {val_metrics['recall']:.4f} "
                f"f1 {val_metrics['f1']:.4f}"
            )
        else:
            val_summary = "loss n/a acc n/a prec n/a rec n/a f1 n/a"

        LOGGER.info("Epoch %d | train %s | val %s", epoch, train_summary, val_summary)

        tb_writer.add_scalar("loss/train", train_metrics["loss"], epoch)
        tb_writer.add_scalar("metrics/train_accuracy", train_metrics["accuracy"], epoch)
        tb_writer.add_scalar("metrics/train_f1", train_metrics["f1"], epoch)
        if val_metrics is not None:
            tb_writer.add_scalar("loss/val", val_metrics["loss"], epoch)
            tb_writer.add_scalar("metrics/val_accuracy", val_metrics["accuracy"], epoch)
            tb_writer.add_scalar("metrics/val_f1", val_metrics["f1"], epoch)
        tb_writer.flush()

        if train_outputs is not None:
            _save_epoch_diagnostics(train_outputs["labels"], train_outputs["probs"], "train", epoch, config.output_dir)
        if val_outputs is not None:
            _save_epoch_diagnostics(val_outputs["labels"], val_outputs["probs"], "val", epoch, config.output_dir)
        train_outputs = None
        val_outputs = None

        # Get model state dict (unwrap DataParallel if needed)
        model_state = model.module.state_dict() if use_multi_gpu else model.state_dict()
        optimizer_state = optimizer.state_dict()
        # 保存调度器状态
        if scheduler["type"] == "warmup" and epoch <= scheduler["warmup_epochs"]:
            scheduler_state = scheduler["warmup"].state_dict()
        else:
            scheduler_state = scheduler["main"].state_dict()

        epoch_payload = {
            **base_metadata,
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "scaler_state": scaler.state_dict() if amp_enabled else None,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        epoch_path = config.output_dir / f"ocec_epoch_{epoch:04d}.pt"
        
        torch.save(epoch_payload, epoch_path)
        _prune_checkpoints(config.output_dir, "ocec_epoch_", 10)

        current_val_loss = val_metrics["loss"] if val_metrics else train_metrics["loss"]
        score_value = val_metrics["f1"] if val_metrics else train_metrics["f1"]
        if score_value > best_f1:
            accuracy_value = _infer_accuracy(train_metrics, val_metrics)
            best_f1 = score_value
            best_val_loss = current_val_loss
            # model_state already unwrapped above
            best_state = {
                "epoch": epoch,
                "model_state": copy.deepcopy(model_state),
                "optimizer_state": copy.deepcopy(optimizer_state),
                "scheduler_state": copy.deepcopy(scheduler_state),
                "scaler_state": copy.deepcopy(scaler.state_dict()) if amp_enabled else None,
                "train_metrics": copy.deepcopy(train_metrics),
                "val_metrics": copy.deepcopy(val_metrics) if val_metrics is not None else None,
                "best_val_loss": current_val_loss,
                "best_accuracy": accuracy_value,
                "best_f1": score_value,
            }
            best_checkpoint = dict(epoch_payload)
            best_checkpoint.update(
                best_val_loss=current_val_loss,
                best_epoch=epoch,
                best_accuracy=accuracy_value,
                best_f1=score_value,
            )
            best_checkpoint_path = config.output_dir / f"ocec_best_epoch{epoch:04d}_f1_{score_value:.4f}.pt"
            best_state["checkpoint_path"] = str(best_checkpoint_path)
            torch.save(best_checkpoint, best_checkpoint_path)
            _prune_checkpoints(config.output_dir, "ocec_best_", 10)
            LOGGER.info("New best model at epoch %d (F1 %.4f).", epoch, score_value)

    if best_state is None or best_checkpoint_path is None:
        raise RuntimeError("Training did not produce a valid model checkpoint.")

    # Load best model state (unwrap DataParallel if needed)
    best_model_state = best_state["model_state"]
    if use_multi_gpu and not any(k.startswith("module.") for k in best_model_state.keys()):
        best_model_state = {f"module.{k}": v for k, v in best_model_state.items()}
    elif not use_multi_gpu and any(k.startswith("module.") for k in best_model_state.keys()):
        best_model_state = {k.replace("module.", ""): v for k, v in best_model_state.items() if k.startswith("module.")}
    model.load_state_dict(best_model_state)

    test_metrics = None
    if test_loader:
        test_metrics, _ = _run_epoch(
            model,
            test_loader,
            criterion,
            device,
            optimizer=None,
            scaler=None,
            autocast_enabled=amp_enabled,
            progress_desc="Test",
            margin_method=config.margin_method,
        )
    LOGGER.info("Test metrics: %s", json.dumps(test_metrics, indent=2) if test_metrics else "n/a")
    if test_metrics:
        step = best_state["epoch"]
        tb_writer.add_scalar("loss/test", test_metrics["loss"], step)
        tb_writer.add_scalar("metrics/test_accuracy", test_metrics["accuracy"], step)
        tb_writer.add_scalar("metrics/test_f1", test_metrics["f1"], step)
        tb_writer.flush()

    with open(config.output_dir / "history.json", "w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)

    if test_loader:
        predictions = _evaluate_predictions(model, test_loader, device)
        pd.DataFrame(predictions).to_csv(config.output_dir / "test_predictions.csv", index=False)

    summary = {
        "checkpoint": str(best_checkpoint_path),
        "best_epoch": best_state["epoch"],
        "best_accuracy": best_state["best_accuracy"],
        "best_f1": best_state["best_f1"],
        "best_val_loss": best_state["best_val_loss"],
        "val_metrics": best_state["val_metrics"],
        "train_metrics": best_state["train_metrics"],
        "amp_enabled": amp_enabled,
        "resume_from": str(config.resume_from) if config.resume_from else None,
        "history_path": str(config.output_dir / "history.json"),
        "test_metrics": test_metrics,
        "retained_checkpoints": {
            "epoch": 10,
            "best": 10,
        },
        "tensorboard_logdir": str(tb_dir),
    }
    with open(config.output_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    tb_writer.close()
    
    # 关闭TensorBoard进程（如果还在运行）
    if tb_process and tb_process.poll() is None:
        LOGGER.info("Stopping TensorBoard server...")
        tb_process.terminate()
        try:
            tb_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tb_process.kill()
        LOGGER.info("TensorBoard server stopped.")
    
    return summary


def _gather_image_paths(inputs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(p for p in path.glob("**/*.png") if p.is_file()))
        else:
            LOGGER.warning("Skipping missing path: %s", item)
    if not paths:
        raise FileNotFoundError("No images found for inference.")
    return paths


def predict_images(
    checkpoint_path: Path,
    inputs: Sequence[str],
    device_spec: str = "auto",
) -> pd.DataFrame:
    device = _resolve_device(device_spec)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = ModelConfig(**checkpoint["model_config"])
    model = OCEC(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    normalization = checkpoint["normalization"]
    mean = normalization.get("mean", DEFAULT_MEAN)
    std = normalization.get("std", DEFAULT_STD)
    image_size_raw = normalization.get("image_size", (112, 112))
    try:
        image_size = _ensure_image_size_tuple(image_size_raw)
    except ValueError:
        LOGGER.warning("Invalid image_size %s in checkpoint; defaulting to 112x112.", image_size_raw)
        image_size = (112, 112)
    _, eval_transform = _build_transforms(image_size, mean, std)

    image_paths = _gather_image_paths(inputs)

    records: List[Dict[str, Any]] = []
    with torch.no_grad():
        for image_path in image_paths:
            pil_image = Image.open(image_path).convert("RGB")
            tensor = eval_transform(pil_image).unsqueeze(0).to(device)
            logits = model(tensor)
            # 处理不同形状的 logits
            if logits.ndim == 2 and logits.size(1) == 2:
                # [1, 2] -> 使用 softmax 取第二列（睁眼概率）
                prob = torch.softmax(logits, dim=1)[0, 1].item()
                logit_value = logits[0, 1].item()  # 使用睁眼类别的 logit
            elif logits.ndim == 2 and logits.size(1) == 1:
                # [1, 1] -> 使用 sigmoid
                prob = torch.sigmoid(logits.squeeze(1))[0].item()
                logit_value = logits.squeeze(1)[0].item()
            else:
                # [1] -> 使用 sigmoid
                prob = torch.sigmoid(logits)[0].item()
                logit_value = logits[0].item()
            records.append(
                {
                    "path": str(image_path),
                    "logit": float(logit_value),
                    "prob_open": float(prob),
                }
            )

    return pd.DataFrame(records)


def run_webcam_inference(
    checkpoint_path: Path,
    camera_index: int = 0,
    device_spec: str = "auto",
    window_name: str = "OCEC Webcam",
    mirror: bool = False,
    detector_model: Optional[Path] = None,
    detector_provider: str = "cpu",
) -> None:
    device = _resolve_device(device_spec)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = ModelConfig(**checkpoint["model_config"])
    model = OCEC(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    normalization = checkpoint["normalization"]
    mean = normalization.get("mean", DEFAULT_MEAN)
    std = normalization.get("std", DEFAULT_STD)
    image_size_raw = normalization.get("image_size", (112, 112))
    try:
        image_size = _ensure_image_size_tuple(image_size_raw)
    except ValueError:
        LOGGER.warning("Invalid image_size %s in checkpoint; defaulting to 112x112.", image_size_raw)
        image_size = (112, 112)
    _, eval_transform = _build_transforms(image_size, mean, std)

    detector_path = Path(detector_model) if detector_model is not None else _DEFAULT_DETECTOR_MODEL
    if not detector_path.exists():
        raise FileNotFoundError(
            f"Mouth detector model not found at {detector_path}. "
            "Provide --detector_model pointing to a valid DEIMv2 ONNX file."
        )
    detector_providers = _resolve_onnx_providers(detector_provider)
    mouth_detector = _create_mouth_detector(detector_path, detector_providers)
    LOGGER.info("Loaded DEIMv2 mouth detector from %s using providers %s.", detector_path, detector_providers)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {camera_index}.")

    LOGGER.info(
        "Starting webcam inference using checkpoint %s on device %s (camera index %d).",
        checkpoint_path,
        device,
        camera_index,
    )
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    margin_top = 2
    margin_bottom = 6
    margin_left = 2
    margin_right = 2

    try:
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    LOGGER.warning("Failed to read frame from camera; stopping.")
                    break

                if mirror:
                    frame = cv2.flip(frame, 1)

                try:
                    boxes = mouth_detector(
                        image=frame,
                        disable_generation_identification_mode=True,
                        disable_gender_identification_mode=True,
                        disable_left_and_right_hand_identification_mode=True,
                        disable_headpose_identification_mode=True,
                    )
                except Exception as exc:  # pragma: no cover - runtime safeguard
                    LOGGER.error("Mouth detector inference failed: %s", exc)
                    break

                mouth_boxes = [box for box in boxes if getattr(box, "classid", None) == 19]
                best_box = max(mouth_boxes, key=lambda box: box.score) if mouth_boxes else None
                crop, crop_coords = _extract_mouth_crop(
                    frame,
                    best_box,
                    margin_top=margin_top,
                    margin_bottom=margin_bottom,
                    margin_left=margin_left,
                    margin_right=margin_right,
                )

                if crop is None:
                    cv2.putText(
                        frame,
                        "Mouth not detected",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        LOGGER.info("Exit requested (key press).")
                        break
                    continue

                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(rgb_crop)
                tensor = eval_transform(pil_crop).unsqueeze(0).to(device)
                logits = model(tensor)
                prob_open = torch.sigmoid(logits)[0].item()

                label = LABEL_MAP[int(prob_open >= CONF_THRESHOLD)]
                color = (0, 200, 0) if label == "open" else (50, 50, 255)
                cv2.putText(
                    frame,
                    f"Prob open: {prob_open:.2%}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Prediction: {label}",
                    (12, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA,
                )

                if crop_coords is not None:
                    x1, y1, x2, y2 = crop_coords
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    LOGGER.info("Exit requested (key press).")
                    break
    finally:
        cap.release()
        cv2.destroyWindow(window_name)


def run_webcam_inference_onnx(
    onnx_model: Path,
    camera_index: int = 0,
    window_name: str = "OCEC Webcam (ONNX)",
    mirror: bool = False,
    model_provider: str = "cpu",
    detector_model: Optional[Path] = None,
    detector_provider: str = "cpu",
    image_size: Optional[tuple[int, int]] = None,
) -> None:
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("onnxruntime is required for ONNX inference. Install with `pip install onnxruntime`.") from exc

    ort.set_default_logger_severity(3)
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3

    providers = _resolve_onnx_providers(model_provider)
    session = ort.InferenceSession(
        str(onnx_model),
        sess_options=session_options,
        providers=providers,
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    input_shape = session.get_inputs()[0].shape
    if image_size is not None:
        height, width = _ensure_image_size_tuple(image_size)
        if len(input_shape) >= 4:
            expected_h, expected_w = input_shape[2], input_shape[3]
            if isinstance(expected_h, int) and isinstance(expected_w, int):
                if (expected_h, expected_w) != (height, width):
                    raise ValueError(
                        f"Specified image_size {(height, width)} does not match ONNX input {(expected_h, expected_w)}."
                    )
    else:
        if len(input_shape) >= 4 and isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
            height = int(input_shape[2])
            width = int(input_shape[3])
        else:
            LOGGER.warning(
                "ONNX model has dynamic spatial dimensions; defaulting to 112x112. "
                "Override with --image_size HEIGHTxWIDTH if needed."
            )
            height, width = (112, 112)
    _, eval_transform = _build_transforms((height, width), DEFAULT_MEAN, DEFAULT_STD)

    detector_path = Path(detector_model) if detector_model is not None else _DEFAULT_DETECTOR_MODEL
    if not detector_path.exists():
        raise FileNotFoundError(
            f"Mouth detector model not found at {detector_path}. "
            "Provide --detector_model pointing to a valid DEIMv2 ONNX file."
        )
    detector_providers = _resolve_onnx_providers(detector_provider)
    mouth_detector = _create_mouth_detector(detector_path, detector_providers)
    LOGGER.info(
        "Loaded ONNX model %s with providers %s (detector providers %s).",
        onnx_model,
        providers,
        detector_providers,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {camera_index}.")

    LOGGER.info(
        "Starting ONNX webcam inference using %s (camera index %d).",
        onnx_model,
        camera_index,
    )
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    margin_top = 5 #2
    margin_bottom = 5 #6
    margin_left = 5 #2
    margin_right = 5 #2

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                LOGGER.warning("Failed to read frame from camera; stopping.")
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            try:
                boxes = mouth_detector(
                    image=frame,
                    disable_generation_identification_mode=True,
                    disable_gender_identification_mode=True,
                    disable_left_and_right_hand_identification_mode=True,
                    disable_headpose_identification_mode=True,
                )
            except Exception as exc:  # pragma: no cover - runtime safeguard
                LOGGER.error("Mouth detector inference failed: %s", exc)
                break

            mouth_boxes = [box for box in boxes if getattr(box, "classid", None) == 19]
            best_box = max(mouth_boxes, key=lambda box: box.score) if mouth_boxes else None
            crop, crop_coords = _extract_mouth_crop(
                frame,
                best_box,
                margin_top=margin_top,
                margin_bottom=margin_bottom,
                margin_left=margin_left,
                margin_right=margin_right,
            )

            if crop is None:
                cv2.putText(
                    frame,
                    "Mouth not detected",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    LOGGER.info("Exit requested (key press).")
                    break
                continue

            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(rgb_crop)
            tensor = eval_transform(pil_crop).unsqueeze(0)
            inputs = {input_name: tensor.cpu().numpy()}
            outputs = session.run([output_name], inputs)
            prob_open = float(outputs[0].flatten()[0])

            label = LABEL_MAP[int(prob_open >= CONF_THRESHOLD)]
            color = (0, 200, 0) if label == "open" else (50, 50, 255)
            cv2.putText(
                frame,
                f"Prob open: {prob_open:.2%}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Prediction: {label}",
                (12, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

            if crop_coords is not None:
                x1, y1, x2, y2 = crop_coords
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                LOGGER.info("Exit requested (key press).")
                break
    finally:
        cap.release()
        cv2.destroyWindow(window_name)


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    opset: int = 17,
    device_spec: str = "auto",
) -> None:
    device = _resolve_device(device_spec)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = ModelConfig(**checkpoint["model_config"])
    model = OCEC(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    normalization = checkpoint["normalization"]
    image_size_raw = normalization.get("image_size", (112, 112))
    try:
        image_size = _ensure_image_size_tuple(image_size_raw)
    except ValueError:
        LOGGER.warning("Invalid image_size %s in checkpoint; defaulting to 112x112.", image_size_raw)
        image_size = (112, 112)

    dummy = torch.randn(1, 3, image_size[0], image_size[1], device=device)

    class _ONNXProbWrapper(nn.Module):
        def __init__(self, base_model: nn.Module) -> None:
            super().__init__()
            self.base_model = base_model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # 导出时禁用 margin-based loss，使用原始 head 输出
            # 临时保存原始 margin_method 状态
            original_margin_method = self.base_model.config.margin_method
            self.base_model.config.margin_method = "none"
            
            # 调用模型，不使用 margin-based loss
            logits = self.base_model(x, labels=None, return_embedding=False)
            
            # 恢复原始状态
            self.base_model.config.margin_method = original_margin_method
            
            # 处理不同形状的 logits
            if logits.ndim == 2 and logits.shape[1] == 2:
                # 如果是 [B, 2] 形状（不应该发生，但为了安全），取正类 logit
                logits = logits[:, 1]
            elif logits.ndim == 2 and logits.shape[1] == 1:
                # 如果是 [B, 1]，squeeze 成 [B]
                logits = logits.squeeze(1)
            
            # 对单个 logit 应用 sigmoid 得到概率
            return torch.sigmoid(logits)

    export_base = copy.deepcopy(model)
    export_base.to(device)
    export_base.eval()
    export_model = _ONNXProbWrapper(export_base)

    torch.onnx.export(
        export_model,
        dummy,
        output_path,
        input_names=["images"],
        output_names=["prob_open"],
        dynamic_axes=None, #{"images": {0: "batch"}, "prob_open": {0: "batch"}},
        do_constant_folding=False,
        opset_version=opset,
        keep_initializers_as_inputs=False,
    )
    LOGGER.info("Exported ONNX model to %s", output_path)

    try:
        import onnx
        from onnxsim import simplify

        onnx_model = onnx.load(output_path)
        simplified_model, check = simplify(onnx_model)
        if check:
            simplified_model = _remove_batchnorm_from_onnx(simplified_model)
            onnx.save(simplified_model, output_path)
            LOGGER.info("Simplified ONNX model with onnxsim at %s", output_path)
        else:
            LOGGER.warning("onnxsim simplification check failed; keeping original model.")
    except Exception as exc:
        LOGGER.warning("onnxsim simplification failed: %s", exc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OCEC training and inference pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the OCEC binary classifier.")
    train_parser.add_argument("--data_root", type=Path, required=True)
    train_parser.add_argument("--output_dir", type=Path, required=True)
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--batch_size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight_decay", type=float, default=1e-4)
    train_parser.add_argument("--num_workers", type=int, default=4)
    train_parser.add_argument(
        "--image_size",
        type=_parse_image_size_arg,
        default=_parse_image_size_arg("48"),
        help="Square size (e.g. 48) or HEIGHTxWIDTH (e.g. 64x48) for resizing input images.",
    )
    train_parser.add_argument("--train_ratio", type=float, default=0.8)
    train_parser.add_argument("--val_ratio", type=float, default=0.2)
    train_parser.add_argument("--test_ratio", type=float, default=0.0)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--base_channels", type=int, default=32)
    train_parser.add_argument("--num_blocks", type=int, default=4)
    train_parser.add_argument("--dropout", type=float, default=0.3)
    train_parser.add_argument(
        "--arch_variant",
        type=str,
        default="baseline",
        choices=["baseline", "inverted_se", "convnext"],
        help="Backbone architecture choice.",
    )
    train_parser.add_argument(
        "--head_variant",
        type=str,
        default="auto",
        choices=["auto", "avg", "avgmax_mlp", "transformer", "mlp_mixer"],
        help="Classification head configuration.",
    )
    train_parser.add_argument(
        "--token_mixer_grid",
        type=_parse_token_mixer_grid_arg,
        default=(2, 3),
        help="Grid for token mixer heads specified as HEIGHTxWIDTH (e.g. '2x3').",
    )
    train_parser.add_argument(
        "--token_mixer_layers",
        type=int,
        default=2,
        help="Number of transformer or MLP-mixer layers when using those head variants.",
    )
    train_parser.add_argument("--device", type=str, default="auto")
    train_parser.add_argument("--resume", type=Path, help="Resume training from a checkpoint file (restores full training state).")
    train_parser.add_argument("--pretrain", type=Path, help="Load pretrained weights from a checkpoint file (only loads model weights, does not restore training state).")
    train_parser.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone layers, only train head and margin_head (for fine-tuning).")
    train_parser.add_argument("--unfreeze_backbone_epoch", type=int, default=None, help="Epoch to unfreeze last 2 layers of backbone (for progressive fine-tuning).")
    train_parser.add_argument("--stage2_lr", type=float, default=None, help="Learning rate for stage 2 (after unfreezing backbone). If not specified, uses lr * 0.05.")
    train_parser.add_argument("--use_amp", action="store_true", help="Enable mixed precision training (CUDA only).")
    train_parser.add_argument(
        "--margin_method",
        type=str,
        default="cosface",
        choices=["none", "arcface", "cosface"],
        help="Margin-based loss method: 'none' (BCE), 'arcface' (ArcFace), 'cosface' (CosFace/AM-Softmax). Default: cosface",
    )
    train_parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Number of epochs for linear learning rate warmup (default: 5, set to 0 to disable).",
    )
    train_parser.add_argument("--verbose", action="store_true")

    predict_parser = subparsers.add_parser("predict", help="Run inference with a trained checkpoint.")
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--inputs", nargs="+", required=True, help="Image files or directories.")
    predict_parser.add_argument("--output", type=Path, help="Optional CSV path to save predictions.")
    predict_parser.add_argument("--device", type=str, default="auto")

    webcam_parser = subparsers.add_parser("webcam", help="Run real-time inference from a webcam.")
    webcam_parser.add_argument("--checkpoint", type=Path, required=True)
    webcam_parser.add_argument("--camera_index", type=int, default=0, help="OpenCV camera index (default: 0).")
    webcam_parser.add_argument("--device", type=str, default="auto")
    webcam_parser.add_argument("--window_name", type=str, default="OCEC Webcam")
    webcam_parser.add_argument("--mirror", action="store_true", help="Mirror frames horizontally before display.")
    webcam_parser.add_argument(
        "--detector_model",
        type=Path,
        help="Optional path to the DEIMv2 ONNX model for mouth detection. Defaults to the bundled dataset weight.",
    )
    webcam_parser.add_argument(
        "--detector_provider",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "tensorrt"],
        help="ONNX Runtime execution provider for the mouth detector (default: cpu).",
    )

    webcam_onnx_parser = subparsers.add_parser("webcam_onnx", help="Run real-time inference from a webcam using an ONNX model.")
    webcam_onnx_parser.add_argument("--model", type=Path, required=True, help="Path to the exported OCEC ONNX model.")
    webcam_onnx_parser.add_argument("--camera_index", type=int, default=0, help="OpenCV camera index (default: 0).")
    webcam_onnx_parser.add_argument("--window_name", type=str, default="OCEC Webcam (ONNX)")
    webcam_onnx_parser.add_argument("--mirror", action="store_true", help="Mirror frames horizontally before display.")
    webcam_onnx_parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "tensorrt"],
        help="ONNX Runtime execution provider for the classifier (default: cpu).",
    )
    webcam_onnx_parser.add_argument(
        "--detector_model",
        type=Path,
        help="Optional path to the DEIMv2 ONNX model for mouth detection. Defaults to the bundled dataset weight.",
    )
    webcam_onnx_parser.add_argument(
        "--detector_provider",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "tensorrt"],
        help="ONNX Runtime execution provider for the mouth detector (default: cpu).",
    )
    webcam_onnx_parser.add_argument(
        "--image_size",
        type=str,
        default=None,
        help="Override classifier input size as HEIGHTxWIDTH (defaults to ONNX model shape).",
    )

    onnx_parser = subparsers.add_parser("exportonnx", help="Export a trained checkpoint to ONNX.")
    onnx_parser.add_argument("--checkpoint", type=Path, required=True)
    onnx_parser.add_argument("--output", type=Path, required=True)
    onnx_parser.add_argument("--opset", type=int, default=17)
    onnx_parser.add_argument("--device", type=str, default="auto")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        config = TrainConfig(
            data_root=args.data_root,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            image_size=args.image_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            base_channels=args.base_channels,
            num_blocks=args.num_blocks,
            dropout=args.dropout,
            arch_variant=args.arch_variant,
            head_variant=args.head_variant,
            token_mixer_grid=args.token_mixer_grid,
            token_mixer_layers=args.token_mixer_layers,
            device=args.device,
            resume_from=args.resume,
            pretrain_from=args.pretrain,
            use_amp=args.use_amp,
            warmup_epochs=args.warmup_epochs,
            margin_method=args.margin_method,
            freeze_backbone=args.freeze_backbone,
            unfreeze_backbone_epoch=args.unfreeze_backbone_epoch,
            stage2_lr=args.stage2_lr,
        )
        train_pipeline(config, verbose=args.verbose)
    elif args.command == "predict":
        df = predict_images(args.checkpoint, args.inputs, device_spec=args.device)
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Saved predictions to {args.output}")
        else:
            print(df.to_string(index=False))
    elif args.command == "webcam":
        run_webcam_inference(
            args.checkpoint,
            camera_index=args.camera_index,
            device_spec=args.device,
            window_name=args.window_name,
            mirror=args.mirror,
            detector_model=args.detector_model,
            detector_provider=args.detector_provider,
        )
    elif args.command == "webcam_onnx":
        parsed_size = _parse_image_size_arg(args.image_size) if args.image_size else None
        run_webcam_inference_onnx(
            args.model,
            camera_index=args.camera_index,
            window_name=args.window_name,
            mirror=args.mirror,
            model_provider=args.provider,
            detector_model=args.detector_model,
            detector_provider=args.detector_provider,
            image_size=parsed_size,
        )
    elif args.command == "exportonnx":
        export_to_onnx(args.checkpoint, args.output, opset=args.opset, device_spec=args.device)
    else:
        parser.error(f"Unknown command: {args.command}")
