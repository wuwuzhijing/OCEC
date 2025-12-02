#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Optional, Any

import cv2
import numpy as np

try:
    import onnxruntime
except ImportError:
    print("ERROR: onnxruntime not installed. Run pip install onnxruntime-gpu or onnxruntime")
    sys.exit(1)

# setproctitle 可能导致 crash → 保护性加载
try:
    import setproctitle
    setproctitle.setproctitle("ocec_classify")
except Exception as e:
    print(f"[WARN] setproctitle failed: {e}")


# ---------------------------------------------------------
#  CLASSIFIER
# ---------------------------------------------------------
class OCECClassifier:
    _onnx_dtypes_to_numpy = {
        "tensor(float)": np.float32,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
    }

    def __init__(self, model_path: str, providers: list):
        print(f"[INFO] Loading ONNX model: {model_path}", flush=True)

        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3

        try:
            self.session = onnxruntime.InferenceSession(
                model_path, sess_options=so, providers=providers
            )
        except Exception as e:
            print(f"[ERROR] Failed to load ONNX model: {e}")
            raise

        print(f"[INFO] Providers enabled: {self.session.get_providers()}", flush=True)

        self.input_names = [i.name for i in self.session.get_inputs()]
        self.input_dtypes = [
            self._onnx_dtypes_to_numpy[i.type] for i in self.session.get_inputs()
        ]
        self.output_names = [o.name for o in self.session.get_outputs()]

        # 解析输入尺寸
        shape = list(self.session.get_inputs()[0].shape)
        self.h = shape[2] if shape[2] else 30
        self.w = shape[3] if shape[3] else 48
        self.swap = (2, 0, 1)

        print(f"[INFO] Model Input Size: {self.w}x{self.h}", flush=True)

    def preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img, (self.w, self.h)).astype(np.float32) / 255.0
        resized = resized.transpose(self.swap)
        return np.ascontiguousarray(resized, dtype=np.float32)

    def infer(self, img_bgr: np.ndarray) -> float:
        inp = self.preprocess(img_bgr)
        inp = np.asarray([inp], dtype=self.input_dtypes[0])

        out = self.session.run(self.output_names, {self.input_names[0]: inp})
        prob = float(np.squeeze(out[0]))
        return float(np.clip(prob, 0.0, 1.0))


# ---------------------------------------------------------
#  UTILS
# ---------------------------------------------------------
def list_images(folder: Path) -> List[Path]:
    files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
        files.extend(folder.rglob(ext))
    return sorted(files)


def write_json(path: Path, prob: float, label: int, thr: float, model_name: str):
    j = {
        "image": str(path),
        "prob_open": prob,
        "label": label,
        "label_name": "open" if label else "closed",
        "threshold": thr,
        "model": model_name,
    }
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(j, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------
# BUILD PROVIDERS
# ---------------------------------------------------------
def build_providers(ep: str, dtype: str, model_dir: Path):
    ep = ep.lower()
    dtype = dtype.lower()
    trt_path = str(model_dir)

    if ep == "cpu":
        print("[INFO] Using CPUExecutionProvider")
        return ["CPUExecutionProvider"]

    if ep == "cuda":
        print(f"[INFO] Using CUDAExecutionProvider (dtype={dtype})")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    if ep == "tensorrt":
        print(f"[INFO] Using TensorRTExecutionProvider (dtype={dtype})")
        trt_options = {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": trt_path,
            "trt_fp16_enable": dtype == "fp16",
            "trt_int8_enable": dtype == "int8",
        }
        return [
            ("TensorrtExecutionProvider", trt_options),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    raise ValueError(f"Unknown provider: {ep}")

def move_to_label_folder(image_path: Path, label: int):
    """
    根据 label（0/1）移动图片及对应的 JSON 到同名文件夹。
      - 0/   -> closed
      - 1/   -> open
    文件名不变
    """
    target_dir = image_path.parent / str(label)
    target_dir.mkdir(exist_ok=True)

    # 移动图片
    img_dst = target_dir / image_path.name
    image_path.rename(img_dst)

    # 移动JSON（如果存在）
    json_path = image_path.with_suffix(".json")
    if json_path.exists():
        json_dst = target_dir / json_path.name
        json_path.rename(json_dst)

    # 返回移动后的图片路径
    return img_dst


# ---------------------------------------------------------
#  MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("OCEC classifier batch processing")

    parser.add_argument(
        "--images_root",
        type=str,
        required=True,
        help="Root dir of cropped eye images, e.g. /10/cvz/.../fatigue/cropped",
    )
    parser.add_argument(
        "--ocec_model",
        type=str,
        default="/103/guochuang/Code/myOCEC/ocec_l.onnx",
        help="Path to OCEC onnx model, e.g. ocec_l.onnx",
    )
    parser.add_argument(
        "--csv_output_dir",
        type=str,
        default="/10/cvz/guochuang/dataset/Classification/fatigue/list",
        help="Directory to save CSV files.",
    )
    parser.add_argument(
        "--rel_prefix",
        type=str,
        default="/10/cvz/guochuang/dataset/Classification/fatigue/cropped",
        help="Prefix used in CSV paths (logical path in your training code).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold on prob_open; >= threshold => label=1(open), else 0(closed).",
    )
    parser.add_argument(
        "--images_per_csv",
        type=int,
        default=12000,
        help="Max number of lines per CSV file.",
    )
    parser.add_argument(
        "--write_csv",
        action="store_true",
        help="write csv file or not",
    )

    parser.add_argument("--ep", type=str, default="cuda",
                        choices=["cpu", "cuda", "tensorrt"])
    parser.add_argument("--dtype", type=str, default="fp32",
                        choices=["fp32", "fp16", "int8"])

    args = parser.parse_args()

    print("\n========== CONFIG ==========")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("============================\n", flush=True)
    
    root = Path(args.images_root).resolve()
    out_dir = Path(args.csv_output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    model_dir = Path(args.ocec_model).resolve().parent
    providers = build_providers(args.ep, args.dtype, model_dir)

    # 加载模型
    classifier = OCECClassifier(args.ocec_model, providers)

    subdirs = [d for d in root.iterdir() if d.is_dir()]
    subdirs.sort()

    total_images = 0
    total_csv = 0

    # ----------------------------------------------------------
    # 处理每个 20000xxxxx 子目录
    # ----------------------------------------------------------
    for sub in subdirs:
        folder = sub.name
        print(f"\n===== Processing folder {folder} =====", flush=True)
        
        # 检查并创建 0/1 目录（如果不存在）
        dir0 = sub / "0"
        dir1 = sub / "1"
        dir0.mkdir(exist_ok=True)
        dir1.mkdir(exist_ok=True)

        # 列出“根目录”里的图片，不进入 0/1 子目录
        imgs = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            imgs.extend(sub.glob(ext))   # 只匹配根目录

        imgs = sorted(imgs)
        N = len(imgs)

        if N == 0:
            print("[WARN] No images found")
            continue

        print(f"[INFO] {N} images found.", flush=True)

        buf = []
        part = 0
        t0 = time.time()

        for idx, path in enumerate(imgs, 1):
            img = cv2.imread(str(path))
            if img is None:
                print(f"[WARN] read failed: {path}")
                continue

            prob = classifier.infer(img)
            label = 1 if prob >= args.threshold else 0
            # write_json(path, prob, label, args.threshold, Path(args.ocec_model).name)
            
            new_path = move_to_label_folder(path, label)
            
            if args.write_csv:
                path = new_path   # 后面 CSV 也用新地址

                rel = path.relative_to(root)
                csv_line = f"{args.rel_prefix}/{rel.as_posix()},{label}"
                buf.append(csv_line)
                total_images += 1

            # ---- 进度条每 1000 张刷新 ----
            if idx % 1000 == 0:
                pct = idx * 100.0 / N
                elapsed = time.time() - t0
                eta = elapsed / idx * (N - idx)
                print(f"[{folder}] {idx}/{N} ({pct:.1f}%)  ETA: {eta/60:.1f} min",
                      flush=True)
                
            if args.write_csv:
            # ---- 写 CSV ----
                if len(buf) >= args.images_per_csv:
                    name = f"{folder}_{part:03d}.csv"
                    with open(out_dir / name, "w", encoding="utf-8") as f:
                        f.write("\n".join(buf))

                    print(f"[INFO] wrote CSV {name} ({len(buf)})", flush=True)
                    buf = []
                    part += 1
                    total_csv += 1

        # ---- 处理剩余 buf ----
        if args.write_csv:
            if buf:
                name = f"{folder}_{part:03d}.csv"
                with open(out_dir / name, "w", encoding="utf-8") as f:
                    f.write("\n".join(buf))
                print(f"[INFO] wrote CSV {name} ({len(buf)})", flush=True)
                total_csv += 1

    print("\n======== DONE ========")
    print(f"Total images processed: {total_images}")
    print(f"Total CSV generated   : {total_csv}")
    print("======================\n", flush=True)

if __name__ == "__main__":
    main()