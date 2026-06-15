#!/bin/bash
# ============================================================
# 预训练 backbone 权重下载脚本
#
# 在有网络的机器上执行此脚本，下载完成后将整个目录
# 复制到训练机对应的路径下。
#
# 用法:
#   bash scripts/download_pretrained_weights.sh                     # 下载到默认目录
#   bash scripts/download_pretrained_weights.sh /path/to/dir        # 下载到指定目录
#
# 默认输出目录: ./pretrained_weights/
# 训练机目标路径: /ssddisk/guochuang/ocec/pretrained_weights/
# ============================================================

set -e

OUTPUT_DIR="${1:-./pretrained_weights}"
mkdir -p "$OUTPUT_DIR"
echo "============================================"
echo "  预训练权重下载"
echo "  输出目录: $(realpath "$OUTPUT_DIR")"
echo "============================================"

# ---------- helper ----------
download_torchvision() {
    local name="$1"
    local url="$2"
    local dst="$OUTPUT_DIR/${name}.pth"
    if [ -f "$dst" ]; then
        echo "  [跳过] ${name}.pth 已存在"
        return
    fi
    echo "  [下载] ${name} <- ${url}"
    wget -q --show-progress -O "$dst" "$url" || curl -L -o "$dst" "$url"
}

download_timm_safetensors() {
    # timm 1.x models are on huggingface hub
    # model_id format: timm/<repo_name>
    local name="$1"
    local repo="$2"
    local dst="$OUTPUT_DIR/${name}.safetensors"
    if [ -f "$dst" ]; then
        echo "  [跳过] ${name}.safetensors 已存在"
        return
    fi
    echo "  [下载] ${name} <- huggingface.co/timm/${repo}"

    python3 - "$name" "$repo" "$OUTPUT_DIR" << 'PYEOF'
import os, sys, shutil
name, repo, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]

# Try huggingface_hub first
try:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=f"timm/{repo}", filename="model.safetensors")
    shutil.copy(path, os.path.join(out_dir, f"{name}.safetensors"))
    print(f"    -> {name}.safetensors")
except ImportError:
    pass
else:
    sys.exit(0)

# Fallback: use timm to trigger download, then copy
import timm, glob
m = timm.create_model(name, pretrained=True, num_classes=0, global_pool='')
# Find the downloaded file in cache
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
pattern = os.path.join(cache_dir, f"models--timm--{repo}", "snapshots", "*", "*.safetensors")
matches = glob.glob(pattern)
if matches:
    shutil.copy(matches[0], os.path.join(out_dir, f"{name}.safetensors"))
    print(f"    -> {name}.safetensors")
else:
    # Also try torch hub cache for older timm
    torch_cache = os.path.expanduser("~/.cache/torch/hub/checkpoints")
    for f in os.listdir(torch_cache):
        if name in f and (f.endswith('.pth') or f.endswith('.pt')):
            shutil.copy(os.path.join(torch_cache, f), os.path.join(out_dir, f"{name}.pth"))
            print(f"    -> {name}.pth (from torch hub)")
            sys.exit(0)
    print(f"    WARNING: could not locate downloaded file for {name}")
PYEOF
}

# ========================================
# 1. torchvision backbones
# ========================================
echo ""
echo "--- torchvision backbones ---"

download_torchvision "mobilenet_v3_small" \
    "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"

download_torchvision "efficientnet_b0" \
    "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth"

download_torchvision "resnet18" \
    "https://download.pytorch.org/models/resnet18-f37072fd.pth"

download_torchvision "resnet34" \
    "https://download.pytorch.org/models/resnet34-b627a593.pth"

# ========================================
# 2. timm RepVGG backbones
# ========================================
echo ""
echo "--- timm RepVGG backbones ---"

download_timm_safetensors "repvgg_a0" "repvgg_a0.rvgg_in1k"
download_timm_safetensors "repvgg_a1" "repvgg_a1.rvgg_in1k"
download_timm_safetensors "repvgg_a2" "repvgg_a2.rvgg_in1k"
download_timm_safetensors "repvgg_b0" "repvgg_b0.rvgg_in1k"
download_timm_safetensors "repvgg_b1" "repvgg_b1.rvgg_in1k"
download_timm_safetensors "repvgg_b2" "repvgg_b2.rvgg_in1k"

# ========================================
# 完成
# ========================================
echo ""
echo "============================================"
echo "  下载完成"
echo "============================================"
ls -lh "$OUTPUT_DIR"/
echo ""
echo "将整个目录复制到训练机:"
echo "  scp -r $OUTPUT_DIR/ user@gpu-server:/ssddisk/guochuang/ocec/pretrained_weights/"
echo "============================================"
