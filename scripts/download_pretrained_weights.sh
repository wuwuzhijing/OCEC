#!/bin/bash
# ============================================================
# 预训练 backbone 权重下载脚本
#
# 所有模型均通过 huggingface 下载（国内可用 hf-mirror.com 镜像）
# 默认自动启用镜像加速，无需 VPN。
#
# 用法:
#   bash scripts/download_pretrained_weights.sh                     # 默认（hf-mirror）
#   bash scripts/download_pretrained_weights.sh /path/to/dir        # 指定输出目录
#   HF_MIRROR="" bash scripts/download_pretrained_weights.sh        # 禁用镜像（直连）
#
# 默认输出: ./pretrained_weights/
# 训练机目标: /ssddisk/guochuang/ocec/pretrained_weights/
# ============================================================

set -e

OUTPUT_DIR="${1:-./pretrained_weights}"
mkdir -p "$OUTPUT_DIR"

# 默认使用 hf 镜像；设为空字符串则直连
HF_MIRROR="${HF_MIRROR:-https://hf-mirror.com}"

echo "============================================"
echo "  预训练权重下载"
echo "  输出目录: $(realpath "$OUTPUT_DIR")"
if [ -n "$HF_MIRROR" ]; then
    echo "  镜像: $HF_MIRROR"
else
    echo "  直连 huggingface / pytorch"
fi
echo "============================================"

# ---------- helpers ----------

download_from_hf() {
    # 从 huggingface (或其镜像) 下载单个文件
    # 用法: download_from_hf <repo> <filename> <local_name>
    local repo="$1"       # e.g. "pytorch/vision" or "timm/repvgg_b0.rvgg_in1k"
    local file="$2"       # e.g. "models/resnet18-f37072fd.pth" or "model.safetensors"
    local local_name="$3" # e.g. "resnet18.pth"
    local dst="$OUTPUT_DIR/$local_name"

    if [ -f "$dst" ]; then
        echo "  [跳过] $local_name 已存在"
        return
    fi

    local base_url
    if [ -n "$HF_MIRROR" ]; then
        base_url="$HF_MIRROR"
    else
        base_url="https://huggingface.co"
    fi

    echo "  [下载] $local_name <- $repo/$file"
    wget -q --show-progress -O "$dst" "$base_url/$repo/resolve/main/$file" || \
    curl -# -L -o "$dst" "$base_url/$repo/resolve/main/$file"
}

# ========================================
# 1. torchvision backbones (from pytorch/vision on huggingface)
# ========================================
echo ""
echo "--- torchvision backbones ---"

download_from_hf "pytorch/vision" "models/mobilenet_v3_small-047dcff4.pth"    "mobilenet_v3_small.pth"
download_from_hf "pytorch/vision" "models/efficientnet_b0_rwightman-7f5810bc.pth" "efficientnet_b0.pth"
download_from_hf "pytorch/vision" "models/resnet18-f37072fd.pth"             "resnet18.pth"
download_from_hf "pytorch/vision" "models/resnet34-b627a593.pth"             "resnet34.pth"

# ========================================
# 2. timm RepVGG backbones (from timm org on huggingface)
# ========================================
echo ""
echo "--- timm RepVGG backbones ---"

for model in a0 a1 a2 b0 b1 b2; do
    download_from_hf "timm/repvgg_${model}.rvgg_in1k" "model.safetensors" "repvgg_${model}.safetensors"
done

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
