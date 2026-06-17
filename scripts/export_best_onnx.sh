#!/bin/bash
# ============================================================
# 通用 ONNX 导出脚本
#
# 自动找到指定 run 目录下 F1 最高的 checkpoint 并导出 ONNX。
# 也支持显式指定 checkpoint 路径。
#
# Usage:
#   bash scripts/export_best_onnx.sh runs/ocec_mrl_v4.3_repvgg_b0
#   bash scripts/export_best_onnx.sh runs/ocec_mrl_v4.3_fatigue
#   bash scripts/export_best_onnx.sh runs/ocec_mrl_v4.3_fatigue/v1/ocec_best_epoch0126_f1_0.9858.pt
#   bash scripts/export_best_onnx.sh runs/ocec_mrl_v4.3_fatigue --output /path/to/output.onnx
# ============================================================

set -e

RUN_DIR="$1"
OUTPUT_ONNX="${2:-}"

if [ -z "$RUN_DIR" ]; then
    echo "Usage: bash scripts/export_best_onnx.sh <run_dir_or_ckpt> [output.onnx]"
    echo ""
    echo "Examples:"
    echo "  bash scripts/export_best_onnx.sh runs/ocec_mrl_v4.3_repvgg_b0"
    echo "  bash scripts/export_best_onnx.sh runs/ocec_mrl_v4.3_fatigue/v1/ocec_best_epoch0126_f1_0.9858.pt"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Resolve checkpoint
if [ -f "$RUN_DIR" ]; then
    # Explicit checkpoint path
    BEST_CKPT="$RUN_DIR"
else
    # Auto-find best checkpoint by F1
    BEST_CKPT=$(ls "$RUN_DIR"/*/ocec_best_epoch*_f1_*.pt 2>/dev/null | sort -t_ -k4 -rn | head -1)
    if [ -z "$BEST_CKPT" ]; then
        echo "ERROR: No checkpoint found in $RUN_DIR"
        exit 1
    fi
fi

# Resolve output path
F1_VAL=$(echo "$BEST_CKPT" | grep -oP 'f1_\K[0-9.]+')
RUN_NAME=$(echo "$RUN_DIR" | sed 's|runs/||' | tr '/' '_')
if [ -z "$OUTPUT_ONNX" ]; then
    OUTPUT_ONNX="${RUN_NAME}_f1_${F1_VAL}.onnx"
fi

echo "============================================"
echo "  ONNX Export"
echo "  Checkpoint: $BEST_CKPT"
echo "  F1:         $F1_VAL"
echo "  Output:     $OUTPUT_ONNX"
echo "============================================"

cd "$PROJECT_DIR"
# Try local weights dir if it exists
WEIGHTS_DIR="/ssddisk/guochuang/ocec/pretrained_weights"
EXTRA=""
[ -d "$WEIGHTS_DIR" ] && EXTRA="--pretrained_weights_dir $WEIGHTS_DIR"

python -m ocec exportonnx \
    --checkpoint "$BEST_CKPT" \
    --output "$OUTPUT_ONNX" \
    --opset 17 \
    $EXTRA

echo ""
echo "Done: $OUTPUT_ONNX"
ls -lh "$OUTPUT_ONNX"
