#!/bin/bash
# ============================================================
# 双数据集联合训练 v4 (预训练 backbone)
# ============================================================
#
# v4 改进 (vs v2):
#   - 使用 torchvision 预训练 MobileNetV3-Small 替换自研小网络
#   - ImageNet 预训练权重提供通用视觉特征，大幅提升跨 subject 泛化能力
#   - image_size: 64x64
#   - 参数量 ~5.5M（含 head，远大于 v2 自研 ~0.3M）
#
# 可选 backbone (torchvision): mobilenet_v3_small | efficientnet_b0 | resnet18 | resnet34
#              (timm): repvgg_a0 | repvgg_a1 | repvgg_a2 | repvgg_b0 | repvgg_b1 | repvgg_b2
#
# ⚠️  严格区分:
#   - MRL v4:  train_mrl_v4.sh      → runs/ocec_mrl_v4/      port 6006
#   - 联合 v4:  train_combined_v4.sh → runs/ocec_combined_v4/ port 6007
# ============================================================

# ========== 续训配置 ==========
RESUME_FROM=""
# RESUME_FROM="auto"
RESUME_LR="5e-5"
# ============================

# ========== 模型配置 ==========
# torchvision:
# BACKBONE="mobilenet_v3_small"
# BACKBONE="efficientnet_b0"
# BACKBONE="resnet18"
# timm:
# BACKBONE="repvgg_a0"
# BACKBONE="repvgg_a2"
BACKBONE="repvgg_b0"
# ==============================

OUTPUT_DIR="runs/ocec_combined_v4"

RESUME_ARG=""
if [ "$RESUME_FROM" = "auto" ]; then
    BEST_CKPT=$(ls -t ${OUTPUT_DIR}/*/ocec_best_epoch*_f1_*.pt 2>/dev/null | head -1)
    if [ -n "$BEST_CKPT" ]; then
        RESUME_ARG="--resume ${BEST_CKPT}"
        echo "✅ 自动找到 best checkpoint: ${BEST_CKPT}"
    else
        echo "⚠️  未找到已有 checkpoint，将从头训练"
    fi
elif [ -n "$RESUME_FROM" ]; then
    if [ -f "$RESUME_FROM" ]; then
        RESUME_ARG="--resume ${RESUME_FROM}"
        echo "✅ 使用指定 checkpoint: ${RESUME_FROM}"
    else
        echo "❌ checkpoint 不存在: ${RESUME_FROM}"
        exit 1
    fi
fi

TRAIN_LR="1e-4"
if [ -n "$RESUME_ARG" ] && [ -n "$RESUME_LR" ]; then
    TRAIN_LR="$RESUME_LR"
    echo "✅ 续训学习率: ${TRAIN_LR}"
fi

# ---------- 确保合并数据存在 ----------
COMBINED_PARQUET="/ssddisk/guochuang/ocec/MRL/ocec_combined/dataset.parquet"
if [ ! -f "$COMBINED_PARQUET" ]; then
    echo "合并数据集不存在，正在创建..."
    python3 scripts/merge_datasets_to_parquet.py \
        --existing data/dataset.parquet \
        --mrl /ssddisk/guochuang/ocec/MRL/mrl_eyes_2018/dataset.parquet \
        --output-dir /ssddisk/guochuang/ocec/MRL/ocec_combined
    exit 1
fi

# ---------- 训练 ----------
export OCEC_ARGS="train \
    --data_root /ssddisk/guochuang/ocec/MRL/ocec_combined/ \
    --output_dir ${OUTPUT_DIR} \
    --epochs 1000 \
    --batch_size 512 \
    --num_workers 16 \
    --train_ratio 0.8 \
    --val_ratio 0.2 \
    --image_size 64x64 \
    --base_channels 64 \
    --num_blocks 6 \
    --arch_variant baseline \
    --margin_method cosface \
    --lr ${TRAIN_LR} \
    --weight_decay 5e-4 \
    --dropout 0.35 \
    --seed 42 \
    --device auto \
    --warmup_epochs 5 \
    --use_amp \
    --neg_class_weight 1.5 \
    --enable_hard_negative_mining \
    --tb_port 6007 \
    --pretrained_backbone ${BACKBONE} \
    ${RESUME_ARG}"

LOG_FILE="logs/train/combined/train_combined_v4_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs/train/combined

nohup bash -c "source \$(conda info --base)/etc/profile.d/conda.sh && conda activate tf_310 && exec -a combined_v4 python - << 'EOF'
import os, shlex
from ocec.__main__ import main
args = shlex.split(os.environ['OCEC_ARGS'])
main(args)
EOF" > ${LOG_FILE} 2>&1 &

echo ""
echo "============================================"
if [ -n "$RESUME_ARG" ]; then
    echo "  联合训练 v4 续训已启动 (resume)"
else
    echo "  联合训练 v4 已启动 (fresh)"
fi
echo "============================================"
echo "  v4 改进: pretrained ${BACKBONE} + 64x64"
echo "  数据: /ssddisk/guochuang/ocec/MRL/ocec_combined/"
echo "  输出: ${OUTPUT_DIR}"
echo "  日志: ${LOG_FILE}"
echo "  查看: tail -f ${LOG_FILE}"
echo "============================================"
