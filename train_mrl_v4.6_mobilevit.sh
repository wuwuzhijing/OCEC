#!/bin/bash
# ============================================================
# MRL Eye Dataset 独立训练 v4.6 (预训练 backbone)
# ============================================================
#
# v4.6 改进 (vs v2):
#   - 使用 torchvision 预训练 MobileNetV3-Small 替换自研小网络
#   - ImageNet 预训练权重提供通用视觉特征，大幅提升跨 subject 泛化能力
#   - image_size: 64x64（匹配预训练 backbone 输入能力）
#   - 参数量 ~1.5M（v2 自研 ~0.3M，v4.6 容量大且特征质量高）
#
# 可选 backbone (torchvision): mobilenet_v3_small | efficientnet_b0 | resnet18 | resnet34
#              (timm): repvgg_a0 | repvgg_a1 | repvgg_a2 | mobilevit | repvgg_b1 | repvgg_b2
#
# 续训: 设置 RESUME_FROM="auto" 或指定路径
# ============================================================

# ========== 续训配置 ==========
RESUME_FROM=""
# RESUME_FROM="auto"
RESUME_LR="5e-5"
# ============================

# ========== 模型配置 ==========
# torchvision:
# timm:
# ========== 模型配置 ==========
# MobileNetV3-Small: 小尺度无 BN 精度损失问题
# RepVGG 在 64×64 下 BN 统计不稳定，MobileNetV3 用 GN+SE 更鲁棒
BACKBONE="mobilevit_xxs"

# 预下载权重目录（训练机离线时使用）
# 提前在有网络的机器上把权重文件下载好，放到这个目录下
# 文件名：{backbone_name}.pth 或 {backbone_name}.safetensors
WEIGHTS_DIR="/ssddisk/guochuang/ocec/pretrained_weights"
# ==============================

OUTPUT_DIR="runs/ocec_mrl_v4.6_mobilevit"

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

export OCEC_ARGS="train \
    --data_root /ssddisk/guochuang/ocec/MRL/mrl_eyes_2018/ \
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
    --tb_port 6014 \
    --pretrained_backbone ${BACKBONE} \
    --pretrained_weights_dir ${WEIGHTS_DIR} \
    ${RESUME_ARG}"

LOG_FILE="logs/train/mrl/train_mrl_v4.6_mobilevit_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs/train/mrl

nohup bash -c "source \$(conda info --base)/etc/profile.d/conda.sh && conda activate tf_310 && exec -a mrl_v4.6 python - << 'EOF'
import os, shlex
from ocec.__main__ import main
args = shlex.split(os.environ['OCEC_ARGS'])
main(args)
EOF" > ${LOG_FILE} 2>&1 &

echo ""
echo "============================================"
if [ -n "$RESUME_ARG" ]; then
    echo "  MRL v4.6 续训已启动 (resume)"
else
    echo "  MRL v4.6 独立训练已启动 (fresh)"
fi
echo "============================================"
echo "  v4.6 改进: pretrained ${BACKBONE} + 64x64"
echo "  数据: /ssddisk/guochuang/ocec/MRL/mrl_eyes_2018/"
echo "  输出: ${OUTPUT_DIR}"
echo "  日志: ${LOG_FILE}"
echo "  查看: tail -f ${LOG_FILE}"
echo "============================================"
