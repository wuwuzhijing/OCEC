#!/bin/bash
# ============================================================
# MRL 独立训练 v4 — 无 margin baseline（实验 A）
#
# 对比 CosFace，验证 margin 是否导致 train/val loss 倒挂和过拟合。
# 标签: closed=0, open=1 (与 CosFace 一致，便于对比)
# ============================================================

# ========== 配置 ==========
RESUME_FROM=""
RESUME_LR="5e-5"
BACKBONE="repvgg_b0"
WEIGHTS_DIR="/ssddisk/guochuang/ocec/pretrained_weights"
OUTPUT_DIR="runs/ocec_mrl_v4_nomargin"
# ==========================

export OCEC_ARGS="train \
    --data_root /ssddisk/guochuang/ocec/MRL/mrl_eyes_2018/ \
    --output_dir ${OUTPUT_DIR} \
    --epochs 100 \
    --batch_size 512 \
    --num_workers 16 \
    --image_size 64x64 \
    --arch_variant baseline \
    --margin_method none \
    --lr 1e-4 \
    --weight_decay 5e-4 \
    --dropout 0.35 \
    --seed 42 \
    --device auto \
    --warmup_epochs 5 \
    --use_amp \
    --tb_port 6008 \
    --pretrained_backbone ${BACKBONE} \
    --pretrained_weights_dir ${WEIGHTS_DIR}"

LOG_FILE="logs/train/mrl/train_mrl_v4_nomargin_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs/train/mrl

nohup bash -c "source \$(conda info --base)/etc/profile.d/conda.sh && conda activate tf_310 && exec -a mrl_nomargin python - << 'EOF'
import os, shlex
from ocec.__main__ import main
args = shlex.split(os.environ['OCEC_ARGS'])
main(args)
EOF" > ${LOG_FILE} 2>&1 &

echo "v4 no-margin baseline 已启动 (port 6008)"
echo "日志: ${LOG_FILE}"
