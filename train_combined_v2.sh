#!/bin/bash
# ============================================================
# 双数据集联合训练 v2 (改进版)
# ============================================================
#
# v2 改进 (vs v1):
#   - MRL 验证集按 subject open-ratio 分层采样 → val 从 14.1%→56.4% open
#   - neg_class_weight 1.2→1.5 (更强惩罚 FP)
#   - dropout 0.3→0.5 (防止过拟合)
#   - weight_decay 1e-4→5e-4 (更强 L2)
#   - 启用 hard_negative_mining
#
# 数据: /10/cvz/guochuang/dataset/ocec_combined/
#   - train: 67,601 (39.8% open)
#   - val:   28,853 (56.4% open)
#
# 续训: 设置 RESUME_FROM 变量
#   RESUME_FROM="auto"           → 自动找 runs/ocec_combined_v2/ 下最新的 best checkpoint
#   RESUME_FROM="/path/to/ckpt"  → 指定 checkpoint 路径
#   RESUME_FROM=""               → 从头训练（默认）
#
# ⚠️  严格区分:
#   - MRL v2:  train_mrl_v2.sh      → runs/ocec_mrl_v2/      port 6006
#   - 联合 v2:  train_combined_v2.sh → runs/ocec_combined_v2/ port 6007
# ============================================================

# ========== 续训配置 ==========
# 设为 "auto" 自动找 best.pt，或指定路径，空字符串=从头训练
RESUME_FROM=""
# RESUME_FROM="auto"
# RESUME_FROM="runs/ocec_combined_v2/v1/ocec_best_epoch0095_f1_0.4244.pt"

# 续训时的学习率（为空则使用默认 lr）
RESUME_LR="5e-5"
# ============================

OUTPUT_DIR="runs/ocec_combined_v2"

# --- 处理续训 ---
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
    if [ $? -ne 0 ]; then
        echo "❌ 数据集合并失败!"
        exit 1
    fi
    echo "✅ 数据集合并完成: $COMBINED_PARQUET"
else
    echo "✅ 合并数据集已存在: $COMBINED_PARQUET"
fi

# ---------- 训练 ----------
export OCEC_ARGS="train \
    --data_root /ssddisk/guochuang/ocec/MRL/ocec_combined/ \
    --output_dir ${OUTPUT_DIR} \
    --epochs 300 \
    --batch_size 1024 \
    --num_workers 16 \
    --train_ratio 0.8 \
    --val_ratio 0.2 \
    --image_size 32x64 \
    --base_channels 32 \
    --num_blocks 4 \
    --arch_variant baseline \
    --margin_method cosface \
    --lr ${TRAIN_LR} \
    --weight_decay 5e-4 \
    --dropout 0.5 \
    --seed 42 \
    --device auto \
    --warmup_epochs 15 \
    --use_amp \
    --neg_class_weight 1.5 \
    --enable_hard_negative_mining \
    --tb_port 6007 \
    ${RESUME_ARG}"

LOG_FILE="logs/train/combined/train_combined_v2_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs/train/combined

nohup bash -c "source \$(conda info --base)/etc/profile.d/conda.sh && conda activate tf_310 && exec -a combined_v2 python - << 'EOF'
import os, shlex
from ocec.__main__ import main
args = shlex.split(os.environ['OCEC_ARGS'])
main(args)
EOF" > ${LOG_FILE} 2>&1 &

echo ""
echo "============================================"
if [ -n "$RESUME_ARG" ]; then
    echo "  联合训练 v2 续训已启动 (resume)"
else
    echo "  联合训练 v2 已启动 (fresh)"
fi
echo "============================================"
echo "  数据: /ssddisk/guochuang/ocec/MRL/ocec_combined/"
echo "  输出: ${OUTPUT_DIR}"
echo "  日志: ${LOG_FILE}"
echo "  查看: tail -f ${LOG_FILE}"
echo "============================================"
