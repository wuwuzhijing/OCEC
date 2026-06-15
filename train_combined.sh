#!/bin/bash
# ============================================================
# 双数据集联合训练脚本 (Existing OCEC + MRL Eye Dataset)
# ============================================================
#
# 数据集组成:
#   - Existing OCEC (real_data): 11,556 samples  (98% closed)
#   - MRL Eye Dataset:           84,898 samples  (50% closed / 50% open)
#   ─────────────────────────────────────────────────────────
#   - Combined:                  96,454 samples  (55% closed / 45% open)
#
# 拆分 (预设):
#   - Train: 82,557 (existing 9,244 + MRL 73,313)
#   - Val:   13,897 (existing 2,312 + MRL 11,585)
#
# ⚠️  严格区分说明:
#   - 独立 MRL 训练:   train_mrl.sh       → runs/ocec_mrl/       数据 /10/.../mrl_eyes_2018/
#   - 联合训练:        train_combined.sh  → runs/ocec_combined/  数据 /10/.../ocec_combined/
#   - 与之前所有训练脚本的输出目录、日志目录、数据目录完全隔离。
# ============================================================

# ---------- 第一步: 合并数据集 (如尚未合并) ----------
COMBINED_PARQUET="/10/cvz/guochuang/dataset/ocec_combined/dataset.parquet"
if [ ! -f "$COMBINED_PARQUET" ]; then
    echo "合并数据集不存在，正在创建..."
    python3 scripts/merge_datasets_to_parquet.py
    if [ $? -ne 0 ]; then
        echo "❌ 数据集合并失败!"
        exit 1
    fi
    echo "✅ 数据集合并完成: $COMBINED_PARQUET"
else
    echo "✅ 合并数据集已存在: $COMBINED_PARQUET"
fi

# ---------- 第二步: 训练 ----------
export OCEC_ARGS="train \
    --data_root /ssddisk/guochuang/ocec/MRL/ocec_combined \
    --output_dir runs/ocec_combined \
    --epochs 1000 \
    --batch_size 1024 \
    --num_workers 16 \
    --train_ratio 0.8 \
    --val_ratio 0.2 \
    --image_size 32x64 \
    --base_channels 32 \
    --num_blocks 4 \
    --arch_variant baseline \
    --margin_method cosface \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --seed 42 \
    --device auto \
    --warmup_epochs 15 \
    --use_amp"

LOG_FILE="logs/train/combined/train_combined_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs/train/combined

nohup bash -c "source \$(conda info --base)/etc/profile.d/conda.sh && conda activate tf_310 && exec -a combined python - << 'EOF'
import os, shlex
from ocec.__main__ import main
args = shlex.split(os.environ['OCEC_ARGS'])
main(args)
EOF" > ${LOG_FILE} 2>&1 &

echo ""
echo "============================================"
echo "  联合训练已启动 (Existing + MRL)"
echo "============================================"
echo "  数据: /10/cvz/guochuang/dataset/ocec_combined/"
echo "  输出: runs/ocec_combined/"
echo "  日志: ${LOG_FILE}"
echo "  TB:   tensorboard --logdir runs/ocec_combined"
echo ""
echo "  查看: tail -f ${LOG_FILE}"
echo "============================================"
