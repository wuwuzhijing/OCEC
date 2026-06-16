#!/bin/bash
# ============================================================
# v4.3 + fatigue 数据集联合训练
#
# 数据集组成:
#   - MRL Eye Dataset:    84,898 samples (49% closed / 51% open)
#   - Fatigue src_dataset: ~170K samples (65% normal / 35% squint)
#   - Fatigue L2+_dataset: ~2K samples (3-class, 2_unknown skipped)
#
# 步骤:
#   1. 转换 fatigue → parquet
#   2. 合并 MRL + fatigue
#   3. 训练 repvgg_b0
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKBONE="repvgg_b0"
WEIGHTS_DIR="/ssddisk/guochuang/ocec/pretrained_weights"
OUTPUT_DIR="runs/ocec_mrl_v4.3_fatigue"

# 1. Convert fatigue dataset
FATIGUE_PARQUET="/ssddisk/guochuang/ocec/fatigue_dataset/dataset.parquet"
if [ ! -f "$FATIGUE_PARQUET" ]; then
    echo "=== Step 1: Convert fatigue → parquet ==="
    python3 "${SCRIPT_DIR}/scripts/convert_fatigue_to_parquet.py"
    [ $? -ne 0 ] && echo "❌ fatigue conversion failed" && exit 1
fi

# 2. Merge MRL + fatigue
MERGED_PARQUET="/ssddisk/guochuang/ocec/mrl_fatigue_combined/dataset.parquet"
if [ ! -f "$MERGED_PARQUET" ]; then
    echo "=== Step 2: Merge MRL + fatigue ==="
    python3 -c "
import pandas as pd
from pathlib import Path

mrl = pd.read_parquet('/ssddisk/guochuang/ocec/MRL/mrl_eyes_2018/dataset.parquet')
fatigue = pd.read_parquet('${FATIGUE_PARQUET}')
print(f'MRL: {len(mrl)} rows')
print(f'Fatigue: {len(fatigue)} rows')

combined = pd.concat([mrl, fatigue], ignore_index=True)
Path('${MERGED_PARQUET}').parent.mkdir(parents=True, exist_ok=True)
combined.to_parquet('${MERGED_PARQUET}', index=False)

for s in ['train','val']:
    sub = combined[combined['split']==s]
    lb = sub['label'].value_counts()
    print(f'{s}: {len(sub)}, open={lb.get(\"open\",0)}, closed={lb.get(\"closed\",0)}')
print(f'Saved: ${MERGED_PARQUET} ({len(combined)} rows)')
"
    [ $? -ne 0 ] && echo "❌ merge failed" && exit 1
fi

# 3. Train
echo "=== Step 3: Train v4.3 + fatigue ==="
export OCEC_ARGS="train \
    --data_root /ssddisk/guochuang/ocec/mrl_fatigue_combined/ \
    --output_dir ${OUTPUT_DIR} \
    --epochs 300 \
    --batch_size 256 \
    --num_workers 16 \
    --image_size 64x64 \
    --arch_variant baseline \
    --margin_method cosface \
    --lr 1e-4 \
    --weight_decay 5e-4 \
    --dropout 0.35 \
    --seed 42 \
    --device auto \
    --warmup_epochs 5 \
    --use_amp \
    --neg_class_weight 1.5 \
    --enable_hard_negative_mining \
    --tb_port 6016 \
    --pretrained_backbone ${BACKBONE} \
    --pretrained_weights_dir ${WEIGHTS_DIR}"

LOG_FILE="logs/train/mrl/train_mrl_v4.3_fatigue_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs/train/mrl

nohup bash -c "source \$(conda info --base)/etc/profile.d/conda.sh && conda activate tf_310 && exec -a mrl_v4.3fat python - << 'EOF'
import os, shlex
from ocec.__main__ import main
args = shlex.split(os.environ['OCEC_ARGS'])
main(args)
EOF" > ${LOG_FILE} 2>&1 &

echo "v4.3 + fatigue 训练已启动 (port 6016)"
echo "日志: ${LOG_FILE}"
