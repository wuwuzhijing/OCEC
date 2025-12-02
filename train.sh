#!/bin/bash

# nohup bash -c "exec -a ocec python -m ocec train \
#     --data_root /ssddisk/guochuang/ocec/parquet \
#     --output_dir runs/ocec \
#     --epochs 10000 \
#     --batch_size 4096 \
#     --num_workers 16 \
#     --train_ratio 0.8 \
#     --val_ratio 0.2 \
#     --image_size 32x64 \
#     --base_channels 64 \
#     --num_blocks 8 \
#     --arch_variant baseline \
#     --seed 42 \
#     --device auto" \
#     > logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &

export OCEC_ARGS="train \
    --data_root /ssddisk/guochuang/ocec/parquet_hq \
    --output_dir runs/ocec_hq \
    --epochs 1000 \
    --batch_size 1024 \
    --num_workers 16 \
    --train_ratio 0.8 \
    --val_ratio 0.2 \
    --image_size 32x64 \
    --base_channels 32 \
    --num_blocks 4 \
    --arch_variant baseline \
    --use_arcface \
    --seed 42 \
    --device auto"

# 激活 conda 环境并运行训练
nohup bash -c "source $(conda info --base)/etc/profile.d/conda.sh && conda activate deploy && exec -a ocec python - << 'EOF'
import os, shlex
from ocec.__main__ import main
args = shlex.split(os.environ['OCEC_ARGS'])
main(args)
EOF" > logs/train/hq_data/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &