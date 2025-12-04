#!/bin/bash
# ConvNext 架构训练脚本
# 针对 ConvNext 架构优化的参数配置

export OCEC_ARGS="train \
    --data_root /ssddisk/guochuang/ocec/parquet_hq \
    --output_dir runs/ocec_hq_convnext \
    --epochs 1000 \
    --batch_size 1024 \
    --num_workers 16 \
    --train_ratio 0.8 \
    --val_ratio 0.2 \
    --image_size 32x64 \
    --base_channels 32 \
    --num_blocks 4 \
    --arch_variant convnext \
    --token_mixer_grid 2x4 \
    --margin_method cosface \
    --lr 1.2e-4 \
    --weight_decay 1e-4 \
    --seed 42 \
    --device auto \
    --warmup_epochs 20 \
    --use_amp"

# 激活 conda 环境并运行训练
nohup bash -c "source $(conda info --base)/etc/profile.d/conda.sh && conda activate deploy && exec -a ocec_t python - << 'EOF'
import os, shlex
from ocec.__main__ import main
args = shlex.split(os.environ['OCEC_ARGS'])
main(args)
EOF" > logs/train/hq_data/train_convnext_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "ConvNext 训练已启动，日志文件: logs/train/hq_data/train_convnext_$(date +%Y%m%d_%H%M%S).log"
echo "输出目录: runs/ocec_hq_convnext"
echo "TensorBoard: tensorboard --logdir runs/ocec_hq_convnext"