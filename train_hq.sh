#!/bin/bash
# 训练脚本示例
# 注意：每次训练会自动在输出目录下创建版本号文件夹（v1, v2, v3...）
# 例如：runs/ocec_hq/v1, runs/ocec_hq/v2 等

export OCEC_ARGS="train \
    --data_root /ssddisk/guochuang/ocec/parquet_hq/ \
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
    --margin_method cosface \
    --seed 42 \
    --device auto \
    --warmup_epochs 15 \
    --use_amp"

# 如果需要加载预训练权重，取消下面的注释并指定checkpoint路径
# --pretrain runs/ocec_hq/v1/checkpoint_best.pt \

# 如果需要恢复训练（包括优化器、调度器等状态），使用 --resume 参数
# --resume runs/ocec_hq/v1/checkpoint_best.pt \


# 激活 conda 环境并运行训练
# 使用绝对路径确保在正确的目录下执行
nohup bash -c "cd '$SCRIPT_DIR' && source \$(conda info --base)/etc/profile.d/conda.sh && conda activate deploy && exec -a ocec python - << 'EOF'
import os, shlex
from ocec.__main__ import main
args = shlex.split(os.environ['OCEC_ARGS'])
main(args)
EOF" > logs/train/hq_data/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &