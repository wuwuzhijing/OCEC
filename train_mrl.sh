#!/bin/bash
# MRL Eye Dataset 独立训练脚本
# 数据集：MRL Eye Dataset (84,898 samples, 37 subjects)
# 标签：0=闭眼(closed), 1=睁眼(open)
# 数据已按 subject 分层划分 train/val (80/20)
#
# 执行方式：bash train_mrl.sh

export OCEC_ARGS="train \
    --data_root /ssddisk/guochuang/ocec/MRL/mrl_eyes_2018/ \
    --output_dir runs/ocec_mrl \
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

LOG_FILE="logs/train/mrl/train_mrl_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs/train/mrl

nohup bash -c "source \$(conda info --base)/etc/profile.d/conda.sh && conda activate tf_310 && exec -a mrl python - << 'EOF'
import os, shlex
from ocec.__main__ import main
args = shlex.split(os.environ['OCEC_ARGS'])
main(args)
EOF" > ${LOG_FILE} 2>&1 &

echo "MRL 训练已启动"
echo "日志文件: ${LOG_FILE}"
echo "TensorBoard: tensorboard --logdir runs/ocec_mrl"
echo ""
echo "查看日志: tail -f ${LOG_FILE}"
echo "查看进度: grep 'Train epoch summary' ${LOG_FILE} | tail -5"
