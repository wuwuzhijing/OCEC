#!/bin/bash
# 微调脚本：在模型A（数据集A训练）的基础上，使用数据集B进行微调
# 
# 使用场景：
# - 模型A：在数据集A（25W public + 5W private）上训练，F1=0.985
# - 数据集B：40W private，分布差异大，直接训练F1=0.776
# - 目标：在模型A基础上微调，提升数据集B上的效果

# ========== 配置区域 ==========
# 设置预训练模型路径（模型A的最佳checkpoint）
# 可以使用通配符，脚本会自动选择最新的
PRETRAIN_MODEL="runs/ocec_6Whq+24WPublic/v8/ocec_best_epoch*.pt"

# 数据集B的路径
DATA_ROOT_B="/ssddisk/guochuang/ocec/parquet_hq"

# 微调输出目录
OUTPUT_DIR="runs/ocec_hq_finetune"
# =============================

# 处理通配符路径
if [[ "$PRETRAIN_MODEL" == *"*"* ]]; then
    PRETRAIN_MATCHES=($(ls -t $PRETRAIN_MODEL 2>/dev/null))
    if [ ${#PRETRAIN_MATCHES[@]} -eq 0 ]; then
        echo "错误: 未找到匹配的预训练模型: $PRETRAIN_MODEL"
        exit 1
    fi
    PRETRAIN_MODEL="${PRETRAIN_MATCHES[0]}"
    echo "找到预训练模型: $PRETRAIN_MODEL"
fi

# 检查预训练模型是否存在
if [ ! -f "$PRETRAIN_MODEL" ]; then
    echo "错误: 预训练模型不存在: $PRETRAIN_MODEL"
    exit 1
fi

export OCEC_ARGS="train \
    --data_root ${DATA_ROOT_B} \
    --output_dir ${OUTPUT_DIR} \
    --pretrain ${PRETRAIN_MODEL} \
    --epochs 200 \
    --batch_size 1024 \
    --num_workers 16 \
    --train_ratio 0.8 \
    --val_ratio 0.2 \
    --image_size 32x64 \
    --base_channels 32 \
    --num_blocks 4 \
    --arch_variant baseline \
    --margin_method cosface \
    --lr 5e-5 \
    --weight_decay 1e-4 \
    --seed 42 \
    --device auto \
    --warmup_epochs 10 \
    --use_amp"

# 激活 conda 环境并运行训练
nohup bash -c "source $(conda info --base)/etc/profile.d/conda.sh && conda activate deploy && exec -a ocec_finetune python - << 'EOF'
import os, shlex
from ocec.__main__ import main
args = shlex.split(os.environ['OCEC_ARGS'])
main(args)
EOF" > logs/train/hq_data/train_finetune_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "=========================================="
echo "微调训练已启动"
echo "=========================================="
echo "预训练模型: ${PRETRAIN_MODEL}"
echo "数据集B: ${DATA_ROOT_B}"
echo "输出目录: ${OUTPUT_DIR}"
echo "日志文件: logs/train/hq_data/train_finetune_$(date +%Y%m%d_%H%M%S).log"
echo "TensorBoard: tensorboard --logdir ${OUTPUT_DIR}"
echo "=========================================="