#!/bin/bash
# 两阶段渐进式微调脚本
# 
# 策略：
# 1. 第一阶段：加载模型A权重 → 冻结backbone → 只训练分类头，5-10个epoch
# 2. 第二阶段：如果第一阶段有效（F1从0.776→0.80+），解冻部分backbone，用小LR继续训练

# ========== 配置区域 ==========
# 模型A的最佳checkpoint路径
PRETRAIN_MODEL="runs/ocec_hq_finetune_progressive/v14/ocec_best_epoch0069_f1_0.7981.pt"

# 数据集B的路径
DATA_ROOT_B="/ssddisk/guochuang/ocec/parquet_hq"

# 微调输出目录
OUTPUT_DIR="runs/ocec_hq_finetune_progressive_v2"

# 第一阶段配置（冻结backbone，只训练head）
STAGE1_EPOCHS=10          # 第一阶段训练轮数
STAGE1_LR=5e-6            # 第一阶段学习率（head可以稍大）

# 第二阶段配置（解冻部分backbone）
STAGE2_EPOCHS=1000        # 第二阶段训练轮数
STAGE2_LR=1e-6            # 第二阶段学习率（backbone需要很小）
UNFREEZE_EPOCH=10         # 在哪个epoch解冻backbone（STAGE1_EPOCHS + 1）

# 目标F1阈值（第一阶段需要达到这个值才继续第二阶段）
TARGET_F1=0.80
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

echo "=========================================="
echo "两阶段渐进式微调"
echo "=========================================="
echo "预训练模型: ${PRETRAIN_MODEL}"
echo "数据集B: ${DATA_ROOT_B}"
echo "输出目录: ${OUTPUT_DIR}"
echo ""
echo "第一阶段: 冻结backbone，训练head (${STAGE1_EPOCHS} epochs, LR=${STAGE1_LR})"
echo "第二阶段: 解冻部分backbone (从epoch ${UNFREEZE_EPOCH}开始, ${STAGE2_EPOCHS} epochs, LR=${STAGE2_LR})"
echo "目标F1: ${TARGET_F1}"
echo "=========================================="

# 第一阶段：冻结backbone，只训练head
export OCEC_ARGS="train \
    --data_root ${DATA_ROOT_B} \
    --output_dir ${OUTPUT_DIR} \
    --pretrain ${PRETRAIN_MODEL} \
    --epochs $((STAGE1_EPOCHS + STAGE2_EPOCHS)) \
    --batch_size 1024 \
    --num_workers 8 \
    --train_ratio 0.8 \
    --val_ratio 0.2 \
    --image_size 32x64 \
    --base_channels 32 \
    --num_blocks 4 \
    --arch_variant baseline \
    --margin_method none \
    --lr ${STAGE1_LR} \
    --stage2_lr ${STAGE2_LR} \
    --weight_decay 1e-4 \
    --seed 42 \
    --device auto \
    --warmup_epochs 3 \
    --freeze_backbone \
    --unfreeze_backbone_epoch ${UNFREEZE_EPOCH} \
    --use_amp"

    # --freeze_backbone \

# 激活 conda 环境并运行训练
LOG_FILE="logs/train/hq_data/train_finetune_progressive_$(date +%Y%m%d_%H%M%S).log"
nohup bash -c "source $(conda info --base)/etc/profile.d/conda.sh && conda activate deploy && exec -a ocec_fprog python - << 'EOF'
import os, shlex
from ocec.__main__ import main
args = shlex.split(os.environ['OCEC_ARGS'])
main(args)
EOF" > ${LOG_FILE} 2>&1 &

echo "微调训练已启动"
echo "日志文件: ${LOG_FILE}"
echo "TensorBoard: tensorboard --logdir ${OUTPUT_DIR}"
echo ""
echo "提示: 第一阶段结束后，检查验证集F1是否达到${TARGET_F1}"
echo "      如果达到，训练会自动进入第二阶段（解冻部分backbone）"

####pkill -f '\<ocec\>'