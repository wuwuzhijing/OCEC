#!/bin/bash
# 三阶段渐进式微调脚本（基于 v5.0 最佳模型继续训练）
# 
# 策略：
# 1. 第一阶段：加载 v5.0 最佳模型权重 → 冻结backbone → 只训练分类头
# 2. 第二阶段：解冻部分backbone，用小LR继续训练
# 3. 第三阶段：在 epoch 1000 后进一步解冻最后 1-2 个 stage，使用更小LR
# 
# 改进：
# - 增强数据增强（CLAHE 0.25, Gamma校正, Exposure扰动）
# - Hard-Negative Mining 加权采样
# - 第三阶段解冻策略

# ========== 配置区域 ==========
# v5.0 最佳模型路径（F1=0.8909）
PRETRAIN_MODEL="runs/ocec_hq_finetune_progressive_v5.0/v1/ocec_best_epoch1004_f1_0.8909.pt"

# 数据集路径
DATA_ROOT_B="/ssddisk/guochuang/ocec/parquet_hq_v3"

# 微调输出目录
OUTPUT_DIR="runs/ocec_hq_finetune_progressive_v5.1_stage3"

# 第一阶段配置（冻结backbone，只训练head）
STAGE1_EPOCHS=10           # 第一阶段训练轮数（快速适应）
STAGE1_LR=1e-4             # 第一阶段学习率

# 第二阶段配置（解冻部分backbone）
STAGE2_EPOCHS=1000         # 第二阶段训练轮数
STAGE2_LR=3e-5             # 第二阶段学习率（比 v5.0 更小，因为从更高起点开始）
UNFREEZE_EPOCH=10          # 在哪个epoch解冻backbone

# 第三阶段配置（进一步解冻最后 1-2 个 stage）
STAGE3_EPOCHS=500          # 第三阶段额外训练轮数
STAGE3_LR=1.5e-5           # 第三阶段学习率（Stage2 的 50%）
UNFREEZE_EPOCH_STAGE3=1000 # 在哪个epoch进入第三阶段
UNFREEZE_RATIO_STAGE3=0.25 # 第三阶段解冻比例（解冻最后 1-2 个 stage）

# 目标F1阈值
TARGET_F1=0.85             # Stage3 需要达到的F1阈值
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
echo "三阶段渐进式微调（基于 v5.0 最佳模型）"
echo "=========================================="
echo "预训练模型: ${PRETRAIN_MODEL}"
echo "数据集B: ${DATA_ROOT_B}"
echo "输出目录: ${OUTPUT_DIR}"
echo ""
echo "第一阶段: 冻结backbone，训练head (${STAGE1_EPOCHS} epochs, LR=${STAGE1_LR})"
echo "第二阶段: 解冻部分backbone (从epoch ${UNFREEZE_EPOCH}开始, ${STAGE2_EPOCHS} epochs, LR=${STAGE2_LR})"
echo "第三阶段: 进一步解冻 (从epoch ${UNFREEZE_EPOCH_STAGE3}开始, 额外 ${STAGE3_EPOCHS} epochs, LR=${STAGE3_LR})"
echo "目标F1: ${TARGET_F1}"
echo "=========================================="

# 计算总epoch数
TOTAL_EPOCHS=$((STAGE1_EPOCHS + STAGE2_EPOCHS + STAGE3_EPOCHS))

# 训练命令
export OCEC_ARGS="train \
    --data_root ${DATA_ROOT_B} \
    --output_dir ${OUTPUT_DIR} \
    --pretrain ${PRETRAIN_MODEL} \
    --epochs ${TOTAL_EPOCHS} \
    --batch_size 8192 \
    --num_workers 8 \
    --train_ratio 0.8 \
    --val_ratio 0.2 \
    --image_size 32x64 \
    --base_channels 32 \
    --num_blocks 4 \
    --arch_variant baseline \
    --margin_method cosface \
    --lr ${STAGE1_LR} \
    --stage2_lr ${STAGE2_LR} \
    --stage3_lr ${STAGE3_LR} \
    --weight_decay 1e-4 \
    --seed 42 \
    --device auto \
    --warmup_epochs 3 \
    --freeze_backbone \
    --unfreeze_backbone_epoch ${UNFREEZE_EPOCH} \
    --unfreeze_backbone_epoch_stage3 ${UNFREEZE_EPOCH_STAGE3} \
    --unfreeze_backbone_ratio 0.5 \
    --unfreeze_backbone_ratio_stage3 ${UNFREEZE_RATIO_STAGE3} \
    --use_amp \
    --pseudo_update_start_epoch 200 \
    --neg_class_weight 1.3 \
    --enable_hard_negative_mining \
    --hard_neg_min_prob 0.7 \
    --hard_neg_weight 2.5"

# 激活 conda 环境并运行训练
LOG_FILE="logs/train/hq_data/train_finetune_progressive_stage3_$(date +%Y%m%d_%H%M%S).log"
nohup bash -c "source $(conda info --base)/etc/profile.d/conda.sh && conda activate deploy && exec -a fprog51_stage3 python - << 'EOF'
import os, shlex
from ocec.__main__ import main
args = shlex.split(os.environ['OCEC_ARGS'])
main(args)
EOF" > ${LOG_FILE} 2>&1 &

echo "三阶段微调训练已启动"
echo "日志文件: ${LOG_FILE}"
echo "TensorBoard: tensorboard --logdir ${OUTPUT_DIR}"
echo ""
echo "预期改进："
echo "- 增强数据增强（CLAHE 0.25, Gamma, Exposure）"
echo "- Hard-Negative Mining 加权采样（权重 2.5x）"
echo "- 第三阶段解冻（epoch 1000+，解冻最后 25% backbone）"
echo "- 目标：F1 从 0.8909 提升到 0.92+"

