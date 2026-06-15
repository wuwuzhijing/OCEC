#!/bin/bash
# 恢复训练并渐进式解冻所有 backbone 的脚本
# 
# 用途：从之前的训练 checkpoint 恢复，采用渐进式解冻策略
#       先解冻50% backbone，等F1恢复到阈值后再解冻所有
# 
# ========== 配置区域 ==========
# 要恢复的 checkpoint 路径（最后一个 epoch 的 checkpoint）
RESUME_CHECKPOINT="runs/ocec_hq_finetune_progressive_v5.0_stage3/v20/ocec_epoch_1116.pt"

# 数据集路径（应该和之前训练时一样）
DATA_ROOT_B="/ssddisk/guochuang/ocec/parquet_hq_v3"

# 输出目录（可以继续使用原来的，或者新建一个）
OUTPUT_DIR="runs/ocec_hq_finetune_progressive_v5.0_stage3"

# 继续训练的配置
TOTAL_EPOCHS=2000             # 总训练 epoch 数（会从 checkpoint 的 epoch + 1 继续训练）
UNFREEZE_LR=1.5e-5            # 解冻所有 backbone 后的学习率（建议很小，如 stage2_lr 的 25%）
UNFREEZE_F1_THRESHOLD=0.89    # 解冻所有backbone的F1阈值
# =============================

# 检查 checkpoint 是否存在
if [ ! -f "$RESUME_CHECKPOINT" ]; then
    echo "错误: Checkpoint 不存在: $RESUME_CHECKPOINT"
    exit 1
fi

echo "=========================================="
echo "恢复训练并渐进式解冻所有 backbone"
echo "=========================================="
echo "Checkpoint: ${RESUME_CHECKPOINT}"
echo "数据集: ${DATA_ROOT_B}"
echo "输出目录: ${OUTPUT_DIR}"
echo "总训练 epoch: ${TOTAL_EPOCHS} (会从 checkpoint 的 epoch + 1 继续)"
echo "解冻后学习率: ${UNFREEZE_LR}"
echo "F1阈值: ${UNFREEZE_F1_THRESHOLD} (达到此值后自动解冻所有backbone)"
echo ""
echo "渐进式解冻策略："
echo "  1. 恢复训练时先解冻50% backbone"
echo "  2. 每个epoch验证后检查F1值"
echo "  3. 当F1 >= ${UNFREEZE_F1_THRESHOLD} 时，自动解冻所有backbone"
echo "=========================================="

# 训练命令（配置与原始训练保持一致）
export OCEC_ARGS="train \
    --data_root ${DATA_ROOT_B} \
    --output_dir ${OUTPUT_DIR} \
    --resume ${RESUME_CHECKPOINT} \
    --epochs ${TOTAL_EPOCHS} \
    --batch_size 1024 \
    --num_workers 8 \
    --train_ratio 0.8 \
    --val_ratio 0.2 \
    --image_size 32x64 \
    --base_channels 32 \
    --num_blocks 4 \
    --arch_variant baseline \
    --margin_method cosface \
    --lr 0.0002 \
    --stage2_lr 6e-05 \
    --stage3_lr ${UNFREEZE_LR} \
    --weight_decay 1e-4 \
    --seed 42 \
    --device auto \
    --warmup_epochs 3 \
    --use_amp \
    --pseudo_update_start_epoch 200 \
    --neg_class_weight 1.3 \
    --enable_hard_negative_mining \
    --hard_neg_min_prob 0.7 \
    --hard_neg_weight 2.0 \
    --unfreeze_backbone_ratio 0.5 \
    --unfreeze_all_after_resume \
    --unfreeze_all_f1_threshold ${UNFREEZE_F1_THRESHOLD}"

# 激活 conda 环境并运行训练
LOG_FILE="logs/train/hq_data/train_resume_unfreeze_all_$(date +%Y%m%d_%H%M%S).log"
nohup bash -c "source $(conda info --base)/etc/profile.d/conda.sh && conda activate deploy && exec -a resume5 python - << 'EOF'
import os, shlex
from ocec.__main__ import main
args = shlex.split(os.environ['OCEC_ARGS'])
main(args)
EOF" > ${LOG_FILE} 2>&1 &

echo "恢复训练已启动（渐进式解冻策略）"
echo "日志文件: ${LOG_FILE}"
echo "TensorBoard: tensorboard --logdir ${OUTPUT_DIR}"
echo ""
echo "注意："
echo "- 训练会从 checkpoint 的 epoch + 1 开始"
echo "- 先解冻50% backbone，等F1恢复到 ${UNFREEZE_F1_THRESHOLD} 后再解冻所有"
echo "- 使用较小的学习率 (${UNFREEZE_LR}) 进行 fine-tuning"
echo "- 查看日志确认何时触发完全解冻"