#!/bin/bash

# --- 请在这里配置你的参数 ---
# 存放所有原始文件夹的父目录
SOURCE_DIR="/10/cvz/guochuang/dataset/Classification/fatigue/cropped"

# 合并后新文件夹的前缀
NEW_FOLDER_PREFIX="merged_"

# 每多少个原始文件夹合并为一个新文件夹
FOLDERS_PER_GROUP=25
# --------------------------

# 确保源目录存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo "错误：源目录 '$SOURCE_DIR' 不存在或不是一个目录。"
    exit 1
fi

# 进入源目录
cd "$SOURCE_DIR" || exit

# 获取所有原始文件夹，并按名称排序
# 使用 mapfile 可以正确处理包含空格的文件名（虽然你的文件名看起来没有空格）
mapfile -t ALL_FOLDERS < <(find . -maxdepth 1 -type d ! -name '.' ! -name '.*' | sort)

TOTAL_ORIGINAL_FOLDERS=${#ALL_FOLDERS[@]}

if [ "$TOTAL_ORIGINAL_FOLDERS" -eq 0 ]; then
    echo "在源目录中没有找到任何文件夹。"
    exit 0
fi

echo "开始处理..."
echo "找到 $TOTAL_ORIGINAL_FOLDERS 个原始文件夹，将合并为 $(( (TOTAL_ORIGINAL_FOLDERS + FOLDERS_PER_GROUP - 1) / FOLDERS_PER_GROUP )) 个新文件夹。"

# 遍历所有原始文件夹
for ((i=0; i<TOTAL_ORIGINAL_FOLDERS; i++)); do
    FOLDER_PATH=${ALL_FOLDERS[$i]}
    FOLDER_NAME=$(basename "$FOLDER_PATH")

    # 计算当前文件夹应该归属的目标组号
    GROUP_NUMBER=$((i / FOLDERS_PER_GROUP))
    
    # 构建新的目标文件夹路径
    TARGET_FOLDER="${NEW_FOLDER_PREFIX}${GROUP_NUMBER}"
    
    # 遍历 '0' 和 '1' 子文件夹
    for SUBFOLDER in 0 1; do
        SOURCE_SUBFOLDER_PATH="${FOLDER_PATH}/${SUBFOLDER}"
        
        # 检查源子文件夹是否存在
        if [ -d "$SOURCE_SUBFOLDER_PATH" ]; then
            TARGET_SUBFOLDER_PATH="/10/cvz/guochuang/dataset/Classification/fatigue/cropped_merge/${TARGET_FOLDER}/${SUBFOLDER}"
            echo $TARGET_SUBFOLDER_PATH
            # 如果目标子文件夹不存在，则创建
            mkdir -p "$TARGET_SUBFOLDER_PATH"
            
            # 移动源子文件夹到目标位置
            # 使用 mv -n 避免覆盖已存在的文件/目录
            echo "  正在移动 '$SOURCE_SUBFOLDER_PATH' 到 '$TARGET_SUBFOLDER_PATH'..."
            cp -n "$SOURCE_SUBFOLDER_PATH"/* "$TARGET_SUBFOLDER_PATH/"
        fi
    done

    # （可选）删除空的原始文件夹
    if [ -d "$FOLDER_PATH" ] && [ -z "$(ls -A "$FOLDER_PATH")" ]; then
        echo "  删除空文件夹 '$FOLDER_PATH'"
        rmdir "$FOLDER_PATH"
    fi

done

echo -e "\n处理完成！"