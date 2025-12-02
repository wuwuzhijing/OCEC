#!/bin/bash

SRC1="/10/cv/gupengli/datasets/DSM/DSM_dataset_v2_eye/normalface_all/squint"
SRC2="/10/cv/gupengli/datasets/DSM/DSM_dataset_v2_eye/normalface_all/staring"
DST="/10/cvz/guochuang/dataset/Classification/fatigue"

# 目标：DST/open、DST/closed
mkdir -p "$DST/open" "$DST/closed"

# 分片大小
CHUNK=5000

# 为 open/closed 建立计数器
open_count=0
closed_count=0

open_idx=1
closed_idx=1

open_dir=$(printf "%s/open/open_%07d" "$DST" "$open_idx")
closed_dir=$(printf "%s/closed/closed_%07d" "$DST" "$closed_idx")
mkdir -p "$open_dir" "$closed_dir"

# 合并两个源目录的全部图片（find 输出绝对路径）
find "$SRC1" "$SRC2" -type f -name "*.jpg" | parallel -j 32 '
    file={}
    base=$(basename "$file")

    if [[ "$base" == *_0.jpg ]]; then
        echo "OPEN $file"
    else
        echo "CLOSED $file"
    fi
' | while read -r tag path; do

    if [[ "$tag" == "OPEN" ]]; then
        if (( open_count >= CHUNK )); then
            open_count=0
            ((open_idx++))
            open_dir=$(printf "%s/open/open_%07d" "$DST" "$open_idx")
            mkdir -p "$open_dir"
        fi
        rsync -a "$path" "$open_dir/"
        ((open_count++))

    else
        if (( closed_count >= CHUNK )); then
            closed_count=0
            ((closed_idx++))
            closed_dir=$(printf "%s/closed/closed_%07d" "$DST" "$closed_idx")
            mkdir -p "$closed_dir"
        fi
        rsync -a "$path" "$closed_dir/"
        ((closed_count++))
    fi

done

echo "全部处理完成！"彻底