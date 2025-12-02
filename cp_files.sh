#!/bin/bash
set -e

SRC="/10/cv/gupengli/datasets/DSM/DSM_dataset_v2_eye/eyes_compact"
DST="/103/guochuang/Code/myOCEC/data/hq/eyes_compact"
SNAP="/tmp/eyes.snapshot"

echo "[1] Incremental fast copy..., src: $SRC, dst: $DST"
cd $(dirname "$SRC")
tar -g "$SNAP" -cf - $(basename "$SRC") | pv | tar xf - -C $(dirname "$DST")

echo "[2] Generating source md5..."
(cd "$SRC" && find . -type f -print0 | sort -z | xargs -0 md5sum) > /tmp/src.md5

echo "[3] Generating dest md5..."
(cd "$DST" && find . -type f -print0 | sort -z | xargs -0 md5sum) > /tmp/dst.md5

echo "[4] Comparing..."
diff /tmp/src.md5 /tmp/dst.md5 && echo "MD5 OK ✔ All files match."