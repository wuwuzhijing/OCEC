#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_DIR="/103/guochuang/Code/myOCEC/logs/ocec_classifier"
mkdir -p "$LOG_DIR"

LOG_PATH="${LOG_DIR}/ocec_classifier_$(date +%Y%m%d_%H%M%S).log"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

pushd ${SCRIPT_DIR}/..

nohup python run_ocec_on_cropped.py --images_root /10/cvz/guochuang/dataset/Classification/fatigue/cropped > ${LOG_PATH} 2>&1 &

popd