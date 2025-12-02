#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_DIR="logs/data_extract"
mkdir -p "$LOG_DIR"

LOG_PATH="${LOG_DIR}/data_extract_$(date +%Y%m%d_%H%M%S).log"

nohup python 03_wholebody34_data_extractor.py -i /10/cvz/guochuang/dataset/Classification/fatigue -ea -j 24 -ep cuda > ${LOG_PATH} 2>&1 &