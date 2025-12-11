import re
import pandas as pd
from pathlib import Path

OCEC_DATA_ROOT='/ssddisk/guochuang/ocec/'
OCEC_CODE_ROOT='/103/guochuang/Code/myOCEC/'

log_file = OCEC_CODE_ROOT + "logs/train/hq_data/train_finetune_progressive_20251205_223950.log"
LOG_FILE = Path(log_file)
if LOG_FILE.suffix and LOG_FILE.is_file():  
    # 是文件 → 提取文件名（不含后缀或含后缀都可）
    CSV_BASE_NAME = LOG_FILE.stem
    CSV_BASE_DIR = LOG_FILE.name
else:
    # 是目录 → 提取最后一级路径名
    CSV_BASE_NAME = LOG_FILE.name
    CSV_BASE_DIR = LOG_FILE.name

TRAIN_ANALYSYS_PATH= Path(OCEC_CODE_ROOT) / "runs/ocec_analysis" / CSV_BASE_NAME
epochs = []
train_f1 = []
val_f1 = []
recall = []
precision = []
train_loss = []
val_loss = []

pattern = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+train loss\s+([0-9.]+).*?f1\s+([0-9.]+)\s+\|\s+val loss\s+([0-9.]+).*?f1\s+([0-9.]+)"
)

with open(log_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            e, tl, tf, vl, vf = match.groups()
            epochs.append(int(e))
            train_loss.append(float(tl))
            train_f1.append(float(tf))
            val_loss.append(float(vl))
            val_f1.append(float(vf))

df = pd.DataFrame({
    "epoch": epochs,
    "train_loss": train_loss,
    "val_loss": val_loss,
    "train_f1": train_f1,
    "val_f1": val_f1,
})

print(df.tail(100))

TRAIN_ANALYSYS_PATH.mkdir(parents=True, exist_ok=True)

df.to_csv(TRAIN_ANALYSYS_PATH / "training_summary.csv", index=False)
print("\nSaved as training_summary.csv ✔")
