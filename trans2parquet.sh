# Option 1: Use default (automatically loads all annotation_*.csv from data/cropped/list/)
# python 04_dataset_convert_to_parquet.py \
# --output data/dataset.parquet \
# --train-ratio 0.8 \
# --seed 42 \
# --embed-images

# Option 2: Specify a directory (loads all annotation_*.csv files from that directory)
# Split into multiple parquet files (50000 rows per file)
python 04_dataset_convert_to_parquet.py \
--annotation /ssddisk/guochuang/ocec/list_hq \
--output /ssddisk/guochuang/ocec/parquet_hq/dataset_hq.parquet \
--train-ratio 0.8 \
--seed 42 \
--max-rows-per-file 50000 \
--embed-images

# Option 3: Specify multiple CSV files
# python 04_dataset_convert_to_parquet.py \
# --annotation data/cropped/list/annotation_0001.csv data/cropped/list/annotation_0002.csv \
# --output data/dataset.parquet \
# --train-ratio 0.8 \
# --seed 42 \
# --embed-images