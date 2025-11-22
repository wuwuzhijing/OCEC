## 03_wholebody34_data_extractor
```bash
#单线程处理（默认）
python 03_wholebody34_data_extractor.py -ea

# 多线程处理（使用4个worker）
python 03_wholebody34_data_extractor.py -ea -j 4

# 仅处理已有数据集
python 03_wholebody34_data_extractor.py -ea --process-dataset-only -j 4

# 仅处理视频文件
python 03_wholebody34_data_extractor.py -ea --process-video-only

# 处理数据集（多线程）+ 指定目录
python 03_wholebody34_data_extractor.py -ea -i /path/to/images --process-dataset-only -j 8
```