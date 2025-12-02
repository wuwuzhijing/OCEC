/10/cv/gupengli/datasets/DSM/DSM_dataset_v2_eye/eyes_compact/squint/

## 缩进错误排查命令
python -m py_compile myOCEC/03_wholebody34_data_extractor.py 2>&1 | head -30

## hq数据集拷贝
```Bash
find /10/cv/gupengli/datasets/DSM/DSM_dataset_v2_eye/eyes_compact/ -type f |  pv -l -s $(find /10/cv/gupengli/datasets/DSM/DSM_dataset_v2_eye/eyes_compact/ -type f | wc -l) |xargs -P 16 -I{} rsync -a {} /103/guochuang/Code/myOCEC/data/hq

find /103/guochuang/Code/myOCEC/data/cropped/100000* -type f |  pv -l -s $(find /103/guochuang/Code/myOCEC/data/cropped/100000* -type f | wc -l) |xargs -P 16 -I{} rsync -a {} /103/guochuang/Code/myOCEC/data/hq

find /103/guochuang/Code/myOCEC/data/cropped/0000000* -type f |  pv -l -s $(find /103/guochuang/Code/myOCEC/data/cropped/0000000* -type f | wc -l) |xargs -P 16 -I{} rsync -a {} /ssddisk/guochuang/ocec/public/

cd /10/cv/gupengli/datasets/DSM/DSM_dataset_v2_eye/
tar cf - eyes_compact | pv -s $(du -sb eyes_compact | cut -f1) | tar xf - -C /103/guochuang/Code/myOCEC/data/hq/
```
## 修改记录

将/ssddisk/guochuang/ocec/public/cropped/list的csv文件中的data/cropped/目录全部改为/ssddisk/guochuang/ocec/public/cropped/

将/ssddisk/guochuang/ocec/hq下面的jpg文件以data/cropped/000001000/00001522_2_cd0c4e4750fe.png,0这种方式写到csv。
注意，
1. 这些文件的最后一个数字1表示闭眼,0表示睁眼。为了和其它数据集保持一致，写的时候要反转一下。如果文件名后面是1，csv中要写0；如果文件名后面是0，csv中要写1。
2. 写绝对路径。