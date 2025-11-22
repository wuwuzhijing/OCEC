import pyarrow.parquet as pq
import os
from PIL import Image
import io

def extract_images(parquet_path, output_dir="images_out"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading parquet: {parquet_path}")
    table = pq.read_table(parquet_path)
    data = table.to_pylist()

    for i, row in enumerate(data):
        img_bytes = row["Image_data"]["file"]   # JPEG bytes
        filename = row["Image_data"]["filename"]

        # 如果 parquet 里没有文件名，就用 index 生成一个
        if not filename:
            filename = f"image_{i}.jpg"

        # 目标路径
        save_path = os.path.join(output_dir, filename)

        # 解码 JPEG bytes
        try:
            img = Image.open(io.BytesIO(img_bytes))
            img.save(save_path)
        except Exception as e:
            print(f"[ERROR] Failed to decode index {i}, filename={filename}, err={e}")
            continue

        if i % 100 == 0:
            print(f"Saved: {save_path}")

    print(f"Done! All images saved in: {output_dir}")


if __name__ == "__main__":
    extract_images("/ssddisk/guochuang/ocec/data/dataset_687.parquet")