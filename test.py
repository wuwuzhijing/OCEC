import pyarrow.parquet as pq

p = "/ssddisk/guochuang/ocec/data/dataset_687.parquet"   # 改成你的路径
table = pq.read_table(p)
print("Schema:")
print(table.schema)

print("\nFirst record example:")
print(table.to_pylist()[0])
