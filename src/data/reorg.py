import pandas as pd
import os

# 1. 读取 Parquet 文件并统计行数
parquet_path = "datasets/DAPO-Math-17k/data/dapo-math-17k.parquet"
df = pd.read_parquet(parquet_path)
num_rows = len(df)
print(f"Total rows in {parquet_path}: {num_rows}")

# 确保输出目录存在
output_dir = "datasets/DAPO-Math-17k/data"
os.makedirs(output_dir, exist_ok=True)

# 2. 保存前 100 行为 JSONL
first_100_path = os.path.join(output_dir, "dapo-math-100.jsonl")
df.head(100).to_json(first_100_path, orient="records", lines=True)
print(f"Saved first 100 rows to {first_100_path}")

# 3. 保存全部数据为 JSONL
full_jsonl_path = os.path.join(output_dir, "dapo-math-17k.jsonl")
df.to_json(full_jsonl_path, orient="records", lines=True)
print(f"Saved full dataset to {full_jsonl_path}")