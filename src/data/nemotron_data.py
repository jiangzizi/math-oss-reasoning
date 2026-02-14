from datasets import load_dataset
import shutil
import os

# 目标保存路径
save_path = "/home/disk1/jiangdazhi/research/math-oss-reasoning/datasets/Nemotron-sft/science.jsonl"
# 加载数据集（指定具体文件）
ds = load_dataset(
    "nvidia/Llama-Nemotron-Post-Training-Dataset",
    data_files={"train": "SFT/science/science.jsonl"},
    split="train"
)
# 先保存到临时目录，再移动到指定路径
temp_path = ds.to_json(save_path + ".tmp")
shutil.move(temp_path, save_path)
# 验证文件
if os.path.exists(save_path):
    print(f"文件下载成功，路径：{save_path}，文件大小：{os.path.getsize(save_path)/1024/1024:.2f} MB")
else:
    print("文件下载失败")