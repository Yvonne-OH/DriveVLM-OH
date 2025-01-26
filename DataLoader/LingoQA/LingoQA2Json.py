import argparse
import pandas as pd
import json
from tqdm import tqdm
import sys

def main(args=None):
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="Process Parquet file and output as JSON")
    parser.add_argument("input_path", type=str, help="Path to the input Parquet file")
    parser.add_argument("output_file", type=str, help="Path to the output JSON file")

    # 解析命令行参数
    args = parser.parse_args(args)

    # 任务名称从路径提取
    TASK = args.input_path.split("/")[-1].split(".")[0]

    # 读取Parquet文件
    df = pd.read_parquet(args.input_path, engine='pyarrow')

    # 打印基本信息
    print("_"*20, "Basic Information", "_"*20)
    print(df.columns)
    print("_"*60)
    print(df.shape)
    print("_"*60)
    print(df.head())
    print("_"*60)

    segment_data = {}

    # 进度条处理
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):
        segment_id = row['segment_id']
        if segment_id not in segment_data:
            segment_data[segment_id] = {
                "segment_id": segment_id,
                "image_id": [path.replace('images/train/',"") for path in row['images']],
                "questions": []
            }
        segment_data[segment_id]["questions"].append({
            "question": row['question'],
            "GT_answer": row['answer'],
            "Gpt_answer": ""
        })

    # 将数据保存为JSON
    with open(args.output_file, 'w') as json_file:
        json.dump(segment_data, json_file, indent=4)

    print(f"Data saved to {args.output_file}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        args = ["/media/workstation/6D3563AC52DC77EA/Data/LingoQA/train/Scene_train.parquet",
                "/media/workstation/6D3563AC52DC77EA/Data/LingoQA/train/Scene_train.json"]
        main(args)
    else:
        main()

