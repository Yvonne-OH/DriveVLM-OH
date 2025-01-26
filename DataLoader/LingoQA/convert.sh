#!/bin/bash

# 定义Python解释器路径
PYTHON_PATH=/usr/bin/python3

# Python脚本路径
SCRIPT_PATH="LingoQA2Json.py"

# 输入文件和输出文件
INPUT_PATH="/media/workstation/6D3563AC52DC77EA/Data/LingoQA/train/Scene_train.parquet"
OUTPUT_FILE="/media/workstation/6D3563AC52DC77EA/Data/LingoQA/train/scene_train_data.json"

# 运行Python脚本，并传递输入输出参数
$PYTHON_PATH $SCRIPT_PATH $INPUT_PATH $OUTPUT_FILE