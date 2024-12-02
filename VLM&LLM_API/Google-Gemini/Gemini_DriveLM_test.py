import json
import os
import time
from tqdm import tqdm

import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageColor
import google.generativeai as genai
from Util.util import check_and_fix_json

# Suppress logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"  # 0: INFO, 1: WARNING, 2: ERROR, 3: FATAL



# 生成配置
Responses_config = genai.types.GenerationConfig(
    candidate_count=1,
    stop_sequences=["x"],
    max_output_tokens=8192,  # 增加 token 数，确保完整返回边界框信息
    temperature=1.0,
)

def Model_initialize(Api_key, Model_name):
    """
    初始化模型
    """
    genai.configure(api_key=Api_key)
    model = genai.GenerativeModel(Model_name)
    return model

if __name__ == '__main__':
    # 加载数据
    Json_path = "/media/oh/0E4A12890E4A1289/DriveLM/data/QA_dataset_nus/test_eval.json"
    Image_path = "/media/oh/0E4A12890E4A1289/DriveLM/data/"
    Save_path = "/media/oh/0E4A12890E4A1289/DriveLM/data/QA_dataset_nus/Multi_choice_perception_test_Gemini.json"
    Result_path = "/media/oh/0E4A12890E4A1289/DriveLM/data/QA_dataset_nus/test_result_Gemini.json"

    # 配置参数
    Api_key = "AIzaSyCQSNKK5sH4yN87JQFnyEVQVhsOcam8VII"
    Model_name = "gemini-exp-1121"

    # 初始化模型
    model = Model_initialize(Api_key, Model_name)

    root = Save_path
    Sample_num = 20

    # 定义变量
    files_list = []
    response_list = []
    output_data = []  # 存储修改后的数据

    # 打开并加载 JSON 文件
    with open(root, 'r') as f:
        Gemini_Json_file = json.load(f)

    # 遍历样本，限制为 Sample_num

    for i, sample in enumerate(tqdm(Gemini_Json_file, desc="Generate responses: ")):

        time.sleep(5)  # 限制请求速率

        if i >= Sample_num:  # 如果处理数量达到限制则停止
            break

        # 确保每个样本的 image 包含 6 张图片
        if len(sample.get('image', [])) != 6:
            print(f"Warning: Sample {i} does not have exactly 6 images. Skipping.")
            continue

        # 加载图片并确保无异常
        try:
            files = [Image.open(image) for image in sample['image']]
        except Exception as e:
            print(f"Error opening files for sample {i}: {e}")
            continue

        # 提取对话内容
        input_1 = sample['conversations'][0]['value'] if len(sample.get('conversations', [])) > 0 else ""

        # 构建对话格式
        input_2 = (
        """The six images displayed above were captured by a vehicle's cameras positioned at the following locations (The order of the 6 photos is as follows): 
            
            {CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT}.

            In the format <object_id,camera_name,x_coord,y_coord>:
            
            object_id represents the unique identifier for the object.
            camera_name specifies the camera that captured the object.
            x_coord and y_coord represent the x and y coordinates of the object's center, expressed as percentages of the image dimensions.
            Coordinates are specified as percentages, adhering to the following conventions:
            
            The origin (0, 0) is located at the top-left corner of the image.
            The bottom-right corner of the image corresponds to (1000, 1000).
            The x-axis extends horizontally, increasing from left (0) to right (1000).
            The y-axis extends vertically, increasing from top (0) to bottom (1000).
                
        
        -------------------------------------------------------------------------------
        Using these images and the information provided, answer the following question:
        For multiple-choice questions, provide a single-letter answer: <A>, <B>, <C>, or <D>  (answer wrap in < >)
        Let's think step by step!
        **Question**:
        {input_1}"""
            f"{input_1}"
        )

        # 构建 Prompt
        Prompt = [
            "input image in the order {CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT}",
            files[0],
            files[1],
            files[2],
            files[3],
            files[4],
            files[5],
            f"input 2: {repr(input_2)}",
        ]

        # 调用生成模型
        for attempt in range(3):
            try:
                response = model.generate_content(
                    Prompt, generation_config=Responses_config
                )
                break
            except Exception as e:
                print(f"Error generating response for sample {i}, attempt {attempt + 1}: {e}")
                if attempt < 2:
                    time.sleep(30)
                else:
                    response = "Error: Could not generate response."

        # 在 conversations 中添加 "Model_Output"
        try:
            modified_sample = sample.copy()  # 复制原始样本
            modified_sample["conversations"].append({
                "from": "Model_Output",
                "value": response.text
            })

            # 将修改后的样本添加到新数据列表
            output_data.append(modified_sample)
        except Exception as e:
            print(f"Error adding response to sample {i}: {e}")

    # 写入新的 JSON 文件
    output_json_path = Result_path

    try:
        with open(output_json_path, "w") as outfile:
            json.dump(output_data, outfile, indent=4)
        print(f"Updated data has been written to {output_json_path}")
    except Exception as e:
        print(f"Error writing to JSON file: {e}")






