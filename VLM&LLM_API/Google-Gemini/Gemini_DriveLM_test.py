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
    max_output_tokens=2048,  # 增加 token 数，确保完整返回边界框信息
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
    Save_path = "/media/oh/0E4A12890E4A1289/DriveLM/data/QA_dataset_nus/test_Gemini.json"
    Result_path = "/media/oh/0E4A12890E4A1289/DriveLM/data/QA_dataset_nus/test_result_Gemini.json"

    # 配置参数
    Api_key = "AIzaSyCQSNKK5sH4yN87JQFnyEVQVhsOcam8VII"
    Model_name = "gemini-1.5-pro"

    prompt = (
        "Return bounding boxes for cars, and trees in the"
        " following format as a list. \n {'car_0' : [ymin, xmin, ymax,"
        " xmax], ...} \n If there are more than one instance of an object, add"
        " them to the dictionary as 'object_0', 'object_1', etc."
    )

    # 初始化模型
    model = Model_initialize(Api_key, Model_name)

    root = Save_path
    Sample_num = 50

    # 定义变量
    files_list = []
    response_list = []
    output_data = []  # 存储修改后的数据

    # 打开并加载 JSON 文件
    with open(root, 'r') as f:
        Gemini_Json_file = json.load(f)

    # 遍历样本，限制为 Sample_num

    for i, sample in enumerate(tqdm(Gemini_Json_file, desc="Generate responses: ")):
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
        """The six images above were captured by a car's cameras positioned as follows: front, left front, right front, rear, left rear, and right rear. Based on these images, answer the following questions. All coordinates are based on the image pixels. The relationships are defined as follows:
        
        - The absolute pixel coordinates are calculated as:
          - `abs_x1 = int(x1 / 1000 * width)`
          - `abs_y1 = int(y1 / 1000 * height)`
        
        - The image resolution is defined as:
          - `width = 1600`
          - `height = 900`
        
        Guidelines for the format:
        - **ID1**: Represents the object identifier, composed of:
          - A letter indicating the object type:
            - **C**: Car
            - **P**: Pedestrian
            - **V**: Truck/Van
            - **T**: Traffic Sign
            - **O**: Other
          - A numeric sequence (e.g., c1, p2, v3).
        - **CAM_ID**: Specifies the camera source from which the object is detected:
          - {CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT}.
        - **x, y**: The object's pixel coordinates within the image, with the origin located at the top-left corner.
        
        Answer format:
        1. For questions requiring a description, use the following format:
           "Firstly notice that <ID1, CAM_ID, x, y>. The object is a [type of object], so the ego vehicle should [action]."
           "Secondly notice that <ID2, CAM_ID, x, y>. The object is a [type of object], so the ego vehicle should [action]."
           "Thirdly notice that <ID3, CAM_ID, x, y>. The object is a [type of object], so the ego vehicle should [action]."
        
        2. For multiple-choice questions, answer with a single letter: A, B, C, or D.
        
        Example:
        "Firstly notice that <c3, CAM_FRONT, 1043, 82>. The object is a traffic sign, so the ego vehicle should keep going ahead at the same speed. Secondly notice that <c1, CAM_BACK, 1088, 497>. The object is turning left, so the ego vehicle should keep going ahead at the same speed. Thirdly notice that <c2, CAM_BACK, 864, 468>. The object is going ahead, so the ego vehicle should keep going ahead at the same speed."
        
        -------------------------------------------------------------------------------
        **Question**:
        {input_1}"""
            f"{input_1}"
        )

        # 构建 Prompt
        Prompt = [
            "input: ",
            files[0],
            files[1],
            files[2],
            files[3],
            files[4],
            files[5],
            f"input 2: {repr(input_2)}",
        ]

        # 调用生成模型
        try:
            response = model.generate_content(
                Prompt, generation_config=Responses_config
            )
        except Exception as e:
            print(f"Error generating response for sample {i}: {e}")
            response = "Error: Could not generate response."

        # 在 conversations 中添加 "Model_Output"
        modified_sample = sample.copy()  # 复制原始样本
        modified_sample["conversations"].append({
            "from": "Model_Output",
            "value": response.text
        })

        # 将修改后的样本添加到新数据列表
        output_data.append(modified_sample)

    # 写入新的 JSON 文件
    output_json_path = Result_path

    try:
        with open(output_json_path, "w") as outfile:
            json.dump(output_data, outfile, indent=4)
        print(f"Updated data has been written to {output_json_path}")
    except Exception as e:
        print(f"Error writing to JSON file: {e}")






