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

def Model_initialize(Api_key, Model_name, System_instruction=""):
    """
    初始化模型
    """
    genai.configure(api_key=Api_key)
    model = genai.GenerativeModel(Model_name,system_instruction=System_instruction)
    return model

def Gemini_VQA_Nusence_COT_benchmark(json_path, image_path, result_path, model_name, task_type , max_request_rate=1 ,sample_num=20):
    """
    Process QA dataset, analyze images, and generate responses using a specified model.

    Args:
        json_path (str): Path to the input JSON file.
        image_path (str): Path to the directory containing images.
        save_path (str): Path to the QA dataset JSON file for multi-choice behavior.
        result_path (str): Path to save the result JSON file.
        model_name (str): Name of the model.
        sample_num (int): Number of samples to process.
    """
    # Load API key from environment variable
    api_key = os.getenv("API_KEY")

    if not api_key:
        print("Error: API_KEY environment variable is not set.")
        return

    if task_type == "behavior":
        system_instruction =("Now you will answer the questions as a driver. "
                             "Please answer the questions in brief"
                             "Note: If you encounter a multiple choice question, please provide the final answer in the following format:</ans> answer</ans>, "
                             "otherwise the answer will not be counted!!!"
                             )
    elif task_type == "perception":
        system_instruction = ("Now you will answer the questions as a driver. "
                             "Please answer the questions in brief"
                             "Note: If you encounter a multiple choice question, please provide the final answer in the following format:</ans> answer</ans>, "
                             "otherwise the answer will not be counted!!!"
                             )
    elif task_type == "planning":
        system_instruction = ("Now you will answer the questions as a driver")
    elif task_type == "prediction":
        system_instruction = ("Now you will answer the questions as a driver")
    else:
        print("Error: Invalid task type.")
        return

    # Initialize the model
    model = Model_initialize(api_key, model_name,system_instruction)

    # Load JSON file
    try:
        with open(json_path, 'r') as f:
            qa_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return

    output_data = []  # Store modified data

    # Iterate through samples with a limit of `sample_num`
    for i, sample in enumerate(tqdm(qa_data, desc="Generating responses: ")):

        time.sleep(int(60/max_request_rate))  # Limit request rate


        if i >= sample_num:
            break

        # Ensure each sample contains 6 images
        if len(sample.get('image', [])) != 6:
            print(f"Warning: Sample {i} does not have exactly 6 images. Skipping.")
            continue

        try:
            files = [Image.open(image) for image in sample['image']]
        except Exception as e:
            print(f"Error opening files for sample {i}: {e}")
            continue

        # Extract question
        Q = sample['conversations'][0]['value'] if len(sample.get('conversations', [])) > 0 else ""

        # Prepare input
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
            The y-axis extends vertically, increasing from top (0) to bottom (1000)."""
        )

        # Start chat session
        if task_type == "behavior":
            try:
                chat = model.start_chat(
                    history=[
                        {"role": "user", "parts": "Now you will answer the questions as a driver. Note: If you encounter a multiple choice question, "
                                                    "please provide the final answer in the following format:</ans> answer</ans>, "
                                                    "otherwise the answer will not be counted"},
                        {"role": "model", "parts": "Great to meet you. What would you like to know?"},
                    ]
                )

                # Step 1: Provide input content
                user_input_1 = [
                    "input image in the order {CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT}",
                    files[0], files[1], files[2], files[3], files[4], files[5],
                    f"input 2: {repr(input_2)}",

                    "Can you help me analyze images and answer questions step by step?"
                ]
                response = chat.send_message(user_input_1)
                print(f"Model: {response.text}")

                # Step 2: Provide detailed context
                user_input_2 = (
                    """Analyze the image and assess the following:
                    What is the current state of the ego vehicle (position, speed, or behavior)?
                    What are the states of the surrounding vehicles and pedestrians, and how might they influence the ego vehicle's actions?
                    Describe the surrounding traffic environment, including road conditions, traffic signs, and signals."""
                )
                response = chat.send_message(user_input_2)
                print(f"Model: {response.text}")

                # Step 3: Derive final answer
                user_input_4 = f"The question is: {Q} Now, using this information, can you provide the final answer in the following format: </ans>answer</ans>"
                final_response = chat.send_message(user_input_4)
                print(f"Model: {final_response.text}")

            except Exception as e:
                print(f"Error during conversation: {e}")
                continue

        elif task_type == "perception":
            try:
                chat = model.start_chat(
                    history=[
                        {"role": "user",
                         "parts": "Now you will answer the questions as a driver. Note: If you encounter a multiple choice question, "
                                  "please provide the final answer in the following format:</ans> answer</ans>, "
                                  "otherwise the answer will not be counted"},
                        {"role": "model", "parts": "Great to meet you. What would you like to know?"},
                    ]
                )

                # Step 1: Provide input content
                user_input_1 = [
                    "input image in the order {CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT}",
                    files[0], files[1], files[2], files[3], files[4], files[5],
                    f"input 2: {repr(input_2)}",
                    f"The question is: {Q} Now, using this information, can you provide the final answer in the following format: </ans>answer</ans>"

                ]

                final_response = chat.send_message(user_input_1)
                print(f"Model: {final_response.text}")

            except Exception as e:
                print(f"Error during conversation: {e}")
                continue



        # Add "Model_Output" to conversations
        try:
            modified_sample = sample.copy()
            modified_sample["conversations"].append({
                "from": "Model_Output",
                "value": final_response.text
            })
            output_data.append(modified_sample)
        except Exception as e:
            print(f"Error adding response to sample {i}: {e}")

    # Write to new JSON file
    try:
        result_path = result_path + task_type + "_test_result_Gemini.json"
        with open(result_path, "w") as outfile:
            json.dump(output_data, outfile, indent=4)
        print(f"Updated data has been written to {result_path}")
    except Exception as e:
        print(f"Error writing to JSON file: {e}")

if __name__ == '__main__':
    # 加载数据
    Json_path = "/media/oh/0E4A12890E4A1289/DriveLM/data/QA_dataset_nus/test_eval.json"
    Image_path = "/media/oh/0E4A12890E4A1289/DriveLM/data/"
    Save_path = "/media/oh/0E4A12890E4A1289/DriveLM/data/QA_dataset_nus/Multi_choice_behavior_test_Gemini.json"
    Result_path = "/media/oh/0E4A12890E4A1289/DriveLM/data/QA_dataset_nus/"

    # 配置参数
    Api_key = "AIzaSyCQSNKK5sH4yN87JQFnyEVQVhsOcam8VII"
    Model_name = "gemini-exp-1121"
    Model_name = "gemini-1.5-pro"


    Gemini_VQA_Nusence_COT_benchmark(Save_path, Image_path, Result_path, Model_name, "behavior", 10, 50)

    # # 初始化模型
    # model = Model_initialize(Api_key, Model_name)
    #
    # root = Save_path
    # Sample_num = 20
    #
    # # 定义变量
    # files_list = []
    # response_list = []
    # output_data = []  # 存储修改后的数据
    #
    # # 打开并加载 JSON 文件
    # with open(root, 'r') as f:
    #     Gemini_Json_file = json.load(f)
    #
    # # 遍历样本，限制为 Sample_num
    #
    # for i, sample in enumerate(tqdm(Gemini_Json_file, desc="Generate responses: ")):
    #
    #     time.sleep(5)  # 限制请求速率
    #
    #     if i >= Sample_num:  # 如果处理数量达到限制则停止
    #         break
    #
    #     # 确保每个样本的 image 包含 6 张图片
    #     if len(sample.get('image', [])) != 6:
    #         print(f"Warning: Sample {i} does not have exactly 6 images. Skipping.")
    #         continue
    #
    #     # 加载图片并确保无异常
    #     try:
    #         files = [Image.open(image) for image in sample['image']]
    #     except Exception as e:
    #         print(f"Error opening files for sample {i}: {e}")
    #         continue
    #
    #     Q = sample['conversations'][0]['value'] if len(sample.get('conversations', [])) > 0 else ""
    #
    #     input_2 = (
    #         """The six images displayed above were captured by a vehicle's cameras positioned at the following locations (The order of the 6 photos is as follows):
    #
    #             {CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT}.
    #
    #             In the format <object_id,camera_name,x_coord,y_coord>:
    #
    #         object_id represents the unique identifier for the object.
    #         camera_name specifies the camera that captured the object.
    #         x_coord and y_coord represent the x and y coordinates of the object's center, expressed as percentages of the image dimensions.
    #         Coordinates are specified as percentages, adhering to the following conventions:
    #
    #         The origin (0, 0) is located at the top-left corner of the image.
    #         The bottom-right corner of the image corresponds to (1000, 1000).
    #         The x-axis extends horizontally, increasing from left (0) to right (1000).
    #         The y-axis extends vertically, increasing from top (0) to bottom (1000)."""
    #     )
    #
    #
    #
    #     # 初始化对话历史
    #     chat = model.start_chat(
    #         history=[
    #             {"role": "user", "parts": "Now you will answer the questions as a driver. Note: If you encounter a multiple choice question, "
    #                                       "please provide the final answer in the following format:</ans> answer</ans>, "
    #                                       "otherwise the answer will not be counted"},
    #             {"role": "model", "parts": "Great to meet you. What would you like to know?"},
    #         ]
    #     )
    #
    #     # 模拟多轮次对话
    #     try:
    #         # Step 1: 提供输入内容
    #         user_input_1 = [
    #         "input image in the order {CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT}",
    #         files[0],
    #         files[1],
    #         files[2],
    #         files[3],
    #         files[4],
    #         files[5],
    #         f"input 2: {repr(input_2)}",
    #         "Can you help me analyze images and answer questions step by step?"
    #
    #     ]
    #         response= chat.send_message(user_input_1)
    #
    #         print(f"Model: {response.text}")
    #
    #         # Step 2: 提供详细上下文
    #         user_input_2 = (
    #             """Analyze the image and assess the following:
    #                 What is the current state of the ego vehicle (position, speed, or behavior)?
    #                 What are the states of the surrounding vehicles and pedestrians, and how might they influence the ego vehicle's actions?
    #                 Describe the surrounding traffic environment, including road conditions, traffic signs, and signals. How might these factors impact the ego vehicle's navigation or decision-making?
    #                 """
    #             )
    #         response = chat.send_message(user_input_2)
    #         print(f"Model: {response.text}")
    #
    #         # Step 3: 逐步推导答案
    #         user_input_3 = "What is one of the objectives that will have the greatest impact on the ego vehicle's decision-making?"
    #         response = chat.send_message(user_input_3)
    #         print(f"Model: {response.text}")
    #
    #         # Step 4: 最终推导结果
    #         user_input_4 = "The question is: "+Q+"Now, using this information, can you provide the final answer in the following format: </ans>answer</ans>"
    #         final_response = chat.send_message(user_input_4)
    #         print(f"Model: {final_response.text}")
    #
    #     except Exception as e:
    #         print(f"Error during conversation: {e}")
    #
    #     # 在 conversations 中添加 "Model_Output"
    #     try:
    #         modified_sample = sample.copy()  # 复制原始样本
    #         modified_sample["conversations"].append({
    #             "from": "Model_Output",
    #             "value": final_response.text
    #         })
    #
    #         # 将修改后的样本添加到新数据列表
    #         output_data.append(modified_sample)
    #     except Exception as e:
    #         print(f"Error adding response to sample {i}: {e}")
    #
    # # 写入新的 JSON 文件
    # output_json_path = Result_path
    #
    # try:
    #     with open(output_json_path, "w") as outfile:
    #         json.dump(output_data, outfile, indent=4)
    #     print(f"Updated data has been written to {output_json_path}")
    # except Exception as e:
    #     print(f"Error writing to JSON file: {e}")






