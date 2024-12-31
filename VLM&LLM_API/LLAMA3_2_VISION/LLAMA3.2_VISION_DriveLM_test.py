import Multi_modal_Infer

import re
import json
import ast
import openai
from PIL import Image
from Util.util import parse_list_boxes_with_label, plot_bounding_boxes

from PIL import Image, ImageDraw, ImageFont

import base64
import re
import json
import ast
import openai
import time
from tqdm import tqdm
import os
from PIL import Image

# Suppress logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"  # 0: INFO, 1: WARNING, 2: ERROR, 3: FATAL


import torch

# 配置参数
model_name = '/media/workstation/6D3563AC52DC77EA/Model/meta-llama/Llama-3.2-11B-Vision-Instruct'
finetuning_path = None  # 如果有微调路径可以设置
image_path = "car.png"  # 替换为本地图片路径
prompt_text = "Describe this image in detail."
temperature = 0.7
top_p = 0.9

# 确保 CUDA 可用
if not torch.cuda.is_available():
    print("CUDA is not available. Please check your GPU configuration.")
    #return

    # 加载模型和处理器
try:
    model, processor = Multi_modal_Infer.load_model_and_processor(
        model_name=model_name,
        finetuning_path=finetuning_path,
        device="auto",  # 自动设备映射
        max_memory={0: "21GB", 1: "6GB"}  # GPU 显存限制
    )
    print("Model and processor loaded successfully.")
except Exception as e:
    print(f"Failed to load model and processor: {e}")
    #return

    # 加载并处理图像
try:
    image =  Multi_modal_Infer.process_image(image_path=image_path)
    print("Image loaded and processed successfully.")
except Exception as e:
    print(f"Failed to process image: {e}")
    #return



# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def Gpt4_VQA_Nusence_COT_benchmark(json_path, image_path, result_path, model_name, task_type , max_request_rate=1 ,sample_num=20):
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
    openai.api_key =  "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"

    # Write to JSON file after each response
    result_path = result_path + task_type + "_test_result_GPT4.json"

    # if not os.getenv("OPENAI_API_KEY"):
    #     print("Error: API_KEY environment variable is not set.")
    #     return

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


    # Load JSON file
    try:
        with open(json_path, 'r') as f:
            qa_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return

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
            files = [encode_image(image) for image in sample['image']]
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

        conversation_history = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Great to meet you. What would you like to know?"},
        ]

        # Start chat session
        if task_type == "behavior":

            try:
                response = openai.chat.completions.create(
                    model="gpt-4o",  # 选择模型，"gpt-4" 或 "gpt-3.5-turbo"
                    messages=conversation_history,
                    temperature=0.7,  # 控制回复的随机性
                )

                # 获取 GPT 的回复
                assistant_message = response.choices[0].message.content
                print(f"GPT: {assistant_message}")
                conversation_history.append({"role": "assistant", "content": assistant_message})

                # Path to your image
                image_path = "../media/traffic.png"

                # Getting the base64 string
                base64_image = encode_image(image_path)


                # Step 1: Provide input content
                user_input_1 = {
                        "role": "user",
                        "content": [
                            {"type": "text", "text" : f"Input images in the order: **CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT**"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{files[0]}"
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{files[1]}"
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{files[2]}"
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{files[3]}"
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{files[4]}"
                                },
                            },
                            {"type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{files[5]}"
                                },
                            },

                            {"type": "text",
                             "text": f"{input_2}"},
                            {"type": "text",
                             "text": "Can you help me analyze the images and answer questions step by step?"},
                        ],
                    }

                # 添加用户消息到对话历史
                conversation_history.append(user_input_1)
                response = openai.chat.completions.create(
                    model="gpt-4o",  # 选择模型，"gpt-4" 或 "gpt-3.5-turbo"
                    messages=conversation_history,
                    temperature=0.7,  # 控制回复的随机性
                )
                # 获取 GPT 的回复
                assistant_message = response.choices[0].message.content
                print("*"*50)
                print(f"GPT: {assistant_message}")
                conversation_history.append({"role": "assistant", "content": assistant_message})

                user_input_2 = {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": """Analyze the image and assess the following:
                    What is the current state of the ego vehicle (position, speed, or behavior)?
                    What are the states of the surrounding vehicles and pedestrians, and how might they influence the ego vehicle's actions?
                    Describe the surrounding traffic environment, including road conditions, traffic signs, and signals."""},
                    ],
                }

                # 添加用户消息到对话历史
                conversation_history.append(user_input_2)
                response = openai.chat.completions.create(
                    model="gpt-4o",  # 选择模型，"gpt-4" 或 "gpt-3.5-turbo"
                    messages=conversation_history,
                    temperature=0.7,  # 控制回复的随机性
                )
                # 获取 GPT 的回复
                assistant_message = response.choices[0].message.content
                print("*"*50)
                print(f"GPT: {assistant_message}")
                conversation_history.append({"role": "assistant", "content": assistant_message})

                user_input_3 = {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": f"The question is: {Q} provide the final answer in the following format: </ans>answer</ans>"},
                    ],
                }

                # 添加用户消息到对话历史
                conversation_history.append(user_input_3)
                response = openai.chat.completions.create(
                    model="gpt-4o",  # 选择模型，"gpt-4" 或 "gpt-3.5-turbo"
                    messages=conversation_history,
                    temperature=0.7,  # 控制回复的随机性
                )
                # 获取 GPT 的回复
                assistant_message = response.choices[0].message.content
                print("*"*50)
                print(f"GPT: {assistant_message}")
                conversation_history.append({"role": "assistant", "content": assistant_message})

                final_response = assistant_message
                print("*"*50)
                print(f"Model: {final_response}")

            except Exception as e:
                print(f"Error during conversation: {e}")
                continue


        # Add "Model_Output" to conversations
        try:
            modified_sample = sample.copy()
            modified_sample["conversations"].append({
                "from": "Model_Output",
                "value": final_response
            })


            with open(result_path, "a") as outfile:
                json.dump(modified_sample, outfile, indent=4)
                outfile.write("\n")  # Add a newline for better readability
            print(f"Sample {i} has been written to {result_path}")

        except Exception as e:
            print(f"Error adding response or writing sample {i}: {e}")

if __name__ == '__main__':
    # 加载数据
    Json_path = "/media/oh/0E4A12890E4A1289/DriveLM/data/QA_dataset_nus/test_eval.json"
    Image_path = "/media/oh/0E4A12890E4A1289/DriveLM/data/"
    Save_path = "/media/oh/0E4A12890E4A1289/DriveLM/data/QA_dataset_nus/Multi_choice_behavior_test_Gemini.json"
    Result_path = "/media/oh/0E4A12890E4A1289/DriveLM/data/QA_dataset_nus/"

    # 配置参数
    Api_key = "AIzaSyCQSNKK5sH4yN87JQFnyEVQVhsOcam8VII"
    #Model_name = "gemini-exp-1121"
    Model_name = "gemini-1.5-pro"

    Gpt4_VQA_Nusence_COT_benchmark(Save_path, Image_path, Result_path, Model_name, "behavior", 20, 50)







