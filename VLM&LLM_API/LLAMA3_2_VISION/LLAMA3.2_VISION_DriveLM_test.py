import Multi_modal_Infer
import json
from PIL import Image, ImageDraw, ImageFont

import base64
import json
import torch

import time
from tqdm import tqdm
import os

# Suppress logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"  # 0: INFO, 1: WARNING, 2: ERROR, 3: FATAL



def assistant(content: str):
    return { "role": "assistant", "content": content }


def user_input(prompt_text: str, images: list = None):
    """
    构造用户输入，可以包含文本和图片。

    Args:
        prompt_text (str): 用户输入的文本内容。
        images (list, optional): 图片 URL 列表。如果没有图片，默认为 None。

    Returns:
        dict: 用户输入的结构化消息。
    """
    content = []
    if images:
        # 添加图片内容
        content.extend([{"type": "image"} for _ in images])
    # 添加文本内容
    content.append({"type": "text", "text": prompt_text})

    return {"role": "user", "content": content}

def system(content: str):
    return { "role": "system", "content": content }


def llama3_VQA_Nusence_COT_benchmark(
        json_path, image_path, result_path,
        model,processor, device,
        task_type  ,sample_num=20, MAX_OUTPUT_TOKENS = 2048):
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

    # Write to JSON file after each response
    result_path = result_path + task_type + "_test_result_LLAMA3.2.json"

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

    # 检查并清空文件
    if os.path.exists(result_path):
        os.remove(result_path)  # 删除已存在的文件
        print(f"{result_path} exists and has been removed.")
    else:
        print(f"{result_path} does not exist, creating a new file.")

    # 初始化文件，写入 JSON 数组的开头
    with open(result_path, "w") as outfile:
        outfile.write("[\n")  # 开始 JSON 数组

    # Iterate through samples with a limit of `sample_num`
    for i, sample in enumerate(tqdm(qa_data, desc="Generating responses: ")):

        if i >= sample_num:
            break

        # Ensure each sample contains 6 images
        if len(sample.get('image', [])) != 6:
            print(f"Warning: Sample {i} does not have exactly 6 images. Skipping.")
            continue

        try:
            images = Multi_modal_Infer.process_image( image_paths = sample['image'], merge='custom_grid', max_dimensions=(1120,1120))
            """Process and validate image input, with optional resizing"""
            #files = [encode_image(image) for image in sample['image']]
        except Exception as e:
            print(f"Error opening files for sample {i}: {e}")
            continue

        # Extract question
        Q = sample['conversations'][0]['value'] if len(sample.get('conversations', [])) > 0 else ""

        # Prepare input
        input_2 = (
            """The six images displayed above were captured by a vehicle's cameras positioned at the following locations (The order of the 6 photos is as follows):

                {CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT;
                 CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT}.

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

        conversation = [
            system(system_instruction),
            user_input(f"Input images in the order: **CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT**", images),
            user_input(input_2),
            #user_input("Can you help me analyze the images and answer questions step by step?"),
        ]

        #print(conversation)


        # Start chat session
        if task_type == "behavior":

            try:
                """
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                inputs = processor(images=images, text=prompt, return_tensors="pt").to(device)
                with torch.no_grad():  # Disable gradient computation to save memory
                    output = model.generate(**inputs, temperature=temperature, top_p=top_p,
                                            max_new_tokens=MAX_OUTPUT_TOKENS)
                assistant_message = processor.decode(output[0])[len(prompt):]

                print(f"LLAMA: {assistant_message}")
                print("*" * 50)
                conversation.append(assistant(assistant_message))
"""

                # 添加用户消息到对话历史
                conversation.append(user_input("""Analyze the image and assess the following:
                    What is the current state of the ego vehicle (position, speed, or behavior)?
                    What are the states of the surrounding vehicles and pedestrians, and how might they influence the ego vehicle's actions?
                    Describe the surrounding traffic environment, including road conditions, traffic signs, and signals."""))


                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                inputs = processor(images=images, text=prompt, return_tensors="pt").to(device)
                with torch.no_grad():  # Disable gradient computation to save memory
                    output = model.generate(**inputs, temperature=temperature, top_p=top_p,
                                            max_new_tokens=MAX_OUTPUT_TOKENS)

                images[0].save("merged_image.png")

                # 提取生成的新部分（避免重复提示）
                generated_tokens = output[:, inputs['input_ids'].shape[1]:]

                # 解码生成的 token
                assistant_message = processor.decode(
                    generated_tokens[0],  # 解码第一个样本
                    skip_special_tokens=True,  # 跳过特殊标记
                    clean_up_tokenization_spaces=True  # 自动清理多余空格
                )

                print(f"LLAMA: {assistant_message}")
                print("*" * 50)
                conversation.append(assistant(assistant_message))

                conversation.append(user_input( f"The question is: {Q} provide the final answer in the following format: </ans>answer</ans>"))

                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                inputs = processor(images=images, text=prompt, return_tensors="pt").to(device)
                with torch.no_grad():  # Disable gradient computation to save memory
                    output = model.generate(**inputs, temperature=temperature, top_p=top_p,
                                            max_new_tokens=MAX_OUTPUT_TOKENS)

                # 提取生成的新部分（避免重复提示）
                generated_tokens = output[:, inputs['input_ids'].shape[1]:]

                # 解码生成的 token
                assistant_message = processor.decode(
                    generated_tokens[0],  # 解码第一个样本
                    skip_special_tokens=True,  # 跳过特殊标记
                    clean_up_tokenization_spaces=True  # 自动清理多余空格
                )

                print(f"LLAMA: {assistant_message}")
                print("*" * 50)
                conversation.append(assistant(assistant_message))

                conversation.append(user_input(
                    "It's a choice question. Compare your answer with the options given and choose the most relevant option. "
                    "Wrap the final answer in the following format "
                    " </ans>Your_Choice</ans>  "
                   ))

                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                inputs = processor(images=images, text=prompt, return_tensors="pt").to(device)
                with torch.no_grad():  # Disable gradient computation to save memory
                    output = model.generate(**inputs, temperature=temperature, top_p=top_p,
                                            max_new_tokens=MAX_OUTPUT_TOKENS)

                # 提取生成的新部分（避免重复提示）
                generated_tokens = output[:, inputs['input_ids'].shape[1]:]

                # 解码生成的 token
                assistant_message = processor.decode(
                    generated_tokens[0],  # 解码第一个样本
                    skip_special_tokens=True,  # 跳过特殊标记
                    clean_up_tokenization_spaces=True  # 自动清理多余空格
                )

                print(f"LLAMA: {assistant_message}")
                print("*" * 50)

                final_response = assistant_message
                print("*"*50)
                print(f"Model: {final_response}")

            except Exception as e:
                print(f"Error during conversation: {e}")
                continue

            # Modify sample and write to file
            modified_sample = sample.copy()
            modified_sample["conversations"].append({
                "from": "Model_Output",
                "value": final_response
            })

            # 写入结果到文件
            try:
                with open(result_path, "a") as outfile:
                    if i > 0:  # 如果不是第一个结果，添加逗号分隔
                        outfile.write(",\n")
                    json.dump(modified_sample, outfile, indent=4)

                print(f"Sample {i} written to {result_path}")

            except Exception as e:
                print(f"Error writing sample {i} to JSON file: {e}")

    # 在文件末尾关闭 JSON 数组
    with open(result_path, "a") as outfile:
        outfile.write("\n]\n")  # 关闭 JSON 数组


def get_optimal_device():
    if not torch.cuda.is_available():
        return "cpu"

    # 获取所有设备的显存信息
    devices = list(range(torch.cuda.device_count()))
    free_memory = [torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i) for i in devices]

    # 选择可用显存最大的设备
    optimal_device = f"cuda:{free_memory.index(max(free_memory))}"
    return optimal_device


if __name__ == '__main__':
    # 加载数据

    Json_path = "/media/workstation/6D3563AC52DC77EA/Data/DriveLM/data/QA_dataset_nus/test_eval.json"
    Image_path = "/media/workstation/6D3563AC52DC77EA/Data/DriveLM/data/"
    Save_path = "/media/workstation/6D3563AC52DC77EA/Data/DriveLM/data/QA_dataset_nus/Multi_choice_behavior_test_Llama.json"
    Result_path = "/media/workstation/6D3563AC52DC77EA/Data/DriveLM/data/QA_dataset_nus/"

    # 配置参数
    model_name = '/media/workstation/6D3563AC52DC77EA/Model/meta-llama/Llama-3.2-11B-Vision-Instruct'
    Lora_name =  '/media/workstation/6D3563AC52DC77EA/Model/meta-llama/Llama-3.2-11B-Vision-Instruct/lora_model'
    #finetuning_path = None  # 如果有微调路径可以设置

    temperature = 0.1
    top_p = 0.3

    # 确保 CUDA 可用
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your GPU configuration.")
        # return

        # 加载模型和处理器
    try:
        model, processor = Multi_modal_Infer.load_model_and_processor(
            model_name=model_name,
            finetuning_path= Lora_name,
            device="sequential",  # 自动设备映射
            max_memory={0: "22GB", 1: "5GB"}  # GPU 显存限制
        )
        print("Model and processor loaded successfully.")
    except Exception as e:
        print(f"Failed to load model and processor: {e}")
        # return

    # 获取当前设备
    device = get_optimal_device()
    print(f"Selected optimal device: {device}")


    llama3_VQA_Nusence_COT_benchmark(
        Save_path, Image_path, Result_path,
        model,processor, device,
        "behavior", 100 , 512)









