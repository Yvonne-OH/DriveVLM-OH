import Multi_modal_Infer
import copy
import json
import json
import torch
import time
from tqdm import tqdm
import os
import Util.util as util

import torch
from tqdm import tqdm

def process_task_COT(model, processor, device, task_desc: str = None, Question: str = None, image_paths: list = None,**kwargs)->str:
    """
    处理一个任务，包括图像合并、对话生成和问题回答。

    参数：
    - sample: 任务的样本数据，包含图像路径和问题等信息。
    - task_desc: 任务描述字符串，包含任务的指令和描述内容。
    - image_paths: 图像路径列表，包含车辆不同视角的图像路径。
    - model: 预训练的模型。
    - processor: 用于图像和文本处理的处理器。
    - device: 设备（例如 CPU 或 GPU）。
    - temperature: 生成文本时的温度，控制生成的随机性。
    - top_p: 生成时使用的 nucleus sampling的概率阈值。
    - MAX_OUTPUT_TOKENS: 生成的最大token数量。
    - resize_to: 图像缩放尺寸。
    - max_dimensions: 图像的最大尺寸。

    返回值：
    - final_response: 模型生成的最终响应。
    """

    temperature = kwargs.get('temperature', 0.15)
    top_p = kwargs.get('top_p', 0.3)
    MAX_OUTPUT_TOKENS = kwargs.get('MAX_OUTPUT_TOKENS', 2048)
    resize_to = kwargs.get('resize_to', (224, 224))
    max_dimensions = kwargs.get('max_dimensions', (1120, 1120))
    debug = kwargs.get('debug', False)  # 默认值 False

    # 提取任务描述中的相关内容
    task_desc_content = util.extract_between_markers(task_desc, "[DESC_START]", "[DESC_END]")[0]
    # 系统指令
    system_instruction = task_desc_content

    # 初始化 MultimodalInputBuilder 和 ImageProcessor
    MultimodalInputBuilder = util.MultimodalInputBuilder("LLAMA")
    image_processor = util.ImagePreprocessor(resize_to=resize_to, max_dimensions=max_dimensions)


    try:
        # 合并车辆视角图像
        images = image_processor.merge_vehicle_camera_views(image_paths=image_paths, merge='custom_grid',
                                                            logical_order=[1, 0, 2, 4, 3, 5])
        images[0].save("merged_image.png")
    except Exception as e:
        print(f"Error opening image files: {e}")
        return None


    conversation = [
        MultimodalInputBuilder.system(system_instruction),
        MultimodalInputBuilder.user_input(
        util.extract_between_markers(task_desc, "[IMAGE_ORDER_START]", "[IMAGE_ORDER_END]")[0],
        images),
    ]

    # 添加用户消息到对话历史
    conversation.append(MultimodalInputBuilder.user_input(util.extract_between_markers(task_desc, "[COT_STEP1_START]", "[COT_STEP1_END]")[0]))

    # 创建和应用聊天模板
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(images=images, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():  # 禁用梯度计算以节省内存
        output = model.generate(**inputs, temperature=temperature, top_p=top_p, max_new_tokens=MAX_OUTPUT_TOKENS)

    # 提取生成的新部分（避免重复提示）
    generated_tokens = output[:, inputs['input_ids'].shape[1]:]
    assistant_message = processor.decode(
        generated_tokens[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    conversation.append(MultimodalInputBuilder.assistant(assistant_message))

    # 添加用户问题并生成回答
    conversation.append(MultimodalInputBuilder.user_input(
        f"The question is: {Question} Based on the above chain of thought, give your answer and explain in the following format: </ans>answer</ans>"
    ))

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(images=images, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():  # 禁用梯度计算以节省内存
        output = model.generate(**inputs, temperature=temperature, top_p=top_p, max_new_tokens=MAX_OUTPUT_TOKENS)

    generated_tokens = output[:, inputs['input_ids'].shape[1]:]

    # 解码生成的 token
    assistant_message = processor.decode(
        generated_tokens[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    conversation.append(MultimodalInputBuilder.assistant(assistant_message))

    # 添加最终选择的用户输入
    conversation.append(MultimodalInputBuilder.user_input(
        "Wrap the final answer in the following format </ans>Your_Choice</ans>"
    ))

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(images=images, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():  # 禁用梯度计算以节省内存
        output = model.generate(**inputs, temperature=temperature, top_p=top_p, max_new_tokens=MAX_OUTPUT_TOKENS)

    generated_tokens = output[:, inputs['input_ids'].shape[1]:]

    # 解码生成的 token
    assistant_message = processor.decode(
        generated_tokens[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    if debug:
        print(f"LLAMA: {assistant_message}")
        print("*" * 50)

    final_response = assistant_message
    if debug:
        print("*" * 50)
        print(f"Model: {final_response}")

    return final_response


if __name__ == '__main__':


    model_name = '/media/workstation/6D3563AC52DC77EA/Model/meta-llama/Llama-3.2-11B-Vision-Instruct'
    Lora_name =  '/media/workstation/6D3563AC52DC77EA/Model/meta-llama/Llama-3.2-11B-Vision-Instruct/lora_model'
    json_path = "/media/workstation/6D3563AC52DC77EA/Data/DriveLM/data/QA_dataset_nus/test.json"
    result_path = "/media/workstation/6D3563AC52DC77EA/Data/DriveLM/data/QA_dataset_nus/test_benchmark.json"
    image_path = "/media/workstation/6D3563AC52DC77EA/Data/DriveLM/data/"

    config = {
        "temperature": 0.1,
        "top_p": 0.3,
        "MAX_OUTPUT_TOKENS": 2048,
        "resize_to": (224, 224),
        "max_dimensions": (1120, 1120),
        "debug": False
    }

    #
    batch_size = 10 # 设置批量写入的间隔
    max_sessions = 2
    processed_sessions = 0  # 记录处理的会话数量
    processed_count = 0  # 记录处理的问题数量

    # 确保 CUDA 可用
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your GPU configuration.")
        # return

        # 加载模型和处理器
    try:
        model, processor = Multi_modal_Infer.load_model_and_processor(
            model_name=model_name,
            finetuning_path= Lora_name,
            device="auto",  # 自动设备映射
            max_memory={0: "22GB", 1: "16GB"}  # GPU 显存限制
        )
        print("Model and processor loaded successfully.")
    except Exception as e:
        print(f"Failed to load model and processor: {e}")
        # return

    # 获取当前设备
    device = model.device
    print(f"Selected optimal device: {device}")

    # 加载数据
    with open ('Task_DESC.txt') as f:
        task_desc = f.read()




    # Load JSON file
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")

    result_data = copy.deepcopy(json_data)


    # 使用 tqdm 包装最外层的会话循环，显示进度条
    for session_id, session_content in tqdm(result_data.items(), desc="Processing Sessions", total=len(result_data)):

        # 处理完所有问题后，继续到下一个会话
        processed_sessions += 1

        if processed_sessions >= max_sessions:
            break

        key_frames = session_content.get('key_frames', {})
        for frame_id, frame_content in key_frames.items():
            qa_questions = frame_content.get('QA', {})
            images = list(frame_content.get("image_paths", []).values())

            # 遍历所有图像路径并进行替换
            for i, image in enumerate(images):
                images[i] = image.replace('../', image_path)


            for section_name, questions in qa_questions.items():
                for question in questions:
                    q = question.get("Q")
                    a = question.get("A")

                    # 添加 GPT 回答到每个问题字典中
                    question['A_GPT'] =     Gpt_answer = process_task_COT(
                                                                                                                        model,
                                                                                                                        processor,
                                                                                                                        device,
                                                                                                                        task_desc = task_desc,
                                                                                                                        Question = q,
                                                                                                                        image_paths = images ,
                                                                                                                        **config )

                    # 更新 question 为有序字典
                    question = {k: question[k] for k in sorted(question)}

                    processed_count += 1

                    # 每处理一定数量的问题，批量写入一次
                    if processed_count % batch_size == 0:
                        print(f"Processed {processed_count} questions, writing to file...")
                        with open(result_path, 'w', encoding='utf-8') as f:
                            json.dump(result_data, f, indent=4, ensure_ascii=False)

                    # 打印输出处理结果（调试）
                    print(f"Processed question: {q}")
                    print(f"Original Answer: {a}")
                    print(f"GPT Answer: {question['A_GPT']}")
                    print("----")

                # 如果已经达到最大处理会话数量，跳出外层循环
                if processed_sessions >= max_sessions:
                    break
            if processed_sessions >= max_sessions:
                break
        if processed_sessions >= max_sessions:
            break

    # 处理完所有问题后，再写入一次（防止最后一批没有写入）
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=4, ensure_ascii=False)










