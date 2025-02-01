import json
import tqdm
import os
import random
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


LingoQA_folder = "/media/workstation/6D3563AC52DC77EA/Data/LingoQA/"
Omni_Driver_folder = "/media/workstation/6D3563AC52DC77EA/Data/Omni_Drive/data_nusc/"
CODA_LM_folder = "/media/workstation/6D3563AC52DC77EA/Data/CODA_LM/"

def select_random_subset(data, ratio):
    """随机选择字典中的一部分数据。

    参数:
    data (dict): 要从中选择数据的字典。
    ratio (float): 选取的数据比例，介于 0 和 1 之间。

    返回:
    dict: 包含选定部分数据的新字典。
    """
    if isinstance(data, dict):
        keys = list(data.keys())
        random.shuffle(keys)
        num_keys_to_select = int(len(keys) * ratio)
        selected_keys = keys[:num_keys_to_select]
        selected_data = {key: data[key] for key in selected_keys}
        return selected_data
    else:
        print("Error: Data provided is not a dictionary.")
        return None

def get_LingoQA_Scene_data(LingoQA_folder, set_split_ratio= 0.2 ):
    with open(LingoQA_folder+"/train/Scene_train.json", 'r') as f:
        Scene_train_data = json.load(f)

        return select_random_subset(Scene_train_data, set_split_ratio)

def get_LingoQA_Action_data(LingoQA_folder, set_split_ratio= 0.2 ):
    with open(LingoQA_folder+"/train/Action_train.json", 'r') as f:
       Action_train_data = json.load(f)

       return select_random_subset(Action_train_data, set_split_ratio)


def get_Omni_Driver_conv_data(Omni_Driver_folder, set_split_ratio= 0.2):
    with open(Omni_Driver_folder+"/conv/train/json_data/conv_train.json", 'r') as f:
        data = json.load(f)

    return select_random_subset(data, set_split_ratio)

def get_Omni_Driver_desc_data(Omni_Driver_folder, set_split_ratio= 0.2):
    with open(Omni_Driver_folder+"/desc/train/json_data/desc_train.json", 'r') as f:
        data = json.load(f)

    return select_random_subset(data, set_split_ratio)

def get_CODA_LM_data(CODA_LM_folder, set_split_ratio= 0.2):
    with open(CODA_LM_folder+"Train.json", 'r') as f:
        data = json.load(f)

    return select_random_subset(data, set_split_ratio)


def convert_conservation_to_unsloth(sample, image_path):
    """
    将 conservation 数据转换为指定格式的 conversation 数据。

    参数:
        conservation (list): 输入的对话数据列表，每个元素是字典，包含 'from' 和 'value'。
        image_folder (str): 存储图片的文件夹路径，用于加载图像。

    返回:
        list: 转换后的 conversation 数据列表。
    """
    conversation = []
    for item in sample:
        role = item['from']  # 提取角色 (system, human, gpt)
        value = item['value']  # 提取对应的文本内容

        # 根据角色构建内容
        if role == "system":
            conversation.append({
                "role": "system",
                "content": [
                    {"type": "text", "text": value}
                ]
            })
        elif role == "human":
            # 判断是否包含图片占位符 "<image>"
            if "<image>" in value:
                # 提取图像名称并加载图像
                conversation.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": value.replace("<image>", "").strip()},
                        {"type": "image", "image": Image.open(image_path)}  # 替换为实际图片路径
                    ]
                })
            else:
                conversation.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": value}
                    ]
                })
        elif role == "gpt":
            conversation.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": value}
                ]
            })
    return {"messages": conversation}

if __name__ == '__main__':

    LingoQA_Scene_data = get_LingoQA_Scene_data(LingoQA_folder, set_split_ratio= 0.2)
    LingoQA_Action_data = get_LingoQA_Scene_data(LingoQA_folder, set_split_ratio=0.2)
    Omni_Driver_conv_data = get_Omni_Driver_conv_data(Omni_Driver_folder, set_split_ratio=0.2)
    Omni_Driver_desc_data = get_Omni_Driver_conv_data(Omni_Driver_folder, set_split_ratio=0.2)
    CODA_LM_data = get_CODA_LM_data(CODA_LM_folder, set_split_ratio=0.2)

    for Token in LingoQA_Action_data:
        Scene =   LingoQA_Action_data[Token]
        pass



