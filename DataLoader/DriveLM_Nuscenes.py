import json
import os
import numpy as np
from textwrap import wrap
import matplotlib.pyplot as plt
from collections import defaultdict

def data_loader(root_path):
    # 构建 JSON 文件路径
    json_file_path = root_path

    # 读取 JSON 文件
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
            print("Open JSON file successfully.")
    except FileNotFoundError:
        print(f"File not found: {json_file_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    return json_data


def process_data(json_data):
    # 遍历数据列表
    for item in json_data[0:5]:
        # 提取 id
        item_id = item.get("id", None)

        # 提取 image 列表
        image_paths = item.get("image", [])

        # 提取 conversations 列表
        conversations = item.get("conversations", [])

        # 打印数据
        print(f"ID: {item_id}")
        print(f"Images: {image_paths}")
        print("Conversations:")
        for convo in conversations:
            from_person = convo.get("from", "unknown")
            value = convo.get("value", "unknown")
            print(f"  From: {from_person}, Value: {value}")
        print("-" * 40)


def merge_objects(data):
    grouped_data = defaultdict(lambda: {"image": [], "conversations": []})

    for item in data:
        # 提取 ID 前缀，忽略最后的序列号
        full_id = item.get("id", "") # 获取id, 如果没有则为空字符串
        """split() 方法通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则分隔 num+1 个子字符串
            取出0-倒数第二个元素，即忽略最后一个元素，最后重新用-连接"""
        id_prefix = "_".join(full_id.split("_")[:-1])  # 忽略最后一部分

        # 合并 image（相同 ID 的 image 是相同的，只保留一个）
        if not grouped_data[id_prefix]["image"]:
            grouped_data[id_prefix]["image"] = item.get("image", [])

        # 合并 conversations
        grouped_data[id_prefix]["conversations"].extend(item.get("conversations", []))

    # 转为列表形式
    merged_data = [
        {"id": id_prefix, "image": value["image"], "conversations": value["conversations"]}
        for id_prefix, value in grouped_data.items()
    ]

    return merged_data

def data_visulization(img_path, merged_data, sample_id = 0):

    data_package = merged_data[sample_id]
    ID = data_package['id']
    images = data_package['image']
    conversations = data_package['conversations']

    fig, axes = plt.subplots(2, 3, figsize=(24, 30))

    # Define the titles for each subplot
    titles = ['frontleft', 'front', 'frontright', 'backleft', 'back', 'backright']

    for ax, cam_id, title in zip(axes.flatten(), images, titles):
        img = plt.imread(os.path.join(img_path, cam_id))
        ax.imshow(img)
        #ax.set_title(title)
        ax.axis('off')

    # Set the main title
    fig.suptitle("Image ID: " + ID, fontsize=32, fontweight='bold',y=0.98)

    # Adjust layout to remove whitespace
    plt.subplots_adjust(top=0.95, bottom=0.01, left=0.01, right=0.99, hspace=0, wspace=0.05)

    # 将 conversations 内容排版并写到图下方
    conversation_texts = [f"{conv['from']}: {conv['value']}" for conv in conversations]
    wrapped_text = "\n".join(["\n".join(wrap(text, width=80)) for text in conversation_texts])  # 自动换行，宽度80字符
    fig.text(0.5, 0.1, wrapped_text, ha='center', va='top', fontsize=20, wrap=True)

    plt.show()


if __name__ == '__main__':
    QA_json_path = '/media/oh/0E4A12890E4A1289/DriveLM/data/QA_dataset_nus/test_llama.json'
    save_path = '/media/oh/0E4A12890E4A1289/DriveLM/data/QA_dataset_nus/test_OH.json'
    root_path = '/media/oh/0E4A12890E4A1289/DriveLM/'
    json_data = data_loader(QA_json_path)
    process_data(json_data)
    #print(type(data))
    # 合并相同 ID 的对象
    merged_data = merge_objects(json_data)
    data_visulization(root_path, merged_data, sample_id=0)

    # 打印结果或保存到文件
    #print(json.dumps(merged_data, indent=4, ensure_ascii=False))

