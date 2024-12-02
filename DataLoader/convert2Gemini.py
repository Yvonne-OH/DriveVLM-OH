
import json
import tqdm


import os

import google.generativeai as genai



def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file


def GminiJson2List(Json_path, Image_path, API_key, Sample_num=3):
    root = Json_path
    genai.configure(api_key=API_key)

    files_list = []
    response_list = []

    # 打开并加载 JSON 文件
    with open(root, 'r') as f:
        Gemini_Json_file = json.load(f)


    # 遍历样本，限制为 Sample_num
    for i, sample in enumerate(Gemini_Json_file):
        if i >= Sample_num:  # 如果处理数量达到限制则停止
            break

        # 确保每个样本的 image 包含 6 张图片
        if len(sample['image']) != 6:
            print(f"Warning: Sample {i} does not have exactly 6 images. Skipping.")
            continue

        # 调用 `upload_to_gemini` 上传文件并生成 `files` 列表
        try:
            files = [upload_to_gemini(image, mime_type="image/jpg") for image in sample['image']]
            files_list.append(files)  # 将上传的文件对象添加到列表
        except Exception as e:
            print(f"Error uploading files for sample {i}: {e}")
            continue

        # 提取对话内容
        input_1 = sample['conversations'][0]['value'] if len(sample['conversations']) > 0 else ""
        output_1 = sample['conversations'][1]['value'] if len(sample['conversations']) > 1 else ""

        # 构建对话格式
        input_2 = (
            "The six images above were taken by a car's front, left front, right front, "
            "rear, left rear, and right rear cameras. Based on the pictures, develop inferences and answer the questions: "
            f"{input_1}"
        )

        response = [
            "input: ",
            files[0],
            files[1],
            files[2],
            files[3],
            files[4],
            files[5],
            f"input 2: {repr(input_2)}",
            f"output: {repr(output_1)}"
        ]
        response_list.append(response)

    print("Successfully uploaded all files and generated responses.")
    return files_list, response_list


def convert2Gemini(Json_path, Image_path, Save_path):



    with open(Json_path, 'r') as f:
        test_file = json.load(f)

    output = []
    tqdm.tqdm.write("Converting to Gemini format...")

    scene_ids = list(test_file.keys())
    for scene_id in tqdm.tqdm(scene_ids, desc="Scenes"):
        scene_data = test_file[scene_id]['key_frames']

        frame_ids = list(scene_data.keys())
        for frame_id in tqdm.tqdm(frame_ids, desc="Frames", leave=False):
            image_paths = scene_data[frame_id]['image_paths']
            image_paths = [image_paths[key].replace("..", Image_path) for key in image_paths.keys()]

            frame_data_qa = scene_data[frame_id]['QA']
            QA_pairs = frame_data_qa["perception"] + frame_data_qa["prediction"] + frame_data_qa["planning"] + \
                       frame_data_qa["behavior"]

            for idx, qa in enumerate(QA_pairs):
                question = qa['Q']
                answer = qa['A']
                output.append(
                    {
                        "id": scene_id + "_" + frame_id + "_" + str(idx),
                        "image": image_paths,
                        "conversations": [
                            {
                                "from": "Input",
                                "value": "<image>\n" + question
                            },
                            {
                                "from": "Output",
                                "value": answer
                            },
                        ]
                    }
                )

    with open(Save_path, 'w') as f:
        json.dump(output, f, indent=4)

def convert2Gemini_MultiChoice(Json_path, Image_path, Save_path):

    with open(Json_path, 'r') as f:
        test_file = json.load(f)

    output = []
    tqdm.tqdm.write("Converting to Gemini format...")

    scene_ids = list(test_file.keys())
    for scene_id in tqdm.tqdm(scene_ids, desc="Scenes"):
        scene_data = test_file[scene_id]['key_frames']

        frame_ids = list(scene_data.keys())
        for frame_id in tqdm.tqdm(frame_ids, desc="Frames", leave=False):
            image_paths = scene_data[frame_id]['image_paths']
            image_paths = [image_paths[key].replace("..", Image_path) for key in image_paths.keys()]

            frame_data_qa = scene_data[frame_id]['QA']
            QA_pairs = frame_data_qa["perception"] + frame_data_qa["prediction"] + frame_data_qa["planning"] + \
                       frame_data_qa["behavior"]

            for idx, qa in enumerate(QA_pairs):
                question = qa['Q']
                answer = qa['A']
                output.append(
                    {
                        "id": scene_id + "_" + frame_id + "_" + str(idx),
                        "image": image_paths,
                        "conversations": [
                            {
                                "from": "Input",
                                "value": "<image>\n" + question
                            },
                            {
                                "from": "Output",
                                "value": answer
                            },
                        ]
                    }
                )

    with open(Save_path, 'w') as f:
        json.dump(output, f, indent=4)

import json
import tqdm

import json
import tqdm
import re
from PIL import Image

def get_image_resolution(image_path):
    """
    获取图像的宽度和高度
    """
    with Image.open(image_path) as img:
        return img.width, img.height

def convert2Gemini_MultiChoice(Json_path, Image_path, Save_path):
    with open(Json_path, 'r') as f:
        test_file = json.load(f)

    output = []
    tqdm.tqdm.write("Converting to Gemini format...")

    scene_ids = list(test_file.keys())
    for scene_id in tqdm.tqdm(scene_ids, desc="Scenes"):
        scene_data = test_file[scene_id]['key_frames']

        frame_ids = list(scene_data.keys())
        for frame_id in tqdm.tqdm(frame_ids, desc="Frames", leave=False):
            image_paths = scene_data[frame_id]['image_paths']
            image_paths = [image_paths[key].replace("..", Image_path) for key in image_paths.keys()]

            # 假设使用第一个图像获取宽度和高度
            if image_paths:
                width, height = get_image_resolution(image_paths[0])
            else:
                width, height = 1600, 900  # 默认值

            frame_data_qa = scene_data[frame_id]['QA']
            QA_pairs = frame_data_qa["perception"] + frame_data_qa["prediction"] + frame_data_qa["planning"] + \
                       frame_data_qa["behavior"]

            for idx, qa in enumerate(QA_pairs):
                question = qa['Q']
                answer = qa['A']

                # 只处理选择题部分，即答案为 A、B、C、D 的内容
                if answer in {"A", "B", "C", "D"}:
                    # 使用正则表达式替换 <c1,CAM_BACK,1088.3,497.5> 的坐标
                    def replace_coordinates(match):
                        obj_id, cam, x, y = match.groups()
                        x_normalized = int(1000*float(x) / width)
                        y_normalized = int(1000*float(y) / height)
                        return f"<{obj_id},{cam},{x_normalized:.4f},{y_normalized:.4f}>"

                    question = re.sub(r"<(c\d+),(\w+),([\d.]+),([\d.]+)>", replace_coordinates, question)

                    output.append(
                        {
                            "id": scene_id + "_" + frame_id + "_" + str(idx),
                            "image": image_paths,
                            "conversations": [
                                {
                                    "from": "Input",
                                    "value": "<image>\n" + question
                                },
                                {
                                    "from": "Output",
                                    "value": answer
                                },
                            ]
                        }
                    )

    with open(Save_path, 'w') as f:
        json.dump(output, f, indent=4)


    with open(Save_path, 'w') as f:
        json.dump(output, f, indent=4)


if __name__ == '__main__':
    # 加载数据
    Json_path = "/media/oh/0E4A12890E4A1289/DriveLM/data/QA_dataset_nus/test_eval.json"
    Image_path = "/media/oh/0E4A12890E4A1289/DriveLM/data/"
    Save_path = "/media/oh/0E4A12890E4A1289/DriveLM/data/QA_dataset_nus/test_Gemini.json"



    #convert2Gemini(Json_path,  Image_path,Save_path)
    convert2Gemini_MultiChoice(Json_path, Image_path, "/media/oh/0E4A12890E4A1289/DriveLM/data/QA_dataset_nus/test_Gemini_MultiChoice.json")










