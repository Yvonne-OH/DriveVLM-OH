import os
import json
import tqdm
import re
from PIL import Image

def convert2Llama(Json_path, Image_path, Save_path, type = "perception"):

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
            #QA_pairs = frame_data_qa["perception"] + frame_data_qa["prediction"] + frame_data_qa["planning"] + frame_data_qa["behavior"]

            QA_pairs = frame_data_qa[type]

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

    Save_path = Save_path+type+"_test_Llama.json"

    with open(Save_path, 'w') as f:
        json.dump(output, f, indent=4)

def convert2Llama_MultiChoice(Json_path, Image_path, Save_path):

    with open(Json_path, 'r') as f:
        test_file = json.load(f)

    output = []
    tqdm.tqdm.write("Converting to Llama format...")

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



def get_image_resolution(image_path):
    """
    获取图像的宽度和高度
    """
    with Image.open(image_path) as img:
        return img.width, img.height

def convert2Llama_MultiChoice(Json_path, Image_path, Save_path, type = "perception"):
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
            #QA_pairs = frame_data_qa["perception"] + frame_data_qa["prediction"] + frame_data_qa["planning"] + frame_data_qa["behavior"]
            QA_pairs = frame_data_qa[type]

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

    Save_path = Save_path+"Multi_choice_"+type+"_test_Llama.json"
    with open(Save_path, 'w') as f:
        json.dump(output, f, indent=4)




if __name__ == '__main__':
    # 加载数据
    Json_path = "/media/workstation/6D3563AC52DC77EA/Data/DriveLM/data/QA_dataset_nus/test_eval.json"
    Image_path = "/media/workstation/6D3563AC52DC77EA/Data//DriveLM/data/"
    Save_path = "/media/workstation/6D3563AC52DC77EA/Data//DriveLM/data/QA_dataset_nus/"


    #convert2Gemini(Json_path,  Image_path,Save_path, type = "prediction")
    convert2Llama_MultiChoice(Json_path, Image_path, Save_path, type = "behavior")









