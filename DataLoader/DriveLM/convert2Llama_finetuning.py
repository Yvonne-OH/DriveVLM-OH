import json
import re
import os
import shutil
from tqdm import tqdm
import Util.util
from PIL import Image

# 定义主函数
def create_nuscene_qa_llama_finetuning_dataset(Json_path, Image_path, Save_path):

    # 设置文件夹路径
    image_folder_path = os.path.join(Save_path, "image_folder")

    # 如果文件夹存在且非空，清空它
    if os.path.exists(Save_path):
        shutil.rmtree(Save_path)
    os.makedirs(image_folder_path)

    # 读取 JSON 数据
    with open(Json_path, 'r') as f:
        data = json.load(f)

    scene_ids = list(data.keys())

    result = []

    # 遍历场景和帧数据
    for scene_id in tqdm(scene_ids, desc="Processing scenes"):
        scene_data = data[scene_id]['key_frames']
        frame_ids = list(scene_data.keys())

        for frame_id in frame_ids:
            frame_data = scene_data[frame_id]
            id = scene_id + "_" + frame_id
            image_paths = frame_data['image_paths']
            image_paths = [image_paths[key].replace("..", Image_path) for key in image_paths.keys()]

            # 处理图像并获取尺寸
            base_width, base_height = Image.open(image_paths[0]).size
            image = Util.util.process_image(image_paths, merge='custom_grid', max_dimensions=(1120, 1120))

            # 保存图片
            image_save_path = os.path.join(image_folder_path, f"{id}.jpg")
            image[0].save(image_save_path)  # 假设 image[0] 是 PIL.Image 对象

            conservation = [
                Util.util.system("""The six images displayed above were captured by a vehicle's cameras positioned at the following locations (The order of the 6 photos is as follows):

                    {CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT;
                     CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT}.

                    In the format <object_id,camera_name,x_coord,y_coord>:

                object_id represents the unique identifier for the object.
                camera_name specifies the camera that captured the object.
                x_coord and y_coord represent the x and y coordinates of the object's center, expressed as percentages of the image dimensions.
                Coordinates are specified as percentages, adhering to the following conventions:


                The origin (0, 0) is located at the top-left corner of the image.
                The bottom-right corner of the image corresponds to (100%, 100%).
                The x-axis extends horizontally, increasing from left (0) to right (100%).
                The y-axis extends vertically, increasing from top (0) to bottom (100%).""")
            ]

            frame_data_qa = scene_data[frame_id]['QA']

            for key in frame_data_qa.keys():
                QA_pairs = frame_data_qa[key]
                for qa in QA_pairs:
                    question = qa['Q']
                    question = re.sub(r"<[^>]*>", lambda match: Util.util.convert_to_percentage(base_width, base_height, match), question)
                    answer = qa['A']
                    conservation.append(
                        {"from": "human", "value": "<image>\n" + question}
                    )
                    conservation.append(
                        {"from": "gpt", "value": answer}
                    )

            result.append({
                "id": id,
                "image": id + ".jpg",
                "conversations": conservation
            })

    # 将结果保存为 JSON 文件
    output_json_path = os.path.join(Save_path, "NusceneQA_Llama_finetuning_dataset.json")
    with open(output_json_path, "w") as f:
        json.dump(result, f, indent=4)

    print("Dataset created and saved to", output_json_path)

if __name__ == '__main__':
    create_nuscene_qa_llama_finetuning_dataset(
        Json_path="/media/workstation/6D3563AC52DC77EA/Data/DriveLM/data/QA_dataset_nus/v1_1_train_nus.json",
        Image_path="/media/workstation/6D3563AC52DC77EA/Data//DriveLM/data/",
        Save_path="/media/workstation/6D3563AC52DC77EA/Data//DriveLM/data/QA_dataset_nus/NusceneQA_Llama_finetuning"
    )

