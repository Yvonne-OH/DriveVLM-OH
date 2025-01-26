import  Util.util as util
import json
import argparse
import tqdm
import Model.llava.llava_multimodal_interface as llava

image_folder_path = "/media/workstation/6D3563AC52DC77EA/Data/LingoQA/train/Scene_images/"
data_path = "/media/workstation/6D3563AC52DC77EA/Data/LingoQA/train/Scene_train.json"
model_path = "/media/workstation/6D3563AC52DC77EA/Model/llava/llava-v1.6-mistral-7b-hf"

with open(data_path, 'r') as f:
    data = json.load(f)

processor = util.ImagePreprocessor()
llava_chatbot = llava.Llava_Interface(model_path)

count = 0

for scene_id, scene_content in tqdm.tqdm(data.items(), desc="Processing Scenes", total=len(data)):

    count += 1
    if count >10:
        break

    llava_conservation = []

    image_subfolder_path = image_folder_path + scene_id + "/"
    image_list = [image_folder_path + _ for _ in  scene_content['image_id']]
    QA_list = scene_content['questions']

    for QA in QA_list:
        Q = QA['question']
        A = QA['GT_answer']


    #merged_images_horizontal = processor.merge_vehicle_camera_views(image_paths=image_list , merge='horizontal',)
    #merged_images_horizontal[0].save("merged_horizontal.jpg")


    conversation_1 = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
    ]

    # 生成回复
    #response = llava_chatbot .generate_response(conversation_1, [merged_images_horizontal[0]])
    #print(response)

