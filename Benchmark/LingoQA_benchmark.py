import Util.util as util
import json
import argparse
import tqdm
import Util.util as util
import Model.llava.llava_multimodal_interface as llava
import Model.llava.llava_util
import copy

image_folder_path = "/media/workstation/6D3563AC52DC77EA/Data/LingoQA/train/Scene_images/"
data_path = "/media/workstation/6D3563AC52DC77EA/Data/LingoQA/train/Scene_train.json"
model_path = "/media/workstation/6D3563AC52DC77EA/Model/llava/llava-v1.6-mistral-7b-hf"
Task_DESC_path =  "/home/workstation/DriveVLM-OH/Benchmark/LingoQA_DESC.txt"
Benchmark_result_path = "/home/workstation/DriveVLM-OH/Benchmark/LingoQA_benchmark_result.json"

with open(data_path, 'r') as f:
    data = json.load(f)

with open(Task_DESC_path, 'r') as f:
    task_desc = f.read()

results = copy.deepcopy(data)

processor = util.ImagePreprocessor()
llava_chatbot = llava.Llava_Interface(model_path)

count = 0

for scene_id, scene_content in tqdm.tqdm(data.items(), desc="Processing Scenes", total=len(data)):

    count += 1
    if count > 10:
        break


    image_subfolder_path = image_folder_path + scene_id + "/"
    image_list = [image_folder_path + _ for _ in scene_content['image_id']]
    QA_list = scene_content['questions']

    for QA in QA_list:
        Q = QA['question']
        A = QA['GT_answer']


        llava_conservation = []
        Model.llava.llava_util.llava_conversation(llava_conservation, "system",
                                                  util.extract_between_markers(task_desc, "[DESC_START]", "[DESC_END]")[0])
        merged_images_horizontal = processor.merge_vehicle_camera_views(image_paths=image_list , merge='horizontal',)
        merged_images_horizontal[0].save("merged_horizontal.jpg")
        Model.llava.llava_util.llava_conversation(llava_conservation, "user",Q+"Please describe your answer concisely and ensure it is precise and specific. Avoid using vague or ambiguous language.", is_image=1)

        # 生成回复
        response = llava_chatbot .generate_response( llava_conservation, [merged_images_horizontal[0]])
        # Output for verification
        print(f"Question: {Q}")
        print("-----")
        print(f"Ground Truth Answer: {A}")
        print("-----")
        print(f"Generated Answer: {response}")
        print("*" * 50)

        # Store results
        QA['Gpt_answer'] = response

    # Update results in the main data structure
    data[scene_id] = scene_content

    # Save results incrementally after each scene
    with open(Benchmark_result_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print("Test results saved to:", Benchmark_result_path)

