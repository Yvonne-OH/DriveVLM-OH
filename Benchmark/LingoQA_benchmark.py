import Util.util as util
import json
import argparse
import tqdm
import Util.util as util
import Model.llava.llava_multimodal_interface as llava
import Model.llava.llava_util
import copy
import os
#export PYTHONPATH="$PYTHONPATH:/home/workstation/DriveVLM-OH/"
#python LingoQA_benchmark.py --image_folder_path "/media/workstation/6D3563AC52DC77EA/Data/LingoQA/train/Scene_images/" --data_path "/media/workstation/6D3563AC52DC77EA/Data/LingoQA/train/Scene_train.json" --model_path "/media/workstation/6D3563AC52DC77EA/Model/llava/llava-v1.6-mistral-7b-hf" --task_desc_path "/home/workstation/DriveVLM-OH/Benchmark/LingoQA_DESC.txt" --benchmark_result_path "/home/workstation/DriveVLM-OH/Benchmark/LingoQA_benchmark_result.json" --verbose 0

image_folder_path = "/media/workstation/6D3563AC52DC77EA/Data/LingoQA/val/images/"
data_path = "/media/workstation/6D3563AC52DC77EA/Data/LingoQA/val/val.json"
model_path = "/media/workstation/6D3563AC52DC77EA/Model/llava/llava-v1.6-mistral-7b-hf"
Task_DESC_path =  "/home/workstation/DriveVLM-OH/Benchmark/LingoQA_DESC.txt"
Benchmark_result_path = "/home/workstation/DriveVLM-OH/Benchmark/LingoQA_benchmark_val_result.json"
verbose = 1

if __name__ == "__main__":

    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Process scenes and generate responses using LLaVA model.")
    parser.add_argument("--image_folder_path", type=str, required=True,
                        help="Path to the folder containing image subfolders.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the JSON file containing data.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the LLaVA model.")
    parser.add_argument("--task_desc_path", type=str, required=True,
                        help="Path to the file containing task description.")
    parser.add_argument("--benchmark_result_path", type=str, required=True,
                        help="Path where the benchmark results will be saved.")
    parser.add_argument("--verbose", type=int, required=True,
                        help="Provides additional information during the execution.")
    parser.add_argument("--max_scene", type=int, required=False,
                        help="Provides max_scene.")
    args = parser.parse_args()

    # Load data and task description
    with open(args.data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    with open(args.task_desc_path, 'r', encoding='utf-8') as file:
        task_desc = file.read()

    try:
        if os.path.exists(args.benchmark_result_path):
            os.remove(args.benchmark_result_path)
    except OSError as e:
        print(f"Error: {e.strerror}. Path: {e.filename}")

    # Deep copy the original data to avoid mutating it during processing
    results = copy.deepcopy(data)

    # Initialize the image processor and chatbot interface
    processor = util.ImagePreprocessor()
    llava_chatbot = llava.Llava_Interface(args.model_path)

    # Process each scene up to a maximum of 10
    count = 0
    for scene_id, scene_content in tqdm.tqdm(data.items(), desc="Processing Scenes", total=len(data)):

        if args.max_scene is not None and count >= args.max_scene:
            print("Reached the maximum number of scenes specified:", args.max_scenes)
            break
        count += 1

        # Construct the path for images and load images
        image_subfolder_path = image_folder_path + scene_id + "/"
        image_list = [image_folder_path + _ for _ in scene_content['image_id']]
        QA_list = scene_content['questions']

        for QA in QA_list:
            Q = QA['question']
            A = QA['GT_answer']

            # Prepare the conversation context for LLaVA
            llava_conversation = []
            Model.llava.llava_util.llava_conversation(llava_conversation, "system",
                                                util.extract_between_markers(task_desc, "[DESC_START]", "[DESC_END]")[
                                                    0])
            merged_images_horizontal = processor.merge_vehicle_camera_views(image_paths=image_list, merge='horizontal')
            merged_images_horizontal[0].save("merged_horizontal.jpg")
            Model.llava.llava_util.llava_conversation(llava_conversation, "user",
                                                Q + " Please describe your answer concisely and ensure it is precise and specific. Avoid using vague or ambiguous language.",
                                                is_image=1)

            # Generate the response using the LLaVA model
            response = llava_chatbot.generate_response(llava_conversation, [merged_images_horizontal[0]])

            # Print the question and the generated answer for verification
            if  args.verbose:
                print(f"Question: {Q}")
                print("-----")
                print(f"Ground Truth Answer: {A}")
                print("-----")
                print(f"Generated Answer: {response}")
                print("*" * 50)

            # Store the generated answer
            QA['Gpt_answer'] = response

        # Update results in the main data structure
        data[scene_id] = scene_content

        # Save results incrementally after each scene
        with open(args.benchmark_result_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print("Test results saved to:", args.benchmark_result_path)
