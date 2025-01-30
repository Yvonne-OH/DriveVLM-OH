import Evaluator.GPT_Score.GPT4_score as GPT4_score
import torch
import json
import copy
import Util.util as util
from tqdm import tqdm
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from transformers import BitsAndBytesConfig

if __name__ == "__main__":

    with open("Open-Ended Questions_Eval.txt", "r") as f:
        template = f.read()

    # Initialize LLaMA model
    model_name = '/media/workstation/6D3563AC52DC77EA/Model/meta-llama/Llama-3.2-3B-Instruct'
    finetuning_path = None
    max_memory = {0: "22GB", 1: "7.6GB"}
    Temperature = 0.5
    Top_p = 0.3

    # Define custom parameters
    custom_bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16  # Use float16 instead of bfloat16 for testing
    )
    custom_device_map = "auto"  # Automatically allocate model across available GPUs
    custom_torch_dtype = torch.float16  # Use float16 precision for speed

    # Pass the custom parameters via kwargs
    llama_evaluator = GPT4_score.GPT_SCORE(
        model=model_name,
        Temperature=Temperature,
        Top_p=Top_p
    )

    llama_evaluator.Llama_Eval_init(
        model_name=model_name,
        finetuning_path=finetuning_path,
        max_memory=max_memory,
        bnb_config=custom_bnb_config,
        device_map=custom_device_map,
        torch_dtype=custom_torch_dtype
    )


    json_path = "LingoQA_benchmark_val_result_1.json"
    result_path = "LingoQA_benchmark_result_score.json"
    task_desc_path = "Open-Ended Questions_Eval.txt"

    # Load JSON data
    try:
        with open(json_path, 'r') as file:
            json_data = json.load(file)
        with open(task_desc_path, 'r', encoding='utf-8') as file:
            Desc  = file.read()
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        exit()


    result_data = copy.deepcopy(json_data)

    # Batch processing settings
    batch_size = 10
    processed_count = 0

    max_sessions = 2
    processed_sessions = 0

    # Iterate through scenes
    for scene_id, scene_content in tqdm(result_data.items(), desc="Processing Scenes", total=len(result_data)):
        if processed_sessions >= max_sessions:
            print("Reached the maximum number of sessions specified:", max_sessions)
            break

        QA_list = scene_content['questions']
        for QA in QA_list:
            Question = QA['question']
            GT = QA['GT_answer']
            PRED = QA['Gpt_answer']

            # Perform inference and replace placeholder in the template
            MCQs_Evaluation_text = Desc.replace("<<QUESTION>>", Question).replace("<<GT>>", GT).replace("<<PRED>>",PRED)

            response = llama_evaluator.score(MCQs_Evaluation_text)
            #response = "TEST TEST TEST"
            print(response)

            # Store the generated answer
            QA['Reason'] = response
            processed_count += 1

            if processed_count % batch_size == 0:
                print(f"Processed {processed_count} questions, writing to file...")
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, indent=4, ensure_ascii=False)

        processed_sessions += 1
        result_data[scene_id] = scene_content

    # Write remaining results if not divisible by batch_size
    if processed_count % batch_size != 0:
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)

