import GPT4_score
import torch
import json
import copy
from tqdm import tqdm
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from transformers import BitsAndBytesConfig

if __name__ == "__main__":

    with open("MCQs_Eval.txt", "r") as f:
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


    json_path = "/media/workstation/6D3563AC52DC77EA/Data/DriveLM/data/QA_dataset_nus/test_benchmark.json"
    result_path = "/media/workstation/6D3563AC52DC77EA/Data/DriveLM/data/QA_dataset_nus/test_benchmark_gpt_score.json"

    # Load JSON file
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")

    result_data = copy.deepcopy(json_data)

    # 设置批量写入的间隔（比如每10个问题写一次）
    batch_size = 10
    processed_count = 0  # 记录处理的问题数量

    max_sessions = 2  # 你可以根据需要设置这个值
    processed_sessions = 0  # 记录处理的会话数量

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
                images[i] = image.replace('../', "/media/workstation/6D3563AC52DC77EA/Data/DriveLM/data/")

            for section_name, questions in qa_questions.items():
                for question in questions:

                    Question = question.get("Q")
                    Desc = "The question evaluates the decision-making capabilities of a self-driving car when encountering a complex intersection, focusing on traffic rules, object recognition, environmental awareness, and reasoning clarity."
                    GT = question.get("A")
                    PRED = question.get("A_GPT")

                    MCQs_Evaluation_text = template.replace("<<QUESTION>>", Question).replace("<<DESC>>", Desc).replace(
                        "<<GT>>",
                        GT).replace(
                        "<<PRED>>", PRED)

                    # Perform inference using LLaMA model
                    response = llama_evaluator.score(MCQs_Evaluation_text)
                    print("LLaMA Model Response:")
                    print(response)

                    # 添加 GPT 回答到每个问题字典中
                    question['GPT_Score'] = response

                    # 更新 question 为有序字典
                    question = {k: question[k] for k in sorted(question)}

                    processed_count += 1

                    # 每处理一定数量的问题，批量写入一次
                    if processed_count % batch_size == 0:
                        print(f"Processed {processed_count} questions, writing to file...")
                        with open(result_path, 'w', encoding='utf-8') as f:
                            json.dump(result_data, f, indent=4, ensure_ascii=False)

                    # 打印输出处理结果（调试）
                    print(f"Processed question: {Question}")
                    print(f"Original Answer: {GT}")
                    print(f"GPT Score: {response}")
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
