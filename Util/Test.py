import json
import copy
from tqdm import tqdm

json_path = "/media/workstation/6D3563AC52DC77EA/Data/DriveLM/data/QA_dataset_nus/test.json"
result_path = "/media/workstation/6D3563AC52DC77EA/Data/DriveLM/data/QA_dataset_nus/test_benchmark.json"


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
                q = question.get("Q")
                a = question.get("A")

                # 添加 GPT 回答到每个问题字典中
                question['A_GPT'] = q

                # 更新 question 为有序字典
                question = {k: question[k] for k in sorted(question)}

                processed_count += 1

                # 每处理一定数量的问题，批量写入一次
                if processed_count % batch_size == 0:
                    print(f"Processed {processed_count} questions, writing to file...")
                    with open(result_path, 'w', encoding='utf-8') as f:
                        json.dump(result_data, f, indent=4, ensure_ascii=False)

                # 打印输出处理结果（调试）
                print(f"Processed question: {q}")
                print(f"Original Answer: {a}")
                print(f"GPT Answer: {question['A_GPT']}")
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



