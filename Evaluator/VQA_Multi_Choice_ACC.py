import json
import os
import re
import numpy as np
from tqdm import tqdm


def extract_answer_from_model_output(model_output):
    """
    从 Model_Output 提取答案。

    Args:
        model_output (str): Model_Output 字符串。

    Returns:
        str: 提取的答案（如 A, B, C, D），如果无法提取，返回 None。
    """
    try:
        # 使用正则表达式提取 <ans> 和 </ans> 之间的内容
        match = re.search(r"<ans>(.*?)</ans>", model_output, re.IGNORECASE)
        if match:
            answer_content = match.group(1).strip()
            # 尝试从内容中提取 A, B, C, D，包括带有句号或其他符号的格式
            match_choice = re.search(r"[A-D](?=[\.\s]|$)", answer_content, re.IGNORECASE)
            if match_choice:
                return match_choice.group(0).upper()  # 返回答案选项

        # 如果答案在 </ans> 前
        match_suffix = re.search(r"</ans>\s*([A-D])", model_output, re.IGNORECASE)
        if match_suffix:
            return match_suffix.group(1).upper()

        return None  # 如果没有找到答案
    except Exception as e:
        print(f"Error extracting answer: {e}")
        return None




if __name__ == "__main__":

    result_json_path = "/media/workstation/6D3563AC52DC77EA/Data/DriveLM/data/QA_dataset_nus/behavior_test_result_LLAMA3.2.json"

    try:
        with open (result_json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading result json: {e}")

    total_samples = 0
    correct_predictions = 0
    no_answer_count = 0

    # 遍历样本并统计正确率
    for sample in data:
        conversations = sample.get("conversations", [])

        # 提取 Output 和 Model_Output
        output = next((conv["value"] for conv in conversations if conv["from"] == "Output"), None)
        model_output = next((conv["value"] for conv in conversations if conv["from"] == "Model_Output"), None)

        if output and model_output:
            total_samples += 1
            # 提取答案
            extracted_answer = extract_answer_from_model_output(model_output)
            if extracted_answer:
                if extracted_answer == output.strip():
                    correct_predictions += 1
            else:
                print(extracted_answer)
                print(output)
                print(model_output)
                no_answer_count += 1

    # 计算正确率
    accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    print(f"Total Samples: {total_samples}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"No Answer Count: {no_answer_count}")
    print(f"Accuracy: {accuracy:.2f}%")

    





