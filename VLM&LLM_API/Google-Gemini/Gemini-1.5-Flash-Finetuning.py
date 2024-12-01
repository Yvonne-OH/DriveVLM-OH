import os
import json
import time
import pandas as pd

import google.generativeai as genai
from DataLoader.convert2Gemini import GminiJson2List

genai.configure(api_key="AIzaSyCQSNKK5sH4yN87JQFnyEVQVhsOcam8VII")

Json_path = "/media/oh/0E4A12890E4A1289/DriveLM/data/QA_dataset_nus/test_eval.json"
Image_path = "/media/oh/0E4A12890E4A1289/DriveLM/data/"
Save_path = "/media/oh/0E4A12890E4A1289/DriveLM/data/QA_dataset_nus/test_Gemini.json"

API_key="AIzaSyCQSNKK5sH4yN87JQFnyEVQVhsOcam8VII"
Sample_num = 2
root = Save_path
genai.configure(api_key=API_key)

files_list = []
response_list = []

# 配置 API 密钥
genai.configure(api_key="YOUR_NEW_API_KEY")  # 替换为新的 API 密钥

# 文件上传示例
def upload_to_gemini(path, mime_type=None):
    try:
        file = genai.upload_file(path, mime_type=mime_type)
        print(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file
    except Exception as e:
        print(f"Error uploading file: {e}")
        raise

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

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
)

Files, training_data = GminiJson2List (Save_path,Image_path, 2)
response = model.generate_content(training_data)

base_model = "models/gemini-1.5-flash-001-tuning"

operation = genai.create_tuned_model(
    # You can use a tuned model here too. Set `source_model="tunedModels/..."`
    display_name="increment",
    source_model=base_model,
    epoch_count=20,
    batch_size=4,
    learning_rate=0.001,
    training_data=training_data,
)

for status in operation.wait_bar():
    time.sleep(10)

result = operation.result()
print(result)
# # You can plot the loss curve with:
snapshots = pd.DataFrame(result.tuning_task.snapshots)
#sns.lineplot(data=snapshots, x='epoch', y='mean_loss')

model = genai.GenerativeModel(model_name=result.name)
result = model.generate_content("III")
print(result.text)  # IV