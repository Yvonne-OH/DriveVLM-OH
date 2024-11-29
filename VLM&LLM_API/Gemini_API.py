import os

# Suppress logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"  # 0: INFO, 1: WARNING, 2: ERROR, 3: FATAL

import google.generativeai as genai
from PIL import Image

# 配置 Google Generative AI 的 API 密钥
genai.configure(api_key="AIzaSyAeDUmN1KvGENCHJh1_jDDI8bea215ZM3I")

# 配置生成模型
model = genai.GenerativeModel("gemini-1.5-flash")

# 设置生成配置
Responses_config = genai.types.GenerationConfig(
    candidate_count=1,
    stop_sequences=["x"],
    max_output_tokens=20,
    temperature=1.0,
)

# 打开图像
organ = Image.open("media/car.png")

# 生成内容
response = model.generate_content(
    ["Tell me about this instrument", organ],
    generation_config=Responses_config,
    stream=True
)

# 处理响应流
for chunk in response:
    print(chunk.text)
    print("_" * 80)
