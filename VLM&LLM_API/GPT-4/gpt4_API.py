import json
import os
from tqdm import tqdm
import base64
from openai import OpenAI

# Suppress logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"



client = OpenAI()

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "path_to_your_image.jpg"

# Getting the base64 string
base64_image = encode_image(image_path)

response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is in this image?",
        },
        {
          "type": "image_url",
          "image_url": {
            "url":  f"data:image/jpeg;base64,{base64_image}"
          },
        },
      ],
    }
  ],
)

print(response.choices[0])


if __name__ == '__main__':
    # File paths
    Json_path = "/media/oh/0E4A12890E4A1289/DriveLM/data/QA_dataset_nus/test_eval.json"
    Image_path = "/media/oh/0E4A12890E4A1289/DriveLM/data/"
    Result_path = "/media/oh/0E4A12890E4A1289/DriveLM/data/QA_dataset_nus/test_result_GPT4.json"

    # Config
    Task_type = "behavior"


