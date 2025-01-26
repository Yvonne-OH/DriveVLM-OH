import base64
import re
import json
import ast
import openai
from PIL import Image
from Util.util import parse_list_boxes_with_label, plot_bounding_boxes

from PIL import Image, ImageDraw, ImageFont


openai.api_key = "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"

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "../media/traffic.png"


# Getting the base64 string
base64_image = encode_image(image_path)

response = openai.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
{
      "role": "system",
      "content": [
        {
          "type": "text",
          "text":
           """Coordinates are specified as percentages, adhering to the following conventions:
            The origin (0, 0) is located at the top-left corner of the image.
            The bottom-right corner of the image corresponds to (100, 100).
            The x-axis extends horizontally, increasing from left (0) to right (100).
            The y-axis extends vertically, increasing from top (0) to bottom (100)."""

        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Return bounding boxes for cars in the"
                  " following format as a list. \n {'car_0' : [ymin, xmin, ymax,"
                  " xmax], ...} \n If there are more than one instance of an object, add"
                  " them to the dictionary as 'object_0', 'object_1', etc."
                  "Just output the results as requested and don't add anything else! Wrap the answer in </ans> <ans/>."
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

print(response.choices[0].message.content)

text = response.choices[0].message.content
# 使用正则表达式提取 <ans> 和 </ans> 中的内容
match = re.search(r"</ans>(.*?)<ans/>", text, re.DOTALL)

if match:
    extracted_data = match.group(1).strip()  # 提取匹配到的内容并去掉两边的空格

    # 将字符串转换为字典
    data_dict = ast.literal_eval(extracted_data)

    # 将字典转换为 JSON 格式
    json_data = json.dumps(data_dict, indent=4, ensure_ascii=False)

    print("转换为 JSON 格式：")
    print(json_data)
else:
    print("未找到匹配的数据。")

print(parse_list_boxes_with_label(json_data))
bounding_boxes = parse_list_boxes_with_label(json_data)

# 加载图像并绘制边界框
image = Image.open(image_path)

plot_bounding_boxes(image, bounding_boxes)
