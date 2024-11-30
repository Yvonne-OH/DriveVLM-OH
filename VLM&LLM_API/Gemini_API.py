import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageColor
import google.generativeai as genai
from Util.util import check_and_fix_json

# Suppress logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"  # 0: INFO, 1: WARNING, 2: ERROR, 3: FATAL

# 生成配置
Responses_config = genai.types.GenerationConfig(
    candidate_count=1,
    stop_sequences=["x"],
    max_output_tokens=200,  # 增加 token 数，确保完整返回边界框信息
    temperature=1.0,
)

bbox_str = None
bbox_hist = ''

additional_colors = [
    colorname for (colorname, colorcode) in ImageColor.colormap.items()
]


def Model_initialize(Api_key, Model_name):
    """
    初始化模型
    """
    genai.configure(api_key=Api_key)
    model = genai.GenerativeModel(Model_name)
    return model

# @title Parsing utils
def parse_list_boxes(text):
  result = []
  for line in text.strip().splitlines():
    # Extract the numbers from the line, remove brackets and split by comma
    try:
      numbers = line.split('[')[1].split(']')[0].split(',')
    except:
      numbers =  line.split('- ')[1].split(',')

    # Convert the numbers to integers and append to the result
    result.append([int(num.strip()) for num in numbers])

  return result


def parse_list_boxes_with_label(text):
    """
    修复和解析 JSON 格式边界框数据，删除未完成的部分。
    """
    try:
        # 清理输入数据
        #text = text.strip("```").strip("json").strip().replace("'", '"').replace(",}", "}")

        # 检测 JSON 的完整性
        parsed_data=check_and_fix_json(text)

        return parsed_data
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}. Returning empty result.")
        return {}

def postproc_bbox_str(height, width):
    """
    后处理边界框字符串，将百分比坐标转换为绝对像素坐标
    :param height:
    :param width:
    :return: str
    """
    x0, y0, x1, y1 = [float(x) for x in bbox_str.removeprefix('[').removesuffix(']').split(',')]
    x0 = int(np.round(x0 / width * 1000))
    y0 = int(np.round(y0 / height * 1000))
    x1 = int(np.round(x1 / width * 1000))
    y1 = int(np.round(y1 / height * 1000))
    return f'{y0} {x0} {y1} {x1}'

def postproc_bbox_hist(height, width):
    """
    后处理边界框字符串，将百分比坐标转换为绝对像素坐标
    :param height:
    :param width:
    :return:
    """
    bbox_strs = bbox_hist.rstrip().split("\n")
    results = []
    for bbox_str in bbox_strs:
        x0, y0, x1, y1 = [float(x) for x in bbox_str.removeprefix('[').removesuffix(']').split(',')]
        x0 = int(np.round(x0 / width * 1000))
        y0 = int(np.round(y0 / height * 1000))
        x1 = int(np.round(x1 / width * 1000))
        y1 = int(np.round(y1 / height * 1000))
        results.append(f" (x = {(x0 + x1) // 2}, y = {(y0 + y1)//2}); ")
    return results

def plot_bounding_boxes(im, noun_phrases_and_positions):
    """
    绘制边界框和文字标注
    """
    # 加载图像
    img = im
    width, height = img.size

    # 创建绘制对象
    draw = ImageDraw.Draw(img)

    # 定义颜色列表
    colors = [
        "red",
        "green",
        "blue",
        "yellow",
        "orange",
        "pink",
        "purple",
        "brown",
        "gray",
        "beige",
        "turquoise",
        "cyan",
        "magenta",
        "lime",
        "navy",
        "maroon",
        "teal",
        "olive",
        "coral",
        "lavender",
        "violet",
        "gold",
        "silver",
    ] + additional_colors

    # 遍历名词短语及其对应位置
    for i, (label, coords) in enumerate(bounding_boxes.items()):
        # 选择颜色
        color = colors[i % len(colors)]

        # 解包边界框坐标
        y1, x1, y2, x2 = coords

        # 转换为绝对像素坐标
        abs_x1 = int(x1 / 1000 * width)
        abs_y1 = int(y1 / 1000 * height)
        abs_x2 = int(x2 / 1000 * width)
        abs_y2 = int(y2 / 1000 * height)

        # 绘制边界框
        draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)

        # 绘制标签文字
        draw.text((abs_x1 + 8, abs_y1 + 6), label, fill=color)

    # 显示图像
    img.show()

def get_bounding_boxes(image_path, model, prompt):
    """
    调用生成模型，获取边界框信息
    """
    try:
        # 打开图像
        image = Image.open(image_path)
        width, height = image.size

        # 调用生成模型
        response = model.generate_content(
            [image, prompt], generation_config=Responses_config
        )

        #print ("Response:", response.text)

        # 解析边界框
        boxes = parse_list_boxes_with_label(response.text)  # 使用解析工具
        print("Parsed bounding boxes:", boxes)

        return boxes
    except Exception as e:
        print(f"Error processing image or model response: {e}")
        return []


if __name__ == "__main__":
    # 配置参数
    Api_key = "AIzaSyAeDUmN1KvGENCHJh1_jDDI8bea215ZM3I"
    Model_name = "gemini-1.5-pro"
    sample_file_1 = "media/traffic.png"
    prompt =     (
        "Return bounding boxes for cars, and trees in the"
        " following format as a list. \n {'car_0' : [ymin, xmin, ymax,"
        " xmax], ...} \n If there are more than one instance of an object, add"
        " them to the dictionary as 'object_0', 'object_1', etc."
    )

    # 初始化模型
    model = Model_initialize(Api_key, Model_name)

    # 获取边界框
    bounding_boxes = get_bounding_boxes(sample_file_1, model, prompt)

    # 加载图像并绘制边界框
    image = Image.open(sample_file_1)

    plot_bounding_boxes(image, bounding_boxes)
