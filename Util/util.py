import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageColor
import google.generativeai as genai


additional_colors = [
    colorname for (colorname, colorcode) in ImageColor.colormap.items()
]

bbox_str = None
bbox_hist = ''


def check_and_fix_json(json_text):
    """
    检查 JSON 是否完整并修复不完整的部分。

    Args:
        json_text (str): 原始 JSON 文本

    Returns:
        dict: 修复后的 JSON 对象
    """
    try:
        # 清理 JSON 文本格式（将单引号替换为双引号，移除多余的换行和标记）
        json_text = json_text.strip("```").strip("json").strip().replace("'", '"')

        # 尝试直接解析 JSON
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"JSON is incomplete or invalid. Attempting to fix... Error: {e}")

        # 尝试修复 JSON
        last_valid_comma = json_text.rfind(',')
        last_valid_close_brace = json_text.rfind('}')

        # 如果没有找到有效的逗号或关闭括号，返回空字典
        if last_valid_comma == -1 and last_valid_close_brace == -1:
            return {}

        # 截断到最后一个有效的逗号
        fixed_text = json_text[:last_valid_comma] + '\n}'
        try:
            return json.loads(fixed_text)
        except json.JSONDecodeError as e:
            print(f"Unable to fix JSON: {e}")
            return {}

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

def plot_bounding_boxes(im, bounding_boxes):
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
        abs_x1 = int(x1 / 100 * width)
        abs_y1 = int(y1 / 100 * height)
        abs_x2 = int(x2 / 100 * width)
        abs_y2 = int(y2 / 100 * height)

        # 绘制边界框
        draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)

        # 绘制标签文字
        draw.text((abs_x1 + 8, abs_y1 + 6), label, fill=color)

    # 显示图像
    img.show()


if __name__ == '__main__':
    # 示例用法
    response_text = """```json
    {
      'car_0': [375, 0, 425, 41],
      'car_1': [465, 61, 545, 132],
      'car_2': [495, 209, 595, 311],
      'car_3': [453, 288, 529, 368],
      'car_4': [370, 458, 429, 516],
      'car_5': [319, 300, 365, 349],
      'car_6': [279, 213, 322, 258],
      'car_7': [170
    """
    # 检查并修复 JSON
    #fixed_json = check_and_fix_json(response_text)
    #print("Fixed JSON:", fixed_json)

    json_path = ""


