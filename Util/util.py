import json


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
    fixed_json = check_and_fix_json(response_text)
    print("Fixed JSON:", fixed_json)

