import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor


class Gemini_ImageProcessor:
        def __init__(self):
            self.additional_colors = [
                colorname for (colorname, colorcode) in ImageColor.colormap.items()
            ]

            self.bbox_str = None
            self.bbox_hist = ''

        @staticmethod
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

        @staticmethod
        def parse_list_boxes(text):
            """
                解析给定的文本并提取嵌套列表结构，返回一个包含整数列表的列表。

                参数：
                    text (str): 包含嵌套列表的文本，每行表示一个列表，可能以不同格式表示。

                返回：
                    list[list[int]]: 解析后的嵌套整数列表。
                """
            result = []  # 用于存储解析后的结果列表
            for line in text.strip().splitlines():  # 遍历文本中的每一行
                try:
                    # 尝试从方括号内提取数字
                    numbers = line.split('[')[1].split(']')[0].split(',')
                except:
                    # 如果方括号解析失败，尝试从 '- ' 后提取数字
                    numbers = line.split('- ')[1].split(',')

                # 将提取到的数字字符串转换为整数，并存入结果列表
                result.append([int(num.strip()) for num in numbers])

            return result  # 返回解析后的结果列表

        def plot_bounding_boxes(self, im, bounding_boxes):
            img = im
            width, height = img.size
            draw = ImageDraw.Draw(img)

            colors = [
                         "red", "green", "blue", "yellow", "orange", "pink", "purple", "brown", "gray", "beige",
                         "turquoise",
                         "cyan", "magenta", "lime", "navy", "maroon", "teal", "olive", "coral", "lavender", "violet",
                         "gold", "silver"
                     ] + self.additional_colors

            for i, (label, coords) in enumerate(bounding_boxes.items()):
                color = colors[i % len(colors)]

                y1, x1, y2, x2 = coords

                abs_x1 = int(x1 / 100 * width)
                abs_y1 = int(y1 / 100 * height)
                abs_x2 = int(x2 / 100 * width)
                abs_y2 = int(y2 / 100 * height)

                draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)
                draw.text((abs_x1 + 8, abs_y1 + 6), label, fill=color)

            img.show()

        def parse_list_boxes_with_label(self,text):
            """
            修复和解析 JSON 格式边界框数据，删除未完成的部分。
            """
            try:
                return self.check_and_fix_json(text)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}. Returning empty result.")
                return {}

        def postproc_bbox_hist(self, height, width):
            """
            后处理边界框字符串，将百分比坐标转换为绝对像素坐标
            :param height:
            :param width:
            :return:
            """
            bbox_strs = self.bbox_hist.rstrip().split("\n")
            results = []
            for bbox_str in bbox_strs:
                x0, y0, x1, y1 = [float(x) for x in bbox_str.removeprefix('[').removesuffix(']').split(',')]
                x0 = int(np.round(x0 / width * 1000))
                y0 = int(np.round(y0 / height * 1000))
                x1 = int(np.round(x1 / width * 1000))
                y1 = int(np.round(y1 / height * 1000))
                results.append(f" (x = {(x0 + x1) // 2}, y = {(y0 + y1) // 2}); ")
            return results

        def plot_bounding_boxes(self,im, bounding_boxes):
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
                         "red", "green", "blue", "yellow", "orange", "pink", "purple", "brown", "gray", "beige",
                         "turquoise",
                         "cyan", "magenta", "lime", "navy", "maroon", "teal", "olive", "coral", "lavender", "violet",
                         "gold", "silver"
                     ] + self.additional_colors

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


class DataPreprocessor:
    def __init__(self, image_paths: list = None, images: list = None,resize_to: tuple = None,max_dimensions: tuple = None):
        self.image_paths = image_paths
        self.images = images
        self.resize_to = resize_to
        self.max_dimensions = max_dimensions

    def _load_images(self):
        """
        Load images from paths or directly from the provided list.
        """
        processed_images = []
        all_image_paths = self.image_paths + [img for img in self.images if isinstance(img, Image.Image)]

        for img_path in all_image_paths:
            if isinstance(img_path, str):
                if not os.path.exists(img_path):
                    raise ValueError(f"Image path does not exist: {img_path}")
                img = Image.open(img_path)
            elif isinstance(img_path, Image.Image):
                img = img_path
            else:
                raise ValueError(f"Invalid image or path: {img_path}")

            if self.resize_to:
                img = img.resize(self.resize_to)

            processed_images.append(img.convert("RGB"))
        return processed_images

    def arrange_images_in_logical_order(self, processed_images: list) -> list:
        """
        Arrange images into a fixed logical order based on their input order.
        """
        fixed_order = [
            "front_left",  # processed_images[1]
            "front",  # processed_images[0]
            "front_right",  # processed_images[2]
            "back_left",  # processed_images[4]
            "back",  # processed_images[3]
            "back_right"  # processed_images[5]
        ]

        ordered_indices = [1, 0, 2, 4, 3, 5]  # Indices corresponding to the fixed order
        ordered_images = [processed_images[idx] for idx in ordered_indices]
        return ordered_images\


    def _add_borders_and_numbering(self, processed_images: list) -> list:
        """
        Add borders and numbering to each image.
        """
        bordered_images = []
        for idx, img in enumerate(processed_images):
            border_size = 10  # Size of the border
            bordered_img = Image.new("RGB", (img.width + 2 * border_size, img.height + 2 * border_size), "black")
            bordered_img.paste(img, (border_size, border_size))

            # Add numbering with dynamic font size
            draw = ImageDraw.Draw(bordered_img)
            font_size = max(20, min(img.width, img.height) // 10)  # Dynamic font size based on image size
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()  # Fallback to default font if arial.ttf is not available
            draw.text((10, 10), f"{idx + 1}", fill="red", font=font)
            bordered_images.append(bordered_img)
        return bordered_images

    def arrange_images_in_logical_order(processed_images: list) -> list:
        """
        Arrange images into a fixed logical order based on their input order.
        """
        # Define the fixed order of the images
        fixed_order = [
            "front_left",  # processed_images[1]
            "front",  # processed_images[0]
            "front_right",  # processed_images[2]
            "back_left",  # processed_images[4]
            "back",  # processed_images[3]
            "back_right"  # processed_images[5]
        ]

        # Reorder images based on fixed logic
        ordered_indices = [1, 0, 2, 4, 3, 5]  # Indices corresponding to the fixed order
        ordered_images = [processed_images[idx] for idx in ordered_indices]

        return ordered_images

    def merge_vehicle_camera_views(self, merge: str = None, grid_size: tuple = None):

        processed_images = []

        image_paths = self.image_paths
        images = self.images
        resize_to = self.resize_to
        max_dimensions = self.max_dimensions

        # Check if all images have the same dimensions before processing
        all_image_paths = []
        if image_paths is not None:
            all_image_paths.extend(image_paths)
        if images is not None:
            all_image_paths.extend(images)

        if len(all_image_paths) > 1:
            dimensions = []
            for img_path in all_image_paths:
                if isinstance(img_path, str) and os.path.exists(img_path):
                    img = Image.open(img_path)
                elif isinstance(img_path, Image.Image):
                    img = img_path
                else:
                    raise ValueError(f"Invalid image or path: {img_path}")
                dimensions.append(img.size)

            if len(set(dimensions)) > 1:
                raise ValueError("All images must have the same dimensions before further processing.")

        if images is not None:
            for img in images:
                if resize_to:
                    img = img.resize(resize_to)
                processed_images.append(img.convert("RGB"))

        if image_paths is not None:
            for path in image_paths:
                if os.path.exists(path):
                    img = Image.open(path)
                    if resize_to:
                        img = img.resize(resize_to)
                    processed_images.append(img.convert("RGB"))
                else:
                    raise ValueError(f"Image path does not exist: {path}")

        if not processed_images:
            raise ValueError("No valid images provided")

        if merge:
            # Add borders and numbering to each image
            bordered_images = []
            for idx, img in enumerate(processed_images):
                border_size = 10  # Size of the border
                bordered_img = Image.new("RGB", (img.width + 2 * border_size, img.height + 2 * border_size), "black")
                bordered_img.paste(img, (border_size, border_size))

                # Add numbering with dynamic font size
                draw = ImageDraw.Draw(bordered_img)
                font_size = max(20, min(img.width, img.height) // 10)  # Dynamic font size based on image size
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except IOError:
                    font = ImageFont.load_default()  # Fallback to default font if arial.ttf is not available
                draw.text((10, 10), f"{idx + 1}", fill="red", font=font)
                bordered_images.append(bordered_img)

            # Update processed_images to bordered_images
            processed_images = bordered_images

            # Determine merging layout
            if merge == 'auto':
                num_images = len(processed_images)
                rows = int(num_images ** 0.5)
                cols = (num_images + rows - 1) // rows  # Ensure all images fit
                grid_size = (rows, cols)
                merge = 'grid'

            if merge == 'horizontal':
                # Horizontal layout
                total_width = sum(img.width for img in processed_images)
                total_height = max(img.height for img in processed_images)
                merged_image = Image.new("RGB", (total_width, total_height), "white")

                x_offset = 0
                for img in processed_images:
                    merged_image.paste(img, (x_offset, 0))
                    x_offset += img.width

            elif merge == 'vertical':
                # Vertical layout
                total_width = max(img.width for img in processed_images)
                total_height = sum(img.height for img in processed_images)
                merged_image = Image.new("RGB", (total_width, total_height), "white")

                y_offset = 0
                for img in processed_images:
                    merged_image.paste(img, (0, y_offset))
                    y_offset += img.height

            elif merge == 'grid' and grid_size:
                # Grid layout
                rows, cols = grid_size
                cell_width = max(img.width for img in processed_images)
                cell_height = max(img.height for img in processed_images)

                total_width = cols * cell_width
                total_height = rows * cell_height
                merged_image = Image.new("RGB", (total_width, total_height), "white")

                for idx, img in enumerate(processed_images):
                    row = idx // cols
                    col = idx % cols
                    x_offset = col * cell_width
                    y_offset = row * cell_height
                    merged_image.paste(img, (x_offset, y_offset))

            if merge == 'custom_grid':
                # Reorder processed images using fixed order
                processed_images = arrange_images_in_logical_order(processed_images)

                # Create the grid layout
                grid_layout = [
                    ("front_left", "front", "front_right"),
                    ("back_left", "back", "back_right")
                ]

                rows, cols = len(grid_layout), len(grid_layout[0])
                cell_width = max(img.width for img in processed_images)
                cell_height = max(img.height for img in processed_images)

                total_width = cols * cell_width
                total_height = rows * cell_height
                merged_image = Image.new("RGB", (total_width, total_height), "white")

                # Paste images in grid layout
                for idx, img in enumerate(processed_images):
                    row, col = divmod(idx, cols)
                    x_offset = col * cell_width
                    y_offset = row * cell_height
                    merged_image.paste(img, (x_offset, y_offset))

            else:

                raise ValueError("Invalid merge option or missing grid_size for grid layout")

            # Scale merged image if it exceeds max_dimensions
            if max_dimensions:
                max_width, max_height = max_dimensions
                if merged_image.width > max_width or merged_image.height > max_height:
                    scale_ratio = min(max_width / merged_image.width, max_height / merged_image.height)
                    new_width = int(merged_image.width * scale_ratio)
                    new_height = int(merged_image.height * scale_ratio)
                    merged_image = merged_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            return [merged_image]

        return processed_images


















def assistant(content: str):
    return { "role": "assistant", "content": content }


def user_input(prompt_text: str, images: list = None):
    """
    构造用户输入，可以包含文本和图片。

    Args:
        prompt_text (str): 用户输入的文本内容。
        images (list, optional): 图片 URL 列表。如果没有图片，默认为 None。

    Returns:
        dict: 用户输入的结构化消息。
    """
    content = []
    if images:
        # 添加图片内容
        content.extend([{"type": "image"} for _ in images])
    # 添加文本内容
    content.append({"type": "text", "text": prompt_text})

    return {"role": "user", "content": content}

def system(content: str, type ):
    if type == 'Fine_tuning':
        return { "from": "system", "value": content}
    else:
        return { "role": "system", "content": content }


def convert_to_percentage(base_width,base_height, match):

    # 提取数值并计算百分比
    values = match.group().split(",")
    if len(values) == 4:
        x_percent = float(values[2]) / base_width * 100
        y_percent = float(values[3].strip(">")) / base_height * 100
        return f"<{values[0]},{values[1]},{x_percent:.2f}%,{y_percent:.2f}%>"
    return match.group()