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

#__________________________________________________________________________________
#__________________________________________________________________________________
class DataPreprocessor:
    def __init__(self, image_paths: list = None, images: list = None,resize_to: tuple = None,max_dimensions: tuple = None):
        self.image_paths = image_paths if image_paths else []
        self.images = images if images else []
        self.resize_to = resize_to
        self.max_dimensions = max_dimensions

    def _load_images(self):
        """
        Load images from paths or directly from the provided list.
        """
        processed_images = []
        # Combine the image paths and PIL.Image objects into one list
        all_image_paths = self.image_paths + [img for img in self.images if isinstance(img, Image.Image)]

        for img_path in all_image_paths:
            if isinstance(img_path, str):
                # If it's a path, load the image
                if not os.path.exists(img_path):
                    raise ValueError(f"Image path does not exist: {img_path}")
                img = Image.open(img_path)
            elif isinstance(img_path, Image.Image):
                # If it's already a PIL image object, use it directly
                img = img_path
            else:
                raise ValueError(f"Invalid image or path: {img_path}")

            if self.resize_to:
                img = img.resize(self.resize_to)

            processed_images.append(img.convert("RGB"))
        return processed_images

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

    def arrange_images_in_logical_order(self,processed_images: list, logical_order: list = [1,0,2,4,3,5]) -> list:
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
        ordered_images = [processed_images[idx] for idx in logical_order]

        return ordered_images

    def _merge_images(self, processed_images: list, merge: str, grid_size: tuple = None, logical_order: list  = None) -> Image:
        """
        Merge the images based on the selected merge type (horizontal, vertical, grid, etc.)
        """

        if logical_order:
            processed_images = self.arrange_images_in_logical_order(processed_images, logical_order)

        if merge == 'horizontal':
            return self._merge_horizontal(processed_images)
        elif merge == 'vertical':
            return self._merge_vertical(processed_images)
        elif merge == 'grid' and grid_size:
            return self._merge_grid(processed_images, grid_size)
        elif merge == 'custom_grid':
            return self._merge_grid(processed_images, (2, 3))  # Assuming fixed 2x3 grid for custom grid
        else:
            raise ValueError("Invalid merge option or missing grid_size for grid layout")

    def _merge_horizontal(self, processed_images: list) -> Image:
        """
        Merge images in a horizontal layout.
        """
        total_width = sum(img.width for img in processed_images)
        total_height = max(img.height for img in processed_images)
        merged_image = Image.new("RGB", (total_width, total_height), "white")

        x_offset = 0
        for img in processed_images:
            merged_image.paste(img, (x_offset, 0))
            x_offset += img.width
        return merged_image

    def _merge_vertical(self, processed_images: list) -> Image:
        """
        Merge images in a vertical layout.
        """
        total_width = max(img.width for img in processed_images)
        total_height = sum(img.height for img in processed_images)
        merged_image = Image.new("RGB", (total_width, total_height), "white")

        y_offset = 0
        for img in processed_images:
            merged_image.paste(img, (0, y_offset))
            y_offset += img.height
        return merged_image

    def _merge_grid(self, processed_images: list, grid_size: tuple) -> Image:
        """
        Merge images in a grid layout.
        """

        rows, cols = grid_size
        cell_width = max(img.width for img in processed_images)
        cell_height = max(img.height for img in processed_images)

        total_width = cols * cell_width
        total_height = rows * cell_height
        merged_image = Image.new("RGB", (total_width, total_height), "white")

        for idx, img in enumerate(processed_images):
            row, col = divmod(idx, cols)
            x_offset = col * cell_width
            y_offset = row * cell_height
            merged_image.paste(img, (x_offset, y_offset))
        return merged_image

    def merge_vehicle_camera_views(self, merge: str = None, grid_size: tuple = None, logical_order: list = None) -> list:
        """
        Main method to merge vehicle camera views.
        """
        processed_images = self._load_images()

        if not processed_images:
            raise ValueError("No valid images provided")

        # Add borders and numbering to images
        processed_images = self._add_borders_and_numbering(processed_images)

        # Merge the images based on the selected option
        merged_image = self._merge_images(processed_images, merge, grid_size, logical_order)

        # Scale merged image if it exceeds max_dimensions
        if self.max_dimensions:
            max_width, max_height = self.max_dimensions
            if merged_image.width > max_width or merged_image.height > max_height:
                scale_ratio = min(max_width / merged_image.width, max_height / merged_image.height)
                new_width = int(merged_image.width * scale_ratio)
                new_height = int(merged_image.height * scale_ratio)
                merged_image = merged_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return [merged_image]



class MultimodalInputBuilder:
    def __init__(self, Model_type: str = None):
        self.content = []
        self.Model_type = Model_type

    def assistant(self, content: str):
        """
        构造助手消息

        Args:
            content (str): 助手的文本内容。

        Returns:
            dict: 助手的结构化消息。
        """
        if self.Model_type == 'LLAMA':
            return {"role": "assistant", "content": content}
        else:
            raise ValueError("Invalid Model type. Must be 'LLAMA'")

    def user_input(self, prompt_text: str, images: list = None):
        """
        构造用户输入，可以包含文本和图片。

        Args:
            prompt_text (str): 用户输入的文本内容。
            images (list, optional): 图片 URL 列表。如果没有图片，默认为 None。

        Returns:
            dict: 用户输入的结构化消息。
        """
        if self.Model_type == 'LLAMA':
            content = []
            if images:
                # 添加图片内容
                content.extend([{"type": "image"} for _ in images])
            # 添加文本内容
            content.append({"type": "text", "text": prompt_text})
            return {"role": "user", "content": content}
        else:
            raise ValueError("Invalid Model type. Must be 'LLAMA'")

    def system(self, content: str, type: str):
        """
        构造系统消息，支持 fine-tuning 类型和其他类型。

        Args:
            content (str): 系统消息的内容。
            type (str): 消息类型，支持 'Fine_tuning' 或其他。

        Returns:
            dict: 系统消息的结构化内容。
        """
        if self.Model_type == 'LLAMA':
            if type == 'Fine_tuning':
                return {"from": "system", "value": content}
            else:
                return {"role": "system", "content": content}
        else:
            raise ValueError("Invalid Model type. Must be 'LLAMA'")

    def convert_to_percentage(self, base_width, base_height, match):
        """
        将坐标转化为百分比形式。

        Args:
            base_width (float): 基准宽度。
            base_height (float): 基准高度。
            match (match object): 正则匹配对象，包含坐标信息。

        Returns:
            str: 转换后的百分比格式。
        """
        # 提取数值并计算百分比
        values = match.group().split(",")
        if len(values) == 4:
            x_percent = float(values[2]) / base_width * 100
            y_percent = float(values[3].strip(">")) / base_height * 100
            return f"<{values[0]},{values[1]},{x_percent:.2f}%,{y_percent:.2f}%>"
        return match.group()  # 如果没有匹配到正确的格式，直接返回原始数据

    def build_input(self, role: str, content: str, images: list = None, system_type: str = None):
        """
        根据角色构建不同类型的输入。

        Args:
            role (str): 输入的角色，'user', 'assistant' 或 'system'。
            content (str): 内容文本。
            images (list, optional): 图片列表，默认为 None。
            system_type (str, optional): 如果角色是系统，提供系统类型，默认为 None。

        Returns:
            dict: 对应角色的结构化输入。
        """
        if role == "assistant":
            return self.assistant(content)
        elif role == "user":
            return self.user_input(content, images)
        elif role == "system":
            return self.system(content, system_type)
        else:
            raise ValueError("Invalid role. Must be 'assistant', 'user', or 'system'.")


if __name__ == '__main__':
    image_paths = ["2.png", "1.png", "3.png", "2.png", "1.png", "3.png"]
    resize_to = (224,224)
    max_dimensions = (1120, 1120)

    processor = DataPreprocessor(image_paths=image_paths, resize_to=resize_to, max_dimensions=max_dimensions)


    merged_images_horizontal = processor.merge_vehicle_camera_views(merge='horizontal', logical_order=[1, 0, 2, 4, 3, 5])
    merged_images_horizontal[0].save("merged_horizontal.jpg")

    merged_images_vertical = processor.merge_vehicle_camera_views(merge='vertical', logical_order=[1, 0, 2, 4, 3, 5])
    merged_images_vertical[0].save("merged_vertical.jpg")

    merged_images_custom_grid = processor.merge_vehicle_camera_views(merge='custom_grid', logical_order=[1, 0, 2, 4, 3, 5])
    merged_images_custom_grid[0].save("merged_custom_grid.jpg")

    Input = MultimodalInputBuilder(Model_type='LLAMA')
    user_input = Input.build_input(role='user', content='Hello, how can I help you?', images=[Image.open("1.png")])
    print (user_input)

    system_message = Input.build_input(role='system', content="Fine-tuning message", system_type="Fine_tuning")
    print(system_message)

