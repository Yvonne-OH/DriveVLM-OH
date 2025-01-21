import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from transformers import BitsAndBytesConfig
import llava_util


class Llava_Interface:
    def __init__(self, model_path: str = None, finetuning_path: str = None, **kwargs):
        self.model_path = model_path
        self.finetuning_path = finetuning_path if finetuning_path else None
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device_map = kwargs.get("device_map", "auto")
        self.max_memory = kwargs.get("max_memory", None)

        # 指定量化配置
        self.quantization_config = kwargs.get("quantization_config", None)
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        # 加载模型并设置到指定设备
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=self.quantization_config,
            device_map= self.device_map,
            low_cpu_mem_usage=True,
    ).to(self.device)

        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(model_path)

    def prepare_prompt(self, conversation):
        """根据对话生成prompt"""
        return self.processor.apply_chat_template(conversation, add_generation_prompt=True)

    def generate_response(self, conversation, images, max_new_tokens=2048, **kwargs):
        """生成模型的回复"""
        # 处理图像和对话内容
        prompt = self.prepare_prompt(conversation)
        inputs = self.processor(images=images, text=prompt, padding=True, return_tensors="pt").to(self.device)

        generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)         # 生成回复

        generated_text = self.processor.decode(generate_ids[0], skip_special_tokens=kwargs.get('skip_special_tokens',True),    # 解码并清理生成的文本
                                               clean_up_tokenization_spaces=kwargs.get(' clean_up_tokenization_spaces',True))

        if kwargs.get(' clean_up_tokenization_spaces',True):
            # 清理文本，只保留回复内容
            cleaned_text = generated_text.split('[INST]')[-1].split('[/INST]')[-1].strip()

        return cleaned_text


# 示例使用
if __name__ == "__main__":
    model_path = "/media/workstation/6D3563AC52DC77EA/Model/llava/llava-v1.6-mistral-7b-hf"
    chatbot = Llava_Interface(model_path)

    # 获取图像
    url_stop = "https://www.ilankelman.org/stopsigns/australia.jpg"
    url_cats = "http://images.cocodataset.org/val2017/000000039769.jpg"

    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    image_stop = Image.open(requests.get(url, stream=True).raw).resize((128, 128))

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image_cats = Image.open(requests.get(url, stream=True).raw).resize((128, 128))

    llava_conservation = [ ]


    # 添加新的用户对话和回答
    llava_util.llava_conversation(llava_conservation, "user", "What about this image? How many birds do you see?", is_image=True)

    conversation_1 = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
    ]

    # 生成回复
    response = chatbot.generate_response(conversation_1, [image_stop, image_cats])
    print(response)
