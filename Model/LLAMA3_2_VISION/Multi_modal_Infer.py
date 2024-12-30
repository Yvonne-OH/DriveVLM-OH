import argparse
import os
import sys
import torch
from accelerate import Accelerator
from PIL import Image as PIL_Image
from peft import PeftModel
import gradio as gr
from transformers import MllamaForConditionalGeneration, MllamaProcessor

from huggingface_hub import HfFolder
# Initialize accelerator
accelerator = Accelerator()
device = accelerator.device

# Constants
DEFAULT_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
MAX_OUTPUT_TOKENS = 2048
MAX_IMAGE_SIZE = (1120, 1120)

def get_hf_token():
    """Retrieve Hugging Face token from the cache or environment."""

    # 检查是否在环境变量中设置了 Hugging Face 的访问令牌。
    # 如果存在，则直接返回这个令牌。
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        return token

    # 如果环境变量未设置，尝试从 Hugging Face 的缓存中获取令牌。
    # 这要求用户通过 `huggingface-cli login` 登录后，令牌被存储在本地。
    token = HfFolder.get_token()
    if token:
        return token

    # 如果两种方法都无法找到令牌，提示用户登录 Hugging Face。
    print("Hugging Face token not found. Please login using `huggingface-cli login`.")
    # 退出程序，返回状态码 1，表示错误。
    sys.exit(1)


def load_model_and_processor(model_name: str, finetuning_path: str = None, device: str = "auto", max_memory: dict = None):
    """
    Load a model and its processor with optional LoRA adapter.

    Args:
        model_name (str): The name or path of the pre-trained model to load.
        finetuning_path (str, optional): The path to the LoRA fine-tuning adapter. Default is None.

    Returns:
        Tuple: A tuple containing the loaded model and processor.
    """

    # Step 1: 打印正在加载的模型名称
    print(f"Loading model: {model_name}")

    # Step 2: 获取 Hugging Face 的访问令牌，用于认证加载模型和处理器
    hf_token = get_hf_token()

    # Step 3: 加载模型
    # 使用 `MllamaForConditionalGeneration` 加载模型：
    # - `torch_dtype=torch.bfloat16`: 使用 `bfloat16` 精度以减少显存占用。
    # - `use_safetensors=True`: 指定使用 `safetensors` 格式以提高加载安全性。
    # - `device_map=device`: 自动分配设备（如 GPU）。
    # - `token=hf_token`: 提供 Hugging Face 的认证令牌。
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        device_map=device,
        max_memory=max_memory,
        token=hf_token
    )

    # Step 4: 加载处理器
    # 使用 `MllamaProcessor` 加载处理器以处理输入数据（如文本和图像）。
    processor = MllamaProcessor.from_pretrained(
        model_name,
        token=hf_token,
        use_safetensors=True
    )

    # Step 5: 如果提供了 LoRA 微调适配器路径，加载适配器
    if finetuning_path and os.path.exists(finetuning_path):
        print(f"Loading LoRA adapter from '{finetuning_path}'...")

        # 使用 `PeftModel` 将 LoRA 适配器加载到模型中：
        # - `is_adapter=True`: 指定这是一个适配器。
        # - `torch_dtype=torch.bfloat16`: 保持与模型一致的精度。
        model = PeftModel.from_pretrained(
            model,
            finetuning_path,
            is_adapter=True,
            torch_dtype=torch.bfloat16
        )
        print("LoRA adapter merged successfully")

    # Step 6: 使用 `accelerator` 工具优化模型和处理器的设备分配
    # - `accelerator.prepare`: 根据加速器（如多 GPU）准备模型和处理器。
    model, processor = accelerator.prepare(model, processor)

    # Step 7: 返回加载后的模型和处理器
    return model, processor

def process_image(image_path: str = None, image = None) -> PIL_Image.Image:
    """Process and validate image input"""
    if image is not None:
        return image.convert("RGB")
    if image_path and os.path.exists(image_path):
        return PIL_Image.open(image_path).convert("RGB")
    raise ValueError("No valid image provided")

def generate_text_from_image(model, processor, image, prompt_text: str, temperature: float, top_p: float):
    """Generate text from image using model"""
    conversation = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(image, prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, temperature=temperature, top_p=top_p, max_new_tokens=MAX_OUTPUT_TOKENS)
    return processor.decode(output[0])[len(prompt):]