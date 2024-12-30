import argparse
import os
import sys
import torch
from accelerate import Accelerator
from PIL import Image as PIL_Image
from peft import PeftModel
from transformers import MllamaForConditionalGeneration, MllamaProcessor

# Initialize accelerator
accelerator = Accelerator()
device = accelerator.device

# Constants
DEFAULT_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
MAX_OUTPUT_TOKENS = 2048
MAX_IMAGE_SIZE = (1120, 1120)


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
    #hf_token = get_hf_token()

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
        #token=hf_token
    )

    # Step 4: 加载处理器
    # 使用 `MllamaProcessor` 加载处理器以处理输入数据（如文本和图像）。
    processor = MllamaProcessor.from_pretrained(
        model_name,
        #token=hf_token,
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

if __name__ == "__main__":
    # 配置参数
    model_name = '/media/workstation/6D3563AC52DC77EA/Model/meta-llama/Llama-3.2-11B-Vision-Instruct'
    finetuning_path = None  # 如果有微调路径可以设置
    image_path = "car.png"  # 替换为本地图片路径
    prompt_text = "Describe this image in detail."
    temperature = 0.7
    top_p = 0.9

    # 确保 CUDA 可用
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your GPU configuration.")
        #return

        # 加载模型和处理器
    try:
        model, processor = load_model_and_processor(
            model_name=model_name,
            finetuning_path=finetuning_path,
            device="auto",  # 自动设备映射
            max_memory={0: "21GB", 1: "6GB"}  # GPU 显存限制
        )
        print("Model and processor loaded successfully.")
    except Exception as e:
        print(f"Failed to load model and processor: {e}")
        #return

        # 加载并处理图像
    try:
        image = process_image(image_path=image_path)
        print("Image loaded and processed successfully.")
    except Exception as e:
        print(f"Failed to process image: {e}")
        #return

        # 生成文本
    try:
        generated_text = generate_text_from_image(
            model=model,
            processor=processor,
            image=image,
            prompt_text=prompt_text,
            temperature=temperature,
            top_p=top_p
        )
        print("Generated Text:")
        print(generated_text)
    except Exception as e:
        print(f"Failed to generate text: {e}")




