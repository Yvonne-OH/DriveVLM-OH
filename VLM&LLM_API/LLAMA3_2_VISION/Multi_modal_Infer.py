import os
import sys
import torch
import argparse
from peft import PeftModel
from accelerate import Accelerator
from huggingface_hub import HfFolder
from Util.util import ImagePreprocessor
from transformers import BitsAndBytesConfig
from torch.quantization import quantize_dynamic
from transformers import MllamaForConditionalGeneration, MllamaProcessor

# Initialize accelerator
accelerator = Accelerator()
device = accelerator.device

# Constants
DEFAULT_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
MAX_OUTPUT_TOKENS = 512
MAX_IMAGE_SIZE = (512,512)

class VisionModelHandler:
    def __init__(self, model_name: str = None, finetuning_path: str = None, max_memory: dict = None, **kwargs):
        # Validate inputs
        assert model_name, "Model name must be specified."
        if finetuning_path:
            assert os.path.exists(finetuning_path), f"Finetuning path does not exist: {finetuning_path}"

        self.model_name = model_name
        self.finetuning_path = finetuning_path
        self.max_memory = eval(max_memory) if isinstance(max_memory, str) else max_memory

        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.model = None
        self.processor = None
        self._initialize_model_and_processor()

    def _get_hf_token(self):
        token = os.getenv("HUGGINGFACE_TOKEN") or HfFolder.get_token()
        if not token:
            raise ValueError("Hugging Face token not found. Please login using `huggingface-cli login`.")
        return token

    def _initialize_model_and_processor(self, bnb_config: BitsAndBytesConfig = None):
        print(f"Loading model: {self.model_name}")
        bnb_config = bnb_config
        """
            bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        """

        try:
            # Load the model with specified configurations
            self.model = MllamaForConditionalGeneration.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                use_safetensors=True,
                torch_dtype= torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto",
                max_memory=self.max_memory
            ).eval()

            self.model.tie_weights()

            # Load the processor
            self.processor = MllamaProcessor.from_pretrained(
                self.model_name,
                use_safetensors=True
            )

            # Optionally load a LoRA adapter
            if self.finetuning_path and os.path.exists(self.finetuning_path):
                print(f"Loading LoRA adapter from '{self.finetuning_path}'...")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.finetuning_path,
                    is_adapter=True,
                    load_in_8bit=True
                )
                print("LoRA adapter merged successfully.")

            # Prepare model and processor with the accelerator
            self.model, self.processor = self.accelerator.prepare(self.model, self.processor)
            print("Model and processor loaded successfully.")

        except Exception as e:
            print(f"Failed to load model and processor: {e}")
            raise

    def generate_text_from_image(self, images, prompt_text, temperature=0.7, top_p=0.9):
        print("Generating text from images...")
        conversation = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in images],
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(images=images, text=prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs, temperature=temperature, top_p=top_p, max_new_tokens=512
            )
        generated_text = self.processor.decode(output[0], skip_special_tokens=True)
        return generated_text


def get_hf_token():
    """Retrieve Hugging Face token from the cache or environment."""
    # Check if a token is explicitly set in the environment
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        return token

    # Automatically retrieve the token from the Hugging Face cache (set via huggingface-cli login)
    token = HfFolder.get_token()
    if token:
        return token

    print("Hugging Face token not found. Please login using `huggingface-cli login`.")
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
    #hf_token = get_hf_token()

    # Step 3: 加载模型
    # 使用 `MllamaForConditionalGeneration` 加载模型：
    # - `torch_dtype=torch.bfloat16`: 使用 `bfloat16` 精度以减少显存占用。
    # - `use_safetensors=True`: 指定使用 `safetensors` 格式以提高加载安全性。
    # - `device_map=device`: 自动分配设备（如 GPU）。
    # - `token=hf_token`: 提供 Hugging Face 的认证令牌。
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = MllamaForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=device,
        max_memory=max_memory,
        #token=hf_token
    ).eval()

    model.tie_weights()

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
            load_in_8bit=True,
        )
        print("LoRA adapter merged successfully")

    # Step 6: 使用 `accelerator` 工具优化模型和处理器的设备分配
    # - `accelerator.prepare`: 根据加速器（如多 GPU）准备模型和处理器。
    model, processor = accelerator.prepare(model, processor)

    # Step 7: 返回加载后的模型和处理器
    return model, processor



def generate_text_from_image(model, processor, images, prompt_text: str, temperature: float, top_p: float):
    """Generate text from image using model"""
    """
    conversation = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}
    ]
    """

    # Construct a single conversation containing all images
    conversation = [
        {
            "role": "user",
            "content": [
                *[{"type": "image"} for _ in images],
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    print (conversation)

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(images=images, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():  # Disable gradient computation to save memory
        output = model.generate(**inputs, temperature=temperature, top_p=top_p, max_new_tokens=MAX_OUTPUT_TOKENS)
    generated_text = processor.decode(output[0], skip_special_tokens=True)

    return generated_text


if __name__ == "__main__":

    torch.cuda.empty_cache()

    # 配置参数
    model_name = '/media/workstation/6D3563AC52DC77EA/Model/meta-llama/Llama-3.2-11B-Vision-Instruct'
    finetuning_path = '/media/workstation/6D3563AC52DC77EA/Model/meta-llama/Llama-3.2-11B-Vision-Instruct/lora_model'  # 如果有微调路径可以设置
    image_paths = ["2.png","1.png","3.png","2.png","1.png","3.png"]  # 替换为本地图片路径
    prompt_text = "Describe this image in detail."
    temperature = 0.7
    top_p = 0.9
    resize_to = (224, 224)  # 设置图像缩放尺寸
    max_dimension = (1120, 1120)

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
            max_memory={0: "22GB", 1: "7.6GB"}  # GPU 显存限制
        )
        model.gradient_checkpointing_enable()
        print("Model and processor loaded successfully.")
    except Exception as e:
        print(f"Failed to load model and processor: {e}")
        #return

        # 加载并处理图像
    try:

        image_processor = ImagePreprocessor(image_paths=image_paths, resize_to=resize_to, max_dimensions=max_dimension)
        image = image_processor.merge_vehicle_camera_views(merge='custom_grid',
                                                                         logical_order=[1, 0, 2, 4, 3, 5])

        image[0].save("merged_image.png")
        print("Image loaded and processed successfully.")
    except Exception as e:
        print(f"Failed to process image: {e}")
        #return

        # 生成文本
    try:
        generated_text = generate_text_from_image(
            model=model,
            processor=processor,
            images=image,
            prompt_text=prompt_text,
            temperature=temperature,
            top_p=top_p
        )
        print("Generated Text:")
        print(generated_text)
    except Exception as e:
        print(f"Failed to generate text: {e}")




