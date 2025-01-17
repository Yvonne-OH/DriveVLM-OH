import re
import json
import ast
import openai
import time
from tqdm import tqdm
import os
from openai import OpenAI
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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import Accelerator

# Suppress logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"  # 0: INFO, 1: WARNING, 2: ERROR, 3: FATAL
os.environ["http_proxy"] = "http://localhost:7897"
os.environ["https_proxy"] = "http://localhost:7897"


class GPT_SCORE:
    def __init__(self, Api_key: str = None, model: str = None, Temperature: float = 0.5, Top_p: float = 0.3, **kwargs):
        """
               Initialize the GPT_SCORE class with parameters for GPT or LLaMA scoring.

               Args:
                   Api_key (str): API key for GPT scoring (if using GPT).
                   model (str): Model name to use for GPT or LLaMA.
                   Temperature (float): Temperature parameter for response generation.
                   Top_p (float): Top-p parameter for response generation.
        """
        self.api_key = Api_key
        self.model = model
        self.temperature = Temperature
        self.top_p = Top_p

        print(model.upper())

        if model and "GPT" in model.upper():
            if not self.api_key:
                raise ValueError("API key is required for GPT scoring.")
            self.scorer_type = "GPT"
        elif model and "LLAMA" in model.upper():
             self.scorer_type = "LLAMA"
        else:
            raise ValueError("Invalid model name. ")

    def Llama_Eval_init(self, model_name: str, finetuning_path: str = None, max_memory: dict = None,**kwargs):

        assert model_name !=  None, "Please provide a model name for evaluation."

        for key, value in kwargs.items():
            setattr(self, key, value)

        accelerator = Accelerator()

        # Step 1: 打印正在加载的模型名称
        print(f"Loading model: {model_name}")

        # Step 2: 配置量化选项
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        try:
            # Step 3: 加载模型和分词器
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_safetensors=True
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                use_safetensors=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else None,  # 自动分配设备
                max_memory=max_memory,
            ).eval()

            # Step 4: 如果提供了 LoRA 微调适配器路径，加载适配器
            if finetuning_path and os.path.exists(finetuning_path):
                from peft import PeftModel

                print(f"Loading LoRA adapter from '{finetuning_path}'...")
                model = PeftModel.from_pretrained(
                    model,
                    finetuning_path,
                    is_adapter=True,
                    load_in_8bit=True,
                )
                print("LoRA adapter merged successfully")

            # Step 5: 使用 accelerator 工具优化模型和分词器
            model, tokenizer = accelerator.prepare(model, tokenizer)

            # 打印加载完成信息
            print("Model and tokenizer are ready for use.")

        except Exception as e:
            raise e

        self.model = model
        self.tokenizer = tokenizer

    def GPT_Score_interface(self, MCQs_Evaluation_text: str) -> str:
        openai.api_key = self.api_key
        try:
            client = OpenAI(
                # defaults to os.environ.get("OPENAI_API_KEY")
                api_key=self.api_key,
            )

            # Ensure the input is formatted as a list of messages
            messages = [
                {"role": "system", "content": "Please give your rating based on the documents given."},
                {"role": "user", "content": MCQs_Evaluation_text}
            ]
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise e

    def Llama_Score_interface(self, text_input: str) -> str:
        """
        Evaluate the input text using the LLaMA model.

        Args:
            text_input (str): The text input to evaluate.

        Returns:
            str: The generated response from the LLaMA model.
        """
        try:
            assert hasattr(self, 'model') and hasattr(self, 'tokenizer'), "LLaMA model and tokenizer must be initialized."

            messages = [
                {"role": "system", "content": "Please give your rating based on the documents given."},
                {"role": "user", "content": text_input},
            ]

            # 转换对话为模型可用的输入
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)

            # 设置终止标志符
            terminators = [
                self.tokenizer.eos_token_id,  # 模型的默认终止符
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")  # 自定义终止符（如有）
            ]

            # 生成输出
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=2048,
                eos_token_id=terminators,
                do_sample=True,  # 启用采样模式
                temperature=self.temperature,  # 控制生成多样性
                top_p=self.top_p,  # 核采样的参数
            )

            # 解码模型的生成结果
            response = outputs[0][input_ids.shape[-1]:]  # 取生成的新增部分
            generated_text = self.tokenizer.decode(response, skip_special_tokens=True)

            return generated_text

        except Exception as e:
            raise e

    def score(self, input_text: str) -> str:
        """
        Unified interface for scoring based on the selected model.

        Args:
            input_text (str): The text input to evaluate.

        Returns:
            str: The generated response from the selected model.
        """
        if self.scorer_type == "GPT":
            return self.GPT_Score_interface(input_text)
        elif self.scorer_type == "LLAMA":
            return self.Llama_Score_interface(input_text)
        else:
            raise ValueError("Invalid scorer type. Ensure the model is properly initialized.")

if __name__ == "__main__":

    with open("MCQs_Eval.txt", "r") as f:
        template = f.read()

    Question = """A self-driving car encounters a complex intersection with multiple objects and conditions to consider. 
                             Based on the rules below, identify the car's primary decision in the scenario:
                            Scenario:
                            The car approaches an intersection with a stop sign on the right and a pedestrian crossing from left to right. 
                            A cyclist is coming from the opposite direction, and the traffic light ahead is green. The weather is clear, but the sun is setting, causing glare.
                            
                            Options:
                            A. Proceed through the intersection immediately.
                            B. Stop and yield to the pedestrian and cyclist, despite the green light.
                            C. Ignore the stop sign and focus on the green light.
                            D. Slow down, assess all conditions, and proceed cautiously while yielding to any right-of-way entities.
    """
    Desc = "The question evaluates the decision-making capabilities of a self-driving car when encountering a complex intersection, focusing on traffic rules, object recognition, environmental awareness, and reasoning clarity."
    GT = "The car should slow down, assess all conditions (stop sign, pedestrian, cyclist, traffic light, and glare), and proceed cautiously while yielding to entities with the right of way.."
    PRED = """
    D. Slow down, assess all conditions, and proceed cautiously while yielding to any right-of-way entities. 
    This option ensures compliance with traffic rules by yielding to the pedestrian and cyclist, who have the right of way, even with a green traffic light.
    It demonstrates proper object recognition by considering the stop sign, pedestrian, cyclist, and traffic light, 
    while also accounting for their locations and movements. Additionally, it acknowledges the environmental factor of glare from the setting sun, 
    ensuring a safe and cautious approach in line with clear and logical reasoning."""

    MCQs_Evaluation_text = template.replace("<<QUESTION>>", Question).replace("<<DESC>>", Desc).replace("<<GT>>",
                                                                                                        GT).replace(
        "<<PRED>>", PRED)

    # 初始化模型和配置
    model_name = '/media/workstation/6D3563AC52DC77EA/Model/meta-llama/Llama-3.2-3B-Instruct'
    finetuning_path = None
    max_memory = {0: "22GB", 1: "7.6GB"}  # GPU 显存限制

    Api_key =  "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"  # Replace with your actual API key
    model = "gpt-4o-mini"  # Correct model name
    Temperature = 0.5
    Top_p = 0.3

    # 创建实例
    llama_evaluator = GPT_SCORE(model=model_name, Temperature=Temperature, Top_p=Top_p)
    llama_evaluator.Llama_Eval_init(model_name=model_name, finetuning_path=finetuning_path, max_memory=max_memory)

    # 推理并打印结果
    response = llama_evaluator.score(MCQs_Evaluation_text)
    print("LLaMA Model Response:")
    print(response)

    print("-"*50)


    GPT_Evaluator = GPT_SCORE(Api_key, model, Temperature, Top_p)
    GPT_score = GPT_Evaluator.GPT_Score_interface(MCQs_Evaluation_text)
    print("GPT4O Model Response:")
    print (GPT_score)








