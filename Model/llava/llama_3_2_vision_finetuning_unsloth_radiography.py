
# %%capture
# !pip install unsloth
# # Also get the latest nightly Unsloth!
# !pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git

"""* We support Llama 3.2 Vision 11B, 90B; Pixtral; Qwen2VL 2B, 7B, 72B; and any Llava variant like Llava NeXT!
* We support 16bit LoRA via `load_in_4bit=False` or 4bit QLoRA. Both are accelerated and use much less memory!
"""

#%%
from PIL import Image
def convert_conservation_to_unsloth(sample, image_path):
    """
    将 conservation 数据转换为指定格式的 conversation 数据。

    参数:
        conservation (list): 输入的对话数据列表，每个元素是字典，包含 'from' 和 'value'。
        image_folder (str): 存储图片的文件夹路径，用于加载图像。

    返回:
        list: 转换后的 conversation 数据列表。
    """
    conversation = []
    for item in sample:
        role = item['from']  # 提取角色 (system, human, gpt)
        value = item['value']  # 提取对应的文本内容

        # 根据角色构建内容
        if role == "system":
            conversation.append({
                "role": "system",
                "content": [
                    {"type": "text", "text": value}
                ]
            })
        elif role == "human":
            # 判断是否包含图片占位符 "<image>"
            if "<image>" in value:
                # 提取图像名称并加载图像
                conversation.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": value.replace("<image>", "").strip()},
                        {"type": "image", "image": Image.open(image_path)}  # 替换为实际图片路径
                    ]
                })
            else:
                conversation.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": value}
                    ]
                })
        elif role == "gpt":
            conversation.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": value}
                ]
            })
    return {"messages": conversation}
#%%

from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
"""
fourbit_models = [
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", # Llama 3.2 vision support
    "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit", # Can fit in a 80GB card!
    "unsloth/Llama-3.2-90B-Vision-bnb-4bit",

    "unsloth/Pixtral-12B-2409-bnb-4bit",              # Pixtral fits in 16GB!
    "unsloth/Pixtral-12B-Base-2409-bnb-4bit",         # Pixtral base model

    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",          # Qwen2 VL support
    "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",

    "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",      # Any Llava variant works!
    "unsloth/llava-1.5-7b-hf-bnb-4bit",
] # More models at https://huggingface.co/unsloth
"""


model, tokenizer = FastVisionModel.from_pretrained(
    "/media/workstation/6D3563AC52DC77EA/Model/meta-llama/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

"""We now add LoRA adapters for parameter efficient finetuning - this allows us to only efficiently train 1% of all parameters.

**[NEW]** We also support finetuning ONLY the vision part of the model, or ONLY the language part. Or you can select both! You can also select to finetune the attention or the MLP layers!
"""

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0.005, # LoRA dropout
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)


#%%
import json

datasets_json_path = "/media/workstation/6D3563AC52DC77EA/Llama3.2-Vision-Finetune-master/NusceneQA_Llama_finetuning/NusceneQA_Llama_finetuning_dataset.json"
image_folder = "/media/workstation/6D3563AC52DC77EA/Llama3.2-Vision-Finetune-master/NusceneQA_Llama_finetuning//image_folder"

with open(datasets_json_path, 'r') as f:
    dataset_json = json.load(f)

converted_dataset = []

for _ in dataset_json:
    image = _["image"]
    conservation = _["conversations"]
    converted_dataset.append(convert_conservation_to_unsloth(conservation, image_folder + '/' + image))

#%%
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2, ####
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        #max_steps=30,
        num_train_epochs = 1 ,# Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",     # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4, #######3
        max_seq_length = 2048,
    ),
)

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
model.save_pretrained("lora_model") # Local saving
tokenizer.save_pretrained("lora_model")


model.save_pretrained("lora_model") # Local saving
tokenizer.save_pretrained("lora_model")
model.save_pretrained_merged("unsloth_finetune", tokenizer,)
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving

"""Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:"""

if False:
    from unsloth import FastVisionModel
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit = True, # Set to False for 16bit LoRA
    )
    FastVisionModel.for_inference(model) # Enable for inference!


"""### Saving to float16 for VLLM

We also support saving to `float16` directly. Select `merged_16bit` for float16. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.
"""

# Select ONLY 1 to save! (Both not needed!)

# Save locally to 16bit

