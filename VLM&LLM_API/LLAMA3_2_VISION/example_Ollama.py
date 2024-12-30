import torch
from PIL import Image
from transformers import AutoTokenizer, MllamaForConditionalGeneration, AutoProcessor
from accelerate import init_empty_weights

torch.cuda.empty_cache()
print("CUDA cache cleared.")

# Model path and device setup
model_id = '/media/workstation/6D3563AC52DC77EA/Model/meta-llama/Llama-3.2-11B-Vision-Instruct'

# Print the number of GPUs available
print("Number of GPUs available:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


# Auto-detect device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

"""
# Initialize model with empty weights
with init_empty_weights():
    model = MllamaForConditionalGeneration.from_pretrained(model_id)
"""

max_memory={0: "22GiB", 1: "6GiB"}

# Load the model
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype= torch.half,
    device_map='auto',
    max_memory=max_memory
)

# Load the processor
processor = AutoProcessor.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

#%%
# Create a message list that includes an image and a text prompt
image = Image.open('car.png')
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Describe what's in the picture. What are the characteristics?"}
    ]}
]

# Prepare inputs using the processor
try:
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, input_text, return_tensors="pt").to(device)
    # print(f"Inputs: {inputs}")

    # Generate output from the model
    output = model.generate(**inputs, max_new_tokens=512)

    # Decode and print the output
    decoded_output = processor.decode(output[0], clean_up_tokenization_spaces=True)
    print(f"Generated text: {decoded_output}")


except Exception as e:
    print(f"An error occurred: {e}")
