import torch
import io
import base64
import requests
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, MllamaForConditionalGeneration
from accelerate import infer_auto_device_map, init_empty_weights, dispatch_model

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

# 设置非交互式后端
matplotlib.use('Agg')

def display_local_image(image_path, save_path='output_image.png'):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')  # 隐藏坐标轴
    plt.savefig(save_path, bbox_inches='tight')  # 保存图片到文件
    print(f"Image saved to {save_path}")

def encode_image(image_path):
        # Open the image file
        with Image.open(image_path) as img:
            # Create a BytesIO object to hold the image data
            buffered = io.BytesIO()
            # Save the image to the BytesIO object in a specific format (e.g., PNG)
            img.save(buffered, format="PNG")
            # Get the byte data from the BytesIO object
            img_bytes = buffered.getvalue()
            # Encode the byte data to base64
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            return img_base64


base64_image = encode_image('car.png')

# 显示并保存图像
display_local_image('car.png', save_path='car_output.png')

#%%
# Print the number of GPUs available
print("Number of GPUs available:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Auto-detect device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Define the directory for the model
model_dir = r'/media/workstation/6D3563AC52DC77EA/Model/meta-llama/Llama-3.2-11B-Vision-Instruct'

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = MllamaForConditionalGeneration.from_pretrained(model_dir, torch_dtype='auto', device_map='sequential')
processor = AutoProcessor.from_pretrained(model_dir)

#%%
while True:
    #print(f'Enter a prompt to generate a response:')
    #prompt = input()



    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
    print('Test point')
    image = Image.open('car.png')
    print('Test point')

    print (image)

    # Create a message list that includes an image and a text prompt
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "If I had to write a haiku for this one, it would be: "}
        ]}
    ]


    text = tokenizer.apply_chat_template(
        messages,
        #tokenize=False,
        add_generation_prompt=True
    )

    #model_input = tokenizer([text], return_tensors='pt').to(device)
    model_input = processor(base64_image, [text], return_tensors="pt", ).to(device)
    attention_mask = torch.ones(model_input.input_ids.shape, dtype=torch.long, device=device)
    generated_ids = model.generate(
        model_input.input_ids,
        max_new_tokens=512,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_input.input_ids, generated_ids)]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f'{response}\n')
