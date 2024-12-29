import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_dir = r'/media/workstation/6D3563AC52DC77EA/Model/meta-llama/Llama-3.2-11B-Vision-Instruct'

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype='auto', device_map='auto')

pipeline = transformers.pipeline(
    task = 'text-generation',
    model=model_dir,
    torch_dtype='auto',
    device_map = 'auto')

while 1:
    print(f'Enter a prompt to generate a response:')
    prompt = input()
    messages = [
        {'role': 'system', 'content': 'aaa'},
        {'role': 'user', 'content': prompt}
    ]


    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_input = tokenizer([text], return_tensors='pt').to(device)
    attention_mask = torch.ones(model_input.input_ids.shape, dtype=torch.long, device=device)
    generated_ids = model.generate(
        model_input.input_ids,
        max_new_tokens=512,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_input.input_ids, generated_ids)]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f'{response} \n')
