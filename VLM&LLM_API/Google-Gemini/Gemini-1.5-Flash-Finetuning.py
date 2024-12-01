import os
import time
import pandas as pd

import google.generativeai as genai
from DataLoader.convert2Gemini import GminiJson2List

genai.configure(api_key="AIzaSyAeDUmN1KvGENCHJh1_jDDI8bea215ZM3I")

Json_path = "/media/oh/0E4A12890E4A1289/DriveLM/data/QA_dataset_nus/test_eval.json"
Image_path = "/media/oh/0E4A12890E4A1289/DriveLM/data/"
Save_path = "/media/oh/0E4A12890E4A1289/DriveLM/data/QA_dataset_nus/test_Gemini.json"

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
)

Files, training_data = GminiJson2List (Json_path,Image_path, 10)
#response = model.generate_content(training_data)

base_model = "models/gemini-1.5-flash-001-tuning"

operation = genai.create_tuned_model(
    # You can use a tuned model here too. Set `source_model="tunedModels/..."`
    display_name="increment",
    source_model=base_model,
    epoch_count=20,
    batch_size=4,
    learning_rate=0.001,
    training_data=training_data,
)

for status in operation.wait_bar():
    time.sleep(10)

result = operation.result()
print(result)
# # You can plot the loss curve with:
snapshots = pd.DataFrame(result.tuning_task.snapshots)
#sns.lineplot(data=snapshots, x='epoch', y='mean_loss')

model = genai.GenerativeModel(model_name=result.name)
result = model.generate_content("III")
print(result.text)  # IV