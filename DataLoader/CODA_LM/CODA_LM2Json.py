import argparse
import pandas as pd
import json
from tqdm import tqdm
import sys
from PIL import Image
import io
import uuid
import os
import shutil

parquet_folder_path ="/media/workstation/6D3563AC52DC77EA/Data/CODA_LM/English/"
output_json_path = "/media/workstation/6D3563AC52DC77EA/Data/CODA_LM/English/Test.json"
directory_path = '/media/workstation/6D3563AC52DC77EA/Data/CODA_LM/English/Image/'
Task ="Test"

parquet_files = [file for file in os.listdir(parquet_folder_path) if Task in file and file.endswith('.parquet')]


# Check if the directory exists
if not (os.path.exists(directory_path)):

    # Create the directory if it does not exist
    os.makedirs(directory_path)
    os.makedirs(directory_path+"/Train")
    os.makedirs(directory_path + "/Val")
    os.makedirs(directory_path + "/Test")
    print(f"Directory created: {directory_path}")

# Initialize segment_data outside the loop to accumulate data from all files
segment_data = {}

# Read and process each Parquet file
for parquet_file in parquet_files:
    # Read Parquet file
    df = pd.read_parquet(os.path.join(parquet_folder_path, parquet_file), engine='pyarrow')

    # Processing each row in the DataFrame
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing " + parquet_file):
        # Generate unique ID for each entry
        id = str(uuid.uuid4())

        # Convert binary data to image
        image_data = row['image']['bytes']
        image = Image.open(io.BytesIO(image_data))

        # Extract and clean conversation data
        question = row['conversations'][0]['value'].replace('<image>', '')
        answer = row['conversations'][1]['value'].replace('<image>', '')

        # Check if this ID already exists (unlikely but safe to check)
        if id not in segment_data:
            segment_data[id] = {"images": [], "questions": []}

        # Append image info
        #image_path = f"output_{id}.png"
        image.save(directory_path+Task+"/"+f"{id}.jpg")  # Save image to disk
        segment_data[id]["images"].append({"id": id+".jpg"})

        # Append question and answer info
        segment_data[id]["questions"].append({
            "question": question,
            "GT_answer": answer,
            "Gpt_answer": ""
        })

# Save the consolidated dictionary to a single JSON file after all files are processed
with open(output_json_path, 'w', encoding='utf-8') as file:
    json.dump(segment_data, file, indent=4, ensure_ascii=False)

print(f"All data successfully saved to {output_json_path}")
