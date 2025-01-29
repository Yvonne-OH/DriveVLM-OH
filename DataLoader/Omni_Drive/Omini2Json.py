import json
import glob
import os
from PIL import Image
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm

#%%
# Define constants
TASK = "desc"
STAGE = "train"
NU_SCENES_DATAROOT = "/media/workstation/6D3563AC52DC77EA/Data/Nuscenes/"
nusc = NuScenes(version='v1.0-trainval', dataroot=NU_SCENES_DATAROOT, verbose=True)
OMNI_PATH = f"/media/workstation/6D3563AC52DC77EA/Data/Omni_Drive/data_nusc/{TASK}/{STAGE}/"
OUTPUT_JSON_PATH = OMNI_PATH + 'json_data/' + f'{TASK}_{STAGE}.json'  # Output JSON file path
OUTPUT_IMAGE_PATH = OMNI_PATH + 'image/'  # Output image directory path
# Sensors to be processed
SENSORS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

#%%
def create_directory(path):
    """Create a directory if it does not exist, otherwise raise a FileExistsError if it is not empty."""
    if os.path.exists(path):
        if os.listdir(path):  # Non-empty directory
            print(f"Directory is not empty: {path}")
            raise FileExistsError(f"Directory already exists: {path}")
        else:
            print(f"Directory is empty: {path}")
    else:
        os.makedirs(path)
        print(f"Directory created: {path}")

#%%
if __name__ == "__main__":
    try:
        create_directory(OUTPUT_IMAGE_PATH)
    except FileExistsError as e:
        print(e)

    files = [f for f in glob.glob(OMNI_PATH + '/*') if os.path.isfile(f)]

    segment_data = {}

    for file in tqdm(files, desc="Processing files"):
        token = os.path.basename(file).split(".")[0]

        try:
            sample = nusc.get('sample', token)
            images = [nusc.get('sample_data', sample['data'][sensor])['filename'].replace('samples/', '') for sensor in SENSORS]

            if token not in segment_data:
                segment_data[token] = {"token": token, "images": images, "questions": []}
            else:
                segment_data[token]["images"].extend(images)

            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if TASK == 'conv':
                for row in data:
                    segment_data[token]["questions"].append({
                        "question": row['question'],
                        "GT_answer": row['answer'],
                        "Gpt_answer": ""
                    })
            elif TASK == 'desc':
                segment_data[token]["questions"].append({
                    "question": data['description'],
                    "GT_answer": data['action'],
                    "Gpt_answer": ""
                })

            token_path = os.path.join(OUTPUT_IMAGE_PATH, token)
            os.makedirs(token_path, exist_ok=True)  # Ensure directory exists
            for image in images:
                img_path = os.path.join(NU_SCENES_DATAROOT, 'samples', image)
                with Image.open(img_path) as img:
                    img.save(os.path.join(token_path, os.path.basename(image)))

        except Exception as e:
            print(f"Failed to find {token}: {str(e)}")
            continue

    os.makedirs(os.path.join(OMNI_PATH, 'json_data'), exist_ok=True)
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as file:
        json.dump(segment_data, file, indent=4, ensure_ascii=False)

    print(f"All data successfully saved to {OUTPUT_JSON_PATH}")