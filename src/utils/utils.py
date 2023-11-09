import matplotlib.pyplot as plt
import numpy as np
import gdown
import os
import zipfile
import json
import shutil
from tqdm import tqdm
import glob


def move_file_to_folder(source_file_path, destination_folder_path):
    try:
        shutil.move(source_file_path, destination_folder_path)
        print(f"Moved {source_file_path} to {destination_folder_path}")
    except Exception as e:
        print(f"Error: {e}")


def download_files_from_google_drive(url, name='train.zip', destination_folder='data/raw'):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    output = os.path.join(destination_folder, name)

    gdown.download(url=url, output=output, quiet=False, fuzzy=True)
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)
        os.remove(output)


def save_dict_as_json(data_dict, json_file_path):
    try:
        with open(json_file_path, 'w') as json_file:
            json.dump(data_dict, json_file, indent=4)
    except Exception as e:
        print(f'Error: {e}')


def load_json_as_dict(json_file_path="outputs/validation.json"):
    try:
        with open(json_file_path, 'r') as json_file:
            data_dict = json.load(json_file)
        return data_dict
    except FileNotFoundError:
        print(f'Error: File not found - {json_file_path}')
        return None
    except json.JSONDecodeError as e:
        print(f'Error: JSON decoding error - {e}')
        return None


def analyze_json(json_file_path="outputs/validation.json"):
    data = load_json_as_dict(json_file_path)
    class_counts = {}

    for class_label in data.values():
        if class_label in class_counts:
            class_counts[class_label] += 1
        else:
            class_counts[class_label] = 1

    # Extract class labels and their corresponding counts
    class_labels = list(class_counts.keys())
    counts = list(class_counts.values())

    # Create a bar chart or histogram
    plt.figure(figsize=(10, 6))
    plt.bar(class_labels, counts)
    plt.xlabel('Class Labels')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution')
    plt.xticks(rotation=45)  
    plt.show()


def number_to_padded_string(number):
    if 0 <= number < 100:
        return str(number).zfill(3)  
    else:
        return "Number out of range"
        

def merge_semi_data(json_file_path="outputs/validation.json"):
    data = load_json_as_dict(json_file_path)
    for key, value in tqdm(data.items()): 
        destination_folder_path = os.path.join("data/raw/Train", f"phase_{1 + value // 10}", f"{number_to_padded_string(value)}")
        current_img_nums = len(glob.glob(os.path.join(destination_folder_path, "*.jpg")))
        move_file_to_folder(os.path.join("data/raw/Val", key), os.path.join(destination_folder_path, f"{current_img_nums+1}.jpg"))


if __name__ == "__main__":
    # download_files_from_google_drive(url="https://drive.google.com/file/d/12gRUJ5nT1RkAxZcRMf2pmnR3yn7kdmAH/view?usp=sharing", name="val.zip")
    # download_files_from_google_drive(url="https://drive.google.com/file/d/1XC2tWUdWN_rUKRwuiLuXNYYl1mTHLpxt/view?usp=sharing", name="train.zip")
    # analyze_json()
    merge_semi_data()