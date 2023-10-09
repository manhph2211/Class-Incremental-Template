import matplotlib.pyplot as plt
import numpy as np
import gdown
import os
import zipfile


def download_files_from_google_drive(url, name='train.zip', destination_folder='data/raw'):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    output = os.path.join(destination_folder, name)

    gdown.download(url=url, output=output, quiet=False, fuzzy=True)
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)
        os.remove(output)

if __name__ == "__main__":
    download_files_from_google_drive(url="https://drive.google.com/file/d/12gRUJ5nT1RkAxZcRMf2pmnR3yn7kdmAH/view?usp=sharing", name="val.zip")
    download_files_from_google_drive(url="https://drive.google.com/file/d/1XC2tWUdWN_rUKRwuiLuXNYYl1mTHLpxt/view?usp=sharing", name="train.zip")