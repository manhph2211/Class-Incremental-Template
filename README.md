Class-Incremental
=====

# Introduction

This project aims to handle the image classification using Class-Incremental Method :smile: 

# Dataset

Now we have 2 sets: train and val (public_test) corresponding to 2 folders in `data`. There are 10 phases and each phases we will have 10 different classes so in total we have 100 classes.

Before going to the project experiment, we need to make sure that the data structure follow the below format!

## Structure
    
    ```
    .
    ├── data
    
    │   ├── raw
    │   │   ├── train
    │   │   │   ├── phase 1
    │   │   │   │   ├── class_0
    │   │   │   │   │   ├── 000.jpg
    │   │   │   │   │   ├── ...
    │   │   │   │   ├── class_1
    │   │   │   │   ├── ...
    │   │   │   │   ├── class_10
    │   │   │   ├── phase 2
    │   │   │   ├── ...
    │   │   │   ├── phase 10
    │   │   ├── public_test
    │   │   │   ├── 00.jpg
    │   │   │   ├── ...
    
    │   ├── processed
    │   │   ├── train
    │   │   ├── public_test
    ```
# Set-up

```
git clone https://github.com/manhph2211/Class-Incremental.git
cd Class-Incremental
conda create -n ci python=3.11
conda activate ci
pip install -r requirements.txt
```

# Usage

```
python src/tools/segment.py --output "" --image_folder "" # optional
python src/tools/train.py 
python src/tools/classify.py
```


