Class-Incremental
=====

# Introduction

This project aims to handle the image classification using Class-Incremental Method, FastSAM, and CLIP model :smile: 

# Dataset

Now suppose that we have 2 sets in raw: train and val (public_test) corresponding to 2 folders in `data/raw`. There are 10 phases and each phases we will have 10 different classes so in total we have 100 classes.

Before going to the project experiment, we need to make sure that the data structure follow the below format!

## Structure
    
    ```
    .
    ├── data
    
    │   ├── raw
    │   │   ├── Train
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
    │   │   ├── Val
    │   │   │   ├── 00.jpg
    │   │   │   ├── ...
    
    │   ├── processed
    │   │   ├── Train
    │   │   ├── Val
    ```
# Set-up

```
git clone https://github.com/manhph2211/Class-Incremental.git
cd Class-Incremental
conda create -n ci python=3.11
conda activate ci
pip install -r requirements.txt
gdown --fuzzy https://drive.google.com/file/d/1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv/view -O src/models/segmentator/weights
```

# Usage

``` 
python src/tools/segment.py --output " " --image_folder " " --text_prompt "eg. a bird" # Optional
 
python src/tools/train.py # Training and validation

python src/tools/classify.py # Testing an image folder and submission

```

# Reference

```
@misc{zhao2023fast,
      title={Fast Segment Anything},
      author={Xu Zhao and Wenchao Ding and Yongqi An and Yinglong Du and Tao Yu and Min Li and Ming Tang and Jinqiao Wang},
      year={2023},
      eprint={2306.12156},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

