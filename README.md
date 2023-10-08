Class-Incremental
=====

# Introduction

In this project, we will aim to handle the image classification using Class-Incremental Method :smile: Hope that we can do the best!

# Dataset

Now we have 2 sets: train and val (public_test) corresponding to 2 folders in `data`. There are 10 phases and each phases we will have 10 different classes so in total we have 100 classes.

# Set-up

```
conda create -n ci python=3.11
conda activate ci
pip install -r requirements.txt
```

# Usage

```
python src/tools/segment_bird.py --output "" --image_folder "" # optional
python src/tools/train.py 
python src/tools/inference.py
```

