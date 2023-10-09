import os
import glob
import torch
import matplotlib.pyplot as plt
import random
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class ImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes()
        self.samples = self._make_dataset()

    def _find_classes(self):
        classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls: int(cls) for i, cls in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self):
        samples = []
        for target_class in self.classes:
            class_dir = os.path.join(self.root, target_class)
            if not os.path.isdir(class_dir):
                continue

            for root, _, fnames in sorted(os.walk(class_dir)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, self.class_to_idx[target_class])
                    samples.append(item)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, target
    

class CustomTrainDataset(Dataset):
    def __init__(self, original_dataset, additional_samples=None):
        self.original_dataset = original_dataset
        self.additional_samples = additional_samples if additional_samples else []
    
    def __len__(self):
        return len(self.original_dataset) + len(self.additional_samples)

    def __getitem__(self, index):
        if index < len(self.original_dataset):
            return self.original_dataset[index]
        else:
            additional_index = index - len(self.original_dataset)
            return self.additional_samples[additional_index]


class CustomTestDataset(Dataset):
    def __init__(self, path_to_folder='data/raw/val', transform=None):
        self.path_to_folder = path_to_folder
        self.transform = transform
        self.image_names = os.listdir(path_to_folder)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.path_to_folder, image_name)
        image = Image.open(image_path).convert('RGB') 

        if self.transform:
            image = self.transform(image)

        return image, image_name