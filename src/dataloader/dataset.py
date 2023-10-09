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
import sys
sys.path.append(".")
from src.dataloader.custom_incremental import CustomTestDataset, CustomTrainDataset, ImageFolder


class DataHandler:
    def __init__(self, root='data/raw/Train', ratio=0.2, transform=None):
        self.root = root
        self.transform = transform
        self.ratio = ratio
        
    def get_classes(self, phase):
        all_classes = []
        for i in glob.glob(os.path.join(self.root,f"phase_{str(phase)}/*")):
            name = i.split("/")[-1]
            all_classes.append((name,int(name)))
        return dict(all_classes)

    def get_one_phase_dataset(self, phase):
        dataset = ImageFolder(root=os.path.join(self.root,f'phase_{str(phase)}'), transform=self.transform)
        total_samples = len(dataset)
        test_size = int(self.ratio * total_samples)
        train_size = total_samples - test_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
        return train_dataset, test_dataset

    def update_one_phase_dateset(self, phase):
        train_dataset, test_dataset = self.get_one_phase_dataset(phase)
        if phase > 1:
            additional_samples = []
            for previous_phase in range(1,phase):
                previous_train_dataset, previous_test_dataset = self.get_one_phase_dataset(previous_phase)
                test_dataset = ConcatDataset([test_dataset, previous_test_dataset])
                track = defaultdict(int)
                for i, (sample, label) in enumerate(previous_train_dataset):
                    if track[label] == 6:
                        continue
                    additional_samples.append((sample, label))
                    track[label]+=1

            train_dataset = CustomTrainDataset(train_dataset, additional_samples)
        return train_dataset, test_dataset

    def visualize_images_per_label(self, phase, num_images_per_label=5):
        dataset, _ = self.update_one_phase_dateset(phase)
        unique_labels = list(range(phase*10))
        images_per_label = {label: [] for label in unique_labels}

        for index in range(len(dataset)):
            image, label = dataset[index]
            if len(images_per_label[label]) < num_images_per_label:
                images_per_label[label].append(image)

        for label, images in images_per_label.items():
            plt.figure(figsize=(12, 2))
            for i, image in enumerate(images):
                plt.subplot(1, num_images_per_label, i + 1)
                plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))  
                plt.title(f"Label: {label}")
                plt.axis('off')
            plt.tight_layout()
        plt.show()


def get_loader(phase, transform):
    datahandler = DataHandler(transform=transform)
    train_dataset, val_dataset = datahandler.update_one_phase_dateset(phase)
    train_loader = DataLoader(
            dataset = train_dataset,
            batch_size = 64,
            num_workers = 2,
            shuffle=True
    )

    val_loader = DataLoader(
            dataset = val_dataset,
            batch_size = 64,
            num_workers = 2,
            shuffle=True
    )
    print("DONE LOADING DATA !")
    return train_loader, val_loader


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    handler = DataHandler(transform = transform)
    handler.visualize_images_per_label(phase=2)
    