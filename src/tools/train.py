import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import sys
sys.path.append(".")
from src.dataloader.dataset import get_loader
from src.utils.rand_augment import RandAugment
from src.models.classifier.baseline import Baseline
from src.models.classifier.pretrained_clip import CLIPClassifier
from src.utils.losses import LabelSmoothingCrossEntropy
import torchvision.transforms as transforms
from tqdm import tqdm 
import numpy as np


class Trainer:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # RandAugment(n=3, m=5),
            transforms.RandomVerticalFlip(),  
            transforms.RandomHorizontalFlip(),  
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])    
            ])
        self.models = {}

    def train_one_phase(self, phase, max_epochs, lr=0.0001):
        best_train_accuracy, best_val_accuracy = 0.0, 0.0
        best_loss = np.inf
        model = CLIPClassifier(pretrained_checkpoint=f"checkpoints/phase_{phase-1}_model.pth", num_classes=phase*10, device=self.device)
        criterion = LabelSmoothingCrossEntropy()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=3, verbose=True)

        train_loader, val_loader = get_loader(phase=phase, transform=self.transform)
        
        for epoch in range(max_epochs):
            model.train()
            running_loss = 0.0
            val_running_loss = 0.0
            train_predictions, train_labels = [], []

            for inputs, labels in tqdm(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs.to(self.device))
                loss = criterion(outputs, labels.to(self.device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_predictions.extend(predicted.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

            train_accuracy = accuracy_score(train_labels, train_predictions)
            if train_accuracy > best_train_accuracy:
                best_train_accuracy = train_accuracy

            val_predictions, val_labels = [], []

            with torch.no_grad():
                for inputs, labels in tqdm(val_loader):
                    outputs = model(inputs.to(self.device))
                    val_loss = criterion(outputs, labels.to(self.device))
                    val_running_loss += val_loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_predictions.extend(predicted.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            val_accuracy = accuracy_score(val_labels, val_predictions)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy

            scheduler.step(val_running_loss)

            print(f"Phase {phase} - Epoch [{epoch + 1}/{max_epochs}] - Val Loss: {val_running_loss:.4f} - Train Acc: {train_accuracy:.4f} - Val Acc: {val_accuracy:.4f}")

            if val_running_loss < best_loss:
                best_loss = val_running_loss
                torch.save(model.state_dict(), f"checkpoints/phase_{phase}_model.pth")

        self.models[phase] = model

        return best_train_accuracy, best_val_accuracy
    
    def train(self):
        for phase in range(1,11):
            print(f"************** Start traning phase {phase} **************")
            train_accuracy, val_accuracy = self.train_one_phase(phase, max_epochs=100)
            print(f"************** Phase {phase} - Best Train Acc: {train_accuracy:.4f} - Best Val Acc: {val_accuracy:.4f} **************")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()