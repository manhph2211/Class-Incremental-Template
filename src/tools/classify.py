import os
import zipfile
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import sys
sys.path.append(".")
from src.dataloader.dataset import CustomTestDataset
from src.models.classifier.mlpmixer import Baseline
from src.models.classifier.pretrained_clip import CLIPClassifier
from src.utils.utils import save_dict_as_json, load_json_as_dict
from torch.nn.functional import softmax


class Inference:
    def __init__(self, path_to_folder, threshold=0.6):
        self.path_to_folder = path_to_folder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = self.load_models()
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])    
        ])
        self.threshold = threshold
        self.good_samples = dict()

    def load_models(self):
        models = []
        for phase in range(1, 11):
            model = Baseline(pretrained_checkpoint=None, num_classes=10*phase, device=self.device)
            model_path = f"checkpoints/phase_{phase}_model.pth"  
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            models.append(model)
        return models

    def infer(self):
        results = []
        image_folder = CustomTestDataset(path_to_folder=self.path_to_folder, transform=self.data_transforms)
        data_loader = DataLoader(image_folder, batch_size=1, shuffle=False)
        results = {}
        for i, model in enumerate(self.models):
            predictions = []
            with torch.no_grad():
              for image, filename in tqdm(data_loader):
                    image = image.to(self.device)
                    output = model(image)
                    #   _, predicted_class = output.max(1)
                    probabilities = softmax(output, dim=1) 

                    max_probability, predicted_class = probabilities.max(1)
                    print(max_probability)
                    if max_probability.item() >= self.threshold:
                        if filename in self.good_samples:
                            if self.good_samples[filename[0]] < max_probability.item():
                                self.good_samples[filename[0]] = predicted_class[0].cpu().detach().numpy().tolist()
                        else:
                            self.good_samples[filename[0]] = predicted_class[0].cpu().detach().numpy().tolist()
                            
                    predictions.append((filename[0], predicted_class[0]))
            results[i+1] = predictions
        save_dict_as_json(self.good_samples, "outputs/validation.json")
        return results

    def submit(self):
        results = self.infer()
        zip_filename = "outputs/results.zip"

        with zipfile.ZipFile(zip_filename, "w") as zipf:
            for phase in results:
                txt_filename = f"result_{phase}.txt"
                with open(txt_filename, "w") as txtf:
                    for image_name, prediction in results[phase]:
                        txtf.write(f"{image_name} {prediction}\n")
                zipf.write(txt_filename)


if __name__ == "__main__":
    inference = Inference(path_to_folder="data/raw/Test")
    inference.submit()