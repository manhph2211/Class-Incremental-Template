import torch.nn as nn
import torch
import clip


class CLIPClassifier(nn.Module):
    def __init__(self, pretrained_checkpoint=None, num_classes=10, device='cuda'):
        super(CLIPClassifier, self).__init__()
        self.extractor, _ = clip.load("ViT-B/32", device='cpu')
        self.extractor = self.extractor.to(device)
        try:
          checkpoint = torch.load(pretrained_checkpoint, map_location=device)
          self.extractor.load_state_dict({k.replace('extractor.', ''): v for k, v in checkpoint.items() if k.startswith('extractor.')})
          print(f"Done load model for phase {num_classes//10}")
        except:
          pass
        self.base = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        ).to(device)

    def forward(self, x):
        x = self.extractor.encode_image(x)
        x = self.base(x)
        return x


if __name__ == "__main__":
    model = Baseline()
    print(model(torch.randn(1, 3, 224, 224).to('cuda')).shape)