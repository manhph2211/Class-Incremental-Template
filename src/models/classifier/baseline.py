from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import MemoryEfficientSwish
import torch.nn as nn
import torch


class Baseline(nn.Module):
    def __init__(self, pretrained_checkpoint=None, num_classes=10, device='cuda'):
        super(Baseline, self).__init__()
        self.extractor = EfficientNet.from_pretrained('efficientnet-b0').to(device)

        try:
          checkpoint = torch.load(pretrained_checkpoint, map_location=device)
          self.extractor.load_state_dict({k.replace('extractor.', ''): v for k, v in checkpoint.items() if k.startswith('extractor.')})
          print(f"Done load model for phase {num_classes//10}")
        except:
          pass
        self.base = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1000, num_classes),
            MemoryEfficientSwish()
        ).to(device)

    def forward(self, x):
        x = self.extractor(x)
        x = self.base(x)
        return x


if __name__ == "__main__":
    model = Baseline()
    print(model)
    print(model(torch.randn(1, 3, 224, 224)).shape)