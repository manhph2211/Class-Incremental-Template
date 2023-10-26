import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import MemoryEfficientSwish
import torch.nn as nn
import torch


class MLPMixerBlock(nn.Module):
    def __init__(self, in_channels, mlp_dim):
        super(MLPMixerBlock, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels, mlp_dim),
            nn.GELU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(mlp_dim, in_channels)
        )
    
    def forward(self, x):
        out = self.mlp1(x)
        out = self.mlp2(out)
        out += x
        return out

class MLPMixer(nn.Module):
    def __init__(self, num_blocks, in_channels, mlp_dim):
        super(MLPMixer, self).__init__()
        self.blocks = nn.ModuleList([MLPMixerBlock(in_channels, mlp_dim) for _ in range(num_blocks)])
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Baseline(nn.Module):
    def __init__(self, pretrained_checkpoint=None, num_classes=10, device='cuda'):
        super(Baseline, self).__init__()
        self.extractor = EfficientNet.from_pretrained('efficientnet-b0').to(device)
        self.extractor._fc = nn.Identity()

        try:
            checkpoint = torch.load(pretrained_checkpoint, map_location=device)
            self.extractor.load_state_dict({k.replace('extractor.', ''): v for k, v in checkpoint.items() if k.startswith('extractor.')})
            print(f"Done loading model for phase {num_classes // 10}")
        except:
            pass

        # for param in self.extractor.parameters():
        #     param.requires_grad = False

        mlp_dim = 512
        num_blocks = 4  
        self.mixer = MLPMixer(num_blocks, in_channels=1280, mlp_dim=mlp_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(0.05),
            nn.Linear(1280, 512),
            MemoryEfficientSwish(),
            nn.Dropout(0.05),
            nn.Linear(512, num_classes),
            MemoryEfficientSwish()
        ).to(device)

    def forward(self, x):
        x = self.extractor(x)
        x = self.mixer(x)  
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = Baseline()
    print(model)
    print(model(torch.randn(1, 3, 224, 224)).shape)
