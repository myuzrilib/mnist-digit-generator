import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=10):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat([noise, self.label_embed(labels)], dim=1)
        img = self.model(x)
        return img.view(-1, 1, 28, 28)
