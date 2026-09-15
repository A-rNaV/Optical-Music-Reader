import torch
import torch.nn as nn
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(
                1, 32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                32, 64,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                64, 128,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                128, 256,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)