import torch
import torch.nn as nn
from models.blocks import AOTBlock, UpConv, SEBlock

class Generator(nn.Module):
    def __init__(self, input_channels=4, residual_blocks=8):
        super(Generator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, 7),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        blocks = []
        for _ in range(residual_blocks):
            blocks.append(AOTBlock(256, [1, 2, 4, 8]))
        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            UpConv(256, 128),
            SEBlock(128),
            UpConv(128, 64),
            SEBlock(64),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x
