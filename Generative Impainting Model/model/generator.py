import torch.nn as nn
from .aot_block import AOTBlock, UpConv

class AOTGenerator(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super(AOTGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.middle = nn.Sequential(
            AOTBlock(256, rates=[1, 2, 4, 8]),
            AOTBlock(256, rates=[1, 2, 4, 8])
        )
        self.decoder = nn.Sequential(
            UpConv(256, 128),
            UpConv(128, 64),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, 7),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x
