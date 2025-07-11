import torch
import torch.nn as nn
import torch.nn.functional as F

def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat

class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            setattr(self, f"block_{str(i).zfill(2)}", nn.Sequential(
                nn.ReflectionPad2d(rate),
                nn.Conv2d(dim, dim // 4, 3, padding=0, dilation=rate),
                nn.ReLU(inplace=True)
            ))
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0)
        )
        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0)
        )

    def forward(self, x):
        out = [getattr(self, f"block_{str(i).zfill(2)}")(x) for i in range(len(self.rates))]
        out = torch.cat(out, dim=1)
        out = self.fuse(out)
        mask = torch.sigmoid(my_layer_norm(self.gate(x)))
        return x * (1 - mask) + out * mask

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale
