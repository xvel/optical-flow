import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Block(nn.Module):
    def __init__(self, dim, ffdim):
        super().__init__()
        self.gc = dim//8
        self.conv3 = nn.Conv2d(self.gc, self.gc, groups=self.gc, kernel_size=3, padding="same")
        self.conv5 = nn.Conv2d(self.gc, self.gc, groups=self.gc, kernel_size=5, padding="same")
        self.conv9 = nn.Conv2d(self.gc, self.gc, groups=self.gc, kernel_size=3, dilation=3, padding="same")
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ffdim*2)
        self.fc2 = nn.Linear(ffdim, dim)
        
    def forward(self, xx):
        sx = torch.split(xx, (self.gc, self.gc, self.gc, 5*self.gc), dim=1)
        x1 = self.conv3(sx[0])
        x2 = self.conv5(sx[1])
        x3 = F.avg_pool2d(sx[2], 3, stride=1, padding=1)
        x3 = self.conv9(x3)
        x = torch.cat([x1, x2, x3, sx[3]], dim=1)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.ln(x)
        x = self.fc1(x)
        x = torch.chunk(x, 2, -1)
        x = self.fc2(F.gelu(x[0]) * x[1])
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return x + xx


class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = self.conv(x)
        return x


class InceptionNext(nn.Module):
    def __init__(self, dims=(128, 256), ffdims=(192, 384), num_layers=(4, 8)):
        super().__init__()
        self.stem = nn.Conv2d(in_channels=6, out_channels=dims[0], kernel_size=(5, 5), stride=(5, 5))
        self.ln = nn.LayerNorm(dims[0])
        
        self.layers1 = nn.ModuleList([Block(dims[0], ffdims[0]) for _ in range(num_layers[0])])
        self.downsample = Downsample(dims[0], dims[1])
        self.layers2 = nn.ModuleList([Block(dims[1], ffdims[1]) for _ in range(num_layers[1])])

        self.out = nn.Conv2d(dims[1], 8, kernel_size=1)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.stem(x)
        
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        
        for layer in self.layers1:
            x = layer(x)
        x = self.downsample(x)
        for layer in self.layers2:
            x = layer(x)
        x = self.out(x)
        return x
