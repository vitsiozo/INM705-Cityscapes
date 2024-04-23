import logging
import torch

from torch import Tensor, nn

class Block(nn.Module):
    def __init__(self, inlen: int, outlen: int):
        super().__init__()
        self.conv = nn.Conv2d(inlen, outlen, kernel_size = 3, padding = 1)
        self.relu = nn.ReLU()
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.relu(x)
        return x

class Downsampler(nn.Module):
    def __init__(self, inlen: int, size: int):
        super().__init__()
        self.conv1 = Block(inlen, size)
        self.conv2 = Block(size, size)
        self.downsample = nn.MaxPool2d(kernel_size = (2, 2), stride = 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.downsample(x)
        return x

class Upsampler(nn.Module):
    def __init__(self, inlen: int, size: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(inlen, size, kernel_size = 2, stride = 2)
        self.conv2 = Block(size, size)
        self.conv1 = Block(size, size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample(x)
        x = self.conv2(x)
        x = self.conv1(x)
        return x

class BaselineNoBatchNormModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.enc1 = Downsampler(in_channels, 64)
        self.enc2 = Downsampler( 64, 128)
        self.enc3 = Downsampler(128, 256)

        self.bottleneck = Block(256, 512)
        self.dottleneck = Block(512, 256)

        self.dec3 = Upsampler(256, 128)
        self.dec2 = Upsampler(128, 64)
        self.dec1 = Upsampler(64, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size = 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.bottleneck(x)
        x = self.dottleneck(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        x = self.final(x)
        return x

