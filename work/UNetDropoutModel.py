import logging
import torch

from torch import Tensor, nn

class Block(nn.Module):
    def __init__(self, inlen: int, outlen: int):
        super().__init__()
        self.conv = nn.Conv2d(inlen, outlen, kernel_size = 3, padding = 1)
        self.norm = nn.BatchNorm2d(outlen)
        self.relu = nn.ReLU()
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class Downsampler(nn.Module):
    def __init__(self, inlen: int, size: int):
        super().__init__()
        self.conv1 = Block(inlen, size)
        self.conv2 = Block(size, size)
        self.downsample = nn.MaxPool2d(kernel_size = (2, 2), stride = 2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        right = x
        x = self.downsample(x)
        return x, right

class Upsampler(nn.Module):
    def __init__(self, inlen: int, size: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(inlen, size, kernel_size = 2, stride = 2)
        self.conv2 = Block(2 * size, size)
        self.conv1 = Block(size, size)

    def forward(self, left: Tensor, x: Tensor) -> Tensor:
        x = self.upsample(x)
        x = torch.cat([left, x], dim = 1)
        x = self.conv2(x)
        x = self.conv1(x)
        return x

class UNetModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.enc1 = Downsampler(in_channels, 64)
        self.enc2 = Downsampler( 64, 128)
        self.enc3 = Downsampler(128, 256)
        self.enc4 = Downsampler(256, 512)

        self.bottleneck = Block(512, 1024)
        self.dottleneck = Block(1024, 1024)

        self.dec4 = Upsampler(1024, 512)
        self.dec3 = Upsampler(512, 256)
        self.dec2 = Upsampler(256, 128)
        self.dec1 = Upsampler(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size = 1)

    def forward(self, x: Tensor) -> Tensor:
        x, res1 = self.enc1(x)
        x, res2 = self.enc2(x)
        x, res3 = self.enc3(x)
        x, res4 = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dottleneck(x)
        x = self.dec4(res4, x)
        x = self.dec3(res3, x)
        x = self.dec2(res2, x)
        x = self.dec1(res1, x)
        x = self.final(x)
        return x
