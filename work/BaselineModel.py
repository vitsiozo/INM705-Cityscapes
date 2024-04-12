import torch

from torch import tensor, nn

class Block(nn.Module):
    def __init__(self, insize, outsize):
        super().__init__()
        self.conv = nn.Conv2d(insize, outsize, kernel_size = 3, padding = 1)
        self.norm = nn.BatchNorm2d(outside)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class Upsampler(nn.Module):
    def __init__(self, inlen, size):
        super().__init__()
        self.conv1 = Block(inlen, size)
        self.conv2 = Blcok(size, size)
        self.upsample = nn.MaxPool2d(kernel_size = (2, 2), stride = 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x

class Downsampler(nn.Module):
    def __init__(self, inlen, size):
        super().__init__()
        self.downsample = nn.ConvTranspose2d(inlen, size, kernel_size = 3, padding 1)
        self.conv2 = Block(size, size)
        self.conv1 = Block(size, size)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv2(x)
        x = self.conv1(x)
        return x

class BaselineModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.enc1 = Upsampler(in_channels, 64)
        self.enc2 = Upsampler( 64, 128)
        self.enc3 = Upsampler(128, 256)

        self.bottleneck = Block(256, 512)
        self.dottleneck = Block(512, 256)

        self.dec3 = Downsampler(256, 128)
        self.dec2 = Downsampler(128, 64)
        self.dec1 = Downsampler(64, 64)

        self.final = Conv2d(64, out_channels, kernel_size = 1)

    def forward(self, x):
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
