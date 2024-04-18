import logging
import torch

from torch import Tensor, nn
import torchvision

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

class Upsampler(nn.Module):
    def __init__(self, inlen: int, size: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(inlen, size, kernel_size = 2, stride = 3, output_padding = 1)
        self.conv2 = Block(size, size)
        self.conv1 = Block(size, size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample(x)
        x = self.conv2(x)
        x = self.conv1(x)
        return x

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torchvision.models.vit_h_14(weights = torchvision.models.ViT_H_14_Weights.DEFAULT)
        self.transformer.heads = nn.Identity()

    def forward(self, x):
        x = self.transformer(x)
        x = x.unsqueeze(2).unsqueeze(3)
        return x

class UNetTransformerModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.transformer = Transformer()

        self.dec5 = Upsampler(1024, 512)
        self.dec4 = Upsampler(512, 256)
        self.dec3 = Upsampler(256, 128)
        self.dec2 = Upsampler(128, 64)
        # self.dec1 = Upsampler(64, 32)

        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size = 1),
            nn.AdaptiveAvgPool2d((224, 224)),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.transformer(x)
        x = self.dec5(x)
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        # x = self.dec1(x)
        x = self.final(x)
        return x
