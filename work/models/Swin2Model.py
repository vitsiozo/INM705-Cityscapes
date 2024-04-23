import logging
import torch
import timm
import torchvision

from torch import Tensor, nn
from torchvision.transforms import Resize

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
        self.upsample = nn.ConvTranspose2d(inlen, size, kernel_size = 2, stride = 2)
        self.conv2 = Block(2 * size, size)
        self.conv1 = Block(size, size)

    def forward(self, left: Tensor, x: Tensor) -> Tensor:
        x = self.upsample(x)
        x = torch.cat([left, x], dim = 1)
        x = self.conv2(x)
        x = self.conv1(x)
        return x

class Upsampler2(nn.Module):
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

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        model_name = 'swinv2_small_window8_256.ms_in1k'
        img_size = 768

        # self.resize = Resize(img_size)
        self.backbone = timm.create_model(model_name, pretrained = True, features_only = True, img_size = img_size)

    def forward(self, x):
        # x = self.resize(x)
        fmaps = self.backbone(x)
        fmap1, fmap2, fmap3, fmap4 = [x.permute(0, 3, 1, 2) for x in fmaps]

        # fmap1: b × 96 × 192 × 192
        # fmap2: b × 192 × 96 × 96
        # fmap3: b × 384 × 48 × 48
        # fmap4: b × 768 × 24 × 24
        return fmap1, fmap2, fmap3, fmap4

class Swin2Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.backbone = Backbone()
        self.dec3 = Upsampler(768, 384)
        self.dec2 = Upsampler(384, 192)
        self.dec1 = Upsampler(192, 96)

        self.final1 = Upsampler2(96, 48)
        self.final2 = Upsampler2(48, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        fm1, fm2, fm3, x = self.backbone(x)
        x = self.dec3(fm3, x)
        x = self.dec2(fm2, x)
        x = self.dec1(fm1, x)
        x = self.final1(x)
        x = self.final2(x)
        return x
