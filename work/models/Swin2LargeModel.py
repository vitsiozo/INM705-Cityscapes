import logging
import torch
import timm
import torchvision

from torch import Tensor, nn
from torchvision.transforms import Resize

class Block(nn.Module):
    def __init__(self, inlen: int, outlen: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv2d(inlen, outlen, kernel_size = 3, padding = 1)
        self.norm = nn.BatchNorm2d(outlen)
        self.relu = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout2d(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class Upsampler(nn.Module):
    def __init__(self, inlen: int, size: int, dropout: float):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(inlen, size, kernel_size = 2, stride = 2)
        self.conv2 = Block(2 * size, size, dropout)
        self.conv1 = Block(size, size, dropout)

    def forward(self, left: Tensor, x: Tensor) -> Tensor:
        x = self.upsample(x)
        x = torch.cat([left, x], dim = 1)
        x = self.conv2(x)
        x = self.conv1(x)
        return x

class Upsampler2(nn.Module):
    def __init__(self, inlen: int, size: int, dropout: float):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(inlen, size, kernel_size = 2, stride = 2)
        self.conv2 = Block(size, size, dropout)
        self.conv1 = Block(size, size, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample(x)
        x = self.conv2(x)
        x = self.conv1(x)
        return x

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        model_name = 'swinv2_large_window12_192.ms_in22k'
        img_size = 768

        # self.resize = Resize(img_size)
        self.backbone = timm.create_model(model_name, pretrained = True, features_only = True, img_size = img_size)

    def forward(self, x):
        # x = self.resize(x)
        fmaps = self.backbone(x)
        fmap1, fmap2, fmap3, fmap4 = [x.permute(0, 3, 1, 2) for x in fmaps]

        # fmap1: b × 192 × 192 × 192
        # fmap2: b × 384 × 96 × 96
        # fmap3: b × 768 × 48 × 48
        # fmap4: b × 1536 × 24 × 24
        return fmap1, fmap2, fmap3, fmap4

class Swin2LargeModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = .1):
        super().__init__()
        self.backbone = Backbone()

        self.bottleneck = Block(1536, 1536, dropout)
        self.dottleneck = Block(1536, 1536, dropout)

        self.dec3 = Upsampler(1536, 768, dropout)
        self.dec2 = Upsampler(768, 384, dropout)
        self.dec1 = Upsampler(384, 192, dropout)

        self.final1 = Upsampler2(192, 96, dropout)
        self.final2 = Upsampler2(96, 48, dropout)
        self.final3 = nn.Conv2d(48, out_channels, kernel_size = 1)

    def forward(self, x: Tensor) -> Tensor:
        fm1, fm2, fm3, fm4 = self.backbone(x)
        x = fm4

        x = self.bottleneck(x)
        x = self.dottleneck(x)

        x = self.dec3(fm3, x)
        x = self.dec2(fm2, x)
        x = self.dec1(fm1, x)
        x = self.final1(x)
        x = self.final2(x)
        x = self.final3(x)
        return x
