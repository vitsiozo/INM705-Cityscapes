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
        model_name = 'swinv2_base_window8_256.ms_in1k'
        img_size = 768

        # self.resize = Resize(img_size)
        self.backbone = timm.create_model(model_name, pretrained = True, features_only = True, img_size = img_size)

    def forward(self, x):
        # x = self.resize(x)
        fmaps = self.backbone(x)
        fmap1, fmap2, fmap3, fmap4 = [x.permute(0, 3, 1, 2) for x in fmaps]

        # fmap1: b × 128 × 192 × 192
        # fmap2: b × 256 × 96 × 96
        # fmap3: b × 512 × 48 × 48
        # fmap4: b × 1024 × 24 × 24
        return fmap1, fmap2, fmap3, fmap4

class Swin2BaseModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = .1):
        super().__init__()
        # The backbone can only work with a 768 × 768 image.
        self.resizedown = Resize((768, 768))

        self.backbone = Backbone()

        self.bottleneck = Block(1024, 1024, dropout)
        self.dottleneck = Block(1024, 1024, dropout)

        # self.dec4 = Upsampler(1536, 768)
        self.dec3 = Upsampler(1024, 512, dropout)
        self.dec2 = Upsampler(512, 256, dropout)
        self.dec1 = Upsampler(256, 128, dropout)

        self.final1 = Upsampler2(128, 64, dropout)
        self.final2 = Upsampler2(64, 32, dropout)
        self.final3 = nn.Conv2d(32, out_channels, kernel_size = 1)

        # We want to resize the result to the same size as the original image.
        # Since we don't know which one this is, we resize dynamically.

    def forward(self, x: Tensor) -> Tensor:
        size = x.shape[2:]
        x = self.resizedown(x)

        fm1, fm2, fm3, fm4 = self.backbone(x)
        x = fm4

        x = self.bottleneck(x)
        x = self.dottleneck(x)

        # x = self.dec4(fm4, x)
        x = self.dec3(fm3, x)
        x = self.dec2(fm2, x)
        x = self.dec1(fm1, x)
        x = self.final1(x)
        x = self.final2(x)
        x = self.final3(x)

        x = torchvision.transforms.functional.resize(x, size = size)
        return x
