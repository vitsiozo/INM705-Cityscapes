import torch
import torchvision

from torch import nn
from torch import FloatTensor

class VGGBaselineModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        backbone = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.decode1 = nn.ConvTranspose2d(2048, 1024, kernel_size = 2, stride = 2)
        self.decode2 = nn.ConvTranspose2d(1024, 512, kernel_size = 2, stride = 2)
        self.decode3 = nn.ConvTranspose2d(512, 256, kernel_size = 2, stride = 2)
        self.decode4 = nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2)
        self.decode5 = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2)
        self.final = nn.Conv2d(64, out_channels, kernel_size = 1)

    def forward(self, x: FloatTensor):
        x = self.features(x)
        x = self.decode1(x)
        x = self.decode2(x)
        x = self.decode3(x)
        x = self.decode4(x)
        x = self.decode5(x)
        x = self.final(x)
        return x
