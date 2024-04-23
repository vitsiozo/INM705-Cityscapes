import torch
from torch import Tensor, nn

class NewBaselineModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size = 1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 5, padding = 2)

        self.conv4 = nn.Conv2d(64, 64, kernel_size = 9, padding = 4)

        self.conv5 = nn.Conv2d(64, 32, kernel_size = 5, padding = 2)
        self.conv6 = nn.Conv2d(32, 32, kernel_size = 3, padding = 1)
        self.conv7 = nn.Conv2d(32, out_channels, kernel_size = 1)

    def forward(self, x: Tensor) -> Tensor:
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.conv3(x)
        print(x.shape)
        x = self.conv4(x)
        print(x.shape)
        x = self.conv5(x)
        print(x.shape)
        x = self.conv6(x)
        print(x.shape)
        x = self.conv7(x)
        print(x.shape)
        return x
