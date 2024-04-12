# models.py
import torch.nn as nn
import torch.nn.functional as F

class SimpleFCN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleFCN, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Decoder
        self.upsample1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2)

    def forward(self, x):
        # Encode
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2) # Pooling to reduce dimensionality
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2) 
        
        # Decode
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x) # output layer
        
        return x
    
class FCN_Skip(nn.Module):
    def __init__(self, num_classes):
        super(FCN_Skip, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Decoder
        self.upsample1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2)
        
        # Skip connections
        self.conv2d_1x1_conv1 = nn.Conv2d(64, 64, kernel_size=1)  # 1x1 convolutions for channel matching
        self.conv2d_1x1_conv2 = nn.Conv2d(128, 64, kernel_size=1)
        
    def forward(self, x):
        # Encode
        c1 = F.relu(self.conv1(x))
        p1 = F.max_pool2d(c1, 2)
        c2 = F.relu(self.conv2(p1))
        p2 = F.max_pool2d(c2, 2)
        c3 = F.relu(self.conv3(p2))
        p3 = F.max_pool2d(c3, 2)
        
        # Decode
        up1 = self.upsample1(p3)
        up1 = up1 + self.conv2d_1x1_conv2(c2)  # Add skip connection
        up2 = self.upsample2(up1)
        up2 = up2 + self.conv2d_1x1_conv1(c1)  # Add skip connection
        up3 = self.upsample3(up2)
        
        return up3