# fcn_train_citysc.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from citysc_dataset import CityscapesCoarseDataset
from logger import Logger

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
        x = F.max_pool2d(x, 2) # Pooling to reduce dimensionality
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2) # Pooling to reduce dimensionality
        
        # Decode
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x) # This layer also acts as the output layer
        
        return x

def train(model, device, train_loader, optimizer, epoch, num_epochs, my_logger):
    model.train()  # Set model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Inspect the unique class indices in the target tensor
        #unique_classes = torch.unique(target)
        #print(f'Unique class indices in batch {batch_idx}: {unique_classes.cpu().numpy()}')

        optimizer.zero_grad()
        output = model(data)
        target = target.long()
        target = target.squeeze(1)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:  # Print log every 10 batches
            my_logger.log({"epoch": epoch, "loss": loss.item(), "batch": batch_idx})
            print(f'Train Epoch: {epoch}/{num_epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


