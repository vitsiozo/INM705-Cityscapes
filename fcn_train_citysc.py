# fcn_train_citysc.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
import numpy as np
import seaborn as sns
from PIL import Image
import wandb
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
            
        if batch_idx % 100 == 0:
             # Convert the predictions and true mask to color images using the apply_palette function
            pred = torch.argmax(output, dim=1)[0].cpu().numpy() 
            true_mask = target[0].cpu().numpy()

              # Apply the palette to colorize the prediction and the true mask
            pred_colored = apply_palette(pred)
            true_mask_colored = apply_palette(true_mask)

              # Convert the numpy arrays to PIL images for logging
            pred_image = Image.fromarray(pred_colored)
            true_mask_image = Image.fromarray(true_mask_colored)

              # Get the corresponding input image as a PIL image
            input_image_pil = to_pil_image(data.cpu().data[0])

              # Log the images side by side by creating a list of images
            images_to_log = [
              wandb.Image(input_image_pil, caption="Input Image"),
              wandb.Image(pred_image, caption="Predicted Mask"),
              wandb.Image(true_mask_image, caption="True Mask")
            ]

             # Log the list of images as a single entry
            my_logger.log({"Input images, Predicted Mask, and True Mask": images_to_log})


def apply_palette(mask):
    # Define a palette
    palette = (255 * np.array(sns.color_palette('husl', 34))).astype(np.uint8)
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    for label in range(34):  
        color_mask[mask == label] = palette[label]
    
    return color_mask

