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
from utils import apply_palette



def train(model, device, train_loader, val_loader, optimizer, num_epochs, my_logger):
    
    best_val_loss = float('inf')    
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, num_epochs + 1):
        model.train()  # Set model to training mode at the start of each epoch
        total_loss = 0
        
        for batch_idx, (data, target, image_names) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            
            # Perform forward pass
            output = model(data)
            
            # Calculate loss
            target = target.long().squeeze(1)  # Ensure target is in the correct shape and type
            loss = criterion(output, target)  # Calculate loss here
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        my_logger.log({"epoch": epoch, "Train Loss": avg_loss})
        print(f"Epoch: {epoch}, Avg Loss: {avg_loss}")

        # Perform validation and calculate validation loss
        if val_loader is not None:
            val_loss = perform_validation(model, val_loader, device)
           
            my_logger.log({"epoch": epoch, "Val Loss": val_loss})
            print(f"Epoch: {epoch}, Val Loss: {val_loss}")

            if val_loss < best_val_loss:
                my_logger.log_model_artifact(model, f"Best_Model_epoch_{epoch}", {
                   "val_loss": val_loss,
                   "epoch": epoch
                })
                best_val_loss = val_loss
                
        # Log sample images at the end of each epoch
        # Use a small batch from either train_loader or val_loader
        model.eval()
        with torch.no_grad():
            for data, target, image_names in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                break  # Use only the first batch for logging
                
        sample_logs = get_sample_images(data, target, output, image_names[0])
        my_logger.log(sample_logs)
        model.train()  # Set model back to training mode for the next epoch
      
def perform_validation(model, val_loader, device):
    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0.0
    criterion = nn.CrossEntropyLoss()  # Define the loss function

    with torch.no_grad():  # Disable gradient computation for validation
        for data, target, _ in val_loader:  # Ignore image names during validation
            data, target = data.to(device), target.to(device)
            target = target.long().squeeze(1)  # Ensure target is in the correct shape and type
            output = model(data)
            loss = criterion(output, target)
            total_val_loss += loss.item()  # Accumulate the total validation loss

    avg_val_loss = total_val_loss / len(val_loader)  # Calculate the average validation loss
  
    model.train()  # Set the model back to training mode
    return avg_val_loss

def get_sample_images(data, target, output, image_name):
    # Convert data, target, output to images, and log them
    pred = torch.argmax(output, dim=1)[0].cpu().numpy()
    true_mask = target[0].cpu().numpy()
    pred_colored = apply_palette(pred)  # Your function to colorize masks
    true_mask_colored = apply_palette(true_mask)
    input_image = to_pil_image(data[0])
    return {
        f"Sample - {image_name}": [wandb.Image(input_image, caption="Input"),
                                   wandb.Image(pred_colored, caption="Prediction"),
                                   wandb.Image(true_mask_colored, caption="True Mask")]
    }