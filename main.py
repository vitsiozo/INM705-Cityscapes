# main.py
from logger import Logger
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from fcn_train_citysc import SimpleFCN
from fcn_train_citysc import train
from citysc_dataset import create_dataloaders

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 10
    batch_size = 4
    root_dir = '/Users/vitsiozo/Code/Image/dataset' # Root directory for the dataset

    # Initialize logger
    experiment_name = "Cityscapes_Semantic_Segmentation"
    my_logger = Logger(experiment_name, project='cityscapes_project')
    my_logger.start()
    
    # Create dataloader for training and validation
    train_loader, val_loader = create_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
    )
    
    model = SimpleFCN(num_classes=34).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Optionally watch model for gradients and model parameters
    my_logger.watch(model)
    
    # Training loop...
    train(model, device, train_loader, val_loader, optimizer, num_epochs, my_logger)
        
            
if __name__ == '__main__':
    main()
