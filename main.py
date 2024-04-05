# main.py
from logger import Logger
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from fcn_train_citysc import SimpleFCN
from fcn_train_citysc import train
from citysc_dataset import CityscapesCoarseDataset



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 10
    
    # Initialize logger
    experiment_name = "Cityscapes_Semantic_Segmentation"
    my_logger = Logger(experiment_name, project='cityscapes_project')
    my_logger.start()
    
    # Load dataset, model, etc.
    train_dataset = CityscapesCoarseDataset(root_dir='/Users/vitsiozo/Code/Image/dataset', split='train')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    model = SimpleFCN(num_classes=34).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Optionally watch model for gradients and model parameters
    my_logger.watch(model)
    
    # Training loop...
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch, num_epochs, my_logger)
        
            
if __name__ == '__main__':
    main()
