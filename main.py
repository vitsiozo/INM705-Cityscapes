# main.py
import torch
import torch.optim as optim
from models import SimpleFCN, FCN_Skip
from fcn_train_citysc import train
from citysc_dataset import create_dataloaders
from logger import Logger
from utils import parse_arguments, read_settings

def main(settings):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device used: {device}')

    # Access and use the settings as needed
    model_type = settings['model']['type']
    num_classes = settings['training']['num_classes']
    training_settings = settings['training']
    logger_settings = settings['logger']

    num_epochs = training_settings['num_epochs']
    batch_size = training_settings['batch_size']
    num_classes = training_settings['num_classes']
    root_dir = training_settings['root_dir']
    learning_rate = training_settings['learning_rate']

    # Initialize logger
    experiment_name = logger_settings['experiment_name']
    project = logger_settings['project']
    entity = logger_settings['entity']
    models_dir = logger_settings['models_dir']
    my_logger = Logger(experiment_name, project, entity, models_dir)
    my_logger.start(settings)
    
    # Create dataloader for training and validation
    train_loader, val_loader = create_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
    )
    
    model = get_model(model_type, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Log gradients and model parameters
    my_logger.watch(model)
    
    # Training loop
    train(model, device, train_loader, val_loader, optimizer, num_epochs, my_logger)        

def get_model(model_name, num_classes):
    models = {
        'SimpleFCN': SimpleFCN,
        'FCN_Skip': FCN_Skip
        # Additional models can be added here.
    }
    try:
        return models[model_name](num_classes)
    except KeyError:
        raise ValueError(f"Unsupported model type: {model_name}")

if __name__ == '__main__':
    args = parse_arguments()

    # Read settings from the YAML file
    settings = read_settings(args.config)

    main(settings)