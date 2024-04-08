import argparse
import logging
import random
import socket
import torch
import wandb

from torch.nn import CrossEntropyLoss
from torch.optim import Adam, AdamW, Adamax
from torch.utils.data import DataLoader

from CityScapesDataset import CityScapesDataset
from Trainer import Trainer
from DiceLoss import DiceLoss

def parse_args(is_hyperion):
    parser = argparse.ArgumentParser(description='Cityscapes!')

    parser.add_argument('--loss-fn', type=str, default = 'cross_entropy', choices=['cross_entropy', 'dice_loss'], help='Loss function.')
    parser.add_argument('--granularity', type=str, default = 'coarse', choices=['fine', 'coarse'], help='Granularity of the dataset.')
    parser.add_argument('--optimiser', type=str, default = 'Adam', choices=['Adam', 'AdamW', 'Adamax'], help='Optimiser.')
    parser.add_argument('--lr', type=float, default = 1e-3, help='Learning rate')
    parser.add_argument('--num-epochs', type = int, default = 100 if is_hyperion else 2, help = 'Number of epochs')

    args = parser.parse_args()

    match args.loss_fn:
        case 'cross_entropy':
            args.loss_fn = CrossEntropyLoss(reduction = 'sum')
            args.accumulate_fn = lambda loss, loader: loss / len(loader.dataset)
        case 'dice_loss':
            args.loss_fn = DiceLoss()
            args.accumulate_fn = lambda loss, loader: loss / len(loader)

    match args.optimiser:
        case 'Adam':
            args.optimiser = Adam
        case 'AdamW':
            args.optimiser = AdamW
        case 'Adamax':
            args.optimiser = Adamax

    return vars(args)

def main():
    is_hyperion = 'hyperion' in socket.gethostname()
    random_seed = random.randint(0, 1000)
    torch.manual_seed(random_seed)

    config = dict(
        random_seed = random_seed,
        n = None if is_hyperion else 10,
        batch_size = 64 if is_hyperion else 1,
        epochs = 300 if is_hyperion else 2,
        ignore_index = 0,
        granularity = 'fine',
        image_size = 512,
        loss_fn = 'dice_loss',
    )
    config |= parse_args(is_hyperion)

    wandb.init(
        project = 'work',
        config = config,
    )
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    train_dataset = CityScapesDataset('data/leftImg8bit/train', 'data/fine/train', n = config['n'], size = config['image_size'])
    val_dataset = CityScapesDataset('data/leftImg8bit/val', 'data/fine/val', n = config['n'], size = config['image_size'])

    train_dataloader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = config['batch_size'], shuffle = True)

    trainer = Trainer(train_dataloader, val_dataloader, config)
    trainer.train(epochs = config['epochs'])

if __name__ == '__main__':
    main()
