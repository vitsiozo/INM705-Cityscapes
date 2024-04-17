import argparse
import logging
import random
import socket
import torch
import wandb

from typing import Any

from torch.nn import CrossEntropyLoss
from torch.optim import Adam, AdamW, Adamax
from torch.utils.data import DataLoader

from CityScapesDataset import CityScapesDataset
from Trainer import Trainer
from DiceLoss import DiceLoss
from BaselineModel import BaselineModel
from UNetModel import UNetModel

device = 'cuda'

def parse_args(is_hyperion: bool) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description = 'Cityscapes!')

    parser.add_argument('--loss-fn', type = str, default = 'cross_entropy', choices = ['cross_entropy', 'dice_loss'], dest = 'loss_fn_name', help = 'Loss function.')
    parser.add_argument('--granularity', type = str, default = 'coarse', choices = ['fine', 'coarse'], help = 'Granularity of the dataset.')
    parser.add_argument('--optimiser', type = str, default = 'AdamW', choices = ['Adam', 'AdamW', 'Adamax'], dest = 'optimiser_name', help = 'Optimiser.')
    parser.add_argument('--lr', type = float, default = 1e-3, help = 'Learning rate')
    parser.add_argument('--epochs', type = int, default = 100 if is_hyperion else 2, help = 'Number of epochs')
    parser.add_argument('--model', default = 'Baseline', choices = ['Baseline', 'UNet'], dest = 'model_name', help = 'Which model to use.')
    parser.add_argument('--batch-size', type = int, nargs = '?', help = 'Batch size')

    args = parser.parse_args()

    if args.loss_fn_name == 'cross_entropy':
        args.loss_fn = CrossEntropyLoss(reduction = 'sum')
        args.accumulate_fn = lambda loss, loader: loss / len(loader.dataset)
    elif args.loss_fn_name == 'dice_loss':
        args.loss_fn = DiceLoss()
        args.accumulate_fn = lambda loss, loader: loss / len(loader)
    else:
        raise ValueError(f'Unknown loss function {args.loss_fn}')

    if args.optimiser_name == 'Adam':
        args.optimiser = Adam
    elif args.optimiser_name == 'AdamW':
        args.optimiser = AdamW
    elif args.optimiser_name == 'Adamax':
        args.optimiser = Adamax
    else:
        raise ValueError(f'Unknown optimiser {args.optimiser_name}')

    if args.model_name == 'Baseline':
        args.model = BaselineModel(3, CityScapesDataset.n_classes)
    elif args.model_name == 'UNet':
        args.model = UNetModel(3, CityScapesDataset.n_classes)
    else:
        raise ValueError(f'Unknown model {args.model_name}')

    if args.batch_size is None:
        del args.batch_size

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
        level = logging.INFO,
        format = '[%(asctime)s] %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S',
    )

    train_dataset = CityScapesDataset('data/leftImg8bit/train', 'data/fine/train', n = config['n'], size = config['image_size'])
    val_dataset = CityScapesDataset('data/leftImg8bit/val', 'data/fine/val', n = config['n'], size = config['image_size'])

    train_dataloader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = config['batch_size'], shuffle = True)

    model = config['model'].to(device)

    trainer = Trainer(model, train_dataloader, val_dataloader, config)
    trainer.train(epochs = config['epochs'])

if __name__ == '__main__':
    main()
