import argparse
import logging
import random
import os
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

from Model import Model

def parse_args(is_hyperion: bool) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description = 'Cityscapes!')

    parser.add_argument('--granularity', type = str, default = 'coarse', choices = ['fine', 'coarse'], help = 'Granularity of the dataset.')
    parser.add_argument('--lr', type = float, default = 1e-3, help = 'Learning rate')
    parser.add_argument('--weight-decay', type = float, default = 0., help = 'L2 weight decay for AdamW classifier')
    parser.add_argument('--gamma', type = float, default = 1, help = 'Learning rate decay every 10 epochs.')
    parser.add_argument('--epochs', type = int, default = 100 if is_hyperion else 2, help = 'Number of epochs')
    parser.add_argument('--batch-size', type = int, nargs = '?', help = 'Batch size')
    parser.add_argument('--loss-fn', type = str, default = 'cross_entropy', choices = ['cross_entropy', 'dice_loss'], dest = 'loss_fn_name', help = 'Loss function.')
    parser.add_argument('--model', default = 'Baseline', choices = Model.keys(), dest = 'model_name', help = 'Which model to use.')
    parser.add_argument('--optimiser', type = str, default = 'AdamW', choices = ['Adam', 'AdamW', 'Adamax'], dest = 'optimiser_name', help = 'Optimiser.')
    parser.add_argument('--label', type = str, help = 'Label for wandb artifact.')
    parser.add_argument('--image-size', type = int, help = 'The square image size to use')
    parser.add_argument('--device', type = str, choices = ['cuda', 'mps', 'cpu'], help = 'Which device to use')
    parser.add_argument('--dropout', type = float, help = 'How much dropout to use (if applicable).')

    args = parser.parse_args()

    args.model = Model.instanciate(
        args.model_name,
        in_channels = 3,
        out_channels = CityScapesDataset.n_classes,
        dropout = args.dropout,
    )

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

    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            raise RuntimeError('No GPU Device (explicitly run --device=cpu for CPU)')

    if args.batch_size is None:
        del args.batch_size

    if args.image_size is None:
        del args.image_size

    return vars(args)

def main():
    is_hyperion = 'hyperion' in socket.gethostname()
    random_seed = random.randint(0, 1000)
    torch.manual_seed(random_seed)

    logging.basicConfig(
        level = logging.INFO,
        format = '[%(asctime)s] %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S',
    )

    config = dict(
        random_seed = random_seed,
        n = None if is_hyperion else 10,
        batch_size = 64 if is_hyperion else 1,
        epochs = 300 if is_hyperion else 2,
        ignore_index = 0,
        image_size = 512,
        loss_fn = 'dice_loss',
    )
    config |= parse_args(is_hyperion)

    wandb.init(
        project = 'work',
        config = config,
    )

    train_dataset = CityScapesDataset('data/leftImg8bit/train', os.path.join('data', config['granularity'], 'train'), n = config['n'], size = config['image_size'], granularity = config['granularity'])
    val_dataset = CityScapesDataset('data/leftImg8bit/val', os.path.join('data', config['granularity'], 'val'), n = config['n'], size = config['image_size'], granularity = config['granularity'])

    train_dataloader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = config['batch_size'], shuffle = True)

    model = config['model'].to(config['device'])

    trainer = Trainer(model, train_dataloader, val_dataloader, config)
    trainer.train(epochs = config['epochs'])

if __name__ == '__main__':
    main()
