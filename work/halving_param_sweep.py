import argparse
import logging
import torch
import socket
import random
import os
import math
import wandb
from itertools import product
from typing import Any

from torch.nn import CrossEntropyLoss
from torch.optim import Adam, AdamW, Adamax
from torch.utils.data import DataLoader

from CityScapesDataset import CityScapesDataset
from Trainer import Trainer
from DiceLoss import DiceLoss
from JaccardLoss import *

from Model import Model
def parse_args(is_hyperion: bool) -> dict[str, Any]:
    losses = dict(
        cross_entropy = CrossEntropyLoss(reduction = 'sum', ignore_index = 0),
        dice_loss = DiceLoss(),
        iou_loss = IoULoss(),
    )

    parser = argparse.ArgumentParser(
            description = 'Halving parameter sweep for CityScapes.',
            epilog = 'At each step, sweeps the parameters and keeps the best half, it then doubles `n`.',
    )

    parser.add_argument('--model', default = 'Baseline', choices = Model.keys(), dest = 'model_name', help = 'Which model to use.')
    parser.add_argument('--sweep-dropout', action = 'store_true', help = 'Whether to parameter sweep the dropout value.')

    parser.add_argument('--batch-size', type = int, nargs = '?', help = 'Batch size')
    parser.add_argument('--granularity', type = str, default = 'coarse', choices = ['fine', 'coarse'], help = 'Granularity of the dataset.')
    parser.add_argument('--optimiser', type = str, default = 'AdamW', choices = ['Adam', 'AdamW', 'Adamax'], dest = 'optimiser_name', help = 'Optimiser.')
    parser.add_argument('--image-size', type = int, default = 768, help = 'The square image size to use')
    parser.add_argument('--device', type = str, default = 'cuda', choices = ['cuda', 'mps', 'cpu'], help = 'Which device to use')

    args = parser.parse_args()

    args.loss_fn = CrossEntropyLoss(reduction = 'sum', ignore_index = 0)
    args.accumulate_fn = lambda loss, loader: loss / len(loader.dataset)

    if args.optimiser_name == 'Adam':
        args.optimiser = Adam
    elif args.optimiser_name == 'AdamW':
        args.optimiser = AdamW
    elif args.optimiser_name == 'Adamax':
        args.optimiser = Adamax
    else:
        raise ValueError(f'Unknown optimiser {args.optimiser_name}')

    if args.batch_size is None:
        del args.batch_size

    return vars(args)

# Returns a list of parameters to sweep.
def parameter_sweep(sweep_dropout: bool) -> list[dict[str, float]]:
    params = dict(
        lr = [1e-2, 1e-3, 1e-4],
        gamma = [1., math.sqrt(1/10), 1/10],
        weight_decay = [0., 1e-4],
    )

    if sweep_dropout:
        params['dropout'] = [0., 1/20, 1/10],

    return [dict(zip(params.keys(), x)) for x in product(*params.values())]

def run_sweep(train_dataloader, val_dataloader, config):
    params = parameter_sweep(config['sweep_dropout'])
    while len(params) > 1:
        config['batches_per_epoch'] = 2994 // config['batch_size'] // len(params)

        logging.info(f"New sweep halving! {len(params)} left; n = {config['batches_per_epoch']}")
        if len(params) <= 16:
            print('\n'.join(params))

        results = []
        for param_set in params:
            logging.info(f'Attempting {param_set}')
            model = Model.instanciate(
                config['model_name'],
                in_channels = 3,
                out_channels = CityScapesDataset.n_classes,
                dropout = param_set.get('dropout', None),
            ).to(config['device'])

            run = wandb.init(project = 'work', config = config)
            trainer = Trainer(model, train_dataloader, val_dataloader, config | param_set, wandb_run = run)
            loss = trainer.train(config['epochs'])
            run.finish()

            results.append((loss, param_set))

        results.sort(key = lambda x: x[0])
        params = results[len(results) // 2:]

    logging.info('We have a winner!')
    return params[0]

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
        batch_size = 16 if is_hyperion else 1,
        epochs = 20 if is_hyperion else 2,
        label = 'sweep',
        no_log_models = True,
    )
    config |= parse_args(is_hyperion)

    train_dataset = CityScapesDataset('data/leftImg8bit/train', os.path.join('data', config['granularity'], 'train'), size = config['image_size'], granularity = config['granularity'])
    val_dataset = CityScapesDataset('data/leftImg8bit/val', os.path.join('data', config['granularity'], 'val'), size = config['image_size'], granularity = config['granularity'])

    train_dataloader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = config['batch_size'], shuffle = True)

    best_loss, best_model = run_sweep(train_dataloader, val_dataloader, config)
    print(f'Best loss: {best_loss}')
    print(f'Best model parameters: {best_model}')

if __name__ == '__main__':
    main()
