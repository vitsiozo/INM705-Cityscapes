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
from JaccardLoss import *

from Model import Model

def parse_args(is_hyperion: bool) -> dict[str, Any]:
    losses = dict(
        cross_entropy = CrossEntropyLoss(reduction = 'sum'),
        cross_entropy_ignore = CrossEntropyLoss(reduction = 'sum', ignore_index = 0),
        dice_loss = DiceLoss(),
        iou_loss = IoULoss(),
    )

    parser = argparse.ArgumentParser(description = 'Cityscapes!')

    model_options = parser.add_argument_group('Model options', 'Options that affect the model and its training')
    model_options.add_argument('--model', default = 'Baseline', choices = Model.keys(), dest = 'model_name', help = 'Which model to use. "Baseline", "EnhancedUNet", and "EnhancedSwin" expand to the corresponding models.')
    model_options.add_argument('--pretrained-model-weights', help = 'Wandb ID to pre-load model weights')

    model_options.add_argument('--lr', type = float, default = 1e-3, help = 'Initial learning rate.')
    model_options.add_argument('--weight-decay', type = float, default = 0., help = 'L2 weight decay.')
    model_options.add_argument('--gamma', type = float, default = 1, help = 'Learning rate decay every 10 epochs.')
    model_options.add_argument('--optimiser', type = str, default = 'AdamW', choices = ['Adam', 'AdamW', 'Adamax'], dest = 'optimiser_name', help = 'Optimiser.')
    model_options.add_argument('--dropout', type = float, help = 'How much dropout to use (if applicable).')
    model_options.add_argument('--loss-fn', type = str, default = 'cross_entropy', choices = losses.keys(), dest = 'loss_fn_name', help = 'Loss function.')

    input_options = parser.add_argument_group('Input image options', 'Options that affect how the training images are read')
    input_options.add_argument('--granularity', type = str, default = 'coarse', choices = ['fine', 'coarse'], help = 'Granularity of the dataset. Only coarse (the default) was used in the final report.')
    input_options.add_argument('--image-size', type = int, help = 'The square image size to use.')
    input_options.add_argument('--no-resize', action = 'store_true', help = 'Do not resize the image.')

    training_options = parser.add_argument_group('Training options', 'Options that affect how the model is trained')
    training_options.add_argument('--epochs', type = int, default = 100 if is_hyperion else 2, help = 'Number of epochs.')
    training_options.add_argument('--batch-size', type = int, nargs = '?', help = 'Batch size.')

    training_options.add_argument('--label', type = str, help = 'Label added to wandb artifacts.')
    training_options.add_argument('--device', type = str, choices = ['cuda', 'mps', 'cpu'], help = 'Which device to use.')

    args = parser.parse_args()

    args.model = Model.instanciate(
        args.model_name,
        in_channels = 3,
        out_channels = CityScapesDataset.n_classes,
        dropout = args.dropout,
    )

    args.loss_fn = losses[args.loss_fn_name]
    if args.loss_fn_name in ['cross_entropy', 'cross_entropy_ignore']:
        args.accumulate_fn = lambda loss, loader: loss / len(loader.dataset)
    else:
        args.accumulate_fn = lambda loss, loader: loss / len(loader)

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

    if args.no_resize:
        args.image_size = None

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
        batches_per_epoch = None,
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
    if config['pretrained_model_weights'] is not None:
        logging.info(f'Using pretrained weights from {config["pretrained_model_weights"]}')
        artifact = wandb.use_artifact(config['pretrained_model_weights'], type = 'model')
        artifact_dir = artifact.download()
        weights_file = os.path.join(artifact_dir, 'model.pth')
        weights = torch.load(weights_file, map_location = config['device'])
        model.load_state_dict(weights)

    trainer = Trainer(model, train_dataloader, val_dataloader, config, eval_losses = {'IoU Score': IoUScore(ignore_index = 0), 'iIoU Score': InstanceIoUScore(ignore_index = 0)})
    trainer.train(epochs = config['epochs'])

if __name__ == '__main__':
    main()
