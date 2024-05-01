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
    parser = argparse.ArgumentParser(description = 'Downloads a Wandb Artifact and evaluates it against the validation or test sets.')

    parser.add_argument('--granularity', type = str, default = 'coarse', choices = ['fine', 'coarse'], help = 'Granularity of the dataset.')
    parser.add_argument('--image-size', type = int, help = 'The square image size to use')
    parser.add_argument('--no-resize', action = 'store_true', help = 'Do not resize the image')

    parser.add_argument('--lr', type = float, default = 1e-3, help = 'Learning rate')
    parser.add_argument('--weight-decay', type = float, default = 0., help = 'L2 weight decay for AdamW classifier')
    parser.add_argument('--gamma', type = float, default = 1, help = 'Learning rate decay every 10 epochs.')
    parser.add_argument('--optimiser', type = str, default = 'AdamW', choices = ['Adam', 'AdamW', 'Adamax'], dest = 'optimiser_name', help = 'Optimiser.')
    parser.add_argument('--dropout', type = float, help = 'How much dropout to use (if applicable).')

    parser.add_argument('--model', default = 'Baseline', choices = Model.keys(), dest = 'model_name', help = 'Which model to use.')
    parser.add_argument('--epochs', type = int, default = 100 if is_hyperion else 2, help = 'Number of epochs')
    parser.add_argument('--batch-size', type = int, nargs = '?', help = 'Batch size')
    parser.add_argument('--loss-fn', type = str, default = 'cross_entropy', choices = losses.keys(), dest = 'loss_fn_name', help = 'Loss function.')

    parser.add_argument('--label', type = str, help = 'Label for wandb artifact.')
    parser.add_argument('--device', type = str, choices = ['cuda', 'mps', 'cpu'], help = 'Which device to use')

    args = parser.parse_args()
