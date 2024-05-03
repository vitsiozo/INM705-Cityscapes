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

from torch import tensor

def parse_args():
    parser = argparse.ArgumentParser(description = 'Downloads a Wandb Artifact and evaluates it against the validation or test sets.')

    parser.add_argument('--batch-size', type = int, default = 16, help = 'Batch size. Lower this if the program runs out of memory.')
    parser.add_argument('--device', choices = ['cuda', 'mps', 'cpu'], help = 'Which device to evaluate on (same as model by default).')
    parser.add_argument('--run-on', choices = ['train', 'val', 'test'], default = 'test', help = 'Which dataset to test this on (test by default)')
    parser.add_argument('model', help = 'Wandb tags or model name')

    return parser.parse_args()

class Evaluator:
    def __init__(self, model, device):
        # We are evaluating in regular cross entropy loss.
        # See report for more details.
        self.criterion = CrossEntropyLoss(reduction = 'sum')
        self.accumulate_fn = lambda loss, loader: loss / len(loader.dataset)
        self.eval_losses = {
            'IoU Score': IoUScore(ignore_index = 0),
            'iIoU Score': InstanceIoUScore(ignore_index = 0),
        }

        # If no device is found, use the device of the model.
        # I think I can can assume all parameters of the model are in the same device;
        # if this is not correct I'm quitting computer science and moving to a farm.
        self.device = device or next(model.parameters()).device

    @torch.no_grad
    def eval_step(self, images, masks):
        outputs = self.model(images)
        extra_losses = {k: f(outputs, masks) for k, f in self.eval_losses.items()}
        return self.criterion(outputs, masks), extra_losses

    @torch.no_grad
    def evaluate(self, dataset):
        total_loss = tensor(0.).to(self.device)
        total_extra_losses = {k: tensor(0.).to(self.device) for k in self.eval_losses.keys()}
        for e, (images, masks) in enumerate(dataset, start = 1):
            images, masks = images.to(self.device), masks.to(self.device)

            loss, extra_losses = self.eval_step(images, masks)
            total_loss += loss
            for k, v in extra_losses.items():
                total_extra_losses[k] += v

            logging.debug(f'Running {e}/{batches}: partial loss = {loss / len(images):g}')

        eval_losses = {k: self.accumulate_fn(v, dataloader) for k, v in total_extra_losses.items()}
        return self.accumulate_fn(total_loss, dataloader), eval_losses

# Gets a run that matches `tag` as either a tag (ie 'EnhancedSwin2') or
# a model name (ie 'wspszqbr') and its latest artifact weights.
# Project name hardcoded for simplicity. Sorry Greg!
def get_run(tag: str) -> tuple[dict[str, Any], str]:
    api = wandb.Api()
    runs = api.runs('mfixman-convolutional-team/work', {'$or': [
        {'tags': tag},
        {'name': tag},
    ]})

    if len(runs) == 0:
        raise NameError(f'No run found with either tag or name "{tag}"')
    
    if len(runs) > 1:
        logging.warning(f'{len(runs)} runs found with this tag or name! Choosing one of them.')

    run = runs[0]
    artifact = max(run.logged_artifacts(), key = lambda x: x.version)
    artifact_dir = artifact.download()

    return run.config, os.path.join(artifact_dir, 'model.pth')

def main():
    logging.basicConfig(
        level = logging.INFO,
        format = '[%(asctime)s] %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S',
    )

    args = parse_args()
    config, artifact = get_run(args.model)

    dataset = CityScapesDataset(
        os.path.join('data', 'leftImg8bit', args.run_on),
        os.path.join('data', config['granularity'], args.run_on),
        size = config.get('image_size'),
        granularity = config['granularity'],
    )

    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True)

    model = Model.instanciate(
        config['model_name'],
        in_channels = 3,
        out_channels = CityScapesDataset.n_classes,
        dropout = config.get('dropout'),
    )
    weights = torch.load(artifact)
    model.load_state_dict(weights)
    model.eval()

    evaluator = Evaluator(model, args.device)
    score, extra_scores = evaluator.evaluate(dataset)
    logging.info('Final score calculated')
    logging.info(f'Final CCE loss for {args.run_on}: {score}')
    for name, loss in extra_losses.items():
        logging.info('Final {name} for {args.run_on}: {loss}')

if __name__ == '__main__':
    main()
