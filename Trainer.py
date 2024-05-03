import logging
import numpy
import torch
import seaborn
import wandb

from numpy import array
from torch import nn, tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image
from wandb import Artifact # type: ignore

from CityScapesDataset import CityScapesDataset
from DiceLoss import DiceLoss

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, config, wandb_run = None, eval_losses = {}):
        self.config = config
        self.device = self.config['device']

        self.n_classes = CityScapesDataset.n_classes

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.val_example = [x[0] for x in next(iter(val_dataloader))]

        self.model = model
        wandb.watch(self.model, log = 'all', log_freq = 10)

        self.artifact_to_delete = None

        self.optimizer = config['optimiser'](self.model.parameters(), lr = config['lr'], weight_decay = config['weight_decay'])
        self.scheduler = StepLR(self.optimizer, step_size = 10, gamma = config['gamma'])

        self.criterion = config['loss_fn']
        self.accumulate_fn = config['accumulate_fn']

        self.eval_losses = eval_losses

        self.wandb = wandb_run or wandb

    def log_model(self, **metadata):
        if self.config.get('no_log_models', False):
            return

        torch.save(self.model.state_dict(), 'model.pth')

        api = wandb.Api()
        wandb_label = f'{wandb.run.id}_best'
        name = 'model_weights'

        artifact = Artifact(name, type = 'model', metadata = metadata)
        artifact.add_file('model.pth')

        labels = [wandb_label]
        if self.config['label'] is not None:
            labels.append(self.config['label'])
        self.wandb.log_artifact(artifact, aliases = labels)

        if self.artifact_to_delete is not None:
            logging.debug(f'Deleting old artifact with ID {self.artifact_to_delete.id}')
            self.artifact_to_delete.delete()
            self.artifact_to_delete = None

        try:
            old_artifact = api.artifact(f'{wandb.run.entity}/{wandb.run.project}/{name}:{wandb_label}')
            old_artifact.aliases = []
            old_artifact.save()

            self.artifact_to_delete = old_artifact
        except wandb.errors.CommError as e:
            logging.debug(f'First artifact, not deleting ({e})')

    def train_step(self, images, masks):
        self.optimizer.zero_grad()

        outputs = self.model(images)
        loss = self.criterion(outputs, masks)
        loss.backward()

        self.optimizer.step()
        return loss

    @torch.no_grad
    def eval_step(self, images, masks):
        outputs = self.model(images)
        extra_losses = {k: f(outputs, masks) for k, f in self.eval_losses.items()}
        return self.criterion(outputs, masks), extra_losses

    def run_epoch(self, dataloader, training):
        # Sets training or eval mode.
        self.model.train(training)

        total_loss = tensor(0.).to(self.device)
        total_extra_losses = {k: tensor(0.).to(self.device) for k in self.eval_losses.keys()}
        batches = self.config['batches_per_epoch'] or len(dataloader)
        for e, (images, masks) in enumerate(dataloader, start = 1):
            if e > batches:
                break

            images, masks = images.to(self.device), masks.to(self.device)

            if training:
                loss = self.train_step(images, masks)
            else:
                loss, extra_losses = self.eval_step(images, masks)
                for k, v in extra_losses.items():
                    total_extra_losses[k] += v.detach()

            lr = self.optimizer.param_groups[0]['lr']
            logging.debug(f'Running {e}/{batches}: lr = {lr:g}; partial loss = {loss / len(images):g}')

            total_loss += loss

        if training:
            self.scheduler.step()
            return self.accumulate_fn(total_loss, dataloader)

        eval_losses = {k: self.accumulate_fn(v, dataloader) for k, v in total_extra_losses.items()}
        return self.accumulate_fn(total_loss, dataloader), eval_losses

    @staticmethod
    def apply_palette(mask, num_classes):
        palette = (255 * array(seaborn.color_palette('husl', num_classes))).astype(numpy.uint8)
        color_mask = numpy.zeros((*mask.shape, 3), dtype = numpy.uint8)

        for label in range(num_classes):
            color_mask[mask == label] = palette[label]

        return color_mask

    def get_sample(self):
        if self.config.get('no_log_models', False):
            return {}

        image, mask_true = self.val_example
        with torch.no_grad():
            mask_pred = self.model(image.unsqueeze(0).to(self.device)).squeeze(0).argmax(dim = 0).cpu()

        return {
            'Image': wandb.Image(to_pil_image(image)),
            'True Mask': wandb.Image(to_pil_image(self.apply_palette(mask_true, self.n_classes))),
            'Pred Mask': wandb.Image(to_pil_image(self.apply_palette(mask_pred, self.n_classes))),
        }

    def train(self, epochs):
        best_loss = float('inf')
        for epoch in range(1, epochs + 1):
            train_loss = self.run_epoch(self.train_dataloader, training = True)
            val_loss, extra = self.run_epoch(self.val_dataloader, training = False)

            if val_loss < best_loss:
                logging.debug('Best loss found!')
                self.log_model(train_loss = train_loss, val_loss = val_loss, epoch = epoch)
                best_loss = val_loss

            sample = self.get_sample()
            losses = {
                'Train loss': train_loss,
                'Val loss': val_loss,
                'Best loss': best_loss,
            } | extra

            logging.debug('\n'.join([f'Epoch {epoch}/{epochs}:'] + [f'{k}:\t{v:g}' for k, v in losses.items()]))
            self.wandb.log({'Epoch': epoch} | sample | losses)

        logging.debug(f'Model final loss is {best_loss}')
        return best_loss
