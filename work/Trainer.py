import logging
import numpy
import torch
import seaborn
import wandb

from numpy import array
from torch import nn, tensor
from torch.optim import Adam
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image
from wandb import Artifact # type: ignore

from CityScapesDataset import CityScapesDataset
from DiceLoss import DiceLoss

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, config):
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
        self.criterion = config['loss_fn']
        self.accumulate_fn = config['accumulate_fn']

    def log_model(self, **metadata):
        torch.save(self.model.state_dict(), 'model.pth')

        api = wandb.Api()
        wandb_label = f'{wandb.run.id}_best'
        name = 'model_weights'

        artifact = Artifact(name, type = 'model', metadata = metadata)
        artifact.add_file('model.pth')

        labels = [wandb_label]
        if self.config['label'] is not None:
            labels.append(self.config['label'])
        wandb.log_artifact(artifact, aliases = labels)

        if self.artifact_to_delete is not None:
            logging.info(f'Deleting old artifact with ID {self.artifact_to_delete.id}')
            self.artifact_to_delete.delete()
            self.artifact_to_delete = None

        try:
            old_artifact = api.artifact(f'{wandb.run.entity}/{wandb.run.project}/{name}:{wandb_label}')
            old_artifact.aliases = []
            old_artifact.save()

            self.artifact_to_delete = old_artifact
        except wandb.errors.CommError as e:
            logging.info(f'First artifact, not deleting ({e})')

    def run_step(self, images, masks, training):
        if not training:
            with torch.no_grad():
                outputs = self.model(images)
                return self.criterion(outputs, masks)

        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, masks)
        loss.backward()
        self.optimizer.step()

        return loss

    def run_epoch(self, dataloader, training):
        total_loss = tensor(0.).to(self.device)
        for e, (images, masks) in enumerate(dataloader, start = 1):
            images, masks = images.to(self.device), masks.to(self.device)
            loss = self.run_step(images, masks, training)
            logging.info(f'Running {e}/{len(dataloader)}: partial loss = {loss / len(images):g}')

            total_loss += loss

        return self.accumulate_fn(total_loss, dataloader)

    @staticmethod
    def apply_palette(mask, num_classes):
        palette = (255 * array(seaborn.color_palette('husl', num_classes))).astype(numpy.uint8)
        color_mask = numpy.zeros((*mask.shape, 3), dtype = numpy.uint8)

        for label in range(num_classes):
            color_mask[mask == label] = palette[label]

        return color_mask

    def get_sample(self):
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
            val_loss = self.run_epoch(self.val_dataloader, training = False)

            if val_loss < best_loss:
                logging.info('Best loss found!')
                self.log_model(train_loss = train_loss, val_loss = val_loss, epoch = epoch)
                best_loss = val_loss

            sample = self.get_sample()
            losses = {
                'Train loss': train_loss,
                'Val loss': val_loss,
                'Best loss': best_loss,
            }

            logging.info(f'Epoch {epoch}/{epochs}: train loss = {train_loss:g}, val loss = {val_loss:g}')
            wandb.log({'Epoch': epoch} | sample | losses)

        logging.info(f'Model final loss is {best_loss}')
