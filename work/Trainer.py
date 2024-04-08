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
from wandb import Artifact

from CityScapesDataset import CityScapesDataset

device = 'cuda'

class Trainer:
    def __init__(self, train_dataloader, val_dataloader, config):
        self.n_classes = CityScapesDataset.n_classes

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.model = self.get_model(self.n_classes)
        wandb.watch(self.model, log = 'all', log_freq = 10)

        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.CrossEntropyLoss(reduction = 'sum', ignore_index = config['ignore_index'])

    @staticmethod
    def get_model(num_classes):
        model = fcn_resnet50(weights = None, progress = True, num_classes = num_classes)
        model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size = (1, 1))
        return model.to(device)

    def log_model(self, **metadata):
        torch.save(self.model.state_dict(), 'model.pth')
        artifact = Artifact('model_weights', type = 'model', metadata = metadata)
        artifact.add_file('model.pth')
        wandb.log_artifact(artifact)

    def run_step(self, images, masks, training):
        if not training:
            with torch.no_grad():
                outputs = self.model(images)['out']
                return self.criterion(outputs, masks)

        self.optimizer.zero_grad()
        outputs = self.model(images)['out']
        loss = self.criterion(outputs, masks)
        loss.backward()
        self.optimizer.step()

        return loss

    def run_epoch(self, dataloader, training):
        total_loss = tensor(0.).to(device)
        for e, (images, masks) in enumerate(dataloader, start = 1):
            images, masks = images.to(device), masks.to(device)
            loss = self.run_step(images, masks, training)
            logging.info(f'Running {e}/{len(dataloader)}: partial loss = {loss / len(images):g}')

            total_loss += loss

        return total_loss / len(dataloader.dataset)

    @staticmethod
    def apply_palette(mask, num_classes):
        palette = (255 * array(seaborn.color_palette('husl', num_classes))).astype(numpy.uint8)
        color_mask = numpy.zeros((*mask.shape, 3), dtype = numpy.uint8)

        for label in range(num_classes):
            color_mask[mask == label] = palette[label]

        return color_mask

    def get_sample(self):
        image, mask_true = [x[0] for x in next(iter(self.val_dataloader))]
        with torch.no_grad():
            mask_pred = self.model(image.unsqueeze(0).to(device))['out'].squeeze(0).argmax(dim = 0).cpu()

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

            sample = self.get_sample()
            losses = {
                'Train loss': train_loss,
                'Val loss': val_loss,
            }

            logging.info(f'Epoch {epoch}/{epochs}: train loss = {train_loss:g}, val loss = {val_loss:g}')
            wandb.log({'Epoch': epoch} | sample | losses)

            if val_loss < best_loss:
                logging.info('Best loss found!')
                self.log_model(train_loss = train_loss, val_loss = val_loss)
                best_loss = val_loss

