import logging
import os
import pickle
import torch
import wandb

from PIL import Image
from torch import nn, tensor
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from wandb import Artifact

device = 'cuda'

class CityScapesDataset(Dataset):
    def __init__(self, image_dir, mask_dir, n = None):
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            lambda x: x.squeeze(0),
        ])

        self.images = []
        self.masks = []
        for city in os.listdir(image_dir):
            city_image_dir = os.path.join(image_dir, city)
            city_mask_dir = os.path.join(mask_dir, city)
            for img_file in os.listdir(city_image_dir):
                if not 'leftImg8bit' in img_file:
                    continue

                mask_file = img_file.replace('leftImg8bit.png', 'gtCoarse_labelIds.png')

                self.images.append(os.path.join(city_image_dir, img_file))
                self.masks.append(os.path.join(city_mask_dir, mask_file))

                if n is not None and len(self.images) >= n:
                    return

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

class Trainer:
    def __init__(self, train_dataloader, val_dataloader):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.model = self.get_model(30)
        wandb.watch(self.model, log = 'all', log_freq = 10)

        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.CrossEntropyLoss(reduction = 'sum')

    @staticmethod
    def get_model(num_classes):
        model = fcn_resnet50(weights = None, progress = True)
        model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size = (1, 1))
        return model.to(device)

    def log_model(self):
        torch.save(self.model.state_dict(), 'model.pth')
        artifact = Artifact('model_weights', type = 'model')
        artifact.add_file('model.pth')
        wandb.log_artifact(artifact)

    def train(self, epochs):
        best_loss = float('inf')
        for epoch in range(1, epochs + 1):
            train_loss = self.run_epoch(self.train_dataloader, training = True)
            val_loss = self.run_epoch(self.val_dataloader, training = False)

            logging.info(f'Epoch {epoch}/{epochs}: train loss = {train_loss:g}, val loss = {val_loss:g}')
            wandb.log({'Epoch': epoch, 'Train loss': train_loss, 'Val loss': val_loss})

            if val_loss < best_loss:
                logging.info('Best loss found!')
                self.log_model()
                best_loss = val_loss

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
            images, masks = images.to(device), masks.to(device).long()
            total_loss += self.run_step(images, masks, training)

            logging.info(f'Running {e}/{len(dataloader)}: loss = {total_loss:g}')

        return total_loss / len(dataloader)

def main():
    config = dict(
        n = None,
        batch_size = 154,
        epochs = 100,
    )

    wandb.init(
        project = 'work',
        config = config,
    )
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    train_dataset = CityScapesDataset('data/leftImg8bit/train', 'data/coarse/train', n = config['n'])
    val_dataset = CityScapesDataset('data/leftImg8bit/val', 'data/coarse/val', n = min(config['n'], config['batch_size']))

    train_dataloader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = config['batch_size'], shuffle = True)

    trainer = Trainer(train_dataloader, val_dataloader)
    trainer.train(epochs = config['epochs'])

if __name__ == '__main__':
    main()
