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

device = 'cuda'

class CityScapesDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
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
                if 'leftImg8bit' in img_file:
                    self.images.append(os.path.join(city_image_dir, img_file))
                    mask_file = img_file.replace('leftImg8bit.png', 'gtCoarse_labelIds.png')
                    self.masks.append(os.path.join(city_mask_dir, mask_file))

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
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.model = self.get_model(30)

        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.CrossEntropyLoss(reduction = 'sum')

    @staticmethod
    def get_model(num_classes):
        model = fcn_resnet50(weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1, progress = True)
        model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size = (1, 1))
        return model.to(device)

    def train(self, epochs = 1):
        for epoch in range(epochs):
            loss = self.run_epoch()

    def run_epoch(self):
        total_loss = tensor(0.)
        for e, (images, masks) in enumerate(self.dataloader):
            images, masks = images.to(device), masks.to(device)

            self.optimizer.zero_grad()
            outputs = self.model(images)['out']

            loss = self.criterion(outputs, masks.long())
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            logging.info(f'Running {e}/{len(self.dataloader)}: loss = {total_loss:g}')

        return total_loss

def main():
    config = {
        'batch_size': 10,
    }

    wandb.init(
        project = 'work',
        config = config,
    )
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    dataset = CityScapesDataset(
        'data/leftImg8bit/train',
        'data/coarse/train',
    )
    dataloader = DataLoader(dataset, batch_size = config['batch_size'], shuffle = True)
    trainer = Trainer(dataloader)
    trainer.train()

if __name__ == '__main__':
    main()
