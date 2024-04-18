import os
import torch

from numpy import array
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class CityScapesDataset(Dataset):
    n_classes = 34

    @staticmethod
    def get_transforms(size):
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        mask_transform = transforms.Compose([
            transforms.Resize((size, size), interpolation = transforms.InterpolationMode.NEAREST),
            lambda x: torch.from_numpy(array(x)).long(),
        ])

        return transform, mask_transform

    def __init__(self, image_dir, mask_dir, n = None, size = 512, granularity = 'coarse'):
        self.transform, self.mask_transform = self.get_transforms(size)

        self.images = []
        self.masks = []
        for city in os.listdir(image_dir):
            city_image_dir = os.path.join(image_dir, city)
            city_mask_dir = os.path.join(mask_dir, city)
            for img_file in os.listdir(city_image_dir):
                if not 'leftImg8bit' in img_file:
                    continue

                mask_file = img_file.replace('leftImg8bit.png', f'gt{granularity.capitalize()}_labelIds.png')

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
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

