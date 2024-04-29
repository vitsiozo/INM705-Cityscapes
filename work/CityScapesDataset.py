import os
import torch

from numpy import array
from PIL import Image
from torchvision.transforms import v2 as transforms
#from torchvision import transforms
from torch.utils.data import Dataset

class CityScapesDataset(Dataset):
    n_classes = 34

    @staticmethod
    def get_transforms(size, train_transforms=False):
        # Resize transformations separately for the image and the mask
        image_resize = transforms.Resize((size, size))
        mask_resize = transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST)

        def apply_transforms(image, mask):
            # Apply the respective resize transformations
            image = image_resize(image)
            mask = mask_resize(mask)

            # Apply random transforms if specified
            if train_transforms:
                #print(f'Hey, I am applying random transforms')
                # Random horizontal flip
                if torch.rand(1) > 0.5:
                    image = transforms.functional.hflip(image)
                    mask = transforms.functional.hflip(mask)

                # Random rotation
                angle = torch.randint(-10, 10, (1,)).item()
                image = transforms.functional.rotate(image, angle)
                mask = transforms.functional.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)

            # Convert the image to a tensor
            image = transforms.functional.to_tensor(image)
            mask = torch.from_numpy(array(mask)).long()

            return image, mask

        return apply_transforms

    def __init__(self, image_dir, mask_dir, n = None, size = 512, granularity = 'coarse', train_transforms = False):
        self.transform = self.get_transforms(size, train_transforms)

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

        image, mask = self.transform(image, mask)

        return image, mask

