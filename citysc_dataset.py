# citysc_dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np


class CityscapesCoarseDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, target_transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and annotations.
            split (string): One of 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform 
        self.images_dir = os.path.join(root_dir, 'leftImg8bit', split)
        self.labels_dir = os.path.join(root_dir, 'gtCoarse', split)
        #self.images = os.listdir(self.images_dir)
        #self.labels = [img.replace('leftImg8bit', 'gtCoarse_labelIds') for img in self.images]

        self.images = []
        self.labels = []
        
        # Iterate over all city directories
        for city_dir in os.listdir(self.images_dir):
            city_images_path = os.path.join(self.images_dir, city_dir)
            city_labels_path = os.path.join(self.labels_dir, city_dir)
            
            for img_file in os.listdir(city_images_path):
                if img_file.endswith(".png"):  # Ensure we're only adding images
                    img_path = os.path.join(city_images_path, img_file)
                    self.images.append(img_path)
                    #print(f'Added to images: {img_path}')  # Print the image file path

                    # Construct the corresponding label path
                    label_file = img_file.replace('leftImg8bit', 'gtCoarse_labelIds')
                    label_path = os.path.join(city_labels_path, label_file)
                    self.labels.append(label_path)
                    #print(f'Added to labels: {label_path}')  # Print the label file path

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.labels[idx]

        # Extract the image name from the path for logging
        image_name = os.path.basename(img_path)
    
        image = Image.open(img_path).convert('RGB') # Load image
        label = Image.open(label_path) # Load label

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label) 

        #image = img_transform()(image)  # Apply image transforms
        #label = label_transform()(label)  # Apply label transforms WITHOUT normalization
    
        return image, label, image_name


    def __len__(self):
        return len(self.images)
    
def create_dataloaders(root_dir, batch_size):
    train_loader = DataLoader(
        CityscapesCoarseDataset(
            root_dir=root_dir, split='train',
            transform=img_transform(), target_transform=label_transform()
        ),
        batch_size=batch_size, shuffle=True
    )

    val_loader = DataLoader(
        CityscapesCoarseDataset(
            root_dir=root_dir, split='val',
            transform=img_transform(), target_transform=label_transform()
        ),
        batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader

def img_transform():
     return transforms.Compose([
        transforms.Resize((512, 512)),  # Resize to a common size
        transforms.ToTensor(),  # Convert to tensor
    ])

def label_transform():
        return transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),  # Resize labels without changing their class values
        lambda x: torch.from_numpy(np.array(x, dtype=np.int64)).long()
    ])



