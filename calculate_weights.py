import torch
import torch.nn as nn
import torch.nn.functional as F

from CityScapesDataset import CityScapesDataset
from torch.utils.data import DataLoader

device = 'cpu'

@torch.no_grad()
def main():
    train_dataset = CityScapesDataset('data/leftImg8bit/train', 'data/coarse/train', n = None, size = 768, granularity = 'coarse', train_transforms = False)
    train_dataloader = DataLoader(train_dataset, batch_size = 256, shuffle = True)

    class_sums = torch.zeros(34).to(device)
    for e, (_, mask) in enumerate(train_dataloader):
        print(f'{e} / {len(train_dataloader)}')
        mask = mask.to(device)
        sums = mask.flatten().bincount()
        class_sums += sums

    print(class_sums.cpu().tolist())

if __name__ == '__main__':
    main()
