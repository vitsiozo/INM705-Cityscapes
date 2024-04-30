import torch
import torch.nn as nn
import torch.nn.functional as F

from CityScapesDataset import CityScapesDataset
from torch.utils.data import DataLoader

device = 'cuda'

@torch.no_grad()
def main():
    train_dataset = CityScapesDataset('data/leftImg8bit/train', 'data/coarse/train', n = None, size = 768, granularity = 'coarse', train_transforms = False)
    train_dataloader = DataLoader(train_dataset, batch_size = 256, shuffle = True)

    class_sums = torch.zeros(34).to(device)
    for e, (_, mask) in enumerate(train_dataloader):
        if e % 100 == 0:
            print(f'{e} / {len(train_dataloader.dataset)}')

        mask = mask.to(device)

        mask_1h = F.one_hot(mask, 34).permute((0, 3, 1, 2))
        sums = mask_1h.sum(dim = (0, 2, 3))
        class_sums += sums

    print(enumerate(class_sums.cpu()))

if __name__ == '__main__':
    main()
