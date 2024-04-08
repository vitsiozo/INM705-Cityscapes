import logging
import socket
import wandb

from torch.utils.data import DataLoader

from CityScapesDataset import CityScapesDataset
from Trainer import Trainer

def main():
    is_hyperion = 'hyperion' in socket.gethostname()

    config = dict(
        n = None if is_hyperion else 10,
        batch_size = 64 if is_hyperion else 1,
        epochs = 300 if is_hyperion else 2,
        ignore_index = 0,
        granularity = 'fine',
        image_size = 512,
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

    train_dataset = CityScapesDataset('data/leftImg8bit/train', 'data/fine/train', n = config['n'], size = config['image_size'])
    val_dataset = CityScapesDataset('data/leftImg8bit/val', 'data/fine/val', n = config['n'], size = config['image_size'])

    train_dataloader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = config['batch_size'], shuffle = True)

    trainer = Trainer(train_dataloader, val_dataloader, config)
    trainer.train(epochs = config['epochs'])

if __name__ == '__main__':
    main()
