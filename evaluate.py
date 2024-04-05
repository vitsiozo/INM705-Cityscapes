import torch
import random
import glob

from PIL import Image
from work import CityScapesDataset, Trainer

class Evaluator:
    def __init__(self, model_file):
        weights = torch.load(model_file, map_location = torch.device('cpu'))
        self.model = Trainer.get_model(30).cpu()
        self.model.load_state_dict(weights)

    @torch.no_grad()
    def eval(self, path = None):
        if path is None:
            path = random.choice(glob.glob('data/leftImg8bit/val/**/*.png'))
            print(f'Evaluating {path}')

        image = CityScapesDataset.transform(Image.open(path).convert('RGB'))
        result = self.model(image.unsqueeze(0))['out']
        return result.squeeze(0).argmax(dim = 0)
