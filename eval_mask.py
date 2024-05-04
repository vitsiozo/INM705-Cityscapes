import argparse
import logging
import os
import torch
import wandb

from typing import Any

from torch import tensor
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms

from PIL import Image
from torchvision.transforms.functional import to_pil_image

from CityScapesDataset import CityScapesDataset
from Model import Model

from Trainer import Trainer

from numpy import array

import re

def parse_args():
    parser = argparse.ArgumentParser(
            description = 'Downloads a Wandb Artifact and calculates its mask',
            epilog = '''Example usages:
python eval_mask.py baseline city_file.png output_mask.png
python eval_mask.py --gt-output ground_truth.png enhanced_swin2 data/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png output.png
            ''',
            formatter_class = argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--batch-size', type = int, default = 16, help = 'Batch size. Lower this if the program runs out of memory.')
    parser.add_argument('--granularity', choices = ['coarse', 'fine'], help = 'Granularity (same as model by default).')
    parser.add_argument('--device', choices = ['cuda', 'mps', 'cpu'], default = 'cuda', help = 'Which device to evaluate on.')
    
    parser.add_argument('--gt-output', help = 'If set, will assume this is an input CityScapes image and create a comparison to its ground truth')
    parser.add_argument('--append-model-name', action = 'store_true', help = 'Append the model name to the name of the output file.')

    parser.add_argument('--model', required = True, help = 'Wandb tags or model name. Use `baseline`, `enhanced_unet`, and `enhanced_swin2` for final models.')
    parser.add_argument('input_file', help = 'Input file. If --compare-to-gt, then it should point to somewhere in the `data` directory.')
    parser.add_argument('output_file', help = 'Output file')

    args = parser.parse_args()
    if args.append_model_name:
        model = args.model.replace('enhanced_', '')
        args.output_file = args.output_file.replace('.png', f'_{model}.png')

    return args

# Gets a run that matches `tag` as either a tag (ie 'EnhancedSwin2') or
# a model name (ie 'wspszqbr') and its latest artifact weights.
# Project name hardcoded for simplicity. Sorry Greg!
def get_run(tag: str) -> tuple[dict[str, Any], str]:
    api = wandb.Api()
    runs = api.runs('mfixman-convolutional-team/work', {'$or': [
        {'tags': tag},
        {'name': tag},
    ]})

    if len(runs) == 0:
        raise NameError(f'No run found with either tag or name "{tag}"')
    
    if len(runs) > 1:
        logging.warning(f'WARNING: {len(runs)} runs found with this tag or name! Choosing one of them.')

    run = runs[0]
    artifact = max(run.logged_artifacts(), key = lambda x: x.version)
    artifact_dir = artifact.download()

    return run.config, os.path.join(artifact_dir, 'model.pth')

def join_imagemask(image, mask):
    mask = to_pil_image(Trainer.apply_palette(mask, 34).squeeze(0)).convert('RGBA')

    alpha = mask.split()[3].point(lambda x: x / 2)
    mask.putalpha(alpha)

    result_image = Image.alpha_composite(to_pil_image(image).convert('RGBA'), mask).convert('RGB')
    return result_image

def get_image(input_file):
    image = Image.open(input_file).convert('RGB')
    return transforms.functional.to_dtype(transforms.functional.to_image(image), dtype = torch.float32, scale = True)

@torch.no_grad()
def get_imagemask(image, model):
    device = next(model.parameters()).device
    output_mask = model(image.unsqueeze(0).to(device)).argmax(dim = 1).cpu()
    return join_imagemask(image, output_mask)

def main():
    logging.basicConfig(
        level = logging.INFO,
        format = '[%(asctime)s] %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S',
    )

    args = parse_args()
    config, artifact = get_run(args.model)
    granularity = args.granularity or config['granularity']

    model = Model.instanciate(
        config['model_name'],
        in_channels = 3,
        out_channels = CityScapesDataset.n_classes,
        dropout = config.get('dropout'),
    ).to(args.device)
    weights = torch.load(artifact, map_location = args.device)
    model.load_state_dict(weights)
    model.eval()

    image = get_image(args.input_file)
    result_image = get_imagemask(image, model)
    result_image.save(args.output_file)

    if args.gt_output is not None:
        gt_mask_file, subs = re.subn(r'(.*)/leftImg8bit/(.*)_leftImg8bit.png', rf'\1/{granularity}/\2_gt{granularity.title()}_labelIds.png', args.input_file)
        if subs < 1:
            raise NameError(f'Equivalent ground truth file not found!')

        gt_mask = torch.from_numpy(array(Image.open(gt_mask_file))).long().unsqueeze(0)
        gt_result = join_imagemask(image, gt_mask).save(args.gt_output)

if __name__ == '__main__':
    main()
