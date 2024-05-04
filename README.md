# CityScapes!
Assessment for Deep Learning for Image Analysis from Martin Fixman and Grigorios Vaitsas.

# Downloading the dataset.
To download the dataset, you must accept the terms and conditions and create an account at https://www.cityscapes-dataset.com. Unfortunately, this makes it impossible to bundle the data in this.

The dataset consists of two packages, which can be downloaded after signing up in the Cityscapes website.
1. **City images** `leftImg8bit_trainvaltext.zip` (11GB): https://www.cityscapes-dataset.com/file-handling/?packageID=3
2. **Coarse masks** `gtCoarse.zip` (1.3GB): https://www.cityscapes-dataset.com/file-handling/?packageID=2

These files should be downloaded to the `data/leftImg8bit` and `data/coarse` directories, respectively. These are later read by `CityScapesDataset.py` when running a model.

# Wandb considerations.
If you are logged to a wandb account, all models and their interesting data can be logged in this account.

In Hyperion, you might want to modify the `WANDB_API_KEY` inside the `sh` file.

All of the input parameters sent to a `Trainer` will appear in the wandb config parameters. This allows for each searching and filtering.

# How to Train a new Model
`train.py` trains a new model with a chosen model and hyperparameters.

Options for the training can be set as command line arguments, which are explained in `train.py --help`.
```
options:
  -h, --help            show this help message and exit

Model options:
  Options that affect the model and its training

  --model {Baseline,EnhancedUNet,EnhancedSwin,...}
                        Which model to use. "Baseline", "EnhancedUNet", and "EnhancedSwin" expand to the corresponding models.
  --pretrained-model-weights PRETRAINED_MODEL_WEIGHTS
                        Wandb ID to pre-load model weights


  --lr LR               Initial learning rate.
  --weight-decay WEIGHT_DECAY
                        L2 weight decay.
  --gamma GAMMA         Learning rate decay every 10 epochs.
  --optimiser {Adam,AdamW,Adamax}
                        Optimiser.
  --dropout DROPOUT     How much dropout to use (if applicable).
  --loss-fn {cross_entropy,dice_loss,iou_loss}
                        Loss function.

Input image options:
  Options that affect how the training images are read

  --granularity {fine,coarse}
                        Granularity of the dataset. Only coarse (the default) was used in the final
                        report.
  --image-size IMAGE_SIZE
                        The square image size to use.
  --no-resize           Do not resize the image.

Training options:
  Options that affect how the model is trained

  --epochs EPOCHS       Number of epochs.
  --batch-size [BATCH_SIZE]
                        Batch size.
  --label LABEL         Label added to wandb artifacts.
  --device {cuda,mps,cpu}
                        Which device to use.
```

Example usage:
```
python train.py --model EnhancedSwin --gamma 0.123 --lr 0.001 --weight-decay 0.00001 --optimiser Adamax
```

# How to continue training a pre-trained model.
The optional `--pretrained-model-weights` arguments in `train.py` can point to a Wandb artifact (NOT a model ID or tag!). This model will be used as a base and will continue training from there.

Note that:
1. The model needs to be identical to the pre-trained one. Ensure that the command-line arguments are the same.
2. The epoch number will start from 1 regardless of the amount of epochs in the previous model.

# How to run a parameter sweep
`halving_param_sweep.py` runs a new halving parameter sweep, which is explained in **Section 2.5** of the report.

The arguments, explained in `python halving_param_sweep.py --help`, are similar to the ones used in `train.py` without most model options (other than a boolean --sweep-dropout to also do a parameter sweep over the dropout).

The values that are swept can be modified inside the `.py` file.

# How to evaluate a model against one image.
`eval_mask.py` evaluates a model against an image (which can be from CityScapes or anywhere else!) and returns a mask to be used for comparison.

It can also optionally create a second image of comparison with a ground truth.

Example usages:
```
python eval_mask.py enhanced_swin2 city_file.png output_mask.png
python eval_mask.py --gt-output ground_truth.png enhanced_swin2 data/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png output_mask.png
```

The model name (`enhanced_swin2`) can be either a Wandb model name or a Wandb tag; the evaluator takes its config and the latest artifact (which is the one with the best validation score).

`baseline`, `enhanced_unet`, and `enhanced_swin2` expand to the final models from our assessment.

# How to get the score of an already-trained model
`eval_score.py` evaluates a model in one of the CityScapes (train, val, test) datasets.

Note that the `test` dataset does not contain valid images when downloaded from the CityScapes website, as these images are secret.

As before, the model can be a tag or a model ID.
`baseline`, `enhanced_unet`, and `enhanced_swin2` expand to the final models from our assessment.

Example usage:
```
python eval_score.py enhanced_swin2
```
