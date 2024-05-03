# CityScapes!
Assessment for Deep Learning for Image Analysis from Martin Fixman and Greg Gregoris.

# Downloading the dataset.
To download the dataset, you must accept the terms and conditions and create an account at https://www.cityscapes-dataset.com. Unfortunately, this makes it impossible to bundle the data in this.

The dataset consists of two packages, which can be downloaded after signing up in the Cityscapes website.
1. **City images** `leftImg8bit_trainvaltext.zip` (11GB): https://www.cityscapes-dataset.com/file-handling/?packageID=3
2. **Coarse masks** `gtCoarse.zip` (1.3GB): https://www.cityscapes-dataset.com/file-handling/?packageID=2

These files should be downloaded to the `data/leftImg8bit` and `data/coarse` directories, respectively. These are later read by `CityScapesDataset.py` when running a model.

# Wandb considerations.
If you are logged to a wandb account, all models and their interesting data can be logged its account.

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

# How to run a parameter sweep
`halving_param_sweep.py` runs a new halving parameter sweep, which is explained in **Section 2.5** of the report.

The arguments, explained in `python halving_param_sweep.py --help`, are similar to the ones used in `train.py` without most model options (other than a boolean --sweep-dropout to also do a parameter sweep over the dropout).

Which values are swept can be modified inside the `.py` file.
