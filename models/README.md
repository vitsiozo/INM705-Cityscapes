# Models
This is a directory with models that we trained at different parts of the workflow.

All of the models can and should accessed via `Model.py` (in the parent directory).
This also includes useful aliases `Baseline`, `EnhancedUNet`, and `EnhancedSwin`.

The final three models are the following.
* **Baseline**: `SimpleFCN.py`.
    * Trained under wandb tag `baseline`.
    * https://wandb.ai/mfixman-convolutional-team/work/runs/j6iippi9
* **Enhanced-UNet**: `UNetNoBatchNorm.py`.
    * Trained under wandb tag `enhanced unet`.
    * https://wandb.ai/mfixman-convolutional-team/work/runs/5ytrv216
* **Enhanced-Swin2**: `Swin2Base.py`
    * Trained under wandb tag `enhanced_swin2`.
    * https://wandb.ai/mfixman-convolutional-team/work/runs/sty5iun8
