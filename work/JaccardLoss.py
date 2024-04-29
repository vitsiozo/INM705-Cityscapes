import torch
from torch import nn

import torch.nn.functional as F
from torch import FloatTensor, LongTensor

class IoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds: FloatTensor, labels: LongTensor):
        one_hot = F.one_hot(labels, num_classes = preds.size(1)).permute((0, 3, 1, 2))

        intersection = torch.sum(preds * one_hot, dim = (2, 3))
        union = torch.sum(preds + labels, dim = (2, 3))

        iou = intersection / union
        return 1 - iou.sum()
