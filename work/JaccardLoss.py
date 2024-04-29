import torch
from torch import nn

import torch.nn.functional as F
from torch import FloatTensor, LongTensor

class IoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds: FloatTensor, labels: LongTensor) -> FloatTensor:
        probs = preds.softmax(dim = 1)
        one_hot = F.one_hot(labels, num_classes = preds.size(1)).permute((0, 3, 1, 2))

        intersection = torch.sum(probs * one_hot, dim = (2, 3))
        union = torch.sum(probs + one_hot, dim = (2, 3)) - intersection

        iou = intersection / union.clamp(min = 1e-6)

        return 1 - iou.mean()
