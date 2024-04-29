import torch
from torch import nn

import torch.nn.functional as F
from torch import FloatTensor, LongTensor

class IoULoss(nn.Module):
    def __init__(self, ignore = None):
        super().__init__()

        if ignore is None:
            ignore = -1

        self.ignore = ignore

    def forward(self, preds: FloatTensor, labels: LongTensor) -> FloatTensor:
        probs = preds.softmax(dim = 1)
        one_hot = F.one_hot(labels, num_classes = preds.size(1)).permute((0, 3, 1, 2))

        mask = (labels != self.ignore).unsqueeze(1).expand_as(one_hot)
        probs = probs * mask
        one_hot = one_hot * mask

        intersection = torch.sum(probs * one_hot, dim = (2, 3))
        union = torch.sum(probs + one_hot, dim = (2, 3)) - intersection

        iou = intersection / union.clamp(min = 1e-6)

        return 1 - iou.mean()

class IoUScore(IoULoss):
    def __init__(self):
        return super().__init__(ignore = 0)

    def forward(self, preds: FloatTensor, labels: LongTensor) -> FloatTensor:
        best = F.one_hot(preds.argmax(dim = 1), num_classes = preds.size(1)).permute((0, 3, 1, 2)).to(torch.float32)
        return 100 * (1 - super().forward(best, labels))

class InstanceIoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds: FloatTensor, labels: LongTensor) -> FloatTensor:
        probs = preds.softmax(dim = 1)
        one_hot = F.one_hot(labels, num_classes = preds.size(1)).permute((0, 3, 1, 2))

        intersection = torch.sum(probs * one_hot, dim = (2, 3))
        union = torch.sum(probs + one_hot, dim = (2, 3)) - intersection

        iou = intersection / union.clamp(min = 1e-6)

        return 1 - iou.mean()
