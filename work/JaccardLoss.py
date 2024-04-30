import torch
from torch import nn
from torch import tensor

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

class IoUScore(nn.Module):
    def __init__(self, ignore_index = None):
        super().__init__()

        if ignore_index is None:
            ignore_index = -1
        self.ignore_index = ignore_index

    def forward(self, preds: FloatTensor, labels: LongTensor) -> FloatTensor:
        num_classes = preds.size(1)

        preds = torch.argmax(preds, dim = 1)
        one_hot_preds = F.one_hot(preds, num_classes = num_classes).permute(0, 3, 1, 2)
        one_hot_labels = F.one_hot(labels, num_classes = num_classes).permute(0, 3, 1, 2)

        mask = torch.unsqueeze(labels != self.ignore_index, dim = 1)
        one_hot_preds = one_hot_preds * mask
        one_hot_labels = one_hot_labels * mask

        tp = torch.sum( one_hot_preds &  one_hot_labels, dim = (2, 3))
        fp = torch.sum( one_hot_preds & ~one_hot_labels, dim = (2, 3))
        fn = torch.sum(~one_hot_preds &  one_hot_labels, dim = (2, 3))

        intersection = tp
        union = tp + fp + fn
        iou = torch.where(union == 0, 0, intersection / union)

        return 100 * iou.mean()

class InstanceIoUScore(nn.Module):
    # Total amount of pixels of this instance in 768 x 768 images.
    # Calculated using calculate_weights.py.
    instance_inv_sums = tensor([
        515257920, 80242488, 22716522, 27322400, 0, 0,
        262198, 510640960, 48721176, 2616676, 896892, 227768096,
        6251469, 9440081, 146116, 655928, 338741, 6654748,
        0, 1184962, 4447795, 148047248, 6715583, 40057328,
        9196087, 845308, 73100160, 2680537, 2653371, 148543,
        70536, 2323841, 667847, 2654887,
     ])
    instance_inv_ratio = instance_inv_sums / instance_inv_sums.sum()

    def __init__(self, ignore_index = None):
        super().__init__()

        if ignore_index is None:
            ignore_index = -1
        self.ignore_index = ignore_index

    def forward(self, preds: FloatTensor, labels: LongTensor) -> FloatTensor:
        num_classes = preds.size(1)

        preds = torch.argmax(preds, dim = 1)
        one_hot_preds = F.one_hot(preds, num_classes = num_classes).permute(0, 3, 1, 2)
        one_hot_labels = F.one_hot(labels, num_classes = num_classes).permute(0, 3, 1, 2)

        # Torch handles dimensionality expansions nicely.
        mask = torch.unsqueeze(labels != self.ignore_index, dim = 1)
        one_hot_preds = one_hot_preds * mask
        one_hot_labels = one_hot_labels * mask

        tp = torch.sum( one_hot_preds &  one_hot_labels, dim = (2, 3))
        fp = torch.sum( one_hot_preds & ~one_hot_labels, dim = (2, 3))
        fn = torch.sum(~one_hot_preds &  one_hot_labels, dim = (2, 3))

        weights = 1 / self.instance_inv_ratio.to(preds.device).clamp(min = 1e-6)
        itp = weights * tp
        ifn = weights * fn

        intersection = itp
        union = itp + fp + ifn
        iiou = torch.where(union == 0, 0, intersection / union)

        return 100 * torch.mean(iiou)
