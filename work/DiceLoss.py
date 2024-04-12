import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda'

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.softmax(inputs, dim=1).to(device)
        targets = targets.long().unsqueeze(1)
        targets_one_hot = torch.eye(inputs.size(1)).to(device)[targets].squeeze(2).to(device)

        inputs = inputs.view(-1)
        targets_one_hot = targets_one_hot.view(-1)
        
        intersection = (inputs * targets_one_hot).sum()                            
        dice = (2.*intersection + smooth) / (inputs.sum() + targets_one_hot.sum() + smooth)  
        
        return 1 - dice
