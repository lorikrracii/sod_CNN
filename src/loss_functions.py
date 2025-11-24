import torch
import torch.nn as nn

#1 binary cross entropy loss
bce_loss_fn = nn.BCELoss() #BCELoss bcs our model outputs sigmoid probabilities (0 to 1)

def bce_loss(pred, target):
    #binary cross entropy. measures pixel-wise difference between mask ang GT mask
    return bce_loss_fn(pred, target)

#2 IoU loss (soft IoU)
def iou_loss(pred, target, eps=1e-6):
    """iou = intersection / union
    Iou loss = 1 - IoU
    """
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection

    iou = (intersection + eps) / (union + eps)
    return 1 - iou.mean()

#3 combined Loss = BCE + 0.5 * Iou Loss

def combined_loss(pred, target):
    """ Combined loss ---> Loss = BCE + 0.5 x (1 - IoU)"""
    bce = bce_loss(pred, target)
    iou = iou_loss(pred, target)
    return bce + 0.5 * iou