# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 13:32:42 2020

@author: 11627
"""
# loss.py
import torch.nn as nn
import torch
import numpy as np




class EntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits):
        p = torch.sigmoid(logits)
        elementwise_entropy = -p * torch.log2(p+ 1e-30)
        if self.reduction == 'none':
            return elementwise_entropy

        sum_entropy = torch.sum(elementwise_entropy, dim=1)
        if self.reduction == 'sum':
            return sum_entropy

        return torch.mean(sum_entropy)



def diceCoeff(pred, gt, eps=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss =  (2 * intersection + eps) / (unionset + eps)
#    loss1= torch.nn.functional.cross_entropy(pred, gt)
    return  loss.sum() / N


class SoftDiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, activation=None, reduction='mean'):
        super(SoftDiceLoss, self).__init__()
        self.activation = activation

    def forward(self, y_pred, y_true):

        
        return 1 - diceCoeff(y_pred, y_true, activation=self.activation)

def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))

def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCELoss()(y_pred, y_truth_tensor)
