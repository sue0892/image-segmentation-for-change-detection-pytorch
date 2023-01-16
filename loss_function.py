# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class DiceLoss:
    def __init__(self,num_classes=2):
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        loss = 0
        eps = 1e-15
        for cls in range(self.num_classes):
            jaccard_target = (targets == cls).float()
            #outputs has done F.softmax
            jaccard_output = outputs[:, cls] # RSAUNet result only does F.softmax
            #jaccard_output = outputs[:, cls].exp() # use .exp() is due to RSAUNet result has done F.log_softmax
            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()
            loss -= torch.log((2*intersection + eps) / (union + eps))
        return loss

class FocalLoss:
    def __init__(self, alpha=None, gamma=0, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def __call__(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        #Actually, outputs has done F.softmax
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
        
class FocalLoss_DiceLoss:
    def __init__(self, dice_weight=0.5, num_classes=2, alpha=0.25, gamma=2):
        self.jaccard_weight = dice_weight
        self.num_classes = num_classes
        self.alpha=alpha 
        self.gamma=gamma

    def __call__(self, outputs, targets):
        #outputs has done F.softmax
        focal_loss = FocalLoss(alpha=self.alpha, gamma=self.gamma)(outputs, targets)
        dice_loss = DiceLoss(num_classes = self.num_classes)(outputs, targets)
        loss = self.jaccard_weight * focal_loss + (1 - self.jaccard_weight) * dice_loss
        return loss