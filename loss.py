import torch
from torch import nn
import torch.nn.functional as F
import pdb

def dice_loss(inputs, targets):
    smooth=1
    inputs = torch.sigmoid(inputs)
    #flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    
    intersection = (inputs * targets).sum()                            
    dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
    
    return dice

class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target)
        return loss.mean()

class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        # loss = self.alpha*self.focal(input, target)
        loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        # loss = torch.log(dice_loss(input, target))
        return loss.mean()
    

NUM_CLASS = 2
class ActiveContourLoss(nn.Module):
    def __init__(self, class_weight=[1] * NUM_CLASS):
        """
        class weight should be a list. 
        """
        super().__init__()
        self.class_weight = torch.tensor(class_weight)
    def forward(self, y_true, y_pred):   
        yTrueOnehot = torch.zeros(y_true.size(0), NUM_CLASS, y_true.size(2), y_true.size(3))
        yTrueOnehot = torch.scatter(yTrueOnehot, 1, y_true, 1)

        active = torch.sum(yTrueOnehot * (1 - y_pred) + (1 - yTrueOnehot) * y_pred, dim=[2, 3])
        loss = torch.sum(active * self.class_weight)
        return loss / (torch.sum(self.class_weight) * y_true.size(0) * y_true.size(2) * y_true.size(3))