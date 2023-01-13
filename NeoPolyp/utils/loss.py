import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import pdb

# def split_mask(neo_mask):
#     ignore_mask = neo_mask[:, [0], :, :]
#     # sum of ignore, neo and non-neo
#     polyp_mask = ignore_mask + neo_mask[:, [1], :, :] + neo_mask[:, [2], :, :]
#     # neo, non-neo and background
#     neo_mask = neo_mask[:, [1, 2, 3], :, :]
    
#     return polyp_mask, neo_mask, ignore_mask


def CELoss(inputs, targets, ignore=None):
    if inputs.shape[1] == 1:
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    else:
        ce_loss = F.cross_entropy(inputs, torch.argmax(targets, axis=1), reduction='none')

    if ignore is not None:
        ignore = 1 - ignore.squeeze()
        ce_loss = ce_loss * ignore

    return ce_loss.mean()


def FocalTverskyLoss(inputs, targets, alpha=0.7, beta=0.3, gamma=4/3, smooth=1, ignore=None):
    if inputs.shape[1] == 1:
        inputs = torch.sigmoid(inputs)
    else:
        inputs = torch.softmax(inputs, dim=1)

    if ignore is None:
        tp = (inputs * targets).sum(dim=(0, 2, 3))
        fp = (inputs).sum(dim=(0, 2, 3)) - tp
        fn = (targets).sum(dim=(0, 2, 3)) - tp
    else:
        ignore = (1-ignore).expand(-1, targets.shape[1], -1, -1)
        tp = (inputs * targets * ignore).sum(dim=(0, 2, 3))
        fp = (inputs * ignore).sum(dim=(0, 2, 3)) - tp
        fn = (targets * ignore).sum(dim=(0, 2, 3)) - tp
    
    ft_score = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    ft_loss = (1 - ft_score) ** gamma
    
    return ft_loss.mean()


class BlazeNeoLoss(nn.Module):
    __name__ = 'blazeneo_loss'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_prs, mask):
        polyp_mask, neo_mask, ignore_mask = split_mask(mask)

        ce_loss = CELoss(y_prs[0], polyp_mask)
        ft_loss = FocalTverskyLoss(y_prs[0], polyp_mask)
        aux_loss = ce_loss + ft_loss

        ce_loss = CELoss(y_prs[-1], neo_mask, ignore=ignore_mask)
        ft_loss = FocalTverskyLoss(y_prs[-1], neo_mask, ignore=ignore_mask)
        main_loss = ce_loss + ft_loss

        return aux_loss + main_loss


class UNetLoss(nn.Module):
    __name__ = 'unet_loss'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pr, mask):
        polyp_mask, neo_mask, ignore_mask = split_mask(mask)

        ce_loss = CELoss(y_pr, neo_mask, ignore=ignore_mask)
        ft_loss = FocalTverskyLoss(y_pr, neo_mask, ignore=ignore_mask)
        main_loss = ce_loss + ft_loss

        return main_loss


class HarDNetMSEGLoss(nn.Module):
    __name__ = 'hardnetmseg_loss'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pr, mask):
        polyp_mask, neo_mask, ignore_mask = split_mask(mask)

        ce_loss = CELoss(y_pr, neo_mask, ignore=ignore_mask)
        ft_loss = FocalTverskyLoss(y_pr, neo_mask, ignore=ignore_mask)
        main_loss = ce_loss + ft_loss

        return main_loss


# class PraNetLoss2(nn.Module):
#     __name__ = 'pranet_loss'

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def forward(self, y_prs, mask):
#         main_loss = 0
#         for y_pr in y_prs:
#             polyp_mask, neo_mask, ignore_mask = split_mask(mask)

#             ce_loss = CELoss(y_pr, neo_mask, ignore=ignore_mask)
#             ft_loss = FocalTverskyLoss(y_pr, neo_mask, ignore=ignore_mask)
#             main_loss += ce_loss + ft_loss

#         return main_loss


class NeoUNetLoss(nn.Module):
    __name__ = 'neounet_loss'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_prs, mask):
        main_loss = 0
        pdb.set_trace()

        neo_gt = mask[:, 1, :, :]
        non_gt = mask[:, 2, :, :]
        polyp_gt = neo_gt + non_gt
        
        for y_pr in y_prs:
            neo_pr = y_pr[:, 0, :, :]
            non_pr = y_pr[:, 1, :, :]
            polyp_pr = (neo_pr > non_pr) * neo_pr + (non_pr > neo_pr) * non_pr

            main_loss += (CELoss(neo_pr, neo_gt) + FocalTverskyLoss(neo_pr, neo_gt) + \
                    CELoss(non_pr, non_gt) + FocalTverskyLoss(non_pr, non_gt) + \
                    CELoss(polyp_pr, polyp_gt) + FocalTverskyLoss(polyp_pr, polyp_gt)) / len(y_prs)

        return main_loss

NUM_CLASS = 3
class ActiveContourLoss(nn.Module):
    def __init__(self, device, class_weight=[1] * NUM_CLASS):
        """
        class weight should be a list. 
        """
        super().__init__()
        self.device = device
        self.class_weight = torch.tensor(class_weight, device=device)
    def forward(self, y_pred, y_true):   
        y_true = y_true.unsqueeze(1)
        yTrueOnehot = torch.zeros(y_true.size(0), NUM_CLASS, y_true.size(2), y_true.size(3), device=self.device)
        yTrueOnehot = torch.scatter(yTrueOnehot, 1, y_true, 1)
        
        active = torch.sum(yTrueOnehot * (1 - y_pred) + (1 - yTrueOnehot) * y_pred, dim=[-2, -1])
        loss = torch.sum(active * self.class_weight)
        return loss / (torch.sum(self.class_weight) * y_true.size(0) * y_true.size(-2) * y_true.size(-1))

# class PraNetLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, pred, mask):   
#         pdb.set_trace()
#         weit = 1 + 5*torch.abs(F.avg_pool2d(mask.type(torch.FloatTensor).to('cuda'), kernel_size=31, stride=1, padding=15) - mask.type(torch.FloatTensor).to('cuda'))
        
#         criterion = nn.CrossEntropyLoss(reduce='None')
#         wbce = criterion(pred, mask)
#         wbce = (weit*wbce).sum(dim=(-2, -1)) / weit.sum(dim=(-2, -1))

#         # pred = torch.sigmoid(pred)
#         probs = torch.softmax(pred, dim=1)
#         pred = torch.argmax(probs, dim=1)
        
#         inter = ((pred * mask)*weit).sum(dim=(-2, -1))
#         union = ((pred + mask)*weit).sum(dim=(-2, -1))
#         wiou = 1 - (inter + 1)/(union - inter+1)
#         return (wbce + wiou).mean() 

class PraNetLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, mask, weights=[1,1]):  
        criterion = [nn.CrossEntropyLoss(), smp.losses.DiceLoss(mode='multiclass')]
        ce_loss = criterion[0](pred, mask)
        dice_loss = criterion[1](pred, mask)

        # loss =(ce_loss +  dice_loss).mean()
        loss = weights[0] * ce_loss +  weights[1] * dice_loss
        return loss
    
class MultiCELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, mask):  
        criterion = nn.CrossEntropyLoss()
        ce_loss = criterion(pred, mask)
        return ce_loss