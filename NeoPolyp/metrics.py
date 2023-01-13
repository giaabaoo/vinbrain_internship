import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import segmentation_models_pytorch as smp
import pdb

def compute_dice_coef(y_true, y_pred, thr=0.5, dim=0, epsilon=0.001):
    # y_pred = torch.sigmoid(y_pred)
    y_true = y_true.to(torch.int64)
    # y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice
    

def compute_F1(output, target):
    # pdb.set_trace()
    tp, fp, fn, tn = smp.metrics.get_stats(output, target, mode='multiclass', num_classes=3)
    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")
    
    return f1_score

def compute_IoU(output, target):
    tp, fp, fn, tn = smp.metrics.get_stats(output, target, mode='multiclass', num_classes=3)
    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
    
    return iou_score

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs