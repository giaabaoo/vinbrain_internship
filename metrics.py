import torch
import pdb
import numpy as np

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
    
    
def all_dice_scores(predictions, labels, threshold):
    batch_size = len(labels)
    
    predictions = torch.sigmoid(predictions)
    predictions = predictions.view(batch_size, -1)
    labels = labels.view(batch_size, -1)
    assert(predictions.shape == labels.shape)

    p = (predictions > threshold).float()
    t = (labels > 0.5).float()

    t_sum = t.sum(-1) # sum of 32 labels --> 0 means image without mask, 1 means image with mask 32, 512x512
    p_sum = p.sum(-1)
    neg_index = torch.nonzero(t_sum == 0) # index of images without mask in a batch
    pos_index = torch.nonzero(t_sum >= 1) # index of images with mask in a batch

    dice_neg = (p_sum == 0).float()
    dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

    dice_neg = dice_neg[neg_index]
    dice_pos = dice_pos[pos_index]
    dice = torch.cat([dice_pos, dice_neg])

    return dice, dice_neg, dice_pos