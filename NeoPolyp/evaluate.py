from torch.utils.data import DataLoader
import torch
import yaml
from dataset import EvalNeoPolyp
import torch.nn.functional as F
from tqdm import tqdm
import pdb
import numpy as np
from metrics import compute_dice_coef, compute_F1, compute_IoU
from albumentations import (Normalize, Resize, Compose)
from albumentations.pytorch import ToTensorV2
import argparse
import cv2
import albumentations as A
import pandas as pd
from utils.helper import get_args_parser, my_post_process
from utils.train_utils import prepare_architecture

def evaluate(config, model, validation_loader):
    model.eval()
    dice_coef_list = []
    F1_list = []
    IoU_list = []

    with torch.no_grad():
        for sample in tqdm(validation_loader, desc="Evaluating", leave=False):
            images, masks = sample['image'], sample['mask']
            if not config['transform']:
                images = images.type(torch.FloatTensor)

            images, masks = images.to(
                config['device']), masks.to(config['device'])

            output = model(images)  # forward
            if "blazeneo" in config['backbone']:
                output = output[1]
            elif "neounet" in config['backbone']:
                output = output[0]
            elif "pranet" in config['backbone']:
                # output  = output[0]
                output  = output[-1]
            elif "deeplabv3" in config['backbone']:
                output = output['out']
                
            if config['probability_correction_strategy']:
                pred  = F.interpolate(output, size=(images.shape[-2], images.shape[-1]), mode='bilinear', align_corners=True)
                for i in range(3):
                    pred[:,i,:,:][torch.where(pred[:,i,:,:]>0)] /= (pred[:,i,:,:]>0).float().mean()
                    pred[:,i,:,:][torch.where(pred[:,i,:,:]<0)] /= (pred[:,i,:,:]<0).float().mean()
                probs = torch.softmax(pred, dim=1)
            else:
                probs = torch.softmax(output, dim=1)
                
            predictions = torch.argmax(probs, dim=1)
            predictions = predictions.detach().cpu().numpy()[0,:,:]
                
            height, width = masks.shape[-2], masks.shape[-1]

            if predictions.shape != (height, width):
                predictions = cv2.resize(predictions, dsize=(width, height), interpolation=cv2.INTER_NEAREST)
            
            predictions = torch.from_numpy(np.array(predictions)).unsqueeze(0).to(config['device'])
            
            # predictions = my_post_process(predictions)

            dice_coef = compute_dice_coef(masks, predictions)
            F1_score = compute_F1(predictions, masks)
            IoU_score = compute_IoU(predictions, masks)
            dice_coef_list.append(dice_coef.cpu().numpy())
            F1_list.append(F1_score.cpu().numpy())
            IoU_list.append(IoU_score.cpu().numpy())

    return np.mean(dice_coef_list), np.mean(F1_list), np.mean(IoU_list)

if __name__ == "__main__":
    # Evaluating on batch size 1 because of different sizes in the test set
    parser = argparse.ArgumentParser("NeoPolyp evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    
    # load yaml file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(config)
    weights_path = config['save_checkpoint']
    valid_transform = Compose([
        Resize(config['image_height'], config['image_width']),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
        ToTensorV2(),
    ])
    
    model = prepare_architecture(config)
    
    validating_data = EvalNeoPolyp(config['root_valid_image_path'], config['root_valid_label_path'], transform=valid_transform)
    validation_loader = DataLoader(validating_data, batch_size=1, shuffle=True)

    print("Evaluating using ", weights_path)
    model.load_state_dict(torch.load(weights_path)['model_state_dict'])
    model.to(config['device'])
    
    dice_coef, F1, IoU = evaluate(config, model, validation_loader)
    print(f"Test. dice score: {dice_coef:.3f}")
    print(f"Test. F1 score: {F1:.3f}")
    print(f"Test. IoU score: {IoU:.3f}")