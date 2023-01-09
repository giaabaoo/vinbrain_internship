import numpy as np
import cv2
import yaml
import torch
import pdb
from albumentations import (Normalize, Resize, Compose)
from albumentations.pytorch import ToTensorV2
from metrics import compute_dice_coef
from pathlib import Path
import argparse
from tqdm import tqdm
import os
from PIL import Image, ImageDraw, ImageFont
from utils.helper import get_concat_h, get_args_parser, valid_postprocess, visualize, read_mask, refine_mask
from utils.train_utils import prepare_architecture
import torch.nn.functional as F


LABEL_TO_COLOR = {0:[0,0,0], 1:[255,0,0], 2:[0,255,0]}
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "NeoPolyp inference script", parents=[get_args_parser()])
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(config)
    
    transform = Compose([
        Resize(config['image_height'], config['image_width']),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
        ToTensorV2(),
    ])

    model = prepare_architecture(config)
    backbone = config['csv_name'] 
    Path(f"./results/{backbone}/hmasks").mkdir(parents=True, exist_ok=True)
    Path(f"./results/{backbone}/valid_blend").mkdir(parents=True, exist_ok=True)
    Path(f"./results/{backbone}/valid_mask").mkdir(parents=True, exist_ok=True)
    weights_path = config['save_checkpoint'] 
    print("Inferencing using ", weights_path)
    model.load_state_dict(torch.load(weights_path)['model_state_dict'])
    model.to(config['device'])
    
    # path = config['root_valid_image_path']
    path = "./example_data/valid"
    # Inference on all images
    for image_name in tqdm(os.listdir(path)):
        image = cv2.imread(os.path.join(path, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        model.eval()
        ground_truth_path = os.path.join(path.replace("valid", "valid_mask"), image_name)
        mask = cv2.imread(ground_truth_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        gt_pil = Image.fromarray(mask).convert('RGB')
        
        augmented = transform(image=image, mask=mask)
        image_transform = augmented['image']
        mask = augmented['mask']

        mask = read_mask(mask).to(config['device'])
        
        with torch.no_grad():
            if not config['transform']:
                image_transform = image_transform.type(torch.FloatTensor)
            image_transform = image_transform.unsqueeze(0).to(config['device'])

            output = model(image_transform)
            if "blazeneo" in config['backbone']:
                output = output[1]
            elif "neounet" in config['backbone']:
                output = output[0]
            elif "pranet" in config['backbone']:
                output  = output[0]
                # output  = output[-1]
            elif "deeplabv3" in config['backbone']:
                output = output['out']
        
        if config['probability_correction_strategy']:
            pred  = F.interpolate(output, size=(image_transform.shape[-2], image_transform.shape[-1]), mode='bilinear', align_corners=True)
            for i in range(3):
                pred[:,i,:,:][torch.where(pred[:,i,:,:]>0)] /= (pred[:,i,:,:]>0).float().mean()
                pred[:,i,:,:][torch.where(pred[:,i,:,:]<0)] /= (pred[:,i,:,:]<0).float().mean()
            probs = torch.softmax(pred, dim=1)
        else:
            probs = torch.softmax(output, dim=1)
            
        prediction = torch.argmax(probs, dim=1)
        
        
        output = valid_postprocess(prediction.cpu().numpy(), image)
        
        ### refine_mask by following the rule: 1 label per polyp only
        output = refine_mask(image_name, output)
        dice_coef = compute_dice_coef(prediction, mask).cpu().numpy()
        
        cv2.imwrite(f"./results/{backbone}/valid_mask/{image_name}", output)
        # pdb.set_trace()
        output_pil = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        output_pil = Image.fromarray(output_pil)
        
        font = ImageFont.truetype("LiberationSans-Regular.ttf", 30)
        d1 = ImageDraw.Draw(gt_pil)
        d1.text((width/2 - 100, height-100), "Ground truth", fill=(255, 255, 255), font=font)

        d2 = ImageDraw.Draw(output_pil)
        d2.text((width/2 - 100, height-100), f"Prediction with scores = {dice_coef.item()}", fill=(255, 255, 255), font=font)

        get_concat_h(output_pil, gt_pil).save(f'results/{backbone}/hmasks/{image_name}')

        blend = visualize(image, output)
        cv2.imwrite(f"results/{backbone}/valid_blend/{image_name}", blend)
        
