import numpy as np
import cv2
import yaml
import torch
import pdb
import segmentation_models_pytorch as smp
from albumentations import (Normalize, Resize, Compose)
from albumentations.pytorch import ToTensorV2
from metrics import compute_dice_coef
from pathlib import Path
import argparse
from tqdm import tqdm
from models.unet import UNet
from models.blazeneo.model import BlazeNeo
import pydicom
import os
from PIL import Image, ImageDraw, ImageFont
import albumentations as A
from utils.helper import mask2rgb, get_concat_h, get_args_parser, valid_postprocess, visualize
import shutil

LABEL_TO_COLOR = {0:[0,0,0], 1:[255,0,0], 2:[0,255,0]}

def mask_to_class(mask):
    binary_mask = np.array(mask)
    binary_mask = np.array(mask)
    binary_mask = cv2.erode(binary_mask, np.ones((5, 5), np.uint8)) 
    binary_mask[:,:,2] = 0
    binary_mask = (binary_mask != 0).astype(np.uint8)
    binary_mask *= 255
    
    rgb = np.array(binary_mask)
    output_mask = np.zeros((rgb.shape[0], rgb.shape[1]))

    for k,v in LABEL_TO_COLOR.items():
        output_mask[np.all(rgb==v, axis=2)] = k
        
    output_mask = torch.from_numpy(output_mask)
    output_mask = output_mask.type(torch.int64)
    
    return output_mask
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

    if "unetplusplus" in config['backbone'].split("."):
        model = smp.UnetPlusPlus(config['backbone'].split(".")[1], encoder_weights=config['encoder_weights'],
                         in_channels=3, classes=len(config['classes']), activation='sigmoid')
    elif "blazeneo" in config['backbone']:
        model = BlazeNeo()
    elif config['backbone'] != "None":
        model = smp.Unet(config['backbone'], encoder_weights=config['encoder_weights'],
                         in_channels=3, classes=len(config['classes']), activation='sigmoid')
    else:
        model = UNet(n_channels=3, n_classes=3)

    backbone = config['backbone'] 
    Path(f"./results/{backbone}/hmasks").mkdir(parents=True, exist_ok=True)
    Path(f"./results/{backbone}/valid_blend").mkdir(parents=True, exist_ok=True)
    Path(f"./results/{backbone}/valid_mask").mkdir(parents=True, exist_ok=True)
    weights_path = config['save_checkpoint']
    print("Inferencing using ", weights_path)
    model.load_state_dict(torch.load(weights_path)['model_state_dict'])
    model.to(config['device'])
    
    path = config['root_valid_image_path']
    # path = "./example_data/valid"
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

        mask = mask_to_class(mask).to(config['device'])
        
        if not config['transform']:
            image_transform = image_transform.type(torch.FloatTensor)
        image_transform = image_transform.unsqueeze(0).to(config['device'])

        output = model(image_transform)
        
        if "blazeneo" in config['backbone']:
            output = output[1]
            
        probs = torch.softmax(output, dim=1)
        prediction = torch.argmax(probs, dim=1)
        dice_coef = compute_dice_coef(prediction, mask).cpu().numpy()
        output = valid_postprocess(prediction.cpu().numpy(), image)
        
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
        
