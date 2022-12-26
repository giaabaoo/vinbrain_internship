import numpy as np
import cv2
import yaml
import torch
import pdb
import segmentation_models_pytorch as smp
from albumentations import (Normalize, Resize, Compose)
from albumentations.pytorch import ToTensor
from pathlib import Path
import argparse
from model import UNET
import pydicom
from unet import UNetWithResnet50Encoder, UNetWithResNext101Encoder
import os
from tqdm import tqdm
import shutil
from PIL import Image, ImageDraw

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_args_parser():
    parser = argparse.ArgumentParser("Parsing arguments", add_help=False)
    parser.add_argument("--config", default="./configs/config.yaml", type=str)

    return parser

def postprocess(x, image):
    height, width, _ = image.shape
    x = np.transpose(x, (1, 2, 0)) 
    x = np.vectorize(lambda value: 0 if value < 0.5 else 255)(x)
    x = cv2.resize(x, (256, 256), 0, 0, interpolation = cv2.INTER_NEAREST)
    
    if len(np.unique(x)) == 2:
        return x, True
    
    return x, False

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

def visualize(image, mask):
    mask = np.repeat(mask[..., np.newaxis], 3, axis=-1).astype(np.uint8)
     
    blend_image = cv2.addWeighted(image, 0.5, mask, 0.5, 0.0)
    
    return blend_image

if __name__ == "__main__":
    # weights_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/vinbrain_internship/checkpoints/model-ckpt-best.pt"
    Path("./results/").mkdir(parents=True, exist_ok = True)
    
    parser = argparse.ArgumentParser("Pneumothorax inference script", parents=[get_args_parser()])
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(config)
    list_transforms = []
    list_transforms.extend(
        [
            Resize(256, 256),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
            ToTensor(),
        ]
    )

    transform = Compose(list_transforms)
    
    if config['backbone'] == 'None':
        model = UNET(in_channels=3, out_channels=1)
    elif config['backbone'] == 'torchvision.resnet50':
        print("Using torchvision")
        model = UNetWithResnet50Encoder()
    elif config['backbone'] == 'torchvision.resnext101':
        model = UNetWithResNext101Encoder()
    else:
        model = smp.Unet(config['backbone'], encoder_weights="imagenet", activation=None)


    weights_path = config['save_checkpoint']
    model.load_state_dict(torch.load(weights_path)['model_state_dict'])
    model.to(config['device'])
    
    Path("./results/hmasks/").mkdir(parents=True, exist_ok=True)
    Path("./results/blend/").mkdir(parents=True, exist_ok=True)
                
    ### Inference on all images
    image_paths = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/dicom/stage_2_train/"
    
    for image_path in tqdm(os.listdir(image_paths)):
        check_path = os.path.join("./results/blend", image_path.replace(".dcm", ".png"))
        if not Path(check_path).exists():
            if ".dcm" in image_path:
                ds = pydicom.dcmread(os.path.join(image_paths, image_path))
                image = ds.pixel_array # img is 2-D matrix
                image = np.stack((image, ) * 3, axis=-1)
            else:
                image = cv2.imread(image_path)
        
            model.eval()
            
            augmented = transform(image=image)
            image_transform = augmented['image'].unsqueeze(0).to(config['device'])
            output = model(image_transform)
            output = torch.sigmoid(output)
            output = output.squeeze(0).detach().cpu().numpy()

            output, have_mask = postprocess(output, image)
            
            if have_mask:
                image_name = image_path.split("/")[-1].replace(".dcm", ".png")            
                ground_truth_path = os.path.join("/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/nguyen/masks", image_name)
                
                try:
                    gt = cv2.imread(ground_truth_path)
                    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
                    gt = Image.fromarray(gt)
                    output_pil = Image.fromarray(output)         
                    
                    d1 = ImageDraw.Draw(gt)
                    d1.text((100, 220), "Ground truth", fill = (255))   
                    
                    d2 = ImageDraw.Draw(output_pil)
                    d2.text((100, 220), "Prediction", fill = (255))   
                    
                    get_concat_h(output_pil, gt).save(f'results/hmasks/{image_name}')

                    image_resize = cv2.resize(image, (256, 256))
                    blend = visualize(image_resize, output)
                    cv2.imwrite(f"results/blend/{image_name}", blend)
                except:
                    continue
                
        else:
            continue
                
            
            
