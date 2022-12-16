import numpy as np
import cv2
import yaml
import torch
import pdb
import segmentation_models_pytorch as smp
from albumentations import (Normalize, Resize, Compose)
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser("Parsing arguments", add_help=False)
    parser.add_argument("--config", default="./configs/config.yaml", type=str)

    return parser

def postprocess(config, x):
    # pdb.set_trace()
    x = np.transpose(x, (1, 2, 0)) 
    x = np.vectorize(lambda value: 0 if value < 0.5 else 255)(x)
    x = cv2.resize(x, (1024, 1024), 0, 0, interpolation = cv2.INTER_NEAREST)
    
    return x

def visualize(image_path, mask):
    # pdb.set_trace()
    image = cv2.imread(image_path)
    mask = np.repeat(mask[..., np.newaxis], 3, axis=-1).astype(np.uint8)
     
    blend_image = cv2.addWeighted(image, 0.5, mask, 0.5, 0.0)
    
    return blend_image

if __name__ == "__main__":
    weights_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/vinbrain_internship/checkpoints/model-ckpt-best-focal.pt"
    Path("./results/").mkdir(parents=True, exist_ok = True)
    
    parser = argparse.ArgumentParser("Pneumothorax inference script", parents=[get_args_parser()])
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    list_transforms = []
    list_transforms.extend(
        [
            Resize(config['image_height'], config['image_width']),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
            ToTensorV2(),
        ]
    )

    transform = Compose(list_transforms)
    
    if config['backbone'] == "resnet34":
        model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
    elif config['backbone'] == "efficientnet-b4":
        model = smp.Unet("efficientnet-b4", encoder_weights="imagenet", activation=None)
        
    model.load_state_dict(torch.load(weights_path)['model_state_dict'])
    model.to(config['device'])
                
    ### Inference on image
    image_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/pngs/original_images/train/1.2.276.0.7230010.3.1.4.8323329.300.1517875162.258081.png"
    image = cv2.imread(image_path)
    # image = Image.fromarray(image)

    model.eval()
    augmented = transform(image=image)
    image = augmented['image'].unsqueeze(0).to(config['device'])
    output = model(image)
    output = torch.sigmoid(output)
    output = output.squeeze(0).detach().cpu().numpy()

    output = postprocess(config, output)
    
    cv2.imwrite("results/output.png", output)
    blend = visualize(image_path, output)
    cv2.imwrite("results/blend.png", blend)
    #comment