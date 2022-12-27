import numpy as np
import cv2
import yaml
import torch
import pdb
import segmentation_models_pytorch as smp
from albumentations import (Normalize, Resize, Compose)
from albumentations.pytorch import ToTensorV2
from unet import UNetWithResnet50Encoder, UNetWithResNext101Encoder

from pathlib import Path
import argparse
from model import UNET

def get_args_parser():
    parser = argparse.ArgumentParser("Parsing arguments", add_help=False)
    parser.add_argument("--config", default="./configs/config.yaml", type=str)

    return parser

def post_process(probability, threshold, min_size):
    # pdb.set_trace()
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    # pdb.set_trace()
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((1024, 1024), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num
def postprocess(x, image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    # pdb.set_trace()
    x = np.transpose(x, (1, 2, 0)) 
    x = np.vectorize(lambda value: 0 if value < 0.5 else 255)(x)
    x = cv2.resize(x, (width, height), 0, 0, interpolation = cv2.INTER_NEAREST)
    
    return x


def visualize(image_path, mask):
    image = cv2.imread(image_path)
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
        
    list_transforms = []
    list_transforms.extend(
        [
            Resize(config['image_height'], config['image_width']),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
            ToTensorV2(),
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
                
    ### Inference on image
    image_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/chest-xray/split/test/MCUCXR_0173_1.png"
    image = cv2.imread(image_path)
    # image = Image.fromarray(image)
    # pdb.set_trace()
    model.eval()
    
    augmented = transform(image=image)
    image = augmented['image'].unsqueeze(0).to(config['device'])
    output = model(image)
    output = torch.sigmoid(output)
    output = output.squeeze(0).detach().cpu().numpy()

    output = postprocess(output, image_path)
    
    cv2.imwrite("results/output_chest.png", output)
    blend = visualize(image_path, output)
    cv2.imwrite("results/blend_chest.png", blend)
    
    mask_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/chest-xray/split/test_mask/MCUCXR_0173_1.png"
    image_name = mask_path.split("/")[-1]
    cv2.imwrite(f"results/{image_name}", cv2.imread(mask_path))
    #comment