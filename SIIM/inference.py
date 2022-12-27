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
from unet import UNetWithResnet50Encoder, UNetWithResNext101Encoder
import pydicom
import os
from PIL import Image, ImageDraw
import albumentations as A
import shutil

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_args_parser():
    parser = argparse.ArgumentParser("Parsing arguments", add_help=False)
    parser.add_argument("--config", default="./configs/config.yaml", type=str)

    return parser

def post_process(probability, threshold, min_size):
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((1024, 1024), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num

def postprocess(x, image):
    height, width, _ = image.shape
    x = np.transpose(x, (1, 2, 0)) 
    x = np.vectorize(lambda value: 0 if value < 0.5 else 255)(x)
    x = cv2.resize(x, (width, height), 0, 0, interpolation = cv2.INTER_NEAREST)
    
    return x


def visualize(image, mask):
    mask = np.repeat(mask[..., np.newaxis], 3, axis=-1).astype(np.uint8)
     
    blend_image = cv2.addWeighted(image, 0.5, mask, 0.5, 0.0)
    
    return blend_image
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')
def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """   
    _transform = []
    if preprocessing_fn:
        _transform.append(A.Lambda(image=preprocessing_fn))
    _transform.append(A.Lambda(image=to_tensor, mask=to_tensor))
        
    return A.Compose(_transform)
if __name__ == "__main__":
    Path("./results/").mkdir(parents=True, exist_ok = True)
    
    parser = argparse.ArgumentParser("Pneumothorax inference script", parents=[get_args_parser()])
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(config)
    list_transforms = []
    list_transforms.extend(
        [
            Resize(config['image_height'], config['image_width']),
            # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
            # ToTensor(),
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
    print("Inferencing using ", weights_path)
    model.load_state_dict(torch.load(weights_path)['model_state_dict'])
    model.to(config['device'])
                
    ### Inference on image
    # image_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/chest-xray/split/test/MCUCXR_0369_1.png"
    image_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/dicom/stage_2_train/1.2.276.0.7230010.3.1.4.8323329.11969.1517875236.686733.dcm"
    
    if ".dcm" in image_path:
        ds = pydicom.dcmread(image_path)
        image = ds.pixel_array # img is 2-D matrix
        image = cv2.equalizeHist(image)
        image = np.stack((image, ) * 3, axis=-1)
    else:
        image = cv2.imread(image_path)
    model.eval()
    augmented = transform(image=image)
    image_transform = augmented['image']
    
    preprocess_input = None
    preprocessing = get_preprocessing(preprocess_input)
    image_transform = torch.from_numpy(preprocessing(image=image_transform)['image']).unsqueeze(0).to(config['device'])
    
    output = model(image_transform)
    
    output = torch.sigmoid(output)
    
    output = output.squeeze(0).detach().cpu().numpy()

    output = postprocess(output, image)
    
    image_name = image_path.split("/")[-1].replace(".dcm", ".png")  
    ground_truth_path = os.path.join("/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/nguyen/pngs/masks", image_name)
    gt = cv2.imread(ground_truth_path)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    gt = Image.fromarray(gt)
    output_pil = Image.fromarray(output).resize((256, 256)) 

    d1 = ImageDraw.Draw(gt)
    d1.text((100, 220), "Ground truth", fill = (255))   

    d2 = ImageDraw.Draw(output_pil)
    d2.text((100, 220), "Prediction", fill = (255))   

    get_concat_h(output_pil, gt).save(f'results/hmasks.png')
    
    blend = visualize(image, output)
    cv2.imwrite("results/blend.png", blend)
