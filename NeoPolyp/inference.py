import numpy as np
import cv2
import yaml
import torch
import pdb
from albumentations import (Normalize, Resize, Compose)
from albumentations.pytorch import ToTensor
from pathlib import Path
import argparse
import pydicom
import os
from PIL import Image, ImageDraw, ImageFont
import albumentations as A
from utils.train_utils import prepare_architecture
from utils.helper import get_concat_h, get_args_parser, postprocess, visualize
import shutil

if __name__ == "__main__":
    Path("./results/").mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(
        "NeoPolyp inference script", parents=[get_args_parser()])
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(config)
    
    transform = Compose([
        Resize(config['image_height'], config['image_width']),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
        ToTensor(),
    ])

    model = prepare_architecture(config)

    weights_path = config['save_checkpoint']
    print("Inferencing using ", weights_path)
    model.load_state_dict(torch.load(weights_path)['model_state_dict'])
    model.to(config['device'])

    # Inference on image
    # image_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/chest-xray/split/test/MCUCXR_0369_1.png"
    image_path = "/home/dhgbao/VinBrain/assignments/vinbrain_internship/NeoPolyp/example_data/train/be9800e6c6f4f219688c7e39b3608a3d.jpeg"
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    model.eval()
    
    augmented = transform(image=image)
    
    image_transform = augmented['image']
    if not config['transform']:
        image_transform = image_transform.type(torch.FloatTensor)
    image_transform = image_transform.unsqueeze(0).to(config['device'])

    output = model(image_transform)

    probs = torch.softmax(output, dim=1)
    prediction = torch.argmax(probs, dim=1).cpu().numpy()

    output = postprocess(prediction, image)
    output_pil = Image.fromarray(output)
    image_name = image_path.split("/")[-1]
    ground_truth_path = os.path.join("/home/dhgbao/VinBrain/assignments/vinbrain_internship/NeoPolyp/example_data/train_mask", image_name)
    
    gt = cv2.imread(ground_truth_path)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    gt = Image.fromarray(gt)

    font = ImageFont.truetype("LiberationSans-Regular.ttf", 30)
    d1 = ImageDraw.Draw(gt)
    d1.text((width/2 - 100, height-100), "Ground truth", fill=(255, 255, 255), font=font)

    d2 = ImageDraw.Draw(output_pil)
    d2.text((width/2 - 100, height-100), "Prediction", fill=(255, 255, 255), font=font)

    get_concat_h(output_pil, gt).save(f'results/hmasks.png')

    blend = visualize(image, output)
    cv2.imwrite("results/blend.png", blend)
