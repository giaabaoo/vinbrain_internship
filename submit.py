import os
import cv2
import yaml
import torch
import numpy as np
import pandas as pd
import pydicom
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from pydicom.pixel_data_handlers.util import apply_voi_lut
import segmentation_models_pytorch as smp
import pdb
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensorV2
from utils.mask_functions import mask2rle
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser("Parsing arguments", add_help=False)
    parser.add_argument("--config", default="./configs/config.yaml", type=str)

    return parser

class TestDataset(Dataset):
    def __init__(self, root_image_path, transform=None):
        self.root_image_path = root_image_path
        self.transform = transform
        
        self.paths = os.listdir(self.root_image_path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_image_path, self.paths[idx])
        image = self.process_dicom(image_path)
        image = np.repeat(image[..., np.newaxis], 3, axis=-1).astype(np.uint8)        
      
        augmented = self.transform(image=image)
        image = augmented['image']
        
        return image

    def process_dicom(self, file_path):
        dicom = pydicom.read_file(file_path)
        
        data = apply_voi_lut(dicom.pixel_array, dicom)
        
        # Correct image inversion.
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data
                
        return data

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

def run_length_encode(component):
    component = component.T.flatten()
    start = np.where(component[1:] > component[:-1])[0]+1
    end = np.where(component[:-1] > component[1:])[0]+1
    length = end-start
    rle = []
    for i in range(len(length)):
        if i == 0:
            rle.extend([start[0], length[0]])
        else:
            rle.extend([start[i]-end[i-1], length[i]])
    rle = ' '.join([str(r) for r in rle])
    return rle
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pneumothorax evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    # load yaml file
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
    
    testing_data = TestDataset("/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/dicom/stage_2_images", transform=transform)    
    
    testing_loader = DataLoader(testing_data, batch_size=config['batch_size'], shuffle=True)
    
    if config['backbone'] == "resnet34":
        model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
    elif config['backbone'] == "efficientnet-b4":
        model = smp.Unet("efficientnet-b4", encoder_weights="imagenet", activation=None)
    weights_path = config['save_checkpoint']
    model.load_state_dict(torch.load(weights_path)['model_state_dict'])
    print("Using weights", weights_path.split("/")[-1])
    model.to(config['device'])
    sample_submission_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/annotations/stage_2_sample_submission.csv"
    df = pd.read_csv(sample_submission_path)

    encoded_pixels = []
    for i, batch in enumerate(tqdm(testing_loader)):
        preds = torch.sigmoid(model(batch.to(config['device'])))
        preds = preds.detach().cpu().numpy()[:, 0, :, :] # (batch_size, 1, size, size) -> (batch_size, size, size)
        for probability in preds:
            if probability.shape != (1024, 1024):
                probability = cv2.resize(probability, dsize=(1024, 1024), interpolation=cv2.INTER_NEAREST)
            predict, num_predict = post_process(probability, 0.5, 3000)
            
            if num_predict == 0:
                encoded_pixels.append('-1')
            else:
                # pdb.set_trace()
                r = run_length_encode(predict)
                encoded_pixels.append(r)
    df['EncodedPixels'] = encoded_pixels
    df.to_csv('submission.csv', columns=['ImageId', 'EncodedPixels'], index=False)