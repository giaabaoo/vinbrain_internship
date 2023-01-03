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
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from pydicom.pixel_data_handlers.util import apply_voi_lut
import segmentation_models_pytorch as smp
import pdb
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensor
from utils.mask_functions import mask2rle
from unet import UNetWithResnet50Encoder, UNetWithResNext101Encoder
import argparse
from model import UNET
import albumentations as A

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
    
class TestDatasetDataframe(Dataset):
    def __init__(self, root, df, transform, preprocessing):
        self.root = root
        self.fnames = list(df["ImageId"])
        self.transform = transform
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname + ".dcm")
        ds = pydicom.dcmread(path)
        image = ds.pixel_array
        image = cv2.equalizeHist(image)
        image = np.stack((image, ) * 3, axis=-1)
        
        images = self.transform(image = image)["image"]
        if self.preprocessing:
            images = self.preprocessing(image=image)['image']
            
        return images

    def __len__(self):
        return len(self.fnames)

def post_process(probability, threshold, min_size):
    # pdb.set_trace()
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
    parser = argparse.ArgumentParser("Pneumothorax evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    # load yaml file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(config)
    list_transforms = []
    list_transforms.extend(
        [
            Resize(config['image_height'], config['image_width']),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
            ToTensor(),
        ]
    )

    transform = Compose(list_transforms)
    
    sample_submission_path = "/home/dhgbao/VinBrain/assignments/dataset/siim/annotations/stage_2_sample_submission.csv"
    df = pd.read_csv(sample_submission_path)
    
    if config['backbone'] == 'None':
        model = UNET(in_channels=3, out_channels=1)
    elif config['backbone'] == 'torchvision.resnet50':
        preprocess_input = None
        print("Using torchvision")
        model = UNetWithResnet50Encoder()
    elif config['backbone'] == 'torchvision.resnext101':
        preprocess_input = None
        model = UNetWithResNext101Encoder()
    else:
        model = smp.Unet(config['backbone'], encoder_weights="imagenet", activation=None)
        preprocess_input = get_preprocessing_fn(config['backbone'], pretrained='imagenet')
    
    if config['annotation_type'] == 'dataframe':
        print("Dataframe")
        testing_data = TestDatasetDataframe(config['root_test_path'], df, transform, get_preprocessing(preprocess_input))
    else:
        testing_data = TestDataset("/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/dicom/stage_2_test", transform=transform)    
    testing_loader = DataLoader(testing_data, batch_size=2, shuffle=True)
    
    
         
    weights_path = config['save_checkpoint']
    model.load_state_dict(torch.load(weights_path)['model_state_dict'])
    print("Using weights", weights_path.split("/")[-1])
    model.to(config['device'])
    model.eval()
   
    encoded_pixels = []
    for i, batch in enumerate(tqdm(testing_loader)):
        preds = torch.sigmoid(model(batch.to(config['device'])))
        preds = preds.detach().cpu().numpy()[:, 0, :, :] # (batch_size, 1, size, size) -> (batch_size, size, size)
        for probability in preds:
            if probability.shape != (1024, 1024):
                probability = cv2.resize(probability, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
            predict, num_predict = post_process(probability, 0.5, 3500)
            
            if num_predict == 0:
                encoded_pixels.append('-1')
            else:
                r = run_length_encode(predict)
                encoded_pixels.append(r)
    df['EncodedPixels'] = encoded_pixels
    df.to_csv('submission.csv', columns=['ImageId', 'EncodedPixels'], index=False)