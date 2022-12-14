from torch.utils.data import Dataset
from PIL import Image
from utils.mask_functions import rle2mask
import json
import cv2
import numpy as np
import torchvision.transforms as transforms
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensorV2

import os
import pdb

ALPHA = 0.5
BETA = 1 - ALPHA

class Pneumothorax(Dataset):
    def __init__(self, root_image_path, root_label_path, transform=None):
        self.root_image_path = root_image_path
        self.root_label_path = root_label_path
        self.transform = transform
        
        with open(self.root_label_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with open(self.root_label_path, 'r') as f:
            self.data = json.load(f)
            
        data_keys = list(self.data.keys())
        image_id = data_keys[idx]
        
        image_path = os.path.join(self.root_image_path, "{}.png".format(image_id))
        # PIL
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        
        all_masks = np.zeros((height, width))
        if self.data[image_id][0] != "-1":
            # check = True
            for annotation in self.data[image_id]:
                mask = rle2mask(annotation, width, height)
                all_masks += mask    
        
        # convert 0-255 to 0-1
        all_masks = all_masks / 255.0
        all_masks = (all_masks >= 1.0).astype('float32') # for overlap cases
        
        augmented = self.transform(image=image, mask=all_masks)
        image = augmented['image']
        all_masks = augmented['mask'].unsqueeze(0)
        
        
        sample = {'image': image, 'mask': all_masks}
        
        return sample
    
    def visualize(self, idx):
        data_dict = self.__getitem__(idx)
        image = data_dict['image']
        mask = data_dict['mask']
        mask = np.repeat(mask[..., np.newaxis], 3, axis=-1).astype(np.uint8)        

        blend_image = cv2.addWeighted(image, ALPHA, mask, BETA, 0.0)
        
        return blend_image

class EvalPneumothorax(Dataset):
    def __init__(self, root_image_path, root_label_path, transform=None):
        self.root_image_path = root_image_path
        self.root_label_path = root_label_path
        self.transform = transform
        
        with open(self.root_label_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with open(self.root_label_path, 'r') as f:
            self.data = json.load(f)
            
        data_keys = list(self.data.keys())
        image_id = data_keys[idx]
        
        image_path = os.path.join(self.root_image_path, "{}.png".format(image_id))
        # PIL
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        
        all_masks = np.zeros((height, width))
        if self.data[image_id][0] != "-1":
            # check = True
            for annotation in self.data[image_id]:
                mask = rle2mask(annotation, width, height)
                all_masks += mask    
        
        # convert 0-255 to 0-1
        all_masks = all_masks / 255.0
        all_masks = (all_masks >= 1.0).astype('float32') # for overlap cases
        
        augmented = self.transform(image=image)
        image = augmented['image']
        
        test_transform = Compose([
            ToTensorV2(),
        ])
        augmented_mask = test_transform(image=all_masks)
        all_masks = augmented_mask['image']
        
        sample = {'image': image, 'mask': all_masks}
        
        return sample
    
if __name__ == "__main__":
    root_image_path = "../dataset/pngs/balanced_images/test"
    root_label_path = "../dataset/annotations/test.json"
    list_transforms = []
    list_transforms.extend(
        [
            Resize(1024, 1024),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
            ToTensorV2(),
        ]
    )

    transform = Compose(list_transforms)
    dataloader = Pneumothorax(root_image_path, root_label_path, transform=transform)
    
    for idx in  range(len(dataloader)):
        data_dict = dataloader[idx]
        image, mask = data_dict['image'], data_dict['mask']
