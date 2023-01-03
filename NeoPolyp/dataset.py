import os
from torch.utils.data import Dataset
import cv2
from albumentations import (Normalize, Resize, Compose)
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import matplotlib
import pdb

ALPHA = 0.8
BETA = 1-ALPHA


class NeoPolyp(Dataset):
    def __init__(self, root_image_path, root_label_path, transform=None):
        self.root_image_path = root_image_path
        self.root_label_path = root_label_path
        self.transform = transform
        self.classes = ['background', 'neoplastic', 'non-neoplastic']
        self.LABEL_TO_COLOR = {0:[0,0,0], 1:[255,0,0], 2:[0,255,0]}

        self.image_names = os.listdir(root_image_path)
        self.label_names = os.listdir(root_label_path)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        label_name = self.label_names[idx]
        image_path = os.path.join(self.root_image_path, image_name)
        mask_path = os.path.join(self.root_label_path, label_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        mask = self.mask_to_class(mask)
        # mask = np.stack((mask, ) * 3, axis=-1).T
        sample = {'image_name': image_name, 'image': image, 'mask': mask}
        
        return sample
    
    def mask_to_class(self, mask):
        binary_mask = np.array(mask)
        binary_mask = cv2.erode(binary_mask, np.ones((5, 5), np.uint8)) 
        binary_mask[:,:,2] = 0
        binary_mask = (binary_mask != 0).astype(np.uint8)
        binary_mask *= 255
        
        # r = np.array(binary_mask[:,:,0])
        # g = np.array(binary_mask[:,:,1])
        # b = np.array(binary_mask[:,:,2])
        
        # cv2.imwrite("./visualization/red.png", r)
        # cv2.imwrite("./visualization/green.png", g)
        # cv2.imwrite("./visualization/blue.png", b)
        
        # convert colors to "flat" labels
        rgb = np.array(binary_mask)
        output_mask = np.zeros((rgb.shape[0], rgb.shape[1]))

        for k,v in self.LABEL_TO_COLOR.items():
            output_mask[np.all(rgb==v, axis=2)] = k
        
        output_mask = torch.from_numpy(output_mask)
        output_mask = output_mask.type(torch.int64)

        return output_mask
        
    def visualize(self, idx):
        sample = self.__getitem__(idx)
        image_name = sample['image_name']
        image = sample['image'].numpy().T 
        mask = sample['mask'].numpy()
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        blend_image = cv2.addWeighted(image, ALPHA, mask, BETA, 0.0)

        Path("./visualization/").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f"./visualization/blend_{image_name}", blend_image)
        
        
class EvalNeoPolyp(Dataset):
    def __init__(self, root_image_path, root_label_path, transform=None):
        self.root_image_path = root_image_path
        self.root_label_path = root_label_path
        self.transform = transform
        self.classes = ['background', 'neoplastic', 'non-neoplastic']
        self.LABEL_TO_COLOR = {0:[0,0,0], 1:[255,0,0], 2:[0,255,0]}

        self.image_names = os.listdir(root_image_path)
        self.label_names = os.listdir(root_label_path)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        label_name = self.label_names[idx]
        image_path = os.path.join(self.root_image_path, image_name)
        mask_path = os.path.join(self.root_label_path, label_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        mask = self.mask_to_class(mask)
        sample = {'image_name': image_name, 'image': image, 'mask': mask}
        
        return sample
    
    def mask_to_class(self, mask):
        binary_mask = np.array(mask)
        binary_mask = cv2.erode(binary_mask, np.ones((5, 5), np.uint8)) 
        binary_mask[:,:,2] = 0
        binary_mask = (binary_mask != 0).astype(np.uint8)
        binary_mask *= 255
        
        # r = np.array(binary_mask[:,:,0])
        # g = np.array(binary_mask[:,:,1])
        # b = np.array(binary_mask[:,:,2])
        
        # cv2.imwrite("./visualization/red.png", r)
        # cv2.imwrite("./visualization/green.png", g)
        # cv2.imwrite("./visualization/blue.png", b)
        
        # convert colors to "flat" labels
        rgb = np.array(binary_mask)
        output_mask = np.zeros((rgb.shape[0], rgb.shape[1]))

        for k,v in self.LABEL_TO_COLOR.items():
            output_mask[np.all(rgb==v, axis=2)] = k
        
        output_mask = torch.from_numpy(output_mask)
        output_mask = output_mask.type(torch.int64)

        return output_mask


if __name__ == "__main__":
    root_image_path = "/home/dhgbao/VinBrain/assignments/vinbrain_internship/NeoPolyp/example_data/train"
    root_label_path = "/home/dhgbao/VinBrain/assignments/vinbrain_internship/NeoPolyp/example_data/train_mask"
    # root_image_path = "/home/dhgbao/VinBrain/assignments/others_code/unet-multiclass-pytorch/data/images_train/images"
    # root_label_path = "/home/dhgbao/VinBrain/assignments/others_code/unet-multiclass-pytorch/data/images_train/masks"

    transform = Compose([
        Resize(512, 512, interpolation=cv2.INTER_NEAREST),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
        ToTensorV2(),
    ])

    dataloader = NeoPolyp(
        root_image_path, root_label_path, transform=transform)
    for idx in tqdm(range(len(dataloader))):
        sample = dataloader[idx]
        image_name, image, mask = sample['image_name'], sample['image'], sample['mask']
        # dataloader.visualize(idx)
        # if len(mask.unique()) > 2:
        #     pdb.set_trace()
