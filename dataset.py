from torch.utils.data import Dataset
from utils.mask_functions import rle2mask, run_length_decode, getMaskAndImg
import json
import cv2
import numpy as np
from albumentations import (Normalize, Resize, Compose)
from albumentations.pytorch import ToTensorV2
import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import os
import pdb
from pathlib import Path
from tqdm import tqdm

ALPHA = 0.5
BETA = 1 - ALPHA

class PneumothoraxImagesPair(Dataset):
    def __init__(self, root_image_path, root_label_path, transform=None):
        self.root_image_path = root_image_path
        self.root_label_path = root_label_path
        self.transform = transform
        
        self.image_paths = os.listdir(self.root_image_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_image_path, self.image_paths[idx])
        mask_path = os.path.join(self.root_label_path, self.image_paths[idx])
        # PIL
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # if "1.2.276.0.7230010.3.1.4.8323329.300.1517875162.258081" in self.image_paths[idx]:
        #     pdb.set_trace()
        mask = cv2.imread(mask_path, 0)
        mask = mask / 255.0
        
        mask = (mask >= 1.0).astype('float32') # for overlap cases

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # no augmentation
            image = torch.from_numpy(image).mT.type(torch.FloatTensor)
            mask = torch.from_numpy(mask).unsqueeze(0)
        
        sample = {'image': image, 'mask': mask}
        
        return sample

class PneumothoraxDataFrame(Dataset):
    def __init__(self, dataframe, fnames, transform=None, preprocessing=None):
        self.dataframe = dataframe
        self.fnames = fnames
        self.transform = transform
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        image_id = self.fnames[idx]
        indice_inDataFrame = self.dataframe.index[self.dataframe['UID'] == image_id].tolist()[0]
        image, mask = getMaskAndImg(self.dataframe, indice_inDataFrame) # img, mask are arrays
        augmentedData = self.transform(image = image, mask = mask)
        image = augmentedData['image']
        mask = augmentedData['mask']
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask[..., np.newaxis])
            image, mask = sample['image'], sample['mask']
        
        sample = {'image': image, 'mask': mask}
        
        return sample

def sampledDataset(df, fold, train_transform, test_transform, preprocessing):
    df_positiveCase = df[df["Pneumothorax"] == 1]
    df_negativeCase = df[df["Pneumothorax"] == 0]
    
    if len(df_negativeCase) != 0: # train on positive cases only
        df_negativeCase = df_negativeCase.sample(len(df_positiveCase) + 1000, random_state = 2019) # resample negative samples
    else:
        print("Training on positive images only")

    newSub_Dataframe = pd.concat([df_positiveCase, df_negativeCase])
    df_split = newSub_Dataframe
        
    kfold = StratifiedKFold(n_splits = 5, random_state = 43, shuffle = True)
    # k_fold validation
    train_idx, val_idx = list(kfold.split(X = df_split["UID"], y = df_split["Pneumothorax"]))[fold]
    train_df, val_df = df_split.iloc[train_idx], df_split.iloc[val_idx]
    f_names_datatrain = train_df.iloc[:, 0].values.tolist()
    f_names_dataval = val_df.iloc[:, 0].values.tolist()
   
    train_dataset = PneumothoraxDataFrame(train_df, f_names_datatrain, train_transform, preprocessing)
    val_dataset = PneumothoraxDataFrame(val_df, f_names_dataval, test_transform, preprocessing)
    return train_dataset, val_dataset

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
        # pdb.set_trace()
        all_masks = np.zeros((height, width))
        if self.data[image_id][0] != "-1":
            # check = True
            for annotation in self.data[image_id][0].split(","):
                mask = run_length_decode(annotation, width, height)
                all_masks += mask    

        if "1.2.276.0.7230010.3.1.4.8323329.10603.1517875224.506264" in image_id:
            Path("./visualize/").mkdir(parents=True, exist_ok=True)
            all_masks *= 255
            cv2.imwrite(f"./visualize/mask_{image_id}.png", all_masks)
        # convert 0-255 to 0-1
        # all_masks = all_masks / 255.0
        all_masks = (all_masks >= 1.0).astype('float32') # for overlap cases
        # pdb.set_trace()
        
        augmented = self.transform(image=image, mask=all_masks)
        
        image = augmented['image']
        all_masks = augmented['mask'].unsqueeze(0)
        
        assert len(np.unique(all_masks) == 2)
                
        sample = {'image': image, 'mask': all_masks}
        
        return sample
    
    def visualize(self, idx):
        data_dict = self.__getitem__(idx)
        data_keys = list(self.data.keys())
        image_id = data_keys[idx]
        image = data_dict['image']
        mask = data_dict['mask']
        mask *= 255  
        
        mask = np.repeat(mask[..., np.newaxis], 3, axis=-1).astype(np.uint8)      
        blend_image = cv2.addWeighted(image, ALPHA, mask, BETA, 0.0)
        
        
        if "1.2.276.0.7230010.3.1.4.8323329.3678.1517875178.953520" in image_id:
            # pdb.set_trace()
            Path("./visualize/").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(f"./visualize/blend_{image_id}.png", blend_image)
        
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
                mask = run_length_decode(annotation, width, height)
                all_masks += mask    
        
        # convert 0-255 to 0-1
        # all_masks = all_masks / 255.0
        all_masks = (all_masks >= 1.0).astype('float32') # for overlap cases
                
        augmented = self.transform(image=image, mask=all_masks)
        image = augmented['image']
        all_masks = augmented['mask'].unsqueeze(0)
        assert len(np.unique(all_masks) == 2)
        # test_transform = Compose([
        #     ToTensorV2(),
        # ])
        # augmented_mask = test_transform(image=all_masks)
        # all_masks = augmented_mask['image']
        # all_masks = torch.from_numpy(all_masks)
                
        sample = {'image': image, 'mask': all_masks}
        
        return sample
    
class EvalPneumothoraxImagesPair(Dataset):
    def __init__(self, root_image_path, root_label_path, transform=None):
        self.root_image_path = root_image_path
        self.root_label_path = root_label_path
        self.transform = transform
        
        self.image_paths = os.listdir(self.root_image_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_image_path, self.image_paths[idx])
        mask_path = os.path.join(self.root_label_path, self.image_paths[idx])
        # PIL
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # if "CHNCXR_0015_0.png" in self.image_paths[idx]:
        #     pdb.set_trace()
        mask = cv2.imread(mask_path, 0)
        mask = mask / 255.0
        
        mask = (mask >= 1.0).astype('float32') # for overlap cases
                
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        # mask = torch.from_numpy(mask)
        sample = {'image': image, 'mask': mask}
        
        return sample
    
if __name__ == "__main__":
    root_image_path = "../dataset/pngs/positive_images/train"
    root_label_path = "../dataset/annotations/positive_samples/train.json"
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
    
    for idx in  tqdm(range(len(dataloader))):
        data_dict = dataloader[idx]
        image, mask = data_dict['image'], data_dict['mask']
        # blend = dataloader.visualize(idx)
       
        
        