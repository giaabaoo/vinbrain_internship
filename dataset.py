from torch.utils.data import Dataset
from PIL import Image
from utils.mask_functions import rle2mask
import json
import cv2
import numpy as np
import torchvision.transforms as transforms

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
        
        # check = False
        
        if self.data[image_id][0] != "-1":
            # check = True
            for annotation in self.data[image_id]:
                mask = rle2mask(annotation, width, height)
                all_masks += mask    
                
        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)
            all_masks = Image.fromarray(all_masks)
            all_masks = self.transform(all_masks)

        sample = {'image': image, 'mask': all_masks}
        
        return sample
    
    def visualize(self, idx):
        data_dict = self.__getitem__(idx)
        image = data_dict['image']
        mask = data_dict['mask']
        mask = np.repeat(mask[..., np.newaxis], 3, axis=-1).astype(np.uint8)        

        blend_image = cv2.addWeighted(image, ALPHA, mask, BETA, 0.0)
        
        return blend_image
        
if __name__ == "__main__":
    root_image_path = "../dataset/pngs/original_images/test"
    root_label_path = "../dataset/annotations/test.json"
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    dataloader = Pneumothorax(root_image_path, root_label_path, transform=transform)
    
    for idx in  range(len(dataloader)):
        data_dict = dataloader[idx]
        image, mask = data_dict['image'], data_dict['mask']
        
        # if check:
        #     visualize_image = dataloader.visualize(idx)
        #     cv2.imwrite("{}_test_dataloader.png".format(image_id), visualize_image)
        #     pdb.set_trace()
        #     print(image_id)
