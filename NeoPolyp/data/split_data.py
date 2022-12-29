from pathlib import Path
from tqdm import tqdm
import os
import shutil
import cv2
import numpy as np

if __name__ == "__main__":
    labels_path = "/home/dhgbao/VinBrain/assignments/dataset/NeoPolyp/train_gt/train_gt"
    images_path = "/home/dhgbao/VinBrain/assignments/dataset/NeoPolyp/train/train"
    
    train_path = "/home/dhgbao/VinBrain/assignments/dataset/NeoPolyp/split/train"
    valid_path = "/home/dhgbao/VinBrain/assignments/dataset/NeoPolyp/split/valid"
    mask_train_path = "/home/dhgbao/VinBrain/assignments/dataset/NeoPolyp/split/train_mask"
    mask_valid_path = "/home/dhgbao/VinBrain/assignments/dataset/NeoPolyp/split/valid_mask"
    
    Path(train_path).mkdir(parents=True, exist_ok=True)
    Path(valid_path).mkdir(parents=True, exist_ok=True)
    Path(mask_train_path).mkdir(parents=True, exist_ok=True)
    Path(mask_valid_path).mkdir(parents=True, exist_ok=True)

    split_idx = int(len(os.listdir(images_path)) * 0.8)
    
    data_keys = list(os.listdir(images_path))
    train_keys = data_keys[:split_idx]
    valid_keys = data_keys[split_idx:]
    
    print("Preparing training folders: ")
    for image_name in tqdm(train_keys):
        image_path =os.path.join(images_path, image_name)
        shutil.copy(image_path, os.path.join(train_path, image_name))
        shutil.copy(os.path.join(labels_path, image_name), os.path.join(mask_train_path, image_name))
    
    print("Preparing validing folders: ")
    for image_name in tqdm(valid_keys):
        image_path =os.path.join(images_path, image_name)
        shutil.copy(image_path, os.path.join(valid_path, image_name))
        shutil.copy(os.path.join(labels_path, image_name), os.path.join(mask_valid_path, image_name))
        
    print("Preparing testing folders: ")
    shutil.copytree("/home/dhgbao/VinBrain/assignments/dataset/NeoPolyp/test/test", "/home/dhgbao/VinBrain/assignments/dataset/NeoPolyp/split/test")
    
        