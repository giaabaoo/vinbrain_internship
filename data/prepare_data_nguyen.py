import json
import pandas as pd
import glob
from pathlib import Path
import cv2
from tqdm import tqdm
import pdb
import os
import shutil

if __name__ == "__main__":
    labels_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/nguyen/masks"
    images_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/nguyen/train"
    
    train_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/nguyen/split/train"
    test_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/nguyen/split/test"
    mask_train_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/nguyen/split/train_mask"
    mask_test_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/nguyen/split/test_mask"
    
    Path(train_path).mkdir(parents=True, exist_ok=True)
    Path(test_path).mkdir(parents=True, exist_ok=True)
    Path(mask_train_path).mkdir(parents=True, exist_ok=True)
    Path(mask_test_path).mkdir(parents=True, exist_ok=True)

    split_idx = int(len(os.listdir(images_path)) * 0.8)
    
    data_keys = list(os.listdir(images_path))
    train_keys = data_keys[:split_idx]
    test_keys = data_keys[split_idx:]
    
    print("Preparing training folders: ")
    for image in tqdm(train_keys):
        shutil.copy(os.path.join(images_path, image), os.path.join(train_path, image))
        shutil.copy(os.path.join(labels_path, image), os.path.join(mask_train_path, image))
    
    print("Preparing testing folders: ")
    for image in tqdm(test_keys):
        shutil.copy(os.path.join(images_path, image), os.path.join(test_path, image))
        shutil.copy(os.path.join(labels_path, image), os.path.join(mask_test_path, image))
        