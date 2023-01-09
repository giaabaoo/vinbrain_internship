import os
import cv2
import random
import numpy as np
import shutil
from pathlib import Path
import pdb
from tqdm import tqdm

if __name__ == "__main__":
    train_folder = "/home/dhgbao/VinBrain/assignments/dataset/NeoPolyp/split/train"
    train_mask_folder = "/home/dhgbao/VinBrain/assignments/dataset/NeoPolyp/split/train_mask"
    train_aug_folder = "/home/dhgbao/VinBrain/assignments/dataset/NeoPolyp/split/train_aug_600"
    train_aug_mask_folder = "/home/dhgbao/VinBrain/assignments/dataset/NeoPolyp/split/train_aug_mask_600"
    
    Path(train_aug_folder).mkdir(parents=True, exist_ok=True)
    Path(train_aug_mask_folder).mkdir(parents=True, exist_ok=True)
    
    all_image_names = os.listdir(train_folder)
    k = len(all_image_names) * 60 // 100
    indicies = random.sample(range(len(all_image_names)), k)
    aug_image_names = [all_image_names[i] for i in indicies]
    
    for train_image1_name in tqdm(aug_image_names):
        train_image1_path = os.path.join(train_folder, train_image1_name)
        train_image1 = cv2.imread(train_image1_path)
        train_image1 = cv2.cvtColor(train_image1, cv2.COLOR_BGR2LAB)
        
        train_image2_name = random.choice(os.listdir(train_folder))
        train_image2_path = os.path.join(train_folder, train_image2_name)
        train_image2 = cv2.imread(train_image2_path)
        train_image2 = cv2.cvtColor(train_image2, cv2.COLOR_BGR2LAB)
        
        mean1 , std1  = train_image1.mean(axis=(0,1), keepdims=True), train_image1.std(axis=(0,1), keepdims=True)
        mean2, std2 = train_image2.mean(axis=(0,1), keepdims=True), train_image2.std(axis=(0,1), keepdims=True)
        
        image = np.uint8((train_image1-mean1)/std1*std2+mean2)
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        
        aug_image_name = train_image1_name.replace(".jpeg", "") +"_aug.jpeg"
        cv2.imwrite(os.path.join(train_aug_folder, aug_image_name), image)
        shutil.copy(os.path.join(train_mask_folder, train_image1_name), os.path.join(train_aug_mask_folder,  aug_image_name))
        
    for train_image1_name in tqdm(all_image_names):
        train_image1_path = os.path.join(train_folder, train_image1_name)
        shutil.copy(train_image1_path, os.path.join(train_aug_folder, train_image1_name))
        shutil.copy(os.path.join(train_mask_folder, train_image1_name), os.path.join(train_aug_mask_folder, train_image1_name))