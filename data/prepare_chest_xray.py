from pathlib import Path
from tqdm import tqdm
import os
import shutil
import cv2
import numpy as np
import pdb

if __name__ == "__main__":
    labels_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/chest-xray/masks"
    images_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/chest-xray/CXR_png"
    
    train_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/chest-xray/split/train"
    test_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/chest-xray/split/test"
    mask_train_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/chest-xray/split/train_mask"
    mask_test_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/chest-xray/split/test_mask"
    
    Path(train_path).mkdir(parents=True, exist_ok=True)
    Path(test_path).mkdir(parents=True, exist_ok=True)
    Path(mask_train_path).mkdir(parents=True, exist_ok=True)
    Path(mask_test_path).mkdir(parents=True, exist_ok=True)


    split_idx = int(len(os.listdir(images_path)) * 0.8)
    
    data_keys = list(os.listdir(images_path))
    train_keys = data_keys[:split_idx]
    test_keys = data_keys[split_idx:]
    
    # print("Preparing training folders: ")
    # for image_name in tqdm(train_keys):
    #     image_path =os.path.join(images_path, image_name)
    #     shutil.copy(image_path, os.path.join(train_path, image_name))
        
    #     try:
    #         shutil.copy(os.path.join(labels_path, image_name), os.path.join(mask_train_path, image_name))
    #     except:
    #         image = cv2.imread(image_path)
    #         height, width, _ = image.shape
    #         mask = np.zeros((height, width))
    #         mask = np.repeat(mask[..., np.newaxis], 3, axis=-1).astype(np.uint8)
    #         cv2.imwrite(os.path.join(mask_train_path, image_name), mask)
    
    print("Preparing testing folders: ")
    for image_name in tqdm(test_keys):
        image_path =os.path.join(images_path, image_name)
        shutil.copy(image_path, os.path.join(test_path, image_name))
        try:
            shutil.copy(os.path.join(labels_path, image_name), os.path.join(mask_test_path, image_name))
        except:
            image = cv2.imread(image_path)
            height, width, _ = image.shape
            mask = np.zeros((height, width))
            mask = np.repeat(mask[..., np.newaxis], 3, axis=-1).astype(np.uint8)
            cv2.imwrite(os.path.join(mask_test_path, image_name), mask)
    
        