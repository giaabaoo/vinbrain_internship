from mask_functions import rle2mask
from pathlib import Path
import json
import os
import cv2
import shutil
import pdb
from tqdm import tqdm
import numpy as np

ALPHA = 0.8
BETA = 1  - ALPHA

if __name__ == "__main__":
    train_annotations_path = "../../dataset/annotations/train.json"
    test_annotations_path = "../../dataset/annotations/test.json"
    
    train_images_path = "../../dataset/pngs/original_images/train"
    test_images_path = "../../dataset/pngs/original_images/test"
    
    Path("../../dataset/pngs/segmentation_masks/train").mkdir(parents=True, exist_ok=True)
    Path("../../dataset/pngs/segmentation_masks/test").mkdir(parents=True, exist_ok=True)
    Path("../../dataset/pngs/segmentation_masks/normal_train").mkdir(parents=True, exist_ok=True)
    Path("../../dataset/pngs/segmentation_masks/normal_test").mkdir(parents=True, exist_ok=True)
    Path("../../dataset/pngs/segmentation_masks/train_mask").mkdir(parents=True, exist_ok=True)
    Path("../../dataset/pngs/segmentation_masks/test_mask").mkdir(parents=True, exist_ok=True)
    
    with open(train_annotations_path, "r") as f:
        train_annotations = json.load(f)
        
    with open(test_annotations_path, "r") as f:
        test_annotations = json.load(f)
    
    print("Visualizing train data... ")
    for image_name, annotations in tqdm(train_annotations.items()):
        image_name += ".png"
        image_path = os.path.join(train_images_path, image_name)
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        
        if annotations[0] == "-1":
                shutil.copy(image_path, os.path.join("../../dataset/pngs/segmentation_masks/normal_train", image_name))
        else:
            all_masks = np.zeros((height, width))
            
            for idx, annotation in enumerate(annotations):
                mask = rle2mask(annotation, width, height)
                all_masks += mask
                
            all_masks = np.repeat(all_masks[..., np.newaxis], 3, axis=-1).astype(np.uint8)
            blend_image = cv2.addWeighted(image, ALPHA, all_masks, BETA, 0.0)
            cv2.imwrite(os.path.join("../../dataset/pngs/segmentation_masks/train_mask", image_name), all_masks)
            cv2.imwrite(os.path.join("../../dataset/pngs/segmentation_masks/train", image_name), blend_image)

    print("Visualizing test data... ")
    for image_name, annotations in tqdm(test_annotations.items()):
        image_name += ".png"
        image_path = os.path.join(test_images_path, image_name)
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        
        if annotations[0] == "-1":
                shutil.copy(image_path, os.path.join("../../dataset/pngs/segmentation_masks/normal_test", image_name))
        else:
            all_masks = np.zeros((height, width))
            
            for idx, annotation in enumerate(annotations):
                mask = rle2mask(annotation, width, height)
                all_masks += mask
            
            all_masks = np.repeat(all_masks[..., np.newaxis], 3, axis=-1).astype(np.uint8)
            blend_image = cv2.addWeighted(image, ALPHA, all_masks, BETA, 0.0)
            cv2.imwrite(os.path.join("../../dataset/pngs/segmentation_masks/test_mask", image_name), all_masks)
            cv2.imwrite(os.path.join("../../dataset/pngs/segmentation_masks/test", image_name), blend_image)