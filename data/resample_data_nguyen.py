import json
import pandas as pd
import glob
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import pdb
import shutil
import os

if __name__ == "__main__":
    annotation_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/annotations/preprocessing_data_bao.csv"
    all_images_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/nguyen/train"
    
    Path("/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/nguyen/balanced_images/train/").mkdir(parents=True, exist_ok=True)
    Path("/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/nguyen/balanced_images/test/").mkdir(parents=True, exist_ok=True)
    dicom_path_dict = {}
    # Get all the image paths in subdirectories
    
    for image_path in os.listdir(all_images_path):
        image_id = Path(image_path).stem
        dicom_path_dict[image_id] = os.path.join(all_images_path, image_path)
    
    df_all = pd.read_csv(annotation_path, usecols=["UID", "EncodedPixels"])
    df = df_all.drop_duplicates('UID')

    df_with_mask = df[df["EncodedPixels"] != "-1"]
    df_with_mask['has_mask'] = 1
    df_without_mask = df[df["EncodedPixels"] == "-1"]
    df_without_mask['has_mask'] = 0
    df_without_mask_sampled = df_without_mask.sample(len(df_with_mask), random_state=69) # random state is imp
    df = pd.concat([df_with_mask, df_without_mask_sampled])
    #NOTE: equal number of positive and negative cases are chosen.
    
    kfold = StratifiedKFold(5, shuffle=True, random_state=69)
    train_idx, test_idx = list(kfold.split(df["UID"], df["has_mask"]))[0]
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
    
    
    train_dict, test_dict = {}, {}
    
    print("Resampling train images...")

    for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
        image_id, encoded_pixels = row["UID"], row["EncodedPixels"]
        
        try:
            path = dicom_path_dict[image_id]
            try:
                train_dict[image_id].append(encoded_pixels)
            except:
                train_dict[image_id] = [encoded_pixels]
            
            shutil.copy(dicom_path_dict[image_id], "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/nguyen/balanced_images/train/" + image_id + ".png")
        except:
            continue
        
    print("Resampling test images...")
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        image_id, encoded_pixels = row["UID"], row["EncodedPixels"]
        
        try:
            path = dicom_path_dict[image_id]
            try:
                test_dict[image_id].append(encoded_pixels)
            except:
                test_dict[image_id] = [encoded_pixels]
                
            shutil.copy(dicom_path_dict[image_id], "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/nguyen/balanced_images/test/" + image_id + ".png")
        except:
           continue
    # print("Generating json annotations for each image...")
    # with open('../../dataset/annotations/balanced_nguyen/train.json', 'w') as f:
    #     json.dump(train_dict, f)
    
    with open('../../dataset/annotations/balanced_nguyen/test.json', 'w') as f:
        json.dump(test_dict, f)
        
    print("Done!")