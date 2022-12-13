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

if __name__ == "__main__":
    annotation_path = "../../dataset/annotations/stage_2_train.csv"
    image_path = "../../dataset/pngs/original_images"
    
    Path("../../dataset/pngs/balanced_images/train/").mkdir(parents=True, exist_ok=True)
    Path("../../dataset/pngs/balanced_images/test/").mkdir(parents=True, exist_ok=True)
    dicom_path_dict = {}
    # Get all the image paths in subdirectories
    all_images_path = glob.glob(image_path + "/train/*.png") + glob.glob(image_path + "/test/*.png")
    
    for image_path in all_images_path:
        image_id = Path(image_path).stem
        dicom_path_dict[image_id] = image_path
        
        
    df_all = pd.read_csv(annotation_path, usecols=["ImageId", "EncodedPixels"])
    

    df = df_all.drop_duplicates('ImageId')

    df_with_mask = df[df["EncodedPixels"] != "-1"]
    df_with_mask['has_mask'] = 1
    df_without_mask = df[df["EncodedPixels"] == "-1"]
    df_without_mask['has_mask'] = 0
    df_without_mask_sampled = df_without_mask.sample(len(df_with_mask), random_state=69) # random state is imp
    df = pd.concat([df_with_mask, df_without_mask_sampled])
    #NOTE: equal number of positive and negative cases are chosen.
    
    kfold = StratifiedKFold(5, shuffle=True, random_state=69)
    train_idx, test_idx = list(kfold.split(df["ImageId"], df["has_mask"]))[0]
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
    
    train_dict, test_dict = {}, {}
    
    print("Resampling train images...")

    for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
        image_id, encoded_pixels = row["ImageId"], row["EncodedPixels"]
        
        try:
            train_dict[image_id].append(encoded_pixels)
        except:
            train_dict[image_id] = [encoded_pixels]
            
        shutil.copy(dicom_path_dict[image_id], "../../dataset/pngs/balanced_images/train/" + image_id + ".png")
        
    print("Resampling test images...")
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        image_id, encoded_pixels = row["ImageId"], row["EncodedPixels"]
        
        try:
            test_dict[image_id].append(encoded_pixels)
        except:
            test_dict[image_id] = [encoded_pixels]
            
        shutil.copy(dicom_path_dict[image_id], "../../dataset/pngs/balanced_images/test/" + image_id + ".png")
       
    print("Generating json annotations for each image...")
    with open('../../dataset/annotations/train.json', 'w') as f:
        json.dump(train_dict, f)
    
    with open('../../dataset/annotations/test.json', 'w') as f:
        json.dump(test_dict, f)
        
    print("Done!")
    
    
    
    
                 
    # print("Generating json annotations for each image...")
    # with open('../../dataset/annotations/train.json', 'w') as f:
    #     json.dump(train_dict, f)
    
    # with open('../../dataset/annotations/test.json', 'w') as f:
    #     json.dump(test_dict, f)
        
    # print("Done!")