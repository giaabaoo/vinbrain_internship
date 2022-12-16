import json
import pandas as pd
import glob
import pydicom
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from pydicom.pixel_data_handlers.util import apply_voi_lut
import pdb

def process_dicom(file_path):
    dicom = pydicom.read_file(file_path)
    
    data = apply_voi_lut(dicom.pixel_array, dicom)
    
    # Correct image inversion.
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
            
    return data

if __name__ == "__main__":
    annotation_path = "../../dataset/annotations/train-rle.csv"
    
    dicom_images_train_glob = "../../dataset/dicom/dicom-images-train/*/*/*.dcm"
    dicom_images_test_glob = "../../dataset/dicom/dicom-images-test/*/*/*.dcm"
    
    dicom_path_dict = {}
    
    all_images_path = glob.glob(dicom_images_train_glob) + glob.glob(dicom_images_test_glob)
    
    for image_path in all_images_path:
        image_id = Path(image_path).stem
        dicom_path_dict[image_id] = image_path
            
    Path("../../dataset/pngs/original_rle_images/train/").mkdir(parents=True, exist_ok=True)
    Path("../../dataset/pngs/original_rle_images/test/").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(annotation_path)
    data_dict = {}
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_id, encoded_pixels = row["ImageId"], row[" EncodedPixels"]
        
        try:
            data_dict[image_id].append(encoded_pixels.lstrip(" "))
        except:
            data_dict[image_id] = [encoded_pixels.lstrip(" ")]
    
    train_dict, test_dict = {}, {}   

    split_idx = int(len(data_dict) * 0.8)
    
    data_keys = list(data_dict.keys())
    train_keys = data_keys[:split_idx]
    test_keys = data_keys[split_idx:]
    
    print("Processing train images (converting from dicom to pngs)...")

    for image_id in tqdm(train_keys):
        train_dict[image_id] = data_dict[image_id]
        
        file_path = dicom_path_dict[image_id]
        image = process_dicom(file_path)
        cv2.imwrite("../../dataset/pngs/original_rle_images/train/" + str(image_id) + ".png", image)
    
    print("Processing test images (converting from dicom to pngs)...")
    for image_id in tqdm(test_keys):
        test_dict[image_id] = data_dict[image_id]
        
        file_path = dicom_path_dict[image_id]
        image = process_dicom(file_path)
        cv2.imwrite("../../dataset/pngs/original_rle_images/test/" + str(image_id) + ".png", image)
    
    
    print("Generating json annotations for each image...")
    with open('../../dataset/annotations/full_rle/train.json', 'w') as f:
        json.dump(train_dict, f)
    
    with open('../../dataset/annotations/full_rle/test.json', 'w') as f:
        json.dump(test_dict, f)
        
    print("Done!")