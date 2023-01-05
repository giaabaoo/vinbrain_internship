import cv2
import os
from pathlib import Path
from tqdm import tqdm

ALPHA = 0.8
BETA = 1-ALPHA

if __name__ == "__main__":
    image_folder = "/home/dhgbao/VinBrain/assignments/dataset/NeoPolyp/original/train/train"
    mask_folder = "/home/dhgbao/VinBrain/assignments/dataset/NeoPolyp/original/train_gt/train_gt"
    Path("/home/dhgbao/VinBrain/assignments/dataset/NeoPolyp/vis_mask").mkdir(parents=True, exist_ok=True)
    for image_name in tqdm(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, image_name)
        mask_path = os.path.join(mask_folder, image_name)
        
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)
        blend_image = cv2.addWeighted(image, ALPHA, mask, BETA, 0.0)
        
        cv2.imwrite(f"/home/dhgbao/VinBrain/assignments/dataset/NeoPolyp/vis_mask/blend_{image_name}", blend_image)