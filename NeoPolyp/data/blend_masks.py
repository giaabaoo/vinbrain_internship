import cv2
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

ALPHA = 0.8
BETA = 1-ALPHA

if __name__ == "__main__":
    image_folder = "/home/dhgbao/VinBrain/assignments/dataset/NeoPolyp/original/train/train"
    mask_folder = "/home/dhgbao/VinBrain/assignments/dataset/NeoPolyp/original/train_gt/train_gt"
    output_path = "/home/dhgbao/VinBrain/assignments/dataset/NeoPolyp/vis_mask/img_gt"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    for image_name in tqdm(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, image_name)
        mask_path = os.path.join(mask_folder, image_name)
        
        image = Image.open(image_path)
        width, height = image.size
        mask = Image.open(mask_path)
                
        font = ImageFont.truetype("LiberationSans-Regular.ttf", 30)
        d1 = ImageDraw.Draw(mask)
        d1.text((width/2 - 100, height-100), "Ground truth", fill=(255, 255, 255), font=font)

        d2 = ImageDraw.Draw(image)
        d2.text((width/2 - 100, height-100), "Original image", fill=(255, 255, 255), font=font)

        get_concat_h(mask, image).save(f'{output_path}/{image_name}')