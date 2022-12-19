import cv2
import os
import numpy as np
from pathlib import Path
import pdb

if __name__ == "__main__":
    sample_images = ["1.2.276.0.7230010.3.1.4.8323329.300.1517875162.258081",
                    "1.2.276.0.7230010.3.1.4.8323329.32752.1517875162.169303",
                    "1.2.276.0.7230010.3.1.4.8323329.32719.1517875161.965007",
                    "1.2.276.0.7230010.3.1.4.8323329.32686.1517875161.807009",
                    "1.2.276.0.7230010.3.1.4.8323329.3678.1517875178.953520"]

    image_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/pngs256/train"
    label_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/pngs256/masks"
    Path("/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/pngs256/blend").mkdir(parents=True, exist_ok=True)

    for idx, image_name in enumerate(sample_images):
        # pdb.set_trace()
        image = cv2.imread(os.path.join(image_path, image_name + ".png"))
        mask = cv2.imread(os.path.join(label_path, image_name+ ".png"))
        
        # pdb.set_trace()
        # mask = np.repeat(mask[..., np.newaxis], 3, axis=-1).astype(np.uint8)
        blend_image = cv2.addWeighted(image, 0.5, mask, 0.5, 0.0)
        cv2.imwrite(os.path.join("/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/pngs256/blend", f"mask_{idx}.png"), blend_image)
        cv2.imwrite(os.path.join("/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/pngs256/blend", f"image_{idx}.png"), image)