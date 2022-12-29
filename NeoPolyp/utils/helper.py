import numpy as np
import argparse
from PIL import Image
import cv2
import pdb

LABEL_TO_COLOR = {0:[0,0,0], 1:[255,0,0], 2:[0,255,0], 3:[0,0,255]}

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def get_args_parser():
    parser = argparse.ArgumentParser("Parsing arguments", add_help=False)
    parser.add_argument("--config", default="./configs/default.yaml", type=str)

    return parser

def mask2rgb(mask):
    rgb = np.zeros(mask.shape+(3,), dtype=np.uint8)
    
    for i in np.unique(mask):
            rgb[mask==i] = LABEL_TO_COLOR[i]
            
    return rgb

def postprocess(prediction, image):
    prediction = mask2rgb(prediction).squeeze(0)
    height, width, _ = image.shape
    prediction = cv2.resize(prediction, (width, height), 0, 0, interpolation=cv2.INTER_NEAREST)

    return prediction


def visualize(image, mask):
    blend_image = cv2.addWeighted(image, 0.5, mask, 0.5, 0.0)

    return blend_image