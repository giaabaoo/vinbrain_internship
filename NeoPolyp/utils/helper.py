import numpy as np
import argparse
from PIL import Image
import cv2
import torch
import pdb

LABEL_TO_COLOR = {0:[0,0,0], 1:[255,0,0], 2:[0,255,0], 3:[0,0,255]}

def mask_to_class(mask):
        binary_mask = np.array(mask)
        binary_mask = cv2.erode(binary_mask, np.ones((5, 5), np.uint8)) 
        binary_mask[:,:,2] = 0
        binary_mask = (binary_mask != 0).astype(np.uint8)
        binary_mask *= 255
        
        # r = np.array(binary_mask[:,:,0])
        # g = np.array(binary_mask[:,:,1])
        # b = np.array(binary_mask[:,:,2])
        
        # cv2.imwrite("./visualization/red.png", r)
        # cv2.imwrite("./visualization/green.png", g)
        # cv2.imwrite("./visualization/blue.png", b)
        
        # convert colors to "flat" labels
        rgb = np.array(binary_mask)
        output_mask = np.zeros((rgb.shape[0], rgb.shape[1]))

        for k,v in LABEL_TO_COLOR.items():
            output_mask[np.all(rgb==v, axis=2)] = k
        
        output_mask = torch.from_numpy(output_mask)
        output_mask = output_mask.type(torch.int64)

        return output_mask

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
    # pdb.set_trace()
    rgb = np.zeros(mask.shape+(3,), dtype=np.uint8)
    
    for i in np.unique(mask):
        rgb[mask==i] = LABEL_TO_COLOR[i]
            
    return rgb

def postprocess(prediction, image):
    height, width, _ = image.shape
    prediction = cv2.resize(prediction.squeeze(0), (width, height), 0, 0, interpolation=cv2.INTER_NEAREST)

    return prediction


def valid_postprocess(prediction, image):
    prediction = mask2rgb(prediction).squeeze(0)
    
    height, width, _ = image.shape
    prediction = cv2.resize(prediction, (width, height), 0, 0, interpolation=cv2.INTER_NEAREST)
    prediction = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)
    
    return prediction


def visualize(image, mask):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    blend_image = cv2.addWeighted(image, 0.5, mask, 0.5, 0.0)

    return blend_image