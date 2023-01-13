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
    
def read_mask(mask):
        mask = np.array(mask)
        # cv2.imwrite("./visualization/original.png", mask)
        output = np.zeros(mask.shape[:2])
        output = np.where(mask[...,0] >127 , 1, output)
        output = np.where(mask[...,1] >127 , 2, output)
        
        return torch.from_numpy(output.astype(np.uint8)).type(torch.int64)
    
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

def refine_mask(image_name, predictions):    
    rgbmask = mask2rgb(predictions)
    bgrmask = cv2.cvtColor(rgbmask, cv2.COLOR_RGB2BGR)
    gray_mask = cv2.cvtColor(bgrmask, cv2.COLOR_BGR2GRAY)
    _, gray_mask = cv2.threshold(gray_mask, 0, 255, cv2.THRESH_BINARY)
    
    # Fill small holes in each components
    gray_mask_filled = gray_mask.copy()
    h, w = gray_mask.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(gray_mask_filled, mask, (0,0), 255)
    gray_mask_filled_inv = cv2.bitwise_not(gray_mask_filled)
    im_out = gray_mask | gray_mask_filled_inv
    # cv2.imwrite("./visualization/gray.png", gray_mask)
    # cv2.imwrite("./visualization/gray_filled.png", gray_mask_filled)
    # cv2.imwrite("./visualization/gray_filled_combine.png", im_out)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(im_out, 4, cv2.CV_32S)

    # For each list of contour points, choose the prominent color and assign it to the whole contour 
    for i in range(1, num_labels):
        # Access the image pixels and create a 1D numpy array then add to list
        pts = np.where(labels == i)
        pts = list(zip(list(pts[0]),list(pts[1])))
    
        ## count number of pixels in red and green labels
        red_count, green_count = 0, 0
        
        for x, y in pts:
            if predictions[x,y] == 1:
                red_count += 1
            elif predictions[x,y] == 2:
                green_count += 1

        # pdb.set_trace()
        ## if red > green --> set all red
        if red_count > green_count:
            for x, y in pts:
                predictions[x,y] = 1
        else:
            for x, y in pts:
                predictions[x,y] = 2

        # pdb.set_trace()

    return predictions
    
def postprocess(prediction, image):
    height, width, _ = image.shape
    prediction = cv2.resize(prediction.squeeze(0), (width, height), 0, 0, interpolation=cv2.INTER_NEAREST)

    return prediction


# def valid_postprocess(prediction, image):
#     # prediction = mask2rgb(prediction).squeeze(0)
    
#     height, width, _ = image.shape
#     prediction = cv2.resize(prediction, (width, height), 0, 0, interpolation=cv2.INTER_NEAREST)
#     # prediction = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)
    
#     return prediction


def visualize(image, mask):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    blend_image = cv2.addWeighted(image, 0.5, mask, 0.5, 0.0)

    return blend_image