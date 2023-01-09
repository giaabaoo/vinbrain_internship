import numpy as np
import cv2
import yaml
import torch
import pdb
from albumentations import (Normalize, Resize, Compose)
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import argparse
from tqdm import tqdm
import pandas as pd
import os
from utils.helper import mask2rgb, get_concat_h, get_args_parser, postprocess, visualize, refine_mask
from utils.train_utils import prepare_architecture
import torch.nn.functional as F

LABEL_TO_COLOR = {0:[0,0,0], 1:[255,0,0], 2:[0,255,0]}

def predict2submit(predict, class_id):
    pixels = predict.flatten()
    pixels = np.where(pixels == class_id, 1, 0)
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return ' '.join(str(x) for x in rle)

def save_results(csv_name, output, image, image_name):
    output = mask2rgb(output)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
    cv2.imwrite(f"./results/{csv_name}/test_mask/{image_name}", output)
    
    blend = visualize(image, output)
    cv2.imwrite(f"./results/{csv_name}/test_blend/{image_name}", blend)
    
def save_csv(image_name, output):
    # mask = Image.fromarray(output)
    mask = np.array(output)
    for channel in range(2):
        ids.append(f'{image_name.replace(".jpeg","")}_{channel}')
        string = predict2submit(mask, channel+1)
        strings.append(string)
    return ids, strings

if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(
        "NeoPolyp inference script", parents=[get_args_parser()])
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(config)
    
    transform = Compose([
        Resize(config['image_height'], config['image_width']),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
        ToTensorV2(),
    ])

    model = prepare_architecture(config)
    
    csv_name = config['csv_name']
    Path(f"./results/{csv_name}/test_blend").mkdir(parents=True, exist_ok=True)
    Path(f"./results/{csv_name}/test_mask").mkdir(parents=True, exist_ok=True)

    weights_path = config['save_checkpoint']
    print("Inferencing using ", weights_path)
    model.load_state_dict(torch.load(weights_path)['model_state_dict'])
    model.to(config['device'])

    path = config['root_test_image_path']
    # path = "/home/dhgbao/VinBrain/assignments/vinbrain_internship/NeoPolyp/example_data/valid"
    ids, strings = [], []
    # Inference on all images
    for image_name in tqdm(os.listdir(path)):
        image = cv2.imread(os.path.join(path, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        model.eval()
                
        augmented = transform(image=image)
        image_transform = augmented['image']
        
        with torch.no_grad():
            if not config['transform']:
                image_transform = image_transform.type(torch.FloatTensor)
            image_transform = image_transform.unsqueeze(0).to(config['device'])

            output = model(image_transform)
            if "blazeneo" in config['backbone']:
                output = output[1]
            elif "neounet" in config['backbone']:
                output = output[0]
            elif "pranet" in config['backbone']:
                output  = output[0]
                # output  = output[-1]
            elif "deeplabv3" in config['backbone']:
                output = output['out']
        
        if config['probability_correction_strategy']:
            pred  = F.interpolate(output, size=(image_transform.shape[-2], image_transform.shape[-1]), mode='bilinear', align_corners=True)
            for i in range(3):
                pred[:,i,:,:][torch.where(pred[:,i,:,:]>0)] /= (pred[:,i,:,:]>0).float().mean()
                pred[:,i,:,:][torch.where(pred[:,i,:,:]<0)] /= (pred[:,i,:,:]<0).float().mean()
            probs = torch.softmax(pred, dim=1)
        else:
            probs = torch.softmax(output, dim=1)
            
        prediction = torch.argmax(probs, dim=1)
        
        output = postprocess(prediction.cpu().numpy(), image)
        
        if config['refine_masks'] :
            ### refine_mask by following the rule: 1 label per polyp only
            output = refine_mask(image_name, output)
        
        ids, strings = save_csv(image_name, output)
        save_results(csv_name, output, image, image_name)

    res = {
            'ids': ids,
            'strings': strings,
        }

    df = pd.DataFrame(columns=['Id', 'Expected'])
    df['Id'] = res['ids']
    df['Expected'] = res['strings']
    
   
    df.to_csv(f'./submission/{csv_name}.csv', index=False)
        
