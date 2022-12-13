from unet import Unet
import numpy as np
import cv2
import yaml
from PIL import Image
import torch
import torchvision.transforms as transforms
import pdb
import torchvision.models as models
from torch import nn, optim
from unet import Unet
import segmentation_models_pytorch as smp


def postprocess(config, x):
    # pdb.set_trace()
    x = np.transpose(x, (1, 2, 0)) 
    # x = np.argmax(x, axis=2) 
    x = np.vectorize(lambda value: 0 if value < 0.5 else 255)(x)
    x = cv2.resize(x, (1024, 1024), 0, 0, interpolation = cv2.INTER_NEAREST)
    
    return x

def visualize(image_path, mask):
    # pdb.set_trace()
    image = cv2.imread(image_path)
    mask = np.repeat(mask[..., np.newaxis], 3, axis=-1).astype(np.uint8)
     
    blend_image = cv2.addWeighted(image, 0.5, mask, 0.5, 0.0)
    
    return blend_image

if __name__ == "__main__":
    weights_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/code/checkpoints/model-ckpt-best.pt"
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    transform = transforms.Compose([
        transforms.Resize((config['image_height'], config['image_width'])),
        transforms.ToTensor()
    ])
    
    model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
    model.load_state_dict(torch.load(weights_path)['model_state_dict'])
    model.to(config['device'])
    
    # for i, child in enumerate(model.children()):
    #     if i <= 7:
    #         for param in child.parameters():
    #             param.requires_grad = False
                
    ### Inference on image
    image_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/pngs/original_images/train/1.2.276.0.7230010.3.1.4.8323329.524.1517875163.298894.png"
    image = cv2.imread(image_path)
    image = Image.fromarray(image)

    model.eval()
    image = transform(image).unsqueeze(0).to(config['device'])
    output = model(image)
    output = output.squeeze(0).detach().cpu().numpy()
    output = postprocess(config, output)
    
    # pdb.set_trace()
    cv2.imwrite("output.png", output)
    blend = visualize(image_path, output)
    cv2.imwrite("blend.png", blend)