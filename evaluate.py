from dataset import Pneumothorax
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import yaml
from tqdm import tqdm
from metrics import all_dice_scores
import numpy as np
import pdb
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensorV2

def evaluate(config, model, testing_loader):
    model.eval()
    
    dices, negative_dices, positive_dices = [], [], []

    with torch.no_grad():
        for sample in tqdm(testing_loader, desc="Evaluating", leave=False):
            images, labels = sample['image'], sample['mask']
            images, labels = images.to(config['device']), labels.to(config['device'])
            
            predictions = model(images)
            dice, dice_neg, dice_pos = all_dice_scores(predictions, labels, 0.5)
            
            dices.extend(dice.cpu().numpy().tolist())
            negative_dices.extend(dice_neg.cpu().numpy().tolist())
            positive_dices.extend(dice_pos.cpu().numpy().tolist())
          
        
    return np.mean(dices), np.mean(negative_dices), np.mean(positive_dices)

if __name__ == "__main__":
    weights_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/vinbrain_internship/checkpoints/model-ckpt-best.pt"

    # load yaml file
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    list_transforms = []
    list_transforms.extend(
        [
            Resize(config['image_height'], config['image_width']),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
            ToTensorV2(),
        ]
    )

    transform = Compose(list_transforms)
    testing_data = Pneumothorax(config['root_test_image_path'], config['root_test_label_path'], transform=transform)    
    
    testing_loader = DataLoader(testing_data, batch_size=config['batch_size'], shuffle=True)
    
    model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
    
    model.load_state_dict(torch.load(weights_path)['model_state_dict'])
    model.to(config['device'])
    
    dice, dice_neg, dice_pos = evaluate(config, model, testing_loader)
    print(f"Test. dice score: {dice:.3f}")
    print(f"Test. dice_neg score: {dice_neg:.3f}")
    print(f"Test. dice_pos score: {dice_pos:.3f}")
   