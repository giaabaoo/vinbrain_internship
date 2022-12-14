from dataset import EvalPneumothorax
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
import argparse
import cv2

def get_args_parser():
    parser = argparse.ArgumentParser("Parsing arguments", add_help=False)
    parser.add_argument("--config", default="./configs/config.yaml", type=str)

    return parser

def post_process(probability, threshold, min_size):
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((1024, 1024), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num

def evaluate(config, model, testing_loader):
    model.eval()
    
    dices, negative_dices, positive_dices = [], [], []

    with torch.no_grad():
        for sample in tqdm(testing_loader, desc="Evaluating", leave=False):
            images, labels = sample['image'], sample['mask']
            images, labels = images.to(config['device']), labels.to(config['device'])
            
            predictions = model(images)
            pdb.set_trace()
            predictions = predictions.detach().cpu().numpy()[:, 0, :, :]
            predictions_resize = []
            for prediction in predictions:
                if prediction.shape != (1024, 1024):
                    prediction = cv2.resize(prediction, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
                predictions_resize.append(prediction)
            
            dice, dice_neg, dice_pos = all_dice_scores(predictions_resize, labels, 0.5)
            
            dices.extend(dice.cpu().numpy().tolist())
            negative_dices.extend(dice_neg.cpu().numpy().tolist())
            positive_dices.extend(dice_pos.cpu().numpy().tolist())
        
    return np.mean(dices), np.mean(negative_dices), np.mean(positive_dices)

if __name__ == "__main__":
    weights_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/vinbrain_internship/checkpoints/model-ckpt-best.pt"
    parser = argparse.ArgumentParser("Pneumothorax evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    
    # load yaml file
    with open(args.config, 'r') as f:
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
    testing_data = EvalPneumothorax(config['root_test_image_path'], config['root_test_label_path'], transform=transform)    
    
    testing_loader = DataLoader(testing_data, batch_size=config['batch_size'], shuffle=True)
    
    if config['backbone'] == "resnet34":
        model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
    elif config['backbone'] == "efficientnet-b4":
        model = smp.Unet("efficientnet-b4", encoder_weights="imagenet", activation=None)
    
    model.load_state_dict(torch.load(weights_path)['model_state_dict'])
    model.to(config['device'])
    
    dice, dice_neg, dice_pos = evaluate(config, model, testing_loader)
    print(f"Test. dice score: {dice:.3f}")
    print(f"Test. dice_neg score: {dice_neg:.3f}")
    print(f"Test. dice_pos score: {dice_pos:.3f}")
   