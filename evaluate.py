from dataset import EvalPneumothorax, EvalPneumothoraxImagesPair, sampledDataset
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import torch
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import yaml
from tqdm import tqdm
from metrics import all_dice_scores
import numpy as np
import pdb
from albumentations import (Normalize, Resize, Compose)
from albumentations.pytorch import ToTensor
from unet import UNetWithResnet50Encoder, UNetWithResNext101Encoder
import argparse
import cv2
from model import UNET
import albumentations as A
import pandas as pd

def get_args_parser():
    parser = argparse.ArgumentParser("Parsing arguments", add_help=False)
    parser.add_argument("--config", default="./configs/config.yaml", type=str)

    return parser

def evaluate(config, model, testing_loader):
    model.eval()
    
    dices, negative_dices, positive_dices = [], [], []
    
    with torch.no_grad():
        for sample in tqdm(testing_loader, desc="Evaluating", leave=False):
            images, labels = sample['image'], sample['mask']
            images, labels = images.to(config['device']), labels.to(config['device'])
            
            predictions = model(images)
            predictions = predictions.detach().cpu().numpy()[:, 0, :, :] # bs x 1 x width x height --> bs x width x height
            # get cv2 height and width from predictions
            height, width = labels.shape[-2], labels.shape[-1]

            predictions_resize = []
            for idx, prediction in enumerate(predictions):
                if prediction.shape != (height, width):
                    prediction = cv2.resize(prediction, dsize=(height, width), interpolation=cv2.INTER_NEAREST)
                predictions_resize.append(prediction)
        
            predictions_resize = torch.from_numpy(np.array(predictions_resize)).to(config['device'])
            dice, dice_neg, dice_pos = all_dice_scores(predictions_resize, labels, 0.5)
            
            dices.extend(dice.cpu().numpy().tolist())
            negative_dices.extend(dice_neg.cpu().numpy().tolist())
            positive_dices.extend(dice_pos.cpu().numpy().tolist())
            
    if config['positive_only']:
        macro_dice_score = np.mean(positive_dices)
    else:
        macro_dice_score = (np.mean(negative_dices)+np.mean(positive_dices))/2
        
    return macro_dice_score, np.mean(negative_dices), np.mean(positive_dices)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')
def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """   
    _transform = []
    if preprocessing_fn:
        _transform.append(A.Lambda(image=preprocessing_fn))
    _transform.append(A.Lambda(image=to_tensor, mask=to_tensor))
        
    return A.Compose(_transform)

if __name__ == "__main__":
    # weights_path = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/vinbrain_internship/checkpoints/model-ckpt-best.pt"
    parser = argparse.ArgumentParser("Pneumothorax evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    
    # load yaml file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(config)
    weights_path = config['save_checkpoint']
    list_transforms = []
    list_transforms.extend(
        [
            Resize(config['image_height'], config['image_width']),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
            ToTensor(),
        ]
    )

    transform = Compose(list_transforms)
    
    if config['backbone'] == 'None':
        model = UNET(in_channels=3, out_channels=1)
    elif config['backbone'] == 'torchvision.resnet50':
        print("Using torchvision")
        model = UNetWithResnet50Encoder()
    elif config['backbone'] == 'torchvision.resnext101':
        model = UNetWithResNext101Encoder()
    else:
        model = smp.Unet(config['backbone'], encoder_weights="imagenet", activation=None)
        preprocess_input = get_preprocessing_fn(config['backbone'], pretrained='imagenet')
    
    if config['annotation_type'] == 'images':
        testing_data = EvalPneumothoraxImagesPair(config['root_test_image_path'], config['root_test_label_path'], transform=transform)    
    elif config['annotation_type'] == 'json':
        testing_data = EvalPneumothorax(config['root_test_image_path'], config['root_test_label_path'], transform=transform)
    elif config['annotation_type'] == 'dataframe':
        df = pd.read_csv(config['root_df_path'])
        training_data, testing_data = sampledDataset(df, 2, transform, transform, get_preprocessing(preprocess_input))
    testing_loader = DataLoader(testing_data, batch_size=config['batch_size'], shuffle=True)
    
    

    print("Evaluating using ", weights_path)
    model.load_state_dict(torch.load(weights_path)['model_state_dict'])
    model.to(config['device'])
    
    dice, dice_neg, dice_pos = evaluate(config, model, testing_loader)
    print(f"Test. dice score: {dice:.3f}")
    print(f"Test. dice_neg score: {dice_neg:.3f}")
    print(f"Test. dice_pos score: {dice_pos:.3f}")
   