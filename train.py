import os
from dataset import Pneumothorax, PneumothoraxImagesPair
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torch import optim, nn
import torch
import yaml
from torchsummary import summary
from tqdm import tqdm
import time
import wandb
from pathlib import Path
import numpy as np
from loss import MixedLoss, WeightedFocalLoss, DiceLoss, DiceBCELoss, FocalLoss
from metrics import all_dice_scores, epoch_time
from albumentations import (Normalize, Resize, Compose, HorizontalFlip, ShiftScaleRotate, GaussNoise)
from albumentations.pytorch import ToTensorV2
import argparse
import torchvision.models as models
from torch.autograd import Variable


from unet import Unet
from model import UNET
import cv2
import albumentations as A
import pdb

def get_args_parser():
    parser = argparse.ArgumentParser("Parsing arguments", add_help=False)
    
    parser.add_argument("--config", default="./configs/config.yaml", type=str)

    return parser

def train(config, model, training_loader, optimizer, criterion):
    epoch_loss = 0
    dices, negative_dices, positive_dices = [], [], []
    
    model.train()

    for sample in tqdm(training_loader, desc="Training", leave=False):
        images, labels = sample['image'], sample['mask']
        images, labels = images.to(config['device']), labels.to(config['device'])
        
        optimizer.zero_grad()
        predictions = model(images)  # forward      
        # predictions = torch.sigmoid(predictions)       
        # predictions = torch.from_numpy(np.vectorize(lambda value: 0.0 if value <= 0.5 else 1.0)(predictions.detach().cpu().numpy())).to('cuda')
        # predictions = Variable(predictions.data, requires_grad=True)
        
        loss = criterion(predictions, labels)

        # pdb.set_trace()
        loss.backward()  # backward
        optimizer.step()  # optimize
        
        dice, dice_neg, dice_pos = all_dice_scores(predictions, labels, 0.5)
            
        dices.extend(dice.cpu().numpy().tolist())
        negative_dices.extend(dice_neg.cpu().numpy().tolist())
        positive_dices.extend(dice_pos.cpu().numpy().tolist())
        epoch_loss += loss.item()
    
    # macro_dice_score = (np.mean(negative_dices)+np.mean(positive_dices))/2
    macro_dice_score = np.mean(positive_dices)

    return epoch_loss / len(training_loader), macro_dice_score, np.mean(negative_dices), np.mean(positive_dices)


def evaluate(config, model, validation_loader, criterion):
    epoch_loss = 0
    
    dices, negative_dices, positive_dices = [], [], []

    model.eval()

    with torch.no_grad():
        for sample in tqdm(validation_loader, desc="Evaluating", leave=False):
            images, labels = sample['image'], sample['mask']
            images, labels = images.to(config['device']), labels.to(config['device'])
            
            
            predictions = model(images)
            # predictions = torch.sigmoid(predictions)       
            # predictions = torch.from_numpy(np.vectorize(lambda value: 0.0 if value < 0.5 else 1.0)(predictions.detach().cpu().numpy())).to('cuda')
            # predictions = Variable(predictions.data, requires_grad=True)
            
            loss = criterion(predictions, labels)
            
            dice, dice_neg, dice_pos = all_dice_scores(predictions, labels, 0.5)

            dices.extend(dice.cpu().numpy().tolist())
            negative_dices.extend(dice_neg.cpu().numpy().tolist())
            positive_dices.extend(dice_pos.cpu().numpy().tolist())
            epoch_loss += loss.item()
            
    # macro_dice_score = (np.mean(negative_dices)+np.mean(positive_dices))/2
    macro_dice_score = np.mean(positive_dices)
    return epoch_loss / len(validation_loader),macro_dice_score, np.mean(negative_dices), np.mean(positive_dices)

def train_and_evaluate(training_loader, validation_loader, model, criterion, optimizer, scheduler, config, start_epoch):
    best_valid_loss = float("inf")

    for epoch in range(start_epoch, start_epoch + config['num_epochs']):
        start_time = time.monotonic()

        train_loss, train_acc, _, _ = train(config, model, training_loader, optimizer, criterion)
        valid_loss, valid_acc, _,_ = evaluate(config, model, validation_loader, criterion)
        
        scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            Path("./checkpoints/").mkdir(parents=True, exist_ok=True)
            print("Saving checkpoints ", config['save_checkpoint'])
            torch.save({'epoch': epoch, 
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_valid_loss,
                        }, config['save_checkpoint'])

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain dice Loss: {train_loss:.3f} | Train dice score: {train_acc:.3f}")
        print(f"\t Val. dice Loss: {valid_loss:.3f} |  Val. dice score: {valid_acc:.3f}")
        wandb.log(
            {
                "Train dice loss": train_loss,
                "Train dice score": train_acc,
                "Val. dice loss": valid_loss,
                "Val. dice score": valid_acc,
            }
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pneumothorax training script", parents=[get_args_parser()])
    args = parser.parse_args()
    
    # load yaml file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    test_transform = Compose([
            Resize(config['image_height'], config['image_width']),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
            ToTensorV2(),
        ])
    
    if config['transform']:
        train_transform = Compose([
                    HorizontalFlip(p=0.5),
                    ShiftScaleRotate(
                        shift_limit=0,  # no resizing
                        scale_limit=0.1,
                        rotate_limit=10, # rotate
                        p=0.5,
                        border_mode=cv2.BORDER_CONSTANT
                    ),
                    GaussNoise(),
                    A.MultiplicativeNoise(multiplier=1.5, p=1),
                    Resize(config['image_height'], config['image_width']),
                    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
                    ToTensorV2(),
                ])
    
    if config['annotation_type'] == 'images':
        training_data = PneumothoraxImagesPair(config['root_train_image_path'], config['root_train_label_path'], transform=test_transform)
        testing_data = PneumothoraxImagesPair(config['root_test_image_path'], config['root_test_label_path'], transform=test_transform)
    elif config['annotation_type'] == 'json':
        training_data = Pneumothorax(config['root_train_image_path'], config['root_train_label_path'], transform=test_transform)
        testing_data = Pneumothorax(config['root_test_image_path'], config['root_test_label_path'], transform=test_transform)
    
    training_loader = DataLoader(training_data, batch_size=config['batch_size'], shuffle=True)
    validation_loader = DataLoader(testing_data, batch_size=config['batch_size'], shuffle=True)
    testing_loader = DataLoader(testing_data, batch_size=config['batch_size'], shuffle=True)
    
    # images, masks = next(iter(training_loader))['image'], next(iter(training_loader))['mask']
    # print(images.shape, masks.shape)
    
    # define loss function
    if config['loss_function'] == 'DiceLoss':
        criterion = DiceLoss() # will assign later
    elif config['loss_function'] == 'MixedLoss':
        criterion = MixedLoss(10.0, 2.0) # alpha * Focal + DiceLoss
    elif config['loss_function'] == 'FocalLoss':
        criterion = FocalLoss(2.0)
    elif config['loss_function'] == 'WeightedFocalLoss':
        criterion = WeightedFocalLoss(10.0, 2.0)
    elif config['loss_function'] == 'DiceBCELoss':
        criterion = DiceBCELoss()
    elif config['loss_function'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()

    
    if config['backbone'] != 'None':
        model = smp.Unet(config['backbone'], encoder_weights="imagenet", activation=None)
    else:
        model = UNET(in_channels=3, out_channels=1)
    # model_ft = models.resnet50(weights=True)
    # model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
    
    # model = Unet(model_ft)
    
        
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])
    
    if config['scheduler'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        
    if config['continue_training']:
        model.load_state_dict(torch.load(config['save_checkpoint'])['model_state_dict'])
        epoch = torch.load(config['save_checkpoint'])['epoch']
    else:
        epoch = 0
    
    model.to(config['device'])
    wandb.init(project="pneumothorax", entity="_giaabaoo_", config=config)
    wandb.watch(model)
                
    print(summary(model,input_size=(3,512,512))) 
    train_and_evaluate(training_loader, validation_loader, model, criterion, optimizer, scheduler, config, epoch)