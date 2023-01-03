import pdb
from models.unet import UNet
from models.blazeneo.model import BlazeNeo
from models.neounet.model import NeoUNet
from models.doubleunet.model import DUNet
import argparse
from utils.train_utils import apply_transform, prepare_dataloaders, prepare_objectives
from metrics import epoch_time, compute_IoU, compute_F1, compute_dice_coef
import numpy as np
import pandas as pd
import torch.nn.functional as F
from pathlib import Path
import wandb
import time
from tqdm import tqdm
from torchsummary import summary
import yaml
import torch
import os
import segmentation_models_pytorch as smp
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from utils.helper import get_args_parser


def train(config, model, training_loader, optimizer, criterion):
    epoch_loss = 0
    model.train()
    F1_list = []
    IoU_list = []
    
    for sample in tqdm(training_loader, desc="Training", leave=False):
        images, masks = sample['image'], sample['mask']
        if "upsample" in config['backbone'].split("."):
            input_size = images.size()[-1]
            size_rates = [0.75, 1.25, 1]
            train_sizes = [int(round(input_size*rate/32)*32) for rate in size_rates]
            
            for size in train_sizes:
                if size != input_size:
                    upsampled_images = F.upsample(images, size=(size, size), mode='nearest')
                    upsampled_masks = F.upsample(masks.unsqueeze(1).type(torch.FloatTensor), size=(size, size), mode='nearest')
                else:
                    upsampled_images = images
                    upsampled_masks = masks
                    
                upsampled_images, upsampled_masks = upsampled_images.to(config['device']), upsampled_masks.to(config['device'])
                pdb.set_trace()
                optimizer.zero_grad()
                output = model.forward(upsampled_images)
                loss = criterion(output, upsampled_masks)
                loss.backward()

                optimizer.step() 
                pdb.set_trace()
                probs = torch.softmax(output, dim=1)
                predictions = torch.argmax(probs, dim=1)
                
                F1_score = compute_F1(predictions, masks)
                IoU_score = compute_IoU(predictions, masks)
                F1_list.append(F1_score.cpu().numpy())
                IoU_list.append(IoU_score.cpu().numpy())
                epoch_loss += loss.item()
        else:
            if not config['transform']:
                images = images.type(torch.FloatTensor)
            images, masks = images.to(config['device']), masks.to(config['device'])

            optimizer.zero_grad()
            
            
            output = model(images)  # forward
            
            if "blazeneo" in config['backbone']:
                output = output[1]
            elif "neounet" in config['backbone']:
                output = output[0]
            
            
            if config['loss_function'] == 'CrossEntropy_TverskyLoss':
                ce_loss = criterion[0](output, masks)
                tversky_loss = criterion[1](output, masks)
                loss = (ce_loss+tversky_loss)/2
            elif config['loss_function'] == 'ActiveContourLoss':
                soft_output = torch.softmax(output, dim=1)
                loss = criterion(soft_output, masks)
            else:
                loss = criterion(output, masks)

            loss.backward()  # backward
            optimizer.step()  # optimize
            
            probs = torch.softmax(output, dim=1)
            predictions = torch.argmax(probs, dim=1)
            F1_score = compute_F1(predictions, masks)
            IoU_score = compute_IoU(predictions, masks)
            F1_list.append(F1_score.cpu().numpy())
            IoU_list.append(IoU_score.cpu().numpy())
            epoch_loss += loss.item()

    if "upsample" in config['backbone'].split("."):
        return epoch_loss / len(training_loader) * len(train_sizes), np.sum(F1_list)/(len(training_loader) * len(train_sizes)), np.sum(IoU_list)/(len(training_loader) * len(train_sizes))
    else:
        return epoch_loss / len(training_loader), np.mean(F1_list), np.mean(IoU_list)


def evaluate(config, model, validation_loader, criterion):
    epoch_loss = 0

    model.eval()
    F1_list = []
    IoU_list = []
    
    with torch.no_grad():
        for sample in tqdm(validation_loader, desc="Evaluating", leave=False):
            images, masks = sample['image'], sample['mask']
            if not config['transform']:
                images = images.type(torch.FloatTensor)

            images, masks = images.to(
                config['device']), masks.to(config['device'])

           
            output = model(images)  # forward
            
            if "blazeneo" in config['backbone']:
                output = output[1]
            elif "neounet" in config['backbone']:
                output = output[0]
            
            
            if config['loss_function'] == 'CrossEntropy_TverskyLoss':
                ce_loss = criterion[0](output, masks)
                tversky_loss = criterion[1](output, masks)
                loss = (ce_loss+tversky_loss)/2
            elif config['loss_function'] == 'ActiveContourLoss':
                soft_output = torch.softmax(output, dim=1)
                loss = criterion(soft_output, masks)
            else:
                loss = criterion(output, masks)
            
            probs = torch.softmax(output, dim=1)
            predictions = torch.argmax(probs, dim=1)
            F1_score = compute_F1(predictions, masks)
            IoU_score = compute_IoU(predictions, masks)
            F1_list.append(F1_score.cpu().numpy())
            IoU_list.append(IoU_score.cpu().numpy())
            epoch_loss += loss.item()

    return epoch_loss / len(validation_loader), np.mean(F1_list), np.mean(IoU_list)


def train_and_evaluate(training_loader, validation_loader, model, criterion, optimizer, scheduler, config, start_epoch):
    best_valid_loss = float("inf")

    for epoch in range(start_epoch, start_epoch + config['num_epochs']):
        start_time = time.monotonic()
        
        train_loss, train_dice_score, train_iou_score = train(config, model, training_loader, optimizer, criterion)        
        valid_loss, valid_dice_score, valid_iou_score = evaluate(config, model, validation_loader, criterion)
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
        print("Training")
        print(
            f"\t Dice Loss: {train_loss:.3f} | Dice score: {train_dice_score:.3f} | IoU score: {train_iou_score:.3f}")
        print("Validating")
        print(
            f"\t Dice Loss: {valid_loss:.3f} |  Dice score: {valid_dice_score:.3f} | IoU score: {valid_iou_score:.3f}")
        print("\n")
        wandb.log(
            {
                "Train dice loss": train_loss,
                "Train dice score": train_dice_score,
                "Train IoU score": train_iou_score,
                
                "Val. dice loss": valid_loss,
                "Val. dice score": valid_dice_score,
                "Val. IoU score": valid_iou_score,
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "NeoPolyp training script", parents=[get_args_parser()])
    args = parser.parse_args()

    # load yaml file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(config)

    train_transform, valid_transform = apply_transform(config)
    training_loader, validation_loader = prepare_dataloaders(
        config, train_transform, valid_transform)

    if "unetplusplus" in config['backbone'].split("."):
        model = smp.UnetPlusPlus(config['backbone'].split(".")[1], encoder_weights=config['encoder_weights'],
                         in_channels=3, classes=len(config['classes']), activation='sigmoid')
    elif "blazeneo" in config['backbone']:
        model = BlazeNeo()
    elif "neounet" in config['backbone']:
        model = NeoUNet(num_classes=3)
    elif "doubleunet" in config['backbone']:
        model = DUNet()
    elif config['backbone'] != "None":
        model = smp.Unet(config['backbone'], encoder_weights=config['encoder_weights'],
                         in_channels=3, classes=len(config['classes']), activation='sigmoid')
    else:
        model = UNet(n_channels=3, n_classes=3)

    model.to(config['device'])

    criterion, optimizer, scheduler = prepare_objectives(config, model)

    if config['continue_training']:
        model.load_state_dict(torch.load(
            config['trained_weights'])['model_state_dict'])
        epoch = torch.load(config['trained_weights'])['epoch']
    else:
        epoch = 0

    wandb.init(project="neopolyp", entity="_giaabaoo_", config=config)
    wandb.watch(model)
    print(summary(model, input_size=(3, 512, 512)))
    train_and_evaluate(training_loader, validation_loader,
                       model, criterion, optimizer, scheduler, config, epoch)
