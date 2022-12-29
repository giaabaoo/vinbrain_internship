import pdb
from models.unet import UNet
import argparse
from utils.train_utils import apply_transform, prepare_dataloaders, prepare_objectives
from metrics import compute_CM, epoch_time, compute_IoU, compute_F1, compute_dice_coef
import numpy as np
import pandas as pd
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
    dice_coef_list = []
    
    for sample in tqdm(training_loader, desc="Training", leave=False):
        images, masks = sample['image'], sample['mask']
        if not config['transform']:
            images = images.type(torch.FloatTensor)
        images, masks = images.to(config['device']), masks.to(config['device'])

        optimizer.zero_grad()
        output = model(images)  # forward
        probs = torch.softmax(output, dim=1)
        predictions = torch.argmax(probs, dim=1)
        
        loss = criterion(output, masks)

        loss.backward()  # backward
        optimizer.step()  # optimize

        dice_coef = compute_dice_coef(masks, predictions)
        dice_coef_list.append(dice_coef.cpu().numpy())
        epoch_loss += loss.item()

    return epoch_loss / len(training_loader), np.mean(dice_coef_list)


def evaluate(config, model, validation_loader, criterion):
    epoch_loss = 0

    model.eval()
    dice_coef_list = []

    with torch.no_grad():
        for sample in tqdm(validation_loader, desc="Evaluating", leave=False):
            images, masks = sample['image'], sample['mask']
            if not config['transform']:
                images = images.type(torch.FloatTensor)

            images, masks = images.to(
                config['device']), masks.to(config['device'])

            output = model(images)  # forward
            probs = torch.softmax(output, dim=1)
            predictions = torch.argmax(probs, dim=1)

            loss = criterion(output, masks)
            dice_coef = compute_dice_coef(masks, predictions)
            dice_coef_list.append(dice_coef.cpu().numpy())
            epoch_loss += loss.item()

    return epoch_loss / len(validation_loader), np.mean(dice_coef_list)


def train_and_evaluate(training_loader, validation_loader, model, criterion, optimizer, scheduler, config, start_epoch):
    best_valid_loss = float("inf")

    for epoch in range(start_epoch, start_epoch + config['num_epochs']):
        start_time = time.monotonic()
        
        train_loss, train_dice_score = train(config, model, training_loader, optimizer, criterion)        
        valid_loss, valid_dice_score = evaluate(config, model, validation_loader, criterion)
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
            f"\t Dice Loss: {train_loss:.3f} | Dice score: {train_dice_score:.3f}")
        print("Validating")
        print(
            f"\t Dice Loss: {valid_loss:.3f} |  Dice score: {valid_dice_score:.3f}")
        print("\n")
        wandb.log(
            {
                "Train dice loss": train_loss,
                "Train dice score": train_dice_score,
                "Val. dice loss": valid_loss,
                "Val. dice score": valid_dice_score,
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

    if config['backbone'] != "None":
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
