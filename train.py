import os
from dataset import Pneumothorax
# from model import UNet
from unet import Unet
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn, optim
import torch
import yaml
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary
from torch.nn.functional import cross_entropy
# import albumentations as A
import pdb
from tqdm import tqdm
import time
import wandb
from pathlib import Path
import numpy as np
from loss import MixedLoss, dice_loss
from metrics import all_dice_scores, epoch_time



def train(config, model, training_loader, optimizer, criterion):
    epoch_loss = 0
    dices, negative_dices, positive_dices = [], [], []
    
    model.train()

    for sample in tqdm(training_loader, desc="Training", leave=False):
        images, labels = sample['image'], sample['mask']
        images = images.type(torch.FloatTensor)
        images, labels = images.to(config['device']), labels.to(config['device'])
        
        optimizer.zero_grad()
        predictions = model(images)  # forward
    
        loss = criterion(predictions, labels)
        dice, dice_neg, dice_pos = all_dice_scores(predictions, labels, 0.5)
        loss.backward()  # backward
        optimizer.step()  # optimize
        
        dices.extend(dice.cpu().numpy().tolist())
        negative_dices.extend(dice_neg.cpu().numpy().tolist())
        positive_dices.extend(dice_pos.cpu().numpy().tolist())

        epoch_loss += loss.item()
        
    

    # print(loss_hist)
    return epoch_loss / len(training_loader), np.mean(dices), np.mean(negative_dices), np.mean(positive_dices)


def evaluate(config, model, validation_loader, criterion):
    epoch_loss = 0
    
    dices, negative_dices, positive_dices = [], [], []

    model.eval()

    with torch.no_grad():
        for sample in tqdm(validation_loader, desc="Evaluating", leave=False):
            images, labels = sample['image'], sample['mask']
            images = images.type(torch.FloatTensor)
            images, labels = images.to(config['device']), labels.to(config['device'])
            
            predictions = model(images)
          
            loss = criterion(predictions, labels)
            
            dice, dice_neg, dice_pos = all_dice_scores(predictions, labels, 0.5)

            dices.extend(dice.cpu().numpy().tolist())
            negative_dices.extend(dice_neg.cpu().numpy().tolist())
            positive_dices.extend(dice_pos.cpu().numpy().tolist())
            epoch_loss += loss.item()

    return epoch_loss / len(validation_loader),np.mean(dices), np.mean(negative_dices), np.mean(positive_dices)

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
            torch.save({'epoch': epoch, 
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_valid_loss,
                        }, os.path.join(config['save_path'], "model-ckpt-best.pt"))

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
    # load yaml file
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    transform = transforms.Compose([
        transforms.Resize((config['image_height'], config['image_width'])),
        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transforms.ToTensor()
    ])
    
    training_data = Pneumothorax(config['root_train_image_path'], config['root_train_label_path'], transform=transform)
    testing_data = Pneumothorax(config['root_test_image_path'], config['root_test_label_path'], transform=transform)
    
    training_loader = DataLoader(training_data, batch_size=config['batch_size'], shuffle=True)
    validation_loader = DataLoader(testing_data, batch_size=config['batch_size'], shuffle=True)
    testing_loader = DataLoader(testing_data, batch_size=config['batch_size'], shuffle=True)
    
    # images, masks = next(iter(training_loader))['image'], next(iter(training_loader))['mask']
    # print(images.shape, masks.shape)
    
    # define loss function
    if config['loss_function'] == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    elif config['loss_function'] == 'DiceLoss':
        criterion = None # will assign later
    elif config['loss_function'] == 'MixedLoss':
        criterion = MixedLoss(10.0, 2.0)
        # criterion = dice_loss()

    # model and optimizer
    # model_ft = models.resnet50(weights=True)
    # model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
    
    # model = Unet(model_ft)
    model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
        
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])
    
    if config['scheduler'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        
    if config['continue_training']:
        model.load_state_dict(torch.load(config['continue_training_path'])['model_state_dict'])
        epoch = torch.load(config['continue_training_path'])['epoch']
        # optimizer.load_state_dict(torch.load(config['continue_training_path'])['optimizer_state_dict'])
    else:
        epoch = 0
    
    model.to(config['device'])
    wandb.init(project="pneumothorax", entity="_giaabaoo_", config=config)
    # pdb.set_trace()
    # wandb.config = config
    wandb.watch(model)

    # for i, child in enumerate(model.children()):
    #     if i <= 7:
    #         for param in child.parameters():
    #             param.requires_grad = False
                
    print(summary(model,input_size=(3,512,512))) 
    
    train_and_evaluate(training_loader, validation_loader, model, criterion, optimizer, scheduler, config, epoch)