import pdb
from metrics import epoch_time, compute_IoU, compute_F1
import numpy as np
import torch.nn.functional as F
from pathlib import Path
import wandb
import time
from tqdm import tqdm
import torch

def train_upsample(sample, config, optimizer, model, criterion):
    images, masks = sample['image'], sample['mask']
    input_size = images.size()[-1]
    size_rates = config['size_rates']
    train_sizes = [int(round(input_size*rate/32)*32) for rate in size_rates]
    
    for size in train_sizes:
        if size != input_size:
            upsampled_images = F.interpolate(images, size=(size, size), mode='nearest')
            upsampled_masks = F.interpolate(masks.unsqueeze(1).type(torch.FloatTensor), size=(size, size), mode='nearest').squeeze(1).type(torch.LongTensor)
        else:
            upsampled_images = images
            upsampled_masks = masks
            
        upsampled_images, upsampled_masks = upsampled_images.to(config['device']), upsampled_masks.to(config['device'])
        optimizer.zero_grad()
        output = model(upsampled_images)
        
        if "blazeneo" in config['backbone']:
            output = output[1]
        elif "neounet" in config['backbone']:
            output = output[0]
        elif "pranet" in config['backbone']:
            output, output4, output3, output2  = output
            # output5, output4, output3, output  = output
        elif "deeplabv3" in config['backbone']:
            output = output['out']
        elif "polyp_pvt_p1" == config['backbone']:
            P1, P2 = output
            output = P1
        elif "polyp_pvt_p2" == config['backbone']:
            P1, P2 = output
            output = P2
        elif "polyp_pvt" == config['backbone']:
            P1, P2 = output
            res = F.interpolate(P1 + P2 , size=(config['image_height'], config['image_width']), mode='bilinear', align_corners=False)
            output = res
        
        if config['loss_function'] == 'CrossEntropy_TverskyLoss':
            ce_loss = criterion[0](output, masks)
            tversky_loss = criterion[1](output, masks)
            loss = (ce_loss+tversky_loss)/2
        elif config['loss_function'] == 'ActiveContourLoss':
            soft_output = torch.softmax(output, dim=1)
            loss = criterion(soft_output, masks)
        elif config['loss_function'] == 'CE_DiceLoss': 
            ce_loss = criterion[0](output, masks)
            dice_loss = criterion[1](output, masks)
            
            loss = 0.4 * ce_loss + 0.6 * dice_loss
        elif config['loss_function'] == 'FocalDiceLoss':
            focal_loss = criterion[0](output, masks)
            dice_loss = criterion[1](output, masks)
            w1, w2 = config['weights']
            
            loss = w1 * focal_loss + w2 * dice_loss
        elif config['loss_function'] == 'PraNetLoss': 
            loss5 = criterion(output5, masks, config['weights'])
            loss4 = criterion(output4, masks, config['weights'])
            loss3 = criterion(output3, masks, config['weights'])
            loss2 = criterion(output, masks, config['weights'])
            loss = loss2 + loss3 + loss4 + loss5
        elif "polyp_pvt" == config['backbone']:
            loss_P1 = criterion(P1, masks)
            loss_P2 = criterion(P2, masks)
            
            loss = loss_P1 + loss_P2
        else:
            loss = criterion(output, masks)
        loss.backward()

        optimizer.step() 
        probs = torch.softmax(output, dim=1)
        predictions = torch.argmax(probs, dim=1)
        
        F1_score = compute_F1(predictions, upsampled_masks)
        IoU_score = compute_IoU(predictions, upsampled_masks)
        
    return loss, F1_score, IoU_score

def default_train(sample, config, optimizer, model, criterion):
    images, masks = sample['image'], sample['mask']
    if not config['transform']:
        images = images.type(torch.FloatTensor)
    images, masks = images.to(config['device']), masks.to(config['device'])

    optimizer.zero_grad()
    output = model(images)  # forward
    
    if "blazeneo" in config['backbone']:
        output = output[1]
    elif "neounet" in config['backbone']:
        output = output[0]
    elif "pranet" in config['backbone']:
        output, output4, output3, output2  = output
        # output5, output4, output3, output  = output
    elif "deeplabv3" in config['backbone']:
        output = output['out']
    elif "polyp_pvt" in config['backbone']:
        P1, P2 = output
        res = F.interpolate(P1 + P2 , size=(config['image_height'], config['image_width']), mode='bilinear', align_corners=False)
        output = res
    elif "polyp_pvt_p1" == config['backbone']:
        P1, P2 = output
        output = P1
    
    elif "polyp_pvt_p2" == config['backbone']:
        P1, P2 = output
        output = P2
            
    if config['loss_function'] == 'CrossEntropy_TverskyLoss':
        ce_loss = criterion[0](output, masks)
        tversky_loss = criterion[1](output, masks)
        loss = (ce_loss+tversky_loss)/2
    elif config['loss_function'] == 'ActiveContourLoss':
        soft_output = torch.softmax(output, dim=1)
        loss = criterion(soft_output, masks)
    elif config['loss_function'] == 'CE_DiceLoss': 
        ce_loss = criterion[0](output, masks)
        dice_loss = criterion[1](output, masks)
        
        loss = 0.4 * ce_loss + 0.6 * dice_loss
    elif config['loss_function'] == 'FocalDiceLoss':
        focal_loss = criterion[0](output, masks)
        dice_loss = criterion[1](output, masks)
        w1, w2 = config['weights']
        
        loss = w1 * focal_loss + w2 * dice_loss
    elif config['loss_function'] == 'PraNetLoss': 
        loss5 = criterion(output5, masks, config['weights'])
        loss4 = criterion(output4, masks, config['weights'])
        loss3 = criterion(output3, masks, config['weights'])
        loss2 = criterion(output, masks, config['weights'])
        loss = loss2 + loss3 + loss4 + loss5
    elif "polyp_pvt" == config['backbone']:
        loss_P1 = criterion(P1, masks)
        loss_P2 = criterion(P2, masks)
        
        loss = loss_P1 + loss_P2
    else:
        loss = criterion(output, masks)

    loss.backward()  # backward
    optimizer.step()  # optimize
    
    probs = torch.softmax(output, dim=1)
    predictions = torch.argmax(probs, dim=1)
    F1_score = compute_F1(predictions, masks)
    IoU_score = compute_IoU(predictions, masks)
    
    return loss, F1_score, IoU_score

def tga_train(sample, config, optimizer, model, criterion):
    images, text_classes = sample['image'], sample['text_classes']
    masks, num_polyps, polyp_sizes = sample['mask'], sample['num_polyps'], sample['polyp_sizes']
    
    images, text_classes = images.to(config['device']), text_classes.to(config['device'])
    masks, num_polyps, polyp_sizes = masks.to(config['device']), num_polyps.to(config['device']), polyp_sizes.to(config['device'])

    optimizer.zero_grad()
    p1, p2, p3 = model(images, text_classes)  # forward
    p2 = torch.softmax(p2, dim=1)
    p3 = torch.softmax(p3, dim=1)
            
    loss_P1 = criterion(p1, masks)
    loss_P2 = criterion(p2, num_polyps)
    loss_P3 = criterion(p3, polyp_sizes)
    
    loss = (loss_P1 + loss_P2 + loss_P3).mean()

    loss.backward()  # backward
    optimizer.step()  # optimize
    
    output = p1
    probs = torch.softmax(output, dim=1)
    predictions = torch.argmax(probs, dim=1)
    F1_score = compute_F1(predictions, masks)
    IoU_score = compute_IoU(predictions, masks)
    
    return loss, F1_score, IoU_score

def train(config, model, training_loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    F1_list = []
    IoU_list = []
    
    for sample in tqdm(training_loader, desc="Training", leave=False):
        if "upsample" in config['backbone'].split("."):
            loss, F1_score, IoU_score = train_upsample(sample, config, optimizer, model, criterion)
        elif "tganet" in config['backbone']:
            loss, F1_score, IoU_score = tga_train(sample, config, optimizer, model, criterion)
        else:
            loss, F1_score, IoU_score = default_train(sample, config, optimizer, model, criterion)
        
        
        F1_list.append(F1_score.cpu().numpy())
        IoU_list.append(IoU_score.cpu().numpy())
        epoch_loss += loss.item()
    
    if "upsample" in config['backbone'].split("."):
        return epoch_loss / len(training_loader) * len(config['size_rates']), np.sum(F1_list)/(len(training_loader) * len(config['size_rates'])), np.sum(IoU_list)/(len(training_loader) * len(config['size_rates']))
    else:
        return epoch_loss / len(training_loader), np.mean(F1_list), np.mean(IoU_list)

def default_evaluate(sample, config, model, criterion):
    images, masks = sample['image'], sample['mask']
    if not config['transform']:
        images = images.type(torch.FloatTensor)

    images, masks = images.to(config['device']), masks.to(config['device'])

    output = model(images)  # forward
    
    if "blazeneo" in config['backbone']:
        output = output[1]
    elif "neounet" in config['backbone']:
        output = output[0]
    elif "pranet" in config['backbone']:
        output, output4, output3, output2  = output
        # output5, output4, output3, output  = output
    elif "deeplabv3" in config['backbone']:
        output = output['out']
    elif "polyp_pvt" in config['backbone']:
        P1, P2 = output
        res = F.interpolate(P1 + P2 , size=(config['image_height'], config['image_width']), mode='bilinear', align_corners=False)
        output = res
    elif "polyp_pvt_p1" == config['backbone']:
        P1, P2 = output
        output = P1
    elif "polyp_pvt_p2" == config['backbone']:
        P1, P2 = output
        output = P2
        
    if config['loss_function'] == 'CrossEntropy_TverskyLoss':
        ce_loss = criterion[0](output, masks)
        tversky_loss = criterion[1](output, masks)
        loss = (ce_loss+tversky_loss)/2
    elif config['loss_function'] == 'ActiveContourLoss':
        soft_output = torch.softmax(output, dim=1)
        loss = criterion(soft_output, masks)
    elif config['loss_function'] == 'CE_DiceLoss': 
        ce_loss = criterion[0](output, masks)
        dice_loss = criterion[1](output, masks)
        
        loss = 0.4 * ce_loss + 0.6 * dice_loss
    elif config['loss_function'] == 'FocalDiceLoss':
        focal_loss = criterion[0](output, masks)
        dice_loss = criterion[1](output, masks)
        w1, w2 = config['weights']
        
        loss = w1 * focal_loss + w2 * dice_loss
    elif config['loss_function'] == 'PraNetLoss': 
        loss5 = criterion(output5, masks, config['weights'])
        loss4 = criterion(output4, masks, config['weights'])
        loss3 = criterion(output3, masks, config['weights'])
        loss2 = criterion(output, masks, config['weights'])
        loss = loss2 + loss3 + loss4 + loss5
    elif "polyp_pvt" == config['backbone']:
        loss_P1 = criterion(P1, masks)
        loss_P2 = criterion(P2, masks)
        
        loss = loss_P1 + loss_P2
    else:
        loss = criterion(output, masks)
    
    probs = torch.softmax(output, dim=1)
    predictions = torch.argmax(probs, dim=1)
    F1_score = compute_F1(predictions, masks)
    IoU_score = compute_IoU(predictions, masks)

    return loss, F1_score, IoU_score

def tga_evaluate(sample, config, model, criterion):
    images, text_classes = sample['image'], sample['text_classes']
    masks, num_polyps, polyp_sizes = sample['mask'], sample['num_polyps'], sample['polyp_sizes']
    
    images, text_classes = images.to(config['device']), text_classes.to(config['device'])
    masks, num_polyps, polyp_sizes = masks.to(config['device']), num_polyps.to(config['device']), polyp_sizes.to(config['device'])

    p1, p2, p3 = model(images, text_classes)  # forward
    p2 = torch.softmax(p2, dim=1)
    p3 = torch.softmax(p3, dim=1)
            
    loss_P1 = criterion(p1, masks)
    loss_P2 = criterion(p2, num_polyps)
    loss_P3 = criterion(p3, polyp_sizes)
    
    loss = (loss_P1 + loss_P2 + loss_P3).mean()
    
    output = p1
    probs = torch.softmax(output, dim=1)
    predictions = torch.argmax(probs, dim=1)
    F1_score = compute_F1(predictions, masks)
    IoU_score = compute_IoU(predictions, masks)

    return loss, F1_score, IoU_score

def evaluate(config, model, validation_loader, criterion):
    epoch_loss = 0

    model.eval()
    F1_list = []
    IoU_list = []
    
    with torch.no_grad():
        for sample in tqdm(validation_loader, desc="Evaluating", leave=False):
            if "tganet" in config['backbone']:
                loss, F1_score, IoU_score = tga_evaluate(sample, config, model, criterion)
            else:
                loss, F1_score, IoU_score = default_evaluate(sample, config, model, criterion)
                
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