from torch.utils.data import DataLoader
from albumentations import Compose, Resize, Normalize, HorizontalFlip, ShiftScaleRotate
from albumentations.pytorch import ToTensorV2
from dataset import NeoPolyp
from torch import optim, nn
import segmentation_models_pytorch as smp
import timm.optim
from .loss import NeoUNetLoss, ActiveContourLoss
import cv2

def apply_transform(config):
    valid_transform = Compose([
            Resize(config['image_height'], config['image_width']),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
            ToTensorV2(),
        ])
    
    if config['transform']:
        train_transform = Compose([
            Resize(config['image_height'], config['image_width']),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(
                shift_limit=0,  # no resizing
                scale_limit=0.1,
                rotate_limit=10, # rotate
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT
            ),
            ToTensorV2(),
        ])
    else:
        train_transform = valid_transform
        
    return train_transform, valid_transform

def prepare_dataloaders(config, train_transform, valid_transform):
    training_data = NeoPolyp(config['root_train_image_path'], config['root_train_label_path'], transform=train_transform)
    validating_data = NeoPolyp(config['root_valid_image_path'], config['root_valid_label_path'], transform=valid_transform)
    
    training_loader = DataLoader(training_data, batch_size=config['batch_size'], shuffle=True)
    validation_loader = DataLoader(validating_data, batch_size=config['batch_size'], shuffle=True)
    
    return training_loader, validation_loader

def prepare_objectives(config, model):
    # define loss function
    if config['loss_function'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    elif config['loss_function'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif config['loss_function'] == 'smpDiceLoss':
        criterion = smp.losses.DiceLoss(mode='multiclass')
    elif config['loss_function'] == 'smpFocalLoss':
        criterion = smp.losses.FocalLoss(mode='multiclass')
    elif config['loss_function'] == 'smpTverskyLoss':
        criterion = smp.losses.TverskyLoss(mode='multiclass')
    elif config['loss_function'] == 'CrossEntropy_TverskyLoss':
        criterion = [nn.CrossEntropyLoss(), smp.losses.TverskyLoss(mode='multiclass')]
    elif config['loss_function'] == 'NeoUNetLoss':
        criterion = NeoUNetLoss()
    elif config['loss_function'] == 'ActiveContourLoss':
        criterion = ActiveContourLoss(config['device'])

    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])
    elif config['optimizer'] == 'Nadam':
        optimizer = timm.optim.Nadam(model.parameters(), lr = config['learning_rate'])
    
    if config['scheduler'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)
    elif config['scheduler'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config['T_max'], verbose=True)

    return criterion, optimizer, scheduler