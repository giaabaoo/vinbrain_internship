from torch.utils.data import DataLoader
from albumentations import Compose, Resize, Normalize, HorizontalFlip, ShiftScaleRotate, VerticalFlip, RandomRotate90, ColorJitter, RandomBrightness, Sharpen
from albumentations.pytorch import ToTensorV2
from dataset import NeoPolyp, TGA_NeoPolyp
from torch import optim, nn
import segmentation_models_pytorch as smp
import timm.optim
from .loss import NeoUNetLoss, ActiveContourLoss, PraNetLoss, MultiCELoss
import cv2
from networks.unet import UNet
from networks.blazeneo.model import BlazeNeo
from networks.neounet.model import NeoUNet
from networks.doubleunet.model import DUNet
from networks.FocalNet.main import FUnet
from networks.hardnetmseg.model import HarDNetMSEG
from networks.pranet.model import PraNet
from networks.sanet.model import SANet
from networks.polyp_pvt.pvt import PolypPVT
from networks.swin.vision_transformer import SwinUnet
from networks.tganet.model import TGAPolypSeg
from torchvision.models.segmentation import deeplabv3_resnet101

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

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
            VerticalFlip(p=0.5),
            # RandomBrightness(p=0.5),
            # Sharpen(p=0.5),
            # ColorJitter(brightness=0.3, contrast=0, saturation=0, hue=0, p=0.5),
            # ShiftScaleRotate(
            #     shift_limit=0,  # no resizing
            #     scale_limit=0.1,
            #     rotate_limit=10, # rotate
            #     p=0.5,
            #     border_mode=cv2.BORDER_CONSTANT
            # ),
            ToTensorV2(),
        ])
    else:
        train_transform = valid_transform
        
    return train_transform, valid_transform

def prepare_dataloaders(config, train_transform, valid_transform):
    if "tganet" in config['backbone']:
        training_data = TGA_NeoPolyp(config['root_train_image_path'], config['root_train_label_path'], transform=train_transform)
        validating_data = TGA_NeoPolyp(config['root_valid_image_path'], config['root_valid_label_path'], transform=valid_transform)
    else:
        training_data = NeoPolyp(config['root_train_image_path'], config['root_train_label_path'], transform=train_transform)
        validating_data = NeoPolyp(config['root_valid_image_path'], config['root_valid_label_path'], transform=valid_transform)
    
    training_loader = DataLoader(training_data, batch_size=config['batch_size'], shuffle=True)
    validation_loader = DataLoader(validating_data, batch_size=config['batch_size'], shuffle=True)
    
    return training_loader, validation_loader

def prepare_objectives(config, model, training_loader):
    # define loss function
    if config['loss_function'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    elif config['loss_function'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif config['loss_function'] == 'smpDiceLoss':
        criterion = smp.losses.DiceLoss(mode='multiclass')
    elif config['loss_function'] == 'smpFocalLoss':
        criterion = smp.losses.FocalLoss(gamma = 2, mode='multiclass')
    elif config['loss_function'] == 'smpTverskyLoss':
        criterion = smp.losses.TverskyLoss(mode='multiclass')
    elif config['loss_function'] == 'CrossEntropy_TverskyLoss':
        criterion = [nn.CrossEntropyLoss(), smp.losses.TverskyLoss(mode='multiclass')]
    elif config['loss_function'] == 'NeoUNetLoss':
        criterion = NeoUNetLoss()
    elif config['loss_function'] == 'ActiveContourLoss':
        criterion = ActiveContourLoss(config['device'])
    elif config['loss_function'] == 'CE_DiceLoss':
        criterion = [nn.CrossEntropyLoss(), smp.losses.DiceLoss(mode='multiclass')]
    elif config['loss_function'] == 'PraNetLoss':
        criterion = PraNetLoss()
    elif config['loss_function'] == 'FocalDiceLoss':
        criterion = [smp.losses.FocalLoss(gamma = 2, mode='multiclass'), smp.losses.DiceLoss(mode='multiclass')]
    elif config['loss_function'] == 'MultiCELoss':
        criterion = MultiCELoss()
        
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
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = len(training_loader), verbose=True)
    elif config['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = len(training_loader), T_mult = config['T_mult'], verbose=True)

    return criterion, optimizer, scheduler

def prepare_architecture(config):
    if "unetplusplus" in config['backbone'].split("."):
        model = smp.UnetPlusPlus(config['backbone'].split(".")[1], encoder_weights=config['encoder_weights'],
                         in_channels=3, classes=len(config['classes']), activation=config['activation'])
    elif "unet." in config['backbone']:
        model = smp.Unet(config['backbone'].split(".")[1], encoder_weights=config['encoder_weights'],
                         in_channels=3, classes=len(config['classes']), activation=config['activation'])
    elif "blazeneo" in config['backbone']:
        model = BlazeNeo()
    elif "neounet" in config['backbone']:
        model = NeoUNet(num_classes=3)
    elif "doubleunet" in config['backbone']:
        model = DUNet()
    elif "focalunet" in config['backbone']:
        model = FUnet()
    elif "hardnet" in config['backbone']:
        model = HarDNetMSEG()
    elif "deeplabv3_resnet101" in config['backbone']:
        model = deeplabv3_resnet101(num_classes=len(config['classes']))
    elif "pranet" in config['backbone']:
        model = PraNet()
    elif "sanet" in config['backbone']:
        model = SANet(num_classes=len(config['classes']))
    elif "polyp_pvt" in config['backbone']:
        model = PolypPVT(num_classes=len(config['classes']))
    elif "swin" in config['backbone']:
        model = SwinUnet(config['image_height'], num_classes=len(config['classes']))
    elif "tganet" in config['backbone']:
        model = TGAPolypSeg(num_classes=len(config['classes']))
    elif config['backbone'] != "None":
        model = smp.Unet(config['backbone'], encoder_weights=config['encoder_weights'],
                         in_channels=3, classes=len(config['classes']), activation=config['activation'])
    else:
        model = UNet(n_channels=3, n_classes=3)
        
    return model