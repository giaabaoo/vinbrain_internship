# vinbrain_internship

## TODO TASKS:
- Experiment on other architectures using other features

## Results table
#### Architecture + Loss functions
Architecture | Loss functions | Validation dice score | Private test dice score | Notes
--- | --- | :---: | :---: | --- |
**UNet** + Resnet34 | Cross Entropy Loss | 0.85 | 0.73 | Resize at 512 x 512
UNet + SE-Resnext101 | Focal Loss | 0.86 | 0.75 | Resize at 512 x 512
UNet + SE-Resnext101 | Focal Loss | 0.88 | 0.78 | Resize at 480 x 640
UNet + SE-Resnext101 | Active Contour Loss | 0.9 | 0.78 | Resize at 480 x 640
UNet + Efficient-net B4 | Focal Loss | 0.8 | 0.76 | Resize at 512 x 512
**DoubleUNet** | Cross Entropy Loss| 0.76 | 0.61 | Resize at 480 x 640
**UNet++** + SE-Resnext101 | Focal Loss | 0.81 | 0.71 | Resize at 480 x 640
**BlazeNeo** | Cross Entropy Loss | 0.89 | 0.76 | Resize at 480 x 640
BlazeNeo | Focal Loss | 0.86 | 0.75 | Resize at 480 x 640
BlazeNeo | Cross Entropy Loss | 0.85 | 0.73 | Upsampling with original size 352x352
**NeoUNet** | Cross Entropy Loss | 0.88 | 0.79 | Resize at 480 x 640
NeoUNet | Tversky Loss | 0.88 | 0.76 | Resize at 480 x 640a
NeoUNet | CE + Tversky Mean Loss| 0.86 | 0.72 | Resize at 480 x 640
NeoUNet | Active Contour Loss | 0.84 | 0.77 | Resize at 480 x 640
NeoUNet | Cross Entropy Loss | 0.82 | 0.7 | Upsampling with original size 352x352
**HarDNet** | Cross Entropy Loss | 0.88 | 0.8 (0.78) | Resize at 480 x 640
HarDNet upsample | Cross Entropy Loss | 0.87 | 0.76 | Upsampling with original size 352x352
**Focal UNet** | Cross Entropy Loss | 0.76 | 0.6 | Resize at 224 x 224
Focal UNet | Focal Loss | 0.77 | 0.6 | Continue training

## DONE:
#### 1. Prepare data
- Retrieve data 
- Understand competition structure —> Train test split into image sub-folders
- Visualize masks

#### 2. Build a full pipeline 
- DataLoader: generate {image, mask} pairs by eroding and binarizing mask's labels + data augmentation
- Upsample the input with different image sizes
- Baseline UNet model + pretrained resnet34
- Resize 512x512
- Logging tracks with wandb
- Training and evaluation using dice score and cross-entropy loss
- Inference module
- Submission module

#### 3. Others
- Final evaluation on full size image with F1 score and IoU score


## PLAN
- Experiment on other losses
