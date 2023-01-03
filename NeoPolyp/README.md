# vinbrain_internship

## TODO TASKS:
- Upsample the input with different image sizes

## Results table
#### Architecture + Loss functions
Architecture | Loss functions | Validation dice score | Private test dice score | Notes
--- | --- | :---: | :---: | --- |
UNet + Resnet34 | Cross Entropy Loss | 0.85 | 0.73 | Resize at 512 x 512
UNet + SE-Resnext101 | Focal Loss | 0.86 | 0.75 | Resize at 512 x 512
UNet + SE-Resnext101 | Focal Loss | 0.88 | 0.78 | Resize at 480 x 640
UNet + SE-Resnext101 | Active Contour Loss | - | - | Resize at 480 x 640
UNet + Efficient-net B4 | Focal Loss | 0.8 | 0.76 | Resize at 512 x 512

BlazeNeo | Cross Entropy Loss | 0.89 | 0.76 | Resize at 480 x 640
BlazeNeo | Focal Loss | 0.86 | 0.75 | Resize at 480 x 640

NeuUNet | Cross Entropy Loss | 0.88 | 0.79 | Resize at 480 x 640
NeoUNet | Tversky Loss | 0.88 | 0.76 | Resize at 480 x 640
NeoUNet | CE + Tversky Mean Loss| 0.86 | 0.72 | Resize at 480 x 640
NeoUNet | Active Contour Loss | - | - | Resize at 480 x 640

DoubleUNet | Cross Entropy Loss| 0.76 | 0.61 | Resize at 480 x 640
UNet++ + SE-Resnext101 | Focal Loss | 0.81 | 0.71 | Resize at 480 x 640

## DONE:
#### 1. Prepare data
- Retrieve data 
- Understand competition structure —> Train test split into image sub-folders
- Visualize masks

#### 2. Build a full pipeline 
- DataLoader: generate {image, mask} pairs by eroding and binarizing mask's labels + data augmentation
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
- Experiment on other architectures