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
UNet + ResNeSt200e | Cross Entropy Loss | 0.87 | 0.73 | Resize at 480 x 640
**DoubleUNet** | Cross Entropy Loss| 0.76 | 0.61 | Resize at 480 x 640
**UNet++** + SE-Resnext101 | Focal Loss | 0.81 | 0.71 | Resize at 480 x 640
UNet++ + ResNeSt200e | Cross Entropy Loss | 0.87 | 0.78 | Resize at 480 x 640
**BlazeNeo** | Cross Entropy Loss | 0.89 | 0.76 | Resize at 480 x 640
BlazeNeo | Focal Loss | 0.86 | 0.75 | Resize at 480 x 640
BlazeNeo | Cross Entropy Loss | 0.85 | 0.73 | Upsampling with original size 352x352
**NeoUNet** | Cross Entropy Loss | 0.88 | 0.79 | Resize at 480 x 640
NeoUNet | Tversky Loss | 0.88 | 0.76 | Resize at 480 x 640a
NeoUNet | CE + Tversky Mean Loss| 0.86 | 0.72 | Resize at 480 x 640
NeoUNet | Active Contour Loss | 0.84 | 0.77 | Resize at 480 x 640
NeoUNet | Cross Entropy Loss | 0.82 | 0.7 | Upsampling with original size 352x352
**HarDNet** | Cross Entropy Loss | 0.88 | 0.8 (0.78) | Resize at 480 x 640
HarDNet | Cross Entropy Loss | 0.87 | 0.76 | Upsampling with original size 352x352
HarDNet | Cross Entropy Loss | 0.85 | 0.77 | Continue training at 960 x 1280
**Focal UNet** | Cross Entropy Loss | 0.76 | 0.6 | Resize at 224 x 224
Focal UNet | Focal Loss | 0.77 | 0.6 | Continue training
**DeepLabV3** Resnet101 | Cross Entropy Loss | 0.86 | 0.78 | Resize at 480 x 640
**PraNet** | Cross Entropy Loss | 0.88 | 0.824 | Resize at 480 x 640, first feature map
PraNet | Cross Entropy Loss | 0.89 | 0.825 | Continue training, first feature map
PraNet | Active Contour Loss | 0.9 | 0.823 | Continue training with best CE ckpt, first feature map
PraNet | Active Contour Loss | 0.91 | 0.84 | Continue training with best CE ckpt, first feature map, postprocess
PraNet | Active Contour Loss | 0.893 | 0.78 | Continue training with best CE AC ckpt 400, last feature map + color jitter
PraNet | Cross Entropy Loss | 0.895 | 0.795 | Continue training with best CE ckpt + color transfer 400, first feature map
PraNet | Cross Entropy Loss | 0.88 | 0.838 | Continue training with best CE ckpt + color transfer 400, first feature map
PraNet | Cross Entropy Loss | 0.911 | 0.856 | Continue training with best CE ckpt + color transfer 400, first feature map, postprocess
PraNet | Focal Loss | 0.88 | 0.79 | Start training with color transfer 400, first feature map, postprocess
PraNet | Focal Loss + Dice Loss | - | - | Start training with color transfer 400, first feature map, postprocess
PraNet Res2Net101 | Cross Entropy Loss | 0.877 | 0.794 | Start training with color transfer 400, last feature map, with color jitter
PraNet Res2Net101 | Cross Entropy Loss | 0.892 | 0.817 | Start training with color transfer 400, first feature map, wo color jitter 
PraNet Res2Net101 | Cross Entropy Loss | 0.892 |0.803 | Continue training with color transfer 400, first feature map, with brightness jitter, postprocess
PraNet Res2Net152 | Cross Entropy Loss | 0.888 | 0.78 | Start training with color transfer 400, first feature map, wo color jitter
PraNet | Cross Entropy Loss | 0.902 | 0.788 | Continue training with best CE ckpt + color transfer 600, last feature map + color jitter
PraNet | Cross Entropy Loss | 0.888 | 0.81 | Continue training with best CE 400 ckpt (0.838)+ color transfer 400, last feature map
PraNet | Multi-Maps Cross Entropy Loss | 0.87 | 0.775 | Resize at 480 x 640, start training with color transfer 400, first feature map, compute loss across 4 feature maps (original paper)
PraNet | Cross Entropy Loss | 0.871 | 0.77 | Resize at 960 x 1280, first feature map
PraNet | CE + Dice Loss | 0.916 | 0.83 | Resize at 480 x 640, start training with color transfer 400, last feature map, compute loss across 4 feature maps (original paper)
PraNet | CE + Dice Loss | 0.917 | 0.825 | Continue training with color transfer 400, last feature map, compute loss across 4 feature maps (original paper)
PraNet | CE + Dice Loss | 0.905 | 0.8 | Continue training with color transfer 400, last feature map, compute loss across 4 feature maps (original paper), color jittering
PraNet | CE + Dice Loss | 0.917 | 0.82 | Continue resize at 480 x 640, start training with color transfer 400, first feature map, compute loss across 4 feature maps (original paper)
PraNet | CE + Dice Loss | 0.93 | 0.845 | Continue resize at 480 x 640, start training with color transfer 400, first feature map, compute loss across 4 feature maps (original paper), postprocess
PraNet | 0.4 * CE + 0.6 * Dice Loss | 0.885 (0.912) | 0.786 (0.812)| Start at 480 x 640, start training with color transfer 600, last feature map, compute loss across 4 feature maps (original paper) (post-process)
PraNet | Cross Entropy Loss | 0.87 | 0.78 | Start training with color transfer 400, last feature map
PraNet | Cross Entropy Loss | 0.834 | 0.73 | Start training with color transfer 400 + size upsampling, first feature map
**SANet** | Cross Entropy Loss | 0.87 | 0.76 | Resize at 480 x 640
SANet | Cross Entropy Loss | 0.904 | 0.814 | Continue training with best CE ckpt + color transfer
SANet | Cross Entropy Loss | 0.88 | 0.775 | Continue training with best CE ckpt + color transfer 400

For more details, refer to this [link](https://docs.google.com/spreadsheets/d/1iSvGNBwq1OicNJdAsUYEsGnScmvbrIdsWGNe16BHItU/edit?usp=sharing)

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
