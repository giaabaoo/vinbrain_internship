# vinbrain_internship

## TODO TASKS:
- Experiment on other losses
- Data augmentation

## DONE:
#### 1. Prepare data
- Retrieve data 
- Understand competition structure —> Preprocess dicoms —> Pngs (+ Bit inversion if needed)  —> Train test split (image folders + json labels to handle multi-annotated images)
- Visualize masks
- Resample balanced data

#### 2. Build a full pipeline 
- DataLoader: {image, mask}
- UNet model + pretrained resnet34
- Resize 512x512
- Logging tracks with wandb
- Training and evaluation (on resize) using dice score and loss
- Inference module
- Submission module

#### 3. Others
- Experiments on focal loss with resample and full dataset
- Investigating model low performance
- Final evaluation on full size image
