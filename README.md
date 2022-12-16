# vinbrain_internship

## TODO TASKS:
- Train and evaluate on positive samples 


## Results table
#### U-Net + pre-trained ResNet34 (10 epochs)
Loss functions | Notes | Dice score on private test |
--- | --- | :---: |
Weighted focal loss | Trained on re-sampled dataset |  0.5328 |
Weighted focal loss | Trained on full dataset | 0.7487 |
Dice loss | Trained on full dataset | 0.774 |
Dice BCE loss | Trained on full dataset | 0.774 |

####  U-Net + pre-trained EfficientNet (10 epochs)
Loss functions | Notes | Dice score on private test |
--- | --- | :---: |
Weighted focal loss | Trained on re-sampled dataset |  0.774 |

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
- Experiments on focal loss with resampled and full dataset
- Investigating model low performance
- Final evaluation on full size image
- Calculate macros dice scores


## PLAN
- Experiment on other losses
- Experiment on other architectures
- Data augmentation