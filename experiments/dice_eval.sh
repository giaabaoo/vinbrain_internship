cd ..
python submit.py --config ./configs/dice_loss.yaml
kaggle competitions submit -c siim-acr-pneumothorax-segmentation -f submission.csv -m "dice loss 10 epochs"