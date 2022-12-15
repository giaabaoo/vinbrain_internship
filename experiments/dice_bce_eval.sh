cd ..
python3 evaluate.py --config ./configs/dice_bce.yaml
# python3 submit.py --config ./configs/dice_bce.yaml
# kaggle competitions submit -c siim-acr-pneumothorax-segmentation -f submission.csv -m "dice_bce 10 epochs"