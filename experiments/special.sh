# sigmoid predictions --> change to 0 and 1 only --> calculate loss with labels
cd ../
# python train.py --config ./configs/special.yaml
# python submit.py --config ./configs/special.yaml
# kaggle competitions submit -c siim-acr-pneumothorax-segmentation -f submission.csv -m "sigmoid and convert before loss"