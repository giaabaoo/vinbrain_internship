cd ../
python train.py --config ./configs/efficient_net.yaml
# python3 submit.py --config ./configs/efficient_net.yaml
# kaggle competitions submit -c siim-acr-pneumothorax-segmentation -f submission.csv -m "mixed loss on full dataset 10 epochs"