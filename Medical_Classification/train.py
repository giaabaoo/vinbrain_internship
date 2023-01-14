import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from dataset import Dataset
from pytorch_lightning.loggers import WandbLogger
from model import Net, Classifier
from utils import get_args_parser
from pathlib import Path
import yaml
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "NeoPolyp training script", parents=[get_args_parser()])
    args = parser.parse_args()

    # load yaml file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(config)

    train_dataloader = DataLoader(Dataset(config['train_image_path'], config['train_label_path']), batch_size=config['batch_size'])
    valid_dataloader = DataLoader(Dataset(config['valid_image_path'], config['valid_label_path']), batch_size=config['batch_size'])
    
    model = Net()
    classifier = Classifier(model=model, num_classes=len(config["classes"]), learning_rate=config["learning_rate"])
    Path("./checkpoints/").mkdir(parents=True, exist_ok=True)
    check_point = pl.callbacks.model_checkpoint.ModelCheckpoint("./checkpoints/", filename=config['saved_checkpoint'],
                                                            monitor="loss", mode="min", save_top_k=1)
    
    wandb_logger = WandbLogger(project="medical_cls")
    PARAMS = {"accelerator": 'gpu', "devices": 1, "benchmark": True, "enable_progress_bar": True,
              "logger": wandb_logger,
              "callbacks": [check_point],
              "log_every_n_steps": 1, "num_sanity_val_steps": 2, "max_epochs": 50,
              }

    trainer = pl.Trainer(**PARAMS)

    trainer.fit(classifier, train_dataloader, valid_dataloader)