import pdb
import argparse
from utils.train_utils import apply_transform, prepare_dataloaders, prepare_objectives, prepare_architecture
import wandb
from torchsummary import summary
import yaml
import torch
from trainer import train_and_evaluate
from utils.helper import get_args_parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "NeoPolyp training script", parents=[get_args_parser()])
    args = parser.parse_args()

    # load yaml file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(config)

    train_transform, valid_transform = apply_transform(config)
    training_loader, validation_loader = prepare_dataloaders(
        config, train_transform, valid_transform)
    
    model = prepare_architecture(config)
    model.to(config['device'])

    criterion, optimizer, scheduler = prepare_objectives(config, model, training_loader)

    if config['continue_training']:
        model.load_state_dict(torch.load(
            config['trained_weights'])['model_state_dict'])
        epoch = torch.load(config['trained_weights'])['epoch']
    else:
        epoch = 0

    wandb.init(project="neopolyp", entity="_giaabaoo_", config=config)
    wandb.watch(model)
    # print(summary(model, input_size=(3, 512, 512)))
    train_and_evaluate(training_loader, validation_loader,
                       model, criterion, optimizer, scheduler, config, epoch)
