import wandb

wandb.init(project="pneumothorax", entity="_giaabaoo_")
wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 128
}
wandb.log({"loss": 123})

# Optional
# wandb.watch(model)