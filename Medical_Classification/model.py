import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm.optim
import torch.optim as optim
from loss import FocalLoss
from metrics import accuracy, f1_score


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # layer 1 (b, 1*28*28) -> (b, 128)
        x = self.layer_1(x)
        x = torch.relu(x)

        # layer 2 (b, 128) -> (b, 256)
        x = self.layer_2(x)
        x = torch.relu(x)

        # layer 3 (b, 256) -> (b, 10)
        x = self.layer_3(x)

        return x


class Classifier(pl.LightningModule):
    def __init__(self, model, num_classes, learning_rate):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        image, y_true = batch
        y_pred = self.model(image)
        loss_focal = FocalLoss(
            self.device, self.class_weight, self.num_classes)(y_true, y_pred)
        acc, f1 = accuracy(y_true, y_pred), f1_score(y_true, y_pred)
        return loss_focal, acc, f1

    def training_step(self, batch):
        loss, acc, f1 = self._step(batch)
        metrics = {"loss": loss, "train_acc": acc, "train_f1": f1}
        self.log_dict(metrics, on_step=True,
                      on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        loss, acc, f1 = self._step(batch)
        metrics = {"test_loss": loss, "test_acc": acc, "test_f1": f1}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def test_step(self, batch):
        loss, acc, f1 = self._step(batch)
        metrics = {"test_loss": loss, "test_acc": acc, "test_f1": f1}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def predict_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        return y_hat.cpu().numpy()

    def configure_optimizers(self):
        optimizer = timm.optim.Nadam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                         factor=0.5, patience=20, verbose=True)
        lr_schedulers = {"scheduler": scheduler, "monitor": "loss"}
        return [optimizer], lr_schedulers
