import pytorch_lightning as pl

import torch
import torch.nn as nn
from torch import optim
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch import flatten
import torchvision.models as models
import torch.nn.functional as F

    
class ResNetFC(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.fc1 = Linear(in_features=1000, out_features=256)
        self.relu1 = ReLU()

        self.fc2 = Linear(in_features=256, out_features=128)
        self.relu2 = ReLU()

        self.final = Linear(in_features=128, out_features=2)

    def embedding(self, x):
        x = self.resnet(x)
        x = self.fc1(x)
        x = self.relu1(x)
        out = self.fc2(x)
        return out

    def forward(self, x):
        x = self.embedding(x)
        x = self.relu2(x)
        out = self.final(x)
        return out
    
    def training_step(self, batch):
        x, y = batch
        y_pred = self(x)
        l = nn.MSELoss()
        loss = l(y_pred, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            y_pred = self(x)
        l = nn.MSELoss()
        loss = l(y_pred, y)
        self.log("val_loss",loss, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class CNNNet(pl.LightningModule):
    def __init__(self, numChannels=3):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=20, kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = Conv2d(in_channels=50, out_channels=50, kernel_size=(5, 5))
        self.relu3 = ReLU()

        self.fc1 = Linear(in_features=22050, out_features=2048)
        self.relu4 = ReLU()

        self.fc2 = Linear(in_features=2048, out_features=128)
        self.relu5 = ReLU()

        self.final = Linear(in_features=128, out_features=2)

    def embedding(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu4(x)

        out = self.fc2(x)
        return out

    def forward(self, x):
        x = self.embedding(x)
        x = self.relu5(x)
        out = self.final(x)
        return out
    
    def training_step(self, batch):
        x, y = batch
        y_pred = self(x)
        l = nn.MSELoss()
        loss = l(y_pred, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            y_pred = self(x)
        l = nn.MSELoss()
        loss = l(y_pred, y)
        self.log("val_loss",loss, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
