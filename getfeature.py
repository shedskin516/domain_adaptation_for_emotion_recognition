import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
from PIL import Image
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch import flatten
import argparse


#Aff-Wild2
df = pd.read_csv('data_aff.csv')
dir = "../Face-Warping-Emotion-Recognition-main/data/Aff-Wild2/cropped_aligned/"
#SEWA
# df = pd.read_csv('data_sewa.csv')
# dir = "../Face-Warping-Emotion-Recognition-main/data/SEWA/prep_SEWA/"

num_total = 1250
train_percent = 0.8
num_train = int(num_total * train_percent)

df = df.sample(frac=1)

class train_data(Dataset):
    def __init__(self):
        df_sub = df[:num_train]
        files = dir + df_sub['file_path']
        self.filenames = files.values.tolist()
        self.labels = df_sub[['valence', 'arousal']].values.tolist()
        print(len(self.filenames))

    def __len__(self):
        return len(self.filenames)
   
    def __getitem__(self, idx):
        img = Image.open(self.filenames[idx])
        img = img.resize((112,112))
        entry = np.asarray(img, dtype=np.float32)
        entry = (entry - np.mean(entry))/np.std(entry)
        entry = torch.tensor(entry)
        entry = entry.permute(2,0,1)
        label = self.labels[idx]
        label = np.asarray(label, dtype=np.float32)

        return entry, label

class val_data(Dataset):
    def __init__(self):
        df_sub = df[num_train:num_total]
        files = dir + df_sub['file_path']
        self.filenames = files.values.tolist()
        self.labels = df_sub[['valence', 'arousal']].values.tolist()
        print(len(self.filenames))

    def __len__(self):
        return len(self.filenames)
   
    def __getitem__(self, idx):
        img = Image.open(self.filenames[idx])
        img = img.resize((112,112))
        entry = np.asarray(img, dtype=np.float32)
        entry = (entry - np.mean(entry))/np.std(entry)
        entry = torch.tensor(entry)
        entry = entry.permute(2,0,1)
        label = self.labels[idx]
        label = np.asarray(label, dtype=np.float32)
        return entry, label

class Net(pl.LightningModule):
    def __init__(self, numChannels=3):
        super().__init__()
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--dataset', default='aff')
    parser.add_argument('--batch', type=int, default='50')
    opt = parser.parse_args()
    print(opt)

    data_train = train_data()
    train_loader = DataLoader(data_train, batch_size=opt.batch, shuffle=True)

    data_val = val_data()
    val_loader = DataLoader(data_val, batch_size=opt.batch, shuffle=False)

    net = Net()

    if opt.train:
        print("Train Begin...")
        trainer = pl.Trainer(max_epochs=10)
        trainer.fit(model=net, train_dataloaders=train_loader, val_dataloaders=val_loader)
        torch.save(net.state_dict(), 'net.pth')
        print("Train End...")
    else:
        print("Load Modal...")
        net.load_state_dict(torch.load("net.pth"))


    features = torch.empty(0, 128)
    with torch.no_grad():
        for batch in iter(train_loader):
            out = net.embedding(batch[0])
            print(out[0][0])
            features = torch.cat((features, out), dim=0)

    features = features.detach().numpy()
    print(features.shape)
    np.save('features.npy', features)


if __name__ == '__main__':
	main()