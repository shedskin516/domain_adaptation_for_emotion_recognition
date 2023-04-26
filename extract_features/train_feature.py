import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import argparse
from net import ResNetFC 

df = pd.read_csv('../mini_datasets/Aff-Wild2/train.csv')
dir = "../data/Aff-Wild2/cropped_aligned/"
num_total = len(df)
train_percent = 0.9
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default='1000')
    opt = parser.parse_args()
    print(opt)

    data_train = train_data()
    train_loader = DataLoader(data_train, batch_size=opt.batch, shuffle=True)

    data_val = val_data()
    val_loader = DataLoader(data_val, batch_size=opt.batch, shuffle=False)

    net = ResNetFC()

    print("Train Begin...")
    trainer = pl.Trainer(max_epochs=50)
    trainer.fit(model=net, train_dataloaders=train_loader, val_dataloaders=val_loader)
    torch.save(net.state_dict(), 'net.pth')
    print("Train End...")


if __name__ == '__main__':
	main()