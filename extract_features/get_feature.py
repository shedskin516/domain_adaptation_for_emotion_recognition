import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import argparse
from net import ResNetFC 

# Aff-Wild2
df = pd.read_csv('../mini_datasets/Aff-Wild2/train.csv')
dir = "../data/Aff-Wild2/cropped_aligned/"

# #SEWA
# df = pd.read_csv('mini_datasets/SEWA/train.csv')
# dir = "../Face-Warping-Emotion-Recognition-main/data/SEWA/prep_SEWA/"
# df = df[:20000]


class all_data(Dataset):
    def __init__(self):
        files = dir + df['file_path']
        self.filenames = files.values.tolist()
        self.labels = df[['valence', 'arousal']].values.tolist()
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

    data = all_data()
    data_loader = DataLoader(data, batch_size=opt.batch, shuffle=False)

    net = ResNetFC()

    print("Load Modal...")
    net.load_state_dict(torch.load("net.pth"))

    print("Getting Features...")
    features = torch.empty(0, 128)
    idx = 1
    with torch.no_grad():
        for batch in iter(data_loader):
            out = net.embedding(batch[0])
            # print(out[0][0])
            print(idx)
            idx += 1
            features = torch.cat((features, out), dim=0)

    features = features.detach().numpy()
    print(features.shape)
    np.save('features_aff.npy', features)


if __name__ == '__main__':
	main()