import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys
sys.path.append('../model/')
import argparse
import time
import math
import pandas
import numpy as np
import random

import torch
import torch.nn as nn
from torch import optim
from PIL import Image
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from losses import SupConLoss
from distance import get_pairs
from model import Encoder

source_file_path = '../mini_datasets/Aff-Wild2/train.csv'
target_file_path = '../mini_datasets/SEWA/train.csv'
batch_size = 256
# mini_batch = 10
output_dim = 128
temp = 0.07
num_epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_clusters = 10

class train_data(Dataset):
    def __init__(self):
        source_file = pandas.read_csv(source_file_path)
        total_source_file = source_file['file_path'].tolist()
        self.source_file_path = total_source_file[:20000]

        target_file = pandas.read_csv(target_file_path)
        total_target_file = target_file['file_path'].tolist()
        self.target_file_path = total_target_file[:20000]

        transform_list = [  transforms.Resize(112),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                            transforms.RandomErasing(scale=(0.02,0.25))]
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.source_file_path)
   
    def __getitem__(self, idx):
        source_img = Image.open('../data/Aff-Wild2/cropped_aligned/'+self.source_file_path[idx])
        target_img = Image.open('../data/SEWA/prep_SEWA/'+self.target_file_path[idx])

        simg = self.transform(source_img)
        timg = self.transform(target_img)

        return simg, timg

def set_loader():
    data_train = train_data()
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    return train_loader

def set_model():
    model = Encoder(output_dim=output_dim)
    criterion = SupConLoss(temperature=temp)
    model = model.to(device)
    return model, criterion

def set_optimizer(model):
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    return optimizer

def train(train_loader, model, criterion, optimizer, scheduler):
    avg_loss = 0
    for index, (simg, timg) in enumerate(train_loader):
        simg, timg = simg.to(device), timg.to(device)
        sfeature = model(simg)
        tfeature = model(timg)
        source_labels, target_labels = get_pairs(sfeature, tfeature, n_clusters)

        sfeature = sfeature.reshape(len(simg), 1, output_dim)
        tfeature = tfeature.reshape(len(timg), 1, output_dim)

        concat_features = torch.cat((sfeature, tfeature), dim=0)

        labels = np.concatenate((source_labels, target_labels))

        loss = criterion(concat_features, torch.from_numpy(labels))
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        avg_loss += loss.item()

    avg_loss /= len(train_loader)
    return avg_loss

        # print(sfeature.shape)
        # print(tfeature.shape)
        # break



def main():
    train_loader = set_loader()
    model, criterion = set_model()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs*len(train_loader))

    for epoch in range(num_epochs):
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, scheduler)
        time2 = time.time()
        # break
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        print('loss: ', loss)
        # tensorboard logger
        #logger.log_value('loss', loss, epoch)
        #logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
    torch.save(model.state_dict(), "../checkpoints/pretrained_model.pt")
        # if epoch % opt.save_freq == 0:
        #     save_file = os.path.join(
        #         opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        #     save_model(model, optimizer, opt, epoch, save_file)


if __name__ == '__main__':
    main()