import os
import sys
import argparse
import time
import math
import pandas

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.optim.lr_scheduler as lr_scheduler

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from losses import SupConLoss
from distance import get_pairs
import numpy as np
import random

source_file_path = '../mini_datasets/Aff-Wild2/train.csv'
target_file_path = '../mini_datasets/SEWA/train.csv'
batch_size = 256
# mini_batch = 10
output_dim = 128
temp = 0.07
num_epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_clusters = 10

class Encoder(nn.Module):
    def __init__(self, output_dim):
            super(Encoder, self).__init__()
            self.encoder = models.resnet18(pretrained=True)
            self.encoder.fc = nn.Linear(self.encoder.fc.in_features, 512)
            self.fc1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, output_dim)
            self.dropout = nn.Dropout(0.1)
            # self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.encoder(x)
        x = self.dropout(self.fc1(x))
        x = F.relu(x)
        x = self.fc2(x)
        return x

class train_data(Dataset):
    def __init__(self):
        source_file = pandas.read_csv(source_file_path)
        total_source_file = source_file['file_path'].tolist()
        self.source_file_path = total_source_file[:5000]

        target_file = pandas.read_csv(target_file_path)
        total_target_file = target_file['file_path'].tolist()
        self.target_file_path = total_target_file[:5000]

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
        if index > 0:
            print(sfeature)
        source_labels, target_labels = get_pairs(sfeature, tfeature, n_clusters)
        batch_loss = 0
        for k in range(n_clusters):
            source_index = [i for i in range(len(source_labels)) if source_labels[i] == k]
            target_index = [i for i in range(len(target_labels)) if target_labels[i] == k]
            length = min(len(source_index), len(target_index))

            # source_cat_features = torch.
            source_cat_features = torch.cat(tuple(sfeature[i].unsqueeze(0) for i in source_index[:length]), dim=0)
            target_cat_features = torch.cat(tuple(tfeature[i].unsqueeze(0) for i in target_index[:length]), dim=0)
            # print(source_cat_features.shape)
            # print(target_cat_features.shape)

            concat_features = torch.cat((source_cat_features, target_cat_features), dim=1)
            concat_features = concat_features.reshape(length, 1, output_dim*2)
            print(concat_features.shape)
            labels = torch.tensor([k for i in range(length)])
            print(labels)

            loss = criterion(concat_features, labels)
            # print(loss_supcon)
            # break
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
            batch_loss += loss.item()
            print(f"{k}: loss:{loss.item()}")
        batch_loss = batch_loss/k
        avg_loss += batch_loss
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

        # if epoch % opt.save_freq == 0:
        #     save_file = os.path.join(
        #         opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        #     save_model(model, optimizer, opt, epoch, save_file)

if __name__ == '__main__':
    main()