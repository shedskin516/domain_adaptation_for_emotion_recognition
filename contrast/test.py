import os
# os.environ['OMP_NUM_THREADS'] = '1'
import sys
import argparse
import time
import math
import pandas
import numpy as np
import random

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

from model import Encoder, Classifier
from metric import get_eval_metrics

source_file_path = '../mini_datasets/Aff-Wild2/train.csv'
target_file_path = '../mini_datasets/SEWA/test.csv'
val_file_path = '../mini_datasets/Aff-Wild2/val.csv'

source_data = '../data/Aff-Wild2/cropped_aligned/'
target_data = '../data/SEWA/prep_SEWA/'
batch_size = 256
# mini_batch = 10
output_dim = 128
temp = 0.07
num_epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_clusters = 10
label = 'valence'


class dataset(Dataset):
    def __init__(self, file_path, data_path, range):
        files = pandas.read_csv(file_path)

        total_source_file = files['file_path'].tolist()
        total_labels = files[label].values.tolist()

        self.source_file_path = total_source_file[range[0]:range[1]]
        self.labels = total_labels[range[0]:range[1]]
        self.data_path = data_path

        transform_list = [  transforms.Resize(112),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                            transforms.RandomErasing(scale=(0.02,0.25))]
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.source_file_path)
   
    def __getitem__(self, idx):
        img = Image.open(self.data_path+self.source_file_path[idx])
        img = self.transform(img)
        label = self.labels[idx]
        label = np.asarray(label, dtype=np.float32)

        return img, label

def set_loader():
    data_train = dataset(source_file_path, source_data, [5000,10000])
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)

    data_val = dataset(val_file_path, source_data, [0,1000])
    val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False)

    data_test = dataset(target_file_path, target_data, [0,3000])
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def set_model():
    pretrained_model = Encoder(output_dim=output_dim)
    pretrained_model.load_state_dict(torch.load("../checkpoints/pretrained_model.pt"))

    model = Classifier()
    model.encoder = pretrained_model.encoder
    model = model.to(device)
    for param in model.encoder.parameters():
        param.requires_grad = False

    criterion = nn.MSELoss()

    return model, criterion


def train(train_loader, model, criterion, optimizer, scheduler):
    """one epoch training"""
    model.train()
    end = time.time()
    train_loss = 0
    train_ccc = 0

    for idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)
        labels = labels.reshape(labels.shape[0],1)
        y_pred = model(images)

        loss = criterion(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        labels, y_pred = labels.detach().cpu(), y_pred.detach().cpu()
        train_ccc += get_eval_metrics(labels, y_pred)
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_ccc /= len(train_loader)
    return train_loss, train_ccc

def eval(val_loader, model):
    model.eval()
    val_ccc = 0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.reshape(labels.shape[0],1)
            y_pred = model(images)

            labels, y_pred = labels.detach().cpu(), y_pred.detach().cpu()
            val_ccc += get_eval_metrics(labels, y_pred)
    val_ccc /= len(val_loader)
    return val_ccc
        
def test(test_loader, model):
    model.eval()
    test_ccc = 0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.reshape(labels.shape[0],1)
            y_pred = model(images)

            labels, y_pred = labels.detach().cpu(), y_pred.detach().cpu()
            test_ccc += get_eval_metrics(labels, y_pred)
    test_ccc /= len(test_loader)
    return test_ccc

def main():
    # build data loader
    train_loader, val_loader, test_loader = set_loader()

    # build model and criterion
    model, criterion = set_model()

    # build optimizer
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs*len(train_loader))

    # training routine
    for epoch in range(0, num_epochs):

        # train for one epoch
        time1 = time.time()
        train_loss, train_ccc = train(train_loader, model, criterion, optimizer, scheduler)
        val_ccc = eval(val_loader, model)
        test_ccc = test(test_loader, model)
        time2 = time.time()
        # break
        print(f"Epoch [{epoch + 1}/{num_epochs}], total time {(time2 - time1):.2f}, Train Loss: {train_loss:.4f}, Train CCC: {train_ccc:.2f}%, Val Acc: {val_ccc:.2f}%, Test Acc: {test_ccc:.2f}%")
        # tensorboard logger
        #logger.log_value('loss', loss, epoch)
        #logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

    torch.save(model.state_dict(), "../checkpoints/model"+label+".pt")

if __name__ == '__main__':
    main()
