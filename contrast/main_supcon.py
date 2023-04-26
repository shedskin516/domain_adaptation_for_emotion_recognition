from __future__ import print_function

import os
import sys
import argparse
import time
import math
import numpy as np
import random
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model

from losses import SupConLoss
from distance import get_pairs
from model import Encoder

path_aff = "../features_aff_20k.npy"
path_sewa = "../features_sewa_20k.npy"

source_file_path = '../mini_datasets/Aff-Wild2/train.csv'
target_file_path = '../mini_datasets/SEWA/train.csv'

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    parser.add_argument('--input_dim', type=int, default=128, help='parameter for input size')
    parser.add_argument('--output_dim', type=int, default=128, help='parameter for output size')
    parser.add_argument('--weight', type=int, default=0.8, help='parameter for output size')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


class train_data(Dataset):
    def __init__(self):
        source_file = pd.read_csv(source_file_path)
        total_source_file = source_file['file_path'].tolist()
        self.source_file_path = total_source_file[:1000]

        target_file = pd.read_csv(target_file_path)
        total_target_file = target_file['file_path'].tolist()
        self.target_file_path = total_target_file[:1000]

        transform_list = [  transforms.Resize(112),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                            transforms.RandomErasing(scale=(0.02,0.25))]
        self.transform = transforms.Compose(transform_list)
    
        self.aff_labels, self.sewa_labels = get_pairs(path_aff, path_sewa)


    def __len__(self):
        return len(self.source_file_path)
   
    def __getitem__(self, idx):
        aff_dir = '../../Face-Warping-Emotion-Recognition-main/data/Aff-Wild2/cropped_aligned/'
        sewa_dir = '../../Face-Warping-Emotion-Recognition-main/data/SEWA/prep_SEWA/'

        entry1 = self.transform(Image.open(aff_dir + self.source_file_path[idx]))
        label1 = self.aff_labels[idx]
        indices = [i for i in range(self.__len__()) if self.sewa_labels[i] == label1]
        random_index = random.choice(indices)
        entry2 = self.transform(Image.open(sewa_dir + self.target_file_path[random_index]))
        label2 = self.sewa_labels[random_index]

        entry3 = self.transform(Image.open(sewa_dir + self.target_file_path[idx]))
        label3 = self.sewa_labels[idx]
        indices = [i for i in range(self.__len__()) if self.sewa_labels[i] == label3]
        random_index = random.choice(indices)
        entry4 = self.transform(Image.open(sewa_dir + self.target_file_path[random_index]))
        label4 = self.sewa_labels[random_index]

        indices = [i for i in range(self.__len__()) if self.aff_labels[i] == label1]
        random_index = random.choice(indices)
        entry5 = self.transform(Image.open(aff_dir + self.source_file_path[random_index]))
        label5 = self.aff_labels[random_index]

        entry = np.concatenate((entry1, entry2, entry3, entry4, entry5), axis=0)
        label = torch.tensor([label1, label2, label3, label4, label5])

        return entry, label
    
class sewa_data(Dataset):
    def __init__(self):
        self.data = np.load(path_sewa)

    def __len__(self):
        return len(self.data)
   
    def __getitem__(self, idx):
        entry = torch.tensor(self.data[idx])
        return entry

def set_loader(opt):
    data_train = train_data()
    train_loader = DataLoader(data_train, batch_size=opt.batch_size, shuffle=True)
    return train_loader

def set_model(opt):
    model = Encoder(output_dim=opt.output_dim)
    criterion = SupConLoss(temperature=opt.temp)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        image1, image2, image3, image4, image5 = torch.split(images, split_size_or_sections=3, dim=1)
        feature1 = model(image1).unsqueeze(1)
        feature2 = model(image2).unsqueeze(1)
        feature3 = model(image3).unsqueeze(1)
        feature4 = model(image4).unsqueeze(1)
        feature5 = model(image5).unsqueeze(1)

        # across aff and sewa
        concat_features1 = torch.cat((feature1, feature2), dim=1)
        labels1, labels2, labels3 = torch.split(labels, split_size_or_sections=[2, 2, 1], dim=1)
        loss_supcon = criterion(concat_features1, labels1[:,0])

        # within sewa
        concat_features2 = torch.cat((feature3, feature4), dim=1)
        loss_infoNCE = criterion(concat_features2, labels2[:,0])

        # within aff
        concat_features3 = torch.cat((feature1, feature5), dim=1)
        loss_infoNCE = loss_infoNCE + criterion(concat_features3, labels1[:,0])

        # loss
        loss = (loss_supcon + opt.weight * loss_infoNCE) / (1 + opt.weight)
        print(loss)
        
        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        print(idx)

        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()
        

    return losses.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    #logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        print('loss: ', loss)
        # tensorboard logger
        #logger.log_value('loss', loss, epoch)
        #logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)


    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


    # get 
    data_sewa = sewa_data()
    sewa_loader = DataLoader(data_sewa, batch_size=100, shuffle=True)
    features = torch.empty(0, opt.output_dim)
    with torch.no_grad():
        for batch in iter(sewa_loader):
            out = model.forward(batch)
            # print(out)
            features = torch.cat((features, out), dim=0)

    features = features.detach().numpy()
    print(features.shape)
    np.save('contrastive_features.npy', features)



if __name__ == '__main__':
    main()
