import os
import sys
import re
import datetime

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.optim.lr_scheduler import LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import CIFAR100train, CIFAR100test
import conf
from torch.utils.data.sampler import SubsetRandomSampler

def creat_net(args):

    if args.net == 'vgg11':
        from vgg_m import VGG11
        net = VGG11()
    elif args.net == 'vgg13':
        from vgg_m import VGG13
        net = VGG13()
    elif args.net == 'vgg16':
        from vgg_m import VGG16
        net = VGG16()
    elif args.net == 'vgg19':
        from vgg_m import VGG19
        net = VGG19()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net

def create_train_loader(mean, std, path, batchszie =16, val_ratio = 0.2, num_workers =2, shuffle = True):
     """ create training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_loader:torch dataloader object
    """
     transform_train = transforms.Compose([
         #transforms.ToPILImage(),
         transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(30),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)
     ])

     transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

     train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
     val_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_val)

     len_valdata = np.floor(val_ratio*len(train_dataset))
     incidies = list(range(len(train_dataset)))

     if shuffle:
         np.random.seed(1)
         np.random.shuffle(incidies)

     index_train = incidies[int(len_valdata):]
     index_val = incidies[:int(len_valdata)]

     #train_dataset = CIFAR100train(path= path,transform= transform_train)
     train_loader = DataLoader(train_dataset, batch_size=batchszie,num_workers=num_workers,
                               sampler=SubsetRandomSampler(index_train))
     val_loader = DataLoader(val_dataset, batch_size=batchszie, num_workers=num_workers,
                             sampler=SubsetRandomSampler(index_val))


     return train_loader, val_loader, int(len(train_dataset)-int(len_valdata)), int(len_valdata)

def create_test_loader(mean, std, path, batchszie =16, num_workers =2, shuffle = True):
    """ creat test dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: test_loader:torch dataloader object
    """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    #test_dataset = CIFAR100test(path=path, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batchszie, num_workers=num_workers, shuffle=shuffle)

    return test_loader

def compute_mean_std(dataset):

    r = np.dstack([dataset[i][1][:,:,0] for i in range(len(dataset))])
    g = np.dstack([dataset[i][1][:,:,1] for i in range(len(dataset))])
    b = np.dstack([dataset[i][1][:,:,2] for i in range(len(dataset))])

    mean = np.mean(r), np.mean(g), np.mean(b)
    std = np.std(r), np.std(g), np.std(b)

    return mean, std

"""
path = './data/cifar-100-python'

train_dataset = CIFAR100train(path=path,transform=None)

train_mean, train_std = compute_mean_std(train_dataset)
print(train_mean, train_std)

train_loader = create_train_loader(train_mean,train_std,path,shuffle=False)

print(len(train_loader))

label, image = train_dataset.__getitem__(33)

print(label, image)
plt.imshow(image)
plt.show()
"""

class WarmupLR(LRScheduler):

    def __init__(self, optimizer, total_iter, last_epoch= -1):
        self.total_iter = total_iter
        super().__init__(optimizer, last_epoch)

    def get_lr(self):

        return [base_lr * self.last_epoch/(self.total_iter+ 1e-8) for base_lr in self.base_lrs]

def most_recent_folder(net_weights, fmt):

    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    final_folders = []
    folders = os.listdir(net_weights)

    for f in folders:
        if len(os.listdir(os.path.join(net_weights,f))):
            final_folders.append(f)

    if len(final_folders) == 0:
        return ''
    final_folders = sorted(final_folders, key= lambda x: datetime.datetime.strptime(x, fmt))

    return final_folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """

    weight_files = os.listdir(weights_folder)

    if len(weight_files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    # 正则化字符匹配，（）代表一个组，[]匹配中括号内的任意一个字符，+ 匹配前面的字符出现1次或多次
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)

    if not weight_file:
        raise Exception('no recent weights were found')

    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]

