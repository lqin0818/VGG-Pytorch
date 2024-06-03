"""
train and test dataset building

"""

import os
import sys
import pickle

from skimage import io
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CIFAR100train(Dataset):

    def __init__(self, path, transform = None):
        with open(os.path.join(path, 'train'), 'rb') as cifar100:
            self.dict = pickle.load(cifar100, encoding='latin1')
        self.transform = transform

    def __len__(self):
        return len(self.dict['filenames'])

    def __getitem__(self, index):
        label = self.dict['fine_labels'][index]

        r = self.dict['data'][index, :1024].reshape(32, 32)
        g = self.dict['data'][index, 1024:2048].reshape(32, 32)
        b = self.dict['data'][index, 2048:].reshape(32, 32)
        image = np.dstack((r,g,b))

        if self.transform:
            image = self.transform(image)
        return label, image

class CIFAR100test(Dataset):

    def __init__(self, path, transform = None):
        with open(os.path.join(path, 'test'), 'rb') as cifar100:
            self.dict = pickle.load(cifar100, encoding='latin1')
        self.transform = transform

    def __len__(self):
        return len(self.dict['filenames'])

    def __getitem__(self, index):
        label = self.dict['fine_labels'][index]
        r = self.dict['data'][index, :1024].reshape(32, 32)
        g = self.dict['data'][index, 1024:2048].reshape(32, 32)
        b = self.dict['data'][index, 2048:].reshape(32, 32)
        image = np.dstack((r,g,b))

        if self.transform:
            image = self.transform(image)
        return label, image

path = './data/cifar-100-python'

traindata = CIFAR100train(path, transform=None)
testdata = CIFAR100test(path, transform=None)

image1 = traindata[5][1]

print(image1)
plt.imshow(image1)
plt.show()
