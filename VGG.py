"""
VGG in pytorch
[1] Karen Simonyan, Andrew Zisserman, Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""

import torch
import torch.nn as nn

cfg = {
    '11' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    '13' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    '16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    '19' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, nc=100):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, nc)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512)
        x = self.classifier(x)

        return  x

def make_layers(cfg, batch_norm = False):

    layers = []
    cin =3

    for layer in cfg:
        if layer == 'M':
            layers.append(nn.MaxPool2d(2,2))
            continue

        layers.append(nn.Conv2d(cin,layer,3,1))

        if batch_norm:
            layers.append(nn.BatchNorm2d(layer))

        layers.append(nn.ReLU())
        cin = layer
    return nn.Sequential(*layers)

def VGG11():
    return VGG(make_layers(cfg['11'], batch_norm=True))

def VGG13():
    return VGG(make_layers(cfg['13'], batch_norm=True))

def VGG16():
    return VGG(make_layers(cfg['16'], batch_norm=True))

def VGG19():
    return VGG(make_layers(cfg['19'], batch_norm=True))

# test the code #

"""

model = VGG11()
print(model)

"""

