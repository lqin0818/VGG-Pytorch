import argparse
from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import conf
from utils import creat_net,create_test_loader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='vgg11', help='net type')
    parser.add_argument('-weights', type=str, default='checkpoint/vgg11/Tuesday_04_June_2024_20h_52m_17s/vgg11-5-best.pth', help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')

    args = parser.parse_args()

    net = creat_net(args)

    test_loader = create_test_loader(conf.test_mean, conf.test_std, path=None,
                                     batchszie=args.b, num_workers=2)

    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()

    correct_1 = 0
    correct_5 = 0
    total = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            print("iteration: {}\ttotal {} iterations".format(i + 1, len(test_loader)))

            if args.gpu:
                images = images.cuda()
                labels = labels.cuda()
                print('GPU info...')
                print(torch.cuda.memory_summary(), end='')

            output = net(images)
            _, preds = output.topk(5, 1, largest=True, sorted=True)
            labels = labels.view(labels.size(0), -1).expand_as(preds)
            correct = (preds == labels)

            #compute top 5
            correct_5 += correct[:, :5].sum().item()

            #compute top1
            correct_1 += correct[:, :1].sum().item()

        if args.gpu:
            print('GPU INFO.....')
            print(torch.cuda.memory_summary(), end='')

        print()
        print("Top 1 err: ", 1 - correct_1 / len(test_loader.dataset))
        print("Top 5 err: ", 1 - correct_5 / len(test_loader.dataset))
        #print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))

