import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import creat_net, create_train_loader, create_test_loader, WarmupLR,\
    most_recent_weights,most_recent_folder,last_epoch,best_acc_weights
import conf

def train(epoch):

    start = time.time()
    net.train()
    for i, (images, labels) in enumerate(train_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n_iter = (epoch -1)*len(train_loader) + i + 1

        #print(len(train_loader))

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter) #para.norm()
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)  #para.norm()

        print("Training Epoch: {epoch} [{trained_samples}/{total_samples}]\t Loss: {:0.4f}\t LR: {:0.6f}".format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch = epoch,
            trained_samples = i * args.b + len(images),
            total_samples = len_train

        ))

         #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time is : {:0.2f}s'.format(epoch, finish-start))
        
@torch.no_grad()
def eval_training(epoch, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in val_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Val set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len_val,
        correct / len_val,
        finish - start
    ))
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Val/Average loss', test_loss / len_val, epoch)
        writer.add_scalar('Val/Accuracy', correct / len_val, epoch)

    return correct / len_val


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='vgg11', help='net type')
    parser.add_argument('-path', type=str,default='./data/cifar-100-python', help='path')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int,default=16, help='batch_size')
    parser.add_argument('-warm',type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')

    args = parser.parse_args()

    net = creat_net(args)
    #print(net)

    train_loader, val_loader, len_train, len_val = create_train_loader(
                               mean=conf.train_mean, std=conf.train_std, val_ratio=0.2,
                               path=args.path, batchszie= args.b, num_workers= 2, shuffle=True)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr= args.lr, momentum= 0.9, weight_decay= 5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones= conf.MILESTONES, gamma=0.2)

    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmupLR(optimizer, iter_per_epoch*args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(conf.CHECKPOINT_PATH, args.net), fmt=conf.DATE_FORMAT)

        if not recent_folder:
            raise Exception('no recent folder was found')
        checkpoint_path = os.path.join(conf.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(conf.CHECKPOINT_PATH, args.net, conf.TIME_NOW)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path,'{net}-{epoch}-{type}.pth')

    #use tensorboard
    if not os.path.exists(conf.LOG_DIR):
        os.mkdir(conf.LOG_DIR)

    writer = SummaryWriter(log_dir=os.path.join(conf.LOG_DIR, args.net, conf.TIME_NOW))
    input_tensor = torch.Tensor(1,3,32,32)
    if args.gpu:
        input_tensor = input_tensor.cuda()
    writer.add_graph(net,input_tensor)                   # 绘制模型结构流程图

    best_acc = 0.0

    if args.resume:
        best_weights = best_acc_weights(os.path.join(conf.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(conf.CHECKPOINT_PATH, args.net, recent_folder,best_weights)
            print('found best weights file: {}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_files = most_recent_weights(os.path.join(conf.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_files:
            raise Exception('no recent weights file were found')

        weights_path = os.path.join(conf.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_files)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(conf.CHECKPOINT_PATH, args.net, recent_folder))

    for epoch in range(1, conf.EPOCH+1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train(epoch)
        acc = eval_training(epoch)

        if epoch >= 1 and acc > best_acc:
            weights_path = checkpoint_path.format(net=args.net, epoch = epoch, type = 'best')
            print('saving weights to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % 10:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    writer.close()





