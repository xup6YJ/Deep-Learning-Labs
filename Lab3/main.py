import pandas as pd
import copy
import torch
import argparse
from dataloader import *
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import torch.optim as optim
import matplotlib.pyplot as plt
from ResNet import *
from torchsummary import summary
from matplotlib.ticker import MaxNLocator
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, CosineAnnealingWarmRestarts
from datetime import datetime
import os
import json
from torchvision import datasets, models, transforms, utils
from utils import *
import torch.backends.cudnn as cudnnbackend
from torch.utils.tensorboard import SummaryWriter


def evaluate(model, loader):
    avg_acc = 0.0
    avg_loss = 0.0
    model.eval()
    criterion = nn.CrossEntropyLoss()

    with torch.set_grad_enabled(False):
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            loss = criterion(outputs, labels.to(device)).item()
            avg_loss += loss

            for i in range(len(labels)):
                if int(pred[i]) == int(labels[i]):
                    avg_acc += 1

        avg_acc = (avg_acc / len(loader.dataset)) * 100
        avg_loss = (avg_loss / len(loader.dataset))
        # print(f'val acc: {avg_acc:.2f}, val loss: {avg_loss:.5f}')

    return avg_acc, avg_loss


def train(args, loader, val_loader, optimizer = "adam", model_n = 'EEGNet'):
    cudnnbackend.benchmark = True
    # print("train() not defined")
    best_acc = 0.0
    best_wts = None
    # avg_acc_list = []
    # test_acc_list = []
    # avg_loss_list = []
    logger = SummaryWriter(os.path.join(args.folder, f'{args.model_n}'))

    # model_list = ['ResNet18', 'ResNet50', 'ResNet152']
    # total_result = {
    #     'train':{
    #         'ResNet18':[],
    #         'ResNet50':[],
    #         'ResNet152':[]
    #     },
    #     'validation':{
    #         'ResNet18':[],
    #         'ResNet50':[],
    #         'ResNet152':[]
    #     }
    # }

    # total_loss = {
    #     'train':{
    #         'ResNet18':[],
    #         'ResNet50':[],
    #         'ResNet152':[]
    #     },
    #     'validation':{
    #         'ResNet18':[],
    #         'ResNet50':[],
    #         'ResNet152':[]
    #     }
    # }

    # for model_n in model_list:
    # args.model = model
    if args.model_n == 'ResNet18':
        model = ResNet18(args)
    elif args.model_n == 'ResNet50':
        model = ResNet50(args)
    else:
        model = ResNet152(args)
        

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)

    if args.sch_n == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 20, T_mult=2, verbose=False)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=30, threshold=1e-3, 
                                threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
    # scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1, verbose=True)
    
    # scheduler = ExponentialLR(optimizer, gamma=0.9)

    model.to(device)
    criterion.to(device)

    for epoch in tqdm(range(1, args.num_epochs+1)):
        model.train()
        with torch.set_grad_enabled(True):
            avg_acc = 0.0
            avg_loss = 0.0 
            for i, data in enumerate(loader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                _, pred = torch.max(outputs.data, 1)
                avg_acc += pred.eq(labels).cpu().sum().item()

            
            avg_loss /= len(loader.dataset)
            # avg_loss_list.append(avg_loss)
            avg_acc = (avg_acc / len(loader.dataset)) * 100
            # print(f'train acc: {avg_acc:.2f}, train loss: {avg_loss:.5f}')

            # avg_acc_list.append(avg_acc)
            # print(f'Epoch: {epoch}')
            # print(f'Loss: {avg_loss}')
            # print(f'Training Acc. (%): {avg_acc:3.2f}%')

            # total_result['train'][model_n].append(avg_acc)
            # total_loss['train'][model_n].append(avg_loss)

            logger.add_scalar(f'train/loss', avg_loss, epoch)
            logger.add_scalar(f'train/acc', avg_acc, epoch)



        test_acc, test_loss = evaluate(model, val_loader)
        # total_result['validation'][model_n].append(test_acc)
        # total_loss['validation'][model_n].append(test_loss)

        logger.add_scalar(f'val/loss', test_loss, epoch)
        logger.add_scalar(f'val/acc', test_acc, epoch)
        
        scheduler.step(test_acc)
        # test_acc_list.append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            best_wts = model.state_dict()
            torch.save(best_wts, './{}/epoch_{}_{}_{}_{:.2f}.pt'.format(args.folder, epoch, args.time, model_n, test_acc))
        # print(f'Test Acc. (%): {test_acc:3.2f}%')

    # return total_result, total_loss


def save_result(csv_path, predict_result, model_n):
    df = pd.read_csv(csv_path)
    new_df = pd.DataFrame()
    new_df['ID'] = df['Path']
    new_df["label"] = predict_result
    new_df.to_csv(f"./{args.time}_411551007_{model_n}.csv", index=False)

def infer(model, test_loader):

    result = []
    model.eval()
    with torch.set_grad_enabled(False):
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            result.append(pred)

    return result

def test(args, test_loader):
    model = args.model_n()
    s_model_path = os.path.join('/media/yclin/3TBNAS/DLP/Lab2/20230719_1405_weight/20230719_1405_EEGNet_drop0.25_relu_1.0_88.15.pt')
    model.load_state_dict(torch.load(s_model_path, map_location=device), strict=False)
    print('model loaded sucessfully')
    model.to(device)

    result = infer(model, test_loader)
    save_result(args.test_path, result, args.model_n)


    # return test_acc, test_loss


if __name__ == "__main__":

    print("Good Luck :)")
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_epochs", type=int, default=300)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-img_x", type=int, default=450)
    parser.add_argument("-img_y", type=int, default=450)
    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-model_n", type=str, default='ResNet18')
    parser.add_argument('-hist', action='store_true', default=False)
    parser.add_argument('-in_ch', type=int, default=3)
    parser.add_argument("-sch_n", type=str, default='plateau')

    parser.add_argument('-five_crop', action='store_true', default=False)
    parser.add_argument("-test_path", type=str, default='/media/yclin/3TBNAS/DLP/Lab3/resnet_18_test.csv')

    args = parser.parse_args()

    now = datetime.now() 
    dt_string = now.strftime("%Y%m%d_%H%M")
    dt_string = now.strftime("%Y%m%d_%H%M")
    args.time = dt_string
    args.folder = args.time + args.model_n + '_weight'
    path = os.getcwd()
    folder = os.path.join(path, args.folder)
    os.makedirs(folder, exist_ok=True)
    print(json.dumps(vars(args), indent=2))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    root = '/media/yclin/3TBNAS/DLP/Lab3/new_dataset'

    train_dataset = LeukemiaLoader(args = args, root = root, mode = 'train')
    valid_dataset = LeukemiaLoader(args = args,root = root, mode = 'valid')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)


    #EEGNet
    train(args, train_loader, valid_loader, model_n = args.model_n)
    # plot_result(args = args, result = result, epochs = args.num_epochs)
    # plot_train_loss(args = args, loss_list = loss_list, epochs = args.num_epochs)



