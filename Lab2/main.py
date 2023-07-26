import copy
import torch
import argparse
import dataloader
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import torch.optim as optim
import matplotlib.pyplot as plt
from model import *
from torchsummary import summary
from matplotlib.ticker import MaxNLocator
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
import os
import json
from torchvision import datasets, models, transforms, utils


class BCIDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = torch.tensor(self.data[index,...], dtype=torch.float32)
        label = torch.tensor(self.label[index], dtype=torch.int64)
        return data, label

    def __len__(self):
        return self.data.shape[0]


def plot_result(result, epochs, model = 'EEGNet'):
    plt.figure(figsize=(10, 5))
    epochs = range(1, epochs +1)

    plt.plot(epochs, result['train']['relu'], 'r', label = 'relu_train')
    plt.plot(epochs, result['test']['relu'], 'b', label = 'relu_test')
    plt.plot(epochs, result['train']['leakyrelu'], 'g', label = 'leakyRelu_train')
    plt.plot(epochs, result['test']['leakyrelu'], 'c', label = 'leakyRelu_test')
    plt.plot(epochs, result['train']['elu'], 'm', label = 'elu_train')
    plt.plot(epochs, result['test']['elu'], 'y', label = 'elu_test')

    plt.title(f'Activation function comparison({model})')
    plt.legend()
    # plt.figure()
    plt.savefig('./{}/Comparison{}_{}_{}_{}.jpeg'.format(args.folder, model, args.dropout_rate, args.activation, args.elu_alpha))
    # plt.show()
    plt.close()



def plot_train_loss(loss_list, epochs, model = 'EEGNet'):
    # TODO plot training and testing accuracy curve
    plt.figure(figsize=(10, 5))
    epochs = range(1, epochs +1)

    plt.plot(epochs, loss_list['train']['relu'], 'r', label = 'relu_train')
    plt.plot(epochs, loss_list['test']['relu'], 'b', label = 'relu_test')
    plt.plot(epochs, loss_list['train']['leakyrelu'], 'g', label = 'leakyRelu_train')
    plt.plot(epochs, loss_list['test']['leakyrelu'], 'c', label = 'leakyRelu_test')
    plt.plot(epochs, loss_list['train']['elu'], 'm', label = 'elu_train')
    plt.plot(epochs, loss_list['test']['elu'], 'y', label = 'elu_test')

    # plt.plot(epochs, train_loss_list, 'b', label = 'Train loss')
    plt.title(f'{model} Train and Test Loss')
    plt.legend()
    # plt.figure()
    plt.savefig('./{}/Loss{}_{}_{}_{}_Loss.jpeg'.format(args.folder, model, args.dropout_rate, args.activation, args.elu_alpha))
    # plt.show()
    plt.close()


def train(args, loader, optimizer = "adam", model_n = 'EEGNet'):
    best_acc = 0.0
    best_wts = None
    # avg_acc_list = []
    # test_acc_list = []
    # avg_loss_list = []

    activation_list = ['elu', 'relu', 'leakyrelu']
    total_result = {
        'train':{
            'elu':[],
            'relu':[],
            'leakyrelu':[]
        },
        'test':{
            'elu':[],
            'relu':[],
            'leakyrelu':[]
        }
    }

    total_loss = {
        'train':{
            'elu':[],
            'relu':[],
            'leakyrelu':[]
        },
        'test':{
            'elu':[],
            'relu':[],
            'leakyrelu':[]
        }
    }

    for act in activation_list:
        args.activation = act
        if model_n == 'EEGNet':
            model = EEGNet(args=args)
        elif model_n == 'DeepConvNet':
            model = DeepConvNet(args=args)
        else:
            model = AEEGNet(args=args)
            

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=30, threshold=1e-3, 
                                    threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
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
                # avg_acc_list.append(avg_acc)
                # print(f'Epoch: {epoch}')
                # print(f'Loss: {avg_loss}')
                # print(f'Training Acc. (%): {avg_acc:3.2f}%')
                total_result['train'][act].append(avg_acc)
                total_loss['train'][act].append(avg_loss)


            test_acc, test_loss = test(model, test_loader)
            total_result['test'][act].append(test_acc)
            total_loss['test'][act].append(test_loss)
            scheduler.step(test_acc)
            # test_acc_list.append(test_acc)
            if test_acc > best_acc:
                best_acc = test_acc
                best_wts = model.state_dict()
                torch.save(best_wts, './{}/{}_{}_drop{}_{}_{}_{:.2f}.pt'.format(args.folder, args.time, model_n, args.dropout_rate, args.activation, args.elu_alpha, test_acc))
            # print(f'Test Acc. (%): {test_acc:3.2f}%')

        # if model_n != 'EEGNet' or model_n != 'DeepConvNet':
        #     break
    return total_result, total_loss


def test(model, loader):
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

    return avg_acc, avg_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_epochs", type=int, default=300)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-dropout_rate", type=float, default=0.25)
    
    parser.add_argument("-elu_alpha", type=float, default=1.0)
    parser.add_argument("-model_n", type=str, default='AEEGNet')
    args = parser.parse_args()

    now = datetime.now() 
    dt_string = now.strftime("%Y%m%d_%H%M")
    dt_string = now.strftime("%Y%m%d_%H%M")
    args.time = dt_string
    args.folder = args.time+'_weight'
    path = os.getcwd()
    folder = os.path.join(path, args.folder)
    os.makedirs(folder, exist_ok=True)
    print(json.dumps(vars(args), indent=2))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    train_dataset = BCIDataset(train_data, train_label)
    test_dataset = BCIDataset(test_data, test_label)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    #EEGNet
    result, loss_list = train(args, train_loader, model_n = args.model_n)
    plot_result(result = result, epochs = args.num_epochs, model = args.model_n)
    plot_train_loss(loss_list, args.num_epochs, model = args.model_n)

    m_1 = max(result['test']['relu'])
    m_2 = max(result['test']['elu'])
    m_3 = max(result['test']['leakyrelu'])
    print(f'Test acc of ReLu: {m_1}, Test acc of ELu: {m_2}, Test acc of LeakyReLu: {m_3}')

    # Accuracy of ReLu: 81.2037037037037, Accuracy of ELu: 82.31481481481482, Accuracy of LeakyReLu: 83.14814814814815

    #DeepConvNet
    # result, train_loss_list = train(args, train_loader, model_n = 'DeepConvNet')
    # plot_result(result = result, epochs = args.num_epochs, model = 'DeepConvNet')
    # plot_train_loss(train_loss_list, args.num_epochs, model = 'DeepConvNet')

    # plot_train_acc(train_acc_list, args.num_epochs)
    # plot_test_acc(test_acc_list, args.num_epochs)

    