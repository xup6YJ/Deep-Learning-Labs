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

def infer(test_loader):
    args.activation = 'relu'
    model = EEGNet(args=args)
    s_model_path = os.path.join('/media/yclin/3TBNAS/DLP/Lab2/20230719_1405_weight/20230719_1405_EEGNet_drop0.25_relu_1.0_88.15.pt')
    model.load_state_dict(torch.load(s_model_path, map_location=device), strict=False)
    print('model loaded sucessfully')
    model.to(device)

    test_acc, test_loss = test(model, test_loader)
    print('Test accuraccy: ', test_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_epochs", type=int, default=300)
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-dropout_rate", type=float, default=0.25)
    
    parser.add_argument("-elu_alpha", type=float, default=1.0)
    parser.add_argument("-model_n", type=str, default='EEGNet')
    args = parser.parse_args()

    print(json.dumps(vars(args), indent=2))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_data, train_label, test_data, test_label = dataloader.read_bci_data()

    test_dataset = BCIDataset(test_data, test_label)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    #EEGNet
    infer(test_loader = test_loader)