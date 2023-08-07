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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
import os
import json
from torchvision import datasets, models, transforms, utils
from utils import *


def evaluate(args, model, loader):

    avg_acc = 0.0
    avg_loss = 0.0
    model.eval()
    criterion = nn.CrossEntropyLoss()
    label_list, pred_list = [], []

    with torch.set_grad_enabled(False):
        for inputs, labels in tqdm(loader):
            if args.five_crop:
                inputs = inputs.to(device)
                # outputs = model(inputs)
                bs, ncrops, c, h, w = inputs.size()
                outputs = model(inputs.view(-1, c, h, w))
                result_avg = outputs.view(bs, ncrops, -1).mean(1)
                _, pred = torch.max(result_avg, 1)

                for i in range(len(labels)):
                    if int(pred[i]) == int(labels[i]):
                        avg_acc += 1

                label_list.extend(labels.detach().cpu().numpy().tolist())
                pred_list.extend(pred.detach().cpu().numpy().tolist())
            else:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, pred = torch.max(outputs, 1)

                loss = criterion(outputs, labels.to(device)).item()
                avg_loss += loss

                label_list.extend(labels.detach().cpu().numpy().tolist())
                pred_list.extend(pred.detach().cpu().numpy().tolist())

                for i in range(len(labels)):
                    if int(pred[i]) == int(labels[i]):
                        avg_acc += 1

        avg_acc = (avg_acc / len(loader.dataset)) * 100
        avg_loss = (avg_loss / len(loader.dataset))

    plot_confusion_matrix(args, label_list, pred_list)

    print('validation acc: ', avg_acc)


def save_result(csv_path, predict_result, model_n):
    df = pd.read_csv(csv_path)
    new_df = pd.DataFrame()
    new_df['ID'] = df['Path']
    new_df["label"] = predict_result
    new_df.to_csv(f"{args.folder}/pred_411551007_{model_n}_5crop_{args.five_crop}.csv", index=False)

def infer(model, test_loader):

    result = []
    model.eval()
    with torch.set_grad_enabled(False):
        for inputs in tqdm(test_loader):
            if args.five_crop:
                inputs = inputs.to(device)
                bs, ncrops, c, h, w = inputs.size()
                outputs = model(inputs.view(-1, c, h, w))
                result_avg = outputs.view(bs, ncrops, -1).mean(1)
                _, pred = torch.max(result_avg, 1)
                result.append(pred.item())

            else:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, pred = torch.max(outputs, 1)
                result.append(pred.item())

    return result

def test(args, test_loader, mode = 'test'):
    model = eval(f'{args.model_n}(args)')
    s_model_path = os.path.join(args.model_w_path)
    model.load_state_dict(torch.load(s_model_path, map_location=device), strict=False)
    print('model loaded sucessfully')
    model.to(device)

    if mode == 'test':
        result = infer(model, test_loader)
        save_result(args.test_path, result, args.model_n)
    else:
        evaluate(args, model, test_loader)


if __name__ == "__main__":

    print("Good Luck :)")
    parser = argparse.ArgumentParser()
    parser.add_argument("-img_x", type=int, default=450)
    parser.add_argument("-img_y", type=int, default=450)
    parser.add_argument("-num_epochs", type=int, default=300)
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-model_n", type=str, default='ResNet18')
    parser.add_argument('-hist', action='store_true', default=False)
    parser.add_argument('-in_ch', type=int, default=3)


    parser.add_argument("-test_path", type=str, default='/media/yclin/3TBNAS/DLP/Lab3/resnet_18_test.csv')
    parser.add_argument("-model_w_path", type=str, default='/media/yclin/3TBNAS/DLP/Lab3/20230804_1228ResNet18_weight/epoch_280_20230804_1228_ResNet18_96.37.pt')
    # parser.add_argument("-folder", type=str, default='/media/yclin/3TBNAS/DLP/Lab3/20230731_1409ResNet18_weight')
    # parser.add_argument('-five_crop', action='store_true', default=False)
    parser.add_argument('-five_crop', type=int, default=5)

    args = parser.parse_args()
    folder, _ = os.path.split(args.model_w_path)
    args.folder = folder

    print(json.dumps(vars(args), indent=2))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    root = '/media/yclin/3TBNAS/DLP/Lab3/new_dataset'

    test_dataset = LeukemiaLoader(args = args,root = root, mode = 'test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    valid_dataset = LeukemiaLoader(args = args,root = root, mode = 'valid')
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    test(args, valid_loader, mode = 'valid')

    test(args, test_loader, mode = 'test')

    




