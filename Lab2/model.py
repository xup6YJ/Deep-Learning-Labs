import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms, utils

class EEGNet(nn.Module):
    def __init__(self, args):
        
        super(EEGNet, self).__init__()
        # self.nb_classes = nb_classes
        self.dropoutRate = args.dropout_rate
        self.activation = args.activation
        self.alpha = args.elu_alpha

        if self.activation == 'elu':
            self.act_fun = nn.ELU(alpha=self.alpha)
        elif self.activation == 'leakyrelu':
            self.act_fun = nn.LeakyReLU()
        else:
            self.act_fun = nn.ReLU()


        # Layer 1
        self.l1 = nn.Sequential(
            nn.Conv2d(1, 16, (1, 51), padding = (0, 25), bias = False),  #(1, 64),
            nn.BatchNorm2d(16, False)
        )

        #DepthwiseConv2D D * F1 (C, 1)
        '''When groups == in_channels and out_channels == K * in_channels, 
        where K is a positive integer, this operation is also known as a “depthwise convolution”.
        '''
        #16, 1, 2, 1
        self.Depthwise = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 1), stride = (1,1), groups=16, bias=False),
            nn.BatchNorm2d(32, False),
            self.act_fun,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(self.dropoutRate)
        )

        # Layer 2
        self.Separable = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size=(1, 15), stride = (1,1), bias=False, padding=(0,7)),
            nn.BatchNorm2d(32, False),
            self.act_fun,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(self.dropoutRate)
        )
        
        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        self.fc1 = nn.Linear(in_features=736, out_features=2, bias=True)

        

    def forward(self, x):
         
        # print('Input x.shape: ', x.shape)  #Input x.shape:  torch.Size([B, 1, 2, 750])

        # Block 
        x = self.l1(x)
        x = self.Depthwise(x)
        x = self.Separable(x)

        # FC Layer
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        # x = F.sigmoid(x)
        # x = F.softmax(x, dim=1) #do not need

        return x
    

class DeepConvNet(nn.Module):
    def __init__(self, args):
        
        super(DeepConvNet, self).__init__()
        # self.nb_classes = nb_classes
        self.dropoutRate = args.dropout_rate
        self.activation = args.activation
        self.alpha = args.elu_alpha

        if self.activation == 'elu':
            self.act_fun = nn.ELU(alpha=self.alpha)
        elif self.activation == 'leakyrelu':
            self.act_fun = nn.LeakyReLU()
        else:
            self.act_fun = nn.ReLU()


        # Layer 1
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 25, (1, 5)),  
            nn.Conv2d(25, 25, (2, 1)),
            nn.BatchNorm2d(25, False),
            self.act_fun,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(self.dropoutRate)
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(25, 50, (1, 5)),  
            nn.BatchNorm2d(50, False),
            self.act_fun,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(self.dropoutRate)
        )

        self.b3 = nn.Sequential(
            nn.Conv2d(50, 100, (1, 5)),  
            nn.BatchNorm2d(100, False),
            self.act_fun,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(self.dropoutRate)
        )

        self.b4 = nn.Sequential(
            nn.Conv2d(100, 200, (1, 5)),  
            nn.BatchNorm2d(200, False),
            self.act_fun,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(self.dropoutRate)
        )

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        self.fc1 = nn.Linear(in_features=8600, out_features=2, bias=True)
        

    def forward(self, x):
         
        # print('Input x.shape: ', x.shape)  #Input x.shape:  torch.Size([64, 1, 2, 750])

        # Block 
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)

        # FC Layer
        x = torch.flatten(x, start_dim=1)
        # print('torch.flatten(x) x.shape', x.shape)
        x = self.fc1(x)
        # print('self.fc1(x) x.shape', x.shape)
        x = F.softmax(x, dim=1)
        # print('F.sigmoid(x) x.shape', x.shape)
        return x
    


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super(SEBlock, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1) #Channel  #Check
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels) #Check
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Global average pooling
        y = self.avg_pool(x).view(batch_size, channels)

        # Squeeze
        y = self.relu(self.fc1(y))

        # Excite
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, channels, 1, 1)

        # Scale
        # x = x * y.expand_as(x)  
        output_tensor = torch.mul(x, y)  #Check
        # print ('SE block', 'output shape: ', output_tensor.shape) 

        return output_tensor

class AEEGNet(nn.Module):
    def __init__(self, args):
        
        super(AEEGNet, self).__init__()
        # self.nb_classes = nb_classes
        self.dropoutRate = args.dropout_rate
        self.activation = args.activation
        self.alpha = args.elu_alpha

        if self.activation == 'elu':
            self.act_fun = nn.ELU(alpha=self.alpha)
        elif self.activation == 'leakyrelu':
            self.act_fun = nn.LeakyReLU()
        else:
            self.act_fun = nn.ReLU()


        # Layer 1
        self.l1 = nn.Sequential(
            nn.Conv2d(1, 16, (1, 51), padding = (0, 25), bias = False),  #(1, 64),
            nn.BatchNorm2d(16, False)
        )

        self.attention1 = SEBlock(in_channels = 16)

        #DepthwiseConv2D D * F1 (C, 1)
        '''When groups == in_channels and out_channels == K * in_channels, 
        where K is a positive integer, this operation is also known as a “depthwise convolution”.
        '''
        #16, 1, 2, 1
        self.Depthwise = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 1), stride = (1,1), groups=16, bias=False),
            nn.BatchNorm2d(32, False),
            self.act_fun,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(self.dropoutRate)
        )

        self.attention2 = SEBlock(in_channels = 32)

        # Layer 2
        self.Separable = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size=(1, 15), stride = (1,1), bias=False, padding=(0,7)),
            nn.BatchNorm2d(32, False),
            self.act_fun,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(self.dropoutRate)
        )

        # self.attention3 = SEBlock(in_channels = 32)
        
        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        self.fc1 = nn.Linear(in_features=736, out_features=2, bias=True)

        

    def forward(self, x):
         
        # print('Input x.shape: ', x.shape)  #Input x.shape:  torch.Size([B, 1, 2, 750])

        # Block 
        x = self.l1(x)
        x = self.attention1(x)

        x = self.Depthwise(x) #(64, 32, 1, 187)
        x = self.attention2(x)
        x = self.Separable(x)
        
        x = self.attention3(x)

        # FC Layer
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        # x = F.sigmoid(x)
        x = F.softmax(x, dim=1)

        return x