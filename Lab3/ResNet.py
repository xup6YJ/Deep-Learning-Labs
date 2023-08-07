# print("Please define your ResNet in this file.")


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
    
#Block
class basic_block(nn.Module):
    def __init__(self, in_channels, out_channels = 64, kernel_size = 3, padding = 1):
        super(basic_block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                        kernel_size=kernel_size, stride = 1, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                        kernel_size=kernel_size, stride = 1, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        
        )

    def forward(self, x):
        out = self.block(x)
        out = F.relu(out+x)
        return out


class down_block(nn.Module):
    def __init__(self, in_channels, out_channels = 64, down_stride = 1, kernel_size = 3, padding = 1):
        super(down_block, self).__init__()

        self.block = nn.Sequential(
            #first conv dif
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                        kernel_size=kernel_size, stride = 1, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_channels)
                
        )

        #downsample
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                        kernel_size=1, stride = down_stride, bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        out = F.relu(out+self.downsample(x))
        return out
    

class bottle_neck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, downsample_stride = None, kernel_size = 3):
        super(bottle_neck, self).__init__()

        self.init = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, 
                      kernel_size=1, stride = 1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True)
        )

        #middle downsample or not
        if downsample_stride is None:
            self.middle = nn.Sequential(
                nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, 
                        kernel_size=3, stride = 1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=mid_channels),
                nn.ReLU(inplace=True)
            )
            self.downsample = None
        else:
            self.middle = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, 
                          kernel_size=3, stride=downsample_stride, padding=1, bias=False),
                nn.BatchNorm2d(num_features=mid_channels),
                nn.ReLU(inplace=True)
            )

            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=downsample_stride, bias=False),
                nn.BatchNorm2d(out_channels)
                )

        self.last = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, 
                      kernel_size=1, stride = 1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        )


    def forward(self, x):
        out = self.init(x)
        out = self.middle(out)
        out = self.last(out)

        if self.downsample is not None: 
            out = F.relu(out+self.downsample(x)) #does not match ([1, 256, 56, 56]), ([1, 256, 28, 28])
        else:
            out = F.relu(out + x)

        return out

#Model 
class ResNet18(nn.Module):
    def __init__(self, args):
        super(ResNet18, self).__init__()

        self.in_ch = args.in_ch
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_ch, out_channels=64, kernel_size=7, stride = 2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
            
        )

        self.conv_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            basic_block(in_channels = 64, out_channels = 64, kernel_size = 3,  padding = 1),
            basic_block(in_channels = 64, out_channels = 64, kernel_size = 3,  padding = 1)
        )

        self.conv_3 = nn.Sequential(
            down_block(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1, down_stride = 2),
            basic_block(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        )

        self.conv_4 = nn.Sequential(
            down_block(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1, down_stride = 2),
            basic_block(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        )

        self.conv_5 = nn.Sequential(
            down_block(in_channels = 256, out_channels = 512, kernel_size = 3, padding = 1, down_stride = 2),
            basic_block(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
        )

        self.gap = nn.AdaptiveAvgPool2d(output_size = (1, 1))
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        o = self.conv_1(x)
        o = self.conv_2(o)
        o = self.conv_3(o)
        o = self.conv_4(o)
        o = self.conv_5(o)
        o = self.gap(o)
        o = self.fc(o.reshape(o.shape[0], -1))

        return o
        

class ResNet50(nn.Module):
    def __init__(self, args):
        super(ResNet50, self).__init__()

        self.in_ch = args.in_ch
        self.conv_1 = nn.Sequential(
            nn.Conv2d(self.in_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)   
        )

        self.conv_2 = nn.Sequential(
            bottle_neck(64, 64, 256, downsample_stride = 1),
            bottle_neck(256, 64, 256),
            bottle_neck(256, 64, 256),

        )

        self.conv_3 = nn.Sequential(
            bottle_neck(256, 128, 512, downsample_stride = 2),
            bottle_neck(512, 128, 512),
            bottle_neck(512, 128, 512),
            bottle_neck(512, 128, 512),

        )

        self.conv_4 = nn.Sequential(
            bottle_neck(512, 256, 1024, downsample_stride = 2),
            bottle_neck(1024, 256, 1024),
            bottle_neck(1024, 256, 1024),
            bottle_neck(1024, 256, 1024),
            bottle_neck(1024, 256, 1024),
            bottle_neck(1024, 256, 1024),

        )

        self.conv_5 = nn.Sequential(
            bottle_neck(1024, 512, 2048, downsample_stride = 2),
            bottle_neck(2048, 512, 2048),
            bottle_neck(2048, 512, 2048),

        )

        self.gap = nn.AdaptiveAvgPool2d(output_size = (1, 1))
        self.fc = nn.Linear(2048, 2)


    def forward(self, x):
        o = self.conv_1(x)  #([1, 64, 56, 56])
        o = self.conv_2(o)
        o = self.conv_3(o)
        o = self.conv_4(o)
        o = self.conv_5(o) #([1, 2048, 7, 7])
        o = self.gap(o)
        o = self.fc(o.reshape(o.shape[0], -1))

        return o
    

def Bottleneck_build(filter_nums, block_nums, downsample_stride = None):
    build_model = []
    build_model.append(bottle_neck(filter_nums[0], filter_nums[1], filter_nums[2], downsample_stride = downsample_stride))
    filter_nums[0] = filter_nums[2]
    for _ in range(block_nums - 1):
        build_model.append(bottle_neck(filter_nums[0], filter_nums[1], filter_nums[2]))
    return nn.Sequential(*build_model)


class ResNet152(nn.Module):
    def __init__(self, args):
        super(ResNet152, self).__init__()

        self.in_ch = args.in_ch
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_ch, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),
            Bottleneck_build(filter_nums=[64, 64, 256], block_nums=3, downsample_stride=1)
        )
        self.conv3 = Bottleneck_build(filter_nums=[256, 128, 512], block_nums=8, downsample_stride=2)
        self.conv4 = Bottleneck_build(filter_nums=[512, 256, 1024], block_nums=36, downsample_stride=2)
        self.conv5 = Bottleneck_build(filter_nums=[1024, 512, 2048], block_nums=3, downsample_stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(2048, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        out = self.fc(x)
        return out
    

if __name__ == '__main__':
    data = torch.randn(1, 3, 224, 224)
    data = data.cuda()
    model = ResNet152()
    model.cuda()

    o = model(data)
    print(o.shape)
