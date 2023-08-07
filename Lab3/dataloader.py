import pandas as pd
from PIL import Image
from torch.utils import data
import os
from PIL import Image, ImageFile
from torchvision import transforms
import torch
from tqdm import tqdm
import cv2

def getData(args, mode):
    
    if args.hist == False:
        if mode == 'train':
            df = pd.read_csv('/media/yclin/3TBNAS/DLP/Lab3/train.csv')
            path = df['Path'].tolist()
            label = df['label'].tolist()
            return path, label
        
        elif mode == "valid":
            df = pd.read_csv('/media/yclin/3TBNAS/DLP/Lab3/valid.csv')
            path = df['Path'].tolist()
            label = df['label'].tolist()
            return path, label
        
        else:
            df = pd.read_csv(args.test_path)
            path = df['Path'].tolist()
            return path
    else:
        if mode == 'train':
            df = pd.read_csv('/media/yclin/3TBNAS/DLP/Lab3/train.csv')
            df['Path'] = df['Path'].str.replace('train','new_train')
            path = df['Path'].tolist()
            label = df['label'].tolist()
            return path, label
        
        elif mode == "valid":
            df = pd.read_csv('/media/yclin/3TBNAS/DLP/Lab3/valid.csv')
            df['Path'] = df['Path'].str.replace('valid','new_valid')
            path = df['Path'].tolist()
            label = df['label'].tolist()
            return path, label
        
        else:
            df = pd.read_csv(args.test_path)
            df['Path'] = df['Path'].str.replace('test','new_test')
            path = df['Path'].tolist()
            return path


def transform(args, mode ='train'):
    size = (args.img_x, args.img_y)

    if mode == 'train':
        transform =  transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize(size),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=20),
            transforms.CenterCrop((384, 384))
        ])
    else:
        if args.five_crop == 10:
            transform = transforms.Compose([
                        # transforms.Resize(size),
                        transforms.TenCrop((352, 352)),
                        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
                        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        ])
        elif args.five_crop == 5:
            transform = transforms.Compose([
                        # transforms.Resize(size),
                        transforms.FiveCrop((352, 352)),
                        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
                        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        ])
        else:
            transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.CenterCrop((384, 384))
                        # transforms.Resize(size),
                        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
    return transform


class LeukemiaLoader(data.Dataset):
    def __init__(self, args, root, mode):
        """
        Args:
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        if mode == 'test':
            self.img_name = getData(args, mode)
        else:
            self.img_name, self.label = getData(args, mode)

        self.mode = mode
        self.transform = transform(args, mode = mode)

        print("> Found %d images..." % (len(self.img_name)))  

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        if self.mode != 'test':
            """
            step1. Get the image path from 'self.img_name' and load it.
                    hint : path = root + self.img_name[index] + '.jpeg' """
            
            f_name = self.img_name[index]
            # img_path = os.path.join(self.root, self.mode, f_name)
            img_path = os.path.join(f_name)
            # img = torch.load(img_path)
            img = Image.open(img_path)

            # img = cv2.imread(img_path)

            """  
            step2. Get the ground truth label from self.label
            """
            label = self.label[index]

            """
            step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                    rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                        
                    In the testing phase, if you have a normalization process during the training phase, you only need 
                    to normalize the data. 
                    
                    hints : Convert the pixel value to [0, 1]
                            Transpose the image shape from [H, W, C] to [C, H, W]"""
            
            img = self.transform(img)

            """                 
                step4. Return processed image and label
            """

            return img, label
        
        else:
            f_name = self.img_name[index]
            img_path = os.path.join(f_name)
            img = Image.open(img_path)

            img = self.transform(img)

            return img
    

if __name__ == '__main__':
    root = '/media/yclin/3TBNAS/DLP/Lab3/new_dataset'
    LeukemiaLoader(root = root, mode = 'train')