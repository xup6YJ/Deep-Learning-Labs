

import cv2
import os
import glob
from tqdm import tqdm

data_path = '/media/yclin/3TBNAS/DLP/Lab3/new_dataset'
datasets = os.listdir(data_path)
import pandas as pd


for dataset in datasets:
    sub_path = os.path.join(data_path, dataset)
    image_paths = sorted(glob.glob(os.path.join(sub_path, '*.bmp')))

    new_path = os.path.join(data_path, f'new_{dataset}')
    if not os.path.exists(new_path):
        os.mkdir(new_path)

    for img_path in tqdm(image_paths):
        # img0 = cv2.imread(img_path)
        img = cv2.imread(img_path, 0)
        # cv2.imshow('Org', img0)
        # cv2.imshow('Gray', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        img_eq = cv2.equalizeHist(img)
        # cv2.imshow('EQ', img_eq)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # img_o = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2BGR)
        # cv2.imshow('o', img_o)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        _, name = os.path.split(img_path)
        cv2.imwrite(os.path.join(new_path, name), img_eq)

        # csv_path = '/media/yclin/3TBNAS/DLP/Lab3/train.csv'




    