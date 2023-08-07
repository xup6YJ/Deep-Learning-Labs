

from utils import *
import matplotlib.pyplot as plt
import pandas as pd

def plot_result():
    plt.figure(figsize=(10, 5))
    epochs = range(1, 300 +1)

    rn18 = pd.read_csv('/media/yclin/3TBNAS/DLP/Lab3/csv/ResNet18.csv')
    rn18_val = pd.read_csv('/media/yclin/3TBNAS/DLP/Lab3/csv/ResNet18_val.csv')
    rn50 = pd.read_csv('/media/yclin/3TBNAS/DLP/Lab3/csv/ResNet50.csv')
    rn50_val = pd.read_csv('/media/yclin/3TBNAS/DLP/Lab3/csv/ResNet50_val.csv')
    rn152 = pd.read_csv('/media/yclin/3TBNAS/DLP/Lab3/csv/ResNet152.csv')
    rn152_val = pd.read_csv('/media/yclin/3TBNAS/DLP/Lab3/csv/ResNet152_val.csv')

    plt.plot(epochs, rn18['Value'], 'r', label = 'ResNet18_train', linestyle="-.")
    plt.plot(epochs, rn18_val['Value'], 'b', label = 'ResNet18_val',  linewidth="1")
    plt.plot(epochs, rn50['Value'], 'g', label = 'ResNet50_train', linestyle="-.")
    plt.plot(epochs, rn50_val['Value'], 'c',  label = 'ResNet50_val', linewidth="1")
    plt.plot(epochs, rn152['Value'], 'm', label = 'ResNet152_train', linestyle="-.")
    plt.plot(epochs, rn152_val['Value'], 'y', label = 'ResNet152_val', linewidth="1")

    plt.title(f'Model accuracy comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    # plt.figure()
    plt.savefig('/media/yclin/3TBNAS/DLP/Lab3/Comparison.jpeg')
    # plt.show()
    plt.close()



plot_result()