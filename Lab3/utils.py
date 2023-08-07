
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn

def plot_result(result, epochs):
    plt.figure(figsize=(10, 5))
    epochs = range(1, epochs +1)

    plt.plot(epochs, result['train']['ResNet18'], 'r', label = 'ResNet18_train')
    plt.plot(epochs, result['validation']['ResNet18'], 'b', label = 'ResNet18_test')
    plt.plot(epochs, result['train']['ResNet50'], 'g', label = 'ResNet50_train')
    plt.plot(epochs, result['validation']['ResNet50'], 'c', label = 'ResNet50_test')
    plt.plot(epochs, result['train']['ResNet152'], 'm', label = 'ResNet152_train')
    plt.plot(epochs, result['validation']['ResNet152'], 'y', label = 'ResNet152_test')

    plt.title(f'Model accuracy comparison')
    plt.legend()
    # plt.figure()
    plt.savefig('/media/yclin/3TBNAS/DLP/Lab3/Comparison.jpeg')
    # plt.show()
    plt.close()



def plot_train_loss(args, loss_list, epochs):
    # TODO plot training and testing accuracy curve
    plt.figure(figsize=(10, 5))
    epochs = range(1, epochs +1)

    plt.plot(epochs, loss_list['train']['ResNet18'], 'r', label = 'ResNet18_train')
    plt.plot(epochs, loss_list['validation']['ResNet18'], 'b', label = 'ResNet18_test')
    plt.plot(epochs, loss_list['train']['ResNet50'], 'g', label = 'ResNet50_train')
    plt.plot(epochs, loss_list['validation']['ResNet50'], 'c', label = 'ResNet50_test')
    plt.plot(epochs, loss_list['train']['ResNet152'], 'm', label = 'ResNet152_train')
    plt.plot(epochs, loss_list['validation']['ResNet152'], 'y', label = 'ResNet152_test')

    # plt.plot(epochs, train_loss_list, 'b', label = 'Train loss')
    plt.title(f'{args.model_n} Train and Test Loss')
    plt.legend()
    # plt.figure()
    plt.savefig('./{}/{}_Loss.jpeg'.format(args.folder, args.model_n))
    # plt.show()
    plt.close()


def plot_confusion_matrix(args, y_true, y_pred, classes=[0, 1],
                          title=None,
                          cmap=plt.cm.Blues,
                          filename=None):
    
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize='true')

    fig, ax = plt.subplots()
    # fig, ax = plt.subplots(1,1, figsize = (10, 9), dpi = 80)

    sn.heatmap(cm, annot=True, ax=ax, cmap=cmap, fmt='.2f')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.xaxis.set_ticklabels(classes, rotation=45)
    ax.yaxis.set_ticklabels(classes, rotation=0)
    plt.title(f'{args.model_n} Confusion matrix')
    filename = '{}/{}_5crop_{}.jpeg'.format(args.folder, args.model_n, args.five_crop)
    plt.savefig(filename, dpi=300)
