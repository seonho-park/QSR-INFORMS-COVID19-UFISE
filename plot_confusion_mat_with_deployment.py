import argparse
import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pylab as pl

from dataset import COVID19DataSet
from Model import mobilenet_v2, ResNetUNet
import utils
from test import validate


def validate(net, testloader, device):
    probs = []
    gts = []
    net.eval() # eval mode

    with torch.no_grad():
        for batch_idx, (imgs, lungsegs, labels) in enumerate(testloader):
            imgs = imgs.to(device)
            lungsegs = lungsegs.to(device)
            logits = net(imgs, lungsegs)
            probs.append(torch.sigmoid(logits))
            gts.append(labels)

    probs = torch.cat(probs, dim=0)
    preds = torch.round(probs).cpu().numpy()
    
    probs = probs.cpu().numpy()
    gts = torch.cat(gts, dim=0).cpu().numpy()

    num_correct = (preds == gts).sum()
    accuracy = num_correct/gts.shape[0]
    print("ACC:", accuracy)    
    return preds

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def main():
    device = utils.get_device()
    utils.set_seed(1, device) # set random seed

    trainset = COVID19DataSet(root = args.datapath, ctonly = False) # load dataset
    testloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False, num_workers = args.nworkers)
    net = mobilenet_v2(task = 'classification', moco = False, ctonly = False).to(device)
    
    state = torch.load("model.pth")
    net.load_state_dict(state['classifier'])
    
    preds = validate(net, testloader, device)
    cm = confusion_matrix(np.array(trainset.labels), preds)
    plot_confusion_matrix(cm, ['NonCOVID', 'COVID'])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default="/home/sean/data/COVID-CT QSR Data Challenge/COVID-CT QSR Data Challenge", help='data path')
    parser.add_argument('--nworkers', type=int, default=2, help='the number of workers used in DataLoader')
    args = parser.parse_args()
    main()