import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from Model import predict


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
    auroc = roc_auc_score(gts, probs)
    precision, recall, thresholds = precision_recall_curve(gts, probs)
    aupr = auc(recall, precision)
    f1 = f1_score(gts, preds)

    num_correct = (preds == gts).sum()
    accuracy = num_correct/gts.shape[0]
    print('  AUROC: %5.4f | AUPR: %5.4f | F1_Score: %5.4f | Accuracy: %5.4f (%d/%d)'%(auroc, aupr, f1, accuracy, num_correct, gts.shape[0]))
    

    return auroc, aupr, f1, accuracy

def test():
    from PIL import Image
    x_test1 = Image.open('/home/sean/data/COVID-CT QSR Data Challenge/COVID-CT QSR Data Challenge/Images-processed/CT_COVID/2020.02.10.20021584-p6-52%2.png')
    x_test2 = Image.open('/home/sean/data/COVID-CT QSR Data Challenge/COVID-CT QSR Data Challenge/Images-processed/CT_NonCOVID/33%1.jpg')
    x_test3 = Image.open('/home/sean/data/COVID-CT QSR Data Challenge/COVID-CT QSR Data Challenge/Images-processed/CT_NonCOVID/1331.png')
    x_test4 = Image.open('/home/sean/data/COVID-CT QSR Data Challenge/COVID-CT QSR Data Challenge/Images-processed/CT_COVID/2020.02.25.20021568-p24-111%2.png')
    x_test1 = np.asarray(x_test1)
    x_test2 = np.asarray(x_test2)
    x_test3 = np.asarray(x_test3)
    x_test4 = np.asarray(x_test4)
    x_test = [x_test1, x_test2, x_test3, x_test4]
    y_pred = predict(x_test)
    print(y_pred)


if __name__ == "__main__":
    test()
