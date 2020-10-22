import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score


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
