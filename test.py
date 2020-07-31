import torch
import numpy as np
# from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score

def validate(net, testloader, device):
    probs = []
    gts = []
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(testloader):
            imgs = imgs.to(device)
            # labels = labels.to(device)
            # optimizer.zero_grad()
            logits = net(imgs)
            probs.append(torch.sigmoid(logits))
            gts.append(labels)
            # loss = criterion(logits, labels)
            # loss.backward()
            # optimizer.step()
            # train_loss += loss.item()
            # print('  Training... Epoch: %4d | Iter: %4d/%4d | Train Loss: %.4f'%(epoch, batch_idx+1, len(trainloader), train_loss/(batch_idx+1)), end = '\r')
        # print('')
        # return net
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
