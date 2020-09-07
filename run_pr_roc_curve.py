import torch
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import numpy as np

import utils
from dataset import COVID19DataSet
from train import split_dataset
from Model import mobilenet_v2, densenet121

def main():
    print("Arguments:")
    print(vars(args))
    device = utils.get_device()

    base_fpr = np.linspace(0,1,1001)
    base_recall = np.linspace(1,0,1001)

    # load model for ctonly
    net = mobilenet_v2(task = 'classification', moco = args.moco, ctonly = True).to(device)
    dataset = COVID19DataSet(root = args.datapath, ctonly = True) # load dataset
    
    precisions_ctonly = []
    tprs_ctonly = []
    aurocs_ctonly = []
    auprs_ctonly = []
    for seed in range(1000,1010):
        print(seed)
        utils.set_seed(seed, device) # set random seed
        trainset, testset = split_dataset(dataset = dataset, root = args.datapath, logger = None)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.bstest, shuffle=False, num_workers = args.nworkers)
        state = torch.load("./chpt/mobilenet_ctonly_%d.pth"%(seed))
        net.load_state_dict(state)
        net.eval()
        probs = []
        gts = []
        with torch.no_grad():
            for batch_idx, (imgs, lungsegs, labels) in enumerate(testloader):
                imgs = imgs.to(device)
                lungsegs = lungsegs.to(device)
                logits = net(imgs, lungsegs)
                probs.append(torch.sigmoid(logits))
                gts.append(labels)

        probs = torch.cat(probs, dim=0).cpu().numpy()
        gts = torch.cat(gts, dim=0).cpu().numpy()
        precision, recall, thresholds = precision_recall_curve(gts, probs)
        fpr, tpr, thresholds = roc_curve(gts, probs)
        auroc = auc(fpr, tpr)
        aupr = auc(recall, precision)
        precision = np.interp(base_recall, recall[::-1], precision[::-1])
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0
        tprs_ctonly.append(tpr)
        precisions_ctonly.append(precision)
        aurocs_ctonly.append(auroc)
        auprs_ctonly.append(aupr)


    # load model for ct + lungseg
    net = mobilenet_v2(task = 'classification', moco = args.moco, ctonly = False).to(device)
    dataset = COVID19DataSet(root = args.datapath, ctonly = False) # load dataset

    precisions = []
    tprs = []
    aurocs = []
    auprs = []
    for seed in range(1000,1010):
        print(seed)
        utils.set_seed(seed, device) # set random seed
        trainset, testset = split_dataset(dataset = dataset, root = args.datapath, logger = None)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.bstest, shuffle=False, num_workers = args.nworkers)
        state = torch.load("./chpt/mobilenet_%d.pth"%(seed))
        net.load_state_dict(state)
        net.eval()
        probs = []
        gts = []
        with torch.no_grad():
            for batch_idx, (imgs, lungsegs, labels) in enumerate(testloader):
                imgs = imgs.to(device)
                lungsegs = lungsegs.to(device)
                logits = net(imgs, lungsegs)
                probs.append(torch.sigmoid(logits))
                gts.append(labels)

        probs = torch.cat(probs, dim=0).cpu().numpy()
        gts = torch.cat(gts, dim=0).cpu().numpy()
        precision, recall, thresholds = precision_recall_curve(gts, probs)
        fpr, tpr, thresholds = roc_curve(gts, probs)
        auroc = auc(fpr, tpr)
        aupr = auc(recall, precision)
        precision = np.interp(base_recall, recall[::-1], precision[::-1])
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0
        tprs.append(tpr)
        precisions.append(precision)
        aurocs.append(auroc)
        auprs.append(aupr)

    tprs_ctonly = np.array(tprs_ctonly)
    tprs = np.array(tprs)
    mean_tprs_ctonly = tprs_ctonly.mean(axis=0)
    std_tprs_ctonly = tprs_ctonly.std(axis=0)
    mean_tprs = tprs.mean(axis=0)
    std_tprs = tprs.std(axis=0)

    precisions_ctonly = np.array(precisions_ctonly)
    precisions = np.array(precisions)
    mean_precisions_ctonly = precisions_ctonly.mean(axis=0)
    std_precisions_ctonly = precisions_ctonly.std(axis=0)
    mean_precisions = precisions.mean(axis=0)
    std_precisions = precisions.std(axis=0)
    
    mean_auroc_ctonly = auc(base_fpr, mean_tprs_ctonly)
    mean_auroc = auc(base_fpr, mean_tprs)
    std_auroc_ctonly = np.std(aurocs_ctonly)
    std_auroc = np.std(aurocs)

    mean_aupr_ctonly = auc(base_recall, mean_precisions_ctonly)
    mean_aupr = auc(base_recall, mean_precisions)
    std_aupr_ctonly = np.std(auprs_ctonly)
    std_aupr = np.std(auprs)

    tprs_upper_ctonly = np.minimum(mean_tprs_ctonly+0.5*std_tprs_ctonly, 1.)
    tprs_lower_ctonly = mean_tprs_ctonly-0.5*std_tprs_ctonly
    tprs_upper = np.minimum(mean_tprs+0.5*std_tprs, 1.)
    tprs_lower = mean_tprs-0.5*std_tprs

    precisions_upper_ctonly = np.minimum(mean_precisions_ctonly+0.5*std_precisions_ctonly, 1.)
    precisions_lower_ctonly = mean_precisions_ctonly-0.5*std_precisions_ctonly
    precisions_upper = np.minimum(mean_precisions+0.5*std_precisions, 1.)
    precisions_lower = mean_precisions-0.5*std_precisions

    
    # ROC curve
    plt.figure(1,figsize=(12,9))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(base_fpr, mean_tprs_ctonly, 'b', alpha = 0.9, label= "CT image (AUC = %0.4f$\pm$%0.4f"%(mean_auroc_ctonly,std_auroc_ctonly))
    plt.fill_between(base_fpr, tprs_lower_ctonly, tprs_upper_ctonly, color='blue', alpha=0.1)
    plt.plot(base_fpr, mean_tprs, 'tab:orange', alpha = 0.9, label= "CT image + Lung segmentation (AUC = %0.4f$\pm$%0.4f"%(mean_auroc,std_auroc))
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='tab:orange', alpha=0.1)
    plt.xlabel('False positive rate', fontsize='x-large')
    plt.ylabel('True positive rate', fontsize='x-large')
    plt.title('ROC curve', fontsize='x-large')
    plt.legend(loc='lower right', fontsize='x-large')
    plt.ylim([-0.01,1.01])
    plt.xlim([-0.01,1.01])
    plt.show()

    # # PR curve
    plt.figure(2,figsize=(12,9))
    plt.step(base_recall, mean_precisions_ctonly, 'b', where='post', alpha = 0.9, label= "CT image (AUC = %0.4f$\pm$%0.4f"%(mean_aupr_ctonly,std_aupr_ctonly))
    plt.fill_between(base_recall, precisions_lower_ctonly, precisions_upper_ctonly, color='blue', alpha=0.1)
    plt.step(base_recall, mean_precisions, 'tab:orange', where='post', alpha = 0.9, label= "CT image + Lung segmentation (AUC = %0.4f$\pm$%0.4f"%(mean_aupr,std_aupr))
    plt.fill_between(base_recall, precisions_lower, precisions_upper, color='tab:orange', alpha=0.1)

    plt.xlabel('Recall', fontsize='x-large')
    plt.ylabel('Precision', fontsize='x-large')
    plt.title('PR curve', fontsize='x-large')
    plt.legend(loc='lower left', fontsize='x-large')
    plt.ylim([-0.01,1.01])
    plt.xlim([-0.01,1.01])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default="/home/sean/data/COVID-CT QSR Data Challenge/COVID-CT QSR Data Challenge", help='data path')
    parser.add_argument('--model', type=str, default='mobilenet', help='backbone architecture mobilenet|densenet')
    parser.add_argument('--bstest', type=int, default=32, help='batch size for testing')
    parser.add_argument('--moco', action='store_true', help='using moco pretraining')
    parser.add_argument('--nworkers', type=int, default=2, help='the number of workers used in DataLoader')

    args = parser.parse_args()
    
    main()
    
