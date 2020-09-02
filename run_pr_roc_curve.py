import torch
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

import utils
from dataset import COVID19DataSet
from train import split_dataset
from Model import mobilenet_v2, densenet121

# def compute_pr(pred, gt, nthres = 100):
#     assert pred.shape == gt.shape
#     # ntest, ntrain = dist_mat.shape
#     # compute thresholds
#     max_pred = pred.max()
#     min_pred = pred.min()
#     thresholds = np.arange(start = min_dist, stop=max_dist, step = (max_dist-min_dist)/nthres)

#     # thresholds = np.sort(np.unique(dist_mat.flatten()))
#     thresholds = np.unique(thresholds)
#     print('thresholds', thresholds[:10], thresholds.shape)

#     precision = np.zeros(thresholds.size)
#     recall = np.zeros(thresholds.size)
#     preds = np.zeros(dist_mat.shape, dtype=np.bool)
#     gt_sum = 

#     for i, thr in enumerate(thresholds):
#         # t0 = time.time()
#         preds = dist_mat>thr
#         tp = gt[preds].sum()
        
#         precision[i] = tp/preds.sum()
#         recall[i] = tp/gt.sum()

#     return precision, recall, thresholds


def main():
    print("Arguments:")
    print(vars(args))
    device = utils.get_device()
    

    
    # load chpt
    utils.set_seed(1003, device) # set random seed
    dataset = COVID19DataSet(root = args.datapath, ctonly = True) # load dataset
    trainset, testset = split_dataset(dataset = dataset, root = args.datapath, logger = None)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bstest, shuffle=False, num_workers = args.nworkers)
    net = mobilenet_v2(task = 'classification', moco = args.moco, ctonly = True).to(device)
    chptpath = "./chpt/mobilenet_ctonly.pth"
    state = torch.load(chptpath)
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
    precision_ctonly, recall_ctonly, thresholds = precision_recall_curve(gts, probs)
    fpr_ctonly, tpr_ctonly, thresholds = roc_curve(gts, probs)

    # load chpt
    utils.set_seed(1004, device) # set random seed
    dataset = COVID19DataSet(root = args.datapath, ctonly = False) # load dataset
    trainset, testset = split_dataset(dataset = dataset, root = args.datapath, logger = None)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bstest, shuffle=False, num_workers = args.nworkers)
    net = mobilenet_v2(task = 'classification', moco = args.moco, ctonly = False).to(device)
    chptpath = "./chpt/mobilenet.pth"
    state = torch.load(chptpath)
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

    # ROC curve
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_ctonly, tpr_ctonly, label="CT image")
    plt.plot(fpr, tpr, label="CT image + Lung segmentation")
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

    # PR curve
    plt.figure(2)
    plt.step(recall_ctonly, precision_ctonly, where='post', label='CT image')
    plt.step(recall, precision, where='post', label='CT image + Lung segmentation')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default="/home/sean/data/COVID-CT QSR Data Challenge/COVID-CT QSR Data Challenge", help='data path')
    parser.add_argument('--seed', type=int, default=1000, help='random seed')
    # parser.add_argument('--type', type=str, default='pr', help='backbone architecture roc|pr')
    parser.add_argument('--model', type=str, default='mobilenet', help='backbone architecture mobilenet|densenet')
    # parser.add_argument('--dataname', type=str, default='Haywrd', help='data name Haywrd|ykdelB')
    # parser.add_argument('--bitlength', type=int, default=96, help='bit code size')
    parser.add_argument('--bstest', type=int, default=32, help='batch size for testing')
    # parser.add_argument('--ctonly', action='store_true', help='using ctimages only for the input of the network')
    parser.add_argument('--moco', action='store_true', help='using moco pretraining')
    parser.add_argument('--nworkers', type=int, default=2, help='the number of workers used in DataLoader')
    # parser.add_argument('--plot', dest='plot', action='store_true', help = 'conduct plotting prcurve')
    # parser.add_argument('--no-plot', dest='plot', action='store_false')
    # parser.set_defaults(plot=True)

    args = parser.parse_args()
    
    main()
    
