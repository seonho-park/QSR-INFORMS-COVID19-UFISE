import os
import argparse
import torch
import pandas as pd
import numpy as np
import time

from dataset import COVID19DataSet
from model import mobilenet_v2, densenet121
import utils
from test import validate


NUM_COVID = 251
NUM_NONCOVID = 292
NTRAIN_RATIO = 0.8 

def train(epoch, net, trainloader, criterion, optimizer, scheduler, model, device):
    net.train() # train mode
    train_loss = 0.
    
    for batch_idx, (imgs, lungsegs, labels) in enumerate(trainloader):
        imgs = imgs.to(device)
        lungsegs = lungsegs.to(device)
        labels = labels.to(device)
        if model in ['densenet']:
            if batch_idx%2 == 0:
                optimizer.zero_grad()
        else:
            optimizer.zero_grad()
        
        logits = net(imgs, lungsegs)
        loss = criterion(logits, labels)
        loss.backward()
        if model in ['densenet']:
            if batch_idx%2 == 1:
                optimizer.step()
        else:
            optimizer.step()

        train_loss += loss.item()
        print('  Training... Epoch: %4d | Iter: %4d/%4d | Train Loss: %.4f'%(epoch, batch_idx+1, len(trainloader), train_loss/(batch_idx+1)), end = '\r')
    print('')

    scheduler.step()
    return net

def split_dataset(dataset, logger):
    imgpath_dict = {os.path.basename(img_fn): j for j, img_fn in enumerate(dataset.imgs)}

    df_covid = pd.read_excel("./CT-MetaInfo.xlsx", sheet_name='COVID-CT-info')  
    df_covid = df_covid[['File name', 'Patient ID']]

    # process COVID
    fn_covid = df_covid['File name'].tolist()[:NUM_COVID]
    pid_covid = df_covid['Patient ID'].tolist()[:NUM_COVID]

    pid_fn_dict = {}
    for pid, fn in zip(pid_covid, fn_covid):
        if pid in pid_fn_dict:
            fn_exist = pid_fn_dict[pid]
            fn_exist.append(fn)
            pid_fn_dict[pid] = fn_exist
        else:
            pid_fn_dict[pid] = [fn]

    pid_covid_unique = sorted(list(set(pid_covid)))
    indices = torch.randperm(len(pid_covid_unique))
    pid_train = indices[:int(NTRAIN_RATIO*len(pid_covid_unique))]
    pid_test = indices[int(NTRAIN_RATIO*len(pid_covid_unique)):]

    indices_covid_train = []    
    for i in pid_train:
        pid = pid_covid_unique[i]
        fns = pid_fn_dict[pid]
        for fn in fns:
            if fn in imgpath_dict:
                indices_covid_train.append(imgpath_dict[fn])
            else:
                raise ValueError

    indices_covid_test = []    
    for i in pid_test:
        pid = pid_covid_unique[i]
        fns = pid_fn_dict[pid]
        for fn in fns:
            if fn in imgpath_dict:
                indices_covid_test.append(imgpath_dict[fn])
            else:
                raise ValueError
    assert len(indices_covid_train) + len(indices_covid_test) == NUM_COVID

    # process NonCOVID
    df_noncovid = pd.read_excel("CT-MetaInfo.xlsx", sheet_name='NonCOVID-CT-info')
    df_noncovid = df_noncovid[['image name', 'patient id']]
    fn_noncovid = df_noncovid['image name'].tolist()[:NUM_NONCOVID]
    pid_noncovid = df_noncovid['patient id'].tolist()[:NUM_NONCOVID]

    pid_fn_dict = {}
    for pid, fn in zip(pid_noncovid, fn_noncovid):
        if pid in pid_fn_dict:
            fn_exist = pid_fn_dict[pid]
            fn_exist.append(fn)
            pid_fn_dict[pid] = fn_exist
        else:
            pid_fn_dict[pid] = [fn]

    pid_noncovid_unique = sorted(list(set(pid_noncovid)))
    indices = torch.randperm(len(pid_noncovid_unique))
    pid_train = indices[:int(NTRAIN_RATIO*len(pid_noncovid_unique))]
    pid_test = indices[int(NTRAIN_RATIO*len(pid_noncovid_unique)):]

    indices_noncovid_train = []    
    for i in pid_train:
        pid = pid_noncovid_unique[i]
        fns = pid_fn_dict[pid]
        for fn in fns:
            if fn in imgpath_dict:
                indices_noncovid_train.append(imgpath_dict[fn])
            else:
                raise ValueError

    indices_noncovid_test = []    
    for i in pid_test:
        pid = pid_noncovid_unique[i]
        fns = pid_fn_dict[pid]
        for fn in fns:
            if fn in imgpath_dict:
                indices_noncovid_test.append(imgpath_dict[fn])
            else:
                raise ValueError
    assert len(indices_noncovid_train) + len(indices_noncovid_test) == NUM_NONCOVID

    indices_train = indices_covid_train+indices_noncovid_train
    indices_test = indices_covid_test+indices_noncovid_test
    assert len(indices_train) == len(set(indices_train))
    assert len(indices_test) == len(set(indices_test))
    assert len(list(set(indices_test) & set(indices_train))) == 0

    trainset = torch.utils.data.Subset(dataset, indices_train)
    testset = torch.utils.data.Subset(dataset, indices_test)
    dataset.set_indices_train(indices_train)

    print("The number of training CT images:", len(indices_train))
    print("The number of test CT images:", len(indices_test))
    if logger is not None:
        logger.write("The number of training CT images:%d\n"%(len(indices_train)))
        logger.write("The number of test CT images:%d\n"%(len(indices_test)))

    return trainset, testset


def main():
    logger, result_dir, _ = utils.config_backup_get_log(args,__file__)

    device = utils.get_device()
    utils.set_seed(args.seed, device) # set random seed

    dataset = COVID19DataSet(root = args.datapath, ctonly = args.ctonly) # load dataset
    trainset, testset = split_dataset(dataset = dataset, logger = logger)
    
    if args.model.lower() in ['mobilenet']:
        net = mobilenet_v2(task = 'classification', moco = False, ctonly = args.ctonly).to(device)
    elif args.model.lower() in ['densenet']:
        net = densenet121(task = 'classification', moco = False, ctonly = args.ctonly).to(device)
    else:
        raise Exception
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)   
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bstrain, shuffle=True, num_workers = args.nworkers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bstest, shuffle=False, num_workers = args.nworkers)

    best_auroc = 0.
    print('==> Start training ..')   
    start = time.time()
    for epoch in range(args.maxepoch):
        net = train(epoch, net, trainloader, criterion, optimizer, scheduler, args.model, device)
        scheduler.step()
        if epoch%5 == 0:
            auroc, aupr, f1_score, accuracy = validate(net, testloader, device)
            logger.write('Epoch:%3d | AUROC: %5.4f | AUPR: %5.4f | F1_Score: %5.4f | Accuracy: %5.4f\n'%(epoch, auroc, aupr, f1_score, accuracy))
            if auroc>best_auroc:
                best_auroc = auroc
                best_aupr = aupr
                best_epoch = epoch
                print("save checkpoint...")
                torch.save(net.state_dict(), './%s/%s.pth'%(result_dir, args.model))

    auroc, aupr, f1_score, accuracy = validate(net, testloader, device)
    logger.write('Epoch:%3d | AUROC: %5.4f | AUPR: %5.4f | F1_Score: %5.4f | Accuracy: %5.4f\n'%(epoch, auroc, aupr, f1_score, accuracy))
    
    if args.batchout:
        with open('temp_result.txt', 'w') as f:
            f.write("%10.8f\n"%(best_auroc))
            f.write("%10.8f\n"%(best_aupr))
            f.write("%d"%(best_epoch))

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Elapsed Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    logger.write("Elapsed Time: {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours),int(minutes),seconds))
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default="/home/sean/data/COVID-CT QSR Data Challenge/COVID-CT QSR Data Challenge", help='data path')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--model', type=str, default='mobilenet', help='backbone architecture mobilenet|densenet')
    parser.add_argument('--bstrain', type=int, default=32, help='batch size for training')
    parser.add_argument('--bstest', type=int, default=64, help='batch size for testing')
    parser.add_argument('--nworkers', type=int, default=2, help='the number of workers used in DataLoader')
    parser.add_argument('--maxepoch', type=int, default=100, help='the number of epoches')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--ctonly', action='store_true', help='using ctimages only for the input of the network')
    parser.add_argument('--suffix', type=str, default='test', help='suffix of result directory')
    parser.add_argument('--batchout', action='store_true', help='batch out')
    args = parser.parse_args()

    main()
