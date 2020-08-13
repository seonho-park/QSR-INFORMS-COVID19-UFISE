import os
import argparse
import torch
import pandas as pd
import numpy as np

from dataset import COVID19DataSet
from Model import mobilenet_v2
import utils
from test import validate


NUM_COVID = 251
NUM_NONCOVID = 292
NTRAIN_RATIO = 0.8 

def train(epoch, net, trainloader, criterion, optimizer, device):
    train_loss = 0.
    for batch_idx, (imgs, lungsegs, labels) in enumerate(trainloader):
        imgs = imgs.to(device)
        lungsegs = lungsegs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = net(imgs, lungsegs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print('  Training... Epoch: %4d | Iter: %4d/%4d | Train Loss: %.4f'%(epoch, batch_idx+1, len(trainloader), train_loss/(batch_idx+1)), end = '\r')
    print('')
    return net

def split_dataset(dataset, root):
    imgpath_dict = {os.path.basename(img_fn): j for j, img_fn in enumerate(dataset.imgs)}

    df_covid = pd.read_excel(os.path.join(root, "CT-MetaInfo.xlsx"), sheet_name='COVID-CT-info')  
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
            # print(i, pid, fns)
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
    df_noncovid = pd.read_excel(os.path.join(root, "CT-MetaInfo.xlsx"), sheet_name='NonCOVID-CT-info')
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

    print("The number of training data:", len(indices_train))
    print("The number of test data:", len(indices_test))

    return trainset, testset


def main():
    device = utils.get_device()
    utils.set_seed(args.seed, device) # set random seed

    dataset = COVID19DataSet(root = args.datapath) # load dataset
    trainset, testset = split_dataset(dataset = dataset, root = args.datapath)
    
    # net = mobilenet_v2(pretrained = True, num_classes = 1).to(device)
    net = mobilenet_v2(pretrained = False, num_classes = 128)
    state = torch.load("./chpt/moco_output.pth.tar")["state_dict"]
    state_dict = {k.replace('encoder_q.',''):state[k] for k in state.keys() if 'encoder_q' in k}
    net.load_state_dict(state_dict)
    net.classifier_new = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1280, 1),
        )
    net = net.to(device)

    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)   
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bstrain, shuffle=True, num_workers = args.nworkers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bstest, shuffle=False, num_workers = args.nworkers)
    for epoch in range(args.maxepoch):
        net = train(epoch, net, trainloader, criterion, optimizer, device)
        scheduler.step()
        if epoch%10 == 0:
            auroc, aupr, f1_score, accuracy = validate(net, testloader, device)
    auroc, aupr, f1_score, accuracy = validate(net, testloader, device)
    net = net.to('cpu')
    state = net.state_dict()
    torch.save(state, 'model.pth')

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default="/home/sean/data/COVID-CT QSR Data Challenge/COVID-CT QSR Data Challenge", help='data path')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--model', type=str, default='mobilenet', help='backbone architecture mobilenet|xxx|xxx')
    parser.add_argument('--bstrain', type=int, default=32, help='batch size for training')
    parser.add_argument('--bstest', type=int, default=64, help='batch size for testing')
    parser.add_argument('--nworkers', type=int, default=2, help='the number of workers used in DataLoader')
    parser.add_argument('--maxepoch', type=int, default=100, help='the number of epoches')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    args = parser.parse_args()
    main()
