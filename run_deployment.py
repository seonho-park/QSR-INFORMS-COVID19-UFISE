import torch
import argparse

from dataset import COVID19DataSet
from Model import mobilenet_v2, densenet121, ResNetUNet
import utils
from train import train
from test import validate


def main():
    # logger, result_dir, _ = utils.config_backup_get_log(args,__file__)

    device = utils.get_device()
    utils.set_seed(args.seed, device) # set random seed

    trainset = COVID19DataSet(root = args.datapath, ctonly = args.ctonly) # load dataset
    if args.model.lower() in ['mobilenet']:
        net = mobilenet_v2(task = 'classification', moco = args.moco, ctonly = args.ctonly).to(device)
    elif args.model.lower() in ['densenet']:
        net = densenet121(task = 'classification', moco = args.moco, ctonly = args.ctonly).to(device)
    else:
        raise Exception
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)   
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bstrain, shuffle=True, num_workers = args.nworkers)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=args.bstest, shuffle=False, num_workers = args.nworkers)
    print('==> Start training ..')   
    for epoch in range(args.maxepoch):
        net = train(epoch, net, trainloader, criterion, optimizer, scheduler, args.model, device)
        scheduler.step()
        if epoch%5 == 0:
            auroc, aupr, f1_score, accuracy = validate(net, testloader, device)
    
    # load lungseg state_dict
    state_dict_lungseg = torch.load("lungseg_net.pth")
    state_dict = {'classifier': net.to('cpu').state_dict() , 'lungseg': state_dict_lungseg}
    torch.save(state_dict,'model.pth')


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
    parser.add_argument('--moco', action='store_true', help='using moco pretraining')
    parser.add_argument('--ctonly', action='store_true', help='using ctimages only for the input of the network')
    parser.add_argument('--suffix', type=str, default='test', help='suffix of result directory')
    parser.add_argument('--batchout', action='store_true', help='batch out')
    args = parser.parse_args()
    args.moco = False
    args.ctonly = False

    main()
