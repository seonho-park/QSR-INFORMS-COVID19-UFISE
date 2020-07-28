import argparse
import torch

from dataset import COVID19DataSet
import model
import utils





def train(epoch, net, trainloader, criterion, optimizer, device):
    train_loss = 0.
    for batch_idx, (imgs, labels) in enumerate(trainloader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = net(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print('  Training... Epoch: %4d | Iter: %4d/%4d | Train Loss: %.4f'%(epoch, batch_idx+1, len(trainloader), train_loss/(batch_idx+1)), end = '\r')
    print('')
    return net


def main():
    device = utils.get_device()
    utils.set_seed(args.seed, device) # set random seed

    dataset = COVID19DataSet(args.datapath) # load dataset
    img, label = dataset[1]
    
    net = model.setup_model(args.model).to(device)
    
    ntrain = int(0.8*len(dataset)) # 80% of the data is set to be a training dataset
    trainset, testset = torch.utils.data.random_split(dataset, (ntrain, len(dataset)-ntrain))
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bstrain, shuffle=True, num_workers = args.nworkers)
    for epoch in range(args.maxepoch):
        net = train(epoch, net, trainloader, criterion, optimizer, device)
    # net = train(net, trainset, device)

    # test(net, test_data)



    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default="/home/sean/data/COVID-CT QSR Data Challenge/COVID-CT QSR Data Challenge/Images-processed", help='data path')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--model', type=str, default='mobilenet', help='backbone architecture mobilenet|xxx|xxx')
    parser.add_argument('--bstrain', type=int, default=32, help='batch size for training')
    parser.add_argument('--bstest', type=int, default=64, help='batch size for testing')
    parser.add_argument('--nworkers', type=int, default=2, help='the number of workers used in DataLoader')
    parser.add_argument('--maxepoch', type=int, default=200, help='the number of epoches')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    args = parser.parse_args()
    main()
