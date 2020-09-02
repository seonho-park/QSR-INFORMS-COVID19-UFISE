import torch
import torch.nn.functional as F
import torchvision
from collections import defaultdict
import os
import glob
import argparse
import numpy as np
import scipy.signal as ss
import scipy.ndimage as snd
import skimage.transform as skt
from PIL import Image

from Model import ResNetUNet
import utils

class LungSegDataSet(torch.utils.data.Dataset):
    def __init__(self, root):
        self.img_list = sorted(glob.glob(os.path.join(root, '2d_images/*.tif')))
        self.mask_list = sorted(glob.glob(os.path.join(root, '2d_masks/*.tif')))
        self.dims = [480,480]
        assert len(self.img_list) == len(self.mask_list)
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):   
        img = Image.open(self.img_list[idx])
        img = np.asarray(img) # convert to numpy array, order: HxWxC
        img = img.astype(np.float32)/255. # normalize value to [0.,1.]
        
        mask = Image.open(self.mask_list[idx])
        mask = np.asarray(mask)
        mask = mask.astype(np.float32)/255.
        
        # preprocessing img and mask
        img = skt.resize(img, (self.dims[0], self.dims[1]), mode='constant', anti_aliasing=False)
        bandwidth = 255
        img = (img - img.min()) / (img.max() - img.min())
        imhist = (img * bandwidth).astype(np.uint8)
        h = np.histogram(imhist.flatten(), bins=bandwidth+1)
        hmed = ss.medfilt(h[0], kernel_size=51)
        hf = snd.gaussian_filter(hmed, sigma=25)
        hf = np.maximum(0, hf - len(img.flatten())*0.001) # reject 0.1% of mass
        if np.max(hf) > 0:
            hmin = np.nonzero(hf)[0][0] /bandwidth
            hmax = np.nonzero(hf)[0][-1]/bandwidth
            img = (img - hmin) / (hmax - hmin)
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img).float()

        mask = skt.resize(mask, (self.dims[0], self.dims[1]), mode='constant', anti_aliasing=False)
        mask = np.expand_dims(mask, 0)
        mask = torch.from_numpy(mask).float()
        
        return img, mask

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
        
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    loss = bce * bce_weight + dice * (1 - bce_weight)
    
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    return loss

def print_metrics(metrics, epoch_samples):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}".format(", ".join(outputs)))    


def train(epoch, net, trainloader, optimizer, device):
    net.train()  # Set model to training mode
    train_loss = 0.
    metrics = defaultdict(float)
    epoch_samples = 0
    for batch_idx, (imgs, labels) in enumerate(trainloader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net(imgs)
        loss = calc_loss(outputs, labels, metrics)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        epoch_samples += imgs.size(0)
        print('  Training... Epoch: %4d | Iter: %4d/%4d | Train Loss: %.4f'%(epoch, batch_idx+1, len(trainloader), train_loss/(batch_idx+1)), end = '\r')
    print('')
    print_metrics(metrics, epoch_samples)
    return net


def main():
    device = utils.get_device()
    utils.set_seed(args.seed, device) # set random seed

    dataset = LungSegDataSet(args.datapath)
    # ntrain = int(0.8*len(dataset)) # 80% of the data is set to be a training dataset
    # trainset, testset = torch.utils.data.random_split(dataset, (ntrain, len(dataset)-ntrain))
    
    net = ResNetUNet(n_class=1).to(device)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)   
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.bstrain, shuffle=True, num_workers = args.nworkers)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bstrain, shuffle=True, num_workers = args.nworkers)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=args.bstest, shuffle=False, num_workers = args.nworkers)
    for epoch in range(args.maxepoch):
        scheduler.step()
        net = train(epoch, net, trainloader, optimizer, device)
    net = net.to('cpu')
    state = net.state_dict()
    torch.save(state, 'lungseg_net.pth')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default="/home/sean/data/COVID-CT QSR Data Challenge/LungSegCT_kaggle", help='data path')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--bstrain', type=int, default=10, help='batch size for training')
    parser.add_argument('--bstest', type=int, default=64, help='batch size for testing')
    parser.add_argument('--nworkers', type=int, default=2, help='the number of workers used in DataLoader')
    parser.add_argument('--maxepoch', type=int, default=100, help='the number of epoches')
    args = parser.parse_args()
    main()