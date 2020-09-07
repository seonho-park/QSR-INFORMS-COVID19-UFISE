import torch
import torchvision
import numpy as np
import scipy.signal as ss
import scipy.ndimage as snd
import skimage.transform as skt
from PIL import Image
import glob
import os
import random

import pandas as pd


class CTImageAdjustment(object):

    def __init__(self, dims=[480,480]):
        self.dims = dims

    def __call__(self, x):
        img = np.asarray(x) # convert to numpy array, order: HxWxC
        img = img.astype(np.float32)/255. # normalize value to [0.,1.]
        # adjust brightness
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
        img = Image.fromarray(img, mode = 'RGB')
        return img



class COVID19DataSet(torch.utils.data.Dataset):
    def __init__(self, root, ctonly, dims=[480,480]): # from the reference... the size of the image is 480 by 480
        self.ctonly = ctonly
        self.dims = dims # dimension of the image 
        imgs_covid = sorted(glob.glob(os.path.join(root, 'Images-processed', 'CT_COVID', '*.*')))
        imgs_noncovid = sorted(glob.glob(os.path.join(root, 'Images-processed', 'CT_NonCOVID', '*.*')))
        self.imgs = imgs_covid + imgs_noncovid
        self.labels = [1]*len(imgs_covid) + [0]*len(imgs_noncovid)

        lungsegs_covid = sorted(glob.glob(os.path.join(root, 'lung_segmentation', 'CT_COVID', '*.*')))
        lungsegs_noncovid = sorted(glob.glob(os.path.join(root, 'lung_segmentation', 'CT_NonCOVID', '*.*')))
        self.lungsegs = lungsegs_covid+lungsegs_noncovid

        assert len(self.imgs) == len(self.lungsegs)
        assert len(self.labels) == len(self.lungsegs)
        self.set_indices_train(list(range(len(self))))

    def set_indices_train(self, indices_train):
        self.indices_train = indices_train

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if random.random() < 0.5 and idx in self.indices_train:
            hflip = True
        else: 
            hflip = False
        
        label = torch.FloatTensor([self.labels[idx]])
        
        img = Image.open(self.imgs[idx]).convert('L') # greyscale
        if hflip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        img = np.asarray(img) # convert to numpy array, order: HxWxC
        img = img.astype(np.float32)/255. # normalize value to [0.,1.]
        # adjust brightness
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
        img = torch.from_numpy(img).unsqueeze(0).float()
        
        # lung segmentation
        if not self.ctonly:
            lungseg = Image.open(self.lungsegs[idx]).convert('L') # greyscale
            if hflip:
                lungseg = lungseg.transpose(Image.FLIP_LEFT_RIGHT)
            lungseg = np.asarray(lungseg)
            lungseg = lungseg.astype(np.float32)/255.
            lungseg = skt.resize(lungseg, (self.dims[0], self.dims[1]), mode='constant', anti_aliasing=False) # resize
            lungseg = torch.from_numpy(lungseg).unsqueeze(0).float()
            return img, lungseg, label
        else:
            pseudo_lungseg = torch.FloatTensor([0.])
            return img, pseudo_lungseg, label
