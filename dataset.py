import torch
import torchvision
import numpy as np
import scipy.signal as ss
import scipy.ndimage as snd
import skimage.transform as skt
from PIL import Image
import glob
import os

import pandas as pd

class COVID19DataSet(torch.utils.data.Dataset):
    def __init__(self, root, dims=[480,480]): # from the reference... the size of the image is 480 by 480
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

    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, idx):
        # img, label = super(COVID19DataSet, self).__getitem__(idx)
        
        img = Image.open(self.imgs[idx]).convert('L')
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
        
        # lung segmentation
        lungseg = Image.open(self.lungsegs[idx]).convert('L')
        lungseg = np.asarray(lungseg)
        lungseg = lungseg.astype(np.float32)/255.
        lungseg = skt.resize(lungseg, (self.dims[0], self.dims[1]), mode='constant', anti_aliasing=False) # resize

        # img = torch.from_numpy(img).unsqueeze(0).float()
        # img = np.transpose(img,(2,0,1)) #change the order to CxHxW
        img = torch.from_numpy(img).unsqueeze(0).float()
        lungseg = torch.from_numpy(lungseg).unsqueeze(0).float()
        label = torch.FloatTensor([self.labels[idx]])
        return img, lungseg, label


# class COVID19DataSetTest(torch.utils.data.Dataset):
#     def __init__(self, x_test):
#         self.x_test = x_test
#         self.dims = [480,480]

#     def __len__(self):
#         return len(self.x_test)

#     def __getitem__(self, idx):
#         x = self.x_test[idx]
#         img = np.asarray(x)
#         img = img.astype(np.float32)/255. # normalize value to [0.,1.]
#         if img.shape[2] > 3: # bypass RGBA case
#             img = img[:,:,0:3]
#         # adjust brightness
#         img = skt.resize(img, (self.dims[0], self.dims[1]), mode='constant', anti_aliasing=False)
#         bandwidth = 255
#         img = (img - img.min()) / (img.max() - img.min())
#         imhist = (img * bandwidth).astype(np.uint8)
#         h = np.histogram(imhist.flatten(), bins=bandwidth+1)
#         hmed = ss.medfilt(h[0], kernel_size=51)
#         hf = snd.gaussian_filter(hmed, sigma=25)
#         hf = np.maximum(0, hf - len(img.flatten())*0.001) # reject 0.1% of mass
#         if np.max(hf) > 0:
#             hmin = np.nonzero(hf)[0][0] /bandwidth
#             hmax = np.nonzero(hf)[0][-1]/bandwidth
#             img = (img - hmin) / (hmax - hmin)
#         # img = torch.from_numpy(img).unsqueeze(0).float()
#         img = np.transpose(img,(2,0,1)) #change the order to CxHxW
#         img = torch.from_numpy(img).float()
#         return img

