import torch
import torchvision
import numpy as np
import scipy.signal as ss
import scipy.ndimage as snd
import skimage.transform as skt
from PIL import Image


class COVID19DataSet(torchvision.datasets.ImageFolder):
    def __init__(self, data_path, dims=[480,480]): # from the reference... the size of the image is 480 by 480
        super(COVID19DataSet, self).__init__(root=data_path)
        self.dims = dims # dimension of the image 
        self.class_to_idx = {'CT_COVID': 1, 'CT_NonCOVID': 0}
        self.targets = (-1*(np.asarray(self.targets)-1)).tolist() # swarp the indices... 1 for covid and 0 for noncovid


    def __getitem__(self, idx):
        # img, label = super(COVID19DataSet, self).__getitem__(idx)
        label = self.targets[idx]
        img = Image.open(self.samples[idx][0]).convert('RGB')
        img = np.asarray(img) # convert to numpy array, order: HxWxC
        img = img.astype(np.float32)/255. # normalize value to [0.,1.]

        if img.shape[2] > 3: # bypass RGBA case
            img = img[:,:,0:3]
        
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
        # img = torch.from_numpy(img).unsqueeze(0).float()
        img = np.transpose(img,(2,0,1)) #change the order to CxHxW
        img = torch.from_numpy(img).float()
        label = torch.FloatTensor([label])
        return img, label


class COVID19DataSetTest(torch.utils.data.Dataset):
    def __init__(self, x_test):
        self.x_test = x_test
        self.dims = [480,480]

    def __len__(self):
        return len(self.x_test)

    def __getitem__(self, idx):
        x = self.x_test[idx]
        img = np.asarray(x)
        img = img.astype(np.float32)/255. # normalize value to [0.,1.]
        if img.shape[2] > 3: # bypass RGBA case
            img = img[:,:,0:3]
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
        # img = torch.from_numpy(img).unsqueeze(0).float()
        img = np.transpose(img,(2,0,1)) #change the order to CxHxW
        img = torch.from_numpy(img).float()
        return img

