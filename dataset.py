import torch
import torchvision
import numpy as np
import scipy.signal as ss
import scipy.ndimage as snd
import skimage.transform as skt

class COVID19DataSet(torchvision.datasets.ImageFolder):
    def __init__(self, data_path, dims=[256,256]):
        super(COVID19DataSet, self).__init__(root=data_path)
        self.dims = dims # dimension of the image 
    
    def __getitem__(self, idx):
        img, label = super(COVID19DataSet, self).__getitem__(idx)
        img = np.asarray(img) # convert to numpy array
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
        label = torch.LongTensor([label])
        return img, label
