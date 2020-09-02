import os
import torch
import argparse
from PIL import Image
import numpy as np
import scipy.signal as ss
import scipy.ndimage as snd
import skimage.transform as skt
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

# from Model import ctprocessing

def ctprocessing(img):
    img = np.asarray(img)
    img = img.astype(np.float32)/255. # normalize value to [0.,1.]

    # adjust brightness
    img = skt.resize(img, (480,480), mode='constant', anti_aliasing=False)
    bandwidth = 255
    img = (img - img.min()) / (img.max() - img.min())
    imhist = (img * bandwidth).astype(np.uint8)
    h = np.histogram(imhist.flatten(), bins=bandwidth+1)
    # plt.hist(imhist.flatten(), bins=bandwidth+1)
    # plt.show()

    hmed = ss.medfilt(h[0], kernel_size=51) # median filter
    hf = snd.gaussian_filter(hmed, sigma=25) 
    hf = np.maximum(0, hf - len(img.flatten())*0.001) # reject 0.1% of mass
    if np.max(hf) > 0:
        hmin = np.nonzero(hf)[0][0] /bandwidth
        hmax = np.nonzero(hf)[0][-1]/bandwidth
        img = (img - hmin) / (hmax - hmin)
        img = np.maximum(0,img)
        img = np.minimum(1.,img)
        # img = np.maximum(hmin,img)
        # img = np.minimum(hmax,img)
    return img

def main():
    raw_images = [
                    # 'CT_COVID/2020.02.25.20021568-p23-108%12.png',
                    'CT_NonCOVID/1868.png',
                    'CT_COVID/2020.03.19.20038539-p10-60.png',
                    'CT_COVID/2020.03.04.20031039-p23-97_2%0.png',
                    # 'CT_NonCOVID/2142.png',
                    'CT_NonCOVID/1101.png',
                    # 'CT_NonCOVID/1309.png',
                    # 'CT_COVID/2020.02.11.20022053-p12-67%0.png',
                    'CT_COVID/2020.03.12.20034686-p17-91-5.png']
    
    ctimages = []    
    processed_ctimages = []
    for raw_img in raw_images:
        ctimagename = os.path.join(args.datapath, 'Images-processed', raw_img)
        ctimage = Image.open(ctimagename).convert('L')
        processed_ctimage = ctprocessing(ctimage)
        processed_ctimage = torch.from_numpy(processed_ctimage).float().unsqueeze(0)
        # processed_ctimage = torch.from_numpy(processed_ctimage).byte().unsqueeze(0)
        ctimage = np.asarray(ctimage)
        ctimage = skt.resize(ctimage, (480,480), mode='constant', anti_aliasing=False)
        ctimage = torch.from_numpy(ctimage).float().unsqueeze(0)
        # ctimage = torch.from_numpy(ctimage).byte().unsqueeze(0)
        # images.append(torch.stack([ctimage, processed_ctimage], 0))
        ctimages.append(ctimage)
        processed_ctimages.append(processed_ctimage)
        
    images = ctimages + processed_ctimages
    images = torch.stack(tuple(images), 0)


    images = make_grid(images, nrow=len(raw_images))
    output_dir = 'figure_output'
    os.makedirs(output_dir, exist_ok=True)
    output_name =  'output.png'
    output_path = os.path.join(output_dir, output_name)

    save_image(images, output_path)
    
    result_img = Image.open(output_path)
    result_img.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default="/home/sean/data/COVID-CT QSR Data Challenge/COVID-CT QSR Data Challenge", help='data path')
    args = parser.parse_args()
    main()


