import glob
import os
import torch
from PIL import Image
import numpy as np
import scipy.signal as ss
import scipy.ndimage as snd
import skimage.transform as skt


from Model import ResNetUNet

# import utils


def process(net, img_fn):
    img = Image.open(img_fn).convert('L') # test image
    img = np.asarray(img) # convert to numpy array, order: HxWxC
    org_dim = img.shape
    img = img.astype(np.float32)/255. # normalize value to [0.,1.]
    img = skt.resize(img, (480, 480), mode='constant', anti_aliasing=False)
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

    output = torch.sigmoid(net(img)).detach().cpu().numpy().squeeze() * 255.
    output = skt.resize(output, org_dim, mode='constant', anti_aliasing=False)
    out_img = Image.fromarray(output).convert("L")

    return out_img


def main():
    input_root = "/home/sean/data/COVID-CT QSR Data Challenge/COVID-CT QSR Data Challenge/Images-processed"
    output_root = "/home/sean/data/COVID-CT QSR Data Challenge/COVID-CT QSR Data Challenge/lung_segmentation"
    covid_img_list = sorted(glob.glob(os.path.join(input_root, "CT_COVID", '*.*')))
    noncovid_img_list = sorted(glob.glob(os.path.join(input_root, "CT_NonCOVID", '*.*')))

    net = ResNetUNet() # load model
    net.load_state_dict(torch.load("lungseg_net.pth"))

    for i, img_fn in enumerate(covid_img_list):
        img_filename = os.path.basename(img_fn)
        print("%d/%d: %s"%(i, len(covid_img_list), img_filename))
        out_img = process(net, img_fn)
        out_img.save(os.path.join(output_root, "CT_COVID", img_filename)) # save lungseg image

    for i, img_fn in enumerate(noncovid_img_list):
        img_filename = os.path.basename(img_fn)
        print("%d/%d: %s"%(i, len(noncovid_img_list), img_filename))
        out_img = process(net, img_fn)
        out_img.save(os.path.join(output_root, "CT_NonCOVID", img_filename)) # save lungseg image


if __name__ == "__main__":

    main()