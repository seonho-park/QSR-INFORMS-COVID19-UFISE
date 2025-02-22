import glob
import argparse
import os
import torch
import numpy as np
from PIL import Image
import skimage.transform as skt
from torchvision.utils import make_grid, save_image

from gradcam import GradCAM, GradCAMpp, visualize_cam
import utils
from Model import mobilenet_v2, densenet121, ctprocessing


def main():
    device = utils.get_device()
    
    # load trained model
    net = mobilenet_v2(task = 'classification', moco = False, ctonly = False).to(device)
    state_dict = torch.load("./model.pth")
    net.load_state_dict(state_dict)
    net.eval() # eval mode

    # load GradCAM
    model_dict = dict(type='mobilenet', arch=net, layer_name='features_18', input_size=(480, 480))
    gradcampp = GradCAMpp(model_dict, True)

    # load image
    raw_images = ['CT_COVID/2020.01.24.919183-p27-132.png',
                  'CT_COVID/2020.02.22.20024927-p18-66%2.png',
                  'CT_COVID/2020.02.26.20026989-p34-114_1%1.png',
                  'CT_NonCOVID/673.png',
                  'CT_NonCOVID/1262.png',
                  'CT_NonCOVID/40%1.jpg']

    images = []    
    for raw_image in raw_images:
        ctimagename = os.path.join(args.datapath, 'Images-processed', raw_image)
        lungimagename = os.path.join(args.datapath, 'lung_segmentation', raw_image)
        ctimage = Image.open(ctimagename).convert('L')
        lungimage = Image.open(lungimagename).convert('L')
        ctimage = ctprocessing(ctimage)
        ctimage = torch.from_numpy(ctimage).float().unsqueeze(0).unsqueeze(0).to(device)

        lungimage = np.asarray(lungimage)
        lungimage = lungimage.astype(np.float32)/255.
        lungimage = skt.resize(lungimage, (480,480), mode='constant', anti_aliasing=False) # resize
        lungimage = torch.from_numpy(lungimage).float().unsqueeze(0).unsqueeze(0).to(device)

        mask_pp, _ = gradcampp(ctimage, lungimage)
        heatmap_pp, result_pp = visualize_cam(mask_pp.cpu(), ctimage)
        images.append(
            torch.stack([
                ctimage.squeeze().cpu().unsqueeze(0).expand(3,-1,-1), \
                lungimage.squeeze().cpu().unsqueeze(0).expand(3,-1,-1), \
                heatmap_pp, result_pp], 0))

    images = make_grid(torch.cat(images, 0), nrow=4) #####

    output_dir = 'gradcam_output'
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



