import argparse
import os
import glob
import numpy as np
from PIL import Image


def main():
    
    lungseg_imgs_fn = sorted(glob.glob(os.path.join(args.datapath,'2d_images','*.*')))
    lungseg_masks_fn = sorted(glob.glob(os.path.join(args.datapath,'2d_masks','*.*')))

    lungseg_imgs = []
    for i, img_fn in enumerate(lungseg_imgs_fn):
        img = Image.open(img_fn).convert('L') # test image
        img = np.asarray(img) # convert to numpy array, order: HxWxC
        lungseg_imgs.append(img)

    lungseg_masks = []
    for i, img_fn in enumerate(lungseg_masks_fn):
        img = Image.open(img_fn).convert('L') # test image
        img = np.asarray(img) # convert to numpy array, order: HxWxC
        lungseg_masks.append(img)


    lungseg_imgs = np.asarray(lungseg_imgs)
    lungseg_masks = np.asarray(lungseg_masks)
    # lungseg_imgs=np.reshape(np.array(lungseg_imgs),[len(lungseg_imgs),])
    # lungseg_masks=np.reshape(np.array(lungseg_masks),[len(lungseg_masks),])

    lungseg_imgs.dump("lungseg_imgs.npy")
    lungseg_masks.dump("lungseg_masks.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default="/home/sean/data/COVID-CT QSR Data Challenge/LungSegCT_kaggle", help='data path')
    args = parser.parse_args()
    main()
    