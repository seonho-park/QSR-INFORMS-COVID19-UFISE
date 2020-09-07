import os
import glob
import argparse
from PIL import Image
import numpy as np
from Model import estimate

def main():
    covid_img_list = sorted(glob.glob(os.path.join(args.datapath, "Images-processed", "CT_COVID", '*.*')))
    noncovid_img_list = sorted(glob.glob(os.path.join(args.datapath, "Images-processed", "CT_NonCOVID", '*.*')))

    X_train = []

    for img_fn in covid_img_list:
        img = Image.open(img_fn)
        img = np.asarray(img) # convert to numpy array, order: HxWxC
        X_train.append(img)
    
    for img_fn in noncovid_img_list:
        img = Image.open(img_fn)
        img = np.asarray(img) # convert to numpy array, order: HxWxC
        X_train.append(img)

    
    y_train = ['COVID']*len(covid_img_list) + ['NonCOVID']*len(noncovid_img_list)
    estimate(X_train, y_train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default="/home/sean/data/COVID-CT QSR Data Challenge/COVID-CT QSR Data Challenge", help='data path')
    args = parser.parse_args()
    main()