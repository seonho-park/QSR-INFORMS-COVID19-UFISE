import torch
import os
import glob
import shutil
import json
import argparse
import numpy as np
from datetime import datetime
from pprint import pprint

def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device:", device)
    return device 


def set_seed(seed, device):
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def config_backup_get_log(args, filename):
    if not os.path.isdir('./results'):
        os.mkdir('./results')
    if not os.path.isdir('./chpt'):
        os.mkdir('./chpt')

    # set result dir
    current_time = str(datetime.now())
    dir_name = '%s_%s'%(current_time, args.suffix)
    result_dir = 'results/%s'%dir_name

    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
        os.mkdir(result_dir+'/codes')

    # deploy codes
    files = glob.iglob('*.py')
    model_files = glob.iglob('./model/*.py')

    for file in files:
        shutil.copy2(file, result_dir+'/codes')
    for model_file in model_files:
        shutil.copy2(model_file, result_dir+'/codes/model')


    # printout information
    print("Export directory:", result_dir)
    print("Arguments:")
    pprint(vars(args))

    # logger = open(result_dir+'/%s.txt'%dir_name,'w')
    logger = open(result_dir+'/log.txt','w')
    logger.write("%s \n"%(filename))
    logger.write("Export directory: %s\n"%result_dir)
    logger.write("Arguments:\n")
    logger.write(json.dumps(vars(args)))
    logger.write("\n")

    return logger, result_dir, dir_name