import os
import csv
from datetime import datetime
current_time = str(datetime.now())

if not os.path.isdir('./batchout'):
        os.mkdir('./batchout')

seeds = [1000,1001,1002,1003,1004,1005,1006,1007,1008,1009]
# ctonlys = [True, False]
# mocos = [True, False]
ablationlevels = [1,2]
# models = ['mobilenet', 'densenet']
models = ['mobilenet']


ntotal_trials = len(seeds) * len(ablationlevels) * len(models)

csv_name = './batchout/results_%s.csv'%(current_time)
file = open(csv_name, 'w', newline ='')
writer = csv.writer(file)

writer.writerow(["ablationlevel","model", "seed","auroc","aupr","best_epoch"])

count = 0
for model in models:
    for ablationlevel in ablationlevels:
        for seed in seeds:
            print("trial:", count, "out of", ntotal_trials)
            args = "--seed=%s"%(seed)
            if model in ['mobilenet']:
                args += " --model=mobilenet --bstrain=32"
            elif model in ['densenet']:
                args += " --model=densenet --bstrain=16"
            if ablationlevel == 1: # ctonly and moco=False
                args += " --ctonly"
            elif ablationlevel == 2: # ctonly=False and moco=False
                args += ""
            elif ablationlevel == 3: # ctonly=False and moco=True
                args += " --moco"

            print(args)
            exec_str = "python ./train.py %s --suffix=\"%s\"  --batchout"%(args, args)
            csvline = [ablationlevel, model, seed]
            os.system("%s"%(exec_str))
            with open('temp_result.txt', 'r') as f:
                auroc = float(f.readline())
                aupr = float(f.readline())
                best_epoch = float(f.readline())
            csvline.append(auroc)
            csvline.append(aupr)
            csvline.append(best_epoch)
            writer.writerow(csvline)
            count += 1

file.close()