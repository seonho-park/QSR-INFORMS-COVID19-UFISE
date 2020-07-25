import argparse
from dataset import COVID19DataSet

def main():
    data = COVID19DataSet(args.datapath)
    img, label = data[1]


    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default="/home/sean/data/COVID-CT QSR Data Challenge/COVID-CT QSR Data Challenge/Images-processed", help='data path')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--bstrain', type=int, default=128, help='batch size for training')
    parser.add_argument('--bstest', type=int, default=128, help='batch size for testing')
    parser.add_argument('--nworkers', type=int, default=2, help='the number of workers used in DataLoader')
    parser.add_argument('--maxepoch', type=int, default=200, help='the number of epoches')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    args = parser.parse_args()
    main()
