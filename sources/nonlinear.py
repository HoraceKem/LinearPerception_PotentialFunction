from utils import *
import numpy as np
import argparse
import re
import math


def generate_data(args):
    """
    :param args: arguments
    :return: data, label
    """
    if args.center == 'auto':
        center = np.random.rand(args.cluster_num, 2) * 10
    else:
        center = re.split(',|;', args.center)
        center = np.array([float(x) for x in center]).reshape(-1, 2)
    data = []
    label = []
    for i in range(args.cluster_num):
        data.append(np.random.rand(args.sample_num, 2) + center[i])
        label.append(np.ones([args.sample_num, 1]) * i)
    data = np.concatenate([data[i] for i in range(args.cluster_num)], axis=0)
    label = np.concatenate([label[i] for i in range(args.cluster_num)], axis=0)
    return data, label


def potential(data, label, args):
    item_all = [[] for n in range(args.cluster_num)]
    for i in range(args.cluster_num):
        first_flag = True
        counter = 0
        while True:
            for j in range(data.shape[0]):
                if label[j] == i:
                    symbol = 1
                else:
                    symbol = -1
                if not first_flag:
                    K = compute_K(data[j][0], data[j][1], i, data, label, item_all[i])
                    if K * symbol < 0:
                        item_all[i].append(j)
                        counter = 0
                    else:
                        counter += 1
                else:
                    item_all[i].append(j)
                    first_flag = False
                if counter == data.shape[0]:
                    break
            if counter == data.shape[0]:
                break
    return item_all


if __name__ == '__main__':
    print(toRed("1. Parsing arguments..."))
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_num", type=int, default=4, help="the number of clusters")
    parser.add_argument("--sample_num", type=int, default=100, help="the number of samples per cluster")
    parser.add_argument("--center", type=str, default="0,-1;2,0;0,2;-2,0", help="coordinates of the centers of each cluster")
    args = parser.parse_args()
    print(toGreen("Done"))
    print(args)
    print("Cluster_num:{}, Sample_num:{}, Center:{}".format(args.cluster_num, args.sample_num, args.center))

    print(toRed("2. Generating patterns..."))
    data, label = generate_data(args)
    print(toGreen("Done"), "\nThe data shape: {}".format(data.shape), "\nThe label shape: {}".format(label.shape))

    print(toRed("3. Perceiving..."))
    item_all = potential(data, label, args)
    print(toGreen("Done"))
    print("The index of the potential function items in the data:")
    print(item_all)
    print(toRed("4. Visualizing..."))
    visualize_nonlinear(args.cluster_num, args.sample_num, data, label, item_all)
