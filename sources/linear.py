from utils import *
import numpy as np
import argparse
import re


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


def perceive(data, label, args):
    """
    Pattern class: w1, w2,..., wm | n=args.cluster_num
    Pattern :X1, X2,..., Xn | m=args.cluster_num * args.sample_num
    :param data: A numpy array with shape[args.cluster_num * args.sample_num, 2]
    :param label: A numpy array with shape[args.cluster_num * args.sample_num, 1]
    :param args: arguments
    :return: W: weights
    """
    # Generate augmented matrix
    augment_colume = np.ones([data.shape[0], 1]).tolist()
    data_colume = data.tolist()
    X = np.array([a + b for a, b in zip(data_colume, augment_colume)])

    # Initialize weights
    W = np.zeros([args.cluster_num, 3])
    d = [[0. for i in range(args.cluster_num)] for i in range(args.cluster_num)]
    while True:
        counter = 0
        for it in range(args.sample_num):
            for j in range(args.cluster_num):
                for k in range(args.cluster_num):
                    d[j][k] = float(np.matmul(W[k], X[j * args.sample_num + it].T))
            for i in range(args.cluster_num):
                index = args.sample_num * i + it
                current_label = int(label[index])
                plus, minus, keep = compare_d(d[i], args.cluster_num, current_label)
                if len(keep) == args.cluster_num:
                    counter += 1
                else:
                    for p in plus:
                        W[p] = W[p] + args.correction * X[index]
                    for m in minus:
                        W[m] = W[m] - args.correction * X[index]
        if counter == data.shape[0]:
            break
    return W


if __name__ == '__main__':
    print(toRed("1. Parsing arguments..."))
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_num", type=int, default=3, help="the number of clusters")
    parser.add_argument("--sample_num", type=int, default=100, help="the number of samples per cluster")
    parser.add_argument("--center", type=str, default="0,-2;2,0;0,1.5", help="coordinates of the centers of each cluster")
    parser.add_argument("--correction", type=float, default=1.0, help="correction correlation")
    args = parser.parse_args()
    print(toGreen("Done"))
    print(args)
    print("Cluster_num:{}, Sample_num:{}, Center:{}".format(args.cluster_num, args.sample_num, args.center))

    print(toRed("2. Generating patterns..."))
    data, label = generate_data(args)
    print(toGreen("Done"), "\nThe data shape: {}".format(data.shape), "\nThe label shape: {}".format(label.shape))

    print(toRed("3. Perceiving..."))
    W = perceive(data, label, args)
    print(W)
    visualize_linear(args.cluster_num, args.sample_num, data, W)



